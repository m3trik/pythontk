# !/usr/bin/python
# coding=utf-8
"""OperationServer -- the operation catalog as an MCP server: one tool per operation.

``python -m pythontk ops mcp`` serves every discovered operation to an agent
over the Model Context Protocol (stdio). Each operation is one tool, named by
its id with ``.`` as ``_`` (``maya.python.exec`` ->
``maya_python_exec``); its input schema is the operation's argument schema
(plus ``scenes`` for an operation a host application runs), and its result is
the run's :class:`~pythontk.Outcome` as structured content.

Two servers split what an agent may do without asking. ``ops mcp`` serves the
reads and the dry runs: nothing it does changes anything (a kept instance may
open the scene files a call names, but never over unsaved changes), so a
client can allow it whole. ``ops mcp --apply`` serves only the tools that make changes,
under the same names. Clients match permission rules by server and tool name
(tool annotations are hints a client may not act on), so the split is what
lets a person allow every plan and still be asked before each change.

Host-application calls run in instances kept for the client's session
(:class:`~pythontk.SessionOperationHost`): the client's id names the session
(:attr:`OperationServer.SESSION_ENV`), so both servers of one client share one
instance per host, and the server ends the ones it started when the client
disconnects. A result that is an image the run made (a viewport capture)
comes back as MCP image content too.

Progress reaches the client as MCP progress notifications when it asks for
them; a cancelled request cancels the run's :class:`~pythontk.CancelScope`, so
the operation stops at its next checkpoint. Operations run in a worker thread;
those running in this process take turns (they share it), while host calls
queue in their instance.

Adopted, not built: the official MCP Python SDK (``mcp``, the
``pythontk[mcp]`` extra, Python 3.10+). Only :meth:`OperationServer.serve`
imports it; :meth:`tools` and :meth:`call` are plain data, so every front end
and test reads the same tool list without the SDK.
"""

from __future__ import annotations

import base64
import copy
import os
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional

from pythontk.core_utils.cancel_scope import CancelScope
from pythontk.core_utils.operation_catalog.batch_operation_host import PureSceneFiles
from pythontk.core_utils.operation_catalog.operation import Operation
from pythontk.core_utils.operation_catalog.outcome import Outcome


class _OperationServerInternal:
    """Internal helpers for :class:`OperationServer`."""

    #: The argument a host operation's tool takes its scene files by.
    SCENES = "scenes"

    #: What the server tells a client, in order: the host part only when it
    #: serves a host's tools, naming those hosts (kept instances or fresh ones,
    #: as its catalog runs them), and the API part only when it serves the
    #: ``api.*`` operations.
    INSTRUCTIONS = (
        "This codebase's operations, one tool each, named by the operation id "
        "with dots as underscores. A tool that changes anything only PLANS on the "
        "`ops` server: its result's `changes` is the plan, and nothing is changed; "
        "the tool of the same name on the `ops-apply` server makes the changes. "
        "Code has no plan, so the code tools (`*_exec`) are on `ops-apply` alone.",
        "Tools led by a host's name ({hosts}) run in this session's own instance "
        "of that application: started on the first call (seconds), kept between "
        "calls and shared by both servers, never a session a person opened; it "
        "restarts when code it loaded changes on disk, and the result says so. "
        "Pass `scenes` (absolute paths) to open scene files first, replacing what "
        "it holds (refused while it holds unsaved changes: save or discard them "
        "from code first); without them a tool works on what it holds. A live run "
        "of a tool that plans saves the scenes it opened and changed; code saves "
        "only what it saves itself, and counts as a change until it does. Nothing "
        "is selected there, so code names the objects it works on.",
        "Tools led by a host's name ({hosts}) run in a FRESH instance of that "
        "application per call, never an open session; pass `scenes` (absolute "
        "paths) to open scene files first, and a live run of a tool that plans saves "
        "the ones it changed (code saves only what it saves itself: the instance "
        "ends with the call). Nothing is selected there, so code names the objects "
        "it works on.",
        "The packages' whole public API is reached through code: `api_search` "
        "finds any public function or class and `api_describe` reads one (its "
        "signature, docstring and the host it runs in); that host's `*_exec` on "
        "`ops-apply` runs code in its instance (`python_exec` for a symbol no "
        "application owns), in a console whose names last as long as that instance "
        "does, and a cancel interrupts the code without losing them; "
        "`*_scene_view` shows its scene. Code that opens windows (a Qt panel) is "
        "not for these instances.",
        "Every result is an Outcome: `status` (done, partial, cancelled, failed), "
        "`result`, `changes`, `issues` (each may carry a `fix`: an operation call "
        "to make) and `error`. A result that is an image the run made (a "
        "`*_scene_view`) comes back as the image too.",
    )
    #: Environment variables naming the client's session, the first set one
    #: winning: pythontk's own, then Claude Code's (it gives every server of
    #: one session the same id, in the server's own environment).
    SESSION_ENV = ("PYTHONTK_OPS_SESSION", "CLAUDE_CODE_SESSION_ID")
    #: Image files a result may name, and the largest one sent back.
    IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    IMAGE_MAX_BYTES = 5 * 1024 * 1024

    @classmethod
    def _input_schema(cls, entry: Mapping[str, Any], kept: bool) -> Dict[str, Any]:
        schema = copy.deepcopy(dict(entry.get("args") or {}))
        schema.setdefault("type", "object")
        properties = schema.setdefault("properties", {})
        if entry.get("host") and entry["host"] != PureSceneFiles.HOST:  # it opens none
            where = (
                f"this session's {entry['host']} instance opens first, replacing what it "
                "holds (refused while that has unsaved changes)"
                if kept
                else f"the fresh {entry['host']} instance opens first"
            )
            properties[cls.SCENES] = {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    f"Scene files {where}, as absolute paths (the first is the main "
                    "one); a live run saves those it changed."
                ),
            }
        return schema

    @classmethod
    def _image(cls, outcome: Outcome) -> Optional[Dict[str, Any]]:
        """The image a run made, as MCP image content: a result naming an
        image file in pythontk's temp root (where a capture writes, and no
        asset of a person's lives), or ``None``."""
        from pythontk.file_utils._file_utils import FileUtils
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        path = outcome.result
        if not outcome.ok or not isinstance(path, str):
            return None
        mime = cls.IMAGE_TYPES.get(os.path.splitext(path)[1].lower())
        # The root itself, never the system temp folder TempArtifacts.contains also counts.
        root = os.path.realpath(TempArtifacts.root())
        if mime is None or not FileUtils.is_under(os.path.realpath(path), root):
            return None
        try:
            if os.path.getsize(path) > cls.IMAGE_MAX_BYTES:
                return None
            with open(path, "rb") as handle:
                data = base64.b64encode(handle.read()).decode("ascii")
        except OSError:
            return None
        return {"type": "image", "data": data, "mimeType": mime}

    #: Schema words that describe without constraining, left out of the
    #: Outcome schema each tool repeats (the instructions describe it once).
    _PROSE = ("description", "title", "$schema", "examples")
    #: Keywords whose values are data, never a schema to trim.
    _DATA = ("default", "enum", "const", "examples")

    @classmethod
    def _terse(cls, schema: Any) -> Any:
        """*schema* with its prose left out: the shape a value must have."""
        if isinstance(schema, list):
            return [cls._terse(s) for s in schema]
        if not isinstance(schema, Mapping):
            return schema
        out = {}
        for key, value in schema.items():
            if key in cls._PROSE:
                continue
            if key in ("properties", "$defs"):
                out[key] = {name: cls._terse(s) for name, s in value.items()}
            else:
                out[key] = value if key in cls._DATA else cls._terse(value)
        return out

    @staticmethod
    def _description(entry: Mapping[str, Any], apply: bool, kept: bool) -> str:
        parts = [entry.get("summary") or entry.get("title") or entry["id"]]
        if entry["effect"] == Operation.READ:
            parts.append("Reads only; it changes nothing.")
        elif entry["effect"] == Operation.CODE:
            parts.append(
                "Runs code: it may change anything it reaches, and has no plan, "
                "so it is on this server alone."
            )
        elif apply:
            parts.append(
                f"Makes its changes ({entry['effect']}). The `ops` server's tool of "
                "the same name returns the plan first, changing nothing."
            )
        else:
            parts.append(
                "Plans only (a dry run): `changes` in the result lists what it "
                "would change, and nothing is changed. The `ops-apply` server's "
                "tool of the same name makes the changes."
            )
        if entry.get("host"):
            parts.append(
                f"Runs in this session's {entry['host']} instance."
                if kept
                else f"Runs in a fresh {entry['host']} instance per call."
            )
        return " ".join(parts)

    @staticmethod
    def _progress_token(meta: Any) -> Any:
        if not meta:
            return None
        get = meta.get if hasattr(meta, "get") else lambda k: getattr(meta, k, None)
        token = get("progress_token")
        return token if token is not None else get("progressToken")  # 0 is a token


class OperationServer(_OperationServerInternal):
    """The operation catalog as MCP tools (see the module docstring).

    Parameters:
        catalog: The operations to serve (``OperationCatalog.discover()``).
        apply: Serve the tools that make changes, and only those; otherwise
            the reads and the dry runs.
        name: The server's name, as a client lists it.

    An operation it cannot serve (a second id on one tool name, an argument
    named ``scenes`` on a host operation) is left out and named in
    :attr:`problems`, as the catalog's discovery does: one never stops the rest.
    """

    def __init__(self, catalog, apply: bool = False, name: str = "ops"):
        self.catalog = catalog
        self.apply = bool(apply)
        self.name = name
        self.problems: List[str] = []
        self._ops: Dict[str, Dict[str, Any]] = {}
        for entry in catalog.describe():
            if not self._serves(entry):
                continue
            tool = self.tool_name(entry["id"])
            if tool in self._ops:
                self.problems.append(
                    f"operations {self._ops[tool]['id']!r} and {entry['id']!r} would "
                    f"both be served as tool {tool!r}; the second is left out"
                )
            elif entry.get("host") and self.SCENES in (entry.get("args") or {}).get(
                "properties", {}
            ):
                self.problems.append(
                    f"operation {entry['id']!r} has an argument named {self.SCENES!r}, "
                    "which its tool takes the scene files by; it is left out"
                )
            else:
                self._ops[tool] = entry
        self._local = threading.Lock()

    @staticmethod
    def tool_name(op_id: str) -> str:
        """The tool an operation is served as: its id with ``.`` as ``_``."""
        return op_id.replace(".", "_")

    def _serves(self, entry: Mapping[str, Any]) -> bool:
        if entry["effect"] == Operation.READ:
            return not self.apply
        if entry["effect"] == Operation.CODE:  # never on the plan server, however declared
            return self.apply
        return self.apply or bool(entry.get("dry_run"))

    @property
    def kept(self) -> bool:
        """Whether host calls run in instances kept for a session (else fresh per call)."""
        return getattr(self.catalog, "session", None) is not None

    @classmethod
    def session_name(cls) -> str:
        """The session this server keeps its host instances for: the client's
        (:attr:`SESSION_ENV`), so every server of one client shares them, else
        this process's own."""
        for var in cls.SESSION_ENV:
            if os.environ.get(var):
                return os.environ[var]
        return f"pid{os.getpid()}"

    def instructions(self) -> str:
        """What this server tells a client about its tools, the hosts it serves
        named (the scene rules: so never the plain Python, which opens none)."""
        hosts = sorted(
            {
                entry["host"]
                for entry in self._ops.values()
                if entry.get("host") and entry["host"] != PureSceneFiles.HOST
            }
        )
        intro, kept, fresh, api, outcome = self.INSTRUCTIONS
        parts = [intro]
        if hosts:
            parts.append(
                (kept if self.kept else fresh).format(
                    hosts=", ".join(f"{h}_" for h in hosts)
                )
            )
        if self.tool_name("api.search") in self._ops:
            parts.append(api)
        return " ".join(parts + [outcome])

    def tools(self) -> List[Dict[str, Any]]:
        """Every tool this server offers, as MCP ``Tool`` mappings, sorted by name."""
        output = self._terse(Outcome.json_schema())
        tools = []
        for tool, entry in sorted(self._ops.items()):
            changes = entry["effect"] != Operation.READ and self.apply
            tools.append(
                {
                    "name": tool,
                    "title": entry.get("title") or entry["id"],
                    "description": self._description(entry, self.apply, self.kept),
                    "inputSchema": self._input_schema(entry, self.kept),
                    "outputSchema": output,
                    "annotations": {
                        "title": entry.get("title") or entry["id"],
                        "readOnlyHint": not changes,
                        "destructiveHint": changes,
                        "idempotentHint": not changes,
                        "openWorldHint": False,
                    },
                }
            )
        return tools

    def call(
        self,
        tool: str,
        arguments: Optional[Mapping[str, Any]] = None,
        scope: Optional[CancelScope] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
    ) -> Outcome:
        """Run the operation *tool* serves; never raises (the outcome says why).

        Parameters:
            tool: The tool's name.
            arguments: Its arguments; ``scenes`` is taken off for the host.
            scope: The cancel scope the run honours.
            on_progress: Called with ``(fraction, message)`` as it goes.
        """
        entry = self._ops.get(tool)
        if entry is None:
            return Outcome.refused(tool, f"this server has no tool {tool!r}")
        args = dict(arguments or {})
        scenes = args.pop(self.SCENES, None) if entry.get("host") else None
        if scenes is not None and not (
            isinstance(scenes, list) and all(isinstance(s, str) for s in scenes)
        ):
            return Outcome.refused(
                entry["id"], f"{self.SCENES!r} must be a list of paths"
            )

        def run() -> Outcome:
            return self.catalog.run(
                entry["id"],
                args,
                apply=self.apply,
                scenes=scenes or (),
                scope=scope,
                on_progress=on_progress,
            )

        if entry.get("host"):
            return run()  # its instance (kept or fresh) orders them: calls may overlap
        with self._local:  # this process: one at a time
            return run()

    async def serve(self) -> None:
        """Serve the tools over stdio until the client disconnects (needs ``mcp``)."""
        import anyio
        import mcp.types as types
        from mcp.server.lowlevel import Server
        from mcp.server.stdio import stdio_server

        import pythontk

        async def list_tools(ctx, params) -> "types.ListToolsResult":
            return types.ListToolsResult(
                tools=[types.Tool.model_validate(t) for t in self.tools()]
            )

        async def call_tool(ctx, params) -> "types.CallToolResult":
            scope = CancelScope("mcp")
            token = self._progress_token(getattr(ctx, "meta", None))
            on_progress = None
            if token is not None:

                def on_progress(fraction: float, message: str) -> None:
                    anyio.from_thread.run(
                        ctx.session.send_progress_notification,
                        token,
                        fraction,
                        1.0,
                        message,
                    )

            try:
                outcome = await anyio.to_thread.run_sync(
                    lambda: self.call(
                        params.name, params.arguments, scope, on_progress
                    ),
                    abandon_on_cancel=True,
                )
            except anyio.get_cancelled_exc_class():
                scope.cancel("the client cancelled the request")
                raise
            content = [{"type": "text", "text": outcome.to_json()}]
            image = self._image(outcome)
            if image is not None:
                content.append(image)
            return types.CallToolResult.model_validate(
                {
                    "content": content,
                    "structuredContent": outcome.to_dict(),
                    "isError": not outcome.ok,
                }
            )

        server = Server(
            f"pythontk-{self.name}",
            version=getattr(pythontk, "__version__", ""),
            instructions=self.instructions(),
            on_list_tools=list_tools,
            on_call_tool=call_tool,
        )
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    def warm(self) -> Optional[threading.Thread]:
        """When this server serves ``api_search``, read every package a search
        covers in the background, so the first search answers from memory (a
        cold read of a large package's source takes seconds).

        Returns:
            The reading thread, or ``None`` when there is nothing to read.
        """
        if self.tool_name("api.search") not in self._ops:
            return None
        from pythontk.core_utils.api_surface.api_surface import ApiSurface

        def read() -> None:
            for package in ApiSurface.registered():
                try:
                    ApiSurface.read(package)
                except Exception:  # a search that names it says why
                    pass

        thread = threading.Thread(target=read, name="api-surface-warm", daemon=True)
        thread.start()
        return thread

    def run(self) -> int:
        """Serve until the client disconnects (the ``ops mcp`` command), then
        end the host instances this server started; the exit code."""
        try:
            import anyio
            import mcp  # noqa: F401 - checked here, used by serve
        except ImportError:
            raise SystemExit(
                "the MCP server needs the official MCP SDK: pip install 'pythontk[mcp]' "
                "(or run it as: uv run --with mcp python -m pythontk ops mcp)"
            )
        self.warm()
        try:
            anyio.run(self.serve)
        finally:
            close = getattr(self.catalog, "close", None)
            if callable(close):
                close()
        return 0
