# !/usr/bin/python
# coding=utf-8
"""OperationCatalog -- every declared operation, read from committed catalogs and run.

The catalog is data: one :class:`~pythontk.OperationCatalogDocument` per
provider, committed beside the provider's declaration module and shipped in its
wheel. Python packages write theirs from their declarations
(:class:`~pythontk.OperationProvider`, which also onboards a package and checks it);
a provider in another language writes its own (a Unity project's, from the
editor's live registry). Every front end reads the same entries -- ``python -m
pythontk ops``, the MCP server, a panel -- so a new operation reaches all of
them as one addition (CODE_STANDARD section 0, Seams).

A provider registers its declaration module under the ``pythontk.operations``
entry-point group of its ``pyproject.toml``::

    [project.entry-points."pythontk.operations"]
    mayatk = "mayatk.operations"

:meth:`OperationCatalog.discover` reads each registered provider's committed
document without importing the provider (an entry point is metadata, and the
file is found beside the module's source), so listing and searching cost
nothing a fresh interpreter has not already paid. A provider not yet written,
and any module named explicitly (``--provider`` on the command line: a
checkout's working declarations, or an overlay declaring operations for a
codebase that does not know about them), is built live from its declarations.
Every document read is validated against the declared shape, so a malformed
one (hand-edited, or from another language's writer) is reported, never run.

A provider whose catalog another language writes registers the module path
its document sits at, and commits the document there; no module is needed,
since nothing is imported to read it. A catalog outside any package (one a
Unity project's editor writes into the project) is named by its file instead:
``--provider path/to/operations.json``, read as it stands.

Running a call validates its arguments against the entry's schema (an unknown
argument is refused, not ignored), decides dry run or live from the entry's
effect and the caller's ``apply``, and hands it to the entry's host
(:mod:`.operation_host`): this process when the entry names none, otherwise the
runner that application's library registers under ``pythontk.operation_hosts``.

Adopted, not built: JSON Schema for the argument shapes (the dialect
``SchemaSpec.json_schema`` already writes, and the one MCP tools take), entry
points for discovery, ``CLI.dispatch`` for the command line. The small schema
check here covers the subset the catalogs write; a full validator
(``jsonschema``) was weighed and left out as a dependency pythontk does not take.
"""

from __future__ import annotations

import difflib
import json
import sys
import threading
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from pythontk.core_utils.api_surface.api_surface import ApiSurface
from pythontk.core_utils.cancel_scope import CancelScope
from pythontk.core_utils.cli import CLI
from pythontk.core_utils.operation_catalog.operation import (
    Operation,
    OperationCatalogDocument,
    OperationEntry,
)
from pythontk.core_utils.operation_catalog.batch_operation_host import (
    BatchOperationHost,
)
from pythontk.core_utils.operation_catalog.operation_host import (
    OperationHost,
    PureOperationHost,
)
from pythontk.core_utils.operation_catalog.operation_provider import OperationProvider
from pythontk.core_utils.operation_catalog.outcome import OpRequest, Outcome
from pythontk.core_utils.operation_catalog.session_operation_host import (
    SessionOperationHost,
)
from pythontk.core_utils.package_manager import PackageManager


class _OperationCatalogInternal:
    """Internal helpers for :class:`OperationCatalog`: the schema check, the
    document reader and the command line."""

    # -- the schema check ------------------------------------------------------
    @staticmethod
    def _is_kind(value: Any, kind: str) -> bool:
        """Whether *value* is the JSON Schema *kind* (an integer-valued number
        is an ``integer``, as JSON Schema counts it)."""
        if kind == "null":
            return value is None
        if kind == "boolean":
            return isinstance(value, bool)
        if kind == "integer":
            if isinstance(value, bool):
                return False
            return isinstance(value, int) or (
                isinstance(value, float) and value.is_integer()
            )
        if kind == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if kind == "string":
            return isinstance(value, str)
        if kind == "array":
            return isinstance(value, (list, tuple))
        if kind == "object":
            return isinstance(value, Mapping)
        return True

    @classmethod
    def _json_kind(cls, value: Any) -> str:
        for kind in ("null", "boolean", "string", "array", "object", "number"):
            if cls._is_kind(value, kind):
                return (
                    "integer" if kind == "number" and isinstance(value, int) else kind
                )
        return type(value).__name__

    @staticmethod
    def _kinds_of(schema: Mapping[str, Any]) -> List[str]:
        kinds = schema.get("type")
        if kinds:
            return [kinds] if isinstance(kinds, str) else list(kinds)
        if "$ref" in schema:
            return [str(schema["$ref"]).rsplit("/", 1)[-1]]
        return [
            k
            for option in schema.get("anyOf", ())
            for k in _OperationCatalogInternal._kinds_of(option)
        ]

    @classmethod
    def _schema_errors(
        cls, schema: Mapping[str, Any], value: Any, path: str, defs: Mapping[str, Any]
    ) -> List[str]:
        """Every way *value* breaks *schema* (the JSON Schema subset the
        catalogs write: type, enum, anyOf, $ref, items, prefixItems,
        properties, required, additionalProperties)."""
        if not schema:
            return []
        if "$ref" in schema:
            name = str(schema["$ref"]).rsplit("/", 1)[-1]
            if name not in defs:
                return [
                    f"{path}: its schema names {schema['$ref']!r}, which the catalog does not define"
                ]
            return cls._schema_errors(defs[name], value, path, defs)
        if "anyOf" in schema:
            options = list(schema["anyOf"])
            tried = [cls._schema_errors(s, value, path, defs) for s in options]
            if any(not errors for errors in tried):
                return []
            # A value that misses an Optional misses its one real shape: say how.
            real = [e for s, e in zip(options, tried) if s.get("type") != "null"]
            if len(real) == 1:
                return real[0]
            expected = " or ".join(
                dict.fromkeys(k for s in options for k in cls._kinds_of(s))
            )
            return [f"{path}: expected {expected}, got {cls._json_kind(value)}"]
        if "enum" in schema and value not in schema["enum"]:
            return [f"{path}: {value!r} is not one of {list(schema['enum'])}"]
        kinds = schema.get("type")
        if kinds:
            kinds = [kinds] if isinstance(kinds, str) else list(kinds)
            if not any(cls._is_kind(value, k) for k in kinds):
                return [
                    f"{path}: expected {' or '.join(kinds)}, got {cls._json_kind(value)}"
                ]
        errors: List[str] = []
        if isinstance(value, (list, tuple)):
            prefix = schema.get("prefixItems")
            if prefix is not None:
                if len(value) != len(prefix):
                    errors.append(
                        f"{path}: expected {len(prefix)} items, got {len(value)}"
                    )
                for i, (item, sub) in enumerate(zip(value, prefix)):
                    errors += cls._schema_errors(sub, item, f"{path}[{i}]", defs)
            elif schema.get("items"):
                for i, item in enumerate(value):
                    errors += cls._schema_errors(
                        schema["items"], item, f"{path}[{i}]", defs
                    )
        if isinstance(value, Mapping):
            properties = schema.get("properties", {})
            extra = schema.get("additionalProperties", True)
            for key, item in value.items():
                if key in properties:
                    errors += cls._schema_errors(
                        properties[key], item, f"{path}.{key}", defs
                    )
                elif extra is False:
                    errors.append(f"{path}: unknown key {key!r}")
                elif isinstance(extra, Mapping):
                    errors += cls._schema_errors(extra, item, f"{path}.{key}", defs)
            for key in schema.get("required", ()):
                if key not in value:
                    errors.append(f"{path}: missing key {key!r}")
        return errors

    # -- documents and references ---------------------------------------------
    @classmethod
    def _entries_of(cls, document: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """*document*'s entries, each tagged with its provider.

        Raises:
            ValueError: A newer format, or a document that breaks
                :class:`OperationCatalogDocument` (the errors listed).
        """
        provider = document.get("provider") if isinstance(document, Mapping) else None
        fmt = document.get("format", 0) if isinstance(document, Mapping) else 0
        if isinstance(fmt, int) and fmt > OperationCatalogDocument.FORMAT:
            raise ValueError(
                f"catalog of {provider!r} is format {fmt}; this pythontk reads up to "
                f"{OperationCatalogDocument.FORMAT}: upgrade pythontk"
            )
        result = OperationCatalogDocument.validate(document)
        if result.errors:
            raise ValueError(
                f"catalog of {provider!r} is invalid: {'; '.join(result.errors)}"
            )
        return [dict(e, provider=provider) for e in document["operations"]]

    @staticmethod
    def _read_document(path: Path) -> Any:
        """The JSON document at *path* (a leading BOM, which some writers add, is skipped)."""
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))

    @staticmethod
    def _host_led(entries: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
        """``{id: why}`` for each host-less entry whose id starts with a host's
        name (compared by the first dotted segment): an id led by a host's name
        runs in that host."""
        entries = list(entries)
        named = {e.get("host") for e in entries} - {None}
        return {
            e["id"]: (
                f"operation {e['id']!r} runs in any Python, but its id starts with "
                f"the host {e['id'].split('.')[0]!r}"
            )
            for e in entries
            if not e.get("host") and e["id"].split(".")[0] in named
        }

    # -- command line ----------------------------------------------------------
    @classmethod
    def _parse_pairs(
        cls, pairs: Sequence[str], raw: Optional[str], properties: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Arguments from ``--args`` (JSON, ``@file`` or ``-`` for stdin) and
        ``NAME=VALUE`` pairs: an argument whose one non-null type is a string
        takes its text as given (``null`` is ``None`` where it may be);
        any other parses as JSON when it can, else is its text."""
        args: Dict[str, Any] = {}
        if raw:
            if raw == "-":
                raw = sys.stdin.read()
            elif raw.startswith("@"):
                raw = Path(raw[1:]).read_text(encoding="utf-8")
            try:
                loaded = json.loads(raw)
            except ValueError as exc:
                raise SystemExit(f"--args is not JSON: {exc}")
            if not isinstance(loaded, dict):
                raise SystemExit("--args must be a JSON object")
            args.update(loaded)
        for pair in pairs or ():
            name, sep, text = pair.partition("=")
            if not sep or not name:
                raise SystemExit(f"argument {pair!r} is not NAME=VALUE")
            kinds = cls._kinds_of(properties.get(name, {}))
            if [k for k in kinds if k != "null"] == ["string"]:
                args[name] = None if (text == "null" and "null" in kinds) else text
                continue
            try:
                args[name] = json.loads(text)
            except ValueError:
                args[name] = text
        return args

    @classmethod
    def _cli_catalog_of(cls, args) -> "OperationCatalog":
        """The discovered catalog; what it could not load goes to stderr."""
        catalog = cls.discover(args.provider)
        for problem in catalog.problems:
            print(f"warning: {problem}", file=sys.stderr)
        return catalog

    @classmethod
    def _cli_list(cls, args) -> int:
        return cls._print_entries(cls._cli_catalog_of(args).describe(), args.json)

    @classmethod
    def _cli_search(cls, args) -> int:
        entries = cls._cli_catalog_of(args).search(" ".join(args.words))
        return cls._print_entries(entries, args.json)

    @staticmethod
    def _print_entries(entries: List[Dict[str, Any]], as_json: bool) -> int:
        if as_json:
            print(json.dumps(entries, indent=2))
            return 0
        # A summary may carry a character the console's code page lacks (a
        # stock Windows console is cp1252): print a stand-in, never raise.
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        width = max((len(e["id"]) for e in entries), default=0)
        for e in entries:
            where = f" @{e['host']}" if e.get("host") else ""
            line = (
                f"{e['id']:<{width}}  {e['effect']:<6}{where}  {e.get('summary', '')}"
            )
            print(line.encode(encoding, "replace").decode(encoding))
        return 0

    @classmethod
    def _cli_describe(cls, args) -> int:
        try:
            entry = cls._cli_catalog_of(args).describe(args.id)
        except KeyError as exc:
            raise SystemExit(exc.args[0])
        print(json.dumps(entry, indent=2))
        return 0

    @classmethod
    def _cli_run(cls, args, apply: bool) -> int:
        catalog = cls._cli_catalog_of(args)
        try:
            properties = catalog.get(args.id).get("args", {}).get("properties", {})
        except KeyError:
            properties = {}
        call_args = cls._parse_pairs(args.pairs, args.args, properties)
        outcome = catalog.run(args.id, call_args, apply=apply, scenes=args.scene)
        print(outcome.to_json())
        # A run that reports an error (a check, a lint) fails the command, as a linter does.
        failed = not outcome.ok or any(i.severity == "error" for i in outcome.issues)
        return 1 if failed else 0

    @classmethod
    def _cli_mcp(cls, args) -> int:
        from pythontk.core_utils.operation_catalog.operation_server import (
            OperationServer,
        )

        catalog = cls.discover(args.provider, session=OperationServer.session_name())
        server = OperationServer(
            catalog, apply=args.apply, name="ops-apply" if args.apply else "ops"
        )
        for problem in catalog.problems + server.problems:
            print(f"ops: {problem}", file=sys.stderr)
        return server.run()


class OperationCatalog(_OperationCatalogInternal):
    """A set of operation entries, and the one way to run them (see the module docstring).

    Parameters:
        entries: Catalog entries (``Operation.describe`` mappings, or entries
            read from a provider's document), each carrying its ``provider``.
        hosts: ``{host name: OperationHost}`` overriding the registered
            runners (``None`` keys the in-process one).
        session: Keep one instance per host for this session name and run
            every call of a registered runner there
            (:class:`~pythontk.SessionOperationHost`; any process given the
            name shares it), for a front end that lives long; ``None`` (a
            one-shot front end): a fresh instance per call. :meth:`close`
            ends the instances this process started.

    Raises:
        ValueError: Two entries share an id, or one breaks
            :class:`OperationEntry`.
    """

    #: The entry-point group a provider registers its declaration module under.
    ENTRY_POINT_GROUP = OperationProvider.ENTRY_POINT_GROUP
    #: The entry-point group an application's library registers its runner under.
    HOST_ENTRY_POINT_GROUP = ApiSurface.HOST_GROUP

    def __init__(
        self,
        entries: Iterable[Mapping[str, Any]] = (),
        hosts: Optional[Mapping[Optional[str], OperationHost]] = None,
        session: Optional[str] = None,
    ):
        self.session = session
        self._entries: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            errors = OperationEntry.validate(dict(entry)).errors
            problem = (
                None if errors else Operation.id_error(entry["id"], entry.get("host"))
            )
            errors += [problem] if problem else []
            if errors:
                raise ValueError(
                    f"operation {entry.get('id')!r} is invalid: {'; '.join(errors)}"
                )
            op_id = entry["id"]
            if op_id in self._entries:
                raise ValueError(
                    f"operation {op_id!r} is declared by both "
                    f"{self._entries[op_id].get('provider')!r} and {entry.get('provider')!r}"
                )
            self._entries[op_id] = dict(entry)
        misled = self._host_led(self._entries.values())
        if misled:
            raise ValueError("; ".join(misled.values()))
        self._hosts: Dict[Optional[str], Optional[OperationHost]] = dict(hosts or {})
        self._hosts_guard = threading.Lock()
        #: What :meth:`discover` could not load or had to ignore, one line each.
        self.problems: List[str] = []

    # -- reading ---------------------------------------------------------------
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        hosts: Optional[Mapping[Optional[str], OperationHost]] = None,
    ) -> "OperationCatalog":
        """The catalog in one committed document.

        Raises:
            ValueError: The document is a newer format or breaks its shape.
        """
        return cls(cls._entries_of(cls._read_document(path)), hosts)

    @classmethod
    def discover(
        cls,
        providers: Sequence[Union[str, Path]] = (),
        hosts: Optional[Mapping[Optional[str], OperationHost]] = None,
        session: Optional[str] = None,
    ) -> "OperationCatalog":
        """Every registered provider's operations, plus *providers*'.

        One provider that fails to load, declares an id another already did,
        or ships a malformed document never costs the rest: its trouble lands
        in :attr:`problems` and the first declaration of an id is kept. The
        same module registered twice (an editable and an installed copy) is
        read once.

        Parameters:
            providers: Providers to add, ahead of the registered ones. A
                declaration module is built live from its declarations, and
                replaces the committed document of an installed provider that
                registers it too (a checkout's working declarations); a
                catalog file (``.json``, a path) is read as it stands (one
                another language's tool writes outside any package).
            hosts: Runner overrides, as for the constructor.
            session: Keep each host's instance for this session, as for the
                constructor.
        """
        # source -> (provider, how it is read: "file", "live" or "committed")
        sources: Dict[str, Tuple[str, str]] = {}
        problems: List[str] = []
        for reference in providers:
            if OperationProvider.is_catalog_file(reference):
                sources[str(Path(reference).resolve())] = ("catalog file", "file")
                continue
            try:
                module = OperationProvider.module_of(reference)
            except ValueError as exc:
                problems.append(str(exc))
                continue
            sources[module] = (module.split(".")[0], "live")
        for ep in PackageManager.entry_points(cls.ENTRY_POINT_GROUP):
            try:
                module = OperationProvider.module_of(ep.value)
            except ValueError as exc:
                problems.append(f"{ep.name}: {exc}")
                continue
            sources.setdefault(module, (ep.name, "committed"))

        entries: Dict[str, Dict[str, Any]] = {}
        for source, (provider, kind) in sources.items():
            try:
                if kind == "file":
                    document = cls._read_document(Path(source))
                elif (
                    kind == "committed"
                    and OperationProvider.catalog_path(source).is_file()
                ):
                    document = cls._read_document(
                        OperationProvider.catalog_path(source)
                    )
                else:
                    document = OperationProvider.document(
                        OperationProvider.declarations(source), provider
                    )
                for entry in cls._entries_of(document):
                    first = entries.setdefault(entry["id"], entry)
                    if first is not entry:
                        problems.append(
                            f"operation {entry['id']!r} is declared by both "
                            f"{first.get('provider')!r} and {entry.get('provider')!r}; "
                            f"{entry.get('provider')!r}'s is ignored"
                        )
            except Exception as exc:
                problems.append(f"{provider} ({source}): {type(exc).__name__}: {exc}")
        for op_id, why in cls._host_led(entries.values()).items():
            problems.append(f"{why}; it is ignored")
            del entries[op_id]
        catalog = cls(entries.values(), hosts, session)
        catalog.problems = problems
        return catalog

    def ids(self) -> List[str]:
        """Every operation id, sorted."""
        return sorted(self._entries)

    def get(self, op_id: str) -> Dict[str, Any]:
        """The entry for *op_id*.

        Raises:
            KeyError: No such operation (the message names the closest ids).
        """
        try:
            return self._entries[op_id]
        except KeyError:
            close = difflib.get_close_matches(op_id, self._entries, n=3)
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            raise KeyError(f"unknown operation {op_id!r}{hint}") from None

    def describe(
        self, op_id: Optional[str] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """One entry (a copy), or every entry sorted by id."""
        if op_id is not None:
            return dict(self.get(op_id))
        return [dict(self._entries[i]) for i in self.ids()]

    def search(self, text: str) -> List[Dict[str, Any]]:
        """The entries whose id, title, summary or arguments name every word of *text*."""
        words = [w for w in text.lower().split() if w]
        found = []
        for op_id in self.ids():
            entry = self._entries[op_id]
            args = entry.get("args", {}).get("properties", {})
            haystack = " ".join(
                [op_id, entry.get("title", ""), entry.get("summary", "")]
                + [f"{k} {v.get('description', '')}" for k, v in args.items()]
            ).lower()
            if all(w in haystack for w in words):
                found.append(dict(entry))
        return found

    # -- running ---------------------------------------------------------------
    def argument_errors(self, op_id: str, args: Mapping[str, Any]) -> List[str]:
        """Every way *args* breaks *op_id*'s argument schema (empty when it fits)."""
        schema = self.get(op_id).get("args") or {}
        properties = schema.get("properties", {})
        defs = schema.get("$defs", {})
        expected = ", ".join(properties) or "none"
        errors = [
            f"unknown argument {name!r} (expected: {expected})"
            for name in args
            if name not in properties
        ]
        errors += [
            f"missing argument {name!r}"
            for name in schema.get("required", ())
            if name not in args
        ]
        for name, value in args.items():
            if name in properties:
                errors += self._schema_errors(properties[name], value, name, defs)
        return errors

    def host(self, name: Optional[str]) -> Optional[OperationHost]:
        """The runner for host *name* (``None``: this process), or ``None`` when
        no installed library registers one. With a :attr:`session`, a
        registered :class:`~pythontk.BatchOperationHost` comes kept: one
        instance for the session (one that serves its run folders itself is
        launched per call still)."""
        with self._hosts_guard:  # one runner per host, however many calls ask at once
            if name not in self._hosts:
                runner: Optional[OperationHost] = None
                if name is None:
                    runner = PureOperationHost()
                else:
                    for ep in PackageManager.entry_points(self.HOST_ENTRY_POINT_GROUP):
                        if ep.name == name:
                            try:
                                runner = ep.load()()
                            except Exception as exc:
                                # A repo folder that hides its package fails as a
                                # bare AttributeError: say why, and the fix.
                                hidden = OperationProvider.import_problem(
                                    ep.value.split(":")[0].split(".")[0]
                                )
                                if not hidden:
                                    raise
                                raise ImportError(
                                    f"{type(exc).__name__}: {exc}. {hidden}"
                                ) from exc
                            break
                    kept = isinstance(runner, BatchOperationHost) and (
                        runner.command is None or runner.serve_args is not None
                    )
                    if kept and self.session is not None:
                        runner = SessionOperationHost(runner, self.session, name)
                self._hosts[name] = runner
            return self._hosts[name]

    def close(self) -> None:
        """End the session instances this process started (:attr:`session`);
        nothing else is closed."""
        for runner in self._hosts.values():
            if isinstance(runner, SessionOperationHost):
                runner.close()

    def run(
        self,
        op_id: str,
        args: Optional[Mapping[str, Any]] = None,
        apply: bool = False,
        scenes: Sequence[Union[str, Path]] = (),
        scope: Optional[CancelScope] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
    ) -> Outcome:
        """Run *op_id* and hand back its :class:`Outcome`; never raises for a
        refused call (the outcome says why).

        An operation that changes anything is a dry run unless *apply*: its
        outcome's ``changes`` is then the plan. One that cannot plan (its
        entry's ``dry_run`` is false) is refused without *apply*.

        Parameters:
            op_id: The operation's id.
            args: Its arguments, by name, as JSON values.
            apply: Make the changes, not just plan them.
            scenes: Scene files the host opens before the call (the first is
                the main one); a live run saves those it changed. Only an
                operation that names its host takes them.
            scope: A cancel scope the run honours.
            on_progress: Called with ``(fraction, message)`` as it goes.
        """
        try:
            entry = self.get(op_id)
        except KeyError as exc:
            return Outcome.refused(op_id, exc.args[0])
        args = dict(args or {})
        dry_run = entry["effect"] != Operation.READ and not apply
        errors = self.argument_errors(op_id, args)
        if errors:
            return Outcome.refused(op_id, "; ".join(errors), dry_run)
        scene_paths = [str(Path(p).resolve()) for p in scenes]
        # A folder too: a host may open one whole (a Unity project).
        missing = [p for p in scene_paths if not Path(p).exists()]
        if scene_paths and not entry.get("host"):
            return Outcome.refused(
                op_id,
                f"{op_id} runs in this process, which opens no scenes: only an "
                "operation that names its host takes scenes",
                dry_run,
            )
        if missing:
            return Outcome.refused(
                op_id, f"no such scene file or folder: {', '.join(missing)}", dry_run
            )
        if dry_run and not entry.get("dry_run"):
            return Outcome.refused(
                op_id, Operation.NO_DRY_RUN.format(op=op_id), dry_run
            )
        try:
            runner = self.host(entry.get("host"))
        except Exception as exc:
            return Outcome.refused(
                op_id,
                f"the runner for host {entry.get('host')!r} failed to load: "
                f"{type(exc).__name__}: {exc}",
                dry_run,
            )
        if runner is None:
            return Outcome.refused(
                op_id,
                f"nothing runs {entry['host']!r} operations here: install the library "
                f"that registers one under the {self.HOST_ENTRY_POINT_GROUP!r} entry points",
                dry_run,
            )
        request = OpRequest(op=op_id, args=args, dry_run=dry_run, scenes=scene_paths)
        return runner.run(entry, request, scope=scope, on_progress=on_progress)

    # -- command line ----------------------------------------------------------
    @classmethod
    def cli(cls, argv: Optional[Sequence[str]] = None) -> int:
        """``python -m pythontk ops <command>``: list, describe, search, run,
        apply, and serve them over MCP. Onboarding and upkeep are operations
        themselves (``operations.init``, ``catalog``, ``check``).

        Returns:
            The exit code: 0 when the command (or the run) succeeded.
        """
        common = [
            (
                "--provider",
                dict(
                    action="append",
                    default=[],
                    metavar="MODULE|FILE",
                    help=(
                        "add a provider: a declaration module, built live from its "
                        "declarations, or a catalog file (.json), read as it stands"
                    ),
                ),
            ),
            ("--json", dict(action="store_true", help="print JSON")),
        ]
        call = [
            ("id", dict(help="the operation id")),
            (
                "pairs",
                dict(
                    nargs="*", metavar="NAME=VALUE", help="an argument (JSON, or text)"
                ),
            ),
            (
                "--args",
                dict(
                    default=None,
                    metavar="JSON",
                    help="the arguments as a JSON object; @FILE reads it from a file, - from stdin",
                ),
            ),
            (
                "--scene",
                dict(
                    action="append",
                    default=[],
                    metavar="PATH",
                    help="a scene file the host opens first (repeat for more); a live run saves it if changed",
                ),
            ),
        ]
        commands = {
            "list": CLI.Command("list every operation", cls._cli_list),
            "describe": CLI.Command(
                "print one operation's entry, its argument schema included",
                cls._cli_describe,
                [("id", dict(help="the operation id"))],
            ),
            "search": CLI.Command(
                "find operations naming every word given",
                cls._cli_search,
                [("words", dict(nargs="+", help="words to match"))],
            ),
            "run": CLI.Command(
                "run an operation; one that changes anything only plans (a dry run)",
                lambda a: cls._cli_run(a, apply=False),
                call,
            ),
            "apply": CLI.Command(
                "run an operation and make its changes",
                lambda a: cls._cli_run(a, apply=True),
                call,
            ),
            "mcp": CLI.Command(
                "serve the operations to an agent over MCP (stdio): reads and dry runs, "
                "or with --apply only the tools that make changes",
                cls._cli_mcp,
                [
                    (
                        "--apply",
                        dict(
                            action="store_true",
                            help="serve the tools that make changes",
                        ),
                    )
                ],
            ),
        }
        return CLI.dispatch(
            "Operations: the named actions every front end calls (python -m pythontk ops).",
            commands,
            argv,
            common=common,
            prog="python -m pythontk ops",
        )
