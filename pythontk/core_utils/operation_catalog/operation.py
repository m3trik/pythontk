# !/usr/bin/python
# coding=utf-8
"""Operation -- one named, typed action that people and agents call alike.

An operation is a plain function (or a static or class method) declared under
an id::

    OPERATIONS = (
        Operation("api.search", "pythontk:ApiSurface.search"),
        Operation("maya.python.exec", "pythontk:PythonConsole.run", effect="code", host="maya"),
    )

The function is the operation: its typed signature is the argument schema, its
Google docstring's summary line and ``Parameters:`` section are the
descriptions, its return annotation the result schema. Arguments arrive as JSON
values (a ``SchemaSpec``-typed argument arrives as the mapping its schema
describes). The declaration adds only what the code cannot say: the id front
ends call it by, its effect (what it may change, and so what undo means) and
the ``host`` it must run in. A provider package lists its declarations in one
module (a tuple named ``OPERATIONS``) and
:class:`~pythontk.OperationCatalog` turns them into the committed catalog every
front end reads. The library code itself never imports any of this: declaring
an operation changes nothing about the function, so a codebase gains or sheds
its operations without an edit to the code they call.

An operation that changes anything takes an optional ``ctx`` keyword (an
:class:`~pythontk.OpContext`) and asks it before each change; that is what
gives it a dry run. One that does not take ``ctx`` has none, and front ends
refuse to dry-run it.

Two payloads are the published forms, declared once here for every reader and
writer, another language's included (CODE_STANDARD section 0, Contracts):
:class:`OperationEntry`, one operation in a catalog, and
:class:`OperationCatalogDocument`, one provider's catalog file.

So an operation written in any language reads alike in one catalog: ids are
``<area>.<action>`` in lower case (``audio.apply``, ``project.sync``), and the
effects are ``read`` (changes nothing; agents run it freely), ``state`` (what
the running application holds: open scenes or documents and their objects; one
undo step), ``files`` (assets, exports, settings; a dry run lists every
file first, version control is the way back) and ``code`` (runs code: it may
change anything, and has no plan and no undo, so only the change server runs
it).
"""

from __future__ import annotations

import collections.abc
import importlib
import inspect
import json
import re
import types
import typing
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from pythontk.core_utils.api_surface.api_surface import ApiSurface
from pythontk.core_utils.operation_catalog.op_context import OpContext
from pythontk.core_utils.schema_spec import SchemaSpec

_field = SchemaSpec.spec_field
#: The forms a union annotation takes (``Union[a, b]``; ``a | b`` from 3.10).
_UNIONS = tuple(o for o in (typing.Union, getattr(types, "UnionType", None)) if o)


@dataclass
class OperationEntry(SchemaSpec):
    """One operation as a catalog lists it: what every front end reads."""

    TYPED = True

    #: The effects, widest change last.
    EFFECTS = ("read", "state", "files", "code")

    id: str = _field(
        help=(
            "'<area>.<action>' in lower case, led by its host when it names one "
            "('maya.mesh.mirror'): what every front end calls it by."
        ),
        required=True,
        example="mesh.mirror",
    )
    title: str = _field(
        help="The label a menu or palette shows.",
        required=True,
        example="Mirror the selected meshes",
    )
    summary: str = _field(
        help="One line on what it does.",
        example="Mirror the selected meshes across an axis.",
    )
    effect: str = _field(
        help=(
            "'read' (changes nothing), 'state' (what the running application "
            "holds: open scenes or documents; one undo step), 'files' (a dry "
            "run lists them first) or 'code' (runs code: may change anything; "
            "no plan, no undo)."
        ),
        required=True,
        choices=EFFECTS,
        example="state",
    )
    host: str = _field(
        help="The application it runs in ('maya', 'blender', 'unity'); absent: any Python.",
        example="maya",
    )
    target: str = _field(
        help="The function it calls, 'module:qualified.name' (Python providers).",
        example="mayatk:EditUtils.mirror",
    )
    dry_run: bool = _field(
        help="Whether it can plan its changes without making them.",
        default=False,
        example=True,
    )
    args: Dict[str, Any] = _field(
        help="Its arguments, as a JSON Schema object; an unknown argument is refused.",
        required=True,
        default_factory=dict,
        example={"type": "object", "properties": {}, "additionalProperties": False},
    )
    result: Dict[str, Any] = _field(
        help="Its result, as a JSON Schema; absent when it returns nothing.",
        example={"type": "object"},
    )


@dataclass
class OperationCatalogDocument(SchemaSpec):
    """One provider's catalog file: what a provider writes and every front end reads."""

    TYPED = True

    #: The format this pythontk writes; a reader refuses a newer one.
    FORMAT = 1

    format: int = _field(
        help="The document's format version.",
        required=True,
        example=1,
    )
    provider: str = _field(
        help="The package (or project) that declares these operations.",
        required=True,
        example="mayatk",
    )
    operations: List[OperationEntry] = _field(
        help="Its operations, sorted by id.",
        required=True,
        nested=[OperationEntry],
        default_factory=list,
    )


# A Google docstring entry: "name: text" or "name (type): text".
_PARAM_LINE = re.compile(r"^\*{0,2}(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
_PARAM_HEADERS = ("Parameters:", "Args:", "Arguments:")


class _OperationInternal:
    """Internal helpers for :class:`Operation`."""

    @staticmethod
    def _parameter_docs(doc: Optional[str]) -> Dict[str, str]:
        """``{name: description}`` from a Google docstring's parameter section.

        Continuation lines (indented past the entry) join their entry; the
        section ends at the next unindented line (``Returns:``).
        """
        out: Dict[str, str] = {}
        current: Optional[str] = None
        entry_indent: Optional[int] = None
        in_section = False
        for line in inspect.cleandoc(doc or "").splitlines():
            stripped = line.strip()
            if stripped in _PARAM_HEADERS:
                in_section, current, entry_indent = True, None, None
                continue
            if not in_section or not stripped:
                continue
            indent = len(line) - len(line.lstrip())
            if indent == 0:
                in_section, current = False, None
                continue
            match = _PARAM_LINE.match(stripped)
            if match and (entry_indent is None or indent <= entry_indent):
                entry_indent = indent
                current = match.group(1)
                out[current] = match.group(2).strip()
            elif current is not None:
                out[current] = f"{out[current]} {stripped}".strip()
        return {name: ApiSurface.markdown(text) for name, text in out.items()}

    @staticmethod
    def _type_hints(op_id: str, fn: Callable) -> Dict[str, Any]:
        """The resolved annotations of *fn*.

        Raises:
            ValueError: One does not resolve at runtime (a name imported only
                under ``TYPE_CHECKING``, ``int | None`` on Python 3.9): a
                schema read from the raw strings would accept anything, so the
                catalog refuses it instead.
        """
        try:
            return typing.get_type_hints(fn)
        except Exception as exc:
            raise ValueError(
                f"operation {op_id!r}: an annotation of "
                f"{getattr(fn, '__qualname__', fn)} does not resolve at runtime "
                f"({type(exc).__name__}: {exc}); import its type at module level, "
                "or use a spelling every supported Python reads"
            ) from exc

    @staticmethod
    def _allows_null(schema: Dict[str, Any]) -> bool:
        """Whether *schema* admits ``null`` (no constraint at all does)."""
        if not schema or schema.get("type") == "null":
            return True
        return any(option.get("type") == "null" for option in schema.get("anyOf", ()))

    @staticmethod
    def _or_null(schema: Dict[str, Any]) -> Dict[str, Any]:
        """*schema* or ``null``, flat: ``Union[int, str]`` with a ``None``
        default reads as ``anyOf[int, str, null]``, as Python before 3.11
        spells it (``Optional[Union[int, str]]`` flattens)."""
        if set(schema) == {"anyOf"}:
            return {"anyOf": list(schema["anyOf"]) + [{"type": "null"}]}
        return {"anyOf": [schema, {"type": "null"}]}

    @staticmethod
    def _json_default(value: Any) -> Any:
        """*value* when it is strict JSON as it stands (no NaN or Infinity,
        which a C# reader refuses), else a sentinel meaning "omit"."""
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            return inspect.Parameter.empty
        return value

    @staticmethod
    def _takes_code(tp: Any) -> bool:
        """Whether *tp* (through ``Optional``) is a callable: code, which no
        caller can send as JSON."""
        if typing.get_origin(tp) in _UNIONS:
            options = [a for a in typing.get_args(tp) if a is not type(None)]
        else:
            options = [tp]
        return bool(options) and all(
            o is callable
            or o is collections.abc.Callable
            or typing.get_origin(o) is collections.abc.Callable
            for o in options
        )

    @staticmethod
    def _schema_of(tp: Any, defs: Dict[str, Any]) -> Dict[str, Any]:
        """*tp*'s JSON Schema, its ``$defs`` merged into *defs* (one per name)."""
        schema = SchemaSpec.annotation_schema(tp)
        defs.update(schema.pop("$defs", {}))
        return schema

    @classmethod
    def _argument_schema(
        cls, op_id: str, target: str, fn: Callable, doc: str
    ) -> Dict[str, Any]:
        """*fn*'s arguments as a JSON Schema object: each parameter but its
        ``ctx`` and the code hooks (a callable with a default, which the call
        keeps)."""
        try:
            parameters = inspect.signature(fn).parameters
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"operation {op_id!r}: {target} has no signature to read ({exc})"
            ) from exc
        if next(iter(parameters), None) == "self":
            raise ValueError(
                f"operation {op_id!r}: {target} is an instance method; an operation "
                "calls a function, a static method or a class method"
            )
        hints = cls._type_hints(op_id, fn)
        docs = cls._parameter_docs(doc)
        named = [
            name
            for name, param in parameters.items()
            if name != Operation.CONTEXT_PARAMETER
            and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        ]
        defs: Dict[str, Any] = {}
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for name in named:
            param = parameters[name]
            if param.kind is param.POSITIONAL_ONLY:
                raise ValueError(
                    f"operation {op_id!r}: parameter {name!r} is positional-only; "
                    "an operation is called with keyword arguments"
                )
            if cls._takes_code(hints.get(name)):
                if param.default is param.empty:
                    raise ValueError(
                        f"operation {op_id!r}: {target} takes code as {name!r} (a callable), "
                        "which no caller can send; give it a default"
                    )
                continue
            schema = cls._schema_of(hints.get(name, Any), defs)
            if param.default is None and not cls._allows_null(schema):
                # ``x: str = None``: Python before 3.11 reads the hint as
                # Optional[str], later ones as str. Spell the null here so the
                # catalog is the same document under every interpreter.
                schema = cls._or_null(schema)
            if docs.get(name):
                schema["description"] = docs[name]
            if param.default is param.empty:
                required.append(name)
            else:
                default = cls._json_default(param.default)
                if default is not inspect.Parameter.empty:
                    schema["default"] = default
            properties[name] = schema
        schema_doc: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            schema_doc["required"] = required
        if defs:
            schema_doc["$defs"] = defs
        return schema_doc

    @classmethod
    def _result_schema(cls, op_id: str, fn: Callable) -> Optional[Dict[str, Any]]:
        hints = cls._type_hints(op_id, fn)
        if "return" not in hints or hints["return"] is type(None):
            return None
        defs: Dict[str, Any] = {}
        schema = cls._schema_of(hints["return"], defs)
        if defs:
            schema["$defs"] = defs
        return schema or None


@dataclass(frozen=True)
class Operation(_OperationInternal):
    """One operation's declaration (see the module docstring).

    Parameters:
        id: ``<area>.<action>`` in lower case, dot-separated (``mesh.mirror``),
            led by the host when it names one (``maya.mesh.mirror``; see
            :meth:`id_error`). Saved data and agent prompts name it, so it is
            published the day it ships (CODE_STANDARD section 5).
        target: The function, as ``"module:qualified.name"``; the module is
            imported and the name walked with ``getattr``
            (``"mayatk:EditUtils.mirror"`` reaches the method through the
            package root).
        effect: :attr:`READ`, :attr:`STATE`, :attr:`FILES` or :attr:`CODE`:
            the widest change it may make.
        title: The label a menu or palette shows; defaults to the summary.
        host: The application it must run in (``"maya"``, ``"blender"``,
            ``"unity"``), in an instance the tooling owns; ``None`` runs in
            any Python process.

    Raises:
        ValueError: The id, effect or target is malformed.
    """

    id: str
    target: str
    effect: str = "read"
    title: str = ""
    host: Optional[str] = None

    #: Changes nothing: agents run it freely.
    READ = "read"
    #: Changes what the running application holds (open scenes or documents
    #: and their objects): one undo step.
    STATE = "state"
    #: Writes files (assets, exports, settings): a dry run lists them first.
    FILES = "files"
    #: Runs code it is given: may change anything, with no plan and no undo
    #: (a console's ``exec``), so it takes no context and only an apply runs it.
    CODE = "code"
    EFFECTS = OperationEntry.EFFECTS
    #: The keyword a function takes its :class:`OpContext` by.
    CONTEXT_PARAMETER = "ctx"
    #: Why a dry run of a change whose function takes no ``ctx`` is refused.
    NO_DRY_RUN = (
        "{op} cannot plan its changes (its function takes no ctx), so it has no "
        "dry run: apply it to run it"
    )
    ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
    #: The longest id: Claude caps a tool name at 64 characters, and a client
    #: names a plugin's MCP tool ``mcp__plugin_<plugin>_<server>__<tool>``
    #: (``mcp__plugin_m3trik_ops-apply__`` is 30; a project server's
    #: ``mcp__ops-apply__`` only 16).
    ID_MAX = 34

    def __post_init__(self):
        problem = self.id_error(self.id, self.host)
        if problem:
            raise ValueError(problem)
        if self.effect not in self.EFFECTS:
            raise ValueError(
                f"operation {self.id!r}: effect {self.effect!r} is not one of "
                f"{list(self.EFFECTS)}"
            )
        module, _, name = (self.target or "").partition(":")
        if not module or not name:
            raise ValueError(
                f"operation {self.id!r}: target {self.target!r} is not "
                "'module:qualified.name'"
            )

    @classmethod
    def id_error(cls, op_id: str, host: Optional[str] = None) -> Optional[str]:
        """Why *op_id* is not a valid id for an operation *host* runs, or ``None``.

        An id is ``<area>.<action>`` in lower case, at most :attr:`ID_MAX`
        characters. One a host application runs starts with that host
        (``maya.scene.summary``, ``blender.scene.summary``): twin libraries
        declare the same action for their own host, and the id names where it
        runs, so every id stays unique and stable when a twin is added.
        """
        if not cls.ID_PATTERN.match(op_id or ""):
            return (
                f"operation id {op_id!r} is not '<area>.<action>' in lower case "
                "(letters, digits and '_', dot-separated)"
            )
        if len(op_id) > cls.ID_MAX:
            return f"operation id {op_id!r} is longer than {cls.ID_MAX} characters"
        if host and not op_id.startswith(f"{host}."):
            return f"operation id {op_id!r} runs in {host!r}, so it must start with '{host}.'"
        return None

    def resolve(self) -> Callable:
        """The function this operation calls (imports its module).

        Raises:
            ImportError: The module does not import.
            AttributeError: The name is not in it.
        """
        return self.resolve_target(self.target)

    @staticmethod
    def resolve_target(target: str) -> Callable:
        """The callable a ``"module:qualified.name"`` target names."""
        module_name, _, qualname = target.partition(":")
        obj: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return obj

    @classmethod
    def takes_context(cls, fn: Callable) -> bool:
        """Whether *fn* takes an :class:`OpContext`: a ``ctx`` keyword left
        unannotated (the convention) or annotated as one. A ``ctx`` of another
        kind (an exporter's own context) keeps its default, so the function
        has no dry run."""
        try:
            param = inspect.signature(fn).parameters.get(cls.CONTEXT_PARAMETER)
        except (TypeError, ValueError):
            return False
        if param is None or param.annotation is param.empty:
            return param is not None
        try:
            hint = typing.get_type_hints(fn).get(cls.CONTEXT_PARAMETER)
        except Exception:
            return False
        if typing.get_origin(hint) in _UNIONS:
            options = [a for a in typing.get_args(hint) if a is not type(None)]
        else:
            options = [hint]
        return all(isinstance(o, type) and issubclass(o, OpContext) for o in options)

    def entry(self) -> OperationEntry:
        """This operation's catalog entry: the declaration plus what the
        function says about itself (imports the target)."""
        fn = self.resolve()
        doc = inspect.getdoc(fn) or ""
        summary = ApiSurface.summary(doc)
        return OperationEntry(
            id=self.id,
            title=self.title or summary.rstrip(".") or self.id,
            summary=summary,
            effect=self.effect,
            host=self.host,
            target=self.target,
            dry_run=self.effect == self.READ or self.takes_context(fn),
            args=self._argument_schema(self.id, self.target, fn, doc),
            result=self._result_schema(self.id, fn),
        )

    def describe(self) -> Dict[str, Any]:
        """:meth:`entry` as the JSON mapping a catalog stores."""
        return self.entry().to_dict()
