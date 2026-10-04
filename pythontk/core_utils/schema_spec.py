# !/usr/bin/python
# coding=utf-8
"""Declarative schema for JSON/YAML *template* files, defined as a dataclass.

A :class:`SchemaSpec` subclass is a normal ``@dataclass`` whose fields carry a
little extra metadata — help text, an example value, whether the key is
required, an optional nested sub-schema, allowed ``choices``, and an optional
``validate`` callable — supplied through :func:`spec_field`.  From that single
definition the base derives, with no extra per-schema code:

* :meth:`validate` — structural validation returning *errors* and *warnings*
  (unknown-but-harmless keys are warnings, not errors — see the module note),
* :meth:`skeleton` — a fully-populated example ``dict`` a user can model a new
  file after,
* :meth:`describe` / :meth:`to_markdown` — human-readable reference docs,
* :meth:`json_schema` — the same shape as a JSON Schema document, the form a
  reader in another language generates its types from.

This is the storage-agnostic *shape* SSoT.  Pair it with a
:class:`~pythontk.core_utils.presets.store.PresetStore` (any codec — JSON or
YAML) through :class:`~pythontk.core_utils.template_set.TemplateSet` to get a
discoverable, user-extensible collection of template files whose schema is
documented and enforced from one place.

The second use is a *payload* shape: a JSON value one program writes and
programs in other languages read (a scene record's payload, a manifest a
glTF runtime binds).  There the field annotations ARE the contract, so such a
spec sets :attr:`SchemaSpec.TYPED` and :meth:`validate` also checks every
present value against its annotation; a template spec leaves it off, because
its annotations describe the parsed object (a ``Tuple`` built from a list)
rather than the file.

Design note — errors vs. warnings
    :meth:`validate` deliberately splits *errors* (unknown method, wrong
    structure, missing required key) from *warnings* (an unrecognised
    top-level key).  Callers raise on errors but tolerate warnings, so a file
    that merely carries an extra annotation key keeps working.  Keys beginning
    with ``_`` (e.g. ``_meta``, ``_comment``) are reserved for annotations and
    never warned about — that is how a generated skeleton can embed help text
    while remaining valid.
"""

from __future__ import annotations

import collections.abc
import copy
import dataclasses
import enum
import logging
import types
import typing
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Type

# Metadata namespace key — keeps schema metadata from colliding with any other
# library that reads ``dataclasses.field(metadata=...)``.
_META = "schema_spec"

# Sentinel: "no explicit example given — derive one from the field default or a
# nested schema." Distinct from ``None``, which is a legitimate example value.
MISSING = dataclasses.MISSING

# ``int | None`` (Python 3.10+) has its own origin; ``Optional[int]`` is Union.
_UNION_ORIGINS = tuple(
    o for o in (typing.Union, getattr(types, "UnionType", None)) if o is not None
)
# Annotation origins read as a JSON array / a JSON object.
_ARRAY_ORIGINS = (
    list,
    set,
    frozenset,
    collections.abc.Sequence,
    collections.abc.MutableSequence,
    collections.abc.Set,
    collections.abc.Collection,
    collections.abc.Iterable,
)
_OBJECT_ORIGINS = (dict, collections.abc.Mapping, collections.abc.MutableMapping)
# The JSON Schema keyword for each scalar annotation -- bool BEFORE int, which
# it subclasses.
_SCALARS = (("boolean", bool), ("integer", int), ("number", float), ("string", str))


class SchemaError(ValueError):
    """Raised by :meth:`ValidationResult.raise_if_errors` when a file is invalid."""


@dataclass
class FieldDoc:
    """One row of a schema's generated reference."""

    name: str
    help: str
    required: bool
    example: Any
    choices: Optional[Sequence[Any]]
    nested: Optional[Type["SchemaSpec"]]
    nested_is_list: bool = False


@dataclass
class ValidationResult:
    """Outcome of :meth:`SchemaSpec.validate` — separated errors and warnings."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """``True`` when there are no errors (warnings are tolerable)."""
        return not self.errors

    def raise_if_errors(self, prefix: str = "") -> None:
        """Raise :class:`SchemaError` joining all errors, or do nothing."""
        if self.errors:
            raise SchemaError(prefix + "; ".join(self.errors))

    def raise_or_warn(
        self,
        *,
        prefix: str = "",
        logger: Optional[logging.Logger] = None,
        strict: bool = False,
    ) -> None:
        """Enforce a validated file: raise on errors, log (or, if *strict*, raise on) warnings.

        The one-call helper every template loader shares — raises
        :class:`SchemaError` (messages joined, *prefix*-tagged) when there are
        errors, plus warnings when *strict*; otherwise emits each warning via
        *logger* (when given) so a tolerable file still loads.
        """
        problems = list(self.errors)
        if strict:
            problems.extend(self.warnings)
        elif logger is not None:
            for w in self.warnings:
                logger.warning("%s%s", prefix, w)
        if problems:
            raise SchemaError(prefix + "; ".join(problems))

    def merge(self, other: "ValidationResult", path: str = "") -> None:
        """Fold *other* in, prefixing each message with *path* (e.g. ``"columns."``)."""
        self.errors.extend(f"{path}{e}" for e in other.errors)
        self.warnings.extend(f"{path}{w}" for w in other.warnings)


class _SchemaSpecInternal(object):
    """Internal helpers for SchemaSpec."""

    @staticmethod
    def _default_for(cls: Type["SchemaSpec"], name: str) -> Any:
        """The dataclass default (value or factory result) for field *name*."""
        for f in fields(cls):
            if f.name == name:
                if f.default is not MISSING:
                    return f.default
                if f.default_factory is not MISSING:  # type: ignore[misc]
                    return f.default_factory()
                return None
        return None

    @staticmethod
    def _compact(value: Any, limit: int = 60) -> str:
        """One-line, length-capped repr of an example value for a doc table cell."""
        import json

        try:
            text = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
        text = text.replace("\n", " ")
        return text if len(text) <= limit else text[: limit - 1] + "…"

    @staticmethod
    def _nested_of(m: dict) -> "tuple[Optional[Type['SchemaSpec']], bool]":
        """Resolve a spec field's ``nested`` metadata to ``(schema, is_list)``.

        A bare ``SchemaSpec`` subclass is a single nested mapping; a one-element
        ``[SchemaSpec]`` (mirroring ``typing.List[X]``) is a list of that schema.
        ``(None, False)`` when the field carries no nested schema.
        """
        nested = m["nested"]
        if isinstance(nested, (list, tuple)):
            if len(nested) != 1:
                raise SchemaError(
                    "nested list must hold exactly one SchemaSpec subclass"
                )
            return nested[0], True
        return nested, False

    @staticmethod
    def _doc_lines(spec: type) -> List[str]:
        """*spec*'s docstring lines -- none for the constructor signature
        ``@dataclass`` synthesises as ``__doc__`` when a subclass has no real
        docstring, which would leak field types / ``<factory>`` / ``=None``
        noise into a generated reference."""
        doc = spec.__doc__ or ""
        if doc.strip().startswith(spec.__name__ + "("):
            return []
        return doc.strip().splitlines()

    @classmethod
    def _summary(cls, spec: type) -> str:
        """The first paragraph of *spec*'s docstring, on one line."""
        lines: List[str] = []
        for line in cls._doc_lines(spec):
            if not line.strip():
                break
            lines.append(line.strip())
        return " ".join(lines)

    @staticmethod
    def _hints(spec: type) -> Dict[str, Any]:
        """``{field: annotation}`` for *spec*'s fields, string annotations (a
        module under ``from __future__ import annotations``) resolved. One that
        cannot be resolved -- a name local to a function -- reads as ``Any``."""
        try:
            hints = typing.get_type_hints(spec)
        except Exception:  # noqa: BLE001 -- NameError, or an exotic annotation
            hints = {}
        resolved: Dict[str, Any] = {}
        for f in fields(spec):
            tp = hints.get(f.name, f.type)
            resolved[f.name] = Any if isinstance(tp, str) else tp
        return resolved

    @staticmethod
    def _json_kind(value: Any) -> str:
        """The JSON type *value* is, for a message."""
        if value is None:
            return "null"
        for kind, py in _SCALARS:
            if isinstance(value, py):
                return kind
        if isinstance(value, Mapping):
            return "object"
        if isinstance(value, (list, tuple)):
            return "array"
        return type(value).__name__

    @classmethod
    def _type_schema(cls, tp: Any, defs: Dict[str, Any]) -> Dict[str, Any]:
        """The JSON Schema of the annotation *tp*. A :class:`SchemaSpec` it
        names is defined once in *defs* (by class name) and referenced."""
        if tp is Any or tp is object:
            return {}
        if tp is type(None):
            return {"type": "null"}
        if isinstance(tp, type) and issubclass(tp, SchemaSpec):
            return {"$ref": cls._define(tp, defs)}
        origin, args = typing.get_origin(tp), typing.get_args(tp)
        if origin in _UNION_ORIGINS:
            return {"anyOf": [cls._type_schema(a, defs) for a in args]}
        if origin is typing.Literal or (
            isinstance(tp, type) and issubclass(tp, enum.Enum)
        ):
            values = list(args) if origin is typing.Literal else [m.value for m in tp]
            kinds = {cls._json_kind(v) for v in values}
            return (
                {"type": kinds.pop(), "enum": values}
                if len(kinds) == 1
                else {"enum": values}
            )
        for kind, py in _SCALARS:
            if tp is py:
                return {"type": kind}
        if origin is tuple or tp is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                return cls._array_schema(args[0], defs)
            if not args:
                return {"type": "array"}
            items = [cls._type_schema(a, defs) for a in args]
            return {
                "type": "array",
                "prefixItems": items,
                "minItems": len(items),
                "maxItems": len(items),
            }
        if origin in _ARRAY_ORIGINS or tp in (list, set, frozenset):
            return cls._array_schema(args[0] if args else Any, defs)
        if origin in _OBJECT_ORIGINS or tp is dict:
            schema: Dict[str, Any] = {"type": "object"}
            values = cls._type_schema(args[1] if len(args) == 2 else Any, defs)
            if values:
                schema["additionalProperties"] = values
            return schema
        return {}

    @classmethod
    def _array_schema(cls, item: Any, defs: Dict[str, Any]) -> Dict[str, Any]:
        schema: Dict[str, Any] = {"type": "array"}
        items = cls._type_schema(item, defs)
        if items:
            schema["items"] = items
        return schema

    @classmethod
    def _define(cls, spec: type, defs: Dict[str, Any]) -> str:
        """Define *spec* in *defs* (once) and return its ``$ref``.

        Raises:
            SchemaError: Another spec of the same class name is already there
                -- two would share one ``$ref`` and one generated type.
        """
        name = spec.__name__
        owner = defs.get(name, {}).get("x-spec")
        if owner is not None and owner is not spec:
            raise SchemaError(
                f"two schemas named {name!r} in one document: "
                f"{owner.__module__} and {spec.__module__}"
            )
        if name not in defs:
            # Claimed BEFORE it is built: a spec nesting itself refers back to
            # the claim instead of recursing forever.
            defs[name] = {"x-spec": spec}
            defs[name].update(cls._object_schema(spec, defs))
        return f"#/$defs/{name}"

    @classmethod
    def _object_schema(cls, spec: type, defs: Dict[str, Any]) -> Dict[str, Any]:
        """*spec* as a JSON Schema object: its fields' types from their
        annotations (a ``nested=`` schema from its metadata), ``help`` as each
        description, ``choices`` as each enum, the required keys.

        Raises:
            SchemaError: A typed spec whose annotation and ``nested=`` name
                different schemas -- one declaration may not say two things.
        """
        hints = cls._hints(spec)
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for name, m in spec._specs().items():
            hinted = cls._type_schema(hints.get(name, Any), defs)
            nested, is_list = cls._nested_of(m)
            if nested is not None:
                ref = {"$ref": cls._define(nested, defs)}
                fragment = {"type": "array", "items": ref} if is_list else ref
                if spec.TYPED and hinted != fragment:
                    raise SchemaError(
                        f"{spec.__name__}.{name}: its annotation and nested= "
                        "name different schemas"
                    )
            else:
                fragment = dict(hinted)
            if m["choices"] is not None:
                fragment["enum"] = list(m["choices"])
            if m["help"]:
                fragment["description"] = m["help"]
            properties[name] = fragment
            if m["required"]:
                required.append(name)
        schema: Dict[str, Any] = {"type": "object"}
        summary = cls._summary(spec)
        if summary:
            schema["description"] = summary
        schema["properties"] = properties
        if required:
            schema["required"] = required
        return schema

    @staticmethod
    def _strip_specs(node: Any) -> Any:
        """*node* without the ``x-spec`` class claims :meth:`_define` leaves."""
        if isinstance(node, dict):
            return {
                k: _SchemaSpecInternal._strip_specs(v)
                for k, v in node.items()
                if k != "x-spec"
            }
        if isinstance(node, list):
            return [_SchemaSpecInternal._strip_specs(v) for v in node]
        return node

    @classmethod
    def _type_errors(
        cls, tp: Any, value: Any, path: str, res: "ValidationResult"
    ) -> None:
        """Every way *value* breaks the annotation *tp*, into *res*, each
        located at *path* (``joints[0].t[2]``). An annotation this cannot read
        checks nothing."""
        if tp is Any or tp is object:
            return
        if tp is type(None):
            if value is not None:
                res.errors.append(f"{path}: expected null, got {cls._json_kind(value)}")
            return
        if isinstance(tp, type) and issubclass(tp, SchemaSpec):
            res.merge(tp.validate(value), path=f"{path}.")
            return
        origin, args = typing.get_origin(tp), typing.get_args(tp)
        if origin in _UNION_ORIGINS:
            # A null meets the Optional; anything else answers to the other
            # option(s), whose own message says what was wrong.
            options = (
                [a for a in args if a is not type(None)] if value is not None else args
            )
            for option in options:
                probe = ValidationResult()
                cls._type_errors(option, value, path, probe)
                if not probe.errors:
                    res.warnings.extend(probe.warnings)
                    return
            if len(options) == 1:
                cls._type_errors(options[0], value, path, res)
            else:
                res.errors.append(
                    f"{path}: {cls._json_kind(value)} matches none of its types"
                )
            return
        if origin is typing.Literal or (
            isinstance(tp, type) and issubclass(tp, enum.Enum)
        ):
            values = list(args) if origin is typing.Literal else [m.value for m in tp]
            # A bool is never the int it equals (True == 1), either way round.
            if not any(
                v == value and isinstance(v, bool) == isinstance(value, bool)
                for v in values
            ):
                res.errors.append(f"{path}: {value!r} is not one of {values}")
            return
        for kind, py in _SCALARS:
            if tp is py:
                ok = isinstance(value, (int, float) if py is float else py)
                if py is not bool and isinstance(value, bool):
                    ok = False
                if not ok:
                    res.errors.append(
                        f"{path}: expected {kind}, got {cls._json_kind(value)}"
                    )
                return
        fixed = (origin is tuple or tp is tuple) and args and args[-1] is not Ellipsis
        if (
            origin is tuple
            or tp is tuple
            or origin in _ARRAY_ORIGINS
            or tp in (list, set, frozenset)
        ):
            if not isinstance(value, (list, tuple)):
                res.errors.append(
                    f"{path}: expected array, got {cls._json_kind(value)}"
                )
                return
            if fixed and len(value) != len(args):
                res.errors.append(
                    f"{path}: expected {len(args)} items, got {len(value)}"
                )
                return
            for i, item in enumerate(value):
                item_tp = args[i] if fixed else (args[0] if args else Any)
                cls._type_errors(item_tp, item, f"{path}[{i}]", res)
            return
        if origin in _OBJECT_ORIGINS or tp is dict:
            if not isinstance(value, Mapping):
                res.errors.append(
                    f"{path}: expected object, got {cls._json_kind(value)}"
                )
                return
            value_tp = args[1] if len(args) == 2 else Any
            for key, item in value.items():
                if not isinstance(key, str):
                    res.errors.append(f"{path}: key {key!r} is not a string")
                cls._type_errors(value_tp, item, f"{path}.{key}", res)


@dataclass
class SchemaSpec(_SchemaSpecInternal):
    """Base for declarative template schemas (see module docstring).

    Subclass with ``@dataclass`` and declare fields via :func:`spec_field`.
    Override :meth:`from_dict` / :meth:`to_dict` for (de)serialisation of
    irregular shapes; the validate/skeleton/docs machinery is inherited.
    """

    #: Whether :meth:`validate` also holds every present value to its field's
    #: annotation (``Optional[float]``, ``Tuple[float, float, float]``,
    #: ``List[OtherSpec]``, ...). On for a payload shape, where the annotation
    #: is the contract; off for a template, whose annotations describe the
    #: parsed object. Unannotated, so it is a class setting, not a field.
    TYPED = False
    #: The dialect :meth:`json_schema` writes.
    JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

    # -- introspection ----------------------------------------------------
    @classmethod
    def _specs(cls) -> Dict[str, dict]:
        """``{field_name: schema-metadata}`` for every spec field, in order."""
        out: Dict[str, dict] = {}
        for f in fields(cls):
            m = f.metadata.get(_META)
            if m is not None:
                out[f.name] = m
        return out

    # -- (de)serialisation ------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaSpec":
        """Build an instance from a raw ``dict``, recursing into nested schemas.

        Unknown keys are ignored (they surface as warnings in :meth:`validate`).
        Subclasses with irregular coercion (e.g. list→tuple) override this.
        """
        kwargs: Dict[str, Any] = {}
        for name, m in cls._specs().items():
            if name not in data:
                continue
            value = data[name]
            schema, is_list = cls._nested_of(m)
            # Deep-copy plain values so the instance never aliases the source
            # dict (a later mutation of one must not silently rewrite the other).
            # Nested schemas recurse and copy on their own.
            if schema is not None and is_list and isinstance(value, list):
                kwargs[name] = [
                    schema.from_dict(v) if isinstance(v, dict) else copy.deepcopy(v)
                    for v in value
                ]
            elif schema is not None and not is_list and isinstance(value, dict):
                kwargs[name] = schema.from_dict(value)
            else:
                kwargs[name] = copy.deepcopy(value)
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON/YAML-safe ``dict``; unset optionals are omitted."""
        out: Dict[str, Any] = {}
        for name in type(self)._specs():
            value = getattr(self, name)
            if value is None:
                continue
            # Deep-copy so a caller mutating the serialised output can't reach
            # back into the live instance (nested schemas serialise + copy below).
            if isinstance(value, SchemaSpec):
                out[name] = value.to_dict()
            elif isinstance(value, list):
                out[name] = [
                    v.to_dict() if isinstance(v, SchemaSpec) else copy.deepcopy(v)
                    for v in value
                ]
            else:
                out[name] = copy.deepcopy(value)
        return out

    # -- validation -------------------------------------------------------
    @classmethod
    def validate(cls, data: Any) -> ValidationResult:
        """Validate a raw ``dict`` against this schema.

        Errors: not-a-mapping, missing required key, bad ``choices`` value,
        nested-schema errors, anything a field's ``validate`` callable
        reports, and -- on a :attr:`TYPED` schema -- a value its annotation
        does not admit.  Warnings: unrecognised top-level keys
        (``_``-prefixed keys are reserved for annotations and skipped).
        """
        res = ValidationResult()
        if not isinstance(data, dict):
            res.errors.append(f"expected a mapping, got {type(data).__name__}")
            return res

        specs = cls._specs()
        hints = cls._hints(cls) if cls.TYPED else {}
        for key in data:
            if key not in specs and not str(key).startswith("_"):
                res.warnings.append(f"unknown key {key!r} (ignored)")

        for name, m in specs.items():
            if name not in data:
                if m["required"]:
                    res.errors.append(f"missing required key {name!r}")
                continue
            value = data[name]

            if m["choices"] is not None and value not in m["choices"]:
                res.errors.append(
                    f"{name}: {value!r} is not one of {list(m['choices'])}"
                )

            schema, is_list = cls._nested_of(m)
            if schema is not None:
                if is_list:
                    if isinstance(value, list):
                        for i, item in enumerate(value):
                            res.merge(schema.validate(item), path=f"{name}[{i}].")
                    else:
                        res.errors.append(
                            f"{name}: expected a list for nested-list schema"
                        )
                elif isinstance(value, dict):
                    res.merge(schema.validate(value), path=f"{name}.")
                else:
                    res.errors.append(f"{name}: expected a mapping for nested schema")
            elif cls.TYPED:
                cls._type_errors(hints.get(name, Any), value, name, res)

            validator = m["validate"]
            if validator is not None:
                try:
                    res.errors.extend(f"{name}: {e}" for e in (validator(value) or []))
                except Exception as exc:  # a buggy validator must not crash a load
                    res.errors.append(f"{name}: validator raised {exc!r}")
        return res

    # -- generation -------------------------------------------------------
    @classmethod
    def skeleton(cls) -> Dict[str, Any]:
        """A fully-populated example ``dict`` to model a new file after.

        Each key uses its ``example`` if given, else a nested schema's skeleton,
        else a valid ``choices`` value, else the field's dataclass default.  The
        result is guaranteed to pass this schema's own :meth:`validate` (a
        ``choices`` key carries an allowed value; a required key is never
        dropped), so it is always a safe copy-me template.
        """
        out: Dict[str, Any] = {}
        for name, m in cls._specs().items():
            choices = m["choices"]
            if m["example"] is not MISSING:
                # Deep-copy: the example lives in shared class metadata, so a
                # caller mutating a skeleton must not poison it (or later skeletons).
                value = copy.deepcopy(m["example"])
            elif m["nested"] is not None:
                schema, is_list = cls._nested_of(m)
                value = [schema.skeleton()] if is_list else schema.skeleton()
            elif choices:
                value = choices[0]
            else:
                value = _SchemaSpecInternal._default_for(cls, name)
            # A ``choices`` field must carry an allowed value or the skeleton
            # fails its own validate(); fall back to the first valid choice.
            if choices and value not in choices:
                value = choices[0]
            if value is None:
                # A required key must appear (an omitted one fails validate());
                # surface it with an empty placeholder. An unset *optional*
                # (no example, ``None`` default) is omitted — a ``null`` in a
                # copy-me skeleton reads as a real value, which it usually isn't.
                if not m["required"]:
                    continue
                value = ""
            out[name] = value
        return out

    @classmethod
    def describe(cls) -> List[FieldDoc]:
        """Structured field-by-field reference (powers :meth:`to_markdown`)."""
        docs: List[FieldDoc] = []
        for f in fields(cls):
            m = f.metadata.get(_META)
            if m is None:
                continue
            example = m["example"]
            if example is MISSING:
                example = _SchemaSpecInternal._default_for(cls, f.name)
            schema, is_list = _SchemaSpecInternal._nested_of(m)
            docs.append(
                FieldDoc(
                    name=f.name,
                    help=m["help"],
                    required=m["required"],
                    example=example,
                    choices=m["choices"],
                    nested=schema,
                    nested_is_list=is_list,
                )
            )
        return docs

    @classmethod
    def to_markdown(cls, title: Optional[str] = None, _level: int = 2) -> str:
        """Markdown reference for this schema, recursing into nested schemas.

        Generated from the same metadata that drives validation, so the doc can
        never drift from the enforced shape.  Suitable for committing as a
        ``*_FORMAT.md`` and linking from a tool's help.
        """
        heading = "#" * _level
        head = title or cls.__name__
        lines: List[str] = [f"{heading} {head}", ""]
        doc = _SchemaSpecInternal._doc_lines(cls)
        if doc:
            lines += [doc[0].strip(), ""]
        lines += ["| Key | Required | Description | Example |", "|---|---|---|---|"]
        nested_specs: List[Type[SchemaSpec]] = []
        for fd in cls.describe():
            req = "yes" if fd.required else "no"
            desc = fd.help or ""
            if fd.choices is not None:
                desc = (desc + f" One of: {', '.join(map(str, fd.choices))}.").strip()
            if fd.nested is not None:
                # A nested block gets its own table below — pointing there reads
                # better than cramming a serialised default into one cell.
                suffix = " (list)" if fd.nested_is_list else ""
                example = f"see *{fd.nested.__name__}* below{suffix}"
                if fd.nested not in nested_specs:
                    nested_specs.append(fd.nested)
            elif fd.example in (None, MISSING):
                example = ""
            else:
                example = f"`{_SchemaSpecInternal._compact(fd.example)}`"
            lines.append(f"| `{fd.name}` | {req} | {desc} | {example} |")
        for ns in nested_specs:
            lines += ["", ns.to_markdown(_level=_level + 1)]
        return "\n".join(lines)

    @classmethod
    def json_schema(cls) -> Dict[str, Any]:
        """This schema as a JSON Schema (draft 2020-12) document.

        Derived from the same declaration :meth:`validate` enforces: each
        field's type from its annotation (``Optional[float]`` is a number or
        null, ``Tuple[float, float, float]`` exactly three numbers, a nested
        schema a ``$ref`` into ``$defs``, by class name), its ``help`` as the
        description, its ``choices`` as an enum, and the required keys.
        Unknown keys are left allowed, as :meth:`validate` tolerates them.
        It is the form a reader in another language generates its types
        from, so the generated types can never drift from this definition.

        Returns:
            dict: ``{"$schema", "title", "description", "type": "object",
            "properties", "required", "$defs"}`` (empty parts omitted).

        Raises:
            SchemaError: Two nested schemas share a class name, or a
                :attr:`TYPED` field's annotation and ``nested=`` disagree.
        """
        defs: Dict[str, Any] = {}
        document: Dict[str, Any] = {
            "$schema": cls.JSON_SCHEMA_DIALECT,
            "title": cls.__name__,
        }
        document.update(_SchemaSpecInternal._object_schema(cls, defs))
        if defs:
            document["$defs"] = defs
        return _SchemaSpecInternal._strip_specs(document)

    @staticmethod
    def spec_field(
        *,
        help: str = "",
        example: Any = MISSING,
        required: bool = False,
        nested: Optional[Type["SchemaSpec"]] = None,
        choices: Optional[Sequence[Any]] = None,
        validate: Optional[Callable[[Any], List[str]]] = None,
        default: Any = MISSING,
        default_factory: Any = MISSING,
    ):
        """A :func:`dataclasses.field` carrying schema metadata.

        ``required`` is a *validation* flag (the key must appear in an input file),
        independent of the dataclass default — every spec field still gets a default
        so the dataclass stays constructible and field-ordering rules never bite.
        When neither *default* nor *default_factory* is given the field defaults to
        ``None``.

        Parameters:
            help: One-line description shown in generated docs.
            example: Value used in :meth:`SchemaSpec.skeleton`/docs.  Omit to fall
                back to the field default (or a nested schema's skeleton).
            required: Whether the key must be present in a validated file.
            nested: A :class:`SchemaSpec` subclass validated recursively for this
                key's (mapping) value. Wrap it in a one-element list —
                ``nested=[MySpec]`` (mirroring ``typing.List[X]``) — for a key
                whose value is a *list* of that schema.
            choices: Allowed values for a scalar field.
            validate: Callable ``(value) -> list[str]`` returning error strings for
                polymorphic/irregular fields the generic checks can't express.
        """
        meta = {
            _META: {
                "help": help,
                "example": example,
                "required": required,
                "nested": nested,
                "choices": choices,
                "validate": validate,
            }
        }
        if default is not MISSING:
            return field(default=default, metadata=meta)
        if default_factory is not MISSING:
            return field(default_factory=default_factory, metadata=meta)
        return field(default=None, metadata=meta)
