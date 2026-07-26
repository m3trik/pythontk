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
* :meth:`describe` / :meth:`to_markdown` — human-readable reference docs.

This is the storage-agnostic *shape* SSoT.  Pair it with a
:class:`~pythontk.core_utils.preset_store.PresetStore` (any codec — JSON or
YAML) through :class:`~pythontk.core_utils.template_set.TemplateSet` to get a
discoverable, user-extensible collection of template files whose schema is
documented and enforced from one place.

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

import copy
import dataclasses
import logging
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

# Metadata namespace key — keeps schema metadata from colliding with any other
# library that reads ``dataclasses.field(metadata=...)``.
_META = "schema_spec"

# Sentinel: "no explicit example given — derive one from the field default or a
# nested schema." Distinct from ``None``, which is a legitimate example value.
MISSING = dataclasses.MISSING


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


@dataclass
class SchemaSpec(_SchemaSpecInternal):
    """Base for declarative template schemas (see module docstring).

    Subclass with ``@dataclass`` and declare fields via :func:`spec_field`.
    Override :meth:`from_dict` / :meth:`to_dict` for (de)serialisation of
    irregular shapes; the validate/skeleton/docs machinery is inherited.
    """

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
        nested-schema errors, and anything a field's ``validate`` callable
        reports.  Warnings: unrecognised top-level keys (``_``-prefixed keys
        are reserved for annotations and skipped).
        """
        res = ValidationResult()
        if not isinstance(data, dict):
            res.errors.append(f"expected a mapping, got {type(data).__name__}")
            return res

        specs = cls._specs()
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
        # Skip the constructor signature ``@dataclass`` auto-synthesises as
        # ``__doc__`` when a subclass has no real docstring — emitting it would
        # leak field types / ``<factory>`` / ``=None`` noise into a committed doc.
        doc_str = cls.__doc__ or ""
        if doc_str.strip().startswith(cls.__name__ + "("):
            doc_str = ""
        doc = doc_str.strip().splitlines()
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
