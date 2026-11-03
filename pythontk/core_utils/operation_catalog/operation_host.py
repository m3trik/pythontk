# !/usr/bin/python
# coding=utf-8
"""Where an operation runs: the :class:`OperationHost` seam and its in-process default.

:class:`~pythontk.OperationCatalog` validates a call and decides dry run or
live; an :class:`OperationHost` runs it. An entry whose ``host`` is unset runs
in this Python process (:class:`PureOperationHost`). An entry naming an
application (``"maya"``, ``"blender"``, ``"unity"``) runs through the host that
application's library registers under the ``pythontk.operation_hosts``
entry-point group. Such a runner's contract is to never reach an instance a
person opened: it runs the call in an instance the tooling launched and owns
(:class:`~pythontk.BatchOperationHost` launches a fresh one per call), hands
it there to :class:`PureOperationHost` directly (never back to a catalog,
which would route it to a runner again) and returns its :class:`Outcome`.
The Protocol cannot enforce that; each runner's own tests do.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import os
import sys
import typing
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, runtime_checkable

from pythontk.core_utils.cancel_scope import CancelScope, OperationCancelled
from pythontk.core_utils.operation_catalog.op_context import OpContext
from pythontk.core_utils.operation_catalog.operation import _UNIONS, Operation
from pythontk.core_utils.operation_catalog.outcome import OpRequest, Outcome
from pythontk.core_utils.schema_spec import SchemaSpec


@runtime_checkable
class OperationHost(Protocol):
    """Runs one validated operation call and hands back its :class:`Outcome`."""

    def run(
        self,
        entry: Mapping[str, Any],
        request: OpRequest,
        scope: Optional[CancelScope] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
    ) -> Outcome:
        """Run *request* (already validated) for *entry*, its catalog entry.

        Parameters:
            entry: The operation's catalog entry (``id``, ``target``, ...).
            request: The call: its arguments (JSON values checked against
                ``entry["args"]``), whether it is a dry run (the catalog never
                asks a host to dry-run an entry whose ``dry_run`` is false) and
                the scenes to open first.
            scope: The cancel scope to honour, if the caller holds one.
            on_progress: Called with ``(fraction, message)`` as it goes.
        """


class _PureOperationHostInternal:
    """Internal helpers for :class:`PureOperationHost`."""

    @classmethod
    def _coerce(cls, tp: Any, value: Any) -> Any:
        """*value* (JSON) as the type *tp* declares: a path's text as the path
        class, an enum's value as its member, a payload's mapping as its
        :class:`SchemaSpec`, through ``Optional``, lists, tuples, sets and
        dict values; anything else as it came."""
        if value is None or tp is Any:
            return value
        origin, args = typing.get_origin(tp), typing.get_args(tp)
        if origin in _UNIONS:
            options = [a for a in args if a is not type(None)]
            return cls._coerce(options[0], value) if len(options) == 1 else value
        if tp is int and isinstance(value, float) and value.is_integer():
            return int(value)  # JSON Schema counts 2.0 an integer; the function gets 2
        if isinstance(tp, type):
            if issubclass(tp, SchemaSpec) and isinstance(value, Mapping):
                return tp.from_dict(value)
            if issubclass(tp, enum.Enum) and not isinstance(value, tp):
                return tp(value)
            if issubclass(tp, (os.PathLike, PurePath)) and isinstance(value, str):
                return Path(value) if tp is os.PathLike else tp(value)
        if isinstance(value, (list, tuple)):
            if origin is tuple:
                if len(args) == 2 and args[1] is Ellipsis:
                    return tuple(cls._coerce(args[0], v) for v in value)
                return tuple(cls._coerce(a, v) for a, v in zip(args, value)) if args else tuple(value)
            item = args[0] if args else Any
            items = [cls._coerce(item, v) for v in value]
            return set(items) if origin in (set, frozenset) else items
        if isinstance(value, Mapping) and len(args) == 2:
            return {k: cls._coerce(args[1], v) for k, v in value.items()}
        return value

    @classmethod
    def _typed_args(cls, fn: Callable, args: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            hints = typing.get_type_hints(fn)
        except Exception:
            hints = {}
        return {name: cls._coerce(hints.get(name, Any), value) for name, value in args.items()}

    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        """*value* as JSON: a payload's ``to_dict``, a dataclass's fields, a
        path's text, a named object's name (a Blender datablock: what a
        caller passes back); anything else unreadable becomes its ``repr``.
        A mapping's keys read the same way, as text."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, enum.Enum):
            return cls._jsonable(value.value)
        if isinstance(value, Mapping):
            return {cls._key(k): cls._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._jsonable(v) for v in value]
        if isinstance(value, (set, frozenset)):
            return sorted((cls._jsonable(v) for v in value), key=repr)
        if isinstance(value, os.PathLike):
            return os.fspath(value)
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return cls._jsonable(to_dict())
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return cls._jsonable(dataclasses.asdict(value))
        name = getattr(value, "name", None)
        return name if isinstance(name, str) else repr(value)

    @classmethod
    def _key(cls, key: Any) -> str:
        """A mapping key as JSON's text: a named object's name, else its ``str``."""
        plain = cls._jsonable(key)
        return plain if isinstance(plain, str) else str(key)

    @classmethod
    def _result_of(cls, value: Any) -> Any:
        """:meth:`_jsonable`, never raising: a result that cannot be read (a
        cycle, a ``to_dict`` that raises) must not cost a run that already
        made its changes its outcome."""
        try:
            return cls._jsonable(value)
        except (Exception, RecursionError) as exc:
            try:
                return f"<unreadable result: {type(exc).__name__}: {exc}> {value!r}"
            except Exception:
                return f"<unreadable result: {type(exc).__name__}: {exc}>"


class PureOperationHost(_PureOperationHostInternal):
    """Runs an operation in this Python process: the default for an entry with no host.

    The arguments reach the function as the types its signature declares
    (see ``_coerce``); the call gets an :class:`OpContext` when it takes
    ``ctx``. Anything it prints goes to
    stderr, so it can never corrupt the JSON a front end writes to stdout
    (the CLI, an MCP server over stdio).
    It opens no scenes: a request's ``scenes`` are for the host that
    launched this process (:class:`~pythontk.BatchOperationHost` opens them
    before it hands the call here), so they are ignored.
    Its status follows from how it ended: ``done``; ``cancelled`` at a
    checkpoint; ``partial`` when a live run raised after changing something,
    or a live run of an operation that keeps no journal raised (what it
    changed is unknown); ``failed`` when it raised before changing anything,
    code that did not compile included (a ``SyntaxError`` from a ``code``
    operation: none of it ran).
    """

    def run(
        self,
        entry: Mapping[str, Any],
        request: OpRequest,
        scope: Optional[CancelScope] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
    ) -> Outcome:
        op_id, dry_run = entry["id"], request.dry_run
        try:
            fn = Operation.resolve_target(entry["target"])
            kwargs = self._typed_args(fn, request.args)
        except Exception as exc:
            return Outcome.refused(
                op_id, f"cannot call {entry.get('target')}: {type(exc).__name__}: {exc}", dry_run
            )
        journal = Operation.takes_context(fn)
        if dry_run and not journal and entry.get("effect") != Operation.READ:
            # The catalog never asks this; a host asked directly still must
            # not run a change it cannot plan.
            return Outcome.refused(op_id, Operation.NO_DRY_RUN.format(op=op_id), dry_run)
        ctx = OpContext(dry_run=dry_run, scope=scope, on_progress=on_progress)
        if journal:
            kwargs[Operation.CONTEXT_PARAMETER] = ctx
        status, error, value = "done", None, None
        try:
            with contextlib.redirect_stdout(sys.stderr):
                value = fn(**kwargs)
        except OperationCancelled as exc:
            status, error = "cancelled", str(exc) or "cancelled"
        except (Exception, SystemExit) as exc:
            # SystemExit too: an operation wrapping a script's main() must
            # never end the process serving it (the CLI, an MCP server).
            error = f"{type(exc).__name__}: {exc}"
            live_change = not dry_run and entry.get("effect") != Operation.READ
            never_ran = entry.get("effect") == Operation.CODE and isinstance(exc, SyntaxError)
            if live_change and not journal and not never_ran:
                status = "partial"
                error += " (it keeps no journal, so what it changed before failing is unknown)"
            else:
                status = "partial" if (ctx.changes and not dry_run) else "failed"
        return Outcome(
            op=op_id,
            status=status,
            dry_run=dry_run,
            result=self._result_of(value),
            changes=list(ctx.changes),
            issues=list(ctx.issues),
            error=error,
        )
