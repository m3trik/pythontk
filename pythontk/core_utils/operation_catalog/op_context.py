# !/usr/bin/python
# coding=utf-8
"""OpContext -- what a running operation is handed: dry run, plan, issues, progress.

An operation that changes anything asks its context before each change, and the
answer is the dry run::

    def mirror(objects: list, axis: str = "x", ctx=None) -> dict:
        for obj in objects:
            if ctx is None or ctx.should_change(obj, f"mirror across {axis}"):
                ...  # make the change
        return {"mirrored": len(objects)}

In a dry run :meth:`OpContext.should_change` records the change and answers
``False``, so the same body that does the work lists it; in a live run it
records it and answers ``True``. :meth:`OpContext.should_create` and
:meth:`OpContext.should_delete` are the same for an object made or removed
(a plan that tells a creation or a deletion from an edit is easier to review),
and :meth:`OpContext.should_write` for a file. Reads and checks run in a dry
run too, which is what makes the plan honest. It is PowerShell's ``ShouldProcess`` /
``-WhatIf`` pattern; an operation in another language asks the same questions
in its own case (``ctx.ShouldChange`` in C#).

The context also carries the run's issues (:meth:`OpContext.issue`), its
progress (:meth:`OpContext.progress`) and its cancellation: every call above is
a checkpoint of the run's :class:`~pythontk.CancelScope` (the scope it was
given, else the ambient one), so an operation that asks before it changes is
cancellable for free. Library code takes the context as an optional ``ctx``
keyword and never imports this module: it is duck-typed, and every change site
asks ``ctx is None or ctx.should_...(...)``, so a function called on its own
(no context) makes its changes. One guard spelling everywhere keeps a missed
site easy to see; a default context object would make the library import the
engine, which pythontk's own providers sit below.
"""

from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Union

from pythontk.core_utils.cancel_scope import CancelScope
from pythontk.core_utils.operation_catalog.outcome import Change, Issue, OpCall


class _OpContextInternal:
    """Internal helpers for :class:`OpContext`."""

    def _ask(self, kind: str, target: Any, action: str) -> bool:
        """Record one change of *kind* and answer whether it may be made."""
        self.check()
        self.changes.append(Change(kind=kind, target=str(target), action=action or None))
        return not self.dry_run

    @staticmethod
    def _as_call(code: str, fix: Union[Mapping[str, Any], OpCall]) -> OpCall:
        """*fix* as an :class:`OpCall`.

        Raises:
            ValueError: A mapping that names no ``op``.
        """
        if isinstance(fix, OpCall):
            return fix
        if not fix.get("op"):
            raise ValueError(f"issue {code!r}: its fix names no 'op' ({dict(fix)!r})")
        return OpCall(op=fix["op"], args=dict(fix.get("args") or {}))


class OpContext(_OpContextInternal):
    """The run state one operation call is handed (see the module docstring).

    Parameters:
        dry_run: Plan, do not change: every ``should_*`` answers ``False``.
        scope: The cancel scope to check at each checkpoint; ``None`` checks
            the ambient one (``CancelScope.check``), if any is active.
        on_progress: Called with ``(fraction, message)`` on each
            :meth:`progress` report (a progress bar, a log, an MCP
            notification).
    """

    def __init__(
        self,
        dry_run: bool = False,
        scope: Optional[CancelScope] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
    ):
        self.dry_run = bool(dry_run)
        self.scope = scope
        self.on_progress = on_progress
        #: Every change asked for, in order: the plan in a dry run, the journal
        #: in a live one.
        self.changes: List[Change] = []
        #: Every issue reported, in order.
        self.issues: List[Issue] = []

    def should_change(self, target: Any, action: str = "") -> bool:
        """Record a change to a scene object; ``True`` when it may be made.

        Parameters:
            target: The object that changes (its name, or anything whose
                ``str`` names it).
            action: What the change does, in a few words.
        """
        return self._ask("change", target, action)

    def should_create(self, target: Any, action: str = "") -> bool:
        """Record an object about to be made; ``True`` when it may be.

        Parameters:
            target: The object's name (or anything whose ``str`` names it).
            action: What it is, in a few words.
        """
        return self._ask("create", target, action)

    def should_delete(self, target: Any, action: str = "") -> bool:
        """Record an object or a file about to be removed; ``True`` when it may be.

        Parameters:
            target: The object's name, or the file's path.
            action: Why, in a few words.
        """
        return self._ask("delete", target, action)

    def should_write(self, path: Any, action: str = "") -> bool:
        """Record a write to a file; ``True`` when it may be made.

        Parameters:
            path: The file written.
            action: What the write is, in a few words.
        """
        return self._ask("write", path, action)

    def issue(
        self,
        code: str,
        message: str,
        severity: str = "error",
        location: Optional[str] = None,
        fix: Optional[Union[Mapping[str, Any], OpCall]] = None,
    ) -> Issue:
        """Report a problem found; it rides on the run's outcome.

        Parameters:
            code: The issue's stable code, declared beside the operation.
            message: What is wrong and what to do.
            severity: ``"error"``, ``"warning"`` or ``"info"``.
            location: Where: an object's name or a file's path.
            fix: The operation call that fixes it: an :class:`OpCall`, or
                ``{"op": id, "args": {...}}``.

        Returns:
            The recorded :class:`Issue`.

        Raises:
            ValueError: *severity* is not one of ``Issue.SEVERITIES``, or *fix*
                names no ``op``.
        """
        if severity not in Issue.SEVERITIES:
            raise ValueError(
                f"severity {severity!r} is not one of {list(Issue.SEVERITIES)}"
            )
        issue = Issue(
            code=code,
            message=message,
            severity=severity,
            location=None if location is None else str(location),
            fix=None if fix is None else self._as_call(code, fix),
        )
        self.issues.append(issue)
        return issue

    def progress(self, fraction: float, message: str = "") -> None:
        """Report how far the run is (0 to 1), and check for a cancel."""
        self.check()
        if self.on_progress is not None:
            self.on_progress(max(0.0, min(1.0, float(fraction))), message)

    def check(self) -> None:
        """A cancel checkpoint: raise ``OperationCancelled`` if the run was stopped."""
        if self.scope is not None:
            self.scope.checkpoint()
        else:
            CancelScope.check()
