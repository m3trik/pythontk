# !/usr/bin/python
# coding=utf-8
"""What an operation is asked and hands back, declared once as data every front end reads.

:class:`OpRequest` is one call as the batch protocol carries it: the
``request.json`` a fresh application instance reads (see its docstring).
:class:`Outcome` is the envelope every run returns, whichever front end asked
(the CLI, MCP, a panel) and wherever it ran (this process, a fresh mayapy, an
application another language serves): its ``status`` (``done``, ``partial``, ``cancelled``,
``failed``), whether it was a dry run, the operation's own ``result``, the
changes it planned or made (:class:`Change`), the issues it found
(:class:`Issue`, each with an optional ``fix`` that is itself an operation
call, :class:`OpCall`) and the ``error`` that stopped it.

They are typed :class:`~pythontk.core_utils.schema_spec.SchemaSpec` payloads:
one definition validates a payload, round-trips it (``to_dict`` /
``from_dict``) and publishes its JSON Schema, the form another language's
types are generated from (CODE_STANDARD section 0, Contracts), so a C# run's
result and a Python run's are the same document. A dry run lists the plan as
data, a run that stops says how far it got, and an issue's fix is an operation
call that an agent, or an operation that repairs what a check found, applies
the same way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from pythontk.core_utils.schema_spec import SchemaSpec

_field = SchemaSpec.spec_field


@dataclass
class OpCall(SchemaSpec):
    """One operation call: an id and its arguments (an issue's fix, a recipe step)."""

    TYPED = True

    op: str = _field(
        help="The operation id, '<area>.<action>' in lower case.",
        required=True,
        example="mesh.mirror",
    )
    args: Dict[str, Any] = _field(
        help="Its arguments, by name; an argument left out takes its default.",
        default_factory=dict,
        example={"axis": "x"},
    )


@dataclass
class OpRequest(SchemaSpec):
    """One call as a batch launch reads it: the batch protocol.

    A runner makes a fresh run folder, writes the request there as
    :attr:`REQUEST_FILE`, launches a NEW instance of the application, and reads
    back the :class:`Outcome` it writes beside it as :attr:`OUTCOME_FILE`; a
    fresh folder per launch means a stale outcome is never read. The
    application opens the request's scenes, runs the call synchronously, saves
    the scenes the call changed (in a dry run: lists them as the writes it
    would make), writes the outcome and exits by itself, never through a quit
    flag that cuts deferred work short. A host written in any language reads
    the same document.
    """

    TYPED = True

    #: The format this pythontk writes; a host refuses a newer one.
    FORMAT = 1
    #: The request's file name in a run folder.
    REQUEST_FILE = "request.json"
    #: The outcome's file name, beside the request.
    OUTCOME_FILE = "outcome.json"

    format: int = _field(
        help="The request's format version; a host refuses a newer one.",
        required=True,
        default=1,
        example=1,
    )
    op: str = _field(
        help="The operation id, '<area>.<action>' in lower case.",
        required=True,
        example="scene.summary",
    )
    args: Dict[str, Any] = _field(
        help="Its arguments, by name, already checked against the entry's schema.",
        default_factory=dict,
        example={},
    )
    dry_run: bool = _field(
        help="Plan only: the outcome's changes are what a live run would make.",
        default=False,
        example=True,
    )
    scenes: List[str] = _field(
        help=(
            "Scene files to open before the call, as absolute paths, in order "
            "(the first is the main one); a live run saves those it changed."
        ),
        default_factory=list,
        example=["C:/project/scenes/shot_010.ma"],
    )


@dataclass
class Change(SchemaSpec):
    """One change an operation planned (a dry run) or made (a live run)."""

    TYPED = True

    #: The kinds of change, one per ``OpContext`` question (C#: ``ShouldCreate``,
    #: ``ShouldChange``, ``ShouldDelete``, ``ShouldWrite``): a scene object
    #: created, changed or deleted, a file written; ``delete`` covers a file too.
    KINDS = ("create", "change", "delete", "write")

    kind: str = _field(
        help=(
            "'create', 'change' or 'delete' for an object in a scene ('delete' "
            "also for a file); 'write' for a file written, new or overwritten."
        ),
        required=True,
        choices=KINDS,
        example="change",
    )
    target: str = _field(
        help="What changes: the object's name or the file's path.",
        required=True,
        example="pCube1",
    )
    action: str = _field(
        help="What the change does, in a few words.",
        example="mirror across x",
    )


@dataclass
class Issue(SchemaSpec):
    """One problem an operation found, and how to fix it when it knows."""

    TYPED = True

    SEVERITIES = ("error", "warning", "info")

    code: str = _field(
        help=(
            "A stable code naming the problem, declared beside the operation "
            "that reports it and never reused for another."
        ),
        required=True,
        example="mesh.non_manifold",
    )
    message: str = _field(
        help="What is wrong, why it matters and what to do, in a sentence.",
        required=True,
        example="pCube1 has 2 non-manifold edges; the export would split them.",
    )
    severity: str = _field(
        help="'error' stops the work the issue names; 'warning' and 'info' do not.",
        choices=SEVERITIES,
        default="error",
        example="warning",
    )
    location: str = _field(
        help="Where: a scene object's name or a file's path.",
        example="pCube1",
    )
    fix: OpCall = _field(
        help="The operation call that fixes it, when one does.",
        nested=OpCall,
    )


@dataclass
class Outcome(SchemaSpec):
    """What one operation run hands back (see the module docstring)."""

    TYPED = True

    #: ``done``: it finished. ``partial``: a live run failed after changing
    #: something (``changes`` lists what it asked to change; the last entry may
    #: be the change that failed), or one that keeps no journal failed, so what
    #: it changed is unknown. ``cancelled``: it was stopped at a checkpoint.
    #: ``failed``: it changed nothing and did not finish, or was refused.
    STATUSES = ("done", "partial", "cancelled", "failed")

    op: str = _field(
        help="The operation id that ran.",
        required=True,
        example="mesh.mirror",
    )
    status: str = _field(
        help="'done', 'partial', 'cancelled' or 'failed'.",
        required=True,
        choices=STATUSES,
        example="done",
    )
    dry_run: bool = _field(
        help="Whether this was a dry run: 'changes' is then the plan, nothing made.",
        default=False,
        example=False,
    )
    result: Any = _field(
        help="The operation's own result, as JSON.",
        example={"mirrored": 1},
    )
    changes: List[Change] = _field(
        help="The changes planned (a dry run) or made (a live run), in order.",
        nested=[Change],
        default_factory=list,
    )
    issues: List[Issue] = _field(
        help="The problems found, in the order found.",
        nested=[Issue],
        default_factory=list,
    )
    error: str = _field(
        help="Why the run stopped or was refused, when it did not finish.",
        example="unknown argument 'axes' (expected: axis, objects)",
    )

    @property
    def ok(self) -> bool:
        """Whether the run finished (``status == "done"``)."""
        return self.status == "done"

    def to_json(self, indent: int = 2) -> str:
        """This outcome as a JSON document (the CLI's and MCP's wire form)."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False, default=str)

    @classmethod
    def refused(cls, op: str, error: str, dry_run: bool = False) -> "Outcome":
        """A run that never started: *error* says why, nothing changed."""
        return cls(op=op, status="failed", dry_run=dry_run, error=error)
