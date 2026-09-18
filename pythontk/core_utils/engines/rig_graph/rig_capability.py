# !/usr/bin/python
# coding=utf-8
"""What ONE target can actually build — the capability manifest.

NO DCC IMPORTS. A builder package (blendertk, mayatk-as-target, unitytk)
declares the SHAPE of its entries alongside its builder registry, and a
conformance run fills in the ``fidelity`` grades by measuring each op against
its fixture. Nobody types ``exact`` anywhere: the grade is a measurement on
that target, which is the whole reason fidelity in this system cannot lie.

The manifest is a committed test artifact, guarded by a test that regenerates
it and fails on a difference — the same drift guard this repo uses for its
vendored template readers. So a builder that silently stops honouring a
parameter is caught by the artifact changing, not by a user noticing a rig
arrive wrong.

This module owns the QUERY side: given a record, which of the planner's
capability rules does this target fail, and why. The planner owns sequencing
and fallback; keeping the two apart is what lets a UI ask "could this target
build that?" without planning a whole graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pythontk.core_utils.engines.rig_graph.rig_model import (
    SCHEMA_VERSION,
    RigGraph,
    RigRecord,
)

#: How faithfully an op reproduces its conformance fixture, best first. Read
#: from a measurement, never declared.
FIDELITIES = ("exact", "approximate", "lossy")


@dataclass
class RigOpCapability:
    """One ``"<shape>/<op>"`` entry: what this target honours, and how well."""

    #: A conformance result. An op with no fixture can still be built but can
    #: never grade ``exact`` -- nothing measured it -- so the planner always
    #: attaches ``verify`` to records that use it.
    fidelity: str = "approximate"
    #: Target channels the builder writes. Empty means the op takes none.
    channels: Tuple[str, ...] = ()
    #: Source roles it understands. Empty means it takes no sources.
    roles: Tuple[str, ...] = ()
    #: Parameter path -> True (any literal) or a list of accepted enum values.
    #: A path absent from this map is unsupported. Nested paths are dotted.
    params: Dict[str, Any] = field(default_factory=dict)
    #: Parameter paths a ``{"plug": ...}`` Value may sit on, prefixed
    #: (``sources.weight``, ``params.twist.start``).
    plugs: Tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RigOpCapability":
        """Build from plain values."""
        return cls(
            fidelity=str(data.get("fidelity", "approximate")),
            channels=tuple(data.get("channels") or ()),
            roles=tuple(data.get("roles") or ()),
            params=dict(data.get("params") or {}),
            plugs=tuple(data.get("plugs") or ()),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Plain-value form, omitting empty sections."""
        out: Dict[str, Any] = {"fidelity": self.fidelity}
        if self.channels:
            out["channels"] = list(self.channels)
        if self.roles:
            out["roles"] = list(self.roles)
        if self.params:
            out["params"] = self.params
        if self.plugs:
            out["plugs"] = list(self.plugs)
        return out

    def accepts_param(self, path: str, value: Any) -> bool:
        """Whether this op honours *value* at parameter *path*.

        Parameters:
            path (str): dotted path below ``params`` (e.g. ``"twist.mode"``).
            value: the literal used, or a plug (checked by :meth:`accepts_plug`
                instead, so any value passes here).

        Returns:
            bool: True when the path is declared and, for an enum, the value
                is listed.
        """
        if path not in self.params:
            return False
        allowed = self.params[path]
        if allowed is True or RigGraph.is_plug(value):
            return True
        if isinstance(allowed, (list, tuple)):
            return value in allowed
        return bool(allowed)

    def accepts_plug(self, path: str) -> bool:
        """Whether a driven Value may sit at *path* (prefixed, e.g.
        ``"params.twist.start"``)."""
        return path in self.plugs


@dataclass
class RigCapability:
    """Everything one target can build, as data."""

    target: str
    target_version: str = ""
    schema_version: int = SCHEMA_VERSION
    shapes: Tuple[str, ...] = ()
    ops: Dict[str, RigOpCapability] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RigCapability":
        """Build from plain values (a committed manifest)."""
        return cls(
            target=str(data.get("target", "")),
            target_version=str(data.get("target_version", "")),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            shapes=tuple(data.get("shapes") or ()),
            ops={
                key: RigOpCapability.from_dict(value)
                for key, value in (data.get("ops") or {}).items()
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Plain-value form, with ops sorted so a regenerated manifest diffs
        cleanly against the committed one."""
        return {
            "target": self.target,
            "target_version": self.target_version,
            "schema_version": self.schema_version,
            "shapes": list(self.shapes),
            "ops": {key: self.ops[key].to_dict() for key in sorted(self.ops)},
        }

    def op(self, record: RigRecord) -> Optional[RigOpCapability]:
        """The entry for *record*'s ``"<shape>/<op>"``, or None."""
        return self.ops.get(record.key)

    def rejects(self, record: RigRecord) -> List[Tuple[str, str]]:
        """Why this target cannot build *record* — the capability rules, in order.

        These are the planner's rules 2-7, kept here because they are questions
        about a TARGET, not about sequencing: a UI can ask them of one record
        without planning a graph. The planner adds enablement, cycles, fidelity
        and the fallback that follows.

        Parameters:
            record (RigRecord): the record to test.

        Returns:
            list: ``(reason, detail)`` pairs, empty when the target can build
                it. ``reason`` is the stable slug a report groups by.
        """
        if record.shape not in self.shapes:
            return [("unsupported_shape", record.shape)]
        entry = self.op(record)
        if entry is None:
            return [("no_builder", record.key)]

        failures: List[Tuple[str, str]] = []
        # `transform` and `points` ops have a FIXED vocabulary -- `space`, `aim`,
        # `curve`, `translate` -- so an undeclared name is a real gap and absent
        # means the op accepts none. A `channel` op does not: its roles are the
        # expression's own variable names and its "channel" is whatever plug it
        # was pointed at, so neither is the target's to declare.
        if record.shape != "channel":
            target = record.target
            if isinstance(target, dict):
                for channel in target.get("channels") or ():
                    if channel not in entry.channels:
                        failures.append(("unsupported_channel", str(channel)))
            for source in record.sources:
                role = source.get("role")
                if role is not None and role not in entry.roles:
                    failures.append(("unsupported_role", str(role)))
        elif entry.channels:
            # A channel op MAY still opt into a restriction (a target that can
            # drive transforms but not blend-shape weights, say).
            channel = RigGraph.split_plug(str(record.target or ""))[1]
            if channel and channel not in entry.channels:
                failures.append(("unsupported_channel", channel))
        for path, value in _flatten_params(record.params):
            if not entry.accepts_param(path, value):
                failures.append(("unsupported_param", f"params.{path}={value!r}"))
        for path, _ in record.plug_paths():
            if not entry.accepts_plug(path):
                failures.append(("undrivable", path))
        return failures


def _flatten_params(params: Dict[str, Any], prefix: str = "") -> List[Tuple[str, Any]]:
    """Dotted leaf paths of a params dict, so a capability can declare nesting.

    A LIST is a leaf: a capability declares ``skip`` or ``weights``, not every
    element of one. A plug is a leaf too -- rule 7 checks those by path.

    Parameters:
        params (dict): a record's params.
        prefix (str): internal, the path accumulated so far.

    Returns:
        list: ``(dotted path, value)`` pairs.
    """
    out: List[Tuple[str, Any]] = []
    for key, value in (params or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and not RigGraph.is_plug(value):
            out.extend(_flatten_params(value, f"{path}."))
        else:
            out.append((path, value))
    return out
