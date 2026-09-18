# !/usr/bin/python
# coding=utf-8
"""Resolve a RigGraph against ONE target into what to build, what to bake, and why.

NO DCC IMPORTS, no side effects. :meth:`RigPlanner.plan` is a pure function of
a :class:`~pythontk.core_utils.engines.rig_graph.rig_model.RigGraph` and a
:class:`~pythontk.core_utils.engines.rig_graph.rig_capability.RigCapability`,
which is what makes a PRE-FLIGHT possible: a consumer can tell a user "12 rigs
will be baked; 3 deformers have no equivalent" before committing to a
ten-minute conversion, with no DCC open and nothing written.

Two rules carry the design:

**Degrade, never break.** Every record decides its own fallback, so an exotic
rig loses the parts a target cannot build and keeps the rest. "Drop and bake"
is not a second code path -- it is this planner against a target that registers
no builders.

**A half-built node is worse than a baked one.** A node stays procedural only
if EVERY record targeting it was built; if any was not, the node is baked and
the other records on it are demoted with it. That cascades (a demoted record's
other targets bake too), so it is resolved to a fixpoint rather than in one
pass.

Every loss is a :class:`ReportEntry` emitted by the code that causes it, so the
report cannot drift from what actually happened the way a hand-maintained list
of caveats always does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from pythontk.core_utils.engines.rig_graph.rig_capability import RigCapability
from pythontk.core_utils.engines.rig_graph.rig_model import RigGraph, RigRecord
from pythontk.core_utils.engines.rig_graph.rig_verify import RigVerify

#: What happened to a record, as a UI can render it.
KINDS = ("native", "baked", "degraded", "diverged", "dropped", "cyclic", "failed")

# How a report kind reads to someone deciding whether to proceed. `baked` is
# INFO on purpose: motion is preserved and only editability is lost, so it is
# the expected outcome for most of a production rig, not a problem.
_SEVERITY = {
    "native": "info",
    "baked": "info",
    "degraded": "warn",
    "diverged": "warn",
    "dropped": "warn",
    "cyclic": "warn",
    "failed": "error",
}


class RigPlanRefused(RuntimeError):
    """A record whose fallback is ``"fail"`` could not be built.

    Raised instead of returning a plan, because ``fail`` is the caller saying
    this relationship is not optional -- a rig without it is not worth shipping.
    """


@dataclass
class ReportEntry:
    """One thing that happened to one record, and what it cost."""

    kind: str
    reason: str
    record: Optional[str] = None
    nodes: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def severity(self) -> str:
        """``info`` / ``warn`` / ``error``, from :data:`KINDS`."""
        return _SEVERITY.get(self.kind, "warn")

    @property
    def recoverable(self) -> bool:
        """Whether the motion survived even though the relationship did not."""
        return self.kind in ("baked", "degraded", "diverged", "cyclic")

    def to_dict(self) -> Dict[str, Any]:
        """Plain-value form, as written to a manifest sidecar."""
        out: Dict[str, Any] = {
            "kind": self.kind,
            "severity": self.severity,
            "reason": self.reason,
            "recoverable": self.recoverable,
        }
        if self.record:
            out["record"] = self.record
        if self.nodes:
            out["nodes"] = self.nodes
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass
class PlanResult:
    """What a target should do with a graph."""

    #: Record ids to build, in document order.
    build: List[str] = field(default_factory=list)
    #: Node ids whose motion must be baked instead.
    bake: List[str] = field(default_factory=list)
    #: Record id -> the ``verify`` spec the planner attached, because the op
    #: did not grade ``exact`` on this target.
    verify: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    report: List[ReportEntry] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        """How many records met each outcome — the one-line pre-flight summary.

        Returns:
            dict: kind -> count, omitting kinds that did not occur.
        """
        tally: Dict[str, int] = {}
        for entry in self.report:
            tally[entry.kind] = tally.get(entry.kind, 0) + 1
        return tally

    def worst_severity(self) -> str:
        """The highest severity present: ``error`` > ``warn`` > ``info``."""
        levels = {entry.severity for entry in self.report}
        for level in ("error", "warn", "info"):
            if level in levels:
                return level
        return "info"

    def entries(self, severity: str) -> List[ReportEntry]:
        """Report entries at exactly *severity*."""
        return [e for e in self.report if e.severity == severity]

    def to_dict(self) -> Dict[str, Any]:
        """Plain-value form, as written beside the graph in a sidecar."""
        return {
            "build": self.build,
            "bake": self.bake,
            "verify": self.verify,
            "report": [e.to_dict() for e in self.report],
        }


class RigPlanner:
    """Resolve one :class:`RigGraph` against one :class:`RigCapability`.

    Stateless and side-effect free by design: the same graph and capability
    always yield the same :class:`PlanResult`, so a caller can show the plan
    as a pre-flight, let a user cancel, and re-plan later against a different
    target without re-reading either DCC.
    """

    @classmethod
    def plan(cls, graph: RigGraph, capability: RigCapability) -> PlanResult:
        """Decide what *capability*'s target can build from *graph*.

        Parameters:
            graph (RigGraph): the rig's intent.
            capability (RigCapability): what the target can build, as measured
                by its conformance run.

        Returns:
            PlanResult: records to build, nodes to bake, verifications to run,
                and a report entry for every record that did not arrive whole.

        Raises:
            RigPlanRefused: a record whose fallback is ``"fail"`` could not be
                built.
        """
        cyclic = cls._cyclic_nodes(graph)
        buildable: Dict[str, RigRecord] = {}
        baked_nodes: List[str] = []
        report: List[ReportEntry] = []
        verify: Dict[str, Dict[str, Any]] = {}
        considered: List[str] = []  # enabled records: what the component rule sees
        unit = str(graph.source.get("linear_unit", "cm"))

        def bake_targets(record: RigRecord) -> None:
            for node_id in record.target_ids():
                if node_id not in baked_nodes:
                    baked_nodes.append(node_id)

        for record in graph.records:
            policy = graph.effective_policy(record)
            if not policy.enabled:
                continue  # switched off by its author: not a loss, not reported
            considered.append(record.id)

            targets = record.target_ids()
            if any(node_id in cyclic for node_id in targets):
                report.append(
                    ReportEntry(
                        kind="cyclic",
                        reason="cyclic",
                        record=record.id,
                        nodes=targets,
                        detail={"op": record.key},
                    )
                )
                bake_targets(record)
                continue

            failures = capability.rejects(record)
            if failures:
                reason, detail = failures[0]
                if policy.fallback == "fail":
                    raise RigPlanRefused(
                        f"{record.id} ({record.key}): {reason} [{detail}] — "
                        "its policy says this relationship is not optional"
                    )
                dropped = policy.fallback == "drop"
                report.append(
                    ReportEntry(
                        kind="dropped" if dropped else "baked",
                        reason=reason,
                        record=record.id,
                        nodes=targets,
                        detail={
                            "op": record.key,
                            "detail": detail,
                            "also": [f"{r}:{d}" for r, d in failures[1:]],
                        },
                    )
                )
                if not dropped:
                    bake_targets(record)
                continue

            buildable[record.id] = record
            entry = capability.op(record)
            if entry is not None and entry.fidelity != "exact":
                # Never trust an approximate solver silently: what the record
                # asked for becomes mandatory. `policy` has already fallen back
                # to the graph's default, so there is nothing further to consult.
                wanted = dict(policy.verify or {})
                # A tolerance nobody stated is a physical centimetre in the
                # graph's unit, never "1.0" of whatever unit that happens to be.
                wanted.setdefault("tolerance", RigVerify.default_tolerance(unit))
                verify[record.id] = dict(wanted, fidelity=entry.fidelity)

        cls._demote_to_fixpoint(buildable, baked_nodes, report)
        cls._demote_components(graph, considered, buildable, baked_nodes, report)

        # A demoted record is baked, so whatever it was going to be measured
        # against is moot.
        verify = {k: v for k, v in verify.items() if k in buildable}

        for record in graph.records:
            if record.id in buildable:
                report.append(
                    ReportEntry(
                        kind="native",
                        reason="built",
                        record=record.id,
                        nodes=record.target_ids(),
                        detail={
                            "op": record.key,
                            **(
                                {"verify": verify[record.id]}
                                if record.id in verify
                                else {}
                            ),
                        },
                    )
                )

        return PlanResult(
            build=[r.id for r in graph.records if r.id in buildable],
            bake=baked_nodes,
            verify=verify,
            report=report,
        )

    @staticmethod
    def _demote_components(
        graph: RigGraph,
        considered: List[str],
        buildable: Dict[str, RigRecord],
        baked_nodes: List[str],
        report: List[ReportEntry],
    ) -> None:
        """The component rule: a rig component is all-or-nothing.

        A component (:meth:`RigGraph.components`) with one record that bakes,
        drops or cycles is baked WHOLE. The records above the baked link would
        drive a bake and the ones below would follow one, so what remained
        would not be a rig but a bake with constraints as clutter -- measured
        on a production module: 119 records built and verified, not one of
        them driving anything an animator could use, because every chain's
        spline IK and auto-bend math had baked underneath them. Runs after the
        node rule; a record it demotes shares every node it bakes with its
        component, so no further fix-point pass is needed.
        """
        why = {e.record: e.reason for e in report if e.record}
        for component in graph.components(considered):
            blockers = [rid for rid in component if rid not in buildable]
            if not blockers or len(blockers) == len(component):
                continue
            for record_id in component:
                record = buildable.pop(record_id, None)
                if record is None:
                    continue
                for node_id in record.target_ids():
                    if node_id not in baked_nodes:
                        baked_nodes.append(node_id)
                report.append(
                    ReportEntry(
                        kind="baked",
                        reason="component",
                        record=record_id,
                        nodes=record.target_ids(),
                        detail={
                            "op": record.key,
                            "blocker": blockers[0],
                            "blocker_reason": why.get(blockers[0], "unknown"),
                            "component": len(component),
                        },
                    )
                )

    @staticmethod
    def _demote_to_fixpoint(
        buildable: Dict[str, RigRecord],
        baked_nodes: List[str],
        report: List[ReportEntry],
    ) -> None:
        """Bake every record left driving a node that is already baked.

        A node with one baked driver cannot be left partly procedural, and
        demoting a record bakes its OTHER targets too — so this repeats until
        nothing changes rather than sweeping once.
        """
        while True:
            doomed = [
                record
                for record in buildable.values()
                if any(node_id in baked_nodes for node_id in record.target_ids())
            ]
            if not doomed:
                return
            for record in doomed:
                del buildable[record.id]
                report.append(
                    ReportEntry(
                        kind="baked",
                        reason="node_baked",
                        record=record.id,
                        nodes=record.target_ids(),
                        detail={
                            "op": record.key,
                            "detail": "another driver of this node could not be built",
                        },
                    )
                )
                for node_id in record.target_ids():
                    if node_id not in baked_nodes:
                        baked_nodes.append(node_id)

    @staticmethod
    def _cyclic_nodes(graph: RigGraph) -> Set[str]:
        """Node ids inside a dependency cycle, found ONCE for the whole graph.

        Tarjan's SCC, iterative so a deep rig cannot exhaust the stack. A
        component of more than one node is a cycle; so is a self-edge. A cycle
        the rigger MEANT still plays once baked — it is only frozen.

        Returns:
            set: the node ids no record may drive procedurally.
        """
        adjacency: Dict[str, List[str]] = {}
        self_edges: Set[str] = set()
        for source, target in graph.iter_edges():
            if source == target:
                self_edges.add(source)
            adjacency.setdefault(source, []).append(target)
            adjacency.setdefault(target, [])

        index: Dict[str, int] = {}
        low: Dict[str, int] = {}
        on_stack: Set[str] = set()
        stack: List[str] = []
        counter = 0
        cyclic: Set[str] = set(self_edges)

        for root in adjacency:
            if root in index:
                continue
            # (node, iterator over its successors) — an explicit call stack.
            work: List[Tuple[str, int]] = [(root, 0)]
            index[root] = low[root] = counter
            counter += 1
            stack.append(root)
            on_stack.add(root)
            while work:
                node, position = work[-1]
                successors = adjacency[node]
                if position < len(successors):
                    work[-1] = (node, position + 1)
                    nxt = successors[position]
                    if nxt not in index:
                        index[nxt] = low[nxt] = counter
                        counter += 1
                        stack.append(nxt)
                        on_stack.add(nxt)
                        work.append((nxt, 0))
                    elif nxt in on_stack:
                        low[node] = min(low[node], index[nxt])
                    continue
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    component: List[str] = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.append(member)
                        if member == node:
                            break
                    if len(component) > 1:
                        cyclic.update(component)
        return cyclic
