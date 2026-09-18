# !/usr/bin/python
# coding=utf-8
"""The RigGraph document — a DCC-agnostic statement of a rig's INTENT.

NO DCC IMPORTS. This is the bottom of the rig-transfer stack; mayatk extracts
into it and blendertk builds from it, and neither can import the other.

The shape every record shares:

    a TARGET receives a value computed by an OPERATOR from some SOURCES.

Targets come in three kinds -- a transform's channels, a single plug, a set of
points -- plus ``order`` (evaluation order) and ``opaque`` (a driver that was
recognised but cannot be described). Those five shapes are validated strictly
here; the ``op`` inside a shape is a registry key validated against a target's
:class:`~pythontk.core_utils.engines.rig_graph.rig_capability.RigCapability`,
never against this module. That is what lets the vocabulary grow without
touching the schema, the validator, or the planner.

Identity is the payload's prim path, so a record joins to a carrier prim -- and
through it to a built object -- with no name matching, no namespace rewriting
and no collision-suffix tolerance. A node's parent is its path's parent, so
there is no second spelling of the hierarchy to keep consistent.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

#: Bumped only when a SHAPE or the envelope changes meaning. Operators are
#: additive and unversioned -- an unknown one is a planner outcome, not an error.
SCHEMA_VERSION = 1

#: The five record shapes. Strictly validated; everything else is open.
SHAPES = ("transform", "channel", "points", "order", "opaque")

#: What a record may do when its target cannot build it.
FALLBACKS = ("bake", "drop", "fail")

#: Used when neither the record nor the graph states one. Baking keeps the
#: motion and loses only editability, which is the safe answer.
DEFAULT_FALLBACK = "bake"

#: Callables a restricted ``expr`` may use. Deliberately small: arithmetic,
#: comparison and these. ``select(c, a, b)`` returns ``a`` or ``b`` -- a
#: FUNCTION, not control flow -- and is what lets a condition node translate.
#: Everything outside this is reported and baked, which is the line that stops
#: the grammar becoming a programming language.
EXPR_FUNCTIONS = frozenset(
    {
        "abs",
        "min",
        "max",
        "clamp",
        "floor",
        "ceil",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "atan2",
        "lerp",
        "remap",
        "select",
    }
)

# AST node types a restricted expression may contain. Anything else -- attribute
# access, subscripting, comprehensions, lambdas, walrus -- is refused, so no
# escape into the object graph exists to begin with.
_EXPR_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


@dataclass
class RigPolicy:
    """What to do with one record when its target cannot build it.

    Carried at the envelope's top level as the graph's DEFAULT and on any record
    that overrides it field by field. Defaults live in the GRAPH, never in a
    target's capability manifest: a manifest says what a target can do, a graph
    says how strictly it must be held to that, and letting a target supply both
    would let it quietly relax the standard it is measured against.
    """

    #: Every field is UNSET (None) until someone states it. That distinction is
    #: load-bearing: filling concrete defaults here would make a record that
    #: says nothing override the graph with a default it never asked for, and
    #: the graph-level default would never apply to anything.
    enabled: Optional[bool] = None
    fallback: Optional[str] = None
    #: ``{"tolerance": float, "frames": [...]}`` -- measure the built result
    #: against ground truth and fall back on a miss. The planner ATTACHES this
    #: to any record whose op did not grade ``exact``, so an approximate
    #: solver is never trusted silently.
    verify: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RigPolicy":
        """Build from plain values; an absent key stays UNSET, not defaulted."""
        data = data or {}
        enabled = data.get("enabled")
        fallback = data.get("fallback")
        return cls(
            enabled=None if enabled is None else bool(enabled),
            fallback=None if fallback is None else str(fallback),
            verify=data.get("verify"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Plain-value form, omitting every field nobody set."""
        out: Dict[str, Any] = {}
        for key in ("enabled", "fallback", "verify"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    def merged(self, default: "RigPolicy") -> "RigPolicy":
        """This policy over *default*, field by field; unset fields take *default*'s.

        Parameters:
            default (RigPolicy): the policy to fall back to.

        Returns:
            RigPolicy: a new policy, still possibly unset where neither stated a
                value -- :meth:`resolved` is what fills those.
        """
        return RigPolicy(
            enabled=self.enabled if self.enabled is not None else default.enabled,
            fallback=self.fallback if self.fallback is not None else default.fallback,
            verify=self.verify if self.verify is not None else default.verify,
        )

    def resolved(self) -> "RigPolicy":
        """This policy with every remaining unset field filled by the schema.

        Returns:
            RigPolicy: fully determined -- what a planner reads.
        """
        return RigPolicy(
            enabled=True if self.enabled is None else self.enabled,
            fallback=self.fallback or DEFAULT_FALLBACK,
            verify=self.verify,
        )


@dataclass
class RigNode:
    """A node a record references. Not the whole scene -- only what is named,
    plus the ancestors those paths imply."""

    id: str
    path: str = ""
    kind: str = "transform"
    rest: Dict[str, Any] = field(default_factory=dict)
    shape: Optional[Dict[str, Any]] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
    locks: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    @property
    def parent(self) -> Optional[str]:
        """The parent node id, derived from this one's path."""
        return RigGraph.parent_of(self.id)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RigNode":
        """Build from plain values."""
        return cls(
            id=str(data.get("id", "")),
            path=str(data.get("path", "")),
            kind=str(data.get("kind", "transform")),
            rest=dict(data.get("rest") or {}),
            shape=data.get("shape"),
            attrs=dict(data.get("attrs") or {}),
            locks=dict(data.get("locks") or {}),
            labels=dict(data.get("labels") or {}),
            tags=list(data.get("tags") or []),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Plain-value form, omitting empty optional sections."""
        out: Dict[str, Any] = {"id": self.id, "kind": self.kind}
        for key in ("path", "rest", "attrs", "locks", "labels", "tags"):
            value = getattr(self, key)
            if value:
                out[key] = value
        if self.shape:
            out["shape"] = self.shape
        return out


@dataclass
class RigRecord:
    """One relationship: a target receives a value computed by an op from sources."""

    id: str
    shape: str
    op: str
    target: Any = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    policy: RigPolicy = field(default_factory=RigPolicy)
    #: Source-app nodes this record was read FROM, by name. Not used to
    #: resolve anything -- it exists so :meth:`RigGraph.coverage` can prove the
    #: extractor accounted for every driver it saw.
    provenance: List[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """``"<shape>/<op>"`` -- how a capability manifest names this record."""
        return f"{self.shape}/{self.op}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RigRecord":
        """Build from plain values."""
        return cls(
            id=str(data.get("id", "")),
            shape=str(data.get("shape", "")),
            op=str(data.get("op", "")),
            target=data.get("target"),
            sources=[dict(s) for s in (data.get("sources") or [])],
            params=dict(data.get("params") or {}),
            policy=RigPolicy.from_dict(data.get("policy")),
            provenance=list(data.get("provenance") or []),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Plain-value form."""
        out: Dict[str, Any] = {"id": self.id, "shape": self.shape, "op": self.op}
        if self.target is not None:
            out["target"] = self.target
        if self.sources:
            out["sources"] = self.sources
        if self.params:
            out["params"] = self.params
        policy = self.policy.to_dict()
        if policy:  # nobody stated one: say nothing rather than a default
            out["policy"] = policy
        if self.provenance:
            out["provenance"] = self.provenance
        return out

    def target_ids(self) -> List[str]:
        """Node ids this record DRIVES, whatever its shape.

        Returns:
            list: node ids, in the order the record names them.
        """
        target = self.target
        if target is None:
            return []
        if isinstance(target, str):  # `channel`: a plug
            return [RigGraph.split_plug(target)[0]]
        if isinstance(target, dict):
            if "chain" in target:  # a solver writing a whole chain
                return [str(i) for i in target.get("chain") or []]
            if "ids" in target:  # `opaque`
                return [str(i) for i in target.get("ids") or []]
            if "id" in target:
                return [str(target["id"])]
        return []

    def source_ids(self) -> List[str]:
        """Node ids this record READS, from both source entries and plug values.

        Returns:
            list: node ids, de-duplicated, order preserved.
        """
        found: List[str] = []
        for source in self.sources:
            if isinstance(source.get("id"), str):
                found.append(source["id"])
            if isinstance(source.get("plug"), str):
                found.append(RigGraph.split_plug(source["plug"])[0])
        for _, plug in self.plug_paths():
            found.append(RigGraph.split_plug(plug)[0])
        seen: Dict[str, None] = {}
        for node_id in found:
            seen.setdefault(node_id, None)
        return list(seen)

    def node_ids(self) -> List[str]:
        """Every node id this record touches -- targets first, then sources --
        de-duplicated. What decides which rig COMPONENT it belongs to
        (:meth:`RigGraph.components`)."""
        seen: Dict[str, None] = {}
        for node_id in self.target_ids() + self.source_ids():
            seen.setdefault(node_id, None)
        return list(seen)

    def plug_paths(self) -> List[Tuple[str, str]]:
        """Every ``{"plug": ...}`` in this record, as ``(dotted path, plug)``.

        The dotted path is what a capability's ``plugs`` list names -- so
        ``sources.weight`` covers that key on ANY source, and nested params
        read ``params.twist.start``. It is the planner's rule-7 input.

        Returns:
            list: ``(path, plug)`` pairs, sources first.
        """
        found: List[Tuple[str, str]] = []

        def walk(value: Any, path: str) -> None:
            if RigGraph.is_plug(value):
                found.append((path, value["plug"]))
            elif isinstance(value, dict):
                for key, sub in value.items():
                    walk(sub, f"{path}.{key}")
            elif isinstance(value, (list, tuple)):
                for sub in value:
                    walk(sub, path)  # a list shares its parent's path

        for source in self.sources:
            for key, value in source.items():
                if key != "plug":  # the source's own reference, not a parameter
                    walk(value, f"sources.{key}")
        walk(self.params, "params")
        return found


@dataclass
class RigGraph:
    """One scene's rig intent: what exists, and what drives what."""

    version: int = SCHEMA_VERSION
    source: Dict[str, Any] = field(default_factory=dict)
    policy: RigPolicy = field(default_factory=RigPolicy)
    nodes: List[RigNode] = field(default_factory=list)
    records: List[RigRecord] = field(default_factory=list)

    # --- the document's vocabulary -------------------------------------------
    # Plug addressing and the expression grammar are stateless, so they are
    # `@staticmethod`s rather than module-level functions: `CODE_STANDARD.md` §4
    # puts logic on classes reached through the class namespace, and the schema
    # these four define IS this document's. Reached as `RigGraph.split_plug(...)`
    # from the record and node shapes above (resolved at call time, which is why
    # their definitions may precede this class) and from the sibling modules.

    @staticmethod
    def is_plug(value: Any) -> bool:
        """True when *value* is a plug reference rather than a literal.

        Parameters:
            value: any parameter value read from a record.

        Returns:
            bool: whether it is ``{"plug": "<id>.<channel>"}``.
        """
        return isinstance(value, dict) and isinstance(value.get("plug"), str)

    @staticmethod
    def split_plug(plug: str) -> Tuple[str, str]:
        """Split ``"<id>.<channel>"`` into its node id and channel.

        A sanitised prim name cannot contain ``.``, so the FIRST dot ends the path
        and everything after it is the channel -- unambiguous with no escaping,
        which is the reason ids are paths rather than names.

        Parameters:
            plug (str): e.g. ``"/rig/ctrls/ctrl_L.translate.x"``.

        Returns:
            tuple: ``(node_id, channel)``; channel is ``""`` when the string holds
                no dot (a bare node reference).
        """
        head, sep, tail = plug.partition(".")
        return (head, tail) if sep else (plug, "")

    @staticmethod
    def parent_of(node_id: str) -> Optional[str]:
        """The parent path of *node_id*, or None for a root.

        Parameters:
            node_id (str): a prim path, e.g. ``"/rig/ctrls/ctrl_L"``.

        Returns:
            str | None: ``"/rig/ctrls"``, or None when there is no parent left.
        """
        head, sep, _ = node_id.rstrip("/").rpartition("/")
        return head if (sep and head) else None

    @staticmethod
    def validate_expression(expr: str, variables: Sequence[str] = ()) -> List[str]:
        """Check *expr* against the restricted grammar; return the reasons it fails.

        Sibling of :meth:`pythontk.MathUtils.eval_expression`, which is a
        CALCULATOR -- it takes no variables, exposes the whole ``math`` module and
        returns a formatted string. A rig expression is a different contract: named
        variables, comparisons, a deliberately narrower allowlist
        (:data:`EXPR_FUNCTIONS`) and a numeric result. They share only the idea of
        walking an AST and refusing everything not named, so they stay apart rather
        than one being bent into the other.

        Parameters:
            expr (str): the expression source.
            variables (Sequence[str]): names the expression may read (a record's
                source roles).

        Returns:
            list: human-readable failures; empty when the expression is acceptable.
        """
        if not isinstance(expr, str) or not expr.strip():
            return ["expression is empty"]
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as error:
            return [f"expression does not parse ({error.msg})"]

        known = set(variables)
        errors: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None)
                if name not in EXPR_FUNCTIONS:
                    errors.append(
                        f"call to {name or 'a computed callee'!s} is not allowed"
                    )
                if node.keywords:
                    errors.append("keyword arguments are not allowed")
                continue
            if isinstance(node, ast.Name):
                # A call's own callee was handled above; a bare name must be a
                # variable the record actually supplies.
                if node.id not in known and node.id not in EXPR_FUNCTIONS:
                    errors.append(f"unknown variable {node.id!r}")
                continue
            if isinstance(node, ast.Constant):
                if not isinstance(node.value, (int, float)) or isinstance(
                    node.value, bool
                ):
                    errors.append(
                        f"only numeric literals are allowed, got {node.value!r}"
                    )
                continue
            if not isinstance(node, _EXPR_NODES):
                errors.append(f"{type(node).__name__} is not allowed in an expression")
        return errors

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RigGraph":
        """Build from plain values (the serialised envelope)."""
        return cls(
            version=int(data.get("version", SCHEMA_VERSION)),
            source=dict(data.get("source") or {}),
            policy=RigPolicy.from_dict(data.get("policy")),
            nodes=[RigNode.from_dict(n) for n in (data.get("nodes") or [])],
            records=[RigRecord.from_dict(r) for r in (data.get("records") or [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Plain-value form of the whole envelope."""
        out: Dict[str, Any] = {"version": self.version, "source": self.source}
        policy = self.policy.to_dict()
        if policy:
            out["policy"] = policy
        out["nodes"] = [n.to_dict() for n in self.nodes]
        out["records"] = [r.to_dict() for r in self.records]
        return out

    def node(self, node_id: str) -> Optional[RigNode]:
        """The node with *node_id*, or None."""
        for candidate in self.nodes:
            if candidate.id == node_id:
                return candidate
        return None

    def record(self, record_id: str) -> Optional[RigRecord]:
        """The record with *record_id*, or None."""
        for candidate in self.records:
            if candidate.id == record_id:
                return candidate
        return None

    def effective_policy(self, record: RigRecord) -> RigPolicy:
        """*record*'s policy over the graph's, over the schema's — fully resolved.

        Parameters:
            record (RigRecord): the record to resolve for.

        Returns:
            RigPolicy: every field determined, so a planner never has to ask
                again where a value came from.
        """
        return record.policy.merged(self.policy).resolved()

    def validate(self) -> List[str]:
        """Check the envelope and the five shapes; return every failure found.

        Validates only what this schema OWNS: the version, record shapes,
        identity, policy and the restricted expression grammar. Operators and
        their parameters are deliberately NOT checked here -- they are a
        target's business, resolved by the planner against a capability
        manifest, which is what keeps the vocabulary open.

        Returns:
            list: human-readable failures; empty means the document is well
                formed (not that any target can build it).
        """
        errors: List[str] = []
        if self.version != SCHEMA_VERSION:
            errors.append(
                f"unknown schema version {self.version} (this build speaks {SCHEMA_VERSION})"
            )

        seen_nodes: Dict[str, None] = {}
        for node in self.nodes:
            if not node.id:
                errors.append("a node has no id")
            elif node.id in seen_nodes:
                errors.append(f"duplicate node id {node.id!r}")
            seen_nodes.setdefault(node.id, None)

        seen_records: Dict[str, None] = {}
        for record in self.records:
            label = record.id or "<unnamed>"
            if not record.id:
                errors.append("a record has no id")
            elif record.id in seen_records:
                errors.append(f"duplicate record id {record.id!r}")
            seen_records.setdefault(record.id, None)

            if record.shape not in SHAPES:
                errors.append(f"{label}: unknown shape {record.shape!r}")
            if not record.op:
                errors.append(f"{label}: no op")
            if self.effective_policy(record).fallback not in FALLBACKS:
                errors.append(
                    f"{label}: fallback must be one of {', '.join(FALLBACKS)}"
                )
            errors.extend(self._validate_shape(record, label))
            for node_id in list(record.target_ids()) + list(record.source_ids()):
                if node_id and node_id not in seen_nodes:
                    errors.append(f"{label}: references unknown node {node_id!r}")
        return errors

    def _validate_shape(self, record: RigRecord, label: str) -> List[str]:
        """Shape-specific structure for one record."""
        errors: List[str] = []
        target = record.target
        if record.shape == "channel":
            if not isinstance(target, str) or "." not in target:
                errors.append(f"{label}: a channel target must be a plug")
            if record.op == "expr":
                roles = [str(s.get("role", "")) for s in record.sources]
                errors.extend(
                    f"{label}: {reason}"
                    for reason in RigGraph.validate_expression(
                        record.params.get("expr", ""), roles
                    )
                )
        elif record.shape == "transform":
            if not isinstance(target, dict) or not (
                "id" in target or "chain" in target
            ):
                errors.append(f"{label}: a transform target needs an id or a chain")
        elif record.shape == "points":
            if not isinstance(target, dict) or "id" not in target:
                errors.append(f"{label}: a points target needs an id")
        elif record.shape == "order":
            ordered = record.params.get("records")
            if not isinstance(ordered, list):
                errors.append(f"{label}: an order record needs params.records")
            else:
                known = {r.id for r in self.records}
                for other in ordered:
                    if other not in known:
                        errors.append(
                            f"{label}: orders unknown record {other!r} — a dangling "
                            "reference would silently order nothing"
                        )
        elif record.shape == "opaque":
            if not isinstance(target, dict) or not target.get("ids"):
                errors.append(f"{label}: an opaque record must name the ids it freezes")
        return errors

    def coverage(self) -> Dict[str, Any]:
        """Prove the extractor accounted for every driver node it saw.

        "Emit ``opaque`` rather than nothing" is the schema's central rule --
        an omitted driver is indistinguishable from "this node is free", so the
        consumer leaves its target STATIC instead of baking it, and the loss
        report says everything is fine. As prose aimed at extractor authors
        that rule is worth nothing: measured on a production scene, a first-cut
        extractor emitted records for 241 of 539 driver nodes and reported a
        clean bill of health for the 55% it never looked at.

        So an extractor states what it SAW in ``source.census`` (source node
        type -> count) and what each record was read FROM in the record's
        ``provenance``; the difference is what nobody accounted for. A consumer
        that finds any is looking at an incomplete graph, whatever the plan says.

        Provenance is counted as DISTINCT source nodes, never as occurrences:
        two records legitimately read the same constraint, and tallying both
        would let one node stand in for another and report a clean bill of
        health while a real driver went unread -- the very false negative this
        exists to catch.

        Returns:
            dict: ``{"seen": int, "accounted": int, "unaccounted": int,
                "by_type": {type: count_unaccounted},
                "uncensused": {type: count}}``. ``seen`` is 0 when no census was
                recorded, which itself means the graph cannot make this promise.
                ``uncensused`` holds provenance for types the census never
                mentioned -- an extractor contradicting itself, and a reason to
                trust neither number.
        """
        census: Dict[str, int] = dict(self.source.get("census") or {})
        seen_nodes: Dict[str, set] = {}
        for record in self.records:
            for entry in record.provenance:
                node_type, _, node = entry.partition(":")
                seen_nodes.setdefault(node_type, set()).add(node or entry)
        by_type = {}
        for node_type, count in census.items():
            missing = count - len(seen_nodes.get(node_type, ()))
            if missing > 0:
                by_type[node_type] = missing
        return {
            "seen": sum(census.values()),
            "accounted": sum(
                min(len(seen_nodes.get(t, ())), c) for t, c in census.items()
            ),
            "unaccounted": sum(by_type.values()),
            "by_type": by_type,
            "uncensused": {
                node_type: len(nodes)
                for node_type, nodes in seen_nodes.items()
                if node_type not in census
            },
        }

    def components(self, record_ids: Optional[Iterable[str]] = None) -> List[List[str]]:
        """Group records into RIG COMPONENTS: two records share one when they
        touch a common node (as target or source), transitively.

        A component is the unit a consumer can honestly call "a rig": one
        chain from the thing an animator grabs to the thing that deforms.
        Any link in it that bakes leaves the links above driving a bake and
        the links below following one, so the planner (section 9.3, the
        component rule) and the verifier take a component back WHOLE.
        Measured on a production module: 119 records built and verified and
        not one of them driving anything usable, because every chain's spline
        IK and auto-bend math had baked underneath them.

        Parameters:
            record_ids: Restrict to these records (the planner passes the
                enabled ones, so a switched-off record never binds two
                components together). Default: every record.

        Returns:
            list: Components, each a list of record ids in ``records`` order,
                the components themselves ordered by their first record.
        """
        wanted = None if record_ids is None else set(record_ids)
        records = [r for r in self.records if wanted is None or r.id in wanted]
        parent: Dict[str, str] = {}

        def find(key: str) -> str:
            root = key
            while parent.setdefault(root, root) != root:
                root = parent[root]
            while parent[key] != root:  # path compression
                parent[key], key = root, parent[key]
            return root

        for record in records:
            for node_id in record.node_ids():
                parent[find("rec:" + record.id)] = find("node:" + node_id)
        groups: Dict[str, List[str]] = {}
        for record in records:
            groups.setdefault(find("rec:" + record.id), []).append(record.id)
        return list(groups.values())

    def iter_edges(self) -> Iterator[Tuple[str, str]]:
        """``(source node, target node)`` for every dependency a record creates.

        The planner's cycle detection runs on these. ``order`` and ``opaque``
        records create none -- one states sequence and the other states that
        nothing is known.

        Yields:
            tuple: ``(source_id, target_id)``.
        """
        for record in self.records:
            if record.shape in ("order", "opaque"):
                continue
            targets = record.target_ids()
            for source_id in record.source_ids():
                for target_id in targets:
                    yield (source_id, target_id)
