# !/usr/bin/python
# coding=utf-8
"""The golden cases every port of the record name rule is held to.

A record's entries (the ``lightmaps`` record's ``objects``, the ``shadows``
record's ``planes``, sources and contacts, the ``articulation`` record's
joints and grabbed parts) find their nodes by one rule,
:meth:`_Media.resolve_entries`: the exact name first, then the leaf (the
name after the last ``:``); objects sharing a name told apart by the entry's
``hierarchy`` against where each node sits; every set of entries whose
candidates overlap settled together; a tie never guessed. It runs outside
Python -- the WebXR viewer's ``kernel/resolve.js``, and every engine reader
of a delivery that ports it -- and a port is only as good
as the cases that pin it, so they are generated from the reference at test
time (``Conformance.cases("record_names")``).

Every case is one file's nodes -- each a name, its parent's index (or
``None``) and whether it carries a mesh (only those are candidates; every
node is a parent) -- and one record's entries; what the reference makes of
them is one result per entry: its status (``exact``, ``leaf``,
``ambiguous``, ``absent``), the node it binds (an index, or ``None``) and,
for a tie, where each candidate sits (its names from the file's root down,
joined by ``/``, sorted). The named cases come first -- one per clause of the
rule -- then random ones, so every earlier case keeps its place.
"""

import random
from typing import Any, Dict, List, Optional, Sequence

from pythontk.file_utils.mesh_convert._media import _Media


class RecordNamesConformance:
    """The golden cases of :meth:`_Media.resolve_entries`."""

    #: Names only: nothing to round, so no quantity has a bound.
    TOLERANCE: Dict[str, float] = {}

    #: The names the random cases draw from: a leaf reused under several
    #: namespaces, so the leaf rule, its ties and the hierarchy all bind.
    NAMES = ("BODY", "lid", "ns:BODY", "rf:BODY", "ns:lid", "grp", "ns:grp", "")

    @classmethod
    def case(
        cls,
        nodes: Sequence[Sequence[Any]],
        entries: Sequence[Any],
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """One case: *nodes* as ``(name, parent, mesh)`` rows, *entries* as
        the record states them, and the reference's answer for each entry."""
        rows = [
            {"name": n, "parent": p, "mesh": bool(m)} for n, p, m in nodes
        ]
        gltf: Dict[str, Any] = {"nodes": []}
        for row in rows:
            node: Dict[str, Any] = {"name": row["name"]}
            if row["mesh"]:
                node["mesh"] = 0
            gltf["nodes"].append(node)
        for index, row in enumerate(rows):
            if row["parent"] is not None:
                gltf["nodes"][row["parent"]].setdefault("children", []).append(index)
        media = _Media(None)
        by_name, leaves, _users, lineage = media.node_index(gltf, meshes=True)
        index_of = {id(node): i for i, node in enumerate(gltf["nodes"])}
        expected = []
        for found, status, candidates in media.resolve_entries(
            list(entries), by_name, leaves, lineage
        ):
            bound = index_of[id(found[0])] if found and len(found) == 1 else None
            if status in ("exact", "leaf") and bound is None:
                raise AssertionError(f"a {status} match bound no single node")
            expected.append(
                {
                    "status": status,
                    "node": bound,
                    "candidates": list(candidates) if status == "ambiguous" else [],
                }
            )
        return {
            "name": name,
            "input": {"nodes": rows, "entries": list(entries)},
            "expected": expected,
        }

    @classmethod
    def named(cls) -> List[Dict[str, Any]]:
        """One case per clause of the rule."""
        c = cls.case
        return [
            c([("BODY", None, 1)], [{"name": "BODY"}], "exact"),
            c([("ns:BODY", None, 1)], [{"name": "BODY"}], "leaf"),
            c([("ns:BODY", None, 1)], [{"name": "rf:BODY"}], "leaf_other_namespace"),
            c([("BODY", None, 1)], [{"name": "body"}], "case_sensitive"),
            c([("a:BODY", None, 1), ("b:BODY", None, 1)], [{"name": "BODY"}], "leaf_tie"),
            # An exact name owns its node; the leaf entry is left its other candidate's tie.
            c(
                [("hand", None, 1), ("other:hand", None, 1)],
                [{"name": "hand"}, {"name": "x:hand"}],
                "exact_owns_its_node",
            ),
            c(
                [("hand", None, 1)],
                [{"name": "hand"}, {"name": "other:hand"}],
                "exact_beats_a_leaf",
            ),
            c(
                [("a:hand", None, 1), ("b:hand", None, 1)],
                [{"name": "c:hand"}, {"name": "d:hand"}],
                "two_leaves_tie",
            ),
            # Two objects named alike, told apart by where each sits.
            c(
                [("roomA", None, 0), ("BODY", 0, 1), ("roomB", None, 0), ("BODY", 2, 1)],
                [
                    {"name": "BODY", "hierarchy": ["roomA", "BODY"]},
                    {"name": "BODY", "hierarchy": ["roomB", "BODY"]},
                ],
                "hierarchy_tells_apart",
            ),
            # A stale third entry whose best candidates both went to better matches.
            c(
                [("roomA", None, 0), ("BODY", 0, 1), ("roomB", None, 0), ("BODY", 2, 1)],
                [
                    {"name": "BODY", "hierarchy": ["roomA", "BODY"]},
                    {"name": "BODY", "hierarchy": ["roomB", "BODY"]},
                    {"name": "BODY", "hierarchy": ["BODY"]},
                ],
                "stale_entry_absent",
            ),
            c(
                [("roomA", None, 0), ("BODY", 0, 1), ("roomB", None, 0), ("BODY", 2, 1)],
                [{"name": "BODY"}],
                "same_name_no_hierarchy_tie",
            ),
            # The hierarchy's namespaces are blind; an extra file root is not compared.
            c(
                [("RootNode", None, 0), ("ns:grp", 0, 0), ("BODY", 1, 1), ("BODY", 0, 1)],
                [{"name": "BODY", "hierarchy": ["grp", "BODY"]}],
                "hierarchy_namespace_blind",
            ),
            # An exact name beats a leaf where the hierarchy cannot tell them apart.
            c(
                [("PROPS_DA:prop352", None, 1), ("PROPS_RF:prop352", None, 1)],
                [{"name": "PROPS_DA:prop352", "hierarchy": ["PROPS_DA:prop352"]}],
                "exact_breaks_a_hierarchy_tie",
            ),
            c([("", None, 1)], [{"name": ""}, {}, {"name": None}], "nameless_entries_absent"),
            c([("BODY", None, 0)], [{"name": "BODY"}], "no_mesh_no_candidate"),
            c([("lid", None, 1)], [{"name": "BODY"}], "absent"),
            # Two groups that do not overlap settle apart.
            c(
                [("BODY", None, 1), ("lid", None, 1), ("ns:lid", None, 1)],
                [{"name": "BODY"}, {"name": "x:lid"}],
                "separate_groups",
            ),
        ]

    @classmethod
    def random_case(cls, rng: random.Random, index: int) -> Dict[str, Any]:
        """A random file of up to 9 nodes and up to 4 entries."""
        nodes = []
        for i in range(rng.randint(1, 9)):
            parent = rng.randrange(i) if i and rng.random() < 0.6 else None
            nodes.append((rng.choice(cls.NAMES), parent, rng.random() < 0.7))
        entries = []
        for _ in range(rng.randint(1, 4)):
            entry: Dict[str, Any] = {"name": rng.choice(cls.NAMES[:-1])}
            roll = rng.random()
            if roll < 0.4:
                # The lineage of a node, as a scene would state it, maybe cut short.
                at: Optional[int] = rng.randrange(len(nodes))
                chain = []
                while at is not None:
                    chain.insert(0, nodes[at][0])
                    at = nodes[at][1]
                entry["hierarchy"] = chain[rng.randrange(len(chain)) :]
            elif roll < 0.6:
                entry["hierarchy"] = [rng.choice(cls.NAMES) for _ in range(rng.randint(1, 3))]
            entries.append(entry)
        return cls.case(nodes, entries, f"random_{index}")

    @classmethod
    def cases(cls, seed: int = 0, count: int = 64) -> Dict[str, Any]:
        """The conformance document: ``{"tolerance", "cases"}``.

        Parameters:
            seed: The random stream's seed -- the same seed, the same cases.
            count: Random cases after the named ones.
        """
        rng = random.Random(seed)
        cases = cls.named() + [cls.random_case(rng, i) for i in range(count)]
        return {"tolerance": dict(cls.TOLERANCE), "cases": cases}
