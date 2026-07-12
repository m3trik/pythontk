# !/usr/bin/python
# coding=utf-8
"""Sort separated mesh parts into repeated-assembly copies.

DCC-neutral clustering behind auto-instancing's ``separate_combined`` flow:
given per-part features (bbox, topology counts, world surface area, center,
material key), decide which parts belong to which copy of a repeated
assembly. The DCC adapters (mayatk/blendertk ``AssemblyReconstructor``)
extract the features and turn the returned index groups into scene
groups/parents; everything here is pure feature-dict math.

Part-feature contract — each part is a dict with:

    ``idx``      int — position in the parts list (== its own index)
    ``bbox``     [xmin, ymin, zmin, xmax, ymax, zmax] world-space AABB
    ``topo``     (nverts, nfaces) tuple — extended in place to
                 (nverts, nfaces, area_class) by :meth:`sort`
    ``area``     float — world-space surface area (rotation invariant)
    ``center``   np.ndarray(3) — world bbox center
    ``volume``   float — world bbox volume
    ``material`` Optional[str] — deterministic material identity key
                 (sorted composite for multi-material parts)

Extra keys (e.g. a DCC node handle) are carried through untouched.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
from functools import reduce
from math import gcd

logger = logging.getLogger(__name__)


class AssemblySorter:
    """Cluster separated parts into copies of repeated assemblies.

    Algorithm (see :meth:`sort`):

    1. Group parts into connected components of the SAME-MATERIAL bbox-touch
       graph. Restricting edges to same-material pairs matters: two assembly
       copies that only connect through a different-material bridge part (a
       deck, a mounting plate) must not fuse into one component — splitting a
       pure-touch component by material afterwards left such cliques glued
       together with no edges between them.
    2. Split genuinely fused components (copies that touch each other) by
       GCD of topology counts, assigning parts touch-first, then by
       internal-distance consistency, then by proximity.
    3. Recover orphaned copies (air gaps) from counts and exemplar
       distances.
    4. Keep a multi-part group only when its part multiset repeats in at
       least one other group (cross-copy support) — an assembly group exists
       to instance copies, so a one-off cluster of touching parts is
       returned as loose singles instead of a speculative assembly.
    """

    def __init__(self, search_radius_mult: float = 1.5, verbose: bool = False):
        self.search_radius_mult = search_radius_mult
        self.verbose = verbose

    def sort(self, parts: List[Dict[str, Any]]) -> List[List[int]]:
        """Sort *parts* into assembly groups; returns lists of part indices.

        Mutates each part's ``topo`` in place, appending a surface-area
        class id (see :meth:`_assign_area_classes`). Single-part groups are
        loose parts; multi-part groups are one assembly copy each. Parts
        that fail every assignment pass are omitted from the result (they
        remain the caller's to keep in the scene).
        """
        if not parts:
            return []

        # Refine part identity with a rotation-invariant size class: vertex/
        # face counts alone alias different parts built the same way (a
        # canister body and its lid are both 12-span cylinders), which
        # poisons every count-based decision downstream. World surface area
        # distinguishes them and, unlike the axis-aligned bbox, is identical
        # for rotated copies.
        self._assign_area_classes(parts)

        adjacency = self._build_adjacency(parts)
        components = self._bfs_group(parts, adjacency)

        if self.verbose:
            logger.info(
                "Connectivity grouping: %s same-material components",
                len(components),
            )

        # Repair connectivity artifacts: parts with a small air gap to their
        # own assembly can chain-link into a fused component (plus orphans)
        # whose counts defeat the GCD split below.
        components = self._repair_fractured_components(parts, components)

        # A component whose FULL part multiset repeats in another component
        # is one copy of an assembly — keep it whole. Without this, a copy
        # with internal symmetry (two brackets + two clips) has gcd 2 and
        # would be halved into two sub-assemblies. Genuinely fused stacks
        # (two copies touching) have no scene-level twin of their doubled
        # multiset, so they still fall through to the count split.
        comp_freq: Dict[Tuple, int] = defaultdict(int)
        for component in components:
            if len(component) > 1:
                comp_freq[self._group_type_key(parts, component)] += 1

        final_groups: List[List[int]] = []
        for component in components:
            if (
                len(component) > 1
                and comp_freq[self._group_type_key(parts, component)] >= 2
            ):
                final_groups.append(component)
                continue
            final_groups.extend(self._split_by_count(parts, component, adjacency))

        # Reassemble orphaned single parts that show consistent internal
        # distances (parts whose air gap kept them out of every component).
        final_groups = self._recover_orphan_assemblies(parts, final_groups)

        # Cross-copy support gate: dissolve speculative one-off groups.
        final_groups = self._dissolve_unsupported_groups(parts, final_groups)

        if self.verbose:
            logger.info("Final assembly count: %s", len(final_groups))

        return final_groups

    @staticmethod
    def _assign_area_classes(parts: List[Dict], rel_tol: float = 1e-3) -> None:
        """Extend each part's ``topo`` key with a surface-area class id.

        Areas are clustered by relative gap (not bucketed by rounding, which
        would split identical copies that straddle a bucket edge): sorted
        areas start a new class when the step from the previous area exceeds
        *rel_tol*. Identical copies differ only by evaluation noise (~1e-7
        relative), far below the threshold.
        """
        order = sorted(range(len(parts)), key=lambda i: parts[i]["area"])
        cls = 0
        prev = None
        for i in order:
            area = parts[i]["area"]
            if prev is not None and area > prev * (1.0 + rel_tol) + 1e-9:
                cls += 1
            parts[i]["topo"] = parts[i]["topo"][:2] + (cls,)
            prev = area

    @staticmethod
    def _topo_counts(parts: List[Dict], indices: List[int]) -> Dict[Tuple, int]:
        counts: Dict[Tuple, int] = defaultdict(int)
        for idx in indices:
            counts[parts[idx]["topo"]] += 1
        return counts

    @classmethod
    def _group_type_key(cls, parts: List[Dict], indices: List[int]) -> Tuple:
        """Hashable assembly-type identity: material + part-topology multiset."""
        return (
            parts[indices[0]]["material"],
            frozenset(cls._topo_counts(parts, indices).items()),
        )

    def _repair_fractured_components(
        self, parts: List[Dict], components: List[List[int]]
    ) -> List[List[int]]:
        """Merge fused-but-fractured components with their nearby orphans.

        A "fractured fusion" holds more copies of a part type than one
        assembly should (multiple assemblies chained together) while its own
        topology counts have gcd 1 (some parts of the chain landed in other
        components), so ``_split_by_count`` would give up on it. Pool such
        components with adjacent single-part components so the counts become
        splittable again.
        """
        import numpy as np

        by_material: Dict[Optional[str], List[int]] = defaultdict(list)
        for i, comp in enumerate(components):
            by_material[parts[comp[0]]["material"]].append(i)

        merged: Set[int] = set()
        result: List[List[int]] = []

        for comp_ids in by_material.values():
            all_indices = [idx for i in comp_ids for idx in components[i]]
            union_counts = self._topo_counts(parts, all_indices)
            union_gcd = reduce(gcd, union_counts.values())
            if union_gcd < 2 or len(comp_ids) < 2:
                continue
            pattern = {t: c // union_gcd for t, c in union_counts.items()}

            fractured = []
            for i in comp_ids:
                counts = self._topo_counts(parts, components[i])
                comp_gcd = reduce(gcd, counts.values())
                if comp_gcd == 1 and any(
                    c > pattern.get(t, 0) for t, c in counts.items()
                ):
                    fractured.append(i)
            if not fractured:
                continue

            # Pool fractured components plus orphan (single-part) components
            # that sit within a part-sized radius of the fractured set.
            pool = list(fractured)
            pooled_indices = [idx for i in fractured for idx in components[i]]
            for i in comp_ids:
                if i in pool or len(components[i]) != 1:
                    continue
                idx = components[i][0]
                center = parts[idx]["center"]
                near = any(
                    np.linalg.norm(center - parts[p]["center"])
                    <= self._anchor_size(parts[p]["bbox"]) * self.search_radius_mult
                    for p in pooled_indices
                )
                if near:
                    pool.append(i)

            if len(pool) < 2:
                continue
            if self.verbose:
                logger.debug(
                    "Repairing fractured fusion: merging %s components", len(pool)
                )
            result.append([idx for i in pool for idx in components[i]])
            merged.update(pool)

        result.extend(comp for i, comp in enumerate(components) if i not in merged)
        return result

    def _recover_orphan_assemblies(
        self, parts: List[Dict], groups: List[List[int]]
    ) -> List[List[int]]:
        """Rebuild assemblies from orphaned single parts.

        Single-part groups whose topologies occur in equal counts (>= 2) are
        candidate copies of one assembly type. They are combined ONLY when a
        consistent part→anchor distance (within the search radius) explains
        every copy — no greedy fallback, so unrelated parts that merely share
        a count are left alone.
        """
        singles = [g[0] for g in groups if len(g) == 1]
        multi = [g for g in groups if len(g) > 1]
        if len(singles) < 2:  # a single orphan can never form a group
            return groups

        by_material: Dict[Optional[str], List[int]] = defaultdict(list)
        for idx in singles:
            by_material[parts[idx]["material"]].append(idx)

        recovered: List[List[int]] = []
        leftover: List[int] = []

        for indices in by_material.values():
            counts = self._topo_counts(parts, indices)
            # Classes of topologies sharing the same repeat count.
            by_count: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
            for topo, count in counts.items():
                by_count[count].append(topo)

            claimed: Set[int] = set()
            for count, topos in by_count.items():
                if count < 2 or len(topos) < 2:
                    continue

                class_members = [
                    idx for idx in indices if parts[idx]["topo"] in topos
                ]
                anchor_topo = max(
                    topos,
                    key=lambda t: max(
                        parts[i]["volume"]
                        for i in class_members
                        if parts[i]["topo"] == t
                    ),
                )
                anchors = [
                    i for i in class_members if parts[i]["topo"] == anchor_topo
                ]
                others = [
                    i for i in class_members if parts[i]["topo"] != anchor_topo
                ]

                clusters: Dict[int, List[int]] = {a: [a] for a in anchors}
                cluster_topo_counts: Dict[int, Dict[Tuple[int, int], int]] = {
                    a: defaultdict(int) for a in anchors
                }
                expected = {t: 1 for t in topos}

                self._assign_consistent_distances(
                    parts, anchors, others, clusters, cluster_topo_counts, expected
                )

                for anchor, members in clusters.items():
                    if len(members) > 1:
                        recovered.append(members)
                        claimed.update(members)
                    # Lone anchors fall through to leftover below.

            leftover.extend(idx for idx in indices if idx not in claimed)

        # Second chance: orphans without enough copies among themselves can
        # still reproduce the internal-distance pattern of assemblies that
        # DID form (e.g. one canister whose lid never bbox-touched its body,
        # amid several correctly reconstructed canisters).
        leftover, bound = self._bind_orphans_to_exemplars(
            parts, multi + recovered, leftover
        )
        recovered.extend(bound)

        if recovered and self.verbose:
            logger.debug(
                "Orphan recovery rebuilt %s assemblies from single parts",
                len(recovered),
            )
        return multi + recovered + [[idx] for idx in leftover]

    def _bind_orphans_to_exemplars(
        self, parts: List[Dict], exemplars: List[List[int]], leftover: List[int]
    ) -> Tuple[List[int], List[List[int]]]:
        """Attach orphan parts by matching learned assembly-internal distances.

        Learns (material, anchor_topo, part_topo) → internal distance from
        already-formed multi-part groups, then binds orphan parts that
        reproduce it. A single exemplar group is enough to form a rule, but a
        bound copy must reproduce the exemplar's COMPLETE part multiset — a
        partial reproduction proves nothing and would glue random parts to a
        lone anchor. (Downstream, the cross-copy support gate and full
        geometric verification keep a coincidental binding from ever
        instancing wrong geometry.)
        """
        import numpy as np

        if not exemplars or len(leftover) < 2:
            return leftover, []

        samples: Dict[Tuple, List[float]] = defaultdict(list)
        group_hits: Dict[Tuple, List[int]] = defaultdict(list)
        exemplar_multisets: Dict[Tuple, Set[frozenset]] = defaultdict(set)
        for group in exemplars:
            anchor = max(group, key=lambda i: parts[i]["volume"])
            key_base = (parts[anchor]["material"], parts[anchor]["topo"])
            in_group: Dict[Tuple, int] = defaultdict(int)
            for p in group:
                if p == anchor:
                    continue
                dist = float(
                    np.linalg.norm(parts[p]["center"] - parts[anchor]["center"])
                )
                key = key_base + (parts[p]["topo"],)
                samples[key].append(dist)
                in_group[key] += 1
            for key, n in in_group.items():
                group_hits[key].append(n)
            exemplar_multisets[key_base].add(
                frozenset(self._topo_counts(parts, group).items())
            )

        # A rule needs a consistent distance across every exemplar sample.
        rules: Dict[Tuple, Tuple[float, float, int]] = {}
        for key, dists in samples.items():
            dists_sorted = sorted(dists)
            median = dists_sorted[len(dists_sorted) // 2]
            eps = max(median * 0.05, 1e-4)
            if all(abs(d - median) <= eps for d in dists):
                rules[key] = (median, eps, max(group_hits[key]))
        if not rules:
            return leftover, []

        bound: List[List[int]] = []
        used: Set[int] = set()
        for anchor in leftover:
            if anchor in used:
                continue
            key_base = (parts[anchor]["material"], parts[anchor]["topo"])
            my_rules = {k: v for k, v in rules.items() if k[:2] == key_base}
            if not my_rules:
                continue

            members = [anchor]
            for (mat, _a_topo, p_topo), (median, eps, budget) in my_rules.items():
                candidates = []
                for p in leftover:
                    if (
                        p in used
                        or p == anchor
                        or parts[p]["topo"] != p_topo
                        or parts[p]["material"] != mat
                    ):
                        continue
                    dist = float(
                        np.linalg.norm(parts[p]["center"] - parts[anchor]["center"])
                    )
                    if abs(dist - median) <= eps:
                        candidates.append((abs(dist - median), p))
                for _, p in sorted(candidates)[:budget]:
                    members.append(p)

            # Commit only complete copies of an exemplar multiset.
            member_multiset = frozenset(self._topo_counts(parts, members).items())
            if len(members) > 1 and member_multiset in exemplar_multisets[key_base]:
                used.update(members)
                bound.append(members)

        remaining = [p for p in leftover if p not in used]
        return remaining, bound

    def _dissolve_unsupported_groups(
        self, parts: List[Dict], groups: List[List[int]]
    ) -> List[List[int]]:
        """Dissolve multi-part groups whose design never repeats.

        An assembly group exists to represent a repeated design. A group is
        SUPPORTED when at least one other group shares its raw topology
        multiset AND has PROPORTIONAL part areas: rigid copies pair area for
        area (ratio ~1), uniformly scaled copies (three sizes of one case
        design) pair at a constant ratio, while a junk chain that merely
        shares part counts pairs at inconsistent ratios. Unsupported groups
        return to loose parts — a connected chain of one-offs must not
        masquerade as an assembly (and, with combining enabled, get merged
        into an arbitrary unit).
        """
        import numpy as np

        def sig(group: List[int]) -> Tuple:
            counts: Dict[Tuple, int] = defaultdict(int)
            for i in group:
                counts[parts[i]["topo"][:2]] += 1
            return (parts[group[0]]["material"], frozenset(counts.items()))

        def area_vector(group: List[int]) -> "np.ndarray":
            # Sorted by (topology, area): proportional scaling preserves the
            # area order within a topology class, so position i pairs the
            # corresponding part across copies.
            order = sorted(
                group, key=lambda i: (parts[i]["topo"][:2], parts[i]["area"])
            )
            return np.array([max(parts[i]["area"], 1e-12) for i in order])

        buckets: Dict[Tuple, List[int]] = defaultdict(list)
        for gi, g in enumerate(groups):
            if len(g) > 1:
                buckets[sig(g)].append(gi)

        supported: Set[int] = set()
        REL = 0.02  # part-to-part ratio spread tolerated within one design
        for gis in buckets.values():
            if len(gis) < 2:
                continue
            vecs = {gi: area_vector(groups[gi]) for gi in gis}
            pool = list(gis)
            while pool:
                seed = pool.pop(0)
                cluster = [seed]
                rest = []
                for gi in pool:
                    ratios = vecs[gi] / vecs[seed]
                    if float(ratios.max()) <= float(ratios.min()) * (1.0 + REL):
                        cluster.append(gi)
                    else:
                        rest.append(gi)
                pool = rest
                if len(cluster) > 1:
                    supported.update(cluster)

        out: List[List[int]] = []
        for gi, g in enumerate(groups):
            if len(g) > 1 and gi not in supported:
                if self.verbose:
                    logger.debug(
                        "Dissolving unsupported %s-part group (one-off design)",
                        len(g),
                    )
                out.extend([i] for i in g)
            else:
                out.append(g)
        return out

    @staticmethod
    def _anchor_size(bbox: List[float]) -> float:
        """Largest bbox dimension — a proxy for the anchor's physical size."""
        return max(bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])

    def _build_adjacency(self, parts: List[Dict]) -> Dict[int, List[int]]:
        """Adjacency graph: bbox touch between SAME-material parts (vectorized).

        Restricting edges to same-material pairs keeps a different-material
        bridge (deck, mounting plate) from fusing unrelated same-material
        clusters into one component. The old pure-touch graph split by
        material afterwards produced exactly those phantom fusions: two
        disjoint same-material cliques in one "component" with no edge
        between them, which the count-based splitter then mis-sorted.
        """
        import numpy as np

        adjacency: Dict[int, List[int]] = defaultdict(list)
        if len(parts) < 2:
            return adjacency

        boxes = np.array([p["bbox"] for p in parts])  # (n, 6): min xyz, max xyz
        mins, maxs = boxes[:, :3], boxes[:, 3:]
        tol = 0.01
        # Boxes touch when they overlap (within tol) on every axis.
        touch = np.all(
            (mins[:, None, :] <= maxs[None, :, :] + tol)
            & (mins[None, :, :] <= maxs[:, None, :] + tol),
            axis=2,
        )
        mat_ids: Dict[Optional[str], int] = {}
        mat_idx = np.array(
            [mat_ids.setdefault(p["material"], len(mat_ids)) for p in parts]
        )
        touch &= mat_idx[:, None] == mat_idx[None, :]
        for i, j in np.argwhere(np.triu(touch, k=1)):
            adjacency[int(i)].append(int(j))
            adjacency[int(j)].append(int(i))
        return adjacency

    @staticmethod
    def _bfs_group(
        parts: List[Dict], adjacency: Dict[int, List[int]]
    ) -> List[List[int]]:
        """Group parts by connectivity using BFS."""
        visited: Set[int] = set()
        groups: List[List[int]] = []

        for start in range(len(parts)):
            if start in visited:
                continue
            queue = deque([start])
            group = []
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                group.append(node)
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            groups.append(group)

        return groups

    def _split_by_count(
        self,
        parts: List[Dict],
        indices: List[int],
        adjacency: Optional[Dict[int, List[int]]] = None,
    ) -> List[List[int]]:
        """Split a fused component into assembly copies from part counts.

        Tries three count models, finest first:
        1. Full identity (topology + area class) — distinguishes a body from
           its same-topology lid.
        2. The repeating CORE of the full identity when one-off extras
           chained into the stack (a base under stacked canisters, a pallet
           under stacked suitcases) defeat the GCD; extras return as loose
           singles for orphan recovery. Only taken when the core forms the
           MAJORITY of the component — a lone assembly whose only repeat is
           a symmetric part pair (two identical clasps on one case) must not
           be shattered into per-pair fragments.
        3. Raw topology (nv, nf) — uniformly SCALED copies (three sizes of
           one case design) share topology but not area classes; the coarse
           counts recover the per-copy split while the touch and distance
           passes keep each size's parts together.
        """
        if len(indices) <= 1:
            return [indices]

        def full_key(idx: int) -> Tuple:
            return parts[idx]["topo"]

        def coarse_key(idx: int) -> Tuple:
            return parts[idx]["topo"][:2]

        def key_counts(idxs, key):
            counts: Dict[Tuple, int] = defaultdict(int)
            for i in idxs:
                counts[key(i)] += 1
            return counts

        counts = key_counts(indices, full_key)
        gcd_val = reduce(gcd, counts.values())
        if gcd_val >= 2:
            return self._split_with_key(
                parts, indices, adjacency, counts, gcd_val, [], full_key
            )

        # The core must span >= 2 distinct identities: a single repeated topo
        # (two identical clips on one bracket, two clasps on one case) is a
        # symmetric part PAIR within one assembly, and "splitting" it would
        # just pair the parts up and shatter their assembly.
        core = {t: c for t, c in counts.items() if c >= 2}
        if len(core) >= 2:
            core_gcd = reduce(gcd, core.values())
            if core_gcd >= 2 and sum(core.values()) * 2 > len(indices):
                extras = [i for i in indices if counts[full_key(i)] < 2]
                kept = [i for i in indices if counts[full_key(i)] >= 2]
                split = self._split_with_key(
                    parts, kept, adjacency, core, core_gcd, extras, full_key
                )
                # A core split is speculative — validate that every cluster
                # is touch-connected. Disconnected clusters mean the core
                # classes belong to DIFFERENT copies (e.g. the big copy's
                # clasp pair grouped with the small copy's) and the split is
                # nonsense; fall through to the coarse model instead.
                if adjacency is None or all(
                    self._cluster_connected(g, adjacency)
                    for g in split
                    if len(g) > 1
                ):
                    return split

        # Same >=2-distinct-identities rule for the coarse model, and it must
        # actually merge identities (else it is the same counts again).
        coarse = key_counts(indices, coarse_key)
        coarse_gcd = reduce(gcd, coarse.values())
        if coarse_gcd >= 2 and len(coarse) >= 2 and len(coarse) < len(counts):
            return self._split_with_key(
                parts, indices, adjacency, coarse, coarse_gcd, [], coarse_key
            )

        return [indices]

    @staticmethod
    def _cluster_connected(
        indices: List[int], adjacency: Dict[int, List[int]]
    ) -> bool:
        """True when *indices* form one touch-connected component."""
        if len(indices) <= 1:
            return True
        members = set(indices)
        seen = {indices[0]}
        stack = [indices[0]]
        while stack:
            for j in adjacency.get(stack.pop(), ()):
                if j in members and j not in seen:
                    seen.add(j)
                    stack.append(j)
        return len(seen) == len(members)

    def _split_with_key(
        self,
        parts: List[Dict],
        indices: List[int],
        adjacency: Optional[Dict[int, List[int]]],
        topo_counts: Dict[Tuple, int],
        gcd_val: int,
        extras: List[int],
        key,
    ) -> List[List[int]]:
        """Split *indices* into ``gcd_val`` clusters under identity *key*."""
        import numpy as np

        expected_per_cluster = {
            topo: count // gcd_val for topo, count in topo_counts.items()
        }

        # Find anchors (largest volume identity class)
        largest_topo = max(
            topo_counts.keys(),
            key=lambda t: max(parts[i]["volume"] for i in indices if key(i) == t),
        )
        anchors = [idx for idx in indices if key(idx) == largest_topo]

        # Calculate distinguishability for each part
        def get_distinguishability(idx: int) -> float:
            center = parts[idx]["center"]
            dists = sorted(
                [np.linalg.norm(center - parts[a]["center"]) for a in anchors]
            )
            if len(dists) < 2:
                return float("inf")
            return dists[1] - dists[0]  # Higher = more distinguishable

        # Initialize clusters with anchors
        clusters: Dict[int, List[int]] = {anchor: [anchor] for anchor in anchors}
        cluster_topo_counts: Dict[int, Dict[Tuple[int, int], int]] = {
            anchor: defaultdict(int) for anchor in anchors
        }
        for anchor in anchors:
            cluster_topo_counts[anchor][largest_topo] = 1

        assigned = set(anchors)
        remaining = [idx for idx in indices if idx not in assigned]

        # Touch-first pass: a part that physically touches exactly one
        # cluster belongs to it — contact beats any distance heuristic.
        # Stacked copies (two suitcases, one atop the other) put a clasp at
        # near-identical distances from both bodies; scalar distances tie or
        # alias and the clasps swap, while the touch graph is unambiguous
        # (each clasp touches only its own body).
        if adjacency is not None:
            remaining = self._grow_by_touch(
                adjacency,
                remaining,
                clusters,
                cluster_topo_counts,
                key=key,
            )

        # Distance-consistency pass next: identical assemblies place a part
        # type at the SAME internal distance from the anchor in every copy,
        # so the correct part→anchor pairing shows up as the distance value
        # shared by all copies. Plain nearest-anchor assignment mis-sorts
        # tightly packed layouts (e.g. stacked assemblies where a lid sits
        # closer to the neighbour's body than to its own).
        remaining = self._assign_consistent_distances(
            parts,
            anchors,
            remaining,
            clusters,
            cluster_topo_counts,
            expected_per_cluster,
            key=key,
        )

        # Greedy proximity fallback for whatever the consistency pass could
        # not settle. Sort by distinguishability: MOST distinguishable first.
        remaining.sort(key=get_distinguishability, reverse=True)

        for idx in remaining:
            topo = key(idx)
            center = parts[idx]["center"]

            # Anchors that still need this topology AND are within the
            # configured search radius.
            candidates = []
            for anchor in anchors:
                if cluster_topo_counts[anchor][topo] >= expected_per_cluster[topo]:
                    continue
                dist = np.linalg.norm(center - parts[anchor]["center"])
                max_dist = (
                    self._anchor_size(parts[anchor]["bbox"]) * self.search_radius_mult
                )
                if dist <= max_dist:
                    candidates.append((anchor, dist))

            if candidates:
                best_anchor = min(candidates, key=lambda x: x[1])[0]
            else:
                best_anchor = self._fallback_anchor(
                    parts,
                    anchors,
                    cluster_topo_counts,
                    expected_per_cluster,
                    topo,
                    center,
                )
                if best_anchor is None:
                    continue  # Truly too far from every anchor — leave unassigned

            clusters[best_anchor].append(idx)
            cluster_topo_counts[best_anchor][topo] += 1

        return list(clusters.values()) + [[e] for e in extras]

    @staticmethod
    def _grow_by_touch(
        adjacency: Dict[int, List[int]],
        remaining: List[int],
        clusters: Dict[int, List[int]],
        cluster_topo_counts: Dict[int, Dict[Tuple, int]],
        key,
    ) -> List[int]:
        """Iteratively assign parts that touch exactly ONE cluster.

        Grows every cluster along the touch graph in rounds. Deliberately
        NOT budget-capped: copies routinely differ in how many of their
        parts register bbox contact (a thin handle can air-gap on the
        straight copy and touch on the rotated one), and a uniform budget
        then forces one copy's genuinely-attached part onto a cluster it
        never touched. Physical contact with a single cluster is attachment
        evidence, period. Parts touching zero clusters (air gap) or several
        (shared contact between stacked copies) are left for the distance
        passes. Returns the unassigned remainder.
        """
        member_of: Dict[int, int] = {}
        for anchor, members in clusters.items():
            for m in members:
                member_of[m] = anchor

        pending = list(remaining)
        changed = True
        while changed and pending:
            changed = False
            deferred = []
            for idx in pending:
                touched = {
                    member_of[j] for j in adjacency.get(idx, ()) if j in member_of
                }
                if len(touched) == 1:
                    anchor = next(iter(touched))
                    clusters[anchor].append(idx)
                    cluster_topo_counts[anchor][key(idx)] += 1
                    member_of[idx] = anchor
                    changed = True
                    continue
                deferred.append(idx)
            pending = deferred
        return pending

    def _assign_consistent_distances(
        self,
        parts: List[Dict],
        anchors: List[int],
        remaining: List[int],
        clusters: Dict[int, List[int]],
        cluster_topo_counts: Dict[int, Dict[Tuple, int]],
        expected_per_cluster: Dict[Tuple, int],
        key=None,
    ) -> List[int]:
        """Assign parts to anchors by internal-distance consistency.

        For each part identity class (*key*; defaults to the full topo),
        bucket every part→anchor distance (within the search radius) and
        look for one distance value that yields a complete budget-respecting
        assignment of ALL parts of that class. Because the metric is a
        distance it is rotation-invariant, so it works for arbitrarily
        rotated copies of the same assembly.

        Returns the parts that could not be settled this way (for the
        greedy fallback).
        """
        import numpy as np

        if key is None:
            key = lambda idx: parts[idx]["topo"]  # noqa: E731
        by_topo: Dict[Tuple, List[int]] = defaultdict(list)
        for idx in remaining:
            by_topo[key(idx)].append(idx)

        mean_anchor_size = float(
            np.mean([self._anchor_size(parts[a]["bbox"]) for a in anchors])
        )
        eps = max(mean_anchor_size * 0.02, 1e-5)

        unassigned: List[int] = []
        for topo, members in by_topo.items():
            expected = expected_per_cluster.get(topo, 0)
            if expected <= 0 or len(anchors) < 2:
                unassigned.extend(members)
                continue

            # Bucket candidate pairs by quantized distance.
            buckets: Dict[int, List[Tuple[float, int, int]]] = defaultdict(list)
            for idx in members:
                center = parts[idx]["center"]
                for anchor in anchors:
                    dist = float(np.linalg.norm(center - parts[anchor]["center"]))
                    max_dist = (
                        self._anchor_size(parts[anchor]["bbox"])
                        * self.search_radius_mult
                    )
                    if dist <= max_dist:
                        buckets[int(round(dist / eps))].append((dist, idx, anchor))

            # Try the most-populated distance values first; commit the first
            # one that covers every member within the per-cluster budget.
            # Adjacent buckets are included per attempt: equal real-world
            # distances that straddle a quantization boundary (float noise,
            # rotated-bbox center wobble) would otherwise split across two
            # buckets and neither would cover all members.
            assignment: Optional[Dict[int, int]] = None
            for bucket_key in sorted(buckets, key=lambda k: (-len(buckets[k]), k)):
                pairs = (
                    buckets.get(bucket_key - 1, [])
                    + buckets[bucket_key]
                    + buckets.get(bucket_key + 1, [])
                )
                trial: Dict[int, int] = {}
                trial_counts: Dict[int, int] = defaultdict(int)
                for dist, idx, anchor in sorted(pairs):
                    if idx in trial:
                        continue
                    if (
                        cluster_topo_counts[anchor][topo] + trial_counts[anchor]
                        >= expected
                    ):
                        continue
                    trial[idx] = anchor
                    trial_counts[anchor] += 1
                if len(trial) == len(members):
                    assignment = trial
                    break

            if assignment is None:
                unassigned.extend(members)
                continue

            if self.verbose:
                logger.debug(
                    "Distance-consistency assigned %s part(s) of topo %s",
                    len(assignment),
                    topo,
                )
            for idx, anchor in assignment.items():
                clusters[anchor].append(idx)
                cluster_topo_counts[anchor][topo] += 1

        return unassigned

    def _fallback_anchor(
        self,
        parts: List[Dict],
        anchors: List[int],
        cluster_topo_counts: Dict[int, Dict[Tuple[int, int], int]],
        expected_per_cluster: Dict[Tuple[int, int], int],
        topo: Tuple[int, int],
        center: "Any",
    ) -> Optional[int]:
        """Nearest anchor within a relaxed radius when the strict pass found none.

        Anchors that still have budget for *topo* are preferred over the
        absolute nearest. When ``search_radius_mult`` is small (< 1.25) the
        user has asked for strict separation (e.g. touching assemblies), so
        the relaxed multiplier is not applied.
        """
        import numpy as np

        effective_mult = self.search_radius_mult
        if self.search_radius_mult >= 1.25:
            effective_mult *= 2.0

        with_budget = [
            a
            for a in anchors
            if cluster_topo_counts[a][topo] < expected_per_cluster[topo]
        ]
        for pool in (with_budget, anchors):
            if not pool:
                continue
            best = min(pool, key=lambda a: np.linalg.norm(center - parts[a]["center"]))
            dist = np.linalg.norm(center - parts[best]["center"])
            if dist <= self._anchor_size(parts[best]["bbox"]) * effective_mult:
                return best
        return None
