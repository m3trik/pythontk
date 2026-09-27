# !/usr/bin/python
# coding=utf-8
"""The DCC-free half of auto-instancing's grouping pass.

mayatk's and blendertk's ``AutoInstancer`` read geometry signatures from their
scenes and turn the resulting groups into instances; between those two scene
steps sits logic that never touches a scene, and it lives here once:

- merging near-identical signature buckets into candidate groups
  (:meth:`InstanceGrouping.merge_similar_signatures`);
- the run summary a pass reports -- its shape
  (:meth:`InstanceGrouping.default_summary`) and its human-readable text
  (:meth:`InstanceGrouping.format_summary`).

Signature contract -- the tuple both DCC ``GeometryMatcher`` classes build:

    ``(verts, edges, faces, pca_sig, materials, uv_signature)``

``sig[:3]`` is the topology, ``sig[3]`` a PCA eigenvalue signature (a tuple of
floats, or empty / ``None`` when unavailable) and ``sig[4:]`` the identity
components (material, UV sets) that must match exactly for two buckets to merge.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, Hashable, List, Optional, Tuple

_log = logging.getLogger(__name__)

__all__ = ["InstanceGrouping"]


class InstanceGrouping:
    """Signature-bucket merging and run-summary reporting for auto-instancing.

    Stateless: every method is a static function of its plain inputs, so the
    DCC adapters call it with the values they already hold (their own logger,
    their strategy's micro-triangle threshold).
    """

    #: Summed absolute PCA difference below which two SAME-topology buckets merge.
    PCA_ABS_TOL = 0.1
    #: Relative PCA difference below which two buckets merge ACROSS topologies
    #: (the combine-mode rescue for a mesh re-triangulated differently).
    PCA_REL_TOL = 0.005

    @staticmethod
    def merge_similar_signatures(
        signature_map: Dict[Tuple, List[Any]],
        logger: Optional[logging.Logger] = None,
    ) -> Dict[Tuple, List[Any]]:
        """Merge signature buckets that are similar enough.

        Only buckets with identical material and UV signature components are
        merged -- geometric similarity must never override
        ``require_same_material`` / ``check_uvs``. Buckets are visited in
        topology order; each absorbs every later bucket that has the same
        topology and a PCA within :attr:`PCA_ABS_TOL` (or, with no PCA on one
        side, an equal PCA), or a different topology but a PCA within
        :attr:`PCA_REL_TOL` relative difference.

        Parameters:
            signature_map: ``{signature: [candidate, ...]}``.
            logger: Receives the debug line for a cross-topology merge;
                defaults to this module's logger.

        Returns:
            A new ``defaultdict(list)`` ``{surviving signature: [candidates]}``;
            the input map and its lists are not modified.
        """
        logger = logger or _log
        sorted_keys = sorted(signature_map.keys(), key=lambda x: x[:3])

        merged_map: Dict[Tuple, List[Any]] = defaultdict(list)
        processed_sigs = set()

        for i, sig in enumerate(sorted_keys):
            if sig in processed_sigs:
                continue

            merged_map[sig].extend(signature_map[sig])
            processed_sigs.add(sig)

            topo = sig[:3]
            pca = sig[3]

            for j in range(i + 1, len(sorted_keys)):
                other_sig = sorted_keys[j]
                if other_sig in processed_sigs:
                    continue
                if other_sig[4:] != sig[4:]:  # materials / UV sets must match
                    continue

                o_pca = other_sig[3]

                if other_sig[:3] == topo:
                    if pca and o_pca:
                        diff = sum(abs(p1 - p2) for p1, p2 in zip(pca, o_pca))
                        if diff > InstanceGrouping.PCA_ABS_TOL:
                            continue
                    elif pca != o_pca:
                        continue

                    merged_map[sig].extend(signature_map[other_sig])
                    processed_sigs.add(other_sig)

                elif pca and o_pca:
                    diff = sum(abs(p1 - p2) for p1, p2 in zip(pca, o_pca))
                    total_mag = sum(pca) + sum(o_pca) + 0.001
                    rel_diff = diff / total_mag

                    if rel_diff < InstanceGrouping.PCA_REL_TOL:
                        logger.debug(
                            "Merging near-identical signature %s into %s "
                            "(topology differs; combine mode)",
                            other_sig[:3],
                            sig[:3],
                        )
                        merged_map[sig].extend(signature_map[other_sig])
                        processed_sigs.add(other_sig)

        return merged_map

    @staticmethod
    def default_summary(micro_threshold: int) -> Dict[str, object]:
        """A zeroed run summary -- the shape of ``AutoInstancer.last_run_summary``.

        Keys: ``matched_groups`` (viable groups of >=2 identical meshes
        discovered), ``instanced_groups`` / ``instances_created`` (converted to
        shared shapes), ``simple_groups`` (identical but below the
        micro-triangle threshold -- combined instead of instanced),
        ``kept_separate_groups`` (left as-is by the strategy: flagged
        individual / non-static), ``micro_threshold`` (the triangle cutoff in
        force) and ``details`` (per-skipped-group records for the console:
        ``{"name", "reason", "count", "tris"}``).

        Parameters:
            micro_threshold: The triangle cutoff in force (the DCC strategy's
                ``MICRO_TRI_THRESHOLD``).

        Returns:
            A fresh dict (its ``details`` list is never shared).
        """
        return {
            "matched_groups": 0,
            "instanced_groups": 0,
            "instances_created": 0,
            "simple_groups": 0,
            "kept_separate_groups": 0,
            "micro_threshold": micro_threshold,
            "details": [],
        }

    @staticmethod
    def format_summary(
        summary: Dict[str, object], output_count: int, micro_threshold: int
    ) -> str:
        """Human-readable, DCC-agnostic description of a run *summary*.

        Returns ASCII plain text with ``-`` bullets (ASCII so a non-UTF-8
        console ``print`` can't raise) -- a slot shows it in its message box
        (newlines -> ``<br>``) and prints it to the console verbatim.

        Parameters:
            summary: A run summary (:meth:`default_summary` shape).
            output_count: The number of live result nodes the caller kept
                (prototypes + instances + combined remainder).
            micro_threshold: The triangle cutoff reported when *summary*
                carries no ``micro_threshold`` of its own.

        Returns:
            The report, one line per outcome.
        """
        matched = summary.get("matched_groups", 0)
        if matched == 0:
            if output_count > 0:
                return (
                    "Auto Instance: no geometrically identical meshes to "
                    f"instance; combined loose geometry into {output_count} "
                    "mesh(es)."
                )
            return "Auto Instance: no geometrically identical meshes were found."

        micro = summary.get("micro_threshold", micro_threshold)
        details = summary.get("details", []) or []
        simple = [d for d in details if d.get("reason") == "too_simple"]
        kept = [d for d in details if d.get("reason") == "kept_separate"]
        instanced = summary.get("instanced_groups", 0)
        instances = summary.get("instances_created", 0)

        def _names(items: List[Dict[Hashable, Any]]) -> str:
            return ", ".join(
                f"{d['name']} (x{d['count']}, {d['tris']} tris)" for d in items
            )

        lines = [f"Auto Instance: {matched} matching group(s) found."]
        if instanced:
            lines.append(
                f"- Instanced {instanced} group(s) -> {instances} new instance(s)."
            )
        if simple:
            lines.append(
                f"- {len(simple)} group(s) too simple to instance (< {micro} "
                f"tris); combined where possible: {_names(simple)}."
            )
        if kept:
            lines.append(
                f"- {len(kept)} group(s) left separate (flagged individual / "
                f"non-static): {_names(kept)}."
            )
        # Only when no line above explains the outcome (e.g. every group's
        # conversion failed) -- otherwise the reasons above already say it.
        if not (instanced or simple or kept):
            lines.append("- Nothing was instanced.")
        return "\n".join(lines)
