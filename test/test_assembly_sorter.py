# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.engines.instancing.assembly_sorter
(AssemblySorter) — the DCC-neutral clustering behind auto-instancing's
separate_combined flow. Scenarios mirror mayatk's TestAssemblySorting regression
class, built here from synthetic part-feature dicts instead of Maya scenes.
"""

import unittest

from pythontk import AssemblySorter

try:
    import numpy as np
except ImportError:
    np = None


def part(idx, center, size, topo, area, material="matA", volume=None):
    """Build a part-feature dict from center + size (axis-aligned box)."""
    cx, cy, cz = center
    sx, sy, sz = size
    bbox = [cx - sx / 2, cy - sy / 2, cz - sz / 2, cx + sx / 2, cy + sy / 2, cz + sz / 2]
    return {
        "idx": idx,
        "bbox": bbox,
        "topo": tuple(topo),
        "area": float(area),
        "center": np.array(center, dtype=float),
        "volume": float(volume) if volume is not None else sx * sy * sz,
        "material": material,
    }


def groups_as_sets(groups):
    return sorted(
        (sorted(g) for g in groups), key=lambda g: (len(g), g)
    )


@unittest.skipIf(np is None, "numpy not available")
class TestAssemblySorter(unittest.TestCase):
    def sort(self, parts, **kw):
        return AssemblySorter(**kw).sort(parts)

    def test_empty(self):
        self.assertEqual(self.sort([]), [])

    def test_stacked_bodies_keep_their_clasps(self):
        """Touch disambiguation: two stacked 3-part units; each clasp stays
        with the body it physically touches (mirror of the stacked-suitcase
        regression)."""
        parts = [
            # Bodies: 6x2x4 cubes stacked at y=1 and y=3 (touch at y=2).
            part(0, (0, 1, 0), (6, 2, 4), (8, 6), area=88.0),
            part(1, (0, 3, 0), (6, 2, 4), (8, 6), area=88.0),
            # Clasps on the front face (z=2) of each body.
            part(2, (-2, 1, 2.05), (1, 1, 0.1), (8, 6), area=2.4),
            part(3, (2, 1, 2.05), (1, 1, 0.1), (8, 6), area=2.4),
            part(4, (-2, 3, 2.05), (1, 1, 0.1), (8, 6), area=2.4),
            part(5, (2, 3, 2.05), (1, 1, 0.1), (8, 6), area=2.4),
        ]
        groups = groups_as_sets(self.sort(parts))
        self.assertEqual(groups, [[0, 2, 3], [1, 4, 5]])

    def test_material_bridge_does_not_fuse(self):
        """A different-material deck touching two units must not fuse them
        into one component."""
        parts = [
            part(0, (0, 0, 0), (2, 2, 2), (8, 6), area=24.0),
            part(1, (0, 1.5, 0), (1, 1, 1), (26, 24), area=6.0),
            part(2, (10, 0, 0), (2, 2, 2), (8, 6), area=24.0),
            part(3, (10, 1.5, 0), (1, 1, 1), (26, 24), area=6.0),
            # Deck spanning both bodies, different material.
            part(4, (5, -1.05, 0), (14, 0.1, 2), (8, 6), area=30.0, material="matB"),
        ]
        groups = groups_as_sets(self.sort(parts))
        self.assertEqual(groups, [[4], [0, 1], [2, 3]])

    def test_one_off_cluster_stays_loose(self):
        """Three touching unique-size parts: no repeated design, no group."""
        parts = [
            part(0, (0, 0, 0), (2, 2, 2), (8, 6), area=24.0),
            part(1, (2, 0, 0), (2, 2, 2), (8, 6), area=30.0),
            part(2, (4, 0, 0), (2, 2, 2), (8, 6), area=38.0),
        ]
        groups = groups_as_sets(self.sort(parts))
        self.assertEqual(groups, [[0], [1], [2]])

    def test_scaled_copies_form_supported_assemblies(self):
        """Uniformly scaled copies (1.0 / 0.8 / 1.25) each keep their own
        parts; the cross-copy support gate pairs them by proportional area."""
        parts = []
        idx = 0
        for x, s in ((0, 1.0), (10, 0.8), (20, 1.25)):
            body_size = 2 * s
            knob_size = 0.6 * s
            parts.append(
                part(
                    idx,
                    (x, 0, 0),
                    (body_size,) * 3,
                    (8, 6),
                    area=24.0 * s * s,
                )
            )
            idx += 1
            for kx in (-0.5 * s, 0.5 * s):
                parts.append(
                    part(
                        idx,
                        (x + kx, body_size / 2 + knob_size / 2 - 0.005, 0),
                        (knob_size,) * 3,
                        (26, 24),
                        area=2.16 * s * s,
                    )
                )
                idx += 1
        groups = groups_as_sets(self.sort(parts))
        self.assertEqual(groups, [[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    def test_orphan_recovery_by_consistent_distance(self):
        """Air-gapped lids (no bbox touch) are recovered onto their bodies
        by internal-distance consistency."""
        parts = [
            part(0, (0, 0, 0), (4, 4, 4), (8, 6), area=96.0),
            part(1, (20, 0, 0), (4, 4, 4), (8, 6), area=96.0),
            # Lids float 1.5 above the body tops (gap > touch tol).
            part(2, (0, 3.5, 0), (1, 1, 1), (26, 24), area=6.0),
            part(3, (20, 3.5, 0), (1, 1, 1), (26, 24), area=6.0),
        ]
        groups = groups_as_sets(self.sort(parts))
        self.assertEqual(groups, [[0, 2], [1, 3]])

    def test_sort_appends_area_class_to_topo(self):
        parts = [
            part(0, (0, 0, 0), (2, 2, 2), (8, 6), area=24.0),
            part(1, (10, 0, 0), (2, 2, 2), (8, 6), area=48.0),
        ]
        self.sort(parts)
        self.assertEqual(len(parts[0]["topo"]), 3)
        # Different areas -> different classes.
        self.assertNotEqual(parts[0]["topo"][2], parts[1]["topo"][2])


if __name__ == "__main__":
    unittest.main()
