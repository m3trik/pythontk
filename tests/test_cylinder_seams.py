# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.geo_utils.uv.cylinder_seams (CylinderSeams).

The seamer works on plain arrays, so every case here is a mesh built in code:
revolved profiles (capped / open / stepped / chamfered) and a torus. The
numbers pin the behavior the seamer had when it moved out of mayatk
(``uv_utils/_cylinder_seams.py``), whose Maya tests (``test_uv_utils.py``)
drive the same core through ``UvUtils.get_auto_seam_edges``.
"""

import math
import unittest

import numpy as np

import pythontk as ptk
from pythontk.geo_utils.uv.cylinder_seams import DEFAULT_VIEW_DIR, RING


def _edge_table(faces):
    """Unique edges in face order -- the table seam ids index into."""
    seen, edges = {}, []
    for f in faces:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen[key] = len(edges)
                edges.append(key)
    return edges


def _revolve(profile, sides=12, caps=True):
    """A turned part: ``profile`` ``[(radius, z), ...]`` swept about Z, with
    n-gon caps on both ends when ``caps``. Returns ``(points, faces, edges)``."""
    pts = [
        (
            r * math.cos(2 * math.pi * j / sides),
            r * math.sin(2 * math.pi * j / sides),
            z,
        )
        for r, z in profile
        for j in range(sides)
    ]
    faces = []
    for i in range(len(profile) - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            faces.append([a, b, b + sides, a + sides])
    if caps:
        faces.append(list(reversed(range(sides))))
        top = (len(profile) - 1) * sides
        faces.append(list(range(top, top + sides)))
    return pts, faces, _edge_table(faces)


def _torus(major=2.0, minor=0.5, nu=16, nv=8):
    pts = []
    for i in range(nu):
        u = 2 * math.pi * i / nu
        for j in range(nv):
            v = 2 * math.pi * j / nv
            rr = major + minor * math.cos(v)
            pts.append((rr * math.cos(u), rr * math.sin(u), minor * math.sin(v)))
    faces = []
    for i in range(nu):
        for j in range(nv):
            a, b = i * nv + j, i * nv + (j + 1) % nv
            c, d = ((i + 1) % nu) * nv + (j + 1) % nv, ((i + 1) % nu) * nv + j
            faces.append([a, d, c, b])
    return pts, faces, _edge_table(faces)


def _shell_count(seamer, cuts):
    """Faces joined across every interior edge that is not cut."""
    parent = list(range(len(seamer.faces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e, fs in seamer.edge_faces.items():
        if len(fs) == 2 and e not in cuts:
            parent[find(fs[0])] = find(fs[1])
    return len({find(i) for i in range(len(seamer.faces))})


CAPPED = [(1, 0), (1, 2)]
STEPPED = [(1, 0), (1, 1), (0.6, 1), (0.6, 2)]
CHAMFER = [(1, 0), (1, 1), (0.6, 1.4), (0.6, 2.4)]  # a 45 degree chamfer


class TestCylinderSeams(unittest.TestCase):
    def _seam(self, mesh, **options):
        seamer = ptk.CylinderSeams(*mesh)
        cuts = seamer.seams(**options)
        return seamer, cuts

    def test_capped_cylinder_is_two_caps_and_one_strip(self):
        seamer, cuts = self._seam(_revolve(CAPPED))
        # Both cap rings (12 each) + one lengthwise column edge.
        self.assertEqual(len(cuts), 25)
        self.assertEqual(_shell_count(seamer, cuts), 3)
        self.assertEqual(seamer.cuts, cuts)

    def test_open_tube_gets_only_the_column(self):
        seamer, cuts = self._seam(_revolve([(1, 0), (1, 1), (1, 2)], caps=False))
        # The rims are 3D boundaries (already UV borders): only the column
        # (one edge per band row) is cut, and the tube unrolls as one strip.
        self.assertEqual(len(cuts), 2)
        self.assertFalse(cuts & seamer.boundary)
        self.assertEqual(_shell_count(seamer, cuts), 1)

    def test_step_is_a_closed_annulus(self):
        seamer, cuts = self._seam(_revolve(STEPPED))
        # 2 caps + 2 walls + the flat step, which gets no lengthwise cut.
        self.assertEqual(_shell_count(seamer, cuts), 5)
        self.assertEqual(len(cuts), 50)

    def test_chamfer_is_a_cone_cut_open_unless_flat_angle_keeps_it(self):
        seamer, cuts = self._seam(_revolve(CHAMFER))
        self.assertEqual(_shell_count(seamer, cuts), 5)
        self.assertEqual(len(cuts), 51)  # the cone carries its own column edge
        _, ring_cuts = self._seam(_revolve(CHAMFER), flat_angle=40.0)
        self.assertEqual(len(ring_cuts), 50)
        self.assertEqual(len(cuts - ring_cuts), 1)

    def test_torus_opens_with_one_ring_and_one_column(self):
        seamer, cuts = self._seam(_torus())
        self.assertEqual(len(cuts), 8 + 16)
        self.assertEqual(_shell_count(seamer, cuts), 1)

    @staticmethod
    def _column(seamer):
        """The one lengthwise (non-ring) cut of a capped cylinder."""
        (col,) = [e for e in seamer.cuts if seamer.edge_class.get(e) != RING]
        return col

    def test_default_column_faces_away_from_the_default_view(self):
        seamer, _ = self._seam(_revolve(CAPPED))
        offset = seamer.edge_mid(self._column(seamer)) - seamer.pts.mean(axis=0)
        self.assertLess(float(np.dot(offset, DEFAULT_VIEW_DIR)), 0.0)

    def test_invert_and_camera_move_only_the_column(self):
        mesh = _revolve(CAPPED)
        _, cuts = self._seam(mesh)
        _, inverted = self._seam(mesh, invert_seam=True)
        _, from_x = self._seam(mesh, camera=(10.0, 0.0, 1.0))
        for other in (inverted, from_x):
            self.assertEqual(len(other), len(cuts))
            self.assertEqual(len(cuts - other), 1)
        # Inverted, the column faces the default view instead of hiding from it.
        seamer, _ = self._seam(mesh, invert_seam=True)
        offset = seamer.edge_mid(self._column(seamer)) - seamer.pts.mean(axis=0)
        self.assertGreater(float(np.dot(offset, DEFAULT_VIEW_DIR)), 0.0)
        # Seen from +X, the hidden column sits on the -X side.
        seamer, _ = self._seam(mesh, camera=(10.0, 0.0, 1.0))
        self.assertLess(float(seamer.edge_mid(self._column(seamer))[0]), 0.0)

    def test_seed_uvs_cover_every_face_vertex(self):
        for mesh in (_revolve(CAPPED), _revolve(STEPPED), _torus()):
            seamer, _ = self._seam(mesh)
            seeds = seamer.seed_uvs()
            self.assertEqual(sorted(seeds), list(range(len(seamer.faces))))
            for f, uvs in seeds.items():
                self.assertEqual(len(uvs), len(seamer.faces[f]))
                self.assertTrue(np.all(np.isfinite(np.asarray(uvs, dtype=float))))

    def test_hard_edges_only_count_when_authored(self):
        pts, faces, edges = _revolve([(1, 0), (1, 1), (1, 2)])
        # Everything hard = no authoring signal: identical to no flags.
        _, plain = self._seam((pts, faces, edges))
        _, all_hard = self._seam((pts, faces, edges, range(len(edges))))
        self.assertEqual(plain, all_hard)


if __name__ == "__main__":
    unittest.main()
