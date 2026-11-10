# !/usr/bin/python
# coding=utf-8
"""Unit tests for MeshOps (PyMeshLab-backed file-level mesh processing).

Strategy (the optional-dep split test_uv_pack.py / test_ktx2_encoder.py
pin): the resolve/available contract and every guard that runs *before*
pymeshlab is touched are covered unconditionally; real-geometry tests are
skipped when pymeshlab is absent; the missing-engine error path is
covered by the inverse skip.

Run with:
    python -m pytest test/test_mesh_ops.py -v
    python test/run_tests.py
"""

import os
import shutil
import unittest

import pythontk as ptk
from pythontk.file_utils.mesh_ops import OPS, MeshOps

_HAVE = MeshOps.available()


class _TempDirTestCase(unittest.TestCase):
    """Per-test scratch dir under ``test/temp_tests/`` (repo convention)."""

    def setUp(self):
        super().setUp()
        self.out_dir = os.path.join(
            os.path.dirname(__file__), "temp_tests", f"meshops_{self._testMethodName}"
        )
        os.makedirs(self.out_dir, exist_ok=True)
        self.addCleanup(shutil.rmtree, self.out_dir, ignore_errors=True)

    # ------------------------------------------------------------ fixtures
    def _sphere(self, name="sphere.obj", subdiv=3, colored=False):
        """Generated sphere OBJ (pymeshlab-only; call from skipUnless tests)."""
        pml = MeshOps.resolve()
        ms = pml.MeshSet()
        ms.create_sphere(radius=1.0, subdiv=subdiv)
        if colored:
            ms.compute_color_by_function_per_vertex(
                x="255*(x+1)/2", y="255*(y+1)/2", z="255*(z+1)/2"
            )
        path = os.path.join(self.out_dir, name)
        ms.save_current_mesh(path)
        return path

    def _jitter_split(self, src, offset=1e-4, name="split.obj"):
        """Every vertex duplicated at ``offset`` — welds only via a real
        close-vertex merge, not exact-duplicate removal."""
        import numpy as np

        pml = MeshOps.resolve()
        ms = pml.MeshSet()
        ms.load_new_mesh(src)
        m = ms.current_mesh()
        v, f = m.vertex_matrix(), m.face_matrix()
        v2 = np.vstack([v, v + offset])
        f2 = np.vstack([f, f + len(v)])
        out = pml.MeshSet()
        out.add_mesh(pml.Mesh(vertex_matrix=v2, face_matrix=f2))
        path = os.path.join(self.out_dir, name)
        out.save_current_mesh(path)
        return path, len(v)


class TestResolveContract(unittest.TestCase):
    """Availability contract — runs with or without pymeshlab installed."""

    def test_resolve_optional_never_raises_import_error(self):
        MeshOps.resolve(required=False)  # must not raise

    def test_available_matches_resolve(self):
        self.assertEqual(
            MeshOps.available(), MeshOps.resolve(required=False) is not None
        )

    def test_install_note_names_the_extra(self):
        self.assertIn("pythontk[mesh]", MeshOps._install_note())

    def test_an_engine_that_cannot_write_is_not_available(self):
        """pymeshlab imports even when its IO plug-ins failed to load (Linux
        without libOpenGL.so.0); every op then failed "Unknown format for
        save" while ``available()`` said True. Probed once, then cached."""
        import sys
        from unittest.mock import MagicMock, patch

        fake = MagicMock()
        fake.MeshSet.return_value.save_current_mesh.side_effect = Exception(
            "Unknown format for save: ply"
        )
        with (
            patch.dict(sys.modules, {"pymeshlab": fake}),
            patch.object(MeshOps, "_io_error", None),
        ):
            self.assertFalse(MeshOps.available())
            with self.assertRaisesRegex(RuntimeError, "libOpenGL.so.0"):
                MeshOps.resolve()
            self.assertEqual(fake.MeshSet.call_count, 1)  # one probe, then cached


class TestPreEngineGuards(unittest.TestCase):
    """Validation that runs before any pymeshlab use — dep-free."""

    def test_preflight_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            MeshOps._preflight(os.path.join("nowhere", "missing.obj"))

    def test_preflight_unsupported_extension(self):
        with self.assertRaises(ValueError):
            MeshOps._preflight(__file__)  # exists, but .py

    def test_decimate_requires_exactly_one_target(self):
        with self.assertRaises(ValueError):
            MeshOps.decimate("x.obj", target_faces=0, target_pct=0.0)
        with self.assertRaises(ValueError):
            MeshOps.decimate("x.obj", target_faces=100, target_pct=50.0)

    def test_run_op_unknown_name(self):
        with self.assertRaises(KeyError):
            MeshOps._run_op(None, None, "not_an_op", {})

    def test_run_op_unknown_param(self):
        with self.assertRaises(TypeError):
            MeshOps._run_op(None, None, "close_holes", {"bogus": 1})

    def test_ops_registry_metric_safe_docs(self):
        # Registry sanity: every spec names a filter and only frozenset params.
        for name, spec in OPS.items():
            self.assertTrue(spec.filter, name)
            self.assertTrue(spec.percent_params <= spec.allowed_params, name)
            self.assertTrue(spec.absolute_params <= spec.allowed_params, name)

    def test_unwritable_output_format_rejected_before_work(self):
        """glTF/GLB load but have no pymeshlab exporter — an output path in a
        read-only format must fail fast with an actionable error, not after a
        long op chain with a raw PyMeshLabException at save time."""
        from pythontk.file_utils.mesh_ops import SAVE_EXTS, SUPPORTED_EXTS

        self.assertTrue(SAVE_EXTS < SUPPORTED_EXTS)  # strictly narrower
        for ext in (".glb", ".gltf"):
            self.assertIn(ext, SUPPORTED_EXTS)
            self.assertNotIn(ext, SAVE_EXTS)
        with self.assertRaises(ValueError) as ctx:
            MeshOps._check_save_ext("scan.glb")
        self.assertIn("read-only", str(ctx.exception))
        MeshOps._check_save_ext("scan.ply")  # writable — must not raise


@unittest.skipIf(_HAVE, "pymeshlab installed - missing-engine path untestable")
class TestMissingEngine(unittest.TestCase):
    def test_resolve_required_raises_runtime_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            MeshOps.resolve(required=True)
        self.assertIn("pythontk[mesh]", str(ctx.exception))


@unittest.skipUnless(_HAVE, "pymeshlab not installed")
class TestMeasure(_TempDirTestCase):
    def test_measure_keys_and_types(self):
        m = MeshOps.measure(self._sphere())
        for key in (
            "faces",
            "vertices",
            "edges",
            "components",
            "boundary_edges",
            "non_two_manifold_edges",
            "holes",
            "surface_area",
            "bbox_diag",
        ):
            self.assertIn(key, m)
        self.assertGreater(m["faces"], 0)
        self.assertTrue(m["two_manifold"])
        self.assertEqual(m["components"], 1)
        self.assertEqual(m["boundary_edges"], 0)

    def test_metric_keys_are_gate_prefix_safe(self):
        # QcGate strips min_/max_ rule prefixes with an unanchored replace;
        # metric names must never contain those substrings — measure() AND
        # compare() alike (hausdorff_peak, not hausdorff_max, for this reason).
        src = self._sphere(subdiv=4)
        keys = set(MeshOps.measure(src))
        dec = MeshOps.decimate(src, target_faces=400)
        keys |= set(MeshOps.compare(dec, src))
        for key in keys:
            self.assertNotIn("min_", key)
            self.assertNotIn("max_", key)

    def test_unmeasured_is_none_never_zero(self):
        # A plane strip is open: volume must be honest, holes computable.
        pml = MeshOps.resolve()
        ms = pml.MeshSet()
        ms.create_grid()  # open plane
        path = os.path.join(self.out_dir, "plane.obj")
        ms.save_current_mesh(path)
        m = MeshOps.measure(path)
        self.assertFalse(m["two_manifold"] and m["boundary_edges"] == 0)


@unittest.skipUnless(_HAVE, "pymeshlab not installed")
class TestClean(_TempDirTestCase):
    def test_merge_distance_is_absolute_units(self):
        """Regression: merge_distance was passed as PercentageValue, making
        the weld a silent no-op (1e-5 read as 0.00001% of the bbox diagonal).
        An absolute-distance weld must actually merge jittered duplicates."""
        src = self._sphere()
        split, n_orig = self._jitter_split(src, offset=1e-4)
        self.assertEqual(MeshOps.measure(split)["vertices"], n_orig * 2)
        cleaned = MeshOps.clean(
            split,
            merge_distance=1e-3,
            remove_isolated_pieces_diameter_percent=0,
            fill_holes_max_edge_count=0,
        )
        self.assertEqual(MeshOps.measure(cleaned)["vertices"], n_orig)

    def test_clean_prunes_isolated_pieces(self):
        pml = MeshOps.resolve()
        ms = pml.MeshSet()
        ms.create_sphere(radius=1.0, subdiv=3)
        ms.create_cube()  # tiny second component
        ms.generate_by_merging_visible_meshes()
        path = os.path.join(self.out_dir, "two_parts.obj")
        ms.save_current_mesh(path)
        self.assertEqual(MeshOps.measure(path)["components"], 2)
        cleaned = MeshOps.clean(path, remove_isolated_pieces_diameter_percent=60.0)
        self.assertEqual(MeshOps.measure(cleaned)["components"], 1)

    def test_default_output_naming(self):
        src = self._sphere()
        out = MeshOps.clean(src)
        self.assertEqual(out, os.path.splitext(src)[0] + "_clean.obj")
        self.assertTrue(os.path.getsize(out) > 0)

    def test_unwritable_output_rejected_before_processing(self):
        src = self._sphere()
        with self.assertRaises(ValueError):
            MeshOps.clean(src, output_path=os.path.join(self.out_dir, "x.glb"))
        # Fail-fast contract: nothing was produced.
        self.assertFalse(os.path.exists(os.path.join(self.out_dir, "x.glb")))


@unittest.skipUnless(_HAVE, "pymeshlab not installed")
class TestRefine(_TempDirTestCase):
    def test_decimate_hits_target(self):
        out = MeshOps.decimate(self._sphere(subdiv=4), target_faces=800)
        self.assertEqual(MeshOps.measure(out)["faces"], 800)

    def test_decimate_curvature_weighted(self):
        out = MeshOps.decimate(
            self._sphere(subdiv=4), target_faces=500, curvature_weighted=True
        )
        self.assertEqual(MeshOps.measure(out)["faces"], 500)

    def test_decimate_by_percent(self):
        src = self._sphere(subdiv=4)  # 5120 faces
        out = MeshOps.decimate(src, target_pct=25.0)
        self.assertAlmostEqual(
            MeshOps.measure(out)["faces"], 5120 * 0.25, delta=5120 * 0.02
        )

    def test_remesh_reaches_target_density(self):
        src = self._sphere(subdiv=4)
        before = MeshOps.measure(src)
        out = MeshOps.remesh(src, target_edge_pct=4.0, iterations=5)
        after = MeshOps.measure(out)
        self.assertGreater(after["avg_edge_length"], before["avg_edge_length"])

    def test_compare_reports_small_nonzero_deviation(self):
        src = self._sphere(subdiv=4)
        dec = MeshOps.decimate(src, target_faces=400)
        d = MeshOps.compare(dec, src)
        self.assertGreater(d["hausdorff_peak"], 0.0)
        self.assertLess(d["hausdorff_peak_pct"], 5.0)
        self.assertGreater(d["samples"], 0)


@unittest.skipUnless(_HAVE, "pymeshlab not installed")
class TestCompareDeviation(_TempDirTestCase):
    """``compare`` against deviations known in closed form.

    A radius-10 UV sphere at 528 faces against the same sphere at 65,280: every
    coarse vertex sits on the true surface, so the true deviation is the chord
    sag inside each coarse face, computed exactly from the coarse mesh alone.
    """

    R = 10.0

    @staticmethod
    def _uv_sphere(seg, rings, r, center=(0.0, 0.0, 0.0)):
        import math

        import numpy as np

        v = [(0.0, 0.0, r)]
        for i in range(1, rings):
            t = math.pi * i / rings
            for j in range(seg):
                p = 2 * math.pi * j / seg
                v.append(
                    (r * math.sin(t) * math.cos(p), r * math.sin(t) * math.sin(p), r * math.cos(t))
                )
        v.append((0.0, 0.0, -r))
        f = [(0, 1 + j, 1 + (j + 1) % seg) for j in range(seg)]
        for i in range(rings - 2):
            a, b = 1 + i * seg, 1 + (i + 1) * seg
            for j in range(seg):
                j2 = (j + 1) % seg
                f += [(a + j, b + j, b + j2), (a + j, b + j2, a + j2)]
        last, a = len(v) - 1, 1 + (rings - 2) * seg
        f += [(last, a + (j + 1) % seg, a + j) for j in range(seg)]
        return np.array(v, float) + np.array(center, float), np.array(f, int)

    def _obj(self, name, parts):
        path, offset = os.path.join(self.out_dir, name), 1
        with open(path, "w") as fh:
            for v, f in parts:
                fh.writelines("v %.9f %.9f %.9f\n" % tuple(p) for p in v)
                fh.writelines("f %d %d %d\n" % tuple(t + offset) for t in f)
                offset += len(v)
        return path

    def setUp(self):
        super().setUp()
        import numpy as np

        self.coarse_mesh = self._uv_sphere(24, 12, self.R)
        self.coarse = self._obj("coarse.obj", [self.coarse_mesh])
        self.fine = self._obj("fine.obj", [self._uv_sphere(256, 128, self.R)])
        v, f = self.coarse_mesh
        tri = v[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1)[:, None]
        self.true_peak = float((self.R - np.abs((n * tri[:, 0]).sum(1))).max())

    def test_the_peak_is_the_true_sag_in_either_order(self):
        # Red before the fix: 0.0013 for a true 0.1689 (vertices of the coarse
        # mesh sampled alone, all of them on the surface).
        for a, b in ((self.coarse, self.fine), (self.fine, self.coarse)):
            d = MeshOps.compare(a, b)
            self.assertAlmostEqual(
                d["hausdorff_peak"], self.true_peak, delta=0.02 * self.true_peak,
                msg=f"{os.path.basename(a)} against {os.path.basename(b)}: {d}",
            )

    def test_a_part_missing_from_the_input_fails_and_is_located(self):
        # Red before the fix: the reference's second sphere was never sampled
        # (0.0003), and the reversed call dropped its samples as too far.
        far = self._uv_sphere(48, 24, 2.0, center=(100.0, 0.0, 0.0))
        both = self._obj("both.obj", [self._uv_sphere(256, 128, self.R), far])
        d = MeshOps.compare(self.coarse, both)
        self.assertGreater(d["hausdorff_peak"], 80.0, d)
        self.assertGreater(d["reference_to_input_peak"], 80.0, d)
        self.assertLess(d["input_to_reference_peak"], 0.2, d)
        self.assertGreater(d["hausdorff_peak_at"][0], 90.0, d)

    def test_a_part_the_reference_lacks_fails_too(self):
        far = self._uv_sphere(48, 24, 2.0, center=(100.0, 0.0, 0.0))
        extra = self._obj("extra.obj", [self.coarse_mesh, far])
        d = MeshOps.compare(extra, self.fine)
        self.assertGreater(d["input_to_reference_peak"], 80.0, d)

    def test_identical_meshes_measure_zero(self):
        d = MeshOps.compare(self.coarse, self.coarse)
        self.assertLess(d["hausdorff_peak"], 1e-6, d)
        self.assertGreater(d["samples"], 0)

    def test_the_heatmap_colours_the_denser_mesh(self):
        heat = os.path.join(self.out_dir, "heat.ply")
        d = MeshOps.compare(self.coarse, self.fine, heatmap_path=heat, heatmap_range=0.2)
        self.assertEqual(d["heatmap"], heat)
        self.assertEqual(d["heatmap_range"], 0.2)
        ms = MeshOps.resolve().MeshSet()
        ms.load_new_mesh(heat)
        m = ms.current_mesh()
        self.assertEqual(m.vertex_number(), 128 * 256 - 256 + 2)  # the fine sphere's
        self.assertTrue(m.has_vertex_color())
        # The colours follow the distance: the pole lies on both spheres
        # (viridis' dark start); the deepest sag, 0.85 of the range, is green.
        green = m.vertex_color_matrix()[:, 1]
        self.assertLess(green[0], 0.05)
        self.assertGreater(green.max(), 0.75)

    def test_a_heatmap_range_at_or_below_zero_is_refused(self):
        # Review: red before, -1 was accepted and 0 read as "unset".
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError, msg=bad):
                MeshOps.compare(
                    self.coarse, self.fine, heatmap_path=os.path.join(self.out_dir, "h.ply"), heatmap_range=bad
                )

    def test_the_same_meshes_always_measure_the_same(self):
        # Montecarlo sampling gave four peaks in four calls at 1,000 samples.
        runs = [MeshOps.compare(self.coarse, self.fine, sample_num=1000) for _ in range(3)]
        self.assertEqual(len({(r["hausdorff_peak"], r["hausdorff_mean"]) for r in runs}), 1, runs)

    def test_a_sample_count_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            MeshOps.compare(self.coarse, self.fine, sample_num=0)

    def test_compare_keys_are_gate_prefix_safe(self):
        for key in MeshOps.compare(self.coarse, self.fine):
            self.assertNotIn("max_", key)
            self.assertNotIn("min_", key)


@unittest.skipUnless(_HAVE, "pymeshlab not installed")
class TestBakeAndSession(_TempDirTestCase):
    def test_bake_vertex_color_writes_mesh_and_texture(self):
        src = self._sphere(colored=True)
        mesh_path, tex_path = MeshOps.bake_vertex_color(src, texture_size=256)
        self.assertTrue(os.path.isfile(mesh_path))
        self.assertTrue(os.path.isfile(tex_path))
        self.assertGreater(os.path.getsize(tex_path), 0)
        # OBJ+MTL must actually bind the texture.
        mtl = mesh_path + ".mtl"
        self.assertTrue(os.path.isfile(mtl))
        with open(mtl, "r", encoding="utf-8", errors="replace") as f:
            self.assertIn(os.path.basename(tex_path), f.read())

    def test_bake_without_vertex_color_raises(self):
        with self.assertRaises(ValueError):
            MeshOps.bake_vertex_color(self._sphere(colored=False))

    def test_session_chains_ops_without_disk_roundtrip(self):
        src = self._sphere(subdiv=4)
        dst = os.path.join(self.out_dir, "chained.obj")
        with MeshOps.session(src) as s:
            before = s.measure()
            s.op("remesh_isotropic", targetlen=3.0, iterations=3)
            s.op("decimate_quadric", targetfacenum=600, autoclean=True)
            out = s.save(dst)
        self.assertEqual(out, dst)
        after = MeshOps.measure(dst)
        self.assertEqual(after["faces"], 600)
        self.assertGreater(before["faces"], after["faces"])

    def test_apply_raw_filter_passthrough(self):
        src = self._sphere()
        out = MeshOps.apply(src, "apply_coord_laplacian_smoothing", stepsmoothnum=1)
        self.assertTrue(os.path.getsize(out) > 0)

    def test_root_export(self):
        self.assertIs(ptk.MeshOps, MeshOps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
