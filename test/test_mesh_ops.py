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
class TestLods(_TempDirTestCase):
    """``MeshOps.lods``: a LOD chain of any mesh file, headless (2026-11-10).

    Each level is ``<stem>_LOD<n>`` beside LOD0, measured against it
    (``compare``). A budget in millimetres is met by measure-and-adjust: no
    simplifier's own error bound held in the LOD bake-off (meshoptimizer's
    under-bounded the measured peak 2.4-3.6x)."""

    def test_a_ladder_halves_each_level_and_names_it(self):
        src = self._sphere(subdiv=5)
        chain = MeshOps.lods(src, output_dir=self.out_dir)
        self.assertEqual([c["level"] for c in chain], [0, 1, 2])
        self.assertEqual(
            [os.path.basename(c["path"]) for c in chain],
            ["sphere_LOD0.ply", "sphere_LOD1.ply", "sphere_LOD2.ply"],
        )
        tris = [c["triangles"] for c in chain]
        self.assertEqual(tris[1], tris[0] // 2)
        self.assertLess(tris[2], tris[1])
        self.assertEqual(chain[0]["peak"], 0.0)
        self.assertGreater(chain[2]["peak"], chain[1]["peak"])

    def test_a_budget_is_met_by_measurement(self):
        src = self._sphere(subdiv=5)
        chain = MeshOps.lods(src, budgets=[0.002, 0.02], output_dir=self.out_dir)
        self.assertEqual(len(chain), 3)
        for level, budget in zip(chain[1:], (0.002, 0.02)):
            self.assertLessEqual(level["peak"], budget, level)
        self.assertLess(chain[2]["triangles"], chain[1]["triangles"])

    def test_budgets_must_rise(self):
        with self.assertRaises(ValueError):
            MeshOps.lods(self._sphere(), budgets=[0.02, 0.002], output_dir=self.out_dir)

    def test_a_bad_ladder_is_refused_before_anything_is_written(self):
        src = self._sphere()
        out = os.path.join(self.out_dir, "levels")
        for bad in ({"levels": 0}, {"ratio": 1.5}):
            with self.assertRaises(ValueError, msg=bad):
                MeshOps.lods(src, output_dir=out, **bad)
        self.assertFalse(os.path.isdir(out) and os.listdir(out), "nothing written")


@unittest.skipUnless(_HAVE, "pymeshlab not installed")
class TestTextured(_TempDirTestCase):
    """A textured mesh keeps its UVs and its texture through decimation
    (2026-11-10). Measured: PyMeshLab's plain quadric collapse leaves a seamed
    atlas 6% of its UV area (the texture smears); the texture-aware collapse
    keeps 94% with no flipped face, but runs only where every face names a
    texture. And a MeshLab save writes the mesh's textures beside the output,
    re-encoding the source's own JPEG when that is where it already is."""

    def _textured(self, texture=True):
        """A sphere with a seamed UV atlas, as an OBJ whose material maps
        ``tex.jpg`` (``texture=False``: the UVs and no material)."""
        import numpy as np

        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        pml = MeshOps.resolve()
        ms = pml.MeshSet()
        ms.create_sphere(radius=1.0, subdiv=4)
        ms.generate_voronoi_atlas_parametrization(regionnum=8)
        ms.set_current_mesh(ms.mesh_number() - 1)
        path = os.path.join(self.out_dir, "scan.obj")
        ms.save_current_mesh(path, save_textures=False)
        if texture:
            pixels = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype("uint8")
            Image.fromarray(pixels).save(os.path.join(self.out_dir, "tex.jpg"), quality=90)
            with open(path, encoding="utf-8") as fh:
                lines = [ln for ln in fh.read().splitlines() if not ln.startswith(("mtllib", "usemtl"))]
            first = next(k for k, ln in enumerate(lines) if ln.startswith("f "))
            lines.insert(first, "usemtl scan")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(["mtllib scan.mtl"] + lines) + "\n")
            with open(os.path.join(self.out_dir, "scan.mtl"), "w", encoding="utf-8") as fh:
                fh.write("newmtl scan\nmap_Kd tex.jpg\n")
        return path

    @staticmethod
    def _uv_area(path):
        import numpy as np

        ms = MeshOps.resolve().MeshSet()
        ms.load_new_mesh(path)
        uv = ms.current_mesh().wedge_tex_coord_matrix().reshape(-1, 3, 2)
        a, b = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
        return float(np.abs(0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])).sum())

    @staticmethod
    def _digest(path):
        import hashlib

        with open(path, "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()

    def test_decimating_a_textured_mesh_keeps_its_uvs(self):
        src = self._textured()
        out = MeshOps.decimate(src, os.path.join(self.out_dir, "low.ply"), target_faces=600)
        self.assertGreater(self._uv_area(out), 0.85 * self._uv_area(src))

    def test_a_save_beside_the_source_leaves_its_texture_as_it_is(self):
        src = self._textured()
        before = self._digest(os.path.join(self.out_dir, "tex.jpg"))
        MeshOps.decimate(src, os.path.join(self.out_dir, "low.ply"), target_faces=600)
        self.assertEqual(self._digest(os.path.join(self.out_dir, "tex.jpg")), before)

    def test_a_save_elsewhere_takes_the_texture_along(self):
        src = self._textured()
        other = os.path.join(self.out_dir, "delivery")
        MeshOps.decimate(src, os.path.join(other, "low.ply"), target_faces=600)
        self.assertTrue(os.path.isfile(os.path.join(other, "tex.jpg")))

    def test_the_session_says_what_its_collapse_applied(self):
        """A textured mesh's UV-keeping collapse weighs no curvature: the
        session states it, so a QC record does not claim it (2026-11-10 review)."""
        src = self._textured()
        with MeshOps.session(src) as s:
            s.decimate(target_faces=600, curvature_weighted=True)
            self.assertEqual(
                s.last_decimation,
                {"filter": "decimate_quadric_texture", "curvature_weighted": False, "preserve_topology": False},
            )

    def test_another_image_of_the_same_name_is_kept_and_said(self):
        """A folder already holding an unrelated ``tex.jpg``: it is not taken
        for the mesh's own, nor written over; the save says so (2026-11-10
        review: it silently pointed the mesh at the other image)."""
        src = self._textured()
        other = os.path.join(self.out_dir, "shared")
        os.makedirs(other)
        with open(os.path.join(other, "tex.jpg"), "wb") as fh:
            fh.write(b"someone else's image")
        with self.assertLogs("pythontk.file_utils.mesh_ops", level="WARNING") as logged:
            MeshOps.decimate(src, os.path.join(other, "low.ply"), target_faces=600)
        self.assertIn("another image", "\n".join(logged.output))
        with open(os.path.join(other, "tex.jpg"), "rb") as fh:
            self.assertEqual(fh.read(), b"someone else's image")

    def test_uvs_without_a_texture_are_said(self):
        src = self._textured(texture=False)
        with self.assertLogs("pythontk.file_utils.mesh_ops", level="WARNING") as logged:
            MeshOps.decimate(src, os.path.join(self.out_dir, "low.ply"), target_faces=600)
        self.assertIn("texture", "\n".join(logged.output))

    def test_a_textured_lod_chain_keeps_its_uvs(self):
        src = self._textured()
        chain = MeshOps.lods(src, output_dir=self.out_dir)
        for level in chain[1:]:
            self.assertGreater(self._uv_area(level["path"]), 0.85 * self._uv_area(src), level)


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
