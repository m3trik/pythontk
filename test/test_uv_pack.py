# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.geo_utils.uv_pack (UvPack — optional xatlas engine).

The engine is an optional dependency, so the packing tests skip cleanly when
xatlas is absent; the resolve/availability contract is testable either way.
"""
import unittest

import pythontk as ptk

try:
    import numpy as np
except ImportError:
    np = None

_XATLAS = ptk.UvPack.available()


class TestResolveContract(unittest.TestCase):
    """resolve() must never ImportError — module or actionable message."""

    def test_resolve_optional_returns_module_or_none(self):
        result = ptk.UvPack.resolve(required=False)
        self.assertTrue(result is None or hasattr(result, "Atlas"))

    def test_available_matches_resolve(self):
        self.assertEqual(ptk.UvPack.available(), ptk.UvPack.resolve(required=False) is not None)

    @unittest.skipIf(_XATLAS, "xatlas installed — missing-engine path untestable")
    def test_resolve_required_raises_with_install_note(self):
        with self.assertRaises(RuntimeError) as ctx:
            ptk.UvPack.resolve(required=True)
        self.assertIn("pip install", str(ctx.exception))


@unittest.skipUnless(_XATLAS and np is not None, "Requires xatlas + numpy")
class TestPackIslands(unittest.TestCase):
    """Packing contract, pinned against behavior verified on xatlas 0.0.11."""

    @staticmethod
    def _quad(scale=1.0, offset=(0.0, 0.0)):
        uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        uvs = uvs * scale + np.asarray(offset, dtype=np.float64)
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        return uvs, tris

    def test_two_islands_pack_without_overlap(self):
        result = ptk.UvPack.pack_islands([self._quad(), self._quad()], rotate=False)
        self.assertEqual(len(result.uvs), 2)
        boxes = [(u.min(0), u.max(0)) for u in result.uvs]
        for (lo, hi) in boxes:
            self.assertTrue((lo >= -1e-6).all() and (hi <= 1 + 1e-6).all())
        (lo_a, hi_a), (lo_b, hi_b) = boxes
        disjoint_u = hi_a[0] <= lo_b[0] + 1e-6 or hi_b[0] <= lo_a[0] + 1e-6
        disjoint_v = hi_a[1] <= lo_b[1] + 1e-6 or hi_b[1] <= lo_a[1] + 1e-6
        self.assertTrue(disjoint_u or disjoint_v, "islands overlap")

    def test_relative_island_scale_is_preserved(self):
        result = ptk.UvPack.pack_islands(
            [self._quad(), self._quad(scale=2.0)], rotate=False
        )
        size_a = (result.uvs[0].max(0) - result.uvs[0].min(0))[0]
        size_b = (result.uvs[1].max(0) - result.uvs[1].min(0))[0]
        self.assertAlmostEqual(size_b / size_a, 2.0, places=3)

    def test_output_is_aspect_true_and_input_aligned(self):
        """Unit-square input islands must come back square (aspect preserved
        despite the non-square atlas) with rows aligned to input indices."""
        result = ptk.UvPack.pack_islands([self._quad(), self._quad()], rotate=False)
        self.assertAlmostEqual(max(result.extent), 1.0, places=6)
        for uv in result.uvs:
            self.assertEqual(len(uv), 4)
            du, dv = (uv.max(0) - uv.min(0)).tolist()
            self.assertAlmostEqual(du, dv, places=3)
            # input order: vertex 0 is the island's min corner in the input;
            # with rotation off it must still be a bbox corner after packing.
            corner = uv[0]
            on_edge_u = min(abs(corner[0] - uv[:, 0].min()), abs(corner[0] - uv[:, 0].max()))
            on_edge_v = min(abs(corner[1] - uv[:, 1].min()), abs(corner[1] - uv[:, 1].max()))
            self.assertLess(float(on_edge_u + on_edge_v), 1e-3)

    def test_written_covers_every_row_including_unreferenced(self):
        """Verified on xatlas 0.0.11: the engine maps EVERY input vertex, even
        one no triangle references — so no row is left holding stale input
        coordinates. `written` records that, and a consumer fitting a transform
        from before/after coordinates can trust the whole array. (Kept as a
        guard: a build that dropped rows would silently corrupt such a fit,
        which is the failure class that broke mayatk's per-shell write-back.)
        """
        uvs, tris = self._quad()
        orphan = np.vstack([uvs, [[9.0, 9.0]]])  # trailing UV no triangle uses
        result = ptk.UvPack.pack_islands([(orphan, tris)], rotate=False)

        written = np.asarray(result.written[0])
        self.assertEqual(sorted(written.tolist()), [0, 1, 2, 3, 4])
        # every row is engine-positioned, so nothing keeps the 9.0 sentinel
        self.assertTrue((result.uvs[0] <= 1.0 + 1e-6).all())

    def test_written_aligns_with_every_mesh(self):
        result = ptk.UvPack.pack_islands([self._quad(), self._quad()], rotate=False)
        self.assertEqual(len(result.written), len(result.uvs))
        for uv, written in zip(result.uvs, result.written):
            self.assertTrue(len(written) <= len(uv))

    def test_fixed_page_mode_fills_the_square_page(self):
        """resolution>0: scale-searched square pages. The whole point is fill —
        the content must reach well past what content-driven aspect-fitting
        gave (measured 0.50 for comparable content)."""
        rect = np.array([[0, 0], [1.333, 0], [1.333, 1], [0, 1]], dtype=np.float64) * 0.33
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        meshes = [(rect.copy(), tris) for _ in range(6)]

        result = ptk.UvPack.pack_islands(meshes, padding=4, resolution=1024, pages=1)

        self.assertEqual(result.page_count, 1)
        self.assertEqual((result.width, result.height), (1024, 1024))
        self.assertEqual(result.extent, (1.0, 1.0))
        fill = sum(float(np.prod(u.max(0) - u.min(0))) for u in result.uvs)
        self.assertGreater(fill, 0.55, f"page fill {fill:.3f} — search regressed")
        for uv in result.uvs:
            self.assertTrue((uv >= -1e-6).all() and (uv <= 1 + 1e-6).all())

    def test_fixed_page_mode_two_pages_balance_and_report(self):
        """pages=2: per-UV page indices come back and both pages are used at
        the searched (maximal) scale; every page's UVs are 0-1 normalized."""
        rect = np.array([[0, 0], [1.333, 0], [1.333, 1], [0, 1]], dtype=np.float64) * 0.33
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        meshes = [(rect.copy(), tris) for _ in range(6)]

        result = ptk.UvPack.pack_islands(meshes, padding=4, resolution=1024, pages=2)

        self.assertEqual(result.page_count, 2)
        used = set()
        for uv, pg in zip(result.uvs, result.pages):
            pg = np.asarray(pg)
            self.assertEqual(len(pg), len(uv))
            # a single-island mesh lives on exactly one page
            self.assertEqual(len(np.unique(pg)), 1)
            used.add(int(pg[0]))
            self.assertTrue((uv >= -1e-6).all() and (uv <= 1 + 1e-6).all())
        self.assertEqual(used, {0, 1})

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            ptk.UvPack.pack_islands([])

    def test_mesh_without_uvs_raises(self):
        with self.assertRaises(ValueError):
            ptk.UvPack.pack_islands([(np.zeros((0, 2)), np.zeros((0, 3)))])


class _RecordingXatlas:
    """Stand-in engine module that captures the PackOptions it was handed."""

    SENTINEL = object()

    def __init__(self, field_names):
        captured = self

        class PackOptions:
            def __init__(self):
                for name in field_names:
                    setattr(self, name, _RecordingXatlas.SENTINEL)

        class Atlas:
            def add_uv_mesh(self, *args):
                pass

            def generate(self, pack_options=None, verbose=False):
                captured.options = pack_options

        self.options = None
        self.PackOptions = PackOptions
        self.Atlas = Atlas


@unittest.skipUnless(_XATLAS and np is not None, "Requires xatlas + numpy")
class TestPackOptionContract(unittest.TestCase):
    """The engine-option surface itself — what is pinned, searched, or refused."""

    @staticmethod
    def _tilted_islands(count=8, seed=1):
        """Irregular islands at random angles: a hull-axis rotation shows up
        here, where axis-aligned quads would hide it."""
        rng = np.random.RandomState(seed)
        uvs, tris = [], []
        for i in range(count):
            sides = 6
            angles = np.sort(rng.rand(sides)) * 2 * np.pi
            radii = 0.25 + rng.rand(sides) * 0.35
            pts = np.column_stack([np.cos(angles) * radii, np.sin(angles) * radii])
            turn = rng.rand() * np.pi
            rot = np.array(
                [[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]]
            )
            pts = pts @ rot.T + np.array([i * 3.0, 0.0])
            base = len(uvs)
            uvs += [tuple(p) for p in pts]
            tris += [(base, base + j, base + j + 1) for j in range(1, sides - 1)]
        return np.array(uvs, np.float64), np.array(tris, np.uint32)

    @staticmethod
    def _packed_area(uvs, tris, resolution=1024, **kwargs):
        """Island area delivered by one fixed-page pack — the fill measure the
        tightness assertions compare."""
        result = ptk.UvPack.pack_islands(
            [(uvs, tris)], padding=4, resolution=resolution, pages=1, **kwargs
        )
        return ptk.UvPack._uv_area(result.uvs[0], tris)

    def test_every_pack_option_is_set_explicitly(self):
        """No PackOptions field may be left at the engine's own default.

        ``rotate_charts_to_axis`` defaulted to True and silently rotated every
        island even with rotation off; this fails the moment a new engine
        version adds a field, forcing the same decision to be made knowingly
        rather than inherited.
        """
        import xatlas

        names = [n for n in dir(xatlas.PackOptions()) if not n.startswith("_")]
        stub = _RecordingXatlas(names)
        ptk.UvPack._generate(
            stub,
            [],
            padding=4,
            rotate=True,
            brute_force=False,
            resolution=1024,
            texels_per_unit=100.0,
        )
        unset = [
            n
            for n in names
            if getattr(stub.options, n) is _RecordingXatlas.SENTINEL
        ]
        self.assertEqual(unset, [], f"PackOptions left at engine defaults: {unset}")

    def test_rotate_false_leaves_every_island_unrotated(self):
        """Regression: the engine's hull-axis pre-rotation is independent of its
        90-degree sibling, so rotate=False used to still spin every island."""
        uvs, tris = self._tilted_islands()
        result = ptk.UvPack.pack_islands(
            [(uvs, tris)], padding=4, resolution=1024, pages=1, rotate=False
        )
        packed = result.uvs[0]
        for tri in tris:
            edge_in = uvs[tri[1]] - uvs[tri[0]]
            edge_out = packed[tri[1]] - packed[tri[0]]
            angle = np.degrees(
                np.arctan2(edge_out[1], edge_out[0])
                - np.arctan2(edge_in[1], edge_in[0])
            )
            # Coordinates come back as float32 texels, so an unrotated island
            # still reads a few hundredths of a degree off (measured max 0.09).
            # The leak this guards against tilted islands by 27-169 degrees, so
            # half a degree clears the noise 5x over and any real leak by 50x.
            self.assertLess(abs((angle + 180) % 360 - 180), 0.5)

    def test_variant_search_beats_every_pinned_variant(self):
        """Auto (align_to_axis=None) must be at least as tight as pinning it."""
        uvs, tris = self._tilted_islands(count=12, seed=3)

        def area(pinned):
            return self._packed_area(uvs, tris, align_to_axis=pinned)

        auto = area(None)
        for pinned in (True, False):
            self.assertGreaterEqual(auto, area(pinned) - 1e-9)

    def test_brute_force_is_never_looser_than_plain(self):
        """Brute placement competes with plain placement rather than replacing
        it, so opting into the slower mode can only ever hold or improve fill.
        It used to be applied as a post-pass whose result was returned
        unconditionally, which could hand back a looser pack than not asking
        for it at all."""
        # Both packing modes: fixed-page ranks candidates by searched scale,
        # content-driven by bounding-square coverage, and the guarantee has to
        # hold either way. Content-driven brute force has no page bounding the
        # engine's exhaustive placement, so that mode gets trivial input —
        # otherwise it alone costs more than the rest of this file combined.
        for resolution, count in ((512, 8), (0, 3)):
            with self.subTest(resolution=resolution):
                uvs, tris = self._tilted_islands(count=count, seed=5)
                plain = self._packed_area(
                    uvs, tris, resolution=resolution, brute_force=False
                )
                brute = self._packed_area(
                    uvs, tris, resolution=resolution, brute_force=True
                )
                self.assertGreaterEqual(brute, plain - 1e-9)

    def test_align_to_axis_contradicting_rotate_raises(self):
        with self.assertRaises(ValueError):
            ptk.UvPack.pack_islands(
                [(np.zeros((4, 2)), np.zeros((1, 3), dtype=np.uint32))],
                rotate=False,
                align_to_axis=True,
            )

    def test_variants_collapse_when_rotation_is_off(self):
        """rotate=False must drop every hull-axis variant, not search them."""
        self.assertTrue(
            all(not v["align_to_axis"] for v in ptk.UvPack._variants(False, None))
        )
        self.assertTrue(
            any(v["align_to_axis"] for v in ptk.UvPack._variants(True, None))
        )


class TestModuleInvariant(unittest.TestCase):
    """Helpers live on classes, not at module scope (package standard)."""

    def test_no_top_level_functions(self):
        import ast
        import inspect

        from pythontk.geo_utils import uv_pack

        tree = ast.parse(inspect.getsource(uv_pack))
        offenders = [
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.assertEqual(offenders, [], f"module-level def(s): {offenders}")


if __name__ == "__main__":
    unittest.main()
