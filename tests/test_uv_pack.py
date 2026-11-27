# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.geo_utils.uv.pack (UvPack — optional xatlas engine).

The engine is an optional dependency, so the packing tests skip cleanly when
xatlas is absent; the resolve/availability contract is testable either way.
"""

import time
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
        self.assertEqual(
            ptk.UvPack.available(), ptk.UvPack.resolve(required=False) is not None
        )

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
        for lo, hi in boxes:
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
            on_edge_u = min(
                abs(corner[0] - uv[:, 0].min()), abs(corner[0] - uv[:, 0].max())
            )
            on_edge_v = min(
                abs(corner[1] - uv[:, 1].min()), abs(corner[1] - uv[:, 1].max())
            )
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
        rect = (
            np.array([[0, 0], [1.333, 0], [1.333, 1], [0, 1]], dtype=np.float64) * 0.33
        )
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
        rect = (
            np.array([[0, 0], [1.333, 0], [1.333, 1], [0, 1]], dtype=np.float64) * 0.33
        )
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

    def test_seam_touching_islands_keep_their_gutter(self):
        """User-reported (tentacle Pack, xatlas + Preserve UV): two islands of
        ONE mesh touching along a cut seam -- coincident coordinates, distinct
        indices -- were welded by the engine into a single chart, so they came
        back 0.0 apart with no gutter, and a per-island write-back then drifted
        them into overlap. Each topological island must pack as its own chart,
        ``2 x padding`` texels apart like any other pair."""
        uvs = np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1], [1, 0], [2, 0], [2, 1], [1, 1]],
            dtype=np.float64,
        )
        tris = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.uint32)

        result = ptk.UvPack.pack_islands(
            [(uvs, tris)], padding=4, rotate=False, resolution=256
        )

        out = result.uvs[0]
        (lo_a, hi_a), (lo_b, hi_b) = (
            (out[:4].min(0), out[:4].max(0)),
            (
                out[4:].min(0),
                out[4:].max(0),
            ),
        )
        gap = float(max((lo_b - hi_a).max(), (lo_a - hi_b).max()))
        self.assertGreater(gap, 4 / 256, f"islands only {gap * 256:.2f} texels apart")

    @staticmethod
    def _shuffled_strips(quads=10000, seed=0):
        """Two long strip islands whose UV indices are shuffled across one
        index space -- the order an edited mesh's cuts, sews and merges leave."""
        rows, q = np.arange(quads + 1), np.arange(quads)
        per = 2 * (quads + 1)
        uvs, tris = [], []
        for s in range(2):
            uvs.append(
                np.column_stack(
                    [np.repeat(rows, 2) * 0.01, np.tile([0.0, 0.01], quads + 1) + s]
                )
            )
            a = s * per + 2 * q
            tris += [
                np.column_stack([a, a + 1, a + 3]),
                np.column_stack([a, a + 3, a + 2]),
            ]
        perm = np.random.RandomState(seed).permutation(2 * per)
        shuffled = np.empty((2 * per, 2))
        shuffled[perm] = np.vstack(uvs)
        return shuffled, perm[np.vstack(tris)]

    def test_island_detection_stays_fast_on_scattered_indices(self):
        """Guard against a pathological regression, not drift. Island detection
        hooked each triangle's corners onto the smallest label among them, which
        crept about a triangle per pass along a long island whose indices are
        scattered: 7.5 s on this 40k-triangle fixture (a 40k-triangle strip took
        5753 passes, 11.8 s). Hooking the tree each corner hangs from as well
        merges whole trees at once -- about a dozen passes, measured 60 ms here;
        the budget is 25x that."""
        uvs, tris = self._shuffled_strips()

        t0 = time.perf_counter()
        out = ptk.UvPack._separate_islands(uvs, tris)
        self.assertLess(time.perf_counter() - t0, 1.5)
        # Still exactly two islands, each moved as one piece.
        self.assertEqual(len(np.unique(np.round(out - uvs, 9), axis=0)), 2)

    @staticmethod
    def _bridged_quads():
        """Two quads one island only through a zero-area face -- a real face
        whose UVs collapsed onto a corner, routine in production layouts (625
        of a 9,544-face door's triangles). Plus a clean quad beside them."""
        uvs = np.array(
            [
                [0, 0], [1, 0], [1, 1], [0, 1],  # quad A
                [1, 1], [2, 1], [2, 2], [1, 2],  # quad B, its corner on A's
                [1, 1],  # the collapsed face's third corner
                [3, 0], [4, 0], [4, 1], [3, 1],  # quad C, an island of its own
            ],
            dtype=np.float64,
        )
        tris = np.array(
            [
                [0, 1, 2], [0, 2, 3],
                [4, 5, 6], [4, 6, 7],
                [2, 4, 8],  # zero area: bridges A and B
                [9, 10, 11], [9, 11, 12],
            ],
            dtype=np.uint32,
        )
        return uvs, tris

    def _assert_rigid(self, before, after, rows, what):
        *_, residual = ptk.UvPack.solve_similarity(before[rows], after[rows])
        span = float((after[rows].max(0) - after[rows].min(0)).max()) or 1.0
        self.assertLess(residual / span, 1e-4, f"{what} did not move as one piece")

    def test_an_island_bridged_by_a_collapsed_face_moves_as_one_piece(self):
        """The engine drops a zero-area face, and an island held together only
        through one came back as separate charts: the face stretched across
        the page between them, and corners only it used kept their input
        coordinates, outside the page, on top of other islands (production
        door: 66% of triangles off the pack's one scale, 53,237 texels inside
        two islands). Every island moves rigidly, inside the page."""
        uvs, tris = self._bridged_quads()
        for resolution in (0, 256):
            result = ptk.UvPack.pack_islands(
                [(uvs, tris)], padding=4, rotate=True, resolution=resolution
            )
            out = result.uvs[0]
            self._assert_rigid(uvs, out, np.arange(9), "the bridged island")
            self._assert_rigid(uvs, out, np.arange(9, 13), "the clean island")
            self.assertTrue(((out >= -1e-6) & (out <= 1 + 1e-6)).all(), out)
            # The collapsed face is still collapsed, not a sliver across the page.
            (bu, bv), (cu, cv) = out[4] - out[2], out[8] - out[2]
            self.assertLess(abs(bu * cv - bv * cu), 1e-9)
            self.assertEqual(result.boxed, [1])

    def test_a_pinched_island_moves_as_one_piece_in_one_round(self):
        """Two quads sharing one lone UV corner are one island to a DCC (its
        shells are vertex-connected) and two charts to the engine (edge-
        connected), which packed them apart over the shared UV: mayatk's
        write-back refused the whole mesh (production lightmap layout, shell
        14). Found before the first pack, so one engine pass does."""
        uvs = np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1], [2, 1], [2, 2], [1, 2], [3, 0], [4, 0], [4, 1], [3, 1]],
            dtype=np.float64,
        )
        tris = np.array(
            [[0, 1, 2], [0, 2, 3], [2, 4, 5], [2, 5, 6], [7, 8, 9], [7, 9, 10]],
            dtype=np.uint32,
        )
        calls = []
        real = ptk.UvPack._generate

        def counted(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        from unittest import mock

        with mock.patch.object(ptk.UvPack, "_generate", staticmethod(counted)):
            result = ptk.UvPack.pack_islands([(uvs, tris)], padding=4, align_to_axis=False)
        self._assert_rigid(uvs, result.uvs[0], np.arange(7), "the pinched island")
        self.assertEqual(result.boxed, [1])
        self.assertEqual(len(calls), 1, "a pinch found before packing takes one engine pass")

    def test_a_second_round_searches_from_the_first_rounds_scale(self):
        """A rigidity round boxes a few islands, so its best scale lies within a
        few percent of the last round's: it searches the winning candidate
        alone from there. Each round ran the whole search (a production door:
        40 engine runs over 2 rounds, 85% of its pack time)."""
        from unittest import mock

        quads = [self._quad(0.08 + 0.03 * i, (0.2 * i, 0.0)) for i in range(6)]
        uvs = np.vstack([q[0] for q in quads])
        tris = np.vstack([q[1] + 4 * i for i, q in enumerate(quads)]).astype(np.uint32)
        runs, rounds = [], []
        real_generate = ptk.UvPack._generate
        real_read = ptk.UvPack._read_back.__func__

        def counted(*args, **kwargs):
            runs.append(1)
            return real_generate(*args, **kwargs)

        def first_round_finds_one(cls, *args):
            out, written, pages, bad = real_read(cls, *args)
            rounds.append(len(runs))
            return out, written, pages, ({1} if len(rounds) == 1 else bad)

        with mock.patch.object(
            ptk.UvPack, "_generate", staticmethod(counted)
        ), mock.patch.object(ptk.UvPack, "_read_back", classmethod(first_round_finds_one)):
            result = ptk.UvPack.pack_islands([(uvs, tris)], padding=4, resolution=256)

        self.assertEqual(result.boxed, [1])
        first, second = rounds[0], rounds[1] - rounds[0]
        self.assertLessEqual(second, 6, f"round 1: {first} runs, round 2: {second}")
        self.assertLess(second, first / 2)

    def test_a_large_page_starts_from_the_scale_a_coarse_page_found(self):
        """The fixed-page search runs in full on a 512 page only; a page of
        twice the size starts from the scale the half-size page found (a run
        there costs about a quarter). A production door at 2048: 14.4 s ->
        5.6 s, fill 0.787 -> 0.786."""
        from unittest import mock

        import pythontk.geo_utils.uv.pack as pack_module

        rng = np.random.default_rng(7)
        quads = [
            self._quad(float(s), tuple(o))
            for s, o in zip(rng.uniform(0.02, 0.2, 60), rng.uniform(0.0, 5.0, (60, 2)))
        ]
        uvs = np.vstack([q[0] for q in quads])
        tris = np.vstack([q[1] + 4 * i for i, q in enumerate(quads)]).astype(np.uint32)
        sizes = []
        real = ptk.UvPack._generate

        def counted(*args, **kwargs):
            sizes.append(args[5])  # (xatlas, arrays, padding, rotate, brute, resolution, ...)
            return real(*args, **kwargs)

        with mock.patch.object(ptk.UvPack, "_generate", staticmethod(counted)):
            fast = ptk.UvPack.pack_islands([(uvs, tris)], padding=4, resolution=2048)
        with mock.patch.object(pack_module, "_COARSE_LEAST", 10**6):
            full = ptk.UvPack.pack_islands([(uvs, tris)], padding=4, resolution=2048)

        self.assertEqual(sorted(set(sizes)), [512, 1024, 2048])
        self.assertLessEqual(sizes.count(2048), 8, sizes)
        self.assertGreater(fast.utilization, full.utilization - 0.02)
        self.assertEqual(ptk.UvPack.island_overlaps([(fast.uvs[0], tris)], 2048), 0)

    def test_a_collapsed_island_lands_on_the_page(self):
        """An island whose UVs all sit on one point has no area to pack: the
        engine left it where it was (pack_uvs then reported success with 268 UVs
        outside 0-1). It lands on the page, rigidly, beside the rest."""
        uvs, tris = self._quad()
        point = np.full((4, 2), 7.5)
        both = np.vstack([uvs, point])
        tris2 = np.vstack([tris, tris + 4]).astype(np.uint32)

        result = ptk.UvPack.pack_islands([(both, tris2)], padding=4)

        out = result.uvs[0]
        self.assertTrue(((out >= -1e-6) & (out <= 1 + 1e-6)).all(), out)
        self.assertLess(float(np.ptp(out[4:], axis=0).max()), 1e-9, "still a point")
        self.assertEqual(result.boxed, [1])

    def test_a_box_is_square_pages_along_its_long_axis(self):
        """A Full box is one page of the map's size; a 1:3 box three pages of
        its short side, stacked, each placed in its own cell; a margin insets
        them all."""
        pages, resolution, place = ptk.UvPack.box_pages(1024)
        self.assertEqual((pages, resolution), (1, 1024))
        np.testing.assert_allclose(place([[0.0, 0.0], [1.0, 1.0]], [0, 0]), [[0, 0], [1, 1]])
        pages, resolution, place = ptk.UvPack.box_pages(900, (1 / 3, 1.0), margin=0.01)
        self.assertEqual((pages, resolution), (3, 300))
        top = place([[0.0, 0.0], [1.0, 1.0]], [2, 2])
        self.assertAlmostEqual(top[0][1], 0.01 + 2 * (0.98 / 3))
        self.assertLessEqual(top[1][1], 0.99 + 1e-9)
        self.assertLessEqual(top[1][0], 1 / 3 - 0.01 + 1e-9)
        with self.assertRaises(ValueError):
            ptk.UvPack.box_pages(512, (0.0, 1.0))

    def test_surface_areas_even_texel_density_across_meshes(self):
        """Two islands of one UV size, one with four times the other's surface:
        packed with ``surface_areas`` the larger surface gets four times the UV
        area (one texel density), without them the same; a count that does not
        match the triangles is refused."""
        small, large = self._quad(0.3), self._quad(0.3, (0.5, 0.5))

        def uv_area(uvs):
            return float(np.ptp(uvs[:, 0]) * np.ptp(uvs[:, 1]))

        plain = ptk.UvPack.pack_islands([small, large], padding=4, rotate=False)
        self.assertAlmostEqual(uv_area(plain.uvs[1]) / uv_area(plain.uvs[0]), 1.0, delta=0.1)
        evened = ptk.UvPack.pack_islands(
            [small, large], padding=4, rotate=False, surface_areas=[[1.0, 1.0], [4.0, 4.0]]
        )
        self.assertAlmostEqual(uv_area(evened.uvs[1]) / uv_area(evened.uvs[0]), 4.0, delta=0.4)
        with self.assertRaises(ValueError):
            ptk.UvPack.pack_islands([small, large], surface_areas=[[1.0], [4.0, 4.0]])

    def test_island_overlaps_counts_islands_on_islands_never_a_fold(self):
        """The check a pack is held to: two islands on each other count, a fold
        inside one island and two islands apart do not, across meshes too."""
        a = self._quad(0.4, (0.05, 0.05))
        b = self._quad(0.4, (0.55, 0.55))
        stacked = self._quad(0.4, (0.25, 0.25))
        fold_uvs = np.array([[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5], [0.3, 0.3]], dtype=np.float64)
        fold_tris = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4]], dtype=np.uint32)  # its third face over its first

        self.assertEqual(ptk.UvPack.island_overlaps([a, b], 256), 0)
        self.assertEqual(ptk.UvPack.island_overlaps([(fold_uvs, fold_tris)], 256), 0)
        self.assertGreater(ptk.UvPack.island_overlaps([a, stacked], 256), 0, "across meshes")
        result = ptk.UvPack.pack_islands([a, stacked, (fold_uvs, fold_tris)], padding=4)
        packed = list(zip(result.uvs, [a[1], stacked[1], fold_tris]))
        self.assertEqual(ptk.UvPack.island_overlaps(packed, 1024), 0)

    def test_triangle_indexing_past_its_uvs_raises_value_error(self):
        """Refused with the documented error before anything reads it: island
        detection indexes the UVs ahead of the engine, where an out-of-range
        corner surfaced as a bare numpy IndexError -- past a caller catching the
        documented errors (the engine itself refused it as a RuntimeError)."""
        uvs, tris = self._quad()
        bad = np.vstack([tris, [[0, 2, 4]]]).astype(np.uint32)

        with self.assertRaises(ValueError):
            ptk.UvPack.pack_islands([(uvs, bad)])

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
            n for n in names if getattr(stub.options, n) is _RecordingXatlas.SENTINEL
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


class TestCellBox(unittest.TestCase):
    """How one layout is packed to fill an atlas cell: the whole number of
    square pages nearest the cell's shape, the rest of it the layout's u
    stretch, the cell's long side the map size, a gutter scaled with the atlas."""

    def test_a_square_cell_is_one_page_of_its_size(self):
        box = ptk.UvPack.cell_box(256, 256, 2048)
        self.assertEqual(box, ((1.0, 1.0), 1.0, 256, 3))

    def test_a_long_cell_is_pages_side_by_side(self):
        """2:1 is two pages along u, 1:3 three along v: no stretch either way."""
        self.assertEqual(ptk.UvPack.cell_box(512, 256, 2048)[:3], ((1.0, 0.5), 1.0, 512))
        coverage, stretch, size, _padding = ptk.UvPack.cell_box(100, 300, 2048)
        self.assertEqual((coverage, stretch, size), ((1 / 3, 1.0), 1.0, 300))

    def test_the_rest_of_the_shape_is_the_layouts_stretch(self):
        """5:2 packs as two pages (2:1): the layout's u is squeezed by 2 / 2.5
        first, so the cell, wider than its box, shows every texel square; a
        shape within 1% of its pages needs none."""
        coverage, stretch, size, _padding = ptk.UvPack.cell_box(250, 100, 2048)
        self.assertEqual((coverage, size), ((1.0, 0.5), 250))
        self.assertAlmostEqual(stretch, 0.8)
        self.assertEqual(ptk.UvPack.cell_box(101, 100, 2048).stretch, 1.0)

    def test_the_gutter_is_three_texels_at_2048_scaled_and_at_least_two(self):
        self.assertEqual(
            [ptk.UvPack.cell_box(64, 64, a).padding for a in (512, 1024, 2048, 4096)],
            [2, 2, 3, 6],
        )


@unittest.skipUnless(np is not None, "Requires numpy")
class TestSimilarity(unittest.TestCase):
    """solve_similarity recovers a rigid move; apply_similarity replays it."""

    _POINTS = [[0.1, 0.2], [0.6, 0.25], [0.5, 0.8], [0.15, 0.7], [0.3, 0.4]]

    def _roundtrip(self, scale, angle, mirrored):
        old = np.asarray(self._POINTS)
        c0 = old.mean(axis=0)
        move = (scale, angle, mirrored, c0, c0 + np.array([0.3, -0.1]))
        new = ptk.UvPack.apply_similarity(old, move)
        solved = ptk.UvPack.solve_similarity(old, new)
        self.assertAlmostEqual(solved[0], scale, places=9)
        self.assertEqual(solved[2], mirrored)
        self.assertLess(solved[5], 1e-9)
        np.testing.assert_allclose(ptk.UvPack.apply_similarity(old, solved), new)

    def test_proper_move(self):
        self._roundtrip(0.37, 33.0, False)

    def test_mirrored_move(self):
        self._roundtrip(1.8, -120.0, True)

    def test_non_rigid_reports_residual(self):
        old = np.asarray(self._POINTS)
        new = old * np.array([2.0, 1.0])  # anisotropic: no similarity fits
        self.assertGreater(ptk.UvPack.solve_similarity(old, new)[5], 1e-3)

    def test_degenerate_translates(self):
        move = ptk.UvPack.solve_similarity([[0.5, 0.5]], [[0.7, 0.1]])
        np.testing.assert_allclose(
            ptk.UvPack.apply_similarity([[0.5, 0.5]], move), [[0.7, 0.1]]
        )


class TestModuleInvariant(unittest.TestCase):
    """Helpers live on classes, not at module scope (package standard)."""

    def test_no_top_level_functions(self):
        import ast
        import inspect

        from pythontk.geo_utils.uv import pack

        tree = ast.parse(inspect.getsource(pack))
        offenders = [
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.assertEqual(offenders, [], f"module-level def(s): {offenders}")


if __name__ == "__main__":
    unittest.main()
