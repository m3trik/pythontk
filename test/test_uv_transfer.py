# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.geo_utils.uv.transfer (UvTransfer -- UV-to-UV texel remap).

Pins the conventions the host adapters rely on (V-up UVs, V-flipped images,
+0.5 texel centers, correspondence by triangle index) and the behaviours that
make the remap correct where a ray-cast bake is not: exact identity, rigid
island transforms (rotate / mirror / scale), multi-source consolidation with
constants, tangent-frame re-encoding of normal maps, coverage + padding.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

import pythontk as ptk

# The unit square as two triangles, V up.
QUAD = np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]], dtype=float)


def _rot90(uv):
    """Rotate UVs 90 degrees CCW about the square's center."""
    return np.stack([1.0 - uv[..., 1], uv[..., 0]], axis=-1)


def _mirror_u(uv):
    return np.stack([1.0 - uv[..., 0], uv[..., 1]], axis=-1)


def _noise(size, channels=3, seed=0):
    return (
        np.random.RandomState(seed)
        .randint(0, 256, (size, size, channels))
        .astype(np.float32)
    )


def _gradient(size):
    """Smooth image: R = u ramp, G = v ramp (V up -> top row is G=1)."""
    u = (np.arange(size) + 0.5) / size
    v = 1.0 - (np.arange(size) + 0.5) / size
    img = np.zeros((size, size, 3), np.float32)
    img[..., 0] = u[None, :] * 255
    img[..., 1] = v[:, None] * 255
    return img


def _relief_normals(size=128, convention="opengl"):
    """An 8-bit tangent-space normal map of a real height field, stored in
    *convention* -- relief the content detector reads with confidence."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    h = np.sin(x / size * 6 * np.pi) * np.cos(y / size * 4 * np.pi) * 8.0
    hx, hy = np.gradient(h, axis=1), np.gradient(h, axis=0)
    green = hy if convention == "opengl" else -hy  # image rows run DOWN
    n = np.stack([-hx, green, np.ones_like(h)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return np.rint((n + 1.0) * 127.5).astype(np.float32)


def _encode(xyz):
    """A unit vector as 8-bit normal-map values."""
    return (np.asarray(xyz, np.float32) + 1.0) * 0.5 * 255.0


class TestBuildContract(unittest.TestCase):
    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ptk.UvTransfer.build(QUAD, QUAD[:1], 8)

    def test_source_ids_length_checked(self):
        with self.assertRaises(ValueError):
            ptk.UvTransfer.build(QUAD, QUAD, 8, source_ids=[0])

    def test_full_coverage_and_passes(self):
        t = ptk.UvTransfer.build(QUAD, QUAD, 16, supersample=2)
        self.assertEqual(t.passes, 4)
        self.assertEqual(t.size, (16, 16))
        self.assertTrue(t.mask.all())
        self.assertTrue(np.allclose(t.coverage, 1.0))
        self.assertEqual(t.overlaps, 0)
        self.assertEqual(t.skipped, 0)

    def test_rect_size_and_partial_coverage(self):
        half = QUAD * [0.5, 1.0]  # left half of the square only
        t = ptk.UvTransfer.build(half, half, (8, 16), supersample=1)
        self.assertEqual(t.size, (8, 16))
        self.assertTrue(t.mask[:, :8].all())
        self.assertFalse(t.mask[:, 8:].any())

    def test_degenerate_triangle_skipped_not_fatal(self):
        bad = np.concatenate([QUAD, np.zeros((1, 3, 2))])
        t = ptk.UvTransfer.build(bad, bad, 8)
        self.assertEqual(t.skipped, 1)
        self.assertTrue(t.mask.all())

    def test_overlapping_target_islands_counted(self):
        src = np.concatenate([QUAD, QUAD * 0.5])  # source: two distinct regions
        dst = np.concatenate([QUAD, QUAD])  # target: both land on the same texels
        t = ptk.UvTransfer.build(src, dst, 8, supersample=1)
        self.assertGreater(t.overlaps, 0)

    def test_memory_accounting(self):
        t = ptk.UvTransfer.build(QUAD, QUAD, 32, supersample=2)
        self.assertEqual(t.nbytes, 4 * 32 * 32 * 8)


class TestIdentityAndRigid(unittest.TestCase):
    """Conventions: identity exact; rotate / mirror match numpy image ops."""

    def test_identity_point_sampled_is_exact(self):
        img = _noise(64)
        t = ptk.UvTransfer.build(QUAD, QUAD, 64, supersample=1)
        out, cov = ptk.UvTransfer.transfer(t, img)
        # uint16 UV quantization: 64/65535 texel -> sub-1-level error on noise
        self.assertLess(np.abs(out - img).max(), 1.0)
        self.assertTrue(np.allclose(cov, 1.0))

    def test_identity_supersampled_smooth_image_exact(self):
        img = _gradient(64)
        t = ptk.UvTransfer.build(QUAD, QUAD, 64, supersample=2)
        out, _ = ptk.UvTransfer.transfer(t, img)
        self.assertLess(np.abs(out - img).max(), 0.5)

    def test_rotate_90_matches_image_rotation(self):
        img = _noise(64)
        t = ptk.UvTransfer.build(QUAD, _rot90(QUAD), 64, supersample=1)
        out, _ = ptk.UvTransfer.transfer(t, img)
        # A CCW rotation in V-up UV space is a CCW rotation of the stored image.
        self.assertLess(np.abs(out - np.rot90(img, 1)).max(), 1.0)

    def test_mirror_u_matches_image_flip(self):
        img = _noise(64)
        t = ptk.UvTransfer.build(QUAD, _mirror_u(QUAD), 64, supersample=1)
        out, _ = ptk.UvTransfer.transfer(t, img)
        self.assertLess(np.abs(out - img[:, ::-1]).max(), 1.0)

    def test_scale_down_box_filters(self):
        # Source: 64px checker (2px cells). Target: same quad at quarter size.
        img = np.zeros((64, 64, 1), np.float32)
        cells = (np.arange(64)[:, None] // 2 + np.arange(64)[None, :] // 2) % 2 == 0
        img[cells] = 255
        dst = QUAD * 0.25
        t = ptk.UvTransfer.build(QUAD, dst, 64, supersample=4)
        out, cov = ptk.UvTransfer.transfer(t, img)
        # Inside the shrunken island every texel averages a 4x4 source block:
        # two cells of each colour -> mid grey.
        inside = cov > 0.99
        self.assertTrue(inside[-16:, :16].all())
        self.assertTrue(np.allclose(out[inside][:, 0], 127.5, atol=2.0))

    def test_nearest_sampling_option(self):
        img = _noise(16)
        t = ptk.UvTransfer.build(QUAD, QUAD, 16, supersample=1)
        out, _ = ptk.UvTransfer.transfer(t, img, bilinear=False)
        self.assertTrue(np.array_equal(out, img))


class TestSources(unittest.TestCase):
    def test_multi_source_consolidation_with_constant(self):
        # Left triangle reads image 0, right triangle is material 1 with NO map.
        img = _noise(32)
        t = ptk.UvTransfer.build(QUAD, QUAD, 32, supersample=1, source_ids=[0, 1])
        out, _ = ptk.UvTransfer.transfer(t, {0: img, 1: (10.0, 20.0, 30.0)})
        # Bottom-right corner is in triangle 0, top-left in triangle 1.
        self.assertLess(np.abs(out[-1, -1] - img[-1, -1]).max(), 1.0)
        self.assertTrue(np.allclose(out[0, 0], (10.0, 20.0, 30.0)))

    def test_missing_source_id_raises(self):
        t = ptk.UvTransfer.build(QUAD, QUAD, 8, source_ids=[0, 1])
        with self.assertRaises(KeyError):
            ptk.UvTransfer.transfer(t, {0: _noise(8)})

    def test_grey_and_rgb_sources_widen_to_rgb(self):
        t = ptk.UvTransfer.build(QUAD, QUAD, 8, source_ids=[0, 1])
        out, _ = ptk.UvTransfer.transfer(t, {0: np.full((8, 8), 7.0), 1: _noise(8)})
        self.assertEqual(out.shape, (8, 8, 3))
        self.assertTrue(np.allclose(out[-1, -1], 7.0))

    def test_a_grey_source_in_an_rgba_layout_is_opaque(self):
        """Grey fills the COLOUR channels; the alpha it does not have is
        opaque. Repeated into all four, a metal-0 map beside a packed
        MetallicSmoothness one wrote smoothness 0 over its whole region (and a
        black albedo beside a cutout one would have punched a hole)."""
        rgba = np.full((8, 8, 4), (10.0, 20.0, 30.0, 100.0), np.float32)
        t = ptk.UvTransfer.build(QUAD, QUAD, 8, supersample=1, source_ids=[0, 1])
        for grey in (np.zeros((8, 8)), (0.0,)):
            with self.subTest(grey=type(grey).__name__):
                out, _ = ptk.UvTransfer.transfer(t, {0: rgba, 1: grey}, value_max=255.0)
                self.assertTrue(np.allclose(out[0, 0], (0.0, 0.0, 0.0, 255.0)))
                self.assertTrue(np.allclose(out[-1, -1], (10.0, 20.0, 30.0, 100.0)))

    def test_rgb_pads_opaque_at_the_stated_scale_not_a_guessed_one(self):
        """An all-black RGB map (black metal, no emission) looks like 0..1
        data, so a guessed scale padded its alpha at 1/255 -- transparent."""
        rgba = np.full((8, 8, 4), 128.0, np.float32)
        t = ptk.UvTransfer.build(QUAD, QUAD, 8, supersample=1, source_ids=[0, 1])
        out, _ = ptk.UvTransfer.transfer(
            t, {0: rgba, 1: np.zeros((8, 8, 3))}, value_max=255.0
        )
        self.assertTrue(np.allclose(out[0, 0], (0.0, 0.0, 0.0, 255.0)))

    def test_source_mask_prefills_gutter_before_sampling(self):
        # Source map: valid only on the left half (white); right half is black
        # gutter. Target samples the source's right edge -> must stay white.
        img = np.zeros((16, 16, 1), np.float32)
        img[:, :8] = 255
        mask = np.zeros((16, 16), bool)
        mask[:, :8] = True
        src = QUAD * [0.5, 1.0]  # triangles cover the left half of the source
        t = ptk.UvTransfer.build(src, QUAD, 16, supersample=2)
        without, _ = ptk.UvTransfer.transfer(t, img)
        with_mask, _ = ptk.UvTransfer.transfer(t, img, source_masks=mask)
        self.assertLess(without[:, -1, 0].min(), 250)  # bleeds the gutter
        self.assertTrue(np.all(with_mask[:, -1, 0] > 254))


class TestNormals(unittest.TestCase):
    @staticmethod
    def _encode(xyz):
        return (np.asarray(xyz, np.float32) + 1.0) * 0.5 * 255.0

    def _flat_map(self, size, xyz):
        img = np.empty((size, size, 3), np.float32)
        img[:] = self._encode(xyz)
        return img

    def test_frames_identity_rotation_mirror(self):
        F = ptk.UvTransfer.triangle_frames
        self.assertTrue(np.allclose(F(QUAD, QUAD), np.eye(2)))
        R = F(QUAD, _rot90(QUAD))[0]
        self.assertTrue(np.allclose(R, [[0, -1], [1, 0]]))
        self.assertAlmostEqual(np.linalg.det(F(QUAD, _mirror_u(QUAD))[0]), -1.0)
        # Uniform scale is not a rotation: identity frame.
        self.assertTrue(np.allclose(F(QUAD, QUAD * 0.5), np.eye(2)))

    def test_unrotated_island_passes_through(self):
        n = (0.6, 0.0, 0.8)
        t = ptk.UvTransfer.build(QUAD, QUAD, 16, supersample=1)
        out, _ = ptk.UvTransfer.transfer_normals(t, self._flat_map(16, n))
        self.assertTrue(np.allclose(out, self._encode(n), atol=0.6))

    def test_rotated_island_rotates_xy_opengl(self):
        # A +X tilt on the source island. Rotating the island 90 CCW in UV
        # space turns its tangent frame with it: the tilt becomes +Y.
        t = ptk.UvTransfer.build(QUAD, _rot90(QUAD), 16, supersample=1)
        out, _ = ptk.UvTransfer.transfer_normals(t, self._flat_map(16, (0.6, 0.0, 0.8)))
        self.assertTrue(np.allclose(out, self._encode((0.0, 0.6, 0.8)), atol=0.6))

    def test_rotated_island_directx_convention(self):
        # Same geometry; DirectX stores -Y, so the SAME physical tilt encodes
        # as -Y after rotation, i.e. the opposite green of the OpenGL case.
        t = ptk.UvTransfer.build(QUAD, _rot90(QUAD), 16, supersample=1)
        gl, _ = ptk.UvTransfer.transfer_normals(
            t, self._flat_map(16, (0.6, 0.0, 0.8)), convention="opengl"
        )
        dx, _ = ptk.UvTransfer.transfer_normals(
            t, self._flat_map(16, (0.6, 0.0, 0.8)), convention="directx"
        )
        self.assertTrue(np.allclose(dx[..., 1], 255.0 - gl[..., 1], atol=0.6))
        self.assertTrue(np.allclose(dx[..., 0], gl[..., 0], atol=0.6))

    def test_mirrored_island_flips_x(self):
        t = ptk.UvTransfer.build(QUAD, _mirror_u(QUAD), 16, supersample=1)
        out, _ = ptk.UvTransfer.transfer_normals(t, self._flat_map(16, (0.6, 0.0, 0.8)))
        self.assertTrue(np.allclose(out, self._encode((-0.6, 0.0, 0.8)), atol=0.6))

    def test_output_is_unit_length_and_16bit_range(self):
        t = ptk.UvTransfer.build(QUAD, _rot90(QUAD), 8, supersample=2)
        src = self._flat_map(8, (0.6, 0.0, 0.8)) / 255.0 * 65535.0
        out, _ = ptk.UvTransfer.transfer_normals(t, src, value_range=(0, 65535))
        vec = out / 65535.0 * 2.0 - 1.0
        self.assertTrue(np.allclose(np.linalg.norm(vec, axis=2), 1.0, atol=1e-3))

    def test_bad_convention_raises(self):
        t = ptk.UvTransfer.build(QUAD, QUAD, 8)
        with self.assertRaises(ValueError):
            ptk.UvTransfer.transfer_normals(
                t, self._flat_map(8, (0, 0, 1)), convention="gl"
            )


class TestAutoSize(unittest.TestCase):
    """Consolidating N texture sets into one layout keeps the size the caller's
    choice, but must never let the resulting density loss go unsaid.

    The production failure (TURRETS_WIRES.glb): two 2048 sets transferred into
    one shared 2048 layout. The set that landed on 9.4% of the target -- having
    owned 94% of its own map -- dropped from ~2014px of content to ~629px, a
    3.2x linear loss that shipped as visibly flattened roughness. 2048 was a
    defensible size for the asset; the defect was that nothing said what it
    cost, so nobody could judge.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="uvxfer_size_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _map(self, name, size):
        from PIL import Image

        path = os.path.join(self.tmp, name)
        Image.new("RGB", (size, size), (128, 128, 128)).save(path)
        return path

    @staticmethod
    def _quad(area, offset=(0.0, 0.0)):
        """The unit quad scaled to cover *area* of UV space."""
        return QUAD * float(np.sqrt(area)) + np.asarray(offset, dtype=float)

    def _consolidation(self, big_share, small_share, px=256):
        """Two sources, each owning ~all of its own *px* map, landing on
        *big_share* / *small_share* of one shared target layout."""
        src = np.concatenate([self._quad(0.90), self._quad(0.94)])
        dst = np.concatenate(
            [self._quad(big_share), self._quad(small_share, offset=(2.0, 2.0))]
        )
        ids = np.array([0, 0, 1, 1], np.int32)
        sources = [
            {"maps": {"baseColor": self._map("turrets.png", px)}, "constants": {}},
            {"maps": {"baseColor": self._map("wires.png", px)}, "constants": {}},
        ]
        return [0, 1], sources, ["baseColor"], src, dst, ids

    def test_a_squeezed_source_is_reported_not_silently_resampled(self):
        used, sources, channels, src, dst, ids = self._consolidation(0.575, 0.094)
        lines = []
        size = ptk.UvTransfer._auto_size(
            used, sources, channels, src, dst, ids, say=lines.append
        )
        # The size is the caller's to choose; a 2k map is ample for a small
        # asset and the tool does not inflate one on its own initiative.
        self.assertEqual(size, 256)
        said = " ".join(lines)
        self.assertIn("wires", said, "the squeezed source is not named")
        self.assertIn("3.16x", said, "the squeeze factor is not quantified")
        self.assertIn("81px of its 256px", said, "the cost is not stated in pixels")

    def test_a_one_to_one_transfer_says_nothing(self):
        """No consolidation, no loss, no line — the common re-bake stays quiet."""
        sources = [{"maps": {"baseColor": self._map("only.png", 512)}, "constants": {}}]
        lines = []
        size = ptk.UvTransfer._auto_size(
            [0],
            sources,
            ["baseColor"],
            QUAD.copy(),
            QUAD.copy(),
            np.zeros(2, np.int32),
            say=lines.append,
        )
        self.assertEqual(size, 512)
        self.assertEqual(lines, [])

    def test_a_roomier_target_is_not_a_complaint(self):
        """Landing on MORE of the target than it owned at source loses nothing."""
        sources = [{"maps": {"baseColor": self._map("s.png", 512)}, "constants": {}}]
        lines = []
        size = ptk.UvTransfer._auto_size(
            [0],
            sources,
            ["baseColor"],
            self._quad(0.25),
            self._quad(1.0),
            np.zeros(2, np.int32),
            say=lines.append,
        )
        self.assertEqual(size, 512)
        self.assertEqual(lines, [])

    def test_the_report_names_the_map_it_measured(self):
        """The label must come from the map that set the size, not whichever
        map happens to be first in the dict -- an unresolvable path or a
        channel this run never touched would otherwise name the culprit."""
        src = np.concatenate([self._quad(0.90), self._quad(0.94)])
        dst = np.concatenate([self._quad(0.575), self._quad(0.094, offset=(2.0, 2.0))])
        ids = np.array([0, 0, 1, 1], np.int32)
        sources = [
            {
                "maps": {"baseColor": self._map("turret_color.png", 256)},
                "constants": {},
            },
            {
                "maps": {
                    # First in the dict, and it does not exist.
                    "roughness": os.path.join(self.tmp, "missing_rough.png"),
                    "baseColor": self._map("wire_color.png", 256),
                },
                "constants": {},
            },
        ]
        lines = []
        ptk.UvTransfer._auto_size(
            [0, 1],
            sources,
            ["roughness", "baseColor"],
            src,
            dst,
            ids,
            say=lines.append,
        )
        said = " ".join(lines)
        self.assertIn("wire_color", said)
        self.assertNotIn("missing_rough", said)

    def test_geometry_is_optional(self):
        """A caller that cannot supply UVs gets the plain floor and no report."""
        sources = [{"maps": {"baseColor": self._map("a.png", 1024)}, "constants": {}}]
        lines = []
        size = ptk.UvTransfer._auto_size([0], sources, ["baseColor"], say=lines.append)
        self.assertEqual(size, 1024)
        self.assertEqual(lines, [])


class TestDominantSource(unittest.TestCase):
    """The source material a layout's assigned copy is modelled on: the one
    covering the most TARGET-UV area, not the most triangles -- a dense
    trim strip must not outvote the panel that fills the layout."""

    def test_area_wins_over_triangle_count(self):
        big = QUAD  # 2 triangles filling the square
        small = np.concatenate([QUAD * 0.1] * 3)  # 6 tiny triangles
        job = {
            "dst": np.concatenate([big, small]),
            "ids": np.array([0, 0] + [1] * 6),
            "sources": [{"name": "panel"}, {"name": "trim"}],
        }
        self.assertEqual(ptk.UvTransfer.dominant_source(job), 0)

    def test_a_job_naming_no_sources_has_none(self):
        self.assertIsNone(
            ptk.UvTransfer.dominant_source({"dst": QUAD, "ids": [0, 0], "sources": []})
        )


class TestMergeLayouts(unittest.TestCase):
    """A layout is the unit: disjoint per-material jobs on one set merge."""

    SOURCES = [{"maps": {}, "constants": {}}]

    @classmethod
    def _job(cls, dst, members):
        return {
            "src": QUAD,
            "dst": dst,
            "ids": np.zeros(2, np.int32),
            "sources": cls.SOURCES,
            "members": members,
        }

    def test_disjoint_jobs_merge_under_the_set_name(self):
        left = self._job(QUAD * [0.5, 1.0], ["a"])
        right = self._job(QUAD * [0.5, 1.0] + [0.5, 0.0], ["b"])
        merged = ptk.UvTransfer.merge_layouts({"matA": left, "matB": right}, "map2")
        self.assertEqual(list(merged), ["map2"])
        self.assertEqual(len(merged["map2"]["dst"]), 4)
        self.assertEqual(merged["map2"]["members"], ["a", "b"])
        # The source registry is shared by every job, not concatenated.
        self.assertIs(merged["map2"]["sources"], self.SOURCES)

    def test_overlapping_jobs_stay_apart(self):
        a = self._job(QUAD, ["a"])
        b = self._job(QUAD * 0.75, ["b"])  # inside a's square
        merged = ptk.UvTransfer.merge_layouts({"matA": a, "matB": b}, "map1")
        self.assertEqual(set(merged), {"matA", "matB"})

    def test_single_job_passes_through(self):
        a = self._job(QUAD, ["a"])
        self.assertEqual(list(ptk.UvTransfer.merge_layouts({"only": a}, "x")), ["only"])


class TestLayoutJobs(unittest.TestCase):
    """A run's outputs: what each target contributed, grouped into LAYOUTS --
    by overlap, never by the names a host reads off the targets."""

    SOURCES = [{"maps": {}, "constants": {}}]

    @staticmethod
    def _part(material, uv_set, dst, member):
        return {
            "material": material,
            "uv_set": uv_set,
            "src": QUAD,
            "dst": dst,
            "ids": np.zeros(2, np.int32),
            "members": [member],
        }

    def test_two_set_names_that_do_not_overlap_are_one_layout(self):
        """A production table whose parts came in through Maya (``map1``) and
        an FBX (``UVChannel_1``) was ONE combined layout, and grouping by set
        name transferred it into two materials (2026-10-03)."""
        jobs = ptk.UvTransfer.layout_jobs(
            [
                self._part("tableMat", "map1", QUAD * [0.5, 1.0], "mat"),
                self._part(
                    "tableMat", "UVChannel_1", QUAD * [0.5, 1.0] + [0.5, 0.0], "legs"
                ),
            ],
            self.SOURCES,
        )
        self.assertEqual(list(jobs), ["map1_UVChannel_1"])
        job = jobs["map1_UVChannel_1"]
        self.assertEqual(len(job["dst"]), 4)
        self.assertEqual(job["members"], ["mat", "legs"])
        self.assertIs(job["sources"], self.SOURCES)

    def test_faces_that_wear_nothing_are_still_a_layout(self):
        """What a target wears says nothing about where its texels go: faces
        whose shading group lost its shader are transferred like any other."""
        jobs = ptk.UvTransfer.layout_jobs(
            [self._part(None, "map1", QUAD, "bare")], self.SOURCES
        )
        self.assertEqual(list(jobs), ["map1"])

    def test_one_material_on_one_set_keeps_its_name(self):
        jobs = ptk.UvTransfer.layout_jobs(
            [self._part("matA", "map1", QUAD, "a")], self.SOURCES
        )
        self.assertEqual(list(jobs), ["matA"])

    def test_parts_of_one_material_and_set_are_one_group(self):
        jobs = ptk.UvTransfer.layout_jobs(
            [
                self._part("matA", "map1", QUAD * [0.5, 1.0], "a"),
                self._part("matA", "map1", QUAD * [0.5, 1.0] + [0.5, 0.0], "b"),
            ],
            self.SOURCES,
        )
        self.assertEqual(list(jobs), ["matA"])
        self.assertEqual(jobs["matA"]["members"], ["a", "b"])

    def test_overlapping_groups_stay_apart_named_by_material(self):
        jobs = ptk.UvTransfer.layout_jobs(
            [
                self._part("matA", "map1", QUAD, "a"),
                self._part("matB", "map1", QUAD * 0.75, "b"),
            ],
            self.SOURCES,
        )
        self.assertEqual(set(jobs), {"matA", "matB"})

    def test_a_material_two_sets_share_is_named_by_each_set(self):
        jobs = ptk.UvTransfer.layout_jobs(
            [
                self._part("matA", "map1", QUAD, "a"),
                self._part("matA", "uv2", QUAD * 0.75, "b"),
            ],
            self.SOURCES,
        )
        self.assertEqual(set(jobs), {"map1_matA", "uv2_matA"})

    def test_nothing_contributed_is_no_job(self):
        self.assertEqual(ptk.UvTransfer.layout_jobs([], self.SOURCES), {})


class TestTransferMaterialsNormals(unittest.TestCase):
    """``transfer_materials`` reads each source normal map's OWN convention."""

    def setUp(self):
        artifacts = ptk.TempArtifacts("uv_transfer_normals", policy="scoped")
        self.addCleanup(artifacts.cleanup)
        self.tmp = artifacts.dir_path()

    def _map(self, name, img):
        return ptk.UvTransfer.save_map(os.path.join(self.tmp, name), img)

    def _normal(self, jobs, size):
        out = ptk.UvTransfer.transfer_materials(
            jobs,
            output_dir=os.path.join(self.tmp, "out"),
            channels=["normal"],
            size=size,
            supersample=1,
            padding=0,
        )
        label = next(iter(jobs))
        return ptk.UvTransfer.load_map(out[label]["normal"])[0]

    def test_an_untagged_directx_map_rotates_as_directx(self):
        relief = _relief_normals(64, "directx")
        path = self._map("rock_Normal.png", relief)
        got = self._normal(
            {
                "rock": {
                    "src": QUAD,
                    "dst": _rot90(QUAD),
                    "ids": np.zeros(2, np.int32),
                    "sources": [{"maps": {"normal": path}, "constants": {}}],
                }
            },
            64,
        )
        table = ptk.UvTransfer.build(QUAD, _rot90(QUAD), 64, supersample=1)
        as_dx, _ = ptk.UvTransfer.transfer_normals(table, relief, convention="directx")
        as_gl, _ = ptk.UvTransfer.transfer_normals(table, relief, convention="opengl")
        self.assertLess(np.abs(got - as_dx).max(), 1.5)
        self.assertGreater(np.abs(got - as_gl).max(), 50.0)

    def test_sources_in_two_conventions_come_out_in_one(self):
        """Consolidated into ONE map, a source in the other convention is
        converted (green flipped) first -- the convention of the source
        covering most of the layout wins -- or that part of the result is lit
        upside down."""
        tilt = (0.0, 0.6, 0.8)
        gl = np.empty((16, 16, 3), np.float32)
        gl[:] = _encode(tilt)
        dx = np.empty((16, 16, 3), np.float32)
        dx[:] = _encode((0.0, -0.6, 0.8))  # the SAME tilt, stored DirectX
        # Flat maps: the content cannot tell, so the filenames decide.
        a = self._map("a_Normal_OpenGL.png", gl)
        b = self._map("b_Normal_DirectX.png", dx)
        got = self._normal(
            {
                "m": {
                    "src": np.concatenate([QUAD, QUAD]),
                    "dst": np.concatenate(
                        [QUAD * [0.75, 1.0], QUAD * [0.25, 1.0] + [0.75, 0.0]]
                    ),
                    "ids": np.array([0, 0, 1, 1], np.int32),
                    "sources": [
                        {"maps": {"normal": a}, "constants": {}},
                        {"maps": {"normal": b}, "constants": {}},
                    ],
                }
            },
            16,
        )
        self.assertLess(np.abs(got - _encode(tilt)).max(), 1.5)


class TestMissingSourceMaps(unittest.TestCase):
    """A source map the material names but the disk does not hold is SAID.

    A production tray material named ``SOLDERING_TRAYS_Emission.png`` after
    the file had become ``..._Emissive.png``; the transfer dropped the
    emission channel without a word (2026-10-03)."""

    def setUp(self):
        artifacts = ptk.TempArtifacts("uv_transfer_missing", policy="scoped")
        self.addCleanup(artifacts.cleanup)
        self.tmp = artifacts.dir_path()

    def _run(self, maps_a, maps_b=None):
        sources = [{"maps": maps_a, "constants": {}}]
        ids = np.zeros(2, np.int32)
        dst = QUAD
        if maps_b is not None:
            sources.append({"maps": maps_b, "constants": {}})
            ids = np.array([0, 0, 1, 1], np.int32)
            dst = np.concatenate([QUAD * [0.5, 1.0], QUAD * [0.5, 1.0] + [0.5, 0.0]])
        said = []
        out = ptk.UvTransfer.transfer_materials(
            {
                "m": {
                    "src": np.concatenate([QUAD] * (len(ids) // 2)),
                    "dst": dst,
                    "ids": ids,
                    "sources": sources,
                }
            },
            output_dir=os.path.join(self.tmp, "out"),
            size=8,
            supersample=1,
            padding=0,
            log=said.append,
        )
        return out["m"], "\n".join(said)

    def test_a_channel_whose_maps_are_all_missing_is_reported(self):
        gone = os.path.join(self.tmp, "tray_Emission.png")
        written, said = self._run({"emission": gone})
        self.assertNotIn("emission", written)
        self.assertIn("tray_Emission.png", said)
        self.assertIn("not on disk", said)

    def test_a_missing_map_beside_a_present_one_is_reported(self):
        here = ptk.UvTransfer.save_map(
            os.path.join(self.tmp, "a_BaseColor.png"),
            np.full((8, 8, 3), 200, np.float32),
        )
        gone = os.path.join(self.tmp, "b_BaseColor.png")
        written, said = self._run({"baseColor": here}, {"baseColor": gone})
        self.assertIn("baseColor", written)
        self.assertIn("b_BaseColor.png", said)


class TestTransferMaterialsOutputs(unittest.TestCase):
    """Where ``transfer_materials`` writes: never over a file it is told to
    avoid, nor over another output of the same run."""

    def setUp(self):
        artifacts = ptk.TempArtifacts("uv_transfer_outputs", policy="scoped")
        self.addCleanup(artifacts.cleanup)
        self.tmp = artifacts.dir_path()
        self.out_dir = os.path.join(self.tmp, "out")

    def _source(self, folder, name="held_BaseColor.png"):
        return ptk.UvTransfer.save_map(os.path.join(folder, name), _noise(16))

    @staticmethod
    def _job(path):
        return {
            "src": QUAD,
            "dst": _mirror_u(QUAD),
            "ids": np.zeros(2, np.int32),
            "sources": [{"name": "held_MAT", "maps": {"baseColor": path}}],
        }

    def _transfer(self, jobs, **kwargs):
        return ptk.UvTransfer.transfer_materials(
            jobs, output_dir=self.out_dir, size=16, supersample=1, padding=0, **kwargs
        )

    @staticmethod
    def _same(a, b):
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(
            os.path.abspath(b)
        )

    def test_a_map_to_avoid_is_never_written_over(self):
        """A re-run named like the last one wrote ``held_BaseColor.png`` over
        the very file a kept SOURCE material reads (an iteration on its own
        previous result): the source's look was lost for good. The host names
        the maps that must survive the run."""
        src = self._source(self.out_dir)
        with open(src, "rb") as f:
            before = f.read()
        out = self._transfer(
            {"held": self._job(src)}, name_format="held_{channel}", avoid=[src]
        )
        with open(src, "rb") as f:
            self.assertEqual(f.read(), before)
        written = out["held"]["baseColor"]
        self.assertFalse(self._same(written, src))
        self.assertTrue(
            self._same(written, os.path.join(self.out_dir, "held_BaseColor_1.png"))
        )
        self.assertTrue(os.path.isfile(written))

    def test_a_re_run_rewrites_its_own_maps(self):
        """Nothing to avoid: a re-run is another attempt at one deliverable and
        writes the same files, never a second set beside them."""
        src = self._source(self.tmp)
        first = self._transfer({"held": self._job(src)})
        self.assertEqual(self._transfer({"held": self._job(src)}), first)

    def test_two_outputs_of_one_run_never_share_a_file(self):
        """Two layouts whose names sanitize alike (``a b``, ``a_b``) wrote ONE
        file, the second over the first, and both materials were handed it."""
        src = self._source(self.tmp)
        out = self._transfer({"a b": self._job(src), "a_b": self._job(src)})
        first, second = out["a b"]["baseColor"], out["a_b"]["baseColor"]
        self.assertFalse(self._same(first, second))
        self.assertTrue(os.path.isfile(first) and os.path.isfile(second))


class TestInvalidSourceNormals(unittest.TestCase):
    """Texels that are no tangent-space normal (Z below zero: a black,
    unpainted background) transfer as they are -- and are SAID, naming the
    map. A production table's normal map was black under some of its faces;
    turned with their island, they came out as a yellow strip that read as a
    transfer bug (2026-10-03)."""

    def setUp(self):
        artifacts = ptk.TempArtifacts("uv_transfer_badnormal", policy="scoped")
        self.addCleanup(artifacts.cleanup)
        self.tmp = artifacts.dir_path()

    def _transfer(self, name, img):
        """``(written channels, what was said)`` for a 90-degree turn of *img*."""
        path = ptk.UvTransfer.save_map(os.path.join(self.tmp, name), img)
        said = []
        out = ptk.UvTransfer.transfer_materials(
            {
                "m": {
                    "src": QUAD,
                    "dst": _rot90(QUAD),
                    "ids": np.zeros(2, np.int32),
                    "sources": [{"name": "mat", "maps": {"normal": path}}],
                }
            },
            output_dir=os.path.join(self.tmp, "out"),
            size=16,
            supersample=1,
            padding=0,
            log=said.append,
        )
        return out["m"], "\n".join(said)

    def test_black_source_texels_are_reported_by_map(self):
        img = np.empty((16, 16, 3), np.float32)
        img[:] = _encode((0.0, 0.0, 1.0))
        img[:, :4] = 0.0  # an unpainted, black strip
        written, said = self._transfer("table_Normal.png", img)
        self.assertIn("normal", written)  # transferred, not dropped
        self.assertIn("table_Normal.png", said)
        self.assertIn("no tangent-space normal", said)

    def test_a_valid_map_says_nothing_of_the_kind(self):
        img = np.empty((16, 16, 3), np.float32)
        img[:] = _encode((0.6, 0.0, 0.8))
        _written, said = self._transfer("ok_Normal.png", img)
        self.assertNotIn("no tangent-space normal", said)


class TestLayoutOverlaps(unittest.TestCase):
    """Whether a lightmap layout can be baked: two surface points on one texel."""

    #: A unit quad in the XZ plane, two triangles, positions matching ``QUAD``.
    PTS = np.array(
        [[[0, 0, 0], [1, 0, 0], [1, 0, 1]], [[0, 0, 0], [1, 0, 1], [0, 0, 1]]],
        dtype=float,
    )

    def test_a_clean_layout_has_no_overlap_on_its_shared_edges(self):
        overlaps, covered = ptk.UvTransfer.layout_overlaps(QUAD * 0.5, self.PTS)
        self.assertEqual(overlaps, 0)
        self.assertGreater(covered, 0)

    def test_two_islands_stacked_on_one_texel_overlap(self):
        """A second quad elsewhere in space, laid on the same UVs (a stacked or
        mirrored island -- fine for a texture, unbakeable for a lightmap)."""
        uv = np.concatenate([QUAD * 0.5, QUAD * 0.5])
        pts = np.concatenate([self.PTS, self.PTS + [0, 2, 0]])
        overlaps, covered = ptk.UvTransfer.layout_overlaps(uv, pts)
        self.assertGreater(overlaps, covered // 2)

    def test_the_same_islands_side_by_side_do_not(self):
        uv = np.concatenate([QUAD * 0.5, QUAD * 0.5 + [0.5, 0]])
        pts = np.concatenate([self.PTS, self.PTS + [0, 2, 0]])
        self.assertEqual(ptk.UvTransfer.layout_overlaps(uv, pts)[0], 0)


class TestOutputLabels(unittest.TestCase):
    """A named re-run reads its own result back as a label; the name must not stack."""

    def test_a_re_run_names_what_the_first_run_named(self):
        first = ptk.UvTransfer.output_labels(
            ["TABLE_ASSETS", "UVChannel_1"], "SolderingTable", suffix="_MAT"
        )
        # The second run's targets wear the first run's materials.
        again = ptk.UvTransfer.output_labels(
            ["SolderingTable_TABLE_ASSETS_MAT", "SolderingTable_UVChannel_1_MAT"],
            "SolderingTable",
            suffix="_MAT",
        )
        self.assertEqual(sorted(first.values()), ["TABLE_ASSETS", "UVChannel_1"])
        self.assertEqual(sorted(again.values()), ["TABLE_ASSETS", "UVChannel_1"])

    def test_every_stacked_copy_comes_off(self):
        # The production scene after four runs.
        label = "SolderingTable_SolderingTable_SolderingTable_TABLE_ASSETS_MAT"
        out = ptk.UvTransfer.output_labels(
            [label, "SolderingTable_UVChannel_1_MAT"], "SolderingTable", suffix="_MAT"
        )
        self.assertEqual(out[label], "TABLE_ASSETS")

    def test_the_material_prefix_comes_off_too(self):
        out = ptk.UvTransfer.output_labels(
            ["MAT_hero_matA", "matB"], "hero", prefix="MAT_"
        )
        self.assertEqual(out, {"MAT_hero_matA": "matA", "matB": "matB"})

    def test_only_a_whole_word_is_the_name(self):
        # `heroic` is not `hero_` + something; the case must match as well.
        out = ptk.UvTransfer.output_labels(["heroic_mat", "Hero_mat2"], "hero")
        self.assertEqual(out, {"heroic_mat": "heroic_mat", "Hero_mat2": "Hero_mat2"})

    def test_a_label_that_is_only_the_name_keeps_it(self):
        out = ptk.UvTransfer.output_labels(
            ["hero", "hero_MAT", "other"], "hero", suffix="_MAT"
        )
        self.assertEqual(out["hero"], "hero")
        self.assertEqual(out["hero_MAT"], "hero_MAT")

    def test_two_layouts_never_share_a_stem(self):
        # Both strip to `matA`, and a third already IS `matA`: nobody moves.
        labels = ["hero_matA", "hero_hero_matA", "matA"]
        out = ptk.UvTransfer.output_labels(labels, "hero")
        self.assertEqual(out, {label: label for label in labels})
        self.assertEqual(len(set(out.values())), len(labels))

    def test_no_name_leaves_every_label_alone(self):
        out = ptk.UvTransfer.output_labels(["x_MAT", "y"], "", suffix="_MAT")
        self.assertEqual(out, {"x_MAT": "x_MAT", "y": "y"})


class TestPad(unittest.TestCase):
    def test_full_fill_leaves_no_background(self):
        dst = QUAD * 0.5
        t = ptk.UvTransfer.build(QUAD, dst, 16, supersample=1)
        out, cov = ptk.UvTransfer.transfer(t, np.full((16, 16, 3), 200.0, np.float32))
        self.assertTrue((out[0, -1] == 0).all())  # gutter before padding
        padded = ptk.UvTransfer.pad(out, cov, -1)
        self.assertTrue(np.allclose(padded, 200.0))

    def test_fixed_width_pads_only_that_far(self):
        dst = QUAD * 0.5
        t = ptk.UvTransfer.build(QUAD, dst, 32, supersample=1)
        out, cov = ptk.UvTransfer.transfer(t, np.full((32, 32, 3), 200.0, np.float32))
        padded = ptk.UvTransfer.pad(out, cov, 2)
        # Island occupies the bottom-left 16x16; two texels beyond it filled.
        self.assertTrue(np.allclose(padded[16:, 16:18], 200.0))
        self.assertTrue((padded[0, -1] == 0).all())

    def test_width_zero_is_copy(self):
        t = ptk.UvTransfer.build(QUAD * 0.5, QUAD * 0.5, 8, supersample=1)
        out, cov = ptk.UvTransfer.transfer(t, np.ones((8, 8, 3), np.float32))
        self.assertTrue(np.array_equal(ptk.UvTransfer.pad(out, cov, 0), out))


class TestNormalConvention(unittest.TestCase):
    """Handedness is read off the map's CONTENT, then its FILENAME through the
    shared map registry (the names below are not files: the filename path).

    Getting it wrong inverts a normal map's green channel, and rotated islands
    mix X into Y, so a miss is not a cosmetic error -- it is a wrong transfer.
    """

    def _write(self, name, img):
        if not hasattr(self, "tmp"):
            artifacts = ptk.TempArtifacts("uv_normal_convention", policy="scoped")
            self.addCleanup(artifacts.cleanup)
            self.tmp = artifacts.dir_path()
        return ptk.UvTransfer.save_map(os.path.join(self.tmp, name), img)

    def test_the_content_outranks_the_filename(self):
        """What the rotation needs is the RELATIVE handedness of X and Y, which
        the content measures; a filename only claims it."""
        path = self._write(
            "rock_Normal_OpenGL.png", _relief_normals(convention="directx")
        )
        self.assertEqual(ptk.UvTransfer.normal_convention(path), "directx")
        path = self._write("rock_NormalDX.png", _relief_normals(convention="opengl"))
        self.assertEqual(ptk.UvTransfer.normal_convention(path), "opengl")

    def test_an_untagged_map_reads_its_content(self):
        """``_Normal`` says nothing; read as OpenGL, a DirectX map's rotated
        islands turned the wrong way."""
        path = self._write("rock_Normal.png", _relief_normals(convention="directx"))
        self.assertEqual(ptk.UvTransfer.normal_convention(path), "directx")

    def test_a_flat_map_falls_back_to_its_filename(self):
        flat = np.empty((32, 32, 3), np.float32)
        flat[:] = (128, 128, 255)
        dx = self._write("flat_NormalDX.png", flat)
        self.assertEqual(ptk.UvTransfer.normal_convention(dx), "directx")
        plain = self._write("flat_Normal.png", flat)
        self.assertEqual(ptk.UvTransfer.normal_convention(plain), "opengl")

    def test_directx_spellings_across_delimiters_and_stems(self):
        for name in (
            "rock_NormalDX.png",
            "rock_Normal_DirectX.png",
            "rock_nrml_dx.png",
            "rock-n-dx.png",
            "rock_NDX.png",
            "rock_dx.png",
        ):
            self.assertEqual(ptk.UvTransfer.normal_convention(name), "directx", name)

    def test_opengl_spellings_and_the_untagged_default(self):
        for name in (
            "rock_NormalGL.png",
            "rock_NRMLOGL.png",
            "rock_Normal_OpenGL.png",
            "rock_Normal.png",  # untagged -- convention unknown, do not flip
            "rock_basecolor.png",  # not a normal map at all
        ):
            self.assertEqual(ptk.UvTransfer.normal_convention(name), "opengl", name)

    def test_the_tag_may_lead_the_token(self):
        """``DX_Normal`` is as real a spelling as ``Normal_DX``.

        Only the token-first order was enumerated, so a leading tag fell
        through to the untagged ``Normal`` type and the map read as OpenGL --
        an inverted green channel, silently. That the abbreviation ``DXN``
        already classified (hand-listed) while ``DXNormal`` did not is what
        marks this a gap rather than a rule: whatever the rule is, those two
        spellings have to agree.
        """
        for name in (
            "rock_DX_Normal.png",
            "rock_DXNormal.png",
            "rock_dx_nrm.png",
            "rock_DirectX_Normal.png",
            "rock-dx-nrml.png",
        ):
            self.assertEqual(ptk.UvTransfer.normal_convention(name), "directx", name)
        for name in (
            "rock_GL_Normal.png",
            "rock_OGL_nrml.png",
            "rock_OpenGLNormal.png",
        ):
            self.assertEqual(ptk.UvTransfer.normal_convention(name), "opengl", name)

    def test_a_tag_detached_from_the_token_still_does_not_count(self):
        """The 2026-08-19 narrowing that must survive: a tag loose in the name
        is not a declaration. Only a tag ADJACENT to the token is a suffix."""
        for name in (
            "DirectX_rock_Normal.png",
            "rock_directx_final_normal.png",
            "C:/tex/dx_project/rock_Normal.png",
        ):
            self.assertEqual(ptk.UvTransfer.normal_convention(name), "opengl", name)

    def test_udim_and_duplicate_tokens_do_not_hide_the_tag(self):
        """The regression the local token regex had: a trailing ``_1`` or tile
        pushed the tag off the end of the name and the map read as OpenGL."""
        for name in (
            "rock_NormalDX.1001.png",
            "rock_NormalDX_1.png",
            "rock_Normal_DirectX_1.png",
        ):
            self.assertEqual(ptk.UvTransfer.normal_convention(name), "directx", name)

    def test_override_wins_over_classification(self):
        self.assertEqual(
            ptk.UvTransfer.normal_convention("rock_NormalDX.png", "opengl"), "opengl"
        )
        self.assertEqual(
            ptk.UvTransfer.normal_convention("rock_Normal.png", "DirectX"), "directx"
        )

    def test_a_full_path_classifies_by_its_filename(self):
        self.assertEqual(
            ptk.UvTransfer.normal_convention("C:/tex/dx_project/rock_NormalGL.png"),
            "opengl",
        )


class TestRemapLightmap(unittest.TestCase):
    """A committed lightmap carried into another lightmap layout: HDR in, HDR
    out, read through the object's own atlas rect."""

    @staticmethod
    def _hdr(size):
        """Smooth HDR content well past 1.0 (and past 8-bit's 255)."""
        g = _gradient(size) / 255.0
        return (g * 300.0 + 0.5).astype(np.float32)

    def test_identity_layout_keeps_hdr_values(self):
        img = self._hdr(32)
        out = ptk.UvTransfer.remap_lightmap(img, QUAD, QUAD)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(out.max(), 255.0)  # never clipped to a display range
        # 1% of the 300 range: the supersampled taps at the map's outer edge
        # (the bound the 8-bit identity tests allow, scaled to this ramp).
        self.assertLess(np.abs(out - img).max(), 3.0)

    def test_a_rotated_target_layout_rotates_the_map(self):
        img = self._hdr(32)
        out = ptk.UvTransfer.remap_lightmap(img, QUAD, _rot90(QUAD), supersample=1)
        self.assertLess(np.abs(out - np.rot90(img, 1)).max(), 1.0)

    def test_an_atlas_rect_reads_only_its_own_cell(self):
        """The object owns the LEFT half of a shared map; its neighbour's much
        brighter texels must not bleed in across the cell edge."""
        atlas = np.full((32, 32, 3), 2.0, np.float32)
        atlas[:, 16:] = 50.0  # another object's lighting
        out = ptk.UvTransfer.remap_lightmap(
            atlas, QUAD, QUAD, scale_offset=[0.5, 1.0, 0.0, 0.0], size=16
        )
        self.assertTrue(np.allclose(out, 2.0, atol=1e-4), out.max())

    def test_the_rect_is_v_up_from_the_bottom(self):
        """``offsetY`` counts from the BOTTOM of the map (Unity's convention),
        and images store row 0 at the top."""
        atlas = np.zeros((32, 32, 1), np.float32)
        atlas[16:] = 7.0  # bottom half of the stored image = V in [0, 0.5]
        out = ptk.UvTransfer.remap_lightmap(
            atlas, QUAD, QUAD, scale_offset=[1.0, 0.5, 0.0, 0.0], size=16
        )
        self.assertTrue(np.allclose(out, 7.0, atol=1e-4))

    def test_default_size_is_the_share_the_object_owned(self):
        img = np.ones((64, 64, 3), np.float32)
        size = lambda so: ptk.UvTransfer.remap_lightmap(  # noqa: E731
            img, QUAD, QUAD, scale_offset=so, supersample=1
        ).shape[0]
        self.assertEqual(size(None), 64)
        self.assertEqual(size([0.25, 0.25, 0.0, 0.0]), 16)
        # A non power-of-two share rounds UP, so no texel density is lost...
        self.assertEqual(size([0.3, 0.2, 0.0, 0.0]), 32)
        # ...but never past the map it came from.
        self.assertEqual(
            ptk.UvTransfer.remap_lightmap(
                np.ones((48, 48, 3), np.float32), QUAD, QUAD, supersample=1
            ).shape[0],
            48,
        )

    def test_an_explicit_size_wins(self):
        out = ptk.UvTransfer.remap_lightmap(self._hdr(32), QUAD, QUAD, size=8)
        self.assertEqual(out.shape[:2], (8, 8))

    def test_the_gutter_is_filled(self):
        """A target island smaller than the map leaves gutter; it is padded
        from the island, not left black for the engine's mips to average in."""
        img = np.full((16, 16, 3), 3.0, np.float32)
        out = ptk.UvTransfer.remap_lightmap(img, QUAD, QUAD * 0.5, size=16)
        self.assertTrue(np.allclose(out, 3.0, atol=1e-4))

    def test_an_interior_atlas_cell_reads_its_own_texels(self):
        """Offsets on both axes, away from the map's edges: the read is cropped
        to the cell, so the crop's own coordinates must land on the same
        texels the whole map would have given."""
        atlas = np.zeros((64, 64, 3), np.float32)
        for row in range(4):
            for col in range(4):
                atlas[row * 16 : (row + 1) * 16, col * 16 : (col + 1) * 16] = (
                    row * 4 + col + 1
                )
        atlas[16:32, 32:48, 0] = _gradient(16)[..., 0] / 255.0 * 9.0 + 100.0
        # Cell (row 1 from the top, column 2): U in [0.5, 0.75], V in [0.5, 0.75].
        rect = [0.25, 0.25, 0.5, 0.5]
        out = ptk.UvTransfer.remap_lightmap(
            atlas, QUAD, QUAD, scale_offset=rect, size=16, supersample=1
        )
        expected = atlas[16:32, 32:48]
        self.assertLess(np.abs(out - expected).max(), 1e-3)

    def test_layouts_match(self):
        self.assertTrue(ptk.UvTransfer.layouts_match(QUAD, QUAD.copy()))
        self.assertFalse(ptk.UvTransfer.layouts_match(QUAD, QUAD + 1e-3))
        self.assertFalse(ptk.UvTransfer.layouts_match(QUAD, QUAD[:1]))
        self.assertTrue(ptk.UvTransfer.layouts_match(QUAD[:0], QUAD[:0]))


class TestResampleLightmaps(unittest.TestCase):
    """``resample_lightmaps`` -- the hosts' shared naming + remap + write."""

    def setUp(self):
        self.reads = []
        self.writes = {}
        self.maps = {"/src/a.exr": np.full((16, 16, 3), 2.0, np.float32)}

    def _read(self, path):
        self.reads.append(path)
        return self.maps[path]

    def _write(self, path, image):
        self.writes[path] = image

    def _job(self, owner, dst=None, path="/src/a.exr"):
        return {
            "owner": owner,
            "name": owner.rsplit("|", 1)[-1],
            "path": path,
            "scale_offset": None,
            "src": QUAD,
            "dst": _rot90(QUAD) if dst is None else dst,
        }

    def _run(self, jobs, **kwargs):
        return ptk.UvTransfer.resample_lightmaps(
            jobs, output_dir="/out", read=self._read, write=self._write, **kwargs
        )

    def test_a_matching_layout_is_left_to_the_host_to_rebind(self):
        out = self._run([self._job("|keep", dst=QUAD.copy()), self._job("|rot")])
        self.assertEqual(list(out), ["|rot"])
        self.assertEqual(list(self.writes), [out["|rot"]])

    def test_one_map_takes_the_output_name_several_append_theirs(self):
        self.assertEqual(
            self._run([self._job("|a|crate")], output_name="hero"),
            {"|a|crate": "/out/hero_Lightmap.exr"},
        )
        out = self._run(
            [self._job("|a|crate"), self._job("|b|barrel")], output_name="hero"
        )
        self.assertEqual(out["|a|crate"], "/out/hero_crate_Lightmap.exr")
        self.assertEqual(out["|b|barrel"], "/out/hero_barrel_Lightmap.exr")
        self.assertEqual(
            self._run([self._job("|a|crate")]),
            {"|a|crate": "/out/crate_Lightmap.exr"},
        )

    def test_a_shared_source_map_is_read_once(self):
        self._run([self._job("|a|one"), self._job("|b|two"), self._job("|c|three")])
        self.assertEqual(self.reads, ["/src/a.exr"])

    def test_interleaved_source_maps_are_each_read_once(self):
        """Only the LAST map was kept, so jobs on maps a, b, a re-read a -- a
        4k float EXR each time. Each job still takes the name its place in
        the given order gives it, whatever order the maps are read in."""
        self.maps["/src/b.exr"] = np.full((16, 16, 3), 3.0, np.float32)
        out = self._run(
            [
                self._job("|q|one"),
                self._job("|p|crate", path="/src/b.exr"),
                self._job("|r|crate"),
            ]
        )
        self.assertEqual(self.reads, ["/src/a.exr", "/src/b.exr"])
        self.assertEqual(out["|p|crate"], "/out/crate_Lightmap.exr")
        self.assertEqual(out["|r|crate"], "/out/crate_Lightmap_1.exr")
        # ...and each from its own map.
        self.assertAlmostEqual(float(self.writes[out["|p|crate"]].max()), 3.0, 5)
        self.assertAlmostEqual(float(self.writes[out["|r|crate"]].max()), 2.0, 5)

    def test_a_name_someone_else_reads_is_never_written_over(self):
        out = self._run(
            [self._job("|a|crate")],
            output_name="hero",
            claims={"hero_lightmap.exr": {"|other"}},
        )
        self.assertEqual(out["|a|crate"], "/out/hero_Lightmap_1.exr")
        # ...while the target's own previous map is its to replace.
        out = self._run(
            [self._job("|a|crate")],
            output_name="hero",
            claims={"hero_lightmap.exr": {"|a|crate"}},
        )
        self.assertEqual(out["|a|crate"], "/out/hero_Lightmap.exr")

    def test_a_source_map_is_never_written_over(self):
        self.maps["/out/hero_Lightmap.exr"] = self.maps["/src/a.exr"]
        out = self._run(
            [self._job("|a|crate", path="/out/hero_Lightmap.exr")], output_name="hero"
        )
        self.assertEqual(out["|a|crate"], "/out/hero_Lightmap_1.exr")

    def test_the_written_map_is_the_remap(self):
        self.maps["/src/a.exr"] = TestRemapLightmap._hdr(16)
        out = self._run([self._job("|a|crate")], supersample=1)
        got = self.writes[out["|a|crate"]]
        self.assertLess(np.abs(got - np.rot90(self.maps["/src/a.exr"], 1)).max(), 1.0)


class TestConcatenationOrder(unittest.TestCase):
    """``UvTransfer.concatenation_order`` -- the parts of a combined mesh.

    Maya's Combine and Blender's Join both append each part's faces after the
    previous part's, its vertex indices offset by the vertices before it
    (measured on Maya 2025 and Blender 5.1). The order is read back from the
    topology; positions only break ties between identical pieces.
    """

    @staticmethod
    def _grid(n_quads, at=(0.0, 0.0, 0.0)):
        """A strip of *n_quads* separate quads standing at *at*."""
        counts = [4] * n_quads
        verts = list(range(4 * n_quads))
        points = [
            (at[0] + i * 2 + dx, at[1] + dy, at[2])
            for i in range(n_quads)
            for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))
        ]
        return counts, verts, points

    @staticmethod
    def _tri_fan(at=(0.0, 0.0, 0.0)):
        """Two triangles sharing an edge -- a topology no quad strip has."""
        points = [
            (at[0] + x, at[1] + y, at[2]) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))
        ]
        return [3, 3], [0, 1, 2, 0, 2, 3], points

    @staticmethod
    def _combine(*parts):
        counts, verts, points = [], [], []
        for c, v, p in parts:
            verts += [i + len(points) for i in v]
            counts += list(c)
            points += list(p)
        return counts, verts, points

    def _order(self, target, parts):
        return ptk.UvTransfer.concatenation_order(target, parts)

    def test_the_combine_order_is_read_back(self):
        quad, fan = self._grid(1), self._tri_fan(at=(5, 0, 0))
        self.assertEqual(self._order(self._combine(fan, quad), [quad, fan]), [1, 0])
        self.assertEqual(self._order(self._combine(quad, fan), [quad, fan]), [0, 1])

    def test_identical_pieces_resolve_by_where_they_stand(self):
        left, right = self._grid(1, at=(0, 0, 0)), self._grid(1, at=(9, 0, 0))
        self.assertEqual(self._order(self._combine(right, left), [left, right]), [1, 0])

    def test_a_target_built_from_some_of_the_parts(self):
        a, b, c = (
            self._grid(1),
            self._tri_fan(at=(4, 0, 0)),
            self._grid(2, at=(8, 0, 0)),
        )
        self.assertEqual(self._order(self._combine(c, a), [a, b, c]), [2, 0])

    def test_a_piece_elsewhere_still_matches_by_topology(self):
        """Topology is the contract; the host warns about the positions."""
        quad, fan = self._grid(1), self._tri_fan()
        moved = self._tri_fan(at=(0, 0, 7))
        self.assertEqual(self._order(self._combine(quad, moved), [quad, fan]), [0, 1])

    def test_a_prefix_match_backtracks(self):
        """One quad is a topological prefix of two: the greedy pick of the
        single quad leaves a face no part can fill, so the search backs up."""
        single, double = self._grid(1), self._grid(2)
        self.assertEqual(self._order(double, [single, double]), [1])

    def test_no_ordering_reproduces_the_target(self):
        quad, fan = self._grid(1), self._tri_fan()
        self.assertIsNone(self._order(self._combine(quad, fan, fan), [quad, fan]))
        self.assertIsNone(self._order(self._grid(3), [quad]))  # faces left over
        counts, verts, points = self._combine(quad, fan)
        stray = (counts, verts, points + [(0.0, 0.0, 99.0)])  # a loose vertex
        self.assertIsNone(self._order(stray, [quad, fan]))
        self.assertIsNone(self._order(([], [], []), [quad]))  # empty target

    def test_hundreds_of_pieces_do_not_recurse(self):
        """Iterative: a kit of more pieces than the recursion limit."""
        import sys

        n = sys.getrecursionlimit() + 200
        parts = [self._grid(1, at=(i * 3, 0, 0)) for i in range(n)]
        shuffled = list(reversed(range(n)))
        target = self._combine(*[parts[i] for i in shuffled])
        self.assertEqual(self._order(target, parts), shuffled)

    @staticmethod
    def _find(meshes, reads=None):
        def read(i):
            if reads is not None:
                reads.append(i)
            return meshes[i]

        return ptk.UvTransfer.find_combined([len(m[0]) for m in meshes], read)

    def test_find_combined_picks_the_target_out_of_a_selection(self):
        a, b = self._grid(1), self._tri_fan(at=(4, 0, 0))
        combined = self._combine(b, a)
        self.assertEqual(self._find([a, combined, b]), (1, [2, 0]))

    def test_find_combined_needs_every_other_mesh(self):
        a, b, c = (
            self._grid(1),
            self._tri_fan(at=(4, 0, 0)),
            self._grid(2, at=(9, 0, 0)),
        )
        self.assertIsNone(self._find([a, b, self._combine(a, c)]))  # b is not in it
        self.assertIsNone(self._find([a, self._combine(a, a)]))  # two: ambiguous

    def test_find_combined_reads_nothing_the_face_counts_rule_out(self):
        """The usual Auto run -- nothing combined -- costs only the counts."""
        reads = []
        a, b, c = self._grid(1), self._tri_fan(), self._grid(2, at=(9, 0, 0))
        self.assertIsNone(self._find([a, b, c], reads))  # 1 + 2 + 2: no half
        self.assertEqual(reads, [])
        combined = self._combine(b, a)
        self._find([a, combined, b], reads)
        self.assertEqual(sorted(reads), [0, 1, 2])  # each read once


if __name__ == "__main__":
    unittest.main()
