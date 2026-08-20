# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.geo_utils.uv_transfer (UvTransfer -- UV-to-UV texel remap).

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
    """Handedness is read off the FILENAME through the shared map registry.

    Getting it wrong inverts a normal map's green channel, and rotated islands
    mix X into Y, so a miss is not a cosmetic error -- it is a wrong bake.
    """

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


if __name__ == "__main__":
    unittest.main()
