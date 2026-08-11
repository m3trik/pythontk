# !/usr/bin/python
# coding=utf-8
"""Tests for the region-mask engine (``engines/textures/region_masks.py``)
and its :meth:`ImgUtils.rasterize_uv_triangles` primitive.
"""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

import pythontk as ptk
from pythontk.core_utils.engines.textures.region_masks import (
    RegionGroup,
    RegionGroupRegistry,
    RegionMaskManifest,
    RegionMaskPacker,
)

# A unit quad covering the left half of UV space (V-up), as two triangles.
LEFT_HALF = [
    [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0)],
    [(0.0, 0.0), (0.5, 1.0), (0.0, 1.0)],
]
# And the right half.
RIGHT_HALF = [
    [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0)],
    [(0.5, 0.0), (1.0, 1.0), (0.5, 1.0)],
]
# A quad overlapping both halves (center strip).
CENTER_STRIP = [
    [(0.4, 0.0), (0.6, 0.0), (0.6, 1.0)],
    [(0.4, 0.0), (0.6, 1.0), (0.4, 1.0)],
]


class RasterizeUvTrianglesTest(unittest.TestCase):
    """The general coverage primitive."""

    def test_left_half_coverage(self):
        cover = ptk.ImgUtils.rasterize_uv_triangles(LEFT_HALF, size=64, supersample=1)
        self.assertEqual(cover.shape, (64, 64))
        self.assertEqual(cover.dtype, np.uint8)
        # Solid inside, empty outside (sample away from the seam).
        self.assertEqual(cover[32, 8], 255)
        self.assertEqual(cover[32, 56], 0)

    def test_supersampled_edges_are_antialiased(self):
        # A triangle with a diagonal edge must produce midtones when
        # supersampled and none when not.
        tri = [[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]]
        hard = ptk.ImgUtils.rasterize_uv_triangles(tri, size=32, supersample=1)
        soft = ptk.ImgUtils.rasterize_uv_triangles(tri, size=32, supersample=4)
        self.assertTrue(set(np.unique(hard)) <= {0, 255})
        mid = (soft > 0) & (soft < 255)
        self.assertTrue(mid.any())

    def test_v_flip(self):
        # A strip along the BOTTOM of UV space (v in [0, 0.25]) must land at
        # the BOTTOM of the image array (high row indices).
        strip = [
            [(0.0, 0.0), (1.0, 0.0), (1.0, 0.25)],
            [(0.0, 0.0), (1.0, 0.25), (0.0, 0.25)],
        ]
        cover = ptk.ImgUtils.rasterize_uv_triangles(strip, size=64, supersample=1)
        self.assertEqual(cover[60, 32], 255)  # bottom rows filled
        self.assertEqual(cover[4, 32], 0)  # top rows empty

    def test_full_coverage_means_exactly_full(self):
        # A consumer thresholding on "this texel lies ENTIRELY inside" -- the
        # lightmap baker's coverage refill, which uses it to tell an island's
        # own texels from the ring a bake renders past its border -- needs 255
        # to mean exactly that. A quad ending mid-texel must leave that texel
        # strictly below 255 and its inside neighbour at 255.
        quad = [
            [(0.0, 0.0), (0.5078125, 0.0), (0.5078125, 1.0)],
            [(0.0, 0.0), (0.5078125, 1.0), (0.0, 1.0)],
        ]
        # size 64 -> the edge falls at texel 32.5: col 31 full, 32 half, 33 out.
        cover = ptk.ImgUtils.rasterize_uv_triangles(quad, size=64, supersample=4)
        self.assertEqual(cover[32, 31], 255)
        self.assertTrue(0 < cover[32, 32] < 255)
        self.assertEqual(cover[32, 33], 0)

    def test_geometry_outside_the_image_is_cropped_not_smeared(self):
        # Clamping a vertex would drag the edges meeting it across the image
        # and paint a wedge along the border; cropping the bbox leaves a
        # wholly-outside triangle contributing nothing.
        tri = [[(1.2, 0.4), (1.6, 0.5), (1.2, 0.6)]]
        cover = ptk.ImgUtils.rasterize_uv_triangles(tri, size=32, supersample=1)
        self.assertEqual(int(cover.max()), 0)


class RegionGroupTest(unittest.TestCase):
    def test_slot_range_enforced(self):
        with self.assertRaises(ValueError):
            RegionGroup(name="x", slot=4)

    def test_coerce_accepts_dict(self):
        g = RegionGroup.coerce({"name": "a", "slot": 1, "default": 0.5})
        self.assertEqual((g.name, g.slot, g.default), ("a", 1, 0.5))

    def test_attr_serializes_only_when_set(self):
        """The keyable-weights attr rides the wire; absent groups omit it."""
        keyed = RegionGroup(name="a", slot=0, attr="emissiveGroup_a")
        self.assertEqual(keyed.to_dict()["attr"], "emissiveGroup_a")
        self.assertNotIn("attr", RegionGroup(name="b", slot=1).to_dict())
        back = RegionGroup.coerce(keyed.to_dict())
        self.assertEqual(back.attr, "emissiveGroup_a")


class RegionMaskManifestTest(unittest.TestCase):
    def test_vertex_color_round_trip(self):
        m = RegionMaskManifest.vertex_color(
            [{"name": "headlights", "slot": 0}, {"name": "leds", "slot": 1, "default": 0.0}]
        )
        data = json.loads(m.to_json())
        self.assertEqual(data["schema"], RegionMaskManifest.SCHEMA_VERSION)
        self.assertEqual(data["encoding"], "vertex-color")
        self.assertEqual(data["color_set"], "emissiveGroups")
        self.assertNotIn("mask", data)  # channels-only fields omitted
        back = RegionMaskManifest.from_json(m.to_json())
        self.assertEqual([g.to_dict() for g in back.groups], [g.to_dict() for g in m.groups])

    def test_channels_round_trip(self):
        m = RegionMaskManifest.channels(
            [{"name": "a", "slot": 0}], mask=r"C:\out\prop_EMask.png", resolution=512
        )
        data = m.to_dict()
        self.assertEqual(data["mask"], "prop_EMask.png")  # basename only
        self.assertEqual(data["uv_channel"], 0)
        self.assertEqual(data["resolution"], 512)
        self.assertNotIn("color_set", data)

    def test_duplicate_slots_rejected(self):
        with self.assertRaises(ValueError):
            RegionMaskManifest.vertex_color(
                [{"name": "a", "slot": 0}, {"name": "b", "slot": 0}]
            )

    def test_duplicate_names_rejected(self):
        with self.assertRaises(ValueError):
            RegionMaskManifest.vertex_color(
                [{"name": "a", "slot": 0}, {"name": "a", "slot": 1}]
            )

    def test_unknown_encoding_rejected(self):
        with self.assertRaises(ValueError):
            RegionMaskManifest(groups=[], encoding="bogus")

    def test_save_load(self):
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "m.json")
            RegionMaskManifest.vertex_color([{"name": "a", "slot": 0}]).save(path)
            self.assertEqual(RegionMaskManifest.load(path).groups[0].name, "a")
        finally:
            shutil.rmtree(tmp)


class RegionGroupRegistryTest(unittest.TestCase):
    """The slot bookkeeping both DCC tools delegate to."""

    def setUp(self):
        self.store = {"value": None}
        self.writes = []

        def load():
            return self.store["value"]

        def save(text):
            self.writes.append(text)
            self.store["value"] = text or None

        self.reg = RegionGroupRegistry(load, save)

    def test_read_of_empty_store_does_not_write(self):
        """Reading must never create the carrier channel."""
        self.assertEqual(self.reg.read()["groups"], {})
        self.assertEqual(self.writes, [])

    def test_add_assigns_lowest_free_slot(self):
        self.assertEqual(self.reg.add("a"), (0, True))
        self.assertEqual(self.reg.add("b"), (1, True))
        self.assertEqual(self.reg.add("a"), (0, False))  # existing → no new slot

    def test_remove_retires_slot(self):
        self.reg.add("a")
        self.reg.add("b")
        self.assertEqual(self.reg.remove("a"), 0)
        self.assertEqual(self.reg.add("c"), (2, True))  # 0 retired, not reused

    def test_compact_reclaims(self):
        self.reg.add("a")
        self.reg.add("b")
        self.reg.remove("a")
        self.assertEqual(self.reg.compact(), [0])
        self.assertEqual(self.reg.add("c"), (0, True))

    def test_slot_exhaustion_raises(self):
        for i in range(4):
            self.reg.add(f"g{i}")
        with self.assertRaises(ValueError):
            self.reg.add("overflow")

    def test_emptying_clears_the_channel(self):
        self.reg.add("a")
        self.reg.remove("a")
        self.reg.compact()
        self.assertIsNone(self.store["value"])
        # ...and a further write on an already-clear store is a no-op.
        writes = len(self.writes)
        self.reg.write(self.reg.empty())
        self.assertEqual(len(self.writes), writes)

    def test_set_default_clamps(self):
        self.reg.add("a")
        self.assertEqual(self.reg.set_default("a", 5.0), 1.0)
        self.assertEqual(self.reg.set_default("a", -2.0), 0.0)
        with self.assertRaises(ValueError):
            self.reg.set_default("nope", 1.0)

    def test_groups_are_slot_ordered(self):
        self.reg.add("a")
        self.reg.add("b")
        self.reg.remove("a")
        self.reg.add("c")
        self.assertEqual([g["name"] for g in self.reg.groups()], ["b", "c"])

    def test_corrupt_payload_resets_instead_of_raising(self):
        self.store["value"] = "{not json"
        self.assertEqual(self.reg.read()["groups"], {})

    def test_manifest_follows_encoding(self):
        self.assertIsNone(self.reg.manifest())  # no groups → no manifest
        self.reg.add("a")
        self.assertEqual(self.reg.manifest().encoding, "vertex-color")
        self.reg.set_encoding("channels", mask="p_EMask.png", resolution=256)
        manifest = self.reg.manifest()
        self.assertEqual(manifest.encoding, "channels")
        self.assertEqual(manifest.mask, "p_EMask.png")
        self.assertEqual(manifest.resolution, 256)

    def test_sanitize(self):
        self.assertEqual(RegionGroupRegistry.sanitize(" head lights! "), "head_lights_")
        with self.assertRaises(ValueError):
            RegionGroupRegistry.sanitize("1bad")
        with self.assertRaises(ValueError):
            RegionGroupRegistry.sanitize("   ")

    def test_set_attr_records_clears_and_rides_the_manifest(self):
        """The keyable-weights opt-in: attr per group, cleared with None."""
        self.reg.add("a")
        self.reg.add("b")
        self.reg.set_attr("a", "emissiveGroup_a")
        groups = {g["name"]: g for g in self.reg.groups()}
        self.assertEqual(groups["a"]["attr"], "emissiveGroup_a")
        self.assertNotIn("attr", groups["b"])
        manifest = json.loads(self.reg.manifest().to_json())
        by_name = {g["name"]: g for g in manifest["groups"]}
        self.assertEqual(by_name["a"]["attr"], "emissiveGroup_a")
        self.assertNotIn("attr", by_name["b"])
        self.reg.set_attr("a", None)
        self.assertNotIn("attr", {g["name"]: g for g in self.reg.groups()}["a"])
        with self.assertRaises(ValueError):
            self.reg.set_attr("nope", "x")


class RegionMaskPackerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_slots_fill_lowest_unused_and_stay_stable(self):
        p = RegionMaskPacker(resolution=64)
        a = p.add_group("a", LEFT_HALF)
        b = p.add_group("b", RIGHT_HALF, slot=3)
        c = p.add_group("c", CENTER_STRIP)
        self.assertEqual((a.slot, b.slot, c.slot), (0, 3, 1))

    def test_slot_clash_raises(self):
        p = RegionMaskPacker(resolution=64)
        p.add_group("a", LEFT_HALF, slot=2)
        with self.assertRaises(ValueError):
            p.add_group("b", RIGHT_HALF, slot=2)

    def test_rasterize_fills_assigned_channels(self):
        p = RegionMaskPacker(resolution=64, padding_px=0, supersample=1)
        p.add_group("left", LEFT_HALF, slot=0)
        p.add_group("right", RIGHT_HALF, slot=1)
        arr = p.rasterize()
        self.assertEqual(arr.shape, (64, 64, 4))
        self.assertEqual(arr[32, 8, 0], 255)  # left region in R
        self.assertEqual(arr[32, 8, 1], 0)
        self.assertEqual(arr[32, 56, 1], 255)  # right region in G
        self.assertEqual(arr[32, 56, 0], 0)
        self.assertTrue((arr[..., 2] == 0).all())  # unused slots stay empty
        self.assertTrue((arr[..., 3] == 0).all())

    def test_padding_extends_coverage(self):
        quad = [
            [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75)],
            [(0.25, 0.25), (0.75, 0.75), (0.25, 0.75)],
        ]
        bare = RegionMaskPacker(resolution=64, padding_px=0, supersample=1)
        bare.add_group("a", quad)
        padded = RegionMaskPacker(resolution=64, padding_px=3, supersample=1)
        padded.add_group("a", quad)
        self.assertGreater(
            (padded.rasterize()[..., 0] > 0).sum(), (bare.rasterize()[..., 0] > 0).sum()
        )

    def test_validate_warns_on_overlap_and_oob(self):
        p = RegionMaskPacker(resolution=64)
        p.add_group("left", LEFT_HALF)
        p.add_group("strip", CENTER_STRIP)
        p.add_group("oob", [[(0.0, 0.0), (1.5, 0.0), (0.0, 1.0)]])
        warnings = p.validate()
        self.assertTrue(any("more than one group" in w for w in warnings))
        self.assertTrue(any("outside 0-1" in w for w in warnings))

    def test_validate_clean(self):
        p = RegionMaskPacker(resolution=64)
        p.add_group("left", LEFT_HALF)
        p.add_group("right", RIGHT_HALF)
        self.assertEqual(p.validate(), [])

    def test_write_outputs_mask_and_manifest(self):
        p = RegionMaskPacker(resolution=64)
        p.add_group("a", LEFT_HALF)
        mask_path = os.path.join(self.tmp, "prop_EMask.png")
        manifest = p.write(mask_path)
        self.assertTrue(os.path.isfile(mask_path))
        manifest_path = os.path.join(self.tmp, "prop_EMask.json")
        self.assertTrue(os.path.isfile(manifest_path))
        with Image.open(mask_path) as im:
            self.assertEqual(im.mode, "RGBA")
            self.assertEqual(im.size, (64, 64))
        loaded = RegionMaskManifest.load(manifest_path)
        self.assertEqual(loaded.encoding, "channels")
        self.assertEqual(loaded.mask, "prop_EMask.png")
        self.assertEqual(loaded.groups[0].slot, manifest.groups[0].slot)

    def test_preview_gates_emissive(self):
        p = RegionMaskPacker(resolution=64, padding_px=0, supersample=1)
        p.add_group("left", LEFT_HALF)
        p.add_group("right", RIGHT_HALF)
        emissive = Image.new("RGB", (64, 64), (200, 100, 50))
        on_off = np.asarray(p.preview(emissive, {"left": 1.0, "right": 0.0}))
        self.assertEqual(tuple(on_off[32, 8]), (200, 100, 50))  # left glows
        self.assertEqual(tuple(on_off[32, 56]), (0, 0, 0))  # right dark
        dim = np.asarray(p.preview(emissive, {"left": 0.5, "right": 0.0}))
        self.assertEqual(dim[32, 8, 0], 100)  # half-weight halves the output

    def test_preview_keeps_ungrouped_texels_lit(self):
        """A texel in NO group must glow as baked, whatever the weights are.

        Otherwise authoring one group silently blacks out every other
        emissive region on the mesh. Mirrors EmissiveGroups.hlsl's
        (1 - membership) term.
        """
        p = RegionMaskPacker(resolution=64, padding_px=0, supersample=1)
        p.add_group("left", LEFT_HALF)  # right half belongs to no group
        emissive = Image.new("RGB", (64, 64), (200, 100, 50))
        for weight in (1.0, 0.0):
            out = np.asarray(p.preview(emissive, {"left": weight}))
            self.assertEqual(
                tuple(out[32, 56]), (200, 100, 50), f"ungrouped @ w={weight}"
            )
        off = np.asarray(p.preview(emissive, {"left": 0.0}))
        self.assertEqual(tuple(off[32, 8]), (0, 0, 0))  # the grouped half obeys

    def test_empty_group_rejected(self):
        p = RegionMaskPacker(resolution=64)
        with self.assertRaises(ValueError):
            p.add_group("a", [])


class OptionalImagingDepsTest(unittest.TestCase):
    """The manifest model must work without numpy/Pillow (vertex-color
    encoding on a vanilla DCC Python); only the packer's imaging paths may
    require them — with a clear error, not an ImportError at module load."""

    def test_manifest_and_intake_work_without_imaging_deps(self):
        from pythontk.core_utils.engines.textures import region_masks as rm

        saved = (rm.np, rm.Image)
        rm.np, rm.Image = None, None
        try:
            m = RegionMaskManifest.vertex_color([{"name": "a", "slot": 0}])
            self.assertEqual(
                RegionMaskManifest.from_json(m.to_json()).groups[0].name, "a"
            )
            p = RegionMaskPacker(resolution=32)
            with self.assertRaises(RuntimeError):
                p.rasterize()
        finally:
            rm.np, rm.Image = saved


class MapRegistryEmissiveMaskTest(unittest.TestCase):
    """The `_EMask` suffix family resolves to the registered map type."""

    def test_suffix_resolution(self):
        reg = ptk.MapRegistry()
        for name in (
            "prop_EMask.png",
            "prop_EmissiveMask.png",
            "prop_EmissiveGroups.png",
        ):
            resolved = reg.resolve_type_from_path(name)
            self.assertEqual(resolved, "Emissive_Mask", name)

    def test_plain_emissive_still_resolves(self):
        reg = ptk.MapRegistry()
        self.assertEqual(reg.resolve_type_from_path("prop_Emissive.png"), "Emissive")


if __name__ == "__main__":
    unittest.main(verbosity=2)
