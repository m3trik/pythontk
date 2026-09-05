# !/usr/bin/python
# coding=utf-8
"""The shadow rigs' web route: ``MeshConvert.apply_glb_shadows`` and the
packaged ``shadow_rig`` viewer script, against the contract in
``mayatk/docs/shadow_rig_morphing.md`` (*Contracts*).

Two halves, deliberately in one module because they share the fixtures:

* Pure Python -- the pass over hand-authored GLBs (a plane wearing a base
  colour texture, a source, a contact, the ``data_export`` carrier), the
  optimiser's keep-as-found rule and the server's auto-activation.
* The page -- headless Edge through Playwright, the real ``viewer.html`` and
  three.js, the real ``PreviewServer``; the shim's placement is asserted
  against ``ShadowProjection.model`` itself, and the horizon shader against an
  analytic pole map rendered back through an offscreen target. Skipped, never
  failed, without the runtime (``pip install playwright`` drives the installed
  Edge; nothing is downloaded).

Fixtures are built here rather than imported from the live viewer module: a
test module is not a library, and these need geometry that one does not.
"""

import base64
import io
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pythontk as ptk  # noqa: E402
from pythontk import ImgUtils, MeshConvert, PreviewServer  # noqa: E402

try:
    import numpy as np
    from PIL import Image
except ImportError:  # pragma: no cover - the imaging stack is a test-only need here
    np = None
    Image = None

ptk.TestSandbox.activate()

MESH_CONVERT_LOGGER = "pythontk.file_utils.mesh_convert._mesh_convert"
TWO_PI = 2.0 * math.pi


# ============================================================================
# Fixtures
# ============================================================================


def _png_bytes(rgba) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgba, dtype=np.uint8), "RGBA").save(buf, "PNG")
    return buf.getvalue()


def _silhouette_png(size: int = 8) -> bytes:
    """A silhouette the DCC would write: black RGB, alpha = the shadow."""
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., 3] = 255
    return _png_bytes(rgba)


def _data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _record(name: str, **overrides):
    """One v2 ``shadow_metadata`` plane record, the contract's shape."""
    record = {
        "name": name,
        "type": "projected",
        "texture": f"{name}.png",
        "intensity": 1.0,
        "source": "shadow_source",
        "source_type": "point",
        "source_size": 0.0,
        "source_angle": 0.0,
        "follow_source": True,
        "contact": "Box_contact_loc",
        "ground": 0.0,
        "radius": 0.6,
        "height": 1.2,
        "max_stretch": 6.0,
        "canvas": [-1.0, 1.0, -0.5, 0.5],
    }
    record.update(overrides)
    return record


def _payload(planes, unit_scale: float = 1.0, version: int = 2) -> dict:
    payload = {"version": version, "planes": planes}
    if version >= 2:
        payload["unit_scale"] = unit_scale
    return payload


def _data_export_node(payload: dict) -> dict:
    """The carrier exactly as FBX2glTF transcribes a Maya ``data_export``."""
    return {
        "name": "data_export",
        "extras": {
            "fromFBX": {
                "userProperties": {
                    MeshConvert.SHADOW_METADATA_KEY: {
                        "type": "eFbxString",
                        "value": json.dumps(payload),
                    }
                }
            }
        },
    }


def _pointer_animation(material: int) -> dict:
    """An authored fade as ``apply_glb_fades`` writes it: alpha 1 -> 0 over 2 s."""
    return {
        "name": "FADE",
        "samplers": [{"input": 3, "output": 4, "interpolation": "LINEAR"}],
        "channels": [
            {
                "sampler": 0,
                "target": {
                    "path": "pointer",
                    "extensions": {
                        "KHR_animation_pointer": {
                            "pointer": (
                                f"/materials/{material}/pbrMetallicRoughness/"
                                "baseColorFactor"
                            )
                        }
                    },
                },
            }
        ],
    }


def _write_glb(path, *, nodes, materials, images, textures, animations=None):
    """A minimal but VALID binary glTF around one shared unit quad.

    The quad is the DCC's plane as the FBX hop delivers it: 1 x 1 in local X/Z
    at y = 0, centred on its pivot, UVs in glTF's top-left convention with
    v = 0 on the -Z (near) edge -- the PNG's top row is the light-side edge.
    Every mesh node references mesh index = its own material index, one mesh
    per material, so a per-material fade has one representative mesh.
    """
    positions = struct.pack(
        "<12f", -0.5, 0, 0.5, 0.5, 0, 0.5, -0.5, 0, -0.5, 0.5, 0, -0.5
    )
    uvs = struct.pack("<8f", 0, 1, 1, 1, 0, 0, 1, 0)
    indices = struct.pack("<6H", 0, 1, 2, 1, 3, 2)
    times = struct.pack("<3f", 0.0, 1.0, 2.0)
    alphas = struct.pack("<12f", 1, 1, 1, 1, 1, 1, 1, 0.5, 1, 1, 1, 0)
    chunks = [positions, uvs, indices, times, alphas]
    views, blob, offset = [], b"", 0
    for raw in chunks:
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw)})
        pad = (4 - len(raw) % 4) % 4
        blob += raw + b"\0" * pad
        offset += len(raw) + pad
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                        "indices": 2,
                        "material": index,
                    }
                ]
            }
            for index in range(len(materials))
        ],
        "materials": materials,
        "images": images,
        "textures": textures,
        "samplers": [{"wrapS": 10497, "wrapT": 10497}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
                "min": [-0.5, 0, -0.5],
                "max": [0.5, 0, 0.5],
            },
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2"},
            {"bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR"},
            {
                "bufferView": 3,
                "componentType": 5126,
                "count": 3,
                "type": "SCALAR",
                "min": [0.0],
                "max": [2.0],
            },
            {"bufferView": 4, "componentType": 5126, "count": 3, "type": "VEC4"},
        ],
    }
    if animations:
        gltf["animations"] = animations
        gltf["extensionsUsed"] = ["KHR_animation_pointer"]
    json_bytes = json.dumps(gltf).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(blob)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total))
        fh.write(struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes)
        fh.write(struct.pack("<I4s", len(blob), b"BIN\0") + blob)
    return path


def _bare_material(name: str) -> dict:
    """A plane material as a REAL Maya export delivers it: no baseColorTexture.

    Measured on the production export (`shadow_e2e_proj.glb`): a
    standardSurface's file texture does not survive the FBX hop, so every
    shadow plane's material arrives with a bare ``baseColorFactor``.
    """
    return {
        "name": name,
        "alphaMode": "OPAQUE",
        "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0]},
    }


def _plane_material(name: str, texture: int = 0, alpha: float = 1.0) -> dict:
    return {
        "name": name,
        "alphaMode": "BLEND",
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, alpha],
            "baseColorTexture": {"index": texture},
        },
    }


def _pole_horizon_png(
    bins, tile, r_min, r_max, pole_xz, pole_radius, pole_height, max_stretch
) -> bytes:
    """An analytic horizon map of one grounded pole, in the contract's encoding.

    For every texel (theta, r) of the grounded tile of bin k -- the ground
    point ``P = r (cos theta, sin theta)`` in the contact frame -- the pole's
    azimuth interval from P is written as the 16-sub-bin occupancy mask of the
    bins it crosses (B = sub-bins 0..7, A = 8..15), and the top of the pole as
    R = cot(elevation) / max_stretch -- the pole is the grounded layer's first
    (and only) run, so G, the later runs' channel, stays 0. The floating tiles
    stay empty.
    """
    width, height = tile
    tiles = 2 * bins
    cols = math.ceil(math.sqrt(tiles))
    rows = math.ceil(tiles / cols)
    img = np.zeros((rows * height, cols * width, 4), np.uint8)
    theta = (np.arange(width) + 0.5) / width * TWO_PI
    v = (np.arange(height) + 0.5) / height
    radius = r_min * (r_max / r_min) ** v
    rr, tt = np.meshgrid(radius, theta, indexing="ij")
    px, pz = rr * np.cos(tt), rr * np.sin(tt)
    dx, dz = pole_xz[0] - px, pole_xz[1] - pz
    dist = np.maximum(np.hypot(dx, dz), 1e-6)
    phi = np.mod(np.arctan2(dz, dx), TWO_PI)
    half = np.arcsin(np.minimum(1.0, pole_radius / dist))
    cot_hi = np.clip(dist / pole_height / max_stretch, 0.0, 1.0)
    step = TWO_PI / bins
    sub = step / 16.0
    lo_abs, hi_abs = phi - half, phi + half
    for k in range(bins):
        mask = np.zeros((height, width), np.int32)
        for j in range(16):
            s0 = k * step + j * sub
            s1 = s0 + sub
            hit = np.zeros((height, width), bool)
            for shift in (-TWO_PI, 0.0, TWO_PI):
                hit |= (lo_abs + shift < s1) & (hi_abs + shift > s0)
            mask |= hit.astype(np.int32) << j
        col, row = k % cols, k // cols
        view = img[row * height : (row + 1) * height, col * width : (col + 1) * width]
        occupied = mask > 0
        view[..., 0] = np.where(occupied, np.round(cot_hi * 255.0), 0).astype(np.uint8)
        view[..., 2] = (mask & 0xFF).astype(np.uint8)
        view[..., 3] = ((mask >> 8) & 0xFF).astype(np.uint8)
    return _png_bytes(img)


def _pole_geometry(point_xz, source, pole_xz, pole_radius, pole_height):
    """The source and the pole as seen from a ground point: the source's
    cotangent of elevation and azimuth, the pole's azimuth, half-width and
    cot(top)."""
    px, pz = point_xz
    lx, ly, lz = source[0] - px, source[1], source[2] - pz
    horizontal = math.hypot(lx, lz)
    cot_source = horizontal / ly if ly > 0 else math.inf
    phi = math.atan2(lz, lx) % TWO_PI
    dx, dz = pole_xz[0] - px, pole_xz[1] - pz
    dist = math.hypot(dx, dz)
    return {
        "cot_source": cot_source,
        "phi": phi,
        "delta": abs((phi - math.atan2(dz, dx) + math.pi) % TWO_PI - math.pi),
        "half": math.asin(min(1.0, pole_radius / dist)),
        "cot_top": dist / pole_height,
    }


def _pole_alpha(
    point_xz, source, pole_xz, pole_radius, pole_height, r_max, max_stretch
):
    """The analytic answer at a ground point: is the source, seen from there,
    behind the pole -- within its azimuth extent, under its top and above the
    reach cap (the contract tests elevation in cotangent space:
    ``cot(top) <= cot(source) <= max_stretch``)?"""
    if math.hypot(*point_xz) > r_max:
        return 0.0
    g = _pole_geometry(point_xz, source, pole_xz, pole_radius, pole_height)
    blocked = g["delta"] <= g["half"] and g["cot_top"] <= g["cot_source"] <= max_stretch
    return 1.0 if blocked else 0.0


def _pole_margins(
    point_xz, source, pole_xz, pole_radius, pole_height, bins, max_stretch
):
    """How far the point sits from every decision boundary -- azimuth in
    radians, elevation and the cap in cotangent units (the space the map is
    quantised in, max_stretch / 255 per level), the bearing in bins -- so a
    fixture point cannot pass by sitting on an edge the bilinear blur would
    smear either way."""
    g = _pole_geometry(point_xz, source, pole_xz, pole_radius, pole_height)
    fk = g["phi"] / (TWO_PI / bins)
    return {
        "azimuth": abs(g["delta"] - g["half"]),
        "elevation": abs(g["cot_source"] - g["cot_top"]),
        "cap": abs(max_stretch - g["cot_source"]),
        "bin": abs(fk - round(fk)),
    }


#: The pole fixture's geometry, shared by every horizon test below: the DCC's
#: defaults, and a pole whose azimuth extent from the shadowed point spans
#: three bins. The source's bearing is the middle of bin 18, so no sample sits
#: on a bin edge the bilinear blur could smear either way.
POLE_BINS, POLE_TILE = 32, (128, 64)
POLE_RMIN, POLE_RMAX = 0.05, 4.0
POLE_RADIUS, POLE_HEIGHT = 0.15, 1.0
POLE_MAX_STRETCH = 6.0


def _pole_geometry_frame():
    """``(azimuth, away, perp, pole)`` -- the bearing the light sits on, the
    unit vectors along and across the shadow, and the pole's base."""
    step = TWO_PI / POLE_BINS
    azimuth = (POLE_BINS // 2 + 2 + 0.5) * step
    away = (math.cos(azimuth - math.pi), math.sin(azimuth - math.pi))
    perp = (-away[1], away[0])
    return azimuth, away, perp, (0.5 * away[0], 0.5 * away[1])


def _quat_from_minus_y(direction):
    """A node rotation whose local -Y points along *direction* -- the FBX light
    convention the shim reads a directional source's shine direction from."""
    d = [float(v) for v in direction]
    n = math.sqrt(sum(v * v for v in d))
    d = [v / n for v in d]
    a = [0.0, -1.0, 0.0]
    dot = sum(x * y for x, y in zip(a, d))
    if dot < -0.999999:  # antiparallel: any perpendicular axis will do
        return [1.0, 0.0, 0.0, 0.0]
    axis = [
        a[1] * d[2] - a[2] * d[1],
        a[2] * d[0] - a[0] * d[2],
        a[0] * d[1] - a[1] * d[0],
    ]
    q = axis + [1.0 + dot]
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q]


def _quat(axis, degrees):
    x, y, z = axis
    half = math.radians(degrees) / 2.0
    s = math.sin(half)
    return [x * s, y * s, z * s, math.cos(half)]


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _quat_rotate(q, v):
    qv = [v[0], v[1], v[2], 0.0]
    x, y, z, w = q
    conj = [-x, -y, -z, w]
    out = _quat_mul(_quat_mul(q, qv), conj)
    return out[:3]


# ============================================================================
# The pass, pure Python
# ============================================================================


@unittest.skipUnless(np is not None and Image is not None, "needs numpy + Pillow")
class TestApplyGlbShadows(unittest.TestCase):
    """``apply_glb_shadows`` over hand-built GLBs."""

    #: The same map SHAPE the live tests use -- the block has to describe the
    #: PNG beside it, so both come from one set of constants; only the pole's
    #: position differs, and this suite never samples the map anyway (the pass
    #: treats it as opaque bytes).
    HORIZON = {
        "texture": "Box_horizon.png",
        "bins": POLE_BINS,
        "tile": list(POLE_TILE),
        "layout": [8, 8],
        "layers": 2,
        "mapping": "logpolar",
        "r_min": POLE_RMIN,
        "r_max": POLE_RMAX,
        "frame_a": [1, 0, 0],
        "frame_b": [0, 0, 1],
        "encoding": 1,
        "rect": [0.5, 0.5, 0.25, 0.25],
    }
    ATLAS_RECT = [0.5, 0.5, 0.0, 0.5]

    @classmethod
    def setUpClass(cls):
        cls.temp = ptk.TempArtifacts("shadow_web_pass", policy="scoped")
        cls.maps = cls.temp.dir_path()
        cls.horizon_png = _pole_horizon_png(
            POLE_BINS,
            POLE_TILE,
            POLE_RMIN,
            POLE_RMAX,
            (0.46, 0.19),
            POLE_RADIUS,
            POLE_HEIGHT,
            POLE_MAX_STRETCH,
        )
        with open(os.path.join(cls.maps, "Box_horizon.png"), "wb") as fh:
            fh.write(cls.horizon_png)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _glb(self, payload, *, nodes=None, materials=None, name="scene"):
        nodes = nodes or [
            {"name": "Box_shadow", "mesh": 0},
            {"name": "shadow_source", "translation": [1.5, 2.0, -0.5]},
            {"name": "Box_contact_loc", "translation": [0.2, 0.0, 0.3]},
            {"name": "Box_horizon_plane", "mesh": 1},
            _data_export_node(payload),
        ]
        materials = materials or [
            _plane_material("Box_shadow_MAT"),
            _plane_material("Box_horizon_MAT", alpha=0.7),
        ]
        path = self.temp.path(extension=".glb")
        return _write_glb(
            path,
            nodes=nodes,
            materials=materials,
            images=[{"name": "shadow_atlas", "uri": _data_uri(_silhouette_png())}],
            textures=[{"source": 0, "sampler": 0}],
        )

    def _v2(self):
        return _payload(
            [
                _record(
                    "Box_shadow",
                    atlas={
                        "texture": "shadow_atlas_projected.png",
                        "rect": self.ATLAS_RECT,
                    },
                ),
                _record(
                    "Box_horizon_plane", type="horizon", horizon=dict(self.HORIZON)
                ),
            ],
            unit_scale=0.01,
        )

    def _read(self, path):
        with MeshConvert.open_glb(path) as edit:
            return json.loads(json.dumps(edit.gltf))

    def test_the_manifest_carries_indices_top_left_rects_and_the_data_sampler(self):
        path = self._glb(self._v2())
        manifest = MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])

        self.assertIsNotNone(manifest)
        self.assertEqual(manifest["version"], 2)
        self.assertEqual(manifest["metadata_version"], 2)
        self.assertAlmostEqual(manifest["unit_scale"], 0.01)
        gltf = self._read(path)
        self.assertEqual(gltf["extras"][MeshConvert.SHADOW_WEB_KEY], manifest)
        by_name = {p["name"]: p for p in manifest["planes"]}
        self.assertEqual(set(by_name), {"Box_shadow", "Box_horizon_plane"})

        projected = by_name["Box_shadow"]
        self.assertEqual(projected["node"], 0)
        self.assertEqual(projected["source_node"], 1)
        self.assertEqual(projected["contact_node"], 2)
        self.assertEqual(projected["material"], 0)
        self.assertEqual(projected["texture_index"], 0, "the material's base colour")
        self.assertEqual(projected["atlas"]["texture_index"], 0)
        self.assertEqual(
            projected["atlas"]["rect"], ImgUtils.flip_rect_v(self.ATLAS_RECT)
        )
        self.assertEqual(projected["atlas"]["rect"], [0.5, 0.5, 0.0, 0.0])
        self.assertTrue(projected["follow_source"])
        # The record's own fields travel untouched, DCC units and all.
        self.assertEqual(projected["radius"], 0.6)
        self.assertEqual(projected["canvas"], [-1.0, 1.0, -0.5, 0.5])

        horizon = by_name["Box_horizon_plane"]
        self.assertEqual(horizon["node"], 3)
        self.assertEqual(horizon["material"], 1)
        self.assertEqual(
            horizon["horizon"]["rect"], ImgUtils.flip_rect_v(self.HORIZON["rect"])
        )
        index = horizon["horizon"]["texture_index"]
        self.assertIsInstance(index, int)
        texture = gltf["textures"][index]
        self.assertEqual(texture["name"], "Box_horizon")
        self.assertEqual(
            gltf["samplers"][texture["sampler"]], MeshConvert.SHADOW_DATA_SAMPLER
        )
        # The atlas texture the material samples keeps ITS sampler.
        self.assertEqual(gltf["textures"][0]["sampler"], 0)
        self.assertEqual(horizon["horizon"]["layers"], 2)

    def test_the_horizon_png_is_embedded_byte_for_byte(self):
        """Never decoded: the map is data, and the session relocates it into
        the BIN on close exactly as it was read."""
        path = self._glb(self._v2())
        manifest = MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])
        index = manifest["planes"][1]["horizon"]["texture_index"]
        with MeshConvert.open_glb(path) as edit:
            image = edit.images[edit.textures[index]["source"]]
            self.assertIn("bufferView", image, "relocated out of the JSON chunk")
            self.assertEqual(image.get("mimeType"), "image/png")
            self.assertEqual(edit._image_payload(image), self.horizon_png)

    def test_rerunning_replaces_the_manifest_and_adds_nothing(self):
        path = self._glb(self._v2())
        first = MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])
        before = self._read(path)
        second = MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])
        after = self._read(path)

        self.assertEqual(first, second)
        self.assertEqual(len(after["images"]), len(before["images"]))
        self.assertEqual(len(after["textures"]), len(before["textures"]))
        self.assertEqual(len(after["samplers"]), len(before["samplers"]))
        self.assertEqual(after["extras"], before["extras"])

    def test_a_v1_channel_reads_as_a_still_projected_plane(self):
        payload = _payload(
            [{"name": "Box_shadow", "texture": "Box_shadow.png", "intensity": 0.8}],
            version=1,
        )
        path = self._glb(payload)
        manifest = MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])

        self.assertEqual(manifest["metadata_version"], 1)
        self.assertEqual(manifest["unit_scale"], 1.0)
        (plane,) = manifest["planes"]
        self.assertEqual(plane["type"], "projected")
        self.assertEqual(plane["intensity"], 0.8)
        self.assertFalse(plane["follow_source"])
        self.assertIsNone(plane["source_node"])
        self.assertIsNone(plane["contact_node"])
        self.assertEqual(plane["texture_index"], 0)
        self.assertNotIn("atlas", plane)
        self.assertNotIn("horizon", plane)
        self.assertEqual(plane["max_stretch"], ptk.ShadowProjection.DEFAULT_MAX_STRETCH)

    def test_a_newer_schema_is_refused_not_guessed(self):
        path = self._glb(_payload([_record("Box_shadow")], version=3))
        with self.assertLogs(MESH_CONVERT_LOGGER, level="WARNING") as log:
            self.assertIsNone(
                MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])
            )
        self.assertTrue(any("newer" in line for line in log.output), log.output)
        self.assertNotIn(MeshConvert.SHADOW_WEB_KEY, self._read(path).get("extras", {}))

    def test_a_missing_horizon_map_skips_that_plane_with_a_warning(self):
        path = self._glb(self._v2())
        with self.assertLogs(MESH_CONVERT_LOGGER, level="WARNING") as log:
            manifest = MeshConvert.apply_glb_shadows(path, search_dirs=[])
        self.assertTrue(
            any("Box_horizon.png" in line for line in log.output), log.output
        )
        self.assertEqual([p["name"] for p in manifest["planes"]], ["Box_shadow"])

    def test_no_channel_is_a_clean_no_op(self):
        path = self._glb(None, nodes=[{"name": "Box_shadow", "mesh": 0}])
        self.assertIsNone(MeshConvert.apply_glb_shadows(path))
        self.assertNotIn(MeshConvert.SHADOW_WEB_KEY, self._read(path).get("extras", {}))

    def test_a_run_that_binds_nothing_removes_a_stale_manifest(self):
        path = self._glb(self._v2())
        MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])
        with MeshConvert.open_glb(path) as edit:
            # Rename the planes away: the channel still names them, the file no
            # longer carries them.
            for node in edit.gltf["nodes"]:
                if "mesh" in node:
                    node["name"] = "gone_" + node["name"]
            edit.dirty = True
        self.assertIsNone(MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps]))
        self.assertNotIn(MeshConvert.SHADOW_WEB_KEY, self._read(path)["extras"])

    def test_a_leaf_name_binds_every_namespaced_plane_but_never_an_ambiguous_source(
        self,
    ):
        nodes = [
            {"name": "A:Box_shadow", "mesh": 0},
            {"name": "B:Box_shadow", "mesh": 0},
            {"name": "A:shadow_source", "translation": [1, 2, 0]},
            {"name": "B:shadow_source", "translation": [1, 2, 0]},
            {"name": "Box_contact_loc"},
            _data_export_node(_payload([_record("Box_shadow")])),
        ]
        path = self._glb(None, nodes=nodes, materials=[_plane_material("m")])
        with self.assertLogs(MESH_CONVERT_LOGGER, level="WARNING") as log:
            manifest = MeshConvert.apply_glb_shadows(path)
        self.assertEqual([p["node"] for p in manifest["planes"]], [0, 1])
        self.assertTrue(all(p["source_node"] is None for p in manifest["planes"]))
        self.assertTrue(all(p["contact_node"] == 4 for p in manifest["planes"]))
        self.assertTrue(any("ambiguous" in line for line in log.output), log.output)

    def test_a_projected_plane_without_any_colour_map_is_skipped(self):
        materials = [
            {"name": "bare", "pbrMetallicRoughness": {"baseColorFactor": [0, 0, 0, 1]}}
        ]
        path = self._glb(
            None,
            nodes=[
                {"name": "Box_shadow", "mesh": 0},
                _data_export_node(
                    _payload([_record("Box_shadow", texture="nowhere.png")])
                ),
            ],
            materials=materials,
        )
        with self.assertLogs(MESH_CONVERT_LOGGER, level="WARNING"):
            self.assertIsNone(MeshConvert.apply_glb_shadows(path))

    def test_a_loose_silhouette_is_bound_when_the_material_lost_it(self):
        with open(os.path.join(self.maps, "Box_shadow.png"), "wb") as fh:
            fh.write(_silhouette_png(4))
        materials = [
            {"name": "bare", "pbrMetallicRoughness": {"baseColorFactor": [0, 0, 0, 1]}}
        ]
        path = self._glb(
            None,
            nodes=[
                {"name": "Box_shadow", "mesh": 0},
                _data_export_node(_payload([_record("Box_shadow")])),
            ],
            materials=materials,
        )
        manifest = MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])
        (plane,) = manifest["planes"]
        gltf = self._read(path)
        self.assertEqual(gltf["textures"][plane["texture_index"]]["name"], "Box_shadow")
        self.assertEqual(
            gltf["samplers"][gltf["textures"][plane["texture_index"]]["sampler"]],
            {"wrapS": 33071, "wrapT": 33071},
            "a colour map takes the atlas precedent's clamp sampler, not the data one",
        )

    def test_an_atlas_in_the_file_becomes_the_planes_colour_map(self):
        """The real Maya shape: the materials lost their textures through the
        FBX, both the atlas and each plane's own PNG are embedded, and the
        record names an atlas. The plane must sample the ATLAS -- its rect
        names a tile of that image, so pointing the manifest at the plane's own
        full-frame PNG leaves the rect meaningless and the viewer unable to
        batch (measured on shadow_e2e_proj.glb: three planes, three different
        texture indices, no InstancedMesh)."""
        with open(os.path.join(self.maps, "Box_shadow.png"), "wb") as fh:
            fh.write(_silhouette_png(4))
        own, atlas_png = _silhouette_png(4), _silhouette_png(8)
        path = self.temp.path(extension=".glb")
        _write_glb(
            path,
            nodes=[
                {"name": "Box_shadow", "mesh": 0},
                _data_export_node(
                    _payload(
                        [
                            _record(
                                "Box_shadow",
                                atlas={
                                    "texture": "shadow_atlas_projected.png",
                                    "rect": self.ATLAS_RECT,
                                },
                            )
                        ]
                    )
                ),
            ],
            materials=[_bare_material("Box_shadow_mat")],
            images=[
                {"name": "Box_shadow.png", "uri": _data_uri(own)},
                {"name": "shadow_atlas_projected.png", "uri": _data_uri(atlas_png)},
            ],
            textures=[{"source": 0, "sampler": 0}, {"source": 1, "sampler": 0}],
        )
        manifest = MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])

        (plane,) = manifest["planes"]
        gltf = self._read(path)
        atlas_image = gltf["textures"][plane["texture_index"]]["source"]
        self.assertEqual(
            gltf["images"][atlas_image]["name"],
            "shadow_atlas_projected.png",
            "the plane must sample the atlas, not its own tile PNG",
        )
        self.assertEqual(plane["atlas"]["texture_index"], plane["texture_index"])
        self.assertEqual(plane["atlas"]["rect"], ImgUtils.flip_rect_v(self.ATLAS_RECT))
        # Resolved from the file: no second copy of the atlas was embedded.
        self.assertEqual(len(gltf["images"]), 2)

    def test_an_unresolvable_atlas_leaves_the_material_map_as_the_tile_source(self):
        """The other production shape: the DCC's material DOES point at the
        atlas (the base-colour sidecar repaired it), and no loose atlas file is
        around. The material's own map is then the atlas the rect describes."""
        path = self._glb(
            None,
            nodes=[
                {"name": "Box_shadow", "mesh": 0},
                _data_export_node(
                    _payload(
                        [
                            _record(
                                "Box_shadow",
                                atlas={
                                    "texture": "nowhere_atlas.png",
                                    "rect": self.ATLAS_RECT,
                                },
                            )
                        ]
                    )
                ),
            ],
            materials=[_plane_material("Box_shadow_MAT")],
        )
        with self.assertLogs(MESH_CONVERT_LOGGER, level="WARNING"):
            manifest = MeshConvert.apply_glb_shadows(path, search_dirs=[])
        (plane,) = manifest["planes"]
        self.assertEqual(plane["texture_index"], 0)
        self.assertEqual(plane["atlas"]["texture_index"], 0)

    def test_the_handoff_names_the_manifest(self):
        envelope = MeshConvert.build_scene_sidecar({}, source={"application": "maya"})
        self.assertIn(
            f"extras.{MeshConvert.SHADOW_WEB_KEY}", envelope["handoff"]["reads"]
        )

    def test_the_optimiser_keeps_every_bound_shadow_map_as_found(self):
        """Neither resized nor re-encoded, in a mode that resizes everything else."""
        big = _png_bytes(np.full((64, 64, 4), 200, np.uint8))
        with open(os.path.join(self.maps, "Box_horizon.png"), "wb") as fh:
            fh.write(self.horizon_png)
        path = self.temp.path(extension=".glb")
        _write_glb(
            path,
            nodes=[
                {"name": "Box_shadow", "mesh": 0},
                {"name": "Box_horizon_plane", "mesh": 1},
                {"name": "other", "mesh": 2},
                _data_export_node(self._v2()),
            ],
            materials=[
                _plane_material("shadow", texture=0),
                _plane_material("horizon"),
                _plane_material("other", texture=1),
            ],
            images=[
                {"name": "shadow_atlas", "uri": _data_uri(big)},
                {"name": "plain", "uri": _data_uri(big)},
            ],
            textures=[{"source": 0, "sampler": 0}, {"source": 1, "sampler": 0}],
        )
        MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])
        MeshConvert.optimize_glb_textures(
            path, max_size=16, image_format="PNG", workers=1
        )
        with MeshConvert.open_glb(path) as edit:
            payloads = {img["name"]: edit._image_payload(img) for img in edit.images}
        self.assertEqual(payloads["shadow_atlas"], big, "the atlas was touched")
        self.assertEqual(
            payloads["Box_horizon.png"], self.horizon_png, "the data map was touched"
        )
        self.assertNotEqual(
            payloads["plain"], big, "the control image should have been resized"
        )
        self.assertEqual(Image.open(io.BytesIO(payloads["plain"])).size, (16, 16))

    def test_fbx_to_glb_runs_the_pass_and_searches_sourceimages_beside_the_fbx(self):
        """The registration, end to end, with the converter replaced by a copy."""
        project = self.temp.dir_path()
        os.makedirs(os.path.join(project, "sourceimages"))
        shutil.copy(
            os.path.join(self.maps, "Box_horizon.png"),
            os.path.join(project, "sourceimages", "Box_horizon.png"),
        )
        fixture = self._glb(self._v2())
        src = os.path.join(project, "scene.fbx")
        with open(src, "wb") as fh:
            fh.write(b"Kaydara FBX Binary  \x00")

        def fake_run(cmd, **kwargs):
            base = cmd[cmd.index("-o") + 1]
            shutil.copy(fixture, base + ".glb")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with (
            patch.object(MeshConvert, "resolve_binary", return_value="FBX2glTF"),
            patch(f"{MESH_CONVERT_LOGGER}.subprocess.run", side_effect=fake_run),
        ):
            out = MeshConvert.fbx_to_glb(src, timeout=60, lightmaps=False)
        manifest = self._read(out)["extras"][MeshConvert.SHADOW_WEB_KEY]
        self.assertEqual(
            {p["name"] for p in manifest["planes"]}, {"Box_shadow", "Box_horizon_plane"}
        )


@unittest.skipUnless(np is not None and Image is not None, "needs numpy + Pillow")
class TestServerAutoActivation(unittest.TestCase):
    """``shadow_rig`` is on whenever the deliverable carries the manifest."""

    def setUp(self):
        self.temp = ptk.TempArtifacts("shadow_web_server", policy="scoped")
        self.server = PreviewServer(root=self.temp.dir_path(), port=0, viewer=False)

    def tearDown(self):
        self.server.stop()
        self.temp.cleanup()

    def _glb(self, with_manifest):
        path = self.temp.path(extension=".glb")
        nodes = [
            {"name": "Box_shadow", "mesh": 0},
            {"name": "shadow_source"},
            {"name": "Box_contact_loc"},
        ]
        if with_manifest:
            nodes.append(_data_export_node(_payload([_record("Box_shadow")])))
        _write_glb(
            path,
            nodes=nodes,
            materials=[_plane_material("m")],
            images=[{"name": "s", "uri": _data_uri(_silhouette_png())}],
            textures=[{"source": 0, "sampler": 0}],
        )
        if with_manifest:
            self.assertIsNotNone(MeshConvert.apply_glb_shadows(path))
        return path

    def test_a_shadow_deliverable_activates_the_script_after_the_named_set(self):
        self.server.set_scripts(["turntable"])
        self.server.publish(self._glb(True))
        self.assertEqual(self.server.scripts, ("turntable", "shadow_rig"))
        self.assertEqual(
            self.server.manifest()["scripts"],
            ["scripts/turntable.js", "scripts/shadow_rig.js"],
        )
        self.assertTrue(
            (Path(self.server.root) / "scripts" / "shadow_rig.js").is_file()
        )

    def test_a_plain_deliverable_activates_nothing(self):
        self.server.publish(self._glb(False))
        self.assertEqual(self.server.scripts, ())

    def test_a_file_that_is_not_a_glb_is_served_without_a_probe_failure(self):
        stub = self.temp.path(extension=".glb")
        with open(stub, "wb") as fh:
            fh.write(b"glTF-stub-0")
        self.server.publish(stub)
        self.assertEqual(self.server.scripts, ())
        self.assertEqual(self.server.version, 1)

    def test_an_already_named_script_is_not_repeated(self):
        self.server.set_scripts(["shadow_rig", "inspect"])
        self.server.publish(self._glb(True))
        self.assertEqual(self.server.scripts, ("shadow_rig", "inspect"))

    def test_an_emptied_registry_opts_out(self):
        with patch.dict(PreviewServer.AUTO_SCRIPTS, clear=True):
            self.server.publish(self._glb(True))
        self.assertEqual(self.server.scripts, ())


# ============================================================================
# The page
# ============================================================================


def _runtime_available():
    """Playwright installed AND an Edge/Chrome channel it can drive."""
    if np is None or Image is None:
        return False
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            browser.close()
        return True
    except Exception:  # noqa: BLE001 -- no browser is a skip, not a failure
        return False


#: The probe: waits for the shim's session (whose load hook runs after this
#: module's, in registration order), lets it settle for two frames, snapshots
#: every plane in MODEL space, runs the test's action, and snapshots again.
PROBE_JS = """
export default function probe(viewer) {
  const report = { ready: false, errors: [], loads: 0 };
  window.__probe = report;
  const THREE = viewer.THREE;
  const waiters = [];
  viewer.on('frame', () => { for (const fn of waiters.splice(0)) fn(); });
  const nextFrames = (n) => new Promise((resolve) => {
    let left = n;
    const tick = () => { left -= 1; if (left <= 0) resolve(); else waiters.push(tick); };
    waiters.push(tick);
  });

  function snapshot(session) {
    const inv = new THREE.Matrix4().copy(session.model.matrixWorld).invert();
    return session.planes.map((plane) => {
      const local = new THREE.Matrix4();
      if (plane.batch) plane.batch.mesh.getMatrixAt(plane.instance, local);
      else local.copy(inv).multiply(plane.node.matrixWorld);
      const p = new THREE.Vector3(); const q = new THREE.Quaternion(); const s = new THREE.Vector3();
      local.decompose(p, q, s);
      const material = plane.batch ? plane.batch.mesh.material : plane.mesh.material;
      const u = material.uniforms || {};
      const horizon = u.uHorizonMap ? u.uHorizonMap.value : null;
      return {
        name: plane.record.name,
        node: plane.record.node,
        follow: plane.follow,
        position: [p.x, p.y, p.z],
        yaw: new THREE.Euler().setFromQuaternion(q, 'YXZ').y,
        scale: [s.x, s.y, s.z],
        instanced: !!plane.batch,
        batchCount: plane.batch ? plane.batch.mesh.count : 0,
        hidden: !plane.mesh.visible,
        isShader: !!material.isShaderMaterial,
        uMode: u.uMode ? u.uMode.value : null,
        uMaxStretch: u.uMaxStretch ? u.uMaxStretch.value : null,
        uRect: plane.batch
          ? [0, 1, 2, 3].map((i) => plane.batch.iRect.array[plane.instance * 4 + i])
          : (u.uRect ? u.uRect.value.toArray() : null),
        uOpacity: plane.batch ? plane.batch.iParams.array[plane.instance * 2] : (u.uParams ? u.uParams.value.x : null),
        uIntensity: plane.batch ? plane.batch.iParams.array[plane.instance * 2 + 1] : (u.uParams ? u.uParams.value.y : null),
        originalOpacity: plane.original.opacity,
        depthWrite: material.depthWrite,
        transparent: material.transparent,
        mapBound: !!(u.uMap && u.uMap.value),
        mapImage: u.uMap && u.uMap.value && u.uMap.value.image
          ? [u.uMap.value.image.width, u.uMap.value.image.height] : null,
        horizonBound: !!horizon,
        horizonColorSpace: horizon ? horizon.colorSpace : null,
        horizonMipmaps: horizon ? horizon.generateMipmaps : null,
        horizonMinFilter: horizon ? horizon.minFilter === THREE.LinearFilter : null,
        horizonWrap: horizon ? horizon.wrapS === THREE.ClampToEdgeWrapping : null,
      };
    });
  }

  viewer.on('load', async (detail) => {
    report.loads += 1;
    try {
      await nextFrames(1);
      const model = detail.model;
      const session = model.userData.shadowRig;
      report.hasSession = !!session;
      if (!session) { report.ready = true; return; }
      await session.ready;
      await nextFrames(2);
      let instanced = 0;
      model.traverse((n) => { if (n.isInstancedMesh) instanced += 1; });
      report.instancedMeshes = instanced;
      report.before = snapshot(session);
      const ctx = { viewer, THREE, model, session, report, nextFrames };
      __ACTION__
      await nextFrames(2);
      report.after = snapshot(session);
      report.ready = true;
    } catch (error) {
      report.errors.push(String(error && error.stack || error));
      report.ready = true;
    }
  });
}
"""

#: Renders the horizon plane alone into a 4 x 4 target from straight above each
#: sample point (an orthographic camera 4 mm wide) and reads the centre texel's
#: alpha -- the shader's own answer, cleared to alpha 0 first, on a private
#: layer so the grid and the rest of the scene do not blend in.
HORIZON_SAMPLES_JS = """
      {
        // Real scale first: the page's fit mode scales the model to 1.5 m of
        // HEIGHT, and this fixture is a flat quad (height 0), so fitted it is a
        // million times too large and the sampling camera sits past its far
        // clip. The page's own 'r' shortcut is the documented switch.
        window.dispatchEvent(new KeyboardEvent('keydown', { key: 'r' }));
        await nextFrames(1);
        const renderer = viewer.renderer;
        const scene = viewer.scene;
        const plane = session.planes.find((p) => p.record.type === 'horizon');
        const target = new THREE.WebGLRenderTarget(4, 4);
        const camera = new THREE.OrthographicCamera(-0.002, 0.002, 0.002, -0.002, 0.001, 50);
        camera.layers.set(7);
        (plane.batch ? plane.batch.mesh : plane.mesh).layers.enable(7);
        const background = scene.background;
        const clear = new THREE.Color();
        renderer.getClearColor(clear);
        const clearAlpha = renderer.getClearAlpha();
        report.samples = [];
        for (const [x, z] of __POINTS__) {
          const world = new THREE.Vector3(x, 0.2, z).applyMatrix4(model.matrixWorld);
          camera.position.copy(world);
          camera.up.set(0, 0, -1);
          camera.lookAt(world.x, world.y - 1, world.z);
          camera.updateMatrixWorld();
          scene.background = null;
          renderer.setRenderTarget(target);
          renderer.setClearColor(0x000000, 0);
          renderer.clear();
          renderer.render(scene, camera);
          const pixel = new Uint8Array(4);
          renderer.readRenderTargetPixels(target, 2, 2, 1, 1, pixel);
          report.samples.push(pixel[3] / 255);
        }
        renderer.setRenderTarget(null);
        renderer.setClearColor(clear, clearAlpha);
        scene.background = background;
        target.dispose();
      }
"""


@unittest.skipUnless(
    _runtime_available(), "needs playwright + an installed Edge/Chrome channel"
)
class TestShadowRigLive(unittest.TestCase):
    """The shim, in the real page, against the Python model."""

    SOURCE = [1.5, 2.0, -0.5]
    CONTACT = [0.2, 0.0, 0.3]

    @classmethod
    def setUpClass(cls):
        cls.temp = ptk.TempArtifacts("shadow_web_live", policy="scoped")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    # ------------------------------------------------------------ fixtures
    def _glb(
        self,
        payload,
        *,
        nodes,
        materials,
        images=None,
        textures=None,
        animations=None,
        search_dirs=(),
    ):
        path = self.temp.path(extension=".glb")
        _write_glb(
            path,
            nodes=nodes,
            materials=materials,
            images=images
            or [{"name": "shadow_atlas", "uri": _data_uri(_silhouette_png())}],
            textures=textures or [{"source": 0, "sampler": 0}],
            animations=animations,
        )
        manifest = MeshConvert.apply_glb_shadows(path, search_dirs=list(search_dirs))
        self.assertIsNotNone(manifest, "the fixture itself must bind")
        return path

    def _following_glb(self, unit_scale=1.0, source_node=None):
        scale = 1.0 / unit_scale
        record = _record(
            "Box_shadow",
            ground=0.0,
            radius=0.6 * scale,
            height=1.2 * scale,
            source_size=0.0,
        )
        if source_node and "rotation" in source_node:
            record["source_type"] = "directional"
        return self._glb(
            None,
            nodes=[
                {"name": "Box_shadow", "mesh": 0, "translation": [3.0, 0.0, 3.0]},
                source_node or {"name": "shadow_source", "translation": self.SOURCE},
                {"name": "Box_contact_loc", "translation": self.CONTACT},
                _data_export_node(_payload([record], unit_scale=unit_scale)),
            ],
            materials=[_plane_material("Box_shadow_MAT")],
        )

    # -------------------------------------------------------------- driver
    def _load(self, glb, action="", points=None):
        from playwright.sync_api import sync_playwright

        probe = self.temp.path(extension=".js")
        source = PROBE_JS.replace("__ACTION__", action).replace(
            "__POINTS__", json.dumps(points or [])
        )
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write(source)

        server = ptk.PreviewServer(viewer=True, title="shadow-test")
        server.start()
        # The shim BEFORE the probe, and the page up before the publish. The
        # page imports the manifest's scripts in order and does not await them
        # before loading the asset, so a 5 KB fixture can be on screen before a
        # 34 KB module has arrived (measured: every run in this process). With
        # the probe last, its presence proves the shim is in, and a push then
        # lands on a page that has it -- the production order too: the server
        # outlives every push. Auto-activation is pinned by the server tests.
        server.add_script("shadow_rig")
        server.add_script("probe", probe)
        console = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    channel="msedge",
                    headless=True,
                    args=["--enable-unsafe-swiftshader"],
                )
                page = browser.new_page()
                page.on(
                    "console",
                    lambda m: (
                        console.append(f"[{m.type}] {m.text}")
                        if m.type in ("error", "warning")
                        else None
                    ),
                )
                page.on("pageerror", lambda e: console.append(f"[pageerror] {e}"))
                page.goto(server.url, wait_until="domcontentloaded", timeout=120_000)
                page.wait_for_function("() => !!window.__probe", timeout=120_000)
                server.publish(glb)
                scripts = server.scripts
                page.wait_for_function(
                    "() => window.__probe.ready === true", timeout=180_000
                )
                found = page.evaluate("() => window.__probe")
                browser.close()
        finally:
            server.stop()
        found["console"] = console
        found["scripts"] = scripts
        self.assertEqual(found["errors"], [], found)
        self.assertTrue(found.get("hasSession"), f"the shim built no session: {found}")
        return found

    # ----------------------------------------------------------- expected
    def _expected(
        self,
        source=None,
        contact=None,
        direction=None,
        radius=0.6,
        height=1.2,
        ground=0.0,
        canvas=(-1, 1, -0.5, 0.5),
        unit_scale=1.0,
    ):
        model = ptk.ShadowProjection.model(
            contact or self.CONTACT,
            light=None if direction is not None else (source or self.SOURCE),
            ground=ground,
            radius=radius,
            height=height,
            direction=direction,
            max_stretch=6.0,
        )
        centre, along, across = model.placement(canvas)
        # *ground* is metres here; the lift is 0.01 DCC units (GROUND_OFFSET).
        return {
            "position": [centre[0], ground + 0.01 * unit_scale, centre[1]],
            "yaw": math.atan2(model.bearing[0], model.bearing[1]),
            "scale": [across, 1.0, along],
        }

    def _assert_placed(self, plane, expected, places=5):
        for axis in range(3):
            self.assertAlmostEqual(
                plane["position"][axis],
                expected["position"][axis],
                places=places,
                msg=plane,
            )
            self.assertAlmostEqual(
                plane["scale"][axis], expected["scale"][axis], places=places, msg=plane
            )
        self.assertAlmostEqual(plane["yaw"], expected["yaw"], places=places, msg=plane)

    # -------------------------------------------------------- pole fixture
    def _pole_fixture(
        self, source_node, *, live_cap=None, baked_at=None, stamp_bake_scale=True
    ):
        """``(hmap, glb)`` -- the analytic pole map on disk, the reference that
        decodes the very bytes the page will sample, and a GLB whose one
        horizon plane samples them.

        *source_node* is the light's glTF node: a ``translation`` for a
        positional source, a ``rotation`` for a directional one. *live_cap* is
        the record's top-level ``max_stretch`` (the artist's placement cap) and
        *baked_at* the scale the map was baked with -- they differ, and with
        *stamp_bake_scale* false the block carries none and the decode must
        fall back to the record's, which is what such a map was baked with.
        """
        baked_at = POLE_MAX_STRETCH if baked_at is None else baked_at
        live_cap = baked_at if live_cap is None else live_cap
        _, _, _, pole = _pole_geometry_frame()
        png = _pole_horizon_png(
            POLE_BINS,
            POLE_TILE,
            POLE_RMIN,
            POLE_RMAX,
            pole,
            POLE_RADIUS,
            POLE_HEIGHT,
            baked_at,
        )
        maps = self.temp.dir_path()
        with open(os.path.join(maps, "Box_horizon.png"), "wb") as fh:
            fh.write(png)
        hmap = ptk.HorizonMap.from_rgba(
            np.asarray(Image.open(io.BytesIO(png)).convert("RGBA")),
            bins=POLE_BINS,
            size=POLE_TILE,
            r_min=POLE_RMIN,
            r_max=POLE_RMAX,
            max_stretch=baked_at,
        )
        block = {
            "texture": "Box_horizon.png",
            "bins": POLE_BINS,
            "tile": list(POLE_TILE),
            "layout": [8, 8],
            "layers": 2,
            "mapping": "logpolar",
            "r_min": POLE_RMIN,
            "r_max": POLE_RMAX,
            "frame_a": [1, 0, 0],
            "frame_b": [0, 0, 1],
            "encoding": 1,
            "rect": [1, 1, 0, 0],
        }
        if stamp_bake_scale:
            block["max_stretch"] = baked_at
        record = _record(
            "Box_horizon_plane",
            type="horizon",
            follow_source=False,
            max_stretch=live_cap,
            source_type="directional" if "rotation" in source_node else "point",
            horizon=block,
        )
        glb = self._glb(
            None,
            nodes=[
                {"name": "Box_horizon_plane", "mesh": 0, "scale": [14.0, 1.0, 14.0]},
                source_node,
                {"name": "Box_contact_loc"},
                _data_export_node(_payload([record])),
            ],
            materials=[_plane_material("Box_horizon_MAT")],
            search_dirs=[maps],
        )
        return hmap, glb

    # ---------------------------------------------------------------- tests
    def test_the_shim_places_a_following_plane_from_the_model(self):
        found = self._load(self._following_glb())

        self.assertEqual(found["scripts"], ("shadow_rig", "probe"))
        (plane,) = found["before"]
        self.assertTrue(plane["follow"])
        self.assertTrue(plane["isShader"])
        self.assertEqual(plane["uMode"], 0)
        self.assertEqual(plane["uRect"], [1, 1, 0, 0])
        self.assertTrue(plane["mapBound"])
        self.assertFalse(plane["depthWrite"])
        self.assertTrue(plane["transparent"])
        self._assert_placed(plane, self._expected())
        self.assertEqual([c for c in found["console"] if "shadow_rig" in c], [])

    def test_a_moved_source_re_places_the_plane_at_runtime_in_dcc_units(self):
        """R2, and the unit scale: the record is in centimetres, the file in
        metres, and the placement must come out the same as the metre fixture."""
        moved = [-1.0, 1.5, 1.2]
        found = self._load(
            self._following_glb(unit_scale=0.01),
            action=f"session.planes[0].source.position.set({moved[0]}, {moved[1]}, {moved[2]});",
        )
        self._assert_placed(found["before"][0], self._expected(unit_scale=0.01))
        self._assert_placed(
            found["after"][0], self._expected(source=moved, unit_scale=0.01)
        )

    def test_a_directional_source_projects_along_its_nodes_local_minus_y(self):
        """Pinned on an asymmetric rotation (yaw 35 deg, pitch -55 deg): the
        shim reads the source node's local -Y as the way it shines -- the FBX
        light convention, which the DCC exporters bake into the node's rotation
        and FBX2glTF keeps (measured: a Maya directionalLight at rotate
        (-50, 35, 0) arrives with local -Y equal to its Maya world -Z)."""
        rotation = _quat_mul(_quat([0, 1, 0], 35.0), _quat([1, 0, 0], -55.0))
        direction = _quat_rotate(rotation, [0.0, -1.0, 0.0])
        self.assertLess(direction[1], 0.0, "the fixture must shine downward")
        found = self._load(
            self._following_glb(
                source_node={
                    "name": "shadow_source",
                    "translation": [9.0, 9.0, 9.0],
                    "rotation": rotation,
                }
            )
        )
        self._assert_placed(
            found["before"][0], self._expected(direction=direction), places=4
        )

    def test_atlas_planes_batch_into_one_instanced_mesh_and_a_faded_one_keeps_its_fade(
        self,
    ):
        rects = {
            "A_shadow": [0.5, 0.5, 0.0, 0.0],
            "B_shadow": [0.5, 0.5, 0.5, 0.0],
            "C_shadow": [0.5, 0.5, 0.5, 0.5],
        }
        records = [
            _record(
                name,
                atlas={"texture": "shadow_atlas_projected.png", "rect": rect},
                intensity=0.9,
            )
            for name, rect in rects.items()
        ]
        glb = self._glb(
            None,
            nodes=[
                {"name": "A_shadow", "mesh": 0},
                {"name": "B_shadow", "mesh": 1},
                {"name": "C_shadow", "mesh": 2},
                {"name": "shadow_source", "translation": self.SOURCE},
                {"name": "Box_contact_loc", "translation": self.CONTACT},
                _data_export_node(_payload(records)),
            ],
            materials=[
                _plane_material("A_MAT"),
                _plane_material("B_MAT"),
                _plane_material("C_MAT"),
            ],
            animations=[_pointer_animation(2)],
        )
        found = self._load(
            glb,
            action="viewer.mixer.timeScale = 1; viewer.mixer.setTime(1.0); viewer.mixer.timeScale = 0;",
        )
        by_name = {p["name"]: p for p in found["before"]}
        self.assertEqual(found["instancedMeshes"], 1)
        for name in ("A_shadow", "B_shadow"):
            self.assertTrue(by_name[name]["instanced"], by_name[name])
            self.assertEqual(by_name[name]["batchCount"], 2)
            self.assertTrue(by_name[name]["hidden"])
        faded = by_name["C_shadow"]
        self.assertFalse(faded["instanced"])
        self.assertTrue(faded["isShader"])
        for name, plane in by_name.items():
            for got, want in zip(plane["uRect"], ImgUtils.flip_rect_v(rects[name])):
                self.assertAlmostEqual(got, want, places=6, msg=name)
            self.assertAlmostEqual(plane["uIntensity"], 0.9, places=6)
            self._assert_placed(plane, self._expected(), places=4)
        # The fade: the page drives the ORIGINAL material (the clip auto-plays,
        # so the first snapshot already sits somewhere down the ramp); the shim
        # follows it, and the seek to 1 s lands it on 0.5 exactly. The batched
        # planes, which no ramp targets, stay at their static alpha.
        self.assertAlmostEqual(faded["uOpacity"], faded["originalOpacity"], places=6)
        self.assertLess(faded["uOpacity"], 1.0, faded)
        after = {p["name"]: p for p in found["after"]}
        self.assertAlmostEqual(after["C_shadow"]["originalOpacity"], 0.5, places=3)
        self.assertAlmostEqual(
            after["C_shadow"]["uOpacity"],
            after["C_shadow"]["originalOpacity"],
            places=6,
        )
        for name in ("A_shadow", "B_shadow"):
            self.assertEqual(after[name]["uOpacity"], 1.0, after[name])

    def test_the_horizon_decode_uses_the_BAKE_scale_not_the_live_cap(self):
        """The map's R/G are cot(elevation) / max_stretch AS BAKED, while the
        record's top-level max_stretch is the live placement cap -- a keyable
        attribute an artist retunes after the bake. Decoding with the live one
        scales every shadow length by the ratio: here the cap is halved (3 vs
        the bake's 6), which doubles every decoded reach, and the point past
        the pole's tip falls inside the shadow when it must not.
        """
        baked_at, live_cap = POLE_MAX_STRETCH, 3.0
        azimuth, away, _, pole = _pole_geometry_frame()
        source = [2.0 * math.cos(azimuth), 3.0, 2.0 * math.sin(azimuth)]
        points = {
            "in the shadow": (pole[0] + 0.5 * away[0], pole[1] + 0.5 * away[1]),
            "past the tip": (pole[0] + 2.5 * away[0], pole[1] + 2.5 * away[1]),
        }
        expected = {
            label: _pole_alpha(
                point, source, pole, POLE_RADIUS, POLE_HEIGHT, POLE_RMAX, baked_at
            )
            for label, point in points.items()
        }
        self.assertEqual(expected["in the shadow"], 1.0)
        self.assertEqual(expected["past the tip"], 0.0)

        _, glb = self._pole_fixture(
            {"name": "shadow_source", "translation": source},
            live_cap=live_cap,
            baked_at=baked_at,
        )
        found = self._load(glb, action=HORIZON_SAMPLES_JS, points=list(points.values()))

        (plane,) = found["before"]
        self.assertEqual(
            plane["uMaxStretch"],
            baked_at,
            "the decode took the live cap instead of the bake's scale",
        )
        samples = dict(zip(points, found["samples"]))
        for label, want in expected.items():
            self.assertAlmostEqual(
                samples[label], want, delta=0.1, msg=(label, samples)
            )

    def test_the_real_export_shape_batches_three_planes_onto_one_atlas(self):
        """The production case, reproduced from the measured export: three
        projected rigs, one shared atlas, and materials that lost their
        textures through the FBX hop (no baseColorTexture at all). They must
        become ONE InstancedMesh of three, each instance on its own tile --
        which is the whole point of the DCC packing an atlas.
        """
        rects = {
            "Box_keyLight_shadow": [0.484375, 0.484375, 0.007812, 0.507813],
            "Post_keyLight_shadow": [0.484375, 0.484375, 0.007812, 0.007813],
            "Crate_keyLight_shadow": [0.484375, 0.484375, 0.507812, 0.507813],
        }
        records = [
            _record(
                name,
                atlas={"texture": "shadow_atlas_projected.png", "rect": rect},
            )
            for name, rect in rects.items()
        ]
        names = list(rects)
        glb = self._glb(
            None,
            nodes=[
                {"name": names[0], "mesh": 0},
                {"name": names[1], "mesh": 1},
                {"name": names[2], "mesh": 2},
                {"name": "shadow_source", "translation": self.SOURCE},
                {"name": "Box_contact_loc", "translation": self.CONTACT},
                _data_export_node(_payload(records)),
            ],
            materials=[_bare_material(f"{n}_mat") for n in names],
            images=[
                {
                    "name": "shadow_atlas_projected.png",
                    "uri": _data_uri(_silhouette_png(16)),
                },
                {
                    "name": "Box_keyLight_shadow.png",
                    "uri": _data_uri(_silhouette_png(8)),
                },
            ],
            textures=[{"source": 0, "sampler": 0}, {"source": 1, "sampler": 0}],
        )
        found = self._load(glb)

        by_name = {p["name"]: p for p in found["before"]}
        self.assertEqual(found["instancedMeshes"], 1, found["before"])
        for name in names:
            plane = by_name[name]
            self.assertTrue(plane["instanced"], plane)
            self.assertEqual(plane["batchCount"], 3)
            self.assertTrue(plane["hidden"], "the original mesh still draws")
            for got, want in zip(plane["uRect"], ImgUtils.flip_rect_v(rects[name])):
                self.assertAlmostEqual(got, want, places=6, msg=name)
            self._assert_placed(plane, self._expected(), places=4)
        # The atlas really is on the batch material -- three blank quads would
        # otherwise satisfy every assertion above.
        self.assertTrue(by_name[names[0]]["mapBound"])
        self.assertEqual(by_name[names[0]]["mapImage"], [16, 16])

    def test_the_page_renders_the_reference_alpha_of_a_pole_map(self):
        """The pin: the rendered pixels ARE `HorizonMap.alpha`.

        The contract names that method the oracle both engine shaders answer
        to, so this test asks the real page, through the real WebGL2 shader
        built from the shared body, for the alpha at a set of ground points and
        compares each against the reference decoding the SAME map.

        The fixture is still the analytic pole -- a map whose right answer is
        known in closed form -- so the reference is checked against physics in
        the same pass. Pinning the shader to the reference alone would pass on
        a wrong reference; pinning it to the analytic answer alone (which is
        what this test used to do) never showed the two implementations agreeing
        at all. Both, together, are what the doc of record has always claimed.
        """
        azimuth, away, perp, pole = _pole_geometry_frame()
        source = [2.0 * math.cos(azimuth), 3.0, 2.0 * math.sin(azimuth)]
        points = {
            "in the shadow": (pole[0] + 0.5 * away[0], pole[1] + 0.5 * away[1]),
            "past the tip": (pole[0] + 2.5 * away[0], pole[1] + 2.5 * away[1]),
            "off bearing": (
                pole[0] + 0.5 * away[0] + 0.35 * perp[0],
                pole[1] + 0.5 * away[1] + 0.35 * perp[1],
            ),
            "past r_max": (pole[0] + 4.5 * away[0], pole[1] + 4.5 * away[1]),
        }
        # What decides each point, and by how much: the bilinear fetch blurs a
        # boundary over a texel, so a point must sit clear of the boundary that
        # decides it. The shadowed point has to be clear of ALL of them.
        deciders = {
            "in the shadow": ("azimuth", "elevation", "cap", "bin"),
            "past the tip": ("elevation",),
            "off bearing": ("azimuth",),
            "past r_max": ("radius",),
        }
        analytic = {}
        for label, point in points.items():
            analytic[label] = _pole_alpha(
                point,
                source,
                pole,
                POLE_RADIUS,
                POLE_HEIGHT,
                POLE_RMAX,
                POLE_MAX_STRETCH,
            )
            margins = _pole_margins(
                point,
                source,
                pole,
                POLE_RADIUS,
                POLE_HEIGHT,
                POLE_BINS,
                POLE_MAX_STRETCH,
            )
            margins["radius"] = abs(math.hypot(*point) - POLE_RMAX)
            for decider in deciders[label]:
                self.assertGreater(margins[decider], 0.08, (label, decider, margins))
        self.assertEqual(analytic["in the shadow"], 1.0)
        self.assertEqual(analytic["past the tip"], 0.0)
        self.assertEqual(analytic["off bearing"], 0.0)
        self.assertEqual(analytic["past r_max"], 0.0)

        # This record carries no `horizon.max_stretch` (a map baked before the
        # field existed), so the decode falls back to the record's top-level
        # value -- which is what such a map was baked with.
        hmap, glb = self._pole_fixture(
            {"name": "shadow_source", "translation": source},
            stamp_bake_scale=False,
        )
        # The oracle, decoding the very bytes the page will sample. Its frame
        # is the contact's, whose origin is the world origin here, so a ground
        # point is (x, ground, z) -- and the reference REPLACES that height
        # with the map's ground before forming the light vector, which is the
        # rule the shared body now follows too.
        expected = dict(
            zip(
                points,
                hmap.alpha([[x, 0.0, z] for x, z in points.values()], light=source),
            )
        )
        for label, want in analytic.items():
            self.assertAlmostEqual(
                expected[label],
                want,
                delta=0.02,
                msg=f"the REFERENCE disagrees with the closed form at {label!r}",
            )
        found = self._load(glb, action=HORIZON_SAMPLES_JS, points=list(points.values()))

        (plane,) = found["before"]
        self.assertEqual(plane["uMode"], 1)
        self.assertEqual(plane["uMaxStretch"], POLE_MAX_STRETCH)
        self.assertTrue(plane["horizonBound"])
        self.assertEqual(plane["horizonColorSpace"], "")
        self.assertFalse(plane["horizonMipmaps"])
        self.assertTrue(plane["horizonMinFilter"])
        self.assertTrue(plane["horizonWrap"])
        samples = dict(zip(points, found["samples"]))
        for label, want in expected.items():
            self.assertAlmostEqual(
                samples[label],
                want,
                delta=0.05,
                msg=f"the SHADER disagrees with the reference at {label!r}: {samples}",
            )

    # -- the scatter, shared by the positional and directional pins --------
    #: Where the pole's penumbra edge actually sits, measured per source kind.
    #: A near point source throws a shadow that WIDENS with range; a
    #: directional one throws a constant-width strip that reaches much further.
    #: One offset list cannot find both edges, so each caller brings its own --
    #: and `_assert_tracks_reference` fails the run if the points it was given
    #: turn out to miss the penumbra entirely.
    POSITIONAL_EDGES = (
        (0.25, (0.15, 0.17, 0.19)),
        (0.6, (0.17, 0.21, 0.23)),
        (1.2, (0.23, 0.25, 0.28)),
    )
    DIRECTIONAL_EDGES = (
        (0.3, (0.13, 0.14, 0.15)),
        (0.9, (0.13, 0.15, 0.17)),
        (1.8, (0.08, 0.11, 0.14)),
        (2.4, (0.09, 0.12, 0.15)),
    )

    @staticmethod
    def _scatter_points(edges, along_axis=None):
        """Ground points down the pole's shadow and across it, plus *edges* --
        points measured to sit ON the penumbra.

        The edge is where the shader has to reproduce the bilinear tap blend,
        the coverage integral and the interval lerp all at once; without it a
        binary in/out grid would pass on a shader that gets none of the three
        right.

        *along_axis* is the direction the shadow actually runs, defaulting to
        the positional fixture's. A directional source on a different bearing
        throws its shadow along a different line, and offsets measured across
        the wrong axis land in the umbra instead of on its edge.
        """
        _, away, perp, pole = _pole_geometry_frame()
        away = tuple(away if along_axis is None else along_axis)
        perp = (-away[1], away[0])

        def at(along, across):
            return (
                pole[0] + along * away[0] + across * perp[0],
                pole[1] + along * away[1] + across * perp[1],
            )

        return [
            at(along, across)
            for along in (0.1, 0.35, 0.7, 1.1, 1.7, 2.4)
            for across in (-0.3, -0.11, 0.0, 0.11, 0.3)
        ] + [
            at(along, sign * across)
            for along, edge in edges
            for across in edge
            for sign in (-1.0, 1.0)
        ]

    def _assert_tracks_reference(self, found, expected, points):
        """Every rendered sample within 0.05 of the reference, with the sample
        counts in the message: a run that rendered a field of zeros, or none at
        all, must not read as a pass."""
        samples = found["samples"]
        self.assertEqual(len(samples), len(points), found)
        shadowed = int((expected > 0.5).sum())
        partial = int(((expected > 0.02) & (expected < 0.98)).sum())
        self.assertGreater(shadowed, 4, "the fixture puts nothing in shadow")
        self.assertGreater(len(points) - shadowed, 4, "the fixture leaves nothing lit")
        self.assertGreater(
            partial, 8, "every point is a hard 0 or 1: the penumbra is untested"
        )
        worst = max(abs(a - b) for a, b in zip(samples, expected))
        detail = (
            f"{len(points)} points, {shadowed} in shadow, {partial} in penumbra, "
            f"worst |shader - reference| = {worst:.4f}"
        )
        for i, (got, want) in enumerate(zip(samples, expected)):
            self.assertAlmostEqual(
                got, float(want), delta=0.05, msg=f"point {i} of {detail}"
            )

    def test_the_page_tracks_the_reference_across_a_scatter_of_ground_points(self):
        """The pin, at width: 48 points through the whole decision surface.

        Four points can be satisfied by a shader that is right about the
        bearing and wrong about everything else. This walks the pole's shadow
        lengthways and across it -- inside, along the penumbra edge, past the
        tip -- and requires the page to agree with `HorizonMap.alpha` at every
        one.
        """
        azimuth, _, _, _ = _pole_geometry_frame()
        source = [2.0 * math.cos(azimuth), 3.0, 2.0 * math.sin(azimuth)]
        points = self._scatter_points(self.POSITIONAL_EDGES)
        hmap, glb = self._pole_fixture({"name": "shadow_source", "translation": source})
        expected = hmap.alpha([[x, 0.0, z] for x, z in points], light=source)
        found = self._load(glb, action=HORIZON_SAMPLES_JS, points=points)
        self._assert_tracks_reference(found, expected, points)

    def test_a_directional_source_shadows_along_the_way_it_shines(self):
        """The other half of the source contract, and the half a sign error
        hides in.

        A directional source travels as the direction it SHINES with ``w = 0``
        -- the reference's vocabulary, negated in the shader -- so a flipped
        sign puts the shadow on the light's side of the pole and every
        positional test still passes. Nothing rendered this path before: the
        viewer's directional coverage was the projected rig's PLACEMENT, and
        Unity's horizon tests are all positional.

        The direction is the one that shines from where the positional fixture
        puts its light, so the shadow lies along the same bearing -- but with a
        constant elevation everywhere instead of a per-point one, which is what
        makes it a different evaluation and not a restatement.
        """
        azimuth, _, _, _ = _pole_geometry_frame()
        # 20 degrees above the horizon: shallow enough that the pole reaches
        # height / tan(e) = 2.75 units, which spans the whole scatter (a steep
        # source throws a shadow too short for most points to decide anything).
        #
        # And 0.3 of a bin OFF the bearing the positional fixture uses. That
        # one is the middle of bin 18, which for a POSITIONAL source is only
        # the bearing at the CONTACT -- every fragment sees the light on its
        # own bearing. A directional source has one bearing for the whole
        # plane, so a bin centre would put s at exactly 0.5 for every fragment
        # at once: a dead tie in side, and a boundary in floor(s * 16), decided
        # by whether the GPU's float32 atan2 lands a hair above or below
        # numpy's float64. Off-centre, s = 0.8 and nothing sits on an edge.
        step = TWO_PI / POLE_BINS
        bearing = azimuth + 0.3 * step
        elevation = math.radians(20.0)
        shine = [
            -math.cos(bearing) * math.cos(elevation),
            -math.sin(elevation),
            -math.sin(bearing) * math.cos(elevation),
        ]
        self.assertLess(shine[1], 0.0, "the fixture must shine downward")

        points = self._scatter_points(
            self.DIRECTIONAL_EDGES,
            along_axis=(math.cos(bearing - math.pi), math.sin(bearing - math.pi)),
        )
        hmap, glb = self._pole_fixture(
            {
                "name": "shadow_source",
                "translation": [
                    9.0,
                    9.0,
                    9.0,
                ],  # a directional source's position is inert
                "rotation": _quat_from_minus_y(shine),
            }
        )
        expected = hmap.alpha([[x, 0.0, z] for x, z in points], direction=shine)
        found = self._load(glb, action=HORIZON_SAMPLES_JS, points=points)
        (plane,) = found["before"]
        self.assertEqual(plane["uMode"], 1)
        self._assert_tracks_reference(found, expected, points)

    def test_a_blender_frame_shadows_too(self):
        """Blender's frame arrives as ``(X, -Z)`` in the file's axes -- its
        bake runs bearing from local X toward local Y, which the Y-up
        conversion turns into -Z -- so its bearing sense is the mirror of
        Maya's ``(X, Z)``. The frame's up is the CONTACT's up, never a cross
        product of A and B: ``cross(B, A)`` is +Y for one sense and -Y for
        the other, which put every Blender-exported source below the horizon
        and drew nothing. Measured before the fix: alpha 0 at every point.

        The map is baked in that mirrored sense (the pole at map coords
        ``(x, -z)``) and the reference is fed frame-mapped points, exactly
        as the shader maps world points through A and B.
        """
        azimuth, _, _, _ = _pole_geometry_frame()
        # World bearing of the light, and the pole's WORLD position; in the
        # (X, -Z) frame the map coords are (x, -z) for both.
        source = [2.0 * math.cos(azimuth), 3.0, 2.0 * math.sin(azimuth)]
        points = self._scatter_points(self.POSITIONAL_EDGES)
        to_frame = lambda x, z: (x, -z)  # noqa: E731 -- dot(P, A), dot(P, B)
        _, _, _, pole_world = _pole_geometry_frame()
        png = _pole_horizon_png(
            POLE_BINS,
            POLE_TILE,
            POLE_RMIN,
            POLE_RMAX,
            to_frame(*pole_world),
            POLE_RADIUS,
            POLE_HEIGHT,
            POLE_MAX_STRETCH,
        )
        maps = self.temp.dir_path()
        with open(os.path.join(maps, "Box_horizon.png"), "wb") as fh:
            fh.write(png)
        hmap = ptk.HorizonMap.from_rgba(
            np.asarray(Image.open(io.BytesIO(png)).convert("RGBA")),
            bins=POLE_BINS,
            size=POLE_TILE,
            r_min=POLE_RMIN,
            r_max=POLE_RMAX,
            max_stretch=POLE_MAX_STRETCH,
        )
        frame_pts = [[fx, 0.0, fz] for fx, fz in (to_frame(x, z) for x, z in points)]
        frame_light = [source[0], source[1], -source[2]]
        expected = hmap.alpha(frame_pts, light=frame_light)
        self.assertGreater(int((expected > 0.5).sum()), 4)

        record = _record(
            "Box_horizon_plane",
            type="horizon",
            follow_source=False,
            max_stretch=POLE_MAX_STRETCH,
            horizon={
                "texture": "Box_horizon.png",
                "bins": POLE_BINS,
                "tile": list(POLE_TILE),
                "layout": [8, 8],
                "layers": 2,
                "mapping": "logpolar",
                "r_min": POLE_RMIN,
                "r_max": POLE_RMAX,
                "frame_a": [1, 0, 0],
                "frame_b": [0, 0, -1],
                "encoding": 1,
                "rect": [1, 1, 0, 0],
                "max_stretch": POLE_MAX_STRETCH,
            },
        )
        glb = self._glb(
            None,
            nodes=[
                {"name": "Box_horizon_plane", "mesh": 0, "scale": [14.0, 1.0, 14.0]},
                {"name": "shadow_source", "translation": source},
                {"name": "Box_contact_loc"},
                _data_export_node(_payload([record])),
            ],
            materials=[_plane_material("Box_horizon_MAT")],
            search_dirs=[maps],
        )
        found = self._load(glb, action=HORIZON_SAMPLES_JS, points=points)
        self._assert_tracks_reference(found, expected, points)


if __name__ == "__main__":
    unittest.main()
