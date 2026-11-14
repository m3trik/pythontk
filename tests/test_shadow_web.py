# !/usr/bin/python
# coding=utf-8
"""The shadow rigs' web route: ``MeshConvert.apply_glb_shadows`` and the
packaged ``shadow_rig`` viewer script, against the contract in
``mayatk/docs/shadow_rig_morphing.md`` (*Contracts*).

Two halves, deliberately in one module because they share the fixtures:

* Pure Python -- the pass over hand-authored GLBs (a plane wearing a base
  colour texture, a source, a contact, the ``data_export`` carrier): the
  ``shadows`` record carried as it stands and every file it names riding as
  media, the prune's and the optimiser's keep-as-found rule, the verifier's
  media gate and the server's auto-activation.
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
from conftest import browser_runtime_available  # noqa: E402

try:
    import numpy as np
    from PIL import Image
except ImportError:  # pragma: no cover - the imaging stack is a test-only need here
    np = None
    Image = None

ptk.TestSandbox.activate()

#: The family logger: every mesh_convert module logs under it.
MESH_CONVERT_LOGGER = "pythontk.file_utils.mesh_convert"
#: Where ``MeshConvert.fbx_to_glb`` runs the converter from.
FBX2GLTF_MODULE = "pythontk.file_utils.mesh_convert._fbx2gltf"
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


#: The horizon block a fixture plane carries where its test bakes none:
#: every v3 plane states its map (the record requires it), and a test that
#: never ships the file draws the plane's silhouette instead.
HORIZON_LAYOUT = {
    "texture": "Box_horizon.png",
    "size": 64,
    "spans": 1,
    "bounds": [-0.6, 0.6, -0.6, 0.6],
    "height_scale": 1.0,
    "frame_a": [1, 0, 0],
    "frame_b": [0, 0, 1],
    "rect": [1, 1, 0, 0],
}


def _record(
    name: str,
    *,
    texture=None,
    atlas=None,
    canvas=(-1.0, 1.0, -0.5, 0.5),
    draw=True,
    horizon=None,
    **overrides,
):
    """One v3 ``shadows`` plane record, the contract's shape: its horizon map
    (*horizon*, :data:`HORIZON_LAYOUT` by default) and its silhouette -- the
    extension -- named *texture* (the plane's own PNG by default), packed
    into *atlas* (``{"texture", "rect"}``, the rect bottom-left), placed by
    *canvas* and drawn where *draw* (the projected rig; false is the horizon
    rig, its silhouette the fallback)."""
    silhouette = {
        "texture": texture or f"{name}.png",
        "draw": draw,
    }
    if atlas:
        silhouette["atlas"] = dict(atlas)
    record = {
        "name": name,
        "source": "shadow_source",
        "contact": "Box_contact_loc",
        "intensity": 1.0,
        "source_type": "point",
        "source_size": 0.0,
        "source_angle": 0.0,
        "follow_source": True,
        "ground": 0.0,
        "radius": 0.6,
        "height": 1.2,
        "max_stretch": 6.0,
        "falloff_power": 1.2,
        "fade_height": 2.4,
        "source_ramp": 1.0,
        "canvas": [float(v) for v in canvas],
        "horizon": dict(horizon or HORIZON_LAYOUT),
        "silhouette": silhouette,
    }
    record.update(overrides)
    return record


def _payload(planes, unit_scale: float = 1.0, version=None) -> dict:
    """The record around *planes*, at the current version unless *version*."""
    current = ptk.SceneRecords.SHADOWS.version
    return {
        "version": current if version is None else version,
        "unit_scale": unit_scale,
        "planes": planes,
    }


def _data_export_node(payload: dict) -> dict:
    """The carrier exactly as FBX2glTF transcribes a Maya ``data_export``."""
    return {
        "name": "data_export",
        "extras": {
            "fromFBX": {
                "userProperties": {
                    ptk.SceneRecords.SHADOWS.key: {
                        "type": "eFbxString",
                        "value": json.dumps(payload),
                    }
                }
            }
        },
    }


def _pointer_animation(material: int) -> dict:
    """An authored fade as ``apply_glb_fades`` writes it: alpha 1 -> 0 over 2 s
    -- what a v2 file's shadow plane carried, which a v3 reader never reads."""
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


def _media_views(gltf: dict) -> dict:
    """``{file name: (view index, extras)}`` of every bufferView riding a file
    (``SceneRecords.GLB_MEDIA``)."""
    media = ptk.SceneRecords.GLB_MEDIA
    return {
        view[media.name]: (index, view.get("extras"))
        for index, view in enumerate(gltf.get("bufferViews") or [])
        if isinstance(view.get(media.name), str)
    }


def _write_glb(
    path,
    *,
    nodes,
    materials,
    images,
    textures,
    animations=None,
    z_up=False,
    uv_rects=None,
):
    """A minimal but VALID binary glTF around one shared unit quad.

    The quad is the DCC's plane as the FBX hop delivers it: 1 x 1 in local X/Z
    at y = 0, centred on its pivot, UVs in glTF's top-left convention with
    v = 0 on the -Z (near) edge -- the PNG's top row is the light-side edge.
    *z_up* writes Blender's instead, as a push measured it: the quad in local
    X/Y at z = 0 (its exporter turns only the ROOT nodes upright and never the
    mesh data), its near edge -Y. Every mesh node references mesh index = its
    own material index, one mesh per material, so a per-material fade has one
    representative mesh. A node named in another's ``children`` is not a root.
    *uv_rects* packs the quads as the DCC packs an atlas: per mesh index, the
    tile ``[scale_u, scale_v, offset_u, offset_v]`` (glTF top-left) its UVs
    are remapped into; a mesh it does not name keeps the unit square.
    """
    if z_up:
        positions = struct.pack(
            "<12f", 0.5, 0.5, 0, -0.5, 0.5, 0, -0.5, -0.5, 0, 0.5, -0.5, 0
        )
        unit_uvs = (1, 0, 0, 0, 0, 1, 1, 1)
        indices = struct.pack("<6H", 0, 1, 2, 0, 2, 3)
        bounds = ([-0.5, -0.5, 0], [0.5, 0.5, 0])
    else:
        positions = struct.pack(
            "<12f", -0.5, 0, 0.5, 0.5, 0, 0.5, -0.5, 0, -0.5, 0.5, 0, -0.5
        )
        unit_uvs = (0, 1, 1, 1, 0, 0, 1, 0)
        indices = struct.pack("<6H", 0, 1, 2, 1, 3, 2)
        bounds = ([-0.5, 0, -0.5], [0.5, 0, 0.5])
    uvs = struct.pack("<8f", *unit_uvs)
    children = {c for node in nodes for c in node.get("children", [])}
    times = struct.pack("<3f", 0.0, 1.0, 2.0)
    alphas = struct.pack("<12f", 1, 1, 1, 1, 1, 1, 1, 0.5, 1, 1, 1, 0)
    chunks = [positions, uvs, indices, times, alphas]
    #: mesh index -> its TEXCOORD_0 accessor: the unit square's (1), or its
    #: tile's, appended after the fixed five.
    texcoords = {}
    for mesh, rect in enumerate(uv_rects or []):
        if rect is None:
            continue
        scale_uv, offset_uv = rect[:2], rect[2:]
        tiled = [v * scale_uv[k % 2] + offset_uv[k % 2] for k, v in enumerate(unit_uvs)]
        texcoords[mesh] = len(chunks)
        chunks.append(struct.pack("<8f", *tiled))
    views, blob, offset = [], b"", 0
    for raw in chunks:
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw)})
        pad = (4 - len(raw) % 4) % 4
        blob += raw + b"\0" * pad
        offset += len(raw) + pad
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [i for i in range(len(nodes)) if i not in children]}],
        "nodes": nodes,
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "TEXCOORD_0": texcoords.get(index, 1),
                        },
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
                "min": bounds[0],
                "max": bounds[1],
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
        ]
        + [
            {"bufferView": view, "componentType": 5126, "count": 4, "type": "VEC2"}
            for view in sorted(texcoords.values())
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


_box = ptk.ShadowHorizonConformance._box


def _pole_map(pole_xz, pole_radius, pole_height, *, reach=None):
    """One grounded pole -- a square post of half-width *pole_radius* -- baked
    by pythontk's own bake into the height-field map the contract ships, over
    a footprint reaching *reach* (``POLE_REACH``) past the post on every side
    so a penumbra has room to fall inside it. Returns ``(png_bytes, hmap)``:
    the bytes the page will sample, and the map decoding them."""
    reach = POLE_REACH if reach is None else reach
    px, pz = pole_xz
    r = pole_radius
    hmap = ptk.ShadowHorizon.bake(
        [_box(px - r, px + r, 0.0, pole_height, pz - r, pz + r)],
        up=1,
        size=POLE_SIZE,
        spans=POLE_SPANS,
        bounds=(px - reach, px + reach, pz - reach, pz + reach),
    )
    return _png_bytes(hmap.to_rgba()), hmap


def _horizon_block(hmap, frame_b=(0, 0, 1), **extra):
    """The record's ``horizon`` block for *hmap*, the contract's shape."""
    block = {
        "texture": "Box_horizon.png",
        "size": hmap.size,
        "spans": hmap.spans,
        "bounds": [float(v) for v in hmap.bounds],
        "height_scale": float(hmap.height_scale),
        "frame_a": [1, 0, 0],
        "frame_b": list(frame_b),
        "rect": [1, 1, 0, 0],
    }
    block.update(extra)
    return block


def _pole_geometry(point_xz, source, pole_xz, pole_radius, pole_height):
    """The source and the post as seen from a ground point: the source's
    cotangent of elevation and azimuth, the post's azimuth, its angular
    half-extent (the widest of its four corners) and cot(top) at its near
    face -- where a ray toward the source is lowest over it."""
    px, pz = point_xz
    lx, ly, lz = source[0] - px, source[1], source[2] - pz
    horizontal = math.hypot(lx, lz)
    cot_source = horizontal / ly if ly > 0 else math.inf
    phi = math.atan2(lz, lx) % TWO_PI
    dx, dz = pole_xz[0] - px, pole_xz[1] - pz
    dist = math.hypot(dx, dz)
    centre = math.atan2(dz, dx)
    half = max(
        abs(
            (
                math.atan2(dz + sz * pole_radius, dx + sx * pole_radius)
                - centre
                + math.pi
            )
            % TWO_PI
            - math.pi
        )
        for sx in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    )
    return {
        "cot_source": cot_source,
        "phi": phi,
        "delta": abs((phi - centre + math.pi) % TWO_PI - math.pi),
        "half": half,
        "cot_top": max(dist - pole_radius, 1e-6) / pole_height,
    }


def _pole_alpha(point_xz, source, pole_xz, pole_radius, pole_height):
    """The closed-form answer at a ground point for a sizeless source: is the
    source, seen from there, behind the post -- within its azimuth extent and
    under its top?"""
    g = _pole_geometry(point_xz, source, pole_xz, pole_radius, pole_height)
    blocked = g["delta"] <= g["half"] and g["cot_top"] <= g["cot_source"]
    return 1.0 if blocked else 0.0


def _pole_margins(point_xz, source, pole_xz, pole_radius, pole_height):
    """How far the point sits from each decision boundary -- azimuth in
    radians, elevation in cotangent units -- so a fixture point cannot pass
    by sitting on an edge a texel could round either way."""
    g = _pole_geometry(point_xz, source, pole_xz, pole_radius, pole_height)
    return {
        "azimuth": abs(g["delta"] - g["half"]),
        "elevation": abs(g["cot_source"] - g["cot_top"]),
    }


#: The pole fixture's geometry, shared by every horizon test below: a square
#: post baked into a footprint with room for its penumbra, and the record's
#: live placement cap (nothing of the map's -- the map has no scale to
#: retune).
POLE_SIZE, POLE_SPANS = 64, 1
POLE_REACH = 0.6
POLE_RADIUS, POLE_HEIGHT = 0.15, 1.0
POLE_MAX_STRETCH = 6.0
#: The finite sources the scatter pins run with: a positional diameter and a
#: directional angular diameter (radians).
POLE_SOURCE_SIZE = 0.3
POLE_SOURCE_ANGLE = 0.06


def _pole_geometry_frame():
    """``(azimuth, away, perp, pole)`` -- the bearing the light sits on, the
    unit vectors along and across the shadow, and the pole's base."""
    azimuth = 18.5 / 32.0 * TWO_PI  # the bearing the fixture has always used
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
    """``apply_glb_shadows`` over hand-built GLBs: the ``shadows`` record
    carried as it stands, every file it names riding as media."""

    #: The same map SHAPE the live tests use -- the record has to describe the
    #: PNG beside it, so both come from one set of constants; only the pole's
    #: position differs, and this suite never samples the map (the pass
    #: carries it as opaque bytes).
    HORIZON = {
        "texture": "Box_horizon.png",
        "size": POLE_SIZE,
        "spans": POLE_SPANS,
        "bounds": [-0.14, 1.06, -0.41, 0.79],
        "height_scale": POLE_HEIGHT,
        "frame_a": [1, 0, 0],
        "frame_b": [0, 0, 1],
        "rect": [0.5, 0.5, 0.25, 0.25],
    }
    ATLAS_RECT = [0.5, 0.5, 0.0, 0.5]

    @classmethod
    def setUpClass(cls):
        cls.temp = ptk.TempArtifacts("shadow_web_pass", policy="scoped")
        cls.maps = cls.temp.dir_path()
        cls.horizon_png, _ = _pole_map((0.46, 0.19), POLE_RADIUS, POLE_HEIGHT)
        cls.atlas_png = _silhouette_png(16)
        cls.own_png = _silhouette_png(8)
        for name, data in (
            ("Box_horizon.png", cls.horizon_png),
            ("shadow_atlas_projected.png", cls.atlas_png),
            ("Box_shadow.png", cls.own_png),
            ("Box_horizon_plane.png", cls.own_png),
        ):
            with open(os.path.join(cls.maps, name), "wb") as fh:
                fh.write(data)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _glb(self, payload, *, nodes=None, materials=None, images=None, textures=None):
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
            images=images
            if images is not None
            else [{"name": "shadow_atlas", "uri": _data_uri(_silhouette_png())}],
            textures=textures if textures is not None else [{"source": 0, "sampler": 0}],
        )

    def _v3(self):
        return _payload(
            [
                _record(
                    "Box_shadow",
                    atlas={
                        "texture": "shadow_atlas_projected.png",
                        "rect": self.ATLAS_RECT,
                    },
                    horizon=self.HORIZON,
                ),
                _record("Box_horizon_plane", draw=False, horizon=self.HORIZON),
            ],
            unit_scale=0.01,
        )

    def _read(self, path):
        with MeshConvert.open_glb(path) as edit:
            return json.loads(json.dumps(edit.gltf))

    def _payload_of(self, path, view):
        with MeshConvert.open_glb(path) as edit:
            entry = edit.gltf["bufferViews"][view]
            start = int(entry.get("byteOffset", 0))
            return bytes(edit.bin_data[start : start + int(entry["byteLength"])])

    def test_the_record_rides_as_it_stands_and_each_file_it_names_as_media(self):
        """One layout, two containers: the root extras hold the record the
        carrier holds, and each file it names -- horizon map, silhouette,
        atlas -- the bufferView named by that file, its bytes as they are (a
        horizon map is data) and its extras stating them: PNG, no scale. No
        glTF image, texture or sampler is made, and nothing is bound by index.
        Added: 2026-11-07"""
        payload = self._v3()
        path = self._glb(payload)
        before = self._read(path)
        carried = MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])

        gltf = self._read(path)
        self.assertEqual(carried, payload)
        self.assertEqual(gltf["extras"][ptk.SceneRecords.SHADOWS.key], payload)
        res = ptk.ShadowRecord.validate(carried)
        self.assertEqual((res.errors, res.warnings), ([], []))
        views = _media_views(gltf)
        self.assertEqual(
            set(views),
            {
                "Box_horizon.png",
                "shadow_atlas_projected.png",
                "Box_shadow.png",
                "Box_horizon_plane.png",
            },
        )
        for name, data in (
            ("Box_horizon.png", self.horizon_png),
            ("shadow_atlas_projected.png", self.atlas_png),
            ("Box_shadow.png", self.own_png),
        ):
            index, extras = views[name]
            self.assertEqual(extras, {"mimeType": "image/png"}, name)
            self.assertEqual(self._payload_of(path, index), data, f"{name} re-encoded")
        for key in ("images", "textures", "samplers", "materials"):
            self.assertEqual(gltf[key], before[key], key)
        self.assertNotIn("texture_index", json.dumps(gltf["extras"]))

    def test_rerunning_adds_nothing(self):
        path = self._glb(self._v3())
        first = MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])
        before = self._read(path)
        # The record in the root extras is the one read: the carrier gone, as
        # the conversion leaves it, the files found again by name.
        second = MeshConvert.apply_glb_shadows(path, search_dirs=[])
        after = self._read(path)
        self.assertEqual(first, second)
        self.assertEqual(after["bufferViews"], before["bufferViews"])
        self.assertEqual(after["extras"], before["extras"])

    def test_a_record_of_another_version_is_refused_not_guessed(self):
        """A reader reads the schema the record declares alone: v2 -- a
        plane's 'type' and its silhouette's fields on the plane, no fade terms
        -- is refused as a newer one is, and said, the cure a re-export; never
        read through a branch for its shape. Added: 2026-10-20"""
        current = ptk.SceneRecords.SHADOWS.version
        for version in (current - 1, current + 1):
            with self.subTest(version=version):
                path = self._glb(_payload([_record("Box_shadow")], version=version))
                with self.assertLogs(MESH_CONVERT_LOGGER, level="WARNING") as log:
                    self.assertIsNone(
                        MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])
                    )
                self.assertTrue(
                    any(
                        f"v{version} " in line and "re-export the file" in line
                        for line in log.output
                    ),
                    log.output,
                )
                gltf = self._read(path)
                self.assertNotIn(ptk.SceneRecords.SHADOWS.key, gltf.get("extras", {}))
                self.assertEqual(_media_views(gltf), {})

    def test_the_retired_plane_defaults_are_the_planes_declared_ones(self):
        """``MeshConvert.SHADOW_PLANE_DEFAULTS`` (retired 2026-10-20, removed
        in 0.18.0: a plane states every field ``ShadowPlane`` declares, and the
        pass fills none it leaves unsaid) warns, and serves the table it was
        in 0.16.1 -- REGRESSION (2026-10-24): it served ``""`` for no source or
        contact, the plane's own spelling, where its readers test ``None``."""
        with self.assertWarns(DeprecationWarning):
            defaults = MeshConvert.SHADOW_PLANE_DEFAULTS
        self.assertEqual(
            defaults,
            {
                "type": "projected",
                "intensity": 1.0,
                "source": None,
                "source_type": "point",
                "source_size": 0.0,
                "source_angle": 0.0,
                "follow_source": False,
                "contact": None,
                "ground": 0.0,
                "radius": 0.5,
                "height": 1.0,
                "max_stretch": 6.0,
                "canvas": [-1.0, 1.0, -0.5, 0.5],
            },
        )

    def test_a_file_found_nowhere_is_warned_once_naming_its_planes(self):
        """A plane whose horizon map is nowhere still ships in the record (its
        silhouette is its fallback): the map is warned ONCE, naming every
        plane that reads it, and the rest ride. Added: 2026-11-07"""
        path = self._glb(self._v3())
        with self.assertLogs(MESH_CONVERT_LOGGER, level="WARNING") as log:
            carried = MeshConvert.apply_glb_shadows(path, search_dirs=[])
        horizon = [line for line in log.output if "Box_horizon.png" in line]
        self.assertEqual(len(horizon), 1, log.output)
        self.assertIn("Box_shadow", horizon[0])
        self.assertIn("Box_horizon_plane", horizon[0])
        self.assertEqual(len(carried["planes"]), 2)
        self.assertEqual(_media_views(self._read(path)), {})

    def test_no_record_is_a_clean_no_op(self):
        path = self._glb(None, nodes=[{"name": "Box_shadow", "mesh": 0}])
        before = self._read(path)
        self.assertIsNone(MeshConvert.apply_glb_shadows(path))
        self.assertEqual(self._read(path), before)

    def test_a_plane_this_file_does_not_carry_ships_no_bytes(self):
        """The record names the export's planes; a GLB of part of it (a plane
        renamed away here) carries the record whole and the files of the
        planes it holds alone -- a plane elsewhere would ship dead bytes.
        Added: 2026-11-07"""
        payload = self._v3()
        payload["planes"][1]["horizon"] = dict(
            self.HORIZON, texture="Elsewhere_horizon.png"
        )
        with open(os.path.join(self.maps, "Elsewhere_horizon.png"), "wb") as fh:
            fh.write(self.horizon_png)
        nodes = [
            {"name": "Box_shadow", "mesh": 0},
            {"name": "shadow_source", "translation": [1.5, 2.0, -0.5]},
            {"name": "Box_contact_loc", "translation": [0.2, 0.0, 0.3]},
            {"name": "Other_plane", "mesh": 1},
            _data_export_node(payload),
        ]
        path = self._glb(None, nodes=nodes)
        with self.assertLogs(MESH_CONVERT_LOGGER, level="INFO") as log:
            carried = MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])
        self.assertEqual(carried, payload)
        views = _media_views(self._read(path))
        self.assertNotIn("Elsewhere_horizon.png", views)
        self.assertNotIn("Box_horizon_plane.png", views)
        self.assertIn("Box_horizon.png", views)
        self.assertTrue(any("1 plane(s) not in this export" in m for m in log.output))

    def test_a_plane_nothing_tells_apart_ships_no_bytes_and_is_said(self):
        """Planes resolve by the rule the lightmaps record does (the exact
        name, else the one after the last ':', a tie never guessed): a bare
        name two namespaced planes share is ambiguous -- warned, its files
        unshipped, never put on the wrong plane. Added: 2026-11-07"""
        nodes = [
            {"name": "A:Box_shadow", "mesh": 0},
            {"name": "B:Box_shadow", "mesh": 0},
            {"name": "shadow_source"},
            {"name": "Box_contact_loc"},
            _data_export_node(_payload([_record("Box_shadow", horizon=self.HORIZON)])),
        ]
        path = self._glb(None, nodes=nodes, materials=[_plane_material("m")])
        with self.assertLogs(MESH_CONVERT_LOGGER, level="WARNING") as log:
            MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])
        self.assertTrue(any("ambiguous" in line for line in log.output), log.output)
        self.assertEqual(_media_views(self._read(path)), {})

    def test_a_map_the_fbx_embedded_for_a_material_rides_from_the_files_image(self):
        """The real Maya shape: the FBX embedded the atlas for the plane's
        material, so the GLB holds it as an image the converter named by its
        source file; with no loose copy anywhere, those bytes are the atlas
        the record names. Added: 2026-11-07"""
        atlas = _silhouette_png(12)
        path = self._glb(
            self._v3(),
            images=[{"name": "shadow_atlas_projected", "uri": _data_uri(atlas)}],
        )
        MeshConvert.apply_glb_shadows(path, search_dirs=[])
        views = _media_views(self._read(path))
        self.assertEqual(
            self._payload_of(path, views["shadow_atlas_projected.png"][0]), atlas
        )

    def test_the_files_own_copy_rides_before_a_loose_one(self):
        """The copy the FBX itself carried wins over a loose one, as it always
        has for the atlas: an image the GLB holds under the file's name rides
        even where a folder holds other bytes of that name (a stale export
        beside the scene). Added: 2026-11-08 (HEAD's order, restored)."""
        atlas = _silhouette_png(12)
        self.assertNotEqual(atlas, self.atlas_png, "the copies must differ")
        path = self._glb(
            self._v3(),
            images=[{"name": "shadow_atlas_projected", "uri": _data_uri(atlas)}],
        )
        MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])
        views = _media_views(self._read(path))
        self.assertEqual(
            self._payload_of(path, views["shadow_atlas_projected.png"][0]), atlas
        )
        # A file the GLB holds no image of rides from the folder.
        self.assertEqual(
            self._payload_of(path, views["Box_horizon.png"][0]), self.horizon_png
        )

    def test_a_carried_record_its_shape_refuses_is_refused_and_said(self):
        """The record in the root extras is held to the current shape as the
        carrier's is: a plane missing a field the record requires (a file an
        earlier v3 build wrote) is refused and said -- its fix a re-export --
        never carried on or given media. Added: 2026-11-08"""
        payload = self._v3()
        del payload["planes"][0]["source_ramp"]
        path = self._glb(None, nodes=[{"name": "Box_shadow", "mesh": 0}])
        with MeshConvert.open_glb(path) as edit:
            edit.gltf.setdefault("extras", {})[ptk.SceneRecords.SHADOWS.key] = payload
            edit.dirty = True
        with self.assertLogs(MESH_CONVERT_LOGGER, level="WARNING") as log:
            self.assertIsNone(MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps]))
        self.assertTrue(
            any("source_ramp" in line and "re-export" in line for line in log.output),
            log.output,
        )
        self.assertEqual(_media_views(self._read(path)), {})

    def test_the_files_survive_the_prune_and_the_optimiser_as_found(self):
        """No texture index reaches a file the record names, so the prune
        cannot drop or renumber it, and the optimiser -- resizing and
        re-encoding everything else -- leaves the data map and the
        silhouettes byte for byte. Added: 2026-11-07"""
        big = _png_bytes(np.full((64, 64, 4), 200, np.uint8))
        path = self.temp.path(extension=".glb")
        _write_glb(
            path,
            nodes=[
                {"name": "Box_shadow", "mesh": 0},
                {"name": "Box_horizon_plane", "mesh": 1},
                {"name": "other", "mesh": 2},
                _data_export_node(self._v3()),
            ],
            materials=[
                _plane_material("shadow", texture=0),
                _plane_material("horizon", texture=0),
                _plane_material("other", texture=1),
            ],
            images=[
                {"name": "orphan", "uri": _data_uri(big)},
                {"name": "plain", "uri": _data_uri(big)},
            ],
            textures=[{"source": 1, "sampler": 0}, {"source": 1, "sampler": 0}],
        )
        MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])
        with MeshConvert.open_glb(path) as edit:
            MeshConvert.prune_glb_unreferenced_textures(edit)
        for image_format in ("PNG", "WEBP"):
            with self.subTest(image_format=image_format):
                MeshConvert.optimize_glb_textures(
                    path, max_size=16, image_format=image_format, workers=1
                )
                gltf = self._read(path)
                views = _media_views(gltf)
                for name, data in (
                    ("Box_horizon.png", self.horizon_png),
                    ("shadow_atlas_projected.png", self.atlas_png),
                    ("Box_shadow.png", self.own_png),
                ):
                    index, extras = views[name]
                    self.assertEqual(extras, {"mimeType": "image/png"}, name)
                    self.assertEqual(self._payload_of(path, index), data, name)
                with MeshConvert.open_glb(path) as edit:
                    sizes = [
                        Image.open(io.BytesIO(edit._image_payload(i))).size
                        for i in edit.images
                    ]
                self.assertEqual(sizes, [(16, 16)], "the control was not resized")

    def test_the_handoff_names_the_record(self):
        """A standalone reader is told the record is there, in the artifact:
        the handoff of a file carrying it names it -- its ``assembly`` row,
        addressed at the record under the part the FBX's record gives it too,
        stating what this file's planes use."""
        path = self._glb(self._v3())
        self.assertIsNotNone(
            MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])
        )
        with MeshConvert.open_glb(path) as edit:
            MeshConvert.embed_glb_handoff(
                edit, ptk.HandoffRecord.block("glb", [], {"application": "maya"})
            )
        handoff = MeshConvert.read_glb_handoff(path)
        block = f"extras.{ptk.SceneRecords.SHADOWS.key}"
        record = ptk.SceneRecords.SHADOWS
        (row,) = [r for r in handoff["assembly"] if r["data"] == [block]]
        self.assertEqual(row["part"], record.key)
        with MeshConvert.open_glb(path) as edit:
            payload = edit.gltf["extras"][record.key]
        self.assertEqual(row["spec"], record.assembly_for(payload))

    def test_verify_glb_holds_the_files_the_record_names(self):
        """The media gate: a file several views hold is held by none and a
        view stating another media type (or a scale) is unreadable -- both
        fail the file; a map a plane here reads that rides nothing is a
        note. Added: 2026-11-07"""
        path = self._glb(self._v3())
        MeshConvert.apply_glb_shadows(path, search_dirs=[self.maps])
        with MeshConvert.open_glb(path) as edit:
            MeshConvert.embed_glb_handoff(
                edit, ptk.HandoffRecord.block("glb", [], {"application": "maya"})
            )
        where = f"extras.{ptk.SceneRecords.SHADOWS.key}"
        clean = MeshConvert.verify_glb(path)
        self.assertEqual([p for p in clean["problems"] if where in p], [])
        self.assertEqual([n for n in clean["notes"] if where in n], [])

        with MeshConvert.open_glb(path) as edit:
            views = edit.gltf["bufferViews"]
            index = _media_views(edit.gltf)["Box_horizon.png"][0]
            views[index]["extras"] = {"mimeType": "image/png", "scale": 2.0}
            atlas = _media_views(edit.gltf)["shadow_atlas_projected.png"][0]
            views.append(dict(views[atlas], name="SHADOW_ATLAS_PROJECTED.png"))
            own = _media_views(edit.gltf)["Box_shadow.png"][0]
            views[own]["name"] = "renamed.png"
            edit.dirty = True
        report = MeshConvert.verify_glb(path)
        problems = "\n".join(report["problems"])
        self.assertIn("the view of Box_horizon.png states 'image/png' with a scale", problems)
        self.assertIn("2 bufferViews hold shadow_atlas_projected.png", problems)
        self.assertTrue(
            any("Box_shadow.png" in note for note in report["notes"]), report["notes"]
        )

    def test_fbx_to_glb_carries_the_record_and_rides_the_maps_beside_the_fbx(self):
        """The registration, end to end, with the converter replaced by a
        copy: the record in the root extras, the horizon map found in the
        Maya project's ``sourceimages`` beside the FBX."""
        project = self.temp.dir_path()
        os.makedirs(os.path.join(project, "sourceimages"))
        shutil.copy(
            os.path.join(self.maps, "Box_horizon.png"),
            os.path.join(project, "sourceimages", "Box_horizon.png"),
        )
        fixture = self._glb(self._v3())
        src = os.path.join(project, "scene.fbx")
        with open(src, "wb") as fh:
            fh.write(b"Kaydara FBX Binary  \x00")

        def fake_run(cmd, **kwargs):
            base = cmd[cmd.index("-o") + 1]
            shutil.copy(fixture, base + ".glb")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with (
            patch.object(MeshConvert, "resolve_binary", return_value="FBX2glTF"),
            patch(f"{FBX2GLTF_MODULE}.subprocess.run", side_effect=fake_run),
        ):
            out = MeshConvert.fbx_to_glb(src, timeout=60, lightmaps=False)
        gltf = self._read(out)
        self.assertEqual(gltf["extras"][ptk.SceneRecords.SHADOWS.key], self._v3())
        self.assertIn("Box_horizon.png", _media_views(gltf))

    def test_a_glb_built_from_a_finished_fbx_rides_the_maps_it_embeds(self):
        """A delivered FBX ships nothing beside it (every map the record names
        embedded, ``FbxMedia.DEPENDENCY_FIELDS``): a GLB built from it reads
        each file out of the FBX itself, with no folder searched.
        Added: 2026-11-07"""
        from test_fbx_media import build_carrier_fbx

        payload = self._v3()
        project = self.temp.dir_path()
        src = build_carrier_fbx(
            os.path.join(project, "delivered.fbx"),
            {ptk.SceneRecords.SHADOWS.key: payload},
        )
        embed_dir = os.path.join(project, "to_embed")
        os.makedirs(embed_dir)
        for name in ("Box_horizon.png", "shadow_atlas_projected.png"):
            shutil.copy(os.path.join(self.maps, name), os.path.join(embed_dir, name))
        ptk.FbxMedia.embed(
            src,
            files=[
                os.path.join(embed_dir, n)
                for n in ("Box_horizon.png", "shadow_atlas_projected.png")
            ],
        )
        shutil.rmtree(embed_dir)
        # Other bytes of the same names beside the FBX and in its project:
        # the FBX's own copies still ride (they are what it delivers).
        stale = _silhouette_png(4)
        os.makedirs(os.path.join(project, "sourceimages"))
        for folder in (project, os.path.join(project, "sourceimages")):
            for name in ("Box_horizon.png", "shadow_atlas_projected.png"):
                with open(os.path.join(folder, name), "wb") as fh:
                    fh.write(stale)
        fixture = self._glb(payload)

        def fake_run(cmd, **kwargs):
            base = cmd[cmd.index("-o") + 1]
            shutil.copy(fixture, base + ".glb")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with (
            patch.object(MeshConvert, "resolve_binary", return_value="FBX2glTF"),
            patch(f"{FBX2GLTF_MODULE}.subprocess.run", side_effect=fake_run),
        ):
            out = MeshConvert.fbx_to_glb(src, timeout=60, lightmaps=False)
        gltf = self._read(out)
        views = _media_views(gltf)
        self.assertEqual(self._payload_of(out, views["Box_horizon.png"][0]), self.horizon_png)
        self.assertEqual(
            self._payload_of(out, views["shadow_atlas_projected.png"][0]), self.atlas_png
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
            ["features/turntable.js", "features/shadow_rig/shadow_rig.js"],
        )
        # Served with the model beside it, which its adapter imports.
        for module in ("shadow_rig.js", "model.js"):
            self.assertTrue(
                (Path(self.server.root) / "features" / "shadow_rig" / module).is_file()
            )

    def test_a_plain_deliverable_activates_nothing(self):
        self.server.publish(self._glb(False))
        self.assertEqual(self.server.scripts, ())

    def test_only_the_records_key_activates_it(self):
        """The script is activated on the record's own key alone: a block a
        file holds under any other -- the one it had before the 2026-10-20
        rename too -- is no block of this reader's, and the file is
        re-exported."""
        path = self._glb(True)
        with MeshConvert.open_glb(path) as edit:
            extras = edit.gltf["extras"]
            extras["shadow_web"] = extras.pop(ptk.SceneRecords.SHADOWS.key)
            edit.dirty = True
        self.server.publish(path)
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
    """The image stack AND a drivable browser (see conftest)."""
    return np is not None and Image is not None and browser_runtime_available()


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
      // The quad's vertices as drawn, in MODEL space, in the file's order.
      const position = (plane.batch ? plane.batch.mesh : plane.mesh).geometry.getAttribute('position');
      const corners = [];
      for (let i = 0; i < position.count; i += 1) {
        corners.push(new THREE.Vector3().fromBufferAttribute(position, i).applyMatrix4(local).toArray());
      }
      const material = plane.batch ? plane.batch.mesh.material : plane.mesh.material;
      const u = material.uniforms || {};
      const horizon = u.uHorizonMap ? u.uHorizonMap.value : null;
      return {
        name: plane.name,
        node: plane.nodeIndex,
        mode: plane.mode,
        opacity: plane.opacity,
        sourceBound: !!plane.source,
        follow: plane.follow,
        position: [p.x, p.y, p.z],
        yaw: new THREE.Euler().setFromQuaternion(q, 'YXZ').y,
        scale: [s.x, s.y, s.z],
        corners,
        instanced: !!plane.batch,
        batchCount: plane.batch ? plane.batch.mesh.count : 0,
        hidden: !plane.mesh.visible,
        isShader: !!material.isShaderMaterial,
        uMode: u.uMode ? u.uMode.value : null,
        uHorizonScale: u.uHorizonScale ? u.uHorizonScale.value : null,
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
        mapColorSpace: u.uMap && u.uMap.value ? u.uMap.value.colorSpace : null,
        mapWrap: u.uMap && u.uMap.value ? [u.uMap.value.wrapS, u.uMap.value.wrapT] : null,
        mapIsOriginal: !!(u.uMap && u.uMap.value && u.uMap.value === plane.original.map),
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
#: alpha, cleared to alpha 0 first, on a private layer so the grid and the rest
#: of the scene do not blend in. As it stands it reads what the plane DRAWS --
#: its cache, by default; HORIZON_SAMPLES_JS turns the cache off first, so the
#: pins read the shader's own per-pixel answer.
_HORIZON_READBACK_JS = """
      {
        const renderer = viewer.renderer;
        const scene = viewer.scene;
        const plane = session.planes.find((p) => p.mode === 1);
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

#: The shader's own answer per pixel: the cache (on by default) would hand back
#: a bilinear blend of its texels instead.
HORIZON_SAMPLES_JS = (
    """
      session.cache = false;
      await nextFrames(2);
"""
    + _HORIZON_READBACK_JS
)

#: What a viewer sees: the plane drawing its cache, and the cache's state.
HORIZON_CACHED_SAMPLES_JS = (
    """
      await nextFrames(2);
      report.cache = session.planes.filter((p) => p.cache).map((p) => ({
        size: p.cache.size,
        atlas: [session.atlas.target.width, session.atlas.target.height],
        valid: p.cache.valid,
        cached: p.material.uniforms.uCached.value,
      }));
"""
    + _HORIZON_READBACK_JS
)


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
        z_up=False,
        uv_rects=None,
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
            z_up=z_up,
            uv_rects=uv_rects,
        )
        manifest = MeshConvert.apply_glb_shadows(path, search_dirs=list(search_dirs))
        self.assertIsNotNone(manifest, "the fixture itself must bind")
        return path

    def _maps(self, files):
        """A folder holding *files* ``{name: bytes}``, for a fixture's
        ``search_dirs``: the maps the record names, as an export writes them."""
        folder = self.temp.dir_path()
        for name, data in files.items():
            with open(os.path.join(folder, name), "wb") as fh:
                fh.write(data)
        return folder

    def _fade(
        self,
        source=None,
        contact=None,
        *,
        direction=None,
        radius=0.6,
        height=1.2,
        ground=0.0,
        intensity=1.0,
        falloff_power=1.2,
        fade_height=2.4,
        unit_scale=1.0,
        source_ramp=1.0,
    ):
        """``ShadowProjection.fade`` of a plane whose source and contact
        stand at *source* and *contact* (metres): the reference the page's
        fade is held to. The lengths are metres; *unit_scale* is the
        record's and *source_ramp* the plane's (record units)."""
        return ptk.ShadowProjection.fade(
            contact or self.CONTACT,
            None if direction is not None else (source or self.SOURCE),
            ground,
            radius,
            height,
            direction=direction,
            max_stretch=6.0,
            intensity=intensity,
            falloff_power=falloff_power,
            fade_height=fade_height,
            # The plane's source_ramp, in metres.
            source_ramp=source_ramp * unit_scale,
        )

    def _following_glb(self, unit_scale=1.0, source_node=None, source_ramp=1.0):
        scale = 1.0 / unit_scale
        record = _record(
            "Box_shadow",
            ground=0.0,
            radius=0.6 * scale,
            height=1.2 * scale,
            fade_height=2.4 * scale,
            source_size=0.0,
            source_ramp=source_ramp,
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
    def _load(self, glb, action="", points=None, around=None):
        """The page's snapshot of *glb* (PROBE_JS). *around* is
        ``(before, after)``: two scripts' sources registered on either side of
        the shim -- each sees the shim's ``load`` from its side."""
        from playwright.sync_api import sync_playwright

        probe = self.temp.path(extension=".js")
        source = PROBE_JS.replace("__ACTION__", action).replace(
            "__POINTS__", json.dumps(points or [])
        )
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write(source)

        # port=0: the default is the production port, and a real preview tab
        # the user has open would poll this test server.
        server = ptk.PreviewServer(viewer=True, title="shadow-test", port=0)
        server.start()
        # The shim BEFORE the probe, and the page up before the publish -- the
        # production order: the server outlives every push. The page imports
        # the manifest's scripts in order, so with the probe last its presence
        # proves the shim is in, and the push then lands on a page that has it.
        # (A push naming the asset and the scripts together is held for them
        # too: test_preview_viewer_live pins that.) Auto-activation is pinned
        # by the server tests.
        if around:
            server.add_script("before_shim", self._script(around[0]))
        server.add_script("shadow_rig")
        if around:
            server.add_script("after_shim", self._script(around[1]))
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
                page_timing = page.evaluate("() => window.__timing || null")
                browser.close()
        finally:
            server.stop()
        found["console"] = console
        found["scripts"] = scripts
        found["timing"] = page_timing
        self.assertEqual(found["errors"], [], found)
        self.assertTrue(found.get("hasSession"), f"the shim built no session: {found}")
        return found

    def _script(self, source):
        """A viewer script holding *source*, on disk."""
        path = self.temp.path(extension=".js")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(source)
        return path

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
        self,
        source_node,
        *,
        mirror=False,
        source_size=0.0,
        source_angle=0.0,
        z_up=False,
    ):
        """``(hmap, glb)`` -- the post's map on disk (baked: the very bytes the
        page will sample), the map decoding them, and a GLB whose one horizon
        plane samples them.

        *source_node* is the light's glTF node: a ``translation`` for a
        positional source, a ``rotation`` for a directional one. *mirror*
        bakes the post at ``(x, -z)``: a Blender frame, whose ``b`` arrives as
        -Z. *z_up* lays the file out as a Blender push delivers it: the quad
        in its local X/Y and the plane and the contact each in a Z-up frame
        turned upright (-90 deg about X), so the record's frame -- written in
        the file's axes -- is NOT the contact node's own. The source's size (a
        diameter) or angle (radians) goes into the record as the shim reads it.
        """
        _, _, _, pole = _pole_geometry_frame()
        pole = (pole[0], -pole[1]) if mirror else pole
        png, hmap = _pole_map(pole, POLE_RADIUS, POLE_HEIGHT)
        maps = self.temp.dir_path()
        with open(os.path.join(maps, "Box_horizon.png"), "wb") as fh:
            fh.write(png)
        record = _record(
            "Box_horizon_plane",
            draw=False,
            follow_source=False,
            max_stretch=POLE_MAX_STRETCH,
            # No falloff: the plane's fade is exactly 1 (its source a unit and
            # more over its grounded contact), so the pins read the march
            # alone -- the fade has its own (test_the_fade_*).
            falloff_power=0.0,
            source_type="directional" if "rotation" in source_node else "point",
            source_size=source_size,
            source_angle=source_angle,
            horizon=_horizon_block(hmap, frame_b=(0, 0, -1) if mirror else (0, 0, 1)),
        )
        upright = _quat((1, 0, 0), -90.0)
        if z_up:
            plane = {
                "name": "Box_horizon_plane",
                "mesh": 0,
                "rotation": upright,
                "scale": [14.0, 14.0, 1.0],
            }
            contact = {"name": "Box_contact_loc", "rotation": upright}
        else:
            plane = {"name": "Box_horizon_plane", "mesh": 0, "scale": [14.0, 1.0, 14.0]}
            contact = {"name": "Box_contact_loc"}
        glb = self._glb(
            None,
            nodes=[
                plane,
                source_node,
                contact,
                _data_export_node(_payload([record])),
            ],
            materials=[_plane_material("Box_horizon_MAT")],
            search_dirs=[maps],
            z_up=z_up,
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

    def test_atlas_planes_batch_into_one_instanced_mesh_each_at_its_own_fade(self):
        """Planes sharing one atlas -- riding the file as media -- are ONE
        InstancedMesh, each instance on its own tile and at its own fade: the
        model's, every frame, its intensity the ceiling. A plane whose
        material a v2 file's fade ramp still animates batches too: a v3
        reader reads no fade off the file (the DCC bakes none).
        Added: 2026-11-07"""
        rects = {
            "A_shadow": [0.5, 0.5, 0.0, 0.0],
            "B_shadow": [0.5, 0.5, 0.5, 0.0],
            "C_shadow": [0.5, 0.5, 0.5, 0.5],
        }
        intensities = {"A_shadow": 0.9, "B_shadow": 0.6, "C_shadow": 0.3}
        records = [
            _record(
                name,
                atlas={"texture": "shadow_atlas_projected.png", "rect": rect},
                intensity=intensities[name],
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
            # Packed as the DCC packs them: each quad's UVs remapped into the
            # tile its record names.
            uv_rects=[ImgUtils.flip_rect_v(rect) for rect in rects.values()],
            search_dirs=[self._maps({"shadow_atlas_projected.png": _silhouette_png(16)})],
        )
        moved = [-1.0, 1.5, 1.2]
        found = self._load(
            glb,
            action=(
                "viewer.mixer.timeScale = 1; viewer.mixer.setTime(1.0); viewer.mixer.timeScale = 0;"
                f"session.planes[0].source.position.set({moved[0]}, {moved[1]}, {moved[2]});"
            ),
        )
        by_name = {p["name"]: p for p in found["before"]}
        self.assertEqual(found["instancedMeshes"], 1)
        after = {p["name"]: p for p in found["after"]}
        for name, plane in by_name.items():
            self.assertTrue(plane["instanced"], plane)
            self.assertEqual(plane["batchCount"], 3)
            self.assertTrue(plane["hidden"])
            self.assertEqual(plane["mode"], 0)
            # Each plane on its own tile: its quad's UV bounds.
            for got, want in zip(plane["uRect"], ImgUtils.flip_rect_v(rects[name])):
                self.assertAlmostEqual(got, want, places=6, msg=name)
            self._assert_placed(plane, self._expected(), places=4)
            # The fade carries the intensity, once.
            self.assertEqual(plane["uIntensity"], 1.0, plane)
            want = self._fade(intensity=intensities[name])
            self.assertLess(want, intensities[name], "the fixture must stretch")
            self.assertAlmostEqual(plane["uOpacity"], want, places=6, msg=name)
            moved_fade = self._fade(source=moved, intensity=intensities[name])
            self.assertAlmostEqual(after[name]["uOpacity"], moved_fade, places=6, msg=name)
        self.assertNotAlmostEqual(
            after["A_shadow"]["uOpacity"], by_name["A_shadow"]["uOpacity"], places=3
        )
        # The atlas the record names is on the batch material, off its view.
        self.assertTrue(by_name["A_shadow"]["mapBound"])
        self.assertEqual(by_name["A_shadow"]["mapImage"], [16, 16])

    def test_the_fade_is_the_models_and_nothing_of_the_material_is_read(self):
        """A plane's alpha is its shadow times the fade the record states,
        evaluated off the placement every frame -- whether or not it follows
        its source -- and never dimmed by its material's alpha (a v2 rule: the
        DCC's baked opacity once rode there). A plane whose source the file
        does not carry draws at its intensity. Added: 2026-11-07"""
        records = [
            _record("Plain_shadow", intensity=0.9),
            _record("Stated_shadow", intensity=0.9, follow_source=False),
            _record("Lost_shadow", intensity=0.7, source="nowhere"),
        ]
        glb = self._glb(
            None,
            nodes=[
                {"name": "Plain_shadow", "mesh": 0},
                {"name": "Stated_shadow", "mesh": 1},
                {"name": "Lost_shadow", "mesh": 2},
                {"name": "shadow_source", "translation": self.SOURCE},
                {"name": "Box_contact_loc", "translation": self.CONTACT},
                _data_export_node(_payload(records)),
            ],
            materials=[
                _plane_material("Plain_MAT", texture=0),
                _plane_material("Stated_MAT", texture=1, alpha=0.45),
                _plane_material("Lost_MAT", texture=2),
            ],
            # One map each (the material's own): no batch, each plane its
            # own uniforms.
            images=[
                {"name": f"map{i}", "uri": _data_uri(_silhouette_png(4 + 2 * i))}
                for i in range(3)
            ],
            textures=[{"source": i, "sampler": 0} for i in range(3)],
        )
        found = self._load(glb)
        by_name = {p["name"]: p for p in found["before"]}
        for name in ("Plain_shadow", "Stated_shadow"):
            plane = by_name[name]
            self.assertFalse(plane["instanced"], plane)
            self.assertAlmostEqual(plane["uOpacity"], self._fade(intensity=0.9), places=6)
            self.assertEqual(plane["uIntensity"], 1.0)
        self.assertTrue(by_name["Plain_shadow"]["follow"])
        self.assertFalse(by_name["Stated_shadow"]["follow"])
        lost = by_name["Lost_shadow"]
        self.assertFalse(lost["sourceBound"])
        self.assertAlmostEqual(lost["uOpacity"], 0.7, places=6)
        # Said once, naming what it lost: a selection export that left the
        # source behind is seen, not read as a flat shadow.
        said = [c for c in found["console"] if "does not resolve" in c]
        self.assertEqual(len(said), 1, found["console"])
        self.assertIn("Lost_shadow", said[0])
        self.assertIn("source nowhere", said[0])
        self.assertIn("draws at its intensity", said[0])

    def test_the_fade_ramps_over_the_planes_source_ramp(self):
        """The source-height term ramps in over the plane's ``source_ramp``
        (record units, times ``unit_scale``): the ramp its producer previews
        -- one centimetre of a Maya cm rig, whether its record is in cm
        (1 x 0.01) or exported in metres (0.01 x 1), one metre of Blender's --
        so the same source half a centimetre over the contact fades a cm
        rig's plane to half and a metre rig's to almost nothing.
        Added: 2026-11-07. REGRESSION (2026-11-08): the page ramped over one
        unit of the record, which a cm rig exported in metres read as a metre.
        """
        low = [self.SOURCE[0], self.CONTACT[1] + 0.005, self.SOURCE[2]]
        for unit_scale, source_ramp in ((0.01, 1.0), (1.0, 0.01), (1.0, 1.0)):
            with self.subTest(unit_scale=unit_scale, source_ramp=source_ramp):
                found = self._load(
                    self._following_glb(
                        unit_scale=unit_scale,
                        source_node={"name": "shadow_source", "translation": low},
                        source_ramp=source_ramp,
                    )
                )
                (plane,) = found["before"]
                want = self._fade(
                    source=low,
                    fade_height=2.4,
                    unit_scale=unit_scale,
                    source_ramp=source_ramp,
                )
                self.assertAlmostEqual(plane["uOpacity"], want, places=6)
        self.assertGreater(
            self._fade(source=low, source_ramp=0.01),
            50 * self._fade(source=low, source_ramp=1.0),
        )

    def test_a_rig_draws_its_silhouette_where_it_says_and_marches_otherwise(self):
        """The mode, per plane: the silhouette where the rig draws it and the
        file holds it; else the horizon march; else -- a horizon map the file
        lacks -- the silhouette, the fallback; a drawn silhouette the file
        lacks marches the horizon, as a reader without the extension does.
        Added: 2026-11-07"""
        png, hmap = _pole_map((0.5, 0.0), POLE_RADIUS, POLE_HEIGHT)
        layout = _horizon_block(hmap)
        records = [
            _record("Drawn_shadow", draw=True, horizon=layout),
            _record("Marched_shadow", draw=False, horizon=layout),
            _record(
                "Fallback_shadow",
                draw=False,
                horizon=dict(layout, texture="Missing_horizon.png"),
            ),
            _record("Unshipped_shadow", draw=True, horizon=layout),
        ]
        names = [r["name"] for r in records]
        maps = self._maps(
            {
                "Box_horizon.png": png,
                "Drawn_shadow.png": _silhouette_png(4),
                "Marched_shadow.png": _silhouette_png(6),
                "Fallback_shadow.png": _silhouette_png(8),
            }
        )
        glb = self._glb(
            None,
            nodes=[{"name": n, "mesh": i, "scale": [4.0, 1.0, 4.0]} for i, n in enumerate(names)]
            + [
                {"name": "shadow_source", "translation": self.SOURCE},
                {"name": "Box_contact_loc"},
                _data_export_node(_payload(records)),
            ],
            materials=[_bare_material(f"{n}_MAT") for n in names],
            search_dirs=[maps],
        )
        found = self._load(glb)
        by_name = {p["name"]: p for p in found["before"]}
        self.assertEqual(
            {n: by_name[n]["mode"] for n in names},
            {"Drawn_shadow": 0, "Marched_shadow": 1, "Fallback_shadow": 0, "Unshipped_shadow": 1},
        )
        self.assertEqual(by_name["Drawn_shadow"]["mapImage"], [4, 4])
        self.assertEqual(by_name["Fallback_shadow"]["mapImage"], [8, 8])
        for name in ("Marched_shadow", "Unshipped_shadow"):
            self.assertTrue(by_name[name]["horizonBound"], name)
            self.assertFalse(by_name[name]["horizonMipmaps"], name)
            self.assertTrue(by_name[name]["horizonWrap"], name)

    def test_the_colour_map_is_the_atlas_then_the_material_then_the_planes_own(self):
        """A silhouette plane's colour map, in the order the page has always
        looked: the atlas once packed, else the plane's material's own map
        (read through the plane's rect: on a packed plane it is the atlas),
        else the plane's own image -- read whole, its rect the identity
        whatever its quad's packed UVs say (it is the full canvas, never a
        tile). A map from the file is colour (sRGB) sampled as glTF's default
        sampler samples it (repeat). Added: 2026-11-08 (HEAD's chain and
        sampler, restored)."""
        tile = [0.5, 0.5, 0.5, 0.0]
        records = [
            _record(
                "Atlas_shadow",
                atlas={"texture": "shadow_atlas_projected.png", "rect": tile},
            ),
            _record("Material_shadow"),
            _record(
                "Own_shadow",
                atlas={"texture": "Nowhere_atlas.png", "rect": tile},
            ),
        ]
        names = [r["name"] for r in records]
        glb = self._glb(
            None,
            nodes=[{"name": n, "mesh": i} for i, n in enumerate(names)]
            + [
                {"name": "shadow_source", "translation": self.SOURCE},
                {"name": "Box_contact_loc", "translation": self.CONTACT},
                _data_export_node(_payload(records)),
            ],
            materials=[
                _bare_material("Atlas_MAT"),
                _plane_material("Material_MAT"),
                _bare_material("Own_MAT"),
            ],
            # The atlas plane and the own-image plane packed into a tile.
            uv_rects=[ImgUtils.flip_rect_v(tile), None, ImgUtils.flip_rect_v(tile)],
            search_dirs=[
                self._maps(
                    {
                        "shadow_atlas_projected.png": _silhouette_png(16),
                        "Material_shadow.png": _silhouette_png(4),
                        "Own_shadow.png": _silhouette_png(6),
                    }
                )
            ],
        )
        found = self._load(glb)
        by_name = {p["name"]: p for p in found["before"]}
        repeat = [1000, 1000]  # THREE.RepeatWrapping
        atlas = by_name["Atlas_shadow"]
        self.assertEqual(atlas["mapImage"], [16, 16])
        for got, want in zip(atlas["uRect"], ImgUtils.flip_rect_v(tile)):
            self.assertAlmostEqual(got, want, places=6)
        material = by_name["Material_shadow"]
        self.assertTrue(material["mapIsOriginal"], material)
        self.assertEqual(material["mapImage"], [8, 8])
        own = by_name["Own_shadow"]
        self.assertFalse(own["mapIsOriginal"], own)
        self.assertEqual(own["mapImage"], [6, 6])
        self.assertEqual(own["uRect"], [1, 1, 0, 0])
        for name in names:
            self.assertEqual(by_name[name]["mapColorSpace"], "srgb", name)
            self.assertEqual(by_name[name]["mapWrap"], repeat, name)

    def test_a_horizon_map_packed_in_an_atlas_is_read_from_its_own_tile(self):
        """A horizon map is packed into its image at the record's rect, origin
        bottom-left; the page flips it to glTF's top-left before it fetches a
        texel. Two maps stacked in one image, the plane's the bottom tile
        (rect ``[1, 0.5, 0, 0]``) and an empty one over it: the page draws the
        bottom tile's ``HeightFieldMap.alpha``. Added: 2026-11-08"""
        azimuth, away, perp, pole = _pole_geometry_frame()
        source = [2.0 * math.cos(azimuth), 3.0, 2.0 * math.sin(azimuth)]
        points = [
            (pole[0] + 0.5 * away[0], pole[1] + 0.5 * away[1]),
            (pole[0] + 2.5 * away[0], pole[1] + 2.5 * away[1]),
            (pole[0] + 0.5 * away[0] + 0.35 * perp[0], pole[1] + 0.5 * away[1] + 0.35 * perp[1]),
        ]
        _, hmap = _pole_map(pole, POLE_RADIUS, POLE_HEIGHT)
        tile = hmap.to_rgba()
        packed = _png_bytes(np.concatenate([np.zeros_like(tile), tile], axis=0))
        maps = self._maps({"Box_horizon.png": packed})
        record = _record(
            "Box_horizon_plane",
            draw=False,
            follow_source=False,
            max_stretch=POLE_MAX_STRETCH,
            falloff_power=0.0,
            horizon=_horizon_block(hmap, rect=[1.0, 0.5, 0.0, 0.0]),
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
        expected = hmap.alpha([[x, 0.0, z] for x, z in points], light=source)
        self.assertEqual(round(float(expected[0])), 1, "the fixture puts its point in shadow")
        found = self._load(glb, action=HORIZON_SAMPLES_JS, points=points)
        (plane,) = found["before"]
        self.assertEqual(plane["uMode"], 1)
        for got, want in zip(found["samples"], expected):
            self.assertAlmostEqual(got, float(want), delta=0.05, msg=found["samples"])

    def test_names_resolve_against_one_index_of_the_file(self):
        """Each source and contact name is resolved alone, by the record rule
        over every node -- against ONE index of the file, built once: a
        file of 30k nodes and 200 planes builds its session in well under a
        second (measured 13.7 s while every name re-indexed the file).
        Added: 2026-11-08"""
        count, filler = 200, 30000
        nodes = [{"name": f"n{i}"} for i in range(filler)]
        for i in range(1, filler):
            nodes[(i - 1) // 4].setdefault("children", []).append(i)
        records = []
        for p in range(count):
            nodes += [
                {"name": f"P{p}_shadow", "mesh": 0},
                {"name": f"S{p}_source", "translation": self.SOURCE},
                {"name": f"C{p}_contact", "translation": self.CONTACT},
            ]
            records.append(
                _record(f"P{p}_shadow", source=f"S{p}_source", contact=f"C{p}_contact")
            )
        nodes.append(_data_export_node(_payload(records)))
        glb = self._glb(None, nodes=nodes, materials=[_plane_material("m")])
        start = "export default function t(viewer) { viewer.on('load', () => { window.__timing = { start: performance.now() }; }); }"
        end = "export default function t(viewer) { viewer.on('load', () => { window.__timing.built = performance.now() - window.__timing.start; }); }"
        found = self._load(glb, around=(start, end))
        self.assertEqual(len(found["before"]), count)
        self.assertTrue(all(p["follow"] for p in found["before"]))
        self.assertLess(found["timing"]["built"], 3000.0, found["timing"])

    def test_a_map_the_page_cannot_read_is_said_and_the_session_still_comes_up(self):
        """A horizon map whose bytes do not decode is said and left unbound
        (HEAD's page rejected the whole session); a silhouette two views hold
        is held by neither -- said, as ``verify_glb`` says it -- and its plane
        marches its horizon, as a reader without the extension does.
        Added: 2026-11-07"""
        png, hmap = _pole_map((0.5, 0.0), POLE_RADIUS, POLE_HEIGHT)
        layout = _horizon_block(hmap)
        records = [
            _record("Broken_shadow", draw=False, horizon=dict(layout, texture="Broken_horizon.png")),
            _record("Tied_shadow", draw=True, horizon=layout),
        ]
        maps = self._maps(
            {
                "Broken_horizon.png": b"not a PNG at all" * 8,
                "Box_horizon.png": png,
                "Tied_shadow.png": _silhouette_png(4),
            }
        )
        glb = self._glb(
            None,
            nodes=[
                {"name": "Broken_shadow", "mesh": 0},
                {"name": "Tied_shadow", "mesh": 1},
                {"name": "shadow_source", "translation": self.SOURCE},
                {"name": "Box_contact_loc"},
                _data_export_node(_payload(records)),
            ],
            materials=[_bare_material("a"), _bare_material("b")],
            search_dirs=[maps],
        )
        with MeshConvert.open_glb(glb) as edit:
            views = edit.gltf["bufferViews"]
            (index,) = ptk.SceneRecords.GLB_MEDIA.views(edit.gltf, "Tied_shadow.png")
            views.append(dict(views[index], name="TIED_SHADOW.png"))
            edit.dirty = True
        found = self._load(glb)
        by_name = {p["name"]: p for p in found["before"]}
        self.assertEqual(by_name["Broken_shadow"]["mode"], 1)
        self.assertFalse(by_name["Broken_shadow"]["horizonBound"])
        self.assertEqual(by_name["Tied_shadow"]["mode"], 1)
        self.assertTrue(by_name["Tied_shadow"]["horizonBound"])
        console = "\n".join(found["console"])
        self.assertIn("could not be read", console)
        self.assertIn("held by several bufferViews", console)
        self.assertIn("Tied_shadow.png", console)

    def test_planes_sources_and_contacts_resolve_by_the_record_rule(self):
        """Names resolve as the lightmaps record's do (resolve.js): exact,
        else the name after the last ':' where one node carries it, a tie
        never guessed. A source that is no mesh resolves over every node; a
        bare plane name two namespaced planes share is left as the file has
        it; a source two nodes share leaves its plane still, at its
        intensity. Added: 2026-11-07"""
        records = [
            _record("A:Box_shadow", source="shadow_source", contact="Box_contact_loc"),
            _record("Crate_shadow", source="twin", intensity=0.8),
            _record("Shared_shadow"),
        ]
        glb = self._glb(
            None,
            nodes=[
                {"name": "A:Box_shadow", "mesh": 0},
                {"name": "Crate_shadow", "mesh": 1},
                {"name": "X:Shared_shadow", "mesh": 2},
                {"name": "Y:Shared_shadow", "mesh": 2},
                {"name": "NS:shadow_source", "translation": self.SOURCE},
                {"name": "NS:Box_contact_loc", "translation": self.CONTACT},
                {"name": "P:twin", "translation": self.SOURCE},
                {"name": "Q:twin", "translation": self.SOURCE},
                _data_export_node(_payload(records)),
            ],
            materials=[_plane_material(f"m{i}") for i in range(3)],
        )
        found = self._load(glb)
        by_name = {p["name"]: p for p in found["before"]}
        self.assertEqual(set(by_name), {"A:Box_shadow", "Crate_shadow"})
        box = by_name["A:Box_shadow"]
        self.assertTrue(box["follow"], box)
        self._assert_placed(box, self._expected(), places=4)
        crate = by_name["Crate_shadow"]
        self.assertFalse(crate["sourceBound"], crate)
        self.assertFalse(crate["follow"], crate)
        self.assertAlmostEqual(crate["uOpacity"], 0.8, places=6)
        self.assertTrue(
            any("ambiguous" in line and "twin" in line for line in found["console"]),
            found["console"],
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
            # Each quad's UVs remapped into its tile, as the DCC packs it.
            uv_rects=[ImgUtils.flip_rect_v(rect) for rect in rects.values()],
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
        """The pin: the rendered pixels ARE `HeightFieldMap.alpha`.

        The contract names that method the oracle both engine shaders answer
        to, so this test asks the real page, through the real WebGL2 shader
        built from the shared body, for the alpha at a set of ground points and
        compares each against the reference decoding the SAME map.

        The fixture is a post whose right answer is known in closed form, so
        the reference is checked against physics in the same pass. Pinning the shader to the reference alone would pass on
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
            "far away": (pole[0] + 4.5 * away[0], pole[1] + 4.5 * away[1]),
        }
        # What decides each point, and by how much: the map is a texel grid,
        # so a point must sit clear of the boundary that decides it. The
        # shadowed point has to be clear of ALL of them.
        deciders = {
            "in the shadow": ("azimuth", "elevation"),
            "past the tip": ("elevation",),
            "off bearing": ("azimuth",),
            "far away": ("elevation",),
        }
        analytic = {}
        for label, point in points.items():
            analytic[label] = _pole_alpha(point, source, pole, POLE_RADIUS, POLE_HEIGHT)
            margins = _pole_margins(point, source, pole, POLE_RADIUS, POLE_HEIGHT)
            for decider in deciders[label]:
                self.assertGreater(margins[decider], 0.08, (label, decider, margins))
        self.assertEqual(analytic["in the shadow"], 1.0)
        self.assertEqual(analytic["past the tip"], 0.0)
        self.assertEqual(analytic["off bearing"], 0.0)
        self.assertEqual(analytic["far away"], 0.0)

        hmap, glb = self._pole_fixture({"name": "shadow_source", "translation": source})
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
        self.assertEqual(plane["uOpacity"], 1.0, "the fixture's fade is exactly 1")
        self.assertAlmostEqual(plane["uHorizonScale"], hmap.height_scale, places=5)
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
    @staticmethod
    def _scatter_points(evaluate, along_axis=None):
        """Ground points down the post's shadow and across it, plus points
        measured to sit ON the penumbra -- found on the reference itself, per
        fixture: across the shadow at five ranges, where the alpha falls from
        0.9 to 0.1, four points along that fall on both sides. The
        penumbra is where the shader has to reproduce the march's clearance
        term and not just its hit test; ``evaluate(points)`` is the
        reference's alpha at ``(x, z)`` points, and ``_assert_tracks_reference``
        fails the run if no point lands there.

        *along_axis* is the direction the shadow actually runs, defaulting to
        the positional fixture's. A directional source on a different bearing
        throws its shadow along a different line.
        """
        _, away, perp, pole = _pole_geometry_frame()
        away = tuple(away if along_axis is None else along_axis)
        perp = (-away[1], away[0])

        def at(along, across):
            return (
                pole[0] + along * away[0] + across * perp[0],
                pole[1] + along * away[1] + across * perp[1],
            )

        grid = [
            at(along, across)
            for along in (0.1, 0.35, 0.7, 1.1, 1.7, 2.4)
            for across in (-0.3, -0.11, 0.0, 0.11, 0.3)
        ]
        edges = []
        for along in (0.5, 0.9, 1.3, 1.7, 2.1):
            across = [0.005 * i for i in range(0, 161)]
            alpha = evaluate([at(along, a) for a in across])
            a0 = next((a for a, v in zip(across, alpha) if v < 0.9), None)
            a1 = next((a for a, v in zip(across, alpha) if v < 0.1), None)
            if a0 is None or a1 is None or a1 <= a0:
                continue
            for f in (0.2, 0.4, 0.6, 0.8):
                for sign in (-1.0, 1.0):
                    edges.append(at(along, sign * (a0 + f * (a1 - a0))))
        return grid + edges

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
        tip -- and requires the page to agree with `HeightFieldMap.alpha` at every
        one.
        """
        azimuth, _, _, _ = _pole_geometry_frame()
        source = [2.0 * math.cos(azimuth), 3.0, 2.0 * math.sin(azimuth)]
        hmap, glb = self._pole_fixture(
            {"name": "shadow_source", "translation": source},
            source_size=POLE_SOURCE_SIZE,
        )

        def evaluate(pts):
            return hmap.alpha(
                [[x, 0.0, z] for x, z in pts],
                light=source,
                source_size=POLE_SOURCE_SIZE,
            )

        points = self._scatter_points(evaluate)
        expected = evaluate(points)
        found = self._load(glb, action=HORIZON_SAMPLES_JS, points=points)
        self._assert_tracks_reference(found, expected, points)

    def test_the_cache_tracks_the_reference_across_a_scatter_of_ground_points(self):
        """The real-time path: the plane samples its shadow marched once into
        its slot of one atlas over its own UVs (three map pixels a texel), so the pin's
        48 points are a bilinear blend of cached texels -- close to the
        reference everywhere and never a different shadow. Measured
        2026-10-10: mean 0.026, worst 0.23, the worst on the 5 cm pole's
        needle-sharp core at its foot, where this fixture's long plane hits
        the slot's 254-texel cap (a softened edge, which the per-pixel path's
        0.05 pin above does not allow)."""
        azimuth, _, _, _ = _pole_geometry_frame()
        source = [2.0 * math.cos(azimuth), 3.0, 2.0 * math.sin(azimuth)]
        hmap, glb = self._pole_fixture(
            {"name": "shadow_source", "translation": source},
            source_size=POLE_SOURCE_SIZE,
        )

        def evaluate(pts):
            return hmap.alpha(
                [[x, 0.0, z] for x, z in pts],
                light=source,
                source_size=POLE_SOURCE_SIZE,
            )

        points = self._scatter_points(evaluate)
        expected = np.asarray(evaluate(points))
        found = self._load(glb, action=HORIZON_CACHED_SAMPLES_JS, points=points)
        (cache,) = found["cache"]
        self.assertTrue(cache["valid"], found)
        self.assertEqual(cache["cached"], 1, "the plane draws its cache")
        self.assertTrue(16 <= min(cache["size"]) and max(cache["size"]) <= 254, cache)
        got = np.asarray(found["samples"])
        self.assertEqual(len(got), len(points), found)
        err = np.abs(got - expected)
        detail = f"cache {cache['size']}, mean {err.mean():.4f}, worst {err.max():.4f}"
        self.assertLess(float(err.mean()), 0.04, detail)
        self.assertLess(float(err.max()), 0.3, detail)

    def test_the_cache_re_marches_only_when_the_light_moves_in_the_contacts_frame(
        self,
    ):
        """The cost model: a still light costs one fetch a pixel, frame after
        frame; the page's turntable spins the model, its light and its plane
        together, which changes nothing in the contact's frame and re-marches
        nothing; a moved source re-marches its plane once."""
        azimuth, _, _, _ = _pole_geometry_frame()
        source = [2.0 * math.cos(azimuth), 3.0, 2.0 * math.sin(azimuth)]
        _, glb = self._pole_fixture(
            {"name": "shadow_source", "translation": source},
            source_size=POLE_SOURCE_SIZE,
        )
        action = """
      await nextFrames(3);
      const bakes = [session.cacheStats.bakes];
      await nextFrames(5);
      bakes.push(session.cacheStats.bakes);
      viewer.pivot.rotation.y += 0.7;
      viewer.pivot.updateMatrixWorld(true);
      await nextFrames(4);
      bakes.push(session.cacheStats.bakes);
      session.planes[0].source.position.x += 0.3;
      await nextFrames(4);
      bakes.push(session.cacheStats.bakes);
      report.bakes = bakes;
"""
        found = self._load(glb, action=action)
        self.assertFalse(found.get("errors"), found.get("errors"))
        still, held, spun, moved = found["bakes"]
        self.assertGreaterEqual(still, 1, "the first frame marched the cache")
        self.assertEqual(held, still, "a still light re-marched")
        self.assertEqual(spun, held, "a turntable spin re-marched")
        self.assertEqual(moved, spun + 1, "a moved source re-marches once")

    def test_a_raised_model_keeps_its_horizon_shadow(self):
        """The horizon's ground is a MODEL-space height, like the placement's.

        The page stands a model on its lowest point, which need not be the
        ground (anything below it -- a pit, a plinth's footing -- lifts the
        whole model), and a viewer script may move the model on top of that.
        The shader works in world space, so the ground has to come through the
        model's world matrix; read as world Y it sat 0.75 m below the raised
        model's ground and every ray started that far low, lengthening every
        shadow. The samples are taken at the same model-space points, so the
        reference is unchanged.
        """
        azimuth, _, _, _ = _pole_geometry_frame()
        source = [2.0 * math.cos(azimuth), 3.0, 2.0 * math.sin(azimuth)]
        hmap, glb = self._pole_fixture(
            {"name": "shadow_source", "translation": source},
            source_size=POLE_SOURCE_SIZE,
        )

        def evaluate(pts):
            return hmap.alpha(
                [[x, 0.0, z] for x, z in pts],
                light=source,
                source_size=POLE_SOURCE_SIZE,
            )

        points = self._scatter_points(evaluate)
        expected = evaluate(points)
        raise_model = "model.position.y += 0.75; model.updateMatrixWorld(true);"
        found = self._load(glb, action=raise_model + HORIZON_SAMPLES_JS, points=points)
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
        # And a little off the bearing the positional fixture uses, so no
        # sample sits on a texel edge the GPU's float32 and numpy's float64
        # could round either way.
        bearing = azimuth + 0.3 * (TWO_PI / 32.0)
        elevation = math.radians(20.0)
        shine = [
            -math.cos(bearing) * math.cos(elevation),
            -math.sin(elevation),
            -math.sin(bearing) * math.cos(elevation),
        ]
        self.assertLess(shine[1], 0.0, "the fixture must shine downward")

        along = (math.cos(bearing - math.pi), math.sin(bearing - math.pi))
        hmap, glb = self._pole_fixture(
            {
                "name": "shadow_source",
                "translation": [
                    9.0,
                    9.0,
                    9.0,
                ],  # a directional source's position is inert
                "rotation": _quat_from_minus_y(shine),
            },
            source_angle=POLE_SOURCE_ANGLE,
        )

        def evaluate(pts):
            return hmap.alpha(
                [[x, 0.0, z] for x, z in pts],
                direction=shine,
                source_angle=POLE_SOURCE_ANGLE,
            )

        points = self._scatter_points(evaluate, along_axis=along)
        expected = evaluate(points)
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

        The map is baked in that mirrored sense (the post at map coords
        ``(x, -z)``) and the reference is fed frame-mapped points, exactly
        as the shader maps world points through A and B.
        """
        self._assert_blender_frame_tracks_reference(z_up=False)

    def test_a_z_up_contact_reads_its_frame_in_the_files_axes(self):
        """A Blender push delivers the contact in its own Z-up frame: the FBX
        exporter turns only the ROOT nodes upright, so a contact's local +Y is
        the file's -Z and its +Z the file's up (measured on blendertk pushes,
        a contact at the root and under one alike). The record's frame is
        written in the file's axes, so the shim reads a Z-up contact's frame
        through the turn back (+90 deg about X) -- told from the plane's quad,
        which lies in its local X/Y. Before: the frame's up was the contact's
        +Y, lying flat along -Z, and ``b`` pointed straight down."""
        self._assert_blender_frame_tracks_reference(z_up=True)

    def _assert_blender_frame_tracks_reference(self, z_up):
        azimuth, _, _, _ = _pole_geometry_frame()
        # World bearing of the light, and the pole's WORLD position; in the
        # (X, -Z) frame the map coords are (x, -z) for both.
        source = [2.0 * math.cos(azimuth), 3.0, 2.0 * math.sin(azimuth)]
        to_frame = lambda x, z: (x, -z)  # noqa: E731 -- dot(P, A), dot(P, B)
        hmap, glb = self._pole_fixture(
            {"name": "shadow_source", "translation": source},
            mirror=True,
            source_size=POLE_SOURCE_SIZE,
            z_up=z_up,
        )
        frame_light = [source[0], source[1], -source[2]]

        def evaluate(pts):
            return hmap.alpha(
                [[fx, 0.0, fz] for fx, fz in (to_frame(x, z) for x, z in pts)],
                light=frame_light,
                source_size=POLE_SOURCE_SIZE,
            )

        points = self._scatter_points(evaluate)
        expected = evaluate(points)
        self.assertGreater(int((expected > 0.5).sum()), 4)
        found = self._load(glb, action=HORIZON_SAMPLES_JS, points=points)
        self._assert_tracks_reference(found, expected, points)

    @staticmethod
    def _blender_corners(source, contact, canvas, radius, height, ground=0.0):
        """The Z-up quad's corners, in the file's order and axes, where
        blendertk's drivers put them -- derived in BLENDER's frame, apart from
        the page: ``ShadowProjection.model`` with up = Z, the quad's local +X
        across the bearing (``w = (u[1], -u[0])``) and +Y along it, lifted
        GROUND_OFFSET; then the exporter's turn, ``(x, y, z) -> (x, z, -y)``.
        """
        model = ptk.ShadowProjection.model(
            (contact[0], -contact[2], contact[1]),
            light=(source[0], -source[2], source[1]),
            ground=ground,
            radius=radius,
            height=height,
            up=2,
            max_stretch=6.0,
        )
        (cx, cy), along, across = model.placement(canvas)
        ux, uy = model.bearing
        wx, wy = model.across
        corners = []
        for vx, vy in ((0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)):
            x = cx + vx * across * wx + vy * along * ux
            y = cy + vx * across * wy + vy * along * uy
            corners.append([x, ground + 0.01, -y])
        return corners

    def test_each_producers_canvas_sides_land_where_its_own_rig_put_them(self):
        """One scene, one lopsided canvas, rigged in both DCCs: the page puts
        each producer's quad on the corners its own rig gave it, measured --
        mayatk's expression (Maya 2025, cm, Y-up: the quad in its local X/Z)
        and blendertk's drivers (Blender 5.1, m, Z-up: the quad in its local
        X/Y, the file in its exporter's axes). The two rigs write the canvas's
        sides along the quad's local +X, each in its own DCC's ground axes, so
        the same record lands mirrored across the bearing between them (here
        0.34 m apart): the page reads a Y-up quad's sides as written and a
        Z-up quad's swapped, told off the quad itself (``quadExtent``) -- as
        unitytk's importer does in Unity's mirrored frame. Added: 2026-11-08"""
        canvas = [-0.3, 1.1, -0.55, 0.35]
        # Maya: the rig's world corners (cm), its source and contact (cm).
        maya_cm = [
            [-298.927425, 0.01, -15.608639],
            [-251.212054, 0.01, 139.466321],
            [77.771017, 0.01, -131.515848],
            [125.486387, 0.01, 23.559113],
        ]
        maya = [[v * 0.01 for v in c] for c in maya_cm]
        # Blender: the rig's world corners, source and contact, Blender's axes.
        to_file = lambda p: [p[0], p[2], -p[1]]  # noqa: E731  the exporter's turn
        blender = [
            to_file(c)
            for c in (
                [1.3608983, -0.5802022, 0.01],
                [0.8837445, 0.9705476, 0.01],
                [-2.8832405, -0.1885246, 0.01],
                [-2.4060867, -1.7392744, 0.01],
            )
        ]
        upright = _quat((1, 0, 0), -90.0)
        cases = {
            "maya": (
                maya,
                0.01,
                _record(
                    "Box_shadow",
                    radius=36.055513,
                    height=120.0,
                    fade_height=240.0,
                    canvas=canvas,
                ),
                [
                    {"name": "Box_shadow", "mesh": 0},
                    {"name": "shadow_source", "translation": [1.5, 2.0, -0.5]},
                    {"name": "Box_contact_loc", "translation": [0.2, 0.0, -0.1]},
                ],
                False,
            ),
            "blender": (
                blender,
                1.0,
                _record(
                    "Box_shadow",
                    contact="Box_contact",
                    radius=0.360555,
                    height=1.2,
                    canvas=canvas,
                ),
                [
                    {"name": "Box_shadow", "mesh": 0, "rotation": upright},
                    {"name": "shadow_source", "translation": to_file([1.5, 0.5, 2.0])},
                    {
                        "name": "Box_contact",
                        "translation": to_file([0.2, 0.1, 0.0]),
                        "rotation": upright,
                    },
                ],
                True,
            ),
        }

        def gap(a, b):
            return max(min(max(abs(p[i] - q[i]) for i in range(3)) for q in b) for p in a)

        self.assertGreater(gap(maya, blender), 0.3, "the producers' sides must differ")
        for producer, (want, unit_scale, record, nodes, z_up) in cases.items():
            with self.subTest(producer=producer):
                glb = self._glb(
                    None,
                    nodes=nodes + [_data_export_node(_payload([record], unit_scale=unit_scale))],
                    materials=[_plane_material("m")],
                    z_up=z_up,
                )
                (plane,) = self._load(glb)["before"]
                self.assertTrue(plane["follow"], plane)
                self.assertLess(gap(plane["corners"], want), 1e-5, (producer, plane["corners"]))

    def test_a_z_up_quad_lies_flat_where_the_blender_drivers_put_it(self):
        """A Blender push delivers its shadow quad in the quad's local X/Y
        (normal +Z) -- Blender's exporter turns the ROOT nodes upright and
        never the mesh data -- where the shim's yaw-only placement assumed a
        Y-up (Maya) quad in local X/Z. Measured on blendertk pushes (a plane
        under its rig group, deeper, and at the root): every following plane
        stood on edge, its model-space height the quad's own size. The quad
        now lies flat on the corners the Blender drivers give it, the canvas
        read across the right side of the bearing (the canvas is lopsided):
        once on its own mesh, once as an instanced batch."""
        canvas = [-0.3, 1.1, -0.55, 0.35]
        upright = _quat((1, 0, 0), -90.0)
        expected = self._blender_corners(self.SOURCE, self.CONTACT, canvas, 0.6, 1.2)
        for names in (["Box_shadow"], ["Box_shadow", "Crate_shadow"]):
            count = len(names)
            nodes = [
                {
                    "name": "rig_shadow_grp",
                    "rotation": upright,
                    "children": list(range(1, count + 1)),
                }
            ]
            nodes += [
                {
                    "name": name,
                    "mesh": i,
                    "translation": [0.4, -0.3, 0.0],
                    "rotation": _quat((0, 0, 1), 25.0),
                    "scale": [1.5, 2.0, 1.0],
                }
                for i, name in enumerate(names)
            ]
            nodes += [
                {"name": "shadow_source", "translation": self.SOURCE},
                {
                    "name": "Box_contact_loc",
                    "translation": self.CONTACT,
                    "rotation": upright,
                },
                _data_export_node(
                    _payload(
                        [
                            _record(name, radius=0.6, height=1.2, canvas=canvas)
                            for name in names
                        ]
                    )
                ),
            ]
            glb = self._glb(
                None,
                nodes=nodes,
                materials=[_plane_material(f"{name}_MAT") for name in names],
                z_up=True,
            )
            found = self._load(glb)
            self.assertEqual(len(found["before"]), count)
            for plane in found["before"]:
                self.assertTrue(plane["follow"], plane)
                self.assertEqual(plane["instanced"], count > 1, plane)
                for got, want in zip(plane["corners"], expected):
                    for axis in range(3):
                        self.assertAlmostEqual(
                            got[axis], want[axis], places=5, msg=(names, plane)
                        )


#: Every ``shadow_projection`` golden case through the feature's model port,
#: imported by its own served path (as an app vendoring it would). Each function
#: is held to its own inputs: the far point to the reference's, the model to
#: the reference's source point, the placement and the fade to the model it
#: computed.
CONFORMANCE_JS = """
import { farPoint, placement, shadowFade, shadowModel } from '../features/shadow_rig/model.js';

export default function probe() {
  const doc = __DOC__;
  const worst = { source: 0, model: 0, placement: 0, fade: 0 };
  const flags = [];
  const gap = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));
  for (const c of doc.cases) {
    const given = c.input;
    if (given.direction) {
      const far = farPoint(given.contact, given.direction, Math.max(given.height, 2 * given.radius));
      worst.source = Math.max(worst.source, gap(far, c.source));
    }
    const m = shadowModel(given.contact, c.source, given.ground, given.radius, given.height, given.max_stretch);
    const e = c.model;
    worst.model = Math.max(
      worst.model,
      gap(m.anchor, e.anchor),
      gap(m.bearing, e.bearing),
      ...[[m.kBase, e.k_base], [m.kTop, e.k_top], [m.reach, e.reach],
        [m.base, e.base], [m.top, e.top], [m.width, e.width]].map(([a, b]) => Math.abs(a - b)),
    );
    if (m.overhead !== e.overhead) flags.push(`${c.name}: overhead ${m.overhead}`);
    const p = placement(m, given.canvas);
    worst.placement = Math.max(
      worst.placement,
      gap(p.centre, c.placement.centre),
      Math.abs(p.along - c.placement.along),
      Math.abs(p.across - c.placement.across),
    );
    const f = shadowFade(
      m, given.contact, c.source, given.ground, given.radius, given.intensity,
      given.falloff_power, given.fade_height, given.source_ramp,
    );
    worst.fade = Math.max(worst.fade, Math.abs(f - c.fade));
  }
  window.__probe = { ready: true, worst, flags, count: doc.cases.length };
}
"""


@unittest.skipUnless(
    browser_runtime_available(), "needs playwright + an installed Edge channel"
)
class TestShadowModelConformance(unittest.TestCase):
    """The JavaScript port against the reference's golden cases
    (``ptk.Conformance.cases("shadow_projection")``) -- unitytk's C# port runs
    the same document."""

    def test_the_js_model_matches_every_conformance_case(self):
        from playwright.sync_api import sync_playwright

        doc = ptk.Conformance.cases("shadow_projection", seed=2, per_kind=8)
        temp = ptk.TempArtifacts("shadow_conformance", policy="scoped")
        probe = temp.path(extension=".js")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write(CONFORMANCE_JS.replace("__DOC__", json.dumps(doc)))
        server = ptk.PreviewServer(viewer=True, title="shadow-conformance", port=0)
        server.start()
        # Active so its folder -- the model beside the adapter -- is served.
        server.add_script("shadow_rig")
        server.add_script("probe", probe)
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    channel="msedge",
                    headless=True,
                    args=["--enable-unsafe-swiftshader"],
                )
                page = browser.new_page()
                page.goto(server.url, wait_until="domcontentloaded", timeout=120_000)
                page.wait_for_function("() => window.__probe?.ready", timeout=120_000)
                found = page.evaluate("() => window.__probe")
                browser.close()
        finally:
            server.stop()
            temp.cleanup()
        self.assertEqual(found["count"], len(doc["cases"]))
        self.assertEqual(found["flags"], [])
        for quantity, tolerance in doc["tolerance"].items():
            self.assertLessEqual(
                found["worst"][quantity], tolerance, (quantity, found["worst"])
            )


if __name__ == "__main__":
    unittest.main()
