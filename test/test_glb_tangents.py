# !/usr/bin/python
# coding=utf-8
"""GlbTangents -- a GLB's shipped tangents, repaired against the UVs they describe.

glTF's bitangent is ``cross(normal, tangent.xyz) * tangent.w`` and a normal
map's green is image-up, the direction of DECREASING V; FBX2glTF ships
``w = +1`` everywhere, so a mirrored UV shell reads its normal map with green
inverted, and it passes a DCC's zero tangents through. Every assertion is
against the rule Blender's MikkTSpace glTF export follows (a mirrored shell:
T along +U, which is -X here, and ``w = -1``). Added: 2026-09-23
"""

import json
import math
import struct
import unittest

from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert
from pythontk.file_utils.mesh_convert.glb_reader import GlbReader
from pythontk.file_utils.mesh_convert.glb_tangents import GlbTangents
from pythontk.file_utils.temp_artifacts import TempArtifacts

#: One quad's corners (x, y) in order; the two triangles are (0, 1, 2) and
#: (0, 2, 3), counter-clockwise seen from +Z, where the normal points.
CORNERS = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
#: Accessor order of every fixture: what a test edits by index.
POSITION, NORMAL, TANGENT = 0, 1, 2
#: An index accessor's componentType -> its struct format.
INDEX_FORMATS = {5121: "B", 5123: "H", 5125: "I"}


def tangent_quads_glb(
    path,
    shells,
    tex_coord=0,
    split=False,
    tangent_stride=16,
    tangent_offset=0,
    index_type=5123,
    size=1.0,
):
    """A GLB of one unit quad per entry in *shells*, laid out as FBX2glTF
    writes a mesh: float attributes, uint16 indices, and the tangent along +U.

    Each entry is ``(layout, w)`` or ``(layout, w, direction)``: *layout*
    ``"plain"`` (U runs along +X, V down the image as glTF has it),
    ``"mirrored"`` (U runs along -X) or ``"degenerate"`` (every corner at one
    UV, so no triangle can say which way the UVs run); *w* the handedness the
    TANGENT ships with, and *direction* its xyz where it is not along +U
    (``(0, 0, 0)`` for the zero tangent a DCC writes on a UV sliver).

    The normal map samples ``TEXCOORD_<tex_coord>``; with *tex_coord* 1, set 0
    carries that set flipped in V, so a vote read off the wrong set comes out
    with the other sign (``w`` answers which way V runs across the shipped
    tangent: a set flipped in U alone would vote alike). *split* gives each
    quad its own primitive over the SAME vertex accessors -- the converter's
    shape for one mesh, two materials.
    *tangent_stride* above 16 interleaves zero padding with each tangent, which
    sits *tangent_offset* bytes into its element (the accessor's
    ``byteOffset``). *index_type* is the indices' componentType, ``None`` for a
    non-indexed list (every triangle corner a vertex of its own, in order).
    *size* scales each quad and its UV extent alike, as on a UV sliver.
    """
    positions, normals, tangents, flipped, shaped, indices = [], [], [], [], [], []
    for i, (layout, w, *direction) in enumerate(shells):
        x0, base = 2.0 * i, 4 * i
        along_u = [-1.0 if layout == "mirrored" else 1.0, 0.0, 0.0]
        for x, y in CORNERS:
            positions += [(x0 + x) * size, y * size, 0.0]
            normals += [0.0, 0.0, 1.0]
            if layout == "mirrored":
                shaped += [(1.0 - x) * size, (1.0 - y) * size]
            elif layout == "degenerate":
                shaped += [0.5, 0.5]
            else:
                shaped += [x * size, (1.0 - y) * size]
            flipped += [shaped[-2], size - shaped[-1]]
            tangents.append(list(direction[0] if direction else along_u) + [w])
        indices.append([base, base + 1, base + 2, base, base + 2, base + 3])

    uv_sets = [shaped] if tex_coord == 0 else [flipped, shaped]
    if index_type is None:
        order = [i for quad in indices for i in quad]

        def spread(flat, width):
            return [flat[i * width + k] for i in order for k in range(width)]

        positions, normals = spread(positions, 3), spread(normals, 3)
        uv_sets = [spread(uvs, 2) for uvs in uv_sets]
        tangents = [tangents[i] for i in order]
    views, accessors, blob = [], [], b""

    def add(payload, count, kind, component=5126, stride=None, extra=None):
        nonlocal blob
        view = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(payload)}
        if stride:
            view["byteStride"] = stride
        views.append(view)
        accessor = {
            "bufferView": len(views) - 1,
            "componentType": component,
            "count": count,
            "type": kind,
        }
        accessor.update(extra or {})
        accessors.append(accessor)
        blob += payload + b"\x00" * ((4 - len(payload) % 4) % 4)
        return len(accessors) - 1

    count = len(positions) // 3
    before, after = bytes(tangent_offset), bytes(tangent_stride - 16 - tangent_offset)
    attributes = {
        "POSITION": add(
            struct.pack(f"<{len(positions)}f", *positions),
            count,
            "VEC3",
            extra={
                "min": [min(positions[k::3]) for k in range(3)],
                "max": [max(positions[k::3]) for k in range(3)],
            },
        ),
        "NORMAL": add(struct.pack(f"<{len(normals)}f", *normals), count, "VEC3"),
        "TANGENT": add(
            b"".join(before + struct.pack("<4f", *t) + after for t in tangents),
            count,
            "VEC4",
            stride=tangent_stride if tangent_stride != 16 else None,
            extra={"byteOffset": tangent_offset} if tangent_offset else None,
        ),
    }
    for index, uvs in enumerate(uv_sets):
        attributes[f"TEXCOORD_{index}"] = add(
            struct.pack(f"<{len(uvs)}f", *uvs), count, "VEC2"
        )

    def triangles(flat):
        fmt = INDEX_FORMATS[index_type]
        return add(
            struct.pack(f"<{len(flat)}{fmt}", *flat), len(flat), "SCALAR", index_type
        )

    if index_type is None:
        primitives = [{"attributes": attributes, "material": 0}]
    else:
        groups = indices if split else [[i for quad in indices for i in quad]]
        primitives = [
            {"attributes": dict(attributes), "indices": triangles(group), "material": 0}
            for group in groups
        ]
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "quads", "mesh": 0}],
        "meshes": [{"primitives": primitives}],
        "materials": [
            {"name": "m", "normalTexture": {"index": 0, "texCoord": tex_coord}}
        ],
        "textures": [{"source": 0}],
        "images": [{"uri": "normal.png"}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    payload = json.dumps(gltf).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    rest = struct.pack("<I4s", len(blob), b"BIN\x00") + blob
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(payload) + len(rest)))
        fh.write(struct.pack("<I4s", len(payload), b"JSON") + payload)
        fh.write(rest)
    return path


def read_tangents(path):
    """Every vertex's TANGENT, as ``(x, y, z, w)``, through the file's own layout."""
    reader = GlbReader.load(path)
    primitive = reader.gltf["meshes"][0]["primitives"][0]
    return reader.accessor(primitive["attributes"]["TANGENT"])


class GlbTangentsTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = TempArtifacts("test_glb_tangents", policy="scoped")

    def tearDown(self):
        self.temp.cleanup()

    def _glb(self, shells, **kw):
        return tangent_quads_glb(self.temp.path(extension=".glb"), shells, **kw)

    def _w(self, path):
        return [round(t[3]) for t in read_tangents(path)]

    def _summary(self, primitives=0, signs=0, directions=0):
        return {"primitives": primitives, "signs": signs, "directions": directions}

    # ------------------------------------------------------------ handedness
    def test_a_mirrored_shell_takes_the_handedness_mikktspace_gives_it(self):
        """REGRESSION (2026-09-23), reported as one button in a production
        assembly rendering bad normals in the WebXR preview. FBX2glTF shipped
        ``w = +1`` on every vertex; on a mirrored UV shell that points the
        bitangent at image-DOWN, and three.js -- which takes a shipped TANGENT
        over its own frame -- inverted green there: 7 of 61 tangent-carrying
        primitives, and on each exactly its mirrored triangles. Blender's
        MikkTSpace export of this very layout writes ``w = -1`` on the mirrored
        quad, and the tangent's own direction (along +U) is left as shipped."""
        path = self._glb([("plain", 1.0), ("mirrored", 1.0)])
        before = read_tangents(path)

        summary = GlbTangents.repair(path)

        self.assertEqual(summary, self._summary(primitives=1, signs=4))
        after = read_tangents(path)
        self.assertEqual([round(t[3]) for t in after], [1] * 4 + [-1] * 4)
        self.assertEqual([t[:3] for t in after], [t[:3] for t in before])

    def test_a_wrong_minus_one_on_a_plain_shell_is_set_back(self):
        """The vote decides the sign both ways: this is a rule, not a
        "mirrored means -1" patch."""
        path = self._glb([("plain", -1.0)])
        self.assertEqual(GlbTangents.repair(path)["signs"], 4)
        self.assertEqual(self._w(path), [1] * 4)

    def test_a_file_whose_tangents_already_agree_is_not_rewritten(self):
        path = self._glb([("plain", 1.0), ("mirrored", -1.0)])
        with open(path, "rb") as fh:
            before = fh.read()

        self.assertEqual(GlbTangents.repair(path), self._summary())
        with open(path, "rb") as fh:
            self.assertEqual(fh.read(), before)

    def test_the_vote_reads_the_uv_set_the_normal_map_samples(self):
        """Tangent space is defined on the normal texture's own texCoord; here
        set 1 -- the one the map samples -- is the mirrored shell, and set 0
        runs V the other way, so a vote read off it would set +1."""
        path = self._glb([("mirrored", 1.0)], tex_coord=1)
        GlbTangents.repair(path)
        self.assertEqual(self._w(path), [-1] * 4)

    def test_a_texture_transform_texcoord_names_the_set_the_map_samples(self):
        """``KHR_texture_transform.texCoord`` overrides the textureInfo's own
        (the spec, and three.js's loader): the map samples set 1 here although
        ``normalTexture.texCoord`` says 0 -- and a vote read off set 0 would
        set +1 on this mirrored shell."""
        path = self._glb([("mirrored", 1.0)], tex_coord=1)
        with MeshConvert.open_glb(path) as edit:
            edit.gltf["materials"][0]["normalTexture"] = {
                "index": 0,
                "texCoord": 0,
                "extensions": {"KHR_texture_transform": {"texCoord": 1}},
            }
            edit.dirty = True
        GlbTangents.repair(path)
        self.assertEqual(self._w(path), [-1] * 4)

    def test_every_index_width_and_a_non_indexed_list_vote_alike(self):
        """uint8 and uint32 indices read as uint16 do, and a primitive with no
        indices is a triangle list over its vertices in order."""
        for index_type in (5121, 5125, None):
            with self.subTest(index_type=index_type):
                path = self._glb(
                    [("plain", 1.0), ("mirrored", 1.0)], index_type=index_type
                )
                GlbTangents.repair(path)
                per_quad = 4 if index_type else 6
                self.assertEqual(self._w(path), [1] * per_quad + [-1] * per_quad)

    def test_a_tangent_shared_by_two_primitives_is_voted_on_by_both(self):
        """One vertex set, two index sets (the converter's shape for a mesh
        with two materials): each primitive's triangles vote for the corners
        they use, and the one rewritten accessor serves both."""
        path = self._glb([("plain", 1.0), ("mirrored", 1.0)], split=True)

        summary = GlbTangents.repair(path)

        self.assertEqual(summary, self._summary(primitives=2, signs=4))
        self.assertEqual(self._w(path), [1] * 4 + [-1] * 4)

    def test_a_vertex_no_triangle_can_vote_for_keeps_its_w(self):
        """Every UV triangle around it degenerate, nothing says which way the
        UVs run -- so nothing is guessed."""
        path = self._glb([("degenerate", -1.0), ("mirrored", 1.0)])
        GlbTangents.repair(path)
        self.assertEqual(self._w(path), [-1] * 4 + [-1] * 4)

    def test_an_interleaved_tangent_is_repaired_at_its_stride(self):
        """The rewrite lands at the view's ``byteStride`` and the accessor's own
        ``byteOffset`` into each element, not at a packed VEC4's 16 bytes -- and
        leaves the bytes around each tangent alone."""
        for offset in (0, 16):
            with self.subTest(byteOffset=offset):
                path = self._glb(
                    [("mirrored", 1.0)], tangent_stride=32, tangent_offset=offset
                )

                self.assertEqual(GlbTangents.repair(path)["signs"], 4)
                self.assertEqual(self._w(path), [-1] * 4)
                edit = MeshConvert._read_glb(path)
                accessor = edit.gltf["accessors"][TANGENT]
                start = edit.gltf["bufferViews"][accessor["bufferView"]]["byteOffset"]
                raw = bytes(edit.bin_data[start : start + 128])
                around = {
                    raw[i * 32 : i * 32 + offset]
                    + raw[i * 32 + offset + 16 : i * 32 + 32]
                    for i in range(4)
                }
                self.assertEqual(around, {bytes(16)})

    # ------------------------------------------------------------ direction
    def test_a_zero_length_tangent_is_rebuilt_along_u(self):
        """REGRESSION (2026-09-23): the same production push shipped 8 zero
        tangents on UV slivers the DCC could not orient -- glTF requires a unit
        tangent, and three.js normalizes the zero vector into NaN. It is rebuilt
        along +U (here -X, the shell being mirrored), with the sign that
        direction needs."""
        path = self._glb([("mirrored", 1.0, (0.0, 0.0, 0.0))])

        summary = GlbTangents.repair(path)

        self.assertEqual(summary, self._summary(primitives=1, signs=4, directions=4))
        for x, y, z, w in read_tangents(path):
            self.assertEqual((round(x, 6), round(y, 6), round(z, 6), w), (-1, 0, 0, -1))

    def test_a_zero_tangent_on_a_small_sliver_is_still_rebuilt_along_u(self):
        """REGRESSION (2026-09-23 review): the UV directions summed at a vertex
        are weighted by each triangle's UV area and 3D extent, so on a sliver
        -- exactly where a DCC leaves a zero tangent -- they are tiny, not
        absent: here 1e-4 of each, a sum near 1e-8. An absolute floor of 1e-6
        on that sum threw the real +U away for "any direction across the
        normal", which on this mirrored shell pointed the rebuilt tangent at -U
        (red inverted). Whether a direction survives is judged against the
        sum's own length."""
        path = self._glb([("mirrored", 1.0, (0.0, 0.0, 0.0))], size=1e-4)

        self.assertEqual(GlbTangents.repair(path)["directions"], 4)
        for x, y, z, w in read_tangents(path):
            self.assertEqual((round(x, 6), round(y, 6), round(z, 6), w), (-1, 0, 0, -1))

    def test_a_zero_tangent_with_no_uv_direction_gets_one_across_the_normal(self):
        """A sliver with no UV extent says nothing about +U either: any unit
        direction perpendicular to the normal beats NaN, and the sign stands."""
        path = self._glb([("degenerate", 1.0, (0.0, 0.0, 0.0))])

        self.assertEqual(GlbTangents.repair(path)["directions"], 4)
        for x, y, z, w in read_tangents(path):
            self.assertAlmostEqual(math.sqrt(x * x + y * y + z * z), 1.0, places=6)
            self.assertAlmostEqual(z, 0.0, places=6)  # the normal is +Z
            self.assertEqual(w, 1.0)

    # -------------------------------------------------------------- refused
    def test_what_it_cannot_rewrite_in_place_is_left_as_shipped(self):
        """A strip is not a triangle list; readers that disagree on the vertex
        set cannot share one tangent; a sparse, compressed or external TANGENT
        is not bytes in this file's BIN to flip; a quantized one is not four
        floats to write; an index past the vertices names none of them (the
        first index accessor re-read as uint32 over its uint16 bytes)."""
        edits = {
            "strip": lambda g: g["meshes"][0]["primitives"][0].update(mode=5),
            "two vertex sets": lambda g: g["meshes"][0]["primitives"][1][
                "attributes"
            ].update(NORMAL=POSITION),
            "sparse": lambda g: g["accessors"][TANGENT].update(sparse={"count": 0}),
            "compressed": lambda g: g["bufferViews"][
                g["accessors"][TANGENT]["bufferView"]
            ].update(extensions={"EXT_meshopt_compression": {}}),
            "external": lambda g: g["buffers"][0].update(uri="quads.bin"),
            "quantized": lambda g: g["accessors"][TANGENT].update(
                componentType=5122, normalized=True
            ),
            "index out of range": lambda g: g["accessors"][
                g["meshes"][0]["primitives"][0]["indices"]
            ].update(componentType=5125, count=3),
        }
        for name, edit_json in edits.items():
            with self.subTest(name):
                path = self._glb([("plain", 1.0), ("mirrored", 1.0)], split=True)
                with MeshConvert.open_glb(path) as edit:
                    edit_json(edit.gltf)
                    edit.dirty = True
                with open(path, "rb") as fh:
                    before = fh.read()
                self.assertEqual(GlbTangents.repair(path), self._summary())
                with open(path, "rb") as fh:
                    self.assertEqual(fh.read(), before)

    def test_meshconvert_entry_point_delegates(self):
        path = self._glb([("mirrored", 1.0)])
        self.assertEqual(
            MeshConvert.fix_glb_tangents(path), self._summary(primitives=1, signs=4)
        )
        self.assertEqual(self._w(path), [-1] * 4)


if __name__ == "__main__":
    unittest.main()
