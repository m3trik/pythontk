# !/usr/bin/python
# coding=utf-8
"""Zero-dependency USD (OpenUSD) file utilities.

Pure-Python mechanism only — **no** ``pxr`` import, per the pythontk charter
(zero-dep, DCC-agnostic). Three composable primitives:

- :class:`UsdFile` — format sniffing (``usda``/``usdc``/``usdz`` by magic bytes,
  extension fallback) and USDZ package inspection.
- :class:`UsdzPackager` — spec-compliant ``.usdz`` writer/verifier over the
  stdlib ``zipfile``: entries stored (never compressed), file data 64-byte
  aligned via zero-padded extra fields, default layer first — the three rules
  the USDZ spec adds on top of zip.
- :class:`UsdMeshWriter` — authors a single textured mesh as a ``.usda`` text
  layer (points / faces / UVs / normals + a ``UsdPreviewSurface`` material),
  plus a minimal OBJ/MTL reader. Composed by :func:`obj_to_usd` /
  :func:`obj_to_usdz` — the no-DCC publish path (e.g. photogrammetry: Metashape
  exports OBJ, this publishes a QuickLook-ready USDZ without Maya/Blender or a
  license).

DCC-side USD I/O (the full-fidelity path) lives downstream in
``mayatk.env_utils.usd`` / ``blendertk.env_utils.usd`` over each app's native
USD runtime; this module is the shared, dependency-free floor beneath them.
"""
from __future__ import annotations

import logging
import os
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

#: Every extension the USD ecosystem reads as a layer/package.
USD_EXTENSIONS = (".usd", ".usda", ".usdc", ".usdz")

_USDA_MAGIC = b"#usda"
_USDC_MAGIC = b"PXR-USDC"
_ZIP_MAGIC = b"PK\x03\x04"


def _unique_arcname(src: str, taken: set, prefix: str = "textures") -> str:
    """A package-unique ``<prefix>/<basename>`` for *src*, suffixing on
    collision (``tex.png`` → ``tex_1.png``) so same-named files from different
    directories never clash. Adds the result to *taken*."""
    base = os.path.basename(src)
    arc, n = f"{prefix}/{base}", 1
    while arc in taken:
        stem, ext = os.path.splitext(base)
        arc, n = f"{prefix}/{stem}_{n}{ext}", n + 1
    taken.add(arc)
    return arc

# USDZ data alignment (bytes) mandated by the spec for zero-copy mmap reads.
_USDZ_ALIGNMENT = 64


def is_usd_file(path: str) -> bool:
    """Return True when *path* looks like a USD layer/package.

    An existing file is classified by content (magic bytes); a non-existent
    or unreadable path falls back to its extension.
    """
    return UsdFile.sniff(path) is not None


class UsdFile:
    """Format sniffing + USDZ package inspection (pure Python, no ``pxr``)."""

    @staticmethod
    def sniff(path: str) -> Optional[str]:
        """Classify *path* as ``'usda'`` / ``'usdc'`` / ``'usdz'``, or ``None``.

        Content wins over extension: an existing readable file is identified by
        magic bytes (``#usda``, ``PXR-USDC``, zip — a ``.usd`` file may be
        either text or crate). A missing/unreadable path classifies by
        extension alone (``.usd`` assumed crate).
        """
        path = str(path)
        ext = os.path.splitext(path)[1].lower()
        head = b""
        try:
            with open(path, "rb") as fh:
                head = fh.read(8)
        except OSError:
            pass
        if head:
            if head.startswith(_USDA_MAGIC):
                return "usda"
            if head.startswith(_USDC_MAGIC):
                return "usdc"
            if head.startswith(_ZIP_MAGIC) and ext in (".usdz", ".usd"):
                return "usdz"
            if ext in USD_EXTENSIONS:
                return None  # claimed USD but content says otherwise
        if ext == ".usda":
            return "usda"
        if ext in (".usdc", ".usd"):
            return "usdc"
        if ext == ".usdz":
            return "usdz"
        return None

    @staticmethod
    def list_package(path: str) -> List[str]:
        """Return the entry names of a ``.usdz`` package, in archive order."""
        with zipfile.ZipFile(path) as zf:
            return zf.namelist()

    @staticmethod
    def default_layer(path: str) -> Optional[str]:
        """The package's default (first) layer name, or ``None`` if the first
        entry is not a USD layer (a malformed package)."""
        names = UsdFile.list_package(path)
        if names and os.path.splitext(names[0])[1].lower() in (
            ".usd", ".usda", ".usdc",
        ):
            return names[0]
        return None


class UsdzPackager:
    """Write and verify spec-compliant ``.usdz`` packages.

    A USDZ file is a zip archive with three extra rules (all enforced here):

    1. Entries are **stored**, never compressed.
    2. Each entry's **file data** starts on a 64-byte boundary (achieved by
       zero-padding the local header's extra field — the same technique
       Pixar's ``usdzip`` uses; readers skip the extra field by length).
    3. The **first** entry is the package's default layer (a ``.usd*`` file).
    """

    # Asset types the USDZ spec admits alongside USD layers.
    _ASSET_EXTENSIONS = (
        ".usd", ".usda", ".usdc",
        ".png", ".jpg", ".jpeg", ".exr", ".avif",
        ".m4a", ".mp3", ".wav",
    )

    @classmethod
    def package(
        cls,
        files: Sequence[Union[str, Tuple[str, str]]],
        output_path: str,
        default_layer: Optional[str] = None,
    ) -> str:
        """Package *files* into a ``.usdz`` at *output_path*.

        Parameters:
            files: Source paths, or ``(source_path, arcname)`` pairs for
                entries that must live at a sub-path inside the package (e.g.
                ``("/abs/tex.png", "textures/tex.png")`` so a layer's
                ``@textures/tex.png@`` reference resolves). Bare paths use
                their basename as arcname.
            output_path: Destination ``.usdz`` (``.usdz`` appended if missing;
                parent dirs created).
            default_layer: Arcname (or source path) of the entry to place
                first. Defaults to the first USD-layer file given.

        Returns:
            The written ``.usdz`` path.

        Raises:
            ValueError: No files, duplicate arcnames, or no USD layer to lead.
            FileNotFoundError: A source file is missing.
        """
        entries: List[Tuple[str, str]] = []
        for item in files:
            src, arc = item if isinstance(item, (tuple, list)) else (item, None)
            src = os.path.abspath(os.path.expandvars(str(src)))
            if not os.path.isfile(src):
                raise FileNotFoundError(f"USDZ input not found: {src}")
            arc = (arc or os.path.basename(src)).replace("\\", "/").lstrip("/")
            entries.append((src, arc))
        if not entries:
            raise ValueError("No files to package.")
        arcs = [arc for _, arc in entries]
        if len(set(arcs)) != len(arcs):
            raise ValueError(f"Duplicate arcnames in package: {sorted(arcs)}")

        def is_layer(arc: str) -> bool:
            return os.path.splitext(arc)[1].lower() in (".usd", ".usda", ".usdc")

        lead = None
        if default_layer:
            want = default_layer.replace("\\", "/")
            for pair in entries:
                if want in (pair[1], pair[0].replace("\\", "/")):
                    lead = pair
                    break
            if lead is None:
                raise ValueError(f"default_layer {default_layer!r} not among inputs.")
        else:
            lead = next((p for p in entries if is_layer(p[1])), None)
            if lead is None:
                raise ValueError("A USDZ package requires at least one USD layer.")
        entries.remove(lead)
        entries.insert(0, lead)

        for _, arc in entries:
            if os.path.splitext(arc)[1].lower() not in cls._ASSET_EXTENSIONS:
                logger.warning(
                    "USDZ entry %r is not a spec-listed asset type "
                    "(packaged anyway; some viewers may ignore it).", arc
                )

        output_path = os.path.abspath(os.path.expandvars(str(output_path)))
        if not output_path.lower().endswith(".usdz"):
            output_path += ".usdz"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for src, arc in entries:
                info = zipfile.ZipInfo.from_file(src, arcname=arc)
                info.compress_type = zipfile.ZIP_STORED
                # Zero-pad the local header's extra field so the entry's DATA
                # begins on a 64-byte boundary: data offset = header offset
                # + 30-byte fixed header + filename + extra. A 1-3 byte pad
                # can't form a well-formed (id, size) extra chunk, so bump it
                # by one full alignment — readers that parse chunks then see
                # zeroed (0, 0) chunks, which every known reader skips.
                header_offset = zf.fp.tell()
                name_len = len(info.filename.encode("utf-8"))
                pad = -(header_offset + 30 + name_len) % _USDZ_ALIGNMENT
                if 0 < pad < 4:
                    pad += _USDZ_ALIGNMENT
                info.extra = b"\x00" * pad
                with open(src, "rb") as fh:
                    zf.writestr(info, fh.read())
        logger.info(f"Packaged USDZ: {output_path} ({len(entries)} entries)")
        return output_path

    @classmethod
    def from_layer(cls, layer_path: str, output_path: str) -> str:
        """Build a self-contained ``.usdz`` from a ``.usda`` text layer.

        Every ``@...@`` asset reference that resolves to a non-USD file on disk
        (textures, audio) is pulled into the package under
        ``textures/<basename>`` (deduped on collision) and the reference is
        rewritten to the in-package path. References to other USD layers are
        left untouched (composition arcs are out of scope here — flatten the
        stage DCC-side first) and logged.

        The DCC wrappers (``mayatk``/``blendertk`` ``env_utils.usd``) compose
        this for their ``.usdz`` export paths: export a temp ``.usda``, then
        package it here.

        Parameters:
            layer_path: Source ``.usda`` (text) layer.
            output_path: Destination ``.usdz``.

        Returns:
            The written ``.usdz`` path.

        Raises:
            ValueError: *layer_path* is not a text (``usda``) layer.
        """
        import re
        import tempfile

        layer_path = os.path.abspath(os.path.expandvars(str(layer_path)))
        if UsdFile.sniff(layer_path) != "usda":
            raise ValueError(
                f"from_layer requires a text (.usda) layer, got: {layer_path}"
            )
        with open(layer_path, "r", encoding="utf-8") as fh:
            text = fh.read()

        layer_dir = os.path.dirname(layer_path)
        arc_by_src: Dict[str, str] = {}
        taken: set = set()

        def rewrite(match: "re.Match[str]") -> str:
            ref = match.group(1)
            src = ref if os.path.isabs(ref) else os.path.join(layer_dir, ref)
            src = os.path.abspath(src)
            ext = os.path.splitext(ref)[1].lower()
            if not os.path.isfile(src):
                return match.group(0)
            if ext in (".usd", ".usda", ".usdc", ".usdz"):
                logger.warning(
                    "Layer reference %r left un-packaged (flatten the stage "
                    "before packaging to include composed layers).", ref
                )
                return match.group(0)
            if src not in arc_by_src:
                arc_by_src[src] = _unique_arcname(src, taken)
            return f"@{arc_by_src[src]}@"

        text = re.sub(r"@([^@\n]+)@", rewrite, text)

        tmp_dir = tempfile.mkdtemp(prefix="ptk_usdz_layer_")
        try:
            staged = os.path.join(tmp_dir, os.path.basename(layer_path))
            with open(staged, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            files: List[Union[str, Tuple[str, str]]] = [
                (staged, os.path.basename(staged))
            ]
            files += [(src, arc) for src, arc in arc_by_src.items()]
            return cls.package(files, output_path)
        finally:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def verify(path: str) -> Dict[str, Any]:
        """Structurally verify a ``.usdz``; returns a report (never raises on
        spec violations — inspect ``report['valid']``).

        Report keys: ``valid`` (bool), ``issues`` (list of strings) and
        ``entries`` — ``(arcname, data_offset, aligned, stored)`` per entry.
        The data offset is read from each entry's **local** header (the
        central directory's extra field may legally differ).
        """
        issues: List[str] = []
        entries: List[Tuple[str, int, bool, bool]] = []
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            if not infos:
                issues.append("empty package")
            elif os.path.splitext(infos[0].filename)[1].lower() not in (
                ".usd", ".usda", ".usdc",
            ):
                issues.append(f"first entry is not a USD layer: {infos[0].filename}")
            with open(path, "rb") as fh:
                for info in infos:
                    fh.seek(info.header_offset + 26)  # name/extra length fields
                    name_len = int.from_bytes(fh.read(2), "little")
                    extra_len = int.from_bytes(fh.read(2), "little")
                    data_offset = info.header_offset + 30 + name_len + extra_len
                    aligned = data_offset % _USDZ_ALIGNMENT == 0
                    stored = info.compress_type == zipfile.ZIP_STORED
                    entries.append((info.filename, data_offset, aligned, stored))
                    if not aligned:
                        issues.append(f"misaligned data: {info.filename} @ {data_offset}")
                    if not stored:
                        issues.append(f"compressed entry: {info.filename}")
        return {"valid": not issues, "issues": issues, "entries": entries}


class UsdMeshWriter:
    """Author a single textured mesh as a ``.usda`` text layer (no ``pxr``).

    The authored prim structure matches what DCC importers and QuickLook
    expect from a simple asset::

        /<name>            (Xform, kind=component, defaultPrim)
          /<name>/Geom     (Mesh: points/faces, primvars:st, normals)
          /<name>/Materials/<name>Mat  (UsdPreviewSurface + UsdUVTexture per channel)

    Texture channels: ``diffuse``, ``normal``, ``roughness``, ``metallic``,
    ``occlusion``, ``emissive`` — single-channel maps read ``r`` with a raw
    color space; ``normal`` applies the standard ``(2, -1)`` scale/bias.
    """

    #: channel -> (UsdPreviewSurface input, output component, value type)
    _CHANNELS = {
        "diffuse": ("diffuseColor", "rgb", "color3f"),
        "normal": ("normal", "rgb", "normal3f"),
        "roughness": ("roughness", "r", "float"),
        "metallic": ("metallic", "r", "float"),
        "occlusion": ("occlusion", "r", "float"),
        "emissive": ("emissiveColor", "rgb", "color3f"),
    }

    # ------------------------------------------------------------------ authoring
    @classmethod
    def write(
        cls,
        path: str,
        points: Sequence[Sequence[float]],
        face_vertex_counts: Sequence[int],
        face_vertex_indices: Sequence[int],
        uvs: Optional[Sequence[Sequence[float]]] = None,
        normals: Optional[Sequence[Sequence[float]]] = None,
        textures: Optional[Dict[str, str]] = None,
        name: str = "Model",
        up_axis: str = "Y",
        meters_per_unit: float = 1.0,
        double_sided: bool = False,
    ) -> str:
        """Write the mesh to *path* as a ``.usda`` layer; returns the path.

        Parameters:
            path: Destination ``.usda`` (extension appended if missing).
            points: ``(x, y, z)`` positions.
            face_vertex_counts: Vertices per face.
            face_vertex_indices: Flattened per-face point indices.
            uvs: ``(u, v)`` pairs — per point (``vertex`` interpolation) or per
                face-vertex (``faceVarying``), decided by length.
            normals: ``(x, y, z)`` — per point or per face-vertex, likewise.
            textures: ``channel -> file path`` (see :attr:`_CHANNELS`). Paths
                are written verbatim as the asset references — pass relative
                paths for a portable layer.
            name: Root prim name (sanitized to a legal USD identifier).
            up_axis: ``"Y"`` or ``"Z"``.
            meters_per_unit: Stage ``metersPerUnit`` metadata.
            double_sided: Author ``doubleSided`` on the mesh.
        """
        if not points or not face_vertex_counts:
            raise ValueError("Mesh has no points or faces.")
        if sum(face_vertex_counts) != len(face_vertex_indices):
            raise ValueError(
                "face_vertex_indices length must equal sum(face_vertex_counts)."
            )
        if up_axis not in ("Y", "Z"):
            raise ValueError(f"up_axis must be 'Y' or 'Z', got {up_axis!r}")
        name = cls._identifier(name)

        path = os.path.abspath(os.path.expandvars(str(path)))
        if not path.lower().endswith(".usda"):
            path += ".usda"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        for ch in textures or {}:
            if ch not in cls._CHANNELS:
                logger.warning(f"Unknown texture channel {ch!r} ignored.")
        textures = {
            ch: tex for ch, tex in (textures or {}).items() if ch in cls._CHANNELS
        }

        lo = [min(p[i] for p in points) for i in range(3)]
        hi = [max(p[i] for p in points) for i in range(3)]

        w: List[str] = []
        w.append("#usda 1.0")
        w.append("(")
        w.append(f'    defaultPrim = "{name}"')
        w.append(f'    metersPerUnit = {cls._f(meters_per_unit)}')
        w.append(f'    upAxis = "{up_axis}"')
        w.append(")")
        w.append("")
        w.append(f'def Xform "{name}" (')
        w.append('    kind = "component"')
        w.append(")")
        w.append("{")
        w.append('    def Mesh "Geom"')
        w.append("    {")
        w.append(f"        float3[] extent = [{cls._vec(lo)}, {cls._vec(hi)}]")
        w.append(
            "        int[] faceVertexCounts = "
            f"[{', '.join(str(int(c)) for c in face_vertex_counts)}]"
        )
        w.append(
            "        int[] faceVertexIndices = "
            f"[{', '.join(str(int(i)) for i in face_vertex_indices)}]"
        )
        w.append(
            "        point3f[] points = "
            f"[{', '.join(cls._vec(p) for p in points)}]"
        )
        if normals:
            interp = cls._interpolation(len(normals), len(points),
                                        len(face_vertex_indices), "normals")
            if interp:
                w.append(
                    "        normal3f[] normals = "
                    f"[{', '.join(cls._vec(n) for n in normals)}] ("
                )
                w.append(f'            interpolation = "{interp}"')
                w.append("        )")
        if uvs:
            interp = cls._interpolation(len(uvs), len(points),
                                        len(face_vertex_indices), "uvs")
            if interp:
                w.append(
                    "        texCoord2f[] primvars:st = "
                    f"[{', '.join(cls._vec2(u) for u in uvs)}] ("
                )
                w.append(f'            interpolation = "{interp}"')
                w.append("        )")
        w.append('        uniform token subdivisionScheme = "none"')
        if double_sided:
            w.append("        uniform bool doubleSided = 1")
        if textures:
            w.append(f"        rel material:binding = </{name}/Materials/{name}Mat>")
        w.append("    }")
        if textures:
            w.extend(cls._material_block(name, textures))
        w.append("}")
        w.append("")

        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(w))
        logger.info(
            f"Wrote USD layer: {path} ({len(points)} points, "
            f"{len(face_vertex_counts)} faces, {len(textures)} texture(s))"
        )
        return path

    @classmethod
    def _material_block(cls, name: str, textures: Dict[str, str]) -> List[str]:
        """The ``Materials`` scope: preview surface + primvar reader + textures."""
        mat = f"/{name}/Materials/{name}Mat"
        w: List[str] = []
        w.append('    def Scope "Materials"')
        w.append("    {")
        w.append(f'        def Material "{name}Mat"')
        w.append("        {")
        w.append(
            "            token outputs:surface.connect = "
            f"<{mat}/Surface.outputs:surface>"
        )
        w.append("")
        w.append('            def Shader "Surface"')
        w.append("            {")
        w.append('                uniform token info:id = "UsdPreviewSurface"')
        for ch in cls._CHANNELS:
            if ch not in textures:
                continue
            inp, comp, vtype = cls._CHANNELS[ch]
            w.append(
                f"                {vtype} inputs:{inp}.connect = "
                f"<{mat}/{ch}Tex.outputs:{comp}>"
            )
        w.append("                token outputs:surface")
        w.append("            }")
        w.append("")
        w.append('            def Shader "stReader"')
        w.append("            {")
        w.append('                uniform token info:id = "UsdPrimvarReader_float2"')
        w.append('                string inputs:varname = "st"')
        w.append("                float2 outputs:result")
        w.append("            }")
        for ch, tex in textures.items():
            _, comp, _ = cls._CHANNELS[ch]
            asset = str(tex).replace("\\", "/")
            w.append("")
            w.append(f'            def Shader "{ch}Tex"')
            w.append("            {")
            w.append('                uniform token info:id = "UsdUVTexture"')
            w.append(f"                asset inputs:file = @{asset}@")
            w.append(
                "                float2 inputs:st.connect = "
                f"<{mat}/stReader.outputs:result>"
            )
            w.append('                token inputs:wrapS = "repeat"')
            w.append('                token inputs:wrapT = "repeat"')
            if ch == "normal":
                w.append("                float4 inputs:scale = (2, 2, 2, 1)")
                w.append("                float4 inputs:bias = (-1, -1, -1, 0)")
            if ch not in ("diffuse", "emissive"):  # data maps read raw; color maps (sRGB) use file metadata
                w.append('                token inputs:sourceColorSpace = "raw"')
            w.append("                float3 outputs:rgb")
            if comp == "r":
                w.append("                float outputs:r")
            w.append("            }")
        w.append("        }")
        w.append("    }")
        return w

    # ------------------------------------------------------------------ OBJ input
    @classmethod
    def from_obj(cls, obj_path: str) -> Dict[str, Any]:
        """Parse a Wavefront OBJ (+ its MTL) into :meth:`write` kwargs.

        Minimal by design: ``v``/``vt``/``vn``/``f`` (negative and 1-based
        indices, n-gons) merged into ONE mesh; the first MTL material carrying
        texture maps supplies the ``textures`` dict (``map_Kd`` → diffuse,
        ``norm``/``map_Bump``/``bump`` → normal, ``map_Pr`` → roughness,
        ``map_Pm`` → metallic, ``map_Ke`` → emissive). UVs/normals are emitted
        faceVarying; either is dropped (logged) if any face corner lacks it.
        """
        obj_path = os.path.abspath(os.path.expandvars(str(obj_path)))
        if not os.path.isfile(obj_path):
            raise FileNotFoundError(f"OBJ not found: {obj_path}")

        points: List[Tuple[float, ...]] = []
        vts: List[Tuple[float, ...]] = []
        vns: List[Tuple[float, ...]] = []
        counts: List[int] = []
        indices: List[int] = []
        uv_fv: List[Tuple[float, ...]] = []
        n_fv: List[Tuple[float, ...]] = []
        uv_ok = n_ok = True
        mtl_files: List[str] = []

        def resolve(idx: str, pool_len: int) -> int:
            i = int(idx)
            return i - 1 if i > 0 else pool_len + i

        with open(obj_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if not parts:
                    continue
                tag = parts[0]
                if tag == "v" and len(parts) >= 4:
                    points.append(tuple(float(x) for x in parts[1:4]))
                elif tag == "vt" and len(parts) >= 3:
                    vts.append(tuple(float(x) for x in parts[1:3]))
                elif tag == "vn" and len(parts) >= 4:
                    vns.append(tuple(float(x) for x in parts[1:4]))
                elif tag == "mtllib" and len(parts) >= 2:
                    mtl_files.append(line.split(None, 1)[1].strip())
                elif tag == "f" and len(parts) >= 4:
                    corners = parts[1:]
                    counts.append(len(corners))
                    for corner in corners:
                        comps = corner.split("/")
                        indices.append(resolve(comps[0], len(points)))
                        if uv_ok and len(comps) > 1 and comps[1]:
                            uv_fv.append(vts[resolve(comps[1], len(vts))])
                        else:
                            uv_ok = False
                        if n_ok and len(comps) > 2 and comps[2]:
                            n_fv.append(vns[resolve(comps[2], len(vns))])
                        else:
                            n_ok = False

        if not uv_ok and uv_fv:
            logger.info("OBJ has faces without UVs; dropping UVs.")
        if not n_ok and n_fv:
            logger.info("OBJ has faces without normals; dropping normals.")

        textures = cls._textures_from_mtl(os.path.dirname(obj_path), mtl_files)
        return {
            "points": points,
            "face_vertex_counts": counts,
            "face_vertex_indices": indices,
            "uvs": uv_fv if uv_ok and uv_fv else None,
            "normals": n_fv if n_ok and n_fv else None,
            "textures": textures or None,
            "name": cls._identifier(os.path.splitext(os.path.basename(obj_path))[0]),
        }

    _MTL_MAP = {
        "map_kd": "diffuse",
        "norm": "normal",
        "map_bump": "normal",
        "bump": "normal",
        "map_pr": "roughness",
        "map_pm": "metallic",
        "map_ke": "emissive",
    }

    @classmethod
    def _textures_from_mtl(
        cls, obj_dir: str, mtl_files: Sequence[str]
    ) -> Dict[str, str]:
        """Texture channels from the first MTL material that has any maps."""
        for mtl_name in mtl_files:
            mtl_path = os.path.join(obj_dir, mtl_name)
            if not os.path.isfile(mtl_path):
                logger.warning(f"MTL not found: {mtl_path}")
                continue
            current: Dict[str, str] = {}
            with open(mtl_path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    parts = line.split(None, 1)
                    if len(parts) != 2:
                        continue
                    key = parts[0].lower()
                    if key == "newmtl" and current:
                        return current  # first material with maps wins
                    channel = cls._MTL_MAP.get(key)
                    if not channel or channel in current:
                        continue
                    # Texture path: the whole remainder (preserves spaces in the
                    # path); only fall back to the last token when option flags
                    # (-bm, -o, …) are present, which rarely coexist with spaces.
                    rest = parts[1].strip()
                    name = rest.split()[-1] if rest.startswith("-") else rest
                    tex = os.path.join(obj_dir, name)
                    if os.path.isfile(tex):
                        current[channel] = os.path.abspath(tex)
                    else:
                        logger.warning(f"MTL texture missing on disk: {tex}")
            if current:
                return current
        return {}

    # ------------------------------------------------------------------ formatting
    @staticmethod
    def _identifier(name: str) -> str:
        """A legal USD prim identifier (alnum + underscore, non-digit start)."""
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in str(name))
        safe = safe or "Model"
        return ("_" + safe) if safe[0].isdigit() else safe

    @staticmethod
    def _interpolation(
        n: int, num_points: int, num_face_verts: int, what: str
    ) -> Optional[str]:
        if n == num_points:
            return "vertex"
        if n == num_face_verts:
            return "faceVarying"
        logger.warning(
            f"{what}: length {n} matches neither points ({num_points}) nor "
            f"face-vertices ({num_face_verts}); dropped."
        )
        return None

    @staticmethod
    def _f(value: float) -> str:
        """Compact float literal (``0.5`` not ``0.500000``; ints stay ints).

        ``.9g`` round-trips single-precision and preserves sub-unit detail for
        large-magnitude (e.g. georeferenced) coordinates that ``.6g`` quantized.
        """
        text = f"{float(value):.9g}"
        return text

    @classmethod
    def _vec(cls, v: Sequence[float]) -> str:
        return f"({cls._f(v[0])}, {cls._f(v[1])}, {cls._f(v[2])})"

    @classmethod
    def _vec2(cls, v: Sequence[float]) -> str:
        return f"({cls._f(v[0])}, {cls._f(v[1])})"


def obj_to_usd(obj_path: str, output_path: Optional[str] = None, **write_opts: Any) -> str:
    """Convert an OBJ to a ``.usda`` layer beside it (or at *output_path*).

    Texture references are written relative to the output when possible, so the
    layer stays portable alongside its textures. Extra *write_opts* override
    the parsed :meth:`UsdMeshWriter.write` kwargs (``up_axis`` etc.).
    """
    data = UsdMeshWriter.from_obj(obj_path)
    if output_path is None:
        output_path = os.path.splitext(os.path.abspath(obj_path))[0] + ".usda"
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if data.get("textures"):
        rel = {}
        for ch, tex in data["textures"].items():
            try:
                rel[ch] = os.path.relpath(tex, out_dir).replace("\\", "/")
            except ValueError:  # different drive — keep absolute
                rel[ch] = tex.replace("\\", "/")
        data["textures"] = rel
    data.update(write_opts)
    return UsdMeshWriter.write(output_path, **data)


def obj_to_usdz(obj_path: str, output_path: Optional[str] = None, **write_opts: Any) -> str:
    """Convert an OBJ (+ MTL textures) to a self-contained ``.usdz``.

    The layer is authored to a temp file with in-package ``textures/<name>``
    references, then packaged (layer first, textures under ``textures/``).
    The no-DCC publish path: OBJ in, AR-ready USDZ out, zero dependencies.
    """
    import tempfile

    data = UsdMeshWriter.from_obj(obj_path)
    if output_path is None:
        output_path = os.path.splitext(os.path.abspath(obj_path))[0] + ".usdz"

    textures = data.get("textures") or {}
    arc_by_src: Dict[str, str] = {}
    taken: set = set()
    for ch, tex in textures.items():
        if tex not in arc_by_src:
            arc_by_src[tex] = _unique_arcname(tex, taken)
        textures[ch] = arc_by_src[tex]
    data["textures"] = textures or None
    data.update(write_opts)

    tmp_dir = tempfile.mkdtemp(prefix="ptk_usdz_")
    try:
        layer_name = data.get("name", "Model")
        layer = UsdMeshWriter.write(os.path.join(tmp_dir, layer_name), **data)
        files: List[Union[str, Tuple[str, str]]] = [
            (layer, os.path.basename(layer))
        ]
        files += [(src, arc) for src, arc in arc_by_src.items()]
        return UsdzPackager.package(files, output_path)
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


__all__ = [
    "USD_EXTENSIONS",
    "is_usd_file",
    "UsdFile",
    "UsdzPackager",
    "UsdMeshWriter",
    "obj_to_usd",
    "obj_to_usdz",
]
