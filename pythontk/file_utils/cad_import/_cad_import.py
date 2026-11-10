# !/usr/bin/python
# coding=utf-8
"""Engineering CAD (STEP, IGES) to meshes, an assembled GLB and a record of the product's truth.

The first stage of a CAD-to-content pipeline, with no DCC and no licence:

    >>> record = CadImport.convert("product.step", profiles=["hero", "realtime"], verify=True)  # doctest: +SKIP

writes, in ``product_cad/`` beside the file:

- ``product_<profile>.glb`` per profile: the whole assembly (tree, names,
  colours, every placement), in metres and +Y up, valid to the Khronos validator.
- ``parts/<part>_<profile>.obj`` per unique part per profile: millimetres, the
  source's axes, exact surface normals, one material per colour. Repeated parts
  (fasteners) are tessellated once and placed by the record.
- ``product.cad.json``: the :class:`~pythontk.CadRecord` -- tree, placements,
  colours, each part's exact size, area and volume, the meshes, and with
  ``verify`` each mesh's measured deviation from its part's exact surfaces.

``verify`` measures against the CAD itself, not another mesh: each triangle's
centroid and edge midpoints are projected onto their face's surface. The
mesher's chord bounds its nodes and edges only, and inside a triangle a surface
can stray further (an analytic sphere's mesh, by up to twice the chord), so a
part over its profile's budget is tessellated again with a tighter chord until
it is within: the profile's tolerance holds for every verified mesh. A later
mesh of the part -- a retopology, a decimated LOD -- is judged against the
``hero`` mesh with :meth:`MeshOps.compare <pythontk.MeshOps.compare>` and a
:class:`~pythontk.QcGate` rule on its keys.

OpenCascade (the ``pythontk[cad]`` extra, Python 3.11+) reads STEP and IGES.
Native files (Creo, CATIA, SolidWorks, NX, JT, Parasolid) need a translator
first; a backend that writes the same record and meshes slots in beside this
one, since the record, not this class, is the contract later stages read.
"""

import hashlib
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from pythontk.core_utils.help_mixin import HelpMixin
from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.cad_import.cad_record import CadRecord
from pythontk.str_utils._str_utils import StrUtils

OCP_PYPI_URL = "https://pypi.org/project/cadquery-ocp-novtk/"

ProfileSpec = Union[str, Mapping[str, Any]]

#: A profile's name becomes part of file names.
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

#: Viridis at every tenth, the heatmap's ramp from 0 to the budget.
_VIRIDIS = (
    (0.267, 0.005, 0.329), (0.283, 0.141, 0.458), (0.254, 0.265, 0.530), (0.207, 0.372, 0.553),
    (0.164, 0.471, 0.558), (0.128, 0.567, 0.551), (0.135, 0.659, 0.518), (0.267, 0.749, 0.441),
    (0.478, 0.821, 0.318), (0.741, 0.873, 0.150), (0.993, 0.906, 0.144),
)
#: A vertex over the budget: off the ramp, so "at the limit" and "past it" differ.
_OVER_BUDGET = (1.0, 0.0, 1.0)


class _CadImportInternal:
    """Helpers for :class:`CadImport`: inputs, identifiers, the writers, the refinement."""

    @classmethod
    def _preflight(cls, path: str) -> "tuple[str, str]":
        """The source's absolute path and format, refused before any work."""
        src = os.path.abspath(path)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"CAD source not found: {src}")
        fmt = cls.EXTENSIONS.get(os.path.splitext(src)[1].lower())
        if fmt is None:
            raise ValueError(
                f"CadImport reads STEP and IGES ({', '.join(sorted(cls.EXTENSIONS))}), not "
                f"{os.path.basename(src)!r}. Export native CAD files (Creo, CATIA, "
                "SolidWorks, NX) to STEP AP242 first."
            )
        return src, fmt

    @staticmethod
    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _part_ids(names: Iterable[str]) -> List[str]:
        """Each name made file-safe, numbered where two would collide.

        Compared ignoring case: the ids name files, and "Bolt" and "bolt" are
        one file on a case-insensitive filesystem.
        """
        ids: List[str] = []
        taken = set()
        for name in names:
            base = StrUtils.sanitize(name, preserve_case=True) or "Part"
            candidate, n = base, 1
            while candidate.casefold() in taken:
                n += 1
                candidate = f"{base}_{n}"
            taken.add(candidate.casefold())
            ids.append(candidate)
        return ids

    @staticmethod
    def _material_name(color: Optional[Sequence[float]]) -> str:
        """One name per colour, so a colour shared across parts is one material."""
        if color is None:
            return "cad_default"
        return "cad_" + "".join(f"{round(min(max(c, 0.0), 1.0) * 255):02x}" for c in color)

    @classmethod
    def _write_obj(cls, path: str, mesh: Dict[str, Any], title: str) -> None:
        """*mesh* as a Wavefront OBJ with vertex normals, its MTL beside it.

        Written here because the OpenCascade wheel ships no OBJ writer
        (``OCP.RWObj``); its glTF writer covers the assembled GLB.
        """
        stem = os.path.splitext(os.path.basename(path))[0]
        names = [cls._material_name(c) for c in mesh["colors"]]
        lines = [f"# {title}\n", f"mtllib {stem}.mtl\n"]
        lines += ["v %.9g %.9g %.9g\n" % tuple(p) for p in mesh["vertices"]]
        lines += ["vn %.6g %.6g %.6g\n" % tuple(n) for n in mesh["normals"]]
        for slot, name in enumerate(names):
            tris = mesh["triangles"][mesh["slots"] == slot] + 1
            if len(tris):
                lines.append(f"usemtl {name}\n")
                lines += ["f %d//%d %d//%d %d//%d\n" % (a, a, b, b, c, c) for a, b, c in tris]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(lines)
        materials = {}
        for name, color in zip(names, mesh["colors"]):
            r, g, b, a = color if color is not None else (0.8, 0.8, 0.8, 1.0)
            materials[name] = f"newmtl {name}\nKd {r:.6g} {g:.6g} {b:.6g}\nd {a:.6g}\n"
        with open(os.path.join(os.path.dirname(path), f"{stem}.mtl"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("# Kd is linear RGB, as the record states colours\n" + "".join(materials.values()))

    @staticmethod
    def _ramp(value: float) -> "tuple[float, float, float]":
        """Viridis at *value* in 0..1; magenta past 1."""
        if value > 1.0:
            return _OVER_BUDGET
        x = max(value, 0.0) * (len(_VIRIDIS) - 1)
        i = min(int(x), len(_VIRIDIS) - 2)
        f = x - i
        return tuple(a + (b - a) * f for a, b in zip(_VIRIDIS[i], _VIRIDIS[i + 1]))

    @classmethod
    def _write_heatmap(cls, path: str, mesh: Dict[str, Any], budget: float) -> None:
        """*mesh* as an ASCII PLY, each vertex coloured by its deviation over *budget*.

        Written here for the OBJ writer's reason (the wheel has no PLY mesh
        writer), and so the CAD import needs no PyMeshLab; its faces come in
        the OBJ's order. Magenta past the budget, where PyMeshLab's ramp would
        clamp, keeps "at the limit" and "past it" apart.
        """
        v, t, d = mesh["vertices"], mesh["triangles"], mesh["vertex_deviation"]
        lines = [
            "ply\nformat ascii 1.0\ncomment deviation from the exact surfaces: viridis 0 to the "
            f"budget ({budget:.6g} mm), magenta over it\n",
            f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n",
            "property uchar red\nproperty uchar green\nproperty uchar blue\n",
            f"element face {len(t)}\nproperty list uchar int vertex_indices\nend_header\n",
        ]
        for p, dev in zip(v, d):
            rgb = cls._ramp(dev / budget if budget > 0 else 0.0)
            lines.append("%.9g %.9g %.9g %d %d %d\n" % (*p, *(round(c * 255) for c in rgb)))
        lines += ["3 %d %d %d\n" % tuple(tri) for tri in t]
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(lines)

    @classmethod
    def _meshes_within(cls, doc, profile: Dict[str, Any], parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Each part's mesh at *profile*, measured, re-tessellated tighter while over budget.

        Each round shrinks the chord by the overshoot with a margin, never by
        more than :attr:`CadImport.REFINE_FLOOR` at once. A round that lowers
        the peak by less than :attr:`CadImport.REFINE_GAIN` stops the loop and
        the best mesh is kept (a mesher clamped by a loose edge tolerance
        would otherwise multiply triangles for nothing). A part still over its
        budget is named in a warning. Returns the final meshes, each with the
        chord it was made at; the document holds their tessellations.
        """
        meshes = []
        for index, (occ_part, part) in enumerate(zip(doc.parts, parts)):
            best_chord = chord = profile["chord"]
            best = doc.mesh(occ_part, measure=True)
            stopped = False
            for _ in range(cls.REFINE_ROUNDS):
                peak = best["deviation"]["peak"]
                if peak <= profile["budget"] or not len(best["triangles"]):
                    break
                chord = best_chord * max(cls.REFINE_MARGIN * profile["budget"] / peak, cls.REFINE_FLOOR)
                doc.tessellate(chord, profile["angle"], parts=[index])
                mesh = doc.mesh(occ_part, measure=True)
                if mesh["deviation"]["peak"] > (1.0 - cls.REFINE_GAIN) * peak:
                    doc.tessellate(best_chord, profile["angle"], parts=[index])  # back to the best
                    stopped = True
                    break
                best, best_chord = mesh, chord
            best["chord"] = best_chord
            if len(best["triangles"]) and best["deviation"]["peak"] > profile["budget"]:
                why = (
                    "tightening the chord stopped helping"
                    if stopped
                    else f"after {cls.REFINE_ROUNDS} tighter tessellations"
                )
                cls.logger.warning(
                    f"{part['name']}: its {profile['name']!r} mesh strays "
                    f"{best['deviation']['peak']:.4f} mm from its surfaces ({why}; it "
                    f"stopped at a {best_chord:.4g} mm chord), over the profile's "
                    f"{profile['budget']:.4f} mm budget."
                )
            meshes.append(best)
        return meshes


class CadImport(HelpMixin, LoggingMixin, _CadImportInternal):
    """STEP and IGES to meshes, an assembled GLB and a :class:`~pythontk.CadRecord`.

    Optional-dependency gated like :class:`~pythontk.MeshOps`: :meth:`available`
    is False without ``pythontk[cad]``, and :meth:`convert` raises the install
    note.
    """

    #: Source extensions and the format each is read as.
    EXTENSIONS = {".step": "step", ".stp": "step", ".iges": "iges", ".igs": "iges"}

    #: Named tessellation profiles: ``chord`` (mm) and ``angle`` (degrees), the
    #: mesher's limits. A studio's own standard adds or replaces entries; any
    #: call also takes a profile as a mapping.
    PROFILES: Dict[str, Dict[str, float]] = {
        "hero": {"chord": 0.005, "angle": 3.0},
        "standard": {"chord": 0.02, "angle": 8.0},
        "realtime": {"chord": 0.1, "angle": 15.0},
    }

    #: How ``verify`` tessellates a part over budget again: at most this many
    #: rounds, each aiming this far under the budget, never shrinking the chord
    #: below this fraction at once, and stopping when a round lowers the peak
    #: by less than this share.
    REFINE_ROUNDS = 4
    REFINE_MARGIN = 0.8
    REFINE_FLOOR = 0.25
    REFINE_GAIN = 0.1

    #: The keys a profile mapping may hold.
    PROFILE_KEYS = ("name", "chord", "angle", "budget")

    #: The axes a source may state as up. The GLBs turn it to glTF's +Y.
    UP_AXES = ("z", "y")

    # ------------------------------------------------------------ gating
    @classmethod
    def resolve(cls, required: bool = True):
        """Return the ``OCP`` module (OpenCascade), or explain how to install it.

        Parameters:
            required: When True (default) a missing module raises RuntimeError
                with install instructions; when False, returns None instead.
        """
        try:
            import OCP
        except ImportError:
            if required:
                raise RuntimeError(cls._install_note())
            return None
        return OCP

    @classmethod
    def available(cls) -> bool:
        """True when OpenCascade imports."""
        return cls.resolve(required=False) is not None

    @staticmethod
    def _install_note() -> str:
        """Actionable message for a missing backend, aimed at the current interpreter."""
        import sys

        return (
            "OpenCascade (the 'OCP' module) is not installed in this interpreter.\n"
            f'Install it with:  "{sys.executable}" -m pip install pythontk[cad]\n'
            f"(Python 3.11 or later; LGPL-2.1 OpenCascade in a binary wheel: {OCP_PYPI_URL})"
        )

    # ------------------------------------------------------------ profiles
    @classmethod
    def profile(cls, spec: ProfileSpec) -> Dict[str, Any]:
        """A profile by name (a :attr:`PROFILES` key) or by value, resolved.

        Parameters:
            spec: A name, or a mapping with ``chord`` (mm) and ``angle``
                (degrees), optionally ``name`` (letters, digits, ``_`` and
                ``-``: it names files) and ``budget`` (mm: the largest
                deviation from the exact surfaces a verified mesh may have,
                default the chord).

        Returns:
            ``{"name", "chord", "angle", "budget"}``.
        """
        if isinstance(spec, str):
            if spec not in cls.PROFILES:
                raise ValueError(
                    f"Unknown profile {spec!r}. Named profiles: {sorted(cls.PROFILES)}; "
                    "or pass {'chord': <mm>, 'angle': <degrees>}."
                )
            values = dict(cls.PROFILES[spec], name=spec)
        elif isinstance(spec, Mapping):
            unknown = sorted(set(spec) - set(cls.PROFILE_KEYS))
            if unknown:
                raise ValueError(f"Unknown profile key(s) {unknown}: a profile holds {list(cls.PROFILE_KEYS)}.")
            values = {"name": "custom", **spec}
        else:
            raise TypeError(f"A profile is a name or a mapping, not {type(spec).__name__}.")
        name = str(values["name"])
        if not _PROFILE_NAME.match(name):
            raise ValueError(f"Profile name {name!r} names files: use letters, digits, '_' and '-'.")
        chord, angle = float(values.get("chord") or 0.0), float(values.get("angle") or 0.0)
        if chord <= 0.0 or not 0.0 < angle < 90.0:
            raise ValueError(f"Profile {name!r} needs chord > 0 mm and 0 < angle < 90 degrees.")
        budget = float(values["budget"]) if values.get("budget") is not None else chord
        if budget <= 0.0:
            raise ValueError(f"Profile {name!r} needs a budget > 0 mm.")
        return {"name": name, "chord": chord, "angle": angle, "budget": budget}

    # ------------------------------------------------------------ convert
    @classmethod
    def convert(
        cls,
        path: str,
        output_dir: Optional[str] = None,
        profiles: Union[ProfileSpec, Sequence[ProfileSpec]] = ("standard",),
        up_axis: str = "z",
        verify: bool = False,
        heatmaps: bool = False,
    ) -> str:
        """Convert a STEP or IGES file; return the path of the record it wrote.

        Parameters:
            path: The source (``.step``/``.stp``/``.iges``/``.igs``), in any
                unit it states; everything written is in millimetres (the GLBs
                in metres, as glTF states them).
            output_dir: Where to write; default ``<stem>_cad`` beside the source.
            profiles: The profiles to tessellate at (names or mappings, see
                :meth:`profile`); one GLB and one mesh per part per profile.
            up_axis: The source's up axis: STEP states none. ``"z"`` (CATIA,
                NX and most AP242 exchange) or ``"y"`` (SolidWorks' and Creo's
                default views).
            verify: Measure every mesh against its part's exact surfaces and
                tessellate a part over its profile's budget again, tighter,
                until it is within (see the module docstring); each mesh's
                deviation and the chord it took are recorded.
            heatmaps: With *verify*, also write each mesh as a PLY coloured by
                its deviation (``parts/<part>_<profile>.deviation.ply``).

        Returns:
            The record's path: ``<output_dir>/<stem>.cad.json``.
        """
        if up_axis not in cls.UP_AXES:
            raise ValueError(f"up_axis is one of {cls.UP_AXES}, not {up_axis!r}.")
        src, fmt = cls._preflight(path)
        if isinstance(profiles, (str, Mapping)):
            profiles = [profiles]
        resolved = [cls.profile(p) for p in profiles]
        names = [p["name"].casefold() for p in resolved]
        if not resolved or len(set(names)) != len(names):
            raise ValueError(
                f"Profiles need distinct names, ignoring case (they name the files): "
                f"{[p['name'] for p in resolved]}."
            )
        cls.resolve()
        from pythontk.file_utils.cad_import._occ import _OccDocument

        doc = _OccDocument(src, fmt)  # read first: a file that fails leaves nothing behind
        stem = os.path.splitext(os.path.basename(src))[0]
        out = os.path.abspath(output_dir or os.path.join(os.path.dirname(src), f"{stem}_cad"))
        os.makedirs(os.path.join(out, "parts"), exist_ok=True)
        ids = cls._part_ids(p["name"] for p in doc.parts)
        parts = []
        for part_id, occ_part in zip(ids, doc.parts):
            part = {"id": part_id, "name": occ_part["name"], **doc.figures(occ_part), "meshes": []}
            if occ_part["color"] is not None:
                part["color"] = occ_part["color"]
            parts.append(part)
        assemblies = {}
        for profile in resolved:
            doc.tessellate(profile["chord"], profile["angle"])
            if verify:
                meshes = cls._meshes_within(doc, profile, parts)
            else:
                meshes = [dict(doc.mesh(p), chord=profile["chord"]) for p in doc.parts]
            assemblies[profile["name"]] = f"{stem}_{profile['name']}.glb"
            doc.write_glb(os.path.join(out, assemblies[profile["name"]]), up_axis)
            for occ_part, part, mesh in zip(doc.parts, parts, meshes):
                if not len(mesh["triangles"]):
                    cls.logger.warning(f"{part['name']}: no surface to mesh at {profile['name']!r}; skipped.")
                    continue
                rel = f"parts/{part['id']}_{profile['name']}.obj"
                cls._write_obj(
                    os.path.join(out, rel), mesh, f"{part['name']}, profile {profile['name']}, mm, source axes"
                )
                entry = {"profile": profile["name"], "path": rel, "chord": float(mesh["chord"]),
                         "triangles": int(len(mesh["triangles"])), "vertices": int(len(mesh["vertices"]))}
                missing, missing_area = doc.unmeshed(occ_part)
                if missing:
                    entry["unmeshed_faces"], entry["unmeshed_area"] = missing, missing_area
                    cls.logger.warning(
                        f"{part['name']}: {missing} of its faces have no mesh at {profile['name']!r} "
                        f"({missing_area:.4g} mm² of surface)."
                    )
                if verify:
                    d = mesh["deviation"]
                    # A missing face counts once its area could hide a gap the
                    # budget would see: a budget-wide square of surface.
                    within = d["peak"] <= profile["budget"] and missing_area <= profile["budget"] ** 2
                    entry["deviation"] = {"peak": d["peak"], "mean": d["mean"], "peak_at": d["peak_at"],
                                          "budget": profile["budget"], "within": within}
                    if heatmaps:
                        heat = f"parts/{part['id']}_{profile['name']}.deviation.ply"
                        cls._write_heatmap(os.path.join(out, heat), mesh, profile["budget"])
                        entry["deviation"]["heatmap"] = heat
                part["meshes"].append(entry)
        nodes = []
        for node in doc.nodes:
            entry = {
                "name": node["name"],
                "parent": node["parent"],
                "matrix": node["matrix"],
                "part": ids[node["part"]] if node["part"] is not None else "",
            }
            if node["color"] is not None:
                entry["color"] = node["color"]
            nodes.append(entry)
        record = {
            "version": CadRecord.VERSION,
            "source": {"file": os.path.basename(src), "sha256": cls._sha256(src), "format": fmt,
                       "unit": "mm", "up_axis": up_axis},
            "profiles": resolved,
            "nodes": nodes,
            "parts": parts,
            "assemblies": assemblies,
        }
        errors = CadRecord.validate(record).errors
        if errors:  # a defect here, never in the source: refuse to publish it
            raise RuntimeError(f"CadImport built a record its own schema refuses: {errors}")
        record_path = os.path.join(out, f"{stem}.cad.json")
        FileUtils.write_json(record_path, record, indent=2)
        return record_path
