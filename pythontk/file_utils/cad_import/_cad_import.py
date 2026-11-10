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
  colours, each part's size, area and volume integrated over its exact
  surfaces, its part number, the properties its CAD system wrote and a
  fingerprint of its shape, the meshes, and with ``verify`` each mesh's
  measured deviation from its part's exact surfaces.

A part's ``id`` names its files and is what a later revision of the source
is matched by. By default it is the part number the STEP product states, else
the part's name; ``part_id`` picks another key: a property (the part number
an NX export keeps in ``"Part Number"``), the ``fingerprint`` (an exact-match
key: the same geometry read again keeps it, renamed or in another unit; a
mirrored copy does not), or a callable.

``verify`` measures against the CAD itself, not another mesh: each triangle's
centroid and edge midpoints are projected onto their face's surface. The
mesher's chord bounds its nodes and edges best, and inside a triangle a surface
can stray further (an analytic sphere's mesh, by up to 1.3 times the chord), so
a part over its profile's budget is tessellated again with a tighter chord
until it is within, and named in a warning where the mesher cannot get it
there. A later
mesh of the part -- a retopology, a decimated LOD -- is judged against the
``hero`` mesh with :meth:`MeshOps.compare <pythontk.MeshOps.compare>` and a
:class:`~pythontk.QcGate` rule on its keys.

OpenCascade (the ``pythontk[cad]`` extra, Python 3.11+) reads STEP and IGES.
Native files (Creo, CATIA, SolidWorks, NX, JT, Parasolid) need a translator
first; a backend that writes the same record and meshes slots in beside this
one, since the record, not this class, is the contract later stages read.
Adopted over the packaged converters: ``cascadio`` (OpenCascade's STEP to GLB,
which ``trimesh`` loads STEP through) writes a mesh but none of the record, no
per-part meshes and no measure against the surfaces; ``build123d`` and
``cadquery`` model CAD rather than import it. Each sits on the same wheel.
"""

import hashlib
import math
import os
import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from pythontk.core_utils.help_mixin import HelpMixin
from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.cad_import.cad_record import CadRecord

OCP_PYPI_URL = "https://pypi.org/project/cadquery-ocp-novtk/"

ProfileSpec = Union[str, Mapping[str, Any]]
#: One way to identify a part: a key (``"part_number"``, ``"name"``,
#: ``"fingerprint"`` or a property's name) or a callable taking the part's facts.
PartKey = Union[str, Callable[[Dict[str, Any]], Optional[str]]]

#: A profile's name becomes part of file names.
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
#: What a part's id never holds: it names files. Letters of any alphabet stay.
_UNSAFE_ID = re.compile(r"[^\w.-]+")

#: Viridis at every tenth, the heatmap's ramp from 0 to the budget.
_VIRIDIS = (
    (0.267, 0.005, 0.329),
    (0.283, 0.141, 0.458),
    (0.254, 0.265, 0.530),
    (0.207, 0.372, 0.553),
    (0.164, 0.471, 0.558),
    (0.128, 0.567, 0.551),
    (0.135, 0.659, 0.518),
    (0.267, 0.749, 0.441),
    (0.478, 0.821, 0.318),
    (0.741, 0.873, 0.150),
    (0.993, 0.906, 0.144),
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
    def _file_safe(name: str) -> str:
        """*name* as an id: letters (of any alphabet), digits, ``.``, ``_`` and
        ``-`` kept, every other run one ``_``, no leading or trailing dot or
        ``_``. A part number keeps its dashes and dots (``HX-100.2``), which
        ``StrUtils.sanitize`` would turn to ``_`` (and drop ``é``), so
        ``HX-100`` and ``HX_100`` stay two parts."""
        return _UNSAFE_ID.sub("_", name).strip("._") or "Part"

    @classmethod
    def _part_ids(cls, names: Iterable[str]) -> List[str]:
        """Each name made file-safe, numbered where two would collide.

        Compared ignoring case: the ids name files, and "Bolt" and "bolt" are
        one file on a case-insensitive filesystem.
        """
        ids: List[str] = []
        taken = set()
        for name in names:
            base = cls._file_safe(name)
            candidate, n = base, 1
            while candidate.casefold() in taken:
                n += 1
                candidate = f"{base}_{n}"
            taken.add(candidate.casefold())
            ids.append(candidate)
        return ids

    @classmethod
    def _part_key(cls, part_id: Union[PartKey, Sequence[PartKey]]) -> List[PartKey]:
        """*part_id* as a list of keys, each a non-empty name or a callable."""
        chain = (
            [part_id]
            if isinstance(part_id, str) or callable(part_id)
            else list(part_id)
        )
        if not chain:
            raise ValueError(
                f"part_id needs a key: {list(cls.PART_KEYS)}, a property's name or a callable."
            )
        for key in chain:
            if not callable(key) and not (isinstance(key, str) and key):
                raise TypeError(
                    f"A part_id key is a name ({list(cls.PART_KEYS)} or a property's) or a callable, not {key!r}."
                )
        return chain

    @classmethod
    def _identify(cls, facts: List[Dict[str, Any]], chain: List[PartKey]) -> List[str]:
        """Each part's identifier before it is made file-safe: the first key in
        *chain* that yields a non-blank value, else the part's name."""
        values = []
        for fact in facts:
            found = None
            for key in chain:
                if callable(key):
                    got = key(
                        {k: fact[k] for k in cls.PART_KEYS}
                        | {"properties": dict(fact["properties"])}
                    )
                elif key in cls.PART_KEYS:
                    got = fact.get(key)
                else:
                    got = fact["properties"].get(key)
                if got is not None and str(got).strip():
                    found = str(got).strip()
                    break
            values.append(found or fact["name"])
        unheld = [
            key
            for key in chain
            if isinstance(key, str)
            and key not in cls.PART_KEYS
            and not any(key in f["properties"] for f in facts)
        ]
        if unheld:
            held = sorted({k for f in facts for k in f["properties"]})
            cls.logger.warning(
                f"No part holds the propert{'y' if len(unheld) == 1 else 'ies'} {unheld} that part_id "
                f"names, so each part took the next key. The file's properties: {held or 'none'}."
            )
        return values

    @classmethod
    def _report_shared_ids(
        cls, parts: List[Dict[str, Any]], values: List[str], ids: List[str]
    ) -> None:
        """Name the parts of different products whose identifiers collide:
        numbering keeps their files apart, but a number depends on the file's
        order, not on the part.

        Pieces of one product (a solid and the PMI points beside it, which
        OpenCascade splits apart) state its part number alike by design, so
        their numbering is not told.
        """
        groups: Dict[str, List[int]] = {}
        for index, value in enumerate(values):
            groups.setdefault(cls._file_safe(value).casefold(), []).append(index)
        for members in groups.values():
            if len({parts[i]["product"] for i in members}) > 1:
                named = ", ".join(
                    f"{parts[i]['name']!r} as {ids[i]!r}" for i in members
                )
                cls.logger.warning(
                    f"{len(members)} parts share the id {values[members[0]]!r}, numbered in file order: "
                    f"{named}. Choose a key that tells them apart with part_id."
                )

    @staticmethod
    def _fingerprint(figures: Dict[str, Any]) -> Optional[str]:
        """A short hash of the part's shape in its own frame: an exact-match key.

        Its moments (centroid, inertia about it) and volume, scaled by the
        root of its area so the numbers are a shape's, not a size's, rounded
        to six places, with the size itself to six significant figures. The
        same geometry read again hashes alike whatever its name or the file's
        unit, and a mirrored copy does not; geometry re-exported through
        another system can differ past the sixth digit and hash apart (3% of
        boxes scaled by one part in ten million did), so match
        those within a tolerance on ``size``, ``area`` and ``volume``.
        """
        moments = figures["moments"]
        if moments is None or figures["area"] <= 0.0:
            return None
        scale, mass = math.sqrt(figures["area"]), moments["mass"]
        numbers = [
            figures["volume"] / scale**3,
            *(c / scale for c in moments["centroid"]),
        ]
        numbers += [i / (mass * scale**2) for i in moments["inertia"]]
        text = "%.6g|" % scale + "|".join("%.6f" % (round(n, 6) + 0.0) for n in numbers)
        return hashlib.sha256(text.encode("ascii")).hexdigest()[:16]

    @staticmethod
    def _material_name(color: Optional[Sequence[float]]) -> str:
        """One name per colour, so a colour shared across parts is one material."""
        if color is None:
            return "cad_default"
        return "cad_" + "".join(
            f"{round(min(max(c, 0.0), 1.0) * 255):02x}" for c in color
        )

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
                lines += [
                    "f %d//%d %d//%d %d//%d\n" % (a, a, b, b, c, c) for a, b, c in tris
                ]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(lines)
        materials = {}
        for name, color in zip(names, mesh["colors"]):
            r, g, b, a = color if color is not None else (0.8, 0.8, 0.8, 1.0)
            materials[name] = f"newmtl {name}\nKd {r:.6g} {g:.6g} {b:.6g}\nd {a:.6g}\n"
        with open(
            os.path.join(os.path.dirname(path), f"{stem}.mtl"),
            "w",
            encoding="utf-8",
            newline="\n",
        ) as fh:
            fh.write(
                "# Kd is linear RGB, as the record states colours\n"
                + "".join(materials.values())
            )

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
            lines.append(
                "%.9g %.9g %.9g %d %d %d\n" % (*p, *(round(c * 255) for c in rgb))
            )
        lines += ["3 %d %d %d\n" % tuple(tri) for tri in t]
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(lines)

    @classmethod
    def _meshes_within(
        cls, doc, profile: Dict[str, Any], parts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Each part's mesh at *profile*, measured, re-tessellated tighter while over budget.

        Each round shrinks the last round's chord by its overshoot with a
        margin, never by more than :attr:`CadImport.REFINE_FLOOR` at once, and
        the best mesh is kept. OpenCascade's mesher is not monotonic in its
        chord (on a NIST B-spline part a tighter chord measured worse, and the
        next tighter one was within), so a round that fails to gain is tried
        tighter; :attr:`CadImport.REFINE_PATIENCE` rounds in a row that remove
        less than :attr:`CadImport.REFINE_GAIN` of the best peak's excess over
        the budget stop the loop (a mesher that cannot do better would
        otherwise multiply triangles for nothing; judged on the excess, a
        mesh just over its budget is not stopped for closing the last few
        percent), and so does a mesh past :attr:`CadImport.REFINE_MAX_GROWTH`
        times the first one's triangles. A part still over its budget is named
        in a warning, with why the loop stopped. Returns
        the final meshes, each with the chord it was made at; the document
        holds their tessellations.
        """
        meshes = []
        for index, (occ_part, part) in enumerate(zip(doc.parts, parts)):
            best_chord = chord = profile["chord"]
            best = latest = doc.mesh(occ_part, measure=True)
            ceiling = cls.REFINE_MAX_GROWTH * len(best["triangles"])
            why, stalls = f"after {cls.REFINE_ROUNDS} tighter tessellations", 0
            for _ in range(cls.REFINE_ROUNDS):
                if best["deviation"]["peak"] <= profile["budget"] or not len(
                    best["triangles"]
                ):
                    break
                overshoot = (
                    cls.REFINE_MARGIN * profile["budget"] / latest["deviation"]["peak"]
                )
                chord *= max(overshoot, cls.REFINE_FLOOR)
                doc.tessellate(chord, profile["angle"], parts=[index])
                latest = doc.mesh(occ_part, measure=True)
                excess = best["deviation"]["peak"] - profile["budget"]
                gained = (
                    best["deviation"]["peak"] - latest["deviation"]["peak"]
                    >= cls.REFINE_GAIN * excess
                )
                if latest["deviation"]["peak"] < best["deviation"]["peak"]:
                    best, best_chord = latest, chord
                stalls = 0 if gained else stalls + 1
                if stalls >= cls.REFINE_PATIENCE:
                    why = "tightening the chord stopped helping"
                    break
                if len(latest["triangles"]) > ceiling:
                    why = f"a tighter mesh passed {cls.REFINE_MAX_GROWTH} times the first one's triangles"
                    break
            if best is not latest:
                doc.tessellate(
                    best_chord, profile["angle"], parts=[index]
                )  # back to the best
            best["chord"] = best_chord
            if len(best["triangles"]) and best["deviation"]["peak"] > profile["budget"]:
                cls.logger.warning(
                    f"{part['name']}: its {profile['name']!r} mesh strays "
                    f"{best['deviation']['peak']:.4f} mm from its surfaces ({why}; the mesh "
                    f"kept is the closest, at a {best_chord:.4g} mm chord), over the profile's "
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
    #: below this fraction at once, stopping once this many rounds in a row
    #: remove less than this share of the peak's excess over the budget, or
    #: once a mesh holds this many times the first one's triangles.
    REFINE_ROUNDS = 4
    REFINE_MARGIN = 0.8
    REFINE_FLOOR = 0.25
    REFINE_PATIENCE = 2
    REFINE_GAIN = 0.1
    REFINE_MAX_GROWTH = 16

    #: The keys a profile mapping may hold.
    PROFILE_KEYS = ("name", "chord", "angle", "budget")

    #: The axes a source may state as up. The GLBs turn it to glTF's +Y.
    UP_AXES = ("z", "y")

    #: The ``part_id`` keys every part has; any other name is a property's.
    PART_KEYS = ("part_number", "name", "fingerprint")

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
                raise ValueError(
                    f"Unknown profile key(s) {unknown}: a profile holds {list(cls.PROFILE_KEYS)}."
                )
            values = {"name": "custom", **spec}
        else:
            raise TypeError(
                f"A profile is a name or a mapping, not {type(spec).__name__}."
            )
        name = str(values["name"])
        if not _PROFILE_NAME.match(name):
            raise ValueError(
                f"Profile name {name!r} names files: use letters, digits, '_' and '-'."
            )
        chord, angle = (
            float(values.get("chord") or 0.0),
            float(values.get("angle") or 0.0),
        )
        if chord <= 0.0 or not 0.0 < angle < 90.0:
            raise ValueError(
                f"Profile {name!r} needs chord > 0 mm and 0 < angle < 90 degrees."
            )
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
        part_id: Union[PartKey, Sequence[PartKey]] = ("part_number", "name"),
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
            part_id: What a part's ``id`` is, tried in order; the first key
                that yields a non-blank value wins, else the part's name.
                A key is ``"part_number"`` (the STEP product's id),
                ``"name"``, ``"fingerprint"``, a property's name (as the CAD
                system wrote it: ``"Part Number"``), or a callable taking the
                part's facts (``name``, ``part_number``, ``fingerprint``,
                ``properties``) and returning a string or None. Ids are made
                file-safe; parts that share one are numbered, and named in a
                warning.

        Returns:
            The record's path: ``<output_dir>/<stem>.cad.json``.

        The profiles are a LOD chain, each level a tessellation of the exact
        surfaces: one line merges them into one GLB of ``MSFT_lod`` groups,
        ``<part>_LOD<n>`` (the layer above this one does the GLB work)::

            ptk.MeshConvert.merge_glb_lods(CadImport.convert(step, profiles=("hero", "realtime")))
        """
        if up_axis not in cls.UP_AXES:
            raise ValueError(f"up_axis is one of {cls.UP_AXES}, not {up_axis!r}.")
        chain = cls._part_key(part_id)
        src, fmt = cls._preflight(path)
        if heatmaps and not verify:
            cls.logger.warning(
                "heatmaps colour each mesh by its measured deviation: pass verify=True too; none written."
            )
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

        doc = _OccDocument(
            src, fmt
        )  # read first: a file that fails leaves nothing behind
        stem = os.path.splitext(os.path.basename(src))[0]
        out = os.path.abspath(
            output_dir or os.path.join(os.path.dirname(src), f"{stem}_cad")
        )
        os.makedirs(os.path.join(out, "parts"), exist_ok=True)
        facts = []
        for occ_part in doc.parts:
            figures = doc.figures(occ_part)
            facts.append(
                {
                    "name": occ_part["name"],
                    "part_number": occ_part["part_number"],
                    "fingerprint": cls._fingerprint(figures),
                    "properties": occ_part["properties"],
                    "figures": figures,
                }
            )
        values = cls._identify(facts, chain)
        ids = cls._part_ids(values)
        cls._report_shared_ids(doc.parts, values, ids)
        parts = []
        for pid, occ_part, fact in zip(ids, doc.parts, facts):
            figures = fact["figures"]
            part = {
                "id": pid,
                "name": occ_part["name"],
                "size": figures["size"],
                "area": figures["area"],
                "volume": figures["volume"],
                "meshes": [],
            }
            for key in ("part_number", "fingerprint"):
                if fact[key] is not None:
                    part[key] = fact[key]
            if fact["properties"]:
                part["properties"] = fact["properties"]
            if occ_part["color"] is not None:
                part["color"] = occ_part["color"]
            if not occ_part["faces"]:
                cls.logger.info(
                    f"{occ_part['name']}: holds no surface (points or curves: construction or "
                    "annotation geometry); recorded without a mesh."
                )
            parts.append(part)
        told = set()  # (part, what): a fact about a part is told once, not per profile
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
                    if occ_part[
                        "faces"
                    ]:  # a part with no surface was told of once, above
                        cls.logger.warning(
                            f"{part['name']}: the mesher made no triangle on any of its "
                            f"{occ_part['faces']} faces at {profile['name']!r}; no mesh written."
                        )
                    continue
                rel = f"parts/{part['id']}_{profile['name']}.obj"
                cls._write_obj(
                    os.path.join(out, rel),
                    mesh,
                    f"{part['name']}, profile {profile['name']}, mm, source axes",
                )
                entry = {
                    "profile": profile["name"],
                    "path": rel,
                    "chord": float(mesh["chord"]),
                    "triangles": int(len(mesh["triangles"])),
                    "vertices": int(len(mesh["vertices"])),
                }
                if mesh["tessellated_faces"]:
                    entry["tessellated_faces"] = mesh["tessellated_faces"]
                    if verify and (part["id"], "tessellated") not in told:
                        told.add((part["id"], "tessellated"))
                        cls.logger.warning(
                            f"{part['name']}: {mesh['tessellated_faces']} of its faces are triangles in "
                            "the source (tessellated geometry), with no exact surface to measure them "
                            "against; they are written as the file states them."
                        )
                missing, missing_area = doc.unmeshed(occ_part)
                if missing:
                    entry["unmeshed_faces"], entry["unmeshed_area"] = (
                        missing,
                        missing_area,
                    )
                    cls.logger.warning(
                        f"{part['name']}: {missing} of its faces have no mesh at {profile['name']!r} "
                        f"({missing_area:.4g} mm² of surface)."
                    )
                if verify:
                    d = mesh["deviation"]
                    # A missing face counts once its area could hide a gap the
                    # budget would see: a budget-wide square of surface.
                    within = (
                        d["peak"] <= profile["budget"]
                        and missing_area <= profile["budget"] ** 2
                    )
                    entry["deviation"] = {
                        "peak": d["peak"],
                        "mean": d["mean"],
                        "peak_at": d["peak_at"],
                        "budget": profile["budget"],
                        "within": within,
                        "source_gap": d["source_gap"],
                    }
                    if d["source_gap"] > profile["budget"]:
                        cls.logger.warning(
                            f"{part['name']}: its CAD edges lie up to {d['source_gap']:.4f} mm off its "
                            f"surfaces (the file's own tolerance), more than the {profile['name']!r} "
                            f"budget of {profile['budget']:.4f} mm. The mesh follows the edges there; "
                            "its deviation is measured from them."
                        )
                    if heatmaps:
                        heat = f"parts/{part['id']}_{profile['name']}.deviation.ply"
                        cls._write_heatmap(
                            os.path.join(out, heat), mesh, profile["budget"]
                        )
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
            "source": {
                "file": os.path.basename(src),
                "sha256": cls._sha256(src),
                "format": fmt,
                "unit": "mm",
                "up_axis": up_axis,
            },
            "profiles": resolved,
            "nodes": nodes,
            "parts": parts,
            "assemblies": assemblies,
        }
        errors = CadRecord.validate(record).errors
        if errors:  # a defect here, never in the source: refuse to publish it
            raise RuntimeError(
                f"CadImport built a record its own schema refuses: {errors}"
            )
        record_path = os.path.join(out, f"{stem}.cad.json")
        FileUtils.write_json(record_path, record, indent=2)
        return record_path
