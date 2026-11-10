# !/usr/bin/python
# coding=utf-8
"""File-level mesh processing via PyMeshLab (optional dependency).

PyMeshLab is the Python binding for MeshLab — measurement, repair,
remeshing, decimation, and attribute baking on mesh *files*, with no DCC
and no license. This module is the ecosystem's headless mesh-processing
floor: photogrammetry post-export polish, QC metrics for gate checks,
deviation measurement between meshes, and a vertex-color → texture bake.

Install: ``pip install pythontk[mesh]`` (gate: :meth:`MeshOps.available`).

Three tiers of surface, mirroring :class:`pythontk.UvUnwrap`'s registry
design:

- **Typed methods** (:meth:`MeshOps.measure`, :meth:`~MeshOps.clean`,
  :meth:`~MeshOps.remesh`, :meth:`~MeshOps.decimate`,
  :meth:`~MeshOps.compare`, :meth:`~MeshOps.bake_vertex_color`) — the
  contract-bearing API: validated, documented, stable.
- **The** :data:`OPS` **registry** — curated single-filter operators with
  validated parameters. Adding one is one :class:`OpSpec` entry; sessions
  run them by name via :meth:`_MeshSession.op`.
- :meth:`MeshOps.apply` — the escape hatch to any PyMeshLab filter by
  name, **unvalidated** and version-dependent by nature.

Filter names and parameter types below were verified against
pymeshlab 2025.7 (``PureValue`` = absolute distance; ``PercentageValue``
= percent of the bounding-box diagonal).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Tuple

from pythontk.core_utils.help_mixin import HelpMixin

logger = logging.getLogger(__name__)

PYMESHLAB_PYPI_URL = "https://pypi.org/project/pymeshlab/"

#: Formats pymeshlab reads reliably. FBX is deliberately absent — its
#: support there is weak; convert via a DCC or ``MeshConvert`` first.
SUPPORTED_EXTS = frozenset(
    {".obj", ".ply", ".stl", ".off", ".dae", ".glb", ".gltf", ".3ds", ".x3d", ".wrl"}
)

#: Formats pymeshlab can WRITE — narrower than the read set: glTF/GLB load
#: but have no exporter (probed against 2025.7). Output paths are validated
#: against this *before* any processing, so a glb input with a defaulted
#: output fails in milliseconds with an actionable error, not after a long
#: op chain with a raw PyMeshLabException at save time.
SAVE_EXTS = frozenset({".obj", ".ply", ".stl", ".off", ".dae", ".3ds", ".x3d", ".wrl"})


@dataclass(frozen=True)
class OpSpec:
    """One curated PyMeshLab filter: name, legal params, wrapped types.

    Adding an operator is one entry in :data:`OPS` — no edits to the
    validation/dispatch logic. ``percent_params`` / ``absolute_params``
    name the parameters that must be wrapped in ``PercentageValue`` /
    ``PureValue`` at call time (the wrapper types cannot appear in
    defaults because pymeshlab is never imported at module scope).
    """

    filter: str
    allowed_params: FrozenSet[str]
    defaults: Dict[str, Any] = field(default_factory=dict)
    percent_params: FrozenSet[str] = frozenset()
    absolute_params: FrozenSet[str] = frozenset()
    doc: str = ""


OPS: Dict[str, OpSpec] = {
    "remove_duplicate_vertices": OpSpec(
        filter="meshing_remove_duplicate_vertices",
        allowed_params=frozenset(),
        doc="Unify bit-identical vertices.",
    ),
    "remove_unreferenced_vertices": OpSpec(
        filter="meshing_remove_unreferenced_vertices",
        allowed_params=frozenset(),
        doc="Drop vertices referenced by no face.",
    ),
    "merge_close_vertices": OpSpec(
        filter="meshing_merge_close_vertices",
        allowed_params=frozenset({"threshold"}),
        absolute_params=frozenset({"threshold"}),
        doc="Weld vertices within an absolute distance (scene units).",
    ),
    "remove_isolated_pieces": OpSpec(
        filter="meshing_remove_connected_component_by_diameter",
        allowed_params=frozenset({"mincomponentdiag", "removeunref"}),
        percent_params=frozenset({"mincomponentdiag"}),
        doc="Drop components smaller than a % of the bbox diagonal.",
    ),
    "repair_non_manifold_edges": OpSpec(
        filter="meshing_repair_non_manifold_edges",
        allowed_params=frozenset({"method"}),
        doc="Repair non-manifold edges ('Remove Faces' or 'Split Vertices').",
    ),
    "close_holes": OpSpec(
        filter="meshing_close_holes",
        allowed_params=frozenset({"maxholesize", "selfintersection", "refinehole"}),
        doc="Fill holes up to a max boundary-edge count.",
    ),
    "decimate_quadric": OpSpec(
        filter="meshing_decimation_quadric_edge_collapse",
        allowed_params=frozenset(
            {
                "targetfacenum",
                "targetperc",
                "qualitythr",
                "preserveboundary",
                "boundaryweight",
                "preservenormal",
                "preservetopology",
                "optimalplacement",
                "planarquadric",
                "qualityweight",
                "autoclean",
            }
        ),
        doc="Quadric edge-collapse simplification.",
    ),
    "remesh_isotropic": OpSpec(
        filter="meshing_isotropic_explicit_remeshing",
        allowed_params=frozenset(
            {
                "iterations",
                "adaptive",
                "targetlen",
                "featuredeg",
                "checksurfdist",
                "maxsurfdist",
            }
        ),
        percent_params=frozenset({"targetlen", "maxsurfdist"}),
        doc="Uniform-density remesh toward a target edge length (% of bbox diag).",
    ),
    "curvature_scalar": OpSpec(
        filter="compute_scalar_by_discrete_curvature_per_vertex",
        allowed_params=frozenset({"curvaturetype"}),
        defaults={"curvaturetype": "ABS Curvature"},
        doc="Per-vertex curvature into vertex quality (feeds qualityweight decimation).",
    ),
    "taubin_smooth": OpSpec(
        filter="apply_coord_taubin_smoothing",
        allowed_params=frozenset({"lambda_", "mu", "stepsmoothnum"}),
        doc="Feature-preserving smoothing without volume shrink.",
    ),
    "laplacian_smooth_surface_preserving": OpSpec(
        filter="apply_coord_laplacian_smoothing_surface_preserving",
        allowed_params=frozenset({"angledeg", "iterations"}),
        doc="Laplacian smoothing constrained to preserve surface detail.",
    ),
}


class _MeshOpsInternal:
    """Internal base carrying the load/validate/measure plumbing.

    Helpers live here rather than as module functions to hold the
    package's helpers-on-a-class rule; :class:`MeshOps` inherits them
    alongside its public classmethods.
    """

    #: Why pymeshlab cannot write a mesh in this process ("" = it can); None
    #: until first probed.
    _io_error: Optional[str] = None

    @classmethod
    def _mesh_io_error(cls, pml) -> str:
        """Why the imported engine cannot write a mesh, or "" when it can.

        pymeshlab imports even when its IO plugins failed to load -- on Linux
        they need ``libOpenGL.so.0``, and without it every save fails "Unknown
        format for save". Probed once with a real one-triangle write.
        """
        if cls._io_error is None:
            import numpy as np

            from pythontk.file_utils.temp_artifacts import TempArtifacts

            try:
                with TempArtifacts("meshops_io_probe", policy="scoped") as tmp:
                    ms = pml.MeshSet()
                    ms.add_mesh(
                        pml.Mesh(
                            vertex_matrix=np.eye(3, dtype=np.float64),
                            face_matrix=np.array([[0, 1, 2]], dtype=np.int32),
                        )
                    )
                    ms.save_current_mesh(tmp.path(extension=".ply"))
                cls._io_error = ""
            except Exception as e:  # any failure means the engine is unusable
                cls._io_error = str(e) or type(e).__name__
        return cls._io_error

    @staticmethod
    def _preflight(input_path: str) -> str:
        """Validate the source file; return its absolute path."""
        src = os.path.abspath(input_path)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Mesh source not found: {src}")
        ext = os.path.splitext(src)[1].lower()
        if ext not in SUPPORTED_EXTS:
            raise ValueError(
                f"Unsupported mesh format {ext!r}: {src}. "
                f"Supported: {sorted(SUPPORTED_EXTS)}"
            )
        if os.path.getsize(src) == 0:
            raise RuntimeError(f"Mesh source is empty: {src}")
        return src

    @staticmethod
    def _postflight(output_path: str) -> str:
        """The output file — not a return code — is the success gate."""
        if not (os.path.isfile(output_path) and os.path.getsize(output_path) > 0):
            raise RuntimeError(f"Mesh output missing or empty: {output_path}")
        return output_path

    @staticmethod
    def _check_save_ext(output_path: str) -> str:
        """Reject un-writable output formats before any work runs."""
        ext = os.path.splitext(output_path)[1].lower()
        if ext not in SAVE_EXTS:
            raise ValueError(
                f"pymeshlab cannot write {ext!r}: {output_path}. Writable "
                f"formats: {sorted(SAVE_EXTS)} — pass an output_path with "
                "one of those extensions (glTF/GLB are read-only there)."
            )
        return output_path

    @staticmethod
    def _default_output(input_path: str, suffix: str, ext: Optional[str] = None) -> str:
        """``<stem>_<suffix><ext>`` beside the input (the ``_clean`` idiom)."""
        stem, in_ext = os.path.splitext(input_path)
        return f"{stem}_{suffix}{ext or in_ext}"

    @classmethod
    def _resolve_output(
        cls,
        input_path: str,
        output_path: Optional[str],
        suffix: str,
        ext: Optional[str] = None,
    ) -> str:
        """Default + validate the output path in one step, before any work."""
        out = output_path or cls._default_output(input_path, suffix, ext)
        return cls._check_save_ext(out)

    @classmethod
    def _run_op(cls, pml, ms, name: str, params: Dict[str, Any]) -> None:
        """Validate ``params`` against ``OPS[name]`` and apply the filter."""
        try:
            spec = OPS[name]
        except KeyError:
            raise KeyError(
                f"Unknown op {name!r}. Curated ops: {sorted(OPS)}. "
                "For arbitrary PyMeshLab filters use apply()."
            )
        unknown = set(params) - spec.allowed_params
        if unknown:
            raise TypeError(
                f"Op {name!r}: unsupported parameter(s) {sorted(unknown)}. "
                f"Supported: {sorted(spec.allowed_params)}"
            )
        merged = {**spec.defaults, **params}
        for key in spec.percent_params & set(merged):
            merged[key] = pml.PercentageValue(merged[key])
        for key in spec.absolute_params & set(merged):
            merged[key] = pml.PureValue(merged[key])
        getattr(ms, spec.filter)(**merged)

    @staticmethod
    def _union_diagonal(a, b) -> float:
        """The diagonal of the box holding both meshes: no point of either lies farther from the other."""
        import numpy as np

        boxes = [m.bounding_box() for m in (a, b)]
        low = np.minimum(boxes[0].min(), boxes[1].min())
        high = np.maximum(boxes[0].max(), boxes[1].max())
        return float(np.linalg.norm(high - low))

    #: The seed every comparison samples with: the same meshes, the same samples.
    _SAMPLE_SEED = 0

    #: Faces per block when weighing them by area: bounds the transient on a
    #: multi-GB scan (the whole mesh at once took ~235 bytes a face).
    _AREA_BLOCK = 1 << 18

    @classmethod
    def _surface_samples(cls, mesh, count: int):
        """Points on *mesh*: up to *count* of its vertices (an even stride) and
        *count* area-weighted points on its faces, from a fixed seed.

        Beyond PyMeshLab's own copies of the vertex and face tables, one number
        per face: the areas, weighed in blocks and turned in place into the
        cumulative distribution the draw searches."""
        import numpy as np

        v = np.asarray(mesh.vertex_matrix(), dtype=np.float64)
        f = np.asarray(mesh.face_matrix())  # PyMeshLab's int32, kept as it is
        verts = v[:: max(1, -(-len(v) // count))] if len(v) else v
        if not len(f):
            return verts
        area = np.empty(len(f))
        for lo in range(0, len(f), cls._AREA_BLOCK):
            tri = v[f[lo : lo + cls._AREA_BLOCK]]
            area[lo : lo + len(tri)] = np.linalg.norm(
                np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
            )
        cdf = np.cumsum(area, out=area)
        if cdf[-1] <= 0:
            return verts
        rng = np.random.default_rng(cls._SAMPLE_SEED)
        pick = np.minimum(np.searchsorted(cdf, rng.random(count) * cdf[-1], side="right"), len(f) - 1)
        r1, r2 = np.sqrt(rng.random(count)), rng.random(count)
        a, b, c = v[f[pick, 0]], v[f[pick, 1]], v[f[pick, 2]]
        faces = (1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b + (r1 * r2)[:, None] * c
        return np.vstack([verts, faces])

    @staticmethod
    def _distances(pml, ms, points, target: int, reach) -> Dict[str, Any]:
        """Each of *points*' distance to mesh *target* of *ms*, summarised."""
        import numpy as np

        if not len(points):
            return {"peak": 0.0, "mean": 0.0, "rms": 0.0, "at": None, "samples": 0}
        ms.add_mesh(pml.Mesh(vertex_matrix=np.asarray(points, dtype=np.float64)), "samples")
        cloud = ms.current_mesh_id()
        ms.compute_scalar_by_distance_from_another_mesh_per_vertex(
            measuremesh=cloud, refmesh=target, signeddist=False, maxdist=reach
        )
        d = np.abs(ms.mesh(cloud).vertex_scalar_array())
        i = int(d.argmax())
        return {
            "peak": float(d[i]),
            "mean": float(d.mean()),
            "rms": float(np.sqrt((d * d).mean())),
            "at": [float(x) for x in points[i]],
            "samples": int(len(d)),
        }

    @staticmethod
    def _measures(ms) -> Dict[str, Any]:
        """Flat, gate-safe metrics dict for the current mesh.

        Every value is numeric or ``None`` — never a fabricated ``0``: a
        zero falsely passes ``max_*`` :class:`pythontk.QcGate` rules, and
        ``None`` is the gate's documented "not measured" skip. Keys avoid
        ``min_``/``max_`` substrings (the gate strips rule prefixes with
        an unanchored replace).
        """
        geo = ms.get_geometric_measures()
        topo = ms.get_topological_measures()
        holes = topo.get("number_holes")
        bbox = geo.get("bbox")
        return {
            "faces": topo.get("faces_number"),
            "vertices": topo.get("vertices_number"),
            "edges": topo.get("edges_number"),
            "components": topo.get("connected_components_number"),
            "boundary_edges": topo.get("boundary_edges"),
            "non_two_manifold_edges": topo.get("non_two_manifold_edges"),
            "non_two_manifold_vertices": topo.get("non_two_manifold_vertices"),
            # -1 = "not computable" (non-manifold input); report honestly.
            "holes": holes if holes is not None and holes >= 0 else None,
            "unreferenced_vertices": topo.get("unreferenced_vertices"),
            "genus": topo.get("genus"),
            "two_manifold": bool(topo.get("is_mesh_two_manifold", False)),
            "surface_area": geo.get("surface_area"),
            # Only present/meaningful on watertight meshes.
            "volume": geo.get("mesh_volume"),
            "avg_edge_length": geo.get("avg_edge_length"),
            "bbox_diag": bbox.diagonal() if bbox is not None else None,
        }


class _MeshSession:
    """One PyMeshLab ``MeshSet`` held across multiple operations.

    Produced by :meth:`MeshOps.session`; avoids the load → save → reload
    round-trip (and its precision/attribute loss) when composing ops.
    Ops chain (each returns the session); nothing touches disk until
    :meth:`save`. Exiting the context drops the MeshSet reference.
    """

    def __init__(self, pml, ms, source_path: str) -> None:
        self._pml = pml
        self._ms = ms
        self.source_path = source_path

    def __enter__(self) -> "_MeshSession":
        return self

    def __exit__(self, *exc) -> bool:
        self._ms = None
        return False

    @property
    def mesh_set(self):
        """The underlying ``pymeshlab.MeshSet`` (raw escape hatch)."""
        return self._ms

    def measure(self) -> Dict[str, Any]:
        """Metrics for the current mesh (see :meth:`MeshOps.measure`)."""
        return _MeshOpsInternal._measures(self._ms)

    def op(self, name: str, **params) -> "_MeshSession":
        """Run one curated :data:`OPS` operator by name, validated."""
        _MeshOpsInternal._run_op(self._pml, self._ms, name, params)
        return self

    def apply(self, filter_name: str, **params) -> "_MeshSession":
        """Run a curated op when ``filter_name`` is in :data:`OPS`,
        otherwise any raw PyMeshLab filter — **unvalidated**."""
        if filter_name in OPS:
            return self.op(filter_name, **params)
        getattr(self._ms, filter_name)(**params)
        return self

    def save(self, output_path: str) -> str:
        """Write the current mesh; returns the (postflighted) path."""
        out = os.path.abspath(_MeshOpsInternal._check_save_ext(output_path))
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        self._ms.save_current_mesh(out)
        return _MeshOpsInternal._postflight(out)


class MeshOps(HelpMixin, _MeshOpsInternal):
    """File-level mesh processing via PyMeshLab (path in → path out).

    Optional-dependency gated: :meth:`available` is False without
    ``pythontk[mesh]``, and every public method raises the actionable
    install note via :meth:`resolve`. Callers that prefer a soft skip
    check :meth:`available` first (the extapps photogrammetry stages do).
    """

    OPS = OPS

    # ------------------------------------------------------------ gating
    @classmethod
    def resolve(cls, required: bool = True):
        """Return the ``pymeshlab`` module, or explain how to install it.

        Parameters:
            required: When True (default) a missing module raises
                RuntimeError with install instructions; when False,
                returns None instead.
        """
        try:
            import pymeshlab
        except ImportError:
            if required:
                raise RuntimeError(cls._install_note())
            return None
        error = cls._mesh_io_error(pymeshlab)
        if error:
            if required:
                raise RuntimeError(
                    f"pymeshlab is installed but cannot write meshes: {error}\n"
                    "On Linux its IO plug-ins need libOpenGL.so.0 "
                    "(Debian/Ubuntu: apt install libopengl0)."
                )
            return None
        return pymeshlab

    @classmethod
    def available(cls) -> bool:
        """True when the pymeshlab engine imports AND can read/write meshes."""
        return cls.resolve(required=False) is not None

    @staticmethod
    def _install_note() -> str:
        """Actionable message for a missing engine, aimed at the *current* interpreter."""
        import sys

        return (
            "The 'pymeshlab' package is not installed in this interpreter.\n"
            f'Install it with:  "{sys.executable}" -m pip install pythontk[mesh]\n'
            f"(GPL-3 licensed binary wheels on PyPI: {PYMESHLAB_PYPI_URL})"
        )

    # ------------------------------------------------------------ session
    @classmethod
    def session(cls, input_path: str) -> _MeshSession:
        """Open a :class:`_MeshSession` on ``input_path`` (context manager).

        Example:
            >>> with MeshOps.session(src) as s:  # doctest: +SKIP
            ...     before = s.measure()
            ...     s.op("remesh_isotropic", targetlen=1.0)
            ...     s.save(dst)
        """
        pml = cls.resolve()
        src = cls._preflight(input_path)
        ms = pml.MeshSet()
        ms.load_new_mesh(src)
        return _MeshSession(pml, ms, src)

    # ------------------------------------------------------------ measure
    @classmethod
    def measure(cls, input_path: str) -> Dict[str, Any]:
        """Geometry + topology metrics for a mesh file.

        Returns a flat dict shaped for ``QcGate.check``: ``faces``,
        ``vertices``, ``edges``, ``components``, ``boundary_edges``,
        ``non_two_manifold_edges``, ``non_two_manifold_vertices``,
        ``holes``, ``unreferenced_vertices``, ``genus``, ``two_manifold``,
        ``surface_area``, ``volume``, ``avg_edge_length``, ``bbox_diag``.
        Unmeasurable values are ``None``, never 0.
        """
        with cls.session(input_path) as s:
            return s.measure()

    @classmethod
    def compare(
        cls,
        input_path: str,
        reference_path: str,
        sample_num: int = 50000,
        heatmap_path: Optional[str] = None,
        heatmap_range: Optional[float] = None,
    ) -> Dict[str, Any]:
        """How far ``input_path`` strays from ``reference_path``, measured both ways.

        The deviation metric for "did that decimation/remesh/LOD/tessellation
        keep the shape?" — run it input=derived, reference=original. Each
        surface is sampled (its vertices and area-weighted points on its faces,
        up to ``sample_num`` of each) and every sample's distance to the other
        surface measured, so a coarse face's sag, a part the input lost and a
        part the reference lacks all count. No sample is dropped for being far:
        the search reaches across both meshes. The sampling is seeded, so the
        same two meshes always measure the same: a gate's verdict never flips
        on a rerun. A peak is the largest distance among the samples, so it
        approaches the true peak from below as ``sample_num`` grows.

        Keys (distances in the files' own units; names say ``peak``, never
        ``max``, because gate rule prefixes are stripped with an unanchored
        replace):

        - ``hausdorff_peak``: the largest distance either way.
        - ``hausdorff_mean`` / ``hausdorff_rms``: the larger direction's.
        - ``hausdorff_peak_pct``: the peak as a percentage of the reference's
          bounding-box diagonal (comparable across assets).
        - ``hausdorff_peak_at``: ``[x, y, z]`` where the peak was sampled.
        - ``input_to_reference_peak``: the input's surface away from the
          reference (geometry the reference lacks, a bulge).
        - ``reference_to_input_peak``: the reference's surface away from the
          input (geometry the input lost, a sag).
        - ``samples``: the samples measured, both directions together.
        - ``reference_bbox_diag``, ``heatmap``, ``heatmap_range``.

        Parameters:
            input_path: The derived mesh.
            reference_path: The mesh it should match.
            sample_num: Face samples per direction (vertex samples are capped
                at the same count, an even stride through them), which bounds
                the cost on any mesh size.
            heatmap_path: When given, write the denser of the two meshes there
                with each vertex coloured by its distance to the other (viridis:
                dark at 0, yellow at ``heatmap_range``); PLY keeps the colours.
            heatmap_range: The distance the ramp tops out at (a budget makes it
                read as "how close to failing"); default the peak.
        """
        if int(sample_num) < 1:
            raise ValueError(f"sample_num is the face samples per direction: at least 1, not {sample_num}.")
        if heatmap_range is not None and not float(heatmap_range) > 0.0:
            raise ValueError(f"heatmap_range is the distance the ramp tops out at: above 0, not {heatmap_range}.")
        pml = cls.resolve()
        src = cls._preflight(input_path)
        ref = cls._preflight(reference_path)
        if heatmap_path:
            heatmap_path = cls._check_save_ext(os.path.abspath(heatmap_path))
        ms = pml.MeshSet()
        ms.load_new_mesh(src)  # id 0
        ms.load_new_mesh(ref)  # id 1
        reach = pml.PureValue(2.0 * cls._union_diagonal(ms.mesh(0), ms.mesh(1)) or 1.0)
        directions = {}
        for key, (sampled, target) in (
            ("input_to_reference", (0, 1)),
            ("reference_to_input", (1, 0)),
        ):
            points = cls._surface_samples(ms.mesh(sampled), int(sample_num))
            directions[key] = cls._distances(pml, ms, points, target, reach)
        fwd, back = directions["input_to_reference"], directions["reference_to_input"]
        worst = fwd if fwd["peak"] >= back["peak"] else back
        ref_diag = ms.mesh(1).bounding_box().diagonal() or None
        result = {
            "hausdorff_peak": worst["peak"],
            "hausdorff_mean": max(fwd["mean"], back["mean"]),
            "hausdorff_rms": max(fwd["rms"], back["rms"]),
            "hausdorff_peak_pct": (worst["peak"] / ref_diag) * 100.0 if ref_diag else None,
            "hausdorff_peak_at": worst["at"],
            "input_to_reference_peak": fwd["peak"],
            "reference_to_input_peak": back["peak"],
            "samples": fwd["samples"] + back["samples"],
            "reference_bbox_diag": ref_diag,
            "heatmap": None,
            "heatmap_range": None,
        }
        if heatmap_path:
            top = float(heatmap_range) if heatmap_range is not None else (worst["peak"] or 1.0)
            dense, other = (
                (0, 1) if ms.mesh(0).vertex_number() >= ms.mesh(1).vertex_number() else (1, 0)
            )
            ms.compute_scalar_by_distance_from_another_mesh_per_vertex(
                measuremesh=dense, refmesh=other, signeddist=False, maxdist=reach
            )
            ms.set_current_mesh(dense)
            ms.compute_color_from_scalar_per_vertex(minval=0.0, maxval=top, colormap="Viridis")
            os.makedirs(os.path.dirname(heatmap_path) or ".", exist_ok=True)
            ms.save_current_mesh(heatmap_path, save_vertex_color=True)
            result["heatmap"] = cls._postflight(heatmap_path)
            result["heatmap_range"] = top
        return result

    # ------------------------------------------------------------ repair
    @classmethod
    def clean(
        cls,
        input_path: str,
        output_path: Optional[str] = None,
        merge_distance: float = 1e-5,
        remove_isolated_pieces_diameter_percent: float = 5.0,
        fill_holes_max_edge_count: int = 500,
        decimate_target_faces: int = 0,
    ) -> str:
        """Repair / clean a mesh file; return the output path.

        The canonical "prepare for downstream DCC" chain — duplicate-vertex
        merge, close-vertex weld, isolated-piece pruning, non-manifold edge
        repair, optional hole-fill, optional quadric decimation. Each step
        is opt-out via parameter.

        Parameters:
            input_path: Source mesh file.
            output_path: Destination. Defaults to ``<stem>_clean<ext>``.
            merge_distance: Weld vertices within this **absolute** distance
                (scene units). 0 = skip.
            remove_isolated_pieces_diameter_percent: Drop islands smaller
                than this % of bbox diagonal. 0 = skip.
            fill_holes_max_edge_count: Fill holes up to this many edges.
                0 = skip.
            decimate_target_faces: Quadric edge-collapse target. 0 = skip
                decimation (preserve source density).
        """
        out = cls._resolve_output(input_path, output_path, "clean")
        with cls.session(input_path) as s:
            s.op("remove_duplicate_vertices")
            s.op("remove_unreferenced_vertices")
            if merge_distance > 0:
                s.op("merge_close_vertices", threshold=merge_distance)
            if remove_isolated_pieces_diameter_percent > 0:
                s.op(
                    "remove_isolated_pieces",
                    mincomponentdiag=remove_isolated_pieces_diameter_percent,
                )
            s.op("repair_non_manifold_edges")
            if fill_holes_max_edge_count > 0:
                try:
                    s.op("close_holes", maxholesize=fill_holes_max_edge_count)
                except Exception as e:
                    logger.warning(f"close_holes failed: {e}")
            if decimate_target_faces > 0:
                s.op(
                    "decimate_quadric",
                    targetfacenum=decimate_target_faces,
                    preserveboundary=True,
                    preservenormal=True,
                    preservetopology=True,
                )
            s.save(out)
        logger.info(f"Mesh cleaned: {input_path} -> {out}")
        return out

    # ------------------------------------------------------------ remesh
    @classmethod
    def remesh(
        cls,
        input_path: str,
        output_path: Optional[str] = None,
        target_edge_pct: float = 1.0,
        iterations: int = 10,
        adaptive: bool = False,
    ) -> str:
        """Isotropic explicit remesh toward a uniform edge length.

        The evenness pass photogrammetry meshes need *before* quadric
        decimation — collapse on wildly uneven density gives poor results.
        ``target_edge_pct`` is the target edge length as a percent of the
        bbox diagonal; ``adaptive`` relaxes uniformity toward curvature.
        """
        out = cls._resolve_output(input_path, output_path, "remeshed")
        with cls.session(input_path) as s:
            s.op(
                "remesh_isotropic",
                targetlen=target_edge_pct,
                iterations=iterations,
                adaptive=adaptive,
            )
            s.save(out)
        logger.info(f"Mesh remeshed: {input_path} -> {out}")
        return out

    # ------------------------------------------------------------ decimate
    @classmethod
    def decimate(
        cls,
        input_path: str,
        output_path: Optional[str] = None,
        target_faces: int = 0,
        target_pct: float = 0.0,
        curvature_weighted: bool = False,
        preserve_boundary: bool = True,
        preserve_normal: bool = True,
        preserve_topology: bool = False,
        quality_threshold: float = 0.3,
    ) -> str:
        """Quadric edge-collapse decimation to a face target.

        Exactly one of ``target_faces`` / ``target_pct`` must be set.
        ``curvature_weighted=True`` first bakes per-vertex ABS curvature
        into vertex quality and collapses with ``qualityweight`` — the
        Decimation-Master-style adaptive density that spends the budget on
        high-curvature areas. ``preserve_topology`` defaults False: on scan
        meshes it routinely blocks the collapse from reaching its target.
        """
        if bool(target_faces) == bool(target_pct):
            raise ValueError("Set exactly one of target_faces / target_pct.")
        out = cls._resolve_output(input_path, output_path, "decimated")
        with cls.session(input_path) as s:
            if curvature_weighted:
                s.op("curvature_scalar")
            s.op(
                "decimate_quadric",
                targetfacenum=int(target_faces),
                targetperc=float(target_pct) / 100.0 if target_pct else 0.0,
                qualitythr=quality_threshold,
                preserveboundary=preserve_boundary,
                preservenormal=preserve_normal,
                preservetopology=preserve_topology,
                qualityweight=curvature_weighted,
                autoclean=True,
            )
            s.save(out)
        logger.info(f"Mesh decimated: {input_path} -> {out}")
        return out

    # ------------------------------------------------------------ bake
    @classmethod
    def bake_vertex_color(
        cls,
        input_path: str,
        output_path: Optional[str] = None,
        texture_size: int = 1024,
        border: int = 2,
    ) -> Tuple[str, str]:
        """Bake per-vertex color to a texture on auto-generated UVs.

        The fully headless textured-mesh path for dense vertex-colored
        meshes (SuGaR output, dense scans): trivial per-wedge
        parametrization → per-vertex attribute transfer. Returns
        ``(mesh_path, texture_path)``; the mesh defaults to
        ``<stem>_baked.obj`` (OBJ+MTL carry the texture binding) with the
        PNG beside it. The parametrization is a bake carrier, not
        production UVs — use :class:`pythontk.UvUnwrap` for real layouts.
        """
        out = cls._resolve_output(input_path, output_path, "baked", ext=".obj")
        out = os.path.abspath(out)
        tex_name = os.path.splitext(os.path.basename(out))[0] + ".png"
        with cls.session(input_path) as s:
            if not s.mesh_set.current_mesh().has_vertex_color():
                raise ValueError(f"Mesh has no per-vertex color to bake: {input_path}")
            s.apply(
                "compute_texcoord_parametrization_triangle_trivial_per_wedge",
                textdim=texture_size,
                border=border,
            )
            # Save first: transfer writes the texture relative to the
            # current mesh's path, so the output dir must be anchored.
            s.save(out)
            s.apply(
                "transfer_attributes_to_texture_per_vertex",
                sourcemesh=0,
                targetmesh=0,
                attributeenum="Vertex Color",
                textname=tex_name,
                textw=texture_size,
                texth=texture_size,
            )
            s.save(out)  # rewrite OBJ+MTL now referencing the texture
        tex_path = cls._postflight(os.path.join(os.path.dirname(out), tex_name))
        logger.info(f"Vertex color baked: {input_path} -> {out} + {tex_path}")
        return out, tex_path

    # ------------------------------------------------------------ escape
    @classmethod
    def apply(
        cls,
        input_path: str,
        filter_name: str,
        output_path: Optional[str] = None,
        **params,
    ) -> str:
        """One filter on a file: curated (validated) when ``filter_name``
        is in :data:`OPS`, otherwise any raw PyMeshLab filter by name —
        unvalidated, and version-dependent by nature."""
        out = cls._resolve_output(input_path, output_path, "out")
        with cls.session(input_path) as s:
            s.apply(filter_name, **params)
            s.save(out)
        return out
