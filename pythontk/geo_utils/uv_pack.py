# !/usr/bin/python
# coding=utf-8
"""UV island packing via the optional ``xatlas`` engine (arrays in -> arrays out).

The pack-only sibling of :class:`pythontk.UvUnwrap`: where that class round-trips
OBJ files through external unwrapper *executables*, this one packs existing UV
islands in-process through the ``xatlas`` Python bindings (MIT licensed;
``pip install xatlas`` — wheels on PyPI). The dependency is optional per the
package charter: everything imports lazily, and :meth:`UvPack.resolve` turns an
absent module into an actionable install message instead of an ImportError.

Two packing modes, chosen by ``resolution``:

- **Content-driven** (``resolution=0``, default): one atlas whose dimensions —
  including its aspect — the engine picks freely. Coordinates come back
  aspect-true, uniformly fitted to the unit square. Best when the consumer has
  no fixed target region; against a square tile the aspect mismatch is wasted
  space (measured: a 6-shell cube filled only 0.50 of the tile this way).
- **Fixed-page** (``resolution > 0``): the engine packs into square pages of
  exactly ``resolution`` texels, and the largest ``texels_per_unit`` that fits
  ``pages`` pages is found by search — so each page comes back edge-to-edge
  full, ready to map onto a square target cell. Per-island page assignment is
  read from the engine's charts.

Verified engine facts this wrapper relies on (xatlas 0.0.11):

- ``add_uv_mesh`` + ``generate`` packs the *given* parametrization — islands are
  detected from UV topology and never re-cut, and the islands' relative input
  scale is preserved exactly (pack-only mode applies one global scale).
- In content-driven mode, returned UVs are normalized 0-1 **per axis** of a
  generally non-square atlas, so equal UV distances are unequal in texture
  space; the wrapper de-normalizes by the atlas dimensions.
- ``get_mesh`` re-indexes vertices; its vertex mapping is scattered back so the
  returned arrays align 1:1 with the caller's input indices, and it maps every
  input vertex (even unreferenced ones), pinned by test.
- ``Chart.atlas_index`` + ``Chart.faces`` give each chart's page and triangle
  rows, which is how per-island page assignment is recovered.
- ``PackOptions.padding`` expands *each* chart, so two adjacent islands land
  ``2 x padding`` apart (measured: 17px at padding 8, 34px at 16) — the same
  per-shell semantics as Maya's ``u3dLayout -shellSpacing``, so one gutter rule
  feeds both engines. The engine insets content from the page border by only
  0.5px, so a border gutter is the caller's to add.
- ``rotate_charts_to_axis`` is **independent of** ``rotate_charts``: left at its
  ``True`` default it rotates every island to its convex-hull axis by an
  arbitrary angle even when ``rotate_charts`` is off (measured: 8/8 charts at
  93/-169/-147/142 degrees). It is therefore always set explicitly — a caller
  asking for no rotation must actually get none.
- ``bilinear`` is inert for ``add_uv_mesh`` input (identical fills across every
  case measured), and ``max_chart_size`` must stay 0 or the engine may split a
  chart and break the input-index mapping. Both are pinned rather than
  inherited so an engine default change cannot silently alter results.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from pythontk.core_utils.help_mixin import HelpMixin

XATLAS_PYPI_URL = "https://pypi.org/project/xatlas/"
XATLAS_REPO_URL = "https://github.com/jpcy/xatlas"

# Fixed-page search bounds: the initial-guess target fill and the bisection
# convergence tolerance.
#
# Convergence is a scale tolerance, so it costs roughly TWICE that in area —
# at the 0.02 used previously it discarded up to 4% fill, which is the same
# order as the spread between pack variants and so hid which one had actually
# won. Tightening it to 0.005 costs ~2 extra bisection steps per variant and
# recovers that fill; holding the variant fixed to isolate it, measured in Maya:
# 4 hard-surface meshes 0.604 -> 0.621 full tile and 0.558 -> 0.570 quarter,
# 4 organic meshes 0.615 -> 0.623 half tile.
_SEARCH_TARGET_FILL = 0.85
_SEARCH_CONVERGENCE = 0.005

# Hull-axis pre-rotation has no good fixed value -- which setting packs
# tightest flips with the island set (measured at 1024/padding 4: 24 mixed
# rects 0.702 -> 0.727 with it off, 24 irregular 0.522 -> 0.564 with it on), so
# it is SEARCHED rather than exposed: no user can predict which suits their
# mesh, and each variant costs one cheap non-brute run. Add a combination here
# to search it.
#
# ``block_align`` deliberately stays False and is NOT a variant. It packs
# cube-like content ~5% tighter, but it snaps each chart to a 4-texel grid, and
# a caller island the engine sees as two charts then has its halves snapped
# INDEPENDENTLY -- about 4 texels of relative drift, ~1e-3 in UV. That is
# enough to break a consumer reconstructing one rigid transform per island
# (mayatk's write-back rejected 2 of 3 real Maya scenes with exactly that
# residual). Re-enabling it needs the caller's island grouping so variants that
# break it can be discarded, not just a flag.
_PACK_VARIANTS = (
    {"align_to_axis": False, "block_align": False},
    {"align_to_axis": True, "block_align": False},
)


@dataclass
class PackIslandsResult:
    """Outcome of one :meth:`UvPack.pack_islands` run.

    ``uvs`` holds one array per input mesh, aligned 1:1 with that mesh's input
    UV indices. Content-driven mode returns aspect-true unit-square
    coordinates (the atlas's longer side spans ``[0, 1]``, the shorter
    ``[0, extent]``); fixed-page mode returns coordinates normalized 0-1
    within each island's own square page, with ``pages`` naming the page.

    ``written`` holds, per mesh, the indices the engine actually repositioned.
    xatlas 0.0.11 maps every input vertex — even one no triangle references —
    so in practice it covers the whole array; it is reported because any row
    NOT listed would still hold its **input** coordinates, and a consumer
    deriving a transform from before/after coordinates must fit on ``written``
    only. Mixing pass-through rows into such a fit corrupts it silently.
    """

    uvs: List[Any]  # one (N, 2) float array per input mesh
    written: List[Any]  # one index array per input mesh (engine-positioned rows)
    width: int  # atlas texel dimensions (fixed-page mode: resolution x resolution)
    height: int
    utilization: float  # engine fill ratio of the atlas rectangle (mean over pages)
    extent: Tuple[float, float]  # (u, v) span in unit space; max(extent) == 1.0
    pages: List[Any] = field(default_factory=list)  # per-uv page index, per mesh
    page_count: int = 1  # pages actually produced (1 in content-driven mode)


class UvPack(HelpMixin):
    """Pack existing UV islands with the optional ``xatlas`` engine.

    Mechanism-only: takes plain UV/triangle arrays, returns plain arrays —
    DCC adapters own extraction and write-back. Currently xatlas is the sole
    engine; adding another means adding a classmethod alongside
    :meth:`pack_islands`, not editing it.
    """

    @classmethod
    def resolve(cls, required: bool = True):
        """Return the ``xatlas`` module, or explain how to install it.

        Parameters:
            required: When True (default) a missing module raises RuntimeError
                with install instructions; when False, returns None instead.
        """
        try:
            import xatlas
        except ImportError:
            if required:
                raise RuntimeError(cls._install_note())
            return None
        return xatlas

    @classmethod
    def available(cls) -> bool:
        """True when the xatlas engine can be imported."""
        return cls.resolve(required=False) is not None

    @staticmethod
    def _install_note() -> str:
        """Actionable message for a missing engine, aimed at the *current* interpreter."""
        import sys

        return (
            "The 'xatlas' Python package is not installed in this interpreter.\n"
            f'Install it with:  "{sys.executable}" -m pip install --user xatlas\n'
            f"(MIT licensed; wheels on PyPI: {XATLAS_PYPI_URL})"
        )

    @staticmethod
    def _variants(rotate: bool, align_to_axis: Optional[bool]) -> List[dict]:
        """The :data:`_PACK_VARIANTS` combinations still in play.

        ``rotate=False`` drops every hull-axis variant — that flag rotates
        islands on its own, so leaving it free would break the no-rotation
        promise — and an explicit *align_to_axis* pins it instead of searching.

        Raises:
            ValueError: ``align_to_axis=True`` with ``rotate=False``. Honouring
                either one silently would contradict the other, and the whole
                point of setting this flag is that the engine's default hides
                exactly that contradiction.
        """
        if align_to_axis and not rotate:
            raise ValueError(
                "align_to_axis=True contradicts rotate=False: aligning islands "
                "to their hull axis IS a rotation. Pass rotate=True, or leave "
                "align_to_axis as None."
            )
        return [
            v
            for v in _PACK_VARIANTS
            if (align_to_axis is None or v["align_to_axis"] == align_to_axis)
            and (rotate or not v["align_to_axis"])
        ]

    @staticmethod
    def _candidates(variants: Sequence[dict], brute_force: bool) -> List[tuple]:
        """``(variant, brute)`` pairs to search, best result wins.

        Brute placement is a different packer, not a refinement of the default
        one, so it *competes* with plain placement instead of replacing it —
        which is what makes asking for it unable to return a looser pack, and
        lets it reorder which variant wins.
        """
        pairs = [(v, False) for v in variants]
        if brute_force:
            pairs += [(v, True) for v in variants]
        return pairs

    @staticmethod
    def _uv_area(uvs, triangles) -> float:
        """Total area of *triangles* indexed into the (N, 2) *uvs*."""
        import numpy as np

        p = np.asarray(uvs, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)]
        return float(
            np.abs(
                (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
            ).sum()
            / 2.0
        )

    @staticmethod
    def _atlas_score(atlas) -> float:
        """Fraction of the atlas's BOUNDING SQUARE that its islands cover.

        Content-driven results are fitted uniformly into a square, so a tightly
        packed but very oblong atlas throws that fit away — engine utilization
        alone would rank it first anyway.
        """
        import numpy as np

        longest = float(max(atlas.width, atlas.height)) or 1.0
        return (
            float(np.mean(np.atleast_1d(atlas.utilization)))
            * atlas.width
            * atlas.height
            / (longest * longest)
        )

    @staticmethod
    def _generate(
        xatlas,
        arrays: Sequence[Tuple[Any, Any]],
        padding: int,
        rotate: bool,
        brute_force: bool,
        resolution: int = 0,
        texels_per_unit: float = 0.0,
        align_to_axis: bool = False,
        block_align: bool = False,
    ):
        """One engine run over *arrays* (a fresh Atlas each time — xatlas
        atlases are single-shot).

        Every ``PackOptions`` field is set explicitly; see the module docstring
        for why each is pinned rather than left at the engine's default.
        """
        atlas = xatlas.Atlas()
        for uv_arr, tri_arr in arrays:
            atlas.add_uv_mesh(uv_arr, tri_arr)
        options = xatlas.PackOptions()
        options.padding = int(padding)
        options.rotate_charts = bool(rotate)
        # Independent of rotate_charts in the engine, and True by default —
        # unset, it rotates islands by arbitrary angles even with rotation off.
        options.rotate_charts_to_axis = bool(rotate and align_to_axis)
        options.blockAlign = bool(block_align)
        options.bruteForce = bool(brute_force)
        options.bilinear = False  # inert for add_uv_mesh input; pinned anyway
        options.max_chart_size = 0  # a split chart breaks the index mapping
        options.create_image = False  # coordinates only — never rasterize
        if resolution:
            options.resolution = int(resolution)
        if texels_per_unit:
            options.texels_per_unit = float(texels_per_unit)
        atlas.generate(pack_options=options, verbose=False)
        return atlas

    @classmethod
    def _fixed_page_generate(
        cls,
        xatlas,
        arrays,
        padding: int,
        rotate: bool,
        resolution: int,
        pages: int,
        candidates: Sequence[tuple],
    ):
        """Largest-scale pack into <= *pages* square pages of *resolution*.

        xatlas's fixed-resolution mode spills into as many pages as the scale
        (``texels_per_unit``) demands, so the largest scale that still fits the
        page budget is found by bisection: ~8-12 engine runs from an area-based
        initial guess. Every entry of *candidates* is searched independently and
        the one reaching the largest scale wins — the caller places islands into
        cells sized from the *page budget*, so delivered fill rises with scale
        alone and the largest scale is simply the tightest pack.
        """
        area = sum(cls._uv_area(uv_arr, tri_arr) for uv_arr, tri_arr in arrays)
        if area <= 0.0:
            raise ValueError("Meshes have no UV area to pack.")

        def bisect(tpu, brute, variant, iterations=16):
            """Grow/shrink until bracketed, then bisect.

            Returns ``(largest fitting scale, its atlas)``, or ``(None, None)``
            if nothing fit.
            """
            lo = hi = best = None
            for _ in range(iterations):
                atlas = cls._generate(
                    xatlas, arrays, padding, rotate, brute, resolution, tpu, **variant
                )
                if atlas.atlas_count <= pages:
                    lo, best = tpu, atlas
                    tpu = tpu * 1.25 if hi is None else (lo + hi) / 2.0
                else:
                    hi = tpu
                    tpu = tpu * 0.5 if lo is None else (lo + hi) / 2.0
                if (
                    lo is not None
                    and hi is not None
                    and (hi - lo) / lo < _SEARCH_CONVERGENCE
                ):
                    break
            return lo, best

        guess = (pages * resolution * resolution * _SEARCH_TARGET_FILL / area) ** 0.5
        lo = atlas = None
        for variant, brute in candidates:
            # Every candidate searches from the SAME guess. Seeding later ones
            # from the running best looks like a free speed-up and is not:
            # whether a scale fits is not monotonic (placement is discrete, so
            # a larger scale sometimes fits where a smaller one did not), and a
            # seed that fails becomes an upper bracket the bisection can never
            # climb back over. That capped better candidates below their true
            # optimum and made the search pick the loser — measured on a
            # 4-mesh Maya scene, a candidate reaching 377 was reported as 365.6.
            found, found_atlas = bisect(guess, brute, variant)
            if found is not None and (lo is None or found > lo):
                lo, atlas = found, found_atlas
        if lo is None:
            raise RuntimeError(
                f"Could not fit the islands into {pages} page(s) of "
                f"{resolution} texels at any scale."
            )
        return atlas

    @classmethod
    def pack_islands(
        cls,
        meshes: Sequence[Tuple[Any, Any]],
        padding: int = 4,
        rotate: bool = True,
        brute_force: bool = False,
        resolution: int = 0,
        pages: int = 1,
        align_to_axis: Optional[bool] = None,
    ) -> PackIslandsResult:
        """Pack every mesh's UV islands together.

        Parameters:
            meshes: One ``(uvs, triangles)`` pair per mesh — ``uvs`` an (N, 2)
                float array-like, ``triangles`` an (M, 3) integer array-like
                indexing into it. Islands are detected from the UV topology;
                their relative input scale is preserved.
            padding: Gutter between islands, in atlas texels. Exact pixels in
                fixed-page mode (the page size is known); approximate in
                content-driven mode (the atlas is content-sized).
            rotate: Allow the packer to re-orient islands where that packs
                tighter. False means *no* rotation at all — including the
                engine's arbitrary-angle hull-axis pre-rotation, which is on by
                default and ignores its 90-degree sibling.
            brute_force: Also search exhaustive placement. It competes with the
                default packer rather than replacing it, so this can only hold
                or improve the result. It doubles the *number* of engine runs
                but costs far more than double in time, since each brute run is
                itself much slower — measured 2-15x the total pack.
            resolution: 0 (default) = content-driven single atlas, aspect
                chosen by the engine. > 0 = fixed-page mode: square pages of
                exactly this many texels, packed edge-to-edge full via a
                scale search — the mode to use against a square target region.
            pages: Fixed-page mode only: the page budget. The scale search
                maximizes island scale subject to fitting this many pages
                (e.g. 2 pages to fill a half-tile region with two stacked
                square cells).
            align_to_axis: Pins the engine's hull-axis pre-rotation instead of
                searching it. None (default) tries both, keeping whichever
                packs tighter — no fixed value wins on all content, so this is
                normally best left alone. Ignored when *rotate* is False, which
                forces it off.

        Returns:
            PackIslandsResult: packed UVs per mesh (input-index aligned) plus
            atlas stats; fixed-page mode also carries per-UV ``pages``.

        Raises:
            RuntimeError: xatlas missing (install note), the engine re-indexed
                a parametrized mesh in a way that can't be mapped back, or the
                fixed-page search could not fit the page budget.
            ValueError: no meshes, a mesh with no UVs/triangles, or zero UV
                area in fixed-page mode.
        """
        import numpy as np

        xatlas = cls.resolve(required=True)

        if not meshes:
            raise ValueError("No meshes to pack.")

        arrays = []
        for i, (uvs, tris) in enumerate(meshes):
            uv_arr = np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
            tri_arr = np.asarray(tris, dtype=np.uint32).reshape(-1, 3)
            if not len(uv_arr) or not len(tri_arr):
                raise ValueError(f"Mesh {i} has no UVs or no triangles.")
            arrays.append((uv_arr, tri_arr))

        candidates = cls._candidates(cls._variants(rotate, align_to_axis), brute_force)
        fixed = resolution > 0
        if fixed:
            atlas = cls._fixed_page_generate(
                xatlas, arrays, padding, rotate, resolution, pages, candidates
            )
            width = height = int(resolution)
            to_unit = np.ones(2, dtype=np.float64)
            extent = (1.0, 1.0)
        else:
            atlas = max(
                (
                    cls._generate(xatlas, arrays, padding, rotate, brute, **variant)
                    for variant, brute in candidates
                ),
                key=cls._atlas_score,
            )
            width, height = int(atlas.width), int(atlas.height)
            longest = float(max(width, height)) or 1.0
            # De-normalize the per-axis 0-1 coordinates to texels, then scale
            # uniformly so aspect survives and the atlas fits the unit square.
            to_unit = np.array([width / longest, height / longest], dtype=np.float64)
            extent = (width / longest, height / longest)

        packed, written, page_arrays = [], [], []
        for i, (uv_arr, _) in enumerate(arrays):
            vmapping, out_tris, new_uvs = atlas.get_mesh(i)
            vmapping = np.asarray(vmapping)
            if len(vmapping) != len(np.unique(vmapping)):
                raise RuntimeError(
                    f"Mesh {i}: engine split a source vertex across charts; "
                    "the result can't be mapped back onto the input indices."
                )
            out = np.array(uv_arr, dtype=np.float64)
            out[vmapping] = np.asarray(new_uvs, dtype=np.float64) * to_unit
            packed.append(out)
            written.append(vmapping)

            # Per-input-UV page index, recovered from the engine's charts.
            page_of_vert = np.zeros(len(new_uvs), dtype=np.int64)
            if fixed and atlas.atlas_count > 1:
                tris_out = np.asarray(out_tris).reshape(-1, 3)
                for j in range(atlas.get_mesh_chart_count(i)):
                    chart = atlas.get_mesh_chart(i, j)
                    rows = np.asarray(chart.faces, dtype=np.int64)
                    verts = np.unique(tris_out[rows].reshape(-1))
                    page_of_vert[verts] = int(chart.atlas_index)
            page_arr = np.zeros(len(uv_arr), dtype=np.int64)
            page_arr[vmapping] = page_of_vert
            page_arrays.append(page_arr)

        return PackIslandsResult(
            uvs=packed,
            written=written,
            width=width,
            height=height,
            utilization=float(np.mean(np.atleast_1d(atlas.utilization))),
            extent=extent,
            pages=page_arrays,
            page_count=int(atlas.atlas_count),
        )
