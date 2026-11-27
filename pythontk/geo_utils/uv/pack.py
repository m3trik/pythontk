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
  never re-cut, and each moves as one rigid, uniformly scaled piece. Their
  relative scale is kept only approximately: on production meshes (hundreds
  to thousands of islands) islands of 16 px and more land within a few
  percent of the median scale (median 1-4%, at most ~9%), and islands of a
  few texels are scaled up toward the engine's least chart size (2026-11-24).
  Charts are found by UV *position*, though: coincident vertices weld, so
  islands touching along a seam would merge into one gutterless chart —
  :meth:`UvPack._separate_islands` moves each topological island clear of the
  others in the engine input first.
- A zero-area triangle is DROPPED by the engine, and charts are
  EDGE-connected while a DCC's islands are vertex-connected: an island held
  together only through a collapsed face, or pinched at a lone shared UV,
  came back as separate charts -- the face stretched across the page, corners
  only it used left at their input coordinates on top of packed islands
  (a production door: 53,237 texels inside two islands after a pack). Such an
  island is packed by its bounding box (:meth:`UvPack._degenerate_islands`,
  :meth:`UvPack._engine_input`) and moved by the similarity the box's corners
  took; every other island is checked for one rigid move after the pack and
  boxed for another round when it was not (``PackIslandsResult.boxed``).
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
from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Tuple

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
# The coarsest page the fixed-page search runs in full; a larger page starts
# from the scale found on one of half its size (coarse to fine).
_COARSE_LEAST = 512

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
    #: Per mesh, the islands packed by their bounds: those with a zero-area
    #: face (the engine drops it and could split the island through it), a
    #: collapsed one, and any the engine did not move as one piece.
    boxed: List[int] = field(default_factory=list)


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
    def _separate_islands(uvs, triangles):
        """Engine-input copy of *uvs* with every topological island in its own cell.

        The engine finds charts by UV *position*, not index: its colocal pass
        welds coincident vertices, so two islands touching along a cut seam --
        distinct indices, identical coordinates -- merge into one chart and come
        back with no gutter between them (measured: a cut sphere's shells 0.0
        apart at padding 4, and a caller's per-island write-back then drifted
        them into overlap). Where an island starts doesn't affect how it packs,
        so each vertex-connected island (the DCC definition of a shell) is
        translated onto a grid of cells one largest-island wide plus a gutter.
        Only the engine's input moves; pass-through rows keep the caller's
        coordinates. Islands stacked exactly on each other (mirrored halves
        sharing texels) come apart the same way: the weld used to fuse them
        into one chart, which kept them stacked.
        """
        import numpy as np

        from pythontk.math_utils._math_utils import MathUtils

        count = len(uvs)
        tri = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        island = np.asarray(MathUtils.connected_components(count, tri))
        islands = int(island.max()) + 1
        if islands < 2:
            return uvs

        lo = np.full((islands, 2), np.inf)
        hi = np.full((islands, 2), -np.inf)
        np.minimum.at(lo, island, uvs)
        np.maximum.at(hi, island, uvs)
        cell = float((hi - lo).max()) * 1.25 or 1.0
        columns = int(np.ceil(np.sqrt(islands)))
        slot = np.arange(islands)
        corner = np.column_stack([slot % columns, slot // columns]) * cell
        return uvs + (corner - lo)[island]

    @staticmethod
    def _triangle_uv_areas(uvs, triangles):
        """Each of *triangles*' area, indexed into the (N, 2) *uvs*: ``(M,)``."""
        import numpy as np

        p = np.asarray(uvs, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)]
        return (
            np.abs(
                (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
            )
            / 2.0
        )

    @classmethod
    def _uv_area(cls, uvs, triangles) -> float:
        """Total area of *triangles* indexed into the (N, 2) *uvs*."""
        return float(cls._triangle_uv_areas(uvs, triangles).sum())

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
        start: Optional[tuple] = None,
    ):
        """Largest-scale pack into <= *pages* square pages of *resolution*.

        xatlas's fixed-resolution mode spills into as many pages as the scale
        (``texels_per_unit``) demands, so the largest scale that still fits the
        page budget is found by bisection: ~8-12 engine runs from an area-based
        initial guess, on a page of :data:`_COARSE_LEAST` texels; a larger page
        starts from the scale the page of half its size found (coarse to fine:
        a production door at 2048, 14.4 s -> 5.6 s, the same fill within 1%).
        Every entry of *candidates* is searched independently and
        the one reaching the largest scale wins — the caller places islands into
        cells sized from the *page budget*, so delivered fill rises with scale
        alone and the largest scale is simply the tightest pack.

        *start*, ``(scale, candidate)`` from a round over nearly the same
        islands (a rigidity round boxes a few), searches that candidate alone
        from that scale in steps of a few percent; the full search runs only
        if it finds nothing.

        Returns:
            ``(atlas, scale, candidate)``.
        """
        area = sum(cls._uv_area(uv_arr, tri_arr) for uv_arr, tri_arr in arrays)
        if area <= 0.0:
            raise ValueError("Meshes have no UV area to pack.")

        def bisect(tpu, brute, variant, grow=1.25, shrink=0.5, iterations=16):
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
                    tpu = tpu * grow if hi is None else (lo + hi) / 2.0
                else:
                    hi = tpu
                    tpu = tpu * shrink if lo is None else (lo + hi) / 2.0
                if (
                    lo is not None
                    and hi is not None
                    and (hi - lo) / lo < _SEARCH_CONVERGENCE
                ):
                    break
            return lo, best

        if start is None and resolution >= 2 * _COARSE_LEAST:
            # Coarse to fine: the whole search on a page of half the size (a
            # run costs about a quarter), its scale doubled as this one's
            # start. Recursive, so only the coarsest page is searched in full.
            coarse = resolution // 2
            try:
                _atlas, found, winner = cls._fixed_page_generate(
                    xatlas,
                    arrays,
                    max(1, int(round(padding * coarse / resolution))),
                    rotate,
                    coarse,
                    pages,
                    candidates,
                )
                start = (found * resolution / coarse, winner)
            except RuntimeError:  # nothing fit the coarse page: search here
                pass
        if start is not None:
            scale, (variant, brute) = start
            found, found_atlas = bisect(scale, brute, variant, grow=1.03, shrink=0.96)
            if found is not None:
                return found_atlas, found, start[1]
        guess = (pages * resolution * resolution * _SEARCH_TARGET_FILL / area) ** 0.5
        lo = atlas = winner = None
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
                lo, atlas, winner = found, found_atlas, (variant, brute)
        if lo is None:
            raise RuntimeError(
                f"Could not fit the islands into {pages} page(s) of "
                f"{resolution} texels at any scale."
            )
        return atlas, lo, winner

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
        surface_areas: Optional[Sequence[Any]] = None,
    ) -> PackIslandsResult:
        """Pack every mesh's UV islands together.

        Parameters:
            meshes: One ``(uvs, triangles)`` pair per mesh — ``uvs`` an (N, 2)
                float array-like, ``triangles`` an (M, 3) integer array-like
                indexing into it. Islands are detected from the UV topology;
                their relative input scale is kept, approximately (see the
                module notes), unless *surface_areas* evens it.
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
            surface_areas: One ``(M,)`` array per mesh, each triangle's area on
                the surface (any one unit for all): every island is first
                scaled about its own centre so its UV area over its surface
                area matches every other's -- one texel density for the whole
                pack (Maya's ``u3dLayout -preScaleMode 1``), done here in one
                pass instead of a per-shell host edit before the pack. None
                keeps the islands' relative input scale, approximately.

        Returns:
            PackIslandsResult: packed UVs per mesh (input-index aligned) plus
            atlas stats; fixed-page mode also carries per-UV ``pages``. Every
            island moved as one piece: one with a collapsed face, a pinched or
            collapsed island, and any the engine split, is packed by its bounds
            (``boxed`` counts them per mesh), so no island lands on another.

        Raises:
            RuntimeError: xatlas missing (install note), the engine re-indexed
                a parametrized mesh in a way that can't be mapped back, the
                fixed-page search could not fit the page budget, or an island
                still did not move as one piece after
                :attr:`_RIGID_ROUNDS` rounds.
            ValueError: no meshes, a mesh with no UVs/triangles, a triangle
                indexing past its mesh's UVs, or zero UV area in fixed-page
                mode.
        """
        import numpy as np

        xatlas = cls.resolve(required=True)

        if not meshes:
            raise ValueError("No meshes to pack.")

        from pythontk.math_utils._math_utils import MathUtils

        plans = []
        for i, (uvs, tris) in enumerate(meshes):
            uv_arr = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
            tri_arr = np.asarray(tris, dtype=np.uint32).reshape(-1, 3)
            if not len(uv_arr) or not len(tri_arr):
                raise ValueError(f"Mesh {i} has no UVs or no triangles.")
            if int(tri_arr.max()) >= len(uv_arr):
                raise ValueError(
                    f"Mesh {i}: a triangle indexes past its {len(uv_arr)} UVs."
                )
            island = np.asarray(
                MathUtils.connected_components(len(uv_arr), tri_arr.astype(np.int64))
            )
            plans.append(
                {
                    "uvs": uv_arr,
                    "tris": tri_arr,
                    "island": island,
                    "boxed": cls._degenerate_islands(uv_arr, tri_arr, island),
                }
            )
        if surface_areas is not None:
            if len(surface_areas) != len(plans):
                raise ValueError(
                    f"{len(surface_areas)} surface-area arrays for {len(plans)} meshes."
                )
            cls._even_density(plans, surface_areas)
        for plan in plans:
            plan["separated"] = cls._separate_islands(
                plan.get("evened", plan["uvs"]), plan["tris"]
            )

        candidates = cls._candidates(cls._variants(rotate, align_to_axis), brute_force)
        fixed = resolution > 0
        # An island the engine did not move as one piece packs again by its
        # bounds: each round boxes the ones the last could not keep rigid. Not
        # every such island can be told up front (an island the engine keeps
        # as one chart can still come back non-rigid), so a later round starts
        # from the last one's scale and candidate instead of searching anew.
        start = None
        for _round in range(cls._RIGID_ROUNDS):
            arrays = [cls._engine_input(plan) for plan in plans]
            if fixed:
                atlas, scale, winner = cls._fixed_page_generate(
                    xatlas,
                    [(a[0], a[1]) for a in arrays],
                    padding,
                    rotate,
                    resolution,
                    pages,
                    candidates,
                    start=start,
                )
                start = (scale, winner)
                width = height = int(resolution)
                to_unit = np.ones(2, dtype=np.float64)
                extent = (1.0, 1.0)
            else:
                atlas = max(
                    (
                        cls._generate(
                            xatlas,
                            [(a[0], a[1]) for a in arrays],
                            padding,
                            rotate,
                            brute,
                            **variant,
                        )
                        for variant, brute in candidates
                    ),
                    key=cls._atlas_score,
                )
                width, height = int(atlas.width), int(atlas.height)
                longest = float(max(width, height)) or 1.0
                # De-normalize the per-axis 0-1 coordinates to texels, then
                # scale uniformly so aspect survives and the atlas fits the
                # unit square.
                to_unit = np.array([width / longest, height / longest], dtype=np.float64)
                extent = (width / longest, height / longest)

            read = [
                cls._read_back(atlas, i, plan, arrays[i], to_unit, fixed)
                for i, plan in enumerate(plans)
            ]
            failed = [r[3] for r in read]
            if not any(failed):
                break
            for plan, bad in zip(plans, failed):
                plan["boxed"] = plan["boxed"] | bad
        else:
            raise RuntimeError(
                "Some islands could not be packed as one piece even by their "
                f"bounds after {cls._RIGID_ROUNDS} rounds: "
                + ", ".join(f"mesh {i}: {len(r[3])}" for i, r in enumerate(read) if r[3])
            )

        return PackIslandsResult(
            uvs=[r[0] for r in read],
            written=[r[1] for r in read],
            width=width,
            height=height,
            utilization=float(np.mean(np.atleast_1d(atlas.utilization))),
            extent=extent,
            pages=[r[2] for r in read],
            page_count=int(atlas.atlas_count),
            boxed=[len(plan["boxed"]) for plan in plans],
        )

    @staticmethod
    def box_pages(
        map_size: int, coverage: Tuple[float, float] = (1.0, 1.0), margin: float = 0.0
    ) -> Tuple[int, int, Callable[[Any, Any], Any]]:
        """How a box of a 0-1 tile is packed as square pages.

        The box -- *coverage* ``(u, v)`` of the tile from its lower-left
        corner, inset by *margin* -- is split along its long axis into the
        whole number of near-square cells nearest its shape (a Full or Quarter
        box one, a half two), each holding one square page of the cell's short
        side in texels at *map_size* (at least 64). :meth:`pack_islands` fills
        exactly those pages (``resolution``, ``pages``), so padding is exact
        texture pixels; in content-driven mode the engine picks the atlas
        shape itself, and fitting that into the box wasted the mismatch (a
        6-shell cube filled 0.50 of a Full tile, 0.25 of a half).

        Returns:
            ``(pages, resolution, place)``: *place(uvs, page)* puts one mesh's
            page coordinates (:class:`PackIslandsResult` ``uvs`` and ``pages``)
            into the box, in tile units.

        Raises:
            ValueError: A coverage side is not positive.
        """
        import numpy as np

        cov_u, cov_v = float(coverage[0]), float(coverage[1])
        short_side, long_side = min(cov_u, cov_v), max(cov_u, cov_v)
        if short_side <= 0:
            raise ValueError(f"coverage must be positive, got {coverage}")
        pages = max(1, int(round(long_side / short_side)))
        resolution = max(64, int(round(map_size * short_side)))
        origin = np.array([margin, margin], dtype=np.float64)
        inner = np.array(
            [max(cov_u - 2 * margin, 1e-6), max(cov_v - 2 * margin, 1e-6)]
        )
        axis = 0 if cov_u >= cov_v else 1  # the long axis the cells stack along
        step = np.zeros(2)
        step[axis] = inner[axis] / pages
        cell = inner.copy()
        cell[axis] = step[axis]
        fit = float(cell.min())  # a square page, fitted uniformly into its cell

        def place(uvs, page):
            return (
                np.asarray(uvs, dtype=np.float64) * fit
                + origin
                + np.asarray(page).reshape(-1, 1) * step
            )

        return pages, resolution, place

    class CellBox(NamedTuple):
        """:meth:`UvPack.cell_box`'s answer: how one layout fills one cell."""

        #: The ``(u, v)`` fraction of the 0-1 tile to pack into
        #: (:meth:`UvPack.box_pages` ``coverage``): a Full tile, or a 1/n strip.
        coverage: Tuple[float, float]
        #: The factor the layout's u is scaled by before the pack (rotation
        #: then off, which would turn it); 1.0 for none.
        stretch: float
        #: The map size to pack at: the cell's long side, so a page texel is
        #: an atlas texel.
        map_size: int
        #: The gutter between the layout's own islands, in atlas texels.
        padding: int

    #: A layout re-packed to fill its cell (:meth:`cell_box`) is worth keeping
    #: only when it needs a cell below this share of the one it had: denser by
    #: 2%. Islands too many for their gutters (a box projection's thousands of
    #: shells) can come back covering less than the layout they replace.
    CELL_GAIN: float = 0.98

    @classmethod
    def cell_box(cls, width: int, height: int, atlas: int) -> "UvPack.CellBox":
        """How one layout is packed to fill a *width* x *height* px cell of an
        *atlas* px atlas (a lightmap's), through :meth:`box_pages`.

        The cell's shape is taken by the whole number of square pages nearest
        it, side by side along its long axis (``coverage``); the rest of the
        shape is the layout's ``stretch``, its u scaled before the pack, so a
        cell that takes its box's shape on the surface shows every texel
        square (none within 1%). ``map_size`` is the cell's long side, so the
        pages are exactly the cell's texels. ``padding`` keeps the islands
        three texels apart at 2048, scaled with the atlas, at least two:
        clear of bleed through the first mip (1.5 texels). A cell is atlas
        texels, so its gutter is the atlas's rule, never one scaled to the
        cell; it is narrower than the gutter between cells, since a
        neighbour's light is the worse leak (game engines' own lightmap
        packing margins default to about 4). Measured on a production cabinet at 2048: two
        texels gave +44% density, four +24%, the whole atlas gutter (8) +14%
        -- a box projection's 2,115 shells each spent it on every side.

        Parameters:
            width, height: The cell, in atlas texels (at least 1 each).
            atlas: The atlas's texels per side.

        Returns:
            :class:`UvPack.CellBox` ``(coverage, stretch, map_size, padding)``.
        """
        width, height = max(1, int(width)), max(1, int(height))
        aspect = width / float(height)
        pages = max(1, int(round(max(aspect, 1.0 / aspect))))
        wide = aspect >= 1.0
        box = float(pages) if wide else 1.0 / pages  # the coverage's own shape
        stretch = box / aspect
        return cls.CellBox(
            coverage=(1.0, 1.0 / pages) if wide else (1.0 / pages, 1.0),
            stretch=1.0 if abs(stretch - 1.0) <= 0.01 else stretch,
            map_size=max(width, height),
            padding=max(2, round(3 * int(atlas) / 2048)),
        )

    #: :meth:`island_overlaps`' probe size bounds, texels per side: 0.1-0.4 s
    #: on a 17,724-triangle production layout at 1024-2048.
    OVERLAP_PROBE = (256, 2048)

    @classmethod
    def island_overlaps(cls, meshes: Sequence[Tuple[Any, Any]], size: int = 1024) -> int:
        """Texels where triangles of two DIFFERENT islands meet -- what a pack
        must never leave, across every mesh given (they share one page).

        A fold inside one island is the layout's own and does not count: the
        pack moves each island as one piece, so it neither makes nor mends one.
        Every island is rasterized with a key of its own
        (``UvTransfer.build``, which counts a texel two triangles claim with
        different keys); triangles of one island share theirs, so a shared
        edge or a fold never counts.

        Parameters:
            meshes: ``(uvs, triangles)`` per mesh, as :meth:`pack_islands`
                takes them; UVs in the page's 0-1 (V up).
            size: The map's texels per side, probed at that size within
                :attr:`OVERLAP_PROBE`. Each texel is judged at its centre, so
                islands apart never count; an overlap under a probe texel can
                go unseen.

        Returns:
            int: Texel samples two islands claim; 0 for a correct pack.
        """
        import math

        import numpy as np

        from pythontk.geo_utils.uv.transfer import UvTransfer
        from pythontk.math_utils._math_utils import MathUtils

        layout, keys, base = [], [], 0
        for uvs, tris in meshes:
            uv_arr = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
            tri_arr = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
            if not len(tri_arr):
                continue
            island = np.asarray(MathUtils.connected_components(len(uv_arr), tri_arr))
            layout.append(uv_arr[tri_arr])
            keys.append(island[tri_arr] + base)
            base += int(island.max()) + 1
        if not layout:
            return 0
        # A 2D key grid: neighbouring keys sit 1/side apart, past the 1e-3 the
        # table tells two keys apart by while side <= 500. Past 250,000
        # islands the keys repeat (id modulo the grid): two islands sharing one
        # are told apart by nothing, so an overlap between exactly those two
        # goes unseen.
        side = min(max(2, math.ceil(math.sqrt(base))), 500)
        ids = np.concatenate(keys) % (side * side)
        key = np.stack([(ids % side + 0.5) / side, (ids // side + 0.5) / side], axis=-1)
        low, high = cls.OVERLAP_PROBE
        probe = int(min(max(int(size), low), high))
        table = UvTransfer.build(key, np.concatenate(layout), probe, supersample=1)
        return int(table.overlaps)

    #: Rounds of re-packing an island the engine did not keep rigid by its
    #: bounds. One is the rule (a degenerate face is boxed before the first);
    #: the rest catch what the engine split on its own. On a small page,
    #: islands of a few texels each round can surface new ones (a production
    #: locker door's 88 px lightmap cell: 3 left after 3 rounds); a later
    #: round starts warm, so it costs a few engine runs.
    _RIGID_ROUNDS: int = 6
    #: A triangle under this share of its island's bounds' area is degenerate
    #: to the engine, which drops it (float32: a collapsed face's corners).
    _DEGENERATE_AREA: float = 1e-9
    #: Max abs error (packed unit space) of an island's one similarity: a rigid
    #: move fits to ~2e-4 at most (float32 inside the engine; mayatk's measured
    #: write-back calibration, ``RESIDUAL_TOLERANCE`` there and in
    #: :mod:`~pythontk.geo_utils.uv.combine`), an island the engine split
    #: through a dropped face or a pinch misses by 1e-1 or more.
    _RIGID_TOLERANCE: float = 1e-3

    @staticmethod
    def _even_density(plans, surface_areas) -> None:
        """Scale every island of every plan about its own centre so its UV area
        over its surface area is the pack's median: one texel density across
        all of them. Sets each plan's ``evened`` UVs; an island with no UV or
        no surface area keeps its scale."""
        import numpy as np

        ratios = []
        for plan, areas in zip(plans, surface_areas):
            areas = np.asarray(areas, dtype=np.float64).reshape(-1)
            if len(areas) != len(plan["tris"]):
                raise ValueError(
                    f"{len(areas)} surface areas for {len(plan['tris'])} triangles."
                )
            uvs, tris, island = plan["uvs"], plan["tris"].astype(np.int64), plan["island"]
            uv_area = UvPack._triangle_uv_areas(uvs, tris)
            count = int(island.max()) + 1
            tri_island = island[tris[:, 0]]
            per_uv = np.bincount(tri_island, uv_area, count)
            per_surface = np.bincount(tri_island, np.abs(areas), count)
            ratio = np.zeros(count)
            ok = (per_uv > 1e-20) & (per_surface > 0)
            ratio[ok] = np.sqrt(per_surface[ok] / per_uv[ok])
            plan["_ratio"] = ratio
            ratios.append(ratio[ok])
        known = np.concatenate(ratios) if ratios else np.zeros(0)
        median = float(np.median(known)) if len(known) else 1.0
        for plan in plans:
            ratio, island, uvs = plan.pop("_ratio"), plan["island"], plan["uvs"]
            scale = np.where(ratio > 0, ratio / median, 1.0)
            count = len(scale)
            members = np.bincount(island, minlength=count).astype(np.float64)
            centre = np.column_stack(
                [np.bincount(island, uvs[:, k], count) for k in (0, 1)]
            ) / np.maximum(members, 1.0)[:, None]
            plan["evened"] = (uvs - centre[island]) * scale[island][:, None] + centre[island]

    @classmethod
    def _degenerate_islands(cls, uvs, tris, island) -> "set":
        """The islands the engine would not keep in one piece, packed by their
        bounds instead (:meth:`_engine_input`) so each moves as one:

        - one with a zero-area triangle: the engine drops the face, and an
          island held together through it came back as separate charts with
          the face stretched between them (or corners only it used left where
          they were);
        - one PINCHED, its triangles meeting at a lone shared UV: the engine's
          charts are edge-connected, so it packs the two sides apart without
          duplicating the shared UV (what mayatk's write-back used to refuse).
        """
        import numpy as np

        from pythontk.math_utils._math_utils import MathUtils

        area = cls._triangle_uv_areas(uvs, tris)
        count = int(island.max()) + 1
        lo = np.full((count, 2), np.inf)
        hi = np.full((count, 2), -np.inf)
        np.minimum.at(lo, island, uvs)
        np.maximum.at(hi, island, uvs)
        box = np.prod(np.maximum(hi - lo, 0.0), axis=1)
        tri_island = island[tris[:, 0].astype(np.int64)]
        repeated = (
            (tris[:, 0] == tris[:, 1])
            | (tris[:, 1] == tris[:, 2])
            | (tris[:, 0] == tris[:, 2])
        )
        degenerate = repeated | (area <= cls._DEGENERATE_AREA * box[tri_island]) | (
            area <= 1e-20
        )
        found = set(np.unique(tri_island[degenerate]).tolist())
        # Pinched: an island whose triangles are one piece through shared UVs
        # but several through shared EDGES. Triangles are joined through each
        # edge they share (its two UVs, either order).
        t = tris.astype(np.int64)
        edges = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
        edges.sort(axis=1)
        owner = np.tile(np.arange(len(t)), 3)
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges, owner = edges[order], owner[order]
        same = (edges[1:] == edges[:-1]).all(axis=1)
        links = np.column_stack([owner[:-1][same], owner[1:][same]])
        by_edge = np.asarray(
            MathUtils.connected_components(
                len(t), np.column_stack([links, links[:, :1]]) if len(links) else np.zeros((0, 3), np.int64)
            )
        )
        pieces = {}
        for isl, piece in zip(tri_island.tolist(), by_edge.tolist()):
            pieces.setdefault(isl, set()).add(piece)
        found |= {isl for isl, ps in pieces.items() if len(ps) > 1}
        return found

    @classmethod
    def _engine_input(cls, plan):
        """``(uvs, triangles, boxes)`` the engine packs for one mesh: every clean
        island's own triangles at its separated position, and each boxed
        island as a quad over its bounds (grown to a sliver's least size, so a
        collapsed island still has an area to place). *boxes* is
        ``{island: first corner row}``."""
        import numpy as np

        sep = plan["separated"]
        tris = plan["tris"]
        island = plan["island"]
        boxed = sorted(plan["boxed"])
        if not boxed:
            return sep.astype(np.float32), tris, {}
        tri_island = island[tris[:, 0].astype(np.int64)]
        keep = tris[~np.isin(tri_island, boxed)]
        largest = float(np.ptp(sep, axis=0).max()) or 1.0
        least = largest * 1e-3
        corners, quads, boxes = [], [], {}
        base = len(sep)
        for k, isl in enumerate(boxed):
            rows = sep[island == isl]
            lo, hi = rows.min(axis=0), rows.max(axis=0)
            mid, half = (lo + hi) / 2.0, np.maximum((hi - lo) / 2.0, least / 2.0)
            lo, hi = mid - half, mid + half
            first = base + 4 * k
            corners.append([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
            quads += [[first, first + 1, first + 2], [first, first + 2, first + 3]]
            boxes[isl] = first
        uvs = np.vstack([sep, np.asarray(corners, dtype=np.float64).reshape(-1, 2)])
        tris = np.vstack([keep, np.asarray(quads, dtype=np.uint32)]).astype(np.uint32)
        return uvs.astype(np.float32), tris, boxes

    @classmethod
    def _read_back(cls, atlas, mesh, plan, engine, to_unit, fixed):
        """``(uvs, written, pages, not_rigid)`` for one mesh out of *atlas*.

        Clean islands take the engine's coordinates; a boxed island moves by
        the similarity its box's corners did. *not_rigid* names the clean
        islands the engine did not move as one piece -- the next round boxes
        them -- and the rows of a boxed island are written with it.
        """
        import numpy as np

        eng_uvs, _eng_tris, boxes = engine
        vmapping, out_tris, new_uvs = atlas.get_mesh(mesh)
        vmapping = np.asarray(vmapping)
        if len(vmapping) != len(np.unique(vmapping)):
            raise RuntimeError(
                f"Mesh {mesh}: engine split a source vertex across charts; "
                "the result can't be mapped back onto the input indices."
            )
        count = len(plan["uvs"])
        out_all = np.array(eng_uvs, dtype=np.float64)
        out_all[vmapping] = np.asarray(new_uvs, dtype=np.float64) * to_unit
        page_all = np.zeros(len(eng_uvs), dtype=np.int64)
        if fixed and atlas.atlas_count > 1:
            tris_out = np.asarray(out_tris).reshape(-1, 3)
            page_of_vert = np.zeros(len(new_uvs), dtype=np.int64)
            for j in range(atlas.get_mesh_chart_count(mesh)):
                chart = atlas.get_mesh_chart(mesh, j)
                rows = np.asarray(chart.faces, dtype=np.int64)
                page_of_vert[np.unique(tris_out[rows].reshape(-1))] = int(
                    chart.atlas_index
                )
            page_all[vmapping] = page_of_vert

        sep, island = plan["separated"], plan["island"]
        out = np.array(plan["uvs"], dtype=np.float64)
        written_mask = np.zeros(count, dtype=bool)
        written_mask[vmapping[vmapping < count]] = True
        pages = np.zeros(count, dtype=np.int64)
        not_rigid = set()
        referenced = np.zeros(count, dtype=bool)
        referenced[plan["tris"].reshape(-1).astype(np.int64)] = True
        for isl in np.unique(island[referenced]):
            rows = np.flatnonzero(island == isl)
            if isl in boxes:
                first = boxes[isl]
                corner_rows = np.arange(first, first + 4)
                move = cls.solve_similarity(eng_uvs[corner_rows], out_all[corner_rows])
                out[rows] = cls.apply_similarity(sep[rows], move)
                pages[rows] = page_all[first]
                written_mask[rows] = True
                continue
            out[rows] = out_all[rows]
            pages[rows] = page_all[rows]
            *_, residual = cls.solve_similarity(sep[rows], out_all[rows])
            if not written_mask[rows].all() or residual > cls._RIGID_TOLERANCE:
                not_rigid.add(int(isl))
        # Unreferenced rows keep what the engine handed back for them.
        loose = ~referenced
        out[loose] = out_all[:count][loose]
        return out, np.flatnonzero(written_mask), pages, not_rigid

    # ------------------------------------------------------------ transforms
    @staticmethod
    def solve_similarity(old, new) -> tuple:
        """Least-squares 2D similarity (reflection-aware) mapping *old* -> *new*.

        The engine moves every island rigidly under one global scale, and it
        mirrors an island where that packs tighter (verified: a mirrored chart
        comes back as a perfect *reflected* similarity, det < 0 -- the same
        liberty Maya's u3dLayout takes), so both the proper and the U-flipped
        solve are tried and the better fit wins. A consumer recovers each
        island's move from its before/after coordinates with this, then
        applies it to anything that moves with the island
        (:meth:`apply_similarity`).

        Parameters:
            old: ``(N, 2)`` coordinates before.
            new: ``(N, 2)`` the same points after.

        Returns:
            ``(scale, angle_degrees, mirrored, c0, c1, residual)``: flip U
            about centroid ``c0`` when *mirrored*, rotate + scale about ``c0``,
            then translate ``c1 - c0``. *residual* is the max abs error over
            the points -- a rigid move fits to float noise, an island the
            engine split does not.
        """
        import math

        import numpy as np

        old = np.asarray(old, dtype=np.float64).reshape(-1, 2)
        new = np.asarray(new, dtype=np.float64).reshape(-1, 2)
        c0, c1 = old.mean(axis=0), new.mean(axis=0)
        d0, d1 = old - c0, new - c1
        denom = float((d0 * d0).sum())
        if denom < 1e-20:  # single-point / degenerate island: translate only
            return 1.0, 0.0, False, c0, c1, float(np.abs(d1).max(initial=0.0))

        def _solve(source):
            a = float((source * d1).sum())
            b = float((source[:, 0] * d1[:, 1] - source[:, 1] * d1[:, 0]).sum())
            scale = math.hypot(a, b) / denom
            angle = math.atan2(b, a)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            rebuilt = np.column_stack(
                [
                    scale * (source[:, 0] * cos_a - source[:, 1] * sin_a),
                    scale * (source[:, 0] * sin_a + source[:, 1] * cos_a),
                ]
            )
            return scale, angle, float(np.abs(rebuilt - d1).max())

        scale, angle, residual = _solve(d0)
        scale_m, angle_m, residual_m = _solve(d0 * np.array([-1.0, 1.0]))
        if residual_m < residual:
            return scale_m, math.degrees(angle_m), True, c0, c1, residual_m
        return scale, math.degrees(angle), False, c0, c1, residual

    @staticmethod
    def apply_similarity(points, transform):
        """*points* moved by a :meth:`solve_similarity` result.

        Parameters:
            points: ``(N, 2)`` coordinates.
            transform: ``(scale, angle_degrees, mirrored, c0, c1, ...)`` -- the
                solve's tuple (a trailing residual is ignored).

        Returns:
            ``(N, 2)`` float64: flipped in U about ``c0`` when mirrored,
            rotated and scaled about ``c0``, translated by ``c1 - c0``.
        """
        import math

        import numpy as np

        scale, angle, mirrored, c0, c1 = transform[:5]
        d = np.asarray(points, dtype=np.float64).reshape(-1, 2) - np.asarray(c0)
        if mirrored:
            d = d * np.array([-1.0, 1.0])
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        return np.column_stack(
            [
                scale * (d[:, 0] * cos_a - d[:, 1] * sin_a),
                scale * (d[:, 0] * sin_a + d[:, 1] * cos_a),
            ]
        ) + np.asarray(c1)
