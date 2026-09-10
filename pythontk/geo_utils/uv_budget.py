# !/usr/bin/python
# coding=utf-8
"""UV texture-budget planning: how many maps, at what texel density (numbers in -> plan out).

The *planning* sibling of :class:`pythontk.UvPack`. Where that class packs real
islands with a real engine, this one answers the question asked **before** any
pack runs, and answers it in microseconds instead of seconds:

    "These surfaces, at this texel density, need how many maps of this size?"
    ...and its inverse: "Given this many maps, what density do I get?"

Nothing here touches geometry, a DCC, or the packer. The input is a list of
:class:`BudgetItem` — an area, a border length, a chart count and optionally an
authored density, per indivisible group — so any adapter that can measure a
mesh can plan with it. Per-item density is what lets one solve cover content
whose sets were authored at different densities: the solvers then scale every
group together rather than flattening them onto a single number.

Why a model instead of just packing
-----------------------------------
A real pack costs seconds per attempt, and answering "how many maps?" means
attempting many. The demand model below is a closed form: it prices a group at
a given density without placing anything, so a full solve is a bisection over
arithmetic. That makes the two inverse questions, and a table of alternates
around the answer, cheap enough to show *before* the user commits to anything.

The demand model
----------------
At ``tpu`` texels per world unit, a group's footprint in a map is not just its
scaled area — every chart also carries a gutter that does not shrink with it::

    demand_px2 = area * tpu**2          # the surfaces themselves
               + pad * perimeter * tpu  # a gutter strip along every border
               + 4 * pad**2 * charts    # the corner squares that strip misses

with ``pad`` the island gutter in pixels and ``perimeter`` the chart border
length in world units. All three terms are px^2, so demand is directly
comparable to a page's ``map_size**2`` of capacity.

Charging every border a full strip over-counts where two charts abut and split
the space between them. A parameter for that was built and then REMOVED: swept
against real packs over four sharing values and eight fill values, the
worst-case margin moved 4% across the whole sharing range against 32% across
fill, so it was a second knob measuring nothing that ``fill`` does not already
absorb.

The quadratic term dominates at high density and the constant term at low
density, which is the whole reason chart *count* matters: 400 small charts and
one large chart of equal area do not cost the same map, and a model that priced
only area would promise a density the packer cannot deliver on fragmented
input. The gutter itself is not a free parameter — it comes from
:meth:`pythontk.MathUtils.calculate_uv_padding`, the same rule the Maya and
RizomUV packers derive their spacing from, so a plan made here and a pack run
later agree on what a gutter is.

What the model does NOT know
----------------------------
It prices demand, not placement. Two effects push in opposite directions and
neither is modelled exactly:

- Real packers interlock concave charts and beat a pure area sum.
- No packer achieves 100% fill, and a page holding one oversized chart wastes
  whatever the chart's bounding box does not cover.

Both are absorbed by ``fill`` (the usable fraction of a page), whose default is
calibrated against real packs rather than guessed — see
:attr:`UvBudget.FILL_DEFAULT`. It stays a parameter rather than a hidden
literal because a plan is a *prediction*, and the honest form of a prediction
names its assumption where the caller can change it.

Packing is NP-hard, so both solvers use the standard approximations with known
bounds rather than pretending to be optimal:

- **First Fit Decreasing** for "how many pages?" — at most ``11/9 * OPT + 6/9``
  pages. Descending order is what makes the bound hold; the large groups that
  constrain the solution are placed while every page is still empty.
- **Longest Processing Time** for "spread across exactly N pages" — the fullest
  page is at most ``4/3 - 1/(3N)`` times optimal. Used when the page count is
  fixed and the goal is even pages rather than few pages.

Both are ``O(n log n)`` and deterministic: the same input always plans the same
way, which matters more here than the last percent of tightness, because the
plan is shown to a person who will re-run it and expect the same answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pythontk.core_utils.help_mixin import HelpMixin
from pythontk.math_utils._math_utils import MathUtils


@dataclass(frozen=True)
class BudgetItem:
    """One indivisible group of surfaces competing for map space.

    "Indivisible" is the caller's call, and it is the single most consequential
    choice in a plan: an item never splits across pages, so item granularity is
    what decides whether a material may span two maps or must fit one. An
    adapter typically offers per-mesh items (a mesh stays whole, a material may
    span maps) or per-material items (a material stays whole, at the cost of
    coarser packing).

    Attributes:
        key: Caller's identifier, echoed back in the page assignment.
        area: World-space surface area of the group, in squared world units.
            This is the *unique* area — a group whose UV shells are stacked or
            instanced occupies one copy of map space, so the adapter must
            divide out that multiplicity before building the item.
        perimeter: Total chart border length, in world units. Sets the gutter
            cost. Zero prices the group as if its charts had no borders, which
            under-predicts demand; measure it rather than defaulting it.
        charts: Number of separate charts (UV islands) in the group. Sets the
            corner cost, and is what makes fragmented input price higher than
            consolidated input of the same area.
        density: This group's OWN target texel density, in texels per world
            unit. Left at 1.0 — the usual case — the solvers' ``density``
            argument is simply the density everything is priced at. Set it per
            item and that argument becomes a uniform SCALE on all of them, so
            1.0 means "every group keeps exactly the density it already has".
            That is the only way to budget content whose sets were authored at
            different densities without flattening them onto one number, and
            flattening is rarely right: a set authored at 11 px/unit and one at
            126 differ by 127x in the map area one shared density would cost.
        payload: Opaque caller data (member node names, a source map size)
            carried through the plan untouched.
    """

    key: str
    area: float
    perimeter: float = 0.0
    charts: int = 1
    density: float = 1.0
    # Excluded from equality (and therefore from the frozen hash): it is
    # whatever the caller stapled on, so two items describing the same demand
    # are the same item regardless of it -- and including it would make an
    # item carrying a dict or a dataclass unhashable, which is a confusing
    # failure in a frozen type that otherwise promises to be a value.
    payload: object = field(default=None, compare=False)

    def demand(self, tpu: float, pad: float) -> float:
        """Pixel-squared footprint at *tpu* (scaling :attr:`density`) and *pad* px gutter."""
        t = self.density * tpu
        return (
            self.area * t * t
            + pad * self.perimeter * t
            + 4.0 * pad * pad * max(self.charts, 0)
        )


@dataclass
class BudgetPage:
    """One map in a plan, and what landed on it."""

    index: int
    keys: List[str] = field(default_factory=list)
    demand: float = 0.0
    capacity: float = 0.0

    @property
    def fill(self) -> float:
        """Fraction of usable page area this page's items occupy (0-1)."""
        return self.demand / self.capacity if self.capacity > 0 else 0.0


@dataclass
class BudgetRow:
    """One candidate answer: a map size and count, and the density it buys.

    Rows are the unit of comparison in a plan — the chosen configuration and
    every alternate are the same shape, so a UI can table them directly.
    """

    map_size: int
    pages: int
    density: float
    padding: float
    fill: float
    items: int = 0
    page_list: List[BudgetPage] = field(default_factory=list)
    feasible: bool = True
    scaled: bool = False
    """Whether :attr:`density` is a multiplier rather than texels per unit.

    Set when the items carried their own :attr:`BudgetItem.density`, in which
    case the solve scaled all of them together and the answer is "0.62x what
    the content already has", not "0.62 texels per unit". Only the row knows,
    so only the row can render itself honestly.
    """

    note: str = ""
    """Why the row is infeasible, or — on a feasible row — what it could not do.

    ``feasible`` disambiguates the two: a note on a feasible row is an aside
    (levelling declined because it would have overflowed a page), not a
    failure.
    """

    @property
    def assignment(self) -> Dict[str, int]:
        """Item key -> page index."""
        return {k: p.index for p in self.page_list for k in p.keys}

    @property
    def utilization(self) -> float:
        """Mean fill across pages — how much of the paid-for map area is used."""
        if not self.page_list:
            return 0.0
        return sum(p.demand for p in self.page_list) / sum(
            p.capacity for p in self.page_list
        )

    @property
    def worst_fill(self) -> float:
        """Fill of the emptiest page.

        The number that catches the failure everyone actually hits: a plan whose
        mean utilization looks healthy while one page holds a single small
        object and wastes an entire map's memory and streaming budget.
        """
        return min((p.fill for p in self.page_list), default=0.0)

    def underfilled(self, threshold: float = 0.5) -> List[int]:
        """Indices of pages filled below *threshold*."""
        return [p.index for p in self.page_list if p.fill < threshold]

    @property
    def texels(self) -> int:
        """Total texels the row spends, across every page."""
        return self.pages * self.map_size * self.map_size

    def __str__(self) -> str:
        if not self.feasible:
            return f"{self.pages}x{self.map_size} - infeasible ({self.note})"
        value = (
            f"{self.density:.3f}x authored"
            if self.scaled
            else f"{self.density:.1f} px/unit"
        )
        return (
            f"{self.pages}x{self.map_size} @ {value}, "
            f"{self.utilization:.0%} used (worst page {self.worst_fill:.0%})"
        )


@dataclass
class BudgetPlan:
    """A chosen row plus the alternates around it, ready to show before committing."""

    chosen: BudgetRow
    alternates: List[BudgetRow] = field(default_factory=list)
    items: List[BudgetItem] = field(default_factory=list)
    target_density: Optional[float] = None

    @property
    def rows(self) -> List[BudgetRow]:
        """Chosen + alternates, ordered by total texels spent."""
        return sorted([self.chosen, *self.alternates], key=lambda r: r.texels)

    @property
    def density_ratio(self) -> Optional[float]:
        """Achieved over asked; None when the solve was driven by page count.

        Below 1.0 the plan under-delivers what was requested. Whether *above*
        1.0 is a win depends on where the target came from: against a target
        measured from the source textures it is not, since density beyond what
        those textures carry is magnified pixels and spends map area to store
        no new detail. Against an arbitrary target it simply means the budget
        had room to spare.
        """
        if not self.target_density:
            return None
        return self.chosen.density / self.target_density

    def __str__(self) -> str:
        lines = [f"chosen: {self.chosen}"]
        lines += [f"   alt: {r}" for r in self.alternates]
        return "\n".join(lines)


class UvBudget(HelpMixin):
    """Plan a UV texture budget: maps needed, density achieved, and the alternates.

    Read-only arithmetic over measured numbers. Nothing here packs, mutates, or
    needs an engine installed — :class:`pythontk.UvPack` does that, and a plan
    from this class is what tells a caller whether to bother.
    """

    FILL_DEFAULT = 0.65
    """Usable fraction of a page assumed when the caller does not say.

    Absorbs everything the closed form cannot see: chart bounding boxes that do
    not tessellate, the packer's own placement heuristics, and the border inset
    a page needs.

    **Calibrated, not chosen.** Planned page assignments were packed for real
    (xatlas via ``mayatk.UvUtils.pack_uvs``, 2 pages at 1024) across three
    content shapes that fail differently — varied primitives, 24 small
    fragments where per-chart gutter cost dominates, and a lopsided set where
    one large object forces a page. The achieved-over-predicted density ratio
    at each fill, worst case across all three::

        0.60 -> 1.081    0.70 -> 0.998
        0.65 -> 1.037    0.75 -> 0.963
                         0.85 -> 0.903

    0.65 is the highest value that stays at or above 1.0 everywhere, so it is
    the default: a plan should under-promise, because a density the packer
    misses is discovered only after the repack. 0.85 would over-promise by
    ~10%, which on a 7-map job is a map that does not fit.

    Raise it toward 1.0 to plan optimistically on content known to pack well;
    lower it for more headroom. It scales page capacity linearly, so it is the
    one knob to turn after comparing a plan against a real pack of the same
    content.
    """

    MIN_TPU = 1e-6
    MAX_BISECT_STEPS = 60

    # ------------------------------------------------------------------ #
    # Gutter
    # ------------------------------------------------------------------ #
    @staticmethod
    def padding_for(
        map_size: int, factor: int = 256, mip_levels: int = 0
    ) -> Tuple[float, bool]:
        """Island gutter in pixels for *map_size*, floored by the mip requirement.

        The ratio rule (:meth:`pythontk.MathUtils.calculate_uv_padding`) scales
        the gutter with the map, which keeps a layout's *normalized* spacing
        constant across resolutions. Mipmapping does not scale that way: each
        mip level halves the map, so a gutter of ``g`` pixels survives only
        ``log2(g)`` levels before neighbouring islands average together and
        bleed. A chain of ``mip_levels`` levels therefore needs at least
        ``2**mip_levels`` pixels of gutter regardless of map size, and on small
        maps that floor is what binds, not the ratio.

        Parameters:
            map_size: Page size in pixels.
            factor: Divisor for the ratio rule; the shipped 256 gives 16px at
                4096. Larger means a tighter gutter.
            mip_levels: Mip levels that must stay bleed-free. 0 disables the
                floor entirely (no mip chain, or bleed handled by dilation).

        Returns:
            tuple[float, bool]: The gutter in pixels, and whether the mip floor
            raised it above the ratio rule. A True flag is worth surfacing —
            it means the plan is paying for mip safety, and that a smaller map
            would pay proportionally more.
        """
        ratio = MathUtils.calculate_uv_padding(map_size, factor=factor)
        floor = float(2**mip_levels) if mip_levels > 0 else 0.0
        return (floor, True) if floor > ratio else (ratio, False)

    # ------------------------------------------------------------------ #
    # Bin packing
    # ------------------------------------------------------------------ #
    @staticmethod
    def first_fit_decreasing(
        sizes: Sequence[Tuple[str, float]], capacity: float
    ) -> Optional[List[List[str]]]:
        """Fewest pages holding every item, by First Fit Decreasing.

        Items are placed largest-first into the first page with room. The
        descending sort is what earns the ``11/9 * OPT + 6/9`` bound — placing
        the constraining items while pages are still empty — and it is also
        what makes the result stable, since ties break on the caller's order.

        Parameters:
            sizes: ``(key, demand)`` pairs. Demand and *capacity* must share
                units (px^2 throughout this module).
            capacity: Usable demand one page can hold.

        Returns:
            list[list[str]] | None: Keys per page, or None when any single item
            exceeds *capacity* — that item cannot fit any page at this density,
            so there is no valid packing to report.
        """
        if capacity <= 0:
            return None
        pages: List[List[str]] = []
        remaining: List[float] = []
        for key, size in sorted(sizes, key=lambda kv: -kv[1]):
            if size > capacity:
                return None
            for i, free in enumerate(remaining):
                if free >= size:
                    pages[i].append(key)
                    remaining[i] = free - size
                    break
            else:
                pages.append([key])
                remaining.append(capacity - size)
        return pages

    @staticmethod
    def partition_lpt(
        sizes: Sequence[Tuple[str, float]], pages: int
    ) -> List[List[str]]:
        """Spread items across exactly *pages* pages, by Longest Processing Time.

        Each item, largest first, goes to whichever page is currently emptiest.
        This minimizes the *fullest* page (within ``4/3 - 1/(3N)`` of optimal),
        which is the right objective once the page count is already fixed: the
        fullest page is what caps the density every page shares.

        Use :meth:`first_fit_decreasing` instead when the goal is fewest pages.
        The two disagree by design — FFD leaves a tail page nearly empty, LPT
        levels every page — and which is wanted depends on whether the caller
        is buying maps or has already bought them.
        """
        if pages < 1:
            return []
        buckets: List[List[str]] = [[] for _ in range(pages)]
        loads = [0.0] * pages
        for key, size in sorted(sizes, key=lambda kv: -kv[1]):
            i = loads.index(min(loads))
            buckets[i].append(key)
            loads[i] += size
        return buckets

    # ------------------------------------------------------------------ #
    # Solve: density -> pages, and pages -> density
    # ------------------------------------------------------------------ #
    @classmethod
    def pages_at_density(
        cls,
        items: Sequence[BudgetItem],
        density: float,
        map_size: int,
        *,
        factor: int = 256,
        mip_levels: int = 0,
        fill: Optional[float] = None,
        level: bool = False,
    ) -> BudgetRow:
        """Fewest maps that hold *items* at *density* texels per world unit.

        The forward question: the density is non-negotiable (it matches the
        source textures) and the map count is whatever that costs.

        Parameters:
            items: Indivisible groups to place.
            density: Target texels per world unit — or, when items carry their
                own :attr:`BudgetItem.density`, a uniform scale on those (1.0
                preserves each group's authored density exactly).
            map_size: Page size in pixels.
            factor: Gutter divisor, see :meth:`padding_for`.
            mip_levels: Mip levels that must stay bleed-free.
            fill: Usable page fraction; None uses :attr:`FILL_DEFAULT`.
            level: Re-balance the solved page count with LPT so pages fill
                evenly instead of leaving FFD's near-empty tail page. Never
                changes the page *count*, only what sits on each.

        Returns:
            BudgetRow: ``feasible`` is False when one item alone exceeds a
            page at this density — the note names it, since that item is the
            thing to split or down-res, and no other page count fixes it.
        """
        fill = cls.FILL_DEFAULT if fill is None else fill
        pad, _ = cls.padding_for(map_size, factor=factor, mip_levels=mip_levels)
        capacity = map_size * map_size * fill
        sizes = [(it.key, it.demand(density, pad)) for it in items]

        row = BudgetRow(
            map_size=map_size,
            pages=0,
            density=density,
            padding=pad,
            fill=fill,
            items=len(items),
            scaled=any(it.density != 1.0 for it in items),
        )
        packed = cls.first_fit_decreasing(sizes, capacity)
        if packed is None:
            worst = max(sizes, key=lambda kv: kv[1], default=("", 0.0))
            row.feasible = False
            row.note = (
                f"'{worst[0]}' needs {worst[1] / capacity:.1f} pages on its own "
                f"at {density:.1f} px/unit"
            )
            return row

        by_key = dict(sizes)
        if level and packed:
            # LPT minimizes the FULLEST page, which is not the same as keeping
            # every page inside capacity, and the gap is reachable rather than
            # theoretical: 5,5,4,4,3,3,3 fits three pages of 9 exactly, while
            # LPT puts 5+3+3 on one and overflows to 11. Levelling is a
            # presentation preference, so a levelled layout that does not fit
            # is simply discarded in favour of the FFD one that does.
            even = cls.partition_lpt(sizes, len(packed))
            if all(sum(by_key[k] for k in page) <= capacity for page in even):
                packed = even
                row.note = ""
            else:
                row.note = "pages left uneven - levelling them overflowed a page"

        row.pages = len(packed)
        row.page_list = [
            BudgetPage(
                index=i,
                keys=list(keys),
                demand=sum(by_key[k] for k in keys),
                capacity=capacity,
            )
            for i, keys in enumerate(packed)
        ]
        return row

    @classmethod
    def density_at_pages(
        cls,
        items: Sequence[BudgetItem],
        pages: int,
        map_size: int,
        *,
        factor: int = 256,
        mip_levels: int = 0,
        fill: Optional[float] = None,
        level: bool = True,
    ) -> BudgetRow:
        """Highest density at which *items* fit in *pages* maps of *map_size*.

        The inverse question: the map count is the fixed budget (an engine
        limit, a memory target) and the density is whatever that buys. With
        items carrying their own :attr:`BudgetItem.density`, the answer is a
        uniform scale on those instead — below 1.0 means the budget cannot hold
        the content at its authored densities, and says by how much.

        Demand rises monotonically with density and page count rises with
        demand, so the answer is a bisection: the largest density whose FFD
        solution still fits within *pages*. The search is bounded above by the
        density at which raw area alone would fill every page — padding can
        only add demand, so no feasible density exceeds it.

        Returns:
            BudgetRow: ``feasible`` is False when even a vanishing density
            needs more than *pages* pages — with the gutter and corner terms
            fixed per chart, sheer chart count can exceed the budget no matter
            how small each chart is drawn. Fewer, larger charts is then the
            only fix, and the note says so.
        """
        fill = cls.FILL_DEFAULT if fill is None else fill
        pad, _ = cls.padding_for(map_size, factor=factor, mip_levels=mip_levels)
        capacity = map_size * map_size * fill
        total_area = sum(it.area * it.density * it.density for it in items)

        def fits(tpu: float) -> bool:
            packed = cls.first_fit_decreasing(
                [(it.key, it.demand(tpu, pad)) for it in items], capacity
            )
            return packed is not None and len(packed) <= pages

        if not items or total_area <= 0:
            return BudgetRow(
                map_size=map_size,
                pages=0,
                density=0.0,
                padding=pad,
                fill=fill,
                feasible=False,
                scaled=any(it.density != 1.0 for it in items),
                note="no measurable surface area at any density",
            )
        if not fits(cls.MIN_TPU):
            fixed = sum(4.0 * pad * pad * it.charts for it in items)
            return BudgetRow(
                map_size=map_size,
                pages=pages,
                density=0.0,
                padding=pad,
                fill=fill,
                items=len(items),
                feasible=False,
                scaled=any(it.density != 1.0 for it in items),
                note=(
                    f"gutters alone need {fixed / capacity:.1f} pages "
                    f"({sum(it.charts for it in items)} charts at {pad:.0f}px) "
                    "- consolidate charts or raise the page count"
                ),
            )

        # Upper bound: the scale at which surfaces alone would fill every page.
        # Gutters only add, so nothing above this can fit. Each item's own
        # density is folded in here, or a scene of mixed densities would be
        # bounded as if every group sat at 1.0 and the search would start far
        # below the answer.
        hi = math.sqrt(pages * capacity / total_area)
        lo = cls.MIN_TPU
        # Bisection assumes `fits` is monotone. Demand per item is monotone in
        # tpu, but FFD is a heuristic, so a higher density could in principle
        # pack luckier and make it briefly non-monotone. That costs at worst a
        # slightly conservative answer, never a wrong one: whatever comes out
        # is re-solved below, so the row always reports a density its own
        # assignment actually fits.
        if fits(hi):
            lo = hi
        else:
            for _ in range(cls.MAX_BISECT_STEPS):
                mid = 0.5 * (lo + hi)
                if fits(mid):
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-4 * max(hi, 1.0):
                    break

        # Re-solve at the answer so the row's pages/assignment are the ones
        # that density actually produces, not the bisection's last probe.
        return cls.pages_at_density(
            items,
            lo,
            map_size,
            factor=factor,
            mip_levels=mip_levels,
            fill=fill,
            level=level,
        )

    # ------------------------------------------------------------------ #
    # Plan
    # ------------------------------------------------------------------ #
    @classmethod
    def plan(
        cls,
        items: Sequence[BudgetItem],
        *,
        map_size: int = 4096,
        density: Optional[float] = None,
        pages: Optional[int] = None,
        factor: int = 256,
        mip_levels: int = 0,
        fill: Optional[float] = None,
        level: bool = False,
        alternates: bool = True,
        map_sizes: Optional[Iterable[int]] = None,
    ) -> BudgetPlan:
        """Solve in whichever direction the caller pinned, and table the neighbours.

        Exactly one of *density* or *pages* drives the solve; the other is the
        answer. Alternates are the configurations a person actually weighs next
        — one map size either side, and one page count either side — each
        solved on its own terms so the efficiency figures are comparable rather
        than extrapolated.

        Parameters:
            items: Indivisible groups to place.
            map_size: Page size in pixels for the chosen row.
            density: Target texels per world unit. Solves for page count.
            pages: Target map count. Solves for density.
            factor: Gutter divisor, see :meth:`padding_for`.
            mip_levels: Mip levels that must stay bleed-free.
            fill: Usable page fraction; None uses :attr:`FILL_DEFAULT`.
            level: Re-balance pages with LPT once the count is known.
            alternates: Build the neighbouring rows. False solves one row only.
            map_sizes: Sizes offered as alternates; None uses one power-of-two
                step either side of *map_size*.

        Returns:
            BudgetPlan: The chosen row, the alternates, and the items solved.

        Raises:
            ValueError: Neither or both of *density* and *pages* were given —
                the direction of the solve has to be unambiguous.
        """
        if (density is None) == (pages is None):
            raise ValueError("pass exactly one of 'density' or 'pages'")
        # Reject rather than solve: zero pages has a defined answer (a density
        # of nothing) that reads like a real one downstream. The realistic way
        # to arrive here is feeding back an infeasible row's `pages`, and that
        # caller needs the error, not a plan promising 0.0 px/unit.
        if pages is not None and pages < 1:
            raise ValueError(f"'pages' must be at least 1, got {pages}")

        kw = dict(factor=factor, mip_levels=mip_levels, fill=fill)
        if density is not None:
            chosen = cls.pages_at_density(items, density, map_size, level=level, **kw)
        else:
            chosen = cls.density_at_pages(items, pages, map_size, level=True, **kw)

        plan = BudgetPlan(chosen=chosen, items=list(items), target_density=density)
        if not alternates:
            return plan

        sizes = (
            list(map_sizes)
            if map_sizes is not None
            else [s for s in (map_size // 2, map_size * 2) if s >= 16]
        )
        rows: List[BudgetRow] = []
        if density is not None:
            # Same density, other map sizes: the trade is page count against
            # per-page memory, and only a solve can say which way it lands.
            rows += [
                cls.pages_at_density(items, density, s, level=level, **kw)
                for s in sizes
                if s != map_size
            ]
            # Same map size, one page either side: what a map costs or saves
            # in density. One page fewer is the interesting one -- it is the
            # only row that can be cheaper than the chosen one.
            if chosen.feasible:
                rows += [
                    cls.density_at_pages(items, p, map_size, level=True, **kw)
                    for p in (chosen.pages - 1, chosen.pages + 1)
                    if p >= 1
                ]
        else:
            rows += [
                cls.density_at_pages(items, pages, s, level=True, **kw)
                for s in sizes
                if s != map_size
            ]
            rows += [
                cls.density_at_pages(items, p, map_size, level=True, **kw)
                for p in (pages - 1, pages + 1)
                if p >= 1
            ]
        # De-duplicate against the chosen row and each other: solving one page
        # either side routinely lands back on the same configuration (asking
        # for 11 pages when 10 suffice re-solves to 10), and a table listing
        # the same answer twice reads as two different options.
        seen = {(chosen.map_size, chosen.pages)}
        plan.alternates = []
        for row in rows:
            config = (row.map_size, row.pages)
            if not row.feasible or config in seen:
                continue
            seen.add(config)
            plan.alternates.append(row)
        return plan
