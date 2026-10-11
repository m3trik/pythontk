# !/usr/bin/python
# coding=utf-8
"""Rectangle packing into one fixed bin: integer texels, gutters, block alignment.

The tile-placement kernel under the ecosystem's texture atlases (sizes in,
integer positions out; no DCC, no image IO, no numpy) -- the placement half of
the atlas family whose layout, inset and assembly helpers live on
:class:`~pythontk.ImgUtils` (``_atlas.py``, :meth:`ImgUtils.pack_atlas_rects`
is the atlas-shaped front door). It sits in ``img_utils`` rather than
``geo_utils/uv`` because the atlas helpers import it and ``geo_utils`` ranks
above ``img_utils`` in this package's layer order; a UV island packer reaches
down to it. Two verbs:

- :meth:`RectPacker.pack` -- place rectangles of given texel sizes, or report
  that they do not fit.
- :meth:`RectPacker.fit` -- the largest UNIFORM scale at which rectangles of
  given natural extents (any consistent unit -- world units for a lightmap,
  UV units for an island set) fit, and their placement there. One scale for
  every rectangle is what keeps texel density equal across an atlas, and each
  keeps its own aspect: nothing is stretched to fill a cell.

Why a packer and not a subdivision. A squarified treemap
(:meth:`pythontk.ImgUtils.compute_atlas_layout`) fills the whole square by
bending each cell's SHAPE away from its content's: a square lightmap layout
pushed into a 2:1 cell gets two texel densities, and the cell is only as good
as the coarser one (a production floor: 0.6 x 1.2 cm per texel). Its gutter
is an inset that a small cell cannot afford, so at a low resolution
neighbouring cells meet with no gutter at all, and a cell under one texel
vanished while its object still sampled the atlas there.

Mechanics (all integers, image orientation: ``x`` right, ``y`` DOWN from the
top row):

- **Skyline, bottom-left with a waste tie-break** (Jylanki, *A Thousand Ways
  to Pack the Bin*): each rectangle goes where its far edge lands shallowest,
  then where it strands the least area under it, then leftmost. Fast -- one
  pass over the skyline per rectangle -- and tight on the many-similar-sizes
  sets atlases are made of (instances are identical rectangles).
- **Padding** is free space reserved around every rectangle on every side:
  two rectangles are at least ``2 * padding`` apart and each is at least
  ``padding`` from the bin's edge. It is the room a dilation fills so
  bilinear taps and mip levels read the rectangle's own colour.
- **Align**: every SLOT (rectangle plus padding) starts on, and spans, whole
  ``align x align`` blocks. Block-compressed formats (BC6H, BC7, ASTC, UASTC)
  encode 4 x 4 texel blocks; a block shared by two rectangles mixes them in
  the encode, a seam no gutter hides. ``align=4`` gives every block one
  owner, and the first two mip levels with it.
- **Deterministic**: equal sizes keep the caller's order, so a set the caller
  sorts by name always lays out the same way, and :meth:`RectPacker.fit`
  treats extents within :attr:`RectPacker.TIE_TOLERANCE` as equal -- copies of
  one instanced mesh never trade places between bakes over float noise in
  their areas.
"""

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Callable, List, Optional, Sequence, Tuple

from pythontk.core_utils.help_mixin import HelpMixin

__all__ = ["RectPacker", "RectPackResult"]


@dataclass(frozen=True)
class RectPackResult:
    """One packing: where each rectangle's CONTENT landed, in input order.

    Attributes:
        positions: ``(x, y)`` of each rectangle's top-left content texel; ``y``
            counts rows DOWN from the bin's top (image orientation).
        sizes: ``(width, height)`` of each rectangle as placed -- the two swap
            for a rotated one.
        rotated: Whether each rectangle was turned 90 degrees.
        bin_size: ``(width, height)`` of the bin, in texels.
        scale: The uniform scale :meth:`RectPacker.fit` found (1.0 from
            :meth:`RectPacker.pack`).
        padding: Texels reserved around every rectangle.
        align: The slot grid, in texels.
    """

    positions: Tuple[Tuple[int, int], ...]
    sizes: Tuple[Tuple[int, int], ...]
    rotated: Tuple[bool, ...]
    bin_size: Tuple[int, int]
    scale: float = 1.0
    padding: int = 0
    align: int = 1

    @property
    def fill(self) -> float:
        """Fraction of the bin's texels the rectangles' content covers."""
        width, height = self.bin_size
        covered = sum(w * h for w, h in self.sizes)
        return covered / float(max(width * height, 1))

    def pixel_rects(self) -> List[Tuple[int, int, int, int]]:
        """``(row0, row1, col0, col1)`` per rectangle, half-open, top-down.

        The shape :meth:`pythontk.ImgUtils.atlas_pixel_rects` returns, so a
        consumer can place, mask and publish from either.
        """
        return [
            (y, y + h, x, x + w) for (x, y), (w, h) in zip(self.positions, self.sizes)
        ]


class _Skyline:
    """The free space of a bin as a skyline: segments ``[x, x + w)`` filled to depth ``y``.

    Parallel lists sorted by ``x``, covering ``[0, width)`` with no gaps;
    adjacent segments never share a depth (they are merged).
    """

    __slots__ = ("width", "height", "xs", "ys", "ws")

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.xs: List[int] = [0]
        self.ys: List[int] = [0]
        self.ws: List[int] = [width]

    def find(self, w: int, h: int) -> Optional[Tuple[Tuple[int, int, int], int]]:
        """``(score, index)`` of the best spot for a ``w x h`` slot, or ``None``.

        A slot starts at a segment's left end; the score is ``(far edge,
        stranded area, x)`` -- lower is better, and the last term makes ties
        resolve left to right, deterministically.
        """
        xs, ys, ws = self.xs, self.ys, self.ws
        width, height = self.width, self.height
        count = len(xs)
        best: Optional[Tuple[Tuple[int, int, int], int]] = None
        for i in range(count):
            x = xs[i]
            if x + w > width:
                break  # segments are sorted by x: every later start is worse
            top, span, j = 0, 0, i
            while span < w:
                if ys[j] > top:
                    top = ys[j]
                span += ws[j]
                j += 1
            if top + h > height:
                continue
            waste, span, k = 0, 0, i
            while span < w:
                covered = ws[k] if span + ws[k] <= w else w - span
                waste += (top - ys[k]) * covered
                span += ws[k]
                k += 1
            score = (top + h, waste, x)
            if best is None or score < best[0]:
                best = (score, i)
        return best

    def place(self, index: int, w: int, h: int) -> Tuple[int, int]:
        """Fill a ``w x h`` slot at segment *index*; returns the slot's ``(x, y)``."""
        xs, ys, ws = self.xs, self.ys, self.ws
        x = xs[index]
        top, span, j = 0, 0, index
        while span < w:
            if ys[j] > top:
                top = ys[j]
            span += ws[j]
            j += 1
        end = x + w
        xs.insert(index, x)
        ys.insert(index, top + h)
        ws.insert(index, w)
        j = index + 1
        while j < len(xs) and xs[j] < end:
            seg_end = xs[j] + ws[j]
            if seg_end <= end:
                del xs[j], ys[j], ws[j]
            else:
                ws[j] = seg_end - end
                xs[j] = end
                break
        k = 0
        while k < len(xs) - 1:
            if ys[k] == ys[k + 1]:
                ws[k] += ws[k + 1]
                del xs[k + 1], ys[k + 1], ws[k + 1]
            else:
                k += 1
        return x, top


class RectPacker(HelpMixin):
    """Pack rectangles into one bin -- integer texels, padding, block alignment.

    Mechanism only: plain sizes in, a :class:`RectPackResult` out. What the
    rectangles ARE (an object's lightmap layout, a UV island's bounds, a
    texture tile) is the caller's; see the module docstring for the rules.
    """

    #: Placement orders :meth:`pack` tries, each a sort key over
    #: ``(slot_width, slot_height, index)``. Height-first is the skyline's
    #: natural order; the others rescue sets it packs badly (one wide strip
    #: among squares). Every key ends in the input index: equal rectangles
    #: keep the caller's order.
    ORDERS: Tuple[Callable[[Tuple[int, int, int]], tuple], ...] = (
        lambda s: (-s[1], -s[0], s[2]),
        lambda s: (-(s[0] * s[1]), -max(s[0], s[1]), s[2]),
        lambda s: (-max(s[0], s[1]), -min(s[0], s[1]), s[2]),
        lambda s: (-s[0], -s[1], s[2]),
    )

    #: Natural extents within this relative difference of each other are one
    #: size (the largest of them) before scaling, so copies of one mesh size,
    #: and therefore order, identically. Maya reports a surface area in
    #: single precision: four instances of one baseboard, rotated in place,
    #: measured 0.8500002 / 0.8500004 / 0.8500005 / 0.8499908 -- enough to
    #: reorder equal cells between two bakes of the same room, and to straddle
    #: a texel boundary. 0.1% is far below a texel on any cell an atlas holds.
    TIE_TOLERANCE: float = 1e-3

    @classmethod
    def _unify_ties(
        cls, extents: Sequence[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """*extents* with every near-equal run snapped to one size (see :attr:`TIE_TOLERANCE`).

        Each run is measured against its FIRST member, so a chain of small
        steps cannot creep a run along; its members all take the run's
        largest extent per axis, so none is sized down.
        """
        tol = float(cls.TIE_TOLERANCE)
        out = [(max(float(w), 0.0), max(float(h), 0.0)) for w, h in extents]
        order = sorted(range(len(out)), key=lambda i: (out[i][0], out[i][1], i))

        def near(a: float, b: float) -> bool:
            return abs(a - b) <= tol * max(abs(a), abs(b), 1e-300)

        start = 0
        while start < len(order):
            head = out[order[start]]
            end = start + 1
            while end < len(order):
                w, h = out[order[end]]
                if not (near(w, head[0]) and near(h, head[1])):
                    break
                end += 1
            run = order[start:end]
            if len(run) > 1:
                size = (max(out[i][0] for i in run), max(out[i][1] for i in run))
                for i in run:
                    out[i] = size
            start = end
        return out

    @staticmethod
    def _slots(
        sizes: Sequence[Tuple[int, int]], padding: int, align: int
    ) -> List[Tuple[int, int]]:
        """Each rectangle's slot in ALIGN blocks: content plus padding, rounded up."""
        return [
            (
                -(-(int(w) + 2 * padding) // align),
                -(-(int(h) + 2 * padding) // align),
            )
            for w, h in sizes
        ]

    @classmethod
    def _pack_order(
        cls,
        slots: Sequence[Tuple[int, int]],
        bin_blocks: Tuple[int, int],
        order: Sequence[int],
        rotate: bool,
    ) -> Optional[List[Tuple[int, int, bool]]]:
        """Slot ``(x, y, rotated)`` in blocks per rectangle, placed in *order*; ``None`` if one fails."""
        sky = _Skyline(*bin_blocks)
        placed: List[Optional[Tuple[int, int, bool]]] = [None] * len(slots)
        for i in order:
            w, h = slots[i]
            best = sky.find(w, h)
            turned = False
            if rotate and w != h:
                alt = sky.find(h, w)
                if alt is not None and (best is None or alt[0] < best[0]):
                    best, turned = alt, True
            if best is None:
                return None
            x, y = sky.place(best[1], *((h, w) if turned else (w, h)))
            placed[i] = (x, y, turned)
        return placed  # type: ignore[return-value]

    @classmethod
    def pack(
        cls,
        sizes: Sequence[Tuple[int, int]],
        bin_size: Tuple[int, int],
        padding: int = 0,
        align: int = 1,
        rotate: bool = False,
    ) -> Optional[RectPackResult]:
        """Place rectangles of given texel sizes in a ``bin_size`` bin.

        Parameters:
            sizes: ``(width, height)`` in texels per rectangle; each is at
                least 1 texel on both axes.
            bin_size: ``(width, height)`` of the bin, in texels.
            padding: Texels kept free around every rectangle on every side
                (two rectangles end up ``2 * padding`` or more apart).
            align: Slot grid in texels: every slot starts on and spans whole
                ``align x align`` blocks (4 for block-compressed textures).
            rotate: Allow 90-degree turns where one packs tighter. An atlas
                bound by a scale/offset (a lightmap's ``scaleOffset``) has no
                rotation term: leave it off there.

        Returns:
            RectPackResult: The placement, or ``None`` when the rectangles do
            not all fit in any of :attr:`ORDERS`. An empty input packs.

        Raises:
            ValueError: A non-positive bin, align or rectangle size, or a
                negative padding.
        """
        width, height = int(bin_size[0]), int(bin_size[1])
        padding, align = int(padding), int(align)
        if width <= 0 or height <= 0 or align <= 0 or padding < 0:
            raise ValueError(
                f"pack: bin {bin_size!r}, align {align} and padding {padding} must "
                "be positive (padding may be 0)."
            )
        sizes = [(int(w), int(h)) for w, h in sizes]
        if any(w < 1 or h < 1 for w, h in sizes):
            raise ValueError("pack: every rectangle needs at least 1 texel per axis.")
        if not sizes:
            return RectPackResult((), (), (), (width, height), 1.0, padding, align)
        slots = cls._slots(sizes, padding, align)
        blocks = (width // align, height // align)
        keyed = [(w, h, i) for i, (w, h) in enumerate(slots)]
        tried = set()
        for key in cls.ORDERS:
            order = tuple(item[2] for item in sorted(keyed, key=key))
            if order in tried:
                continue
            tried.add(order)
            placed = cls._pack_order(slots, blocks, order, rotate)
            if placed is None:
                continue
            positions, out_sizes, turned = [], [], []
            for (x, y, rot), (w, h) in zip(placed, sizes):
                positions.append((x * align + padding, y * align + padding))
                out_sizes.append((h, w) if rot else (w, h))
                turned.append(rot)
            return RectPackResult(
                tuple(positions),
                tuple(out_sizes),
                tuple(turned),
                (width, height),
                1.0,
                padding,
                align,
            )
        return None

    @classmethod
    def fit(
        cls,
        extents: Sequence[Tuple[float, float]],
        bin_size: Tuple[int, int],
        padding: int = 0,
        align: int = 1,
        min_size: int = 1,
        rotate: bool = False,
        tolerance: float = 0.002,
        fold: bool = False,
    ) -> Optional[RectPackResult]:
        """The largest uniform scale at which rectangles of natural *extents* fit, packed.

        Each rectangle is ``ceil(scale * extent)`` texels per axis (never
        under *min_size*), so every one gets the same texels per unit and
        keeps its own aspect. The scale is bisected from the bin's area
        bound down to *tolerance*; placement is discrete, so the largest
        FOUND fitting scale is kept rather than assuming monotonic fit.

        Parameters:
            extents: ``(width, height)`` per rectangle, in any one unit;
                zero or negative extents size at *min_size*.
            bin_size: ``(width, height)`` of the bin, in texels.
            padding: See :meth:`pack`.
            align: See :meth:`pack`.
            min_size: Smallest content size, in texels per axis.
            rotate: See :meth:`pack`.
            tolerance: Relative precision the scale is found to.
            fold: Let a rectangle longer than the bin at the scale being
                tried be folded toward square, keeping its area, instead of
                capping the scale for every rectangle: one 100:1 strip would
                otherwise hold a whole set to the scale at which it alone
                fits. Only that rectangle's aspect gives -- only as far as it
                must, at the scale the set reaches -- and its share of texels
                stays its own.

        Returns:
            RectPackResult: The placement at the scale found (its
            :attr:`~RectPackResult.scale`), or ``None`` when even every
            rectangle at *min_size* does not fit.

        Raises:
            ValueError: As :meth:`pack`, or a *min_size* under 1.
        """
        min_size = int(min_size)
        if min_size < 1:
            raise ValueError("fit: min_size must be at least 1 texel.")
        natural = cls._unify_ties(extents)
        width, height = int(bin_size[0]), int(bin_size[1])
        room_w, room_h = max(width - 2 * padding, 1), max(height - 2 * padding, 1)

        def shaped(w: float, h: float, scale: float) -> Tuple[float, float]:
            # fold: the extent the bin can hold at this scale, at w x h's area
            # -- never folded past square, where only the scale can help.
            if not fold or scale <= 0.0:
                return w, h
            lw, lh = room_w / scale, room_h / scale
            if w <= lw and h <= lh:
                return w, h
            area, side = w * h, sqrt(w * h)
            if w > lw:
                w = max(lw, side)
                return w, (area / w if w > 0.0 else 0.0)
            h = max(lh, side)
            return (area / h if h > 0.0 else 0.0), h

        def sizes_at(scale: float) -> List[Tuple[int, int]]:
            out = []
            for w, h in natural:
                w, h = shaped(w, h, scale)
                out.append(
                    (
                        max(min_size, int(ceil(scale * w - 1e-9))),
                        max(min_size, int(ceil(scale * h - 1e-9))),
                    )
                )
            return out

        def attempt(scale: float) -> Optional[RectPackResult]:
            result = cls.pack(sizes_at(scale), bin_size, padding, align, rotate)
            if result is None:
                return None
            return RectPackResult(
                result.positions,
                result.sizes,
                result.rotated,
                result.bin_size,
                scale,
                result.padding,
                result.align,
            )

        floor = attempt(0.0)
        if floor is None or not natural:
            return floor
        total = sum(w * h for w, h in natural)
        if total <= 0.0:
            return floor
        # No packing beats the bin's area, nor lets one rectangle (plus its
        # padding) outgrow the bin on an axis -- or, folding, outgrow the
        # bin's area: a fold keeps its area, in whatever shape the bin
        # holds. (The bin's SHORTER side squared under-read a long bin: in
        # 1000 x 10 a 100 x 1 strip that fits unfolded at 10 was held to 1.)
        high = sqrt(width * height / total)
        for w, h in natural:
            long_side, short_side = max(w, h), min(w, h)
            if fold:
                if w * h > 0.0:
                    high = min(high, sqrt(room_w * room_h / (w * h)))
            elif rotate:
                if long_side > 0:
                    high = min(high, max(room_w, room_h) / long_side)
                if short_side > 0:
                    high = min(high, min(room_w, room_h) / short_side)
            else:
                if w > 0:
                    high = min(high, room_w / w)
                if h > 0:
                    high = min(high, room_h / h)
        best = attempt(high)
        if best is not None:
            return best
        low, best = 0.0, floor
        while high - low > tolerance * high:
            mid = 0.5 * (low + high)
            if not low < mid < high:
                break  # adjacent floats: a tolerance of 0 must not spin forever
            found = attempt(mid)
            if found is not None:
                low, best = mid, found
            else:
                high = mid
        return best
