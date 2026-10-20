# !/usr/bin/python
# coding=utf-8
"""What a look match compares: the reference image, and the regions measured on both sides.

Coordinates are normalised to the frame (0..1, origin top-left), so a target
works at any resolution. Each region places its shape once per side: a
reference's geometry -- an AI render's especially -- differs a little from ours.
"""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a declared dependency
    np = None  # type: ignore
try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - Pillow is a declared dependency
    Image = ImageDraw = None  # type: ignore

from pythontk.file_utils._file_utils import FileUtils
from pythontk.img_utils._img_utils import ImgUtils

SIDES = ("ours", "ref")


class RegionShape:
    """One shape a region's side can take, keyed in its spec (``{"rect": [...]}``).

    A shape owns everything that depends on its geometry: the region kind it
    serves, its pixels at a size, its outline, where its label anchors, and the
    values a measure of its kind samples. :class:`LookRegion` and the sheet ask
    the shape, so a new one is a subclass plus :meth:`LookRegion.register_shape`.
    """

    #: The spec key it answers to.
    key = ""
    #: The region kind it serves (:attr:`LookRegion.KINDS`).
    kind = "area"
    #: Whether an editor draws it from points (:meth:`from_points`): painted
    #: and object shapes come from elsewhere.
    drawn = False

    def mask(
        self, region: "LookRegion", spec: Mapping[str, Any], size: Tuple[int, int]
    ) -> "np.ndarray":
        """Its pixels at *size* ``(width, height)``: those whose centre is inside."""
        raise NotImplementedError

    def outline(
        self, spec: Mapping[str, Any]
    ) -> Optional[Tuple[List[Tuple[float, float]], bool]]:
        """``(normalised points, closed)``, or None: the mask's own edge is the outline."""
        return None

    def anchor(self, spec: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
        """The normalised top-left a label sits at, or None: the mask's bounds."""
        return None

    def sample(self, region: "LookRegion", frame: "np.ndarray", side: str) -> Any:
        """What its kind's measures read from *frame*: an area's pixels ``(n, C)``."""
        height, width = frame.shape[:2]
        return frame.reshape(-1, frame.shape[-1])[region.indices(side, (width, height))]

    def from_points(
        self, points: Sequence[Sequence[float]], size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Its spec from *points* drawn in pixels at *size* ``(width, height)``
        -- the inverse of :meth:`outline` (a :attr:`drawn` shape implements it).

        Raises:
            ValueError: points it cannot be drawn from.
        """
        raise ValueError(f"A {self.key!r} shape is not drawn from points.")

    @staticmethod
    def _unit(
        points: Sequence[Sequence[float]], size: Tuple[int, int]
    ) -> List[List[float]]:
        """*points* in pixels as normalised ``[x, y]`` inside the frame."""
        width, height = size
        return [
            [
                round(min(1.0, max(0.0, float(x) / width)), 6),
                round(min(1.0, max(0.0, float(y) / height)), 6),
            ]
            for x, y in points
        ]

    @classmethod
    def _corners(cls, points, size) -> Tuple[float, float, float, float]:
        """The normalised box two drawn corners span, in either order.

        Raises:
            ValueError: not two corners spanning an area.
        """
        if len(points) < 2:
            raise ValueError("A box is drawn from two corners.")
        (x0, y0), (x1, y1) = cls._unit(points[:2], size)
        x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        if x0 == x1 or y0 == y1:
            raise ValueError("The corners span no area.")
        return x0, y0, x1, y1


class _Rect(RegionShape):
    key = "rect"
    drawn = True

    def from_points(self, points, size):
        return {"rect": list(self._corners(points, size))}

    def mask(self, region, spec, size):
        width, height = size
        x0, y0, x1, y1 = spec["rect"]
        xs, ys = np.arange(width) + 0.5, np.arange(height) + 0.5
        inside_x = (xs >= x0 * width) & (xs < x1 * width)
        inside_y = (ys >= y0 * height) & (ys < y1 * height)
        return inside_y[:, None] & inside_x[None, :]

    def outline(self, spec):
        x0, y0, x1, y1 = spec["rect"]
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], True

    def anchor(self, spec):
        return spec["rect"][0], spec["rect"][1]


class _Ellipse(RegionShape):
    key = "ellipse"
    drawn = True

    def from_points(self, points, size):  # the box it fills
        x0, y0, x1, y1 = self._corners(points, size)
        return {
            "ellipse": [
                round((x0 + x1) / 2, 6),
                round((y0 + y1) / 2, 6),
                round((x1 - x0) / 2, 6),
                round((y1 - y0) / 2, 6),
            ]
        }

    def mask(self, region, spec, size):
        width, height = size
        cx, cy, rx, ry = spec["ellipse"]
        xs = (np.arange(width) + 0.5 - cx * width) / max(rx * width, 1e-6)
        ys = (np.arange(height) + 0.5 - cy * height) / max(ry * height, 1e-6)
        return (xs[None, :] ** 2 + ys[:, None] ** 2) <= 1.0

    def outline(self, spec):
        cx, cy, rx, ry = spec["ellipse"]
        t = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        return [
            (float(cx + rx * np.cos(a)), float(cy + ry * np.sin(a))) for a in t
        ], True

    def anchor(self, spec):
        cx, cy, rx, ry = spec["ellipse"]
        return cx - rx, cy - ry


class _Poly(RegionShape):
    key = "poly"
    drawn = True

    def from_points(self, points, size):
        if len(points) < 3:
            raise ValueError("A polygon is drawn from three points or more.")
        return {self.key: self._unit(points, size)}

    def mask(self, region, spec, size):
        width, height = size
        canvas = Image.new("L", (width, height), 0)
        ImageDraw.Draw(canvas).polygon(
            [(x * width, y * height) for x, y in spec[self.key]], fill=255
        )
        return np.asarray(canvas) > 127

    def outline(self, spec):
        return [tuple(p) for p in spec[self.key]], True

    def anchor(self, spec):
        xs, ys = zip(*spec[self.key])
        return min(xs), min(ys)


class _Quad(_Poly):
    """A surface rectified onto a grid: its corners top-left, top-right, bottom-right, bottom-left."""

    key = "quad"
    kind = "quad"
    drawn = False  # its grid is chosen, not drawn

    def anchor(self, spec):
        return tuple(spec["quad"][0])

    def sample(self, region, frame, side):
        return region.rectify(frame, side)


class _Painted(RegionShape):
    """Strokes of one colour on a transparent marker layer (:meth:`LookTarget.painted`)."""

    key = "mask"

    def mask(self, region, spec, size):
        return region._owner("a painted").painted(spec["mask"], spec["color"], size)


class _Objects(RegionShape):
    """The pixels of named objects in the target's object-ID render (:meth:`LookTarget.objects`)."""

    key = "objects"

    def mask(self, region, spec, size):
        return region._owner("an object").objects(spec["objects"], size)


class _Line(RegionShape):
    """A profile across the frame: ``{"line": [[x0, y0], [x1, y1]], "width": w}``."""

    key = "line"
    kind = "line"
    drawn = True
    #: The samples a measure reads along it.
    SAMPLES = 48

    @staticmethod
    def points(spec, size, n: int = 64) -> Tuple["np.ndarray", "np.ndarray"]:
        (x0, y0), (x1, y1) = spec["line"]
        t = np.linspace(0.0, 1.0, n)
        return (x0 + (x1 - x0) * t) * size[0] - 0.5, (y0 + (y1 - y0) * t) * size[
            1
        ] - 0.5

    def mask(self, region, spec, size):
        mask = np.zeros((size[1], size[0]), bool)
        xs, ys = self.points(spec, size)
        mask[
            np.clip(ys.astype(int), 0, size[1] - 1),
            np.clip(xs.astype(int), 0, size[0] - 1),
        ] = True
        return mask

    def outline(self, spec):
        return [tuple(p) for p in spec["line"]], False

    def from_points(self, points, size):
        if len(points) < 2:
            raise ValueError("A line is drawn from two points.")
        start, end = self._unit(points[:2], size)
        if start == end:
            raise ValueError("The line has no length.")
        return {"line": [start, end]}

    def anchor(self, spec):
        return tuple(spec["line"][0])

    def sample(self, region, frame, side):
        return region.sample_line(frame, side, n=self.SAMPLES)


class LookRegion:
    """A named part of the frame, compared between our render and the reference.

    Parameters:
        name: Unique within its target.
        kind: ``"area"`` (a filled shape), ``"line"`` (a profile across the
            frame) or ``"quad"`` (a surface rectified onto a grid: a board, a mat).
        ours: Our side's shape:
            area -- ``{"rect": [x0, y0, x1, y1]}``, ``{"ellipse": [cx, cy, rx, ry]}``,
            ``{"poly": [[x, y], ...]}``, a painted marker ``{"mask": "file.png",
            "color": [r, g, b]}`` (a transparent layer of strokes, one colour per
            region: :meth:`LookTarget.painted`) or, on our side, ``{"objects":
            [patterns]}`` (pixels of the objects named in the target's object-ID
            render: :meth:`LookTarget.objects`);
            line -- ``{"line": [[x0, y0], [x1, y1]], "width": w}`` (``width`` optional);
            quad -- ``{"quad": [top-left, top-right, bottom-right, bottom-left],
            "grid": [columns, rows]}``.
        ref: The reference's shape (default: the same as ours).
        measures: The measures to take (names in :class:`LookMeasures`);
            ``None`` takes the kind's defaults.
        weight: How much the region counts in a score or a fit.
        exclude: ``{"ours": [area shapes], "ref": [area shapes]}`` cut from the
            region -- what stands in front of it (a lamp pole, an object).
        notes: Free text.
    """

    KINDS = ("area", "line", "quad")
    #: Every shape a side can take, by spec key (:meth:`register_shape` adds one).
    SHAPES: Dict[str, RegionShape] = {}
    #: A quad's ``(columns, rows)`` when its shape names no ``grid``.
    GRID = (64, 16)

    def __init__(
        self,
        name: str,
        kind: str,
        ours: Mapping[str, Any],
        ref: Optional[Mapping[str, Any]] = None,
        measures: Optional[Sequence[str]] = None,
        weight: float = 1.0,
        exclude: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
        notes: str = "",
    ):
        if kind not in self.KINDS:
            raise ValueError(f"Unknown region kind {kind!r}; one of {self.KINDS}.")
        self.name, self.kind = name, kind
        self.shapes = {
            "ours": copy.deepcopy(dict(ours)),
            "ref": copy.deepcopy(dict(ref or ours)),
        }
        for spec in self.shapes.values():
            self._shape(spec)
        if kind == "quad":  # its sides are compared cell for cell
            grids = [tuple(s.get("grid", self.GRID)) for s in self.shapes.values()]
            if grids[0] != grids[1]:
                raise ValueError(
                    f"Region {name!r}: a quad has one grid on both sides; got {grids[0]} and {grids[1]}."
                )
        self.exclude = {
            side: [copy.deepcopy(dict(s)) for s in (exclude or {}).get(side, [])]
            for side in SIDES
        }
        for specs in self.exclude.values():
            for spec in specs:
                self._shape(spec, "area")
        self.measures = list(measures) if measures is not None else None
        self.weight = float(weight)
        self.notes = notes
        # Geometry per (side, size): shapes never change after construction, and a
        # fit scores hundreds of frames of one size.
        self._masks: Dict[Tuple, "np.ndarray"] = {}
        self._keeps: Dict[Tuple, "np.ndarray"] = {}
        # The target it belongs to: what a painted or object shape reads.
        self.target: Optional["LookTarget"] = None

    # ------------------------------------------------------------------ shapes
    @classmethod
    def drawable(cls) -> Dict[str, RegionShape]:
        """The shapes an editor draws, by spec key (:attr:`RegionShape.drawn`)."""
        return {key: shape for key, shape in cls.SHAPES.items() if shape.drawn}

    @classmethod
    def spec_from_points(
        cls, key: str, points: Sequence[Sequence[float]], size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """The spec of shape *key* drawn through *points* (pixels at *size*).

        Raises:
            ValueError: no such drawable shape, or points it cannot be drawn from.
        """
        shape = cls.SHAPES.get(key)
        if shape is None or not shape.drawn:
            raise ValueError(
                f"No drawable shape {key!r}; one of {sorted(cls.drawable())}."
            )
        return shape.from_points(points, size)

    @classmethod
    def register_shape(cls, shape: RegionShape) -> None:
        """Add (or replace) the shape a spec keyed :attr:`RegionShape.key` takes."""
        cls.SHAPES[shape.key] = shape

    def _shape(
        self, spec: Mapping[str, Any], kind: Optional[str] = None
    ) -> RegionShape:
        """The registered shape *spec* holds, checked against *kind* (default: the region's)."""
        keys = [k for k in spec if k in self.SHAPES]
        if len(keys) != 1:
            raise ValueError(
                f"Region {self.name!r}: a shape holds exactly one of {sorted(self.SHAPES)}; got {sorted(spec)}."
            )
        shape = self.SHAPES[keys[0]]
        kind = kind or self.kind
        if shape.kind != kind:
            raise ValueError(
                f"Region {self.name!r}: a {kind} cannot take a {shape.key!r} shape (one of a {shape.kind})."
            )
        return shape

    def _owner(self, what: str) -> "LookTarget":
        """The target a shape reading its files (markers, object IDs) reads them from."""
        if self.target is None:
            raise ValueError(
                f"Region {self.name!r}: {what} shape reads its target's files -- add the region to a LookTarget."
            )
        return self.target

    def mask(self, side: str, size: Tuple[int, int]) -> "np.ndarray":
        """The region's pixels on *side* at *size* ``(width, height)``, its exclusions cut out.

        A line covers the pixels its samples touch; a quad, its outline's interior.
        """
        key = (side, tuple(size))
        if key not in self._masks:
            self._masks[key] = self._build_mask(side, size)
        return self._masks[key]

    def indices(self, side: str, size: Tuple[int, int]) -> "np.ndarray":
        """The mask's flat pixel indices (cached): ``frame.reshape(-1, C)[indices]``."""
        key = ("indices", side, tuple(size))
        if key not in self._masks:
            self._masks[key] = np.flatnonzero(self.mask(side, size))
        return self._masks[key]

    def _area(self, spec: Mapping[str, Any], size: Tuple[int, int]) -> "np.ndarray":
        """An area shape's pixels (an exclusion's): whatever area shape it is."""
        return self._shape(spec, "area").mask(self, spec, size)

    def _build_mask(self, side: str, size: Tuple[int, int]) -> "np.ndarray":
        spec = self.shapes[side]
        mask = self._shape(spec).mask(self, spec, size)
        for cut in self.exclude[side]:  # a new array: a shape's may be shared
            mask = mask & ~self._area(cut, size)
        return mask

    def outline(self, side: str) -> Optional[Tuple[List[Tuple[float, float]], bool]]:
        """The region's outline on *side*: ``(normalised points, closed)``, or None
        when its pixels' own edge is the outline (painted and object shapes)."""
        spec = self.shapes[side]
        return self._shape(spec).outline(spec)

    def anchor(
        self, side: str, size: Tuple[int, int] = (1000, 1000)
    ) -> Tuple[float, float]:
        """The normalised top-left a label of the region sits at on *side*."""
        spec = self.shapes[side]
        point = self._shape(spec).anchor(spec)
        if point is not None:
            return tuple(point)
        x0, y0, _x1, _y1 = self.window(side, size)
        return x0 / size[0], y0 / size[1]

    def sample(self, frame: "np.ndarray", side: str) -> Any:
        """What the region's measures read from *frame* on *side*: an area's pixels
        ``(n, C)``, a line's profile, a quad's ``(grid, keep)``."""
        spec = self.shapes[side]
        return self._shape(spec).sample(self, frame, side)

    def window(self, side: str, size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """``(x0, y0, x1, y1)`` pixel bounds of the region on *side* at *size*."""
        ys, xs = np.nonzero(self.mask(side, size))
        if not len(xs):
            return (0, 0, 0, 0)
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    # ------------------------------------------------------------------ lines
    def line_points(
        self, side: str, size: Tuple[int, int], n: int = 64
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        """*n* evenly spaced pixel-centre coordinates along the line, end to end."""
        return _Line.points(self.shapes[side], size, n)

    def sample_line(
        self, image: "np.ndarray", side: str, n: int = 64, across: int = 3
    ) -> "np.ndarray":
        """The image along the line: *n* samples, each the mean of *across* samples
        spread over the line's width (``width``, default one pixel).

        Returns ``(n,)`` for a 2D image, ``(n, C)`` for channels.
        """
        height, width = image.shape[:2]
        xs, ys = self.line_points(side, (width, height), n)
        (x0, y0), (x1, y1) = self.shapes[side]["line"]
        dx, dy = (x1 - x0) * width, (y1 - y0) * height
        length = float(np.hypot(dx, dy)) or 1.0
        nx, ny = -dy / length, dx / length  # unit normal
        half = max(self.shapes[side].get("width", 0.0) * max(width, height), 1.0) / 2.0
        offsets = np.linspace(-half, half, across) if across > 1 else np.zeros(1)
        img = np.asarray(image, np.float64)
        samples = [
            ImgUtils.sample_bilinear(img, xs + o * nx, ys + o * ny) for o in offsets
        ]
        return np.mean(samples, axis=0).astype(np.float32)

    # ------------------------------------------------------------------ quads
    def rectify(
        self, image: "np.ndarray", side: str
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        """The quad resampled onto its grid, and which cells to keep.

        Returns:
            ``(grid, keep)``: ``(rows, columns[, C])`` values and a ``(rows,
            columns)`` boolean -- False where an exclusion covers half a cell or more.
        """
        shape = self.shapes[side]
        columns, rows = shape.get("grid", self.GRID)
        height, width = image.shape[:2]
        corners = [(x * width, y * height) for x, y in shape["quad"]]
        # up to 6 x 6 samples per cell: a cell is an average, not a resample of every pixel
        grid = ImgUtils.warp_quad(
            image,
            corners,
            (columns, rows),
            supersample=self._samples(corners, columns, rows),
        )
        key = (side, (width, height))
        if key not in self._keeps:
            keep = np.ones((height, width), np.float32)
            for cut in self.exclude[side]:
                keep[self._area(cut, (width, height))] = 0.0
            self._keeps[key] = ImgUtils.warp_quad(keep, corners, (columns, rows)) >= 0.5
        return grid, self._keeps[key]

    @staticmethod
    def _samples(corners, columns: int, rows: int, cap: int = 6) -> int:
        q = np.asarray(corners, np.float64)
        extent = max(
            np.linalg.norm(q[1] - q[0]) / columns,
            np.linalg.norm(q[3] - q[0]) / rows,
        )
        return int(min(cap, max(1, np.ceil(extent - 1e-9))))

    # ------------------------------------------------------------------ io
    def to_dict(self) -> Dict[str, Any]:
        out = {
            "name": self.name,
            "kind": self.kind,
            "ours": copy.deepcopy(self.shapes["ours"]),
        }
        if self.shapes["ref"] != self.shapes["ours"]:
            out["ref"] = copy.deepcopy(self.shapes["ref"])
        if self.measures is not None:
            out["measures"] = list(self.measures)
        if self.weight != 1.0:
            out["weight"] = self.weight
        if any(self.exclude.values()):
            out["exclude"] = {
                side: shapes for side, shapes in self.exclude.items() if shapes
            }
        if self.notes:
            out["notes"] = self.notes
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LookRegion":
        return cls(
            data["name"],
            data["kind"],
            data["ours"],
            data.get("ref"),
            data.get("measures"),
            data.get("weight", 1.0),
            data.get("exclude"),
            data.get("notes", ""),
        )


for _shape in (_Rect(), _Ellipse(), _Poly(), _Quad(), _Painted(), _Objects(), _Line()):
    LookRegion.register_shape(_shape)


class LookTarget:
    """A reference image and the regions compared against it.

    Parameters:
        reference: Path to the reference image (a display-referred sRGB image).
        regions: The :class:`LookRegion` s.
        focus: The region whose pixels are "the subject" -- where the brightest
            light should land (a frame measure reads it); ``None`` for none.
        notes: Free text.
        ids: Our side's object-ID render, for ``objects`` shapes: ``{"image":
            png, "legend": [{"color": [r, g, b], "names": [root ... node],
            "material": name}]}`` (``LookCapture.shoot_ids``).
        base: The folder relative files (markers, the ID image) are read from;
            :meth:`load` sets it to the target file's.
    """

    #: How close (0..255 per channel, Euclidean) a painted pixel must be to its
    #: region's colour, and how opaque, to count -- a soft stroke's edge does.
    MARKER_DISTANCE = 60.0
    MARKER_ALPHA = 26

    VERSION = 1

    def __init__(
        self,
        reference: str,
        regions: Iterable[LookRegion] = (),
        focus: Optional[str] = None,
        notes: str = "",
        ids: Optional[Mapping[str, Any]] = None,
        base: Optional[str] = None,
    ):
        self.reference = str(reference)
        self.regions: List[LookRegion] = list(regions)
        self.ids = copy.deepcopy(dict(ids)) if ids else None
        self.base = base
        for region in self.regions:
            region.target = self
        names = [r.name for r in self.regions]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Region names must be unique; repeated: {dupes}.")
        self.focus = focus
        self.notes = notes
        self._cache: Dict[Any, "np.ndarray"] = {}

    def region(self, name: str) -> LookRegion:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"No region {name!r}; regions: {[r.name for r in self.regions]}")

    def add(self, region: LookRegion) -> LookRegion:
        """Add *region* (its name new here); returns it.

        Raises:
            ValueError: a region of that name exists.
        """
        if any(r.name == region.name for r in self.regions):
            raise ValueError(f"A region named {region.name!r} exists.")
        region.target = self
        self.regions.append(region)
        return region

    def remove(self, name: str) -> LookRegion:
        """Take region *name* out (and off the focus); returns it.

        Raises:
            KeyError: no such region.
        """
        region = self.region(name)
        self.regions.remove(region)
        region.target = None
        if self.focus == name:
            self.focus = None
        return region

    def rename(self, name: str, new: str) -> LookRegion:
        """Call region *name* *new* (the focus follows); returns it.

        Raises:
            KeyError: no such region.
            ValueError: another region is called *new*.
        """
        region = self.region(name)
        if new != name and any(r.name == new for r in self.regions):
            raise ValueError(f"A region named {new!r} exists.")
        region.name = new
        if self.focus == name:
            self.focus = new
        return region

    def _file(self, path: str) -> str:
        if os.path.isabs(path) or not self.base:
            return path
        return os.path.normpath(os.path.join(self.base, path))

    @staticmethod
    def _coverage(mask: "np.ndarray", size: Tuple[int, int]) -> "np.ndarray":
        """A full-resolution mask at *size*: a pixel is in when most of it was."""
        if (mask.shape[1], mask.shape[0]) == tuple(size):
            return mask
        cover = ImgUtils.resample_area(mask.astype(np.float32)[..., None], size)[..., 0]
        return cover >= 0.5

    def painted(
        self, path: str, color: Sequence[int], size: Tuple[int, int]
    ) -> "np.ndarray":
        """The pixels of a marker layer painted in *color*, at *size*.

        A marker layer is a transparent image of strokes, one colour per region,
        painted over the frame it marks: a pixel belongs to the region whose
        colour is nearest (within :attr:`MARKER_DISTANCE`) where it is opaque
        enough (:attr:`MARKER_ALPHA`), so a soft, half-transparent stroke reads
        whole. Every region painting the same file shares one assignment.
        """
        color = tuple(int(c) for c in color)
        key = ("painted", path, color, tuple(size))
        if key not in self._cache:
            palette = sorted(
                {
                    tuple(int(c) for c in shape["color"])
                    for region in self.regions
                    for shape in [
                        *region.shapes.values(),
                        *sum(region.exclude.values(), []),
                    ]
                    if shape.get("mask") == path
                }
                | {color}
            )
            nearest, distance, opaque = self._assignment(path, palette)
            full = (
                (nearest == palette.index(color))
                & (distance <= self.MARKER_DISTANCE)
                & opaque
            )
            self._cache[key] = self._coverage(full, size)
        return self._cache[key]

    def _assignment(
        self, path: str, palette: List[Tuple[int, ...]]
    ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Marker layer *path*'s pixels against *palette*, once per file and
        palette: ``(nearest colour's index, its distance, opaque enough)``."""
        key = ("painted", path, tuple(palette))
        if key not in self._cache:
            with Image.open(self._file(path)) as img:
                rgba = np.asarray(img.convert("RGBA"), np.float32)
            dist = np.stack(
                [
                    np.linalg.norm(rgba[..., :3] - np.asarray(c, np.float32), axis=-1)
                    for c in palette
                ]
            )
            self._cache[key] = (
                dist.argmin(axis=0),
                dist.min(axis=0),
                rgba[..., 3] >= self.MARKER_ALPHA,
            )
        return self._cache[key]

    def objects(self, patterns: Sequence[str], size: Tuple[int, int]) -> "np.ndarray":
        """The pixels of the objects *patterns* name in our side's object-ID render.

        A pattern is a regular expression matched against an object's name, an
        ancestor's or its material's; one starting with ``!`` keeps what it
        matches out (``LookCapture``'s ``black`` grammar).

        Raises:
            ValueError: the target has no object-ID render.
        """
        if not self.ids:
            raise ValueError(
                "No object-ID render on this target (LookCapture.shoot_ids)."
            )
        key = ("objects", tuple(patterns), tuple(size))
        if key not in self._cache:
            take = [re.compile(p) for p in patterns if not p.startswith("!")]
            keep = [re.compile(p[1:]) for p in patterns if p.startswith("!")]
            with Image.open(self._file(self.ids["image"])) as img:
                rgb = np.asarray(img.convert("RGB"), np.int32)
            code = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
            wanted = []
            for entry in self.ids.get("legend") or []:
                names = list(entry.get("names") or []) + [entry.get("material") or ""]
                if any(p.search(n) for p in take for n in names) and not any(
                    p.search(n) for p in keep for n in names
                ):
                    r, g, b = (int(c) for c in entry["color"])
                    wanted.append((r << 16) | (g << 8) | b)
            full = np.isin(code, wanted)
            self._cache[key] = self._coverage(full, size)
        return self._cache[key]

    #: Marker colours, as far apart as 20 colours go (Trubetskoy's list): one
    #: per region, so a stroke's colour says which region it marks.
    PALETTE = (
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
        (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
        (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
        (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
        (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    )  # fmt: skip

    def paint_template(
        self,
        folder: str,
        side: str = "ref",
        frame: Optional[str] = None,
        regions: Optional[Sequence[str]] = None,
        layer: Optional[str] = None,
    ) -> Dict[str, List[int]]:
        """Start a marker layer for *side*'s area regions, and point them at it.

        Writes ``<layer>.png`` -- a transparent layer, each area region's
        current shape filled half-transparent in a colour of its own -- and
        ``<layer>_guide.png`` (the frame under the markers, with a legend).
        Each area region's *side* shape becomes ``{"mask": "<side>_markers.png",
        "color": ...}``, so it reads exactly what it read before; repaint the
        layer (any editor; save the layer alone, transparent) and the regions
        follow the strokes. Lines and quads keep their shapes.

        Parameters:
            folder: Where the files go -- the target's own folder, so its
                relative paths hold.
            side: ``"ref"`` (the reference) or ``"ours"``.
            frame: The image *side*'s markers lie over (default: the reference;
                for ``"ours"``, a frame of ours -- an exact capture).
            regions: The area regions to paint (default: every one). A pixel is
                one stroke, so regions on one layer may not overlap -- a part of
                an object (a rim) goes on a layer of its own.
            layer: The layer's file stem (default ``<side>_markers``).

        Returns:
            ``{region: color}`` for the regions now painted.

        Raises:
            ValueError: two of the regions overlap.
        """
        background = frame or self.reference
        with Image.open(background) as img:
            under = img.convert("RGB")
        size = under.size
        stem = layer or f"{side}_markers"
        name = f"{stem}.png"
        markers = np.zeros((size[1], size[0], 4), np.uint8)
        palette: Dict[str, List[int]] = {}
        areas = [
            r
            for r in self.regions
            if r.kind == "area" and (regions is None or r.name in regions)
        ]
        taken = np.zeros((size[1], size[0]), bool)
        owner: Dict[int, str] = {}
        for i, region in enumerate(areas):
            color = list(self.PALETTE[i % len(self.PALETTE)])
            mask = region.mask(side, size)
            if (taken & mask).any():
                other = next(
                    n
                    for j, n in owner.items()
                    if (areas[j].mask(side, size) & mask).any()
                )
                raise ValueError(
                    f"Regions {other!r} and {region.name!r} overlap on {side}: paint "
                    "them on separate layers (layer=)."
                )
            taken |= mask
            owner[i] = region.name
            markers[mask] = color + [128]
            palette[region.name] = color
        os.makedirs(folder, exist_ok=True)
        Image.fromarray(markers, "RGBA").save(os.path.join(folder, name))
        guide = under.convert("RGBA")
        guide.alpha_composite(Image.fromarray(markers, "RGBA"))
        legend_h = 22 * ((len(palette) + 3) // 4) + 8
        sheet = Image.new("RGB", (size[0], size[1] + legend_h), (24, 24, 24))
        sheet.paste(guide.convert("RGB"), (0, 0))
        draw = ImageDraw.Draw(sheet)
        for i, (region_name, color) in enumerate(palette.items()):
            x = 8 + (i % 4) * (size[0] // 4)
            y = size[1] + 6 + (i // 4) * 22
            draw.rectangle([x, y, x + 16, y + 16], fill=tuple(color))
            draw.text((x + 22, y + 2), region_name, fill=(230, 230, 230))
        sheet.save(os.path.join(folder, f"{stem}_guide.png"))
        self.base = self.base or folder
        for region in areas:
            region.shapes[side] = {"mask": name, "color": palette[region.name]}
            region._masks.clear()
        self._cache = {  # the reference's own entries are keyed by size (or None)
            k: v
            for k, v in self._cache.items()
            if not (isinstance(k, tuple) and k and k[0] == "painted")
        }
        return palette

    def reference_size(self) -> Tuple[int, int]:
        with Image.open(self.reference) as img:
            return img.size

    def reference_display(self, size: Optional[Tuple[int, int]] = None) -> "np.ndarray":
        """The reference as sRGB-encoded 0..1 float32, area-averaged in linear display light.

        Averaging encoded values would darken every edge between light and dark;
        averaging the light it shows is what an eye at a distance sees.
        """
        key = tuple(size) if size else None
        if key not in self._cache:
            self._cache[key] = self.display_at(self.reference, size)
        return self._cache[key]

    @staticmethod
    def display_at(
        image: Union[str, "np.ndarray"], size: Optional[Tuple[int, int]] = None
    ) -> "np.ndarray":
        """A display image (a file, or an sRGB array: uint8, or float 0..1) as
        sRGB-encoded 0..1 float32 at *size*, area-averaged in linear display light."""
        if isinstance(image, str):
            with Image.open(image) as img:
                rgb = np.asarray(img.convert("RGB"), np.float32) / 255.0
        else:
            rgb = np.asarray(image)
            rgb = (
                rgb.astype(np.float32) / 255.0
                if rgb.dtype == np.uint8
                else rgb.astype(np.float32)
            )[..., :3]
        if size and tuple(size) != (rgb.shape[1], rgb.shape[0]):
            rgb = ImgUtils.linear_to_srgb(
                ImgUtils.resample_area(ImgUtils.srgb_to_linear(rgb), size)
            )
        return np.asarray(rgb, np.float32)

    # ------------------------------------------------------------------ io
    def to_dict(self, base: Optional[str] = None) -> Dict[str, Any]:
        reference = self.reference
        if base:
            try:
                reference = os.path.relpath(self.reference, base).replace("\\", "/")
            except ValueError:  # another drive: keep it absolute
                pass
        out = {
            "version": self.VERSION,
            "reference": reference,
            "regions": [r.to_dict() for r in self.regions],
        }
        if self.focus:
            out["focus"] = self.focus
        if self.notes:
            out["notes"] = self.notes
        if self.ids:
            out["ids"] = copy.deepcopy(self.ids)
        return out

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], base: Optional[str] = None
    ) -> "LookTarget":
        version = data.get("version", cls.VERSION)
        if version > cls.VERSION:
            raise ValueError(
                f"Look target version {version} is newer than this pythontk reads ({cls.VERSION})."
            )
        reference = data["reference"]
        if base and not os.path.isabs(reference):
            reference = os.path.normpath(os.path.join(base, reference))
        return cls(
            reference,
            [LookRegion.from_dict(r) for r in data.get("regions", [])],
            data.get("focus"),
            data.get("notes", ""),
            ids=data.get("ids"),
            base=base,
        )

    def save(self, path: str) -> str:
        """Write the target as JSON, the reference spelled relative to *path*'s folder."""
        FileUtils.write_json(
            path, self.to_dict(os.path.dirname(os.path.abspath(path))), indent=1
        )
        return path

    @classmethod
    def load(cls, path: str) -> "LookTarget":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f), os.path.dirname(os.path.abspath(path)))
