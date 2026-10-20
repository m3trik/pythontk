# !/usr/bin/python
# coding=utf-8
"""Comparison sheets: what a person judges a look match by.

One image: our frame beside the reference (regions outlined), a difference
map, then each region close up on both sides with its scores and, given
contributions, which lights make its light.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a declared dependency
    np = None  # type: ignore
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - Pillow is a declared dependency
    Image = ImageDraw = ImageFont = None  # type: ignore

from pythontk.core_utils.engines.lighting.look_match.measures import LookScore
from pythontk.core_utils.engines.lighting.look_match.target import (
    LookRegion,
    LookTarget,
)
from pythontk.img_utils._img_utils import ImgUtils


class LookSheet:
    """Renders comparison sheets.

    Parameters:
        width: The sheet's width in pixels; frames and crops scale to it.
        crop_height: The height of a region row.
        padding: How much of a region's own size its close-up adds around it.
    """

    BACKGROUND = (24, 24, 24)
    TEXT = (225, 225, 225)
    DIM = (150, 150, 150)
    OURS_OUTLINE = (255, 200, 60)
    #: The frames' captions' height at the sheet's scale: region names stay below it.
    CAPTION_BAND = 24
    REF_OUTLINE = (90, 200, 255)
    GOOD, BAD = (110, 200, 110), (230, 110, 100)
    BAR_COLORS = (
        (236, 160, 80),
        (110, 170, 230),
        (150, 210, 120),
        (210, 120, 200),
        (230, 220, 110),
        (130, 130, 130),
    )

    def __init__(
        self, width: int = 1640, crop_height: int = 260, padding: float = 0.35
    ):
        self.width = int(width)
        self.crop_height = int(crop_height)
        self.padding = float(padding)
        self.last_layout: Dict[str, object] = {}
        try:
            self.font = ImageFont.load_default(size=14)
            self.small = ImageFont.load_default(size=12)
        except TypeError:  # Pillow < 10.1: one bitmap size
            self.font = self.small = ImageFont.load_default()

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def _to_image(frame: "np.ndarray") -> "Image.Image":
        return Image.fromarray((np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8))

    def _outline(
        self,
        image: "Image.Image",
        region: LookRegion,
        side: str,
        color,
        top: float = 0.0,
    ) -> None:
        draw = ImageDraw.Draw(image)
        w, h = image.size
        outline = region.outline(side)
        if outline is not None:
            points, closed = outline
            line = [(x * w, y * h) for x, y in points]
            draw.line(line + line[:1] if closed else line, fill=color, width=2)
        else:  # its own edge, lightly tinted
            mask = region.mask(side, (w, h))
            inner = mask.copy()
            inner[1:, :] &= mask[:-1, :]
            inner[:-1, :] &= mask[1:, :]
            inner[:, 1:] &= mask[:, :-1]
            inner[:, :-1] &= mask[:, 1:]
            pixels = np.asarray(image).copy()
            tint = np.asarray(color[:3], np.float32)
            pixels[mask] = (0.8 * pixels[mask] + 0.2 * tint).astype(np.uint8)
            pixels[mask & ~inner] = tint.astype(np.uint8)
            image.paste(Image.fromarray(pixels))
            draw = ImageDraw.Draw(image)
        draw.text(
            self._label_origin(region, side, (w, h), top),
            region.name,
            fill=color,
            font=self.small,
        )

    def _label_origin(
        self, region: LookRegion, side: str, size, top: float = 0.0
    ) -> tuple:
        """Where *region*'s name goes: just above its top-left, else just
        inside it -- never in the band above *top* (the frame's caption, in
        the image's own pixels): a back wall's band put its name under it."""
        w, h = size
        x0, y0 = region.anchor(side, (w, h))
        above = y0 * h - 15
        return (x0 * w + 3, above if above >= top else max(y0 * h + 3, top))

    def _tag(self, draw, xy, text: str, fill, font) -> None:
        """*text* at *xy* on a dark plate: readable over any frame."""
        box = draw.textbbox(xy, text, font=font)
        draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(0, 0, 0))
        draw.text(xy, text, fill=fill, font=font)

    def _bounds(
        self, region: LookRegion, side: str, size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """The region's window on *side*, padded and kept inside the frame."""
        x0, y0, x1, y1 = region.window(side, size)
        pad_x = int((x1 - x0) * self.padding) + 4
        pad_y = int((y1 - y0) * self.padding) + 4
        x0, x1 = x0 - pad_x, x1 + pad_x
        y0, y1 = y0 - pad_y, y1 + pad_y
        least = (x1 - x0) / 3.0  # a thin region keeps some context above and below
        if y1 - y0 < least:
            cy = (y0 + y1) / 2.0
            y0, y1 = int(cy - least / 2), int(cy + least / 2)
        w, h = size
        return max(0, x0), max(0, y0), min(w, x1), min(h, y1)

    def difference(self, ours: "np.ndarray", ref: "np.ndarray") -> "Image.Image":
        """Where *ours* is brighter or darker than *ref* (display frames of one
        size): the log2 display-luma ratio, red brighter, blue darker, +/-1
        stop full scale -- the sheet's difference row, and a viewer's."""
        lo, lr = ImgUtils.luminance(ours), ImgUtils.luminance(ref)
        d = np.clip(np.log2((lo + 1 / 255) / (lr + 1 / 255)), -1.0, 1.0)
        rgb = np.zeros(d.shape + (3,), np.float32)
        rgb[..., 0] = np.clip(d, 0, 1)
        rgb[..., 2] = np.clip(-d, 0, 1)
        rgb[..., 1] = 0.12 * (1 - np.abs(d))
        return self._to_image(rgb)

    def _score_lines(self, results) -> List[Tuple[str, Tuple[int, int, int]]]:
        lines = []
        for name, result in results.items():
            score = result.score
            if score != score:  # nan
                continue
            if result.unit == "log":
                text = f"{name} x{np.exp(score):.2f}"
            else:
                text = f"{name} {score:+.1f} {result.unit}"
            # within the tolerance as LookScore.summary judges it (its default)
            lines.append((text, self.GOOD if result.miss <= 0.1 else self.BAD))
        return lines

    #: The measures charted under a region of each kind (when it took them).
    CHARTS = {"quad": ("columns", "rows"), "line": ("profile",), "area": ()}

    def _region_row(
        self, region, frames, outlined, size, score, shares
    ) -> "Image.Image":
        """One region close up: both sides, its scores, profile charts, its lights."""
        half = self.width // 2
        results = score.region(region.name) if score is not None else {}
        panels = []
        if (
            region.kind == "quad"
        ):  # what is compared: the rectified grids, cut-out cells dimmed
            for frame, side, x in ((frames[0], "ours", 0), (frames[1], "ref", half)):
                grid, keep = region.rectify(frame, side)
                img = self._to_image(np.where(keep[..., None], grid, grid * 0.3))
                rows_, cols = grid.shape[:2]
                h = int(
                    min(self.crop_height * 0.6, max(24, (half - 8) * rows_ / cols * 3))
                )
                panels.append((img.resize((half - 8, h), Image.NEAREST), x))
        else:  # the outlined frame around the region, so the measured pixels show
            for img, side, x in ((outlined[0], "ours", 0), (outlined[1], "ref", half)):
                x0, y0, x1, y1 = self._bounds(region, side, size)
                if x1 > x0 and y1 > y0:
                    crop = img.crop((x0, y0, x1, y1))
                else:
                    crop = Image.new("RGB", (1, 1))
                scale = min((half - 8) / crop.width, self.crop_height / crop.height)
                new_size = (
                    max(1, int(crop.width * scale)),
                    max(1, int(crop.height * scale)),
                )
                panels.append((crop.resize(new_size, Image.LANCZOS), x))
        charts = [key for key in self.CHARTS[region.kind] if key in results]
        panel_h = max(p.height for p, _ in panels)
        chart_h = 96 if charts else 0
        bars_h = 36 if shares else 0
        row = Image.new(
            "RGB", (self.width, 22 + panel_h + chart_h + bars_h + 4), self.BACKGROUND
        )
        for img, x in panels:
            row.paste(img, (x + (half - img.width) // 2, 20))
        draw = ImageDraw.Draw(row)
        draw.text((6, 3), region.name, fill=self.TEXT, font=self.font)
        cursor = 200
        for text, color in self._score_lines(results):
            draw.text((cursor, 4), text, fill=color, font=self.small)
            cursor += int(draw.textlength(text, font=self.small)) + 18
        y = 22 + panel_h
        if charts:
            width = (self.width - 16 - 8 * (len(charts) - 1)) // len(charts)
            for i, key in enumerate(charts):
                result = results[key]
                self._chart(
                    draw,
                    8 + i * (width + 8),
                    y + 4,
                    width,
                    chart_h - 10,
                    result.ours,
                    result.ref,
                    key,
                )
            y += chart_h
        if shares:
            self._bars(draw, 8, y + 2, half - 16, shares)
        return row

    def _chart(
        self, draw, x: int, y: int, w: int, h: int, ours, ref, label: str
    ) -> None:
        """Two profiles on one scale: ours (orange) against the reference (blue)."""
        o = np.asarray(ours, np.float64).ravel()
        r = np.asarray(ref, np.float64).ravel()
        draw.rectangle([x, y, x + w, y + h], outline=(60, 60, 60))
        draw.text(
            (x + 4, y + 2),
            f"{label}: ours vs reference",
            fill=self.DIM,
            font=self.small,
        )
        if o.size < 2 or r.size < 2:
            return
        top = max(float(o.max()), float(r.max()), 1e-6)

        def points(values):
            step = (w - 2) / (len(values) - 1)
            return [
                (x + 1 + i * step, y + h - 2 - (v / top) * (h - 18))
                for i, v in enumerate(values)
            ]

        draw.line(points(r), fill=self.REF_OUTLINE, width=2)
        draw.line(points(o), fill=self.OURS_OUTLINE, width=2)

    def _bars(
        self, draw, x: int, y: int, width: int, shares: Mapping[str, float]
    ) -> int:
        """A stacked bar of the largest shares; returns the height used."""
        top = sorted(shares.items(), key=lambda kv: -kv[1])[:5]
        rest = 1.0 - sum(v for _, v in top)
        if rest > 0.005:
            top.append(("other", rest))
        cursor = x
        for i, (name, share) in enumerate(top):
            seg = int(round(share * width))
            draw.rectangle(
                [cursor, y, cursor + seg, y + 14],
                fill=self.BAR_COLORS[i % len(self.BAR_COLORS)],
            )
            cursor += seg
        labels = "  ".join(f"{name} {share:.0%}" for name, share in top)
        draw.text((x, y + 17), labels, fill=self.DIM, font=self.small)
        return 36

    # ------------------------------------------------------------------ sheet
    def region_row(
        self,
        ours: "np.ndarray",
        ref: "np.ndarray",
        target: LookTarget,
        region: str,
        score: Optional[LookScore] = None,
        contributions: Optional[Mapping[str, float]] = None,
    ) -> "Image.Image":
        """Region *region* close up, as the sheet shows it: both sides outlined,
        its scores, its charts, its lights (*contributions*, ``{layer:
        share}``) -- a panel's detail of one region, the same picture a sheet
        exports.

        Raises:
            KeyError: *target* has no such region.
        """
        chosen = target.region(region)
        size = (ours.shape[1], ours.shape[0])
        o_img, r_img = self._to_image(ours), self._to_image(ref)
        self._outline(o_img, chosen, "ours", self.OURS_OUTLINE, 0)
        self._outline(r_img, chosen, "ref", self.REF_OUTLINE, 0)
        return self._region_row(
            chosen, (ours, ref), (o_img, r_img), size, score, contributions
        )

    def render(
        self,
        ours: "np.ndarray",
        ref: "np.ndarray",
        target: LookTarget,
        score: Optional[LookScore] = None,
        regions: Optional[Sequence[str]] = None,
        contributions: Optional[Mapping[str, Mapping[str, float]]] = None,
        title: str = "",
    ) -> "Image.Image":
        """The sheet as an image.

        Parameters:
            ours, ref: Display frames (sRGB 0..1) of one size.
            target: Whose regions to outline and crop.
            score: Its scores, printed per region and for the frame.
            regions: The regions to show close up (default: all).
            contributions: ``{region: {layer: share}}`` -- drawn as bars.
            title: A heading.
        """
        size = (ours.shape[1], ours.shape[0])
        chosen = (
            [target.region(n) for n in regions]
            if regions is not None
            else list(target.regions)
        )
        half = self.width // 2
        frame_h = int(round(half * size[1] / size[0]))
        rows: List[Image.Image] = []

        head = Image.new("RGB", (self.width, 26), self.BACKGROUND)
        draw = ImageDraw.Draw(head)
        draw.text((8, 5), title or "look match", fill=self.TEXT, font=self.font)
        if score is not None:
            frame_lines = self._score_lines(score.frame())
            draw.text(
                (half, 5),
                "   ".join(t for t, _ in frame_lines),
                fill=self.DIM,
                font=self.small,
            )
        rows.append(head)

        o_img, r_img = self._to_image(ours), self._to_image(ref)
        for region in chosen:  # names clear of the captions (CAPTION_BAND)
            for img, side, color in (
                (o_img, "ours", self.OURS_OUTLINE),
                (r_img, "ref", self.REF_OUTLINE),
            ):
                top = self.CAPTION_BAND * img.height / frame_h
                self._outline(img, region, side, color, top)
        frames = Image.new("RGB", (self.width, frame_h), self.BACKGROUND)
        frames.paste(o_img.resize((half, frame_h), Image.LANCZOS), (0, 0))
        frames.paste(r_img.resize((half, frame_h), Image.LANCZOS), (half, 0))
        captions = ImageDraw.Draw(frames)
        self._tag(captions, (6, 4), "OURS", self.OURS_OUTLINE, self.font)
        self._tag(captions, (half + 6, 4), "REFERENCE", self.REF_OUTLINE, self.font)
        rows.append(frames)

        diff_h = frame_h // 2
        diff = Image.new("RGB", (self.width, diff_h), self.BACKGROUND)
        diff.paste(
            self.difference(ours, ref).resize((half // 1, diff_h), Image.BILINEAR),
            (0, 0),
        )
        ImageDraw.Draw(diff).text(
            (half + 8, 8),
            "Difference (log2 luma, +/-1 stop):\nred = ours brighter, blue = darker.\nApproximate where the geometry differs.",
            fill=self.DIM,
            font=self.small,
        )
        rows.append(diff)

        for region in chosen:
            rows.append(
                self._region_row(
                    region,
                    (ours, ref),
                    (o_img, r_img),
                    size,
                    score,
                    (contributions or {}).get(region.name),
                )
            )

        sheet = Image.new(
            "RGB", (self.width, sum(r.height for r in rows)), self.BACKGROUND
        )
        cursor = 0
        for row in rows:
            sheet.paste(row, (0, cursor))
            cursor += row.height
        self.last_layout = {
            "regions": [r.name for r in chosen],
            "frame_height": frame_h,
        }
        return sheet

    def save(self, path: str, ours, ref, target: LookTarget, **kwargs) -> str:
        """Render (see :meth:`render`) and write *path* (PNG)."""
        self.render(ours, ref, target, **kwargs).save(path)
        return path

    #: Where a pair looks closer (frame fractions ``(x0, y0, x1, y1)``): the
    #: subject, what stands behind it, what it stands on.
    CROPS = {
        "desk": (0.17, 0.33, 0.83, 0.70),
        "behind": (0.05, 0.0, 0.95, 0.36),
        "floor": (0.0, 0.62, 1.0, 1.0),
    }

    def pair(
        self,
        ours: "np.ndarray",
        ref: "np.ndarray",
        crops: Optional[Mapping[str, Tuple[float, float, float, float]]] = None,
        title: str = "",
    ) -> "Image.Image":
        """Ours beside the reference: the frames, then each crop enlarged.

        The image a look is judged by -- by eye, before any number: what the
        measures cannot say (a pool's shape, where a highlight sits, how dark
        the room reads) is in front of whoever looks.

        Parameters:
            ours, ref: Display frames (sRGB 0..1, any size; each is fitted to
                half the sheet's width).
            crops: ``{name: (x0, y0, x1, y1)}`` in frame fractions (default
                :attr:`CROPS`).
        """
        half = self.width // 2
        rows = []
        for name, box in [("frame", (0.0, 0.0, 1.0, 1.0))] + list(
            (crops if crops is not None else self.CROPS).items()
        ):
            row = []
            for frame in (ours, ref):
                image = self._to_image(np.asarray(frame))
                w, h = image.size
                piece = image.crop(
                    (int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))
                )
                scale = half / piece.width
                row.append(
                    piece.resize(
                        (half, max(1, int(piece.height * scale))), Image.LANCZOS
                    )
                )
            rows.append((name, row))
        height = 26 + sum(row[0].height + 20 for _, row in rows)
        sheet = Image.new("RGB", (self.width, height), self.BACKGROUND)
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (8, 6), title or "ours | reference", fill=(230, 230, 230), font=self.font
        )
        y = 26
        for name, (left, right) in rows:
            draw.text((8, y + 2), name, fill=(170, 170, 170), font=self.small)
            sheet.paste(left, (0, y + 18))
            sheet.paste(right, (half, y + 18))
            y += left.height + 20
        return sheet

    def tiles(
        self, frames: Mapping[str, "np.ndarray"], columns: int = 2
    ) -> "Image.Image":
        """Labelled frames in a grid, row by row: candidates side by side -- the
        reference among them -- to choose between by eye.

        Parameters:
            frames: ``{label: display frame}`` (sRGB 0..1); each tile is
                ``width / columns`` wide, at the first frame's aspect.
            columns: Tiles per row.
        """
        items = list(frames.items())
        if not items:
            raise ValueError("No frames to tile.")
        tile_w = self.width // columns
        first = np.asarray(items[0][1])
        tile_h = max(1, round(tile_w * first.shape[0] / first.shape[1]))
        rows = -(-len(items) // columns)
        sheet = Image.new("RGB", (self.width, rows * tile_h), self.BACKGROUND)
        draw = ImageDraw.Draw(sheet)
        for i, (label, frame) in enumerate(items):
            x, y = (i % columns) * tile_w, (i // columns) * tile_h
            tile = self._to_image(np.asarray(frame)).resize(
                (tile_w, tile_h), Image.LANCZOS
            )
            sheet.paste(tile, (x, y))
            self._tag(draw, (x + 6, y + 4), label, self.TEXT, self.small)
        return sheet
