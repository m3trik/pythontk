# !/usr/bin/python
# coding=utf-8
"""Pure image-compositing engine — alpha-composite layered texture maps
and auto-generate the complementary DirectX/OpenGL normal map.

No Qt, no UI imports. Status messages are written to ``self.logger``
(provided by :class:`ptk.LoggingMixin`); UI layers route output to a
text widget by calling ``self.logger.setup_logging_redirect(widget)``.
Progress-bar updates go through a thin ``progress_callback``.
"""

import os
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import pythontk as ptk

Layers = List[Tuple[str, Image.Image]]
SortedImages = Dict[str, Layers]
ProgressCallback = Callable[[float], None]


class BatchResult(Enum):
    """Outcome of a full composite + retry-with-mask cycle."""

    SUCCESS = "success"  # All maps composited on the first pass.
    RETRIED = "retried"  # Some required a mask retry; all eventually saved.
    MASK_FAILURE = "mask_failure"  # Some failed and no mask was available to recover.


class NormalOutputMode(Enum):
    """How the engine handles DirectX/OpenGL normal-map output."""

    BOTH = "both"  # Save the provided format + auto-generate the complement (default).
    OPENGL_ONLY = "opengl_only"  # Always output OpenGL; convert DirectX inputs.
    DIRECTX_ONLY = "directx_only"  # Always output DirectX; convert OpenGL inputs.
    NONE = "none"  # Pass inputs through as-is; do not synthesize a complement.


@dataclass(frozen=True)
class _MapInfo:
    """Per-map descriptor passed between engine helpers."""

    mode: str
    bit_depth: str
    ext: str
    width: int
    height: int


class MapCompositor(ptk.LoggingMixin):
    """Alpha-composite layered texture maps and auto-generate the
    complementary DirectX/OpenGL normal map when one is missing.

    Status messages are emitted via ``self.logger`` with HTML colouring.
    Attach to a Qt text widget with ``self.logger.setup_logging_redirect(widget)``.
    Progress-bar updates flow through ``progress_callback(percent)``.

    Alpha handling at partial-alpha edges
    -------------------------------------
    Source pixels with ``0 < alpha < 255`` have their RGB rewritten to the
    resolved background colour before each composite/paste so the blend
    reduces to ``bg ↔ bg`` at edges instead of ``bg ↔ 0``. This kills the
    dark-rim halos exporters seed by leaving ``RGB=0`` in transparent
    regions. The trade-off is intentional: partial-alpha pixels lose their
    authored RGB. That's correct for value maps (Roughness, Metallic,
    Ambient_Occlusion, Height) where alpha is purely a content mask. For
    colour maps with deliberate partial-alpha content (e.g. feathered
    foliage in Albedo_Transparency, soft-blended Base_Color edges), edge
    colour will be flattened to the registry default — bake the blend
    upstream if that matters.
    """

    def __init__(
        self,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        super().__init__()
        self._progress_cb: ProgressCallback = (
            progress_callback if progress_callback is not None else (lambda _p: None)
        )
        self.remove_normal_map: bool = True
        self.optimize_output: bool = False
        self.normal_output_mode: NormalOutputMode = NormalOutputMode.BOTH
        # Optional pythontk workflow preset key (e.g. WF.HDRP). When set, the
        # composited output is post-processed by MapFactory.prepare_maps so
        # files are packed/named for the target engine. None = composite-only.
        self.output_template: Optional[str] = None
        self.total_len: int = 0
        self.total_progress: int = 0
        self.masks: List[Image.Image] = []
        # Snapshot of the batch's full map-type inventory. The retry
        # pass only sees the failed subset, but normal-format decisions
        # (e.g. "skip auto-invert because Normal_OpenGL is already on
        # disk") must reason about the original source set.
        self._batch_map_types: set = set()
        # Background each retried type was filled with, keyed by map type.
        # The second composite pass uses it instead of re-probing corners —
        # a correctly-masked island that touches a corner would otherwise
        # make the filled layer look non-uniform again and fail silently.
        self._known_bg: Dict[str, Tuple[int, int, int, int]] = {}
        # Every file this batch wrote to disk, in write order. Scopes the
        # output-template post-pass to the batch's own output — without it
        # the post-pass would re-scan output_dir and sweep in unrelated
        # texture sets that happen to share the folder (e.g. a project-wide
        # sourceimages library).
        self._written_paths: List[str] = []
        # Drops the noisy fully-qualified logger name prefix from every
        # record without sacrificing the level tag (which still carries
        # colour information).
        self.logger.hide_logger_name()

    # Back-compat alias for the original camelCase attribute name.
    @property
    def removeNormalMap(self) -> bool:
        return self.remove_normal_map

    @removeNormalMap.setter
    def removeNormalMap(self, value: bool) -> None:
        self.remove_normal_map = value

    @property
    def written_paths(self) -> List[str]:
        """Files written by the most recent batch, in write order.

        Cleared by :meth:`reset` (and therefore by every
        :meth:`process_batch` call). Auto-generated normal complements are
        included; files the engine wrote and then removed are not.
        """
        return list(self._written_paths)

    def reset(self) -> None:
        """Clear per-session state (masks, progress counters).

        Call at the start of each independent batch — :meth:`process_batch`
        does this for you.
        """
        self.masks = []
        self.total_progress = 0
        self.total_len = 0
        self._batch_map_types = set()
        self._known_bg = {}
        self._written_paths = []

    def process_batch(
        self,
        sorted_images: SortedImages,
        output_dir: str,
        name: str = "",
    ) -> BatchResult:
        """Drive a full composite → retry-with-mask → re-composite cycle."""
        self.reset()
        sorted_images = self._normalize_bit_depth(sorted_images)
        self.total_len = sum(len(layers) for layers in sorted_images.values())
        self._batch_map_types = set(sorted_images.keys())
        failed = self.composite_images(sorted_images, output_dir, name)
        if not failed:
            self.apply_output_template(output_dir)
            return BatchResult.SUCCESS
        # Blank line above the phase marker so it visually separates the
        # first composite pass from the mask-retry pass — matches the
        # leading-newline convention log_group already uses.
        self.logger.log_raw("")
        self.logger.info(
            "Processing additional maps that require a mask ..", preset="italic"
        )
        if not self.masks:
            # Seeded lazily — a clean batch never pays for mask creation.
            self.masks = self._seed_masks(sorted_images)
        if not self.masks:
            # Nothing to key a mask off: no type has a detectable background,
            # i.e. the sources are dilated edge-to-edge (Painter's default
            # infinite padding). One diagnosis beats a per-file error for
            # each layer that was never going to succeed.
            self.logger.error(
                "No map type has a detectable background (checked "
                + ", ".join(f"<b>{typ}</b>" for typ in sorted_images)
                + ") — the sources are painted edge-to-edge, so nothing "
                "marks where each object ends."
            )
            return BatchResult.MASK_FAILURE
        retried = self.retry_failed(failed, name)
        if not retried:
            return BatchResult.MASK_FAILURE
        still_failed = self.composite_images(retried, output_dir, name)
        for typ, layers in still_failed.items():
            for filepath, _ in layers:
                self.logger.error(f"Composite failed: <b>{name}_{typ}: {filepath}</b>")
        if still_failed:
            return BatchResult.MASK_FAILURE
        self.apply_output_template(output_dir)
        return BatchResult.RETRIED

    def apply_output_template(
        self, output_dir: str, files: Optional[List[str]] = None
    ) -> List[str]:
        """Post-process composited output for a target workflow.

        No-op when ``output_template`` is unset. Otherwise loads the named
        pythontk workflow preset (see :class:`pythontk.core_utils.engines.textures.map_registry.WF`)
        and runs :meth:`pythontk.MapFactory.prepare_maps` on the files just
        written to ``output_dir``. The composited files stay on disk; the
        workflow adds packed / format-converted siblings alongside them.

        The post-pass is scoped to this batch's own output. ``files``
        defaults to :attr:`written_paths` — the paths the composite pass
        recorded. Only when the engine has written nothing (a standalone
        call on a pre-existing folder) does it fall back to scanning
        ``output_dir``. Scoping matters because ``output_dir`` is routinely
        a shared library folder: an unscoped scan would group every
        unrelated texture set in there into its own set and generate
        packed / converted siblings for materials the user never selected.

        Returns the list of files produced by the workflow (empty list when
        the template is unset, the dir is invalid, or no inputs were found).
        """
        if not self.output_template:
            return []

        if not output_dir or not os.path.isdir(output_dir):
            self.logger.warning(
                f"Skipping output template: <b>{output_dir!r}</b> is not a directory."
            )
            return []

        presets = ptk.MapRegistry().get_workflow_presets()
        if self.output_template not in presets:
            self.logger.warning(
                f"Unknown output template: <b>{self.output_template}</b>. "
                f"Skipping post-processing."
            )
            return []

        if files is None:
            # Path-only listing for the fallback: prepare_maps wants paths, and
            # get_images would decode every texture in the folder just to throw
            # the pixels away (whole 4K sets, for nothing). Same extension set.
            files = self._written_paths or ptk.FileUtils.get_dir_contents(
                output_dir,
                "filepath",
                inc_files=[f"*.{ext}" for ext in ptk.ImgUtils.texture_file_types],
            )
        # A batch can prune its own output (normal-format modes), and a
        # caller-supplied list may be stale — feed prepare_maps live paths.
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            return []

        # Strip non-config metadata before forwarding to prepare_maps.
        workflow_config = {
            k: v
            for k, v in presets[self.output_template].items()
            if k != "description"
        }

        self.logger.log_raw("")
        self.logger.info(
            f"Applying output template: <b>{self.output_template}</b> ..",
            preset="italic",
        )

        # Surface per-set progress in the UI panel — prepare_maps' own logger
        # writes to its class stream, which the engine's UI handler doesn't
        # see. logger= below catches everything else.
        #
        # Collected, not logged per tick: each log record renders as its own
        # paragraph in a text-widget handler, so a per-set line turned a
        # 20-set batch into 20 blank-line-separated sections. One log_group
        # after the run says the same thing as a single block.
        ticks: List[str] = []

        def _progress(current: int, total: int, message: str) -> None:
            ticks.append(f"[{current}/{total}] {message}")

        try:
            results = ptk.MapFactory.prepare_maps(
                files,
                output_dir=output_dir,
                logger=self.logger,
                progress_callback=_progress,
                output_profile=self.output_template,
                **workflow_config,
            )
        except Exception as e:
            # Report what completed BEFORE the error, not after it: a `finally`
            # here would print the progress block below the failure it preceded.
            # Guarded on ticks — the logger's log_group degrades an empty list
            # to a bare title, i.e. "Prepared 0 set(s)" for a failure that
            # happened before any set was reached.
            if ticks:
                self.logger.log_group(
                    f"Prepared {len(ticks)} set(s) before the failure", ticks
                )
            self.logger.error(f"Output template failed: {e}")
            return []
        if ticks:
            self.logger.log_group(f"Prepared {len(ticks)} set(s)", ticks)

        if isinstance(results, dict):
            return [p for paths in results.values() for p in paths]
        return list(results or [])

    def composite_images(
        self,
        sorted_images: SortedImages,
        output_dir: str,
        name: str = "",
    ) -> SortedImages:
        """Composite each map type and write the result.

        Returns the subset of map types whose layers had non-uniform
        backgrounds — those defer to :meth:`retry_failed`.
        """
        sorted_images = self._normalize_bit_depth(sorted_images)
        failed: SortedImages = {}
        for typ, layers in sorted_images.items():
            if not self._composite_type(typ, layers, sorted_images, output_dir, name):
                failed[typ] = layers
        return failed

    def retry_failed(self, failed: SortedImages, name: str) -> SortedImages:
        """Fill the masked area of each failed layer with the map-type's
        known default background, so a second composite pass can succeed.

        Masks come from :meth:`_seed_masks` (a union over every map type in
        the batch) and are aligned positionally — ``self.masks[n]`` is
        assumed to apply to the n-th layer of any map type. This relies on
        all map types having the same per-layer ordering. The fill colour
        is recorded per type in ``_known_bg`` so the second composite pass
        uses it instead of re-probing corners.
        """
        registry = ptk.MapRegistry()
        map_backgrounds = registry.get_map_backgrounds()
        map_modes = registry.get_map_modes()

        out: SortedImages = {}
        for typ, layers in failed.items():
            key = ptk.MapFactory.resolve_map_type(typ)
            bg = map_backgrounds.get(key)
            if bg is None:
                # No registered default: average the first layer's corners
                # once so every layer of the type shares ONE fill colour.
                bg = ptk.get_background(layers[0][1], "RGBA", average=True)
            bg = tuple(bg)
            target_mode = map_modes.get(key)
            for n, (filepath, image) in enumerate(layers):
                try:
                    mask = self.masks[n]
                except IndexError:
                    self.logger.error(
                        f"Composite failed: <b>{name}_{typ}: {filepath}</b>"
                    )
                    continue
                im = ptk.fill_masked_area(image, bg, mask)
                if target_mode is not None:
                    im = im.convert(target_mode)
                out.setdefault(typ, []).append((filepath, im))
            if typ in out:
                self._known_bg[typ] = bg
        return out

    @staticmethod
    def _normalize_bit_depth(sorted_images: SortedImages) -> SortedImages:
        """Reduce high-bit-depth grayscale layers (``I``, ``I;16*``, ``F``)
        to 8-bit ``L`` up front, rescaling the full range.

        The composite pipeline is 8-bit RGBA (``alpha_composite`` /
        ``paste``), and Pillow implements ``I;16 -> L/RGBA`` as a CLIP at
        255 — a Painter 16-bit Height map (mid-grey ~32767) converted
        implicitly loaded as solid white: its corners looked uniform, the
        composite came out white, and any mask seeded from it was empty.
        Routing through :meth:`ptk.ImgUtils.convert_i_to_l` (÷257, dtype-
        dispatched so float sources take the 0..1 rule) fixes every
        downstream ``convert`` at once. Idempotent — 8-bit layers pass
        through untouched.
        """
        out: SortedImages = {}
        for typ, layers in sorted_images.items():
            out[typ] = [
                (
                    (fp, ptk.ImgUtils.convert_i_to_l(im))
                    if im.mode == "I" or im.mode.startswith(("I;", "F"))
                    else (fp, im)
                )
                for fp, im in layers
            ]
        return out

    # Share of a layer's pixels the candidate bg must occupy for a type to
    # count as a mask source. Guards against a corner colour that is merely
    # content: real padding-off exports run 40-70% bg (a tightly packed set
    # still >10%), whereas an edge-to-edge (fully dilated) export's most
    # common colour is a flat *content* value that rarely exceeds a few
    # percent outside Height/Normal.
    _BG_MIN_SHARE = 0.1

    @classmethod
    def _solid_background(
        cls, arrays: List[np.ndarray]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Detect a type's background across its layers' RGBA arrays.

        Four identical corners are the clean signal, but a padding-off
        export routinely runs an island into a corner, so a majority (>= 2
        of 4, pooled over every layer) is accepted when that colour also
        holds :attr:`_BG_MIN_SHARE` of every layer. A transparent bg is
        the same rule keyed on alpha alone — exporters write arbitrary RGB
        under alpha 0. Returns the RGBA tuple, or None when nothing
        qualifies.
        """
        counts: Dict[Tuple[int, int, int, int], int] = {}
        for arr in arrays:
            for px in (arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]):
                key = tuple(int(v) for v in px)
                counts[key] = counts.get(key, 0) + 1
        if not counts:
            return None
        # Rank the corner colours by the AREA they actually cover, not by how
        # many corners they touch. A padding-off export runs content into the
        # corners, so a large island can tie the true background on corner
        # count -- and `max()` breaks a tie by scan order, which is arbitrary.
        # Picking the loser there inverts every mask: the composite then fills
        # the islands instead of the background, and because the wrong colour
        # is recorded the second pass "succeeds" via `_known_bg`, so the batch
        # reports RETRIED over silently destroyed maps. Coverage cannot tie
        # that way, and the gates below are unchanged.
        best_share, best = -1.0, None
        for candidate, n in counts.items():
            if n < 2 * len(arrays):
                continue
            share = min(
                float(cls._background_pixels(arr, candidate).mean())
                for arr in arrays
            )
            if share >= cls._BG_MIN_SHARE and share > best_share:
                best_share, best = share, candidate
        return best

    @staticmethod
    def _background_pixels(
        arr: np.ndarray, bg: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Boolean map of pixels that are ``bg``: alpha == 0 for a
        transparent bg, exact RGBA match for an opaque one."""
        if bg[3] == 0:
            return arr[:, :, 3] == 0
        return np.all(arr == np.array(bg, dtype=arr.dtype), axis=-1)

    # Share of two layer masks' union they may have in common before a
    # source is considered unreliable (per source) or the batch is warned
    # (final union). Sets composited into one map occupy disjoint UV space,
    # so anything past a few percent is dilation padding, a mis-detected
    # background, or content genuinely baked into both sets.
    _MASK_OVERLAP_MAX = 0.05

    def _seed_masks(self, sorted_images: SortedImages) -> List[Image.Image]:
        """Build one content mask per layer index from the map types in the
        batch that have a detectable background (:meth:`_solid_background`,
        transparent or opaque).

        Per-layer masks are OR-combined across types: content that is flat
        in one map (a Height map's mid-grey is both its bg AND its
        undisplaced surface; a Normal map's ``(127,127,255)`` likewise) is
        textured in another, so no single type is trusted to outline the
        islands. Types whose layer count differs from the majority are
        skipped so positional alignment holds.

        Sources are also vetted for reliability: a type whose own layer
        masks overlap past :attr:`_MASK_OVERLAP_MAX` (dilated islands, or a
        "background" that is really content) is dropped whenever at least
        one clean type exists, so one bad export cannot bleed a neighbour's
        padding into the union. Only when every source agrees the layers
        overlap is the batch warned — that is a genuine data issue.

        Returns ``[]`` when no type qualifies — the caller reports that.
        """
        if not sorted_images:
            return []
        n_layers = max(len(layers) for layers in sorted_images.values())
        per_type: Dict[str, List[np.ndarray]] = {}
        for typ, layers in sorted_images.items():
            if len(layers) != n_layers:
                continue
            arrays = [np.asarray(im.convert("RGBA")) for _, im in layers]
            bg = self._solid_background(arrays)
            if bg is None:
                continue
            per_type[typ] = [~self._background_pixels(arr, bg) for arr in arrays]

        if not per_type:
            return []
        overlaps = {typ: self._max_pairwise_overlap(m) for typ, m in per_type.items()}
        clean = [t for t, o in overlaps.items() if o <= self._MASK_OVERLAP_MAX]
        used = clean or list(per_type)
        for typ in per_type:
            if typ not in used:
                self.logger.info(
                    f"Skipping <b>{typ}</b> as a mask source — its layers overlap "
                    f"by {overlaps[typ]:.0%} (dilated islands or a background that "
                    "is really content); a cleaner source is available.",
                    preset="italic",
                )
        self.logger.info(
            f"Creating masks from <b>{len(used)}</b> source type(s): {', '.join(used)}",
            preset="italic",
        )
        combined: List[Optional[np.ndarray]] = [None] * n_layers
        for typ in used:
            for i, content in enumerate(per_type[typ]):
                combined[i] = content if combined[i] is None else (combined[i] | content)
        masks = [Image.fromarray(c.astype(np.uint8) * 255, mode="L") for c in combined]
        self._warn_on_mask_overlap(masks)
        return masks

    @staticmethod
    def _max_pairwise_overlap(masks: List[np.ndarray]) -> float:
        """Largest share of any two masks' union that both cover (0..1)."""
        worst = 0.0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                union = (masks[i] | masks[j]).sum()
                if union:
                    worst = max(worst, (masks[i] & masks[j]).sum() / union)
        return worst

    def _warn_on_mask_overlap(self, masks: List[Image.Image]) -> None:
        """Layers that combine into one map should occupy disjoint UV
        regions; heavy overlap means the sets share UV space or the
        detected background is really content — the composite there
        resolves last-layer-wins."""
        arrs = [np.array(m) > 0 for m in masks]
        for i in range(len(arrs)):
            for j in range(i + 1, len(arrs)):
                union = (arrs[i] | arrs[j]).sum()
                if not union:
                    continue
                overlap = (arrs[i] & arrs[j]).sum() / union
                if overlap > self._MASK_OVERLAP_MAX:
                    self.logger.warning(
                        f"Layer masks {i} and {j} overlap by {overlap:.0%} of "
                        "their union in every mask source — an object baked "
                        "into both sets, or shared UV space; overlapping "
                        "regions resolve last-layer-wins."
                    )

    def _composite_type(
        self,
        typ: str,
        layers: Layers,
        sorted_images: SortedImages,
        output_dir: str,
        name: str,
    ) -> bool:
        """Composite one map type. Returns False to defer to mask retry."""
        filepath0, first_image = layers[0]
        second_image = layers[1][1] if len(layers) > 1 else first_image
        remaining = layers[1:]
        width, height = first_image.size
        mode = first_image.mode
        ext = ptk.format_path(filepath0, "ext")
        key = ptk.MapFactory.resolve_map_type(typ)
        registry = ptk.MapRegistry()
        map_modes = registry.get_map_modes()
        # Mode-less packed types (MSAO/MRAO have mode=None, filtered out of
        # get_map_modes) and unresolved/None keys fall back to the image's
        # natural mode — exactly what mode=None is documented to intend.
        # Mirrors the safe .get() lookup already used in retry_failed.
        target_mode = map_modes.get(key, mode)
        bit_depth = ptk.ImgUtils.format_bit_depth(target_mode)

        bg = self._known_bg.get(typ)
        if bg is None:
            bg = ptk.get_background(first_image, "RGBA")
            bg2 = ptk.get_background(second_image, "RGBA")
            if not (bg and bg == bg2):
                return False  # non-uniform / mismatched bg → mask retry path

        title = (
            f"{typ.rstrip('_')} {target_mode} {bit_depth} "
            f"{ext.upper()} {width}x{height}:"
        )
        self.logger.log_group(title, [ptk.format_path(fp, "file") for fp, _ in layers])

        # Resolve the effective fill colour used both for the alpha_composite
        # pre-fill and the final solid bg. When corners are transparent, fall
        # back to the registered default; otherwise honour the artist's
        # opaque bg so we don't override a deliberate non-default choice.
        if bg[3] == 0:
            fill_bg = registry.get_map_backgrounds().get(key, bg)
        else:
            fill_bg = bg

        composited = self._alpha_composite_layers(
            first_image, remaining, bg, filepath0, fill_bg=fill_bg
        )

        # Replace src RGB with fill_bg at partial-alpha pixels so the paste
        # below blends bg↔bg at edges instead of bg↔0 — kills the dark/light
        # rim halo exporters seed by leaving RGB=0 in transparent regions.
        composited = self._fill_transparent_rgb(composited, fill_bg)

        result = Image.new("RGBA", composited.size, fill_bg[:3] + (255,))
        result.paste(composited, mask=composited)
        result = ptk.ImgUtils.set_bit_depth(result, key)
        mode = result.mode
        bit_depth = ptk.ImgUtils.format_bit_depth(mode)
        out_path = os.path.join(output_dir, f"{name}_{typ}.{ext}")
        result.save(out_path)
        # Track the post-optimization path — optimization can resolve a
        # different filename, and the template post-pass must be handed the
        # file that actually exists.
        self._record_written(self._maybe_optimize(out_path, key))

        info = _MapInfo(
            mode=mode, bit_depth=bit_depth, ext=ext, width=width, height=height
        )
        self._maybe_convert_normal(
            result,
            typ,
            sorted_images,
            output_dir,
            name,
            info,
            source=first_image,
            source_path=filepath0,
        )
        return True

    def _record_written(self, path: str) -> None:
        """Track a file this batch wrote. Idempotent — the mask-retry pass
        re-saves the same path, and the post-pass must see it once.
        """
        if path not in self._written_paths:
            self._written_paths.append(path)

    def _discard_written(self, path: str) -> None:
        """Untrack a file the engine wrote and then removed (the
        OPENGL_ONLY / DIRECTX_ONLY prune of the opposite-format source).
        """
        try:
            self._written_paths.remove(path)
        except ValueError:
            pass

    def _maybe_optimize(self, out_path: str, map_type: str) -> str:
        """Run MapOptimizer.optimize_map on the just-saved file when enabled.

        Optimization rewrites the file in place with map-type-correct bit
        depth and (optionally) a tighter mode. No-op when disabled.

        Returns the path the optimized map actually landed on — ``optimize_map``
        resolves its own output name (affix / extension normalization), so the
        result is not guaranteed to be ``out_path``. Falls back to ``out_path``
        when disabled or on failure, so the caller always gets a live path to
        track.
        """
        if not self.optimize_output:
            return out_path
        try:
            return (
                ptk.MapOptimizer.optimize_map(
                    out_path,
                    map_type=map_type,
                    optimize_bit_depth=True,
                )
                or out_path
            )
        except Exception as e:
            # Optimization is best-effort — never abort the batch.
            self.logger.warning(
                f"optimize_map failed for <b>{os.path.basename(out_path)}</b>: {e}"
            )
            return out_path

    def _alpha_composite_layers(
        self,
        first_image: Image.Image,
        remaining: Layers,
        bg: Tuple[int, int, int, int],
        first_filepath: str,
        fill_bg: Optional[Tuple[int, int, int, int]] = None,
    ) -> Image.Image:
        composited = first_image.convert("RGBA")
        if fill_bg is not None:
            composited = self._fill_transparent_rgb(composited, fill_bg)
        self._tick(first_filepath)
        for filepath, im in remaining:
            self._tick(filepath)
            im = ptk.replace_color(im, from_color=bg, mode="RGBA")
            if fill_bg is not None:
                im = self._fill_transparent_rgb(im, fill_bg)
            try:
                composited = Image.alpha_composite(composited, im.convert("RGBA"))
            except ValueError as e:
                self.logger.error(
                    f"alpha_composite failed for <b>{ptk.format_path(filepath, 'file')}</b>: {e}"
                )
        return composited

    @staticmethod
    def _fill_transparent_rgb(
        image: Image.Image, bg: Tuple[int, int, int, int]
    ) -> Image.Image:
        """Overwrite RGB with ``bg`` wherever alpha < 255.

        Prevents dark/light rim halos when a subsequent ``alpha_composite``
        or ``paste(..., mask=...)`` blends the source against a solid bg.
        Common failure mode: exporters write (0,0,0,α) in semi-transparent
        edges; the later blend then biases the result toward 0 instead of
        toward bg. After this pass the blend reduces to bg↔bg at edges
        (i.e. stays at bg).

        Destructive on partial alpha — partial-alpha pixels lose their
        authored RGB and adopt ``bg`` instead. Fully-opaque pixels are
        untouched. This is correct for value maps where alpha is a content
        mask; for colour maps with deliberate partial-alpha content, this
        flattens edge colour. See :class:`MapCompositor` docstring for the
        wider rationale. No-op for non-RGBA inputs.
        """
        if image.mode != "RGBA":
            return image
        arr = np.array(image)
        mask = arr[:, :, 3] < 255
        if not mask.any():
            return image
        arr[mask, 0:3] = bg[:3]
        return Image.fromarray(arr, mode="RGBA")

    def _tick(self, filepath: str) -> None:
        """Advance the global progress counter.

        File-name logging is handled up-front by ``log_group`` so the whole
        category renders as one block; ticking here is progress-only.
        """
        self.total_progress += 1
        if self.total_len:
            self._progress_cb((self.total_progress / self.total_len) * 100)

    def _maybe_convert_normal(
        self,
        result: Image.Image,
        typ: str,
        sorted_images: SortedImages,
        output_dir: str,
        name: str,
        info: _MapInfo,
        source: Optional[Image.Image] = None,
        source_path: Optional[str] = None,
    ) -> None:
        """Generate / suppress the complementary normal map according to
        ``normal_output_mode``:

        * ``BOTH`` — emit the missing complement (existing behavior)
        * ``OPENGL_ONLY`` — emit only Normal_OpenGL; delete the DirectX
          variant if it was just written
        * ``DIRECTX_ONLY`` — symmetric to OPENGL_ONLY
        * ``NONE`` — never auto-convert
        """
        mode = self.normal_output_mode

        if mode is NormalOutputMode.NONE:
            return

        map_types = ptk.MapRegistry().get_map_types()
        in_dx = typ in map_types["Normal_DirectX"]
        in_gl = typ in map_types["Normal_OpenGL"]
        if not (in_dx or in_gl):
            return  # not a normal map at all

        # Probe the on-disk source. The in-memory ``source`` may have been
        # rewritten by the retry pass (mask + map_backgrounds fill), which
        # seeds a faint gradient at the mask boundary and pushes
        # borderline integrability correlations across the detector
        # threshold — producing a false-positive format-mismatch warning.
        probe = None
        if source_path:
            try:
                probe = ptk.ImgUtils.load_image(source_path)
            except Exception:
                probe = None
        if probe is None:
            probe = source if source is not None else result

        # Decide complement existence against the batch-wide inventory,
        # not just ``sorted_images``. The retry pass only carries the
        # failed subset; without the batch snapshot the BOTH branch would
        # re-invert and clobber a user-provided complement on retry.
        inventory = self._batch_map_types or set(sorted_images.keys())

        if mode is NormalOutputMode.BOTH:
            # Emit the complement of whichever format this map is, and only
            # when it's genuinely absent from the batch — that keeps the
            # anti-clobber guard while staying symmetric. Testing the
            # *counterpart* rather than short-circuiting on OpenGL matters:
            # a GL-only batch is the common case, and the blanket guard left
            # it with no DirectX complement at all. The DX/GL variant sets
            # are disjoint, so at most one branch applies.
            if in_dx and "Normal_OpenGL" not in inventory:
                if self._try_invert_normal(
                    result,
                    typ,
                    "Normal_DirectX",
                    "Normal_OpenGL",
                    output_dir,
                    name,
                    info,
                ):
                    self._warn_if_normal_format_mismatch(probe, declared="DirectX")
            elif in_gl and "Normal_DirectX" not in inventory:
                if self._try_invert_normal(
                    result,
                    typ,
                    "Normal_OpenGL",
                    "Normal_DirectX",
                    output_dir,
                    name,
                    info,
                ):
                    self._warn_if_normal_format_mismatch(probe, declared="OpenGL")
            return

        if mode is NormalOutputMode.OPENGL_ONLY:
            target_format, src_set, dst_set, declared = (
                "OpenGL",
                "Normal_DirectX",
                "Normal_OpenGL",
                "DirectX",
            )
        elif mode is NormalOutputMode.DIRECTX_ONLY:
            target_format, src_set, dst_set, declared = (
                "DirectX",
                "Normal_OpenGL",
                "Normal_DirectX",
                "OpenGL",
            )
        else:
            return  # unexpected mode — fail closed instead of misrouting

        # Source already matches target → no conversion, but delete any
        # opposite-format file we wrote earlier this same batch wouldn't
        # exist (each typ is processed once).
        if (target_format == "OpenGL" and in_gl) or (
            target_format == "DirectX" and in_dx
        ):
            return

        # Source is the opposite format → invert into target, then delete
        # the source file we just saved.
        if self._try_invert_normal(
            result, typ, src_set, dst_set, output_dir, name, info
        ):
            self._warn_if_normal_format_mismatch(probe, declared=declared)
            pruned = os.path.join(output_dir, f"{name}_{typ}.{info.ext}")
            try:
                os.remove(pruned)
            except OSError:
                pass
            self._discard_written(pruned)

    def _warn_if_normal_format_mismatch(
        self, image: Image.Image, declared: str
    ) -> None:
        """Surface-integrability check: warn when the actual pixel content
        of a normal map disagrees with its declared format. Best-effort —
        swallows exceptions and numpy's zero-variance RuntimeWarning.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                detected = ptk.MapFactory.detect_normal_map_format(image)
        except Exception:
            return
        if detected and detected != declared:
            self.logger.warning(
                f"Normal map declared <b>{declared}</b> but pixel analysis "
                f"suggests <b>{detected}</b>. The auto-generated complement "
                f"may be incorrect — verify the source file's naming."
            )

    def _try_invert_normal(
        self,
        result: Image.Image,
        typ: str,
        src_set: str,
        dst_set: str,
        output_dir: str,
        name: str,
        info: _MapInfo,
    ) -> bool:
        registry = ptk.MapRegistry()
        if typ not in registry.get_map_types()[src_set]:
            return False
        # Swap the convention tag rather than pairing the two alias tuples by
        # index — that read required both lists to stay the same length and in
        # lockstep order, and raised IndexError for the one alias (`DXN`) that
        # had no counterpart. `sort_images_by_type` keys by canonical type today,
        # so this normally returns `dst_set` unchanged; it stays spelling-aware
        # because the membership test above accepts any alias.
        new_type = registry.counterpart_normal_spelling(typ, dst_set)
        inverted = ptk.invert_channels(result, "g")
        inverted_path = os.path.join(output_dir, f"{name}_{new_type}.{info.ext}")
        inverted.save(inverted_path)
        self._record_written(inverted_path)
        title = (
            f"{new_type.rstrip('_')} {info.mode} {info.bit_depth} "
            f"{info.ext.upper()} {info.width}x{info.height}:"
        )
        self.logger.log_group(title, [f"Created using {name}_{typ}.{info.ext}"])
        return True
