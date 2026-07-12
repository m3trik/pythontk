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

    def reset(self) -> None:
        """Clear per-session state (masks, progress counters).

        Call at the start of each independent batch — :meth:`process_batch`
        does this for you.
        """
        self.masks = []
        self.total_progress = 0
        self.total_len = 0
        self._batch_map_types = set()

    def process_batch(
        self,
        sorted_images: SortedImages,
        output_dir: str,
        name: str = "",
    ) -> BatchResult:
        """Drive a full composite → retry-with-mask → re-composite cycle."""
        self.reset()
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
        retried = self.retry_failed(failed, name)
        if not retried:
            return BatchResult.MASK_FAILURE
        self.composite_images(retried, output_dir, name)
        self.apply_output_template(output_dir)
        return BatchResult.RETRIED

    def apply_output_template(self, output_dir: str) -> List[str]:
        """Post-process composited output for a target workflow.

        No-op when ``output_template`` is unset. Otherwise loads the named
        pythontk workflow preset (see :class:`pythontk.core_utils.engines.textures.map_registry.WF`)
        and runs :meth:`pythontk.MapFactory.prepare_maps` on the files just
        written to ``output_dir``. The composited files stay on disk; the
        workflow adds packed / format-converted siblings alongside them.

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

        files = ptk.get_images(output_dir)
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

        def _progress(current: int, total: int, message: str) -> None:
            # Surface per-set progress in the UI panel — prepare_maps' own
            # logger writes to its class stream, which the engine's UI
            # handler doesn't see. logger= below catches everything else.
            self.logger.info(f"  [{current}/{total}] {message}")

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
            self.logger.error(f"Output template failed: {e}")
            return []

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
        failed: SortedImages = {}
        for typ, layers in sorted_images.items():
            if not self._composite_type(typ, layers, sorted_images, output_dir, name):
                failed[typ] = layers
        return failed

    def retry_failed(self, failed: SortedImages, name: str) -> SortedImages:
        """Fill the masked area of each failed layer with the map-type's
        known default background, so a second composite pass can succeed.

        Masks were captured from a *different* map type's layers during the
        first pass and are aligned positionally — ``self.masks[n]`` is
        assumed to apply to the n-th layer of any map type. This relies on
        all map types having the same per-layer ordering.
        """
        registry = ptk.MapRegistry()
        map_backgrounds = registry.get_map_backgrounds()
        map_modes = registry.get_map_modes()

        out: SortedImages = {}
        for typ, layers in failed.items():
            for n, (filepath, image) in enumerate(layers):
                try:
                    mask = self.masks[n]
                except IndexError:
                    self.logger.error(
                        f"Composite failed: <b>{name}_{typ}: {filepath}</b>"
                    )
                    continue

                key = ptk.MapFactory.resolve_map_type(typ)
                bg = map_backgrounds.get(key)
                if bg is None:
                    bg = ptk.get_background(image, "RGBA", average=True)
                    im = ptk.fill_masked_area(image, bg, mask)
                else:
                    im = ptk.fill_masked_area(image, bg, mask)
                    target_mode = map_modes.get(key)
                    if target_mode is not None:
                        im = im.convert(target_mode)

                out.setdefault(typ, []).append((filepath, im))
        return out

    def _seed_masks(
        self,
        sorted_images: SortedImages,
        fallback_typ: str,
        fallback_layers: Layers,
        fallback_bg: Tuple[int, int, int, int],
    ) -> List[Image.Image]:
        """Build per-layer masks from every map type in the batch whose
        layers carry an alpha channel over a transparent background.

        Per-layer alpha masks are OR-combined across sources so an
        antialiased / noisy boundary in one source is filled in by the
        others — far more robust than the legacy single-source,
        exact-color-match approach.

        Falls back to :func:`ptk.create_mask` against ``fallback_typ`` if
        no alpha-capable source qualifies.
        """
        n_layers = len(fallback_layers)
        sources: List[Tuple[str, Layers]] = []
        for typ, layers in sorted_images.items():
            if len(layers) != n_layers:
                continue
            first_image = layers[0][1]
            if "A" not in first_image.getbands():
                continue
            bg = ptk.get_background(first_image, "RGBA")
            if not bg or bg[3] != 0:
                continue
            sources.append((typ, layers))

        if not sources:
            self.logger.info(
                f"Attempting to create masks using source <b>{fallback_typ}</b> ..",
                preset="italic",
            )
            return ptk.create_mask([img for _, img in fallback_layers], fallback_bg)

        self.logger.info(
            f"Creating masks from <b>{len(sources)}</b> alpha source(s): "
            f"{', '.join(typ for typ, _ in sources)}",
            preset="italic",
        )

        masks: List[Image.Image] = []
        for i in range(n_layers):
            combined: Optional[np.ndarray] = None
            for _, layers in sources:
                im = layers[i][1].convert("RGBA")
                content = np.array(im)[:, :, 3] > 0
                combined = content if combined is None else (combined | content)
            mask_arr = combined.astype(np.uint8) * 255
            masks.append(Image.fromarray(mask_arr, mode="L"))
        return masks

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
        target_mode = map_modes[key]
        bit_depth = ptk.ImgUtils.format_bit_depth(target_mode)

        # PIL mode "I" (32bit int) cannot be created directly; route via RGB.
        if mode == "I":
            first_image = first_image.convert("RGB")

        bg = ptk.get_background(first_image, "RGBA")
        bg2 = ptk.get_background(second_image, "RGBA")
        if not (bg and bg == bg2):
            return False  # non-uniform / mismatched bg → mask retry path

        if not self.masks and bg[3] == 0:
            self.masks = self._seed_masks(sorted_images, typ, layers, bg)

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
            first_image, remaining, bg, mode, filepath0, fill_bg=fill_bg
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
        self._maybe_optimize(out_path, key)

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

    def _maybe_optimize(self, out_path: str, map_type: str) -> None:
        """Run MapOptimizer.optimize_map on the just-saved file when enabled.

        Optimization rewrites the file in place with map-type-correct bit
        depth and (optionally) a tighter mode. No-op when disabled.
        """
        if not self.optimize_output:
            return
        try:
            ptk.MapOptimizer.optimize_map(
                out_path,
                map_type=map_type,
                optimize_bit_depth=True,
            )
        except Exception as e:
            # Optimization is best-effort — never abort the batch.
            self.logger.warning(
                f"optimize_map failed for <b>{os.path.basename(out_path)}</b>: {e}"
            )

    def _alpha_composite_layers(
        self,
        first_image: Image.Image,
        remaining: Layers,
        bg: Tuple[int, int, int, int],
        mode: str,
        first_filepath: str,
        fill_bg: Optional[Tuple[int, int, int, int]] = None,
    ) -> Image.Image:
        composited = first_image.convert("RGBA")
        if fill_bg is not None:
            composited = self._fill_transparent_rgb(composited, fill_bg)
        self._tick(first_filepath)
        for filepath, im in remaining:
            self._tick(filepath)
            if mode == "I":
                im = im.convert("RGB")
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
            if "Normal_OpenGL" in inventory:
                return
            if self._try_invert_normal(
                result, typ, "Normal_DirectX", "Normal_OpenGL", output_dir, name, info
            ):
                self._warn_if_normal_format_mismatch(probe, declared="DirectX")
                return
            if "Normal_DirectX" not in inventory:
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
            try:
                os.remove(os.path.join(output_dir, f"{name}_{typ}.{info.ext}"))
            except OSError:
                pass

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
        map_types = ptk.MapRegistry().get_map_types()
        try:
            index = map_types[src_set].index(typ)
        except ValueError:
            return False
        new_type = map_types[dst_set][index]
        inverted = ptk.invert_channels(result, "g")
        inverted.save(os.path.join(output_dir, f"{name}_{new_type}.{info.ext}"))
        title = (
            f"{new_type.rstrip('_')} {info.mode} {info.bit_depth} "
            f"{info.ext.upper()} {info.width}x{info.height}:"
        )
        self.logger.log_group(title, [f"Created using {name}_{typ}.{info.ext}"])
        return True
