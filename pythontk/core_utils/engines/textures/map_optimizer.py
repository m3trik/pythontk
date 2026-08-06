# !/usr/bin/python
# coding=utf-8
"""Plan, assess, and apply map (texture) optimizations.

Split out of ``ImgUtils`` so the decision branches consumed by both
:meth:`MapOptimizer.optimize_map` and :meth:`MapOptimizer.assess`
live in a single planner. Prevents drift between "would change" predictions
and actual mutations.

Architecture:
    plan(image, **opts) -> [Op, ...]   # pure decisions, no IO
    apply(image, plan, ...) -> Image   # executes ops via ImgUtils helpers
    optimize_map(path, ...) -> str     # orchestrator: load + plan + apply + save
    assess(path, ...) -> dict          # wraps plan() with image read + reporting
"""
from __future__ import annotations

import os
import math

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

try:
    from PIL import Image
except ImportError as e:
    print(f"# ImportError: {__file__}\n\t{e}")
    Image = None  # type: ignore

# From this package:
from pythontk.core_utils.help_mixin import HelpMixin
from pythontk.file_utils._file_utils import FileUtils
from pythontk.img_utils._img_utils import ImgUtils
from pythontk.core_utils.engines.textures.map_factory import MapFactory
from pythontk.core_utils.engines.textures.map_registry import MapRegistry
from pythontk.core_utils.engines.textures.output_template import (
    DeliveryBudget,
    OutputSpec,
    OutputTemplates,
)


# Map-type-driven mode coercion rules. Mirrors the tolerated-mode lists in
# the original optimize_map body — defined once here so both plan() and
# apply() reference the same source.
_MAP_TYPE_TOLERATED: Dict[str, Tuple[str, ...]] = {
    "RGB": ("RGB", "P", "L"),
    "RGBA": ("RGBA", "PA", "P"),
    "L": ("L", "P"),
}

# Legacy map-type-key heuristics for keys without a MapRegistry entry. Kept
# verbatim from optimize_map's pre-extraction fallback so call-site
# behavior is preserved exactly.
_LEGACY_MAP_KEY_RULES: Dict[Tuple[str, ...], Tuple[str, Tuple[str, ...]]] = {
    ("Normal", "Normal_OpenGL", "Normal_DirectX"): ("RGB", ("RGB",)),
    ("MSAO", "MaskMap"): ("RGBA", ("RGBA",)),
    ("ORM",): ("RGB", ("RGB", "RGBA")),
}
_LEGACY_GRAYSCALE_KEYS = (
    "Ambient_Occlusion",
    "Roughness",
    "Metallic",
    "Smoothness",
    "Height",
    "Bump",
)


@dataclass
class Op:
    """One operation in an optimization plan.

    ``kind`` drives which ImgUtils helper :func:`apply` dispatches to. Each
    op carries a human-readable ``description`` so :meth:`assess` can surface
    the same wording without re-deriving it.
    """

    kind: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)


class MapOptimizer(HelpMixin):
    """Plan, assess, and apply map (texture) optimizations.

    All decision logic lives in :meth:`plan` — :meth:`apply` is a thin
    dispatcher that mutates the image according to the plan's ops, never
    making its own mode/size choices. :meth:`optimize_map` orchestrates
    load + plan + apply + save; :meth:`assess` is the read-only twin.

    Single source of truth: ``apply`` does not call ``ImgUtils.set_bit_depth``
    so its idiosyncratic ``bit_depth_mapping`` middle step can't introduce
    drift between predicted (``assess``) and actual (``optimize_map``)
    outputs. ``set_bit_depth`` remains available for direct callers.
    """

    @staticmethod
    def _fit_within(width: int, height: int, max_size: int) -> Tuple[int, int]:
        """Cap the longest edge at *max_size*, deriving the other from aspect."""
        if width >= height:
            return max_size, max(1, round(height * max_size / width))
        return max(1, round(width * max_size / height)), max_size

    @staticmethod
    def _snap_pot(
        width: int,
        height: int,
        mode: str = "nearest",
        max_size: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Snap both edges to a power of two.

        ``mode="down"`` never grows an edge — what a *budget* needs, since
        rounding to nearest inflates every dimension in the upper half of a
        band. ``max_size`` is a separate, harder rule: a stated ceiling must
        survive the snap, so any edge that crossed it is halved back under.
        Nearest still wins wherever it stays legal.
        """
        snap = math.floor if mode == "down" else round

        def _edge(value: int) -> int:
            snapped = max(1, 2 ** int(snap(math.log2(value))))
            while max_size and snapped > max_size and snapped > 1:
                snapped //= 2
            return snapped

        return _edge(width), _edge(height)

    @classmethod
    def plan(
        cls,
        image: "Image.Image",
        max_size: Optional[int] = None,
        force_pot: bool = False,
        optimize_bit_depth: bool = True,
        map_type_key: Optional[str] = None,
        allow_palette: bool = False,
        pot_mode: str = "nearest",
    ) -> List[Op]:
        """Return the ordered list of operations :meth:`apply` would run.

        Pure function: no file IO, no mutation. The planner tracks a
        ``logical`` mode/size as each op would change them, so downstream
        decisions (e.g. mode coercion before resize) see the post-prior-op
        state — same as if optimize_map were executing.

        Parameters:
            image: Source image (only its size/mode/info are read).
            max_size: Max edge length for the resize step. None disables.
            force_pot: Snap to a power-of-two if not already POT.
            optimize_bit_depth: Enable the strict-mode + wide-gamut fallback
                step (formerly delegated to set_bit_depth).
            map_type_key: Canonical map-type key from
                ``MapFactory.resolve_map_type(..., key=True)``. Drives the
                map-type mode coercion step.
            allow_palette: Preserve paletted images instead of upcasting.
            pot_mode: How ``force_pot`` snaps — ``"nearest"`` (a caller asking
                for POT wants the closest match) or ``"down"``. A *budget*
                always passes ``"down"``: rounding to nearest makes every
                dimension in the upper half of a POT band grow (1536 -> 2048,
                +78% pixels), so a delivery budget could inflate the very asset
                it exists to shrink, and then report itself as satisfied.

        Returns:
            list[Op]: Ordered ops; empty when no changes would be applied.
        """
        ops: List[Op] = []
        width, height = image.size
        mode = image.mode

        # --- Pre-compute resize and POT decisions (will_resize gates depalettize)
        # The resize caps the LONGEST edge and derives the other from the source
        # aspect: driving both edges from one scalar squashed every non-square
        # map to a square (harmless while only an explicit max_size reached it,
        # automatic for every budgeted profile once enforce_budget existed).
        resize_to: Optional[Tuple[int, int]] = (
            cls._fit_within(width, height, max_size)
            if max_size and max(width, height) > max_size
            else None
        )
        # max_size is passed into the snap, not pre-applied as a mode: a stated
        # ceiling must survive POT rounding whether or not the resize above
        # fired. Keying off "did a resize happen" misses the case where the
        # source already fits but the snap crosses the limit on its own (200
        # under a 250 ceiling snaps to 256).
        pot_target: Optional[Tuple[int, int]] = None
        if force_pot and width > 0 and height > 0:
            rw, rh = resize_to if resize_to else (width, height)
            pot = cls._snap_pot(rw, rh, pot_mode, max_size)
            if (rw, rh) != pot:
                pot_target = pot
        will_resize = resize_to is not None or pot_target is not None

        # --- 1. Depalettize before resize (preserves high-quality resampling)
        if will_resize and mode in ("P", "PA"):
            new_mode = (
                "RGBA"
                if (mode == "PA" or "transparency" in image.info)
                else "RGB"
            )
            ops.append(
                Op(
                    kind="depalettize",
                    description=f"Depalettize for resize: {mode} -> {new_mode}",
                    params={"target_mode": new_mode},
                )
            )
            mode = new_mode

        # --- 2. Map-type mode coercion (lines 1395+ of the original body)
        map_def = MapRegistry().get(map_type_key) if map_type_key else None
        if map_def and getattr(map_def, "mode", None):
            target_mode = map_def.mode
            tolerated = _MAP_TYPE_TOLERATED.get(target_mode, (target_mode,))
            if mode not in tolerated:
                ops.append(
                    Op(
                        kind="mode_coerce",
                        description=(
                            f"Mode (map_type={map_type_key}): "
                            f"{mode} -> {target_mode}"
                        ),
                        params={"target_mode": target_mode},
                    )
                )
                mode = target_mode
        elif map_type_key:
            # Legacy fallback for keys not in the registry.
            for keys, (target, tolerated) in _LEGACY_MAP_KEY_RULES.items():
                if map_type_key in keys and mode not in tolerated:
                    ops.append(
                        Op(
                            kind="mode_coerce",
                            description=(
                                f"Mode (legacy {map_type_key}): "
                                f"{mode} -> {target}"
                            ),
                            params={"target_mode": target},
                        )
                    )
                    mode = target
                    break
            else:
                if map_type_key in _LEGACY_GRAYSCALE_KEYS and mode == "P":
                    ops.append(
                        Op(
                            kind="mode_coerce",
                            description=(
                                f"Mode (legacy grayscale {map_type_key}): "
                                f"P -> L"
                            ),
                            params={"target_mode": "L"},
                        )
                    )
                    mode = "L"

        # --- 3. Resize
        if resize_to is not None:
            rw, rh = resize_to
            ops.append(
                Op(
                    kind="resize",
                    description=(
                        f"Resize: {width}x{height} -> "
                        f"{rw}x{rh} (exceeds max_size={max_size})"
                    ),
                    params={"size": resize_to},
                )
            )
            width, height = rw, rh

        # --- 4. Force POT (recompute against current dims, post-resize)
        if force_pot and width > 0 and height > 0:
            pw, ph = cls._snap_pot(width, height, pot_mode, max_size)
            if (width, height) != (pw, ph):
                ops.append(
                    Op(
                        kind="force_pot",
                        description=f"Force POT: {width}x{height} -> {pw}x{ph}",
                        params={"size": (pw, ph)},
                    )
                )
                width, height = pw, ph

        # --- 5. Strict-mode enforcement (post step-2 cleanups).
        # Mirrors set_bit_depth's two intentional branches: strict palette
        # upcast when a map_type target exists, and the wide-gamut fallback.
        # The bit_depth_mapping middle step in the original set_bit_depth is
        # deliberately NOT mirrored — it's the quirk that caused drift, and
        # bypassing it here makes plan the single source of truth.
        if optimize_bit_depth and (mode != "P" or not allow_palette):
            sb_target = (
                MapRegistry().get_map_modes().get(map_type_key)
                if map_type_key
                else None
            )
            sb_target_mode: Optional[str] = None
            sb_reason: Optional[str] = None

            if sb_target and mode != sb_target:
                # Catches the strict-palette upcast for tolerated-but-non-
                # target inputs (e.g. P -> RGB when allow_palette is False).
                sb_target_mode = sb_target
                sb_reason = f"Strict mode (map_type={map_type_key})"
            elif mode in ("HSV", "LAB", "CMYK", "YCbCr"):
                sb_target_mode = "RGBA" if mode == "CMYK" else "RGB"
                sb_reason = f"Unsupported mode {mode}"

            if sb_target_mode and sb_target_mode != mode:
                ops.append(
                    Op(
                        kind="mode_coerce",
                        description=(
                            f"{sb_reason}: {mode} -> {sb_target_mode}"
                        ),
                        params={"target_mode": sb_target_mode},
                    )
                )
                mode = sb_target_mode

        return ops

    @staticmethod
    def project(
        plan: List[Op],
        width: int,
        height: int,
        mode: str,
    ) -> Tuple[int, int, str]:
        """Replay ``plan``'s op params to get the post-apply size and mode.

        Pure arithmetic over the plan — no pixel work — so a caller that only
        wants to *show* the outcome (a dry run, a report row) doesn't pay for
        a resize. Every op that changes size or mode carries the resulting
        value in its params, making this a replay rather than a second copy
        of :meth:`plan`'s decision logic.

        Parameters:
            plan: Ops from :meth:`plan`.
            width, height, mode: The source image's starting state.

        Returns:
            tuple: ``(width, height, mode)`` after the plan would run.
        """
        for op in plan:
            if op.kind in ("depalettize", "mode_coerce"):
                mode = op.params.get("target_mode", mode)
            elif op.kind in ("resize", "force_pot"):
                size = op.params.get("size")
                if size is not None:
                    width, height = size
        return width, height, mode

    @classmethod
    def apply(
        cls,
        image: "Image.Image",
        plan: List[Op],
    ) -> "Image.Image":
        """Execute ``plan`` against ``image``. Returns the mutated image.

        Pure dispatcher: each branch performs the direct PIL mutation
        implied by the op's params. Apply never re-decides mode or size —
        ``plan`` has already made those choices.
        """
        for op in plan:
            if op.kind == "depalettize":
                image = ImgUtils.depalettize_image(image)
            elif op.kind == "mode_coerce":
                target = op.params["target_mode"]
                if image.mode != target:
                    image = image.convert(target)
            elif op.kind in ("resize", "force_pot"):
                # Both resize to the size the PLAN chose. force_pot must not
                # call ImgUtils.ensure_pot: that re-derives its own nearest-POT
                # target, which is exactly the apply-re-decides-a-size drift
                # this split exists to prevent (and it would silently ignore
                # a budget's round-down).
                w, h = op.params["size"]
                image = ImgUtils.resize_image(image, w, h)
        return image

    @classmethod
    def _resolve_profile(
        cls,
        output_profile: Optional[str],
        map_type_key: Optional[str],
        output_type: Optional[str],
        max_size: Optional[int],
        force_pot: Optional[bool],
        enforce_budget: bool,
    ) -> Tuple[
        Optional[OutputSpec],
        Optional[DeliveryBudget],
        Optional[str],
        Optional[int],
        bool,
        str,
    ]:
        """Apply an output profile's two tiers to a call's arguments.

        Both :meth:`optimize_map` and its dry-run twin :meth:`assess` have to
        read a profile the same way or the prediction stops describing the run —
        the same reason the decision branches live in :meth:`plan`. So the
        precedence rules live here once:

        - **hard tier** — the profile's :class:`OutputSpec` supplies the
          container, but only where the caller named none.
        - **advisory tier** — the profile's :class:`DeliveryBudget` supplies
          ``max_size`` / ``force_pot`` *only* when ``enforce_budget`` is set, and
          even then an explicit value from the caller still wins. Both use
          ``is None`` to mean "caller said nothing": ``force_pot or
          budget.force_pot`` could not tell an explicit ``False`` from unset, so
          the documented opt-out did not work.

        Returns:
            tuple: ``(spec, budget, output_type, max_size, force_pot, pot_mode)``.
            ``spec`` and ``budget`` are None when no profile is active.
            ``pot_mode`` is ``"down"`` only when the POT snap came from the
            budget — a budget must never grow an asset.
        """
        pot_mode = "nearest"
        if not output_profile:
            return None, None, output_type, max_size, bool(force_pot), pot_mode

        spec = OutputTemplates.resolve(map_type_key, output_profile)
        if not output_type:
            output_type = spec.ext

        budget = OutputTemplates.budget(output_profile)
        if enforce_budget:
            if max_size is None:
                max_size = budget.max_size
            if force_pot is None:
                force_pot = budget.force_pot
                pot_mode = "down"

        return spec, budget, output_type, max_size, bool(force_pot), pot_mode

    @classmethod
    def optimize_map(
        cls,
        texture_path: str,
        output_dir: str = None,
        output_type: str = None,
        max_size: int = None,
        force_pot: Optional[bool] = None,
        suffix_old: str = None,
        suffix_opt: str = None,
        old_files_folder: str = None,
        optimize_bit_depth: bool = True,
        check_existing: bool = False,
        map_type: str = None,
        allow_palette: bool = False,
        output_profile: str = None,
        enforce_budget: bool = False,
    ) -> str:
        """Optimizes a texture by resizing, setting bit depth, and adjusting image type.

        Parameters:
            texture_path (str): Path to the texture file.
            output_dir (str, optional): Directory for the optimized texture. Defaults to same directory.
            output_type (str, optional): Output image format (e.g., PNG, TGA). If None, keeps original.
            max_size (int, optional): Maximum size for the longest dimension. Only applies if the image is larger. Defaults to None.
            force_pot (bool): Force Power of Two dimensions.
            suffix_old (str, optional): Suffix to rename the original file before optimization.
            suffix_opt (str, optional): Suffix to append to the optimized file (None = overwrite).
            old_files_folder (str, optional): Name of the folder to store old files.
            optimize_bit_depth (bool): Adjusts bit depth to match the map type.
            check_existing (bool): If True, returns existing optimized file if it exists and is newer.
            map_type (str, optional): The type of map (e.g., "Normal", "MaskMap") to enforce specific modes.
            allow_palette (bool): If True, palette (P) inputs may be preserved
                when the target mode is RGB/RGBA. Default False (strict) — this
                prevents PNG palette-transparency from being read as alpha by
                downstream FBX/DCC pipelines.
            output_profile (str, optional): Workflow profile (a ``WF`` key) whose
                output template drives the container / bit depth / compression.
            enforce_budget (bool): Apply the profile's advisory
                ``DeliveryBudget`` (max_size / POT) instead of only reporting it.
                Default False — an over-budget map is correct, just expensive, so
                it is not silently resampled. An explicit ``max_size`` /
                ``force_pot`` always outranks the budget.

        Returns:
            str: Path to the optimized texture.
        """
        ImgUtils.assert_pathlike(texture_path, "texture_path")

        if output_dir is None:
            output_dir = os.path.dirname(texture_path)
        os.makedirs(output_dir, exist_ok=True)

        map_type_suffix = MapFactory.resolve_map_type(texture_path, key=False)
        if map_type_suffix is None:
            map_type_suffix = ""
        map_type_key = map_type or MapFactory.resolve_map_type(
            texture_path, key=True
        )

        # An active profile drives the output format and, on opt-in, the size
        # budget — resolved by the same helper assess uses, so the two agree.
        spec, budget, output_type, max_size, force_pot, pot_mode = cls._resolve_profile(
            output_profile,
            map_type_key,
            output_type,
            max_size,
            force_pot,
            enforce_budget,
        )
        target_bit_depth = spec.bit_depth if spec else None
        target_compression = spec.compression if spec else None

        # Calculate output path early to check for existence
        temp_path = MapFactory.resolve_texture_filename(
            texture_path,
            map_type_suffix,
            suffix=suffix_opt,
            ext=output_type,
        )
        final_output_path = os.path.join(output_dir, os.path.basename(temp_path))

        if check_existing and os.path.exists(final_output_path):
            if os.path.getmtime(final_output_path) > os.path.getmtime(texture_path):
                print(
                    f"Skipping optimization (existing/newer): "
                    f"{os.path.basename(final_output_path)}"
                )
                return final_output_path

        # Read the source's on-disk size before any archive/overwrite step can
        # move or replace it — it's the "from" half of the size report below.
        size_before = (
            os.path.getsize(texture_path) if os.path.isfile(texture_path) else None
        )

        image = ImgUtils.ensure_image(texture_path)
        dims_before = image.size

        plan = cls.plan(
            image,
            max_size=max_size,
            force_pot=force_pot,
            optimize_bit_depth=optimize_bit_depth,
            map_type_key=map_type_key,
            allow_palette=allow_palette,
            pot_mode=pot_mode,
        )

        if any(op.kind == "resize" for op in plan):
            # Match the original log line so existing tooling that scrapes
            # logs continues to work.
            print(
                f"Resizing {texture_path} from {image.size[0]}x{image.size[1]} "
                f"to {max_size}x{max_size} .."
            )

        image = cls.apply(image, plan)

        # File rename / archive handling — orchestrator concern, not planner.
        old_texture_path = (
            MapFactory.resolve_texture_filename(
                texture_path, map_type_suffix, suffix=suffix_old
            )
            if suffix_old
            else None
        )

        if old_files_folder:
            old_folder = os.path.join(output_dir, old_files_folder)
            FileUtils.move_file(
                texture_path,
                old_folder,
                new_name=(
                    os.path.basename(old_texture_path) if old_texture_path else None
                ),
            )

        # Route through the capability-aware writer (single save SSoT) so the
        # correct backend handles each format (PIL for most, cv2 for EXR/HDR).
        # The extension on final_output_path drives format dispatch; the profile
        # template (if any) supplies bit depth / compression.
        ImgUtils.save_image(
            image,
            final_output_path,
            optimize=True,
            bit_depth=target_bit_depth,
            compression=target_compression,
        )

        print(
            f"Saved optimized texture: {final_output_path} "
            f"({cls.format_result(final_output_path, size_before, dims_before, image)})"
        )

        # Reported against what was actually written, so an explicit max_size that
        # overshot the profile's budget still surfaces. Enforced runs land inside
        # the budget by construction and so report nothing.
        if budget:
            for message in budget.check(*image.size):
                print(
                    f"# Warning: {os.path.basename(final_output_path)} "
                    f"[{output_profile}]: {message}"
                )
        return final_output_path

    @staticmethod
    def format_result(
        output_path: str,
        size_before: Optional[int],
        dims_before: Optional[Tuple[int, int]],
        image: "Image.Image",
    ) -> str:
        """Render the one-line result summary for an optimized map.

        File size is the point of optimizing, so it leads the unchanged
        dimension/bit-depth pair with a ``before -> after`` transition. A
        dimension pair that didn't change renders once rather than as a
        no-op arrow.

        Parameters:
            output_path: Path just written (stat'd for the resulting size).
            size_before: Source byte count, or None when unknown.
            dims_before: Source ``(width, height)``, or None when unknown.
            image: The saved image (supplies final dims + bit depth).

        Returns:
            str: e.g. ``"4096x4096 -> 2048x2048, 24bit (8x3), 12.00 MB ->
                3.00 MB (-75%)"``.
        """
        width, height = image.size
        dims = f"{width}x{height}"
        if dims_before and tuple(dims_before) != (width, height):
            dims = f"{dims_before[0]}x{dims_before[1]} -> {dims}"

        size_after = (
            os.path.getsize(output_path) if os.path.isfile(output_path) else None
        )
        sizes = FileUtils.format_bytes_delta(size_before, size_after)

        return f"{dims}, {ImgUtils.format_bit_depth(image)}, {sizes}"

    @classmethod
    def batch_optimize_maps(cls, directory: str, **kwargs):
        """Batch optimizes all maps in a directory.

        Parameters:
            directory (str): Directory containing the maps to optimize.
            **kwargs: Forwarded to :meth:`optimize_map`.
        """
        ImgUtils.assert_pathlike(directory, "directory")

        textures = ImgUtils.get_images(directory)
        print(f"Optimizing maps in: {directory}")
        for texture_path in textures.keys():
            cls.optimize_map(texture_path, **kwargs)
        print(f"{len(textures)} maps optimized.")

    @classmethod
    def assess(
        cls,
        texture_path: str,
        max_size: int = None,
        force_pot: Optional[bool] = None,
        optimize_bit_depth: bool = True,
        map_type: str = None,
        allow_palette: bool = False,
        image: "Image.Image" = None,
        output_type: str = None,
        output_profile: str = None,
        predict_size: bool = False,
        enforce_budget: bool = False,
    ) -> Dict[str, Any]:
        """Predict whether :meth:`optimize_map` would change ``texture_path``.

        Read-only wrapper around :meth:`plan`. Returns a dict the UI / report
        callers can render without re-deriving decision strings. This is the
        dry-run twin of :meth:`optimize_map` — nothing on disk is touched.

        Parameters:
            texture_path: Path to the texture file.
            max_size, force_pot, optimize_bit_depth, map_type, allow_palette:
                Same semantics as :meth:`optimize_map`.
            image: Optional pre-loaded ``PIL.Image.Image`` to skip the
                redundant header read for callers that already have one open.
            output_type: Output format the run would write (e.g. "png"). Only
                affects the predicted extension / size; None keeps the source's.
            output_profile: Output-template profile the run would use. Resolved
                through the same :meth:`_resolve_profile` helper the real run
                uses, so a profile that dictates the extension / bit depth /
                compression is reflected in the prediction rather than silently
                ignored.
            predict_size: Also report the resulting *byte* count. Costs a real
                encode (the plan is applied to a copy and written to a scratch
                file that is immediately discarded), so it is opt-in — bulk
                report callers assessing every texture in a scene shouldn't
                pay for it. Compression ratios can't be derived from
                dimensions and mode, so this is the only honest way to
                answer "how much smaller?" before committing.
            enforce_budget: Same semantics as :meth:`optimize_map` — predict a
                run that applies the profile's advisory ``DeliveryBudget``
                rather than one that only reports it.

        Returns:
            dict with:
                recommended (bool): True if the plan is non-empty.
                reasons (list[str]): Per-op descriptions from the plan.
                warnings (list[str]): Advisory ``DeliveryBudget`` violations the
                    predicted output would still carry. Always present, empty
                    when there is no profile, no budget, or nothing to flag —
                    distinct from ``reasons``, which describe changes that
                    *would* be made.
                current (dict): {path, name, width, height, mode, format,
                    size_bytes, bit_depth, map_type}.
                predicted (dict): {width, height, mode, bit_depth, ext, path,
                    size_bytes}. ``path`` is where a default-location run would
                    write. ``size_bytes`` is None unless *predict_size* was
                    requested (or the encode failed, in which case
                    ``size_error`` carries the reason).
                target_mode (str | None): Map-type-driven target mode, when
                    one exists.
                error (str): Only present when the file could not be read.
        """
        ImgUtils.assert_pathlike(texture_path, "texture_path")

        if not os.path.exists(texture_path) and image is None:
            return {
                "recommended": False,
                "reasons": [],
                "warnings": [],
                "error": f"File not found: {texture_path}",
                "current": {
                    "path": texture_path,
                    "name": os.path.basename(texture_path),
                },
                "predicted": {},
                "target_mode": None,
            }

        size_bytes = (
            os.path.getsize(texture_path) if os.path.exists(texture_path) else None
        )

        try:
            if image is None:
                with ImgUtils.allow_large_images():
                    image = ImgUtils.ensure_image(texture_path)
            width, height = image.size
            mode = image.mode
            img_format = image.format
        except Exception as e:
            return {
                "recommended": False,
                "reasons": [],
                "warnings": [],
                "error": f"Failed to read image: {e}",
                "current": {
                    "path": texture_path,
                    "name": os.path.basename(texture_path),
                    "size_bytes": size_bytes,
                },
                "predicted": {},
                "target_mode": None,
            }

        map_type_key = map_type or MapFactory.resolve_map_type(
            texture_path, key=True
        )

        # Resolved before planning (the budget can add ops), through the same
        # helper optimize_map uses — so the dry run plans what the real run would.
        spec, budget, output_type, max_size, force_pot, pot_mode = cls._resolve_profile(
            output_profile,
            map_type_key,
            output_type,
            max_size,
            force_pot,
            enforce_budget,
        )

        ops = cls.plan(
            image,
            max_size=max_size,
            force_pot=force_pot,
            optimize_bit_depth=optimize_bit_depth,
            map_type_key=map_type_key,
            allow_palette=allow_palette,
            pot_mode=pot_mode,
        )

        # Surface the target mode the planner picked (first mode_coerce op),
        # if any. UI callers use this to render the current/target pair side
        # by side.
        target_mode: Optional[str] = None
        for op in ops:
            if op.kind == "mode_coerce":
                target_mode = op.params.get("target_mode")
                break

        # Replay the plan for the projected size/mode (cheap), then optionally
        # encode for the projected byte count (not cheap — hence opt-in).
        new_width, new_height, new_mode = cls.project(ops, width, height, mode)
        out_ext = (
            (output_type or FileUtils.format_path(texture_path, "ext"))
            .lower()
            .lstrip(".")
        )
        predicted: Dict[str, Any] = {
            "width": new_width,
            "height": new_height,
            "mode": new_mode,
            "bit_depth": ImgUtils.format_bit_depth(new_mode),
            "ext": out_ext,
            # Same resolve call optimize_map makes, so a dry run names the file
            # the real run would write (the map-type suffix gets normalized
            # here — predicting it by hand would drift).
            "path": MapFactory.resolve_texture_filename(
                texture_path,
                MapFactory.resolve_map_type(texture_path, key=False) or "",
                ext=output_type,
            ),
            "size_bytes": None,
        }
        if predict_size:
            predicted_bytes, size_error = cls._encoded_size(
                image,
                ops,
                out_ext,
                bit_depth=spec.bit_depth if spec else None,
                compression=spec.compression if spec else None,
            )
            predicted["size_bytes"] = predicted_bytes
            if size_error:
                predicted["size_error"] = size_error

        return {
            "recommended": bool(ops),
            "reasons": [op.description for op in ops],
            # Against the *predicted* dimensions — the question a caller is asking
            # is whether the file this run would write lands within budget, not
            # whether the source did.
            "warnings": budget.check(new_width, new_height) if budget else [],
            "current": {
                "path": texture_path,
                "name": os.path.basename(texture_path),
                "width": width,
                "height": height,
                "mode": mode,
                "format": img_format,
                "size_bytes": size_bytes,
                "bit_depth": ImgUtils.format_bit_depth(mode),
                "map_type": map_type_key,
            },
            "predicted": predicted,
            "target_mode": target_mode,
        }

    @classmethod
    def _encoded_size(
        cls,
        image: "Image.Image",
        plan: List[Op],
        ext: str,
        bit_depth: Optional[int] = None,
        compression: Optional[str] = None,
    ) -> Tuple[Optional[int], Optional[str]]:
        """Byte count ``plan`` would produce, measured by a throwaway encode.

        Routes through :meth:`ImgUtils.save_image` — the same writer
        :meth:`optimize_map` uses, with the same profile-driven bit depth and
        compression — so format dispatch (Pillow / cv2 / DDS) and mode fixups
        are identical to the real run instead of an approximation that could
        drift from it.

        The caller's image is passed through uncopied: every op in
        :meth:`apply` rebinds rather than mutating in place, which is the same
        invariant :meth:`optimize_map` already relies on, and copying an 8K
        texture just to throw it away is a real cost on the dry-run path.

        Returns:
            tuple: ``(size_bytes, error)`` — exactly one is non-None.
        """
        import shutil
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        scratch = TempArtifacts("map_optimizer_dryrun").dir_path()
        try:
            probe = os.path.join(scratch, f"probe.{ext}")
            ImgUtils.save_image(
                cls.apply(image, plan),
                probe,
                optimize=True,
                bit_depth=bit_depth,
                compression=compression,
            )
            return os.path.getsize(probe), None
        except Exception as e:
            return None, str(e)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
