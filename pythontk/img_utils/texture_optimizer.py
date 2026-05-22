# !/usr/bin/python
# coding=utf-8
"""Plan, assess, and apply texture optimizations.

Split out of ``ImgUtils`` so the decision branches consumed by both
:meth:`TextureOptimizer.optimize_texture` and :meth:`TextureOptimizer.assess`
live in a single planner. Prevents drift between "would change" predictions
and actual mutations.

Architecture:
    plan(image, **opts) -> [Op, ...]   # pure decisions, no IO
    apply(image, plan, ...) -> Image   # executes ops via ImgUtils helpers
    optimize_texture(path, ...) -> str # orchestrator: load + plan + apply + save
    assess(path, ...) -> dict          # wraps plan() with image read + reporting
"""
from __future__ import annotations

import os
import math

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage

try:
    from PIL import Image
except ImportError as e:
    print(f"# ImportError: {__file__}\n\t{e}")
    Image = None  # type: ignore

# From this package:
from pythontk.core_utils.help_mixin import HelpMixin
from pythontk.file_utils._file_utils import FileUtils
from pythontk.img_utils._img_utils import ImgUtils
from pythontk.img_utils.map_factory import MapFactory
from pythontk.img_utils.map_registry import MapRegistry


# Map-type-driven mode coercion rules. Mirrors the tolerated-mode lists in
# the original optimize_texture body — defined once here so both plan() and
# apply() reference the same source.
_MAP_TYPE_TOLERATED: Dict[str, Tuple[str, ...]] = {
    "RGB": ("RGB", "P", "L"),
    "RGBA": ("RGBA", "PA", "P"),
    "L": ("L", "P"),
}

# Legacy map-type-key heuristics for keys without a MapRegistry entry. Kept
# verbatim from optimize_texture's pre-extraction fallback so call-site
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


class TextureOptimizer(HelpMixin):
    """Plan, assess, and apply texture optimizations.

    All decision logic lives in :meth:`plan` — :meth:`apply` is a thin
    dispatcher that mutates the image according to the plan's ops, never
    making its own mode/size choices. :meth:`optimize_texture` orchestrates
    load + plan + apply + save; :meth:`assess` is the read-only twin.

    Single source of truth: ``apply`` does not call ``ImgUtils.set_bit_depth``
    so its idiosyncratic ``bit_depth_mapping`` middle step can't introduce
    drift between predicted (``assess``) and actual (``optimize_texture``)
    outputs. ``set_bit_depth`` remains available for direct callers.
    """

    @classmethod
    def plan(
        cls,
        image: "Image.Image",
        max_size: Optional[int] = None,
        force_pot: bool = False,
        optimize_bit_depth: bool = True,
        map_type_key: Optional[str] = None,
        allow_palette: bool = False,
        generate_mipmaps: bool = False,
    ) -> List[Op]:
        """Return the ordered list of operations :meth:`apply` would run.

        Pure function: no file IO, no mutation. The planner tracks a
        ``logical`` mode/size as each op would change them, so downstream
        decisions (e.g. mode coercion before resize) see the post-prior-op
        state — same as if optimize_texture were executing.

        Parameters:
            image: Source image (only its size/mode/info are read).
            max_size: Max edge length for the resize step. None disables.
            force_pot: Snap to nearest power-of-two if not already POT.
            optimize_bit_depth: Enable the strict-mode + wide-gamut fallback
                step (formerly delegated to set_bit_depth).
            map_type_key: Canonical map-type key from
                ``MapFactory.resolve_map_type(..., key=True)``. Drives the
                map-type mode coercion step.
            allow_palette: Preserve paletted images instead of upcasting.
            generate_mipmaps: Append the mipmap-generation step.

        Returns:
            list[Op]: Ordered ops; empty when no changes would be applied.
        """
        ops: List[Op] = []
        width, height = image.size
        mode = image.mode

        # --- Pre-compute resize and POT decisions (will_resize gates depalettize)
        resize_to: Optional[int] = (
            max_size if max_size and max(width, height) > max_size else None
        )
        pot_target: Optional[Tuple[int, int]] = None
        if force_pot and width > 0 and height > 0:
            pw, ph = (
                2 ** round(math.log2(width)),
                2 ** round(math.log2(height)),
            )
            if (width, height) != (pw, ph):
                pot_target = (pw, ph)
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
            ops.append(
                Op(
                    kind="resize",
                    description=(
                        f"Resize: {width}x{height} -> "
                        f"{resize_to}x{resize_to} (exceeds max_size={resize_to})"
                    ),
                    params={"size": resize_to},
                )
            )
            width = height = resize_to

        # --- 4. Force POT (recompute against current dims, post-resize)
        if force_pot and width > 0 and height > 0:
            pw, ph = (
                2 ** round(math.log2(width)),
                2 ** round(math.log2(height)),
            )
            if (width, height) != (pw, ph):
                ops.append(
                    Op(
                        kind="force_pot",
                        description=f"Force POT: {width}x{height} -> {pw}x{ph}",
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
                ImgUtils.map_modes.get(map_type_key) if map_type_key else None
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

        # --- 6. Mipmaps
        if generate_mipmaps:
            ops.append(
                Op(kind="mipmaps", description="Generate mipmaps")
            )

        return ops

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
            elif op.kind == "resize":
                s = op.params["size"]
                image = ImgUtils.resize_image(image, s, s)
            elif op.kind == "force_pot":
                image = ImgUtils.ensure_pot(image)
            elif op.kind == "mipmaps":
                image = ImgUtils.generate_mipmaps(image)
        return image

    @classmethod
    def optimize_texture(
        cls,
        texture_path: str,
        output_dir: str = None,
        output_type: str = None,
        max_size: int = None,
        force_pot: bool = False,
        suffix_old: str = None,
        suffix_opt: str = None,
        old_files_folder: str = None,
        generate_mipmaps: bool = False,
        optimize_bit_depth: bool = True,
        check_existing: bool = False,
        map_type: str = None,
        allow_palette: bool = False,
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
            generate_mipmaps (bool): Generates mipmaps if enabled.
            optimize_bit_depth (bool): Adjusts bit depth to match the map type.
            check_existing (bool): If True, returns existing optimized file if it exists and is newer.
            map_type (str, optional): The type of map (e.g., "Normal", "MaskMap") to enforce specific modes.
            allow_palette (bool): If True, palette (P) inputs may be preserved
                when the target mode is RGB/RGBA. Default False (strict) — this
                prevents PNG palette-transparency from being read as alpha by
                downstream FBX/DCC pipelines.

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

        image = ImgUtils.ensure_image(texture_path)

        plan = cls.plan(
            image,
            max_size=max_size,
            force_pot=force_pot,
            optimize_bit_depth=optimize_bit_depth,
            map_type_key=map_type_key,
            allow_palette=allow_palette,
            generate_mipmaps=generate_mipmaps,
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

        save_kwargs = {"optimize": True}
        image.save(
            final_output_path, format=output_type or image.format, **save_kwargs
        )

        print(
            f"Saved optimized texture: {final_output_path} "
            f"({image.size[0]}x{image.size[1]}, {ImgUtils.format_bit_depth(image)})"
        )
        return final_output_path

    @classmethod
    def batch_optimize_textures(cls, directory: str, **kwargs):
        """Batch optimizes all textures in a directory.

        Parameters:
            directory (str): Directory containing the textures to optimize.
            **kwargs: Forwarded to :meth:`optimize_texture`.
        """
        ImgUtils.assert_pathlike(directory, "directory")

        textures = ImgUtils.get_images(directory)
        print(f"Optimizing textures in: {directory}")
        for texture_path in textures.keys():
            cls.optimize_texture(texture_path, **kwargs)
        print(f"{len(textures)} textures optimized.")

    @classmethod
    def assess(
        cls,
        texture_path: str,
        max_size: int = None,
        force_pot: bool = False,
        optimize_bit_depth: bool = True,
        map_type: str = None,
        allow_palette: bool = False,
        generate_mipmaps: bool = False,
        image: "Image.Image" = None,
    ) -> Dict[str, Any]:
        """Predict whether :meth:`optimize_texture` would change ``texture_path``.

        Read-only wrapper around :meth:`plan`. Returns a dict the UI / report
        callers can render without re-deriving decision strings.

        Parameters:
            texture_path: Path to the texture file.
            max_size, force_pot, optimize_bit_depth, map_type, allow_palette,
            generate_mipmaps: Same semantics as :meth:`optimize_texture`.
            image: Optional pre-loaded ``PIL.Image.Image`` to skip the
                redundant header read for callers that already have one open.

        Returns:
            dict with:
                recommended (bool): True if the plan is non-empty.
                reasons (list[str]): Per-op descriptions from the plan.
                current (dict): {path, name, width, height, mode, format,
                    size_bytes, bit_depth, map_type}.
                target_mode (str | None): Map-type-driven target mode, when
                    one exists.
                error (str): Only present when the file could not be read.
        """
        ImgUtils.assert_pathlike(texture_path, "texture_path")

        if not os.path.exists(texture_path) and image is None:
            return {
                "recommended": False,
                "reasons": [],
                "error": f"File not found: {texture_path}",
                "current": {
                    "path": texture_path,
                    "name": os.path.basename(texture_path),
                },
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
                "error": f"Failed to read image: {e}",
                "current": {
                    "path": texture_path,
                    "name": os.path.basename(texture_path),
                    "size_bytes": size_bytes,
                },
                "target_mode": None,
            }

        map_type_key = map_type or MapFactory.resolve_map_type(
            texture_path, key=True
        )

        ops = cls.plan(
            image,
            max_size=max_size,
            force_pot=force_pot,
            optimize_bit_depth=optimize_bit_depth,
            map_type_key=map_type_key,
            allow_palette=allow_palette,
            generate_mipmaps=generate_mipmaps,
        )

        # Surface the target mode the planner picked (first mode_coerce op),
        # if any. UI callers use this to render the current/target pair side
        # by side.
        target_mode: Optional[str] = None
        for op in ops:
            if op.kind == "mode_coerce":
                target_mode = op.params.get("target_mode")
                break

        return {
            "recommended": bool(ops),
            "reasons": [op.description for op in ops],
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
            "target_mode": target_mode,
        }
