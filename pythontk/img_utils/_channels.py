# !/usr/bin/python
# coding=utf-8
"""Per-channel image operations: inversion, swizzling, packing grayscale maps
into channels and extracting them back out, and reading a normal map's
handedness from how its X and Y channels integrate (the bodies behind the
:class:`ImgUtils` facade).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Union, Any, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore
try:
    from PIL import Image, ImageOps, ImageChops
except ImportError:
    Image = ImageOps = ImageChops = None  # type: ignore


class _ImgChannelInternal:
    """Bodies of :class:`ImgUtils`' channel operations (an ``ImgUtils`` base).

    The public signatures and docstrings stay on the facade; each delegates
    here.
    """

    @classmethod
    def invert_channels(cls, image, channels="RGBA"):
        """Body of :meth:`ImgUtils.invert_channels`."""
        im = cls.ensure_image(image)
        split_channels = im.split()

        # Use the image's real band names (e.g. ('L','A'), ('R','G','B','A')) so
        # channels map correctly for every mode instead of the fixed "RGBA"[:n]
        # labels, which mislabel LA's alpha as 'G' and crash on merge.
        bands = im.getbands()
        requested = channels.upper()

        def _is_requested(name: str) -> bool:
            name = name.upper()
            # A single-band image's lone band carries the value data whatever
            # its mode label (L, P, I, F, 1), so it responds to an L/R/G/B
            # request — preserving the historical default where a grayscale
            # image is inverted. Keying on the 'L' label alone silently
            # no-opped paletted/bilevel/deep single-band images.
            if len(bands) == 1:
                return any(c in requested for c in "LRGB")
            return name in requested

        inverted = [
            ImageChops.invert(band) if _is_requested(name) else band
            for name, band in zip(bands, split_channels)
        ]

        if len(inverted) == 1:  # Single-band (e.g. grayscale) image
            return inverted[0]
        return Image.merge(im.mode, tuple(inverted))

    @classmethod
    def swizzle_channels(cls, image, mapping):
        """Body of :meth:`ImgUtils.swizzle_channels`."""
        im = cls.ensure_image(image)
        rgba = im.convert("RGBA")
        bands = dict(zip("RGBA", rgba.split()))

        def resolve(token):
            token = str(token).strip().upper()
            if token in ("0", "1"):
                return Image.new("L", rgba.size, 0 if token == "0" else 255)
            if token in bands:
                return bands[token]
            raise ValueError(
                f"swizzle_channels: invalid source '{token}'; expected one of "
                "R, G, B, A, 0, 1."
            )

        if isinstance(mapping, str):
            order = mapping.strip()
            if not 1 <= len(order) <= 4:
                raise ValueError(
                    "swizzle_channels: string mapping must be 1-4 characters."
                )
            out_bands = [resolve(c) for c in order]
            if len(out_bands) == 1:
                return out_bands[0]
            out_mode = {2: "LA", 3: "RGB", 4: "RGBA"}[len(out_bands)]
            return Image.merge(out_mode, tuple(out_bands))

        # dict mapping — output RGB, gaining alpha when the input already has
        # one or the mapping explicitly addresses the ``A`` destination.
        remap = {str(k).strip().upper(): v for k, v in mapping.items()}
        has_alpha = "A" in remap or "A" in im.getbands()
        dest_order = "RGBA" if has_alpha else "RGB"
        out_bands = [resolve(remap.get(dest, dest)) for dest in dest_order]
        return Image.merge(dest_order, tuple(out_bands))

    @classmethod
    def pack_channels(
        cls,
        channel_files: dict[str, str | Image.Image],
        channels: list[str] = None,
        out_mode: str = None,
        fill_values: dict[str, int] = None,
        output_path: str = None,
        output_format: str = "PNG",
        grayscale_to_rgb: bool = False,
        invert_channels: list[str] = None,
        **kwargs,
    ) -> str | Image.Image:
        """Body of :meth:`ImgUtils.pack_channels`."""
        if channels is None:
            channels = ["R", "G", "B", "A"]
        if fill_values is None:
            fill_values = {ch: 0 for ch in "RGB"}
            fill_values["A"] = 255
        if invert_channels is None:
            invert_channels = []

        has_alpha = bool(channel_files.get("A"))
        out_mode = out_mode or ("RGBA" if has_alpha else "RGB")
        n_channels = 4 if out_mode == "RGBA" else 3

        # Get first valid image for sizing
        first_file = next(
            (f for f in (channel_files.get(ch) for ch in channels) if f), None
        )
        if first_file is None:
            raise ValueError("No input images provided")
        size = cls.ensure_image(first_file).size

        # Determine if we should replicate grayscale to RGB (duplicate if only one RGB channel is used)
        used_rgb_channels = [ch for ch in "RGB" if channel_files.get(ch)]
        allow_duplicate = grayscale_to_rgb and len(used_rgb_channels) == 1
        r_img = (
            cls.ensure_image(channel_files.get("R"), mode="L").resize(size)
            if channel_files.get("R")
            else None
        )

        bands = []
        for ch in channels[:n_channels]:
            img_input = channel_files.get(ch)
            if img_input:
                # Load image once to avoid double I/O
                img_obj = cls.ensure_image(img_input)

                # Optimization: Check if image is constant
                # This avoids expensive resizing artifacts for small constant maps
                is_const, const_color = cls.is_image_constant(img_obj)

                if is_const:
                    # Convert constant color to grayscale
                    # Create 1x1 temp image to handle color conversion correctly
                    temp_img = Image.new(img_obj.mode, (1, 1), const_color)
                    gray_val = temp_img.convert("L").getpixel((0, 0))
                    band = cls.create_image("L", size, color=gray_val)
                else:
                    band = img_obj.convert("L").resize(size)
            elif ch in "GB" and allow_duplicate and r_img is not None:
                # Duplicate R into G/B if only R is used
                band = r_img
            else:
                band = cls.create_image("L", size, color=fill_values.get(ch, 0))

            if ch in invert_channels:
                band = ImageOps.invert(band)

            bands.append(band)

        img = Image.merge(out_mode, bands)

        if output_path:
            cls.save_image(img, output_path, format=output_format, **kwargs)
            return output_path
        return img

    @classmethod
    def pack_channel_into_alpha(
        cls,
        image: Union[str, Image.Image],
        alpha: Union[str, Image.Image],
        output_path: Optional[str] = None,
        invert_alpha: bool = False,
        resize_alpha: bool = True,
        preserve_existing_alpha: bool = False,
    ) -> str | Image.Image:
        """Body of :meth:`ImgUtils.pack_channel_into_alpha`."""
        base_img = cls.ensure_image(image).convert("RGBA")
        r, g, b, existing_alpha_channel = base_img.split()

        alpha_img = cls.ensure_image(alpha)

        final_alpha = alpha_img
        invert_list = ["A"] if invert_alpha else []

        if preserve_existing_alpha:
            # Pre-process alpha for multiplication
            if invert_alpha:
                alpha_img = cls.invert_grayscale_image(alpha_img)
                invert_list = []  # Already inverted

            alpha_img = alpha_img.convert("L")

            # Handle resizing for multiplication
            if alpha_img.size != base_img.size:
                if resize_alpha:
                    # Optimization: Check if alpha is constant
                    is_const, const_color = cls.is_image_constant(alpha_img)
                    if is_const:
                        alpha_img = cls.create_image(
                            "L", base_img.size, color=const_color[0]
                        )
                    else:
                        alpha_img = alpha_img.resize(
                            base_img.size, Image.Resampling.LANCZOS
                        )
                else:
                    raise ValueError(
                        f"Alpha image size {alpha_img.size} does not match base {base_img.size} and resize is disabled."
                    )

            final_alpha = ImageChops.multiply(existing_alpha_channel, alpha_img)

        return cls.pack_channels(
            channel_files={"R": r, "G": g, "B": b, "A": final_alpha},
            output_path=output_path,
            invert_channels=invert_list,
        )

    @classmethod
    def extract_channels(
        cls,
        image_path: Union[str, "Image.Image"],
        channel_config: Dict[str, Dict[str, Any]],
        output_dir: str = None,
        base_name: str = None,
        save: bool = True,
        **kwargs,
    ) -> Dict[str, Union[str, "Image.Image"]]:
        """Body of :meth:`ImgUtils.extract_channels`."""
        # Load image
        img = cls.ensure_image(image_path)

        # Determine output directory and base name
        if base_name is None:
            if isinstance(image_path, str):
                base_name = cls.get_base_texture_name(image_path)
            else:
                base_name = "texture"

        if output_dir is None:
            if isinstance(image_path, str):
                output_dir = os.path.dirname(image_path)
            else:
                output_dir = os.getcwd()

        if save and output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Extract format/extension from kwargs
        output_format = kwargs.pop("output_format", "PNG")
        ext = kwargs.pop("ext", "png")
        if not ext.startswith("."):
            ext = f".{ext}"

        results = {}

        # Helper to get channel safely
        def get_channel_data(source_mode, channel_name, default_val=None):
            # Handle RGB extraction
            if channel_name == "RGB":
                return img.convert("RGB")

            # Handle single channel extraction
            # Check if channel exists in image
            if channel_name in img.getbands():
                return img.getchannel(channel_name)

            # Handle fallback/default
            if default_val is not None:
                # Create constant image
                return Image.new("L", img.size, default_val)

            # If requesting R/G/B from L image, return the L image
            if source_mode == "L" and channel_name in "RGB":
                return img.copy()

            return None

        for src_chan, config in channel_config.items():
            suffix = config.get("suffix", f"_{src_chan}")
            invert = config.get("invert", False)
            default = config.get("default", None)

            extracted = get_channel_data(img.mode, src_chan, default)

            if extracted is None:
                # print(f"// Warning: Channel '{src_chan}' not found in image.")
                continue

            # Ensure L mode for single channels if they aren't already (getchannel returns L)
            if len(src_chan) == 1 and src_chan in "RGBA" and extracted.mode != "L":
                extracted = extracted.convert("L")

            # Invert if requested
            if invert:
                extracted = ImageOps.invert(extracted)

            if not save:
                results[src_chan] = extracted
                continue

            # Save
            out_path = os.path.join(output_dir, f"{base_name}{suffix}{ext}")
            cls.save_image(extracted, out_path, format=output_format, **kwargs)
            results[src_chan] = out_path

        return results

    @classmethod
    def detect_normal_map_format(
        cls,
        image: Union[str, "Image.Image"],
        threshold: float = 0.25,
        min_gradient_std: float = 1.0,
    ) -> Optional[str]:
        """Body of :meth:`ImgUtils.detect_normal_map_format`."""
        try:
            # convert("RGB") always returns our own copy (PIL copies even when
            # the mode already matches), so the in-place thumbnail() below never
            # mutates a caller-supplied Image.
            img = cls.ensure_image(image).convert("RGB")

            # Reducing first keeps the common case cheap, but it is a LOW-PASS
            # over exactly the gradients this statistic reads, so it cannot be
            # the last word: on maps whose relief is fine and shallow it
            # averages the signal flat (measured on real OpenGL bakes: r fell
            # -0.368 -> -0.105 and -0.19 -> -0.09, both under the threshold, a
            # correct answer downgraded to "don't know"). So the reduction is a
            # FAST PATH -- taken when it answers, re-read at native resolution
            # when it does not. Full res costs ~68 ms on a 2048 map against the
            # ~50 ms already spent decoding it, so the escalation is cheap and
            # only the indeterminate minority pays it.
            reduced = img
            if max(img.size) > 512:
                # `resize` rather than `copy() + thumbnail()`: thumbnail is
                # in-place, and the full-size image has to survive for the
                # re-read below, so taking it that way costs a full-size copy
                # first (48 MB on a 4k map). Same `reducing_gap` two-step, same
                # aspect rule, byte-identical output, and measurably faster.
                width, height = img.size
                scale = 512 / max(width, height)
                reduced = img.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    reducing_gap=2.0,
                )

            correlation = cls._normal_handedness_correlation(reduced, min_gradient_std)
            if (correlation is None or abs(correlation) <= threshold) and (
                reduced is not img
            ):
                full = cls._normal_handedness_correlation(img, min_gradient_std)
                if full is not None:
                    correlation = full

            if correlation is None:
                return None
            if correlation < -threshold:
                return "OpenGL"
            if correlation > threshold:
                return "DirectX"
            return None

        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Error detecting normal map format: {e}"
            )
            return None

    @staticmethod
    def _normal_handedness_correlation(
        img: "Image.Image", min_gradient_std: float
    ) -> Optional[float]:
        """``corr(dR/dy, dG/dx)`` for *img*, or ``None`` if it is meaningless.

        The integrability statistic behind :meth:`detect_normal_map_format`,
        split out so the same computation serves both the reduced fast path and
        the native-resolution re-read. Negative = OpenGL, positive = DirectX;
        the caller owns the threshold.

        ``None`` means "no usable signal here", not "flat": either channel's
        gradient falling under *min_gradient_std* (8-bit units) makes the
        correlation noise, and a non-finite result (a constant channel) is the
        same answer arrived at by division.
        """
        # Only R and G carry the signal, so only R and G are materialized --
        # `np.array(img)` would build the blue plane too, a third of the
        # allocation for nothing (measured on a 2048 map: 218 -> 201 MB peak,
        # and marginally faster). Identical correlation to 0e+00.
        dRy = np.gradient(  # dR/dy along image rows
            np.asarray(img.getchannel("R"), dtype=np.float32), axis=0
        ).ravel()
        dGx = np.gradient(  # dG/dx along image cols
            np.asarray(img.getchannel("G"), dtype=np.float32), axis=1
        ).ravel()
        # Variance floor: flat or near-flat inputs produce meaningless
        # correlations (often NaN, often spuriously signed).
        if dRy.std() < min_gradient_std or dGx.std() < min_gradient_std:
            return None
        correlation = np.corrcoef(dRy, dGx)[0, 1]
        return float(correlation) if np.isfinite(correlation) else None
