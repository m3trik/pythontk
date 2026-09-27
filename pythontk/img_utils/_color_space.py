# !/usr/bin/python
# coding=utf-8
"""Colour-space transfer functions: sRGB <-> linear, black-body colour and
HDR-for-web encoding (the bodies behind the :class:`ImgUtils` facade).
"""

from __future__ import annotations

import math
from typing import Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore
try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


class _ImgColorSpaceInternal:
    """Bodies of :class:`ImgUtils`' colour-space transfers (an ``ImgUtils`` base).

    The public signatures and docstrings stay on the facade; each delegates
    here.
    """

    @staticmethod
    def kelvin_to_linear_rgb(
        kelvin: float, normalize: bool = True
    ) -> Tuple[float, float, float]:
        """Body of :meth:`ImgUtils.kelvin_to_linear_rgb`."""
        temperature = max(1000.0, min(40000.0, float(kelvin))) / 100.0

        if temperature <= 66.0:
            red = 255.0
            green = 99.4708025861 * math.log(temperature) - 161.1195681661
        else:
            red = 329.698727446 * ((temperature - 60.0) ** -0.1332047592)
            green = 288.1221695283 * ((temperature - 60.0) ** -0.0755148492)

        if temperature >= 66.0:
            blue = 255.0
        elif temperature <= 19.0:
            blue = 0.0
        else:
            blue = 138.5177312231 * math.log(temperature - 10.0) - 305.0447927307

        srgb = [max(0.0, min(255.0, channel)) / 255.0 for channel in (red, green, blue)]
        # sRGB EOTF -- the fit outputs display-referred values.
        linear = [
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb
        ]
        if normalize:
            peak = max(linear) or 1.0
            linear = [c / peak for c in linear]
        return tuple(linear)

    @staticmethod
    def _srgb_to_linear_np(arr):
        """Convert sRGB values to linear.

        Accepts a NumPy array or array-like. Values can be either 0-255 or 0-1.
        Returns float32 in [0,1]. Alpha channel (if present) is preserved.
        """
        a = np.asarray(arr)
        # Convert to float32 for calculation
        if a.dtype != np.float32 and a.dtype != np.float64:
            a = a.astype(np.float32)

        alpha = None
        if a.ndim == 3 and a.shape[-1] == 4:
            alpha = a[..., 3:4]
            a = a[..., :3]

        # Normalize to [0,1] if needed
        if a.max() > 1.0:
            a = a / 255.0

        a = np.clip(a, 0.0, 1.0)
        k0 = 0.04045
        out = np.empty_like(a, dtype=np.float32)
        low = a <= k0
        out[low] = a[low] / 12.92
        out[~low] = ((a[~low] + 0.055) / 1.055) ** 2.4

        if alpha is not None:
            out = np.concatenate([out, alpha], axis=-1)
        return out

    @staticmethod
    def _linear_to_srgb_np(arr):
        """Convert linear values to sRGB.

        Accepts a NumPy array in [0,1] and returns float32 in [0,1].
        """
        a = np.asarray(arr)
        if a.dtype != np.float32 and a.dtype != np.float64:
            a = a.astype(np.float32)

        alpha = None
        if a.ndim == 3 and a.shape[-1] == 4:
            alpha = a[..., 3:4]
            a = a[..., :3]

        a = np.clip(a, 0.0, 1.0)
        k1 = 0.0031308
        out = np.empty_like(a, dtype=np.float32)
        low = a <= k1
        out[low] = a[low] * 12.92
        out[~low] = 1.055 * (a[~low] ** (1.0 / 2.4)) - 0.055

        if alpha is not None:
            out = np.concatenate([out, alpha], axis=-1)
        return out

    @classmethod
    def _srgb_to_linear_image(cls, img: Image.Image) -> Image.Image:
        """Convert a PIL image (L/RGB/RGBA) from sRGB to linear, returned as 8-bit per channel.

        Alpha channel (if present) is preserved untouched.
        """
        arr = np.array(img, dtype=np.float32)
        if img.mode in ("L", "RGB", "RGBA"):
            # Normalize to [0,1] (incl. alpha) before the helper so its
            # `a.max() > 1.0` re-normalization gate never trips on the RGB
            # slice and leaves alpha at 0-255 (which then clips to 255).
            lin = cls._srgb_to_linear_np(arr / 255.0)
            lin_8 = np.clip(lin * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(lin_8, mode=img.mode)
        # For other modes, fall back to converting to RGB
        return cls._srgb_to_linear_image(img.convert("RGBA"))

    @classmethod
    def _linear_to_srgb_image(cls, img: Image.Image) -> Image.Image:
        """Convert a PIL image (L/RGB/RGBA) from linear to sRGB, returned as 8-bit per channel.

        Alpha channel (if present) is preserved untouched.
        """
        arr = np.array(img, dtype=np.float32)
        if img.mode in ("L", "RGB", "RGBA"):
            srgb = cls._linear_to_srgb_np(arr / 255.0)
            srgb_8 = np.clip(srgb * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(srgb_8, mode=img.mode)
        return cls._linear_to_srgb_image(img.convert("RGBA"))

    @classmethod
    def srgb_to_linear(cls, data):
        """Body of :meth:`ImgUtils.srgb_to_linear`."""
        if isinstance(data, Image.Image):
            return cls._srgb_to_linear_image(data)
        # Accept lists/tuples/arrays
        return cls._srgb_to_linear_np(data)

    @classmethod
    def linear_to_srgb(cls, data):
        """Body of :meth:`ImgUtils.linear_to_srgb`."""
        if isinstance(data, Image.Image):
            return cls._linear_to_srgb_image(data)
        return cls._linear_to_srgb_np(data)

    @classmethod
    def encode_hdr_for_web(cls, path, percentile=None):
        """Body of :meth:`ImgUtils.encode_hdr_for_web`."""
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                "OpenCV (cv2) is required to encode HDR images for the web."
            ) from e

        percentile = cls.HDR_WEB_PERCENTILE if percentile is None else float(percentile)
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError(f"Unreadable image: {path}")
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, None].repeat(3, axis=2)
        bgr = arr[:, :, :3]  # lightmaps are opaque; drop any alpha
        # A renderer's fireflies survive into the file, and both non-finite
        # kinds poison the encode SILENTLY: `inf > 0` is True, so enough of
        # them make the percentile itself inf and the whole map encodes BLACK
        # against an inf scalar; NaN passes the percentile's own filter but
        # propagates through np.clip and casts to uint8 as undefined garbage.
        if not np.isfinite(bgr).all():
            bgr = np.nan_to_num(bgr, nan=0.0, posinf=0.0, neginf=0.0)

        lit = bgr[bgr > 0.0]
        scalar = float(np.percentile(lit, percentile)) if lit.size else 1.0
        scalar = max(scalar, 1e-6)

        srgb = cls._linear_to_srgb_np(np.clip(bgr / scalar, 0.0, 1.0))
        out = np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
        # cv2 reads AND writes BGR: no swap needed. Max compression -- the
        # bytes go straight into a GLB, where they are the file size.
        ok, buf = cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise ValueError(f"PNG encode failed for {path}")
        return buf.tobytes(), scalar
