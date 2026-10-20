# !/usr/bin/python
# coding=utf-8
"""Colour-space transfer functions: sRGB <-> linear, scene-linear primaries,
black-body colour and HDR-for-web encoding (the bodies behind the
:class:`ImgUtils` facade).
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

    @classmethod
    def kelvin_tint(cls, kelvin) -> Tuple[float, float, float]:
        """Body of :meth:`ImgUtils.kelvin_tint`."""
        from pythontk.core_utils.color import Color

        if not kelvin:
            return (1.0, 1.0, 1.0)
        rgb = cls.kelvin_to_linear_rgb(kelvin)
        y = sum(w * c for w, c in zip(Color.REC709_LUMINANCE, rgb))
        return tuple(c / y for c in rgb)

    @staticmethod
    def luminance(linear):
        """Body of :meth:`ImgUtils.luminance`."""
        from pythontk.core_utils.color import Color

        return np.asarray(linear, dtype=np.float64)[..., :3] @ np.asarray(
            Color.REC709_LUMINANCE
        )

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

    #: Scene-linear colour spaces -> their 3x3 matrix TO ACES2065-1 (AP0), the
    #: numbers Maya's default OCIO config (``Maya2022-default``) defines them by.
    #: Any two convert through it: ``inv(M_dst) @ M_src``.
    _SCENE_LINEAR_TO_AP0 = {
        "aces2065-1": (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        "acescg": (
            (0.695452241357, 0.140678696470, 0.163869062172),
            (0.044794563372, 0.859671118456, 0.095534318172),
            (-0.005525882558, 0.004025210306, 1.001500672252),
        ),
        "scene-linear rec.709-srgb": (
            (0.439632981919, 0.382988698152, 0.177378319929),
            (0.089776442959, 0.813439428749, 0.096784128292),
            (0.017541170383, 0.111546553302, 0.870912276314),
        ),
        "scene-linear dci-p3 d65": (
            (0.518933487598, 0.286256586387, 0.194809926015),
            (0.073859383047, 0.819845163937, 0.106295453016),
            (-0.000307011368, 0.043807050254, 0.956499961115),
        ),
        "scene-linear rec.2020": (
            (0.679085634707, 0.157700914643, 0.163213450650),
            (0.046002003080, 0.859054673003, 0.094943323917),
            (-0.000573943188, 0.028467768408, 0.972106174780),
        ),
    }
    #: The same spaces under the names the ACES 1.x / OCIO v2 studio configs
    #: give them, and every name and alias Blender 5.1's config does
    #: (``datafiles/colormanagement/config.ocio``), matched case-insensitively.
    _SCENE_LINEAR_ALIASES = {
        "aces - acescg": "acescg",
        "lin_ap1": "acescg",
        "lin_ap1_scene": "acescg",
        "linear acescg": "acescg",
        "acescg: linear - ap1": "acescg",
        "aces - aces2065-1": "aces2065-1",
        "lin_ap0": "aces2065-1",
        "lin_ap0_scene": "aces2065-1",
        "linear aces": "aces2065-1",
        "aces2065_1": "aces2065-1",
        "aces: linear - ap0": "aces2065-1",
        "utility - linear - srgb": "scene-linear rec.709-srgb",
        "utility - linear - rec.709": "scene-linear rec.709-srgb",
        "linear rec.709 (srgb)": "scene-linear rec.709-srgb",
        "linear rec.709": "scene-linear rec.709-srgb",
        "lin_rec709": "scene-linear rec.709-srgb",
        "lin_rec709_scene": "scene-linear rec.709-srgb",
        "lin_rec709_srgb": "scene-linear rec.709-srgb",
        "lin_srgb": "scene-linear rec.709-srgb",
        "linrec709": "scene-linear rec.709-srgb",
        "linear": "scene-linear rec.709-srgb",
        "linear bt.709": "scene-linear rec.709-srgb",
        "linear bt.709 i-d65": "scene-linear rec.709-srgb",
        "linear tristimulus": "scene-linear rec.709-srgb",
        "cgi: linear - rec.709": "scene-linear rec.709-srgb",
        "utility - linear - p3-d65": "scene-linear dci-p3 d65",
        "linear dci-p3 d65": "scene-linear dci-p3 d65",
        "linear dci-p3 i-d65": "scene-linear dci-p3 d65",
        "linear p3-d65": "scene-linear dci-p3 d65",
        "lin_p3d65": "scene-linear dci-p3 d65",
        "lin_p3d65_scene": "scene-linear dci-p3 d65",
        "apple dci-p3 d65": "scene-linear dci-p3 d65",
        "utility - linear - rec.2020": "scene-linear rec.2020",
        "linear rec.2020": "scene-linear rec.2020",
        "linear bt.2020": "scene-linear rec.2020",
        "linear bt.2020 i-d65": "scene-linear rec.2020",
        "lin_rec2020": "scene-linear rec.2020",
        "lin_rec2020_scene": "scene-linear rec.2020",
    }

    @classmethod
    def _scene_linear_to_ap0(cls, space: str) -> "np.ndarray":
        key = str(space).strip().lower()
        key = cls._SCENE_LINEAR_ALIASES.get(key, key)
        if key not in cls._SCENE_LINEAR_TO_AP0:
            raise KeyError(f"not a known scene-linear colour space: {space!r}")
        return np.array(cls._SCENE_LINEAR_TO_AP0[key], dtype=np.float64)

    @classmethod
    def convert_scene_linear(
        cls, image, src, dst="scene-linear Rec.709-sRGB", bgr=False
    ):
        """Body of :meth:`ImgUtils.convert_scene_linear`."""
        image = np.asarray(image)
        a, b = cls._scene_linear_to_ap0(src), cls._scene_linear_to_ap0(dst)
        if np.array_equal(a, b):
            return image
        matrix = np.linalg.solve(b, a)  # inv(dst) @ src
        if bgr:
            matrix = matrix[::-1, ::-1]
        out = np.array(image, dtype=np.result_type(image.dtype, np.float32), copy=True)
        out[..., :3] = image[..., :3].astype(np.float64) @ matrix.T
        return out

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

    # three.js ACESFilmicToneMapping (Stephen Hill's fit of the ACES RRT + ODT):
    # sRGB -> the fit's input space, and back.
    _ACES_IN = (
        (0.59719, 0.35458, 0.04823),
        (0.07600, 0.90834, 0.01566),
        (0.02840, 0.13383, 0.83777),
    )
    _ACES_OUT = (
        (1.60475, -0.53108, -0.07367),
        (-0.10208, 1.10813, -0.00605),
        (-0.00327, -0.07276, 1.07602),
    )
    # Bjorn Ottosson's OKLab: linear sRGB -> LMS, and the cube-rooted LMS -> Lab.
    _OKLAB_LMS = (
        (0.4122214708, 0.5363325363, 0.0514459929),
        (0.2119034982, 0.6806995451, 0.1073969566),
        (0.0883024619, 0.2817188376, 0.6299787005),
    )
    _OKLAB_LAB = (
        (0.2104542553, 0.7936177850, -0.0040720468),
        (1.9779984951, -2.4285922050, 0.4505937099),
        (0.0259040371, 0.7827717662, -0.8086757660),
    )

    @classmethod
    def _tonemap_aces_filmic(cls, c: "np.ndarray") -> "np.ndarray":
        """three.js ``ACESFilmicToneMapping`` (its ``/ 0.6`` exposure bias included)."""
        c = (c / 0.6) @ np.asarray(cls._ACES_IN).T
        a = c * (c + 0.0245786) - 0.000090537
        b = c * (0.983729 * c + 0.4329510) + 0.238081
        return (a / b) @ np.asarray(cls._ACES_OUT).T

    @classmethod
    def _tonemap_operators(cls):
        return {"aces_filmic": cls._tonemap_aces_filmic, "linear": lambda c: c}

    @classmethod
    def tonemap(cls, linear, operator: str = "aces_filmic", exposure: float = 1.0):
        """Body of :meth:`ImgUtils.tonemap`."""
        operators = cls._tonemap_operators()
        if operator not in operators:
            raise ValueError(
                f"Unknown tone-map operator {operator!r}; one of {sorted(operators)}."
            )
        c = np.clip(np.asarray(linear, dtype=np.float64), 0.0, None) * float(exposure)
        return np.clip(operators[operator](c), 0.0, 1.0).astype(np.float32)

    @classmethod
    def linear_to_oklab(cls, linear):
        """Body of :meth:`ImgUtils.linear_to_oklab`."""
        lms = np.asarray(linear, dtype=np.float64) @ np.asarray(cls._OKLAB_LMS).T
        return (np.cbrt(lms) @ np.asarray(cls._OKLAB_LAB).T).astype(np.float32)

    #: Seed of the 8-bit encode's dither: fixed, so a map always encodes to the
    #: same bytes (digest de-duplication downstream depends on it).
    _DITHER_SEED = 0x5EED

    @classmethod
    def quantize_8bit(cls, unit):
        """Body of :meth:`ImgUtils.quantize_8bit`."""
        unit = np.asarray(unit, dtype=np.float32)
        rng = np.random.default_rng(cls._DITHER_SEED)
        r = rng.random(unit.shape[:-1], dtype=np.float32)[..., None]
        return np.clip(np.floor(unit * 255.0 + r), 0, 255).astype(np.uint8)

    @classmethod
    def encode_hdr_pixels_for_web(cls, rgb, percentile=None):
        """Body of :meth:`ImgUtils.encode_hdr_pixels_for_web`."""
        rgb = np.asarray(rgb, dtype=np.float32)[..., :3]
        # A renderer's fireflies survive into the file, and both non-finite
        # kinds poison the encode SILENTLY: `inf > 0` is True, so enough of
        # them make the divisor itself inf and the whole map encodes BLACK
        # against an inf scalar; NaN passes the divisor's own filter but
        # propagates through the divide and casts to uint8 as undefined garbage.
        if not np.isfinite(rgb).all():
            rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
        lit = rgb[rgb > 0.0]
        if not lit.size:
            scalar = 1.0
        elif percentile is None:
            floor = float(np.percentile(lit, cls.HDR_WEB_PERCENTILE))
            scalar = min(float(lit.max()), cls.HDR_WEB_HEADROOM * floor)
        else:
            scalar = float(np.percentile(lit, float(percentile)))
        scalar = max(scalar, 1e-6)
        unit = np.maximum(rgb / scalar, 0.0)
        # Past full scale the triple is scaled down whole, never clipped per
        # channel: a warm light stays warm instead of bleaching to white.
        unit /= np.maximum(unit.max(axis=-1, keepdims=True), 1.0)
        return cls.quantize_8bit(cls._linear_to_srgb_np(unit)), scalar

    @classmethod
    def encode_hdr_for_web(cls, path, percentile=None):
        """Body of :meth:`ImgUtils.encode_hdr_for_web`."""
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                "OpenCV (cv2) is required to encode HDR images for the web."
            ) from e

        img = cls.cv2_read(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError(f"Unreadable image: {path}")
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, None].repeat(3, axis=2)
        # BGR in, BGR out: the encode treats the channels alike, and cv2 reads
        # AND writes BGR. Lightmaps are opaque; any alpha is dropped.
        out, scalar = cls.encode_hdr_pixels_for_web(arr[:, :, :3], percentile)
        # Max compression -- the bytes go straight into a GLB, where they are
        # the file size.
        ok, buf = cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise ValueError(f"PNG encode failed for {path}")
        return buf.tobytes(), scalar

    @classmethod
    def encode_hdr_radiance(cls, path):
        """Body of :meth:`ImgUtils.encode_hdr_radiance`."""
        try:
            import cv2
        except ImportError as e:
            raise ImportError("OpenCV (cv2) is required to encode HDR images.") from e

        img = cls.cv2_read(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError(f"Unreadable image: {path}")
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, None].repeat(3, axis=2)
        # cv2 reads BGR; the pixels' form takes RGB.
        return cls.encode_hdr_radiance_pixels(arr[:, :, 2::-1])

    @classmethod
    def encode_hdr_radiance_pixels(cls, rgb):
        """Body of :meth:`ImgUtils.encode_hdr_radiance_pixels`."""
        try:
            import cv2
        except ImportError as e:
            raise ImportError("OpenCV (cv2) is required to encode HDR images.") from e

        arr = np.asarray(rgb, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, None].repeat(3, axis=2)
        bgr = np.ascontiguousarray(arr[:, :, 2::-1])
        # As encode_hdr_for_web: a firefly's inf or NaN would encode as an
        # undefined exponent, and RGBE has no sign, so negatives clamp to 0.
        bgr = np.nan_to_num(bgr, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(bgr, 0.0, out=bgr)
        # The Radiance codec stages through the temp folder: see cv2_temp.
        with cls.cv2_temp():
            ok, buf = cv2.imencode(".hdr", bgr)
        if not ok:
            raise ValueError("Radiance HDR encode failed")
        return buf.tobytes()
