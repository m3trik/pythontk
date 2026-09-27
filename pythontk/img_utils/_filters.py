# !/usr/bin/python
# coding=utf-8
"""Spatial image filters: Gaussian blur, dilation, edge-aware denoise and
empty-texel fill (the bodies behind the :class:`ImgUtils` facade).
"""

from __future__ import annotations

import math
from typing import Union, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore
try:
    from PIL import Image, ImageFilter
except ImportError:
    Image = ImageFilter = None  # type: ignore


class _ImgFilterInternal:
    """Bodies of :class:`ImgUtils`' spatial filters (an ``ImgUtils`` base).

    The public signatures and docstrings stay on the facade; each delegates
    here. Helpers resolve through ``cls`` so a patch on ``ImgUtils`` (e.g. the
    ``_cv2`` probe) reaches them.
    """

    @classmethod
    def gaussian_blur(
        cls,
        image: Union[str, "Image.Image", "np.ndarray"],
        radius: float = 2.0,
        channel: Optional[str] = None,
    ) -> Union["Image.Image", "np.ndarray"]:
        """Body of :meth:`ImgUtils.gaussian_blur`."""
        if radius <= 0:
            if isinstance(image, np.ndarray):
                return image.copy()
            im = cls.ensure_image(image)
            return im.copy()

        # Numpy path
        if isinstance(image, np.ndarray):
            return cls._gaussian_blur_array(image, radius, channel)

        # PIL path
        im = cls.ensure_image(image)
        if channel and im.mode in ("RGBA", "LA"):
            bands = list(im.split())
            band_names = im.getbands()  # ('L','A') or ('R','G','B','A')
            ch = channel.upper()
            if ch not in band_names:
                raise ValueError(
                    f"Channel {channel!r} not present in image mode {im.mode!r}"
                )
            idx = band_names.index(ch)
            bands[idx] = bands[idx].filter(ImageFilter.GaussianBlur(radius=radius))
            return Image.merge(im.mode, bands)
        return im.filter(ImageFilter.GaussianBlur(radius=radius))

    @staticmethod
    def _gaussian_blur_array(
        arr: "np.ndarray", radius: float, channel: Optional[str]
    ) -> "np.ndarray":
        """Numpy-array blur. Uses PIL when present (avoids pulling in scipy/cv2); falls back to a
        pure-numpy separable Gaussian when PIL is unavailable, so dependency-light callers (e.g.
        ``rasterize_silhouette`` under Blender's PIL-less Python) keep working."""
        # Non-uint8 arrays (float [0,1]/HDR, uint16, …) must not be truncated to
        # uint8 for the PIL path; route them to the range-agnostic pure-numpy
        # blur which preserves their values.
        if Image is None or arr.dtype != np.uint8:
            return _ImgFilterInternal._gaussian_blur_array_numpy(arr, radius, channel)
        # 2D grayscale
        if arr.ndim == 2:
            src = Image.fromarray(
                arr if arr.dtype == np.uint8 else arr.astype(np.uint8)
            )
            blurred = src.filter(ImageFilter.GaussianBlur(radius=radius))
            out = np.asarray(blurred)
            return out.astype(arr.dtype, copy=False)

        # 3D: HxWxC
        if arr.ndim == 3:
            chans = arr.shape[2]
            mode = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}.get(chans)
            if mode is None:
                raise ValueError(f"Unsupported channel count: {chans}")
            src = Image.fromarray(
                arr if arr.dtype == np.uint8 else arr.astype(np.uint8), mode=mode
            )
            if channel and mode in ("RGBA", "LA"):
                bands = list(src.split())
                band_names = src.getbands()  # ('L','A') or ('R','G','B','A')
                ch = channel.upper()
                if ch not in band_names:
                    raise ValueError(
                        f"Channel {channel!r} not present in mode {mode!r}"
                    )
                idx = band_names.index(ch)
                bands[idx] = bands[idx].filter(ImageFilter.GaussianBlur(radius=radius))
                blurred = Image.merge(mode, bands)
            else:
                blurred = src.filter(ImageFilter.GaussianBlur(radius=radius))
            out = np.asarray(blurred)
            return out.astype(arr.dtype, copy=False)

        raise ValueError(f"Unsupported array shape: {arr.shape}")

    @staticmethod
    def _gaussian_blur_array_numpy(
        arr: "np.ndarray", radius: float, channel: Optional[str]
    ) -> "np.ndarray":
        """Pure-numpy separable Gaussian blur (PIL-free fallback for :meth:`_gaussian_blur_array`).

        Treats ``radius`` as the kernel std-dev (sigma), matching PIL's ``GaussianBlur(radius=…)``,
        and pads with reflection so edges don't darken. Returns the input dtype (uint8 inputs are
        rounded). For a 3D RGBA/LA array, ``channel`` (``"R"``/``"G"``/``"B"``/``"A"``) restricts the
        blur to one channel, mirroring the PIL path."""
        sigma = max(float(radius), 1e-6)
        rad = max(1, int(round(3.0 * sigma)))
        x = np.arange(-rad, rad + 1, dtype=np.float64)
        k = np.exp(-(x * x) / (2.0 * sigma * sigma))
        k /= k.sum()

        def blur2d(a2d: "np.ndarray") -> "np.ndarray":
            a2d = a2d.astype(np.float64, copy=False)
            pad = len(k) // 2
            ap = np.pad(a2d, ((0, 0), (pad, pad)), mode="reflect")
            a2d = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, ap)
            ap = np.pad(a2d, ((pad, pad), (0, 0)), mode="reflect")
            return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, ap)

        def cast(out: "np.ndarray") -> "np.ndarray":
            if arr.dtype == np.uint8:
                return (out + 0.5).clip(0, 255).astype(np.uint8)
            return out.astype(arr.dtype, copy=False)

        if arr.ndim == 2:
            return cast(blur2d(arr))
        if arr.ndim == 3:
            chans = arr.shape[2]
            # Derive the channel index from the array's real band layout so 'A'
            # maps to index 1 on a 2-channel LA array (not 3, which the fixed
            # RGBA map produced -- silently blurring every channel instead).
            band_names = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}.get(chans, "")
            idx = (
                band_names.find(channel.upper()) if channel else -1
            )  # -1 = absent/none
            targets = [idx] if idx >= 0 else range(chans)
            out = arr.astype(np.float64, copy=True)
            for c in targets:
                out[:, :, c] = blur2d(arr[:, :, c])
            return cast(out)
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    @staticmethod
    def dilate_image(
        image: "np.ndarray",
        mask: Optional["np.ndarray"] = None,
        iterations: int = -1,
        connectivity: int = 8,
        return_mask: bool = False,
    ) -> "np.ndarray":
        """Body of :meth:`ImgUtils.dilate_image`."""
        arr = np.asarray(image)
        out = arr.astype(np.float32, copy=True)
        squeeze = out.ndim == 2
        if squeeze:
            out = out[..., None]
        h, w, _ = out.shape

        if mask is None:
            valid = (out > 0).any(axis=2)
        else:
            valid = np.asarray(mask).astype(bool)
            if valid.shape != (h, w):
                raise ValueError(f"mask shape {valid.shape} != image {(h, w)}")
        out[~valid] = 0.0  # empties must not contribute color until filled

        if connectivity == 8:
            offsets = [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
        elif connectivity == 4:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            raise ValueError("connectivity must be 4 or 8")

        # Source/destination slice pair per neighbor offset, precomputed: the
        # accumulators are filled by IN-PLACE slice adds, never through a
        # freshly zeroed full-size shift buffer. A shift temporary per offset
        # is 16 whole-image allocations per pass (8 colour + 8 count) -- at a
        # 4096 atlas with an 18px gutter that is tens of GB of pure alloc and
        # zero-fill traffic for an answer the slices give directly.
        windows = []
        for dy, dx in offsets:
            ys, yd = (
                slice(max(dy, 0), h + min(dy, 0)),
                slice(max(-dy, 0), h + min(-dy, 0)),
            )
            xs, xd = (
                slice(max(dx, 0), w + min(dx, 0)),
                slice(max(-dx, 0), w + min(-dx, 0)),
            )
            windows.append((ys, xs, yd, xd))

        color_acc = np.empty_like(out)
        count_acc = np.empty((h, w), dtype=np.float32)
        vf = np.empty((h, w), dtype=np.float32)
        it = 0
        while not valid.all() and (iterations < 0 or it < iterations):
            color_acc[...] = 0.0
            count_acc[...] = 0.0
            np.copyto(vf, valid)
            # `out` is already zero at every invalid pixel (and stays so until
            # the pass that fills it also flips it valid), so out == out*vf --
            # accumulate `out` directly; only the count needs the validity mask.
            for ys, xs, yd, xd in windows:
                color_acc[yd, xd] += out[ys, xs]
                count_acc[yd, xd] += vf[ys, xs]
            fillable = (~valid) & (count_acc > 0)
            if not fillable.any():
                break  # remaining empties are unreachable from any valid pixel
            out[fillable] = color_acc[fillable] / count_acc[fillable][..., None]
            valid[fillable] = True
            it += 1

        if squeeze:
            out = out[..., 0]
        if np.issubdtype(arr.dtype, np.integer):
            # A neighbour average is fractional; truncating it on the way back
            # to an integer dtype biases every gutter texel dark by up to one
            # LSB, systematically (127.9 -> 127).
            np.rint(out, out=out)
        result = out.astype(arr.dtype, copy=False)
        return (result, valid) if return_mask else result

    @classmethod
    def denoise_image(
        cls,
        image: "np.ndarray",
        mask: Optional["np.ndarray"] = None,
        radius: int = 2,
        strength: float = 3.0,
        noise: Optional[float] = None,
        outliers: float = 5.0,
    ) -> "np.ndarray":
        """Body of :meth:`ImgUtils.denoise_image`."""
        arr = np.asarray(image)
        cv2 = cls._cv2()
        # float32 through cv2 (its filters' native type), float64 in numpy,
        # where the integral images' running sums need the headroom.
        dtype = np.float32 if cv2 is not None else np.float64
        work = arr.astype(dtype, copy=False)
        squeeze = work.ndim == 2
        if squeeze:
            work = work[..., None]
        h, w, channels = work.shape
        finite = np.isfinite(work).all(axis=2)
        if mask is None:
            # A bad sample is still where the renderer wrote content.
            with np.errstate(invalid="ignore"):
                valid = ~finite | (work > 0).any(axis=2)
        else:
            valid = np.asarray(mask).astype(bool)
            if valid.shape != (h, w):
                raise ValueError(f"mask shape {valid.shape} != image {(h, w)}")
        # A NaN or inf texel -- a renderer's rare bad sample -- is read as
        # absent: one would poison every statistic it touches (0 * NaN is NaN,
        # so even a zero-weighted texel leaks into its windows), and the whole
        # map came back NaN. Where it is the caller's content it is written
        # from its neighbourhood; outside the mask it comes back untouched.
        original = work
        heal = None
        if not finite.all():
            heal = valid & ~finite
            work = np.where(finite[..., None], work, dtype(0))
            valid = valid & finite
        if not valid.any() or radius < 1:
            return arr.copy()

        # PLANAR from here on: one contiguous 2D plane per channel. A numpy op
        # broadcasting an HxW plane against HxWx3 runs a strided inner loop
        # (measured: 24 ms at 1024^2 against ~1 ms for the same op plane by
        # plane), and the filter is a few dozen such ops.
        planes = (
            list(cv2.split(work))
            if cv2 is not None and channels > 1
            else [np.ascontiguousarray(work[..., c]) for c in range(channels)]
        )
        # The guide is the channel MEAN, not a luma: callers hand in RGB and
        # cv2's BGR alike, and the guide only has to carry the structure.
        luma = planes[0].copy()
        for plane in planes[1:]:
            luma += plane
        luma *= dtype(1.0 / channels)
        # A floor far under the content, so a black texel inside the mask is a
        # very dark value rather than log(0) = -inf poisoning every window.
        sample = cls._sample(luma[valid & (luma > 0)])
        floor = dtype(1e-6 * (float(np.median(sample)) if sample.size else 1.0))
        guide = cls._log(np.maximum(luma, floor))
        # Centred on the map's own level: the variance is a difference of box
        # means of G and G^2, and in float32 that cancellation is exact only
        # while G sits near zero. The fit is shift-invariant, so the centre is
        # added back at the end.
        centre = dtype(np.median(cls._sample(guide[valid])))
        guide -= centre
        # Each channel floored against its OWN texel's brightness, not the
        # map's: a channel at exactly zero beside lit neighbours (a saturated
        # colour, a coloured light's edge) otherwise sits ~14 log units under
        # them and drags that channel's window means down -- a colour fringe
        # (measured: x0.64 on the next texel's blue). A thousandth of the
        # texel's own level is invisible after any display transform.
        chroma_floor = np.maximum(luma * dtype(1e-3), floor)
        logs = [cls._log(np.maximum(plane, chroma_floor)) - centre for plane in planes]

        if noise is None:
            noise = cls._log_noise_sigma(guide, valid)
        weight = valid.astype(dtype)
        inv_count = 1.0 / np.maximum(cls._box_sum(weight, radius), dtype(1e-6))
        if outliers and noise > 0:
            # The median reads a mask-outside neighbour as the local mean of
            # the mask texels around it, so a gutter cannot pull it.
            local = cls._box_sum(weight * guide, radius) * inv_count
            median = cls._median3(np.where(valid, guide, local), valid)
            deviation = guide - median
            spike = valid & (np.abs(deviation) > dtype(float(outliers) * float(noise)))
            if spike.any():
                # Brightness to the neighbourhood's, the texel's own colour kept.
                shift = deviation[spike]
                for plane in logs:
                    plane[spike] -= shift
                guide[spike] = median[spike]
        eps = dtype(max(float(strength) * float(noise), 1e-6) ** 2)

        weighted_g = weight * guide
        mean_g = cls._box_sum(weighted_g, radius) * inv_count
        var_g = cls._box_sum(weighted_g * guide, radius) * inv_count
        var_g -= mean_g * mean_g
        np.maximum(var_g, 0.0, out=var_g)
        inv_var = 1.0 / (var_g + eps)
        out_planes = []
        for plane, log in zip(planes, logs):
            weighted = np.multiply(log, weight, out=log)
            mean_p = cls._box_sum(weighted, radius) * inv_count
            a = cls._box_sum(weighted * guide, radius) * inv_count
            a -= mean_g * mean_p
            a *= inv_var
            b = mean_p - a * mean_g
            fitted = cls._box_sum(a * weight, radius) * inv_count
            fitted *= guide
            fitted += cls._box_sum(b * weight, radius) * inv_count
            fitted += centre
            result = np.where(valid, cls._exp(fitted), plane)
            if heal is not None:
                # A bad sample's own value says nothing, so it takes its
                # window's (log) mean of the good texels around it.
                result = np.where(heal, cls._exp(mean_p + centre), result)
            out_planes.append(result)
        out = (
            cv2.merge(out_planes)
            if cv2 is not None and channels > 1
            else np.stack(out_planes, axis=2)
        )
        if heal is not None:
            untouched = ~finite & ~heal  # bad samples outside the mask
            out[untouched] = original[untouched]

        if squeeze:
            out = out[..., 0]
        return out.astype(arr.dtype, copy=False)

    @staticmethod
    def _cv2():
        """cv2 when it imports, else ``None``: the fast path, never a need."""
        try:
            import cv2

            return cv2
        except ImportError:
            return None

    @classmethod
    def _log(cls, values: "np.ndarray") -> "np.ndarray":
        """``log`` of a float32 plane through cv2 (~1.6x numpy's), else numpy."""
        cv2 = cls._cv2()
        if cv2 is not None and values.dtype == np.float32:
            return cv2.log(values)
        return np.log(values)

    @classmethod
    def _exp(cls, values: "np.ndarray") -> "np.ndarray":
        """``exp`` of a float32 plane through cv2 (~1.8x numpy's), else numpy."""
        cv2 = cls._cv2()
        if cv2 is not None and values.dtype == np.float32:
            return cv2.exp(values)
        return np.exp(values)

    @staticmethod
    def _sample(values: "np.ndarray", limit: int = 1 << 18) -> "np.ndarray":
        """*values*, strided down to about *limit* for a median: a quarter of a
        million texels pins a level as well as all of them, at a fraction of the
        partition."""
        step = max(1, values.size // limit)
        return values[::step] if step > 1 else values

    @classmethod
    def _box_sum(cls, values: "np.ndarray", radius: int) -> "np.ndarray":
        """Sum of *values* (HxW or HxWxC) over the ``(2 * radius + 1)^2`` window
        at each texel, clipped at the frame: cv2's box filter on float32 (zero
        border = clipped), an integral image otherwise."""
        k = 2 * radius + 1
        cv2 = cls._cv2()
        if cv2 is not None and values.dtype == np.float32:
            planes = values.shape[2] if values.ndim == 3 else 0
            if planes == 1:
                return cv2.boxFilter(
                    values[..., 0],
                    -1,
                    (k, k),
                    normalize=False,
                    borderType=cv2.BORDER_CONSTANT,
                )[..., None]
            if planes <= 4:
                return cv2.boxFilter(
                    values, -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT
                )
        pad = [(radius + 1, radius), (radius + 1, radius)] + [(0, 0)] * (
            values.ndim - 2
        )
        table = np.pad(values, pad).cumsum(axis=0).cumsum(axis=1)
        return table[k:, k:] - table[:-k, k:] - table[k:, :-k] + table[:-k, :-k]

    @classmethod
    def _median3(
        cls, values: "np.ndarray", mask: "np.ndarray", band: int = 256
    ) -> "np.ndarray":
        """Each texel's 3x3 median, *values* already filled outside *mask*.

        cv2's median filter on float32; without it, nine shifted planes per
        band of rows -- all nine of a 4K map at once would be over a gigabyte
        inside the DCC running the bake.
        """
        cv2 = cls._cv2()
        if cv2 is not None:
            return cv2.medianBlur(values.astype(np.float32, copy=False), 3).astype(
                values.dtype, copy=False
            )
        h, w = values.shape
        padded = np.pad(values, 1, mode="edge")
        out = np.empty_like(values)
        for top in range(0, h, band):
            bottom = min(top + band, h)
            planes = [
                padded[top + dy : bottom + dy, dx : dx + w]
                for dy in range(3)
                for dx in range(3)
            ]
            out[top:bottom] = np.median(np.stack(planes), axis=0)
        return out

    @classmethod
    def _log_noise_sigma(cls, log_image: "np.ndarray", mask: "np.ndarray") -> float:
        """Per-texel noise of *log_image* over *mask*, as a standard deviation.

        The median absolute Laplacian, scaled to a Gaussian's sigma: robust to
        the edges and gradients a map is made of, since those occupy few
        texels or add almost nothing to a Laplacian. White noise of sigma s
        gives a 4-neighbour Laplacian of sigma ``s * sqrt(1.25)``. Read on
        every other row: half a megapixel of 1024^2 samples pins a median as
        well as the whole map does.
        """
        centre = log_image[1:-1:2, 1:-1]
        up, down = log_image[0:-2:2, 1:-1], log_image[2::2, 1:-1]
        left, right = log_image[1:-1:2, :-2], log_image[1:-1:2, 2:]
        rows = min(len(centre), len(up), len(down))
        inner = (
            mask[1:-1:2, 1:-1][:rows]
            & mask[0:-2:2, 1:-1][:rows]
            & mask[2::2, 1:-1][:rows]
            & mask[1:-1:2, :-2][:rows]
            & mask[1:-1:2, 2:][:rows]
        )
        if not inner.any():
            return 0.0
        lap = centre[:rows] - 0.25 * (
            up[:rows] + down[:rows] + left[:rows] + right[:rows]
        )
        residual = cls._sample(lap[inner]).astype(np.float64)
        mad = float(np.median(np.abs(residual - np.median(residual))))
        return 1.4826 * mad / math.sqrt(1.25)

    @classmethod
    def fill_empty_texels(
        cls,
        image: "np.ndarray",
        mask: Optional["np.ndarray"] = None,
    ) -> "np.ndarray":
        """Body of :meth:`ImgUtils.fill_empty_texels`."""
        arr = np.asarray(image)
        if mask is None:
            valid = (arr > 0).any(axis=2) if arr.ndim == 3 else arr > 0
        else:
            valid = np.asarray(mask).astype(bool)
            if valid.shape != arr.shape[:2]:
                raise ValueError(f"mask shape {valid.shape} != image {arr.shape[:2]}")
        if valid.all():
            return arr.copy()
        if not valid.any():
            return arr.copy()  # nothing to spread from

        try:
            import cv2
        except ImportError:
            return cls._fill_pyramid(arr, valid)

        # Distance transform on the EMPTY set with pixel-index labels: each
        # empty texel's label is its nearest VALID texel, one pass, exact.
        empty_u8 = (~valid).astype(np.uint8)
        _, labels = cv2.distanceTransformWithLabels(
            empty_u8, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
        )
        # Labels index the zero-pixels of the input (the valid set) in row
        # scan order; map label -> flat pixel index, then gather.
        valid_flat = np.flatnonzero(valid.ravel())
        out = arr.copy()
        flat = out.reshape(-1, arr.shape[2]) if arr.ndim == 3 else out.reshape(-1)
        empty_flat = ~valid.ravel()
        # Gather only the empty texels' labels: indexing the whole label image
        # would materialize a full-size int array to use a fraction of it.
        src = valid_flat[labels.ravel()[empty_flat] - 1]
        flat[empty_flat] = flat[src]
        return out

    @classmethod
    def _fill_pyramid(
        cls, image: "np.ndarray", valid: "np.ndarray", ring: int = 4
    ) -> "np.ndarray":
        """cv2-less :meth:`fill_empty_texels`: a near ring by neighbour averaging,
        the far field from an image pyramid.

        The previous fallback, :meth:`dilate_image` ``iterations=-1``, costs one
        full-image pass per texel of distance to the nearest valid texel, so a
        map's cost is set by its FARTHEST background texel: a 1024 lightmap whose
        corners sit ~60 texels from any island paid ~60 passes -- measured 1.8 s
        per per-object map inside Blender (which ships no cv2), 48% of the whole
        bake loop, and it quadruples with each doubling of resolution.

        Only the texels within *ring* of valid content are ever sampled by a
        bilinear tap or a fine mip level, and those are filled exactly as before
        (neighbour averaging at full resolution). Everything beyond takes its
        value from a pyramid: each level is 2x2 valid-only averaged from the one
        below, filled with the same ring, and its leftovers take the next-coarser
        level's value on the way back up, so the far field costs O(log n) passes
        over shrinking images instead of O(distance) passes over the full one
        (measured 0.2 s on the same map). A far texel feeds nothing but coarse
        mips, and a pooled average is what the GPU would compute there anyway.

        Parameters:
            image: HxW or HxWxC array (any dtype; float32 internally).
            valid: HxW bool mask, at least one True.
            ring: Neighbour-averaging passes per level before the pyramid
                takes over -- the width, in texels, of the exact border.

        Returns:
            Image with every invalid texel filled; same shape/dtype as input.
        """
        arr = np.asarray(image)
        squeeze = arr.ndim == 2
        cur = arr.astype(np.float32, copy=True)
        if squeeze:
            cur = cur[..., None]
        cur_valid = np.asarray(valid).astype(bool)
        cur[~cur_valid] = 0.0

        levels = []  # (image, filled mask) per level, fine to coarse
        while not cur_valid.all() and min(cur_valid.shape) > 2:
            cur, filled = cls.dilate_image(
                cur, mask=cur_valid, iterations=ring, return_mask=True
            )
            if filled.all():
                cur_valid = filled
                break
            levels.append((cur, filled))
            # 2x2 valid-only average pooling; an odd edge is padded with an
            # empty row/column so it pools rather than drops.
            h, w, c = cur.shape
            ph, pw = h + (h & 1), w + (w & 1)
            pooled = np.zeros((ph, pw, c), dtype=np.float32)
            pooled[:h, :w] = cur  # invalid texels are zero, so a plain sum is masked
            count = np.zeros((ph, pw), dtype=np.float32)
            count[:h, :w] = filled
            pooled = pooled.reshape(ph // 2, 2, pw // 2, 2, c).sum(axis=(1, 3))
            count = count.reshape(ph // 2, 2, pw // 2, 2).sum(axis=(1, 3))
            cur_valid = count > 0
            cur = pooled / np.maximum(count, 1.0)[..., None]
        if not cur_valid.all():
            # The coarsest level is a handful of texels: flood it outright.
            cur = cls.dilate_image(cur, mask=cur_valid, iterations=-1)

        for fine, fine_valid in reversed(levels):
            up = np.repeat(np.repeat(cur, 2, axis=0), 2, axis=1)
            missing = ~fine_valid
            fine[missing] = up[: fine.shape[0], : fine.shape[1]][missing]
            cur = fine

        if squeeze:
            cur = cur[..., 0]
        if np.issubdtype(arr.dtype, np.integer):
            np.rint(cur, out=cur)  # see dilate_image: never truncate toward dark
        return cur.astype(arr.dtype, copy=False)
