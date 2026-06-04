#!/usr/bin/python
# coding=utf-8
"""Cross-set exposure / white-balance equalization.

Useful when frames come from multiple captures of the same subject at
different times of day, with different lighting / cameras / WB. Runs
*before* the SfM tool so its color-calibration step has less variance
to chew through (and texture seams in the final diffuse stay tight).

Uses cv2 (already an optional dep for ``FrameExtractor``); no
``colour-science`` requirement. The algorithm is Reinhard mean+std
matching per channel in LAB space — cheap, robust, and orientation-
independent.
"""
import logging
import os
from typing import List, Optional, Sequence

try:
    import cv2
    import numpy as np

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    np = None
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


class ExposureEqualizer:
    """Equalize exposure / WB across a list of source directories.

    The first directory's per-image statistics define the target
    distribution unless ``reference_dir`` is set explicitly. Other
    directories are remapped to match.
    """

    def is_available(self) -> bool:
        return CV2_AVAILABLE

    def equalize_directories(
        self,
        source_dirs: Sequence[str],
        output_root: str,
        reference_dir: Optional[str] = None,
        suffix: str = "_eq",
        sample_count: int = 20,
        strength: float = 1.0,
        reference_strategy: str = "first",
        quality: int = 100,
        preserve_exif: bool = True,
    ) -> List[str]:
        """Equalize every image in ``source_dirs`` against the reference set.

        Returns the list of output directory paths (one per input
        directory). When the reference is one of the source dirs, its
        images are still re-saved under the new root for a uniform set.

        ``strength`` (0–1) blends the matched result with the original in
        LAB space: ``1.0`` is a full Reinhard match (can flatten local
        contrast); lower values nudge exposure/WB toward the reference
        while preserving each frame's own contrast — useful before tools
        that re-balance internally (e.g. RealityCapture's MosaicBlending).

        ``reference_strategy`` chooses the target distribution when
        ``reference_dir`` is not given explicitly:

        * ``"first"``  — first source dir (legacy default).
        * ``"median"`` — the source dir whose mean luma is closest to the
          set-wide mean, so no single capture's color cast dominates.
        * ``"global"`` — the average of every source dir's stats.

        Output is written to **minimize re-encode damage**: ``quality`` (default
        100) + 4:4:4 chroma for JPEG, and ``preserve_exif`` carries the source
        EXIF through (focal-length / camera priors a downstream SfM solver uses).
        This matters because the equalized set may be fed straight to alignment —
        a lossy re-save (the old silent default ~q95, EXIF dropped) injects
        compression artifacts that degrade feature matching. Even so, for
        *alignment* prefer leaving radiometry to the SfM tool; equalization's real
        payoff is seam-free texturing across **multiple** captures.
        """
        if not self.is_available():
            logger.error("cv2 not available; cannot equalize exposures.")
            return []
        if not source_dirs:
            return []

        if reference_dir is not None:
            if not os.path.isdir(reference_dir):
                logger.error(f"Reference directory missing: {reference_dir}")
                return []
            ref_mean, ref_std = self._sample_stats(reference_dir, sample_count)
        else:
            ref_mean, ref_std = self._reference_stats(
                source_dirs, reference_strategy, sample_count
            )
        if ref_mean is None:
            logger.error("Reference stats unavailable; aborting.")
            return []
        logger.info(
            f"Reference ({reference_strategy if reference_dir is None else reference_dir}) "
            f"LAB mean={ref_mean} std={ref_std} strength={strength}"
        )

        out_dirs: List[str] = []
        for src in source_dirs:
            stem = os.path.basename(os.path.normpath(src)) + suffix
            out_dir = os.path.join(output_root, stem)
            os.makedirs(out_dir, exist_ok=True)
            count = self._equalize_dir(
                src, out_dir, ref_mean, ref_std, strength, quality, preserve_exif
            )
            out_dirs.append(out_dir)
            logger.info(f"Equalized {count} images from {src} → {out_dir}")
        return out_dirs

    # ------------------------------------------------------------------ helpers

    def _reference_stats(
        self,
        source_dirs: Sequence[str],
        strategy: str,
        sample_count: int,
    ):
        """Resolve the (mean, std) LAB target for the chosen strategy."""
        valid = {"first", "median", "global"}
        if strategy not in valid:
            raise ValueError(
                f"reference_strategy must be one of {sorted(valid)}; got {strategy!r}"
            )
        per_dir = []  # (dir, mean, std)
        for d in source_dirs:
            if not os.path.isdir(d):
                continue
            m, s = self._sample_stats(d, sample_count)
            if m is not None:
                per_dir.append((d, m, s))
        if not per_dir:
            return None, None
        if strategy == "global":
            return (
                np.mean([m for _, m, _ in per_dir], axis=0),
                np.mean([s for _, _, s in per_dir], axis=0),
            )
        if strategy == "median":
            global_l = float(np.mean([m[0] for _, m, _ in per_dir]))
            d, m, s = min(per_dir, key=lambda t: abs(t[1][0] - global_l))
            logger.info(f"median reference dir: {d}")
            return m, s
        # "first" (legacy)
        return per_dir[0][1], per_dir[0][2]

    def _list_images(self, directory: str) -> List[str]:
        return sorted(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(IMAGE_EXTS)
        )

    def _sample_stats(self, directory: str, sample_count: int):
        files = self._list_images(directory)
        if not files:
            return None, None
        sampled = files[:: max(1, len(files) // sample_count)][:sample_count]
        means, stds = [], []
        for path in sampled:
            img = cv2.imread(path)
            if img is None:
                continue
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype("float32")
            means.append(lab.reshape(-1, 3).mean(axis=0))
            stds.append(lab.reshape(-1, 3).std(axis=0))
        if not means:
            return None, None
        return np.mean(means, axis=0), np.mean(stds, axis=0)

    def _equalize_dir(
        self, src_dir: str, out_dir: str, ref_mean, ref_std, strength: float = 1.0,
        quality: int = 100, preserve_exif: bool = True,
    ) -> int:
        strength = float(np.clip(strength, 0.0, 1.0))
        count = 0
        for path in self._list_images(src_dir):
            # Read the *stored* pixels (don't let cv2 bake in EXIF orientation):
            # _save_image copies the source EXIF verbatim, so the saved pixels must
            # stay in the same frame as the copied Orientation tag. Otherwise cv2
            # would right the image while the tag still says "rotate", and an
            # EXIF-aware consumer (RealityScan / Metashape) rotates a second time.
            img = cv2.imread(path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
            if img is None:
                continue
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype("float32")
            flat = lab.reshape(-1, 3)
            mu = flat.mean(axis=0)
            sigma = flat.std(axis=0)
            sigma = np.where(sigma < 1e-3, 1.0, sigma)
            normalized = (flat - mu) / sigma
            mapped = normalized * ref_std + ref_mean
            if strength < 1.0:
                # Blend toward the match, preserving each frame's own
                # local contrast in proportion to (1 - strength).
                mapped = flat * (1.0 - strength) + mapped * strength
            mapped = np.clip(mapped, 0, 255).reshape(lab.shape).astype("uint8")
            bgr = cv2.cvtColor(mapped, cv2.COLOR_LAB2BGR)
            out_path = os.path.join(out_dir, os.path.basename(path))
            if self._save_image(out_path, bgr, path, quality, preserve_exif):
                count += 1
        return count

    @staticmethod
    def _save_image(out_path, bgr, src_path, quality: int, preserve_exif: bool) -> bool:
        """Write *bgr* to *out_path* with minimal re-encode loss.

        JPEG: a single high-``quality`` encode at 4:4:4 chroma, carrying the
        source EXIF when ``preserve_exif`` (via PIL, if available). PNG/TIFF/BMP
        are lossless, so a plain write suffices. Falls back to ``cv2.imwrite`` if
        PIL is missing.
        """
        ext = os.path.splitext(out_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            try:
                from PIL import Image

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                kw = {"quality": int(quality), "subsampling": 0}
                if preserve_exif:
                    try:
                        with Image.open(src_path) as s:
                            exif = s.info.get("exif")
                        if exif:
                            kw["exif"] = exif
                    except Exception:
                        pass
                Image.fromarray(rgb).save(out_path, **kw)
                return True
            except Exception:
                params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
                sf = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
                sf444 = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444", None)
                if sf is not None and sf444 is not None:
                    params += [int(sf), int(sf444)]
                return bool(cv2.imwrite(out_path, bgr, params))
        return bool(cv2.imwrite(out_path, bgr))
