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
import shutil
from typing import List, Optional, Sequence

try:
    import cv2
    import numpy as np

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    np = None
    CV2_AVAILABLE = False

# From this package:
from pythontk.img_utils._img_utils import ImgUtils

logger = logging.getLogger(__name__)


class ExposureEqualizer:
    """Equalize exposure / WB across a list of source directories.

    The first directory's per-image statistics define the target
    distribution unless ``reference_dir`` is set explicitly. Other
    directories are remapped to match.
    """

    #: Frames written without EXIF in the last :meth:`equalize_directories`
    #: run (0 = every frame kept its EXIF). Counts both cv2-fallback writes
    #: and PIL writes whose EXIF read failed — each frame at most once.
    last_fallback_count: int = 0

    #: Per-run first-occurrence log gates for the two distinct EXIF-loss
    #: causes, so one cause's message never suppresses the other's.
    _exif_loss_warned: bool = False
    _cv2_fallback_warned: bool = False

    #: Per-run memo for :meth:`_sample_stats` — ``_reference_stats`` and the
    #: per-capture loop sample the same directories; without it every source
    #: dir is decoded twice.
    _stats_cache: Optional[dict] = None

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
        per_image: bool = False,
        overwrite_output: bool = True,
    ) -> List[str]:
        """Equalize every image in ``source_dirs`` against the reference set.

        Returns the list of output directory paths (one per input
        directory). When the reference is one of the source dirs, its
        images are still re-saved under the new root for a uniform set.

        By default one transform is computed **per capture directory** (from
        that directory's sampled stats) and applied identically to every
        frame in it. That matches the stated goal — align *captures* to each
        other — while preserving legitimate intra-capture variation (a dark
        underside frame stays darker than a lit top-down one) and keeping the
        gain bounded: per-image matching (``per_image=True``, the legacy
        mode) normalizes each frame by its *own* std, which applies an
        unbounded ``ref_std/σ`` gain to low-contrast/dark frames and
        amplifies exactly the noise SfM feature matching is most sensitive
        to.

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

        When ``overwrite_output`` is True (default) any pre-existing
        per-source output directory is **emptied** before being repopulated.
        Output filenames track the (curated) input set, so without the purge
        a re-run after tighter curation leaves previously-equalized,
        since-culled frames mixed into the set a downstream ``add_images``
        ingests — quality then degrades run over run with no code change.
        Check :attr:`last_fallback_count` after a run: nonzero means that
        many frames were written without EXIF (cv2 fallback or EXIF read
        failure — see :meth:`_save_image`).
        """
        if not self.is_available():
            logger.error("cv2 not available; cannot equalize exposures.")
            return []
        if not source_dirs:
            return []
        self.last_fallback_count = 0
        self._exif_loss_warned = False
        self._cv2_fallback_warned = False
        self._stats_cache = {}

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
            f"LAB mean={ref_mean} std={ref_std} strength={strength} "
            f"mode={'per-image' if per_image else 'per-capture'}"
        )

        out_dirs: List[str] = []
        stems = ImgUtils.unique_dir_stems(source_dirs)
        for src, stem in zip(source_dirs, stems):
            out_dir = os.path.join(output_root, stem + suffix)
            if overwrite_output and os.path.isdir(out_dir):
                # Purge stale survivors from any previous run over a
                # different (e.g. more tightly curated) input set — unless
                # the output resolves onto the source dir itself (e.g.
                # output_root=parent + suffix=""), where the purge would
                # delete the capture before it is ever read.
                if os.path.normcase(os.path.normpath(out_dir)) == os.path.normcase(
                    os.path.normpath(src)
                ):
                    logger.warning(
                        f"Output dir resolves onto the source dir ({out_dir}); "
                        f"skipping purge and overwriting in place."
                    )
                else:
                    shutil.rmtree(out_dir, ignore_errors=True)
            os.makedirs(out_dir, exist_ok=True)
            src_stats = None
            if not per_image:
                src_stats = self._sample_stats(src, sample_count)
                if src_stats[0] is None:
                    src_stats = None  # unreadable dir → per-image fallback
            count = self._equalize_dir(
                src, out_dir, ref_mean, ref_std, strength, quality,
                preserve_exif, src_stats=src_stats,
            )
            out_dirs.append(out_dir)
            logger.info(f"Equalized {count} images from {src} → {out_dir}")
        if self.last_fallback_count:
            logger.error(
                f"{self.last_fallback_count} frame(s) were written without "
                f"EXIF (incl. Orientation + focal length) — downstream SfM "
                f"loses camera priors and portrait frames may load sideways. "
                f"See the per-cause error(s) above (cv2 fallback and/or EXIF "
                f"read failure)."
            )
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
        return ImgUtils.list_image_files(directory, full_paths=True)

    def _sample_stats(self, directory: str, sample_count: int):
        cache = self._stats_cache
        key = (os.path.normcase(os.path.normpath(directory)), sample_count)
        if cache is not None and key in cache:
            return cache[key]
        result = self._compute_sample_stats(directory, sample_count)
        if cache is not None:
            cache[key] = result
        return result

    def _compute_sample_stats(self, directory: str, sample_count: int):
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
        quality: int = 100, preserve_exif: bool = True, src_stats=None,
    ) -> int:
        """Equalize one directory. With ``src_stats`` (the capture's sampled
        ``(mean, std)``) a single affine map is applied to every frame
        (per-capture mode); without it each frame is normalized by its own
        stats (legacy per-image mode)."""
        strength = float(np.clip(strength, 0.0, 1.0))
        if src_stats is not None:
            cap_mu, cap_sigma = src_stats
            cap_sigma = np.where(cap_sigma < 1e-3, 1.0, cap_sigma)
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
            if src_stats is not None:
                mu, sigma = cap_mu, cap_sigma
            else:
                mu = flat.mean(axis=0)
                sigma = flat.std(axis=0)
                sigma = np.where(sigma < 1e-3, 1.0, sigma)
            normalized = (flat - mu) / sigma
            mapped = normalized * ref_std + ref_mean
            if strength < 1.0:
                # Blend toward the match, preserving each frame's own
                # local contrast in proportion to (1 - strength).
                mapped = flat * (1.0 - strength) + mapped * strength
            # rint before the cast: a bare astype() truncates, biasing every
            # channel ~0.5 LSB down per pass and worsening banding.
            mapped = np.rint(np.clip(mapped, 0, 255)).reshape(lab.shape)
            mapped = mapped.astype("uint8")
            bgr = cv2.cvtColor(mapped, cv2.COLOR_LAB2BGR)
            out_path = os.path.join(out_dir, os.path.basename(path))
            if self._save_image(out_path, bgr, path, quality, preserve_exif):
                count += 1
        return count

    def _save_image(self, out_path, bgr, src_path, quality: int, preserve_exif: bool) -> bool:
        """Write *bgr* to *out_path* with minimal re-encode loss.

        JPEG: a single high-``quality`` encode at 4:4:4 chroma, carrying the
        source EXIF when ``preserve_exif`` (via PIL, if available). PNG/BMP
        are lossless, so a plain write suffices. TIFF is written via
        ``cv2.imwrite`` — EXIF is not carried and (since pixels are read
        8-bit) 16-bit sources are truncated; a one-time warning flags it.

        The cv2 fallback for JPEG (PIL missing/broken) keeps quality/4:4:4
        but **cannot write EXIF** — and the pixels were deliberately read
        orientation-ignored, so the output loses both camera priors and the
        rotation tag. Every fallback is counted in
        :attr:`last_fallback_count` and logged, never silent: a run that
        loses EXIF looks identical to a healthy one in the output listing
        but aligns measurably worse.
        """
        ext = os.path.splitext(out_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            frame_counted = False
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
                    except Exception as e:  # noqa: BLE001
                        # The frame is still written, but without EXIF — that
                        # is the same loss the cv2 fallback causes, so count
                        # it the same way instead of passing silently.
                        self.last_fallback_count += 1
                        frame_counted = True
                        if not self._exif_loss_warned:
                            self._exif_loss_warned = True
                            logger.error(
                                f"EXIF read failed ({e}); frame(s) will be "
                                f"written without EXIF (first: {src_path})."
                            )
                Image.fromarray(rgb).save(out_path, **kw)
                return True
            except Exception as e:
                if not frame_counted:
                    self.last_fallback_count += 1
                if not self._cv2_fallback_warned:
                    self._cv2_fallback_warned = True
                    logger.error(
                        f"PIL save failed ({e}); falling back to cv2.imwrite "
                        f"— EXIF/orientation will be dropped for affected "
                        f"frames (first: {src_path})."
                    )
                params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
                sf = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
                sf444 = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444", None)
                if sf is not None and sf444 is not None:
                    params += [int(sf), int(sf444)]
                return bool(cv2.imwrite(out_path, bgr, params))
        if ext in (".tif", ".tiff") and preserve_exif and not getattr(
            self, "_tiff_warned", False
        ):
            self._tiff_warned = True
            logger.warning(
                "TIFF output is written via cv2: EXIF is not carried over and "
                "16-bit sources are truncated to 8-bit (pixels are read 8-bit "
                "for LAB matching). Prefer JPEG/PNG sources for equalization."
            )
        return bool(cv2.imwrite(out_path, bgr))
