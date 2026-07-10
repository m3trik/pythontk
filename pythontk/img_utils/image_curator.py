#!/usr/bin/python
# coding=utf-8
"""Perceptual-hash + sharpness curation for large image sets.

Built for the "I extracted 6000 video frames; help me get to a quality
set" case:

1. Compute a 64-bit difference hash (dHash) per image.
2. Compute Laplacian-variance sharpness per image.
3. Cluster images by Hamming distance over the dHash.
4. Keep the sharpest ``keep_per_cluster`` images of each cluster, and
   drop anything below ``sharpness_floor``.
5. Copy kept files into a single curated output directory per source.

Every stage is opt-in: the default ``hash_threshold=0`` disables
clustering entirely (every frame survives dedup — even bit-identical
hashes are not merged) and the sharpness floors default off, so a
no-arg ``curate()`` is a safe pass-through copy.
Near-duplicate frames of a moving camera carry exactly the
small-baseline parallax SfM triangulates from, so destructive settings
belong to the caller: ``5`` catches near-identical frames only,
``10–15`` aggressively culls redundant static photo sets.
"""
import logging
import os
import shutil
from typing import List, Optional, Sequence, Tuple

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


class ImageCurator:
    """Pre-SfM content-dedup + sharpness culling."""

    def is_available(self) -> bool:
        return CV2_AVAILABLE

    @staticmethod
    def dhash(image, size: int = 8) -> int:
        """Difference hash. Returns a ``size*size``-bit integer (default
        64-bit at size=8). Robust to slight crops / exposure shifts.

        Note: ``hash_threshold`` in :meth:`curate` is the max Hamming
        distance — scale it with ``size`` (e.g. 5 at size=8 / 64 bits ≈ 20
        at size=16 / 256 bits). :meth:`curate` now defaults it to 0, which
        disables hash clustering entirely.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        bits = 0
        for v in diff.flatten():
            bits = (bits << 1) | int(v)
        return bits

    @staticmethod
    def hamming(a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    @staticmethod
    def sharpness(image) -> float:
        """Variance-of-Laplacian sharpness. Absolute values scale with
        image resolution — the curator computes this on a
        :attr:`SCAN_WIDTH`-wide thumbnail, so callers tuning
        ``sharpness_floor`` should empirically inspect the score
        distribution rather than reuse full-resolution thresholds from
        elsewhere.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ------------------------------------------------------------------ curation

    def curate(
        self,
        source_dirs: Sequence[str],
        output_root: str,
        hash_threshold: int = 0,
        sharpness_floor: float = 0.0,
        sharpness_floor_percentile: Optional[float] = None,
        min_sharpness_fraction_of_median: float = 0.0,
        keep_per_cluster: int = 1,
        suffix: str = "_curated",
        progress: Optional[callable] = None,
        overwrite_output: bool = True,
    ) -> List[str]:
        """Curate every image across ``source_dirs`` → write the kept set
        to ``output_root/<stem><suffix>/``, where ``<stem>`` is the source
        dir's basename, parent-qualified if needed to stay unique across
        ``source_dirs`` (see :meth:`ImgUtils.unique_dir_stems`).

        Returns one output directory per input directory (preserving the
        per-source partition so downstream stages can still treat them
        as independent captures).

        ``sharpness_floor`` is an absolute variance-of-Laplacian cutoff;
        because that metric scales with image content it's hard to pick a
        portable constant. ``sharpness_floor_percentile`` (0–100) instead
        derives the floor from the sharpness distribution of the *cluster
        representatives* (the sharpest-per-cluster frames actually headed
        downstream) — e.g. ``10`` drops the blurriest ~tenth of that
        survivor set regardless of the scene's absolute sharpness scale.
        Computing it over representatives (rather than every scanned frame)
        keeps a mass of blurry near-duplicates from dragging the cutoff
        down and under-culling. ``min_sharpness_fraction_of_median`` (0–1) adds a
        *median-relative* guard — drop any representative below that fraction of
        the survivor-median sharpness — which catches catastrophically defocused
        frames the percentile misses (the percentile only removes a fixed bottom
        slice, so when blur exceeds that slice the boundary cases slip through; a
        frame far below the median is physically blurred no matter how many there
        are). The strictest of the three floors wins. Because the floor is applied
        *after* keeping the sharpest of each dHash cluster, only clusters that are
        uniformly blurry are dropped — parallax-bearing sharp views are preserved.

        When ``overwrite_output`` is True (default) any pre-existing
        per-source output directory is **emptied** before being
        repopulated. This guards against re-runs at a different
        threshold leaving stale files mixed with new ones.
        """
        if not self.is_available():
            logger.error("cv2 not available; cannot curate images.")
            return list(source_dirs)
        if not source_dirs:
            return []
        os.makedirs(output_root, exist_ok=True)

        # 1. Scan; 2. cluster; 3. select — shared with :meth:`preview` (DRY).
        scanned = self._scan_images(source_dirs, progress=progress)
        if not scanned:
            logger.warning("No readable images found.")
            return list(source_dirs)
        logger.info(f"Scanned {len(scanned)} images across {len(source_dirs)} dir(s)")

        clusters = self._cluster(scanned, hash_threshold)
        representatives, kept, sharpness_floor = self._select(
            clusters,
            keep_per_cluster,
            sharpness_floor,
            sharpness_floor_percentile,
            min_sharpness_fraction_of_median,
        )

        logger.info(
            f"Curated {len(scanned)} → {len(kept)} images "
            f"({len(clusters)} clusters, {keep_per_cluster}/cluster, "
            f"sharpness floor {sharpness_floor})"
        )

        # 4. Write per-source output dirs. Stems are collision-proofed:
        # same-basename sources (capA/images + capB/images) would otherwise
        # map to one out_dir and the second source's purge would delete the
        # first's just-copied files.
        out_dirs: List[str] = []
        kept_by_src = {d: [] for d in source_dirs}
        for record in kept:
            kept_by_src[record[0]].append(record)
        stems = ImgUtils.unique_dir_stems(source_dirs)
        for src_dir, stem in zip(source_dirs, stems):
            out_dir = os.path.join(output_root, stem + suffix)
            if overwrite_output and os.path.isdir(out_dir):
                # Purge stale survivors from any previous run at a
                # different threshold.
                shutil.rmtree(out_dir, ignore_errors=True)
            os.makedirs(out_dir, exist_ok=True)
            count = 0
            for _src, path, _h, _s in kept_by_src.get(src_dir, []):
                dest = os.path.join(out_dir, os.path.basename(path))
                try:
                    shutil.copy2(path, dest)
                    count += 1
                except Exception as e:
                    logger.error(f"Copy failed {path} -> {dest}: {e}")
            out_dirs.append(out_dir)
            logger.info(f"  {src_dir}: kept {count}")
        return out_dirs

    # ------------------------------------------------------ shared stages (DRY)
    # Analysis width for the sharpness thumbnail. At 256px a 10-px motion
    # blur on a 4K frame is sub-pixel — invisible to the Laplacian — so blur
    # ranking there is mostly noise; 1024px keeps real defocus/motion blur
    # measurable while staying ~14x cheaper than full-res on 4K sources.
    SCAN_WIDTH = 1024

    def _scan_images(self, source_dirs, progress=None):
        """Scan every image: dHash + variance-of-Laplacian sharpness on a
        :attr:`SCAN_WIDTH`-wide thumbnail. Returns
        ``[(src_dir, path, hash, sharpness), ...]``. Shared by
        :meth:`curate` and :meth:`preview`."""
        scanned: List[Tuple[str, str, int, float]] = []
        for src_dir in source_dirs:
            if not os.path.isdir(src_dir):
                logger.warning(f"Source dir missing, skipping: {src_dir}")
                continue
            files = ImgUtils.list_image_files(src_dir)
            for i, name in enumerate(files):
                path = os.path.join(src_dir, name)
                img = cv2.imread(path)
                if img is None:
                    continue
                # Downsize for hash/sharpness speed. INTER_AREA, not the
                # default INTER_LINEAR: a ~4x+ decimation through a 2x2
                # linear tap aliases, and that aliasing feeds pseudo-random
                # high-frequency energy straight into the Laplacian-variance
                # sharpness ranking (dhash() does its own INTER_AREA resize,
                # so the hash is safe either way).
                h, w = img.shape[:2]
                if w > self.SCAN_WIDTH:
                    # max(1, ...): an extreme panorama strip (w > 1024*h)
                    # would otherwise round to height 0 and cv2.resize
                    # asserts, aborting the whole scan.
                    img = cv2.resize(
                        img,
                        (self.SCAN_WIDTH,
                         max(1, int(h * (float(self.SCAN_WIDTH) / w)))),
                        interpolation=cv2.INTER_AREA,
                    )
                scanned.append((src_dir, path, self.dhash(img), self.sharpness(img)))
                if progress is not None:
                    try:
                        progress("scan", i + 1, len(files), src_dir)
                    except Exception:
                        pass
        return scanned

    @staticmethod
    def _cluster(scanned, hash_threshold):
        """Greedy-by-anchor hash clustering: each record joins the first existing
        cluster whose anchor is within *hash_threshold* Hamming distance, else
        opens a new one. Not single-linkage — a slow pan segments into contiguous
        chunks rather than chaining into one giant cluster.

        ``hash_threshold <= 0`` disables clustering outright (every frame its
        own cluster). Hamming-0 matching would still merge *bit-identical*
        hashes — routine for consecutive frames of a paused camera on a
        thumbnail-sized dHash — silently contradicting the documented
        "0 = no dedup, keep all" contract every runner/tooltip states."""
        if hash_threshold <= 0:
            return [[record] for record in scanned]
        clusters: List[List[Tuple[str, str, int, float]]] = []
        anchor_hashes: List[int] = []
        for record in scanned:
            h = record[2]
            assigned = False
            for ci, anchor in enumerate(anchor_hashes):
                if ImageCurator.hamming(h, anchor) <= hash_threshold:
                    clusters[ci].append(record)
                    assigned = True
                    break
            if not assigned:
                clusters.append([record])
                anchor_hashes.append(h)
        return clusters

    @staticmethod
    def _select(
        clusters,
        keep_per_cluster,
        sharpness_floor,
        sharpness_floor_percentile,
        min_sharpness_fraction_of_median=0.0,
    ):
        """Keep the top-K sharpest of each cluster, then drop those below the
        resolved sharpness floor (percentile + median-fraction taken over the
        representatives — see :meth:`curate`). Returns
        ``(representatives, kept, resolved_floor)``."""
        representatives: List[Tuple[str, str, int, float]] = []
        for cluster in clusters:
            cluster.sort(key=lambda r: r[3], reverse=True)  # sharp first
            representatives.extend(cluster[:keep_per_cluster])
        rep_sharp = [r[3] for r in representatives]
        if sharpness_floor_percentile is not None and representatives:
            pct_floor = float(np.percentile(rep_sharp, sharpness_floor_percentile))
            if pct_floor > sharpness_floor:
                logger.info(
                    f"sharpness floor p{sharpness_floor_percentile:g} "
                    f"= {pct_floor:.1f} over {len(representatives)} reps "
                    f"(was {sharpness_floor})"
                )
                sharpness_floor = pct_floor
        # Median-relative guard: catch *catastrophically* defocused frames the
        # percentile misses (a frame far below the survivor median is physically
        # blurred, regardless of how many such frames there are). Self-calibrating
        # — taken over representatives so near-duplicate blur can't dilute it.
        if min_sharpness_fraction_of_median and representatives:
            med_floor = float(np.median(rep_sharp)) * min_sharpness_fraction_of_median
            if med_floor > sharpness_floor:
                logger.info(
                    f"sharpness floor {min_sharpness_fraction_of_median:g}x median "
                    f"= {med_floor:.1f} over {len(representatives)} reps "
                    f"(was {sharpness_floor})"
                )
                sharpness_floor = med_floor
        kept = [r for r in representatives if r[3] >= sharpness_floor]
        return representatives, kept, sharpness_floor

    def preview(
        self,
        source_dirs,
        hash_thresholds=(5,),
        keep_per_cluster=1,
        sharpness_floor=0.0,
        sharpness_floor_percentile=None,
        min_sharpness_fraction_of_median=0.0,
        progress=None,
    ):
        """Dry-run curation report — scan **once**, evaluate one or more
        ``hash_thresholds`` *without copying any files*. Use it to tune curation
        on a real set before committing a run: it shows how many frames survive
        at each threshold and the sharpness distribution (the blurry-tail size).

        :param source_dirs: Sequence of image directories (as in :meth:`curate`).
        :param hash_thresholds: int, or iterable of dHash Hamming thresholds to sweep.
        :param keep_per_cluster: Representatives kept per cluster (as in curate).
        :param sharpness_floor / sharpness_floor_percentile: Floor controls (as in curate).
        :return: ``{"n_scanned": int, "sharpness": {min,p5,p25,median,p75,max},
                 "thresholds": [{hash_threshold, n_clusters, n_kept, reduction_pct,
                 sharpness_floor}, ...]}``.
        :raises RuntimeError: if cv2 is unavailable.
        """
        if not self.is_available():
            raise RuntimeError("cv2 not available; cannot preview curation.")
        if isinstance(hash_thresholds, int):
            hash_thresholds = (hash_thresholds,)
        scanned = self._scan_images(source_dirs, progress=progress)
        n = len(scanned)
        report = {"n_scanned": n, "sharpness": {}, "thresholds": []}
        if n == 0:
            return report
        svals = np.array([r[3] for r in scanned], dtype=float)
        report["sharpness"] = {
            "min": float(svals.min()),
            "p5": float(np.percentile(svals, 5)),
            "p25": float(np.percentile(svals, 25)),
            "median": float(np.percentile(svals, 50)),
            "p75": float(np.percentile(svals, 75)),
            "max": float(svals.max()),
        }
        for t in hash_thresholds:
            clusters = self._cluster(scanned, t)
            _reps, kept, floor = self._select(
                clusters,
                keep_per_cluster,
                sharpness_floor,
                sharpness_floor_percentile,
                min_sharpness_fraction_of_median,
            )
            report["thresholds"].append(
                {
                    "hash_threshold": t,
                    "n_clusters": len(clusters),
                    "n_kept": len(kept),
                    "reduction_pct": round(100.0 * (1.0 - len(kept) / n), 1),
                    "sharpness_floor": round(floor, 1),
                }
            )
        return report
