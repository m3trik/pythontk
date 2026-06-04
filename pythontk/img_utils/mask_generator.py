#!/usr/bin/python
# coding=utf-8
"""Background mask generation via rembg (optional dependency).

``rembg`` is ONNX-based; install with ``pip install rembg`` or
``pip install rembg[gpu]``. When unavailable, :class:`MaskGenerator`
constructs cleanly but :meth:`is_available` returns False and
:meth:`generate_masks` short-circuits with a logged error — matching
the pattern used by ``FrameExtractor`` for cv2.

Output filenames default to ``{basename}_mask.png`` so they pair with
Metashape's ``importMasks(template="{filename}_mask.png")``.
"""
import io
import logging
import os
from typing import List, Optional

try:
    from rembg import new_session, remove

    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    new_session = None
    remove = None

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


class MaskGenerator:
    """Run rembg over a directory of images and write binary masks."""

    DEFAULT_MODEL = "u2net"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.session = new_session(model_name) if REMBG_AVAILABLE else None

    def is_available(self) -> bool:
        return REMBG_AVAILABLE and PIL_AVAILABLE and self.session is not None

    def generate_masks(
        self,
        input_dir: str,
        output_dir: str,
        suffix: str = "_mask",
        out_ext: str = ".png",
        skip_existing: bool = True,
        progress: Optional[callable] = None,
    ) -> List[str]:
        """Generate alpha-channel masks for every image in ``input_dir``.

        Returns the list of mask paths written. When ``rembg`` or ``PIL``
        is missing, logs an error and returns an empty list — matches the
        cv2-guarded posture of :class:`FrameExtractor`.
        """
        if not self.is_available():
            logger.error(
                "Mask generation unavailable. "
                f"rembg={REMBG_AVAILABLE}, PIL={PIL_AVAILABLE}."
            )
            return []
        if not os.path.isdir(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            return []

        os.makedirs(output_dir, exist_ok=True)
        sources = sorted(
            f for f in os.listdir(input_dir)
            if f.lower().endswith(IMAGE_EXTS)
        )
        if not sources:
            logger.warning(f"No images found in {input_dir}")
            return []

        written: List[str] = []
        for i, name in enumerate(sources):
            stem, _ = os.path.splitext(name)
            out_path = os.path.join(output_dir, f"{stem}{suffix}{out_ext}")
            if skip_existing and os.path.exists(out_path):
                written.append(out_path)
                continue
            try:
                with open(os.path.join(input_dir, name), "rb") as f:
                    raw = f.read()
                cut = remove(raw, session=self.session)
                # rembg returns RGBA; we save the alpha as a grayscale mask
                img = Image.open(io.BytesIO(cut)).convert("RGBA")
                alpha = img.split()[-1]
                alpha.save(out_path)
                written.append(out_path)
            except Exception as e:
                logger.error(f"Mask generation failed for {name}: {e}")
            if progress is not None:
                try:
                    progress(i + 1, len(sources))
                except Exception:
                    pass
        logger.info(f"Generated {len(written)}/{len(sources)} masks.")
        return written
