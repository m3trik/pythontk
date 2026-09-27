# !/usr/bin/python
# coding=utf-8
"""Image utilities for Python.

All classes are lazy-loaded via pythontk root package.
Import from pythontk directly: from pythontk import ImgUtils

``ImgUtils`` (``_img_utils.py``) is the facade: format table, IO and the public
signatures + docstrings. Its bodies are split by job into internal bases:
``_codecs`` (per-format encoders), ``_image_header`` (sizes / integrity without
decoding), ``_channels``, ``_filters``, ``_atlas``, ``_rasterize`` and
``_color_space``.
"""

# Lazy-loaded via parent package - no explicit imports needed


# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
