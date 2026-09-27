# !/usr/bin/python
# coding=utf-8
"""The GLB family of ``mesh_convert``: the container and the passes built on it.

``edit`` holds :class:`GlbEdit`, the one GLB container parser and writer every
pass edits through; ``reader`` (:class:`GlbReader`) only reads; ``clips``,
``fades``, ``key_reduction`` and ``tangents`` rewrite animation and geometry;
``pipeline`` (:class:`GlbPipeline`) is the one FBX -> GLB build.

Public names are registered by the pythontk root (``DEFAULT_INCLUDE``), so this
``__init__`` imports nothing: ``from pythontk import GlbReader``.
"""
