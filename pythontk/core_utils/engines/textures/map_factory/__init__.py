# !/usr/bin/python
# coding=utf-8
"""Texture Map Factory for PBR workflow preparation.

Dynamic, extensible factory for processing and preparing texture maps for
various PBR workflows (Unity, Unreal, glTF, etc.).

Architecture (split out of the original single-file module):
    conversions  -- ``MapConversion`` / ``ConversionRegistry`` registry plumbing
    processor    -- ``TextureProcessor`` shared per-set processing context
    handlers     -- ``WorkflowHandler`` strategies (ORM, MRAO, mask, ...)
    _map_factory -- ``MapFactory`` orchestrator (the public entry point)

``processor`` and ``handlers`` call MapFactory's stateless primitives at runtime
but cannot import it at module load -- MapFactory names the handler classes at
class-definition time -- so each resolves the name through its own module-level
``__getattr__``. Importing this package has no side effects.

Public API is unchanged: ``from pythontk import MapFactory`` resolves through the
lazy root exactly as before. The engine was relocated from ``img_utils`` into the
``core_utils/engines/textures`` domain-engine namespace, so the *internal* path is
now ``from pythontk.core_utils.engines.textures.map_factory import MapFactory``.
"""

from .conversions import MapConversion, ConversionRegistry
from .processor import TextureProcessor
from .handlers import (
    WorkflowHandler,
    BaseColorHandler,
    NormalMapHandler,
    ORMMapHandler,
    MRAOMapHandler,
    MaskMapHandler,
    MetallicSmoothnessHandler,
    OutputFallbackHandler,
    SeparateMetallicRoughnessHandler,
)
from ._map_factory import MapFactory

__all__ = [
    "MapFactory",
    "MapConversion",
    "ConversionRegistry",
    "TextureProcessor",
    "WorkflowHandler",
    "BaseColorHandler",
    "NormalMapHandler",
    "ORMMapHandler",
    "MRAOMapHandler",
    "MaskMapHandler",
    "MetallicSmoothnessHandler",
    "OutputFallbackHandler",
    "SeparateMetallicRoughnessHandler",
]
