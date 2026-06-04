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

Public surface is unchanged: ``from pythontk import MapFactory`` and
``from pythontk.img_utils.map_factory import MapFactory`` resolve as before.
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

# Late-bind MapFactory into the submodules that call its stateless primitive
# library at runtime. processor/handlers cannot import it at module load --
# MapFactory references the handler classes at class-definition time, which
# would form a circular import -- so the resolved class is injected here, after
# every collaborator is defined.
from . import processor as _processor
from . import handlers as _handlers

_processor.MapFactory = MapFactory
_handlers.MapFactory = MapFactory

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
