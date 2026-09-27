# !/usr/bin/python
# coding=utf-8
"""Texture Map Factory for PBR workflow preparation.

Dynamic, extensible factory for processing and preparing texture maps for
various PBR workflows (Unity, Unreal, glTF, etc.).

Architecture (split out of the original single-file module):
    conversions  -- ``MapConversion`` / ``ConversionRegistry`` registry plumbing
    processor    -- ``TextureProcessor`` shared per-set processing context
    handlers     -- ``WorkflowHandler`` strategies (ORM, MRAO, mask, ...)
    _map_factory -- ``MapFactory`` orchestrator (the public entry point); its
                    public methods keep their signatures and docstrings and
                    delegate to four internal bases, one per job:
                    ``_texture_sets`` (classification, base names, tiles, set
                    grouping), ``_map_inventory`` (redundant-map pruning),
                    ``_converters`` (the converters ``register_conversions``
                    wires in) and ``_channel_packing`` (pack / unpack)

``processor`` and ``handlers`` call MapFactory's stateless primitives at runtime
but cannot import it at module load -- MapFactory names the handler classes at
class-definition time -- so each resolves the name through its own module-level
``__getattr__``. Importing this package loads none of them: the names below
resolve on first use (``lazy_exports``, CODE_STANDARD section 4), since the
root's scan imports this ``__init__`` on every ``import pythontk``.

Public API is unchanged: ``from pythontk import MapFactory`` resolves through the
lazy root exactly as before. The engine was relocated from ``img_utils`` into the
``core_utils/engines/textures`` domain-engine namespace, so the *internal* path is
now ``from pythontk.core_utils.engines.textures.map_factory import MapFactory``.
"""

from pythontk.core_utils.module_resolver import lazy_exports

lazy_exports(
    globals(),
    {
        "conversions": ("MapConversion", "ConversionRegistry"),
        "processor": "TextureProcessor",
        "handlers": (
            "WorkflowHandler",
            "BaseColorHandler",
            "NormalMapHandler",
            "ORMMapHandler",
            "MRAOMapHandler",
            "MaskMapHandler",
            "MetallicSmoothnessHandler",
            "OutputFallbackHandler",
            "SeparateMetallicRoughnessHandler",
        ),
        "_map_factory": "MapFactory",
    },
)
