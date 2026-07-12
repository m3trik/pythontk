# coding=utf-8
"""CSV mapping resolver — interprets JSON mapping files.

A mapping file is a ``.json`` file that declaratively specifies how CSV columns
map to :class:`BuilderStep` fields and how derived values (e.g. audio objects)
are resolved. See :mod:`._mapping` for the file format and the full docstring.

Package facade: the implementation lives in :mod:`._mapping` (kept out of
``__init__`` per the package convention). The public API (below) is re-exported
here so ``from ...mapping import X`` and a ``...mapping.X`` mock patch keep
working. Private helpers (the ``_audio_*`` / ``_build_*`` builders) are not
re-exported — import or patch them at ``...mapping._mapping.<name>``.
"""
from pythontk.core_utils.engines.shots.manifest.mapping._mapping import (  # noqa: F401
    DEFAULT_DIR,
    discover,
    load_mapping,
    resolve,
    templates,
)
from pythontk.core_utils.engines.shots.manifest.mapping._spec import (  # noqa: F401
    MappingSpec,
    AUDIO_METHODS,
    format_markdown,
)

__all__ = [
    "DEFAULT_DIR",
    "discover",
    "load_mapping",
    "resolve",
    "templates",
    "format_markdown",
    "MappingSpec",
    "AUDIO_METHODS",
]
