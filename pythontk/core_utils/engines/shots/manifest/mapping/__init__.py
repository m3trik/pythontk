# coding=utf-8
"""CSV mapping resolver — interprets JSON mapping files.

A mapping file is a ``.json`` file that declaratively specifies how CSV columns
map to :class:`BuilderStep` fields and how derived values (e.g. audio objects)
are resolved. See :mod:`._mapping` for the file format and the full docstring.

Package facade: the implementation lives in :mod:`._mapping` / :mod:`._spec`
(kept out of ``__init__`` per the package convention). The public classes are
re-exported here so ``from ...mapping import Mapping`` and a
``...mapping.Mapping.<method>`` mock patch keep working. Private helpers (the
``_audio_*`` / ``_build_*`` builders) live on :class:`Mapping`'s internal base.
"""

from pythontk.core_utils.engines.shots.manifest.mapping._mapping import (  # noqa: F401
    Mapping,
    DEFAULT_DIR,
)
from pythontk.core_utils.engines.shots.manifest.mapping._spec import (  # noqa: F401
    MappingSpec,
    AUDIO_METHODS,
)

__all__ = [
    "Mapping",
    "MappingSpec",
    "DEFAULT_DIR",
    "AUDIO_METHODS",
]
