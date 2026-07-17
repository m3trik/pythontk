# !/usr/bin/python
# coding=utf-8
"""Shared infrastructure — the non-data-type half of pythontk.

Where the ``*_utils`` siblings hold general primitives placed by data type
(strings, images, math, …), this package holds the *mechanisms* the ecosystem
runs on. Deliberately flat: these module paths are import contracts for every
downstream package (uitk, mayatk, blendertk, tentacle, extapps), so modules are
not re-nested for cosmetics. The clusters, for navigation:

- **General helpers** — :mod:`._core_utils` (``CoreUtils``).
- **Class infrastructure** — :mod:`.class_property`, :mod:`.help_mixin`,
  :mod:`.logging_mixin`, :mod:`.singleton_mixin`, :mod:`.namedtuple_container`,
  :mod:`.namespace_handler`.
- **App/process orchestration** — :mod:`.app_launcher`, :mod:`.app_installer`,
  :mod:`.app_handoff`, :mod:`.script_run`, :mod:`.script_template`,
  :mod:`.process_stream`, :mod:`.execution_monitor`.
- **Config & persistence** — :mod:`.user_config`, :mod:`.preset_store`,
  :mod:`.schema_spec`, :mod:`.template_set`.
- **Package/dev infrastructure** — :mod:`.module_resolver`,
  :mod:`.module_reloader`, :mod:`.package_manager`, :mod:`.git`, :mod:`.cli`,
  :mod:`.symbol_record`.
- **Pipeline primitives** — :mod:`.task_factory`, :mod:`.qc_log`.
- **Data-structure utilities** — :mod:`.hierarchy_utils`.
- **Homeless shared primitives** — :mod:`.color` (``Color``/``ColorPair``/
  ``Palette``). A single-module value primitive with no dedicated ``*_utils``
  home: a root ``<name>_utils`` subpackage earns its place with a ``*_utils``
  facade (``StrUtils``, ``MathUtils``, …) or a multi-module primitive *family*
  (``geo_utils``); a lone module clears neither bar, and wrapping it in a
  package or inventing a ``ColorUtils`` facade would be nesting-for-cosmetics.
  So it lodges here beside the other cross-cutting shared code — the same reason
  ``file_utils`` hosts the ``Metadata`` primitive.
- **Domain engines** — :mod:`.engines` (placement charter: ``pythontk/CLAUDE.md``).

All classes are lazy-loaded via the pythontk root package.
Import from pythontk directly: ``from pythontk import CoreUtils``.
"""

# Lazy-loaded via parent package - no explicit imports needed
