# !/usr/bin/python
# coding=utf-8
"""Shared infrastructure — the non-data-type half of pythontk.

Where the ``*_utils`` siblings hold general primitives placed by data type
(strings, images, math, …), this package holds the *mechanisms* the ecosystem
runs on. Flat by default, never re-nested for cosmetics: a module moves into a
subpackage only when a CODE_STANDARD §3 trigger fires (a second cohesive
module, private internals, an owned asset directory). Downstream packages still
import many of these module paths directly, so every move follows §14: grep the
workspace, and alias an externally imported path for one release. The
clusters, for navigation:

- **General helpers** — :mod:`._core_utils` (``CoreUtils``).
- **Class infrastructure** — :mod:`.class_property`, :mod:`.help_mixin`,
  :mod:`.logging_mixin`, :mod:`.singleton_mixin`, :mod:`.namedtuple_container`,
  :mod:`.namespace_handler`.
- **App/process orchestration** — :mod:`.app_launcher`, :mod:`.app_installer`,
  :mod:`.handoff` (``app_handoff``, ``script_template``, ``script_run``,
  ``manifest``, ``manifest_plan``), :mod:`.process_stream`,
  :mod:`.execution_monitor`, :mod:`.cancel_scope`.
- **Config & persistence** — :mod:`.user_config`, :mod:`.presets`,
  :mod:`.schema_spec`, :mod:`.template_set`.
- **Package/dev infrastructure** — :mod:`.module_resolver`,
  :mod:`.module_reloader`, :mod:`.package_manager`, :mod:`.cli`,
  :mod:`.symbol_record`, :mod:`.status_badge`, :mod:`.test_sandbox`,
  :mod:`.doc_audit`.
- **Pipeline primitives** — :mod:`.task_factory`, :mod:`.qc_log`,
  :mod:`.step_toggle`.
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

Human-facing module map with per-module summaries: ``README.md`` in this
directory.
"""

# Lazy-loaded via parent package - no explicit imports needed
