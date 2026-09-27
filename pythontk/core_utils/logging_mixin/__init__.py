# !/usr/bin/python
# coding=utf-8
"""Class-scoped logging toolkit.

``LoggerExt`` patches stdlib loggers with custom levels (PROGRESS/SUCCESS/
RESULT/NOTICE), HTML color presets, raw block output (boxes, groups,
dividers, tables), a managed file tee, and a capped in-memory ring buffer.
``LoggingMixin`` exposes one such patched logger per class.

One feature, formerly one 2,000-line module:

* :mod:`._logging_mixin` -- ``LoggingMixin``, the facade: a patched logger per
  class and the class-scoped sink controls.
* :mod:`.logger_ext` -- ``LoggerExt``, the logger patch (levels, sinks, raw
  blocks, duplicate suppression, file tee, ring buffer), with the formatters
  (``StripHtmlFormatter``, ``LevelAwareFormatter``) and handlers
  (``DefaultTextLogHandler``, ``RingBufferHandler``) it attaches.
* :mod:`.table_mixin` -- ``TableMixin``: tables and titled groups, one record
  each, on any class with a ``logger``.
* :mod:`._text_layout` -- ``TextLayout`` (internal): display widths, wrapping,
  box and table geometry -- the one owner of the column math, reached through
  the instance ``LoggerExt._layout`` holds.

The root registers the public names (``from pythontk import LoggingMixin``).
This package path resolves the same objects, loading none of the modules until
a name is first used (``lazy_exports``, CODE_STANDARD section 4).
"""

from pythontk.core_utils.module_resolver import lazy_exports

lazy_exports(
    globals(),
    {
        "_logging_mixin": "LoggingMixin",
        "logger_ext": (
            "LoggerExt",
            "StripHtmlFormatter",
            "LevelAwareFormatter",
            "DefaultTextLogHandler",
            "RingBufferHandler",
        ),
        "table_mixin": "TableMixin",
    },
)
