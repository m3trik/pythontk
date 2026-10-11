# !/usr/bin/python
# coding=utf-8
"""Naming convention -- the ecosystem's one table of type-based name affixes.

* :mod:`._naming_convention` -- :class:`NamingConvention`, the facade (read,
  edit and persist the table) and :class:`AffixRule`, one entry.
* ``presets/`` -- the shipped, read-only snapshots of the table that
  ``NamingConvention.preset_store()`` offers beside the user's own
  (``default.json``: the shipped baseline).

The root registers both names (``from pythontk import NamingConvention``); this
package path resolves the same classes, loading no module until first use
(``lazy_exports``, CODE_STANDARD section 4).
"""

from pythontk.core_utils.module_resolver import lazy_exports

lazy_exports(
    globals(),
    {
        "_naming_convention": [
            "AffixRule",
            "NamingConvention",
            "AFFIX_MODES",
            "CONFIG_ENV_VAR",
            "CONFIG_NAME",
            "CONFIG_PACKAGE",
        ]
    },
)
