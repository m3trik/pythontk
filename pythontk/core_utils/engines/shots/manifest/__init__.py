# !/usr/bin/python
# coding=utf-8
"""Shot Manifest engine — pure CSV → shot-plan core.

The DCC-agnostic half of the Shot Manifest: parse a structured production CSV
into an ordered step/object graph, map spreadsheet columns to logical fields,
resolve behavior keying recipes and shot ranges, and plan the shots — all with
no ``maya`` / ``bpy`` / Qt import.  The DCC toolkits (mayatk, blendertk) subclass
:class:`~pythontk.core_utils.engines.shots.manifest.manifest_engine.ShotManifest`
and override its scene hooks (fps, audio measurement, key emission, object
discovery); the panel layer lives in the DCC package.

Sub-modules:
    ``manifest_model``   — dataclasses (BuilderStep/Object, PlannedShot, statuses),
                           ``ColumnMap``, ``parse_csv``, ``detect_behaviors``.
    ``mapping``          — column-mapping templates + audio/behavior pipeline (JSON).
    ``behaviors``        — behavior spec + template loading + ``resolve_keys`` math.
    ``range_resolver``   — merge user/detected ranges into resolved shot bounds.
    ``manifest_engine``  — the ``ShotManifest`` planner + DCC-hooked commit/assess.
    ``manifest_edits``   — ``ManifestEdits``: a user's edits, composed over a sheet.
    ``manifest_export``  — ``ManifestExport``: the manifest's rows as CSV, JSON or
                           Markdown, for a file or the clipboard.
    ``manifest_tags``    — ``ManifestTags``: steps' colour tags, the user's and
                           the template's automatic ``color_by``.

All symbols are lazy-loaded via the pythontk root package.
"""
