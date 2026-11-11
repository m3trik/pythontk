# !/usr/bin/python
# coding=utf-8
"""The ``lods`` record's payload, declared once.

:attr:`pythontk.SceneRecords.LODS` names :class:`LodsRecord` as its shape: what
both DCCs' ``LodUtils.export_record`` publishes on the ``data_export`` carrier
-- per LOD group (:class:`pythontk.LodChain`) its node, its levels and its
switch heights -- and what the record's engine consumer builds one LOD group
per group from (``SceneRecords.LODS.consumers``). An engine's FBX import may
read the ``<base>_LOD<n>`` names alone: one engine's merged every level-named
mesh of a file into ONE group on the file's root, by level index, and ignored
the FBX's LodGroup thresholds (measured 2026-11-10). A GLB states LODs in glTF's own
terms (``MSFT_lod``) and a USD layer in a variant set, so neither carries this
record.

A typed :class:`~pythontk.SchemaSpec`: this one definition validates a payload
(``LodsRecord.validate(payload)``), publishes its JSON Schema, and is what
``m3trik/scripts/sync_scene_records.py`` generates the consumers' C# record types
from. Change a field together with both producers and bump
``SceneRecords.LODS.version``.
"""

from dataclasses import dataclass
from typing import List

from pythontk.core_utils.schema_spec import SchemaSpec

_field = SchemaSpec.spec_field


@dataclass
class LodGroupEntry(SchemaSpec):
    """One LOD group: its node, its levels most detailed first, and the
    screen height each level is drawn down to."""

    TYPED = True

    name: str = _field(
        help=(
            "The group's node: the object whose children are its levels, by "
            "name, as every record binds a node (exactly, else its "
            "namespace-stripped leaf)."
        ),
        required=True,
        example="chair",
    )
    levels: List[str] = _field(
        help=(
            "The level nodes under it, LOD0 (the most detailed) first: "
            "'<base>_LOD<n>' by the convention engines read; each a mesh or a "
            "subtree of them."
        ),
        required=True,
        example=["chair_LOD0", "chair_LOD1", "chair_LOD2"],
        default_factory=list,
    )
    heights: List[float] = _field(
        help=(
            "One switch height per level: the fraction of the screen's height "
            "(0..1) below which that level is no longer drawn, falling level "
            "by level; the last is the cull height, 0 for never culled: an "
            "engine's screen-relative transition height."
        ),
        required=True,
        example=[0.5, 0.25, 0.0],
        default_factory=list,
    )


@dataclass
class LodsRecord(SchemaSpec):
    """The ``lods`` record's payload: the scene's LOD groups."""

    TYPED = True

    version: int = _field(
        help="The record's schema version (SceneRecords.LODS.version).",
        required=True,
        example=1,
    )
    groups: List[LodGroupEntry] = _field(
        help="Every LOD group the export ships, by node name.",
        required=True,
        nested=[LodGroupEntry],
        default_factory=list,
    )
