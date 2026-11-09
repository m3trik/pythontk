# !/usr/bin/python
# coding=utf-8
"""Held until 0.22.0: ``RegionMaskManifest``, for its ``schema`` notice.

v0.20.0 noticed ``RegionMaskManifest.schema`` for removal in 0.18.0 on
2026-10-24, so its window closes 2026-11-23, after 0.21.0 ships. The region
masks themselves retired on 2026-11-07 (an emissive group is a material copy,
``EmissiveGroupsModel``; its weight rides the ``render_effects`` record), so
the manifest is restored here from v0.20.0 -- its model and wire form, no
packer -- for a caller of the old name. Delete this module at 0.22.0
(``pythontk/__init__.py``, "Held until 0.22.0").
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

from pythontk.core_utils.deprecation import Deprecation

ENCODING_VERTEX_COLOR = "vertex-color"
ENCODING_CHANNELS = "channels"
ENCODING_ID = "id"  # reserved

#: slot index -> mask channel. The ceiling for both live encodings.
SLOT_CHANNELS = ("R", "G", "B", "A")


@dataclass
class _RegionGroup:
    """One named region group: its encoding ``slot`` (0=R ... 3=A), its
    ``default`` weight and the scene attribute keying it (``attr``)."""

    name: str
    slot: int
    default: float = 1.0
    attr: Optional[str] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("RegionGroup requires a non-empty name.")
        if not 0 <= int(self.slot) < len(SLOT_CHANNELS):
            raise ValueError(
                f"RegionGroup {self.name!r}: slot {self.slot} outside "
                f"0-{len(SLOT_CHANNELS) - 1}."
            )
        self.slot = int(self.slot)
        self.default = float(self.default)

    def to_dict(self) -> dict:
        data = {"name": self.name, "slot": self.slot, "default": self.default}
        if self.attr:
            data["attr"] = self.attr
        return data

    @classmethod
    def coerce(cls, group: Union["_RegionGroup", dict]) -> "_RegionGroup":
        """Accept a group or its plain-dict form."""
        if isinstance(group, cls):
            return group
        if hasattr(group, "to_dict"):
            return cls(**group.to_dict())
        return cls(**group)


@dataclass
class RegionMaskManifest:
    """Retired 2026-11-07 (removed after 0.21.0): an emissive group is a
    material copy (``EmissiveGroupsModel.render_effect_materials``).

    As v0.20.0 declared it: the wire schema joining DCC-authored groups to
    their game-engine consumer, at the retired ``emissive_groups`` record's
    version.
    """

    #: The document's schema version: the retired record's.
    VERSION = 2
    #: Retired 2026-10-20 (removed after 0.21.0): the document states ``version``.
    SCHEMA_VERSION = Deprecation.class_attribute(
        VERSION, "RegionMaskManifest.VERSION", remove_in="0.18.0", since="2026-10-24"
    )

    groups: List[_RegionGroup] = field(default_factory=list)
    encoding: str = ENCODING_VERTEX_COLOR
    version: int = VERSION
    color_set: Optional[str] = None
    mask: Optional[str] = None
    uv_channel: Optional[int] = None

    _ENCODINGS = (ENCODING_VERTEX_COLOR, ENCODING_CHANNELS, ENCODING_ID)

    @property
    @Deprecation.symbol(
        "RegionMaskManifest.version", remove_in="0.18.0", since="2026-10-24"
    )
    def schema(self) -> int:
        """The document's version (:attr:`version`)."""
        return self.version

    def __post_init__(self):
        if self.encoding not in self._ENCODINGS:
            raise ValueError(
                f"Unknown encoding {self.encoding!r}; expected one of {self._ENCODINGS}."
            )
        self.groups = [_RegionGroup.coerce(g) for g in self.groups]
        slots = [g.slot for g in self.groups]
        if len(set(slots)) != len(slots):
            raise ValueError(f"Duplicate slot assignment in groups: {slots}")
        names = [g.name for g in self.groups]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate group name in groups: {names}")

    @classmethod
    def vertex_color(
        cls, groups: Sequence[Union[_RegionGroup, dict]], color_set: str = "emissiveGroups"
    ) -> "RegionMaskManifest":
        """Manifest for membership riding in a mesh color set."""
        return cls(groups=list(groups), encoding=ENCODING_VERTEX_COLOR, color_set=color_set)

    @classmethod
    @Deprecation.parameter(
        "resolution",
        remove_in="0.18.0",
        drop=True,
        reason="The mask image states its own size.",
        since="2026-10-24",
    )
    def channels(
        cls,
        groups: Sequence[Union[_RegionGroup, dict]],
        mask: str,
        resolution: Optional[int] = None,
        uv_channel: int = 0,
    ) -> "RegionMaskManifest":
        """Manifest for membership rasterized into an RGBA mask texture;
        *resolution* is retired and ignored, as v0.20.0 had it."""
        if resolution is not None:  # passed by position: the decorator sees keywords only
            Deprecation.warn(
                "RegionMaskManifest.channels parameter 'resolution'",
                "nothing -- the value already has no effect",
                remove_in="0.18.0",
                reason="The mask image states its own size.",
                kind="parameter",
                stacklevel=2,
                since="2026-10-24",
            )
        return cls(
            groups=list(groups),
            encoding=ENCODING_CHANNELS,
            mask=os.path.basename(mask),
            uv_channel=int(uv_channel),
        )

    def to_dict(self) -> dict:
        """Wire form: encoding-irrelevant fields are omitted, not null."""
        data = {
            "version": self.version,
            "encoding": self.encoding,
            "groups": [g.to_dict() for g in self.groups],
        }
        if self.encoding == ENCODING_VERTEX_COLOR:
            data["color_set"] = self.color_set or "emissiveGroups"
        elif self.encoding == ENCODING_CHANNELS:
            data["mask"] = self.mask
            data["uv_channel"] = self.uv_channel or 0
        return data

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "RegionMaskManifest":
        """*data* read as the version this reads (:attr:`VERSION`).

        Raises:
            ValueError: *data* states another version, or none.
        """
        if data.get("version") != cls.VERSION:
            raise ValueError(
                f"Region mask manifest v{data.get('version')} is not the "
                f"v{cls.VERSION} this reader reads; re-export it."
            )
        known = {"version", "encoding", "groups", "color_set", "mask", "uv_channel"}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, text: str) -> "RegionMaskManifest":
        return cls.from_dict(json.loads(text))

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(self.to_json(indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str) -> "RegionMaskManifest":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())
