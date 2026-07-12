# !/usr/bin/python
# coding=utf-8
"""Per-map output-format templates — the "export preset" layer.

Separates *delivery format* (container, bit depth, optional GPU compression) from
*content correctness* (color space, channels, normal convention), which lives on
:class:`~pythontk.core_utils.engines.textures.map_registry.MapType`. A template maps each map type to
an :class:`OutputSpec` for a target profile — the per-map export preset you'd see in
Substance Painter.

:class:`OutputTemplates` owns the built-in catalogue (keyed by
:class:`~pythontk.core_utils.engines.textures.map_registry.WF` profile) and resolution — the read-only
tier. The templates are deliberately plain data (``to_dict``/``from_dict``) so a future
user-editable layer (``pythontk.PresetStore`` built-in + user tiers) can wrap them
without rework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from pythontk.core_utils.engines.textures.map_registry import WF


@dataclass(frozen=True)
class OutputSpec:
    """How a single map is written to disk.

    Attributes:
        ext: Container/extension — "png", "tga", "tiff", "exr", "dds".
        bit_depth: Per-channel bit depth — 8, 16, or 32 (32 = float, EXR/HDR).
        compression: None (uncompressed) or a DDS block format. "DXT1"/"DXT3"/
            "DXT5"/"BC5" are written by Pillow directly; "BC7"/"BC6H" require an
            external codec registered via ``ImgUtils.register_dds_codec``.
    """

    ext: str = "png"
    bit_depth: int = 8
    compression: Optional[str] = None

    def to_dict(self) -> dict:
        return {"ext": self.ext, "bit_depth": self.bit_depth, "compression": self.compression}

    @classmethod
    def from_dict(cls, d: dict) -> "OutputSpec":
        return cls(
            ext=d.get("ext", "png"),
            bit_depth=int(d.get("bit_depth", 8)),
            compression=d.get("compression"),
        )


@dataclass(frozen=True)
class OutputTemplate:
    """A profile's per-map output formats: a default spec + per-map-type overrides."""

    default: OutputSpec = field(default_factory=OutputSpec)
    overrides: Dict[str, OutputSpec] = field(default_factory=dict)

    def resolve(self, map_type: Optional[str]) -> OutputSpec:
        """Return the :class:`OutputSpec` for *map_type* (falls back to ``default``)."""
        if map_type and map_type in self.overrides:
            return self.overrides[map_type]
        return self.default

    def to_dict(self) -> dict:
        return {
            "default": self.default.to_dict(),
            "overrides": {k: v.to_dict() for k, v in self.overrides.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OutputTemplate":
        return cls(
            default=OutputSpec.from_dict(d.get("default", {})),
            overrides={
                k: OutputSpec.from_dict(v) for k, v in (d.get("overrides") or {}).items()
            },
        )


class OutputTemplates:
    """Registry of the built-in per-profile output templates and their resolution.

    Owns the read-only built-in tier — the per-:class:`~pythontk.core_utils.engines.textures.map_registry.WF`
    catalogue plus the lookup helpers — so there is a single surface a future
    user-editable layer (``pythontk.PresetStore`` built-in + user tiers) can wrap.
    The plain-data :class:`OutputSpec` / :class:`OutputTemplate` above carry the
    values; this class owns the catalogue and resolution logic.
    """

    # Maps whose surface detail benefits from 16-bit precision (parallax /
    # tessellation / displacement) — banding here reads as visible stepping in-engine.
    _PRECISION_16: Dict[str, OutputSpec] = {
        "Height": OutputSpec("png", 16),
        "Displacement": OutputSpec("png", 16),
        "Bump": OutputSpec("png", 16),
    }
    _NORMAL_TYPES = ("Normal", "Normal_OpenGL", "Normal_DirectX")

    # Profile-agnostic fallback (no profile, or an unknown one).
    DEFAULT = OutputTemplate(default=OutputSpec("png", 8), overrides=dict(_PRECISION_16))

    # Built-in per-profile templates (read-only tier). Populated below the class so
    # the catalogue can be assembled with the class's own ``_build`` helper.
    BUILTIN: Dict[str, OutputTemplate] = {}

    @classmethod
    def _build(cls, default_ext: str, normal_ext: Optional[str] = None) -> OutputTemplate:
        """Build a template: ``default_ext`` for everything, 16-bit for height-like
        maps, and an optional distinct container for normal maps."""
        overrides: Dict[str, OutputSpec] = dict(cls._PRECISION_16)
        if normal_ext:
            for n in cls._NORMAL_TYPES:
                overrides[n] = OutputSpec(normal_ext, 8)
        return OutputTemplate(default=OutputSpec(default_ext, 8), overrides=overrides)

    @classmethod
    def get(cls, profile: Optional[str]) -> OutputTemplate:
        """Return the built-in template for *profile* (a ``WF`` key), or the default."""
        if profile and profile in cls.BUILTIN:
            return cls.BUILTIN[profile]
        return cls.DEFAULT

    @classmethod
    def resolve(
        cls, map_type: Optional[str], profile: Optional[str] = None
    ) -> OutputSpec:
        """Resolve the :class:`OutputSpec` for *map_type* under *profile*."""
        return cls.get(profile).resolve(map_type)


# Built-in catalogue — engine-import oriented: correct *uncompressed* source per map.
# Assembled here (post-class) so it can use ``OutputTemplates._build``. Tune here or,
# later, via an editable PresetStore layer.
OutputTemplates.BUILTIN = {
    WF.STD: OutputTemplates._build("png"),
    WF.URP: OutputTemplates._build("png"),
    WF.HDRP: OutputTemplates._build("png"),
    WF.GODOT: OutputTemplates._build("png"),
    WF.GLTF: OutputTemplates._build("png"),  # glTF references PNG/JPG
    WF.UE: OutputTemplates._build("tga", normal_ext="tga"),  # UE commonly prefers TGA
    WF.SPEC: OutputTemplates._build("png"),
}
