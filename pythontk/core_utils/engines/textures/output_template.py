# !/usr/bin/python
# coding=utf-8
"""Per-map output-format templates — the "export preset" layer.

Separates *delivery format* (container, bit depth, optional GPU compression) from
*content correctness* (color space, channels, normal convention), which lives on
:class:`~pythontk.core_utils.engines.textures.map_registry.MapType`. A template maps each map type to
an :class:`OutputSpec` for a target profile — the per-map export preset you'd see in
Substance Painter.

A template carries two tiers, split by whether a violation is *wrong* or merely
*expensive*:

- :class:`OutputSpec` — **hard**. Container, bit depth, and GPU compression are
  handed straight to the writer. A 16-bit height map written as 8-bit is not a
  budget overrun, it's banding; there is nothing to negotiate.
- :class:`DeliveryBudget` — **advisory**. Texture-size ceiling and power-of-two
  expectation. An over-budget map is still correct, it just costs the target
  platform more memory than it wants to spend, so callers are *warned* by default
  and opt in to enforcement (``enforce_budget=True`` on
  :meth:`~pythontk.core_utils.engines.textures.map_optimizer.MapOptimizer.optimize_map`)
  rather than having pixels resampled behind their back.

:class:`OutputTemplates` owns the built-in catalogue (keyed by
:class:`~pythontk.core_utils.engines.textures.map_registry.WF` profile) and resolution — the read-only
tier. The templates are deliberately plain data (``to_dict``/``from_dict``) so a future
user-editable layer (``pythontk.PresetStore`` built-in + user tiers) can wrap them
without rework.

Note on WebXR: it is not a profile of its own — the WebXR runtime material model
*is* glTF 2.0 (ORM packing, +Y normals), so ``WF.GLTF`` is the profile to pick, and
its budget carries the tighter web/headset delivery ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pythontk.core_utils.engines.textures.map_registry import WF, MapRegistry


@dataclass(frozen=True)
class OutputSpec:
    """How a single map is written to disk.

    Attributes:
        ext: Container/extension — "png", "tga", "tiff", "exr", "dds", or the
            delivery container "ktx2" (encode-only; needs the ``toktx`` encoder
            — see ``ImgUtils.DELIVERY_FORMATS``).
        bit_depth: Per-channel bit depth — 8, 16, or 32 (32 = float, EXR/HDR).
            ``ktx2`` is 8-bit LDR regardless.
        compression: None (uncompressed) or a GPU compression scheme for the
            container. For "dds": a block format — "DXT1"/"DXT3"/"DXT5"/"BC5"
            are written by Pillow directly; "BC7"/"BC6H" require an external
            codec registered via ``ImgUtils.register_dds_codec``. For "ktx2": a
            Basis codec — "UASTC" (high quality: normals, packed/linear data)
            or "ETC1S" (low bitrate: unpacked sRGB color). None on a ktx2
            target means *derive per map type* —
            ``MapOptimizer.resolve_compression`` picks ETC1S exactly where
            ``MapRegistry.is_lossy_safe`` allows a lossy codec and UASTC
            everywhere else, and refuses (upgrades) an explicit ETC1S on a map
            that cannot survive it.
        quality: None (lossless) or a 1-100 lossy quality for the container's
            codec. Distinct from ``compression``, which selects a *GPU block
            format*: this is CPU-side container compression (WebP/JPEG), it
            changes pixels rather than layout, and the two are independent.
            Every built-in template leaves this None — see the catalogue note at
            the bottom of this module. It exists so a profile *can* express
            "ship albedo at q90" once a user-editable preset layer lands, and so
            the value round-trips through ``to_dict``/``from_dict`` when it does.
            A quality set here still passes the per-map-type safety gate in
            ``MapRegistry.is_lossy_safe`` before reaching the writer.
    """

    ext: str = "png"
    bit_depth: int = 8
    compression: Optional[str] = None
    quality: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "ext": self.ext,
            "bit_depth": self.bit_depth,
            "compression": self.compression,
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OutputSpec":
        quality = d.get("quality")
        return cls(
            ext=d.get("ext", "png"),
            bit_depth=int(d.get("bit_depth", 8)),
            compression=d.get("compression"),
            quality=int(quality) if quality is not None else None,
        )


@dataclass(frozen=True)
class DeliveryBudget:
    """A profile's advisory delivery limits — reported by default, not enforced.

    The soft tier of a template (see the module docstring). These describe what
    the target *platform* wants to pay for, which is a judgement call the
    content owner makes: a 4K albedo bound for a standalone headset is a
    problem, the same file bound for a cinematic is not. So a violation
    produces a warning, and resampling happens only when a caller opts in.

    Applies to the whole template rather than per map type — a budget is a
    property of the delivery target, and per-map resolution intent is already
    modelled upstream by ``MapType.resolution_critical``.

    Attributes:
        max_size: Advisory ceiling for the longest edge, in pixels. None =
            unbudgeted (authoring / offline targets, where the map is an
            intermediate rather than a shipped asset).
        force_pot: Target platforms expect power-of-two dimensions.
    """

    max_size: Optional[int] = None
    force_pot: bool = False

    @staticmethod
    def _is_pot(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    def check(self, width: int, height: int) -> List[str]:
        """Return one message per budget rule ``width`` x ``height`` violates.

        The wording lives here rather than at the call sites for the same reason
        :class:`~pythontk.core_utils.engines.textures.map_optimizer.Op` carries its own
        ``description``: the dry-run twin and the real run must say the same
        thing, and two copies of a sentence are two chances to drift.

        Returns:
            list[str]: Empty when the dimensions are within budget.
        """
        messages: List[str] = []
        if self.max_size and max(width, height) > self.max_size:
            messages.append(
                f"Over delivery budget: {width}x{height} exceeds the profile's "
                f"advisory max_size of {self.max_size}"
            )
        if self.force_pot and not (self._is_pot(width) and self._is_pot(height)):
            messages.append(
                f"Non-power-of-two: {width}x{height} — this profile's target "
                f"platforms expect POT dimensions"
            )
        return messages

    def to_dict(self) -> dict:
        return {"max_size": self.max_size, "force_pot": self.force_pot}

    @classmethod
    def from_dict(cls, d: dict) -> "DeliveryBudget":
        max_size = d.get("max_size")
        return cls(
            max_size=int(max_size) if max_size else None,
            force_pot=bool(d.get("force_pot", False)),
        )


@dataclass(frozen=True)
class OutputTemplate:
    """A profile's per-map output formats: a default spec + per-map-type overrides.

    Plus the profile's advisory :class:`DeliveryBudget` — the two tiers travel
    together because they answer the same question ("how does this target want
    its textures?"), and separating them would let a UI offer a format preset
    whose size expectations silently belong to a different target.
    """

    default: OutputSpec = field(default_factory=OutputSpec)
    overrides: Dict[str, OutputSpec] = field(default_factory=dict)
    budget: DeliveryBudget = field(default_factory=DeliveryBudget)

    def resolve(self, map_type: Optional[str]) -> OutputSpec:
        """Return the :class:`OutputSpec` for *map_type* (falls back to ``default``)."""
        if map_type and map_type in self.overrides:
            return self.overrides[map_type]
        return self.default

    def to_dict(self) -> dict:
        return {
            "default": self.default.to_dict(),
            "overrides": {k: v.to_dict() for k, v in self.overrides.items()},
            "budget": self.budget.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OutputTemplate":
        return cls(
            default=OutputSpec.from_dict(d.get("default", {})),
            overrides={
                k: OutputSpec.from_dict(v) for k, v in (d.get("overrides") or {}).items()
            },
            budget=DeliveryBudget.from_dict(d.get("budget") or {}),
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
    # The registry owns the taxonomy; a local copy would silently miss a normal
    # type added there (a new one would keep the default container instead of
    # the profile's).
    _NORMAL_TYPES = MapRegistry.NORMAL_TYPES

    # Advisory ceilings by delivery class, not by engine — the same numbers recur
    # across profiles, and naming the *reason* keeps the catalogue below readable.
    _BUDGET_NONE = DeliveryBudget()  # authoring / offline: the map is an intermediate
    _BUDGET_REALTIME = DeliveryBudget(max_size=4096)  # high-end PC / current-gen console
    _BUDGET_MOBILE = DeliveryBudget(max_size=2048)  # mobile, standalone VR, mid-tier PC
    # Web/WebXR ships over the network and decodes on the client, so it inherits the
    # mobile ceiling *and* wants POT (mip generation on the GL/WebGPU backends).
    _BUDGET_WEB = DeliveryBudget(max_size=2048, force_pot=True)

    # Profile-agnostic fallback (no profile, or an unknown one). Unbudgeted: with no
    # target named, any ceiling would be a guess presented as a recommendation.
    DEFAULT = OutputTemplate(default=OutputSpec("png", 8), overrides=dict(_PRECISION_16))

    # Built-in per-profile templates (read-only tier). Populated below the class so
    # the catalogue can be assembled with the class's own ``_build`` helper.
    BUILTIN: Dict[str, OutputTemplate] = {}

    @classmethod
    def _build(
        cls,
        default_ext: str,
        normal_ext: Optional[str] = None,
        budget: Optional[DeliveryBudget] = None,
    ) -> OutputTemplate:
        """Build a template: ``default_ext`` for everything, 16-bit for height-like
        maps, an optional distinct container for normal maps, and the profile's
        advisory :class:`DeliveryBudget` (default: unbudgeted)."""
        overrides: Dict[str, OutputSpec] = dict(cls._PRECISION_16)
        if normal_ext:
            for n in cls._NORMAL_TYPES:
                overrides[n] = OutputSpec(normal_ext, 8)
        return OutputTemplate(
            default=OutputSpec(default_ext, 8),
            overrides=overrides,
            budget=budget or DeliveryBudget(),
        )

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

    @classmethod
    def budget(cls, profile: Optional[str]) -> DeliveryBudget:
        """Return the advisory :class:`DeliveryBudget` for *profile*.

        Unlike :meth:`resolve`, this is not per map type — see
        :class:`DeliveryBudget`. An unknown or absent profile yields the
        unbudgeted default rather than raising, so callers can pass a profile
        through without branching on whether one was set.
        """
        return cls.get(profile).budget

    # ------------------------------------------------------------------
    # Selection SSoT — what a UI offers, and what a selection resolves to.
    #
    # Six panels across four packages (mayatk/blendertk game_shader,
    # blendertk mat_updater + scene_exporter, the compositor and converter)
    # each rebuilt this list and re-implemented the sentinel semantics. The
    # copies had already drifted: some spell the defer-to-template option
    # "Profile default", the converter spells its keep-the-source-container
    # option "Original", and each re-derived "empty means defer" inline. A
    # profile list is a property of the catalogue, so it belongs here with
    # the catalogue — the panels keep only their widget wiring.
    # ------------------------------------------------------------------

    #: Label for "let the selected profile pick each map's container".
    PROFILE_DEFAULT_LABEL = "Profile default"
    #: Label for "keep whatever container each source already has".
    ORIGINAL_LABEL = "Original"

    @classmethod
    def profile_choices(cls) -> List[Tuple[str, str]]:
        """``(name, description)`` for every selectable workflow profile.

        Ordered as the registry declares them. The description is the tooltip
        every panel shows — sourced from ``MapRegistry``'s workflow settings so
        the answer to "which one is right for me" is written once.

        Returns:
            list[tuple[str, str]]: ``(profile_name, description)`` pairs;
            description is ``""`` when the profile declares none.
        """
        return [
            (name, (preset or {}).get("description", ""))
            for name, preset in MapRegistry().get_workflow_presets().items()
        ]

    @classmethod
    def format_choices(
        cls,
        sentinel: Optional[str] = PROFILE_DEFAULT_LABEL,
        writable: Optional[Tuple[str, ...]] = None,
        sentinel_first: bool = False,
    ) -> List[Tuple[str, str]]:
        """``(label, value)`` for an output-container combo.

        Parameters:
            sentinel: Label for the deferring option —
                :attr:`PROFILE_DEFAULT_LABEL` (defer to the profile's template),
                :attr:`ORIGINAL_LABEL` (keep each source's container), or None
                for concrete formats only. Its value is always ``""``, which is
                what :meth:`resolve_selection` reads as "nothing forced".
            writable: Override the container list (defaults to
                ``ImgUtils.writable``). Injectable so a caller with a narrower
                target — a container set a specific exporter accepts — is not
                forced to rebuild the sentinel rule.
            sentinel_first: Put the sentinel at index 0 instead of last.
                **Position is a compatibility contract, not a style choice:**
                panels persist a combobox by index, so moving the sentinel
                silently re-points every saved selection (a user on "PNG" comes
                back to "JPG"). Existing panels therefore keep the order they
                shipped with — the converter's "Original" leads, game_shader's
                "Profile default" trails — and new callers should take the
                default and append.

        Returns:
            list[tuple[str, str]]: ``(label, value)`` pairs.
        """
        from pythontk.img_utils._img_utils import ImgUtils

        exts = ImgUtils.writable if writable is None else writable
        choices = [(ext.upper(), ext) for ext in exts]
        if sentinel:
            if sentinel_first:
                choices.insert(0, (sentinel, ""))
            else:
                choices.append((sentinel, ""))
        return choices

    @classmethod
    def resolve_selection(
        cls, profile: Optional[str], ext: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Turn a (profile, container) UI selection into call arguments.

        Encodes the one rule the copies each reimplemented: **a concrete
        container outranks the profile's template.** Naming an extension means
        "write everything as this", so the profile must not also drive per-map
        containers — but it still supplies the budget, which is why the profile
        is returned rather than dropped.

        Parameters:
            profile: Selected profile name, or None/"" for none.
            ext: Selected container (``""`` / None = the sentinel was chosen).

        Returns:
            tuple: ``(output_profile, output_type)`` as
            :meth:`~pythontk.core_utils.engines.textures.map_optimizer.MapOptimizer.optimize_map`
            takes them. ``output_type`` is None when the profile (or the
            source) should decide.
        """
        ext = (ext or "").lower().lstrip(".")
        return (profile or None), (ext or None)


# Built-in catalogue — engine-import oriented: correct *uncompressed* source per map.
# Assembled here (post-class) so it can use ``OutputTemplates._build``. Tune here or,
# later, via an editable PresetStore layer.
#
# Containers stay uncompressed across the board *including* the web profile: the real
# web/XR saving is KTX2 + Basis supercompression (KHR_texture_basisu), not PNG->JPG, and
# defaulting to a lossy container would look like that win while delivering a fraction
# of it — with no way for a caller to opt back out per map.
#
# KTX2 delivery is therefore an explicit *step*, not a template default: pass
# ``output_type="ktx2"`` to ``MapOptimizer.optimize_map`` (per-map Basis codecs are
# derived — see ``OutputSpec.compression``) for loose files, or
# ``MeshConvert.optimize_glb_textures(image_format="KTX2")`` for a packaged GLB. The
# uncompressed template output remains the correct *source* for both.
#
# Lossy delivery is therefore opt-in, never a default: a caller asks for it explicitly
# via ``MapOptimizer.optimize_map(lossy_quality=...)`` (or, later, a preset layer setting
# ``OutputSpec.quality``), and either route is filtered per map type by
# ``MapRegistry.is_lossy_safe`` so the maps that cannot survive a lossy codec — normals,
# ORM/MSAO, every linear channel — are refused rather than quietly degraded.
OutputTemplates.BUILTIN = {
    WF.STD: OutputTemplates._build("png", budget=OutputTemplates._BUDGET_NONE),
    WF.URP: OutputTemplates._build("png", budget=OutputTemplates._BUDGET_MOBILE),
    WF.HDRP: OutputTemplates._build("png", budget=OutputTemplates._BUDGET_REALTIME),
    WF.GODOT: OutputTemplates._build("png", budget=OutputTemplates._BUDGET_MOBILE),
    # glTF references PNG/JPG. Also the WebXR delivery format — hence the web budget.
    WF.GLTF: OutputTemplates._build("png", budget=OutputTemplates._BUDGET_WEB),
    WF.UE: OutputTemplates._build(  # UE commonly prefers TGA
        "tga", normal_ext="tga", budget=OutputTemplates._BUDGET_REALTIME
    ),
    # Spec/Gloss is a conversion source, not a delivery target — unbudgeted.
    WF.SPEC: OutputTemplates._build("png", budget=OutputTemplates._BUDGET_NONE),
}
