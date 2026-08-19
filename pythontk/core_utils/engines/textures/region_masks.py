# !/usr/bin/python
# coding=utf-8
"""Region-mask engine — named face-group masks that gate texture regions at runtime.

The SSoT both DCC emissive-group tools (mayatk / blendertk) and the Unity
importer templates (unitytk) share so a group authored in a DCC resolves to
the same channel slot everywhere. A *region group* is a named set of faces
that toggles as a unit; the engine models how group membership is encoded
for the game engine and produces the artifacts:

- ``RegionGroup`` — one named group: stable ``slot`` (its encoding channel)
  and a ``default`` weight.
- ``RegionMaskManifest`` — the wire schema (JSON). Rides to Unity either as
  a ``data_export`` FBX user property (vertex-color encoding) or as a
  sidecar next to the mask texture (channels encoding).
- ``RegionMaskPacker`` — rasterizes each group's UV triangles into its slot
  channel of an RGBA mask texture (channels encoding only; vertex-color
  encoding needs no image work).

Encodings (``RegionMaskManifest.encoding``):

- ``vertex-color`` — membership rides in a mesh color set (one group per
  RGBA channel). No textures; the DCC writes per-face-vertex colors.
- ``channels`` — membership rasterized into an RGBA mask texture (one group
  per channel). Needed when emissive detail is painted sub-face.
- ``id`` — reserved (per-pixel group index + control LUT); not produced yet.

Both live encodings share the 4-slot ceiling: slot N gates shader channel N
(``dot(mask, weights)``), so slots are a *contract* — reassigning one breaks
every downstream scene keyed against it. The DCC layer persists assignments;
the engine only validates them.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

# Optional imaging deps: only the packer (rasterize/write/preview) needs
# them. The manifest model is pure JSON and must resolve without either —
# vertex-color encoding never touches an image (e.g. vanilla Blender's
# Python has neither numpy-fed PIL paths nor Pillow).
try:
    import numpy as np
except ImportError as error:
    logging.getLogger(__name__).debug(f"# ImportError: {__file__}\n\t{error}")
    np = None  # type: ignore
try:
    from PIL import Image
except ImportError as error:
    logging.getLogger(__name__).debug(f"# ImportError: {__file__}\n\t{error}")
    Image = None  # type: ignore
import pythontk as ptk

# Encoding identifiers (manifest wire values).
ENCODING_VERTEX_COLOR = "vertex-color"
ENCODING_CHANNELS = "channels"
ENCODING_ID = "id"  # reserved

#: slot index -> mask channel. The ceiling for both live encodings.
SLOT_CHANNELS = ("R", "G", "B", "A")


@dataclass
class RegionGroup:
    """One named region group.

    ``slot`` is the group's channel in the encoding (0=R … 3=A) and is a
    stable contract — the DCC layer assigns it once and never reshuffles.
    ``default`` is the weight a consumer applies when nothing overrides it
    (1.0 = on, matching an all-on emissive bake). ``attr``, when set, names
    the DCC scene attribute whose animation curve carries this group's keyed
    weight (the *keyable weights* opt-in) — consumers that find a matching
    animated custom property rebind it to their live weight.
    """

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
    def coerce(cls, group: Union["RegionGroup", dict]) -> "RegionGroup":
        """Accept a ``RegionGroup`` or its plain-dict form."""
        if isinstance(group, cls):
            return group
        return cls(**group)


@dataclass
class RegionMaskManifest:
    """The wire schema joining DCC-authored groups to their game-engine consumer.

    Producer: the DCC bake step. Consumers: the Unity importer template
    (``EmissiveGroupController``) and re-imports of the mask texture.
    ``schema`` is an integer version consumers compare against the version
    they support (warn-and-best-effort on newer payloads).
    """

    SCHEMA_VERSION = 1

    groups: List[RegionGroup] = field(default_factory=list)
    encoding: str = ENCODING_VERTEX_COLOR
    schema: int = SCHEMA_VERSION
    # vertex-color encoding:
    color_set: Optional[str] = None
    # channels encoding:
    mask: Optional[str] = None  # mask texture filename (no path)
    uv_channel: Optional[int] = None
    resolution: Optional[int] = None

    _ENCODINGS = (ENCODING_VERTEX_COLOR, ENCODING_CHANNELS, ENCODING_ID)

    def __post_init__(self):
        if self.encoding not in self._ENCODINGS:
            raise ValueError(
                f"Unknown encoding {self.encoding!r}; expected one of {self._ENCODINGS}."
            )
        self.groups = [RegionGroup.coerce(g) for g in self.groups]
        slots = [g.slot for g in self.groups]
        if len(set(slots)) != len(slots):
            raise ValueError(f"Duplicate slot assignment in groups: {slots}")
        names = [g.name for g in self.groups]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate group name in groups: {names}")

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    @classmethod
    def vertex_color(
        cls,
        groups: Sequence[Union[RegionGroup, dict]],
        color_set: str = "emissiveGroups",
    ) -> "RegionMaskManifest":
        """Manifest for membership riding in a mesh color set."""
        return cls(groups=list(groups), encoding=ENCODING_VERTEX_COLOR, color_set=color_set)

    @classmethod
    def channels(
        cls,
        groups: Sequence[Union[RegionGroup, dict]],
        mask: str,
        resolution: int,
        uv_channel: int = 0,
    ) -> "RegionMaskManifest":
        """Manifest for membership rasterized into an RGBA mask texture."""
        return cls(
            groups=list(groups),
            encoding=ENCODING_CHANNELS,
            mask=os.path.basename(mask),
            resolution=int(resolution),
            uv_channel=int(uv_channel),
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Wire form: encoding-irrelevant fields are omitted, not null."""
        data = {
            "schema": self.schema,
            "encoding": self.encoding,
            "groups": [g.to_dict() for g in self.groups],
        }
        if self.encoding == ENCODING_VERTEX_COLOR:
            data["color_set"] = self.color_set or "emissiveGroups"
        elif self.encoding == ENCODING_CHANNELS:
            data["mask"] = self.mask
            data["uv_channel"] = self.uv_channel or 0
            data["resolution"] = self.resolution
        return data

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "RegionMaskManifest":
        known = {
            "schema",
            "encoding",
            "groups",
            "color_set",
            "mask",
            "uv_channel",
            "resolution",
        }
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


class RegionGroupRegistry:
    """Slot-assignment model for region groups — persistence injected.

    The DCC-agnostic half of the Emissive Groups tools: mayatk and blendertk
    hold identical *membership* mechanics (Maya sets vs. Blender face
    attributes) but identical *bookkeeping* — which slot a group owns, which
    slots are retired, the default weights, and the manifest that bookkeeping
    produces. That bookkeeping is plain dict work with no DCC call in it, so
    it lives here once rather than drifting in two copies.

    Persistence is a ``(load, save)`` callable pair over a single JSON string
    (each DCC's scene carrier — see their ``node_utils.data_nodes``):
    ``load() -> str | None`` and ``save(str) -> None``. Every operation is a
    read-modify-write, so the scene stays the single source of truth and two
    tools can't hold divergent copies.

    **Slots are a contract.** A slot is assigned once and *retires* when its
    group is removed — never silently reused, because engine-side scenes and
    animations bind to the slot index. :meth:`compact` is the explicit,
    binding-breaking reclaim.
    """

    MAX_SLOTS = len(SLOT_CHANNELS)
    SCHEMA_VERSION = RegionMaskManifest.SCHEMA_VERSION
    DEFAULT_COLOR_SET = "emissiveGroups"

    def __init__(self, load, save, *, max_slots: Optional[int] = None, logger=None):
        self._load = load
        self._save = save
        self.max_slots = int(max_slots or self.MAX_SLOTS)
        self._logger = logger

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def empty(self) -> dict:
        return {
            "schema": self.SCHEMA_VERSION,
            "encoding": ENCODING_VERTEX_COLOR,
            "groups": {},
            "retired_slots": [],
        }

    def read(self) -> dict:
        """The stored registry, or a fresh empty one (never raises)."""
        raw = self._load()
        if not raw:
            return self.empty()
        try:
            data = json.loads(raw)
        except ValueError:
            if self._logger is not None:
                self._logger.warning(
                    "Region-group registry is not valid JSON; resetting."
                )
            return self.empty()
        data.setdefault("groups", {})
        data.setdefault("retired_slots", [])
        data.setdefault("encoding", ENCODING_VERTEX_COLOR)
        return data

    def write(self, registry: dict) -> None:
        """Persist *registry*, or clear the channel when it holds nothing.

        A registry with no groups and no retired slots is indistinguishable
        from "this tool was never used", so it is cleared rather than stored
        — the scene must not accumulate an empty carrier channel. The clear
        is skipped when nothing was stored, so reading alone never writes.
        """
        if not registry.get("groups") and not registry.get("retired_slots"):
            if self._load() is not None:
                self._save("")
            return
        self._save(json.dumps(registry))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @staticmethod
    def sanitize(name: str) -> str:
        """Coerce *name* to a DCC-safe identifier, or raise."""
        clean = re.sub(r"[^A-Za-z0-9_]", "_", str(name).strip())
        if not clean or clean[0].isdigit():
            raise ValueError(f"Invalid group name {name!r}.")
        return clean

    def groups(self, registry: Optional[dict] = None) -> List[dict]:
        """Groups in slot order as ``{"name", "slot", "default"[, "attr"]}``
        dicts — the shape :class:`RegionMaskManifest` and
        :meth:`RegionMaskPacker.add_group` both consume. ``attr`` appears only
        for groups whose weight is keyable (see :meth:`set_attr`)."""
        registry = self.read() if registry is None else registry
        out = []
        for name, data in sorted(
            registry["groups"].items(), key=lambda kv: kv[1]["slot"]
        ):
            entry = {
                "name": name,
                "slot": data["slot"],
                "default": data.get("default", 1.0),
            }
            if data.get("attr"):
                entry["attr"] = data["attr"]
            out.append(entry)
        return out

    def next_slot(self, registry: dict) -> int:
        """Lowest slot that is neither used nor retired."""
        used = {g["slot"] for g in registry["groups"].values()}
        used |= set(registry.get("retired_slots", []))
        for slot in range(self.max_slots):
            if slot not in used:
                return slot
        raise ValueError(
            f"All {self.max_slots} slots are used or retired. Remove a group "
            "and run compact_slots() (breaks existing engine bindings), or "
            "use the channels encoding on a second mask page."
        )

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add(self, name: str, default: float = 1.0) -> Tuple[int, bool]:
        """Register *name* (no-op for an existing group).

        Returns:
            ``(slot, is_new)`` — the caller writes membership either way.
        """
        registry = self.read()
        existing = registry["groups"].get(name)
        if existing is not None:
            return existing["slot"], False
        slot = self.next_slot(registry)
        registry["groups"][name] = {"slot": slot, "default": float(default)}
        self.write(registry)
        return slot, True

    def remove(self, name: str) -> Optional[int]:
        """Drop *name* and retire its slot. Returns the retired slot, if any."""
        registry = self.read()
        data = registry["groups"].pop(name, None)
        if data is not None:
            retired = set(registry.get("retired_slots", []))
            retired.add(data["slot"])
            registry["retired_slots"] = sorted(retired)
        self.write(registry)
        return data["slot"] if data else None

    def set_default(self, name: str, default: float) -> float:
        """Set a group's default gate weight (clamped 0-1). Returns it."""
        registry = self.read()
        if name not in registry["groups"]:
            raise ValueError(f"Unknown group {name!r}.")
        value = max(0.0, min(1.0, float(default)))
        registry["groups"][name]["default"] = value
        self.write(registry)
        return value

    def set_attr(self, name: str, attr: Optional[str]) -> None:
        """Record — or clear, with ``attr=None`` — the DCC animation attribute
        carrying a group's keyed weight (the *keyable weights* opt-in). The
        attr name rides into the manifest so consumers can rebind the matching
        animated custom property to their live weight.
        """
        registry = self.read()
        if name not in registry["groups"]:
            raise ValueError(f"Unknown group {name!r}.")
        if attr:
            registry["groups"][name]["attr"] = str(attr)
        else:
            registry["groups"][name].pop("attr", None)
        self.write(registry)

    def compact(self) -> List[int]:
        """Reclaim retired slots. Returns the reclaimed indices.

        Explicitly binding-breaking: any engine scene keyed against a
        previously-exported slot layout must be re-wired after the next
        bake/export. Callers should say so out loud.
        """
        registry = self.read()
        reclaimed = list(registry.get("retired_slots", []))
        registry["retired_slots"] = []
        self.write(registry)
        return reclaimed

    def set_encoding(self, encoding: str, **info) -> None:
        """Record the encoding the last bake produced (plus mask info)."""
        registry = self.read()
        registry["encoding"] = encoding
        registry.update({k: v for k, v in info.items() if v is not None})
        self.write(registry)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def manifest(
        self, color_set: Optional[str] = None
    ) -> Optional[RegionMaskManifest]:
        """The manifest for the current registry, or None when it has no groups."""
        registry = self.read()
        groups = self.groups(registry)
        if not groups:
            return None
        if registry.get("encoding") == ENCODING_CHANNELS:
            return RegionMaskManifest.channels(
                groups,
                mask=registry.get("mask") or "",
                resolution=registry.get("resolution") or 512,
                uv_channel=registry.get("uv_channel", 0),
            )
        return RegionMaskManifest.vertex_color(
            groups, color_set=color_set or self.DEFAULT_COLOR_SET
        )


class _RegionMaskPackerInternal:
    """Implementation detail base for :class:`RegionMaskPacker`."""

    @staticmethod
    def _coverage(triangles, resolution: int, supersample: int) -> "np.ndarray":
        """(H, W) uint8 anti-aliased coverage for one group's UV triangles."""
        return ptk.ImgUtils.rasterize_uv_triangles(
            triangles, size=resolution, supersample=supersample
        )

    @staticmethod
    def _dilate(channel: "np.ndarray", padding_px: int) -> "np.ndarray":
        """Extend coverage outward so bilinear/mip sampling never reads a
        zero gutter inside a lit region's footprint. Mirrors bake padding."""
        if padding_px <= 0:
            return channel
        grown = ptk.ImgUtils.dilate_image(
            channel, mask=channel > 0, iterations=padding_px
        )
        return np.clip(np.rint(grown), 0, 255).astype(np.uint8).reshape(channel.shape)

    @staticmethod
    def _erode1(cover: "np.ndarray") -> "np.ndarray":
        """Shrink a boolean coverage by one 4-connected pixel ring.

        Overlap detection intersects *eroded* covers so groups that merely
        share a UV seam edge (adjacent shells, split panels) don't trip the
        warning on the 1px boundary line both rasterize onto.
        """
        p = np.pad(cover, 1, constant_values=False)
        return (
            cover
            & p[:-2, 1:-1]
            & p[2:, 1:-1]
            & p[1:-1, :-2]
            & p[1:-1, 2:]
        )

    @staticmethod
    def _uv_bounds_warning(name: str, triangles) -> Optional[str]:
        tris = np.asarray(triangles, dtype=float)
        if tris.size and ((tris < -1e-4).any() or (tris > 1.0 + 1e-4).any()):
            return (
                f"Group {name!r} has UVs outside 0-1; coverage is clipped to "
                "the image edge (tile-crossing shells won't gate correctly)."
            )
        return None


class RegionMaskPacker(ptk.LoggingMixin, _RegionMaskPackerInternal):
    """Rasterize named UV face-groups into a channel-packed RGBA mask texture.

    Channels-encoding producer: each added group fills its slot channel
    (0=R … 3=A) with anti-aliased, edge-padded coverage; :meth:`write` saves
    the mask plus its :class:`RegionMaskManifest` sidecar. The shader gate is
    ``dot(mask, weights)``, so anything the rasterizer leaves at 0 never
    glows and coverage edges fade the gate rather than corrupt it.

    A packer instance is single-use per bake: add groups, then
    ``rasterize``/``write``. UV triangles come in as plain ``(N, 3, 2)``
    arrays in normalized UV space (V up) — the DCC layer owns harvesting.
    """

    def __init__(
        self,
        resolution: int = 512,
        padding_px: int = 4,
        supersample: int = 4,
        log_level: str = "WARNING",
    ):
        self.logger.setLevel(log_level)
        self.resolution = int(resolution)
        self.padding_px = int(padding_px)
        self.supersample = int(supersample)
        self._groups: List[RegionGroup] = []
        self._triangles: Dict[str, "np.ndarray"] = {}

    # ------------------------------------------------------------------
    # Group intake
    # ------------------------------------------------------------------

    @property
    def groups(self) -> List[RegionGroup]:
        return list(self._groups)

    def add_group(
        self,
        name: str,
        uv_triangles,
        *,
        slot: Optional[int] = None,
        default: float = 1.0,
        attr: Optional[str] = None,
    ) -> RegionGroup:
        """Register a group and its UV coverage.

        Parameters:
            name: Group name (unique per packer).
            uv_triangles: (N, 3, 2) array-like of triangle UVs (V up).
            slot: Channel slot 0-3. Defaults to the lowest unused slot —
                pass the persisted slot explicitly for stable re-bakes.
            default: Default gate weight for consumers (1.0 = on).
            attr: DCC animation attribute carrying the group's keyed weight,
                if any — rides into the manifest sidecar unchanged.

        Returns:
            The registered :class:`RegionGroup`.
        """
        if any(g.name == name for g in self._groups):
            raise ValueError(f"Group {name!r} already added.")
        used = {g.slot for g in self._groups}
        if slot is None:
            slot = next(
                (i for i in range(len(SLOT_CHANNELS)) if i not in used), None
            )
            if slot is None:
                raise ValueError(
                    f"All {len(SLOT_CHANNELS)} slots used; channels encoding "
                    "caps at one group per channel."
                )
        if slot in used:
            raise ValueError(f"Slot {slot} already used (adding group {name!r}).")
        tris = np.asarray(uv_triangles, dtype=float).reshape(-1, 3, 2)
        if not tris.size:
            raise ValueError(f"Group {name!r} has no UV triangles.")
        group = RegionGroup(name=name, slot=slot, default=default, attr=attr)
        self._groups.append(group)
        self._triangles[name] = tris
        return group

    # ------------------------------------------------------------------
    # Validation / rasterization
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Non-fatal authoring warnings (hard errors raise in ``add_group``).

        Returns:
            Warning strings: out-of-range UVs and inter-group texel overlap
            (overlapping texels glow when *either* group is on).
        """
        warnings: List[str] = []
        for group in self._groups:
            msg = self._uv_bounds_warning(group.name, self._triangles[group.name])
            if msg:
                warnings.append(msg)
        if len(self._groups) > 1:
            # Overlap on raw (pre-padding) coverage — padding gutters are
            # expected to touch and never carry emissive texels. Covers are
            # eroded 1px so seam-sharing neighbors don't false-positive.
            covers = {
                g.name: self._erode1(
                    self._coverage(self._triangles[g.name], self.resolution, 1) > 0
                )
                for g in self._groups
            }
            stack = np.stack(list(covers.values()))
            multi = (stack.sum(axis=0) > 1).sum()
            lit = (stack.any(axis=0)).sum()
            if multi:
                pct = 100.0 * multi / max(lit, 1)
                warnings.append(
                    f"{multi} texels ({pct:.1f}% of covered area) belong to "
                    "more than one group; they glow when either group is on."
                )
        for msg in warnings:
            self.logger.warning(msg)
        return warnings

    def rasterize(self) -> "np.ndarray":
        """Fill each group's slot channel; returns (H, W, 4) uint8 RGBA."""
        if np is None:
            raise RuntimeError(
                "numpy is required for mask rasterization (channels encoding)."
            )
        if not self._groups:
            raise ValueError("No groups added.")
        out = np.zeros((self.resolution, self.resolution, 4), dtype=np.uint8)
        for group in self._groups:
            cover = self._coverage(
                self._triangles[group.name], self.resolution, self.supersample
            )
            out[..., group.slot] = self._dilate(cover, self.padding_px)
        return out

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def write(
        self,
        mask_path: str,
        manifest_path: Optional[str] = None,
        uv_channel: int = 0,
    ) -> RegionMaskManifest:
        """Save the packed mask texture and its manifest sidecar.

        Parameters:
            mask_path: Output image path (extension selects the format).
            manifest_path: Manifest JSON path; defaults to the mask path
                with a ``.json`` extension.
            uv_channel: Game-engine UV channel the mask is sampled with.

        Returns:
            The written :class:`RegionMaskManifest`.
        """
        if Image is None:
            raise RuntimeError(
                "Pillow is required to write mask textures (channels encoding)."
            )
        arr = self.rasterize()
        Image.fromarray(arr, mode="RGBA").save(mask_path)
        manifest = RegionMaskManifest.channels(
            self._groups, mask_path, self.resolution, uv_channel=uv_channel
        )
        if manifest_path is None:
            manifest_path = os.path.splitext(mask_path)[0] + ".json"
        manifest.save(manifest_path)
        self.logger.info(
            f"Wrote {os.path.basename(mask_path)} + "
            f"{os.path.basename(manifest_path)} "
            f"({len(self._groups)} group(s), {self.resolution}px)."
        )
        return manifest

    def preview(
        self,
        emissive,
        weights: Optional[Dict[str, float]] = None,
    ) -> "Image.Image":
        """Preview which texels glow for a given weight combination.

        Applies the runtime gate so a DCC panel can show the exact engine
        result without a viewport shader. The formula is the Python twin of
        ``EmissiveGroups.hlsl``'s ``EmissiveGroupGate`` — including the
        ``(1 - membership)`` term that keeps texels belonging to NO group lit
        (see that file for the full rationale). Keep the two in step.

        Parameters:
            emissive: All-on emissive map (path or PIL image).
            weights: ``{group_name: weight}``; unlisted groups use their
                ``default``.

        Returns:
            RGB PIL image at the packer resolution.
        """
        if Image is None:
            raise RuntimeError("Pillow is required for previews.")
        weights = weights or {}
        mask = self.rasterize().astype(np.float32) / 255.0
        wvec = np.zeros(4, dtype=np.float32)
        for group in self._groups:
            wvec[group.slot] = float(weights.get(group.name, group.default))
        membership = np.clip(mask.sum(axis=2), 0.0, 1.0)
        gate = np.clip((mask * wvec).sum(axis=2) + (1.0 - membership), 0.0, 1.0)
        em = ptk.ImgUtils.ensure_image(emissive).convert("RGB")
        if em.size != (self.resolution, self.resolution):
            em = em.resize((self.resolution, self.resolution))
        out = np.asarray(em).astype(np.float32) * gate[..., None]
        return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")
