# !/usr/bin/python
# coding=utf-8
"""Write authored per-object material ramps into a GLB as ``KHR_animation_pointer`` channels.

glTF animates four things -- translation, rotation, scale and morph weights --
and a material property is none of them. The Khronos answer is
``KHR_animation_pointer``, which lets a channel target any property by JSON
pointer. Writing it is what makes a fade (or a highlight) part of the
DELIVERABLE rather than a note attached to it: a viewer that implements the
extension plays it with nothing explained to it, and one that does not still
loads the file (it is declared in ``extensionsUsed``, never
``extensionsRequired``).

ONE pass, a TABLE of channels. Each :class:`PointerChannel` row says which
``visibility_tracks`` key carries its ramp, which material property it targets,
how many components that property has, whether the clone needs ``alphaMode
BLEND`` (only alpha does -- an additive emissive sorts as opaque), and how a
ramp sample composes with the material's own value. The pass isolates a node's
materials ONCE for every channel it carries, and guards per pointer TARGET
rather than per file -- so a channel added after another was written is still
written, and a second run of the same set writes nothing.

three.js does not implement the extension -- checked against its
``GLTFLoader``, which is the runtime this pipeline's own preview uses; the
loader simply skips a channel with no ``target.node``. So the preview page
carries a small shim that reads these same channels back out of the file and
hands them to the mixer as ordinary keyframe tracks. ONE statement of each
ramp, the glTF's; no side block, no second encoding, and the day three.js
ships support the shim deletes without touching the exporter.

Materials are ISOLATED per animated subtree before being animated, because a
material is shared by whatever samples it: animating the one a node happens
to use would animate every other object using it too. A material already used
by nothing outside the subtree is animated in place; one that is shared is
cloned. The clone costs no binary -- a glTF material is JSON, and so is the
mesh copy taken when a mesh is shared with something outside the subtree.
"""

from __future__ import annotations

import copy
import logging
import struct
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["GlbFades", "PointerChannel", "CHANNELS"]

logger = logging.getLogger(__name__)

#: The extension this pass writes. Declared in ``extensionsUsed`` only: a
#: viewer without it must still open the file and simply not animate.
EXTENSION = "KHR_animation_pointer"
#: The alpha channel's target, kept under its historical name. The alpha is the
#: fourth component of the base colour factor, and glTF has no pointer to a
#: single component -- so the whole VEC4 is animated, holding the material's
#: own RGB constant.
POINTER = "/materials/{index}/pbrMetallicRoughness/baseColorFactor"

Rgb = Sequence[float]


def _opacity_values(base: List[float], sample: float, _color: Rgb) -> List[float]:
    """Alpha rides the fourth lane; the material's own RGB is carried through."""
    return [base[0], base[1], base[2], sample]


def _highlight_values(base: List[float], sample: float, color: Rgb) -> List[float]:
    """Additive over the material's own emissive, clamped to glTF's LDR factor.

    An LED panel that is also highlighted keeps glowing at intensity 0, and a
    white surface highlighted blue reads blue rather than replacing its albedo.
    """
    return [min(1.0, max(0.0, base[i] + color[i] * sample)) for i in range(3)]


@dataclass(frozen=True)
class PointerChannel:
    """One animatable material property and how a published ramp reaches it.

    Attributes:
        name: The ``visibility_tracks`` key carrying the ramp (``[[frame, v]]``).
        pointer: JSON pointer template with ``{index}`` for the material.
        property: Path of the material property inside the material dict, as
            ``("pbrMetallicRoughness", "baseColorFactor")`` or ``("emissiveFactor",)``.
        default: The property's per-spec default when the material omits it.
        blend: Whether an animated clone must switch to ``alphaMode BLEND``.
        color_key: Optional sibling track key carrying a per-node RGB.
        values: ``(base, sample, color) -> components`` for one key.
    """

    name: str
    pointer: str
    property: Tuple[str, ...]
    default: Tuple[float, ...]
    blend: bool
    color_key: Optional[str]
    values: Callable[[List[float], float, Rgb], List[float]]

    @property
    def components(self) -> int:
        return len(self.default)

    @property
    def accessor_type(self) -> str:
        return {3: "VEC3", 4: "VEC4"}[self.components]

    def base(self, gltf: Dict[str, Any], index: int) -> List[float]:
        """The material's own value for this property, defaulted per spec."""
        holder: Dict[str, Any] = (gltf.get("materials") or [])[index]
        for key in self.property[:-1]:
            holder = holder.setdefault(key, {})
        value = holder.get(self.property[-1])
        if not (isinstance(value, list) and len(value) == self.components):
            value = list(self.default)
            holder[self.property[-1]] = value
        return [float(v) for v in value]


#: The channel table. ``opacity`` is the fade; ``highlight`` an additive
#: emissive intensity with a per-node colour. Order is write order.
CHANNELS: Dict[str, PointerChannel] = {
    "opacity": PointerChannel(
        name="opacity",
        pointer=POINTER,
        property=("pbrMetallicRoughness", "baseColorFactor"),
        default=(1.0, 1.0, 1.0, 1.0),
        blend=True,
        color_key=None,
        values=_opacity_values,
    ),
    "highlight": PointerChannel(
        name="highlight",
        pointer="/materials/{index}/emissiveFactor",
        property=("emissiveFactor",),
        default=(0.0, 0.0, 0.0),
        blend=False,
        color_key="highlight_color",
        values=_highlight_values,
    ),
}

#: A highlight with no published colour is white: the ramp still reads.
DEFAULT_COLOR: Tuple[float, float, float] = (1.0, 1.0, 1.0)


class _GlbFadesInternal:
    """Internal helpers for :class:`GlbFades`."""

    @staticmethod
    def _subtree(gltf: Dict[str, Any], roots: Sequence[int]) -> List[int]:
        """Every node index at or under *roots* (cycle-safe)."""
        nodes = gltf.get("nodes") or []
        seen: set = set()
        stack = list(roots)
        while stack:
            index = stack.pop()
            if index in seen or not 0 <= index < len(nodes):
                continue
            seen.add(index)
            stack.extend((nodes[index] or {}).get("children") or [])
        return sorted(seen)

    @staticmethod
    def _mesh_users(gltf: Dict[str, Any]) -> Dict[int, List[int]]:
        """``mesh index -> [node index, ...]``, so a shared mesh is recognisable."""
        users: Dict[int, List[int]] = {}
        for index, node in enumerate(gltf.get("nodes") or []):
            mesh = (node or {}).get("mesh")
            if isinstance(mesh, int):
                users.setdefault(mesh, []).append(index)
        return users

    @staticmethod
    def _material_users(
        gltf: Dict[str, Any], users: Dict[int, List[int]]
    ) -> Dict[int, Set[int]]:
        """``material index -> {node index, ...}`` across every primitive."""
        by_material: Dict[int, Set[int]] = {}
        for mesh_index, mesh in enumerate(gltf.get("meshes") or []):
            for primitive in (mesh or {}).get("primitives") or []:
                material = primitive.get("material")
                if isinstance(material, int):
                    by_material.setdefault(material, set()).update(
                        users.get(mesh_index, [])
                    )
        return by_material

    @classmethod
    def _isolate(
        cls,
        gltf: Dict[str, Any],
        nodes: Sequence[int],
        users: Dict[int, List[int]],
        blend: bool = True,
    ) -> List[int]:
        """Give *nodes* materials nothing outside them shares; return their indices.

        Two levels of copying, both JSON-only:

        * A MESH used by a node outside this subtree is duplicated first, so
          repointing its primitives cannot reach the outsider. The copy shares
          every accessor, so it adds no geometry.
        * A MATERIAL is cloned unless it is used by nothing outside AND already
          carries what the channel needs -- for alpha, a ``BLEND`` clone this
          pass made on an earlier run; for an additive channel, any exclusive
          material. The original is never mutated: the sidecar repairs it by
          name, and its authored ``alphaMode`` stays. This is what makes the
          pass idempotent and lets two channels on one node share one clone.

        *blend* switches the isolated materials to ``BLEND`` -- for alpha,
        because the ramp is continuous and ``MASK`` is binary at its cutoff
        (the pop this pass exists to replace). An additive channel leaves the
        mode alone: a blended surface sorts differently from an opaque one,
        and a highlight has no reason to pay that.
        """
        meshes = gltf.setdefault("meshes", [])
        materials = gltf.setdefault("materials", [])
        inside = set(nodes)
        material_users = cls._material_users(gltf, users)
        clones: Dict[int, int] = {}
        touched: List[int] = []
        repointed: set = set()
        for index in nodes:
            node = (gltf.get("nodes") or [])[index]
            mesh_index = node.get("mesh")
            if not isinstance(mesh_index, int) or not 0 <= mesh_index < len(meshes):
                continue
            if any(user not in inside for user in users.get(mesh_index, [])):
                meshes.append(copy.deepcopy(meshes[mesh_index]))
                mesh_index = len(meshes) - 1
                node["mesh"] = mesh_index
                users.setdefault(mesh_index, []).append(index)
            if mesh_index in repointed:
                # Instanced: shared with another node in this SAME subtree, so
                # its primitives already point at this call's clones. Walking
                # them again would read a CLONE as the source material and clone
                # that, stranding the first with no primitive while still
                # returning it -- and the caller writes one channel per returned
                # material. (A mesh copied just above is always a fresh index,
                # so it is never the one skipped here.)
                continue
            repointed.add(mesh_index)
            for primitive in meshes[mesh_index].get("primitives") or []:
                source = primitive.get("material")
                if not isinstance(source, int) or not 0 <= source < len(materials):
                    continue
                if source not in clones:
                    # In place only when the material is used by nothing
                    # outside AND already carries what the channel needs. For
                    # alpha that means it is already a BLEND clone (an earlier
                    # run's): the ORIGINAL keeps its authored mode untouched,
                    # because the sidecar repairs it by name and an authored
                    # MASK turned BLEND would pop at no cutoff at all. An
                    # additive channel mutates nothing authored, so any
                    # exclusive material will do.
                    exclusive = material_users.get(source, set()) <= inside
                    if exclusive and (
                        not blend or materials[source].get("alphaMode") == "BLEND"
                    ):
                        clones[source] = source
                    else:
                        clone = copy.deepcopy(materials[source])
                        # The NAME is kept, deliberately: glTF does not require
                        # unique material names, and the lightmap manifest
                        # (`lightmap_web.materials`) is keyed by name -- a
                        # renamed clone loses its lightmap in every reader that
                        # binds that way, the preview included.
                        materials.append(clone)
                        clones[source] = len(materials) - 1
                    if blend:
                        materials[clones[source]]["alphaMode"] = "BLEND"
                    touched.append(clones[source])
                primitive["material"] = clones[source]
        return touched

    @staticmethod
    def _existing_pointers(animations: Sequence[Dict[str, Any]]) -> Set[str]:
        """Every pointer target the file already animates."""
        found: Set[str] = set()
        for animation in animations:
            for channel in animation.get("channels") or []:
                target = channel.get("target") or {}
                if target.get("path") != "pointer":
                    continue
                pointer = ((target.get("extensions") or {}).get(EXTENSION) or {}).get(
                    "pointer"
                )
                if pointer:
                    found.add(str(pointer))
        return found

    @staticmethod
    def _per_clip(
        ramp: Sequence[Tuple[float, float]],
        animations: Sequence[Dict[str, Any]],
        windows: Dict[str, Tuple[float, float]],
        zeros: Dict[str, float],
        fps: float,
    ) -> Dict[str, Tuple[List[float], List[Any]]]:
        """``{clip: (times, samples)}`` -- the ramp cut to each clip that it moves in."""
        from pythontk.file_utils.mesh_convert.glb_clips import GlbClips

        per_clip: Dict[str, Tuple[List[float], List[Any]]] = {}
        for animation in animations:
            clip = str(animation.get("name") or "")
            window, zero = windows.get(clip), zeros.get(clip)
            if window is None or zero is None:
                continue
            lo = (window[0] - zero) / fps
            cut, samples = GlbClips._window(
                [(frame - zero) / fps for frame, _v in ramp],
                [(value,) for _f, value in ramp],
                "LINEAR",
                False,
                lo,
                (window[1] - zero) / fps,
                0.25 / fps,
            )
            # ``_window`` rebases onto the window it cut, which is right for a
            # clip BUILT to that window and wrong here: this channel joins a
            # clip that already has a zero, and the two are the same only when
            # the clip starts where its window does. The retained whole-timeline
            # stack is the case where they differ -- its zero is the timeline's,
            # so the ramp arrived shifted by the first shot's start frame
            # (measured: 7 frames).
            cut = [k + lo for k in cut]
            if len(cut) < 2 or len({round(s[0], 6) for s in samples}) < 2:
                continue  # this clip holds a constant value: nothing to animate
            per_clip[clip] = (cut, samples)
        return per_clip


class GlbFades(_GlbFadesInternal):
    """Publish authored material ramps as animated material channels."""

    CHANNELS = CHANNELS

    @classmethod
    def apply(
        cls,
        edit: Any,
        fades: Dict[str, Sequence[Sequence[float]]],
        windows: Dict[str, Tuple[float, float]],
        zeros: Dict[str, float],
        fps: float,
    ) -> Optional[Dict[str, Any]]:
        """Write one alpha channel per faded node per clip.

        The opacity-only entry point, kept for its callers;
        :meth:`apply_channels` is the general form.

        Parameters:
            edit: An open ``MeshConvert.GlbEdit``.
            fades: ``{node name: [[frame, alpha], ...]}`` -- the authored ramps.
            windows: ``{clip name: (start frame, end frame)}``.
            zeros: ``{clip name: authoring frame the clip puts at t=0}``.
            fps: The rate the frame numbers are quoted in.

        Returns:
            ``{"nodes", "materials", "channels"}``, or ``None`` when there was
            nothing to write.
        """
        return cls.apply_channels(edit, {"opacity": fades}, {}, windows, zeros, fps)

    @classmethod
    def apply_channels(
        cls,
        edit: Any,
        ramps: Dict[str, Dict[str, Sequence[Sequence[float]]]],
        colors: Dict[str, Dict[str, Rgb]],
        windows: Dict[str, Tuple[float, float]],
        zeros: Dict[str, float],
        fps: float,
    ) -> Optional[Dict[str, Any]]:
        """Write every channel of every node, per clip, in one pass.

        Parameters:
            edit: An open ``MeshConvert.GlbEdit``.
            ramps: ``{channel name: {node name: [[frame, value], ...]}}`` --
                channel names are :data:`CHANNELS` keys; unknown ones are skipped
                with a warning.
            colors: ``{channel name: {node name: (r, g, b)}}`` for channels
                whose row names a ``color_key``.
            windows: ``{clip name: (start frame, end frame)}``.
            zeros: ``{clip name: authoring frame the clip puts at t=0}``.
            fps: The rate the frame numbers are quoted in.

        Returns:
            ``{"nodes", "materials", "channels", "by_channel"}``, or ``None``
            when there was nothing to write.
        """
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        gltf = edit.gltf
        animations = gltf.get("animations") or []
        if not ramps or not animations or fps <= 0:
            return None

        unknown = sorted(set(ramps) - set(CHANNELS))
        if unknown:
            logger.warning(
                "Pointer channels: no row for %s -- those ramps ship as data only.",
                ", ".join(unknown),
            )

        by_name: Dict[str, List[int]] = {}
        for index, node in enumerate(gltf.get("nodes") or []):
            name = (node or {}).get("name")
            if name is not None:
                by_name.setdefault(str(name).split("|")[-1], []).append(index)

        # Which node moves inside which clip, per channel, resolved BEFORE
        # anything is cloned. Isolating a subtree can switch its materials to
        # BLEND, and a blended surface sorts differently from an opaque one --
        # so doing it for a ramp that turns out to hold a constant value in
        # every clip would change how the object renders in exchange for no
        # animation.
        missing: List[str] = []
        # node -> [(channel, per_clip)]
        resolved: Dict[str, List[Tuple[PointerChannel, Dict[str, Any]]]] = {}
        for channel_name, spec in CHANNELS.items():
            for name, keys in sorted((ramps.get(channel_name) or {}).items()):
                if name not in by_name:
                    missing.append(f"{name}.{channel_name}")
                    continue
                ramp = sorted((float(k[0]), float(k[1])) for k in keys)
                per_clip = cls._per_clip(ramp, animations, windows, zeros, fps)
                if per_clip:
                    resolved.setdefault(name, []).append((spec, per_clip))
        if missing:
            logger.warning(
                "Pointer channels: %d ramp(s) name nodes not in this GLB (%s) -- "
                "they ship as data only.",
                len(missing),
                ", ".join(sorted(missing)),
            )
        if not resolved:
            return None

        existing = cls._existing_pointers(animations)
        users = cls._mesh_users(gltf)
        payloads: List[bytes] = []
        slots: Dict[bytes, int] = {}

        def slot(values: Sequence[float]) -> int:
            raw = struct.pack(f"<{len(values)}f", *values)
            if raw not in slots:
                slots[raw] = len(payloads)
                payloads.append(raw)
            return slots[raw]

        by_clip: Dict[str, List[Dict[str, Any]]] = {}
        animated_nodes: Set[str] = set()
        by_channel: Dict[str, int] = {}
        for name, entries in resolved.items():
            # One isolation per node for every channel it carries; BLEND only
            # when one of those channels is alpha.
            materials = cls._isolate(
                gltf,
                cls._subtree(gltf, by_name[name]),
                users,
                blend=any(spec.blend for spec, _pc in entries),
            )
            if not materials:
                # A locator drives the ramp but carries no geometry of its own;
                # nothing under it renders, so there is nothing to animate.
                continue
            for spec, per_clip in entries:
                color = (colors.get(spec.name) or {}).get(name) or DEFAULT_COLOR
                for material in materials:
                    pointer = spec.pointer.format(index=material)
                    if pointer in existing:
                        # Already written on THIS target. Writing again would
                        # stack a second channel on the same property, which is
                        # undefined; the file's own channel is the statement.
                        continue
                    base = spec.base(gltf, material)
                    for clip, (cut, samples) in per_clip.items():
                        flat = [
                            c
                            for s in samples
                            for c in spec.values(base, float(s[0]), color)
                        ]
                        by_clip.setdefault(clip, []).append(
                            {
                                "pointer": pointer,
                                "type": spec.accessor_type,
                                "material": material,
                                "channel": spec.name,
                                "input": slot(cut),
                                "output": slot(flat),
                                "count": len(cut),
                                "min": cut[0],
                                "max": cut[-1],
                            }
                        )
                    animated_nodes.add(name)
                    by_channel[spec.name] = by_channel.get(spec.name, 0) + 1
        planned = [
            (animation, by_clip[str(animation.get("name") or "")])
            for animation in animations
            if by_clip.get(str(animation.get("name") or ""))
        ]
        if not planned:
            return None

        added = MeshConvert._append_bin_views(edit, payloads)
        if not added:
            logger.warning(
                "Pointer channels: this GLB's buffer is external, so the "
                "channels have nowhere to live -- the ramps ship as data only."
            )
            return None

        accessors = gltf.setdefault("accessors", [])
        written = 0
        for animation, specs in planned:
            samplers = animation.setdefault("samplers", [])
            channels = animation.setdefault("channels", [])
            for spec in specs:
                accessors.append(
                    {
                        "bufferView": added[spec["input"]],
                        "componentType": 5126,
                        "count": spec["count"],
                        "type": "SCALAR",
                        "min": [spec["min"]],
                        "max": [spec["max"]],
                    }
                )
                input_index = len(accessors) - 1
                accessors.append(
                    {
                        "bufferView": added[spec["output"]],
                        "componentType": 5126,
                        "count": spec["count"],
                        "type": spec["type"],
                    }
                )
                samplers.append(
                    {
                        "input": input_index,
                        "output": len(accessors) - 1,
                        "interpolation": "LINEAR",
                    }
                )
                channels.append(
                    {
                        "sampler": len(samplers) - 1,
                        "target": {
                            # No ``node``: a pointer channel targets the
                            # document, not the scene graph.
                            "path": "pointer",
                            "extensions": {EXTENSION: {"pointer": spec["pointer"]}},
                        },
                    }
                )
                written += 1

        used = gltf.setdefault("extensionsUsed", [])
        if EXTENSION not in used:
            used.append(EXTENSION)
        edit.dirty = True

        materials = sorted(
            {spec["material"] for _a, specs in planned for spec in specs}
        )
        logger.info(
            "Pointer channels: %s written as %d %s channel(s) on %d material(s), "
            "across %d clip(s).",
            ", ".join(f"{n} x{c}" for n, c in sorted(by_channel.items())),
            written,
            EXTENSION,
            len(materials),
            len(planned),
        )
        return {
            "nodes": len(animated_nodes),
            "materials": len(materials),
            "channels": written,
            "by_channel": by_channel,
        }
