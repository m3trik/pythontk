# !/usr/bin/python
# coding=utf-8
"""Write authored opacity ramps into a GLB as ``KHR_animation_pointer`` channels.

glTF animates four things -- translation, rotation, scale and morph weights --
and alpha is none of them. The Khronos answer is ``KHR_animation_pointer``,
which lets a channel target any property by JSON pointer, including
``/materials/N/pbrMetallicRoughness/baseColorFactor``. Writing it is what makes
a fade part of the DELIVERABLE rather than a note attached to it: a viewer that
implements the extension plays the fade with nothing explained to it, and one
that does not still loads the file (it is declared in ``extensionsUsed``, never
``extensionsRequired``).

three.js does not implement it -- checked against its ``GLTFLoader``, which is
the runtime this pipeline's own preview uses; the loader simply skips a channel
with no ``target.node``. So the preview page carries a small shim that reads
these same channels back out of the file and hands them to the mixer as
ordinary keyframe tracks. ONE statement of the fade, the glTF's; no side block,
no second encoding, and the day three.js ships support the shim deletes without
touching the exporter.

Materials are CLONED per faded subtree before being animated, because a
material is shared by whatever samples it: fading the one a node happens to use
would fade every other object using it too. The clone costs no binary -- a glTF
material is JSON, and so is the mesh copy taken when a mesh is shared with
something outside the fade.
"""

from __future__ import annotations

import copy
import logging
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["GlbFades"]

logger = logging.getLogger(__name__)

#: The extension this pass writes. Declared in ``extensionsUsed`` only: a
#: viewer without it must still open the file and simply not fade.
EXTENSION = "KHR_animation_pointer"
#: What a channel targets. The alpha is the fourth component of the base colour
#: factor, and glTF has no pointer to a single component -- so the whole VEC4
#: is animated, holding the material's own RGB constant.
POINTER = "/materials/{index}/pbrMetallicRoughness/baseColorFactor"


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

    @classmethod
    def _isolate(
        cls, gltf: Dict[str, Any], nodes: Sequence[int], users: Dict[int, List[int]]
    ) -> List[int]:
        """Give *nodes* materials nothing outside them shares; return their indices.

        Two levels of copying, both JSON-only:

        * A MESH used by a node outside this fade is duplicated first, so
          repointing its primitives cannot reach the outsider. The copy shares
          every accessor, so it adds no geometry.
        * Each MATERIAL is then cloned and switched to ``BLEND``, because
          animating alpha on a shared material fades everything that samples it.
        """
        meshes = gltf.setdefault("meshes", [])
        materials = gltf.setdefault("materials", [])
        inside = set(nodes)
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
                    clone = copy.deepcopy(materials[source])
                    # BLEND, not MASK: the ramp is continuous, and a MASK
                    # material is binary at its cutoff -- which is the pop this
                    # whole pass exists to replace.
                    clone["alphaMode"] = "BLEND"
                    # The NAME is kept, deliberately: glTF does not require
                    # unique material names, and the lightmap manifest
                    # (`lightmap_web.materials`) is keyed by name -- a
                    # renamed clone loses its lightmap in every reader that
                    # binds that way, the preview included.
                    materials.append(clone)
                    clones[source] = len(materials) - 1
                    touched.append(clones[source])
                primitive["material"] = clones[source]
        return touched

    @staticmethod
    def _base_color(gltf: Dict[str, Any], index: int) -> List[float]:
        """The material's own base colour factor, defaulted per spec."""
        material = (gltf.get("materials") or [])[index]
        pbr = material.setdefault("pbrMetallicRoughness", {})
        factor = pbr.get("baseColorFactor")
        if not (isinstance(factor, list) and len(factor) == 4):
            factor = [1.0, 1.0, 1.0, 1.0]
            pbr["baseColorFactor"] = factor
        return [float(v) for v in factor]


class GlbFades(_GlbFadesInternal):
    """Publish authored alpha ramps as animated material channels."""

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
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert
        from pythontk.file_utils.mesh_convert.glb_clips import GlbClips

        gltf = edit.gltf
        animations = gltf.get("animations") or []
        if not fades or not animations or fps <= 0:
            return None
        if any(
            (channel.get("target") or {}).get("path") == "pointer"
            for animation in animations
            for channel in (animation.get("channels") or [])
        ):
            # Already written. Running again would clone the clones and stack
            # a second channel on every material; the file's own channels are
            # the statement, and this pass has nothing to add to them.
            logger.debug("Fades: this GLB already carries pointer channels.")
            return None

        by_name: Dict[str, List[int]] = {}
        for index, node in enumerate(gltf.get("nodes") or []):
            name = (node or {}).get("name")
            if name is not None:
                by_name.setdefault(str(name).split("|")[-1], []).append(index)

        # Which node fades inside which clip, resolved BEFORE anything is
        # cloned. Isolating a subtree switches its materials to BLEND, and a
        # blended surface sorts differently from an opaque one -- so doing it
        # for a ramp that turns out to hold a constant alpha in every clip
        # would change how the object renders in exchange for no animation.
        missing: List[str] = []
        resolved: List[
            Tuple[str, List[int], Dict[str, Tuple[List[float], List[Any]]]]
        ] = []
        for name, keys in sorted(fades.items()):
            roots = by_name.get(name)
            if not roots:
                missing.append(name)
                continue
            ramp = sorted((float(k[0]), float(k[1])) for k in keys)
            per_clip: Dict[str, Tuple[List[float], List[Any]]] = {}
            for animation in animations:
                clip = str(animation.get("name") or "")
                window, zero = windows.get(clip), zeros.get(clip)
                if window is None or zero is None:
                    continue
                lo = (window[0] - zero) / fps
                cut, alphas = GlbClips._window(
                    [(frame - zero) / fps for frame, _a in ramp],
                    [(alpha,) for _f, alpha in ramp],
                    "LINEAR",
                    False,
                    lo,
                    (window[1] - zero) / fps,
                    0.25 / fps,
                )
                # ``_window`` rebases onto the window it cut, which is right for
                # a clip BUILT to that window and wrong here: this channel joins
                # a clip that already has a zero, and the two are the same only
                # when the clip starts where its window does. The retained
                # whole-timeline stack is the case where they differ -- its zero
                # is the timeline's, so the ramp arrived shifted by the first
                # shot's start frame (measured: 7 frames).
                cut = [k + lo for k in cut]
                if len(cut) < 2 or len({round(a[0], 6) for a in alphas}) < 2:
                    continue  # this clip holds a constant alpha: nothing to animate
                per_clip[clip] = (cut, alphas)
            if per_clip:
                resolved.append((name, roots, per_clip))
        if missing:
            logger.warning(
                "Fades: %d faded node(s) are not in this GLB (%s) -- their "
                "ramps ship as data only.",
                len(missing),
                ", ".join(sorted(missing)),
            )
        if not resolved:
            return None

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
        faded = 0
        for name, roots, per_clip in resolved:
            materials = cls._isolate(gltf, cls._subtree(gltf, roots), users)
            if not materials:
                # A locator drives the fade but carries no geometry of its own;
                # nothing under it renders, so there is no alpha to animate.
                continue
            faded += 1
            for clip, (cut, alphas) in per_clip.items():
                for material in materials:
                    rgb = cls._base_color(gltf, material)[:3]
                    flat = [c for a in alphas for c in (*rgb, a[0])]
                    by_clip.setdefault(clip, []).append(
                        {
                            "material": material,
                            "input": slot(cut),
                            "output": slot(flat),
                            "count": len(cut),
                            "min": cut[0],
                            "max": cut[-1],
                        }
                    )
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
                "Fades: this GLB's buffer is external, so the alpha channels "
                "have nowhere to live -- the ramps ship as data only."
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
                        "type": "VEC4",
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
                            "extensions": {
                                EXTENSION: {
                                    "pointer": POINTER.format(index=spec["material"])
                                }
                            },
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
            "Fades: %d authored ramp(s) written as %s channels on %d cloned "
            "material(s), across %d clip(s).",
            faded,
            EXTENSION,
            len(materials),
            len(planned),
        )
        return {
            "nodes": faded,
            "materials": len(materials),
            "channels": written,
        }
