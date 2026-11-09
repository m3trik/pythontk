# !/usr/bin/python
# coding=utf-8
"""Write animated render effects into a GLB as ``KHR_animation_pointer`` channels.

glTF animates four things -- translation, rotation, scale and morph weights --
and a material property is none of them. The Khronos answer is
``KHR_animation_pointer``, which lets a channel target any property by JSON
pointer. Writing it is what makes a fade, a highlight or an animated emission
part of the DELIVERABLE rather than a note attached to it: a viewer that
implements the extension plays it with nothing explained to it, and one that
does not still loads the file (it is declared in ``extensionsUsed``, never
``extensionsRequired``).

ONE pass, a TABLE of channels. Each :class:`PointerChannel` row says which
``render_effects`` channel it plays, whether its target is a NODE (every
material under the node's subtree) or a MATERIAL (every material of that
name), which material property it targets and how a curve's sample composes
with the material's own value. The curves are the file's own -- each take's
curve proxy, read off the FBX (``MeshConvert.render_effect_curves``) -- never
the record, which only names what carries what (``render_effects`` v2).

Opaque at rest: no channel changes a material's ``alphaMode``. A reader blends
a material only while its animated alpha is below 1 and hides its renderer at
0 (the record's contract); the presence gate (``MeshConvert.apply_glb_visibility``)
states the hiding for every glTF viewer. A highlight keeps an emissive MAP in
the file: glTF cannot animate a texture's presence, so the channel states the
spans its highlight is above 0 (``extras.render_effects.active``), over which
the contract has a reader set the map aside -- the glow is then the
highlight -- and at rest the map is back.

three.js does not implement the extension -- checked against its
``GLTFLoader``, which is the runtime this pipeline's own preview uses; the
loader simply skips a channel with no ``target.node``. So the preview page
carries a small shim that reads these same channels back out of the file and
hands them to the mixer as ordinary keyframe tracks.

A node's materials are ISOLATED before being animated, because a material is
shared by whatever samples it: animating the one a node happens to use would
animate every other object using it too. A material used by nothing outside
the subtree is animated in place; one that is shared is cloned (JSON only: a
clone costs no binary, nor does the mesh copy taken when a mesh is shared
with something outside the subtree). A material target is the material
itself, wherever it is drawn.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from pythontk.core_utils.color import ColorStops
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit, _SamplerBatch

__all__ = ["GlbFades", "PointerChannel", "CHANNELS"]

logger = logging.getLogger(__name__)

#: The extension this pass writes. Declared in ``extensionsUsed`` only: a
#: viewer without it must still open the file and simply not animate.
EXTENSION = "KHR_animation_pointer"
#: Where a material's emission above 1 lives (glTF caps ``emissiveFactor`` at 1).
STRENGTH = "KHR_materials_emissive_strength"
#: The alpha channel's target, kept under its historical name. The alpha is the
#: fourth component of the base colour factor, and glTF has no pointer to a
#: single component -- so the whole VEC4 is animated, holding the material's
#: own RGB constant.
POINTER = "/materials/{index}/pbrMetallicRoughness/baseColorFactor"
#: How long, in seconds, a cut is given in a LINEAR sampler: a step is stated
#: as a hold this long before the key it cuts to. Small enough to read as a
#: cut at any playback rate, large enough to survive float32 times in a clip
#: of tens of minutes.
STEP_SECONDS = 1e-3

Rgb = Sequence[float]
#: One resolved ``(r, g, b)`` per stop, in ``ColorStops.keys`` order.
Stops = Tuple[Rgb, ...]
#: A curve: ``[(seconds, value, interpolation), ...]`` -- interpolation the
#: segment after the key, ``"step"`` or ``"linear"`` (``"smooth"`` reads as
#: linear: the reader samples a cubic segment at the file's rate).
Curve = Sequence[Sequence[Any]]
#: ``{(target, channel): [node path, ...]}``: the nodes a node channel's curve
#: proxies sit under -- the nodes their curves play on (``GlbFades.carriers``).
Carriers = Dict[Tuple[str, str], List[str]]


def _mapped(material: Optional[Dict[str, Any]]) -> bool:
    """Whether *material* has an emissive map; ``None`` (a preview's call,
    with no material) has none."""
    return isinstance((material or {}).get("emissiveTexture"), dict)


def _opacity_values(
    base: List[float],
    sample: float,
    _stops: Stops,
    _material: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """Alpha rides the fourth lane; the material's own RGB is carried through."""
    return [base[0], base[1], base[2], min(1.0, max(0.0, sample))]


def _highlight_values(
    base: List[float],
    sample: float,
    stops: Stops,
    material: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """A glow BETWEEN the channel's two stops, clamped to glTF's LDR factor.

    Over a uniform emission it adds to the material's own: an LED panel that
    is also highlighted keeps glowing, and a white surface highlighted blue
    reads blue. Over an emissive MAP the map is set aside while the highlight
    is above 0 -- the glow is the highlight alone (glTF emission is factor x
    map, and a map hides an additive glow: measured 2026-09-05 on a production
    board, the channel was written and nothing changed on screen) -- and at
    rest the factor is the material's own, the map back. A low stop of black
    is the legacy shape -- ``base + hi * sample`` exactly.
    """
    hi = stops[0]
    lo = stops[1] if len(stops) > 1 else ColorStops.BLACK
    if _mapped(material):
        if sample <= 0.0:
            return list(base)
        own = (0.0, 0.0, 0.0)
    else:
        own = base
    return [
        min(1.0, max(0.0, own[i] + lo[i] + (hi[i] - lo[i]) * sample)) for i in range(3)
    ]


def _emission_values(
    base: List[float],
    sample: float,
    stops: Stops,
    _material: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """The weight scales the emission colour, already divided by the strength
    the material is given (``stops`` here is ``(colour x strength / given,)``)."""
    colour = stops[0]
    return [min(1.0, max(0.0, colour[i] * sample)) for i in range(3)]


def _opacity_rest(sample: float, _stops: Stops, _material: Dict[str, Any]) -> bool:
    """A held alpha of 1 leaves the material as authored, and a held 0 is the
    presence gate's to hide; only a held partial alpha changes the render."""
    return sample <= 0.0 or sample >= 1.0


def _highlight_rest(sample: float, stops: Stops, material: Dict[str, Any]) -> bool:
    """A held intensity of 0 adds the LOW stop -- nothing when it is black,
    and nothing over a map, which is back at rest."""
    low = stops[1] if len(stops) > 1 else ColorStops.BLACK
    return sample <= 0.0 and (_mapped(material) or not any(float(c) for c in low))


@dataclass(frozen=True)
class PointerChannel:
    """One animatable material property and how a curve reaches it.

    Attributes:
        name: The ``render_effects`` channel it plays (a transport channel).
        pointer: JSON pointer template with ``{index}`` for the material.
        property: Path of the material property inside the material dict, as
            ``("pbrMetallicRoughness", "baseColorFactor")`` or ``("emissiveFactor",)``.
        default: The property's per-spec default when the material omits it.
        target: ``"node"`` -- the materials under a track's node -- or
            ``"material"`` -- every material of a ``materials`` entry's name.
        color_stops: Optional :class:`ColorStops` naming the record entry's
            colour key(s) for this channel.
        values: ``(base, sample, stops, material=None) -> components`` for
            one key, where *stops* is one resolved ``(r, g, b)`` per stop and
            *material* the glTF material it composes on; omitted -- a
            panel's preview swatch, handed this very function so it cannot
            drift from what ships, calls it ``(base, sample, stops)`` -- it
            reads as a material with no map.
        rest: ``(sample, stops, material) -> bool``: a value that, HELD
            through a clip, leaves the render as the material has it -- such a
            clip writes nothing. ``None``: every held value does.
        active: Whether the channel states, per clip, the spans its curve is
            above 0 on a material with an emissive map (the highlight's
            map rule).
    """

    name: str
    pointer: str
    property: Tuple[str, ...]
    default: Tuple[float, ...]
    target: str
    color_stops: Optional[ColorStops]
    values: Callable[..., List[float]]
    rest: Optional[Callable[[float, Stops, Dict[str, Any]], bool]] = None
    active: bool = False

    @property
    def color_key(self) -> Optional[str]:
        """The HIGH stop's record key.

        Deprecated read-through kept for one release: the channel carries a
        :class:`ColorStops` now, and a caller that wants every key wants
        ``color_stops.keys``.
        """
        return self.color_stops.hi if self.color_stops else None

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


#: The channel table, in write order: ``opacity`` the fade, ``highlight`` a
#: glow with a per-node colour ramp, ``emission`` a material's animated
#: emission weight. ``visibility``, the one stepped channel, plays as the
#: presence gate (a ``scale`` channel), not here.
CHANNELS: Dict[str, PointerChannel] = {
    "opacity": PointerChannel(
        name="opacity",
        pointer=POINTER,
        property=("pbrMetallicRoughness", "baseColorFactor"),
        default=(1.0, 1.0, 1.0, 1.0),
        target="node",
        color_stops=None,
        values=_opacity_values,
        rest=_opacity_rest,
    ),
    "highlight": PointerChannel(
        name="highlight",
        pointer="/materials/{index}/emissiveFactor",
        property=("emissiveFactor",),
        default=(0.0, 0.0, 0.0),
        target="node",
        color_stops=ColorStops("highlight_color", "highlight_color_dim"),
        values=_highlight_values,
        rest=_highlight_rest,
        active=True,
    ),
    "emission": PointerChannel(
        name="emission",
        pointer="/materials/{index}/emissiveFactor",
        property=("emissiveFactor",),
        default=(0.0, 0.0, 0.0),
        target="material",
        color_stops=None,
        values=_emission_values,
    ),
}

#: Deprecated: each stop now carries its own fallback (an unstated HIGH is
#: white so the ramp still reads, an unstated LOW is black so a legacy asset
#: does not invert). Kept for one release; read ``spec.color_stops`` instead.
DEFAULT_COLOR: Tuple[float, float, float] = ColorStops.WHITE


class _GlbFadesInternal:
    """Internal helpers for :class:`GlbFades`."""

    # ------------------------------------------------------------- the curves
    @staticmethod
    def _keys(curve: Curve) -> List[Tuple[float, float, str]]:
        """*curve* as ``(seconds, value, interpolation)``, time-ordered (ties
        keep their order: the later key is the state from that instant)."""
        keys = []
        for key in curve or ():
            try:
                time, value = float(key[0]), float(key[1])
            except (TypeError, ValueError, IndexError):
                continue
            interpolation = str(key[2]) if len(key) > 2 else "linear"
            keys.append((time, value, interpolation))
        return sorted(keys, key=lambda k: k[0])

    @staticmethod
    def _strictly_increasing(
        samples: Sequence[Tuple[float, Any]],
    ) -> List[Tuple[float, Any]]:
        """*samples* on strictly increasing times, the LAST value at a repeated
        time winning: glTF requires an animation input to strictly increase."""
        merged: List[Tuple[float, Any]] = []
        for time, value in samples:
            if merged and merged[-1][0] >= time:
                merged[-1] = (merged[-1][0], value)
            else:
                merged.append((time, value))
        return merged

    @classmethod
    def _linearized(cls, curve: Curve) -> List[Tuple[float, float]]:
        """*curve* as ``(seconds, value)`` keys a LINEAR sampler reproduces:
        a stepped segment holds its value until :data:`STEP_SECONDS` before
        the next key, which it then cuts to."""
        keys = cls._keys(curve)
        out: List[Tuple[float, float]] = []
        for index, (time, value, interpolation) in enumerate(keys):
            out.append((time, value))
            if index + 1 >= len(keys) or interpolation != "step":
                continue
            following = keys[index + 1][0]
            if following - time > STEP_SECONDS:
                out.append((following - STEP_SECONDS, value))
        return cls._strictly_increasing(out)

    @classmethod
    def _positive_runs(cls, curve: Curve, start: float = 0.0) -> List[Tuple[float, int]]:
        """Where *curve* is above 0, as stepped runs ``[(seconds, 0|1), ...]``
        from *start* (the clip's t=0) on.

        A ramping segment is above 0 wherever either end is -- it leaves 0 the
        instant it starts rising and reaches it only at its end -- and a
        stepped one where its own key is. Before the first key the first
        key's value holds (a curve is read held at its ends), and after the
        last, the last's.
        """
        keys = cls._keys(curve)
        if not keys:
            return []
        runs: List[Tuple[float, int]] = [(start, 1 if keys[0][1] > 0.0 else 0)]
        for index, (time, value, interpolation) in enumerate(keys):
            if index + 1 < len(keys) and interpolation != "step":
                on = value > 0.0 or keys[index + 1][1] > 0.0
            else:
                on = value > 0.0
            runs.append((time, 1 if on else 0))
        return cls._collapsed(runs)

    @classmethod
    def _glowing_runs(
        cls, shaped: Sequence[Tuple[float, float]], start: float = 0.0
    ) -> List[Tuple[float, int]]:
        """Where the keys a highlight's factor is written at (*shaped*,
        :meth:`_with_boundaries`) glow, as stepped runs ``[(seconds, 0|1),
        ...]`` from *start* (the clip's t=0) on: each key's state from its
        instant, on where its value is above 0 -- so at every key the runs
        and the factor agree (the factor is the glow exactly where the value
        is above 0, the material's own elsewhere). A run never starts at a
        ramp's first key, whose value is still 0, but at the edge key
        :data:`STEP_SECONDS` inside it, where the glow starts."""
        if not shaped:
            return []
        runs = [(start, 1 if shaped[0][1] > 0.0 else 0)]
        runs.extend((time, 1 if value > 0.0 else 0) for time, value in shaped)
        return cls._collapsed(runs)

    @classmethod
    def _stepped_runs(cls, curve: Curve, start: float = 0.0) -> List[Tuple[float, int]]:
        """A stepped channel (``visibility``) as runs: on at 0.5 and above,
        each key the state from its instant on, whatever the file's
        interpolation (a DCC that bakes a step writes a one-frame ramp)."""
        keys = cls._keys(curve)
        if not keys:
            return []
        runs = [(start, 1 if keys[0][1] >= 0.5 else 0)]
        runs.extend((time, 1 if value >= 0.5 else 0) for time, value, _i in keys)
        return cls._collapsed(runs)

    @classmethod
    def _collapsed(cls, runs: Sequence[Tuple[float, int]]) -> List[Tuple[float, int]]:
        """*runs* on strictly increasing times, then with each run that changes
        nothing dropped."""
        out: List[Tuple[float, int]] = []
        for time, value in cls._strictly_increasing(runs):
            if not out or out[-1][1] != value:
                out.append((time, value))
        return out

    @staticmethod
    def _both(*runs: Sequence[Tuple[float, int]]) -> List[Tuple[float, int]]:
        """The runs on where every one of *runs* is (each held at its ends)."""
        present = [r for r in runs if r]
        if not present:
            return []
        times = sorted({t for r in present for t, _v in r})

        def at(run, time):
            value = run[0][1]
            for t, v in run:
                if t <= time:
                    value = v
                else:
                    break
            return value

        out: List[Tuple[float, int]] = []
        for time in times:
            value = 1 if all(at(r, time) for r in present) else 0
            if not out or out[-1][1] != value:
                out.append((time, value))
        return out

    @classmethod
    def _with_boundaries(cls, curve: Curve) -> List[Tuple[float, float]]:
        """:meth:`_linearized`, plus a key :data:`STEP_SECONDS` inside each
        segment that leaves or reaches 0, so a value that jumps there (the
        highlight over a map: the factor is the material's own at 0 and the
        glow above it) is stated at the edge, not blended across it."""
        keys = cls._linearized(curve)
        out: List[Tuple[float, float]] = []
        for index, (time, value) in enumerate(keys):
            out.append((time, value))
            if index + 1 >= len(keys):
                continue
            nxt_time, nxt_value = keys[index + 1]
            span = nxt_time - time
            if span <= 2 * STEP_SECONDS or (value > 0.0) == (nxt_value > 0.0):
                continue
            edge = time + STEP_SECONDS if value <= 0.0 else nxt_time - STEP_SECONDS
            out.append((edge, value + (nxt_value - value) * (edge - time) / span))
        return cls._strictly_increasing(sorted(out, key=lambda k: k[0]))

    # ------------------------------------------------------- the file's parts
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
    ) -> List[int]:
        """Give *nodes* materials nothing outside them shares; return their indices.

        Two levels of copying, both JSON-only:

        * A MESH used by a node outside this subtree is duplicated first, so
          repointing its primitives cannot reach the outsider. The copy shares
          every accessor, so it adds no geometry.
        * A MATERIAL used by anything outside is cloned; one used by nothing
          outside is animated where it stands. Nothing authored changes on
          either -- no channel switches an alpha mode or drops a map -- so the
          pass is idempotent: a second run finds every clone exclusive.

        The clone keeps the NAME, deliberately: glTF does not require unique
        material names, and the lightmap manifest (``lightmaps.materials``)
        and the ``materials`` record are keyed by name -- a renamed clone
        loses its lightmap in every reader that binds that way.
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
                # Instanced inside this subtree: its primitives already point
                # at this call's materials; walking them again would read a
                # CLONE as a source and clone it.
                continue
            repointed.add(mesh_index)
            for primitive in meshes[mesh_index].get("primitives") or []:
                source = primitive.get("material")
                if not isinstance(source, int) or not 0 <= source < len(materials):
                    continue
                if source not in clones:
                    if material_users.get(source, set()) <= inside:
                        clones[source] = source
                    else:
                        materials.append(copy.deepcopy(materials[source]))
                        clones[source] = len(materials) - 1
                    touched.append(clones[source])
                primitive["material"] = clones[source]
        return touched

    @staticmethod
    def _named(gltf: Dict[str, Any], name: str) -> List[int]:
        """Every material *name* joins: its own name, or the name after the
        last ``:`` on either side (an import that dropped a namespace)."""
        leaf = str(name).split(":")[-1]
        return [
            index
            for index, material in enumerate(gltf.get("materials") or [])
            if isinstance(material, dict)
            and (
                material.get("name") == name
                or str(material.get("name") or "").split(":")[-1] == leaf
            )
        ]

    @staticmethod
    def _parents(gltf: Dict[str, Any]) -> Dict[int, int]:
        """``child index -> parent index`` over the file's nodes."""
        parents: Dict[int, int] = {}
        for index, node in enumerate(gltf.get("nodes") or []):
            for child in (node or {}).get("children") or []:
                if isinstance(child, int):
                    parents.setdefault(child, index)
        return parents

    @classmethod
    def _paths(cls, gltf: Dict[str, Any]) -> List[str]:
        """Every node's path from its root, by index (``"|grp|a:Box"``)."""
        nodes = gltf.get("nodes") or []
        parents = cls._parents(gltf)
        paths: List[str] = []
        for index in range(len(nodes)):
            steps: List[str] = []
            seen: Set[int] = set()
            at: Optional[int] = index
            while at is not None and at not in seen:
                seen.add(at)
                steps.append(str((nodes[at] or {}).get("name") or ""))
                at = parents.get(at)
            paths.append("|" + "|".join(reversed(steps)))
        return paths

    @classmethod
    def _resolve(
        cls,
        name: str,
        channels: Sequence[str],
        carriers: Carriers,
        paths: Sequence[str],
    ) -> Tuple[List[int], str]:
        """The nodes a record's track *name* plays its *channels* on, by the
        transport's rule (``SceneRecords.TRANSPORT.join``): its proxies'
        parent wherever the file carried a proxy, else its name."""
        leaf = SceneRecords.TRANSPORT.target_name(name)
        wanted = {path for channel in channels for path in carriers.get((leaf, channel), ())}
        carried = [index for index, path in enumerate(paths) if path in wanted]
        return SceneRecords.TRANSPORT.join(name, paths, carried)

    @staticmethod
    def _existing_pointers(animations: Sequence[Dict[str, Any]]) -> Set[Tuple[str, str]]:
        """``(clip, pointer)`` for every pointer target a clip already animates."""
        found: Set[Tuple[str, str]] = set()
        for animation in animations:
            clip = str(animation.get("name") or "")
            for channel in animation.get("channels") or []:
                target = channel.get("target") or {}
                if target.get("path") != "pointer":
                    continue
                pointer = ((target.get("extensions") or {}).get(EXTENSION) or {}).get(
                    "pointer"
                )
                if pointer:
                    found.add((clip, str(pointer)))
        return found

    @staticmethod
    def _held(values: Sequence[float]) -> Optional[float]:
        """The one value *values* hold, or ``None`` when they move."""
        if not values:
            return None
        first = round(float(values[0]), 6)
        return first if all(round(float(v), 6) == first for v in values) else None

    # ----------------------------------------------------- the emission entry
    @staticmethod
    def _emission_strength(material: Dict[str, Any]) -> float:
        extension = (material.get("extensions") or {}).get(STRENGTH) or {}
        try:
            return float(extension.get("emissiveStrength", 1.0))
        except (TypeError, ValueError):
            return 1.0

    @classmethod
    def _give_strength(cls, gltf: Dict[str, Any], index: int, peak: float) -> float:
        """Give material *index* the strength its emission needs -- the curve's
        *peak* and its own emission, whichever is brighter, at least 1 --
        keeping its own emission as it renders; returns the strength given."""
        material = gltf["materials"][index]
        own = cls._emission_strength(material)
        factor = [float(c) for c in material.get("emissiveFactor") or (0.0, 0.0, 0.0)]
        given = max(1.0, peak, max(factor) * own)
        if abs(given - own) > 1e-9:
            material["emissiveFactor"] = [
                min(1.0, c * own / given) for c in factor[:3]
            ]
        extensions = material.setdefault("extensions", {})
        if given > 1.0:
            extensions[STRENGTH] = {"emissiveStrength": given}
            used = gltf.setdefault("extensionsUsed", [])
            if STRENGTH not in used:
                used.append(STRENGTH)
        else:
            extensions.pop(STRENGTH, None)
        if not extensions:
            material.pop("extensions", None)
        return given


class GlbFades(_GlbFadesInternal):
    """Publish a file's animated render effects as animated material channels."""

    CHANNELS = CHANNELS
    STEP_SECONDS = STEP_SECONDS

    @classmethod
    def carriers(cls, gltf: Dict[str, Any]) -> Carriers:
        """``{(target, channel): [node path, ...]}``: where each node channel's
        curve proxy sits in *gltf* -- its parent, the node its curve plays on
        (``SceneRecords.TRANSPORT.join``). A proxy is a node marked
        ``SceneRecords.TRANSPORT.marker``, never a name alone. Read it BEFORE
        the proxies are stripped (``MeshConvert.strip_glb_curve_proxies``)
        and hand it to the passes after: the paths name the parents, which
        the strip leaves where they are."""
        transport = SceneRecords.TRANSPORT
        nodes = gltf.get("nodes") or []
        parents = cls._parents(gltf)
        paths = cls._paths(gltf)
        carriers: Carriers = {}
        for index, node in enumerate(nodes):
            if index not in parents or not GlbEdit.user_property(node, transport.marker):
                continue
            parsed = transport.parse(str((node or {}).get("name") or ""))
            if parsed is None or parsed[1] not in transport.node_channels:
                continue
            found = carriers.setdefault(parsed, [])
            if paths[parents[index]] not in found:
                found.append(paths[parents[index]])
        return carriers

    @classmethod
    def apply_curves(
        cls,
        edit: Any,
        record: Dict[str, Any],
        curves: Dict[str, Dict[Tuple[str, str], Curve]],
        carriers: Optional[Carriers] = None,
    ) -> Optional[Dict[str, Any]]:
        """Write every pointer channel the *record* names, per clip, from *curves*.

        Parameters:
            edit: An open ``GlbEdit``.
            record: The ``render_effects`` record (v2): its ``tracks`` name the
                node channels each node carries (and the highlight's colour
                stops), its ``materials`` the material channels.
            curves: ``{clip name: {(target, channel): curve}}`` -- each curve
                ``[(seconds, value, interpolation), ...]`` from the clip's t=0
                (``MeshConvert.render_effect_curves``). A clip of another name,
                and a curve the record does not name, write nothing.
            carriers: Where each node channel's proxy sat (:meth:`carriers`,
                read before the proxies were stripped); ``None`` reads them
                off *edit* itself. A node channel plays on its proxy's
                parent; a track with no proxy joins by name, a tie never
                guessed (``SceneRecords.TRANSPORT.join``).

        Returns:
            ``{"nodes", "materials", "channels", "by_channel"}``, or ``None``
            when there was nothing to write.
        """
        gltf = edit.gltf
        animations = gltf.get("animations") or []
        if not animations or not curves:
            return None
        clips = {str(a.get("name") or ""): a for a in animations}

        # What moves where, per target and channel, resolved BEFORE anything is
        # cloned: a curve that holds its rest in every clip costs no copy.
        missing: List[str] = []
        tied: List[str] = []
        planned: List[Tuple[str, str, PointerChannel, Dict[str, Any], Stops]] = []
        paths = cls._paths(gltf)
        if carriers is None:
            carriers = cls.carriers(gltf)
        wanted: List[Tuple[str, str, Dict[str, Any]]] = []
        for track in record.get("tracks") or ():
            if isinstance(track, dict) and track.get("node"):
                for channel in track.get("channels") or ():
                    wanted.append(("node", str(track["node"]), {**track, "channel": channel}))
        for entry in record.get("materials") or ():
            if isinstance(entry, dict) and entry.get("material"):
                for channel in entry.get("channels") or ():
                    wanted.append(("material", str(entry["material"]), {**entry, "channel": channel}))
        for kind, target, entry in wanted:
            spec = CHANNELS.get(entry["channel"])
            if spec is None or spec.target != kind:
                continue
            # The record names its target as the delivery (a material) or
            # the host (a node) does; a curve is keyed by the transport's
            # name for it, as its proxy is named.
            joined = SceneRecords.TRANSPORT.target_name(target)
            per_clip = {
                clip: keys
                for clip, keys in (
                    (clip, (by_target or {}).get((joined, spec.name)))
                    for clip, by_target in curves.items()
                )
                if keys and clip in clips
            }
            if not per_clip:
                continue
            roots: List[int] = []
            if kind == "node":
                roots, status = cls._resolve(target, (spec.name,), carriers, paths)
                if not roots:
                    (tied if status == "ambiguous" else missing).append(f"{target}.{spec.name}")
                    continue
            if kind == "material" and not cls._named(gltf, target):
                missing.append(f"{target}.{spec.name}")
                continue
            stops = (
                spec.color_stops.resolve([entry.get(k) for k in spec.color_stops.keys])
                if spec.color_stops is not None
                else ()
            )
            planned.append(
                (kind, target, spec, {"curves": per_clip, "entry": entry, "roots": roots}, stops)
            )
        if tied:
            logger.warning(
                "Pointer channels: %d curve(s) name a node several nodes share, "
                "and no curve proxy says which (%s) -- never guessed, they play "
                "nowhere here. Give each node its own name, or re-export so "
                "each curve rides its node's proxy.",
                len(tied),
                ", ".join(sorted(tied)),
            )
        if missing:
            logger.warning(
                "Pointer channels: %d curve(s) name a target not in this GLB (%s) "
                "-- they play nowhere here.",
                len(missing),
                ", ".join(sorted(missing)),
            )
        if not planned:
            return None

        existing = cls._existing_pointers(animations)
        users = cls._mesh_users(gltf)
        isolated: Dict[str, List[int]] = {}
        batch = _SamplerBatch()
        by_clip: Dict[str, List[Dict[str, Any]]] = {}
        animated: Set[str] = set()
        by_channel: Dict[str, int] = {}
        doubled: Set[str] = set()
        for kind, target, spec, plan, stops in planned:
            materials = cls._targets(gltf, kind, target, spec, plan, stops, users, isolated)
            for material in materials:
                pointer = spec.pointer.format(index=material)
                material_def = gltf["materials"][material]
                given = stops
                if spec.name == "emission":
                    given = cls._emission_stops(gltf, material, plan)
                base = spec.base(gltf, material)
                wrote = False
                for clip, curve in plan["curves"].items():
                    if (clip, pointer) in existing:
                        # Already animated in this clip: a second channel on
                        # one property is undefined -- the file's own stands.
                        doubled.add(f"{target}.{spec.name}")
                        continue
                    shaped = (
                        cls._with_boundaries(curve)
                        if spec.active and _mapped(material_def)
                        else cls._linearized(curve)
                    )
                    held = cls._held([v for _t, v in shaped])
                    if held is not None and (
                        (spec.rest is not None and spec.rest(held, given, material_def))
                        or spec.values(base, held, given, material_def) == base
                    ):
                        continue
                    if len(shaped) < 2:
                        shaped = [shaped[0], (shaped[0][0] + STEP_SECONDS, shaped[0][1])]
                    times = [t for t, _v in shaped]
                    flat = [
                        c
                        for _t, value in shaped
                        for c in spec.values(base, value, given, material_def)
                    ]
                    note: Dict[str, Any] = {"channel": spec.name, kind: target}
                    if spec.active and _mapped(material_def):
                        # Read off the keys the factor is written at, so the
                        # map is set aside exactly where the factor glows.
                        note["active"] = [
                            [round(t, 6) + 0.0, on] for t, on in cls._glowing_runs(shaped)
                        ]
                    by_clip.setdefault(clip, []).append(
                        {
                            "pointer": pointer,
                            "type": spec.accessor_type,
                            "material": material,
                            "input": batch.slot(times),
                            "output": batch.slot(flat),
                            "note": note,
                        }
                    )
                    existing.add((clip, pointer))
                    wrote = True
                if wrote:
                    animated.add(f"{kind}:{target}")
                    by_channel[spec.name] = by_channel.get(spec.name, 0) + 1
        if doubled:
            logger.warning(
                "Pointer channels: %s already animated in some clip -- the "
                "file's own channel stands there.",
                ", ".join(sorted(doubled)),
            )
        writes = [
            (clips[clip], specs) for clip, specs in by_clip.items() if specs
        ]
        if not writes:
            return None

        if not batch.commit(edit):
            logger.warning(
                "Pointer channels: this GLB's buffer is external, so the "
                "channels have nowhere to live -- the effects do not play."
            )
            return None

        written = 0
        for animation, specs in writes:
            channels = animation.setdefault("channels", [])
            for spec in specs:
                channels.append(
                    {
                        "sampler": batch.sampler(
                            animation, spec["input"], spec["output"], spec["type"]
                        ),
                        "target": {
                            # No ``node``: a pointer channel targets the
                            # document, not the scene graph.
                            "path": "pointer",
                            "extensions": {EXTENSION: {"pointer": spec["pointer"]}},
                        },
                        # What the channel plays, for a reader of the record:
                        # the record's channel and target, and -- for a
                        # highlight over a map -- the spans it is above 0.
                        "extras": {"render_effects": spec["note"]},
                    }
                )
                written += 1

        used = gltf.setdefault("extensionsUsed", [])
        if EXTENSION not in used:
            used.append(EXTENSION)
        edit.dirty = True

        materials = sorted({spec["material"] for _a, specs in writes for spec in specs})
        logger.info(
            "Pointer channels: %s written as %d %s channel(s) on %d material(s), "
            "across %d clip(s).",
            ", ".join(f"{n} x{c}" for n, c in sorted(by_channel.items())),
            written,
            EXTENSION,
            len(materials),
            len(writes),
        )
        return {
            "nodes": len({a for a in animated if a.startswith("node:")}),
            "materials": len(materials),
            "channels": written,
            "by_channel": by_channel,
        }

    @classmethod
    def _targets(
        cls,
        gltf: Dict[str, Any],
        kind: str,
        target: str,
        spec: PointerChannel,
        plan: Dict[str, Any],
        stops: Stops,
        users: Dict[int, List[int]],
        isolated: Dict[str, List[int]],
    ) -> List[int]:
        """The material indices *target*'s channel animates -- a node's
        materials (under the nodes it resolved to, ``plan["roots"]``)
        isolated once for every channel it carries, and only when some clip
        moves it from its rest."""
        if kind == "material":
            return cls._named(gltf, target)
        roots = list(plan["roots"])
        key = "|".join(str(i) for i in roots)
        if key in isolated:
            return isolated[key]
        moves = False
        for curve in plan["curves"].values():
            held = cls._held([v for _t, v in cls._linearized(curve)])
            if held is None or spec.rest is None:
                moves = True
                break
            for index in cls._candidates(gltf, roots):
                if not spec.rest(held, stops, gltf["materials"][index]):
                    moves = True
            if moves:
                break
        if not moves:
            return []
        isolated[key] = cls._isolate(gltf, cls._subtree(gltf, roots), users)
        return isolated[key]

    @classmethod
    def _candidates(cls, gltf: Dict[str, Any], roots: Sequence[int]) -> List[int]:
        """The material indices drawn under *roots*, as they stand."""
        meshes = gltf.get("meshes") or []
        found: List[int] = []
        for index in cls._subtree(gltf, roots):
            mesh = (gltf.get("nodes") or [])[index].get("mesh")
            if isinstance(mesh, int) and 0 <= mesh < len(meshes):
                for primitive in meshes[mesh].get("primitives") or []:
                    material = primitive.get("material")
                    if isinstance(material, int) and material not in found:
                        found.append(material)
        return found

    @classmethod
    def _emission_stops(
        cls, gltf: Dict[str, Any], index: int, plan: Dict[str, Any]
    ) -> Stops:
        """``(colour x strength / given,)`` for material *index*'s emission
        entry: the entry's colour and strength, else the material's own, over
        the strength the material is given for the curve's peak."""
        material = gltf["materials"][index]
        entry = plan["entry"]
        colour = entry.get("emission_color")
        if not (isinstance(colour, (list, tuple)) and len(colour) >= 3):
            colour = material.get("emissiveFactor") or (0.0, 0.0, 0.0)
        strength = entry.get("emission_strength")
        if not isinstance(strength, (int, float)):
            strength = cls._emission_strength(material)
        scaled = [float(c) * float(strength) for c in list(colour)[:3]]
        weights = [
            value for curve in plan["curves"].values() for _t, value in cls._linearized(curve)
        ]
        peak = max(scaled) * max([0.0, *weights])
        given = plan.setdefault("given", {}).get(index)
        if given is None:
            given = plan["given"][index] = cls._give_strength(gltf, index, peak)
        return (tuple(c / given for c in scaled),)
