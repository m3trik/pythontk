# !/usr/bin/python
# coding=utf-8
"""The light rig engine every host shares: save, manage, swap and restore named rigs.

A *rig* is the lights under one scene group, plus the plugs it overrides on
nodes it does not own (the office lights turned off, a material's emission),
and how each light fixture of the scene lights the room (``fixtures``,
:class:`~pythontk.LightRigFixture`). Rigs are plain JSON payloads
(:class:`~pythontk.LightRigPayload`) in the rig library every host and tool
shares, so a rig saved in one scene lights another.

Everything the hosts do alike is written once on :class:`LightRig`: the
payload's canonical form (:meth:`LightRig.normalize`), the comparison
(:meth:`LightRig.diff`, :meth:`LightRig.compare`), the fingerprint "edited"
reads, the scene record (``ptk.SceneRecords.LIGHT_RIGS``), the swap and its
snapshot, and the library verbs. Every place that reaches into a scene is a
call on :attr:`LightRig.HOST` (a :class:`LightRigHost`). A host library binds
its adapter and its tables on a class of its own, keeping this one's name
(configuration), and overrides nothing.

Canonical form, version 1::

    {"version": 1,
     "group": "MOODY_LIGHTS",                    # the scene group its lights live under
     "lights": [{"name": ..., "type": <host type>,
                 "pos": [x, y, z], "rot": [x, y, z],        # world space: cm, degrees
                 "scale": [...], "rotate_order": n, "visible": False,   # when not the default
                 "attrs": {attr: value},          # what differs from a fresh light, host names
                 "textures": {attr: {"file": ..., "colorSpace": ..., "plug": "outColor"}},
                 "user_attrs": {attr: {"type": ..., "value": ...}},
                 "filters": [{"type": ..., "attrs": {...}, "textures": {...}}]}],
     "existing": {node: {attr: value}},          # overrides on scene lights it does not own
     "materials": {node: {attr: value}},         # overrides on materials
     ...}                                        # anything else (viewer, source) is carried

A light may also be spelled the host-neutral way (``ptk.LightRigLight``):
``type`` ``spot`` / ``point`` ..., ``cone``, ``radius``, ``aim``, ``kelvin``,
``gobo`` ...; :meth:`LightRig.normalize` maps each name through the binding's
tables (:attr:`LightRig.TYPES`, :attr:`LightRig.FRIENDLY`). Lengths are
centimetres, angles degrees and the world is Y-up (right-handed), whatever the
host's scene works in: each host converts at its own boundary.

Applying a rig never deletes a light it does not own. Its overrides win where
the host can hand them back (:meth:`LightRigHost.set_overrides`): swap back and
undo restore what they replaced.
"""

import contextlib
import copy
import dataclasses
import hashlib
import json
import math
import os
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    ContextManager,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.core_utils.engines.lighting.light_rig_payload import (
    LightRigFixture,
    LightRigLight,
    LightRigPayload,
)
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords

if TYPE_CHECKING:
    from pythontk.core_utils.presets.store import PresetStore


# --------------------------------------------------------------------------- the host adapter
@runtime_checkable
class LightRigHost(Protocol):
    """What the light rig engine asks of the host it runs in.

    Lengths and angles cross this seam in the payload's units (centimetres,
    degrees, Y-up): a host converts on its side. A node is named as the host
    names it; a *plug* is ``"<node>.<attr>"``.
    """

    #: The host library, for the messages that tell a user to update it.
    package: str

    def payload_units(self) -> ContextManager:
        """A block whose scene reads and writes are in the payload's units."""

    def scene_units(self) -> Any:
        """The scene's own units, read now; handed back to :meth:`set_overrides`."""

    def undo_step(self, name: str) -> ContextManager:
        """A block that is ONE step of the host's undo, named *name*."""

    def record_store(self) -> Any:
        """The scene store ``SceneRecords.LIGHT_RIGS`` reads and writes."""

    def exists(self, node: str) -> bool:
        """Whether the scene holds *node*."""

    def defaults(self, kind: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """``({attr: value of a fresh node}, {child attr: compound parent})`` for light type *kind*."""

    def prepare(self, kinds: set, group: str) -> None:
        """Ready the scene for building *kinds* into *group* (outside the undo step)."""

    def capture_lights(self, group: str) -> List[Dict[str, Any]]:
        """The lights under *group*, each a canonical light entry."""

    def remove_group(self, group: str) -> List[str]:
        """Delete *group*, its lights and what only they used; returns what went."""

    def group_parent(self, group: str) -> Optional[str]:
        """The group *group* sits under, or ``None`` at the scene's root."""

    def group_lights(
        self, made: List[str], group: str, parent: Optional[str]
    ) -> Tuple[str, List[str]]:
        """Put the lights *made* under a new group *group* (under *parent*):
        ``(the group as built, its lights' names)``."""

    def build_light(self, light: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
        """Build canonical light *light*: ``(the light, or None when the host
        cannot build it, lines left alone with the reason)``."""

    def gobo_filter(self, path: str, rotate: Optional[float]) -> Dict[str, Any]:
        """The canonical filter entry a neutral ``gobo`` becomes."""

    def plug(self, node: str, attr: str) -> Optional[Tuple[str, str]]:
        """``(named, landed)`` for an override of ``node.attr``: the plug it
        names and the one it lands on (they differ where one plug follows
        another); ``None`` when there is no such attribute."""

    def plug_key(self, plug: str) -> str:
        """*plug* spelled one way, so two spellings of one plug compare equal."""

    def set_overrides(
        self, node: str, attrs: Dict[str, Any], scene_units: Any = None
    ) -> Tuple[List[str], List[str]]:
        """Set *attrs* on *node*: ``(left alone, taken over)``, one line each."""

    def current_values(
        self, section: Optional[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        """The scene's current values of the plugs *section* names (missing ones dropped)."""


class _MemoryStore(object):
    """A scene store in memory (``read`` / ``write`` by scope and key)."""

    def __init__(self):
        self.data: Dict[Tuple[Any, str], str] = {}

    def read(self, scope, key: str) -> Optional[str]:
        return self.data.get((scope, key))

    def write(self, scope, key: str, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None if self.data.pop((scope, key), None) is None else key
        self.data[(scope, key)] = text
        return key


class PureLightRigHost(object):
    """The :class:`LightRigHost` with no host behind it: a scene in memory.

    Groups hold light entries in the canonical form, other nodes hold plain
    attribute values (``nodes``), and the record rides a store in memory. What
    a rig runs on in plain Python, and the base a fake host overrides a part of.
    """

    package = "pythontk"

    def __init__(self):
        #: ``{group: {light name: canonical light entry}}``.
        self.groups: Dict[str, Dict[str, Dict[str, Any]]] = {}
        #: ``{node: {attr: value}}``: what overrides land on.
        self.nodes: Dict[str, Dict[str, Any]] = {}
        #: ``{group: its parent group}``.
        self.parents: Dict[str, str] = {}
        self.store = _MemoryStore()

    def payload_units(self) -> ContextManager:
        return contextlib.nullcontext()

    def scene_units(self) -> Any:
        return None

    def undo_step(self, name: str) -> ContextManager:
        return contextlib.nullcontext()

    def record_store(self) -> Any:
        return self.store

    def exists(self, node: str) -> bool:
        return node in self.groups or node in self.nodes or any(
            node in lights for lights in self.groups.values()
        )

    def defaults(self, kind: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
        return {}, {}

    def prepare(self, kinds: set, group: str) -> None:
        pass

    def capture_lights(self, group: str) -> List[Dict[str, Any]]:
        return [copy.deepcopy(light) for light in self.groups.get(group, {}).values()]

    def remove_group(self, group: str) -> List[str]:
        if self.groups.pop(group, None) is None:
            return []
        self.parents.pop(group, None)
        return [group]

    def group_parent(self, group: str) -> Optional[str]:
        return self.parents.get(group)

    def group_lights(
        self, made: List[str], group: str, parent: Optional[str]
    ) -> Tuple[str, List[str]]:
        staged = self.groups.pop(None, {})
        self.groups[group] = {name: staged[name] for name in made}
        if parent:
            self.parents[group] = parent
        return group, list(made)

    def build_light(self, light: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
        self.groups.setdefault(None, {})[light["name"]] = copy.deepcopy(light)
        return light["name"], []

    def gobo_filter(self, path: str, rotate: Optional[float]) -> Dict[str, Any]:
        out = {"type": "gobo", "textures": {"image": {"file": path}}}
        if rotate is not None:
            out["attrs"] = {"rotate": rotate}
        return out

    def plug(self, node: str, attr: str) -> Optional[Tuple[str, str]]:
        plug = f"{node}.{attr}"
        return plug, plug

    def plug_key(self, plug: str) -> str:
        return plug

    def set_overrides(
        self, node: str, attrs: Dict[str, Any], scene_units: Any = None
    ) -> Tuple[List[str], List[str]]:
        if node not in self.nodes:
            return [f"{node}: not in the scene"], []
        self.nodes[node].update(copy.deepcopy(attrs))
        return [], []

    def current_values(
        self, section: Optional[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for node, attrs in (section or {}).items():
            for attr in attrs:
                if attr in self.nodes.get(node, {}):
                    out.setdefault(node, {})[attr] = copy.deepcopy(
                        self.nodes[node][attr]
                    )
        return out


# --------------------------------------------------------------------------- the engine
class _LightRigInternal(object):
    """The host-free values behind :class:`LightRig`."""

    #: Rotate orders by index (the payload's numbering): the axes in
    #: the order they apply.
    ROTATE_ORDERS = ("xyz", "yzx", "zxy", "xzy", "yxz", "zyx")

    @classmethod
    def _rotation_matrix(cls, rot, order: int = 0) -> List[List[float]]:
        """The 3x3 rotation of Euler *rot* (degrees) in rotate order *order*."""
        out = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        for axis in cls.ROTATE_ORDERS[order or 0]:
            a = math.radians(rot["xyz".index(axis)])
            c, s = math.cos(a), math.sin(a)
            m = {
                "x": [[1, 0, 0], [0, c, -s], [0, s, c]],
                "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
                "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]],
            }[axis]
            out = [
                [sum(m[i][k] * out[k][j] for k in range(3)) for j in range(3)]
                for i in range(3)
            ]
        return out

    @classmethod
    def _same_rotation(cls, a, order_a: int, b, order_b: int, tolerance: float) -> bool:
        """Whether two Eulers (each in its own rotate order) are one orientation."""
        ma, mb = cls._rotation_matrix(a, order_a), cls._rotation_matrix(b, order_b)
        return all(
            abs(ma[i][j] - mb[i][j]) <= tolerance for i in range(3) for j in range(3)
        )

    @staticmethod
    def same(a: Any, b: Any, tolerance: float) -> bool:
        """Whether *a* and *b* are one value: floats within *tolerance* (relative
        above 1), lists and dicts item by item, paths by their normal form."""
        if isinstance(a, bool) or isinstance(b, bool):
            return bool(a) == bool(b)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(
                _LightRigInternal.same(x, y, tolerance) for x, y in zip(a, b)
            )
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(
                _LightRigInternal.same(a[k], b[k], tolerance) for k in a
            )
        if isinstance(a, str) and isinstance(b, str):
            return a == b or os.path.normcase(os.path.normpath(a)) == os.path.normcase(
                os.path.normpath(b)
            )
        return a == b

    @staticmethod
    def _fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(_LightRigInternal._fmt(v) for v in value) + "]"
        return json.dumps(value) if isinstance(value, (dict, str)) else str(value)

    @classmethod
    def _diff_dict(
        cls, prefix: str, a: Dict[str, Any], b: Dict[str, Any], tolerance: float
    ) -> List[str]:
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in b:
                out.append(f"{prefix}{key} {cls._fmt(a[key])} -> (unset)")
            elif key not in a:
                out.append(f"{prefix}{key} (unset) -> {cls._fmt(b[key])}")
            elif not cls.same(a[key], b[key], tolerance):
                out.append(f"{prefix}{key} {cls._fmt(a[key])} -> {cls._fmt(b[key])}")
        return out

    @staticmethod
    def _plugs(section: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        return {
            f"{node}.{attr}": v
            for node, attrs in (section or {}).items()
            for attr, v in attrs.items()
        }

    @staticmethod
    def _rounded(value: Any) -> Any:
        if isinstance(value, float):
            return float(f"{value:.5g}") + 0.0  # + 0.0: -0.0 and 0.0 print alike
        if isinstance(value, (list, tuple)):
            return [_LightRigInternal._rounded(v) for v in value]
        if isinstance(value, dict):
            return {k: _LightRigInternal._rounded(v) for k, v in value.items()}
        return value

    @staticmethod
    def _mapped(table: Dict[str, Any], names: Tuple[str, ...]) -> Dict[str, Any]:
        """*names* mapped through a binding's *table*: a KeyError on a name the
        contract gained that the binding does not map yet."""
        return dict(zip(names, map(table.__getitem__, names)))


class LightRig(_LightRigInternal, LoggingMixin):
    """Save, manage, swap and restore named light rigs (see the module docstring).

    A host library binds it on a class of its own: :attr:`HOST` (its
    :class:`LightRigHost`) and its tables -- :attr:`HOST_TYPES` and
    :attr:`HOST_SETTINGS` (the neutral light types and settings,
    ``ptk.LightRigLight``, onto its own), :attr:`HOST_FIXTURE_MODES` and
    :attr:`DIFFUSE_VISIBILITY` (``ptk.LightRigFixture`` onto its attributes)
    and :attr:`KELVIN_ATTRS`. The neutral tables (:attr:`TYPES`,
    :attr:`FRIENDLY`, :attr:`FIXTURE_MODES`) are built from them when the
    binding is defined, and a name the contract declares that a table does not
    map fails it.

    Parameters:
        group: The scene group a rig's lights live under when neither the call
            nor the payload names one.
        store: The rig library; default ``ptk.LightRigPayload.library()``,
            shared with every scene, host and tool.

    Example:
        >>> rig = LightRig(group="MOODY_LIGHTS")
        >>> rig.save("moody_v1", description="first pass")
        >>> rig.apply("moody_v2")   # the scene's rig swapped; the outgoing one kept
        >>> rig.swap_back()         # and back again
    """

    PAYLOAD_VERSION = LightRigPayload.VERSION
    DEFAULT_GROUP = "LIGHT_RIG"

    #: The host adapter every scene call goes through; a host library's
    #: binding sets its own.
    HOST: ClassVar[LightRigHost] = PureLightRigHost()
    #: Neutral light type -> the host's light type (every ``LightRigLight.TYPES``).
    HOST_TYPES: ClassVar[Dict[str, str]] = {t: t for t in LightRigLight.TYPES}
    #: Neutral setting -> the host's attribute (every ``LightRigLight.SETTINGS``).
    HOST_SETTINGS: ClassVar[Dict[str, str]] = {s: s for s in LightRigLight.SETTINGS}
    #: Fixture mode -> ``(its meshes seen by diffuse rays, its lamps on)``.
    HOST_FIXTURE_MODES: ClassVar[Dict[str, Tuple[bool, bool]]] = {
        "glow": (False, False),
        "emissive": (True, False),
        "lamp": (False, True),
    }
    #: What lets a mesh's emission light the room: its visibility to diffuse rays.
    DIFFUSE_VISIBILITY: ClassVar[Tuple[str, ...]] = ("diffuse_visibility",)
    #: ``(the switch, the temperature)`` a neutral ``kelvin`` sets.
    KELVIN_ATTRS: ClassVar[Tuple[str, str]] = ("use_kelvin", "kelvin")

    #: Neutral light types -> the host's (built from :attr:`HOST_TYPES`).
    TYPES: ClassVar[Dict[str, str]] = {}
    #: Neutral settings -> the host's attributes (built from :attr:`HOST_SETTINGS`).
    FRIENDLY: ClassVar[Dict[str, str]] = {}
    #: Fixture modes -> ``(meshes seen by diffuse rays, lamps on)``.
    FIXTURE_MODES: ClassVar[Dict[str, Tuple[bool, bool]]] = {}
    #: Keys a light may carry that build nothing (look-match notes).
    NOTES = LightRigLight.NOTES
    #: The contract's structural keys: everything a light declares but its settings and notes.
    _LIGHT_KEYS = tuple(
        f.name
        for f in dataclasses.fields(LightRigLight)
        if f.name not in LightRigLight.SETTINGS + LightRigLight.NOTES
    )
    #: Payload keys the engine reads; every other top-level key is carried untouched.
    _OWN_KEYS = ("version", "group", "lights", "existing", "materials", "applied")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._bind_tables()

    @classmethod
    def _bind_tables(cls) -> None:
        """The neutral tables, from the binding's own (a KeyError names a gap)."""
        cls.TYPES = cls._mapped(cls.HOST_TYPES, LightRigLight.TYPES)
        cls.FRIENDLY = cls._mapped(cls.HOST_SETTINGS, LightRigLight.SETTINGS)
        cls.FIXTURE_MODES = cls._mapped(cls.HOST_FIXTURE_MODES, LightRigFixture.MODES)

    def __init__(
        self,
        group: str = DEFAULT_GROUP,
        store: Optional["PresetStore"] = None,
        log_level: str = "WARNING",
    ):
        super().__init__()
        self.logger.setLevel(log_level)
        self.group = group
        self._store = store

    @property
    def store(self) -> "PresetStore":
        """The rig library."""
        if self._store is None:
            self._store = LightRigPayload.library()
        return self._store

    # ------------------------------------------------------------- payloads
    @staticmethod
    def aim_rotation(pos, aim) -> List[float]:
        """World rotation (xyz, degrees) pointing a light's -Z from *pos* at *aim*."""
        d = [a - p for a, p in zip(aim, pos)]
        n = math.sqrt(sum(c * c for c in d)) or 1.0
        dx, dy, dz = (c / n for c in d)
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, dy))))
        yaw = math.degrees(math.atan2(-dx, -dz))
        return [pitch, yaw, 0.0]

    @classmethod
    def normalize(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        """*payload* in the canonical form (module docstring); neutral spellings translated.

        Raises:
            ValueError: the payload is newer than this version reads, or a light
                carries a key nothing builds from.
        """
        version = payload.get("version", cls.PAYLOAD_VERSION)
        if version > cls.PAYLOAD_VERSION:
            raise ValueError(
                f"Light rig payload version {version} is newer than this "
                f"{cls.HOST.package} reads ({cls.PAYLOAD_VERSION}); update "
                f"{cls.HOST.package}."
            )
        out = copy.deepcopy(payload)
        out["version"] = cls.PAYLOAD_VERSION
        out["lights"] = [
            cls._normalize_light(light) for light in payload.get("lights") or []
        ]
        existing = cls.fixture_overrides(payload.get("fixtures"))
        for node, attrs in (payload.get("existing") or {}).items():  # named outright: wins
            existing.setdefault(node, {}).update(
                {cls.FRIENDLY.get(k, k): v for k, v in attrs.items()}
            )
        out["existing"] = existing
        out["materials"] = copy.deepcopy(payload.get("materials") or {})
        return out

    @classmethod
    def fixture_overrides(
        cls, fixtures: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        """The overrides that put each fixture in its mode (``ptk.LightRigFixture``):
        ``{node: {attr: value}}`` on its meshes (seen by diffuse rays or not)
        and its lamps (on at their intensity, or off). :meth:`normalize` sets
        them as ``existing`` overrides, under what the rig names outright, so
        apply, compare, swap back and undo treat them like any other.

        Raises:
            ValueError: a fixture outside the contract (an unknown mode, no meshes).
        """
        out: Dict[str, Dict[str, Any]] = {}
        intensity = cls.FRIENDLY["intensity"]
        for fixture in fixtures or []:
            checked = LightRigFixture.validate(fixture)
            if checked.errors:
                raise ValueError(
                    f"Fixture {fixture.get('name')!r}: " + "; ".join(checked.errors)
                )
            lit, lamps_on = cls.FIXTURE_MODES[fixture["mode"]]
            for mesh in fixture["meshes"]:
                out.setdefault(mesh, {}).update(
                    {attr: lit for attr in cls.DIFFUSE_VISIBILITY}
                )
            for lamp, level in (fixture.get("lamps") or {}).items():
                out.setdefault(lamp, {})[intensity] = float(level) if lamps_on else 0.0
        return out

    @classmethod
    def _normalize_light(cls, light: Dict[str, Any]) -> Dict[str, Any]:
        unknown = set(light) - set(cls._LIGHT_KEYS) - set(cls.FRIENDLY) - set(cls.NOTES)
        if unknown:
            raise ValueError(
                f"Light {light.get('name')!r}: unknown key(s) {sorted(unknown)}."
            )
        attrs = copy.deepcopy(light.get("attrs") or {})
        for key, attr in cls.FRIENDLY.items():
            if key in light:
                attrs[attr] = copy.deepcopy(light[key])
        if "kelvin" in light and cls.FRIENDLY["color"] not in attrs:
            switch, temperature = cls.KELVIN_ATTRS
            attrs.update({switch: 1, temperature: light["kelvin"]})
        out = {
            "name": light["name"],
            "type": cls.TYPES.get(light["type"], light["type"]),
            "pos": list(light.get("pos") or (0, 0, 0)),
        }
        if light.get("rot") is not None:
            out["rot"] = list(light["rot"])
        elif light.get("aim") is not None:
            out["rot"] = cls.aim_rotation(out["pos"], light["aim"])
        for key in ("scale", "rotate_order", "visible", "textures", "user_attrs"):
            if light.get(key) not in (None, {}, [], 0) or (
                key == "visible" and light.get(key) is False
            ):
                out[key] = copy.deepcopy(light[key])
        if attrs:
            out["attrs"] = attrs
        filters = copy.deepcopy(light.get("filters") or [])
        if light.get("gobo"):
            filters.append(cls.HOST.gobo_filter(light["gobo"], light.get("gobo_rotate")))
        if filters:
            out["filters"] = filters
        return out

    @classmethod
    def diff(
        cls, a: Dict[str, Any], b: Dict[str, Any], tolerance: float = 1e-4
    ) -> List[str]:
        """What changes from rig *a* to rig *b*, one line each (empty when they match).

        Floats match within *tolerance* (relative above 1); an attribute set on
        one side only reads as ``(unset)`` -- compare two captures, or a capture
        with a saved capture, for a default-blind comparison.
        """
        a, b = cls.normalize(a), cls.normalize(b)
        la = {light["name"]: light for light in a["lights"]}
        lb = {light["name"]: light for light in b["lights"]}
        out = [f"- {n} ({la[n]['type']})" for n in sorted(set(la) - set(lb))]
        out += [f"+ {n} ({lb[n]['type']})" for n in sorted(set(lb) - set(la))]
        for name in sorted(set(la) & set(lb)):
            x, y = la[name], lb[name]
            if x["type"] != y["type"]:
                out.append(f"{name}: type {x['type']} -> {y['type']}")
            if not cls._same_rotation(
                x.get("rot") or [0, 0, 0],
                x.get("rotate_order", 0),
                y.get("rot") or [0, 0, 0],
                y.get("rotate_order", 0),
                tolerance,
            ):
                out.append(
                    f"{name}: rot {cls._fmt(x.get('rot') or [0, 0, 0])} -> {cls._fmt(y.get('rot') or [0, 0, 0])}"
                )
            for key, default in (
                ("pos", [0, 0, 0]),
                ("scale", [1, 1, 1]),
                ("rotate_order", 0),
                ("visible", True),
            ):
                if not cls.same(x.get(key, default), y.get(key, default), tolerance):
                    out.append(
                        f"{name}: {key} {cls._fmt(x.get(key, default))} -> {cls._fmt(y.get(key, default))}"
                    )
            for key in ("attrs", "textures", "user_attrs"):
                prefix = (
                    f"{name}: "
                    if key == "attrs"
                    else f"{name}: {key[:-1].replace('_', ' ')} "
                )
                out += cls._diff_dict(
                    prefix, x.get(key) or {}, y.get(key) or {}, tolerance
                )
            fx, fy = x.get("filters") or [], y.get("filters") or []
            if [f["type"] for f in fx] != [f["type"] for f in fy]:
                out.append(
                    f"{name}: filters {[f['type'] for f in fx]} -> {[f['type'] for f in fy]}"
                )
                continue
            for i, (f, g) in enumerate(zip(fx, fy)):
                for key in ("attrs", "textures"):
                    out += cls._diff_dict(
                        f"{name}: {f['type']}[{i}] ",
                        f.get(key) or {},
                        g.get(key) or {},
                        tolerance,
                    )
        for section in ("existing", "materials"):
            out += cls._diff_dict(
                f"{section} ",
                cls._plugs(a.get(section)),
                cls._plugs(b.get(section)),
                tolerance,
            )
        return out

    @classmethod
    def fingerprint(cls, payload: Dict[str, Any]) -> str:
        """A short hash of a rig's lights, blind to float noise (what "edited" compares)."""
        lights = cls._rounded(cls.normalize(payload)["lights"])
        return hashlib.sha1(
            json.dumps(lights, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]

    @classmethod
    def landed_overrides(
        cls, payload: Dict[str, Any]
    ) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], List[str]]:
        """*payload*'s overrides as they land: ``({section: {node: {attr:
        value}}}, yielded)``.

        An override of a plug that follows another (:meth:`LightRigHost.plug`)
        lands on the plug it follows. Where the payload also names that plug
        itself, its own value wins and the follower is dropped -- one
        *yielded* line when the two disagree. Apply, compare and a layer's
        emission target read the overrides through this.
        """
        host = cls.HOST
        sections = {s: payload.get(s) or {} for s in ("existing", "materials")}
        named = {
            host.plug_key(f"{node}.{attr}"): value
            for nodes in sections.values()
            for node, attrs in nodes.items()
            if host.exists(node)
            for attr, value in attrs.items()
        }
        landed: Dict[str, Dict[str, Dict[str, Any]]] = {}
        yielded: List[str] = []
        for section, nodes in sections.items():
            for node, attrs in nodes.items():
                kept = {}
                for attr, value in attrs.items():
                    found = host.plug(node, attr) if host.exists(node) else None
                    follows = found and found[1] != found[0]
                    key = host.plug_key(found[1]) if follows else None
                    if key in named:
                        if named[key] != value:
                            yielded.append(
                                f"{node}.{attr}: follows {found[1]}, which the rig "
                                f"sets to {named[key]!r}"
                            )
                        continue
                    kept[attr] = value
                if kept:
                    landed.setdefault(section, {})[node] = kept
        return landed, yielded

    # ------------------------------------------------------------- the scene
    def capture(self, group: Optional[str] = None) -> Dict[str, Any]:
        """The lights under *group* as a payload (their overrides are not included:
        see :meth:`save`), in centimetres and degrees whatever the scene's units."""
        group = group or self.group
        with self.HOST.payload_units():
            lights = self.HOST.capture_lights(group)
        return {
            "version": self.PAYLOAD_VERSION,
            "group": group,
            "lights": sorted(lights, key=lambda light: light["name"]),
        }

    # ------------------------------------------------------------- scene state
    @classmethod
    def _record(cls) -> Dict[str, Any]:
        """This scene's ``SceneRecords.LIGHT_RIGS``: ``{group: entry}``."""
        return dict(SceneRecords.LIGHT_RIGS.load(cls.HOST.record_store(), {}) or {})

    @staticmethod
    def _key(group: str) -> str:
        return group.split("|")[-1]

    def _entry(self, group: str) -> Dict[str, Any]:
        """The record's entry for *group*: ``rig``, ``print``, ``extras``, ``previous``."""
        return dict(self._record().get(self._key(group)) or {})

    def _write_entry(self, group: str, entry: Dict[str, Any]) -> None:
        record = self._record()
        record[self._key(group)] = entry
        SceneRecords.LIGHT_RIGS.save(self.HOST.record_store(), record)

    def applied(self, group: Optional[str] = None) -> Tuple[Optional[str], bool]:
        """``(rig name, edited since)`` for the rig in *group*; ``(None, False)`` when none went in."""
        group = group or self.group
        entry = self._entry(group)
        if not self.HOST.exists(group) or "print" not in entry:
            return None, False
        name = entry.get("rig") or None
        return name, self.fingerprint(self.capture(group)) != entry["print"]

    def compare(
        self, rig: Union[str, Dict[str, Any]], group: Optional[str] = None
    ) -> List[str]:
        """What applying *rig* would change in the scene, one line each (empty: nothing).

        Unlike :meth:`diff`, an attribute one side leaves at its default is
        compared as that default, and only the plugs *rig* overrides count.
        The scene is read in the payload's centimetres and degrees.
        """
        with self.HOST.payload_units():
            target = self.normalize(self.load(rig) if isinstance(rig, str) else rig)
            group = group or target.get("group") or self.group
            scene = self.capture(group)
            landed, _yielded = self.landed_overrides(target)
            for section in ("existing", "materials"):
                target[section] = landed.get(section) or {}
                scene[section] = self.HOST.current_values(target[section])
            return self.diff(
                self._with_defaults(scene, target), self._with_defaults(target, scene)
            )

    @classmethod
    def _with_defaults(
        cls, payload: Dict[str, Any], other: Dict[str, Any]
    ) -> Dict[str, Any]:
        """*payload* with every attribute *other* sets on a same-named light filled in at its default."""
        out = cls.normalize(payload)
        twins = {light["name"]: light for light in cls.normalize(other)["lights"]}
        for light in out["lights"]:
            twin = twins.get(light["name"])
            if not twin or twin["type"] != light["type"]:
                continue
            defaults, parents = cls.HOST.defaults(light["type"])
            attrs = light.setdefault("attrs", {})
            for attr in twin.get("attrs") or {}:
                if attr in attrs:
                    continue
                children = [c for c, p in parents.items() if p == attr]
                if children:
                    attrs[attr] = [defaults[c] for c in children]
                elif attr in defaults:
                    attrs[attr] = defaults[attr]
        return out

    def scene_rigs(self) -> List[Tuple[str, Optional[str]]]:
        """``(group, rig name)`` for every rig group in the scene."""
        return [
            (group, entry.get("rig") or None)
            for group, entry in self._record().items()
            if "print" in entry and self.HOST.exists(group)
        ]

    def can_swap_back(self, group: Optional[str] = None) -> bool:
        """Whether :meth:`swap_back` has a rig to return to (in *group*, or any group)."""
        record = self._record()
        if group:
            return bool((record.get(self._key(group)) or {}).get("previous"))
        return any(entry.get("previous") for entry in record.values())

    def remove(self, group: Optional[str] = None) -> List[str]:
        """Delete *group*, its lights, and what only they used."""
        group = group or self.group
        if not self.HOST.exists(group):
            return []
        with self.HOST.undo_step("Light Rig: Remove"):
            return self.HOST.remove_group(group)

    def apply(
        self,
        rig: Union[str, Dict[str, Any]],
        group: Optional[str] = None,
        snapshot: bool = True,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Swap the scene's rig for *rig* (a library name or a payload).

        The lights under the group are replaced; the overrides are set on the
        nodes they name. With *snapshot*, the outgoing rig -- its lights and the
        current values of every plug either rig overrides -- is kept first, so
        :meth:`swap_back` returns to it. One undo reverts the swap. The rig is
        read in centimetres and degrees whatever the scene's units; the scene
        keeps its own.

        Returns:
            dict: ``name``, ``group``, ``lights`` (built), ``skipped``
            (overrides, attributes and lights left alone, with the reason) and
            ``disconnected`` (driven plugs the rig took over; swap back and
            undo reconnect them).
        """
        # A driver the rig hands back is reconnected in these (set_overrides).
        scene_units = self.HOST.scene_units()
        with self.HOST.payload_units():
            if isinstance(rig, str):
                name = name or rig
                rig = self.load(rig)
            canon = self.normalize(rig)
            canon.pop("applied", None)
            group = group or canon.get("group") or self.group
            kinds = {light["type"] for light in canon["lights"]}
            kinds |= {
                f["type"]
                for light in canon["lights"]
                for f in light.get("filters") or []
            }
            self.HOST.prepare(kinds, group)  # outside the swap's undo step
            previous = None
            if snapshot:
                held, _ = self.applied(group)
                previous = self._snapshot(group, also=canon)
                if held:
                    previous["applied"] = held
            return self._build(canon, group, name, previous, scene_units)

    def swap_back(self, group: Optional[str] = None) -> Dict[str, Any]:
        """Return to the rig the last :meth:`apply` replaced in *group* (again to toggle back).

        With no *group*: the one group in this scene with a rig to return to.

        Raises:
            LookupError: nothing has been replaced in this scene, or several
                groups could swap back and none was named.
        """
        if group is None:
            groups = [g for g, e in self._record().items() if e.get("previous")]
            if len(groups) > 1:
                raise LookupError(
                    f"Several rig groups can swap back ({', '.join(groups)}): name one."
                )
            group = groups[0] if groups else self.group
        previous = self._entry(group).get("previous")
        if not previous:
            raise LookupError(
                "No rig to swap back to: no Apply has replaced one in this scene."
            )
        return self.apply(previous, group=group, name=previous.get("applied"))

    def _build(
        self,
        canon: Dict[str, Any],
        group: str,
        name: Optional[str],
        previous: Optional[Dict[str, Any]] = None,
        scene_units: Any = None,
    ) -> Dict[str, Any]:
        host = self.HOST
        report = {
            "name": name,
            "group": group,
            "lights": [],
            "skipped": [],
            "disconnected": [],
        }
        with host.undo_step("Light Rig: Apply"):
            kept = self._entry(group).get("previous")  # a build without a snapshot keeps it
            # The group keeps its place: a scene parks its rig under a group the exporter skips.
            parent = host.group_parent(group) if host.exists(group) else None
            self.remove(group)
            landed, report["skipped"] = self.landed_overrides(canon)
            for section in ("existing", "materials"):
                for node, attrs in (landed.get(section) or {}).items():
                    skipped, disconnected = host.set_overrides(node, attrs, scene_units)
                    report["skipped"] += skipped
                    report["disconnected"] += disconnected
            made = []
            for light in canon["lights"]:
                built, skipped = host.build_light(light)
                if built:
                    made.append(built)
                report["skipped"] += skipped
            report["group"], report["lights"] = host.group_lights(made, group, parent)
            entry = {
                "rig": name or "",
                "print": self.fingerprint(self.capture(report["group"])),
                "extras": {
                    k: v
                    for k, v in canon.items()
                    if k not in ("version", "group", "lights", "applied")
                },
            }
            if previous is not None or kept:
                entry["previous"] = previous if previous is not None else kept
            self._write_entry(report["group"], entry)  # inside the undo step: undo restores it
        return report

    def _snapshot(
        self, group: str, also: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """The rig in *group* as a payload: its lights, the current values of the
        plugs it (and *also*) overrides, and the sections it was applied with."""
        payload = self.capture(group)
        extras = self._entry(group).get("extras") or {}
        for section in ("existing", "materials"):
            plugs = {**{n: dict(a) for n, a in (extras.get(section) or {}).items()}}
            for node, attrs in ((also or {}).get(section) or {}).items():
                plugs.setdefault(node, {}).update(attrs)
            values = self.HOST.current_values(plugs)
            if values:
                payload[section] = values
        for key, value in extras.items():
            if key not in ("existing", "materials"):
                payload[key] = value
        return payload

    # ------------------------------------------------------------- the library
    def names(self) -> List[str]:
        """The saved rigs, sorted."""
        return sorted(self.store.list())

    def load(self, name: str) -> Dict[str, Any]:
        """A saved rig's payload. Raises ``KeyError`` when there is none."""
        return self.store.load(name)

    def save(
        self,
        name: str,
        description: Optional[str] = None,
        group: Optional[str] = None,
        force: bool = False,
    ) -> str:
        """Save the rig in *group* as *name*; returns the file written.

        The payload holds the group's lights, the current values of the plugs
        the rig it came from overrides, and that rig's carried sections, in
        centimetres and degrees whatever the scene's units; the library entry
        records *description* and the rig it came from (``parent``).

        Raises:
            PermissionError: *name* is locked (``PresetReadOnlyError``).
        """
        group = group or self.group
        with self.HOST.payload_units():
            payload = self._snapshot(group)
            parent, _ = self.applied(group)
        path = self.store.save(name, payload, force=force)
        fields = {"lights": len(payload["lights"]), "group": group}
        if description is not None:
            fields["description"] = description
        if parent and parent != name:
            fields["parent"] = parent
        self.store.set_info(name, **fields)
        if self.HOST.exists(group):  # the scene's rig IS this saved rig now
            entry = self._entry(group)
            entry.update(
                rig=name,
                print=self.fingerprint(payload),
                extras={
                    k: v
                    for k, v in payload.items()
                    if k not in ("version", "group", "lights")
                },
            )
            self._write_entry(group, entry)
        return str(path)

    def delete(self, name: str) -> bool:
        """Delete a saved rig (a locked one stays). Returns ``True`` when one went."""
        return self.store.delete(name)

    def rename(self, old: str, new: str) -> bool:
        """Rename a saved rig; ``False`` when *new* is taken or *old* is locked or missing."""
        return self.store.rename(old, new)


LightRig._bind_tables()
