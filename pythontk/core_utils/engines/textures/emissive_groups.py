# !/usr/bin/python
# coding=utf-8
"""Emissive groups: named face sets whose glow a game engine plays, as materials.

A group is materials, authored at the source: its faces wear a copy of their
material made for the group -- one per material and group, named
``<base>_<group>`` -- sharing the base's maps, its emission keyed by the
group's weight (0..1) so the viewport shows what ships. In the delivery each
copy is a plain material (the ``materials`` record) and each animated one a
``render_effects`` material entry whose ``emission`` curve rides the
animated-value transport (``ptk.SceneRecords.TRANSPORT``: a
``<material>__emission`` proxy under the export's carrier node, its
``scale.x`` the weight). Costs, by decision: one submesh per material and
group, face-level regions, no face in two groups (a face wears one material).

State lives on the materials themselves: a copy is tagged with its group, its
base and the base's emission strength it was keyed against. The weight is the
copy's emission divided by that strength: keyed, the animation owns it.

A tint multiplies a copy's emission colour (white: none), and a state is a
named tint and weight (an indicator's off, green, amber, red...): emission is
colour x tint x weight x strength. A keyed tint rides the transport as the
``tint_r``, ``tint_g`` and ``tint_b`` channels; the states and the tint the
delivered material shows ride its ``render_effects`` entry.

In step with its base: a changed base value or map reaches its copies when
the tool syncs them -- :meth:`EmissiveGroupsModel.sync`, run by the panel's
Sync, never implicitly: no other operation changes a copy to match its base.
It runs as a check, changing nothing, in :meth:`EmissiveGroupsModel.validate`
(the panel's Check) and before an export, where the stager says each copy out
of step and ships it as it stands (:meth:`stage_export_proxies`: never a
swap at export).

Named as the delivery names it: a copy's ``render_effects`` entry names the
material as the FBX and the GLB carry it, namespace included (measured
2026-11-07 on a real export: a copy imported under ``ns`` ships as
``ns:<copy>``), and its proxy and curve go by the transport's name for it
(``ptk.SceneRecords.TRANSPORT.target_name``: the leaf), which every reader
joins on. A shader whose scalar only gates its emission (OpenPBR's luminance,
in nits) is stated at its family's published weight
(:meth:`EmissiveGroupsHost.published_strength`), as the ``materials`` record
states its base.

The old tool (masks: a vertex-colour set or an ``_EMask`` texture over slots,
a registry on the internal carrier, an ``emissive_groups`` record) converts in
one action: :meth:`EmissiveGroupsModel.convert_legacy`.

This module is host-free: every scene read and write goes through the host
adapter (:class:`EmissiveGroupsHost`), which speaks faces as
``{object: [face index, ...]}`` and materials by name; each host library's
``EmissiveGroups`` is this model bound to its host's half (built on
:class:`PureEmissiveGroupsHost`).
"""

import contextlib
import json
import re
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import pythontk as ptk

#: ``{object: [face index, ...]}``: faces as the model speaks them.
FaceMap = Dict[str, List[int]]
#: ``(frame, value)`` keys of a material's emission, in the host's frames.
Keys = List[Tuple[float, float]]


@runtime_checkable
class EmissiveGroupsHost(Protocol):
    """A host's half of Emissive Groups: what :class:`EmissiveGroupsModel` asks
    of the scene. :class:`PureEmissiveGroupsHost` holds the answers with no
    scene, and each host's adapter is built on it.

    A material's *emission* is the shader's own emission scalar (the value
    that scales its emission colour or map); a copy's *tag* names its group,
    its base material and the base's emission strength the copy was keyed
    against.
    """

    #: The host's scene-record store (``DataNodes``): read only to find and
    #: clear the old tool's records.
    data_nodes: Any

    # Faces and materials -------------------------------------------------

    def faces(self, faces=None) -> FaceMap:
        """*faces* (a ``FaceMap``, ``[]`` for every face, or the host's own
        spelling) or, for ``None``, the selected faces; empty when none."""

    def face_materials(self, face_map: FaceMap) -> Dict[str, FaceMap]:
        """The faces of *face_map* by the material each wears now."""

    def material_faces(self, material: str) -> FaceMap:
        """Every face wearing *material*."""

    def face_count(self, face_map: FaceMap) -> int:
        """How many faces *face_map* names, a face several objects show (one
        mesh under linked duplicates or instances) counted once."""

    def material(self, name: str) -> Optional[str]:
        """The material *name* resolves to here, or ``None``."""

    def assign(self, material: str, face_map: FaceMap) -> None:
        """Make the faces of *face_map* wear *material*."""

    def select(self, face_map: FaceMap) -> None:
        """Select exactly these faces."""

    # Copies --------------------------------------------------------------

    def copies(self) -> Dict[str, Dict[str, Any]]:
        """Every tagged copy: ``{copy: {"group", "base", "strength"}}``,
        ``base`` ``None`` when the base material is gone."""

    def make_copy(self, base: str, name: str) -> str:
        """A new material copied from *base*, sharing its maps, named *name*
        (or the host's unique spelling of it), wearing no face; untagged."""

    def tag(self, copy: str, group: str, base: str, strength: float) -> None:
        """Stamp *copy* as *group*'s copy of *base*, keyed against *strength*."""

    def delete_material(self, material: str) -> None:
        """Delete *material* (and what the host made for it alone)."""

    def material_state(self, material: str) -> Dict[str, Any]:
        """Every value and map of *material* but its emission scalar and its
        tag: what a copy keeps in step with its base, compared key by key."""

    def sync_copy(self, copy: str, base: str) -> None:
        """Make *copy*'s state *base*'s; its emission, keys and tag stay."""

    # Emission ------------------------------------------------------------

    def emission(self, material: str) -> Optional[float]:
        """*material*'s emission scalar now; ``None`` for a shader with none."""

    def emission_color(self, material: str) -> Optional[List[float]]:
        """*material*'s uniform emission colour, linear RGB; ``None`` when a
        map drives it."""

    def set_emission(self, material: str, value: float) -> None:
        """Set *material*'s emission scalar (its keys, if any, still own it)."""

    def emission_keys(self, material: str) -> Keys:
        """*material*'s emission keys; empty when it is not keyed."""

    def emission_scale(self, material: str) -> float:
        """What the plug the host keys is scaled by against :meth:`emission`:
        1 where it keys the emission scalar itself, a colour channel's
        reference where it keys a colour (the scalar is that channel over it)."""

    def published_strength(self, material: str, strength: float) -> float:
        """*strength* -- *material*'s emission scalar (:meth:`emission`) -- as
        the delivery states it: the ``materials`` record's emission weight
        for its shader. The scalar itself where it scales the emission (a
        Standard Surface's weight, a Principled BSDF's strength); 1 (0 at 0)
        where it only gates it -- a luminance in nits (OpenPBR's) never
        scales the delivered emission."""

    def key_emission(
        self, material: str, value: float, frame: Optional[float] = None, stepped: bool = False
    ) -> None:
        """Key *material*'s emission at *value*, at *frame* (``None``: now);
        *stepped* holds it until the next key (a state's switch)."""

    def clear_emission_keys(self, material: str) -> None:
        """Delete *material*'s emission keys, keeping its value now."""

    def stepped_emission_frames(self, material: str) -> List[float]:
        """The frames of *material*'s emission keys that hold until the next
        (a state's switch); empty when none does."""

    # Tint: the colour a copy's emission colour is multiplied by --------------

    def tint_refusal(self, material: str) -> Optional[str]:
        """Why *material*'s emission cannot be tinted (a mapped colour, a
        shader that glows by its colour alone), or ``None`` when it can."""

    def emission_tint(self, material: str) -> List[float]:
        """*material*'s tint now: its emission colour over the base colour it
        was tinted against; white for one never tinted."""

    def set_emission_tint(self, material: str, tint: Sequence[float]) -> None:
        """Make *material*'s emission colour its base colour times *tint* (its
        tint keys, if any, still own it)."""

    def key_emission_tint(
        self, material: str, tint: Sequence[float], frame: Optional[float] = None, stepped: bool = False
    ) -> None:
        """Key *material*'s tint at *frame* (``None``: now), each channel."""

    def emission_tint_keys(self, material: str) -> List[Tuple[float, List[float], bool]]:
        """*material*'s tint keys, ``(frame, [r, g, b], stepped)``; empty when
        its tint is not keyed."""

    def clear_emission_tint_keys(self, material: str) -> None:
        """Delete *material*'s tint keys, keeping its tint now."""

    def emission_tint_channels(self, material: str) -> List[Any]:
        """What keys each of *material*'s three tint channels (a plug, an
        fcurve), keyed in its colour's units: over :meth:`emission_color`'s
        channel, the tint. Empty when it cannot be tinted."""

    def emission_states(self, material: str) -> Dict[str, Dict[str, Any]]:
        """The named states stored on *material* (:meth:`set_emission_states`)."""

    def set_emission_states(self, material: str, states: Dict[str, Dict[str, Any]]) -> None:
        """Store *states* (``{name: {"tint", "weight"}}``) on *material*."""

    def copy_emission(self, src: str, dst: str, factor: float) -> None:
        """*dst*'s emission becomes *src*'s times *factor*: its value, and its
        keys with their shape."""

    def scale_emission(self, material: str, factor: float) -> None:
        """Scale *material*'s emission value and keys by *factor*."""

    # Export --------------------------------------------------------------

    def stage_emission_proxy(
        self, material: str, name: str, factor: float, tint: Optional[int] = None
    ) -> Optional[Any]:
        """Stage the transport proxy *name* under the export's carrier node:
        a marked transform whose curve channel carries *material*'s emission
        keys times *factor* -- with *tint* (0, 1, 2), that channel of its
        tint's keys instead (its colour's over :meth:`emission_color`'s, times
        *factor*). ``None`` when *name* is taken."""

    def emission_proxies(self) -> List[str]:
        """The names of the proxies :meth:`stage_emission_proxy` staged."""

    def remove_emission_proxies(self, names: Optional[Iterable[str]] = None) -> List[str]:
        """Delete the staged proxies named *names* (``None``: every one), and
        forget them wherever the host's writer keeps its staged nodes."""

    def undo_step(self, label: str) -> ContextManager[Any]:
        """One undo step named *label* around an operation."""

    # The old tool's data (conversion) -----------------------------------

    def legacy_store_names(self, prefix: str) -> List[str]:
        """The group names of the old membership stores (``<prefix><name>``)."""

    def legacy_members(self, store: str) -> Optional[FaceMap]:
        """An old membership store's faces; ``None`` when it is gone."""

    def adopt_legacy_weight(self, attr: str, copy: str, factor: float) -> bool:
        """Key *copy*'s emission with the old carrier weight *attr*'s keys
        times *factor*, their shape kept; ``False`` when it has none."""

    def remove_legacy(self, stores: List[str], attrs: List[str], color_set: str) -> List[str]:
        """Delete the old stores, carrier weights (keys included) and every
        mesh's *color_set*; returns what was removed."""

    def extra_warnings(self) -> List[str]:
        """The host's own validation warnings (leftover export proxies)."""


class PureEmissiveGroupsHost:
    """The :class:`EmissiveGroupsHost` with no scene behind it: the pure
    defaults, and the base each host's adapter is built on. It never guesses:
    every read finds nothing -- no selection, material, copy or key -- and
    every write refuses (``RuntimeError``), naming the binding: with no scene
    there is nowhere to keep a group, and a write that went nowhere would read
    as done."""

    data_nodes: Any = None

    @staticmethod
    def _refuse(what: str) -> None:
        raise RuntimeError(
            f"No scene to {what} in: Emissive Groups runs on a host's scene "
            "(a host library's EmissiveGroups, or an EmissiveGroupsHost bound "
            "as EmissiveGroupsModel.HOST)."
        )

    def faces(self, faces=None) -> FaceMap:
        if faces is None:
            return {}  # no scene, no selection
        return {str(obj): [int(i) for i in indices] for obj, indices in dict(faces).items()}

    def face_materials(self, face_map: FaceMap) -> Dict[str, FaceMap]:
        self._refuse("read the faces' materials")
        return {}

    def material_faces(self, material: str) -> FaceMap:
        return {}

    def face_count(self, face_map: FaceMap) -> int:
        return sum(len(set(indices)) for indices in face_map.values())

    def material(self, name: str) -> Optional[str]:
        return None

    def assign(self, material: str, face_map: FaceMap) -> None:
        self._refuse(f"assign {material!r}")

    def select(self, face_map: FaceMap) -> None:
        self._refuse("select faces")

    def copies(self) -> Dict[str, Dict[str, Any]]:
        return {}

    def make_copy(self, base: str, name: str) -> str:
        self._refuse(f"copy {base!r}")
        return ""

    def tag(self, copy: str, group: str, base: str, strength: float) -> None:
        self._refuse(f"tag {copy!r}")

    def delete_material(self, material: str) -> None:
        return None  # no material to delete

    def material_state(self, material: str) -> Dict[str, Any]:
        return {}

    def sync_copy(self, copy: str, base: str) -> None:
        self._refuse(f"sync {copy!r}")

    def emission(self, material: str) -> Optional[float]:
        return None

    def emission_color(self, material: str) -> Optional[List[float]]:
        return None

    def set_emission(self, material: str, value: float) -> None:
        self._refuse(f"set {material!r}'s emission")

    def emission_keys(self, material: str) -> Keys:
        return []

    def emission_scale(self, material: str) -> float:
        return 1.0

    def published_strength(self, material: str, strength: float) -> float:
        return float(strength)  # a scalar that scales: the delivery states it

    def key_emission(
        self, material: str, value: float, frame: Optional[float] = None, stepped: bool = False
    ) -> None:
        self._refuse(f"key {material!r}'s emission")

    def clear_emission_keys(self, material: str) -> None:
        return None  # no keys to delete

    def stepped_emission_frames(self, material: str) -> List[float]:
        return []

    def tint_refusal(self, material: str) -> Optional[str]:
        return f"{material!r} is in no scene"

    def emission_tint(self, material: str) -> List[float]:
        return [1.0, 1.0, 1.0]

    def set_emission_tint(self, material: str, tint: Sequence[float]) -> None:
        self._refuse(f"tint {material!r}")

    def key_emission_tint(
        self, material: str, tint: Sequence[float], frame: Optional[float] = None, stepped: bool = False
    ) -> None:
        self._refuse(f"key {material!r}'s tint")

    def emission_tint_keys(self, material: str) -> List[Tuple[float, List[float], bool]]:
        return []

    def clear_emission_tint_keys(self, material: str) -> None:
        return None

    def emission_tint_channels(self, material: str) -> List[Any]:
        return []

    def emission_states(self, material: str) -> Dict[str, Dict[str, Any]]:
        return {}

    def set_emission_states(self, material: str, states: Dict[str, Dict[str, Any]]) -> None:
        self._refuse(f"store {material!r}'s states")

    def copy_emission(self, src: str, dst: str, factor: float) -> None:
        self._refuse(f"key {dst!r}'s emission")

    def scale_emission(self, material: str, factor: float) -> None:
        self._refuse(f"scale {material!r}'s emission")

    def stage_emission_proxy(
        self, material: str, name: str, factor: float, tint: Optional[int] = None
    ) -> Optional[Any]:
        self._refuse(f"stage {name!r}")
        return None

    def emission_proxies(self) -> List[str]:
        return []

    def remove_emission_proxies(self, names: Optional[Iterable[str]] = None) -> List[str]:
        return []

    def undo_step(self, label: str) -> ContextManager[Any]:
        return contextlib.nullcontext()

    def legacy_store_names(self, prefix: str) -> List[str]:
        return []

    def legacy_members(self, store: str) -> Optional[FaceMap]:
        return None

    def adopt_legacy_weight(self, attr: str, copy: str, factor: float) -> bool:
        return False

    def remove_legacy(self, stores: List[str], attrs: List[str], color_set: str) -> List[str]:
        return []

    def extra_warnings(self) -> List[str]:
        return []


class _EmissiveGroupsModelInternal:
    """Implementation-detail base for :class:`EmissiveGroupsModel`."""

    #: Joins a copy's base and group: ``<base>_<group>``.
    SEPARATOR = "_"
    #: The old tool's membership stores and carrier weights: ``<prefix><group>``.
    LEGACY_PREFIX = "emissiveGroup_"
    #: The old tool's vertex-colour encoding.
    LEGACY_COLOR_SET = "emissiveGroups"
    #: Where the old tool kept its registry on the internal carrier (its
    #: earliest spelling last), and its record on the export carrier.
    LEGACY_REGISTRY_KEYS = ("emissive_registry", "emissive_groups")
    LEGACY_RECORD_KEY = "emissive_groups"

    @staticmethod
    def _leaf(name: str) -> str:
        return str(name).split("|")[-1].split(":")[-1]

    @classmethod
    def _members(cls, name: str, copies=None) -> Dict[str, Dict[str, Any]]:
        """Group *name*'s copies; ``{}`` when it has none."""
        copies = cls.HOST.copies() if copies is None else copies
        return {c: i for c, i in copies.items() if i.get("group") == name}

    @classmethod
    def _known(cls, name: str) -> Dict[str, Dict[str, Any]]:
        members = cls._members(name)
        if not members:
            raise ValueError(f"Unknown group {name!r}.")
        return members

    @classmethod
    def _weight_of(cls, members: Dict[str, Dict[str, Any]]) -> Optional[float]:
        """The group's weight now: its first copy's emission over its strength."""
        for copy, info in sorted(members.items()):
            strength = float(info.get("strength") or 0.0)
            value = cls.HOST.emission(copy)
            if strength > 0 and value is not None:
                return max(0.0, min(1.0, float(value) / strength))
        return None

    @classmethod
    def _keyed_copy(cls, members: Dict[str, Dict[str, Any]]) -> Optional[str]:
        return next((c for c in sorted(members) if cls.HOST.emission_keys(c)), None)

    @classmethod
    def _rest(cls, members, weight: float, ctx) -> Tuple[float, List[str]]:
        """Rest *members*' un-keyed copies at *weight* (clamped 0..1), inside
        the caller's undo step: ``(weight, copies rested)``."""
        weight = max(0.0, min(1.0, float(weight)))
        rested = []
        for copy, info in sorted(members.items()):
            if cls.HOST.emission_keys(copy):
                continue
            if ctx is None or ctx.should_change(copy, f"weight {weight:g}"):
                cls.HOST.set_emission(copy, float(info["strength"]) * weight)
                rested.append(copy)
        return weight, rested

    @staticmethod
    def _clean_tint(tint) -> List[float]:
        values = [float(c) for c in list(tint)[:3]]
        if len(values) != 3 or any(c < 0.0 for c in values):
            raise ValueError(f"A tint is three linear RGB values of 0 or more, got {tint!r}.")
        return values

    @classmethod
    def _clean_states(cls, states) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for state, entry in dict(states or {}).items():
            entry = dict(entry or {})
            out[cls.sanitize(state)] = {
                "tint": cls._clean_tint(entry.get("tint", (1.0, 1.0, 1.0))),
                "weight": max(0.0, min(1.0, float(entry.get("weight", 1.0)))),
            }
        if not out:
            raise ValueError("A group's states need at least one state.")
        return out

    @classmethod
    def _own_states(cls, members) -> Dict[str, Any]:
        """The states *members* store (their first copy's that has some); ``{}``
        when none does."""
        return next((t for t in (cls.HOST.emission_states(c) for c in sorted(members)) if t), {})

    @classmethod
    def _tintable(cls, members) -> bool:
        """Whether every copy of *members* can carry a tint."""
        return not any(cls.HOST.tint_refusal(c) for c in members)

    @classmethod
    def _defaults(cls, tintable: bool) -> Dict[str, Any]:
        """The states a group takes when given none: the indicator's, or its
        weight's alone where a copy cannot be tinted."""
        return cls.DEFAULT_STATES if tintable else cls.WEIGHT_STATES

    @classmethod
    def _table(cls, members) -> Dict[str, Any]:
        """*members*' states: their own, else the defaults they take."""
        return cls._own_states(members) or cls._defaults(cls._tintable(members))

    @classmethod
    def _stated(cls, name: str, state: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], bool]:
        """``(members, entry, tinted)`` for putting group *name* in *state*: its
        copies, the state's cleaned ``{"tint", "weight"}`` (its own states,
        else the defaults') and whether it sets the tint. Refused
        (``ValueError``) before any change: an unknown group or state, or a
        tint a copy cannot carry."""
        members = cls._known(name)
        table = cls._table(members)
        if state not in table:
            raise ValueError(
                f"Group {name!r} has no state {state!r}; it has {', '.join(sorted(table))}."
            )
        found = table[state]
        entry = {"tint": cls._clean_tint(found.get("tint", (1.0, 1.0, 1.0))), "weight": float(found.get("weight", 1.0))}
        tinted = cls._coloured(members, entry, table)
        if tinted:
            cls._require_tint(name, members)
        return members, entry, tinted

    @classmethod
    def _adopt_defaults(cls, members, ctx) -> None:
        """Store the defaults on a group put in a state with none of its own,
        inside the caller's undo step: the states it shows are then its own,
        and ship with it (:meth:`EmissiveGroupsModel.render_effect_materials`)."""
        if cls._own_states(members):
            return
        table = cls._clean_states(cls._defaults(cls._tintable(members)))
        for copy in sorted(members):
            if ctx is None or ctx.should_change(copy, f"states {', '.join(table)}"):
                cls.HOST.set_emission_states(copy, table)

    @classmethod
    def _key_state(cls, members, state: str, entry, tinted: bool, frame, ctx) -> List[str]:
        """Key *members* into *state* (*entry*, as :meth:`_stated` gave it) at
        *frame*, stepped, inside the caller's undo step: the copies keyed."""
        cls._adopt_defaults(members, ctx)
        keyed = []
        for copy, info in sorted(members.items()):
            if ctx is not None and not ctx.should_change(copy, f"key state {state}"):
                continue
            cls.HOST.key_emission(copy, float(info["strength"]) * entry["weight"], frame, True)
            if tinted:
                cls.HOST.key_emission_tint(copy, entry["tint"], frame, True)
            keyed.append(copy)
        return keyed

    @classmethod
    def _tinted(cls, members) -> bool:
        """Whether a copy of *members* carries a tint (keyed, or not white)."""
        for copy in members:
            if cls.HOST.tint_refusal(copy):
                continue
            if cls.HOST.emission_tint_keys(copy) or any(
                abs(c - 1.0) > 1e-6 for c in cls.HOST.emission_tint(copy)
            ):
                return True
        return False

    @classmethod
    def _coloured(cls, members, entry, table) -> bool:
        """Whether a state (*entry*, of *table*) sets its group's tint: the
        state is coloured, the group already tinted, or any state of its table
        colours it -- then every state keys the tint, so each switch is whole."""
        white = [1.0, 1.0, 1.0]
        if entry["tint"] != white or cls._tinted(members):
            return True
        return any(cls._clean_tint(e.get("tint", white)) != white for e in table.values())

    @classmethod
    def _require_tint(cls, name: str, members) -> None:
        """Refuse (``ValueError``, naming why) a tint on a group a copy of
        which cannot carry one."""
        for copy in sorted(members):
            reason = cls.HOST.tint_refusal(copy)
            if reason:
                raise ValueError(f"Group {name!r} cannot be tinted: {reason}.")

    @classmethod
    def _join(cls, members, copy: str) -> None:
        """A copy joining *members* takes their states and their tint (its
        keys, or the colour they rest at)."""
        src = next(iter(sorted(c for c in members if c != copy)), None)
        if src is None:
            return
        states = cls.HOST.emission_states(src)
        if states:
            cls.HOST.set_emission_states(copy, states)
        if cls.HOST.tint_refusal(src) or cls.HOST.tint_refusal(copy):
            return
        keys = cls.HOST.emission_tint_keys(src)
        for frame, tint, stepped in keys:
            cls.HOST.key_emission_tint(copy, tint, frame, stepped)
        if not keys:
            tint = cls.HOST.emission_tint(src)
            if any(abs(c - 1.0) > 1e-6 for c in tint):
                cls.HOST.set_emission_tint(copy, tint)

    @classmethod
    def _state_of(cls, members, table) -> Optional[str]:
        """The state of *table* the group shows now (its tint and weight), or
        ``None``."""
        weight = cls._weight_of(members)
        if weight is None:
            return None
        copy = next((c for c in sorted(members) if not cls.HOST.tint_refusal(c)), None)
        tint = cls.HOST.emission_tint(copy) if copy else [1.0, 1.0, 1.0]
        for state, entry in sorted(table.items()):
            if abs(entry["weight"] - weight) > 1e-3:
                continue
            if entry["weight"] <= 0.0 or all(abs(a - b) <= 1e-3 for a, b in zip(entry["tint"], tint)):
                return state
        return None

    @classmethod
    def _key(cls, members, value: Optional[float], frame, ctx) -> List[str]:
        """Key *members*' weight at *value* (``None``: the weight now), inside
        the caller's undo step: the copies keyed."""
        if value is None:
            value = cls._weight_of(members)
        value = max(0.0, min(1.0, float(1.0 if value is None else value)))
        keyed = []
        for copy, info in sorted(members.items()):
            if ctx is None or ctx.should_change(copy, f"key weight {value:g}"):
                cls.HOST.key_emission(copy, float(info["strength"]) * value, frame)
                keyed.append(copy)
        return keyed

    @classmethod
    def _strength_of(cls, base: str) -> float:
        """*base*'s emission strength, refused (``ValueError``, with its fix)
        when its shader has no emission scalar or it does not glow."""
        strength = cls.HOST.emission(base)
        if strength is None:
            raise ValueError(
                f"Material {base!r} has no emission weight to key: give its "
                "faces a surface shader with an emission weight (a Standard "
                "Surface, a Principled BSDF) before grouping them."
            )
        if float(strength) <= 0.0:
            raise ValueError(
                f"Material {base!r} does not glow (its emission weight is "
                f"{strength:g}): set its emission first; a group's weight "
                "scales it."
            )
        colour = cls.HOST.emission_color(base)
        if colour is not None and max(float(c) for c in list(colour)[:3]) <= 0.0:
            raise ValueError(
                f"Material {base!r} does not glow (its emission colour is "
                "black): give it an emission colour first; a group's weight "
                "scales it."
            )
        return float(strength)

    @classmethod
    def _plan(cls, name: str, by_material: Dict[str, FaceMap], copies) -> list:
        """``(material, base, strength, faces)`` for every material whose faces
        join group *name*; refused whole before any change."""
        plan = []
        for material, faces in sorted(by_material.items()):
            info = copies.get(material)
            if info is not None and info.get("group") == name:
                continue  # already in the group
            if info is not None and not info.get("base"):
                raise ValueError(
                    f"{material!r} is a copy whose base material is gone: "
                    "remove its group, or give its faces a material, first."
                )
            base = info["base"] if info is not None else material
            plan.append((material, base, cls._strength_of(base), faces))
        return plan

    @classmethod
    def _prune(cls, materials: Iterable[str], ctx) -> List[str]:
        """Delete each copy among *materials* that no face wears any more."""
        copies = cls.HOST.copies()
        gone = []
        for material in sorted(set(materials)):
            if material in copies and not cls.HOST.material_faces(material):
                if ctx is None or ctx.should_delete(material, "a copy no face wears"):
                    cls.HOST.delete_material(material)
                    gone.append(material)
        return gone

    @classmethod
    def _differences(cls, base: str, copy: str) -> List[str]:
        """The values and maps where *copy* is not its *base*'s."""
        want = cls.HOST.material_state(base)
        have = cls.HOST.material_state(copy)
        return sorted(
            key
            for key in set(want) | set(have)
            if not cls._same(want.get(key), have.get(key))
        )

    @classmethod
    def _same(cls, a, b) -> bool:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) <= 1e-6
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(cls._same(x, y) for x, y in zip(a, b))
        return a == b

    @classmethod
    def _ships(cls, copy: str, ctx) -> bool:
        """Whether a face of a mesh the write ships wears *copy* (every copy
        some face wears, for a write with no scope)."""
        faces = cls.HOST.material_faces(copy)
        if not faces:
            return False  # no face wears it: no file carries it
        return ctx is None or any(ctx.ships(obj) for obj in faces)

    @classmethod
    def _animated(cls, ctx) -> List[Tuple[str, float]]:
        """``(copy, strength)`` for each copy a runtime plays that the write
        ships (:meth:`EmissiveGroupsModel.render_effect_materials` says
        which), in name order: keyed, or holding named states a runtime
        switches. *strength* is the tag's, in the shader's own units -- what
        the copy's weight scales, so its keys over it are the weight."""
        out = []
        for copy, info in sorted(cls.HOST.copies().items()):
            strength = float(info.get("strength") or 0.0)
            if strength <= 0.0:
                continue
            played = (
                cls.HOST.emission_keys(copy)
                or cls.HOST.emission_tint_keys(copy)
                or cls.HOST.emission_states(copy)
            )
            if not played:
                continue
            if cls._ships(copy, ctx):
                out.append((copy, strength))
        return out

    @classmethod
    def _legacy_registry(cls) -> Optional[dict]:
        """The old tool's registry, as it stored it; ``None`` when absent."""
        nodes = cls.HOST.data_nodes
        if nodes is None:
            return None
        for key in cls.LEGACY_REGISTRY_KEYS:
            text = nodes.read(ptk.Scope.PRIVATE, key)
            if not text:
                continue
            try:
                data = json.loads(text)
            except ValueError:
                continue
            if isinstance(data, dict) and isinstance(data.get("groups"), dict):
                return data
        return None


class EmissiveGroupsModel(_EmissiveGroupsModelInternal, ptk.LoggingMixin, ptk.HelpMixin):
    """Author named emissive face groups as materials (see module doc).

    All state lives in the scene, so the class holds only its host's half:
    each library binds it by setting :attr:`HOST` (each host library's
    ``EmissiveGroups``), and every method is a classmethod over it. Every
    operation that changes the scene is one undo step and takes an optional
    ``ctx`` (``ptk.OpContext``) it asks before each change.
    """

    #: The host's scene operations (an :class:`EmissiveGroupsHost`), set by the
    #: host library's binding; the pure default has no scene.
    HOST: EmissiveGroupsHost = PureEmissiveGroupsHost()

    @staticmethod
    def sanitize(name: str) -> str:
        """*name* as a scene-safe identifier, or ``ValueError``."""
        clean = re.sub(r"[^A-Za-z0-9_]", "_", str(name).strip())
        if not clean or clean[0].isdigit():
            raise ValueError(f"Invalid group name {name!r}.")
        return clean

    @classmethod
    def copy_name(cls, base: str, group: str) -> str:
        """The name of *group*'s copy of *base*: ``<base>_<group>``."""
        return f"{cls._leaf(base)}{cls.SEPARATOR}{group}"

    # ------------------------------------------------------------------
    # Authoring
    # ------------------------------------------------------------------

    @classmethod
    def add_group(
        cls, name: str, faces=None, weight: Optional[float] = None, ctx=None
    ) -> List[str]:
        """Create a group from faces (or the selection), or extend one.

        Each face takes the group's copy of its material, made the first time
        (one per material and group, sharing the base's maps); a face already
        in another group moves, and a copy left with no face is deleted.

        Parameters:
            name: Group name (sanitized to safe characters).
            faces: ``{object: [face indices]}`` (``[]`` = every face), or the
                host's own spelling; ``None`` takes the selected faces.
            weight: The weight a new copy rests at (0..1); ``None`` keeps the
                group's own, or 1 for a new group. A keyed group's new copy
                plays its curve.
            ctx: An optional ``ptk.OpContext``.

        Returns:
            The group's copies.

        Raises:
            ValueError: No faces, or a material whose shader has no emission
                weight or does not glow (named, with its fix): nothing changes.
        """
        name = cls.sanitize(name)
        face_map = cls.HOST.faces(faces)
        if not face_map:
            raise ValueError("No faces in the input or selection.")
        with cls.HOST.undo_step(f"Emissive Groups: add {name}"):
            copies = cls.HOST.copies()
            members = cls._members(name, copies)
            plan = cls._plan(name, cls.HOST.face_materials(face_map), copies)
            if cls._tinted(members):  # every copy of a tinted group carries the tint
                for _material, base, _strength, _faces in plan:
                    reason = cls.HOST.tint_refusal(base)
                    if reason:
                        raise ValueError(
                            f"Group {name!r} is tinted and {base!r} cannot be: {reason}."
                        )
            if weight is None:
                weight = cls._weight_of(members)
            weight = 1.0 if weight is None else max(0.0, min(1.0, float(weight)))
            keyed = cls._keyed_copy(members)
            left: List[str] = []
            for material, base, strength, faces_on in plan:
                copy = next((c for c, i in members.items() if i.get("base") == base), None)
                if copy is None:
                    proposed = cls.copy_name(base, name)
                    if ctx is not None and not ctx.should_create(
                        proposed, f"{name}'s copy of {base}"
                    ):
                        continue
                    copy = cls.HOST.make_copy(base, proposed)
                    cls.HOST.tag(copy, name, base, strength)
                    if keyed is not None:
                        cls.HOST.copy_emission(keyed, copy, strength / members[keyed]["strength"])
                    else:
                        cls.HOST.set_emission(copy, strength * weight)
                    cls._join(members, copy)
                    members[copy] = {"group": name, "base": base, "strength": strength}
                count = sum(len(i) for i in faces_on.values())
                if ctx is None or ctx.should_change(copy, f"{count} face(s) of {material} join"):
                    cls.HOST.assign(copy, faces_on)
                    if material in copies:
                        left.append(material)
            cls._prune(left, ctx)
        cls.logger.info(f"Group {name!r}: {', '.join(sorted(members)) or 'no copy'}.")
        return sorted(members)

    @classmethod
    def remove_group(cls, name: str, ctx=None) -> List[str]:
        """Restore the group's faces' base materials and delete its copies.

        Returns:
            The copies deleted. A copy whose base is gone is kept (named).
        """
        members = cls._known(name)
        gone = []
        with cls.HOST.undo_step(f"Emissive Groups: remove {name}"):
            for copy, info in sorted(members.items()):
                base = info.get("base")
                if not base:
                    cls.logger.warning(
                        f"{copy!r}: its base material is gone, so its faces keep it; "
                        "give them a material, then delete it."
                    )
                    continue
                faces = cls.HOST.material_faces(copy)
                if faces and (ctx is None or ctx.should_change(base, f"faces of {copy} back")):
                    cls.HOST.assign(base, faces)
                if ctx is None or ctx.should_delete(copy, f"group {name} removed"):
                    cls.HOST.delete_material(copy)
                    gone.append(copy)
        cls.logger.info(f"Removed group {name!r}.")
        return gone

    @classmethod
    def remove_faces(cls, faces=None, ctx=None) -> FaceMap:
        """Take faces (or the selection) out of their groups: each wears its
        base material again, and a copy left with no face is deleted.

        Returns:
            The faces that left a group.
        """
        face_map = cls.HOST.faces(faces)
        left: FaceMap = {}
        with cls.HOST.undo_step("Emissive Groups: remove faces"):
            copies = cls.HOST.copies()
            touched = []
            for material, faces_on in sorted(cls.HOST.face_materials(face_map).items()):
                base = (copies.get(material) or {}).get("base")
                if not base:
                    continue
                if ctx is None or ctx.should_change(base, f"faces of {material} back"):
                    cls.HOST.assign(base, faces_on)
                    touched.append(material)
                    for obj, indices in faces_on.items():
                        left.setdefault(obj, []).extend(indices)
            cls._prune(touched, ctx)
        return left

    @classmethod
    def list_groups(cls) -> Dict[str, dict]:
        """``{name: {"weight", "keyed", "weight_keyed", "tint_keyed",
        "materials", "bases", "faces", "tint", "states", "state"}}`` by name:
        the weight now; whether a copy's weight or tint is keyed (``keyed``:
        either), the copies and their bases (``None`` for a lost one), the face
        count, the tint now (``None`` when the group cannot be tinted), the
        names of its states -- its own, else the defaults a state put on it
        stores (:meth:`default_states`) -- and the one of its own it shows now
        (``None``: none, or none of its own yet)."""
        grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for copy, info in cls.HOST.copies().items():
            grouped.setdefault(info.get("group"), {})[copy] = info
        out = {}
        for name in sorted(grouped):
            members = grouped[name]
            worn: FaceMap = {}
            for copy in members:
                for obj, indices in cls.HOST.material_faces(copy).items():
                    worn.setdefault(obj, []).extend(indices)
            tintable = cls._tintable(members)
            own = cls._own_states(members)
            weight_keyed = cls._keyed_copy(members) is not None
            tint_keyed = any(cls.HOST.emission_tint_keys(c) for c in members)
            out[name] = {
                "weight": cls._weight_of(members),
                "keyed": weight_keyed or tint_keyed,
                "weight_keyed": weight_keyed,
                "tint_keyed": tint_keyed,
                "materials": sorted(members),
                "bases": [members[c].get("base") for c in sorted(members)],
                "faces": cls.HOST.face_count(worn),
                "tint": cls.HOST.emission_tint(sorted(members)[0]) if tintable else None,
                # As authored: an indicator's order.
                "states": list(own or cls._defaults(tintable)),
                "state": cls._state_of(members, own) if own else None,
            }
        return out

    @classmethod
    def select_group(cls, name: str) -> None:
        """Select the group's faces."""
        faces: FaceMap = {}
        for copy in sorted(cls._known(name)):
            for obj, indices in cls.HOST.material_faces(copy).items():
                faces.setdefault(obj, []).extend(indices)
        if not faces:
            raise ValueError(f"Group {name!r} has no faces.")
        cls.HOST.select({o: sorted(i) for o, i in faces.items()})

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    @classmethod
    def set_weight(cls, name: str, weight: float, ctx=None) -> float:
        """Rest the group's weight at *weight* (clamped 0..1): each un-keyed
        copy's emission becomes its strength times it; a keyed copy is the
        animation's, left alone (:meth:`key_weight` writes it).

        Returns:
            The weight set.
        """
        members = cls._known(name)
        with cls.HOST.undo_step(f"Emissive Groups: weight {name}"):
            return cls._rest(members, weight, ctx)[0]

    @classmethod
    def set_weights(cls, weights: Dict[str, float], ctx=None) -> Dict[str, float]:
        """:meth:`set_weight` for several groups, in one undo step (the
        panel's All On / All Off).

        Returns:
            ``{group: weight}`` for each group whose copies rest at it; a group
            every copy of which is keyed is left out (its curve plays).
        """
        known = {name: cls._known(name) for name in weights}
        out = {}
        with cls.HOST.undo_step("Emissive Groups: weights"):
            for name, members in sorted(known.items()):
                weight, rested = cls._rest(members, weights[name], ctx)
                if rested:
                    out[name] = weight
        return out

    @classmethod
    def key_weight(
        cls, name: str, value: Optional[float] = None, frame: Optional[float] = None, ctx=None
    ) -> List[str]:
        """Key the group's weight on every copy (its strength times *value*).

        Parameters:
            name: Group name.
            value: The weight (clamped 0..1); ``None`` keys the weight now.
            frame: Where; ``None`` = the current time.

        Returns:
            The copies keyed.
        """
        members = cls._known(name)
        with cls.HOST.undo_step(f"Emissive Groups: key {name}"):
            return cls._key(members, value, frame, ctx)

    @classmethod
    def key_weights(
        cls, names: Optional[Iterable[str]] = None, frame: Optional[float] = None, ctx=None
    ) -> List[str]:
        """:meth:`key_weight` at each group's weight now, in one undo step.

        Parameters:
            names: Groups to key; ``None`` = every group.
            frame: Where; ``None`` = the current time.

        Returns:
            The copies keyed.
        """
        copies = cls.HOST.copies()
        wanted = sorted({i.get("group") for i in copies.values()} if names is None else names)
        known = {name: cls._known(name) for name in wanted}
        keyed: List[str] = []
        with cls.HOST.undo_step("Emissive Groups: key weights"):
            for name, members in known.items():
                keyed += cls._key(members, None, frame, ctx)
        return keyed

    @classmethod
    def remove_keys(cls, name: str, ctx=None) -> List[str]:
        """Delete the group's weight and tint keys on every copy, keeping its
        weight and tint now.

        Returns:
            The copies that had keys.
        """
        cleared = []
        with cls.HOST.undo_step(f"Emissive Groups: remove keys {name}"):
            for copy in sorted(cls._known(name)):
                keyed = cls.HOST.emission_keys(copy) or cls.HOST.emission_tint_keys(copy)
                if keyed and (ctx is None or ctx.should_change(copy, "remove keys")):
                    cls.HOST.clear_emission_keys(copy)
                    cls.HOST.clear_emission_tint_keys(copy)
                    cleared.append(copy)
        return cleared

    # ------------------------------------------------------------------
    # States: a tint and a weight under a name
    # ------------------------------------------------------------------

    #: An indicator light's states, as data: a group stores its own
    #: (:meth:`set_states`), these when it is given none. A tint multiplies the
    #: base's emission colour (linear RGB), so over a white base it is the
    #: colour that shows. Prior art weighed: glTF's ``KHR_materials_variants``
    #: swaps a whole material per variant name; a state tints the one copy's
    #: emission instead, so a clip keys it and no material is made per state.
    DEFAULT_STATES: Dict[str, Dict[str, Any]] = {
        "off": {"tint": [1.0, 1.0, 1.0], "weight": 0.0},
        "green": {"tint": [0.0, 1.0, 0.0], "weight": 1.0},
        "blue": {"tint": [0.0, 0.0, 1.0], "weight": 1.0},
        "amber": {"tint": [1.0, 0.5, 0.0], "weight": 1.0},
        "red": {"tint": [1.0, 0.0, 0.0], "weight": 1.0},
    }
    #: The states of a group whose emission cannot be tinted (a map, a black
    #: channel): its weight's alone.
    WEIGHT_STATES: Dict[str, Dict[str, Any]] = {
        "off": {"tint": [1.0, 1.0, 1.0], "weight": 0.0},
        "on": {"tint": [1.0, 1.0, 1.0], "weight": 1.0},
    }

    @classmethod
    def default_states(cls, name: str) -> Dict[str, Any]:
        """The states group *name* takes when given none:
        :attr:`DEFAULT_STATES`, or :attr:`WEIGHT_STATES` when a copy of it
        cannot be tinted (``EmissiveGroupsHost.tint_refusal``)."""
        return cls._defaults(cls._tintable(cls._known(name)))

    @classmethod
    def set_states(cls, name: str, states: Optional[Dict[str, Any]] = None, ctx=None) -> Dict[str, Any]:
        """Give the group its named states, stored on every copy (a copy that
        joins later takes them).

        Parameters:
            name: Group name.
            states: ``{state: {"tint": [r, g, b], "weight": w}}``, a tint
                (default white) multiplying the base's emission colour and a
                weight (0..1, default 1); ``None``: :meth:`default_states`.
            ctx: An optional ``ptk.OpContext``.

        Returns:
            The states stored, cleaned.

        Raises:
            ValueError: No state, a malformed one, or a coloured state on a
                group whose emission cannot be tinted (named, with why).
        """
        members = cls._known(name)
        clean = cls._clean_states(cls.default_states(name) if states is None else states)
        if any(entry["tint"] != [1.0, 1.0, 1.0] for entry in clean.values()):
            cls._require_tint(name, members)
        with cls.HOST.undo_step(f"Emissive Groups: states {name}"):
            for copy in sorted(members):
                if ctx is None or ctx.should_change(copy, f"states {', '.join(clean)}"):
                    cls.HOST.set_emission_states(copy, clean)
        return clean

    @classmethod
    def states(cls, name: str) -> Dict[str, Any]:
        """The group's stored states (``{}`` before :meth:`set_states`, or a
        state put on it, stores them)."""
        return cls._own_states(cls._known(name))

    @classmethod
    def groups_on(cls, objects: Iterable[str]) -> List[str]:
        """The groups whose faces lie on *objects* (named as the host's face
        maps name them: a long path or a bare object name, as the host keeps it), sorted:
        what a part and the parts under it wear."""
        wanted = set(objects)
        found = set()
        for copy, info in cls.HOST.copies().items():
            group = info.get("group")
            if group and group not in found and not wanted.isdisjoint(cls.HOST.material_faces(copy)):
                found.add(group)
        return sorted(found)

    @classmethod
    def ensure_state(cls, names: Iterable[str], state: str, ctx=None) -> Tuple[str, str]:
        """Make sure every group of *names* has the state *state*: one with no
        states of its own stores its defaults (:meth:`default_states`), one
        with states keeps them. Whole or not at all: where one group's states
        (its own, else its defaults) lack *state*, nothing changes. Nothing is
        keyed: switching to it is the business of whatever runs the steps.

        Returns:
            ``("given", "")`` where it stored states (one undo step),
            ``("kept", "")`` where every group had them, else
            ``("skipped", why)``: no group, or one lacking *state*.
        """
        copies = cls.HOST.copies()
        members = {name: cls._members(name, copies) for name in names}
        if not members:
            return "skipped", "it wears no emissive group"
        for name, held in sorted(members.items()):
            if not held:
                raise ValueError(f"Unknown group {name!r}.")
            table = cls._table(held)
            if state not in table:
                return "skipped", f"its emissive group {name!r} has no state {state!r} (it has {', '.join(table)})"
        bare = [name for name, held in sorted(members.items()) if not cls._own_states(held)]
        if not bare:
            return "kept", ""
        with cls.HOST.undo_step(f"Emissive Groups: states {', '.join(bare)}"):
            for name in bare:
                cls._adopt_defaults(members[name], ctx)
        return "given", ""

    @classmethod
    def set_tint(cls, name: str, tint: Sequence[float], ctx=None) -> List[float]:
        """Rest the group's tint at *tint* (linear RGB, each 0 or more): each
        copy whose tint is not keyed shows its base colour times it.

        Raises:
            ValueError: A copy whose emission cannot be tinted (named, why).
        """
        members = cls._known(name)
        tint = cls._clean_tint(tint)
        cls._require_tint(name, members)
        with cls.HOST.undo_step(f"Emissive Groups: tint {name}"):
            for copy in sorted(members):
                if cls.HOST.emission_tint_keys(copy):
                    continue
                if ctx is None or ctx.should_change(copy, f"tint {tint}"):
                    cls.HOST.set_emission_tint(copy, tint)
        return tint

    @classmethod
    def key_tint(
        cls, name: str, tint: Sequence[float], frame: Optional[float] = None, stepped: bool = False, ctx=None
    ) -> List[str]:
        """Key the group's tint on every copy at *frame* (``None``: now);
        *stepped* holds it to the next key.

        Returns:
            The copies keyed.
        """
        members = cls._known(name)
        tint = cls._clean_tint(tint)
        cls._require_tint(name, members)
        keyed = []
        with cls.HOST.undo_step(f"Emissive Groups: key tint {name}"):
            for copy in sorted(members):
                if ctx is None or ctx.should_change(copy, f"key tint {tint}"):
                    cls.HOST.key_emission_tint(copy, tint, frame, stepped)
                    keyed.append(copy)
        return keyed

    @classmethod
    def set_state(cls, name: str, state: str, ctx=None) -> Dict[str, Any]:
        """Rest the group in its *state*: its tint and weight, on each copy
        not keyed (a keyed copy is the animation's: :meth:`key_state`). A
        group with no states of its own stores its defaults
        (:meth:`default_states`) in the same undo step, so it ships the states
        it shows.

        Returns:
            The state's ``{"tint", "weight"}``.
        """
        members, entry, tinted = cls._stated(name, state)
        with cls.HOST.undo_step(f"Emissive Groups: {name} {state}"):
            cls._adopt_defaults(members, ctx)
            cls._rest(members, entry["weight"], ctx)
            if tinted:
                for copy in sorted(members):
                    if not cls.HOST.emission_tint_keys(copy) and (
                        ctx is None or ctx.should_change(copy, f"tint {entry['tint']}")
                    ):
                        cls.HOST.set_emission_tint(copy, entry["tint"])
        return entry

    @classmethod
    def key_state(cls, name: str, state: str, frame: Optional[float] = None, ctx=None) -> List[str]:
        """Key the group into its *state* at *frame* (``None``: now), stepped:
        it holds until the next key, a switch rather than a fade. Its weight
        and, on a group that is or becomes tinted, its tint; a group with no
        states of its own stores its defaults, as :meth:`set_state` does.

        Returns:
            The copies keyed.
        """
        members, entry, tinted = cls._stated(name, state)
        with cls.HOST.undo_step(f"Emissive Groups: key {name} {state}"):
            return cls._key_state(members, state, entry, tinted, frame, ctx)

    @classmethod
    def key_states(cls, states: Dict[str, str], frame: Optional[float] = None, ctx=None) -> List[str]:
        """:meth:`key_state` for several groups, in one undo step (the panel's
        Key States).

        Parameters:
            states: ``{group: state}``.
            frame: Where; ``None`` = the current time.
            ctx: An optional ``ptk.OpContext``.

        Returns:
            The copies keyed.

        Raises:
            ValueError: An unknown group or state, or a tint a copy cannot
                carry: refused whole, before any change.
        """
        planned = {name: cls._stated(name, state) for name, state in states.items()}
        keyed: List[str] = []
        with cls.HOST.undo_step("Emissive Groups: key states"):
            for name, (members, entry, tinted) in sorted(planned.items()):
                keyed += cls._key_state(members, states[name], entry, tinted, frame, ctx)
        return keyed

    # ------------------------------------------------------------------
    # In step with the base
    # ------------------------------------------------------------------

    @classmethod
    def sync(cls, names=None, check_only: bool = False, ctx=None) -> List[str]:
        """Keep each copy in step with its base: a value or map the base
        changed reaches the copy, and a changed base emission strength
        rescales the copy's emission and keys (its weight kept).

        Parameters:
            names: Groups to sync; ``None`` = every group.
            check_only: Report, change nothing (the export's check).
            ctx: An optional ``ptk.OpContext``.

        Returns:
            One line per copy out of step (or whose base is gone), saying what.
        """
        wanted = None if names is None else {str(n) for n in names}
        found: List[str] = []
        # A check changes nothing, so it leaves no undo step (a host may push
        # one for every step opened: every export's check added an empty one).
        step = contextlib.nullcontext() if check_only else cls.HOST.undo_step("Emissive Groups: sync")
        with step:
            for copy, info in sorted(cls.HOST.copies().items()):
                if wanted is not None and info.get("group") not in wanted:
                    continue
                base = info.get("base")
                if not base:
                    found.append(
                        f"{copy!r} (group {info.get('group')!r}): its base material "
                        "is gone, so it stays as it is; give its faces a material "
                        "or remove the group."
                    )
                    continue
                said = []
                diffs = cls._differences(base, copy)
                if diffs:
                    said.append(f"{', '.join(diffs[:6])}{' ...' if len(diffs) > 6 else ''}")
                strength = cls.HOST.emission(base)
                old = float(info.get("strength") or 0.0)
                rescale = (
                    strength is not None
                    and float(strength) > 0.0
                    and old > 0.0
                    and abs(float(strength) - old) > 1e-6
                )
                if rescale:
                    said.append(f"emission strength {old:g} -> {float(strength):g}")
                dark = strength is not None and float(strength) <= 0.0 and old > 0.0
                if dark:
                    # No weight scales a glow of 0: the copy keeps its own, and
                    # says so rather than reading as in step.
                    said.append(
                        "its base no longer glows (emission 0); the copy keeps "
                        f"strength {old:g} until the base glows again"
                    )
                if not said:
                    continue
                found.append(f"{copy!r} is out of step with {base!r}: {'; '.join(said)}.")
                if check_only or not (ctx is None or ctx.should_change(copy, "in step with its base")):
                    continue
                if diffs:
                    cls.HOST.sync_copy(copy, base)
                if rescale:
                    cls.HOST.scale_emission(copy, float(strength) / old)
                    cls.HOST.tag(copy, info["group"], base, float(strength))
        return found

    @classmethod
    def validate(cls) -> List[str]:
        """Non-fatal authoring warnings (empty list = clean): copies out of step
        or lost, the old tool's data still to convert, the host's own."""
        warnings = cls.sync(check_only=True)
        if cls._legacy_registry() is not None or cls.HOST.legacy_store_names(cls.LEGACY_PREFIX):
            warnings.append(
                "The scene holds emissive groups made by the old tool (masks and "
                "slots): convert them in one action with convert_legacy() (the "
                "panel's Convert Old Groups)."
            )
        warnings += cls.HOST.extra_warnings()
        for msg in warnings:
            cls.logger.warning(msg)
        return warnings

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @classmethod
    def render_effect_materials(cls, ctx=None) -> List[dict]:
        """Each copy a runtime plays -- keyed, or holding named states it
        switches -- as a ``render_effects`` material entry:
        ``{"material", "channels", "emission_color", "emission_strength",
        "tint"?, "states"?}`` -- the channels its keys animate (none for
        states alone; ``emission``,
        its weight; ``tint_r/g/b``, its tint, where its colour is not black),
        the tint it shows now and its named states (a tintable copy's, where
        it has them), the colour its weight scales (white over a map, which
        glTF multiplies) and its base's strength as the delivery
        states it (:meth:`EmissiveGroupsHost.published_strength`: the
        ``materials`` record's emission weight -- 1 for a scalar that only
        gates, OpenPBR's nits), so weight 1 is the base's own glow. The copy
        is named as the delivery names it, namespace included; its curve is
        keyed by the transport's name for it
        (``ptk.SceneRecords.TRANSPORT.target_name``). Pure: it reads the
        copies and writes nothing.

        A scoped write names the copies a face of a mesh it ships wears, and
        no other (the record is export-scoped: a push of one prop names its
        own groups); a write with no scope names every copy some face wears. A
        copy no face wears at all (its object deleted) is no material of any
        file. The stager stages exactly these (:meth:`stage_export_proxies`,
        :meth:`scope_export_proxies`), so the record and the proxies agree.

        Parameters:
            ctx: The write's ``ptk.ExportContext`` (its ``ships``); ``None``:
                no scope.
        """
        entries = []
        for copy, strength in cls._animated(ctx):
            colour = cls.HOST.emission_color(copy)
            entry = {
                "material": copy,
                "channels": cls._channels_of(copy),
                "emission_color": [float(c) for c in (colour or (1.0, 1.0, 1.0))[:3]],
                "emission_strength": float(cls.HOST.published_strength(copy, strength)),
            }
            if not cls.HOST.tint_refusal(copy):
                tint = [float(c) for c in cls.HOST.emission_tint(copy)[:3]]
                if any(abs(c - 1.0) > 1e-6 for c in tint) or cls.HOST.emission_tint_keys(copy):
                    entry["tint"] = tint
            states = cls.HOST.emission_states(copy)
            if states:
                entry["states"] = [
                    {"name": name, "tint": list(state["tint"]), "weight": float(state["weight"])}
                    for name, state in cls._clean_states(states).items()
                ]
            entries.append(entry)
        return entries

    @classmethod
    def _channels_of(cls, copy: str) -> List[str]:
        """The channels *copy*'s keys animate: ``emission`` (its weight) and
        the tint channels (``ptk.SceneRecords.TRANSPORT.tint_channels``) its
        colour does not leave black."""
        channels = ["emission"] if cls.HOST.emission_keys(copy) else []
        if cls.HOST.emission_tint_keys(copy):
            colour = cls.HOST.emission_color(copy) or (1.0, 1.0, 1.0)
            tints = ptk.SceneRecords.TRANSPORT.tint_channels
            channels += [name for name, c in zip(tints, colour) if float(c) > 0.0]
        return channels

    @classmethod
    def weight_curves(cls, keys_of: Callable[[str], Sequence[Sequence[Any]]], ctx=None) -> List[dict]:
        """Each animated copy's weight curve as its ``<copy>__emission`` proxy
        carries it (:meth:`stage_export_proxies`), in the scene's frames:
        ``{"target": copy leaf, "channel": "emission", "keys": [[frame,
        weight, interpolation?], ...]}`` -- the curves a write with no take
        states for its build (the hosts' ``RenderEffects.effect_curves``,
        ``ptk.MeshConvert.effect_preview_curves``). Pure: it reads the copies.

        Parameters:
            keys_of: The host's reader of a copy's emission keys, ``[frame,
                value, interpolation]`` each, its segment's interpolation
                stated (a curved segment sampled per frame); the values are
                divided by the copy's strength in the shader's own units (its
                tag's) and by what the keyed plug is scaled by
                (:meth:`EmissiveGroupsHost.emission_scale`: a colour
                channel's reference), which makes them the weight.
            ctx: The export's ``ptk.ExportContext`` (as
                :meth:`render_effect_materials`).
        """
        curves = []
        for copy, strength in cls._animated(ctx):
            scale = strength * cls.HOST.emission_scale(copy)
            keys = cls._scaled(keys_of(copy), scale)
            if keys:
                curves.append(
                    {
                        "target": ptk.SceneRecords.TRANSPORT.target_name(copy),
                        "channel": "emission",
                        "keys": keys,
                    }
                )
        return curves

    @classmethod
    def tint_curves(cls, keys_of: Callable[[Any], Sequence[Sequence[Any]]], ctx=None) -> List[dict]:
        """Each animated copy's tint curves as its ``<copy>__tint_r/g/b``
        proxies carry them (:meth:`stage_export_proxies`), in the scene's
        frames -- :meth:`weight_curves`' twin, for the same writes.

        Parameters:
            keys_of: The host's reader of a channel's keys (as
                :meth:`weight_curves`' reader), handed each of
                :meth:`EmissiveGroupsHost.emission_tint_channels`; the values
                are divided by the copy's colour's channel, which makes them
                the tint.
            ctx: The export's ``ptk.ExportContext`` (as
                :meth:`render_effect_materials`).
        """
        curves = []
        for copy, _strength in cls._animated(ctx):
            if not cls.HOST.emission_tint_keys(copy):
                continue
            colour = cls.HOST.emission_color(copy) or (1.0, 1.0, 1.0)
            plugs = cls.HOST.emission_tint_channels(copy)
            for name, plug, scale in zip(ptk.SceneRecords.TRANSPORT.tint_channels, plugs, colour):
                keys = cls._scaled(keys_of(plug), float(scale)) if float(scale) > 0.0 else []
                if keys:
                    curves.append(
                        {"target": ptk.SceneRecords.TRANSPORT.target_name(copy), "channel": name, "keys": keys}
                    )
        return curves

    @staticmethod
    def _scaled(keys, scale: float) -> List[list]:
        """``[frame, value / scale, interpolation?]`` per key of *keys*."""
        return [
            [float(k[0]), float(k[1]) / scale, *[str(i) for i in k[2:3]]]
            for k in keys or ()
            if len(k) >= 2
        ]

    @classmethod
    def stage_export_proxies(cls, ctx=None) -> List[Any]:
        """Stage each animated copy's weight on its ``<copy>__emission`` proxy
        (``ptk.SceneRecords.TRANSPORT``), and a keyed tint on its
        ``<copy>__tint_r/g/b`` ones, for an FBX write -- staged by the
        ``render_effects`` stager, which hands it the write's context. The
        weight curve is the copy's emission over its strength, 0..1; a tint
        curve its colour's channel over the base colour's.
        Copies out of step with their base are said (a warning, the export's
        check) and ship as they stand: never a swap at export. Stale proxies
        are cleaned first.

        Parameters:
            ctx: The write's ``ptk.ExportContext``: only the copies it ships
                are staged (:meth:`render_effect_materials`). ``None`` -- a
                writer that stages before it knows its scope -- stages every
                one, narrowed once the scope is known
                (:meth:`scope_export_proxies`).

        Returns:
            The staged proxies (the host's handles).
        """
        cls.HOST.remove_emission_proxies()
        for line in cls.sync(check_only=True):
            cls.logger.warning(f"{line} It exports as it stands; Sync puts it in step.")
        staged = []
        for copy, strength in cls._animated(ctx):
            for channel in cls._channels_of(copy):
                name = ptk.SceneRecords.TRANSPORT.proxy_name(copy, channel)
                if channel == "emission":
                    proxy = cls.HOST.stage_emission_proxy(copy, name, 1.0 / strength)
                else:
                    tint = ptk.SceneRecords.TRANSPORT.tint_channels.index(channel)
                    proxy = cls.HOST.stage_emission_proxy(copy, name, 1.0, tint)
                if proxy is None:
                    what = "weight" if channel == "emission" else "tint"
                    cls.logger.warning(
                        f"Curve-proxy name {name!r} is taken by another object: "
                        f"{copy}'s {what} animation does not ship this export."
                    )
                    continue
                staged.append(proxy)
        return staged

    @classmethod
    def scope_export_proxies(cls, ctx) -> List[str]:
        """Narrow what is staged to the write's scope, once it is known: the
        proxy of every copy the write does not ship goes. A writer that stages
        before it publishes its scoped records (a Scene Exporter, a hand-off
        that stages in its bracket) hands its scope over here through
        ``FbxUtils.publish``. Nothing goes for a write with no scope.

        Returns:
            The proxies removed.
        """
        if ctx is None or getattr(ctx, "nodes", None) is None:
            return []
        target_name = ptk.SceneRecords.TRANSPORT.target_name
        keep = {target_name(copy) for copy, _strength in cls._animated(ctx)}
        drop = [
            name
            for name in cls.HOST.emission_proxies()
            if (ptk.SceneRecords.TRANSPORT.parse(name) or ("", ""))[0] not in keep
        ]
        return cls.HOST.remove_emission_proxies(drop) if drop else []

    @classmethod
    def remove_export_proxies(cls) -> List[str]:
        """Delete every staged weight proxy (the stager's finish). Idempotent."""
        return cls.HOST.remove_emission_proxies()

    # ------------------------------------------------------------------
    # The old tool's groups
    # ------------------------------------------------------------------

    @classmethod
    def convert_legacy(cls, ctx=None) -> Dict[str, Any]:
        """Convert the old tool's groups (masks over slots) to materials, in
        one action and one undo step.

        Face-level membership converts exactly: each group's faces take its
        copies, resting at the group's default, a keyed carrier weight
        becoming the copies' keys. What a material cannot hold is reported by
        name with its fix, never guessed: a face in two groups stays in the
        lower slot's; a mask-encoded group's region painted inside a face (the
        mask texture) cannot be a material -- the fix is to cut those faces at
        the source. Then the old stores, weights, colour set and records go.

        Returns:
            ``{"converted": {group: faces}, "copies": {group: [copy]},
            "reported": [line], "removed": [what]}``; empty for a scene with
            nothing to convert.
        """
        registry = cls._legacy_registry() or {}
        stored = cls.HOST.legacy_store_names(cls.LEGACY_PREFIX)
        entries = dict(registry.get("groups") or {})
        for name in stored:
            entries.setdefault(name, {"slot": len(entries) + 100, "default": 1.0})
        report: Dict[str, Any] = {"converted": {}, "copies": {}, "reported": [], "removed": []}
        if not entries and not registry:
            return report
        ordered = sorted(entries, key=lambda n: (entries[n].get("slot", 99), n))
        if registry.get("encoding") == "channels":
            report["reported"].append(
                f"Groups {', '.join(ordered)} used the mask texture "
                f"{registry.get('mask') or '(unnamed)'}: their faces convert "
                "exactly, but a region painted inside a face cannot be a "
                "material -- cut those faces at the source, then add them to "
                "their group."
            )
        claimed: Dict[Tuple[str, int], str] = {}
        with cls.HOST.undo_step("Emissive Groups: convert"):
            for name in ordered:
                store = f"{cls.LEGACY_PREFIX}{name}"
                members = cls.HOST.legacy_members(store)
                if name not in stored and members is None:
                    report["reported"].append(f"Group {name!r}: its faces are gone; nothing converts.")
                    continue
                faces: FaceMap = {}
                for obj, indices in sorted((members or {}).items()):
                    taken = sorted(i for i in indices if (obj, i) in claimed)
                    if taken:
                        owner = claimed[(obj, taken[0])]
                        report["reported"].append(
                            f"Group {name!r}: {len(taken)} face(s) of {obj} are also in "
                            f"{owner!r}; a face wears one material, so they stay in "
                            f"{owner!r} -- split them at the source and add them to "
                            f"{name!r} if both should glow."
                        )
                    kept = [i for i in indices if (obj, i) not in claimed]
                    for i in kept:
                        claimed[(obj, i)] = name
                    if kept:
                        faces[obj] = kept
                cls._convert_group(name, entries[name], faces, report, ctx)
            removed = []
            attrs = [f"{cls.LEGACY_PREFIX}{n}" for n in ordered]
            stores = [f"{cls.LEGACY_PREFIX}{n}" for n in sorted(set(stored) | set(ordered))]
            if ctx is None or ctx.should_delete("the old tool's data", "converted"):
                removed = cls.HOST.remove_legacy(stores, attrs, cls.LEGACY_COLOR_SET)
                nodes = cls.HOST.data_nodes
                for scope, key in [(ptk.Scope.PRIVATE, k) for k in cls.LEGACY_REGISTRY_KEYS] + [
                    (ptk.Scope.DELIVERABLE, cls.LEGACY_RECORD_KEY)
                ]:
                    if nodes is not None and nodes.read(scope, key) is not None:
                        nodes.write(scope, key, None)
                        removed.append(key)
            report["removed"] = removed
        for line in report["reported"]:
            cls.logger.warning(line)
        cls.logger.info(f"Converted emissive groups: {report['converted'] or 'none'}.")
        return report

    @classmethod
    def _convert_group(cls, name, entry, faces: FaceMap, report, ctx) -> None:
        """One old group's faces onto copies, material by material (a material
        that cannot be grouped is reported and the rest still convert)."""
        if not faces:
            report["reported"].append(f"Group {name!r}: no face left to convert.")
            return
        default = float(entry.get("default", 1.0))
        attr = f"{cls.LEGACY_PREFIX}{name}"
        count = 0
        for material, faces_on in sorted(cls.HOST.face_materials(faces).items()):
            try:
                copies = cls.add_group(name, faces_on, weight=default, ctx=ctx)
            except ValueError as error:
                report["reported"].append(f"Group {name!r}, faces on {material!r}: {error}")
                continue
            count += sum(len(i) for i in faces_on.values())
            report["copies"][name] = copies
        if not count:
            return
        report["converted"][name] = count
        for copy, info in sorted(cls._members(name).items()):
            if not cls.HOST.emission_keys(copy) and (
                ctx is None or ctx.should_change(copy, "keyed by the old weight")
            ):
                cls.HOST.adopt_legacy_weight(attr, copy, float(info["strength"]))

    # ------------------------------------------------------------------
    # Hand-off crossings (``ptk.SceneRecords.EMISSIVE_REGISTRY``)
    # ------------------------------------------------------------------

    @classmethod
    def transfer_out(cls, ctx: "ptk.TransferContext") -> Optional[dict]:
        """The groups' hand-off payload: per copy its group, base, strength and
        emission keys (``stepped``: the frames of those that hold), and where
        it has them its named states and its tint (``tint`` now,
        ``tint_keys`` ``[frame, [r, g, b], stepped]``) -- an interchange file
        carries the materials and their faces, never the tags or the material
        animation. Scoped to the copies a face of ``ctx.objects`` wears;
        ``None`` without groups."""
        copies = cls.HOST.copies()
        if not copies:
            return None
        scope = None
        if ctx.objects is not None:
            scope = {cls._leaf(o) for o in ctx.objects}
        out = {}
        for copy, info in sorted(copies.items()):
            if not info.get("base"):
                continue
            if scope is not None and not any(
                cls._leaf(o) in scope for o in cls.HOST.material_faces(copy)
            ):
                continue
            entry = {
                "group": info["group"],
                "base": info["base"],
                "strength": float(info["strength"]),
                "keys": [[float(f), float(v)] for f, v in cls.HOST.emission_keys(copy)],
            }
            stepped = cls.HOST.stepped_emission_frames(copy)
            if stepped:
                entry["stepped"] = sorted(float(f) for f in stepped)
            states = cls.HOST.emission_states(copy)
            if states:
                entry["states"] = cls._clean_states(states)
            if not cls.HOST.tint_refusal(copy):
                tint_keys = cls.HOST.emission_tint_keys(copy)
                tint = [float(c) for c in cls.HOST.emission_tint(copy)]
                if tint_keys:
                    entry["tint_keys"] = [[float(f), [float(c) for c in t], bool(s)] for f, t, s in tint_keys]
                if tint_keys or any(abs(c - 1.0) > 1e-6 for c in tint):
                    entry["tint"] = tint
            out[copy] = entry
        return {"copies": out} if out else None

    @classmethod
    def transfer_in(cls, payload: dict, ctx: "ptk.TransferContext") -> None:
        """Land a received payload (:meth:`transfer_out`'s shape, from either
        host): each arrived copy is tagged again, keyed when it arrived with no
        keys (a stepped key held), given its states, and its tint against its
        base's colour (the copy put in step with its base first: it arrives
        showing the tinted colour). A copy or base that did not arrive is
        noted. Names go
        through ``ctx.spell`` (the record does not respell on its own): a
        material this side renamed on import (a ``.001`` suffix, a ``_``
        for a ``.``) is found under its new name."""
        for copy_name, entry in ((payload or {}).get("copies") or {}).items():
            copy = cls.HOST.material(ctx.spell(copy_name))
            base = cls.HOST.material(ctx.spell(entry.get("base") or ""))
            if copy is None or base is None:
                ctx.note(
                    f"Emissive group {entry.get('group')!r}: {copy_name!r} or its "
                    f"base {entry.get('base')!r} did not arrive; it is a plain "
                    "material here."
                )
                continue
            cls.HOST.tag(copy, entry["group"], base, float(entry["strength"]))
            if entry.get("keys") and not cls.HOST.emission_keys(copy):
                held = {round(float(f), 4) for f in entry.get("stepped") or ()}
                for frame, value in entry["keys"]:
                    cls.HOST.key_emission(copy, float(value), float(frame), round(float(frame), 4) in held)
            if entry.get("states"):
                cls.HOST.set_emission_states(copy, cls._clean_states(entry["states"]))
            if (entry.get("tint") or entry.get("tint_keys")) and not cls.HOST.emission_tint_keys(copy):
                # It arrives showing its base's colour x the tint it had: put in
                # step with its base first, the colour a tint multiplies.
                cls.HOST.sync_copy(copy, base)
                reason = cls.HOST.tint_refusal(copy)
                if reason:
                    ctx.note(f"Emissive group {entry['group']!r}: {copy!r} cannot be tinted here ({reason}).")
                    continue
                for frame, tint, stepped in entry.get("tint_keys") or ():
                    cls.HOST.key_emission_tint(copy, cls._clean_tint(tint), float(frame), bool(stepped))
                if not entry.get("tint_keys"):
                    cls.HOST.set_emission_tint(copy, cls._clean_tint(entry["tint"]))
