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

    def key_emission(self, material: str, value: float, frame: Optional[float] = None) -> None:
        """Key *material*'s emission at *value*, at *frame* (``None``: now)."""

    def clear_emission_keys(self, material: str) -> None:
        """Delete *material*'s emission keys, keeping its value now."""

    def copy_emission(self, src: str, dst: str, factor: float) -> None:
        """*dst*'s emission becomes *src*'s times *factor*: its value, and its
        keys with their shape."""

    def scale_emission(self, material: str, factor: float) -> None:
        """Scale *material*'s emission value and keys by *factor*."""

    # Export --------------------------------------------------------------

    def stage_emission_proxy(self, material: str, name: str, factor: float) -> Optional[Any]:
        """Stage the transport proxy *name* under the export's carrier node:
        a marked transform whose curve channel carries *material*'s emission
        keys times *factor*. ``None`` when *name* is taken."""

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

    def key_emission(self, material: str, value: float, frame: Optional[float] = None) -> None:
        self._refuse(f"key {material!r}'s emission")

    def clear_emission_keys(self, material: str) -> None:
        return None  # no keys to delete

    def copy_emission(self, src: str, dst: str, factor: float) -> None:
        self._refuse(f"key {dst!r}'s emission")

    def scale_emission(self, material: str, factor: float) -> None:
        self._refuse(f"scale {material!r}'s emission")

    def stage_emission_proxy(self, material: str, name: str, factor: float) -> Optional[Any]:
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
        """``(copy, strength)`` for each keyed copy the write ships
        (:meth:`EmissiveGroupsModel.render_effect_materials` says which), in
        name order: *strength* the tag's, in the shader's own units -- what
        the copy's weight scales, so its keys over it are the weight."""
        out = []
        for copy, info in sorted(cls.HOST.copies().items()):
            strength = float(info.get("strength") or 0.0)
            if strength <= 0.0 or not cls.HOST.emission_keys(copy):
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
        """``{name: {"weight", "keyed", "materials", "bases", "faces"}}`` by
        name: the weight now, whether a copy is keyed, the copies and their
        bases (``None`` for a lost one) and the face count."""
        grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for copy, info in cls.HOST.copies().items():
            grouped.setdefault(info.get("group"), {})[copy] = info
        out = {}
        for name in sorted(grouped):
            members = grouped[name]
            out[name] = {
                "weight": cls._weight_of(members),
                "keyed": cls._keyed_copy(members) is not None,
                "materials": sorted(members),
                "bases": [members[c].get("base") for c in sorted(members)],
                "faces": sum(
                    len(i)
                    for c in members
                    for i in cls.HOST.material_faces(c).values()
                ),
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
        animation's, left alone.

        Returns:
            The weight set.
        """
        weight = max(0.0, min(1.0, float(weight)))
        members = cls._known(name)
        with cls.HOST.undo_step(f"Emissive Groups: weight {name}"):
            for copy, info in sorted(members.items()):
                if cls.HOST.emission_keys(copy):
                    continue
                if ctx is None or ctx.should_change(copy, f"weight {weight:g}"):
                    cls.HOST.set_emission(copy, float(info["strength"]) * weight)
        return weight

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
        if value is None:
            value = cls._weight_of(members)
        value = max(0.0, min(1.0, float(1.0 if value is None else value)))
        keyed = []
        with cls.HOST.undo_step(f"Emissive Groups: key {name}"):
            for copy, info in sorted(members.items()):
                if ctx is None or ctx.should_change(copy, f"key weight {value:g}"):
                    cls.HOST.key_emission(copy, float(info["strength"]) * value, frame)
                    keyed.append(copy)
        return keyed

    @classmethod
    def remove_keys(cls, name: str, ctx=None) -> List[str]:
        """Delete the group's weight keys on every copy, keeping the weight now.

        Returns:
            The copies that had keys.
        """
        cleared = []
        with cls.HOST.undo_step(f"Emissive Groups: remove keys {name}"):
            for copy in sorted(cls._known(name)):
                if cls.HOST.emission_keys(copy) and (
                    ctx is None or ctx.should_change(copy, "remove weight keys")
                ):
                    cls.HOST.clear_emission_keys(copy)
                    cleared.append(copy)
        return cleared

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
        with cls.HOST.undo_step("Emissive Groups: sync"):
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
        """Each animated copy as a ``render_effects`` material entry:
        ``{"material", "channels": ["emission"], "emission_color",
        "emission_strength"}`` -- the colour its weight scales (white over a
        map, which glTF multiplies) and its base's strength as the delivery
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
            entries.append(
                {
                    "material": copy,
                    "channels": ["emission"],
                    "emission_color": [float(c) for c in (colour or (1.0, 1.0, 1.0))[:3]],
                    "emission_strength": float(cls.HOST.published_strength(copy, strength)),
                }
            )
        return entries

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
            keys = [
                [float(k[0]), float(k[1]) / scale, *[str(i) for i in k[2:3]]]
                for k in keys_of(copy) or ()
                if len(k) >= 2
            ]
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
    def stage_export_proxies(cls, ctx=None) -> List[Any]:
        """Stage each animated copy's weight on its ``<copy>__emission`` proxy
        (``ptk.SceneRecords.TRANSPORT``) for an FBX write -- staged by the
        ``render_effects`` stager, which hands it the write's context. The
        curve is the copy's emission over its strength: the weight, 0..1.
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
            name = ptk.SceneRecords.TRANSPORT.proxy_name(copy, "emission")
            proxy = cls.HOST.stage_emission_proxy(copy, name, 1.0 / strength)
            if proxy is None:
                cls.logger.warning(
                    f"Curve-proxy name {name!r} is taken by another object: "
                    f"{copy}'s weight animation does not ship this export."
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
        emission keys -- an interchange file carries the materials and their
        faces, never the tags or the material animation. Scoped to the copies
        a face of ``ctx.objects`` wears; ``None`` without groups."""
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
            out[copy] = {
                "group": info["group"],
                "base": info["base"],
                "strength": float(info["strength"]),
                "keys": [[float(f), float(v)] for f, v in cls.HOST.emission_keys(copy)],
            }
        return {"copies": out} if out else None

    @classmethod
    def transfer_in(cls, payload: dict, ctx: "ptk.TransferContext") -> None:
        """Land a received payload (:meth:`transfer_out`'s shape, from either
        host): each arrived copy is tagged again, and keyed when it arrived
        with no keys. A copy or base that did not arrive is noted."""
        for copy_name, entry in ((payload or {}).get("copies") or {}).items():
            copy = cls.HOST.material(copy_name)
            base = cls.HOST.material(entry.get("base") or "")
            if copy is None or base is None:
                ctx.note(
                    f"Emissive group {entry.get('group')!r}: {copy_name!r} or its "
                    f"base {entry.get('base')!r} did not arrive; it is a plain "
                    "material here."
                )
                continue
            cls.HOST.tag(copy, entry["group"], base, float(entry["strength"]))
            if entry.get("keys") and not cls.HOST.emission_keys(copy):
                for frame, value in entry["keys"]:
                    cls.HOST.key_emission(copy, float(value), float(frame))
