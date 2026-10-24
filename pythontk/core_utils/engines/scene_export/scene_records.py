# !/usr/bin/python
# coding=utf-8
"""Scene records -- every piece of tool-authored scene metadata, declared once.

A *record* is one JSON channel a DCC tool keeps on the scene: the shot store,
a lightmap manifest, the keyed-visibility tracks.  Each has exactly one
declaration here (:class:`SceneRecords`), and everything that used to spell a
channel name, version or description on its own -- the DCC carriers, the
producers, the FBX handoff block, the doc tables, the Unity importer gate --
derives from that declaration instead.  Four ideas, one module each in this
engine (``scene_records`` / ``scene_store`` / ``export_snapshot`` /
``record_transfer``):

- **Declare once, derive everything.** :class:`RecordSpec` is the declaration:
  key, scope, version, description, kind and dependencies -- and, where
  another language reads it, its payload's shape and the block a GLB
  carries it as (:class:`GlbBlock`).  Nothing else spells a channel name,
  and the other languages' types are generated from the shapes.
- **Storage is dumb, records are smart.** :class:`SceneStoreBase` is the whole
  contract a DCC implements: ``read`` / ``write`` / ``keys`` / ``values`` per
  :class:`Scope`, strings only.  Encoding, the version envelope, tolerant
  decoding and clear-on-empty live on the record, so the two DCC mirrors
  cannot diverge on semantics.
- **Compute, then commit, once.** A producer RETURNS a :class:`Record` and
  never writes.  :meth:`ExportSnapshot.assemble` orders the producers by
  their declared dependencies, hands each the :class:`ExportContext` (the
  exporter's decisions as INPUT, plus every record produced before it) and
  :meth:`ExportSnapshot.commit` writes the whole snapshot in one pass and
  stamps the handoff block.  The same snapshot object is what the sidecar,
  the export log and the verifier consume, so nothing is read back off the
  node and patched.

- **Cross, by declaration.** A record also says how it crosses into
  another scene (:class:`Merge`, :attr:`RecordSpec.portable`), so every
  route a record takes -- a referenced module imported into its host, a
  DCC hand-off, a legacy fold -- is one engine (:class:`RecordTransfer`)
  over the declarations.  A record's own semantics plug in as a codec (its
  merge) and a DCC owner (what it keeps beside the record); a new record
  crosses correctly by being declared, with no route learning its name.

Two scopes, because the carriers differ in what they must guarantee:
:attr:`Scope.PRIVATE` records persist with the scene and never leave it;
:attr:`Scope.DELIVERABLE` records ride every deliverable as user properties.
The DCC picks the primitive that makes each guarantee structural (Maya: a
``network`` node and a hidden transform; Blender: a scene ID property group
and an Empty) -- this module never knows.

Zero-dep and DCC-agnostic, like everything in ``pythontk``.
"""

import copy
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from pythontk.core_utils.class_property import ClassProperty
from pythontk.core_utils.deprecation import Deprecation

logger = logging.getLogger(__name__)


class Scope(str, Enum):
    """Where a record lives, and therefore what it can never do."""

    #: Persists with the scene file and never leaves it -- app state, restore
    #: manifests, registries.  Structurally outside every export.
    PRIVATE = "private"
    #: Rides every deliverable (FBX user properties, glTF extras, USD custom
    #: attributes) on the one shared carrier node.
    DELIVERABLE = "deliverable"


class Kind(str, Enum):
    """What a record is a function of, which decides WHEN it is refreshed."""

    #: A projection of state the artist authors (a store, markers, a rig).
    #: Republished at authoring time; refreshed again by an export PIPELINE,
    #: which is the authority on every channel.  A hand-off that merely ships
    #: the carrier leaves it as authored: a producer with nothing to say
    #: CLEARS its record, and a bridge is not the authority on a bake the
    #: scene's markers no longer describe (measured: a preview push wiped a
    #: lightmap manifest and previewed the asset unlit).
    AUTHORED = "authored"
    #: Computed from the live scene (the curves themselves) and therefore
    #: stale the moment an artist edits; refreshed before EVERY write,
    #: hand-offs included.
    DERIVED = "derived"


class Merge(str, Enum):
    """What happens to a record when another scene's copy arrives beside the
    scene's own -- a referenced module imported into its host, a hand-off
    landing in a scene that already has one.  Declared per record
    (:attr:`RecordSpec.merge`), so :class:`RecordTransfer` is one loop over
    the declarations and a new record decides its merge where it is declared.
    """

    #: Produced again from the merged scene, never combined as data.  A
    #: deliverable is a projection -- of per-object markers, a private
    #: registry, the curves -- and the other scene's copy spells names as THAT
    #: scene did, so the merged scene publishes it afresh instead.
    DERIVE = "derive"
    #: The scene's own copy stands and the other's is dropped: it describes
    #: its own scene and nothing else (an export hierarchy baseline).
    OWN = "own"
    #: Entries combine by identity -- a mapping's keys, or a list's
    #: :attr:`RecordSpec.merge_key` field -- and the scene's own entry wins a
    #: collision (noted).  A list keeps the scene's own entries LAST, so a
    #: LIFO stack's newest entry stays on top.
    UNION = "union"
    #: A domain merge owns it (:meth:`SceneRecords.codec`): the shot store
    #: renumbers shots, the emissive registry re-slots a group whose slot is
    #: taken.
    CODEC = "codec"


#: A validator message naming no key of its own (``SchemaSpec.validate``'s
#: "missing required key 'x'", "expected a mapping, got str"), which a nested
#: spec's path joins with a dot ("tracks[0].missing required key 'node'").
_PATHLESS = re.compile(
    r"^(?:(?P<path>.+?)\.)?"
    r"(?P<message>(?:missing required key|expected a mapping, got) .*)$"
)


def _located(where: str, error: str) -> str:
    """*error* -- a ``SchemaSpec.validate`` line -- said at *where* as every
    problem line is: where it is, then what is wrong, behind a colon."""
    found = _PATHLESS.match(error)
    if found is None:
        return f"{where}.{error}"
    path = f"{where}.{found['path']}" if found["path"] else where
    return f"{path}: {found['message']}"


def _declared(key: str, shape: Any, body: Dict[str, Any]) -> Dict[str, Any]:
    """*body* pruned to *shape* (:meth:`SchemaSpec.prune`), each key it cut
    warned under the record's *key*: a writer's drift said, never shipped."""
    dropped: List[str] = []
    body = shape.prune(body, dropped)
    if dropped:
        # One name per field, however many entries carried it.
        fields = sorted({re.sub(r"\[\d+\]", "[]", path) for path in dropped})
        logger.warning(
            "%s: left out what its shape does not declare (%s).",
            key,
            ", ".join(fields),
        )
    return body


class _Unknown:
    """A held record whose content is not at hand (:attr:`Clause.UNKNOWN`)."""

    def __repr__(self) -> str:
        return "Clause.UNKNOWN"


@dataclass(frozen=True)
class Clause:
    """A sentence of an assembly that only some files need.

    It applies when the payload holds *value* at *path* -- keys in turn, a
    list fanning out over its entries -- or, with *value* left out, holds
    anything there but null or an empty value. So a file states only what
    its own content uses: a shadows record of projected planes carries no
    horizon march, a scene without lightmaps no lightmap rule. Without a
    payload to test (:meth:`RecordSpec.assembly_for` with none) every clause
    applies -- the full statement the generated docs print.

    Parameters:
        text: The sentence or sentences.
        path: Keys into the payload; empty, it always applies.
        value: What some entry at *path* must equal, type included (``True``
            is not ``1``); ``None`` for anything present.
    """

    #: A payload known to be held but not at hand: whatever a clause looks
    #: for in it may be there, so the clause applies.
    UNKNOWN: ClassVar[object] = _Unknown()

    text: str
    path: Tuple[str, ...] = ()
    value: Any = None

    def applies(self, payload: Any) -> bool:
        """Whether *payload* calls for this clause."""
        return Clause.holds(payload, self.path, self.value)

    @staticmethod
    def holds(payload: Any, path: Tuple[str, ...], value: Any = None) -> bool:
        """Whether *payload* holds *value* at *path* (or, with *value*
        ``None``, anything there but null or an empty value) -- the test a
        clause applies by, for anything else a file's content decides
        (``HandoffRecord.RENDERING_NEEDS``). An empty path always holds."""
        if not path:
            return True
        found = [payload]
        for key in tuple(path) + (None,):
            step = []
            for item in found:
                if item is Clause.UNKNOWN:
                    return True
                for entry in item if isinstance(item, list) else (item,):
                    if key is None:
                        step.append(entry)
                    elif isinstance(entry, Mapping) and key in entry:
                        step.append(entry[key])
            found = step
        if value is None:
            return any(v is not None and v != "" and v != {} for v in found)
        return any(type(v) is type(value) and v == value for v in found)

    @property
    def condition(self) -> str:
        """When the clause applies, in words (for the generated docs)."""
        if not self.path:
            return "always"
        where = ".".join(self.path)
        if self.value is None:
            return f"when `{where}` is present"
        return f"when `{where}` is {json.dumps(self.value)}"

    @staticmethod
    def join(text: str, clauses: Iterable["Clause"], payload: Any = None) -> str:
        """*text* and each of *clauses* that *payload* calls for -- every one
        when *payload* is ``None`` -- as one statement."""
        parts = [text] + [
            c.text for c in clauses if payload is None or c.applies(payload)
        ]
        return " ".join(p.strip() for p in parts if p and p.strip())


@dataclass(frozen=True)
class GlbBlock:
    """How a deliverable record rides a GLB: a block in its root ``extras``.

    A GLB conversion (``MeshConvert``) reads the record off the carrier, binds
    it to the file -- names resolved to glTF node indices, maps to texture
    indices -- and writes the result in the root ``extras`` under the
    record's own key (:attr:`RecordSpec.key`): one name for the record in
    every file, the key ``PreviewServer.AUTO_SCRIPTS`` activates a viewer
    script on.

    Declared here, beside the record, for the same reason the record is: one
    statement of each block, whichever exporter writes it -- the GLB's
    handoff takes its ``assembly`` row from these.

    Parameters:
        version: The block's own schema version, which its writer stamps
            (independent of the record's: the binding can change alone).
        shape: ``(module, class)`` of the :class:`~pythontk.SchemaSpec`
            declaring the block (:meth:`SceneRecords.glb_shape`); ``None``
            while it is undeclared.
        assembly: What the block makes of the approved scene: its row's
            ``spec`` in the GLB handoff's ``assembly``, before its
            *clauses* (:meth:`assembly_for`).
        node_fields: Where the block binds by glTF NODE INDEX, as paths
            (``("planes", "node")``: each plane's ``node``) -- what a pass
            that renumbers the file's nodes follows.
        clauses: The :class:`Clause` s of the assembly that only some blocks
            need, tested against the block.
    """

    version: int = 1
    shape: Optional[Tuple[str, str]] = None
    assembly: str = ""
    node_fields: Tuple[Tuple[str, ...], ...] = ()
    clauses: Tuple[Clause, ...] = ()

    def assembly_for(self, payload: Any = None) -> str:
        """The assembly a block *payload* needs (:meth:`Clause.join`)."""
        return Clause.join(self.assembly, self.clauses, payload)

    @property
    @Deprecation.symbol(
        "RecordSpec.key -- a block takes its record's key",
        remove_in="0.18.0",
        since="2026-10-24",
    )
    def key(self) -> str:
        """The record's key, which the block is written under; empty for a
        block no :class:`RecordSpec` holds."""
        return self.__dict__.get("_record_key", "")


@Deprecation.symbol(
    "GlbBlock -- a record's block is named for the file it rides",
    remove_in="0.18.0",
    since="2026-10-24",
)
class WebProjection(GlbBlock):
    """Retired 2026-10-20 (removed in 0.18.0): :class:`GlbBlock`.

    Builds as it did in 0.16.1 -- ``WebProjection(key, version=1,
    shape=None)`` -- its *key* what :attr:`GlbBlock.key` reads until a
    :class:`RecordSpec` holds it."""

    def __init__(
        self, key: str, version: int = 1, shape: Optional[Tuple[str, str]] = None
    ) -> None:
        super().__init__(version=version, shape=shape)
        object.__setattr__(self, "_record_key", key)


@dataclass(frozen=True)
class Record:
    """One produced value of a :class:`RecordSpec`, ready to store."""

    spec: "RecordSpec"
    payload: Any

    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def text(self) -> str:
        """The stored form (JSON)."""
        return self.spec.encode(self.payload)

    def save(self, store) -> Optional[str]:
        """Write this record through *store*; see :meth:`RecordSpec.save`."""
        return self.spec.write_text(store, self.text)


@dataclass(frozen=True)
class RecordSpec:
    """The one declaration of a record: identity, contract and codec.

    Parameters:
        key: The channel name on the carrier -- stable, it is the wire contract
            every reader keys on (a Unity importer, a GLB applier).
        scope: :class:`Scope`.
        version: The schema this declaration describes.  An enveloped record
            carries it in its payload; a reader reads that schema alone
            (:meth:`is_current`).
        owner: The tool that produces it (a role name, DCC-agnostic).
        description: What the record holds, for the handoff block and the
            docs -- the one sentence a standalone reader gets.
        kind: :class:`Kind`; decides when the record is refreshed.
        after: Keys of the records this one's producer READS, so the
            coordinator orders producers by declaration rather than by dict
            order and a comment.
        envelope: True for a dict payload carrying its version (every
            deliverable record); False for a payload stored as-is (a store's
            own document).
        consumers: Names of the readers, for the docs and the cross-package
            gates (``"unity"``: the C# importers; ``"glb"``: the GLB appliers;
            ``"verifier"``: the deliverable gates).
        merge: :class:`Merge` -- what happens when another scene's copy of
            this record arrives beside the scene's own.  A deliverable is
            re-derived (the default); every private record declares one.
        merge_key: For a :attr:`Merge.UNION` list, the field an entry is
            identified by (``"id"``); ``None`` for a mapping, whose keys are
            the identity.
        respell: Whether the payload spells scene names, which a crossing
            respells through :attr:`TransferContext.rename` before combining.
            False for a payload of group names or file paths, where a node
            that happens to share a name must not rename an entry.
        portable: Whether the record crosses a DCC hand-off (a bridge's
            sidecar): its payload means the same in both DCCs once names are
            respelled, or its DCC owner makes it so.  A record bound to one
            DCC's constructs (a restore manifest of Maya plugs, parked curves)
            does not; a deliverable never does -- the far side re-derives it.
        section: The hand-off sidecar section a portable record rides as a
            whole (``"shots"`` -- the section it had before records were
            generic, which older producers still write); ``None`` rides the
            one generic ``records`` section, keyed by :attr:`key`.
        paths: Whether the payload is a mapping whose string VALUES are file
            or folder paths, each spelled relative to the scene's own project
            (``FileUtils.portable_path``).  Such a record is re-spelled when
            its scene is saved into another project
            (:meth:`SceneRecords.rebase_paths`), leaves a hand-off absolute
            and arrives spelled from the receiving scene's project
            (:class:`TransferContext`).  A tuple of keys names the only values
            that are paths, in a payload that is otherwise data: the hierarchy
            baseline's writer stamp, beside a path set and a hash.
        stamps: Whether the payload's values are WRITER STAMPS -- the scene
            file that made each entry (``SceneStoreBase.writer_stamp``), ``""``
            for one made while the scene was unsaved -- and are given the
            file the scene is first saved as (:meth:`SceneRecords.stamp_unsaved`).
            A tuple of keys names the only values that are (the hierarchy
            baseline's ``scene``).  A stamp is a path too: declare it in
            :attr:`paths` as well.
        shape: ``(module, class)`` of the :class:`~pythontk.SchemaSpec`
            declaring the payload (:meth:`SceneRecords.shape`) -- resolved
            lazily, like :attr:`SceneRecords.CODECS`, so declaring a record
            never imports its model. It validates a payload, and the types
            another language reads it through are generated from it
            (``m3trik/scripts/sync_scene_records.py``). ``None`` while the
            payload is described only in prose.
        glb: The record's :class:`GlbBlock` -- the block a GLB carries it
            as; ``None`` when a GLB carries none.
        export_scoped: Whether the record's producer narrows it to the export
            set (:attr:`ExportContext.scope`): its entries are scene nodes and
            the files they name ride the deliverable, so an entry for a node
            the write leaves out ships that node's files with nothing to bind
            them to. The scene's own copy stays whole: a scoped assembly keeps
            it (:attr:`ExportSnapshot.scene_wide`) for the exporter to put back.
        assembly: What the record makes of the approved scene, field by field
            -- the handoff block's ``assembly`` row for it, so a reader holding
            nothing but the deliverable rebuilds what the record describes
            instead of only knowing it is there.  Declarative: what the scene
            IS, never a step to take outside the file.  Empty for a private
            record, which never ships.  The part every payload needs; what
            only some do is a *clause* (:meth:`assembly_for`).
        transient: Whether the record describes one WRITE rather than the
            scene: produced from the scene as staged for that write (an
            export's converted maps, its baked keys), carried by the file, and
            never held by the scene -- the exporter clears it once the write
            is done (:meth:`ExportSnapshot.restore_scene_wide`) and no crossing
            takes one in (:class:`RecordTransfer`).  So a later write that
            does not produce it (a File > Export) ships none rather than a
            stale one naming files the export's restore deleted.  A
            :attr:`Kind.DERIVED` record: it is refreshed for every write.
        renamed_from: The keys a scene stored the record under before *key*;
            a carrier still holding one has it moved to *key*, once
            (:meth:`SceneStoreBase.migrate`). No reader reads them.
        clauses: The :class:`Clause` s of the assembly that only some
            payloads need: a file carries the ones its record calls for.
    """

    key: str
    scope: Scope
    version: int
    owner: str
    description: str
    kind: Kind = Kind.AUTHORED
    after: Tuple[str, ...] = ()
    envelope: bool = True
    consumers: Tuple[str, ...] = ()
    merge: Merge = Merge.DERIVE
    merge_key: Optional[str] = None
    respell: bool = True
    portable: bool = False
    section: Optional[str] = None
    paths: Union[bool, Tuple[str, ...]] = False
    stamps: Union[bool, Tuple[str, ...]] = False
    shape: Optional[Tuple[str, str]] = None
    glb: Optional[GlbBlock] = None
    export_scoped: bool = False
    assembly: str = ""
    transient: bool = False
    renamed_from: Tuple[str, ...] = ()
    clauses: Tuple[Clause, ...] = ()

    def assembly_for(self, payload: Any = None) -> str:
        """The assembly a record *payload* needs (:meth:`Clause.join`): every
        clause without one."""
        return Clause.join(self.assembly, self.clauses, payload)

    def __post_init__(self) -> None:
        if self.transient and self.kind is not Kind.DERIVED:
            raise ValueError(
                f"{self.key!r} is transient, so it is produced for every write: "
                "declare it Kind.DERIVED."
            )
        if self.glb is not None:
            # What the retired ``GlbBlock.key`` serves until its removal, on a
            # block of this record's own: stamped on the one handed over, a
            # ``dataclasses.replace`` of this record would re-key the
            # original's. A copy, not a ``replace``, which would rebuild a
            # ``WebProjection`` through an ``__init__`` that wants its key.
            glb = copy.copy(self.glb)
            object.__setattr__(glb, "_record_key", self.key)
            object.__setattr__(self, "glb", glb)

    @property
    @Deprecation.symbol(
        "RecordSpec.glb -- a record's block is named for the file it rides",
        remove_in="0.18.0",
        since="2026-10-24",
    )
    def web(self) -> Optional[GlbBlock]:
        """The record's GLB block (:attr:`glb`)."""
        return self.glb

    @property
    @Deprecation.symbol(
        '"version" -- every record states its version under that key',
        remove_in="0.18.0",
        since="2026-10-24",
    )
    def version_key(self) -> str:
        """The key every record's version rides under: ``"version"``."""
        return "version"

    def is_current(self, payload: Any) -> bool:
        """Whether *payload* is this record in the schema this declaration
        describes -- the one a reader reads. An older payload is refused as a
        newer one is: a file written before the current schema is re-exported,
        never read through a branch for its shape."""
        return isinstance(payload, Mapping) and payload.get("version") == self.version

    @property
    def path_keys(self) -> Optional[Tuple[str, ...]]:
        """The payload keys whose values are paths (:attr:`paths`): the named
        ones, or ``None`` -- every value -- for a whole path mapping."""
        return self.paths if isinstance(self.paths, tuple) else None

    @property
    def stamp_keys(self) -> Optional[Tuple[str, ...]]:
        """The payload keys whose values are writer stamps (:attr:`stamps`):
        the named ones, or ``None`` -- every value -- for a whole mapping."""
        return self.stamps if isinstance(self.stamps, tuple) else None

    # ------------------------------------------------------------------ codec
    def make(self, payload: Any) -> Record:
        """A :class:`Record` of *payload*, the envelope's ``version`` applied.

        An enveloped record's payload must be a mapping; its ``version`` is
        stamped from this declaration when absent, so a producer never spells
        the number. A declared record's payload is pruned to its shape
        (:meth:`SchemaSpec.prune`): a file states what its readers read and
        nothing more, and what a producer added beyond the shape is warned --
        its drift said, never shipped. A non-enveloped payload is taken as
        given.

        Raises:
            TypeError: An enveloped record handed a non-mapping payload.
        """
        if self.envelope:
            if not isinstance(payload, Mapping):
                raise TypeError(
                    f"{self.key!r} is an enveloped record and needs a mapping "
                    f"payload, not {type(payload).__name__}."
                )
            # Version FIRST (a producer's own value wins), the order every
            # producer wrote by hand before the declaration stamped it.
            body = {"version": self.version}
            body.update(payload)
            if self.shape:
                body = _declared(
                    self.key, SceneRecords.resolve_class(*self.shape), body
                )
            return Record(self, body)
        return Record(self, payload)

    def encode(self, payload: Any) -> str:
        """The stored text for *payload*.

        ``default=str`` on purpose: a value json cannot encode (a Path in a
        manifest) is recorded as its string form rather than failing the
        write that carries it.
        """
        return json.dumps(payload, default=str)

    def decode(self, text: Optional[str], default: Any = None) -> Any:
        """*text* parsed, or *default* when absent, cleared, not JSON, or newer
        than this declaration knows.

        Tolerant of both halves of "cannot be read".  A newer version is the
        one refusal: the reader would misread it, and a warning names the
        package that needs updating.
        """
        if not text:
            return default
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            return default
        if self.envelope and isinstance(payload, Mapping):
            found = payload.get("version")
            if isinstance(found, (int, float)) and found > self.version:
                logger.warning(
                    "%s record is version %s; this reader knows %s -- ignored.",
                    self.key,
                    found,
                    self.version,
                )
                return default
        return payload

    # ------------------------------------------------------------------ store
    def read_text(self, store) -> Optional[str]:
        """The raw stored text, or ``None``."""
        return store.read(self.scope, self.key)

    def write_text(self, store, text: Optional[str]) -> Optional[str]:
        """Store *text*; empty or ``None`` clears without creating a carrier."""
        return store.write(self.scope, self.key, text or None)

    def load(self, store, default: Any = None) -> Any:
        """The decoded payload from *store*, or *default*."""
        return self.decode(self.read_text(store), default)

    def save(self, store, payload: Any) -> Optional[str]:
        """Publish *payload* (the publish / clear idiom in one call).

        A falsy payload CLEARS the record rather than storing an empty one --
        deleting the last shot must not leave the previous takes riding into
        the next export, and a clear never creates a carrier just to hold
        nothing.  Returns what the store's ``write`` returns (its carrier
        name, or ``None`` when a clear had nothing to do).
        """
        if not payload:
            return self.clear(store)
        return self.make(payload).save(store)

    def clear(self, store) -> Optional[str]:
        """Clear the record; never creates a carrier (see :meth:`save`)."""
        return self.write_text(store, None)

    def is_present(self, store) -> bool:
        """Whether *store* holds a non-empty value for this record."""
        return bool(self.read_text(store))


# The texts a record and its block share, stated once. The shadows record's
# runtime restates pythontk's ShadowProjection.model / ShadowModel.placement
# and the horizon map's layout and march (HeightFieldMap.to_rgba / .alpha, the
# reference every engine shader is pinned to) so a reader with only the file
# rebuilds every plane: change them together. test_scene_records pins the
# constants quoted here to the code's and implements the text cold.
_LIGHTMAP_PROBE = Clause(
    "'probe' is the reflection probe the bake captured: the scene's "
    "environment (rendering.environment).",
    ("probe",),
)
_SHADOW_PLANE = (
    "Each plane is a ground quad, centred on its node's origin, showing the "
    "shadow its source casts from the prop standing on its contact; every "
    "length in the record times 'unit_scale' is metres, and up is +Y. It "
    "draws in its "
    "material's colour at alpha = shadow * min(the material's opacity, "
    "'intensity'). Its source places the shadow alone and lights nothing."
)
_SHADOW_PROJECTED = (
    "A projected plane's shadow is the alpha of {map} through the quad's own "
    "UVs, which then address its tile; its colour is the material's times "
    "that map's."
)
_SHADOW_PLACEMENT = Clause(
    "With 'follow_source' a runtime re-places the quad every frame (otherwise "
    "its imported keys stand). All in metres, with .xz a point's horizontal "
    "(x, z) part, C the contact's position and L the source's -- a "
    "directional source's is "
    "C - s*1e6*max(height, 2*radius), s the unit direction it shines, its "
    "node's local -Y: d = C.xz - L.xz, dist = |d|, u = d/dist ((0, 1) when "
    "dist < 1e-6), k_max = 1 + max_stretch, k_base = clamp((L.y - "
    "ground)/max(1e-4, L.y - C.y), 0, k_max), k_top = clamp((L.y - "
    "ground)/max(1e-4, L.y - C.y - height), 0, min(k_max, k_base + "
    "max_stretch*height/max(dist, 1e-6))), reach = max(0, dist*(k_top - "
    "k_base)), width = 2*radius*max(k_top, k_base), anchor = L.xz + d*k_base "
    "and far = max(reach + canvas[1]*radius*k_top, 0). Turn the quad so its "
    "along axis -- local +Z on a quad lying in its local X/Z, +Y on one lying "
    "in X/Y (a Z-up exporter's) -- runs along u and its normal (+Y, or +Z) "
    "up, which puts its own +X along up x u (X/Z) or u x up (X/Y); then scale "
    "it from its own size (its mesh's extent along each axis) and move it, "
    "0.01 of the record's unit above the ground, to span canvas[0]*far..far "
    "along u and canvas[2]*width..canvas[3]*width along its own +X, both from "
    "the anchor -- the across span follows the quad's +X, as its silhouette's "
    "u axis does, so it mirrors with the quad.",
    ("planes", "follow_source"),
    True,
)
_SHADOW_HORIZON = (
    "A horizon plane's shadow is marched from its map, {map}, at "
    "'horizon.rect' ([scale_u, scale_v, offset_u, offset_v] of the image from "
    "its {origin}). The map is data, its texels read whole (never filtered or "
    "colour-decoded; what is interpolated below is decoded values): S = "
    "'size' texels tall and S*(K + 2) wide, K = 'spans', S x S tiles side by "
    "side whose texel (i, j) -- column i, row j counted from the map's top "
    "row -- is the footprint pixel whose low corner is (a0 + i*pa, b0 + j*pb), "
    "with 'bounds' [a0, a1, b0, b1], pa = (a1 - a0)/S, pb = (b1 - b0)/S, p = "
    "min(pa, pb) and aspect = max(pa, pb)/p. With w = 256*first byte + "
    "second, H16 = w/65535*height_scale and H8 = byte/255*height_scale: tile "
    "k < K holds a span per pixel, from lo = H16(R, G) to hi = H16(B, A) "
    "above the ground (hi = 0: none), a pixel holding any span being solid; "
    "tile K the distance D to the nearest solid pixel, w(R, G)/64 in p, and "
    "that pixel's hull, top H8(B), bottom H8(A); tile K + 1, in its left "
    "half, a min/max pyramid that only speeds the march (a reader may ignore "
    "it) and, when its texel (S/2, S/2) is (83, 72, 1, 165), at texel "
    "(S/2 + i/2, j/2) a solid pixel's distance I to the nearest empty one "
    "(0 on an empty pixel), byte/4 in p: R for even i and j, G odd i, B odd "
    "j, A both odd. The "
    "shadow at a fragment F is marched in the contact's frame -- origin C, "
    "axes a = 'frame_a', b = 'frame_b' and up = +Y, each turned first +90 "
    "degrees about X when the quad lies in its local X/Y, then by the "
    "contact's world rotation (never its scale) -- from the ground point "
    "(P, g) under F: P = ((F - C).a, (F - C).b), g = ((0, 'ground', 0) - "
    "C).up, F's own height dropped. v is the unit vector from (P, g) toward "
    "the source, in that frame (against the way a directional source "
    "shines, its node's local -Y), with v.ab its (a, b) part and v.up its up "
    "part; rho = asin(min(1, 'source_size'/2/the source's distance from "
    "(P, g))), "
    "or for a directional source 'source_angle'/2 ('source_angle' is in "
    "radians); no shadow where v.up <= 0 or c = |v.ab| < 1e-9. From rise = "
    "side = 0, clear = infinity and no hit, visit in order each pixel the "
    "line P + t*v.ab/c (t >= 0) crosses inside the footprint, over [t0, t1]: "
    "h0 = t0*v.up/c, h1 = t1*v.up/c, tm = (t0 + t1)/2. A span blocks when "
    "h1 >= lo and h0 <= hi, otherwise its gap is max(lo - h1, h0 - hi); "
    "rise = max(rise, (hi - h0)*c^2/tm) over every span with hi >= h0, "
    "blocking or not. A blocking pixel sets hit and side = max(side, max(I - "
    "aspect/2, 0)*p*c/tm) (infinite without I); before any hit, any other "
    "sets clear = min(clear, e/tm), e its least gap*c^2 when solid, otherwise "
    "hypot(max(D - aspect/2, 0)*p*c, c^2*the gap from [h0, h1] to [bottom, "
    "top], 0 where they overlap). I, D, bottom and top are read bilinearly "
    "at the line's point at tm, texel centres at +0.5, each corner clamped "
    "to the tile. Stop once hit and min(rise, side) >= rho. shadow = (hit ? "
    "1 : 0) when rho = 0, else disc(hit ? -min(rise, side)/rho : clear/rho), "
    "disc(x) = (acos(x) - x*sqrt(1 - x^2))/pi over x clamped to [-1, 1]: the "
    "share of the source's disc one edge hides, an approximation that is the "
    "approved look. Cheapest: march into a texture over the quad's UVs, a "
    "texel per 3 map pixels, sampled bilinearly, again only when the source, "
    "the quad or the ground moves relative to the contact."
)


def _shadow_clauses(projected_map: str, horizon_map: str, origin: str):
    """The shadows assembly's clauses in one carrier's addresses: a plane
    type's clause where some plane is of it, the placement where some plane
    follows its source."""
    return (
        Clause(
            _SHADOW_PROJECTED.format(map=projected_map),
            ("planes", "type"),
            "projected",
        ),
        _SHADOW_PLACEMENT,
        Clause(
            _SHADOW_HORIZON.format(map=horizon_map, origin=origin),
            ("planes", "type"),
            "horizon",
        ),
    )


class SceneRecords:
    """The registry: every record, declared once, and what derives from it.

    Add a record HERE and nowhere else.  The DCC producer tables key on these
    specs (an unregistered key cannot be produced), the FBX handoff block
    describes whatever of these the carrier holds, the doc tables and the
    Unity importer gate are generated from the same rows.
    """

    # -- deliverable ---------------------------------------------------------
    #: Produced from the scene as staged for each write (the hosts'
    #: ``MaterialRecords.export_record``, through ``MaterialFamilies``) and
    #: never held by it: the export's own converted maps are what the record
    #: names, and the restore that puts the scene back clears it. The GLB is
    #: built from the file that carries it (``MeshConvert.fbx_materials``), so
    #: the two deliverables state every material alike; a file without one
    #: (a File > Export) is read into the same families from its own
    #: materials.
    MATERIALS = RecordSpec(
        "materials",
        Scope.DELIVERABLE,
        1,
        owner="Export",
        description=(
            "every material as glTF 2.0 states it: its factors, the embedded "
            "map behind each, alpha mode, double-sidedness and unlit"
        ),
        kind=Kind.DERIVED,
        transient=True,
        consumers=("unity", "glb"),
        shape=(
            "pythontk.core_utils.engines.scene_export.material_record",
            "MaterialRecord",
        ),
        export_scoped=True,
        assembly=(
            "Every material's inputs are exactly as its 'materials' entry "
            "states them, whatever an importer makes of the file's own "
            "material (how each is lit is the lighting parts'): glTF 2.0 "
            "metallic-roughness, each factor linear and multiplied by its map, "
            "every map on UV set 0, colour and emission maps sRGB, the others "
            "linear with metal, roughness and occlusion read from R, opacity "
            "the base colour's alpha (MASK cuts at 'alphaCutoff', BLEND "
            "blends), 'unlit' shows the base colour alone. An entry's 'name' "
            "is the material's in the file, namespace included; where an "
            "import drops the namespace, the name after the last ':' joins."
        ),
    )
    SHOTS = RecordSpec(
        "animation",
        Scope.DELIVERABLE,
        1,
        owner="Shots",
        renamed_from=("shot_metadata",),
        description=(
            "shot definitions -- per clip its frame range ('start'/'end', the "
            "take the clip is cut from) and any description; the scene fps and "
            "the declared clip mode; the clip name is the join key to the "
            "imported animation clip"
        ),
        # DERIVED, so every write refreshes it, hand-offs included: the shot
        # store is its one authority and it reads the live scene (which shots
        # are stale). Authored, a WebXR or Unity push shipped the last Scene
        # Exporter run's shots or none (measured 2026-10-21, both DCCs).
        kind=Kind.DERIVED,
        consumers=("unity", "glb", "verifier"),
        shape=("pythontk.core_utils.engines.shots.animation_record", "AnimationRecord"),
        # The record's shots inherited whole, and the file's clips by glTF
        # animation index (MeshConvert.apply_glb_animations). Version 2
        # (2026-10-21): the shots ride whatever clips the file holds -- v1
        # listed only the clips, so a shot with none (a 'full' file) was lost.
        glb=GlbBlock(
            version=2,
            shape=(
                "pythontk.core_utils.engines.shots.animation_record",
                "AnimationBlock",
            ),
            assembly=(
                "The clips, by glTF animation index, open on 'default_clip'. "
                "'shots' are the scene's shots whatever clips the file holds: "
                "a shot plays the clip its 'clip' names, else (a 'full' file, "
                "or a take that was not cut) its window 'start'..'end' of the "
                "whole-timeline clip, the one no shot names. A clip's "
                "'zero_frame' is the authoring frame its t=0 sits on, so a "
                "frame f is (f - zero_frame) / fps seconds into it. A STEP "
                "'scale' channel driving a "
                "node to zero is its keyed visibility, not authored scale: glTF "
                "has no visibility channel. KHR_animation_pointer channels on a "
                "material are its authored opacity fade (baseColorFactor alpha) "
                "and highlight (emissiveFactor, added to the material's own); a "
                "runtime without the extension plays a fade by binding each "
                "channel to that material's alpha."
            ),
        ),
        assembly=(
            "The animation is the file's takes, each shot joined to its take "
            "by 'clip', the take's name. 'clip_mode' says how the takes "
            "were cut: 'shots', one take per shot; 'full', one whole-timeline "
            "take in which a shot is the window 'start'..'end'; 'both' (or "
            "absent), both, the whole-timeline take being the longest one no "
            "shot names. 'start' and 'end' are frames of the authoring "
            "timeline at 'fps'; 'description' says what a shot shows."
        ),
    )
    #: Retired 2026-10-20 (removed in 0.18.0): the take list a file written
    #: before 0.11.0 carried. No reader reads it; a scene still holding one has
    #: it cleared (:attr:`RETIRED`).
    FBX_TAKES = Deprecation.class_attribute(
        RecordSpec(
            "fbx_takes",
            Scope.DELIVERABLE,
            1,
            owner="Shots",
            description="the take list a file written before 0.11.0 carried",
            envelope=False,
        ),
        "SceneRecords.SHOTS -- each shot's range rides its clip",
        remove_in="0.18.0",
        since="2026-10-24",
    )
    AUDIO = RecordSpec(
        "audio",
        Scope.DELIVERABLE,
        2,
        owner="Audio Clips",
        renamed_from=("audio_manifest",),
        description=(
            "audio events with the frames they fire on, scoped to their clip, "
            "and 'fps', the rate those frames count at"
        ),
        # Read off the keyed tracks / sound strips an artist moves between one
        # push and the next, so every write refreshes it, a hand-off included
        # (a Unity-bridge FBX shipped the last export's events otherwise).
        kind=Kind.DERIVED,
        after=("animation",),
        consumers=("unity",),
        shape=("pythontk.audio_utils.audio_record", "AudioRecord"),
        assembly=(
            "Each event plays its audio track 'frame' frames into its 'clip' "
            "(at the record's 'fps'); an event whose 'clip' is empty plays in "
            "every clip, its frame counted from the first frame the file's "
            "animation holds. The track's file rides the file as media of its "
            "own, joined to the event's 'name' by file stem without regard "
            "to case."
        ),
    )
    #: The payload's model is ``LightmapRecord``: one ``unit_scale`` for every
    #: length, the probe's box as ``{min, max}``, no direction on a point light.
    LIGHTMAPS = RecordSpec(
        "lightmaps",
        Scope.DELIVERABLE,
        2,
        owner="Lightmap Baker",
        renamed_from=("lightmap_metadata",),
        description=(
            "per-object baked-lightmap records: map file name, uvIndex, "
            "scaleOffset, and the object's scene hierarchy (which "
            "tells apart objects that share a name); and the bake's reflection "
            "probe, when it captured one: its HDR's file name, capture point and "
            "projection box; and the lights the maps were baked from, for the "
            "highlights a lightmap cannot hold -- every length times "
            "unit_scale is metres"
        ),
        consumers=("unity", "glb"),
        shape=(
            "pythontk.core_utils.engines.scene_export.lightmap_record",
            "LightmapRecord",
        ),
        # Per material its intensity alone -- its map is its bound texture, its
        # UV set that texture's texCoord -- and the embedded probe with its
        # highlight copies (MeshConvert.apply_glb_lightmaps).
        glb=GlbBlock(
            version=2,
            shape=(
                "pythontk.core_utils.engines.scene_export.lightmap_record",
                "LightmapsBlock",
            ),
            assembly=(
                "Each material in 'materials' carries a baked lightmap in the "
                "texture slot 'carrier' names (occlusion: occlusionTexture) -- "
                "its RGB, read through that texture's texCoord and "
                "KHR_texture_transform, not glTF's red-only occlusion -- 8-bit "
                "sRGB decoded to linear, times its 'intensity': the "
                "surface's diffuse light, direct and indirect, and not ambient "
                "occlusion."
            ),
            clauses=(_LIGHTMAP_PROBE,),
        ),
        export_scoped=True,
        assembly=(
            "Each 'objects' entry lights one object with a baked lightmap: "
            "the image 'map' names, wired to no material, read through UV set "
            "'uvIndex' (0-based: "
            "1 is the second UV set) at uv * [scaleX, scaleY] + [offsetX, "
            "offsetY] from 'scaleOffset' (in the file's own V-up UVs), "
            "non-directional and applied as "
            "stored, linear. The object is the node 'name' names (its "
            "namespace aside), 'hierarchy' -- its path from the scene root, "
            "namespaces kept, its own name last -- "
            "telling apart objects that share a name; a tie nothing settles "
            "is left unlit rather than guessed. The map is the object's whole "
            "diffuse light and its material stays as authored "
            "(rendering.lightmappedMaterials). Every length in the record "
            "times 'unit_scale' is metres."
        ),
        clauses=(
            _LIGHTMAP_PROBE,
            Clause(
                "'lights' are the "
                "lights the maps were baked from, for their HIGHLIGHTS alone -- "
                "the diffuse light of each is already in the maps, so a reader "
                "that lights them must light their specular only, or the scene "
                "is lit twice. Each is 'type' point, spot or rect at 'position', a "
                "spot or rect facing 'direction' (a rect's width along 'right'), "
                "its 'color' linear Rec.709, "
                "'intensity' its radiant intensity for a point or spot (the "
                "irradiance it gives one unit away) and its radiance for a rect, "
                "a spot's "
                "half 'angle' (radians) and soft 'penumbra' share, a rect's "
                "'width' and 'height', a source's 'radius'.",
                ("lights",),
            ),
            Clause(
                "'probe.highlights' "
                "are the bake's lights already drawn for groups of "
                "receivers, so a reader that uses them lights none of 'lights' "
                "(the highlights would count twice): each names a 'map', a "
                "Radiance layer in the probe's layout and units to add to the "
                "probe (box-filtered down to the layer's size) and prefilter as "
                "the probe is, and the 'objects' ('name' and 'hierarchy', as above) "
                "that take that sum in the probe's place (diffuse irradiance "
                "and reflections alike), box-projected from the same 'position' "
                "and 'box'.",
                ("probe", "highlights"),
            ),
        ),
    )
    SHADOWS = RecordSpec(
        "shadows",
        Scope.DELIVERABLE,
        2,
        owner="Shadow Rig",
        renamed_from=("shadow_metadata",),
        description=(
            "projected-shadow planes: per plane its node name, type, "
            "silhouette texture and intensity, the source and contact it "
            "follows and the projection model's constants, and, when packed "
            "or baked, its atlas rect and horizon map -- every length times "
            "unit_scale is metres"
        ),
        consumers=("unity", "glb"),
        shape=("pythontk.geo_utils.shadow_record", "ShadowRecord"),
        # Each plane, its source and contact as node indices, its maps as
        # texture indices (MeshConvert.apply_glb_shadows).
        glb=GlbBlock(
            version=2,
            shape=("pythontk.geo_utils.shadow_record", "ShadowBlock"),
            node_fields=(
                ("planes", "node"),
                ("planes", "source_node"),
                ("planes", "contact_node"),
            ),
            assembly=(
                "A plane, its source and its contact are the glTF nodes "
                "'node', 'source_node' and 'contact_node' (null where the file "
                "holds no single node of that name). " + _SHADOW_PLANE
            ),
            clauses=_shadow_clauses(
                "texture 'texture_index' (its atlas once packed)",
                "texture 'horizon.texture_index'",
                "top-left",
            ),
        ),
        export_scoped=True,
        assembly=(
            "A plane, its source and its contact are the nodes named 'name', "
            "'source' and 'contact'. " + _SHADOW_PLANE
        ),
        clauses=_shadow_clauses(
            "the image 'texture' names ('atlas.texture' once packed)",
            "the image 'horizon.texture' names",
            "bottom-left",
        ),
    )
    ARTICULATION = RecordSpec(
        "articulation",
        Scope.DELIVERABLE,
        1,
        owner="Articulated Rig",
        description=(
            "articulated rigs: per rig its joints (node name, parent, rest "
            "translate and orient, rotate order, channels with limits and "
            "grab weights) and the parts a hand grabs, each with the joint "
            "it rides"
        ),
        consumers=("unity", "glb"),
        shape=("pythontk.geo_utils.articulation.record", "ArticulationRecord"),
        # Every joint and grabbed part bound to its glTF node
        # (MeshConvert.apply_glb_articulation).
        glb=GlbBlock(
            shape=("pythontk.geo_utils.articulation.record", "ArticulationBlock"),
            node_fields=(("rigs", "joints", "node"), ("rigs", "grab", "node")),
            assembly=(
                "Per rig its joints -- the glTF node index, the parent joint, "
                "the rest translate and rotation in the parent's space, the "
                "rotate order and the channels (rx/ry/rz degrees, tx/ty/tz a "
                "slide along the rest axes) with their limits -- and the parts "
                "a hand grabs, each with its node index and the joint it "
                "rides. A joint poses as rotation = rest (x) Euler(channels) "
                "and translation = rest + rest.(tx, ty, tz), every channel "
                "within its limits: the built state is that pose at the clips' "
                "keys, the rest pose without them. A hand that grabs a part "
                "moves the channels of the joints from it up to the rig's "
                "root, each in proportion to its 'weight' and within its "
                "limits; a runtime measures its unit factor off the rest "
                "translations."
            ),
        ),
        assembly=(
            "Each rig poses its joints as rotation = rest (x) Euler(channels) "
            "and translation = rest + rest.(tx, ty, tz), every channel within "
            "its limits (degrees), and a grabbed part rides its joint; a "
            "grab moves the channels of the joints from the part up to the "
            "rig's root, each in proportion to its 'weight' and within its "
            "limits. Joints are found by "
            "node name; the rest translations are in the file's own axes, "
            "their unit the ratio of the imported rest translations to them. "
            "The imported pose is the rest pose, and the takes play the "
            "joints' own keys."
        ),
    )
    EMISSIVE_GROUPS = RecordSpec(
        "emissive_groups",
        Scope.DELIVERABLE,
        2,
        owner="Emissive Groups",
        description="named emissive material groups and their weights",
        # The payload is a RegionMaskManifest document (its VERSION is this).
        consumers=("unity",),
        shape=(
            "pythontk.core_utils.engines.textures.region_masks",
            "EmissiveGroupsRecord",
        ),
        assembly=(
            "Each group gates the emission of the region its 'slot' selects "
            "(0-3: R, G, B, A of the mask 'encoding' names -- the vertex "
            "colours 'color_set', or the texture 'mask' read through UV set "
            "'uv_channel'), its weight starting at 'default'. A group naming "
            "'attr' has that weight animated: a keyed property of that name "
            "on the model's root, or a child node named 'attr' whose scale.x "
            "carries the curve."
        ),
    )
    VISIBILITY = RecordSpec(
        "render_effects",
        Scope.DELIVERABLE,
        1,
        owner="Render Effects",
        renamed_from=("visibility_tracks",),
        shape=(
            "pythontk.file_utils.mesh_convert.render_effects_record",
            "RenderEffectsRecord",
        ),
        description=(
            "each node's keyed render effects: 'visibility' as stepped on/off "
            "frames, the authored 'opacity' ramp, and the 'highlight' ramp "
            "with its colour stops ('highlight_color', 'highlight_color_dim'); "
            "with 'fps' and each take's first/last authored frame "
            "('clip_span'); on a shot scene keying none, the exporter's "
            "measured whole-timeline clip origin alone"
        ),
        kind=Kind.DERIVED,
        after=("animation",),
        consumers=("glb", "verifier"),
        assembly=(
            "Render effects over time, per 'tracks' entry and the subtree "
            "under its 'node', each a list of [frame, value] keys: "
            "'visibility' shows (1) or hides (0) it, stepped; 'opacity' is "
            "its materials' alpha, hidden at 0; 'highlight' adds an emission "
            "to its materials' own, blended from 'highlight_color_dim' at 0 "
            "to 'highlight_color' at 1. The same curves ride the file itself: "
            "each affected object carries 'opacity' / 'highlight' (and "
            "'highlightColor' / 'highlightColorDim') properties, keyed "
            "through a child '<object>__opacity' / '<object>__highlight' "
            "node whose scale.x holds the curve -- transport, not content."
        ),
    )
    #: Version 2 (2026-10-14) added ``assembly`` and published ``rendering``
    #: in the carrier's own addresses. Version 3 (2026-10-20) is the shape
    #: every block keeps: the fundamentals at the top (``instructions``,
    #: ``assembly`` -- each part named as on every carrier, its ``data`` the
    #: addresses it reads -- and ``rendering``, its values alone, their
    #: meaning in the lighting rows' specs) and the rest under ``extras``
    #: (``source``), where a later addition needs no new version. Which block
    #: describes a file is the node it rides, never its version: a referenced
    #: module's block -- a version-1 one says the maps are not embedded and
    #: that ``intensity`` is a multiplier to apply -- describes that module
    #: alone, whichever is newer.
    HANDOFF = RecordSpec(
        "handoff",
        Scope.DELIVERABLE,
        3,
        owner="Export",
        description=(
            "the standalone-reader contract: the approved scene part by part "
            "and the lighting it was approved under"
        ),
        kind=Kind.DERIVED,
        # Unity's importer reads its lighting recipe; the GLB build reads the
        # FBX's to state the GLB's, and the glTF runtimes the GLB's.
        consumers=("unity", "glb"),
        shape=(
            "pythontk.core_utils.engines.scene_export.handoff_record",
            "HandoffBlock",
        ),
    )
    #: Keys a scene may still hold for a record retired whole, which nothing
    #: reads: cleared from a carrier (:meth:`SceneStoreBase.migrate`) and
    #: dropped from a written deliverable. A renamed record names its old keys
    #: itself (:attr:`RecordSpec.renamed_from`).
    RETIRED: Tuple[str, ...] = ("fbx_takes",)
    #: Root-``extras`` keys a GLB written before 0.17.0 carries for a block
    #: since renamed (the ``*_web`` blocks, now under their record's key) or
    #: retired whole (the ``scene_sidecar`` envelope), which nothing reads: a
    #: reader names each one it finds stale, so the file is re-exported (the
    #: viewer's ``RETIRED_BLOCK_KEYS``, generated by ``sync_scene_records.py``).
    RETIRED_BLOCKS: Tuple[str, ...] = (
        "animation_web",
        "lightmap_web",
        "shadow_web",
        "articulation_web",
        "scene_sidecar",
    )

    # -- private -------------------------------------------------------------
    SHOT_STORE = RecordSpec(
        "shot_store",
        Scope.PRIVATE,
        1,
        owner="Shots",
        description=(
            "the shot store's full app state: its shots (id, name, start, end, "
            "objects, metadata, locked, description), the hidden and pinned "
            "objects, markers, locked gaps, the manifest's edits and the "
            "panel's settings"
        ),
        envelope=False,
        merge=Merge.CODEC,
        portable=True,
        section="shots",
    )
    KEY_STASH = RecordSpec(
        "key_stash",
        Scope.PRIVATE,
        1,
        owner="Key Stash",
        description=(
            "the clip manifest of parked keys: per clip its id, label, "
            "objects, parked curves, source shot and metadata"
        ),
        envelope=False,
        # Declared after the shot store: a parked clip follows its source
        # shot through the renumbering the shot store's merge publishes.
        merge=Merge.CODEC,
    )
    SMART_BAKE_SESSIONS = RecordSpec(
        "smart_bake_sessions",
        Scope.PRIVATE,
        2,
        owner="SmartBake",
        description=(
            "LIFO stack of bake-session restore manifests, oldest first, each "
            "under its 'id': what the bake replaced, to put it back"
        ),
        envelope=False,
        merge=Merge.UNION,
        merge_key="id",
    )
    HIERARCHY_BASELINE = RecordSpec(
        "hierarchy_baseline",
        Scope.PRIVATE,
        1,
        owner="Hierarchy check",
        description=(
            "the export hierarchy baseline (a HierarchyBaseline record): the "
            "exported node paths, their hash and count, and the scene file "
            "that recorded them"
        ),
        envelope=False,
        merge=Merge.OWN,
        paths=("scene",),
        stamps=("scene",),
    )
    EMISSIVE_REGISTRY = RecordSpec(
        "emissive_registry",
        Scope.PRIVATE,
        1,
        owner="Emissive Groups",
        description=(
            "the group registry: per group its slot, default weight and keyed "
            "attr, the membership encoding and the retired slots"
        ),
        renamed_from=("emissive_groups",),
        envelope=False,
        merge=Merge.CODEC,
        respell=False,
        portable=True,
    )
    RENDER_EFFECTS_BINDINGS = RecordSpec(
        "render_effects_bindings",
        Scope.PRIVATE,
        1,
        owner="Render Effects",
        description=(
            "legacy, Maya only: the bindings the retired viewport material "
            "mode left, per material channel its object and the authored "
            "values -- read to restore them, then cleared"
        ),
        envelope=False,
        merge=Merge.UNION,
    )
    #: Each texture combine the scene holds, by name: the meshes it moved
    #: (their original UVs and the material each face wore), the source maps
    #: it read, its settings and table overrides -- what restores the
    #: originals exactly, and re-combines in place when a source map is
    #: revised.  Scene nodes and absolute source paths, so it never crosses.
    TEXTURE_COMBINE = RecordSpec(
        "texture_combine",
        Scope.PRIVATE,
        1,
        owner="Texture Combine",
        description=(
            "texture combines by name: per mesh its original UVs and face "
            "materials, the source maps read, the settings and overrides"
        ),
        # A bare ``{name: entry}`` mapping (each entry carries its own
        # ``schema``), so a crossing unites combines by name.
        envelope=False,
        merge=Merge.UNION,
    )
    #: The light rigs applied in this scene, by rig group name: which saved rig
    #: each group holds, its fingerprint when it went in (what "edited since"
    #: compares), the plugs it overrides and the sections it carried, and the
    #: rig the last swap replaced (what Swap Back returns to). Scene state, kept
    #: off the group: a node attribute rides every FBX as a user property.
    LIGHT_RIGS = RecordSpec(
        "light_rigs",
        Scope.PRIVATE,
        1,
        owner="Light Rigs",
        description=(
            "light rigs by group: the rig applied, its fingerprint, the plugs "
            "it overrides and the rig a swap replaced"
        ),
        envelope=False,
        merge=Merge.UNION,
    )
    AUDIO_FILE_MAP = RecordSpec(
        "audio_file_map",
        Scope.PRIVATE,
        1,
        owner="Audio Clips",
        description="track id to audio file path",
        envelope=False,
        merge=Merge.UNION,
        respell=False,
        paths=True,
    )
    #: Where each baked lightmap was written, by lower-case file name -- the
    #: build-time hint a GLB build and the texture tools locate a map by.  It
    #: rode every per-object ``lightmapInfo`` marker as ``dir`` until
    #: 2026-09-23, and a marker is a node attribute, so every FBX carried the
    #: authoring folder: build-setup data on the deliverable.  Portable, so a
    #: scene pulled across the bridge still finds its maps; file names, never
    #: scene names, so a crossing does not respell it.
    LIGHTMAP_DIRS = RecordSpec(
        "lightmap_dirs",
        Scope.PRIVATE,
        1,
        owner="Lightmap Baker",
        description="lightmap file name to the folder it was written to",
        envelope=False,
        merge=Merge.UNION,
        respell=False,
        portable=True,
        paths=True,
    )
    #: Which scene file wrote each baked lightmap, by lower-case file name --
    #: what lets a re-bake delete the maps it superseded
    #: (``FileDependencies.remove_superseded``) and never one another scene
    #: file still reads: a Save As copy carries its source's markers and
    #: folder record, so those alone cannot tell the two scenes apart.  A
    #: path record like ``lightmap_dirs``, so a copy saved into another
    #: project still names its source.  ``""`` is a map written while the
    #: scene was unsaved, the scene's own while it still is; its first save
    #: stamps it with the file written (:meth:`SceneRecords.stamp_unsaved`),
    #: so a copy made later names its source like any other entry.
    LIGHTMAP_WRITERS = RecordSpec(
        "lightmap_writers",
        Scope.PRIVATE,
        1,
        owner="Lightmap Baker",
        description="lightmap file name to the scene file that wrote it",
        envelope=False,
        merge=Merge.UNION,
        respell=False,
        portable=True,
        paths=True,
        stamps=True,
    )
    #: The reflection probe the lightmap bake captured -- the room it lit, as
    #: an HDR from one point: the file's name, that point, and the box its
    #: reflections project onto, in scene units with ``unit_scale`` (metres
    #: per unit) beside them, read at the same moment.  Published inside
    #: :attr:`LIGHTMAPS` as its
    #: ``probe``; the file's folder and writer ride :attr:`LIGHTMAP_DIRS` and
    #: :attr:`LIGHTMAP_WRITERS` like any map's.  It describes its own scene,
    #: so another scene's copy is dropped.
    LIGHTMAP_PROBE = RecordSpec(
        "lightmap_probe",
        Scope.PRIVATE,
        1,
        owner="Lightmap Baker",
        description=(
            "the reflection probe the lightmap bake captured: file name, "
            "capture point, projection box and unit_scale"
        ),
        merge=Merge.OWN,
        respell=False,
    )
    #: What each lightmap was baked AGAINST, per object: its transform,
    #: bounds, geometry and lightmap-UV fingerprints, material group and the
    #: atlas cell it was written to, and the lighting each bake saw -- the
    #: baseline a re-bake of what changed diffs the scene against
    #: (``BakeBaseline``).  Never on the marker, which rides every FBX and is
    #: read by the GLB applier and WebXR, nor in :attr:`LIGHTMAPS` (the Unity
    #: contract): build-time state keyed by this scene's own node names, so
    #: another scene's copy is dropped, and nothing respells it.
    LIGHTMAP_BAKE_STATE = RecordSpec(
        "lightmap_bake_state",
        Scope.PRIVATE,
        1,
        owner="Lightmap Baker",
        description=(
            "per object the state its lightmap was baked against (transform, "
            "bounds, geometry and UV fingerprints, material, atlas cell) and "
            "the lighting each bake saw (a BakeBaseline record)"
        ),
        merge=Merge.OWN,
        respell=False,
    )

    #: The domain codecs: record key -> (module, class) whose classmethod
    #: ``merge_record(own, other, ctx)`` is a :attr:`Merge.CODEC` record's
    #: merge.  Resolved lazily, like a DCC's producer table -- the codec lives
    #: with the model it merges, and this module stays the lighter import.
    # Keyed by the declarations, never a spelling: a renamed record keeps
    # its codec.
    CODECS: Dict[str, Tuple[str, str]] = {
        SHOT_STORE.key: (
            "pythontk.core_utils.engines.shots.shot_transfer",
            "ShotTransfer",
        ),
        KEY_STASH.key: (
            "pythontk.core_utils.engines.key_stash.key_stash_model",
            "KeyStash",
        ),
        EMISSIVE_REGISTRY.key: (
            "pythontk.core_utils.engines.textures.region_masks",
            "RegionGroupRegistry",
        ),
    }

    @staticmethod
    def resolve_class(module: str, name: str) -> Any:
        """The class a ``(module, name)`` row names -- :attr:`CODECS`',
        :attr:`SceneStoreBase.OWNERS`' -- imported on first use, so declaring
        a record never imports its engine."""
        import importlib

        return getattr(importlib.import_module(module), name)

    @classmethod
    def codec(cls, spec: RecordSpec) -> Optional[Any]:
        """The codec class of a :attr:`Merge.CODEC` record, else ``None``."""
        row = cls.CODECS.get(spec.key) if spec.merge is Merge.CODEC else None
        if row is None:
            return None
        return cls.resolve_class(*row)

    @classmethod
    def shape(cls, item: Union[RecordSpec, str]) -> Optional[Any]:
        """The :class:`~pythontk.SchemaSpec` declaring *item*'s payload
        (:attr:`RecordSpec.shape`), imported on first use; ``None`` while the
        payload is undeclared."""
        spec = cls.resolve(item)
        return cls.resolve_class(*spec.shape) if spec.shape else None

    @classmethod
    def glb_shape(cls, item: Union[RecordSpec, str]) -> Optional[Any]:
        """The :class:`~pythontk.SchemaSpec` declaring *item*'s GLB block
        (:attr:`GlbBlock.shape`), or ``None``."""
        block = cls.resolve(item).glb
        return cls.resolve_class(*block.shape) if block and block.shape else None

    @classmethod
    def put_glb_block(
        cls,
        gltf: Dict[str, Any],
        item: Union[RecordSpec, str],
        block: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Write *block* as *item*'s block in the glTF document *gltf* -- its
        root ``extras``, under the record's key -- the block's version
        stamped (a writer's own wins) and pruned to its declared shape, as
        :meth:`RecordSpec.make` does a record. The one writer of every block,
        beside :meth:`glb_block`, its one reader. Returns the block written."""
        spec = cls.resolve(item)
        body: Dict[str, Any] = {"version": spec.glb.version}
        body.update(block)
        shape = cls.glb_shape(spec)
        if shape is not None:
            body = _declared(f"extras.{spec.key}", shape, body)
        extras = gltf.get("extras")
        if not isinstance(extras, dict):
            extras = gltf["extras"] = {}
        extras[spec.key] = body
        return body

    @classmethod
    def glb_block(
        cls, gltf: Any, item: Union[RecordSpec, str]
    ) -> Optional[Dict[str, Any]]:
        """*item*'s block in the glTF document *gltf* -- its root ``extras``,
        under the record's key -- in the current schema
        (:attr:`GlbBlock.version`); ``None`` otherwise. The block itself, so
        a pass that edits it edits the document."""
        spec = cls.resolve(item)
        extras = gltf.get("extras") if isinstance(gltf, Mapping) else None
        if not spec.glb or not isinstance(extras, Mapping):
            return None
        block = extras.get(spec.key)
        current = isinstance(block, dict) and block.get("version") == spec.glb.version
        return block if current else None

    @classmethod
    def glb_records(cls) -> List[RecordSpec]:
        """The records a GLB carries a block for (:attr:`RecordSpec.glb`), in
        declaration order."""
        return [s for s in cls.all() if s.glb]

    @classmethod
    def problems(cls, item: Union[RecordSpec, str], payload: Any) -> List[str]:
        """What *payload* breaks of *item*'s declaration, one line each: a
        version other than the declared one (:meth:`RecordSpec.is_current`),
        or what its declared shape refuses (``SchemaSpec.validate``); empty
        when a reader reads it."""
        spec = cls.resolve(item)
        return cls._problems(spec.key, spec.version, cls.shape(spec), payload)

    @classmethod
    def block_problems(cls, item: Union[RecordSpec, str], block: Any) -> List[str]:
        """What *block* -- *item*'s GLB block -- breaks of its declaration
        (:attr:`RecordSpec.glb`), as :meth:`problems` says it of a record."""
        spec = cls.resolve(item)
        if not spec.glb:
            return [f"extras.{spec.key}: {spec.key!r} declares no GLB block"]
        shape = cls.glb_shape(spec)
        return cls._problems(f"extras.{spec.key}", spec.glb.version, shape, block)

    @staticmethod
    def _problems(where: str, version: int, shape: Any, payload: Any) -> List[str]:
        """:meth:`problems`' lines for *payload* at *where*."""
        if not isinstance(payload, Mapping):
            return [f"{where}: not an object"]
        found = payload.get("version")
        if found != version:
            return [f"{where}: v{found} is not the v{version} a reader reads"]
        if shape is None:
            return []
        return [_located(where, error) for error in shape.validate(payload).errors]

    # Retired 2026-10-20 (removed in 0.18.0): a record's block is named for
    # the file it rides, not for one of its readers.
    @classmethod
    @Deprecation.symbol(
        "SceneRecords.glb_shape", remove_in="0.18.0", since="2026-10-24"
    )
    def web_shape(cls, item: Union[RecordSpec, str]) -> Optional[Any]:
        """:meth:`glb_shape`."""
        return cls.glb_shape(item)

    @classmethod
    @Deprecation.symbol(
        "SceneRecords.glb_records", remove_in="0.18.0", since="2026-10-24"
    )
    def web_projected(cls) -> List[RecordSpec]:
        """:meth:`glb_records`."""
        return cls.glb_records()

    @classmethod
    def portable(cls) -> List[RecordSpec]:
        """The records that cross a DCC hand-off, in declaration order."""
        return [s for s in cls.all() if s.portable]

    # The handoff's names here retired 2026-10-20 (removed in 0.18.0): the
    # block is declared once, for every carrier, by ``HandoffRecord``, which builds
    # on this registry -- so each alias reads it at call time.
    @staticmethod
    @Deprecation.symbol(
        "HandoffRecord.rendering_policy", remove_in="0.18.0", since="2026-10-24"
    )
    def rendering_policy(
        overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """What a deliverable claims about how it should be lit
        (``HandoffRecord.rendering_policy``)."""
        from pythontk.core_utils.engines.scene_export.handoff_record import (
            HandoffRecord,
        )

        return HandoffRecord.rendering_policy(overrides)

    @ClassProperty
    @Deprecation.symbol(
        "HandoffRecord.instructions('fbx')", remove_in="0.18.0", since="2026-10-24"
    )
    def HANDOFF_INSTRUCTIONS(cls) -> str:
        """The FBX block's reading contract (``HandoffRecord.instructions``)."""
        from pythontk.core_utils.engines.scene_export.handoff_record import (
            HandoffRecord,
        )

        return HandoffRecord.instructions("fbx")

    @ClassProperty
    @Deprecation.symbol(
        "HandoffRecord.CARRIERS['fbx'].own", remove_in="0.18.0", since="2026-10-24"
    )
    def HANDOFF_ASSEMBLY(cls) -> Tuple[Dict[str, Any], ...]:
        """The FBX block's rows for the file's own content (``HandoffRecord.CARRIERS``)."""
        from pythontk.core_utils.engines.scene_export.handoff_record import (
            HandoffRecord,
        )

        return HandoffRecord.CARRIERS["fbx"].own

    # ------------------------------------------------------------- catalogue
    @classmethod
    def all(cls) -> List[RecordSpec]:
        """Every declared record, in declaration order."""
        return [v for v in vars(cls).values() if isinstance(v, RecordSpec)]

    @classmethod
    def with_paths(cls) -> List[RecordSpec]:
        """The records whose values are project-relative paths
        (:attr:`RecordSpec.paths`), in declaration order."""
        return [s for s in cls.all() if s.paths]

    @classmethod
    def with_stamps(cls) -> List[RecordSpec]:
        """The records holding writer stamps (:attr:`RecordSpec.stamps`), in
        declaration order."""
        return [s for s in cls.all() if s.stamps]

    @classmethod
    def transients(cls) -> List[RecordSpec]:
        """The records that describe one write (:attr:`RecordSpec.transient`),
        in declaration order -- what a scene clears when it holds one outside
        a write (an import adopted it; a save)."""
        return [s for s in cls.all() if s.transient]

    @classmethod
    def stamp_unsaved(cls, store, stamp: str) -> int:
        """Give every writer stamp still ``""`` in *store* the file *stamp*
        names -- what a DCC's save hook calls at a scene's FIRST save.

        ``""`` stamps an entry made while the scene was unsaved, and is the
        scene's own only while it still is (``FileDependencies.written_here``):
        once saved, a Save As copy would carry the same ``""`` and nothing
        could tell the two files apart.  Left as it was, everything made
        before the first save was nobody's from then on -- a lightmap
        re-bake never retired those maps, and the hierarchy baseline was set
        aside as another scene's.  Stamped as the file is first written, it
        reads as that file's own, and a later copy names its source.

        Which write is the first save is the hook's to say: a copy of an
        unsaved scene (an export, an autosave) is not one, and keeps ``""``.

        Parameters:
            store: The scene store (a :class:`SceneStoreBase`).
            stamp: The first-saved file, spelled as a writer stamp
                (``SceneStoreBase.writer_stamp_of``).  ``""`` does nothing.

        Returns:
            int: How many entries were stamped.
        """
        if not stamp:
            return 0

        def stamped(spec: RecordSpec, payload: Mapping) -> Dict[str, Any]:
            keys = spec.stamp_keys
            return {
                key: (stamp if value == "" and (keys is None or key in keys) else value)
                for key, value in payload.items()
            }

        return cls._remap_records(store, cls.with_stamps(), stamped)

    @classmethod
    def adopt_gone_stamps(cls, store, stamp: str, base: Optional[str]) -> int:
        """Give every writer stamp in *store* that names a scene file now gone
        (``FileDependencies.writer_gone``: renamed or moved) *stamp* -- what a
        DCC's save hook calls at every save of the open scene.

        Gone, the old name reads as the scene's own; but only while nothing is
        there. A file that came back to that path -- a Save As back to the old
        name, a restored backup -- then owned the same maps, and its re-bake
        wrote over what this scene reads. Written down at the save, the
        ownership no longer depends on what is missing.

        Parameters:
            store: The scene store (a :class:`SceneStoreBase`).
            stamp: The heir, spelled as a writer stamp from *base*.
            base: The project the stamps are spelled from.

        Returns:
            int: How many entries were stamped.
        """
        if not stamp:
            return 0
        from pythontk.file_utils.file_dependencies import FileDependencies

        def adopted(spec: RecordSpec, payload: Mapping) -> Dict[str, Any]:
            keys = spec.stamp_keys
            return {
                key: (
                    stamp
                    if isinstance(value, str)
                    and (keys is None or key in keys)
                    and value != stamp
                    and FileDependencies.writer_gone(value, base)
                    else value
                )
                for key, value in payload.items()
            }

        return cls._remap_records(store, cls.with_stamps(), adopted)

    @staticmethod
    def _remap_records(
        store,
        specs: List[RecordSpec],
        remap: Callable[[RecordSpec, Mapping], Mapping],
    ) -> int:
        """Rewrite each of *specs*' mapping payloads in *store* through
        *remap* (``(spec, payload) -> new payload``), saving only a record
        that changed; returns how many entries changed."""
        changed = 0
        for spec in specs:
            payload = spec.load(store)
            if not isinstance(payload, Mapping) or not payload:
                continue
            new = remap(spec, payload)
            diff = sum(1 for key in payload if new.get(key) != payload.get(key))
            if diff:
                spec.save(store, new)
                changed += diff
        return changed

    @staticmethod
    def map_paths(
        payload: Any,
        spell: Callable[[str], str],
        keys: Optional[Tuple[str, ...]] = None,
    ) -> Any:
        """*payload* (a :attr:`RecordSpec.paths` mapping) with every string
        value put through *spell* -- only those under *keys* when given
        (:attr:`RecordSpec.path_keys`); anything else -- a non-mapping, a
        non-string value -- as it is.  A new mapping, never a mutation."""
        if not isinstance(payload, Mapping):
            return payload
        return {
            key: (
                spell(value)
                if isinstance(value, str) and value and (keys is None or key in keys)
                else value
            )
            for key, value in payload.items()
        }

    @classmethod
    def rebase_paths(
        cls, store, old_base: Optional[str], new_base: Optional[str]
    ) -> int:
        """Re-spell every :attr:`RecordSpec.paths` record in *store* from
        *old_base* to *new_base* -- the scene's own project before and after a
        save moved it (the DCC's save hook) -- and return how many entries
        changed.  The same files, spelled from where the scene lives now
        (``FileUtils.rebase_portable_path``).

        With the two bases equal it normalizes: an absolute entry (written
        before the rule, or one that arrived across a hand-off) is re-spelled
        relative wherever a relative spelling reaches.  A record that does not
        change is not written.

        Parameters:
            store: The scene store (a :class:`SceneStoreBase`).
            old_base: The project the records are spelled from now; ``None``
                while the scene was unsaved (its entries are absolute).
            new_base: The project they are spelled from after the save.

        Returns:
            int: How many entries changed.
        """
        from pythontk.file_utils._file_utils import FileUtils

        return cls._remap_records(
            store,
            cls.with_paths(),
            lambda spec, payload: cls.map_paths(
                payload,
                lambda v: FileUtils.rebase_portable_path(v, old_base, new_base),
                spec.path_keys,
            ),
        )

    @classmethod
    def deliverable(cls) -> List[RecordSpec]:
        return [s for s in cls.all() if s.scope is Scope.DELIVERABLE]

    @classmethod
    def private(cls) -> List[RecordSpec]:
        return [s for s in cls.all() if s.scope is Scope.PRIVATE]

    @classmethod
    def by_key(cls, key: str, scope: Optional[Scope] = None) -> Optional[RecordSpec]:
        """The declaration for *key* (in *scope*, or the deliverable one first
        when the key exists in both), else ``None``."""
        matches = [s for s in cls.all() if s.key == key]
        if scope is not None:
            matches = [s for s in matches if s.scope is scope]
        else:
            matches.sort(key=lambda s: s.scope is not Scope.DELIVERABLE)
        return matches[0] if matches else None

    @classmethod
    def resolve(cls, item: Union[RecordSpec, str]) -> RecordSpec:
        """*item* as a spec: a spec passes through; a key is looked up
        (deliverable first).

        Raises:
            KeyError: *item* is a key no record is declared under.
        """
        if isinstance(item, RecordSpec):
            return item
        spec = cls.by_key(str(item))
        if spec is None:
            raise KeyError(f"No scene record is declared under {item!r}.")
        return spec

    @classmethod
    def ordered(cls, specs: Iterable[RecordSpec]) -> List[RecordSpec]:
        """*specs* in dependency order: every record after the ones its
        producer reads (``after``), declaration order breaking ties.

        A dependency that is not in *specs* is simply not waited for -- a
        hand-off refreshing only the derived records still orders them.

        Raises:
            ValueError: The dependencies form a cycle (never guessed around).
        """
        wanted = {s.key: s for s in specs}
        rank = {s.key: i for i, s in enumerate(cls.all())}
        done: List[RecordSpec] = []
        placed: Set[str] = set()
        pending = sorted(wanted.values(), key=lambda s: rank.get(s.key, len(rank)))
        while pending:
            ready = [
                s
                for s in pending
                if all(dep not in wanted or dep in placed for dep in s.after)
            ]
            if not ready:
                raise ValueError(
                    "Scene records depend on each other in a cycle: "
                    + ", ".join(s.key for s in pending)
                )
            for s in ready:
                done.append(s)
                placed.add(s.key)
            pending = [s for s in pending if s.key not in placed]
        return done

    @classmethod
    def check_producers(cls, table: Mapping[Any, Any]) -> List[RecordSpec]:
        """Validate a DCC producer *table* against the registry.

        Every key must resolve to a DELIVERABLE record that is not the handoff
        (which the snapshot stamps itself), so a producer registered under a
        misspelt or private key fails at import of its test, not at export.

        Returns:
            The resolved specs, in *table* order.

        Raises:
            KeyError: A key no record is declared under.
            ValueError: A private record, or the handoff.
        """
        specs = []
        for item in table:
            spec = cls.resolve(item)
            if spec.scope is not Scope.DELIVERABLE:
                raise ValueError(
                    f"{spec.key!r} is a private record; it has no producer."
                )
            if spec is cls.HANDOFF:
                raise ValueError(
                    "The handoff record is stamped by the snapshot, not produced."
                )
            specs.append(spec)
        return specs

    @classmethod
    def declared_takes(cls, read: Callable[[str], Any]) -> List[Dict[str, Any]]:
        """The take list a deliverable declares.

        The one reader of the take list, as ``{"name", "start", "end"}``
        entries: each clip of the shots record that carries its range.

        Parameters:
            read: Channel key -> that channel's DECODED payload, ``None`` when
                absent -- ``dict.get`` over a decoded carrier, a GLB channel
                reader, a context's ``record``.

        Returns:
            The take entries, empty when none are declared.
        """
        meta = read(cls.SHOTS.key)
        shots = meta.get("shots") if isinstance(meta, Mapping) else None
        return [
            {"name": s["clip"], "start": s["start"], "end": s["end"]}
            for s in (shots if isinstance(shots, list) else ())
            if isinstance(s, Mapping) and s.get("clip") and "start" in s and "end" in s
        ]

    # ---------------------------------------------------------------- handoff
    @classmethod
    @Deprecation.symbol(
        "HandoffRecord.block('fbx', channels, source, rendering)",
        remove_in="0.18.0",
        since="2026-10-24",
    )
    def handoff_block(
        cls,
        channels: Union[Iterable[str], Mapping[str, Any]],
        source: Optional[Mapping[str, str]] = None,
        rendering: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """The FBX's handoff block for a carrier holding *channels*, or ``{}``
        when it holds no record (``HandoffRecord.block``)."""
        from pythontk.core_utils.engines.scene_export.handoff_record import (
            HandoffRecord,
        )

        held = HandoffRecord.carried(channels)
        return HandoffRecord.block("fbx", channels, source, rendering) if held else {}

    @classmethod
    def describe(cls) -> List[Dict[str, Any]]:
        """One row per record, for the docs generator and the gates."""
        return [
            {
                "key": s.key,
                "scope": s.scope.value,
                "version": s.version,
                "kind": s.kind.value,
                "owner": s.owner,
                "after": list(s.after),
                "envelope": s.envelope,
                "consumers": list(s.consumers),
                "merge": s.merge.value,
                "portable": s.portable,
                "transient": s.transient,
                "glb": s.glb.version if s.glb else None,
                "description": s.description,
            }
            for s in cls.all()
        ]
