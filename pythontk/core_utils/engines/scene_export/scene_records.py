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
  another language reads it, its payload's shape and whether a GLB carries
  it as it stands (:attr:`RecordSpec.glb_verbatim`).  Nothing else spells a channel name,
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

Which deliverable carrier is the scene's: in a written file every node
``SceneStoreBase.is_carrier`` names is a carrier, and the scene's own is the
one named exactly ``data_export`` at the root of the file's hierarchy
(``SceneStoreBase.is_scene_carrier``). A referenced module's
``NS:data_export``, a numbered copy an import made, and a grouped module's
own-named ``data_export`` under a group are modules'. A writer writes only the
scene's own, made at the root when the scene has none, and never writes over
or deletes a module's; a reader takes each record from the scene's own, a
module's only where the scene's carries none (the first in file order), and
never a module's ``handoff`` (``SceneStoreBase.reader_values``, the one
statement of that precedence). A build's overlay of a converted file
(``MeshConvert.overlay_data_export``) states a channel for that one build: it
takes the channel off every carrier, a module's included, so no reader falls
back to a module's stale copy, and writes it onto the scene's own.

Zero-dep and DCC-agnostic, like everything in ``pythontk``.
"""

import json
import math
import logging
import re
import sys
from dataclasses import asdict, dataclass
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
    Sequence,
    Set,
    Tuple,
    Union,
)

from pythontk.core_utils.class_property import ClassProperty
from pythontk.core_utils.deprecation import Deprecation
from pythontk.iter_utils._iter_utils import IterUtils

logger = logging.getLogger(__name__)


class NodeNames:
    """The names of the nodes an export ships, matched as a reader of the file
    binds a record entry to its node: a name exactly, else its leaf -- what
    follows a DAG path's last ``|`` and a namespace's last ``:`` -- against
    theirs (the GLB and Unity appliers' join). Built once for many records
    (:meth:`RecordSpec.narrowed`, ``ExportContext.ships``): ``name in
    names``."""

    __slots__ = ("names", "leaves")

    def __init__(self, names: Iterable[str]) -> None:
        self.names = frozenset(n.rsplit("|", 1)[-1] for n in names if n)
        self.leaves = frozenset(self.leaf(n) for n in self.names)

    @staticmethod
    def leaf(name: str) -> str:
        return name.rsplit("|", 1)[-1].rsplit(":", 1)[-1]

    def __contains__(self, name: Any) -> bool:
        if not isinstance(name, str) or not name:
            return False
        name = name.rsplit("|", 1)[-1]
        return name in self.names or self.leaf(name) in self.leaves

    @staticmethod
    def resolve(
        name: Optional[str],
        nodes_by_name: Mapping[str, Sequence[Any]],
        leaf_index: Mapping[str, Sequence[str]],
    ) -> Tuple[Optional[List[Any]], str, List[str]]:
        """The nodes a record names *name*, by name alone: the one ladder
        every reader binds a name by -- exact first, then the one name its
        namespace-stripped leaf matches, a leaf several names share never
        guessed. ``MeshConvert``'s record resolver (``_Media.resolve_node``),
        the transport's name fallback (:meth:`Transport.join`) and the WebXR
        viewer's ``kernel/resolve.js`` read it so.

        Parameters:
            name: The name as the record states it.
            nodes_by_name: ``{node name: [node, ...]}`` -- names are not
                unique, so a name holds every node of it.
            leaf_index: ``{leaf: [node name, ...]}``, each name's leaf the
                part after its last ``:``.

        Returns:
            ``(nodes, status, leaves)``: *status* is ``"exact"``, ``"leaf"``,
            ``"ambiguous"`` (in the file, unbindable; *leaves* names the
            candidates) or ``"absent"``.
        """
        if not name:
            return None, "absent", []  # "" indexes the file's NAMELESS nodes
        nodes = nodes_by_name.get(name)
        if nodes:
            return list(nodes), "exact", []
        leaves = list(leaf_index.get(name.rsplit(":", 1)[-1]) or [])
        if len(leaves) == 1:
            return list(nodes_by_name[leaves[0]]), "leaf", leaves
        if len(leaves) > 1:
            return None, "ambiguous", leaves
        return None, "absent", []


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
    #: renumbers shots.
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
        glb_verbatim: Whether a GLB carries the record: ITSELF, as the FBX's
            carrier holds it, under its key in the root ``extras``, in the
            record's own version and shape, its names resolved by each reader
            in its own scene (one layout, two containers) -- the one way a
            GLB carries a record, none bound to the file's indices. False for
            a deliverable record a GLB does not carry: one the GLB states in
            glTF's own terms (the ``materials`` record: the GLB's materials
            are what it describes) or carries otherwise (the ``handoff``).
        export_scoped: Whether the record is narrowed to the export set
            (:attr:`ExportContext.scope`): its entries describe scene nodes,
            and an entry for a node the write leaves out ships that node's
            data (and the files it names) with nothing to bind it to. Its
            producer narrows it, and the snapshot narrows what the producer
            returns and a copy shipped as stored by :attr:`nodes`. The
            scene's own copy stays whole: a scoped assembly keeps it
            (:attr:`ExportSnapshot.scene_wide`) for the exporter to put back.
            A timeline-level record (the shots, the audio) is not: every
            write ships it whole.
        nodes: Where an export-scoped record lists the nodes it describes:
            ``(list field, name field)``, one entry per node -- the name field
            a key, or a path into the entry (``"*"`` walking a list), an entry
            kept when any node it names ships (:meth:`narrowed`). ``None``
            when the payload names no node a reader could narrow it by (a
            material entry names a material, not a node).
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
        in_frame: Whether this version states its values in
            :attr:`SceneRecords.FRAME`; a record moves into it with a
            version of its own, and until then its assembly states its units.
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
    glb_verbatim: bool = False
    export_scoped: bool = False
    assembly: str = ""
    transient: bool = False
    renamed_from: Tuple[str, ...] = ()
    clauses: Tuple[Clause, ...] = ()
    nodes: Optional[Tuple[str, Union[str, Tuple[str, ...]]]] = None
    in_frame: bool = False

    @property
    @Deprecation.symbol(
        "RecordSpec.glb_verbatim -- a GLB carries each record as it stands",
        remove_in="0.18.0",
        since="2026-10-24",
    )
    def web(self) -> Optional[Any]:
        """Held until 0.22.0: the record's GLB block, a ``GlbBlock`` keyed by
        the record in its own version and shape where a GLB carries it
        (:attr:`glb_verbatim`), else ``None``."""
        from pythontk.core_utils.engines.scene_export._retired import record_block

        return record_block(self)

    def narrowed(self, payload: Any, nodes: Iterable[str]) -> Any:
        """*payload* with only the entries (:attr:`nodes`) describing a node
        among *nodes*, matched as a reader binds them (a name exactly, else its
        namespace-stripped leaf); ``None`` when none is, or when the payload
        holds no entry list. A record that declares no node entries is
        returned as it is, and so is a payload every entry of which ships.
        *nodes* may be a :class:`NodeNames` built once for many records.

        The one narrowing every reader of the rule applies: a scoped export's
        snapshot (:class:`ExportSnapshot`), and the post-write embed reading a
        referenced module's copy (``FbxMedia.reader_records``).
        """
        if self.nodes is None:
            return payload
        names = nodes if isinstance(nodes, NodeNames) else NodeNames(nodes)
        field_name, name_path = self.nodes
        path = (name_path,) if isinstance(name_path, str) else tuple(name_path)
        entries = payload.get(field_name) if isinstance(payload, Mapping) else None
        if not isinstance(entries, list):
            return None
        kept = [
            entry
            for entry in entries
            if any(value in names for value in IterUtils.values_at(entry, path))
        ]
        if not kept:
            return None
        if len(kept) == len(entries):
            return payload
        return {**payload, field_name: kept}

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


# The shadows record's runtime, stated once: pythontk's ShadowProjection.model /
# ShadowModel.placement, the fade (ShadowProjection.fade) and the horizon map's
# layout and march (HeightFieldMap.to_rgba / .alpha, the reference every engine
# shader is pinned to), so a reader with only the file rebuilds every plane:
# change them together. test_scene_records pins the constants quoted here to
# the code's and implements the text cold.
_SHADOW_PLANE = (
    "A plane, its source and its contact are the nodes named 'name', "
    "'source' and 'contact'. Each plane is a ground quad, centred on its "
    "node's origin, showing the shadow its source casts from the prop "
    "standing on its contact; every length in the record times 'unit_scale' "
    "is metres, and up is +Y. It draws in its material's colour at alpha = "
    "shadow * fade (both below), its shadow marched from its horizon map. "
    "Its source places the shadow alone and lights nothing."
)
_SHADOW_MODEL = (
    "All in metres, with .xz a point's horizontal (x, z) part, C the "
    "contact's position and L the source's -- a directional source's is "
    "C - s*1e6*max(height, 2*radius), s the unit direction it shines, its "
    "node's local -Y: d = C.xz - L.xz, dist = |d|, u = d/dist ((0, 1) when "
    "dist < 1e-6), k_max = 1 + max_stretch, k_base = clamp((L.y - "
    "ground)/max(1e-4, L.y - C.y), 0, k_max), k_top = clamp((L.y - "
    "ground)/max(1e-4, L.y - C.y - height), 0, min(k_max, k_base + "
    "max_stretch*height/max(dist, 1e-6))), reach = max(0, dist*(k_top - "
    "k_base)), width = 2*radius*max(k_top, k_base) and anchor = L.xz + "
    "d*k_base."
)
_SHADOW_FADE = (
    "Every plane fades, evaluated each frame from those terms whether or not "
    "it follows its source: fade = clamp(intensity/max(0.001, "
    "stretch^falloff_power) * clamp((L.y - C.y)/source_ramp, 0, 1) * "
    "clamp(1 - max(0, C.y - ground)/max(0.001, fade_height), 0, 1), 0, 1), "
    "with stretch = max(1, (reach + radius*(k_base + k_top))/max(1e-4, "
    "2*radius)): the shadow lightens as it stretches, ramps in over "
    "'source_ramp' as the source rises over the contact, and fades out as "
    "the prop rises off the ground. A plane whose source or contact does not "
    "resolve draws at 'intensity'."
)
_SHADOW_PLACEMENT = Clause(
    "With 'follow_source' a runtime re-places the quad every frame (otherwise "
    "its imported keys stand), from its 'canvas' and far = max(reach + "
    "canvas[1]*radius*k_top, 0). Turn the quad so its "
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
    "A plane's shadow is marched from its map, the image 'horizon.texture' "
    "names, at "
    "'horizon.rect' ([scale_u, scale_v, offset_u, offset_v] of the image from "
    "its bottom-left). The map is data, its texels read whole (never filtered or "
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
#: Only a plane that carries one: the projected silhouette, an extension.
_SHADOW_SILHOUETTE = Clause(
    "'silhouette' is the plane's projected silhouette, an extension a reader "
    "may ignore (it then draws the horizon). With 'silhouette.draw' a reader "
    "holding it draws the plane's shadow as the alpha of the image "
    "'silhouette.texture' names ('silhouette.atlas.texture' once packed) "
    "through the quad's own UVs, which then address its tile, its colour the "
    "material's times that map's; without it, the silhouette is the "
    "fallback of a reader that cannot march the horizon.",
    ("planes", "silhouette"),
)
_SHADOW_CLAUSES = (_SHADOW_PLACEMENT, _SHADOW_SILHOUETTE)


@dataclass(frozen=True)
class Frame:
    """The frame every record value is stated in: glTF 2.0's, so a reader in
    any engine converts from one frame, in one place, and never learns which
    producer wrote a file (:attr:`SceneRecords.FRAME`).

    A record moves into it with a version of its own and says so
    (:attr:`RecordSpec.in_frame`); until then its assembly states its own
    units.
    """

    length: str = "metre"
    up: str = "+Y"
    forward: str = "+Z"
    handedness: str = "right"
    rotation: str = "unit quaternion (x, y, z, w)"
    time: str = "second"

    @property
    def text(self) -> str:
        """The frame in one phrase, for the generated references."""
        return (
            f"{self.length}s, {self.handedness}-handed with {self.up} up and "
            f"{self.forward} forward, rotations as a {self.rotation} and times "
            f"in {self.time}s (glTF 2.0's frame)"
        )


@dataclass(frozen=True)
class Transport:
    """How a value animated over time rides a file, one way for every record
    (:attr:`SceneRecords.TRANSPORT`).

    An FBX importer may flatten every animated custom property onto the root,
    so objects sharing a property's name collapse into one curve, while a
    transform's own animation keeps its node. So in an FBX a curve rides a
    transform of its own, a curve proxy, named
    ``<target><separator><channel>`` and carrying the :attr:`marker` user
    property, its :attr:`curve` holding the values, in each take the clip
    plays; a reader rebinds it and drops the node. Its target is a NODE (the
    proxy a child of that node, for :attr:`node_channels`) or a MATERIAL (the
    proxy a child of the export's carrier node, for
    :attr:`material_channels`). A node channel's node is the proxy's PARENT
    wherever the file carries the proxy; only a target with no proxy is
    joined by name (:meth:`join`). A GLB states the same curve per clip on the
    property itself, through :attr:`glb` -- the presence channels
    (:attr:`stepped`) as a stepped ``scale`` of the node, which glTF can
    play.

    A record that animates anything else states its own carrier in its
    assembly.
    """

    marker: str = "curveProxy"
    separator: str = "__"
    curve: str = "scale.x"
    glb: str = "KHR_animation_pointer"
    #: The channels a node's proxy carries (``render_effects``' tracks).
    node_channels: Tuple[str, ...] = ("visibility", "opacity", "highlight")
    #: The channels a material's proxy carries (``render_effects``' materials).
    material_channels: Tuple[str, ...] = ("emission",)
    #: The channels read as steps (on at 1, off at 0), every key a cut.
    stepped: Tuple[str, ...] = ("visibility",)

    @property
    def channels(self) -> Tuple[str, ...]:
        """Every channel that rides the transport, node channels first."""
        return self.node_channels + self.material_channels

    @staticmethod
    def target_name(name: str) -> str:
        """*name* -- a node or material as a record or a host names it -- as
        the transport names a curve's target: its leaf, with no DAG path and
        no namespace (``|grp|ns:Box`` -> ``Box``). A proxy is named by it
        (:meth:`proxy_name`) and a curve is keyed by it (:meth:`parse`), so a
        reader joins a record's target to its CURVE by it: a record names a
        material as the delivery does, namespace included, and a node as its
        host does. The leaf names a curve, never a node: two nodes may share
        it (``a:Box``, ``b:Box``), and the node a curve plays on is its
        proxy's parent (:meth:`join`)."""
        return str(name).split("|")[-1].split(":")[-1]

    def join(
        self,
        name: str,
        paths: Sequence[str],
        carriers: Iterable[int] = (),
    ) -> Tuple[List[int], str]:
        """The node a record's track *name* plays on, among a file's nodes.

        The one rule every reader and producer of a node channel shares: the
        node is the PARENT of the track's curve proxy (``<leaf>__<channel>``,
        :meth:`proxy_name`) wherever the file carries one -- *carriers*, the
        indices of the nodes such a proxy sits under. Several carriers (two
        proxies of one name) narrow to those the name states exactly, else to
        those of its leaf. Only a track with no proxy falls back to its name,
        by the one ladder every reader binds a name by (:meth:`NodeNames.resolve`:
        exact, then the one name its leaf matches), a DAG path in the name
        narrowing the nodes it names. Nodes sharing one name are one name
        (glTF names are not unique: every one plays it); a tie between names
        is never guessed: it is ``"ambiguous"``, which a reader says.

        Parameters:
            name: The track's node as the record names it (a host's name: a
                leaf, a namespaced name or a DAG path).
            paths: Every node of the file, by index, as its path from the root
                (``"|grp|a:Box"``); a node's name alone is a one-step path.
            carriers: The indices of the nodes a proxy of the track's sits
                under.

        Returns:
            ``(indices, status)``: *status* is ``"proxy"``, ``"exact"``,
            ``"leaf"``, ``"ambiguous"`` or ``"absent"`` (no node).
        """
        steps = [step for step in str(name or "").split("|") if step]
        if not steps:
            return [], "absent"

        def exact(index: int) -> bool:
            own = [step for step in str(paths[index] or "").split("|") if step]
            return own[-len(steps):] == steps

        def named(index: int) -> str:
            return str(paths[index] or "").split("|")[-1]

        carried = sorted({i for i in carriers if 0 <= i < len(paths)})
        if carried:
            if len(carried) == 1:
                return carried, "proxy"
            narrowed = [i for i in carried if exact(i)] or [
                i for i in carried if self.target_name(paths[i]) == self.target_name(steps[-1])
            ]
            if narrowed and len({named(i) for i in narrowed}) == 1:
                return narrowed, "proxy"
            return [], "ambiguous"
        by_name: Dict[str, List[int]] = {}
        for index in range(len(paths)):
            if paths[index]:
                by_name.setdefault(named(index), []).append(index)
        leaf_index: Dict[str, List[str]] = {}
        for full in by_name:
            leaf_index.setdefault(full.rsplit(":", 1)[-1], []).append(full)
        nodes, status, _leaves = NodeNames.resolve(steps[-1], by_name, leaf_index)
        if not nodes:
            return [], status
        if len(steps) > 1:  # a DAG path narrows the nodes its name names
            nodes = [i for i in nodes if exact(i)] or nodes
        return sorted(nodes), status

    def proxy_name(self, target: str, channel: str) -> str:
        """The proxy carrying *target*'s *channel*: ``<target>__<channel>``,
        the target by :meth:`target_name` -- a child of the node it animates
        for a node channel (its node is its proxy's parent, :meth:`join`), of
        the export's carrier node for a material channel.

        Raises:
            ValueError: A channel the transport does not carry.
        """
        if channel not in self.channels:
            raise ValueError(
                f"{channel!r} is no transport channel; expected one of "
                f"{', '.join(self.channels)}."
            )
        return f"{self.target_name(target)}{self.separator}{channel}"

    def parse(self, name: str) -> Optional[Tuple[str, str]]:
        """``(target, channel)`` a proxy named *name* carries, or ``None``
        for a name no channel ends; the target by :meth:`target_name`."""
        leaf = self.target_name(name)
        target, sep, channel = leaf.rpartition(self.separator)
        if not sep or not target or channel not in self.channels:
            return None
        return target, channel

    @property
    def text(self) -> str:
        """The transport in one phrase, for the generated references."""
        return (
            f"An animated value rides, in an FBX, on a curve proxy per take: a "
            f"transform named `<target>{self.separator}<channel>`, the target by "
            f"its leaf name (no path, no namespace: a reader joins a record's node "
            f"or material to its curve by that leaf), carrying the "
            f"`{self.marker}` user property, whose `{self.curve}` holds the "
            f"curve -- a child of the node for a node channel "
            f"({', '.join(self.node_channels)}), of the export's carrier node "
            f"for a material channel ({', '.join(self.material_channels)}). "
            f"A node channel plays on its proxy's parent wherever the file "
            f"carries the proxy; only a track with no proxy joins a node by "
            f"name, the exact name first, then the one node of its leaf, a "
            f"tie never guessed; "
            f"{', '.join(self.stepped)} steps. In a GLB it rides on the property "
            f"itself, per clip, through `{self.glb}` channels, presence as a "
            f"stepped scale of the node. A record animating anything else "
            f"states its own carrier in its assembly"
        )



@dataclass(frozen=True)
class GlbMedia:
    """How a file a record names rides a GLB, one way for every record whose
    files ride one (:attr:`SceneRecords.GLB_MEDIA`).

    A record names its files by FILE NAME. An FBX embeds each file itself
    under that name; a GLB holds each as a bufferView whose glTF ``name`` is
    the file's name, joined to the record's name by file name -- no folder on
    either side -- lower-cased on both (:meth:`fold`: Python's ``str.lower``,
    JavaScript's ``toLowerCase``); two views that fold alike hold neither file
    (:meth:`views`, a tie a reader reports). The bufferView's ``extras``
    state its bytes: always their :attr:`mime_type`; where a glTF reader
    cannot read the file's own format, the bytes are a web encoding of it --
    an HDR lightmap as 8-bit sRGB (:attr:`hdr_web`) whose decoded linear
    texels times :attr:`scale` are the map, an HDR probe or highlight copy as
    Radiance RGBE (:attr:`radiance`). The files that ride a GLB this way are
    the ``lightmaps`` record's (its maps, its probe and the probe's copies)
    and the ``shadows`` record's (each plane's horizon map, silhouette and
    atlas: PNG, a format a glTF reader reads, so ``image/png`` with no
    scale); the ``audio`` record's sounds ride none yet.

    Weighed before it (CODE_STANDARD section 15): glTF's own
    ``images[].bufferView`` + ``mimeType`` embed image bytes, but an image is
    there for a texture to sample -- one no texture samples is dead to every
    reader, glTF has no image type for Radiance, and the texture passes
    re-encode images by the slot that samples them; ``KHR_texture_basisu``
    carries KTX2 for a texture likewise. A record's file has no texture: the
    record names it, and its reader decodes it by what its view states. The
    Khronos glTF Validator (2.0.0-dev.3.10, measured 2026-11-07) reports such a
    view as ``UNUSED_OBJECT``, an info, never an error or a warning.
    """

    #: The glTF field of a bufferView that holds the file's name.
    name: str = "name"
    #: The ``extras`` key stating the bytes' media type.
    mime_type: str = "mimeType"
    #: The ``extras`` key an 8-bit web encoding of an HDR map states its
    #: range by: decoded linear texels times it are the map.
    scale: str = "scale"
    #: The web encodings of an HDR lightmap: 8-bit sRGB (a WebP delivery's
    #: lossless WebP, a KTX2 delivery's UASTC).
    hdr_web: Tuple[str, ...] = ("image/png", "image/webp", "image/ktx2")
    #: The encoding of an HDR probe or highlight copy: Radiance RGBE.
    radiance: str = "image/vnd.radiance"

    @staticmethod
    def file_name(name: Any) -> str:
        """*name* as a record names a file: its file name, no folder."""
        return str(name or "").replace("\\", "/").rsplit("/", 1)[-1]

    @classmethod
    def fold(cls, name: Any) -> str:
        """*name* as a join compares it: its file name (:meth:`file_name`),
        lower-cased -- ``str.lower`` here, ``toLowerCase`` in JavaScript, the
        one fold both languages spell alike."""
        return cls.file_name(name).lower()

    def views(self, gltf: Any, name: Any) -> List[int]:
        """The indices of every bufferView of *gltf* (a parsed glTF document)
        holding the file *name*, joined by :meth:`fold` on both sides."""
        wanted = self.fold(name)
        views = gltf.get("bufferViews") if isinstance(gltf, Mapping) else None
        if not wanted or not isinstance(views, list):
            return []
        return [
            index
            for index, view in enumerate(views)
            if isinstance(view, Mapping)
            and isinstance(view.get(self.name), str)
            and self.fold(view[self.name]) == wanted
        ]

    def view(self, gltf: Any, name: Any) -> Optional[int]:
        """The index of the one bufferView of *gltf* holding the file *name*
        (:meth:`views`); ``None`` when none does, and when several do -- a tie
        is never guessed (:meth:`views` says which)."""
        found = self.views(gltf, name)
        return found[0] if len(found) == 1 else None

    def extras(self, mime_type: str, scale: Optional[float] = None) -> Dict[str, Any]:
        """The ``extras`` of a bufferView holding bytes of *mime_type*: with
        *scale* for an 8-bit web encoding of an HDR map (:attr:`hdr_web`).

        Raises:
            ValueError: A *scale* on bytes that are no such encoding, or one
                that is not a finite positive number (glTF JSON holds no NaN
                or infinity, and no map is scaled by zero or less).
        """
        extras: Dict[str, Any] = {self.mime_type: mime_type}
        if scale is not None:
            if mime_type not in self.hdr_web:
                raise ValueError(
                    f"{mime_type!r} is no 8-bit web encoding of an HDR map "
                    f"({', '.join(self.hdr_web)}): it states no scale."
                )
            value = float(scale)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"A map's scale is a finite positive number, not {scale!r}.")
            extras[self.scale] = value
        return extras

    @property
    def text(self) -> str:
        """The media transport in one phrase, for the generated references."""
        return (
            "The files the lightmaps record names -- its maps, its probe and "
            "the probe's highlight copies -- and the shadows record names -- "
            "each plane's horizon map, silhouette and atlas -- ride a GLB "
            "that carries the record, each "
            f"as the bufferView whose glTF `{self.name}` is the file's name "
            "(the audio record's sounds ride none yet). A reader joins the "
            "record's name to that view by file name, both lower-cased "
            "(str.lower, toLowerCase); two views that join one name hold "
            "neither. The bufferView's "
            f"`extras.{self.mime_type}` states its bytes -- the file itself "
            "where a glTF reader reads its format, else a web encoding of "
            "it: an HDR lightmap as 8-bit sRGB ("
            + ", ".join(f"`{m}`" for m in self.hdr_web)
            + f") whose decoded linear texels times `extras.{self.scale}` (a "
            "finite positive number) are the map, an HDR probe or highlight "
            f"copy as Radiance RGBE (`{self.radiance}`). An FBX embeds each "
            "file itself, under the same name"
        )


class SceneRecords:
    """The registry: every record, declared once, and what derives from it.

    Add a record HERE and nowhere else.  The DCC producer tables key on these
    specs (an unregistered key cannot be produced), the FBX handoff block
    describes whatever of these the carrier holds, the doc tables and the
    Unity importer gate are generated from the same rows.
    """

    #: The vocabularies a :class:`RecordSpec` declares a record with, on the
    #: registry that holds every record -- ``SceneRecords.Kind.DERIVED`` --
    #: since their bare names are too generic for the package root.
    Kind = Kind
    Merge = Merge

    #: The frame every record value is stated in (:class:`Frame`).
    FRAME = Frame()
    #: How a value animated over time rides a file (:class:`Transport`).
    TRANSPORT = Transport()
    #: How a file a record names rides a GLB that carries the record as it
    #: stands (:class:`GlbMedia`).
    GLB_MEDIA = GlbMedia()

    # -- deliverable ---------------------------------------------------------
    #: Produced from the scene as staged for each write (the hosts'
    #: ``MaterialRecords.export_record``, through ``MaterialFamilies``) and
    #: never held by it: the export's own converted maps -- and the
    #: metallicRoughness image it packs from them -- are what the record
    #: names, and the restore that puts the scene back clears it. Version 2
    #: (2026-11-05): literal glTF 2.0 -- a glTF document of materials,
    #: textures, images and samplers -- so the Khronos validator checks it,
    #: the GLB build copies it (``MeshConvert.apply_glb_materials``) and any
    #: glTF reader reads it. One layout, two containers: the FBX carries it
    #: by name on its carrier; a GLB's materials ARE its native materials,
    #: built from it, so a GLB repeats no copy. A file without one (a
    #: File > Export) is read into the same families from its own materials.
    #: Version 1 (the flat shape: separate metal and roughness maps, emission
    #: above 1 in its factor) is refused: re-export.
    MATERIALS = RecordSpec(
        "materials",
        Scope.DELIVERABLE,
        2,
        owner="Export",
        description=(
            "every material as glTF 2.0 states it, literally: a glTF document "
            "of the file's materials, textures, images (by file name) and "
            "samplers -- metal and roughness packed into one image, "
            "KHR_texture_transform, KHR_materials_unlit and "
            "KHR_materials_emissive_strength"
        ),
        kind=Kind.DERIVED,
        transient=True,
        # Its producer reads the lightmaps the write ships: a material only
        # their objects wear states no occlusion map (``MaterialFamilies.
        # export_record``).
        after=("lightmaps",),
        consumers=("unity", "glb"),
        shape=(
            "pythontk.core_utils.engines.scene_export.material_record",
            "MaterialRecord",
        ),
        export_scoped=True,
        assembly=(
            "Every material's inputs are exactly as its 'materials' entry "
            "states them, whatever an importer makes of the file's own "
            "material (how each is lit is the lighting parts'): the record is "
            "a glTF 2.0 document -- the record's 'version' aside -- read as "
            "glTF reads one: metallic-roughness, each factor linear and "
            "multiplied by its map, every map on UV set 0 through its "
            "KHR_texture_transform, colour and emission maps sRGB, the others "
            "linear, metal from B and roughness from G of the one packed "
            "image (occlusion from R), opacity the base colour's alpha (MASK "
            "cuts at 'alphaCutoff', BLEND blends), emission the factor times "
            "KHR_materials_emissive_strength, KHR_materials_unlit the base "
            "colour alone. Each image is the file the delivery carries under "
            "its 'uri'. An entry's 'name' is the material's in the file, "
            "namespace included; where an import drops the namespace, the "
            "name after the last ':' joins. A GLB's materials are the record's "
            "own, so it carries no copy."
        ),
    )
    SHOTS = RecordSpec(
        "animation",
        Scope.DELIVERABLE,
        3,
        owner="Shots",
        renamed_from=("shot_metadata",),
        description=(
            "shot definitions -- per shot its clip name, the name of the "
            "file's take for it, and what a clip cannot hold: its description "
            "and its metadata -- and the scene's frame rate (a scene that "
            "animates and declares no shot states the rate alone); the clip "
            "name is the join key to the imported animation clip"
        ),
        # DERIVED, so every write refreshes it, hand-offs included: the shot
        # store is its one authority and it reads the live scene (which shots
        # are stale). Authored, a WebXR or Unity push shipped the last Scene
        # Exporter run's shots or none (measured 2026-10-21, both DCCs).
        kind=Kind.DERIVED,
        consumers=("unity", "glb", "verifier"),
        shape=("pythontk.core_utils.engines.shots.animation_record", "AnimationRecord"),
        # Version 3 (2026-11-05): one take per shot, so the takes are the
        # timing and the record keeps only what a clip cannot hold -- v1's
        # 'start', 'end' and 'clip_mode' restated the takes -- and the frame
        # rate, which glTF cannot hold. The GLB carries the record itself
        # (one layout, two containers), under the key HEAD's published GLB
        # block rode at version 2 -- so 2 is skipped: two shapes under one
        # key and version would be read half-way. Version 1 (HEAD's FBX
        # record) and 2 (HEAD's GLB block) are refused: re-export. The
        # block retired; a reader joins a shot to the animation of its
        # clip's name.
        glb_verbatim=True,
        assembly=(
            "The animation is the file's takes, one per shot, each named after "
            "its shot and spanning its range on the timeline; a file with no "
            "shots carries one take of the whole timeline, and a record with "
            "no shots stating its rate (none when nothing is animated). Each "
            "shot joins its take by 'clip', the take's name: 'description' "
            "says what it shows and 'metadata' is the key/value data its author "
            "wrote for it (the Shot Manifest's keys; the tools' bookkeeping "
            "stays in the scene). 'fps' is the scene's frame rate, for a "
            "reader of a format that holds none (a glTF, whose animations are "
            "in seconds); a reader of one that states its own (an FBX's time "
            "mode) uses that. In a GLB the record rides as it stands, under "
            "its key in the root extras, and each take is the glTF animation "
            "of its name: the takes played in turn are the whole sequence."
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
    #: Version 3 (2026-11-05): cues per clip, in seconds from the clip's
    #: start, each with its sound (the delivered file's name), its offset
    #: into the sound and its stop -- a sound across a cut is cut per clip --
    #: and no frame or rate: the 'animation' record keeps the scene's rate
    #: for a reader that needs one. Version 2 (frames at the record's 'fps',
    #: one event per start, no stop) and version 1 (the flat wire string)
    #: are refused: re-export. One layout, two containers: a GLB carries the
    #: record as it stands. The WebXR viewer plays no audio and reads none
    #: of it; the GLB build carries it (its 'glb' reader).
    AUDIO = RecordSpec(
        "audio",
        Scope.DELIVERABLE,
        3,
        owner="Audio Clips",
        renamed_from=("audio_manifest",),
        description=(
            "the scene's sounds cut into cues per clip: each the sound (the "
            "delivered audio file's name), when it starts and stops in seconds "
            "from the clip's start, and where in the sound it starts; the "
            "cues are the source's, and what a Unity user adds in the model "
            "importer's Events tab is Unity's"
        ),
        # Read off the keyed tracks / sound strips an artist moves between one
        # push and the next, so every write refreshes it, a hand-off included
        # (a Unity-bridge FBX shipped the last export's events otherwise).
        kind=Kind.DERIVED,
        after=("animation",),
        consumers=("unity", "glb"),
        shape=("pythontk.audio_utils.audio_record", "AudioRecord"),
        glb_verbatim=True,
        assembly=(
            "Each cue plays its 'sound' in its 'clip' -- the file's take of "
            "that name, or the file's one take when 'clip' is empty (a file "
            "with no shots) -- from 'time' to 'stop', seconds from the clip's "
            "start, starting 'offset' seconds into the sound; a sound also "
            "ends when the clip that started it stops playing. A sound that "
            "spans several clips has a cue in each, its offset carrying on "
            "where the cut left it. The cues are the source's, rewritten at "
            "every delivery like the keys they time; what a Unity user adds "
            "in the model importer's Events tab is Unity's and stays. The "
            "sound's file rides the file as media of its own, joined to "
            "'sound' by file name without regard to case; 'media', where the "
            "file states it, gives the SHA-256 of the bytes it carries by "
            "that name, and a same-named file found elsewhere stands in for "
            "it only when its bytes hash alike. In a GLB the record rides as "
            "it stands, under its key in the root extras, each clip the glTF "
            "animation of its name."
        ),
    )
    #: The payload's model is ``LightmapRecord``: one ``unit_scale`` for every
    #: length, the probe's box as ``{min, max}``, no direction on a point light.
    #: Version 3 (2026-11-07): one layout, two containers -- a GLB carries the
    #: record as it stands (the index-bound v2 block, its carrier slot, its
    #: materials by name and its probe by bufferView, retired), each file it
    #: names riding as media (:attr:`SceneRecords.GLB_MEDIA`); no entry names
    #: a UV set, the contract reading every map through the second (a producer
    #: whose set sits elsewhere publishes none: ``LightmapRecord.
    #: check_uv_sets``); the probe's map holds the bake's lights as its
    #: capture point sees them, and its highlights are complete copies of it.
    #: Version 2 is refused: re-export.
    LIGHTMAPS = RecordSpec(
        "lightmaps",
        Scope.DELIVERABLE,
        3,
        owner="Lightmap Baker",
        renamed_from=("lightmap_metadata",),
        description=(
            "per-object baked-lightmap records: map file name, scaleOffset, and "
            "the object's scene hierarchy (which tells apart objects that share "
            "a name); the bake's reflection probe, when it captured one -- its "
            "HDR's file name, capture point and projection box -- with copies "
            "of it for groups of receivers; and the lights the maps were baked "
            "from -- every length times unit_scale is metres"
        ),
        consumers=("unity", "glb"),
        shape=(
            "pythontk.core_utils.engines.scene_export.lightmap_record",
            "LightmapRecord",
        ),
        glb_verbatim=True,
        export_scoped=True,
        nodes=("objects", "name"),
        assembly=(
            "Each 'objects' entry lights one object with a baked lightmap: "
            "the image 'map' names, wired to no material, read through the "
            "object's second UV set (index 1) -- where Unity's lightmaps and "
            "glTF's TEXCOORD_1 read one -- at uv * [scaleX, scaleY] + "
            "[offsetX, offsetY] from 'scaleOffset' (in the file's own V-up "
            "UVs), non-directional and applied as stored, linear. The object "
            "is the node 'name' names (its namespace aside), 'hierarchy' -- "
            "its path from the scene root, namespaces kept, its own name "
            "last -- telling apart objects that share a name; a tie nothing "
            "settles is left unlit rather than guessed. The map is the "
            "object's whole diffuse light, its occlusion included ('materials' "
            "says what that leaves of its material's occlusion map). Every "
            "length in the record times 'unit_scale' is metres."
        ),
        clauses=(
            Clause(
                "'probe' is the reflection probe the bake captured: the "
                "scene's environment (rendering.environment). Its 'map' holds "
                "the bake's lights as well, drawn in as the capture point sees "
                "them.",
                ("probe",),
            ),
            Clause(
                "'lights' are the lights the maps were baked from, for a "
                "delivery with no probe: while the record has one, a reader "
                "lights none of them -- the probe's map holds their highlights, "
                "which would count twice. Otherwise their HIGHLIGHTS alone: the "
                "diffuse light of each is already in the maps, so a reader that "
                "lights them must light their specular only, or the scene is "
                "lit twice. Each is 'type' point, spot or rect at 'position', a "
                "spot or rect facing 'direction' (a rect's width along "
                "'right'), its 'color' linear Rec.709, 'intensity' its radiant "
                "intensity for a point or spot (the irradiance it gives one "
                "unit away) and its radiance for a rect, a spot's half 'angle' "
                "(radians) and soft 'penumbra' share, a rect's 'width' and "
                "'height', a source's 'radius'.",
                ("lights",),
            ),
            Clause(
                "'probe.highlights' are COMPLETE copies of the probe, one per "
                "group of receivers: each names a 'map', the environment with "
                "the bake's lights drawn where box projection shows them true "
                "for that group, and the 'objects' ('name' and 'hierarchy', as above) "
                "that take it in the probe's place (diffuse irradiance and "
                "reflections alike), box-projected from the same 'position' "
                "and 'box'; a node is under one copy at most.",
                ("probe", "highlights"),
            ),
        ),
    )
    #: The payload's model is ``ShadowRecord``. Version 3 (2026-11-07): one
    #: layout, two containers -- a GLB carries the record as it stands (the
    #: index-bound v2 block, its planes, sources, contacts and maps by index,
    #: retired), each file it names riding as media (:attr:`GLB_MEDIA`);
    #: every plane carries its horizon map and its fade's terms, which every
    #: reader evaluates each frame (``ShadowProjection.fade``) where v2's
    #: readers played baked opacity keys; the projected silhouette is an
    #: extension, the plane's draw choice in it (a rig's type is its own).
    #: Version 2 is refused: re-export.
    SHADOWS = RecordSpec(
        "shadows",
        Scope.DELIVERABLE,
        3,
        owner="Shadow Rig",
        renamed_from=("shadow_metadata",),
        description=(
            "shadow planes: per plane its node name, the source and contact it "
            "follows, its horizon map, the projection model's constants and "
            "its fade's terms, and its projected silhouette (an extension) -- "
            "every length times unit_scale is metres"
        ),
        consumers=("unity", "glb"),
        shape=("pythontk.geo_utils.shadow_record", "ShadowRecord"),
        glb_verbatim=True,
        export_scoped=True,
        nodes=("planes", "name"),
        assembly=" ".join(
            (_SHADOW_PLANE, _SHADOW_MODEL, _SHADOW_FADE, _SHADOW_HORIZON)
        ),
        clauses=_SHADOW_CLAUSES,
    )
    #: The payload's model is ``ArticulationRecord``. Version 2 (2026-11-08):
    #: one layout, two containers -- a GLB carries the record as it stands
    #: (the index-bound v1 block, its joints and grabbed parts by node index,
    #: retired), each reader resolving the joints and parts by name; the
    #: record is stated in glTF's frame (:attr:`FRAME`: metres, where v1's
    #: lengths were the file's own and a runtime measured their unit off the
    #: rest pose); and every joint carries an anchor frame its channels turn
    #: and slide in, the identity where a DCC rig nests the joint on its
    #: pivot. Version 1 is refused: re-export.
    ARTICULATION = RecordSpec(
        "articulation",
        Scope.DELIVERABLE,
        2,
        owner="Articulated Rig",
        description=(
            "articulated rigs: per rig its joints (node name, parent joint, "
            "rest translate and rotation, the anchor frame its channels turn "
            "and slide in, rotate order, channels with limits and grab "
            "weights) and the parts a hand grabs, each with the joint it "
            "rides -- in metres"
        ),
        consumers=("unity", "glb"),
        shape=("pythontk.geo_utils.articulation.record", "ArticulationRecord"),
        glb_verbatim=True,
        in_frame=True,
        # A rig ships when any of its joints does: an export of one prop of a
        # scene of rigs states that prop's rig alone.
        export_scoped=True,
        nodes=("rigs", ("joints", "*", "name")),
        assembly=(
            "Each rig poses its joints along its joint chain: a joint's pose "
            "is its 'parent' joint's -- for a root, the space that root's own "
            "node hangs in, so each root's chain poses in its own root's "
            "space (a rig's roots may hang under different nodes) -- times "
            "rest . A . pose(channels) . "
            "A^-1, where rest is the joint's 't' and 'q' in its parent "
            "joint's frame, A its 'anchor' ('t' and 'q', in the rest frame) "
            "and pose(channels) the slide (tx, ty, tz) along A's axes after "
            "the turn Euler(rotate_order; rx, ry, rz) about them in degrees, "
            "every channel within its limits. The identity anchor -- 't' 0, "
            "'q' (0, 0, 0, 1), a joint whose part nests on its pivot -- poses "
            "rest . pose(channels). A part is placed at its joint's pose "
            "whatever node it hangs under (a flat model's parts all under "
            "the root chain alike), and a grabbed part rides its joint; a "
            "grab moves the channels of the joints from the part up to the "
            "rig's root, each in proportion to its 'weight' and within its "
            "limits. Joints and grabbed parts are found by node name (the "
            "name after the last ':' where an import drops the namespace; a "
            "name two nodes answer binds neither). Every value is in its "
            "parent joint's frame as the file holds that node: lengths (the "
            "rest and anchor translations, a slide and its limits) in metres "
            "as glTF frames the file -- an FBX's own value x 0.01 whatever its "
            "UnitScaleFactor, a GLB node's local exactly -- and rotations as "
            "quaternions (x, y, z, w); a scale a node above carries (a "
            "writer's unit on the export's root, a scaled prop) scales the "
            "rig with it. A reader whose import converts the file's unit into "
            "the locals (an FBX importer applying the file's UnitScaleFactor "
            "and its own global scale) "
            "scales every length by that conversion x 100. The imported pose "
            "is the rest pose, and the takes play the joints' own keys."
        ),
    )
    #: Version 2 (2026-11-05): the record names which nodes and materials
    #: carry which animated channels and the colours their curves scale; the
    #: curves themselves ride the transport (:attr:`TRANSPORT`), in the takes,
    #: so they scope to clips by construction. v1's ramps (frames at the
    #: record's 'fps', placed on a whole-timeline clip origin) are refused:
    #: re-export. One layout, two containers: a GLB carries the record as it
    #: stands, its curves as the clips' own pointer channels. (No file ever
    #: carried another version under this key or 'visibility_tracks', so the
    #: number is the next one.)
    VISIBILITY = RecordSpec(
        "render_effects",
        Scope.DELIVERABLE,
        2,
        owner="Render Effects",
        renamed_from=("visibility_tracks",),
        shape=(
            "pythontk.file_utils.mesh_convert.render_effects_record",
            "RenderEffectsRecord",
        ),
        description=(
            "which nodes carry an animated render effect -- 'visibility', "
            "'opacity', 'highlight' with its colour stops 'highlight_color' and "
            "'highlight_color_dim' -- and which materials an animated "
            "'emission'; each curve rides the file's animated-value transport, "
            "per clip"
        ),
        kind=Kind.DERIVED,
        after=("animation",),
        consumers=("unity", "glb", "verifier"),
        # Narrowed by its producer alone (no ``nodes``): a material entry
        # names no node a reader could narrow it by.
        export_scoped=True,
        glb_verbatim=True,
        assembly=(
            "Render effects over time, per 'tracks' entry on the subtree under "
            "its 'node' and per 'materials' entry on every material of that "
            "name, each channel it names a curve the file carries on its "
            "animated-value transport, in each clip it plays in (a clip with "
            "no curve for it holds the rest value). 'visibility' shows the "
            "subtree at 1 and hides it at 0, stepped. 'opacity' is its "
            "materials' alpha: a material keeps its authored alpha mode "
            "(opaque, sorted, shadowed, writing depth) while its opacity is 1, "
            "is blended only while its opacity is below 1, and its renderer is "
            "hidden at 0. 'highlight' is a glow, blended from "
            "'highlight_color_dim' (default black) at 0 to 'highlight_color' "
            "(default white) at 1: while it is above 0 the material's emission "
            "is the glow added to its uniform emission, and a mapped emission "
            "is set aside for those moments only -- at rest the map is back. "
            "'emission' is a material's emission weight, 0 to 1, scaling "
            "'emission_color' times 'emission_strength' (the material's own "
            "where absent), its map kept. In an FBX each curve is the "
            "'<target>__<channel>' proxy's scale.x: transport, not content."
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
        # The GLB build reads the FBX's to state the GLB's, and the glTF
        # runtimes the GLB's. Unity lights with its pipeline's own Lit and
        # reads none of it.
        consumers=("glb",),
        shape=(
            "pythontk.core_utils.engines.scene_export.handoff_record",
            "HandoffBlock",
        ),
    )
    #: Keys a scene may still hold for a record retired whole, which nothing
    #: reads: cleared from a carrier (:meth:`SceneStoreBase.migrate`) and
    #: dropped from a written deliverable. A renamed record names its old keys
    #: itself (:attr:`RecordSpec.renamed_from`).
    RETIRED: Tuple[str, ...] = ("fbx_takes", "emissive_groups")
    #: What a reader of an old scene or file is told for a retired key it
    #: finds (``ExportVerifier.check_fbx_records`` names it with this fix).
    #: ``emissive_groups`` retired 2026-11-07: a group is materials, its faces
    #: wearing a copy of their material per group whose emission the
    #: ``render_effects`` record's material entries animate, so the mask
    #: encodings, the slots and this record went with it.
    RETIRED_FIXES: Dict[str, str] = {
        "emissive_groups": (
            "emissive groups are materials now: open the scene, convert its "
            "groups in one action (Emissive Groups > Convert Old Groups, "
            "EmissiveGroups.convert_legacy), then re-export"
        ),
    }
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
    #: Version 2 (2026-11-07): a group is materials, each copy tagged on the
    #: material itself, so nothing is stored here any more; the key is the
    #: hand-off's: what crosses a bridge for the groups (per copy its group,
    #: base, strength and emission keys -- an interchange file carries the
    #: materials and their faces, never the tags or the material animation),
    #: produced by the owner's ``transfer_out`` and landed by its
    #: ``transfer_in``. A scene the old tool authored still holds version 1
    #: (slots, defaults, encoding), which ``EmissiveGroups.convert_legacy``
    #: reads and removes; an imported scene's is adopted where this scene has
    #: none (``UNION``), so a referenced old-style module converts too.
    EMISSIVE_REGISTRY = RecordSpec(
        "emissive_registry",
        Scope.PRIVATE,
        2,
        owner="Emissive Groups",
        description=(
            "what crosses a hand-off for the emissive groups: per group copy "
            "its group, base material, emission strength and emission keys "
            "(the old tool's slot registry, version 1, is converted and removed)"
        ),
        renamed_from=("emissive_groups",),
        envelope=False,
        merge=Merge.UNION,
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
    }

    #: The entry-point group a package registers its record hooks in: each
    #: entry named ``<host>.<role>.<key>`` (``maya.producer.lightmaps``) and
    #: pointing at ``module:Class.method``, or at ``module:Class`` for an owner.
    #: A tool outside a library lists itself there instead of in the library's
    #: tables (:meth:`registered_hooks`).
    HOOK_GROUP = "m3trik.scene_records"
    #: The group's ``(name, value)`` pairs and the ``sys.path`` they were read
    #: under (:meth:`registered_hooks`); ``None`` until first asked.
    _hook_points: ClassVar[Optional[Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]]]] = None

    @classmethod
    def registered_hooks(
        cls, host: str, role: str
    ) -> Dict[str, Tuple[str, str, Optional[str]]]:
        """The record hooks installed packages register for *host* and *role*.

        Read from entry-point metadata alone, so nothing is imported: an
        installed tool's hooks are found by a batch export that imports one
        library and never the tool. A host's producer table, stager table and
        record owners (``FbxUtils.PRODUCERS``, ``STAGERS``,
        :attr:`SceneStoreBase.OWNERS`) merge these under their own rows.

        Parameters:
            host: The DCC the hooks serve (``"maya"``, ``"blender"``).
            role: ``"producer"``, ``"stager"`` or ``"owner"``, or a single
                call a library makes into tool code (``FbxUtils.HOOKS``:
                ``take_stash``, ``offset_fold``, ``dependency_dirs``; see
                :meth:`hook_row`).

        Returns:
            ``{key: (module, class, method)}``, *method* ``None`` for an owner.
            A key keeps any ``.slot`` suffix its entry carries (a stager's
            ``effects.prepare`` / ``effects.finish``).

        The metadata is read once per ``sys.path`` -- installed packages do not
        change under a running DCC, record owners are asked often, and a path
        that changed (a package put on it) reads it again, as does
        :meth:`forget_registered_hooks`. Two distributions naming one entry:
        the first on the path wins, as its import would, and the other is
        warned about. *method* is a direct attribute of the class; a dotted
        path past it is not followed.
        """
        path = tuple(sys.path)
        if cls._hook_points is None or cls._hook_points[0] != path:
            from pythontk.core_utils.package_manager import PackageManager

            cls._hook_points = (
                path,
                tuple(
                    (point.name, point.value)
                    for point in PackageManager.entry_points(cls.HOOK_GROUP)
                ),
            )
        rows: Dict[str, Tuple[str, str, Optional[str]]] = {}
        seen: Dict[str, str] = {}
        for name, value in cls._hook_points[1]:
            if name in seen:
                if value != seen[name]:
                    logger.warning(
                        "Record hook %r is registered twice; %s is used, %s ignored.",
                        name, seen[name], value,
                    )
                continue
            seen[name] = value
            parts = name.split(".", 2)
            if len(parts) != 3 or parts[0] != host or parts[1] != role:
                continue
            module, _, target = value.split("[", 1)[0].strip().partition(":")
            owner, _, method = target.strip().partition(".")
            if module and owner:
                rows[parts[2]] = (module.strip(), owner, method or None)
        return dict(sorted(rows.items()))

    @classmethod
    def forget_registered_hooks(cls) -> None:
        """Read the hook group's metadata again on the next
        :meth:`registered_hooks` -- after installing a package that registers
        hooks, in the same session."""
        cls._hook_points = None

    @classmethod
    def hook_row(
        cls, row: Optional[Tuple[str, str, str]], host: str, role: str
    ) -> Optional[Tuple[str, str, str]]:
        """The row a single-hook seam resolves: *row*, the library's own
        ``(module, class, method)``, while it has one, else the one installed
        packages register for *host* and *role* (:meth:`registered_hooks`, any
        key) -- what a library's call into tool code becomes once the tool
        moves out (``FbxUtils.HOOKS``). Several registered: the first by key
        is used and the rest warned about. ``None`` when neither names one.

        Parameters:
            row: The library's own row, or ``None`` once the tool left it.
            host: The DCC (``"maya"``, ``"blender"``).
            role: The seam (``"take_stash"``, ``"offset_fold"`` ...).
        """
        if row:
            return row
        found = {
            key: hook for key, hook in cls.registered_hooks(host, role).items() if hook[2]
        }
        if not found:
            return None
        first, *rest = sorted(found)
        if rest:
            logger.warning(
                "%d packages register a %r hook for %s; %r is used, %s ignored.",
                len(found), role, host, first, ", ".join(map(repr, rest)),
            )
        module, owner, method = found[first]
        return (module, owner, method)

    @classmethod
    def producer_rows(
        cls, table: Mapping[Any, Tuple[str, str, str]], host: str
    ) -> Dict[Any, Tuple[str, str, str]]:
        """A host's producer *table* (``FbxUtils.PRODUCERS``: record spec ->
        ``(module, class, method)``) with the producers installed packages
        register for *host* (:meth:`registered_hooks`).

        A record both name keeps *table*'s row, and an entry naming a record
        nobody declares, or no method, is skipped with a warning: the library
        decides what it ships, and a typo must not pass silently.

        Parameters:
            table: The library's own rows.
            host: The DCC (``"maya"``, ``"blender"``).
        """
        rows: Dict[Any, Tuple[str, str, str]] = dict(table)
        taken = {cls.resolve(spec).key for spec in rows}
        for key, (module, owner, method) in cls.registered_hooks(host, "producer").items():
            try:
                spec = cls.resolve(key)
            except KeyError:
                logger.warning("Registered producer %r names no scene record; skipped.", key)
                continue
            if not method:
                logger.warning("Registered producer %r names no method; skipped.", key)
                continue
            if spec.key not in taken:
                rows[spec] = (module, owner, method)
        return rows

    @classmethod
    def stager_rows(
        cls, table: Mapping[str, Tuple[str, str, str, Optional[str]]], host: str
    ) -> Dict[str, Tuple[str, str, str, Optional[str]]]:
        """A host's stager *table* (``FbxUtils.STAGERS``: name ->
        ``(module, class, prepare, finish)``) with the stagers installed
        packages register for *host*: two entries per stager,
        ``<host>.stager.<name>.prepare`` and an optional ``.finish``, on one
        class. A name both declare keeps *table*'s row; a stager without a
        ``prepare``, or whose slots name two classes, is skipped with a warning.

        Parameters:
            table: The library's own rows.
            host: The DCC (``"maya"``, ``"blender"``).
        """
        rows: Dict[str, Tuple[str, str, str, Optional[str]]] = dict(table)
        slots: Dict[str, Dict[str, Tuple[str, str, Optional[str]]]] = {}
        for key, row in cls.registered_hooks(host, "stager").items():
            name, _, slot = key.rpartition(".")
            if name and slot in ("prepare", "finish") and row[2]:
                slots.setdefault(name, {})[slot] = row
            else:
                logger.warning(
                    "Registered stager %r names no prepare/finish slot or no method; skipped.", key
                )
        for name, parts in sorted(slots.items()):
            if name in rows:
                continue
            homes = {(module, owner) for module, owner, _method in parts.values()}
            if "prepare" not in parts or len(homes) != 1:
                logger.warning("Registered stager %r needs a prepare on one class; skipped.", name)
                continue
            module, owner = homes.pop()
            finish = parts.get("finish")
            rows[name] = (module, owner, parts["prepare"][2], finish[2] if finish else None)
        return rows

    @staticmethod
    def resolve_row(
        label: str,
        module: str,
        owner: str,
        *methods: Optional[str],
        log: Optional[logging.Logger] = None,
    ) -> Optional[Tuple[Optional[Callable], ...]]:
        """The callables a producer, stager or hook row names: *owner*'s
        *methods*, imported from *module* on first use.

        ``None`` when the module or the class cannot be imported -- an
        uninstalled subsystem is fine, and only debug-logged. A method the
        class does not have resolves to ``None`` with a WARNING: that is a
        misspelt row, and skipping it quietly would stop a record shipping (or
        a stager finishing) with no sign of why. A method the row leaves
        ``None`` resolves to ``None`` quietly: a declared one-way stage.

        Parameters:
            label: What the log calls the row (``"Producer for 'materials'"``).
            module: The module the class lives in.
            owner: The class.
            methods: The method names, in the row's order.
            log: The logger the row's library reports through; default this
                module's.
        """
        import importlib

        log = log or logger
        try:
            found = getattr(importlib.import_module(module), owner)
        except Exception:  # an uninstalled subsystem is fine
            log.debug("%s unavailable; skipped.", label, exc_info=True)
            return None
        resolved = []
        for method in methods:
            if not method:  # a column the row leaves empty: a one-way stage
                resolved.append(None)
                continue
            fn = getattr(found, method, None)
            if fn is None:
                log.warning(
                    "%s names %s.%s.%s, which does not exist; skipped.",
                    label,
                    module,
                    owner,
                    method,
                )
            resolved.append(fn)
        return tuple(resolved)

    @classmethod
    def resolved_producers(
        cls,
        table: Mapping[Any, Tuple[str, str, str]],
        host: str,
        only: Optional[Iterable[Any]] = None,
        log: Optional[logging.Logger] = None,
    ) -> Dict[RecordSpec, Callable]:
        """A host's producer *table* with the registered ones
        (:meth:`producer_rows`), each resolved to its callable
        (:meth:`resolve_row`); a row that does not resolve is skipped.

        Parameters:
            table: The library's own rows (``FbxUtils.PRODUCERS``).
            host: The DCC (``"maya"``, ``"blender"``).
            only: Specs or keys narrowing the table; ``None``: all.
            log: As :meth:`resolve_row`'s.
        """
        wanted = None if only is None else {cls.resolve(k).key for k in only}
        out: Dict[RecordSpec, Callable] = {}
        for spec, row in cls.producer_rows(table, host).items():
            if wanted is not None and spec.key not in wanted:
                continue
            resolved = cls.resolve_row(f"Producer for {spec.key!r}", *row, log=log)
            if resolved and resolved[0] is not None:
                out[spec] = resolved[0]
        return out

    @classmethod
    def resolved_stagers(
        cls,
        table: Mapping[str, Tuple[str, str, str, Optional[str]]],
        host: str,
        names: Optional[Iterable[str]] = None,
        log: Optional[logging.Logger] = None,
    ) -> Dict[str, Tuple[Optional[Callable], ...]]:
        """A host's stager *table* with the registered ones
        (:meth:`stager_rows`), each resolved to its ``(prepare, finish)``
        callables (:meth:`resolve_row`); a row whose class does not import is
        skipped.

        Parameters:
            table: The library's own rows (``FbxUtils.STAGERS``).
            host: The DCC (``"maya"``, ``"blender"``).
            names: The stagers wanted; ``None``: all.
            log: As :meth:`resolve_row`'s.
        """
        out: Dict[str, Tuple[Optional[Callable], ...]] = {}
        for name, row in cls.stager_rows(table, host).items():
            if names is not None and name not in names:
                continue
            resolved = cls.resolve_row(f"Stager {name!r}", *row, log=log)
            if resolved is not None:
                out[name] = resolved
        return out

    # -- stack-shaped records ------------------------------------------------
    # A record holding a list of ``{"id": ...}`` entries, oldest first, that a
    # host pushes onto and takes back off (``SMART_BAKE_SESSIONS``): the latest
    # entry, or the latest with an id, read the same in every host.
    @staticmethod
    def peek_entry(
        entries: Optional[List[dict]], entry_id: Optional[str] = None
    ) -> Optional[dict]:
        """The latest of *entries* (oldest first), or the latest whose ``id``
        is *entry_id*; ``None`` when there is none."""
        if not entries:
            return None
        if entry_id is None:
            return entries[-1]
        for entry in reversed(entries):
            if entry.get("id") == entry_id:
                return entry
        return None

    @staticmethod
    def pop_entry(
        entries: Optional[List[dict]], entry_id: Optional[str] = None
    ) -> Optional[dict]:
        """Remove and return what :meth:`peek_entry` would, from *entries* in
        place; ``None`` (and *entries* untouched) when there is none."""
        if not entries:
            return None
        if entry_id is None:
            return entries.pop()
        for i in range(len(entries) - 1, -1, -1):
            if entries[i].get("id") == entry_id:
                return entries.pop(i)
        return None

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

    # Held until 0.22.0 (noticed for 0.18.0 on 2026-10-24, its window closing
    # 2026-11-23): a GLB carries each record as it stands, so its block's shape
    # is the record's.
    @classmethod
    @Deprecation.symbol("SceneRecords.shape", remove_in="0.18.0", since="2026-10-24")
    def web_shape(cls, item: Union[RecordSpec, str]) -> Optional[Any]:
        """:meth:`shape`."""
        return cls.shape(item)

    @classmethod
    def shape(cls, item: Union[RecordSpec, str]) -> Optional[Any]:
        """The :class:`~pythontk.SchemaSpec` declaring *item*'s payload
        (:attr:`RecordSpec.shape`), imported on first use; ``None`` while the
        payload is undeclared."""
        spec = cls.resolve(item)
        return cls.resolve_class(*spec.shape) if spec.shape else None

    @classmethod
    def put_glb_block(
        cls,
        gltf: Dict[str, Any],
        item: Union[RecordSpec, str],
        block: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Write *block* -- *item*'s record as a GLB carries it
        (:attr:`RecordSpec.glb_verbatim`) -- in the glTF document *gltf*: its
        root ``extras``, under the record's key, the record's version stamped
        (a writer's own wins) and pruned to its declared shape, as
        :meth:`RecordSpec.make` does a record. The one writer of every
        record a GLB carries, beside :meth:`glb_block`, its one reader.
        Returns the block written.

        A "block" is the GLB's word for a root-``extras`` entry, and every
        one is now its record itself (no index-bound projection is left,
        articulation v2): the name stays, as the viewer's ``readBlock`` /
        ``BLOCKS`` and :attr:`RETIRED_BLOCKS` do, for the root-extras entry
        it writes, and what it writes is the record."""
        spec = cls.resolve(item)
        body: Dict[str, Any] = {"version": spec.version}
        body.update(block)
        shape = cls.shape(spec)
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
        """*item*'s record in the glTF document *gltf* -- its root ``extras``,
        under the record's key -- in the current schema (the record's
        version); ``None`` otherwise, and for a record a GLB does not carry
        (:attr:`RecordSpec.glb_verbatim`). The block itself -- the record as
        it stands under its key (a "block" is a root-``extras`` entry,
        :meth:`put_glb_block`) -- so a pass that edits it edits the document."""
        spec = cls.resolve(item)
        extras = gltf.get("extras") if isinstance(gltf, Mapping) else None
        if not spec.glb_verbatim or not isinstance(extras, Mapping):
            return None
        block = extras.get(spec.key)
        current = isinstance(block, dict) and block.get("version") == spec.version
        return block if current else None

    @classmethod
    def glb_records(cls) -> List[RecordSpec]:
        """The records a GLB carries in its root ``extras``, each as it stands
        (:attr:`RecordSpec.glb_verbatim`), in declaration order: the records
        themselves, the blocks :meth:`glb_block` reads."""
        return [s for s in cls.all() if s.glb_verbatim]

    @classmethod
    def problems(cls, item: Union[RecordSpec, str], payload: Any) -> List[str]:
        """What *payload* breaks of *item*'s declaration, one line each: a
        version other than the declared one (:meth:`RecordSpec.is_current`),
        or what its declared shape refuses (``SchemaSpec.validate``); empty
        when a reader reads it. The same lines hold a GLB's copy, said where
        the file holds it (``extras.`` before each)."""
        spec = cls.resolve(item)
        return cls._problems(spec.key, spec.version, cls.shape(spec), payload)

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
    def json_schema(cls) -> Dict[str, Any]:
        """The contract as one JSON Schema (draft 2020-12) document.

        Its root is what a file's ``data_export`` node holds: every
        deliverable record under its key, a ``$ref`` to its declared shape,
        with its version (``x-version``) and readers (``x-readers``).
        ``$defs`` holds every shape, and ``GlbExtras`` the blocks a GLB's
        root ``extras`` holds, by the same keys. ``x-frame``,
        ``x-transport`` and ``x-media`` state :attr:`FRAME`,
        :attr:`TRANSPORT` and :attr:`GLB_MEDIA`, and each record's
        ``x-in-frame`` whether its version is in the frame. A
        reader in any language validates a delivery against it or generates
        its types from it (``m3trik/scripts/sync_scene_records.py`` publishes
        it). A shape one record holds whole and another nests is defined once.

        Returns:
            dict: The document, plain JSON values.

        Raises:
            ValueError: Two shapes define one name differently.
        """
        from pythontk.core_utils.schema_spec import SchemaSpec

        defs: Dict[str, Any] = {}

        def define(shape) -> str:
            schema = shape.json_schema()
            nested = schema.pop("$defs", {})
            # What only a document's root carries: a nested copy of the same
            # shape has neither, and must compare equal to this one.
            schema.pop("$schema", None)
            schema.pop("title", None)
            for name, body in list(nested.items()) + [(shape.__name__, schema)]:
                if defs.setdefault(name, body) != body:
                    raise ValueError(
                        "SceneRecords.json_schema: two shapes define "
                        f"{name!r} differently"
                    )
            return f"#/$defs/{shape.__name__}"

        def entry(spec, shape, version) -> Dict[str, Any]:
            out: Dict[str, Any] = {"description": spec.description}
            if shape is not None:
                out["$ref"] = define(shape)
            out["x-version"] = version
            if spec.consumers:
                out["x-readers"] = list(spec.consumers)
            return out

        records = {}
        for spec in cls.deliverable():
            records[spec.key] = entry(spec, cls.shape(spec), spec.version)
            records[spec.key]["x-in-frame"] = spec.in_frame
        blocks = {
            spec.key: entry(spec, cls.shape(spec), spec.version)
            for spec in cls.glb_records()
        }
        handoff = cls.HANDOFF
        blocks[handoff.key] = entry(handoff, cls.shape(handoff), handoff.version)
        defs["GlbExtras"] = {
            "type": "object",
            "description": (
                "A GLB's root extras: each record's block under the record's key."
            ),
            "properties": blocks,
        }
        return {
            "$schema": SchemaSpec.JSON_SCHEMA_DIALECT,
            "title": "SceneRecords",
            "description": (
                "What a deliverable's data_export node holds: each record's "
                "JSON under its key. A record whose x-in-frame is true "
                f"states its values in {cls.FRAME.text}; any other states its "
                f"units in its assembly. {cls.TRANSPORT.text}. "
                f"{cls.GLB_MEDIA.text}. A field naming "
                "its readers (x-readers) is an optional extension the others "
                "ignore."
            ),
            "x-frame": asdict(cls.FRAME),
            "x-transport": asdict(cls.TRANSPORT),
            "x-media": asdict(cls.GLB_MEDIA),
            "type": "object",
            "properties": records,
            "$defs": defs,
        }

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
                "glb": s.version if s.glb_verbatim else None,
                "description": s.description,
            }
            for s in cls.all()
        ]


# Held until 0.22.0 (``pythontk/__init__.py``, "Held until 0.22.0"): the GLB
# block names v0.20.0 noticed for removal, whose window closes 2026-11-23.
from pythontk.core_utils.engines.scene_export._retired import (  # noqa: E402, F401 (re-exported)
    GlbBlock,
    WebProjection,
)
