# !/usr/bin/python
# coding=utf-8
"""Scene records -- every piece of tool-authored scene metadata, declared once.

A *record* is one JSON channel a DCC tool keeps on the scene: the shot store,
a lightmap manifest, the keyed-visibility tracks.  Each has exactly one
declaration here (:class:`SceneRecords`), and everything that used to spell a
channel name, version or description on its own -- the DCC carriers, the
producers, the FBX handoff block, the doc tables, the Unity importer gate --
derives from that declaration instead.  Three ideas, three classes:

- **Declare once, derive everything.** :class:`RecordSpec` is the declaration:
  key, scope, version, description, kind and dependencies.  Nothing else
  spells a channel name.
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
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

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
            carries it in its payload; a reader refuses a NEWER one.
        owner: The tool that produces it (a role name, DCC-agnostic).
        description: What the record holds, for the handoff block and the
            docs -- the one sentence a standalone reader gets.
        kind: :class:`Kind`; decides when the record is refreshed.
        after: Keys of the records this one's producer READS, so the
            coordinator orders producers by declaration rather than by dict
            order and a comment.
        envelope: True for a dict payload carrying its version (every
            deliverable record); False for a legacy shape stored as-is (a bare
            list, a store's own document).
        version_key: The payload key the version rides under -- ``"version"``
            unless the payload is a document with its own schema key (the
            emissive manifest's ``"schema"``, which its Unity reader gates on):
            one version per record, never two that could disagree.
        deprecated_by: The key of the record that supersedes this one.  Still
            written while present (a one-release dual write); readers prefer
            the successor.
        remove_in: The pythontk release a deprecated record stops being written.
        consumers: Names of the readers, for the docs and the cross-package
            gates (``"unity"``: the C# importers; ``"glb"``: the GLB appliers;
            ``"verifier"``: the deliverable gates).
    """

    key: str
    scope: Scope
    version: int
    owner: str
    description: str
    kind: Kind = Kind.AUTHORED
    after: Tuple[str, ...] = ()
    envelope: bool = True
    version_key: str = "version"
    deprecated_by: Optional[str] = None
    remove_in: Optional[str] = None
    consumers: Tuple[str, ...] = ()

    # ------------------------------------------------------------------ codec
    def make(self, payload: Any) -> Record:
        """A :class:`Record` of *payload*, the envelope's ``version`` applied.

        An enveloped record's payload must be a mapping; its ``version`` is
        stamped from this declaration when absent, so a producer never spells
        the number.  A non-enveloped payload is taken as given.

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
            body = {self.version_key: self.version}
            body.update(payload)
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
            found = payload.get(self.version_key)
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


class SceneRecords:
    """The registry: every record, declared once, and what derives from it.

    Add a record HERE and nowhere else.  The DCC producer tables key on these
    specs (an unregistered key cannot be produced), the FBX handoff block
    describes whatever of these the carrier holds, the doc tables and the
    Unity importer gate are generated from the same rows.
    """

    # -- deliverable ---------------------------------------------------------
    SHOTS = RecordSpec(
        "shot_metadata",
        Scope.DELIVERABLE,
        1,
        owner="Shots",
        description=(
            "shot definitions -- per clip its frame range ('start'/'end', the "
            "take the clip is cut from), objects, and any description and "
            "section; the scene fps and the declared clip mode; the clip name "
            "is the join key to the imported animation clip"
        ),
        consumers=("unity", "glb", "verifier"),
    )
    #: No longer written (the ranges ride ``shot_metadata``'s clips); declared
    #: so a file written before 0.11.0 still reads (:meth:`declared_takes`),
    #: and cleared from a scene the next time its shots are published.
    FBX_TAKES = RecordSpec(
        "fbx_takes",
        Scope.DELIVERABLE,
        1,
        owner="Shots",
        description=(
            "the take list an older file carries, one per shot -- superseded "
            "by shot_metadata's per-clip ranges and no longer written"
        ),
        envelope=False,
        deprecated_by="shot_metadata",
        consumers=("glb", "verifier"),
    )
    AUDIO = RecordSpec(
        "audio_manifest",
        Scope.DELIVERABLE,
        2,
        owner="Audio Clips",
        description="audio events with the frames they fire on, scoped to their clip",
        after=("shot_metadata",),
        consumers=("unity",),
    )
    LIGHTMAPS = RecordSpec(
        "lightmap_metadata",
        Scope.DELIVERABLE,
        1,
        owner="Lightmap Baker",
        description=(
            "per-object baked-lightmap records: map file name, uvIndex, "
            "intensity, scaleOffset"
        ),
        consumers=("unity", "glb"),
    )
    SHADOWS = RecordSpec(
        "shadow_metadata",
        Scope.DELIVERABLE,
        2,
        owner="Shadow Rig",
        description=(
            "projected-shadow planes: per plane, the plane node name, its "
            "silhouette texture file name, and the authored intensity"
        ),
        consumers=("unity", "glb"),
    )
    EMISSIVE_GROUPS = RecordSpec(
        "emissive_groups",
        Scope.DELIVERABLE,
        1,
        owner="Emissive Groups",
        description="named emissive material groups and their weights",
        # A RegionMaskManifest document versions itself under "schema"
        # (RegionMaskManifest.SCHEMA_VERSION, pinned equal by the tests).
        version_key="schema",
        consumers=("unity",),
    )
    VISIBILITY = RecordSpec(
        "visibility_tracks",
        Scope.DELIVERABLE,
        1,
        owner="Render Effects",
        description=(
            "keyed visibility per node, as stepped on/off frames, with the "
            "authored opacity ramp and each take's first/last authored frame"
        ),
        kind=Kind.DERIVED,
        after=("shot_metadata",),
        consumers=("glb", "verifier"),
    )
    HANDOFF = RecordSpec(
        "handoff",
        Scope.DELIVERABLE,
        1,
        owner="Export",
        description=(
            "the standalone-reader contract: what each channel present on the "
            "carrier holds"
        ),
        kind=Kind.DERIVED,
    )

    # -- private -------------------------------------------------------------
    SHOT_STORE = RecordSpec(
        "shot_store",
        Scope.PRIVATE,
        1,
        owner="Shots",
        description="the shot store's full app state",
        envelope=False,
    )
    KEY_STASH = RecordSpec(
        "key_stash",
        Scope.PRIVATE,
        1,
        owner="Key Stash",
        description="the clip manifest of parked keys",
        envelope=False,
    )
    SMART_BAKE_SESSIONS = RecordSpec(
        "smart_bake_sessions",
        Scope.PRIVATE,
        2,
        owner="SmartBake",
        description="LIFO stack of bake-session restore manifests",
        envelope=False,
    )
    HIERARCHY_BASELINE = RecordSpec(
        "hierarchy_baseline",
        Scope.PRIVATE,
        1,
        owner="Hierarchy check",
        description="the export hierarchy baseline (a HierarchyBaseline record)",
        envelope=False,
    )
    EMISSIVE_REGISTRY = RecordSpec(
        "emissive_groups",
        Scope.PRIVATE,
        1,
        owner="Emissive Groups",
        description="the group registry: slots, defaults, encoding",
        envelope=False,
    )
    RENDER_EFFECTS_BINDINGS = RecordSpec(
        "render_effects_bindings",
        Scope.PRIVATE,
        1,
        owner="Render Effects",
        description=(
            "the viewport material bindings a preview drives, so a suspend "
            "and rebind round-trips"
        ),
        envelope=False,
    )
    AUDIO_FILE_MAP = RecordSpec(
        "audio_file_map",
        Scope.PRIVATE,
        1,
        owner="Audio Clips",
        description="track id to audio file path",
        envelope=False,
    )

    @staticmethod
    def rendering_policy() -> Dict[str, Any]:
        """What a deliverable claims about how it should be lit.

        The GLB's handoff section and the FBX's handoff record publish the
        same policy, and it is the reference viewer's contract
        (``MeshConvert.RENDERING_POLICY``, pinned against the viewer's own
        literals by ``test_preview_server``), so it is read from there rather
        than declared twice.  Imported lazily: this module is the lighter
        dependency and must import first.
        """
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        return copy.deepcopy(MeshConvert.RENDERING_POLICY)

    #: The standalone-reader contract for the carrier the block rides on.
    #: Plain declarative sentences about the file's own structure, no
    #: instructions to the reader (which is what makes it safe for an agent to
    #: read as untrusted content), and no container format named: the same
    #: records are read out of a GLB, a sidecar or a dump, where "the FBX"
    #: misleads.  The load-bearing sentence is the one about the lightmaps NOT
    #: being embedded by the materials: a lighting-only bake deliberately
    #: leaves the map unwired so the authored material survives.
    HANDOFF_INSTRUCTIONS = (
        "The file carrying this block embeds every texture its MATERIALS "
        "reference, so material assignment needs no external files or paths. "
        "Tool-authored metadata rides on the 'data_export' node, each channel "
        "value a JSON string; 'reads' names each channel present and what it "
        "holds. 'lightmap_metadata' names each baked object's map by FILE "
        "NAME, with 'uvIndex' (0-based: 1 is the second UV set), 'intensity' "
        "(the multiplier restoring the bake's range) and 'scaleOffset' "
        "([scaleX, scaleY, offsetX, offsetY] within a shared atlas); each "
        "baked object also carries its own 'lightmapInfo' property. Those maps "
        "are NOT embedded by the materials -- a lighting-only bake leaves them "
        "unwired so the authored material survives -- so the file name is a "
        "join token against maps supplied separately; the manifest names no "
        "folder. The asset carries no lights; 'rendering' records the setup "
        "the reference viewer approved it under, so lighting it differently "
        "renders differently without either side being wrong. Check "
        "'version' against the schema you expect."
    )

    # ------------------------------------------------------------- catalogue
    @classmethod
    def all(cls) -> List[RecordSpec]:
        """Every declared record, in declaration order."""
        return [v for v in vars(cls).values() if isinstance(v, RecordSpec)]

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
        """The take list a deliverable declares, however old the file.

        The one reader of the take list, as ``{"name", "start", "end"}``
        entries: each ``shot_metadata`` clip that carries its range (every
        file since 0.11.0), else the legacy ``fbx_takes`` channel an older
        file holds instead.

        Parameters:
            read: Channel key -> that channel's DECODED payload, ``None`` when
                absent -- ``dict.get`` over a decoded carrier, a GLB channel
                reader, a context's ``record``.

        Returns:
            The take entries, empty when none are declared.
        """
        meta = read(cls.SHOTS.key)
        shots = meta.get("shots") if isinstance(meta, Mapping) else None
        ranged = [
            {"name": s["clip"], "start": s["start"], "end": s["end"]}
            for s in (shots if isinstance(shots, list) else ())
            if isinstance(s, Mapping) and s.get("clip") and "start" in s and "end" in s
        ]
        if ranged:
            return ranged
        takes = read(cls.FBX_TAKES.key)
        if not isinstance(takes, list):
            return []
        return [t for t in takes if isinstance(t, dict)]

    # ---------------------------------------------------------------- handoff
    @classmethod
    def handoff_block(
        cls,
        channels: Union[Iterable[str], Mapping[str, Any]],
        source: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        """The standalone-reader contract for an FBX, ready to store.

        *channels* is what the carrier actually holds -- a mapping of channel
        name to stored value, of which only the STRING values are described
        (the carrier also holds keyable float attrs, animated custom
        properties the manifest beside them already describes, and the block
        says every channel value is a JSON string), or an iterable of names.
        Taken from the carrier rather than assumed so the block describes THIS
        file.  Empty dict when there is nothing to describe: a carrier with no
        metadata has no handoff to make.

        *source* is producer identity and provenance (``application``,
        ``version``, ``scene``); empty entries are dropped rather than
        published as nulls.
        """
        if isinstance(channels, Mapping):
            channels = [k for k, v in channels.items() if isinstance(v, str) and v]
        present = sorted(c for c in channels if c != cls.HANDOFF.key)
        if not present:
            return {}
        described = {s.key: s.description for s in cls.deliverable()}
        return {
            "version": cls.HANDOFF.version,
            "source": {k: v for k, v in (source or {}).items() if v} or None,
            "instructions": cls.HANDOFF_INSTRUCTIONS,
            "reads": {
                f"data_export.{name}": described.get(name, "tool-authored channel")
                for name in present
            },
            "rendering": cls.rendering_policy(),
        }

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
                "version_key": s.version_key,
                "deprecated_by": s.deprecated_by,
                "remove_in": s.remove_in,
                "consumers": list(s.consumers),
                "description": s.description,
            }
            for s in cls.all()
        ]


class SceneStoreBase:
    """The storage contract a DCC implements -- strings per scope, nothing more.

    Subclass with the four primitives as classmethods (a store is a namespace
    over the scene, like the DCC ``DataNodes`` classes) and ``name``.  The
    inspection surface (:meth:`dump` / :meth:`format_dump`) is inherited, so
    the "Scene Metadata" viewer and the sidecar snapshot read every mirror the
    same way.  Records never call anything else on a store.
    """

    #: The carrier's name per scope, the grouping key ``dump`` reports under.
    NAMES: Dict[Scope, str] = {
        Scope.PRIVATE: "data_internal",
        Scope.DELIVERABLE: "data_export",
    }

    @classmethod
    def name(cls, scope: Scope) -> str:
        """The carrier name *scope* reports under (:attr:`NAMES`)."""
        return cls.NAMES[Scope(scope)]

    @classmethod
    def read(cls, scope: Scope, key: str) -> Optional[str]:
        """The string channel *key* in *scope*, or ``None`` when the carrier,
        the channel or a value is absent (a cleared channel reads as ``None``;
        a non-string attribute is not a channel)."""
        raise NotImplementedError

    @classmethod
    def write(cls, scope: Scope, key: str, text: Optional[str]) -> Optional[str]:
        """Store *text* on *key*; ``None`` clears without creating a carrier.
        Returns the carrier's name, or ``None`` when a clear had nothing to do."""
        raise NotImplementedError

    @classmethod
    def values(cls, scope: Scope) -> Dict[str, Any]:
        """Every user value the carrier holds, strings and otherwise, in a
        JSON-serializable form -- the inspection read."""
        raise NotImplementedError

    @classmethod
    def keys(cls, scope: Scope) -> List[str]:
        """The names of everything :meth:`values` reports for *scope*."""
        return list(cls.values(scope))

    @classmethod
    def channels(cls, scope: Scope) -> Dict[str, str]:
        """The non-empty STRING channels of *scope* -- what a handoff describes."""
        return {k: v for k, v in cls.values(scope).items() if isinstance(v, str) and v}

    @staticmethod
    def _decode(raw: str) -> Any:
        """*raw* parsed as JSON, or unchanged when it is not (a few channels
        carry plain wire strings)."""
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return raw

    @classmethod
    def dump(cls, decode: bool = True) -> Dict[str, Dict[str, Any]]:
        """Every channel the scene actually carries, grouped by carrier name.

        Discovers whatever is stored rather than the declared keys, so a new
        producer appears with no edit here.  String values that are valid
        JSON are parsed when *decode*; non-string values (a keyed enum, a
        weight float) are returned as-is; cleared channels are skipped; an
        absent carrier contributes an empty dict.
        """
        return {
            cls.name(scope): cls._dumped(cls.values(scope), decode)
            for scope in (Scope.PRIVATE, Scope.DELIVERABLE)
        }

    @classmethod
    def _dumped(cls, values: Mapping[str, Any], decode: bool) -> Dict[str, Any]:
        """*values* by :meth:`dump`'s rules -- absent and cleared channels
        skipped, strings JSON-decoded when *decode*, the rest as-is -- for a
        store that dumps one carrier of several (``dump_export_nodes``)."""
        return {
            key: cls._decode(value) if decode and isinstance(value, str) else value
            for key, value in values.items()
            if value is not None and value != ""
        }

    @classmethod
    def format_dump(cls, decode: bool = True) -> str:
        """Pretty JSON of :meth:`dump`, or ``""`` when nothing is stored."""
        data = cls.dump(decode=decode)
        if not any(data.values()):
            return ""
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)


@dataclass
class ExportContext:
    """What an export DECIDES, handed to every producer as input.

    The exporter's choices used to be patched onto records after the
    producers ran -- the clip mode onto the shot envelope, the clip origin
    onto the visibility tracks -- and were silently overwritten whenever the
    producers ran again.  Here they are inputs: a producer reads them, and
    re-running produces the same record.

    Attributes:
        mode: :attr:`PIPELINE` (an export pipeline, the authority on every
            record), :attr:`HANDOFF` (a bridge that ships the carrier: only
            :attr:`Kind.DERIVED` records refresh) or :attr:`AUTHORING` (a
            tool republishing its own record).
        clip_mode: The run's Animation Clips mode (``full`` / ``shots`` /
            ``both``), declared on the shot record; ``None`` = undeclared.
        clip_span: The first and last frame the exported stack CARRIES, the
            frame every GLB clip is cut against; measured by the pipeline
            from the final keys, ``None`` when nothing measured it.
        source: Producer identity and provenance for the handoff block
            (``application``, ``version``, ``scene``).
        records: Every record produced so far in this assembly, by key --
            how a producer reads another's output (the audio manifest scopes
            its events against the takes the shots producer just built).
    """

    PIPELINE = "pipeline"
    HANDOFF = "handoff"
    AUTHORING = "authoring"

    mode: str = "pipeline"
    clip_mode: Optional[str] = None
    clip_span: Optional[Tuple[float, float]] = None
    source: Dict[str, Any] = field(default_factory=dict)
    records: Dict[str, Optional[Record]] = field(default_factory=dict)

    def record(
        self, spec: Union[RecordSpec, str], store=None, default: Any = None
    ) -> Any:
        """The payload of *spec* as produced in THIS assembly, else -- when a
        *store* is given -- as currently stored, else *default*.

        The one read a consumer of another producer's record needs: fresh
        when both run in one assembly, stored when a tool republishes its own
        record alone at authoring time.
        """
        spec = SceneRecords.resolve(spec)
        if spec.key in self.records:
            produced = self.records[spec.key]
            return produced.payload if produced is not None else default
        if spec.deprecated_by in self.records:
            # A legacy record lives only beside its successor: its successor
            # was produced here without it, so the stored copy is stale.
            return default
        if store is not None:
            return spec.load(store, default)
        return default

    def refreshes(self, spec: RecordSpec) -> bool:
        """Whether this context's mode refreshes *spec*."""
        if self.mode == self.HANDOFF:
            return spec.kind is Kind.DERIVED
        return True


Producer = Callable[[ExportContext], Union[Record, Iterable[Record], None]]


class ExportSnapshot:
    """The records one export ships, assembled once and committed once.

    :meth:`assemble` runs the producers; :meth:`commit` writes the result.
    Between the two the snapshot is a plain object the exporter can read
    (:meth:`channels`, :meth:`summary`), and after the commit it is still
    the truth about what shipped -- no consumer re-reads the carrier.
    """

    def __init__(self, ctx: ExportContext) -> None:
        self.ctx = ctx
        #: key -> the record produced (``None`` = the producer ran and had
        #: nothing to say, so the stored record is cleared on commit).
        self.produced: Dict[str, Optional[Record]] = {}
        #: Keys whose producer raised: logged, and left untouched on commit
        #: rather than cleared -- a failing subsystem must not erase what the
        #: scene last published.
        self.failed: Set[str] = set()
        #: What :meth:`commit` wrote, key -> text (``None`` = cleared).
        self.written: Dict[str, Optional[str]] = {}

    # ---------------------------------------------------------------- assemble
    @classmethod
    def assemble(
        cls,
        producers: Mapping[Union[RecordSpec, str], Producer],
        ctx: Optional[ExportContext] = None,
        only: Optional[Iterable[Union[RecordSpec, str]]] = None,
    ) -> "ExportSnapshot":
        """Run *producers* in dependency order and collect their records.

        Parameters:
            producers: Record spec (or key) -> callable taking the context
                and returning a :class:`Record`, several, or ``None``.
            ctx: The :class:`ExportContext`; a pipeline context by default.
            only: Narrow the run to these records (specs or keys).  The mode's
                own filter still applies (a hand-off never refreshes an
                authored record, even when named).

        Each producer is isolated: one that raises, or returns something other
        than records, is logged and recorded in :attr:`failed`, and every other
        still runs.  A producer may return records for keys other than its own
        (the shots producer also writes the legacy take list); those ride the
        snapshot like any other.  *ctx* holds ONE assembly's records: a reused
        context starts empty, so no producer reads what an earlier assembly
        produced.

        Returns:
            The snapshot, not yet committed.

        Raises:
            KeyError, ValueError: *producers* names an undeclared, private or
                handoff record (:meth:`SceneRecords.check_producers`).
        """
        ctx = ctx or ExportContext()
        ctx.records.clear()
        snapshot = cls(ctx)
        table = dict(zip(SceneRecords.check_producers(producers), producers.values()))
        wanted = None if only is None else {SceneRecords.resolve(k).key for k in only}
        selected = [
            spec
            for spec in table
            if (wanted is None or spec.key in wanted) and ctx.refreshes(spec)
        ]
        for spec in SceneRecords.ordered(selected):
            try:
                result = table[spec](ctx)
                records = [
                    r
                    for r in (
                        [result]
                        if isinstance(result, Record) or result is None
                        else list(result)
                    )
                    if r is not None
                ]
                if not all(isinstance(r, Record) for r in records):
                    raise TypeError(
                        f"The {spec.key!r} producer returned a bare payload; it "
                        "must return Record(s) or None (spec.make(payload))."
                    )
            except Exception:  # one subsystem's failure must not block the others
                logger.warning(
                    "Scene record %r was not produced.", spec.key, exc_info=True
                )
                snapshot.failed.add(spec.key)
                continue
            own = next((r for r in records if r.spec.key == spec.key), None)
            # The producer's own record first, then any it returned for other
            # keys (the shots producer also writes the legacy take list), so
            # the snapshot reads in producer order.
            snapshot.produced[spec.key] = own
            # Kept even as None: a reader later in this assembly must see
            # "cleared", not fall back to the stored copy the commit is about
            # to clear.
            ctx.records[spec.key] = own
            for record in records:
                if record is not own:
                    snapshot.produced[record.spec.key] = record
                    ctx.records[record.spec.key] = record
        return snapshot

    @classmethod
    def publish(
        cls,
        store,
        records: Mapping[Union[RecordSpec, str], Any],
        ctx: Optional[ExportContext] = None,
    ) -> "ExportSnapshot":
        """Commit *records* that are already in hand -- the AUTHORING-time
        publish a tool makes of its own record, in one call.

        *records* maps a spec (or key) to a :class:`Record`, a payload (made
        into one), or a falsy value (the record is cleared).  The handoff is
        restamped like any commit, so a tool republishing at authoring time
        never leaves the carrier describing what it no longer holds.

        Returns:
            The committed snapshot.

        Raises:
            ValueError: A PRIVATE record.  The snapshot commits the deliverable
                carrier, and a private record can share a deliverable one's
                key (the emissive registry and manifest do), so clearing it
                here would clear the wrong carrier; use
                :meth:`RecordSpec.save`.
        """
        snapshot = cls(ctx or ExportContext(mode=ExportContext.AUTHORING))
        for item, value in records.items():
            spec = SceneRecords.resolve(item)
            if spec.scope is not Scope.DELIVERABLE:
                raise ValueError(
                    f"{spec.key!r} is a private record; commit it with "
                    "spec.save(store, payload), not a deliverable snapshot."
                )
            if isinstance(value, Record):
                record: Optional[Record] = value
            elif value:
                record = spec.make(value)
            else:
                record = None
            snapshot.produced[spec.key] = record
        snapshot.commit(store)
        return snapshot

    # ------------------------------------------------------------------ commit
    def commit(self, store) -> Dict[str, Optional[str]]:
        """Write every produced record to *store* in one pass, then stamp the
        handoff block from what the deliverable carrier now holds.

        A record produced as ``None`` is cleared (its producer ran and had
        nothing to say); a failed producer's record is left as stored.  The
        handoff is stamped only when the carrier holds another channel --
        never manufactured onto an empty carrier -- and cleared when nothing
        else remains, so a lone self-referential channel cannot survive.
        Never creates a carrier for a clear, and never raises for the handoff
        (self-description must not be able to fail an export).  Each write is
        isolated like each producer: one the store refuses (a locked
        attribute) is logged and added to :attr:`failed`, and the rest are
        still written.

        Returns:
            :attr:`written` -- key -> the text stored (``None`` = cleared).
        """

        def write(spec: RecordSpec, text: Optional[str]) -> None:
            try:
                spec.write_text(store, text)
            except Exception:  # one record's write must not cost the others
                logger.warning(
                    "Scene record %r was not written; left as stored.",
                    spec.key,
                    exc_info=True,
                )
                self.failed.add(spec.key)
                return
            self.written[spec.key] = text

        for key, record in self.produced.items():
            if key in self.failed:
                continue
            if record is None:
                write(SceneRecords.resolve(key), None)
            else:
                write(record.spec, record.text)
        # A deprecated record lives only beside its successor: when the
        # successor's producer ran and did not also produce the legacy record
        # (the shots producer returns nothing for an empty store), the legacy
        # copy is cleared too -- deleting the last shot must not leave the old
        # take list riding into the next export.
        for spec in SceneRecords.all():
            if (
                spec.deprecated_by in self.produced
                and spec.key not in self.produced
                and spec.key not in self.failed
            ):
                write(spec, None)
        try:
            handoff = SceneRecords.HANDOFF
            present = store.channels(Scope.DELIVERABLE)
            block = SceneRecords.handoff_block(present, source=self.ctx.source)
            if block:
                text = handoff.make(block).text
                handoff.write_text(store, text)
                self.written[handoff.key] = text
            elif handoff.key in present:
                handoff.clear(store)
                self.written[handoff.key] = None
        except Exception:  # noqa: BLE001 - a missing description never costs the export
            logger.debug("Handoff record not stamped.", exc_info=True)
        return self.written

    # ------------------------------------------------------------------- reads
    @property
    def records(self) -> Dict[str, Record]:
        """The produced records by key (no ``None`` entries)."""
        return {k: r for k, r in self.produced.items() if r is not None}

    def record(self, spec: Union[RecordSpec, str], default: Any = None) -> Any:
        """The payload produced for *spec*, or *default*."""
        found = self.produced.get(SceneRecords.resolve(spec).key)
        return found.payload if found is not None else default

    def channels(self, scope: Scope = Scope.DELIVERABLE) -> Dict[str, Any]:
        """Produced payloads of *scope*, by key -- the sidecar's snapshot."""
        return {k: r.payload for k, r in self.records.items() if r.spec.scope is scope}

    def summary(self) -> str:
        """One line for the export log: each record that shipped and how
        much it holds, then any producer that failed.  A record produced as
        ``None`` shipped nothing and is left out -- listing every empty
        producer buried the records that did ship."""
        parts = []
        for key, record in self.records.items():
            if key in self.failed:  # produced, but the write was refused
                continue
            payload = record.payload
            count = None
            if isinstance(payload, list):
                count = len(payload)
            elif isinstance(payload, Mapping):
                for value in payload.values():
                    if isinstance(value, list):
                        count = len(value)
                        break
            parts.append(
                f"{key} ({count} entr{'y' if count == 1 else 'ies'})"
                if count is not None
                else key
            )
        for key in sorted(self.failed):
            parts.append(f"{key} (FAILED, left as stored)")
        return ", ".join(parts)
