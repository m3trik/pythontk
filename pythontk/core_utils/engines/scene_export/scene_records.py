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

import json
import logging
from dataclasses import dataclass
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
    merge: Merge = Merge.DERIVE
    merge_key: Optional[str] = None
    respell: bool = True
    portable: bool = False
    section: Optional[str] = None
    paths: Union[bool, Tuple[str, ...]] = False

    @property
    def path_keys(self) -> Optional[Tuple[str, ...]]:
        """The payload keys whose values are paths (:attr:`paths`): the named
        ones, or ``None`` -- every value -- for a whole path mapping."""
        return self.paths if isinstance(self.paths, tuple) else None

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
            "intensity, scaleOffset, and the object's scene hierarchy (which "
            "tells apart objects that share a name)"
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
        merge=Merge.CODEC,
        portable=True,
        section="shots",
    )
    KEY_STASH = RecordSpec(
        "key_stash",
        Scope.PRIVATE,
        1,
        owner="Key Stash",
        description="the clip manifest of parked keys",
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
        description="LIFO stack of bake-session restore manifests",
        envelope=False,
        merge=Merge.UNION,
        merge_key="id",
    )
    HIERARCHY_BASELINE = RecordSpec(
        "hierarchy_baseline",
        Scope.PRIVATE,
        1,
        owner="Hierarchy check",
        description="the export hierarchy baseline (a HierarchyBaseline record)",
        envelope=False,
        merge=Merge.OWN,
        paths=("scene",),
    )
    EMISSIVE_REGISTRY = RecordSpec(
        "emissive_groups",
        Scope.PRIVATE,
        1,
        owner="Emissive Groups",
        description="the group registry: slots, defaults, encoding",
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
            "the viewport material bindings a preview drives, so a suspend "
            "and rebind round-trips"
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
    #: scene was unsaved: the scene's own only until its first save (a Save
    #: As copy carries the same ``""``), and it never crosses to another.
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
    )

    #: The domain codecs: record key -> (module, class) whose classmethod
    #: ``merge_record(own, other, ctx)`` is a :attr:`Merge.CODEC` record's
    #: merge.  Resolved lazily, like a DCC's producer table -- the codec lives
    #: with the model it merges, and this module stays the lighter import.
    CODECS: Dict[str, Tuple[str, str]] = {
        "shot_store": (
            "pythontk.core_utils.engines.shots.shot_transfer",
            "ShotTransfer",
        ),
        "key_stash": (
            "pythontk.core_utils.engines.key_stash.key_stash_model",
            "KeyStash",
        ),
        "emissive_groups": (
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
    def portable(cls) -> List[RecordSpec]:
        """The records that cross a DCC hand-off, in declaration order."""
        return [s for s in cls.all() if s.portable]

    @staticmethod
    def rendering_policy(
        overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """What a deliverable claims about how it should be lit.

        The GLB's handoff section and the FBX's handoff record publish the
        same policy, and it is the reference viewer's contract
        (``MeshConvert.RENDERING_POLICY``, pinned against the viewer's own
        literals by ``test_preview_server``), so it is read from there rather
        than declared twice -- *overrides* included: an export's choices
        (``ExportContext.rendering``) merge by ``MeshConvert.rendering_policy``
        on both carriers.  Imported lazily: this module is the lighter
        dependency and must import first.
        """
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        return MeshConvert.rendering_policy(overrides)

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
    def with_paths(cls) -> List[RecordSpec]:
        """The records whose values are project-relative paths
        (:attr:`RecordSpec.paths`), in declaration order."""
        return [s for s in cls.all() if s.paths]

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

        changed = 0
        for spec in cls.with_paths():
            payload = spec.load(store)
            if not payload:
                continue
            moved = cls.map_paths(
                payload,
                lambda v: FileUtils.rebase_portable_path(v, old_base, new_base),
                spec.path_keys,
            )
            diff = sum(1 for k in payload if moved.get(k) != payload.get(k))
            if diff:
                spec.save(store, moved)
                changed += diff
        return changed

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
        rendering: Optional[Mapping[str, Mapping[str, Any]]] = None,
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

        *rendering* is the export's choices over the lighting recipe
        (:meth:`rendering_policy`); ``None`` publishes the policy as it stands.
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
            "rendering": cls.rendering_policy(rendering),
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
                "merge": s.merge.value,
                "portable": s.portable,
                "description": s.description,
            }
            for s in cls.all()
        ]
