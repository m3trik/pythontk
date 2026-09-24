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

import contextlib
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
            (:class:`TransferContext`).
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
    paths: bool = False

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
    def map_paths(payload: Any, spell: Callable[[str], str]) -> Any:
        """*payload* (a :attr:`RecordSpec.paths` mapping) with every string
        value put through *spell*; anything else -- a non-mapping, a
        non-string value -- as it is.  A new mapping, never a mutation."""
        if not isinstance(payload, Mapping):
            return payload
        return {
            key: spell(value) if isinstance(value, str) and value else value
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


class SceneStoreBase:
    """The storage contract a DCC implements -- strings per scope, nothing more.

    Subclass with the four primitives as classmethods (a store is a namespace
    over the scene, like the DCC ``DataNodes`` classes) and ``name``.  The
    inspection surface (:meth:`dump` / :meth:`format_dump`) is inherited, so
    the "Scene Metadata" viewer and the sidecar snapshot read every mirror the
    same way.  Records never call anything else on a store.

    So are the crossings -- a hand-off's sidecar sections
    (:meth:`transfer_sections` / :meth:`receive_sections`) and another scene's
    carriers merged or discarded (:meth:`merge_carriers`), each record by its
    rule (:class:`RecordTransfer`).  A DCC supplies its record owners
    (:attr:`OWNERS`) and the few carrier hooks below them; the orchestration
    is written once, here.
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
    def project_root(cls) -> Optional[str]:
        """This scene's own project root -- what its :attr:`RecordSpec.paths`
        records are spelled from (``FileUtils.portable_path``): the project
        the scene FILE lives in (:meth:`project_root_of`), never the one a
        session has set.  ``None`` while the scene is unsaved, and in this
        base: a DCC store answers for its open file."""
        return None

    @staticmethod
    def project_root_of(scene_path: Optional[str]) -> Optional[str]:
        """The project *scene_path* lives in: its nearest marked ancestor, else
        its own folder (``Workspace.for_path``).  ``None`` without a path, or
        when its folder does not exist."""
        if not scene_path:
            return None
        from pythontk.file_utils.workspace import Workspace

        workspace = Workspace.for_path(scene_path)
        return workspace.root if workspace is not None else None

    @classmethod
    def rebase_paths(cls, old_base: Optional[str], new_base: Optional[str]) -> int:
        """Re-spell this scene's path records from *old_base* to *new_base*
        (:meth:`SceneRecords.rebase_paths`) -- what a DCC's save hook calls
        when a save moves the scene into another project.  Returns how many
        entries changed."""
        changed = SceneRecords.rebase_paths(cls, old_base, new_base)
        if changed:
            logger.info(
                "%d scene-record path(s) re-spelled for the project %s.",
                changed,
                new_base or "(none)",
            )
        return changed

    @classmethod
    def format_dump(cls, decode: bool = True) -> str:
        """Pretty JSON of :meth:`dump`, or ``""`` when nothing is stored."""
        data = cls.dump(decode=decode)
        if not any(data.values()):
            return ""
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)

    # --------------------------------------------------------------- crossings
    #: The DCC owners of records that keep state BESIDE the record -- a
    #: membership set, a parked curve, an in-memory store to reload -- as
    #: record key -> ``(module, class)``, resolved lazily (:meth:`owners`).  An
    #: owner defines any of the hooks the crossings look for:
    #:
    #: - ``transfer_out(ctx)`` -- the record's hand-off payload (without it,
    #:   the stored record crosses as data);
    #: - ``transfer_in(payload, ctx)`` -- land a received payload (without it,
    #:   the payload merges by the record's rule);
    #: - ``merge_carrier(carriers, other, ctx)`` -- another scene's carriers
    #:   were merged into this scene's (an imported reference): carry over
    #:   what sits beside the records and refresh whatever caches them;
    #: - ``discard_carrier(carriers, other, ctx)`` -- they were discarded
    #:   instead: remove what sat beside the records nobody keeps, and drop
    #:   whatever caches them (a discarded carrier the import adopted was
    #:   the scene's own until now);
    #: - ``flush_pending()`` -- store what the owner holds but has not
    #:   written yet (:meth:`flush_owners`, before any crossing reads the
    #:   records).
    #:
    #: *other* is ``{RecordSpec: payload}`` (:meth:`RecordTransfer.payloads`).
    #: A record needing none of this has no row: declaring it
    #: (:class:`SceneRecords`) is all it takes to cross -- a record with a
    #: section of its own through its codec's ``section_out`` /
    #: ``section_in`` (the section's wire shape is the codec's, never the
    #: bare record's), any other as the stored record, respelled.
    OWNERS: Dict[str, Tuple[str, str]] = {}

    @classmethod
    def owners(cls) -> Dict[str, Any]:
        """:attr:`OWNERS` resolved to classes.  One that does not import is
        left out, and its record then crosses as plain data."""
        resolved: Dict[str, Any] = {}
        for key, (module, owner) in cls.OWNERS.items():
            try:
                resolved[key] = SceneRecords.resolve_class(module, owner)
            except Exception:  # noqa: BLE001 - a missing owner never costs a crossing
                logger.debug("Record owner %s.%s unavailable.", module, owner)
        return resolved

    @classmethod
    def transfer_sections(
        cls, spell: Optional[Callable[[str], str]] = None, objects=None
    ) -> Dict[str, Any]:
        """What a hand-off producer adds to its sidecar: every portable record
        this scene holds (:meth:`RecordTransfer.sections`), names spelled by
        *spell* -- the carrier's spelling, the one the sidecar's other
        sections use -- and scoped to *objects* where an owner scopes
        (memberships, ledger claims).  ``{}`` when there is nothing to send; a
        record that could not be sent is logged, never fatal."""
        ctx = TransferContext(
            rename=spell,
            objects=None if objects is None else [str(o) for o in objects],
            path_base=cls.project_root(),
        )
        cls.flush_owners()
        sections = RecordTransfer.sections(cls, ctx, cls.owners())
        cls._log_notes(ctx)
        return sections

    @classmethod
    def receive_sections(
        cls,
        manifest: Optional[Mapping[str, Any]],
        resolve: Optional[Callable[[str], Optional[str]]] = None,
        source: str = "",
        **adapters: Any,
    ) -> "TransferContext":
        """Land a hand-off sidecar's records in this scene
        (:meth:`RecordTransfer.receive`): each to its owner's ``transfer_in``,
        else merged by its rule.  *resolve* maps the carrier's spelling to the
        imported object; *adapters* are what an owner asks for by name
        (``converted``, ``frame_offset``).  Best-effort per record: the
        returned context's notes are the report, each logged as well."""
        ctx = TransferContext(
            rename=resolve,
            source=source,
            adapters=adapters,
            path_base=cls.project_root(),
        )
        cls.flush_owners()
        RecordTransfer.receive(manifest or {}, cls, ctx, cls.owners())
        cls._log_notes(ctx)
        return ctx

    @classmethod
    def flush_owners(cls) -> None:
        """Store what every owner (:attr:`OWNERS`) holds but has not written
        yet (its ``flush_pending``) -- a crossing reads the records, and a
        store that writes on idle can hold a change its record lacks.

        The hand-off routes call it themselves.  An importer calls it BEFORE
        the other scene's nodes land (:meth:`merge_carriers` cannot): a scene
        with no carrier of its own adopts the import's, and flushed after,
        the scene's pending state would be written into that carrier -- over
        the records a merge was to keep, or into the one a discard removes.
        """
        for key, owner in cls.owners().items():
            flush = getattr(owner, "flush_pending", None)
            if flush is None:
                continue
            try:
                flush()
            except Exception:  # noqa: BLE001 - the crossing reads what is stored
                logger.warning(
                    "%s: pending changes were not stored.", key, exc_info=True
                )

    @classmethod
    def merge_plan(cls, carriers: Mapping[Any, Any]) -> "RecordTransfer":
        """*carriers* (another scene's, by scope) against this scene's own:
        what a merge would bring in.  ``is_empty`` means nothing a merge
        keeps, so there is no question to ask before the carriers go."""
        return RecordTransfer.between(
            cls,
            {
                scope: cls._carrier_values(carrier)
                for scope, carrier in cls._foreign_carriers(carriers).items()
            },
        )

    @classmethod
    def merge_carriers(
        cls,
        carriers: Mapping[Any, Any],
        rename=None,
        source: str = "",
        adapters: Optional[Mapping[str, Any]] = None,
        source_path_base: Optional[str] = None,
    ) -> "TransferContext":
        """Merge another scene's *carriers* -- an imported reference's, made
        local -- into this scene's, then remove them.

        Each record merges by its declared rule (:meth:`RecordTransfer.apply`);
        what a carrier holds beside its records (a keyed attribute) moves to
        this scene's carrier (:meth:`_carry_attributes`); every owner
        (:attr:`OWNERS`) carries over what sits beside its record.  Then the
        carriers go, so nothing is left holding records no tool reads, and
        the deliverables they held are produced again from the merged scene
        (:meth:`_rederive`).  A carrier that IS this scene's own (the import
        adopted it: this scene had none) is left alone.  The importer calls
        :meth:`flush_owners` before the other scene's nodes land.

        Parameters:
            carriers: ``{scope: carrier}`` of the other scene's carriers.
            rename: The other scene's names -> this scene's
                (:attr:`TransferContext.rename`); ``None`` when nothing moved.
            source: The other scene's name, for the notes.
            adapters: DCC facts an owner's hook asks for by name
                (:attr:`TransferContext.adapters`) -- what the crossing knows
                and the records do not, such as which datablocks the other
                scene brought.
            source_path_base: The other scene's own project root -- its
                :attr:`RecordSpec.paths` records are spelled from it, and
                arrive re-spelled from this scene's (:meth:`project_root`).

        Returns:
            TransferContext: its ``notes`` are what the merge changed or could
            not keep -- the report the caller shows (each is logged too).
        """
        return cls._settle_carriers(
            carriers,
            rename,
            source,
            keep=True,
            adapters=adapters,
            source_path_base=source_path_base,
        )

    @classmethod
    def discard_carriers(
        cls,
        carriers: Mapping[Any, Any],
        rename=None,
        source: str = "",
        adapters: Optional[Mapping[str, Any]] = None,
        source_path_base: Optional[str] = None,
    ) -> "TransferContext":
        """Remove another scene's *carriers* without merging their records --
        the answer "don't merge" to :meth:`merge_plan`'s question.  Owners
        remove what sat beside the dropped records, and the deliverables the
        carriers held are produced again, as a merge does.  Unlike a merge,
        a carrier the import adopted as this scene's own goes too: it was
        adopted only because this scene had none.  As for a merge, the
        importer calls :meth:`flush_owners` before the other scene's nodes
        land -- an owner reloads what it holds from the records left."""
        return cls._settle_carriers(
            carriers,
            rename,
            source,
            keep=False,
            adapters=adapters,
            source_path_base=source_path_base,
        )

    @classmethod
    def _settle_carriers(
        cls,
        carriers: Mapping[Any, Any],
        rename,
        source: str,
        keep: bool,
        adapters: Optional[Mapping[str, Any]] = None,
        source_path_base: Optional[str] = None,
    ) -> "TransferContext":
        """:meth:`merge_carriers` (*keep*) / :meth:`discard_carriers`."""
        ctx = TransferContext(
            rename=rename,
            source=source,
            adapters=dict(adapters or {}),
            path_base=cls.project_root(),
            source_path_base=source_path_base,
        )
        live = cls._foreign_carriers(carriers) if keep else cls._live_carriers(carriers)
        if not live:
            return ctx
        plan = RecordTransfer.between(
            cls,
            {scope: cls._carrier_values(carrier) for scope, carrier in live.items()},
        )
        other = plan.payloads(ctx)
        hook_name = "merge_carrier" if keep else "discard_carrier"
        with cls._crossing():
            if keep:
                plan.apply(cls, ctx)
                for scope, carrier in live.items():
                    cls._carry_attributes(carrier, scope, ctx)
            elif not plan.is_empty:
                ctx.note(
                    f"{source or 'The other scene'}: not merged -- "
                    + "; ".join(plan.summary())
                    + "."
                )
            for key, owner in cls.owners().items():
                hook = getattr(owner, hook_name, None)
                if hook is None:
                    continue
                try:
                    hook(live, other, ctx)
                except Exception as error:  # noqa: BLE001 - one owner never costs the rest
                    logger.warning("%s: %s failed.", key, hook_name, exc_info=True)
                    ctx.note(f"{key}: its scene state was not settled ({error}).")
            for carrier in live.values():
                cls._delete_carrier(carrier)
            # From what the scene keeps: while the other carriers stood, one a
            # discard adopted as the scene's own still held the dropped records.
            if plan.rederive:
                cls._rederive(plan.rederive, ctx)
        cls._log_notes(ctx)
        return ctx

    @staticmethod
    def _log_notes(ctx: "TransferContext") -> None:
        for note in ctx.notes:
            logger.warning(note)

    # -- the DCC's half of a crossing: override what its carriers need --------
    @classmethod
    def _live_carriers(cls, carriers: Mapping[Any, Any]) -> Dict[Scope, Any]:
        """*carriers* that still exist, by scope (default: every one given)."""
        return {Scope(s): c for s, c in (carriers or {}).items() if c is not None}

    @classmethod
    def _foreign_carriers(cls, carriers: Mapping[Any, Any]) -> Dict[Scope, Any]:
        """:meth:`_live_carriers` less any that is this scene's own carrier of
        its scope -- one an import adopted because the scene had none
        (default: none is)."""
        return cls._live_carriers(carriers)

    @classmethod
    def _carrier_values(cls, carrier: Any) -> Dict[str, Any]:
        """Every value another scene's *carrier* holds, by :meth:`values`'
        rules."""
        raise NotImplementedError

    @classmethod
    def _carry_attributes(cls, carrier: Any, scope: Scope, ctx) -> None:
        """Move what *carrier* holds beside its records -- a keyed attribute
        and its curve -- to this scene's carrier of *scope* (default:
        nothing to move)."""

    @classmethod
    def _rederive(cls, specs: List[RecordSpec], ctx) -> None:
        """Produce the deliverables *specs* again from the merged scene
        (default: a store without producers has none to run)."""

    @classmethod
    def _delete_carrier(cls, carrier: Any) -> None:
        """Remove another scene's settled *carrier*."""
        raise NotImplementedError

    @classmethod
    def _crossing(cls):
        """The context a settle runs in -- the DCC's one undo step (default:
        none)."""
        return contextlib.nullcontext()


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
        rendering: The export's choices over the lighting recipe the handoff
            block publishes (``ExportRun.rendering``: ``{section: {field:
            value}}`` over ``MeshConvert.RENDERING_POLICY``); empty = the
            policy as it stands.
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
    rendering: Dict[str, Any] = field(default_factory=dict)
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
            block = SceneRecords.handoff_block(
                present, source=self.ctx.source, rendering=self.ctx.rendering
            )
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


@dataclass
class TransferContext:
    """What a record crossing between scenes needs from the DCC, and what the
    crossing learns on the way.

    One context for every route: a referenced module's carrier merging into
    its host (the DCC maps the module's names, namespace stripped, to where
    the import put them), a hand-off leaving (names spelled as the carrier
    writes them) or arriving (the carrier's spelling resolved to the imported
    nodes).

    Attributes:
        rename: A name as the OTHER side spells it -> as this side does, or
            ``None`` / the name unchanged when it has no counterpart.  The DCC
            owns its spellings -- DAG paths, plugs, ``object|path|index`` curve
            keys -- so no record respells on its own: a codec record's codec
            puts the names it holds through it (``respell_record``), and every
            string of any other :attr:`RecordSpec.respell` payload, mapping
            keys included, goes through it.
        source: The other scene's name, for the notes (a reference's
            namespace, ``"MOD"``).
        objects: A hand-off's exported set in this scene's naming, when the
            route has one (a DCC owner scopes what it sends to it).
        adapters: DCC callables an owner asks for by name -- the hooks a
            codec's DCC half needs that no other record does (the shot store's
            curve resolution, an importer's frame offset).
        remaps: What an earlier record's merge renumbered, by kind: the shot
            store publishes ``{"shot_id": {old: new}}``, and the key stash,
            declared after it, follows its clips' source shots through it.
        notes: What the crossing changed or could not keep, one sentence
            each -- the report the caller shows, so nothing is renamed or
            dropped silently.
        path_base: This scene's own project root: a
            :attr:`RecordSpec.paths` record leaving is resolved absolute from
            it (the other scene lives elsewhere), and one arriving is spelled
            from it.  ``None`` (an unsaved scene) keeps arrivals absolute.
        source_path_base: The other scene's own project root, when its paths
            arrive spelled from it (a referenced module's carrier); a
            hand-off's arrive absolute and need none.
    """

    rename: Optional[Callable[[str], Optional[str]]] = None
    source: str = ""
    objects: Optional[List[str]] = None
    adapters: Dict[str, Any] = field(default_factory=dict)
    remaps: Dict[str, Dict[Any, Any]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    path_base: Optional[str] = None
    source_path_base: Optional[str] = None

    def note(self, text: str) -> None:
        """Record one sentence for the report."""
        self.notes.append(text)

    def adapter(self, name: str, default: Any = None) -> Any:
        """The DCC adapter *name*, else *default*."""
        return self.adapters.get(name, default)

    def spell(self, name: str) -> str:
        """*name* as this side spells it -- :attr:`rename`'s answer, else
        *name* unchanged (no counterpart, or no rename at all)."""
        spelled = self.rename(name) if self.rename is not None else None
        return spelled if isinstance(spelled, str) and spelled else name

    def respell(self, value: Any) -> Any:
        """*value* with every string -- mapping keys included -- put through
        :meth:`spell`; containers are rebuilt, never mutated.  For a payload
        whose strings are all names; a codec that holds other strings too
        picks its names itself (``respell_record``)."""
        if self.rename is None:
            return copy.deepcopy(value)

        def walk(item: Any) -> Any:
            if isinstance(item, str):
                return self.spell(item)
            if isinstance(item, list):
                return [walk(v) for v in item]
            if isinstance(item, dict):
                return {walk(k): walk(v) for k, v in item.items()}
            return item

        return walk(value)


class RecordTransfer:
    """Another scene's records meeting this scene's -- one engine for every
    route a record crosses by, driven by the declarations.

    **Merging another scene's carrier** (a module's, once its reference is
    imported): build it from the two carriers' string channels per scope
    (:meth:`between`), ask the one question worth asking a user first -- does
    the other scene bring anything a merge would keep (:attr:`is_empty`,
    :meth:`summary`) -- then :meth:`apply` merges each record by its
    :class:`Merge` rule.  A carrier holding only deliverables (the merged
    scene re-derives them, :attr:`rederive`), its own baseline, or nothing,
    is empty: it can simply go, with no question.

    **A DCC hand-off**: :meth:`sections` is what the producer adds to its
    sidecar -- every :attr:`RecordSpec.portable` record the scene holds --
    and :meth:`receive` lands a sidecar's records in the consumer's scene.
    Each record's DCC owner may take over its half (``transfer_out(ctx)`` /
    ``transfer_in(payload, ctx)``) when it keeps state beside the record the
    generic path cannot see; otherwise the stored record crosses as data,
    respelled, and merges by its rule.

    Pure: the DCC reads its carriers, supplies the :class:`TransferContext`,
    writes through its store, re-derives :attr:`rederive` with its
    producers, and carries over what lives on its carrier beside the records.
    """

    #: The hand-off sidecar section a portable record rides when it names no
    #: section of its own (``HandoffManifest.RECORDS``).
    RECORDS_SECTION = "records"

    def __init__(
        self,
        own: Mapping[Any, Mapping[str, Any]],
        other: Mapping[Any, Mapping[str, Any]],
    ) -> None:
        def strings(values: Mapping[str, Any]) -> Dict[str, str]:
            return {k: v for k, v in values.items() if isinstance(v, str) and v}

        self.own = {Scope(s): strings(v) for s, v in own.items()}
        self.other = {Scope(s): strings(v) for s, v in other.items()}

    @classmethod
    def between(cls, store, other: Mapping[Any, Mapping[str, Any]]) -> "RecordTransfer":
        """*store*'s channels, per scope, against *other*'s (``{scope: values}``)."""
        return cls({scope: store.channels(scope) for scope in Scope}, other)

    # ------------------------------------------------------------- the plan
    def _entries(self) -> List[Tuple[Scope, str, Optional[RecordSpec]]]:
        """The other scene's channels: declared records in declaration order
        (a codec may read what an earlier one remapped), then the undeclared
        ones by name."""
        declared = [
            (spec.scope, spec.key, spec)
            for spec in SceneRecords.all()
            if spec.key in self.other.get(spec.scope, {})
        ]
        known = {(scope, key) for scope, key, _ in declared}
        undeclared = sorted(
            (scope, key, None)
            for scope, values in self.other.items()
            for key in values
            if (scope, key) not in known
        )
        return declared + undeclared

    @property
    def incoming(self) -> List[Tuple[Scope, str]]:
        """What a merge would bring in: every :attr:`Merge.UNION` /
        :attr:`Merge.CODEC` record the other scene holds, and every channel no
        record declares (adopted when this scene has none of its own) --
        each one that holds something (:meth:`_holds_nothing`)."""
        return [
            (scope, key)
            for scope, key, spec in self._entries()
            if (spec is None or spec.merge in (Merge.UNION, Merge.CODEC))
            and not self._holds_nothing(self.other[scope][key])
        ]

    @staticmethod
    def _holds_nothing(text: str) -> bool:
        """Whether a stored payload holds no entry: an empty list or mapping,
        or a document whose every list and mapping is empty -- settings
        beside them are no entry (a key stash with no clip:
        ``{"schema": 1, "clips": [], "next_id": 1}``).  A payload that is not
        JSON, or a mapping of settings alone, holds something."""
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            return False
        if isinstance(payload, list):
            return not payload
        if not isinstance(payload, Mapping):
            return False
        containers = [v for v in payload.values() if isinstance(v, (list, Mapping))]
        return not any(containers) if containers else not payload

    @property
    def rederive(self) -> List[RecordSpec]:
        """The deliverables to produce again once the merge is applied: every
        :attr:`Merge.DERIVE` record the other scene held."""
        return [
            spec
            for _scope, _key, spec in self._entries()
            if spec is not None and spec.merge is Merge.DERIVE
        ]

    @property
    def is_empty(self) -> bool:
        """Nothing to decide: the other scene brings no record a merge keeps."""
        return not self.incoming

    def summary(self) -> List[str]:
        """One line per record the merge would bring in, for the prompt --
        what it is and how many entries the other scene holds.  A record
        whose counted list is empty holds something else (a shot store with
        markers alone), so it is named without a count."""
        lines = []
        for scope, key in self.incoming:
            spec = SceneRecords.by_key(key, scope)
            label = spec.owner if spec is not None else key
            count = self._count(self.other[scope][key])
            lines.append(
                f"{label}: {count} entr{'y' if count == 1 else 'ies'}"
                if count
                else label
            )
        return lines

    @staticmethod
    def _count(text: str) -> Optional[int]:
        """How many entries a stored payload holds: a list's length; a keyed
        mapping's (every value a container, or none is); a document's first
        list or mapping (``{"schema": 1, "groups": {...}}`` counts groups)."""
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            return None
        if isinstance(payload, list):
            return len(payload)
        if not isinstance(payload, Mapping):
            return None
        containers = [v for v in payload.values() if isinstance(v, (list, Mapping))]
        if not containers or len(containers) == len(payload):
            return len(payload)
        return len(containers[0])

    def payloads(self, ctx: Optional[TransferContext] = None) -> Dict[RecordSpec, Any]:
        """The other scene's declared records, decoded -- and respelled through
        *ctx* (:meth:`respell`) when given -- by declaration: what a DCC owner
        reads to carry over, or clean up, what sits beside them.  A record
        that does not decode is left out.  Keyed by spec, not key: a private
        record and a deliverable may share a key."""
        out: Dict[RecordSpec, Any] = {}
        for scope, key, spec in self._entries():
            if spec is None:
                continue
            payload = spec.decode(self.other[scope][key])
            if payload is None:
                continue
            out[spec] = payload if ctx is None else self.respell(spec, payload, ctx)
        return out

    # --------------------------------------------------------------- apply
    def apply(self, store, ctx: Optional[TransferContext] = None) -> TransferContext:
        """Merge the other scene's records into *store*, each by its rule.

        The scene's own copy is read from *store* at this moment, not from
        when the merge was planned.  A record the other scene brings and this
        scene lacks is adopted; one both hold is combined by its rule.
        :attr:`Merge.OWN` and :attr:`Merge.DERIVE` records are not written
        (the caller re-derives :attr:`rederive`).  An undeclared channel is
        adopted when *store* has none and otherwise left as the scene's own,
        noted.  One record's failure is noted and the others still merge.

        Returns:
            The context; its :attr:`~TransferContext.notes` are the report.
        """
        ctx = ctx if ctx is not None else TransferContext()
        for scope, key, spec in self._entries():
            if spec is not None and spec.merge in (Merge.DERIVE, Merge.OWN):
                continue
            text = self.other[scope][key]
            try:
                if spec is None:
                    self._adopt_undeclared(store, scope, key, text, ctx)
                    continue
                other = spec.decode(text)
                if other is None:
                    ctx.note(f"{spec.owner}: the other scene's copy could not be read.")
                    continue
                self.merge_record(store, spec, other, ctx)
            except Exception as error:  # noqa: BLE001 - one record never costs the rest
                logger.warning("Scene record %r was not merged.", key, exc_info=True)
                ctx.note(f"{key}: not merged ({error}).")
        return ctx

    @staticmethod
    def _adopt_undeclared(store, scope: Scope, key: str, text: str, ctx) -> None:
        if store.read(scope, key):
            ctx.note(
                f"{key}: this scene's own copy was kept; the other scene's was "
                "not merged (no merge rule is declared for it)."
            )
            return
        store.write(scope, key, text)

    @classmethod
    def merge_record(
        cls,
        store,
        spec: RecordSpec,
        other: Any,
        ctx: TransferContext,
        respelled: bool = False,
    ) -> Any:
        """Merge *other* (another scene's decoded payload of *spec*) into
        *store*'s copy by *spec*'s rule and write the result -- the one merge
        every route calls, a DCC owner's ``transfer_in`` included.

        *other* is respelled first (:meth:`respell`) unless *respelled* says
        its names already are this scene's -- an owner that decoded the
        payload against this scene has resolved them.  Returns the payload
        written (``None`` when the rule writes nothing).
        """
        if spec.merge in (Merge.DERIVE, Merge.OWN) or other is None:
            return None
        if not respelled:
            other = cls.respell(spec, other, ctx)
        if spec.paths:
            other = cls.arriving_paths(other, ctx)
        own = spec.load(store)
        if spec.merge is Merge.CODEC:
            codec = SceneRecords.codec(spec)
            if codec is None:
                raise LookupError(f"no codec is registered for {spec.key!r}")
            merged = codec.merge_record(own, other, ctx)
        else:
            merged = cls.union(own, other, spec, ctx)
        spec.save(store, merged)
        return merged

    @staticmethod
    def absolute_paths(payload: Any, ctx: TransferContext) -> Any:
        """A :attr:`RecordSpec.paths` *payload* resolved absolute from
        ``ctx.path_base`` -- how it leaves the scene it is spelled for."""
        from pythontk.file_utils._file_utils import FileUtils

        return RecordTransfer._crossing_paths(
            payload, lambda v: FileUtils.resolve_portable_path(v, ctx.path_base)
        )

    @staticmethod
    def arriving_paths(payload: Any, ctx: TransferContext) -> Any:
        """A :attr:`RecordSpec.paths` *payload* from the other scene spelled
        from this one's project: resolved from ``ctx.source_path_base`` (an
        absolute value needs none), re-spelled from ``ctx.path_base``."""
        from pythontk.file_utils._file_utils import FileUtils

        return RecordTransfer._crossing_paths(
            payload,
            lambda v: FileUtils.rebase_portable_path(
                v, ctx.source_path_base, ctx.path_base
            ),
        )

    @staticmethod
    def _crossing_paths(payload: Any, spell: Callable[[str], str]) -> Any:
        """:meth:`SceneRecords.map_paths` for a crossing, less every EMPTY
        value: a writer entry made while its scene was unsaved means "this
        scene" and names no file, so it cannot leave that scene -- landed, it
        would name the scene it lands in (``FileDependencies.written_here``),
        whose re-bake would then delete what the other still reads."""
        if isinstance(payload, Mapping):
            payload = {k: v for k, v in payload.items() if v != ""}
        return SceneRecords.map_paths(payload, spell)

    @staticmethod
    def respell(spec: RecordSpec, payload: Any, ctx: TransferContext) -> Any:
        """*payload* of *spec* with its names put through ``ctx.rename``.

        A codec that knows which of its strings are names respells those
        alone (its ``respell_record(payload, ctx)``) -- a shot named like a
        renamed node keeps its name; any other record that spells names has
        every string respelled (:meth:`TransferContext.respell`), and one that
        does not (:attr:`RecordSpec.respell` False) crosses as it is.
        """
        if not spec.respell or not payload or ctx.rename is None:
            return payload
        codec = SceneRecords.codec(spec)
        hook = getattr(codec, "respell_record", None)
        return hook(payload, ctx) if hook is not None else ctx.respell(payload)

    @staticmethod
    def union(own: Any, other: Any, spec: RecordSpec, ctx: TransferContext) -> Any:
        """The :attr:`Merge.UNION` rule: entries by identity, the scene's own
        entry winning a collision (noted when the two differ).

        A mapping unites by key.  A list unites by ``spec.merge_key`` (or by
        equality without one) with the other scene's new entries FIRST, so a
        LIFO stack's newest -- the scene's own -- stays on top.  Either side
        absent adopts the other.
        """
        if not own:
            return other
        if not other:
            return own
        if isinstance(own, Mapping) and isinstance(other, Mapping):
            merged = dict(other)
            for key, value in own.items():
                if key in merged and merged[key] != value:
                    ctx.note(f"{spec.owner}: {key!r} kept as this scene has it.")
                merged[key] = value
            return merged
        if isinstance(own, list) and isinstance(other, list):
            field_name = spec.merge_key

            def ident(entry: Any) -> Any:
                if field_name and isinstance(entry, Mapping):
                    return entry.get(field_name)
                return json.dumps(entry, sort_keys=True, default=str)

            mine = {ident(e): e for e in own}
            fresh = []
            for entry in other:
                found = mine.get(ident(entry))
                if found is None:
                    fresh.append(entry)
                elif found != entry:
                    ctx.note(
                        f"{spec.owner}: {ident(entry)!r} kept as this scene has it."
                    )
            return fresh + list(own)
        ctx.note(f"{spec.owner}: the two copies differ in shape; this scene's kept.")
        return own

    # ------------------------------------------------------------ hand-off
    @classmethod
    def sections(
        cls,
        store,
        ctx: TransferContext,
        owners: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """What a hand-off producer adds to its sidecar: every portable record
        the scene holds, each under its :attr:`RecordSpec.section` or keyed in
        the generic :attr:`RECORDS_SECTION`.

        A record's DCC owner (*owners*, record key -> class) produces the
        payload when it defines ``transfer_out(ctx)`` -- it keeps state beside
        the record (memberships, keyed channels) the stored record does not
        hold.  Otherwise a record with a section of its own crosses in that
        section's wire shape, which its codec defines (``section_out(record,
        ctx)`` -- the shot store's is :meth:`ShotTransfer.encode`'s envelope,
        what every consumer decodes, never the bare record); any other
        stored record crosses respelled (:meth:`respell`) through
        ``ctx.rename`` -- the carrier's spelling.  One record's failure is
        noted and the others still ship.
        """
        out: Dict[str, Any] = {}
        for spec in SceneRecords.portable():
            owner = (owners or {}).get(spec.key)
            hook = getattr(owner, "transfer_out", None)
            section_out = cls._section_hook(spec, "section_out")
            try:
                if hook is not None:
                    payload = hook(ctx)
                elif section_out is not None:
                    payload = section_out(spec.load(store), ctx)
                else:
                    payload = cls.respell(spec, spec.load(store), ctx)
                if spec.paths:
                    # Spelled from THIS scene's project, which the other side
                    # does not share: the far side spells them from its own.
                    payload = cls.absolute_paths(payload, ctx)
            except Exception as error:  # noqa: BLE001 - one record never costs the rest
                logger.warning("Scene record %r was not sent.", spec.key, exc_info=True)
                ctx.note(f"{spec.owner}: not sent ({error}).")
                continue
            if not payload:
                continue
            if spec.section:
                out[spec.section] = payload
            else:
                out.setdefault(cls.RECORDS_SECTION, {})[spec.key] = payload
        return out

    @staticmethod
    def _section_hook(spec: RecordSpec, name: str) -> Optional[Callable[..., Any]]:
        """The codec's ``section_out`` / ``section_in`` for a record that
        crosses under a section of its own, else ``None`` -- a section's wire
        shape is the codec's to define, a generic record's is the record."""
        if not spec.section:
            return None
        return getattr(SceneRecords.codec(spec), name, None)

    @classmethod
    def receive(
        cls,
        manifest: Mapping[str, Any],
        store,
        ctx: TransferContext,
        owners: Optional[Mapping[str, Any]] = None,
    ) -> TransferContext:
        """Land a hand-off sidecar's records in *store*'s scene.

        Each portable record found in *manifest* (under its own section, or
        in :attr:`RECORDS_SECTION`) goes to its DCC owner's
        ``transfer_in(payload, ctx)`` when it defines one; otherwise a
        section is first read back by its codec (``section_in(payload,
        ctx)``, the inverse of what :meth:`sections` wrote, names resolved
        through ``ctx.rename``), and the record merges by its rule
        (:meth:`merge_record`; a generic payload is respelled there through
        ``ctx.rename`` -- the carrier's spelling resolved to this scene).
        Best-effort per record: a bad payload is noted and never costs the
        import.

        Returns:
            *ctx*, its notes the report.
        """
        generic = manifest.get(cls.RECORDS_SECTION) if manifest else None
        generic = generic if isinstance(generic, Mapping) else {}
        for spec in SceneRecords.portable():
            payload = (
                manifest.get(spec.section) if spec.section else generic.get(spec.key)
            )
            if not payload:
                continue
            owner = (owners or {}).get(spec.key)
            hook = getattr(owner, "transfer_in", None)
            section_in = cls._section_hook(spec, "section_in")
            try:
                if hook is not None:
                    hook(payload, ctx)
                elif section_in is not None:
                    record = section_in(payload, ctx)
                    cls.merge_record(store, spec, record, ctx, respelled=True)
                else:
                    cls.merge_record(store, spec, payload, ctx)
            except Exception as error:  # noqa: BLE001 - a record, never a failed import
                logger.warning(
                    "Scene record %r was not received.", spec.key, exc_info=True
                )
                ctx.note(f"{spec.owner}: not received ({error}).")
        return ctx
