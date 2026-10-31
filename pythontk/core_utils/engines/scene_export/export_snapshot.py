# !/usr/bin/python
# coding=utf-8
"""Export assembly: compute every record once, then commit once.

**Compute, then commit, once.**  A producer RETURNS a
:class:`~.scene_records.Record` and never writes.
:meth:`ExportSnapshot.assemble` orders the producers by their declared
dependencies, hands each the :class:`ExportContext` (the exporter's decisions
as INPUT, plus every record produced before it) and
:meth:`ExportSnapshot.commit` writes the whole snapshot in one pass and stamps
the handoff block.  The same snapshot object is what the sidecar, the export
log and the verifier consume, so nothing is read back off the node and patched.

Zero-dep and DCC-agnostic, like everything in ``pythontk``.
"""

import contextlib
import logging
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from pythontk.core_utils.engines.scene_export.handoff_record import HandoffRecord
from pythontk.core_utils.engines.scene_export.scene_records import (
    Kind,
    Record,
    RecordSpec,
    SceneRecords,
    Scope,
    _NodeNames,
)

logger = logging.getLogger(__name__)


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
            :attr:`Kind.DERIVED` records refresh, and an export-scoped one it
            ships as stored is narrowed, :meth:`ExportSnapshot.commit`) or
            :attr:`AUTHORING` (a
            tool republishing its own record -- no write, so a
            :attr:`RecordSpec.transient` record named is cleared, never
            produced).
        clip_mode: The run's Animation Clips mode (``full`` / ``shots`` /
            ``both``), declared on the shot record; ``None`` = undeclared.
        clip_span: The first and last frame the exported stack CARRIES, the
            frame every GLB clip is cut against; measured by the pipeline
            from the final keys, ``None`` when nothing measured it. A
            statement for readers of the carrier: a GLB conversion measures
            the written file itself and logs a disagreement
            (``MeshConvert._stamp_clip_spans``).
        source: Producer identity and provenance for the handoff block
            (``application``, ``version``, ``scene``).
        rendering: The export's choices over the lighting recipe the handoff
            block publishes (``ExportRun.rendering``: ``{section: {field:
            value}}`` over ``HandoffRecord.RENDERING_POLICY``); empty = the policy
            as it stands.
        records: Every record produced so far in this assembly, by key --
            how a producer reads another's output (the audio manifest scopes
            its events against the takes the shots producer just built).
        notes: What a producer left out of its record, or could not keep,
            one sentence each (:meth:`note`) -- the report an exporter shows,
            so nothing is dropped from a deliverable silently (the shots
            producer names the stale shots it did not declare).  One
            assembly's, like :attr:`records`.
        scope: The export set, as the host spells its nodes -- the roots an
            export or a hand-off writes; ``None`` is the whole scene (an
            authoring publish, a raw export).  The producer of an
            :attr:`RecordSpec.export_scoped` record keeps only the entries
            whose node is one of these or sits under one.
        nodes: The names of every node the write ships -- *scope*'s roots
            and all under them, each as the file names it (a DAG node's leaf,
            namespace kept) -- filled once by the host that knows its scene
            (``FbxUtils.export_context``); ``None`` names none (no scope, or
            a context built without a host).  What the snapshot narrows an
            export-scoped record by (:attr:`RecordSpec.nodes`), the produced
            copy and, on a hand-off, the one shipped as stored; and what a
            producer asks of a node (:meth:`ships`).  Shots and audio are the
            timeline's and ship whole.
        unit_scale: The metres one scene unit is in the file this write
            produces -- stated by its WRITER (:attr:`WRITER_UNITS`, filled by
            the host's ``FbxUtils.export_context``), and the label every
            producer gives the lengths it records in scene units (a record's
            ``unit_scale``). No producer converts: the lengths stay as the
            scene holds them, and the label says what the writer made of
            them. ``None`` when no writer stated one: the producer labels its
            record with the scene's unit.
    """

    #: Each writer's effective metres per scene unit, stated once: ``{(host,
    #: format): rule}``, a rule taking the scene's metres per unit and the
    #: write's own options (the writer's keyword arguments) and answering the
    #: metres one scene unit is in the file it writes. Where both hosts'
    #: writers reach it; a new writer is one row.
    WRITER_UNITS: ClassVar[
        Dict[Tuple[str, str], Callable[[float, Mapping[str, Any]], float]]
    ] = {
        # Maya's writers write the scene's unit and say so in the file (the
        # FBX UnitScaleFactor, USD's metersPerUnit): a converted unit changes
        # the numbers, never the metres.
        ("maya", "fbx"): lambda scene, options: scene,
        ("maya", "usd"): lambda scene, options: scene,
        # io_scene_fbx scales by scale_length only with apply_unit_scale on
        # (its default), and by global_scale always: off, one BU is one metre.
        ("blender", "fbx"): lambda scene, options: (
            (scene if options.get("apply_unit_scale", True) else 1.0)
            * float(options.get("global_scale", 1.0))
        ),
        # Blender's native glTF writer ignores scale_length: one BU is one
        # metre (measured 2026-10-17: a 1 BU = 1 cm scene's node at 100 BU
        # wrote 100).
        ("blender", "gltf"): lambda scene, options: 1.0,
    }

    @classmethod
    def writer_unit(
        cls,
        host: str,
        writer: str,
        scene_unit: float,
        options: Optional[Mapping[str, Any]] = None,
    ) -> float:
        """The metres one scene unit is in the file *host*'s *writer* writes
        with *options*, the scene's being *scene_unit* (:attr:`WRITER_UNITS`).

        Raises:
            KeyError: A writer with no row -- a write whose unit nobody stated.
        """
        try:
            rule = cls.WRITER_UNITS[(host, writer)]
        except KeyError:
            raise KeyError(
                f"No unit is stated for the {host} {writer!r} writer; add its "
                "row to ExportContext.WRITER_UNITS."
            ) from None
        return float(rule(float(scene_unit), dict(options or {})))

    def unit(self, scene_unit: Callable[[], float]) -> float:
        """The unit label a producer gives its record: :attr:`unit_scale`, or
        -- no writer stated one -- *scene_unit()*, the scene's metres per
        unit, read only then."""
        return self.unit_scale if self.unit_scale is not None else scene_unit()

    PIPELINE = "pipeline"
    HANDOFF = "handoff"
    AUTHORING = "authoring"

    mode: str = "pipeline"
    clip_mode: Optional[str] = None
    clip_span: Optional[Tuple[float, float]] = None
    source: Dict[str, Any] = field(default_factory=dict)
    rendering: Dict[str, Any] = field(default_factory=dict)
    records: Dict[str, Optional[Record]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    scope: Optional[Tuple[str, ...]] = None
    nodes: Optional[FrozenSet[str]] = None
    unit_scale: Optional[float] = None
    #: :attr:`nodes` as a reader matches them, built on first use.
    _names: Optional[_NodeNames] = field(
        default=None, init=False, repr=False, compare=False
    )

    def ships(self, name: Any) -> bool:
        """Whether the node *name* ships (:attr:`nodes`): matched as a reader
        binds a record to its node -- a name exactly, else its leaf (a DAG
        path's last part, a namespace's) -- so a producer asks in its host's
        own spelling. Every node ships when :attr:`nodes` is ``None``."""
        names = self._node_names()
        return True if names is None else name in names

    def narrowed(self, spec: RecordSpec, payload: Any) -> Any:
        """*payload* of *spec* narrowed to the nodes the write ships
        (:meth:`RecordSpec.narrowed`); as it is when :attr:`nodes` is ``None``
        or *spec* is not export-scoped."""
        names = self._node_names()
        if names is None or not spec.export_scoped:
            return payload
        return spec.narrowed(payload, names)

    def _node_names(self) -> Optional[_NodeNames]:
        if self.nodes is None:
            return None
        if self._names is None:
            self._names = _NodeNames(self.nodes)
        return self._names

    def note(self, text: str) -> None:
        """Record one sentence for the report."""
        self.notes.append(text)

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
        #: ``ctx.notes`` by the record whose producer noted them, so a host can
        #: offer each note's remedy where it can be acted on.
        self.noted: Dict[str, List[str]] = {}
        #: What :meth:`commit` wrote, key -> text (``None`` = cleared).
        self.written: Dict[str, Optional[str]] = {}
        #: The scene's own copy of each record this write's differs from, which
        #: :meth:`restore_scene_wide` puts back once the write is done: for a
        #: scoped assembly (``ctx.scope``), each export-scoped record the scope
        #: narrowed, as the whole scene produces it (``None`` = cleared), or as
        #: the scene stored it when the write ships it narrowed as stored
        #: (:meth:`commit`); and ``None`` for every
        #: :attr:`RecordSpec.transient` record produced, which the scene never
        #: holds.
        self.scene_wide: Dict[str, Optional[Record]] = {}

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
        still runs.  A producer may return records for keys other than its
        own; those ride the snapshot like any other.  *ctx* holds ONE assembly's records and
        notes: a reused context starts empty, so no producer reads what an
        earlier assembly produced.  Each note is logged as a warning, and
        stays on ``snapshot.ctx`` for an exporter's own report.

        Returns:
            The snapshot, not yet committed.

        Raises:
            KeyError, ValueError: *producers* names an undeclared, private or
                handoff record (:meth:`SceneRecords.check_producers`).
        """
        ctx = ctx or ExportContext()
        ctx.records.clear()
        ctx.notes.clear()
        snapshot = cls(ctx)
        table = dict(zip(SceneRecords.check_producers(producers), producers.values()))
        wanted = None if only is None else {SceneRecords.resolve(k).key for k in only}
        selected = [
            spec
            for spec in table
            if (wanted is None or spec.key in wanted) and ctx.refreshes(spec)
        ]
        for spec in SceneRecords.ordered(selected):
            if spec.transient and ctx.mode == ctx.AUTHORING:
                # An authoring publish is no write: a transient record is
                # cleared, never produced -- the scene never holds one (a
                # merge re-deriving what an import brought clears it here).
                snapshot.produced[spec.key] = ctx.records[spec.key] = None
                continue
            before = len(ctx.notes)
            try:
                records = cls._narrowed(ctx, cls._records_of(spec, table[spec](ctx)))
            except Exception:  # one subsystem's failure must not block the others
                logger.warning(
                    "Scene record %r was not produced.", spec.key, exc_info=True
                )
                snapshot.failed.add(spec.key)
                continue
            finally:
                if len(ctx.notes) > before:
                    snapshot.noted[spec.key] = ctx.notes[before:]
            own = next((r for r in records if r.spec.key == spec.key), None)
            # The producer's own record first, then any it returned for other
            # keys, so the snapshot reads in producer order.
            snapshot.produced[spec.key] = own
            # Kept even as None: a reader later in this assembly must see
            # "cleared", not fall back to the stored copy the commit is about
            # to clear.
            ctx.records[spec.key] = own
            for record in records:
                if record is not own:
                    snapshot.produced[record.spec.key] = record
                    ctx.records[record.spec.key] = record
        for note in ctx.notes:
            logger.warning(note)
        if ctx.scope is not None or ctx.nodes is not None:
            snapshot._assemble_scene_wide(table, selected)
        for key, record in snapshot.produced.items():
            # A transient record describes this write alone: the scene's own
            # copy of it is none, which the exporter's restore puts back.
            if record is not None and record.spec.transient:
                snapshot.scene_wide[key] = None
        return snapshot

    @staticmethod
    def _records_of(spec: RecordSpec, result: Any) -> List[Record]:
        """A producer's *result* as a list of records (``None`` dropped).

        Raises:
            TypeError: The producer returned a bare payload.
        """
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
        return records

    @staticmethod
    def _narrowed(ctx: ExportContext, records: List[Record]) -> List[Record]:
        """*records* narrowed to the nodes the write ships
        (:meth:`ExportContext.narrowed`); one left with no entry is dropped,
        so its key reads as cleared. The producer's own scoping comes first;
        this holds every export-scoped record to the rule whatever its
        producer did (an articulated rig names its joints alone)."""
        out = []
        for record in records:
            payload = ctx.narrowed(record.spec, record.payload)
            if payload is record.payload:
                out.append(record)
            elif payload is not None:
                out.append(Record(record.spec, payload))
        return out

    def _assemble_scene_wide(
        self, table: Mapping[RecordSpec, Producer], selected: Iterable[RecordSpec]
    ) -> None:
        """Fill :attr:`scene_wide`: run each export-scoped producer of this
        scoped assembly again over the whole scene, and keep its record where
        the scope narrowed it. Same scene state as the scoped run, so the
        restore puts back exactly what an unscoped export would have left.

        Silent (:meth:`_silenced`): what a producer says here is either said
        already, about a node the write ships, or about one it does not --
        and the export log is the deliverable's. A producer that fails keeps
        the narrow copy (logged)."""
        whole = replace(
            self.ctx, scope=None, nodes=None, records=dict(self.ctx.records), notes=[]
        )
        for spec in selected:
            if not spec.export_scoped or spec.transient or spec.key in self.failed:
                continue
            try:
                with self._silenced():
                    result = table[spec](whole)
                records = self._records_of(spec, result)
            except Exception:  # noqa: BLE001 - the write's own copy still ships
                logger.warning(
                    "Scene record %r was not produced for the whole scene; the "
                    "scene keeps the export's narrowed copy.",
                    spec.key,
                    exc_info=True,
                )
                continue
            own = next((r for r in records if r.spec.key == spec.key), None)
            shipped = self.produced.get(spec.key)
            if (own.text if own else None) != (shipped.text if shipped else None):
                self.scene_wide[spec.key] = own

    @staticmethod
    @contextlib.contextmanager
    def _silenced():
        """Drop every log record for the duration (``logging.disable``), then
        put back the level that was disabled before."""
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            yield
        finally:
            logging.disable(previous)

    def restore_scene_wide(self, store) -> Optional["ExportSnapshot"]:
        """Put the scene's own copy of every record this write's differed from
        back on *store* (:attr:`scene_wide`) -- the exporter's restore once
        the write is done, so the carrier the scene keeps (and a later
        hand-off ships as stored) describes all of the scene and nothing of
        the write: a narrowed record whole again, a transient one cleared,
        and a carrier left holding nothing removed
        (:meth:`SceneStoreBase.discard_if_empty`).

        Returns:
            The authoring snapshot committed, or ``None`` when the write's
            records were the scene's own.
        """
        if not self.scene_wide:
            return None
        # The export's provenance and lighting choices, so the restamped
        # handoff block reads as the export's commit left it.
        ctx = ExportContext(
            mode=ExportContext.AUTHORING,
            source=dict(self.ctx.source),
            rendering=dict(self.ctx.rendering),
        )
        restored = type(self).publish(store, self.scene_wide, ctx)
        if not store.channels(Scope.DELIVERABLE):
            # The write's records were all it held: a carrier the export made
            # for them goes with them.
            store.discard_if_empty(Scope.DELIVERABLE)
        return restored

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
                key (the emissive registry and manifest did until the
                registry took ``emissive_registry``), so clearing it here
                would clear the wrong carrier; use
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

        A write that names the nodes it ships (``ctx.nodes``) ships every
        export-scoped record narrowed to them, the ones this assembly did not
        produce too: a hand-off refreshes only the derived records, so the
        authored ones (the lightmaps, the shadow planes) would otherwise ship
        the whole scene's entries, and the post-write embed their maps. Each
        stored copy narrowed is committed narrowed and kept whole in
        :attr:`scene_wide`, for the exporter's restore. A failed producer's
        record stays as stored.

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

        # A scene written before a record's rename holds it under the old
        # key: moved first, so the writes and the handoff see current keys
        # alone. Never the export's cost.
        try:
            store.migrate()
        except Exception:  # noqa: BLE001
            logger.warning("Scene records were not migrated.", exc_info=True)
        if self.ctx.nodes is not None and self.ctx.mode != ExportContext.AUTHORING:
            self._narrow_stored(store)
        for key, record in self.produced.items():
            if key in self.failed:
                continue
            if record is None:
                write(SceneRecords.resolve(key), None)
            else:
                write(record.spec, record.text)
        try:
            handoff = SceneRecords.HANDOFF
            present = store.channels(Scope.DELIVERABLE)
            held = HandoffRecord.carried(present)
            # A carrier holding no record has nothing to describe: no block,
            # and no carrier made to hold one.
            # The payloads, not just the keys: each row states what its
            # record calls for and nothing more.
            block = (
                HandoffRecord.block(
                    "fbx", present, source=self.ctx.source, rendering=self.ctx.rendering
                )
                if held
                else {}
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

    def _narrow_stored(self, store) -> None:
        """Narrow each export-scoped record *store* holds that this assembly
        did not produce (:attr:`RecordSpec.nodes`) to the nodes the write
        ships, as :meth:`commit` writes it, keeping the stored copy in
        :attr:`scene_wide`. A record its producer failed for stays as stored."""
        for spec in SceneRecords.deliverable():
            if not spec.export_scoped or spec.nodes is None:
                continue
            if spec.key in self.produced or spec.key in self.failed:
                continue
            try:
                stored = spec.load(store, None)
            except Exception:  # noqa: BLE001 - an unreadable copy ships as it is
                logger.debug("Scene record %r not read.", spec.key, exc_info=True)
                continue
            if stored is None:
                continue
            narrowed = self.ctx.narrowed(spec, stored)
            if narrowed is stored:
                continue
            self.scene_wide[spec.key] = Record(spec, stored)
            self.produced[spec.key] = (
                Record(spec, narrowed) if narrowed is not None else None
            )

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
