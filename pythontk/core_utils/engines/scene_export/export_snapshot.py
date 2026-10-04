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

import logging
from dataclasses import dataclass, field
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

from pythontk.core_utils.engines.scene_export.scene_records import (
    Kind,
    Record,
    RecordSpec,
    SceneRecords,
    Scope,
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
            :attr:`Kind.DERIVED` records refresh) or :attr:`AUTHORING` (a
            tool republishing its own record).
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
            value}}`` over ``MeshConvert.RENDERING_POLICY``); empty = the
            policy as it stands.
        records: Every record produced so far in this assembly, by key --
            how a producer reads another's output (the audio manifest scopes
            its events against the takes the shots producer just built).
        notes: What a producer left out of its record, or could not keep,
            one sentence each (:meth:`note`) -- the report an exporter shows,
            so nothing is dropped from a deliverable silently (the shots
            producer names the stale shots it did not declare).  One
            assembly's, like :attr:`records`.
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
    notes: List[str] = field(default_factory=list)

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
        #: ``ctx.notes`` by the record whose producer noted them, so a host can
        #: offer each note's remedy where it can be acted on.
        self.noted: Dict[str, List[str]] = {}
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
        snapshot like any other.  *ctx* holds ONE assembly's records and
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
            before = len(ctx.notes)
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
            finally:
                if len(ctx.notes) > before:
                    snapshot.noted[spec.key] = ctx.notes[before:]
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
        for note in ctx.notes:
            logger.warning(note)
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
