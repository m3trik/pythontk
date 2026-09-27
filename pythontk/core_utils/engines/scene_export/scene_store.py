# !/usr/bin/python
# coding=utf-8
"""The storage contract every DCC scene store implements (:class:`SceneStoreBase`).

**Storage is dumb, records are smart.**  A DCC implements ``read`` / ``write``
/ ``keys`` / ``values`` per :class:`~.scene_records.Scope`, strings only;
encoding, the version envelope, tolerant decoding and clear-on-empty live on
the record (:class:`~.scene_records.RecordSpec`), so the two DCC mirrors cannot
diverge on semantics.  The inspection surface (``dump`` / ``format_dump``), the
project-relative path spelling and the scene crossings (a hand-off's sidecar
sections, another scene's carriers merged or discarded through
:class:`~.record_transfer.RecordTransfer`) are written once, here.

Zero-dep and DCC-agnostic, like everything in ``pythontk``.
"""

import contextlib
import json
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)

from pythontk.core_utils.engines.scene_export.scene_records import (
    RecordSpec,
    SceneRecords,
    Scope,
)
from pythontk.core_utils.engines.scene_export.record_transfer import (
    RecordTransfer,
    TransferContext,
)

logger = logging.getLogger(__name__)


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
    def scene_path(cls) -> str:
        """The open scene's file, ``""`` while it is unsaved -- and in this
        base: a DCC store answers for its open file.  What the project root
        and the writer stamp derive from."""
        return ""

    @classmethod
    def project_root(cls) -> Optional[str]:
        """This scene's own project root -- what its :attr:`RecordSpec.paths`
        records are spelled from (``FileUtils.portable_path``): the project
        the scene FILE lives in (:meth:`project_root_of` of
        :meth:`scene_path`), never the one a session has set.  ``None`` while
        the scene is unsaved."""
        return cls.project_root_of(cls.scene_path())

    @classmethod
    def writer_stamp(cls) -> str:
        """This scene's file as a record stamps its writer: spelled from the
        scene's own project (``FileUtils.portable_path``), ``""`` while
        unsaved.  What a record stores to say which scene file made it -- a
        Save As copy carries every record verbatim (the lightmap writers, the
        hierarchy baseline) -- read back by :meth:`written_here`.  The record
        declares the stamp a path (:attr:`RecordSpec.paths`), so a save into
        another project re-spells it and the copy still names its source."""
        return cls.writer_stamp_of(cls.scene_path())

    @classmethod
    def writer_stamp_of(cls, scene_path: Optional[str]) -> str:
        """*scene_path* as a writer stamp: spelled from ITS own project
        (:meth:`project_root_of`), ``""`` for none -- :meth:`writer_stamp`'s
        answer for the open scene, and a save hook's for the file about to be
        written (:meth:`SceneRecords.stamp_unsaved`)."""
        from pythontk.file_utils._file_utils import FileUtils

        if not scene_path:
            return ""
        return FileUtils.portable_path(scene_path, cls.project_root_of(scene_path))

    @classmethod
    def written_here(cls, stamp: Optional[str]) -> bool:
        """Whether a record stamped *stamp* (:meth:`writer_stamp`) is this
        scene's own (``FileDependencies.written_here``): this file, a file
        that is gone (the scene was renamed or moved), or ``""`` while the
        scene is still unsaved -- never a Save As copy's source, which is
        still on disk.  ``None`` (no stamp) is nobody's."""
        from pythontk.file_utils.file_dependencies import FileDependencies

        return FileDependencies.written_here(
            stamp, cls.scene_path(), cls.project_root()
        )

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
    def respell_for_write(
        cls, target: str, old_base: Optional[str], first_save: bool = False
    ) -> Dict[Tuple[Scope, str], Optional[str]]:
        """Ready the path records for this scene written as *target* -- a
        save, a copy, an export into a scene file -- and return them as they
        were.

        Re-spelled from *old_base*, the project they are spelled from now, to
        *target*'s (:meth:`rebase_paths`), so the written file resolves them
        from where IT lives; on the scene's *first_save* every ``""`` writer
        stamp is given *target*'s (:meth:`SceneRecords.stamp_unsaved`).  What
        both DCCs' save hooks call; a write that leaves the open scene where
        it was -- a copy, an autosave, a failed save -- hands the returned
        snapshot to :meth:`restore_path_records` afterwards.

        Parameters:
            target: The file being written.
            old_base: The project the records are spelled from now (``None``
                while the scene was unsaved: its entries are absolute).
            first_save: Whether this write gives an unsaved scene its file.

        Returns:
            The path records' stored text before the write
            (:meth:`path_records`).
        """
        snapshot = cls.path_records()
        new_base = cls.project_root_of(target)
        if new_base is not None:
            cls.rebase_paths(old_base, new_base)
        if first_save:
            SceneRecords.stamp_unsaved(cls, cls.writer_stamp_of(target))
        return snapshot

    @classmethod
    def path_records(cls) -> Dict[Tuple[Scope, str], Optional[str]]:
        """The stored text of every path record (:meth:`SceneRecords.with_paths`),
        by ``(scope, key)`` -- the snapshot :meth:`restore_path_records` puts
        back exactly, where re-spelling back could normalize an entry."""
        return {
            (spec.scope, spec.key): cls.read(spec.scope, spec.key)
            for spec in SceneRecords.with_paths()
        }

    @classmethod
    def restore_path_records(
        cls, snapshot: Mapping[Tuple[Scope, str], Optional[str]]
    ) -> int:
        """Put the path records back as *snapshot* (:meth:`path_records`) holds
        them; returns how many records changed."""
        changed = 0
        for (scope, key), text in snapshot.items():
            if cls.read(scope, key) != text:
                cls.write(scope, key, text)
                changed += 1
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
