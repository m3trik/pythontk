# !/usr/bin/python
# coding=utf-8
"""Records crossing between scenes, by declaration (:class:`RecordTransfer`).

**Cross, by declaration.**  A record says how it crosses into another scene
(:class:`~.scene_records.Merge`, :attr:`~.scene_records.RecordSpec.portable`),
so every route a record takes -- a referenced module imported into its host, a
DCC hand-off, a legacy fold -- is one engine over the declarations.  A record's
own semantics plug in as a codec (its merge) and a DCC owner (what it keeps
beside the record); a new record crosses correctly by being declared, with no
route learning its name.  :class:`TransferContext` is what a crossing needs
from the DCC (its name spellings) and what it learns on the way.

Zero-dep and DCC-agnostic, like everything in ``pythontk``.
"""

import copy
import json
import logging
from dataclasses import dataclass, field
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
    Merge,
    RecordSpec,
    SceneRecords,
    Scope,
)

logger = logging.getLogger(__name__)


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
            other = cls.arriving_paths(other, ctx, spec.path_keys)
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
    def absolute_paths(
        payload: Any,
        ctx: TransferContext,
        keys: Optional[Tuple[str, ...]] = None,
    ) -> Any:
        """A :attr:`RecordSpec.paths` *payload* resolved absolute from
        ``ctx.path_base`` -- how it leaves the scene it is spelled for.  *keys*
        as :meth:`SceneRecords.map_paths` takes them."""
        from pythontk.file_utils._file_utils import FileUtils

        return RecordTransfer._crossing_paths(
            payload, lambda v: FileUtils.resolve_portable_path(v, ctx.path_base), keys
        )

    @staticmethod
    def arriving_paths(
        payload: Any,
        ctx: TransferContext,
        keys: Optional[Tuple[str, ...]] = None,
    ) -> Any:
        """A :attr:`RecordSpec.paths` *payload* from the other scene spelled
        from this one's project: resolved from ``ctx.source_path_base`` (an
        absolute value needs none), re-spelled from ``ctx.path_base``.  *keys*
        as :meth:`SceneRecords.map_paths` takes them."""
        from pythontk.file_utils._file_utils import FileUtils

        return RecordTransfer._crossing_paths(
            payload,
            lambda v: FileUtils.rebase_portable_path(
                v, ctx.source_path_base, ctx.path_base
            ),
            keys,
        )

    @staticmethod
    def _crossing_paths(
        payload: Any,
        spell: Callable[[str], str],
        keys: Optional[Tuple[str, ...]] = None,
    ) -> Any:
        """:meth:`SceneRecords.map_paths` for a crossing, less every EMPTY
        path: a writer entry made while its scene was unsaved means "this
        scene" and names no file, so it cannot leave that scene -- landed, it
        would name the scene it lands in (``FileDependencies.written_here``),
        whose re-bake would then delete what the other still reads."""
        if isinstance(payload, Mapping):
            payload = {
                k: v
                for k, v in payload.items()
                if v != "" or (keys is not None and k not in keys)
            }
        return SceneRecords.map_paths(payload, spell, keys)

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
                    payload = cls.absolute_paths(payload, ctx, spec.path_keys)
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
