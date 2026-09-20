# !/usr/bin/python
# coding=utf-8
"""Shot transfer codec -- the shot store as a DCC-neutral hand-off section.

Neither FBX nor USD has a place for a shot list, a marker, a locked gap or the
ledger of samples the sequencer planted on shot bounds, so a scene sent from
one DCC to the other arrived with its animation and none of its shots.  The
``.manifest.json`` sidecar every hand-off already carries (materials, lights,
the scene clock, ...) gains a ``shots`` section, and this class is the ONE
codec both producers and both consumers run: mayatk and blendertk write the
section through :meth:`ShotTransfer.encode` and read it back through
:meth:`ShotTransfer.decode`, so a store reconstructs 1:1 in either direction.

What is DCC-specific is injected, never known here:

* how a scene name is spelled on the carrier (``spell``): Maya's long DAG
  path becomes the short name FBX writes, or the sanitized prim a USD stage
  holds -- the same spelling the manifest's other sections use, so the
  consumer resolves every section through one matcher;
* what a ledger curve key names (``curve_ref``): a Maya claim is keyed by
  animCurve node, a Blender one by ``object|data_path|index`` -- both resolve
  to an ``(object, channel label)`` pair, the label in mayatk's
  ``translateX`` vocabulary that blendertk's sequencer already speaks;
* how the receiving scene finds an object (``resolve``) and a curve
  (``curve_key``), and whether a key really sits where a claim says
  (``key_exists``) -- a claim without its key is debris (the importer reduced
  it away, or the carrier never had it) and is dropped rather than carried.

Times travel in the SOURCE scene's frames with its ``scene_fps``; the decoder
rescales to the receiving clock and shifts by the importer's frame offset
(Blender's FBX importer lands frame N at N + ``anim_offset``), the same two
corrections the ``scene`` and ``visibility`` sections already apply.

The section is the store's own :meth:`~pythontk.ShotStore.to_dict` shape with
the names respelled and the ledger regrouped by object and channel -- so a
future store field travels without a codec change, and ``from_dict`` reads the
decoded result as it reads a saved scene.
"""

from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from pythontk.core_utils.engines.shots.shot_ledger import ShotEditLedger
from pythontk.core_utils.engines.shots.shot_model import ShotStore

__all__ = ["ShotTransfer"]

#: ``name -> carrier spelling`` (producer side).
Spell = Callable[[str], str]
#: ``ledger curve key -> (object name, channel label)`` or ``None`` (producer side).
CurveRef = Callable[[str], Optional[Tuple[str, str]]]
#: ``carrier spelling -> live object name`` or ``None`` (consumer side).
Resolve = Callable[[str], Optional[str]]
#: ``(object name, channel label) -> ledger curve key`` or ``None`` (consumer side).
CurveKey = Callable[[str, str], Optional[str]]
#: ``(ledger curve key, time) -> whether a key sits there`` (consumer side).
KeyExists = Callable[[str, float], bool]


class _ShotTransferInternal(object):
    """Internal helpers for ShotTransfer."""

    #: Ledger registers, in the ledger's own dict spelling.
    _REGISTERS = ("steps", "keys")
    #: Per-shot metadata that names scene objects.
    _CSV_OBJECTS = "csv_objects"
    _OBJECT_STATUS = "object_status"

    @staticmethod
    def _unique(names: Iterable[str]) -> List[str]:
        """*names* de-duplicated, first occurrence kept."""
        return list(dict.fromkeys(n for n in names if n))

    @staticmethod
    def _identity(name: str) -> str:
        return str(name)

    @classmethod
    def _spell_metadata(cls, metadata: Dict[str, Any], spell: Spell) -> Dict[str, Any]:
        """*metadata* with its object-naming entries respelled (others verbatim)."""
        out = dict(metadata or {})
        csv = out.get(cls._CSV_OBJECTS)
        if isinstance(csv, list):
            respelled = []
            for entry in csv:
                if isinstance(entry, dict) and entry.get("name"):
                    respelled.append({**entry, "name": spell(str(entry["name"]))})
                elif isinstance(entry, str):
                    respelled.append(spell(entry))
                else:
                    respelled.append(entry)
            out[cls._CSV_OBJECTS] = respelled
        status = out.get(cls._OBJECT_STATUS)
        if isinstance(status, dict):
            out[cls._OBJECT_STATUS] = {spell(str(k)): v for k, v in status.items()}
        return out

    @staticmethod
    def _retime(records: List[list], ratio: float, offset: float) -> List[list]:
        """Ledger records with their time (index 0) rescaled then shifted."""
        out = []
        for rec in records:
            # A bare time is the ledger's legacy record shape; it tolerates it,
            # so the transfer does too (the ledger canonicalises it on load).
            rec = list(rec) if isinstance(rec, (list, tuple)) else [rec]
            try:
                rec[0] = round(float(rec[0]) * ratio + offset, 4)
            except (TypeError, ValueError, IndexError):
                continue
            out.append(rec)
        return out

    @staticmethod
    def _ratio(source_fps: Any, target_fps: Optional[float]) -> float:
        """Frame-scale from *source_fps* to *target_fps* (``1.0`` when either is unknown)."""
        try:
            src = float(source_fps or 0.0)
        except (TypeError, ValueError):
            return 1.0
        if target_fps is None or not src or abs(float(target_fps) - src) < 0.01:
            return 1.0
        return float(target_fps) / src

    @staticmethod
    def _frame(value: Any, ratio: float, offset: float, snap: bool) -> float:
        """*value* on another clock: scaled by *ratio*, then shifted by
        *offset* -- onto a whole frame when *snap*."""
        t = float(value) * ratio + offset
        return float(round(t)) if snap else round(t, 4)

    @classmethod
    def _on_clock(
        cls,
        store: Dict[str, Any],
        ratio: float,
        offset: float = 0.0,
        snap: Optional[bool] = None,
        ledger: bool = True,
    ) -> Dict[str, Any]:
        """A copy of *store* (a ``to_dict`` shape) with its times on another
        clock -- scaled by *ratio*, then shifted by *offset*: shot bounds
        (whole frames when *snap* -- the RECEIVING store's policy, default
        this store's own), markers, the gap (a length, so scaled only) and,
        unless *ledger* is off, the ledger claims (:meth:`decode` rebuilds
        the ledger from the section's own, so it skips that work).  What
        :meth:`~pythontk.ShotStore.rescale_to_fps` does to a live store,
        done on data."""
        out = deepcopy(store)
        if snap is None:
            snap = bool(store.get("snap_whole_frames", True))
        for shot in out.get("shots") or []:
            shot["start"] = cls._frame(shot.get("start", 0.0), ratio, offset, snap)
            shot["end"] = cls._frame(shot.get("end", 0.0), ratio, offset, snap)
        for marker in out.get("markers") or []:
            if "time" in marker:
                marker["time"] = round(float(marker["time"]) * ratio + offset, 4)
        if ratio != 1.0:
            out["gap"] = round(float(store.get("gap") or 0.0) * ratio, 2)
        if ledger and out.get("edit_ledger"):
            # Canonical first, so a legacy record shape retimes too.
            canonical = ShotEditLedger.from_dict(out["edit_ledger"]).to_dict()
            out["edit_ledger"] = {
                reg: {
                    curve: cls._retime(records, ratio, offset)
                    for curve, records in claims.items()
                }
                for reg, claims in canonical.items()
            }
        return out


class ShotTransfer(_ShotTransferInternal):
    """Encode a shot store into a manifest section and decode it into a store.

    Every method is pure: the scene is reached only through the callables the
    caller passes.  See the module docstring for the contract each one meets.
    """

    #: The manifest key the section lives under.
    SECTION = "shots"
    #: Highest section schema this codec writes and reads. v2 added the
    #: ``channels`` and ``audio`` payloads; a v1 section still reads.
    VERSION = 2
    #: Per-object keyed custom channels -- the render-effect channels
    #: (``opacity``, ``highlight``, its colour stops) and any other keyed
    #: user attribute -- as ``{object: {label: {"value", "keys"}}}`` where a
    #: key is ``[time, value, interpolation]`` and the label is mayatk's
    #: attribute spelling (a colour's leaves ``highlightColorR/G/B``). Neither
    #: carrier animates a custom property, so a fade or a pulse crossed as
    #: nothing; a new effect is a new row in the DCCs' channel tables and
    #: travels with no transport change.
    CHANNELS_KEY = "channels"
    #: The audio clips the sequencer shows, ``[{name, file, start, end,
    #: offset}]`` -- neither carrier holds a sound (Maya's audio nodes and
    #: Blender's sound strips both stay behind). ``offset`` is the head trim
    #: in frames, kept for the side that can express it.
    AUDIO_KEY = "audio"
    #: The one interpolation vocabulary a key travels with: what the far side
    #: can reproduce exactly (a hold, a straight line) and the rest.
    KEY_INTERPOLATIONS = ("step", "linear", "smooth")
    #: Store keys holding plain object-name lists.
    OBJECT_LIST_KEYS = ("hidden_objects", "pinned_objects")
    #: Channel labels the Y-up <-> Z-up crossing exchanges. Every Maya <-> Blender
    #: hand-off makes it, on both carriers, and both importers apply it to ROOT
    #: objects only -- a child keeps its parent-space channels (measured 2026-09-17
    #: on the FBX and USD pulls: a root's ``translateZ`` keys land on Blender's
    #: ``location[1]``, a child's on ``location[2]``). Signs never matter here:
    #: a claim names a channel, not a value.
    UP_AXIS_SWAP = {
        "translateY": "translateZ",
        "translateZ": "translateY",
        "rotateY": "rotateZ",
        "rotateZ": "rotateY",
        "scaleY": "scaleZ",
        "scaleZ": "scaleY",
    }

    @classmethod
    def swap_up_axis(cls, label: str) -> str:
        """*label* as the far side of a Y-up / Z-up crossing spells the channel."""
        return cls.UP_AXIS_SWAP.get(label, label)

    # ------------------------------------------------------------------ encode
    @classmethod
    def encode(
        cls,
        state: Dict[str, Any],
        *,
        spell: Optional[Spell] = None,
        curve_ref: Optional[CurveRef] = None,
        objects: Optional[Iterable[str]] = None,
        channels: Optional[Dict[str, Dict[str, Any]]] = None,
        audio: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """The ``shots`` section for a store's :meth:`~pythontk.ShotStore.to_dict`.

        Parameters:
            state: The store's dict form.
            spell: How a scene name is spelled on the carrier (default: as is).
            curve_ref: Maps a ledger curve key to ``(object, channel label)``;
                a ``None`` answer (or no callable) leaves that curve's claims
                behind -- the claim is about a key the far side cannot find.
            objects: The exported set, in the producer's own naming.  When
                given, membership, hidden / pinned lists, ledger claims and
                channels are scoped to it (compared through *spell*, so long
                and short forms agree); shots themselves always travel -- a
                bound is a scene fact, not a member's.
            channels: :attr:`CHANNELS_KEY`'s payload in the producer's own
                naming (the DCC's ``RenderEffects.channel_records``).
            audio: :attr:`AUDIO_KEY`'s payload, scene-wide.

        Returns:
            The section, or ``None`` when there is nothing to say (no shots,
            markers, channels or audio).
        """
        spell = spell or cls._identity
        state = state or {}
        has_content = state.get("shots") or state.get("markers")
        if not has_content and not channels and not audio:
            return None
        scope = None
        if objects is not None:
            scope = {spell(str(o)) for o in objects}

        def keep(name: str) -> bool:
            return scope is None or spell(str(name)) in scope

        def spell_list(names: Iterable[str]) -> List[str]:
            return cls._unique(spell(str(n)) for n in names or [] if keep(n))

        store: Dict[str, Any] = {}
        for key, value in state.items():
            if key == "edit_ledger":
                continue
            store[key] = deepcopy(value)
        store["shots"] = [
            {
                **shot,
                "objects": spell_list(shot.get("objects") or []),
                "metadata": cls._spell_metadata(shot.get("metadata") or {}, spell),
            }
            for shot in state.get("shots") or []
        ]
        for key in cls.OBJECT_LIST_KEYS:
            store[key] = spell_list(state.get(key) or [])
        store["markers"] = [dict(m) for m in state.get("markers") or []]

        ledger: Dict[str, Dict[str, Dict[str, List[list]]]] = {
            reg: {} for reg in cls._REGISTERS
        }
        if curve_ref is not None:
            # Canonical records (a legacy bare time list becomes an owner-less claim).
            canon = ShotEditLedger.from_dict(state.get("edit_ledger")).to_dict()
            for reg in cls._REGISTERS:
                for curve, records in (canon.get(reg) or {}).items():
                    ref = curve_ref(curve)
                    if not ref:
                        continue
                    obj, label = ref
                    if not obj or not label or not keep(obj):
                        continue
                    by_label = ledger[reg].setdefault(spell(str(obj)), {})
                    by_label.setdefault(str(label), []).extend(
                        [list(r) for r in records]
                    )
        section: Dict[str, Any] = {
            "version": cls.VERSION,
            "store": store,
            "ledger": ledger,
        }
        spelled: Dict[str, Dict[str, Any]] = {}
        for obj_name, records in (channels or {}).items():
            if not records or not keep(obj_name):
                continue
            spelled[spell(str(obj_name))] = {
                str(label): {
                    "value": rec.get("value"),
                    "keys": [list(k) for k in rec.get("keys") or []],
                }
                for label, rec in records.items()
            }
        if spelled:
            section[cls.CHANNELS_KEY] = spelled
        if audio:
            section[cls.AUDIO_KEY] = [dict(clip) for clip in audio]
        return section

    # ------------------------------------------------------------------ decode
    @classmethod
    def decode(
        cls,
        section: Dict[str, Any],
        *,
        resolve: Optional[Resolve] = None,
        curve_key: Optional[CurveKey] = None,
        key_exists: Optional[KeyExists] = None,
        scene_fps: Optional[float] = None,
        frame_offset: float = 0.0,
        converted: Optional[Callable[[str], bool]] = None,
        write_channels: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        write_audio: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
    ) -> Dict[str, Any]:
        """A store dict (``from_dict`` shape) for a ``shots`` section.

        Parameters:
            section: What :meth:`encode` wrote.
            resolve: Carrier spelling -> live object name; ``None`` drops the
                name from membership and the hidden list (a pinned name is
                kept as spelled -- pinning exists to track what is missing).
                Default: names pass through unchanged.
            curve_key: ``(object, label)`` -> the receiving ledger's curve key;
                ``None`` drops those claims.  Without it no claim travels.
            key_exists: Guards each claim against the receiving curve; a claim
                whose key is not there is dropped.  Default: every claim kept.
            scene_fps: The receiving scene's frame rate.  When it differs from
                the section's ``scene_fps`` every time is rescaled, as
                :meth:`~pythontk.ShotStore.rescale_to_fps` would.
            frame_offset: Added to every time AFTER the rescale (the importer's
                shift, in receiving frames).
            write_channels: ``write_channels(resolved object, records)``:
                lands that object's :attr:`CHANNELS_KEY` records, times
                already on the receiving clock.  Called BEFORE the ledger is
                read, so a claim on a channel this call creates finds its key.
                Default: channels are not landed.
            write_audio: ``write_audio(clips)``: lands the retimed
                :attr:`AUDIO_KEY` clips.  Default: audio is not landed.
            converted: ``converted(resolved object) -> bool``: whether the
                importer put that object through the up-axis crossing, in which
                case its claims' labels go through :meth:`swap_up_axis` before
                *curve_key* sees them (the importers convert root objects; a
                consumer passes "has no parent").  Default: no object was.

        Raises:
            ValueError: The section declares a schema newer than this reader.
        """
        version = int(section.get("version") or 0)
        if version > cls.VERSION:
            raise ValueError(
                f"shots section schema v{version}; this reader accepts up to "
                f"v{cls.VERSION}"
            )
        resolve = resolve or cls._identity
        source = dict(section.get("store") or {})
        ratio = cls._ratio(source.get("scene_fps"), scene_fps)
        offset = float(frame_offset or 0.0)
        snap = bool(source.get("snap_whole_frames", True))

        def resolve_list(names: Iterable[str], keep_missing: bool) -> List[str]:
            out = []
            for name in names or []:
                hit = resolve(str(name))
                if hit:
                    out.append(str(hit))
                elif keep_missing:
                    out.append(str(name))
            return cls._unique(out)

        def resolve_or_keep(name: str) -> str:
            return str(resolve(str(name)) or name)

        # The ledger is rebuilt below from the section's own claims.
        store = cls._on_clock(source, ratio, offset, ledger=False)
        store["shots"] = [
            {
                **shot,
                "objects": resolve_list(shot.get("objects") or [], keep_missing=False),
                "metadata": cls._spell_metadata(
                    shot.get("metadata") or {}, resolve_or_keep
                ),
            }
            for shot in store.get("shots") or []
        ]
        store["hidden_objects"] = resolve_list(
            source.get("hidden_objects") or [], keep_missing=False
        )
        store["pinned_objects"] = resolve_list(
            source.get("pinned_objects") or [], keep_missing=True
        )
        if scene_fps is not None:
            store["scene_fps"] = float(scene_fps)

        # Content first: a ledger claim names a key on a channel these calls
        # may be the ones to create.
        if write_channels is not None:
            for obj_name, records in (section.get(cls.CHANNELS_KEY) or {}).items():
                node = resolve(str(obj_name))
                if not node or not records:
                    continue
                write_channels(
                    str(node),
                    {
                        str(label): {
                            "value": rec.get("value"),
                            "keys": cls._retime(rec.get("keys") or [], ratio, offset),
                        }
                        for label, rec in records.items()
                    },
                )
        if write_audio is not None and section.get(cls.AUDIO_KEY):
            clips = []
            for clip in section[cls.AUDIO_KEY]:
                clip = dict(clip)
                for key in ("start", "end"):
                    if clip.get(key) is not None:
                        clip[key] = cls._frame(clip[key], ratio, offset, snap)
                if clip.get("offset"):
                    clip["offset"] = round(float(clip["offset"]) * ratio, 4)
                clips.append(clip)
            write_audio(clips)

        ledger: Dict[str, Dict[str, List[list]]] = {}
        if curve_key is not None:
            for reg in cls._REGISTERS:
                by_object = (section.get("ledger") or {}).get(reg) or {}
                for obj_name, by_label in by_object.items():
                    node = resolve(str(obj_name))
                    if not node:
                        continue
                    swapped = converted is not None and bool(converted(str(node)))
                    for label, records in (by_label or {}).items():
                        label = cls.swap_up_axis(str(label)) if swapped else str(label)
                        key = curve_key(str(node), label)
                        if not key:
                            continue
                        retimed = cls._retime(records, ratio, offset)
                        if key_exists is not None:
                            retimed = [r for r in retimed if key_exists(key, r[0])]
                        if retimed:
                            ledger.setdefault(reg, {}).setdefault(key, []).extend(
                                retimed
                            )
        # Through the ledger so records are canonical and sorted, as a save is.
        store["edit_ledger"] = ShotEditLedger.from_dict(ledger).to_dict()
        return store

    # ------------------------------------------------------------------- merge
    @classmethod
    def merge(
        cls,
        existing: Optional[Dict[str, Any]],
        incoming: Dict[str, Any],
        id_map: Optional[Dict[int, int]] = None,
    ) -> Dict[str, Any]:
        """*incoming* (a decoded store dict) folded into *existing*'s.

        A scene without shots adopts the incoming store whole -- settings
        included -- which is the 1:1 case (a clean-slate send, a pull into a
        new scene, a reference bake).  A scene that already has shots keeps
        its own settings and clock and gains the incoming shots after its last
        one under fresh ids: memberships, hidden / pinned lists, markers,
        locked gaps and ledger claims come along, with every shot id (a locked
        gap's pair, a claim's owner) remapped to the id the shot was given,
        and every time rescaled onto the scene's clock when the two stores'
        ``scene_fps`` differ -- the DCC lands the other scene's keys at the
        same seconds, so its shots must land there too.

        A merge renames a shot only to resolve a clash: an incoming name that
        exports as one of the scene's shots already does (by clip spelling,
        ignoring case -- two stores can each be unique and still collide) is
        numbered through :meth:`~pythontk.ShotStore.unique_among`, a legal
        name whatever the incoming spelling.  Any other name arrives as it
        was saved, a legacy spelling included, as a store adopted whole
        keeps it.

        *id_map*, when given, is filled with each incoming shot id -> the id
        it has in the result (identity when the store is adopted whole).
        """
        if not incoming.get("shots") and not incoming.get("markers"):
            # Channels or audio alone: the scene's own store stands as it is.
            return deepcopy(existing) if existing else deepcopy(incoming)
        if not existing or not existing.get("shots"):
            if id_map is not None:
                id_map.update(
                    {
                        int(s["shot_id"]): int(s["shot_id"])
                        for s in incoming.get("shots") or []
                    }
                )
            return deepcopy(incoming)
        out = deepcopy(existing)
        ratio = cls._ratio(incoming.get("scene_fps"), out.get("scene_fps"))
        if ratio != 1.0:
            # This scene's settings stand, its snap policy included: a module
            # saved without snapping must not land fractional bounds here.
            incoming = cls._on_clock(
                incoming, ratio, snap=out.get("snap_whole_frames", True)
            )
        taken = [int(s["shot_id"]) for s in out.get("shots") or []]
        next_id = (max(taken) + 1) if taken else 1
        local_ids: Dict[int, int] = {}
        names = [str(s.get("name") or "") for s in out.get("shots") or []]
        clips = {ShotStore._clip_key(name) for name in names}
        for shot in sorted(
            incoming.get("shots") or [],
            key=lambda s: (s.get("start", 0.0), s["shot_id"]),
        ):
            shot = deepcopy(shot)
            local_ids[int(shot["shot_id"])] = next_id
            shot["shot_id"] = next_id
            next_id += 1
            name = str(shot.get("name") or "")
            if ShotStore._clip_key(name) in clips:
                name = ShotStore.unique_among(name, names)
            shot["name"] = name
            names.append(name)
            clips.add(ShotStore._clip_key(name))
            out["shots"].append(shot)
        for key in cls.OBJECT_LIST_KEYS:
            out[key] = cls._unique(
                list(out.get(key) or []) + list(incoming.get(key) or [])
            )
        out["markers"] = list(out.get("markers") or []) + [
            dict(m) for m in incoming.get("markers") or []
        ]
        gaps = [list(pair) for pair in out.get("locked_gaps") or []]
        for left, right in incoming.get("locked_gaps") or []:
            if int(left) in local_ids and int(right) in local_ids:
                gaps.append([local_ids[int(left)], local_ids[int(right)]])
        out["locked_gaps"] = gaps
        # Canonical first (fresh lists): the ledger loads its legacy bare-time
        # records as unowned samples, so the merge takes them too.
        ledger = ShotEditLedger.from_dict(out.get("edit_ledger")).to_dict()
        incoming_ledger = ShotEditLedger.from_dict(
            incoming.get("edit_ledger")
        ).to_dict()
        for reg in cls._REGISTERS:
            target = ledger.setdefault(reg, {})
            for curve, records in (incoming_ledger.get(reg) or {}).items():
                have = target.setdefault(curve, [])
                for rec in records:
                    if reg == "keys" and len(rec) > 1:
                        try:
                            rec[1] = local_ids.get(int(rec[1]), rec[1])
                        except (TypeError, ValueError):
                            pass
                    if rec not in have:
                        have.append(rec)
            if not target:
                ledger.pop(reg, None)
        out["edit_ledger"] = ledger
        if id_map is not None:
            id_map.update(local_ids)
        return out

    # ---------------------------------------------------- the record's section
    @classmethod
    def section_out(cls, state: Dict[str, Any], ctx: Any) -> Optional[Dict[str, Any]]:
        """The ``shots`` section for a store crossing with no DCC owner
        (``RecordTransfer.sections``): :meth:`encode` in the carrier's
        spelling, scoped to what ships -- the store alone, since no DCC is
        there to add channels or audio.  The section's shape is this codec's
        (what every consumer :meth:`decode`\\ s), so the bare record never
        rides under its name."""
        return cls.encode(state, spell=ctx.spell, objects=ctx.objects)

    @classmethod
    def section_in(cls, section: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
        """A store dict for a received ``shots`` section with no DCC owner to
        land it (``RecordTransfer.receive``): :meth:`decode`, names resolved
        through ``ctx.rename`` and the importer's ``frame_offset`` /
        ``converted`` adapters when it carries them.  The clock is the
        merge's (:meth:`merge_record` rescales onto this scene's)."""
        return cls.decode(
            section,
            resolve=ctx.rename,
            frame_offset=float(ctx.adapter("frame_offset", 0.0) or 0.0),
            converted=ctx.adapter("converted"),
        )

    @classmethod
    def merge_record(
        cls,
        own: Optional[Dict[str, Any]],
        other: Optional[Dict[str, Any]],
        ctx: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """The ``shot_store`` record's merge (``SceneRecords.CODECS``).

        :meth:`merge`, for another scene's store arriving beside this one's --
        a referenced module whose reference is imported, a hand-off landing
        in a scene with shots.  *other* is already respelled for this scene.
        The renumbering is published as ``ctx.remaps["shot_id"]`` (the key
        stash follows its clips' source shots through it); a shot numbered to
        resolve a clash, and a rescale onto this scene's clock, are noted --
        nothing changes silently.
        """
        if not other:
            return own
        id_map: Dict[int, int] = {}
        merged = cls.merge(own, other, id_map=id_map)
        if ctx is not None:
            ctx.remaps["shot_id"] = dict(id_map)
            by_id = {int(s["shot_id"]): s for s in merged.get("shots") or []}
            for shot in other.get("shots") or []:
                arrived = by_id.get(id_map.get(int(shot["shot_id"])))
                if arrived is not None and arrived.get("name") != shot.get("name"):
                    ctx.note(
                        f"Shot {shot.get('name')!r} arrives as "
                        f"{arrived.get('name')!r} (one of this scene's shots "
                        "already exports under that name)."
                    )
            shots = other.get("shots") or []
            ratio = cls._ratio(other.get("scene_fps"), (own or {}).get("scene_fps"))
            if shots and (own or {}).get("shots") and ratio != 1.0:
                ctx.note(
                    f"Shots: {len(shots)} shot(s) rescaled from "
                    f"{float(other['scene_fps']):g} to {float(own['scene_fps']):g} fps."
                )
        return merged

    @classmethod
    def respell_record(cls, state: Optional[Dict[str, Any]], ctx: Any) -> Any:
        """*state* (a store dict another scene saved) with the names it holds
        put through ``ctx.spell`` -- the ``shot_store`` record's respelling
        (``RecordTransfer.respell``).

        Exactly the fields :meth:`encode` spells: each shot's members and
        object metadata, the hidden / pinned lists, and the ledger's curve
        keys (the ledger canonicalised first, so a legacy shape respells
        too).  Shot names, markers and every other string stay as they are:
        a shot named like a node the crossing renamed keeps its name.
        """
        if not state:
            return state
        spell = ctx.spell
        out = deepcopy(state)
        out["shots"] = [
            {
                **shot,
                "objects": [spell(str(o)) for o in shot.get("objects") or []],
                "metadata": cls._spell_metadata(shot.get("metadata") or {}, spell),
            }
            for shot in out.get("shots") or []
        ]
        for key in cls.OBJECT_LIST_KEYS:
            if key in out:
                out[key] = [spell(str(n)) for n in out.get(key) or []]
        if out.get("edit_ledger"):
            canon = ShotEditLedger.from_dict(out["edit_ledger"]).to_dict()
            out["edit_ledger"] = {
                reg: (
                    {spell(str(curve)): recs for curve, recs in claims.items()}
                    if isinstance(claims, dict)
                    else claims
                )
                for reg, claims in canon.items()
            }
        return out
