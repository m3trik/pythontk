# coding=utf-8
"""Behaviors — load JSON keying recipes and resolve them to keyframe math.

A behavior template defines attribute keyframe patterns (e.g. fade-in,
fade-out) anchored to a time range's start or end.  Shared across all
tools in the ``shots`` engine.

This is the **pure** half of the behavior system: template discovery,
loading, schema validation, the anchor/offset/duration → absolute keyframe
math (:func:`resolve_keys`), duration summation (:func:`compute_duration`),
and the build loop that applies a store's behaviors through injected scene
callables (:func:`apply_to_shots`).  The scene-touching appliers
(``apply_behavior``, ``verify_behavior``, ``apply_audio_clip``) live in the DCC
toolkits, which import this module for the pure core.

A template that names an ``effect`` (the built-in fades, highlight and clip)
is keyed from the scene's :class:`~pythontk.EffectRecipe` -- the one the
Render Effects / Audio Clips panels edit -- so a Build keys the same fade or
pulse an artist keys by hand; :func:`keyed` states such a template's channels
and envelope as ``attributes`` for every reader that does not key.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pythontk import Codec, TemplateSet

from pythontk.core_utils.engines.shots.effect_recipe import EffectRecipe
from pythontk.core_utils.engines.shots.manifest.behaviors._spec import BehaviorSpec

# Log under the package name, not this private impl module, so the logger name
# stays stable across the __init__ -> _behaviors split (user log filters and
# tests key on ``...manifest.behaviors``).
log = logging.getLogger(__name__.rpartition(".")[0])

_BEHAVIORS_DIR = Path(__file__).parent


class _BehaviorsInternal(object):
    """Internal helpers for Behaviors."""

    @staticmethod
    def _json_codec() -> Codec:
        """JSON codec for the behavior store.

        Behavior templates ship as ``.json`` (stdlib ``json`` — zero-dependency, and
        loadable in every DCC's bundled Python; Blender's, unlike Maya's, ships no
        PyYAML).  The mapping templates use JSON for the same reason.
        """
        return Codec(
            ext=".json",
            load=json.loads,
            dump=lambda data: json.dumps(data, indent=2),
        )

    @staticmethod
    @functools.lru_cache(maxsize=64)
    def _load_behavior_validated(name: str, tier: str, stamp: int) -> Dict[str, Any]:
        """Two-tier store load + schema-validate, cached per resolved FILE.

        Loads through the :class:`~pythontk.TemplateSet` store — its codec and
        sanitized path match the ``source``/``names`` lookup, so there's no
        hand-rolled re-read or raw-name mismatch — then validates like
        :func:`load_mapping`: hard errors raise ``SchemaError`` with a precise
        message; unknown keys are logged, not fatal.  Cached so a build that
        re-requests the same behavior once per object reads + validates only
        once; *tier* and *stamp* (the file's mtime) only key the cache, so a
        user template added over a loaded built-in, or edited, is read again --
        keyed by name alone, either waited for a restart.
        """
        data = Behaviors.templates().raw(name)
        BehaviorSpec.validate(data).raise_or_warn(
            prefix=f"behavior {name!r}: ", logger=log
        )
        return data

    @staticmethod
    def _binds(fn, *args, **kwargs) -> bool:
        """True when *fn* accepts the given call shape (no call is made).

        Callables without an introspectable signature (some builtins, C
        extensions, mocks) are assumed to accept the full modern contract.
        """
        try:
            inspect.signature(fn).bind(*args, **kwargs)
            return True
        except TypeError:
            return False
        except ValueError:
            return True

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def _load_behavior_cached(name: str, base: Path) -> Dict[str, Any]:
        """Single-directory cached loader (explicit ``search_path`` / back-compat).

        Arguments must be hashable."""
        path = base / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Behavior template not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))


class Behaviors(_BehaviorsInternal):
    """Load JSON keying recipes and resolve them to keyframe math.

    A behavior template names an attribute keyframe pattern (fade-in,
    fade-out, ...) anchored to the start or end of a time range; this
    resolves one to concrete key times and values.
    """

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def templates() -> TemplateSet:
        """The shared :class:`~pythontk.TemplateSet` backing behavior discovery.

        A cached singleton (built on first use, so importing this module touches the
        filesystem lazily). Built-in behaviors in :data:`_BEHAVIORS_DIR` (read-only)
        plus the user's own under ``user_config_root()/shots/manifest_behaviors/`` (a
        user file shadows a built-in of the same name) — the same machinery the CSV
        mappings use, so both extend, validate, and document identically.
        """
        return TemplateSet(
            "manifest_behaviors",
            BehaviorSpec,
            "shots",
            builtin_dir=_BEHAVIORS_DIR,
            codec=_BehaviorsInternal._json_codec(),
        )

    @staticmethod
    def load_behavior(name: str, search_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load a JSON behavior template by stem name.

        Results are cached per ``(name, search_path)`` pair so repeated
        lookups (e.g. many objects sharing the same behavior within one
        build) avoid redundant disk I/O and JSON parsing.

        Parameters:
            name: Template name without extension (e.g. ``"fade_in"``).
            search_path: Directory to search. When given, only that directory is
                used (back-compat / tests). When omitted, the two-tier set is used:
                a user template shadows a built-in of the same name.

        Returns:
            Parsed template dict.

        Raises:
            FileNotFoundError: If the template file does not exist.
        """
        if search_path is not None:
            return _BehaviorsInternal._load_behavior_cached(name, Path(search_path))
        templates = Behaviors.templates()
        tier = templates.source(name)
        if tier is None:
            raise FileNotFoundError(f"Behavior template not found: {name}")
        try:
            stamp = templates.path(name, tier).stat().st_mtime_ns
        except OSError:
            stamp = 0
        return _BehaviorsInternal._load_behavior_validated(name, tier, stamp)

    @staticmethod
    def phrase_table() -> Dict[str, List[Any]]:
        """``{name: [compiled detect phrases]}`` for every loadable template
        (in name order; a template without phrases maps to ``[]``) -- built
        once and handed to :meth:`detect` / :meth:`from_cell` by a caller that
        reads many texts, such as a sheet parse."""
        import re

        table: Dict[str, List[Any]] = {}
        for name in Behaviors.list_behaviors():
            try:
                phrases = Behaviors.load_behavior(name).get("detect") or ()
            except (FileNotFoundError, ValueError):
                continue  # an invalid template is skipped, never fatal here
            table[name] = [re.compile(p, re.I | re.M) for p in phrases]
        return table

    #: Words a doc puts before an object that are not its name
    #: (``"Highlight the 2 screws"`` names ``screws``).
    _DETERMINERS = r"(?:the|a|an|both|all|each|\d+)\s+"

    #: A subject longer than this many words is a sentence a phrase over-read,
    #: not an object's name -- dropped rather than listed.
    SUBJECT_MAX_WORDS = 5

    @staticmethod
    def subjects(
        text: str, table: Optional[Dict[str, List[Any]]] = None
    ) -> List[Tuple[str, str]]:
        """``[(subject, behavior)]`` -- the objects *text* names by a behavior.

        A template's ``detect`` phrase that captures a named group ``object``
        names the thing the behavior is for (``"^highlight\\s+(?P<object>.+)"``
        reads ``"Highlight Red Door"`` as ``("Red Door", "highlight")``); a
        phrase without the group only detects.  Leading determiners
        (``the``, ``both``, ``2``) and trailing punctuation are trimmed, and a
        capture over :attr:`SUBJECT_MAX_WORDS` words is dropped.  Pairs come
        in template-name order, each once.

        Parameters:
            text: One line or cell of a doc's prose.
            table: A :meth:`phrase_table` to reuse (built here when omitted).
        """
        import re

        if not text:
            return []
        table = Behaviors.phrase_table() if table is None else table
        lead = re.compile(f"^(?:{Behaviors._DETERMINERS})+", re.I)
        found: List[Tuple[str, str]] = []
        for name, rxs in table.items():
            for rx in rxs:
                if "object" not in rx.groupindex:
                    continue
                for match in rx.finditer(text):
                    subject = " ".join((match.group("object") or "").split())
                    subject = lead.sub("", subject).strip(" .,;:!?'\"")
                    words = len(subject.split())
                    if 0 < words <= Behaviors.SUBJECT_MAX_WORDS:
                        if (subject, name) not in found:
                            found.append((subject, name))
        return found

    @staticmethod
    def detect(text: str, table: Optional[Dict[str, List[Any]]] = None) -> List[str]:
        """Behaviors whose template ``detect`` phrases occur in *text*.

        Each template is tested independently, in name order, so text naming
        both "fades in" and "fades out" yields ``["fade_in", "fade_out"]``.
        A user template brings its own phrases.  *table* is a
        :meth:`phrase_table` to reuse (built here when omitted).
        """
        if not text:
            return []
        table = Behaviors.phrase_table() if table is None else table
        return [name for name, rxs in table.items() if any(r.search(text) for r in rxs)]

    @staticmethod
    def from_cell(cell: str, table: Optional[Dict[str, List[Any]]] = None) -> List[str]:
        """Behaviors listed in one behaviors-column cell, in order.

        Items split on commas, semicolons and lines.  Each is a template name
        (``"Fade In"`` -> ``fade_in``) or, failing that, phrases a template
        detects (``"fades away"`` -> ``fade_out``).  An item matching neither
        is kept, spelled as a name, so Assess can report it rather than lose
        it.
        """
        import re

        table = Behaviors.phrase_table() if table is None else table
        out: List[str] = []
        for item in re.split(r"[,;\n]", cell or ""):
            item = item.strip()
            if not item:
                continue
            name = re.sub(r"[^0-9a-z]+", "_", item.lower()).strip("_")
            names = [name] if name in table else Behaviors.detect(item, table) or [name]
            out.extend(n for n in names if n not in out)
        return out

    @staticmethod
    def list_behaviors(
        search_path: Optional[Path] = None, kind: Optional[str] = None
    ) -> List[str]:
        """Return stem names of all available behavior templates.

        Parameters:
            search_path: Directory to scan. When omitted, the two-tier set is used
                (built-in + user templates, unioned). When given, only that folder
                is scanned (back-compat / tests).
            kind: When provided, only return behaviors whose ``kind`` list
                includes this value (e.g. ``"scene"`` or ``"audio"``).
                Templates without a ``kind`` key default to ``["scene"]``.
        """
        if search_path is not None:
            base = Path(search_path)
            names = sorted(p.stem for p in base.glob("*.json")) if base.is_dir() else []

            def _load(n: str) -> Dict[str, Any]:
                return Behaviors.load_behavior(n, base)

        else:
            names = Behaviors.templates().names()

            def _load(n: str) -> Dict[str, Any]:
                return Behaviors.load_behavior(n)

        if kind is None:
            return names
        result = []
        for name in names:
            try:
                tmpl = _load(name)
            except (FileNotFoundError, ValueError) as exc:
                # A missing or invalid template (load_behavior now schema-validates,
                # raising SchemaError — a ValueError — on a bad one) must not hide
                # every valid behavior from the picker. Skip it here; validation
                # still raises at apply time so the user gets a precise error then.
                log.warning("Skipping behavior %r in listing: %s", name, exc)
                continue
            if kind in tmpl.get("kind", ["scene"]):
                result.append(name)
        return result

    @staticmethod
    def resolve_keys(
        block_def: Dict,
        start: float,
        end: float,
    ) -> List[Dict[str, Any]]:
        """Resolve an ``in`` or ``out`` block to absolute keyframe dicts.

        Parameters:
            block_def: Dict with ``offset``, ``duration``, ``values``,
                and optionally ``tangent`` and ``anchor``.
            start: First frame of the target range.
            end: Last frame of the target range.

        The ``anchor`` value may be:

        - ``"start"`` — place the block at the beginning of the range.
        - ``"end"`` — place the block at the end of the range.
        - A **float** between 0.0 and 1.0 — interpolate linearly between
          the start and end positions.  ``0.0`` is equivalent to
          ``"start"`` and ``1.0`` to ``"end"``.

        Returns:
            List of ``{"time": float, "value": float, "tangent": str}`` dicts.
        """
        anchor = block_def.get("anchor", "start")
        offset = block_def.get("offset", 0)
        dur = block_def.get("duration", 0)
        values = block_def.get("values", [])
        tangent = block_def.get("tangent", "linear")

        if isinstance(anchor, (int, float)) and not isinstance(anchor, bool):
            # Fractional anchor: interpolate between the anchored endpoints so
            # 0.0 is exactly the "start" placement (start + offset) and 1.0 is
            # exactly the "end" placement (end - dur - offset) — including the
            # offset's sign, which flips between the two ends.
            start_pos = start + offset
            end_pos = end - dur - offset
            base = start_pos + anchor * (end_pos - start_pos)
        elif anchor == "end":
            base = end - dur - offset
        else:
            base = start + offset

        n = len(values)
        keys = []
        for i, v in enumerate(values):
            t = base + (dur * i / max(n - 1, 1))
            keys.append({"time": t, "value": v, "tangent": tangent})
        return keys

    @staticmethod
    def effect_of(behavior: Any) -> Optional[str]:
        """The recipe effect a behavior keys (``"fade_in"``, ``"pulse"``, ...),
        or ``None`` for a template keying its own ``attributes`` -- or a name
        no template answers.

        Parameters:
            behavior: A template name or a loaded template dict.
        """
        if isinstance(behavior, str):
            try:
                behavior = Behaviors.load_behavior(behavior)
            except (FileNotFoundError, ValueError):
                return None
        effect = (behavior or {}).get("effect")
        return effect if effect in EffectRecipe.EFFECTS else None

    @staticmethod
    def keyed(
        behavior: Any,
        recipe: Optional[EffectRecipe] = None,
        fps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """A template as it is keyed: an ``effect`` stated as ``attributes``.

        Every reader that sizes, verifies or lists a behavior's channels speaks
        ``attributes`` blocks; an effect template has none of its own (the
        recipe keys it), so this states the effect's channel and envelope in
        that shape (:meth:`EffectRecipe.envelope`). A template that keys its
        own ``attributes`` comes back as it is.

        Parameters:
            behavior: A template name or a loaded template dict.
            recipe: The scene's recipe (the defaults when omitted -- fine for
                a reader of the SHAPE: channels, phases).
            fps: The scene's rate, for the pulse's lengths.

        Raises:
            FileNotFoundError: A name no template answers.
        """
        tmpl = (
            Behaviors.load_behavior(behavior)
            if isinstance(behavior, str)
            else dict(behavior or {})
        )
        effect = Behaviors.effect_of(tmpl)
        if effect is None:
            return tmpl
        recipe = recipe or EffectRecipe()
        return dict(
            tmpl, attributes=recipe.envelope(effect, Behaviors.place_of(tmpl), fps)
        )

    @staticmethod
    def place_of(tmpl: Dict[str, Any]) -> Any:
        """Where an effect template puts its effect: its ``place``, else where
        the effect naturally goes (a fade out at the end, the rest at the
        start)."""
        place = (tmpl or {}).get("place")
        if place is None:
            place = "end" if tmpl.get("effect") == "fade_out" else "start"
        return place

    @staticmethod
    def phase_durations(tmpl: Dict[str, Any]) -> Tuple[float, float]:
        """Sum a template's ``in`` / ``out`` phase durations across all attributes.

        The single source of the phase-walk math shared by
        :func:`compute_duration` and the engine's ``resolve_duration`` — an
        object's minimum content length is ``in_total + out_total`` laid out
        without overlap.

        Parameters:
            tmpl: A behavior template as :meth:`keyed` states it -- an
                ``effect`` template has no phases of its own, so a raw one
                sums to ``(0.0, 0.0)``.

        Returns:
            ``(in_total, out_total)`` in frames.
        """
        d_in = 0.0
        d_out = 0.0
        for attr_def in tmpl.get("attributes", {}).values():
            for phase in ("in", "out"):
                block = attr_def.get(phase)
                if not block:
                    continue
                d = float(block.get("duration", 0) or 0)
                if phase == "in":
                    d_in += d
                else:
                    d_out += d
        return d_in, d_out

    @staticmethod
    def anchor_overrides(behaviors: List[str]) -> List[Optional[float]]:
        """The ``anchor_override`` each of one object's *behaviors* is keyed at.

        A **point** behavior (blocks in one phase: ``fade_in``, ``fade_out``)
        is placed by its order among the object's points, spread ``0.0 .. 1.0``
        over the range, so "fades in, out, then in again" reads in order. A
        **span** (an ``in`` AND an ``out`` block: ``highlight``) covers the
        range from both ends and keeps its own anchors: forcing both blocks to
        one point collapsed them onto the same frames. ``None`` = the
        template's own anchor -- every span, a lone point, and a name no
        template answers (its apply records that failure).

        The one placement rule for the build, the per-object re-apply and
        Assess's ``exact`` check, so a verify never models an anchor the keys
        were not placed at.

        Parameters:
            behaviors: One object's behavior names, in doc order.

        Returns:
            One override per name, aligned with *behaviors*.
        """
        is_point = []
        for name in behaviors:
            try:
                tmpl = Behaviors.keyed(name)
            except (FileNotFoundError, ValueError):
                is_point.append(False)
                continue
            phases = {
                phase
                for attr_def in (tmpl.get("attributes") or {}).values()
                for phase in ("in", "out")
                if attr_def.get(phase)
            }
            is_point.append(len(phases) == 1)
        points = sum(is_point)
        overrides: List[Optional[float]] = []
        index = 0
        for point in is_point:
            if point and points > 1:
                overrides.append(index / (points - 1))
                index += 1
            else:
                overrides.append(None)
        return overrides

    @staticmethod
    def compute_duration(
        behavior_entries: List[Dict[str, str]],
        fallback: float = 30,
        fps: Optional[float] = None,
        audio_duration_fn: Optional[Callable[[str], Optional[float]]] = None,
        resolve_source_fn: Optional[Callable[[str, str], Optional[str]]] = None,
        recipe: Optional[EffectRecipe] = None,
    ) -> float:
        """Derive duration from the behavior templates referenced in *behavior_entries*.

        For each entry, the durations of all its behaviors are summed
        (since all get applied to the same object).  Audio templates whose
        ``duration`` field is the string ``"from_source"`` are resolved by
        calling *audio_duration_fn* with the entry's source key — so an audio
        shot is sized to the full clip length.  The result is the maximum
        across all entries.

        Parameters:
            behavior_entries: List of dicts with a ``"behavior"`` key, or
                ``BuilderObject``-like objects with a ``.behaviors`` list
                and optional ``.kind`` / ``.source_path`` attributes.
            fallback: Duration when no behavior-driven duration exists.
            fps: The scene's rate: an effect template's lengths are the
                recipe's, the pulse's in seconds (:meth:`keyed`). Audio is
                not resolved here -- a DCC layer binds its rate into
                *audio_duration_fn* (via closure).
            audio_duration_fn: Optional callable ``(source_key) -> Optional[float]``
                returning a clip length in frames for ``"from_source"`` audio
                templates.  Injected by the DCC layer (which wraps its own audio
                duration measurement).  When ``None`` — or when it
                returns ``None`` / a non-positive value — the ``from_source`` entry
                contributes nothing, so an all-audio manifest with no resolver
                falls back to *fallback*.
            resolve_source_fn: Optional callable ``(entry_name, entry_kind) ->
                Optional[str]`` returning a source key for an entry whose own
                ``source_path`` is empty.  Injected by a DCC layer whose audio
                system can resolve a registered track's path from the entry name
                (e.g. Maya's ``audio_clips`` tracks, populated independently of
                the manifest CSV).  ``None`` skips the fallback.
            recipe: The scene's effect recipe (the defaults when omitted).

        Returns:
            Duration in frames.
        """
        max_dur = 0.0
        has_any = False
        # Phase-layout tracking: when different objects carry start-anchored
        # ("in") and end-anchored ("out") behaviors, the shot must be long
        # enough for both phases laid out sequentially.
        global_max_in = 0.0
        global_max_out = 0.0

        for entry in behavior_entries:
            # Support both dict format {"behavior": "name"} and
            # BuilderObject with .behaviors list
            if isinstance(entry, dict):
                behaviors = [entry.get("behavior", "")]
                source_path = entry.get("source_path", "") or ""
                entry_name = entry.get("name", "") or ""
                entry_kind = entry.get("kind", "") or ""
            else:
                behaviors = getattr(entry, "behaviors", [])
                source_path = getattr(entry, "source_path", "") or ""
                entry_name = getattr(entry, "name", "") or ""
                entry_kind = getattr(entry, "kind", "") or ""

            # Source fallback: an entry with no source_path may still have a
            # resolvable source via the DCC's own registry (see resolve_source_fn).
            if not source_path and entry_name and resolve_source_fn is not None:
                try:
                    source_path = resolve_source_fn(entry_name, entry_kind) or ""
                except Exception as exc:
                    log.debug("source fallback failed for '%s': %s", entry_name, exc)

            obj_total = 0.0
            obj_in = 0.0
            obj_out = 0.0
            for behavior in behaviors:
                if not behavior:
                    continue
                try:
                    tmpl = Behaviors.keyed(behavior, recipe, fps)
                except FileNotFoundError:
                    continue

                dur_field = tmpl.get("duration")
                if dur_field == "from_source":
                    # An unresolvable source leaves has_any unchanged so the
                    # caller falls back instead of collapsing the shot to 0.
                    if not source_path or audio_duration_fn is None:
                        continue
                    try:
                        dur_frames = audio_duration_fn(source_path)
                    except Exception as exc:
                        log.debug("from_source duration probe failed: %s", exc)
                        continue
                    if dur_frames is None or dur_frames <= 0:
                        continue
                    obj_total += float(dur_frames)
                    has_any = True
                    continue

                has_any = True
                d_in, d_out = Behaviors.phase_durations(tmpl)
                obj_total += d_in + d_out
                obj_in += d_in
                obj_out += d_out
            if obj_total > max_dur:
                max_dur = obj_total
            global_max_in = max(global_max_in, obj_in)
            global_max_out = max(global_max_out, obj_out)
        if not has_any:
            return fallback
        # Ensure the duration accommodates both start-anchored and
        # end-anchored behaviors laid out without overlap.
        phase_total = global_max_in + global_max_out
        return max(max_dur, phase_total)

    @staticmethod
    def apply_to_shots(
        shots: list,
        apply_fn: Callable,
        exists_fn: Optional[Callable] = None,
        has_keys_fn: Optional[Callable] = None,
        store: Any = None,
        resolve_fn: Optional[Callable[[str], str]] = None,
        conflict_fn: Optional[Callable] = None,
        release_fn: Optional[Callable] = None,
    ) -> Dict[str, list]:
        """Apply declared behaviors from shot metadata through *apply_fn*.

        The build loop both DCCs run -- each host binds only what it alone can
        answer (*exists_fn*, *has_keys_fn*; their ``Behaviors.apply_to_shots``
        pass the host's defaults). Reads ``metadata["behaviors"]`` from each
        shot; locked and zero-length shots are never touched.

        Audio-grow (expanding ``shot.end`` to fit audio clips and rippling
        downstream shots) is handled upstream by ``ShotManifest._compute_plan``
        / ``_execute_plan``: ``shot.start`` / ``shot.end`` are final here.

        Two passes per shot:

        1. **Audio** -- clips first, so the shot range is final before the
           other behaviors compute positional anchors. A clip already placed
           (*has_keys_fn*) is left; otherwise the keys it wrote last time are
           released (*release_fn*) and it is placed again.
        2. **Everything else** over the finalized range, each placed by
           :meth:`anchor_overrides`. Every guard is settled before anything is
           keyed, and every behavior's previous keys are released before any is
           re-keyed (released one at a time, a later release deleted the key an
           earlier behavior had just written on a shared frame).

        Parameters:
            shots: ``ShotBlock``-like instances to process.
            apply_fn: ``(obj, behavior, start, end)`` keying one behavior and
                returning the ``(curve, time)`` keys it wrote. May accept
                ``source_path`` and ``anchor_override`` keywords -- detected
                once by signature and the richest form used.
            exists_fn: ``(name[, entry]) -> bool`` -- whether an object (or an
                audio entry) can be keyed. Default: everything exists.
            has_keys_fn: ``(obj, start, end[, entry]) -> bool`` -- keys already
                in the range (for audio: the clip already placed). Default:
                nothing is keyed.
            store: Accepted for the DCC signature; unused.
            resolve_fn: ``(doc_name) -> node`` mapping a metadata name to the
                scene node to key (a namespaced reference); identity when
                omitted. Ledger and result records keep the doc name.
            conflict_fn: ``(node, behavior, start, end) -> bool``: True when
                keys the system did not write sit on the channels *behavior*
                keys (the manifest's ledger check). Replaces *has_keys_fn* for
                non-audio entries when given, so an object animated on other
                channels still gets its fade.
            release_fn: ``(shot, doc_name, behavior)`` run before a behavior is
                applied: takes out the keys it wrote last time. A behavior
                whose release raises is recorded failed and not applied.

        Returns:
            ``{"applied", "skipped", "failed"}`` lists of ``{object, behavior,
            shot, shot_id}`` records; an applied one carries the ``keys`` its
            applier reported, a failed one its ``error``. A failing entry (a
            locked channel, a template no file answers) is recorded and the
            batch goes on.
        """

        def _is_audio(entry):
            return (entry.get("kind") == "audio") or bool(entry.get("source_path"))

        if exists_fn is None:

            def exists_fn(name, entry=None):
                return True

        if has_keys_fn is None:

            def has_keys_fn(name, start, end, entry=None):
                return False

        # Signature support is probed once via ``inspect`` binding -- calling
        # inside ``except TypeError`` conflated "wrong signature" with genuine
        # TypeErrors raised *inside* the callable, silently re-invoking it
        # with reduced arguments and masking real failures as successes.
        binds = _BehaviorsInternal._binds
        exists_takes_entry = binds(exists_fn, "", None)
        has_keys_takes_entry = binds(has_keys_fn, "", 0.0, 0.0, None)
        apply_takes_anchor = binds(
            apply_fn, "", "", 0.0, 0.0, source_path="", anchor_override=0.0
        )
        apply_takes_source = binds(apply_fn, "", "", 0.0, 0.0, source_path="")

        def _call_exists(obj_name, entry):
            if exists_takes_entry:
                return exists_fn(obj_name, entry)
            return exists_fn(obj_name)

        def _call_has_keys(obj_name, start, end, entry):
            if has_keys_takes_entry:
                return has_keys_fn(obj_name, start, end, entry)
            return has_keys_fn(obj_name, start, end)

        if resolve_fn is None:

            def resolve_fn(name):
                return name

        def _call_apply(node, behavior, shot, source_path, anchor):
            """The richest form *apply_fn* takes; returns what it reports keying."""
            if anchor is not None and apply_takes_anchor:
                return apply_fn(
                    node,
                    behavior,
                    shot.start,
                    shot.end,
                    source_path=source_path,
                    anchor_override=anchor,
                )
            if apply_takes_source:
                return apply_fn(
                    node, behavior, shot.start, shot.end, source_path=source_path
                )
            return apply_fn(node, behavior, shot.start, shot.end)

        applied: list = []
        skipped: list = []
        failed: list = []

        def _record_failure(obj_name, behavior, shot, exc):
            log.warning(
                "Behavior '%s' on '%s' (shot %s) failed: %s",
                behavior,
                obj_name,
                shot.name,
                exc,
            )
            failed.append(
                {
                    "object": obj_name,
                    "behavior": behavior,
                    "shot": shot.name,
                    "error": str(exc),
                }
            )

        def _release(shot, name, behavior) -> bool:
            if release_fn is None:
                return True
            try:
                release_fn(shot, name, behavior)
            except Exception as exc:
                _record_failure(name, behavior, shot, exc)
                return False
            return True

        for shot in shots:
            if shot.locked:
                continue  # user-finalized -- never modified
            if abs(shot.end - shot.start) < 1e-6:
                continue  # nothing to key over

            entries = shot.metadata.get("behaviors", [])

            # Pass 1 -- audio entries first, so the shot range is final before
            # the non-audio behaviors compute positional anchors.
            for entry in entries:
                obj_name = entry.get("name", "")
                behavior = entry.get("behavior", "")
                if not behavior or not obj_name or not _is_audio(entry):
                    continue
                if not _call_exists(obj_name, entry):
                    continue
                rec = {
                    "object": obj_name,
                    "behavior": behavior,
                    "shot": shot.name,
                    "shot_id": shot.shot_id,
                }
                if _call_has_keys(obj_name, shot.start, shot.end, entry):
                    skipped.append(rec)  # clip already placed
                    continue
                if not _release(shot, obj_name, behavior):
                    continue
                try:
                    written = _call_apply(
                        obj_name, behavior, shot, entry.get("source_path") or "", 0.0
                    )
                except Exception as exc:
                    _record_failure(obj_name, behavior, shot, exc)
                    continue
                rec["keys"] = list(written or [])
                applied.append(rec)

            # Pass 2 -- non-audio entries over the finalized range.
            non_audio = [
                e
                for e in entries
                if e.get("name") and e.get("behavior") and not _is_audio(e)
            ]
            # Where each behavior lands: Behaviors.anchor_overrides, the one
            # placement rule (points spread over the shot in doc order; a span
            # such as the highlight keeps its own anchors).
            per_obj: Dict[str, List[str]] = {}
            for entry in non_audio:
                per_obj.setdefault(entry["name"], []).append(entry["behavior"])
            anchors = {
                name: iter(Behaviors.anchor_overrides(names))
                for name, names in per_obj.items()
            }
            # Every guard is settled BEFORE anything is keyed: a behavior just
            # applied keys the object, and a check made after it would read
            # those fresh keys as the animator's and skip the object's other
            # behaviors (a "fade_in, fade_out" object only ever got its fade_in).
            nodes: Dict[str, str] = {}
            blocked: Dict[tuple, bool] = {}
            for entry in non_audio:
                name, behavior = entry["name"], entry["behavior"]
                node = nodes.setdefault(name, resolve_fn(name))
                if conflict_fn is not None:
                    blocked[(name, behavior)] = bool(
                        conflict_fn(node, behavior, shot.start, shot.end)
                    )
                elif (name, None) not in blocked:
                    blocked[(name, None)] = bool(
                        _call_has_keys(node, shot.start, shot.end, entry)
                    )

            def _guard(name, behavior):
                return (name, behavior) if conflict_fn is not None else (name, None)

            # Take out the keys each behavior wrote last time -- ALL of them,
            # before any is re-keyed.  One whose release failed is not keyed
            # again (as in pass 1): its old keys are still there.
            unreleased: set = set()
            for entry in non_audio:
                name, behavior = entry["name"], entry["behavior"]
                if not blocked.get(_guard(name, behavior)) and _call_exists(
                    nodes[name], entry
                ):
                    if not _release(shot, name, behavior):
                        unreleased.add((name, behavior))

            for entry in non_audio:
                obj_name, behavior = entry["name"], entry["behavior"]
                node = nodes[obj_name]
                anchor = next(anchors[obj_name])
                if not _call_exists(node, entry):
                    continue  # missing object -- surfaced by assess, not here
                rec = {
                    "object": obj_name,
                    "behavior": behavior,
                    "shot": shot.name,
                    "shot_id": shot.shot_id,
                }
                if blocked.get(_guard(obj_name, behavior)):
                    skipped.append(rec)  # keys the animator owns -- never overwritten
                    continue
                if (obj_name, behavior) in unreleased:
                    continue  # recorded as failed by its release
                try:
                    written = _call_apply(
                        node, behavior, shot, entry.get("source_path") or "", anchor
                    )
                except Exception as exc:
                    # A missing template (an unknown name from a doc column),
                    # a locked or linked channel: recorded, and the build goes on.
                    _record_failure(obj_name, behavior, shot, exc)
                    continue
                rec["keys"] = list(written or [])
                applied.append(rec)

        return {"applied": applied, "skipped": skipped, "failed": failed}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Duration computation
# ---------------------------------------------------------------------------
