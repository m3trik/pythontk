# coding=utf-8
"""Behaviors — load JSON keying recipes and resolve them to keyframe math.

A behavior template defines attribute keyframe patterns (e.g. fade-in,
fade-out) anchored to a time range's start or end.  Shared across all
tools in the ``shots`` engine.

This is the **pure** half of the behavior system: template discovery,
loading, schema validation, and the anchor/offset/duration → absolute
keyframe math (:func:`resolve_keys`) plus duration summation
(:func:`compute_duration`).  The scene-touching appliers (``apply_behavior``,
``verify_behavior``, ``apply_audio_clip``, ``apply_to_shots``) live in the DCC
toolkits, which import this module for the pure core.
"""
import functools
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pythontk import Codec, TemplateSet

from pythontk.core_utils.engines.shots.manifest.behaviors._spec import BehaviorSpec

# Log under the package name, not this private impl module, so the logger name
# stays stable across the __init__ -> _behaviors split (user log filters and
# tests key on ``...manifest.behaviors``).
log = logging.getLogger(__name__.rpartition(".")[0])

_BEHAVIORS_DIR = Path(__file__).parent


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
        codec=_json_codec(),
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


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
        return _load_behavior_cached(name, Path(search_path))
    if templates().source(name) is None:
        raise FileNotFoundError(f"Behavior template not found: {name}")
    return _load_behavior_validated(name)


@functools.lru_cache(maxsize=64)
def _load_behavior_validated(name: str) -> Dict[str, Any]:
    """Two-tier store load + schema-validate, cached per name.

    Loads through the :class:`~pythontk.TemplateSet` store — its codec and
    sanitized path match the ``source``/``names`` lookup, so there's no
    hand-rolled re-read or raw-name mismatch — then validates like
    :func:`load_mapping`: hard errors raise ``SchemaError`` with a precise
    message; unknown keys are logged, not fatal.  Cached so a build that
    re-requests the same behavior once per object reads + validates only once.
    """
    data = templates().raw(name)
    BehaviorSpec.validate(data).raise_or_warn(prefix=f"behavior {name!r}: ", logger=log)
    return data


@functools.lru_cache(maxsize=32)
def _load_behavior_cached(name: str, base: Path) -> Dict[str, Any]:
    """Single-directory cached loader (explicit ``search_path`` / back-compat).

    Arguments must be hashable."""
    path = base / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Behavior template not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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
            return load_behavior(n, base)

    else:
        names = templates().names()

        def _load(n: str) -> Dict[str, Any]:
            return load_behavior(n)

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


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Duration computation
# ---------------------------------------------------------------------------


def phase_durations(tmpl: Dict[str, Any]) -> Tuple[float, float]:
    """Sum a template's ``in`` / ``out`` phase durations across all attributes.

    The single source of the phase-walk math shared by
    :func:`compute_duration` and the engine's ``resolve_duration`` — an
    object's minimum content length is ``in_total + out_total`` laid out
    without overlap.

    Parameters:
        tmpl: A loaded behavior template dict (see :class:`BehaviorSpec`).

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


def compute_duration(
    behavior_entries: List[Dict[str, str]],
    fallback: float = 30,
    fps: Optional[float] = None,
    audio_duration_fn: Optional[Callable[[str], Optional[float]]] = None,
    resolve_source_fn: Optional[Callable[[str, str], Optional[str]]] = None,
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
        fps: Reserved for the caller — this pure core does not resolve
            audio itself; a DCC layer binds its frame-rate into
            *audio_duration_fn* (via closure) when it needs one.
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
                tmpl = load_behavior(behavior)
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
            d_in, d_out = phase_durations(tmpl)
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
