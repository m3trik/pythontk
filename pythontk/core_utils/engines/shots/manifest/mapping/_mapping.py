# coding=utf-8
"""CSV mapping resolver — interprets JSON mapping files.

A mapping file is a ``.json`` file that declaratively specifies how
CSV columns map to :class:`BuilderStep` fields and how derived values
(e.g. audio objects) are resolved.

Example ``my_project.json``::

    {
        "columns": {
            "step_id": ["Step"],
            "description": ["Step Contents"],
            "assets": ["Asset Names"],
            "audio": ["Voice Support"],
            "exclude_steps": ["SETUP"],
            "exclude_values": {"assets": ["N/A"]},
            "metadata_pass": {"priority": ["Priority"]}
        },
        "audio_resolve": {
            "method": "prefix",
            "directory": "//server/project/audio",
            "extensions": [".wav", ".mp3"]
        },
        "default_behaviors": {
            "audio": ["set_clip"],
            "scene": ["fade_in"]
        }
    }

Supported ``audio_resolve`` methods:

- ``"prefix"``: Match files starting with ``{step_id}_``.
- ``"regex"``: Match files against a regex pattern with
  ``{step_id}`` placeholder.
- ``"map"``: Explicit ``step_id → clip_stem`` lookup table.
- ``"derive"``: Build clip name from ``step_id`` + first N words
  of the audio text (PascalCase).

API
---
:func:`discover` — list available mapping names in a directory.
:func:`load_mapping` — read a JSON mapping and return a callable.
:func:`resolve` — ``(csv_path, mapping_name) → List[BuilderStep]``.
"""
import functools
import json
import logging
import re as _re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from pythontk import TemplateSet

from pythontk.core_utils.engines.shots.manifest.manifest_model import (
    BuilderObject,
    BuilderStep,
    ColumnMap,
    parse_csv,
)
from pythontk.core_utils.engines.shots.manifest.mapping._spec import MappingSpec

# Log under the package name, not this private impl module, so the logger name
# stays stable across the __init__ -> _mapping split.
log = logging.getLogger(__name__.rpartition(".")[0])

__all__ = [
    "DEFAULT_DIR",
    "discover",
    "load_mapping",
    "resolve",
    "templates",
]

DEFAULT_DIR: Path = Path(__file__).parent
"""Directory of built-in (read-only) mapping JSON files shipped with the tool."""


@functools.lru_cache(maxsize=None)
def templates() -> TemplateSet:
    """The shared :class:`~pythontk.TemplateSet` backing mapping discovery.

    A cached singleton (built on first use, so importing this module never
    touches the filesystem). Two tiers: built-in files in :data:`DEFAULT_DIR`
    plus the user's own under ``user_config_root()/shots/manifest_mappings/``
    (a user mapping shadows a built-in of the same name — "duplicate to edit").
    Exposes ``names``/``source``/``user_dir``/``skeleton``/``write_skeleton`` for
    the UI's source-tagging and empty-folder seeding (an example to copy).
    """
    return TemplateSet(
        "manifest_mappings",
        MappingSpec,
        "shots",
        builtin_dir=DEFAULT_DIR,
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover(directory: Optional[str] = None) -> List[str]:
    """List available mapping names (without ``.json``).

    With no *directory* (the normal case) this returns the union of built-in and
    user mappings via :func:`templates`, so a user's own files appear alongside
    the shipped ones. Either way ``_``-prefixed stems are excluded (the repo's
    private/partial convention — a user can park ``_draft.json`` without it
    surfacing in the picker). A *directory* override scans just that folder
    (back-compat / tests).
    """
    if directory is None:
        return [n for n in templates().names() if not n.startswith("_")]
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json") if not p.stem.startswith("_"))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_mapping(
    name: str,
    directory: Optional[str] = None,
) -> Dict[str, Any]:
    """Read a mapping JSON by *name*, validate it, and return the parsed dict.

    Resolution order:
        * *name* ending in ``.json`` → that path directly.
        * *directory* given → ``{directory}/{name}.json`` (back-compat / tests).
        * otherwise → resolve through :func:`templates` (built-in + user tiers).

    The file is validated against :class:`MappingSpec`: warnings (e.g. an
    unrecognised key) are logged; hard errors (unknown audio method, missing
    required key, wrong structure) raise :class:`~pythontk.SchemaError` so the
    caller can surface a precise message instead of a stack trace.

    Parameters:
        name: Mapping stem (e.g. ``"c5m_training"``) or a full ``.json`` path.
        directory: Optional folder override.

    Raises:
        FileNotFoundError: If no matching mapping exists.
        pythontk.SchemaError: If the mapping is structurally invalid.
    """
    if name.endswith(".json"):
        path = Path(name)
        if not path.is_file():
            raise FileNotFoundError(f"Mapping not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
    elif directory is not None:
        path = Path(directory) / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Mapping not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            data = templates().raw(name)
        except KeyError as exc:
            raise FileNotFoundError(str(exc)) from exc

    MappingSpec.validate(data).raise_or_warn(prefix=f"mapping {name!r}: ", logger=log)
    return data


# ---------------------------------------------------------------------------
# Resolution  (JSON → List[BuilderStep])
# ---------------------------------------------------------------------------


def resolve(
    csv_path: str,
    mapping: Optional[Dict[str, Any]] = None,
    *,
    name: Optional[str] = None,
    directory: Optional[str] = None,
) -> List[BuilderStep]:
    """Parse a CSV through a mapping and return fully resolved steps.

    Provide *mapping* (already-loaded dict) **or** *name* (to load from
    disk).  If neither is given, uses default ``ColumnMap``.

    Parameters:
        csv_path: Path to the CSV file.
        mapping: Pre-loaded mapping dict.
        name: Mapping file stem to load via :func:`load_mapping`.
        directory: Search directory for :func:`load_mapping`.
    """
    if mapping is None and name is not None:
        mapping = load_mapping(name, directory)

    col_map, post = _build_pipeline(mapping or {})
    return parse_csv(csv_path, columns=col_map, post_process=post)


# ---------------------------------------------------------------------------
# Pipeline builder  (JSON dict → ColumnMap + post_process)
# ---------------------------------------------------------------------------


def _build_pipeline(
    mapping: Dict[str, Any],
) -> tuple:
    """Convert a mapping dict into ``(ColumnMap, post_process)``."""
    col_map = _build_column_map(mapping.get("columns", {}))

    processors: List[Callable[[BuilderStep], None]] = []

    audio_cfg = mapping.get("audio_resolve")
    if audio_cfg:
        processors.append(_build_audio_resolver(audio_cfg))

    behav_cfg = mapping.get("default_behaviors")
    if behav_cfg:
        processors.append(_build_default_behaviors(behav_cfg))

    post = _chain(processors) if processors else None
    return col_map, post


def _build_column_map(columns: Dict[str, Any]) -> ColumnMap:
    """Construct a :class:`ColumnMap` from the ``columns`` section."""
    if not columns:
        return ColumnMap()
    return ColumnMap.from_dict(columns)


# ---------------------------------------------------------------------------
# Audio resolvers  (JSON config → callable)
# ---------------------------------------------------------------------------


# Resolver builders keyed by audio_resolve method — same key set as
# :data:`._spec.AUDIO_METHODS` (``test_mapping_spec`` guards against drift).
# Adding a method is one entry here + one descriptor there; the dispatcher and
# the validator/docs never change (OCP).
_AUDIO_BUILDERS: Dict[str, Callable[[Dict[str, Any]], Callable[[BuilderStep], None]]] = {
    "prefix": lambda cfg: _audio_prefix(
        cfg.get("directory", ""), cfg.get("extensions", (".wav", ".mp3"))
    ),
    "regex": lambda cfg: _audio_regex(
        cfg.get("directory", ""),
        cfg["pattern"],
        cfg.get("extensions", (".wav", ".mp3")),
    ),
    "map": lambda cfg: _audio_map(cfg.get("clips", {})),
    "derive": lambda cfg: _audio_derive(
        cfg.get("words", 3),
        cfg.get("separator", "_"),
        cfg.get("directory", ""),
        cfg.get("extensions", (".wav", ".mp3")),
    ),
}


def _build_audio_resolver(cfg: Dict[str, Any]) -> Callable[[BuilderStep], None]:
    """Dispatch to the resolver builder for ``cfg["method"]`` (default ``prefix``)."""
    method = cfg.get("method", "prefix")
    try:
        builder = _AUDIO_BUILDERS[method]
    except KeyError:
        raise ValueError(f"Unknown audio_resolve method: {method!r}") from None
    return builder(cfg)


def _audio_prefix(
    directory: str,
    extensions: Sequence[str] = (".wav", ".mp3"),
) -> Callable[[BuilderStep], None]:
    """Match audio files by ``{step_id}_*`` prefix."""
    audio_dir = Path(directory)
    ext_set = {e.lower() for e in extensions}

    def _resolve(step: BuilderStep) -> None:
        if not audio_dir.is_dir():
            return
        prefix = f"{step.step_id}_".lower()
        for f in audio_dir.iterdir():
            if f.suffix.lower() in ext_set and f.stem.lower().startswith(prefix):
                step.objects.append(
                    BuilderObject(
                        name=f.stem,
                        kind="audio",
                        source_path=str(f),
                    )
                )
                return

    return _resolve


def _audio_regex(
    directory: str,
    pattern: str,
    extensions: Sequence[str] = (".wav", ".mp3"),
) -> Callable[[BuilderStep], None]:
    """Match audio files by regex with ``{step_id}`` placeholder."""
    audio_dir = Path(directory)
    ext_set = {e.lower() for e in extensions}

    def _resolve(step: BuilderStep) -> None:
        if not audio_dir.is_dir():
            return
        # Substitute the placeholder literally — str.format would parse
        # regex brace quantifiers like \d{2} as format fields and crash.
        compiled = _re.compile(
            pattern.replace("{step_id}", _re.escape(step.step_id)),
            _re.IGNORECASE,
        )
        for f in audio_dir.iterdir():
            if f.suffix.lower() in ext_set and compiled.search(f.stem):
                step.objects.append(
                    BuilderObject(
                        name=f.stem,
                        kind="audio",
                        source_path=str(f),
                    )
                )
                return

    return _resolve


def _audio_map(
    clips: Dict[str, str],
) -> Callable[[BuilderStep], None]:
    """Set audio from an explicit ``{step_id: clip_stem}`` dict."""

    def _resolve(step: BuilderStep) -> None:
        clip = clips.get(step.step_id, "")
        if clip:
            step.objects.append(BuilderObject(name=clip, kind="audio"))

    return _resolve


def _audio_derive(
    words: int = 3,
    separator: str = "_",
    directory: str = "",
    extensions: Sequence[str] = (".wav", ".mp3"),
) -> Callable[[BuilderStep], None]:
    """Derive audio clip from ``step_id`` + first N words of ``audio``.

    Generates a PascalCase clip name like ``A01_WelcomeToThe`` from the
    step's voiceover text.  If *directory* is provided and a matching
    file exists, ``source_path`` is also set on the audio object.

    Parameters:
        words: Number of leading words to take from the audio text.
        separator: Join character between step_id and the word block.
        directory: Optional directory to scan for a matching file.
        extensions: File extensions to consider when scanning.
    """
    audio_dir = Path(directory) if directory else None
    ext_set = {e.lower() for e in extensions}
    # Strip non-alphanumeric except underscores for safe file names
    _clean_re = _re.compile(r"[^\w]+")

    def _resolve(step: BuilderStep) -> None:
        if not step.audio or step.audio.upper() == "N/A":
            return
        tokens = step.audio.split(None, words)[:words]
        if not tokens:
            return
        clean = [_clean_re.sub("", w).capitalize() for w in tokens]
        clip_name = step.step_id + separator + "".join(clean)
        source = ""

        # If a directory is configured, try to find the actual file
        if audio_dir and audio_dir.is_dir():
            target = clip_name.lower()
            for f in audio_dir.iterdir():
                if f.suffix.lower() in ext_set and f.stem.lower() == target:
                    source = str(f)
                    break

        step.objects.append(
            BuilderObject(name=clip_name, kind="audio", source_path=source)
        )

    return _resolve


# ---------------------------------------------------------------------------
# Default-behaviors applicator  (JSON config → callable)
# ---------------------------------------------------------------------------


def _build_default_behaviors(
    cfg: Dict[str, List[str]],
) -> Callable[[BuilderStep], None]:
    """Assign default behaviors to objects by *kind*.

    ``cfg`` maps object kinds to behavior name lists::

        {"audio": ["set_clip"], "scene": ["fade_in", "fade_out"]}

    Only behaviors not already present on the object are added.
    """

    def _apply(step: BuilderStep) -> None:
        for obj in step.objects:
            defaults = cfg.get(obj.kind, [])
            for b in defaults:
                if b not in obj.behaviors:
                    obj.behaviors.append(b)

    return _apply


# ---------------------------------------------------------------------------
# Internal combinators
# ---------------------------------------------------------------------------


def _chain(
    processors: List[Callable[[BuilderStep], None]],
) -> Callable[[BuilderStep], None]:
    """Chain multiple post-process callables into one."""
    if len(processors) == 1:
        return processors[0]

    def _chained(step: BuilderStep) -> None:
        for proc in processors:
            proc(step)

    return _chained
