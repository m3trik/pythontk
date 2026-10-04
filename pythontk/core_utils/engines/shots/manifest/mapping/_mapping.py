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

from __future__ import annotations

import functools
import json
import logging
import re as _re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pythontk import TemplateSet

from pythontk.core_utils.engines.shots.manifest.manifest_model import (
    ManifestModel,
    BuilderObject,
    BuilderStep,
    ColumnMap,
)
from pythontk.core_utils.engines.shots.manifest.mapping._spec import (
    MappingSpec,
    _MappingSpecInternal,
)

# Log under the package name, not this private impl module, so the logger name
# stays stable across the __init__ -> _mapping split.
log = logging.getLogger(__name__.rpartition(".")[0])

__all__ = [
    "DEFAULT_DIR",
    "Mapping",
]

DEFAULT_DIR: Path = Path(__file__).parent
"""Directory of built-in (read-only) mapping JSON files shipped with the tool."""


class _MappingInternal(object):
    """Internal helpers for Mapping."""

    @staticmethod
    def choice_values(opt: Dict[str, Any]) -> Tuple[Any, ...]:
        """A choice option's values: its schema ``field``'s allowed values,
        else its authored ``choices`` keys."""
        if "field" in opt:
            return _MappingSpecInternal.field_choices(str(opt["field"])) or ()
        return tuple(opt.get("choices") or {})

    @staticmethod
    def _build_pipeline(
        mapping: Dict[str, Any],
    ) -> tuple:
        """Convert a mapping dict into ``(ColumnMap, post_process)``."""
        col_map = _MappingInternal._build_column_map(mapping.get("columns", {}))

        processors: List[Callable[[BuilderStep], None]] = []

        audio_cfg = mapping.get("audio_resolve")
        if audio_cfg:
            processors.append(_MappingInternal._build_audio_resolver(audio_cfg))

        behav_cfg = mapping.get("default_behaviors")
        if behav_cfg:
            processors.append(_MappingInternal._build_default_behaviors(behav_cfg))

        post = _MappingInternal._chain(processors) if processors else None
        return col_map, post

    @staticmethod
    def _build_column_map(columns: Dict[str, Any]) -> ColumnMap:
        """Construct a :class:`ColumnMap` from the ``columns`` section."""
        if not columns:
            return ColumnMap()
        return ColumnMap.from_dict(columns)

    @staticmethod
    def _build_audio_resolver(cfg: Dict[str, Any]) -> Callable[[BuilderStep], None]:
        """Dispatch to the resolver builder for ``cfg["method"]`` (default ``prefix``)."""
        method = cfg.get("method", "prefix")
        try:
            builder = _AUDIO_BUILDERS[method]
        except KeyError:
            raise ValueError(f"Unknown audio_resolve method: {method!r}") from None
        return builder(cfg)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _audio_map(
        clips: Dict[str, str],
    ) -> Callable[[BuilderStep], None]:
        """Set audio from an explicit ``{step_id: clip_stem}`` dict."""

        def _resolve(step: BuilderStep) -> None:
            clip = clips.get(step.step_id, "")
            if clip:
                step.objects.append(BuilderObject(name=clip, kind="audio"))

        return _resolve

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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


class Mapping(_MappingInternal):
    """Resolve a CSV's columns through a declarative JSON mapping file.

    A mapping file states which CSV column feeds which ``BuilderStep``
    field and how derived values are computed, so a new studio's column
    layout is a data file rather than a code change.
    """

    @staticmethod
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

    @staticmethod
    def seed_user_folder(ts: Optional[TemplateSet] = None) -> bool:
        """Seed an empty user mapping folder with an example + format reference.

        What a user opening the mappings folder for the first time needs: a
        mapping to copy (``ts.write_skeleton("example")``) and the schema
        reference beside it (``MAPPING_FORMAT.md``, generated from the same
        ``MappingSpec.format_markdown`` SSoT as the shipped doc, so it is always
        current).  A no-op once the folder holds anything, so it never clobbers
        the user's files or re-creates ones they deleted on purpose.  Dotfiles
        do not count: selecting a mapping writes the ``.active`` last-used
        pointer there, and the first open after a selection must still seed.
        Failures are swallowed -- seeding is a convenience, not a precondition.

        Parameters:
            ts: The template set whose user folder to seed; default
                :meth:`templates`.

        Returns:
            True when the folder was empty and seeding was attempted.
        """
        ts = ts if ts is not None else Mapping.templates()
        folder = Path(ts.user_dir)
        try:
            if any(p for p in folder.iterdir() if not p.name.startswith(".")):
                return False
        except OSError:
            return False
        try:
            ts.write_skeleton("example")
        except Exception:
            pass
        try:
            (folder / "MAPPING_FORMAT.md").write_text(
                MappingSpec.format_markdown(), encoding="utf-8"
            )
        except Exception:
            pass
        return True

    @staticmethod
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
            return [n for n in Mapping.templates().names() if not n.startswith("_")]
        d = Path(directory)
        if not d.is_dir():
            return []
        return sorted(p.stem for p in d.glob("*.json") if not p.stem.startswith("_"))

    #: Retired built-in templates -> ``(replacement, option values)``: what a
    #: saved ``.active`` pointer to one resolves to while its notice runs.
    _RETIRED: Dict[str, Tuple[str, Dict[str, Any]]] = {
        "speedrun": ("default", {"audio": "derive"}),
    }

    @staticmethod
    def retired(name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """``(replacement, option values)`` for a retired template, else ``None``."""
        if name in Mapping.templates().names():
            return None  # a user template of that name wins
        return Mapping._RETIRED.get(name)

    @staticmethod
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
            name: Mapping stem (e.g. ``"default"``) or a full ``.json`` path.
            directory: Optional folder override.

        Raises:
            FileNotFoundError: If no matching mapping exists.
            pythontk.SchemaError: If the mapping is structurally invalid.
        """
        retired = Mapping.retired(name) if directory is None else None
        if retired is not None:
            from pythontk.core_utils.deprecation import Deprecation

            replacement, values = retired
            Deprecation.warn(
                f"shot-manifest mapping {name!r}",
                f"{replacement!r} with options {values!r}",
                remove_in="0.14.0",
                since="2026-10-01",
                kind="value",
            )
            data = Mapping.load_mapping(replacement)
            options = data.get("options") or {}
            for key, value in values.items():
                if key in options:
                    options[key] = dict(options[key], default=value)
            return data
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
                data = Mapping.templates().raw(name)
            except KeyError as exc:
                raise FileNotFoundError(str(exc)) from exc

        MappingSpec.validate(data).raise_or_warn(
            prefix=f"mapping {name!r}: ", logger=log
        )
        return data

    @staticmethod
    def option_specs(mapping: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """The settings *mapping* exposes, one plain dict each, in file order.

        Shaped for a UI widget factory (uitk's ``AttributeSpec`` reads these
        names): ``key``, ``kind`` (``bool`` | ``choice``), ``label``,
        ``tooltip``, ``default``, and for a choice ``choices`` as
        ``[(label, value, tooltip), ...]`` -- a ``field`` choice offers that
        schema field's allowed values (labelled from ``choices`` where given,
        else spelled out), so the list can never drift from what the field
        accepts.
        """
        specs: List[Dict[str, Any]] = []
        for key, opt in ((mapping or {}).get("options") or {}).items():
            kind = opt.get("kind", "bool")
            spec: Dict[str, Any] = {
                "key": key,
                "kind": kind,
                "label": opt.get("label", key),
                "tooltip": opt.get("tooltip", ""),
            }
            if kind == "choice":
                authored = opt.get("choices") or {}
                values = _MappingInternal.choice_values(opt)
                spec["choices"] = [
                    (
                        authored.get(v, {}).get(
                            "label", str(v).replace("_", " ").title()
                        ),
                        v,
                        authored.get(v, {}).get("tooltip", ""),
                    )
                    for v in values
                ]
                spec["default"] = opt.get("default", next(iter(values), None))
            else:
                spec["default"] = bool(opt.get("default", False))
            specs.append(spec)
        return specs

    @staticmethod
    def apply_options(
        mapping: Optional[Dict[str, Any]],
        values: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """The effective template: *mapping* with each option's patch merged in.

        Every option contributes the patch for its value -- *values[key]*
        when that is a valid value, else the option's default -- merged over
        the template (``UserConfig.deep_merge``: dicts merge, lists and
        scalars replace).  The ``options`` block itself is dropped, so the
        result is an ordinary template and applying it again is a no-op.

        Parameters:
            mapping: A loaded template (``load_mapping``).
            values: ``{option key: value}``, e.g. a UI's saved selections.
        """
        from pythontk.core_utils.user_config import UserConfig

        mapping = mapping or {}
        values = values or {}
        result = {k: v for k, v in mapping.items() if k != "options"}
        for spec in Mapping.option_specs(mapping):
            opt = mapping["options"][spec["key"]]
            value = values.get(spec["key"], spec["default"])
            if spec["kind"] == "choice" and "field" in opt:
                if value not in _MappingInternal.choice_values(opt):
                    value = spec["default"]
                patch = value  # the field's value, nested along its path
                for part in reversed(str(opt["field"]).split(".")):
                    patch = {part: patch}
            elif spec["kind"] == "choice":
                choice = opt.get("choices", {}).get(value)
                if choice is None:
                    choice = opt.get("choices", {}).get(spec["default"], {})
                patch = choice.get("set", {})
            else:
                patch = opt.get("set", {}) if bool(value) else opt.get("unset", {})
            result = UserConfig.deep_merge(result, patch)
        return result

    @staticmethod
    def resolve(
        csv_path: str,
        mapping: Optional[Dict[str, Any]] = None,
        *,
        name: Optional[str] = None,
        directory: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[BuilderStep]:
        """Parse a CSV through a mapping and return fully resolved steps.

        Provide *mapping* (already-loaded dict) **or** *name* (to load from
        disk).  If neither is given, uses default ``ColumnMap``.

        Parameters:
            csv_path: Path to the CSV file, or an ``http(s)`` URL.
            mapping: Pre-loaded mapping dict.
            name: Mapping file stem to load via :func:`load_mapping`.
            directory: Search directory for :func:`load_mapping`.
            options: Values for the template's options (:meth:`apply_options`);
                an option left out takes its default.
        """
        if mapping is None and name is not None:
            mapping = Mapping.load_mapping(name, directory)

        mapping = Mapping.apply_options(mapping, options)
        col_map, post = _MappingInternal._build_pipeline(mapping)
        return ManifestModel.parse_csv(csv_path, columns=col_map, post_process=post)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Resolution  (JSON → List[BuilderStep])
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pipeline builder  (JSON dict → ColumnMap + post_process)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Audio resolvers  (JSON config → callable)
# ---------------------------------------------------------------------------


# Resolver builders keyed by audio_resolve method — same key set as
# :data:`._spec.AUDIO_METHODS` (``test_mapping_spec`` guards against drift).
# Adding a method is one entry here + one descriptor there; the dispatcher and
# the validator/docs never change (OCP).
_AUDIO_BUILDERS: Dict[
    str, Callable[[Dict[str, Any]], Callable[[BuilderStep], None]]
] = {
    "prefix": lambda cfg: _MappingInternal._audio_prefix(
        cfg.get("directory", ""), cfg.get("extensions", (".wav", ".mp3"))
    ),
    "regex": lambda cfg: _MappingInternal._audio_regex(
        cfg.get("directory", ""),
        cfg["pattern"],
        cfg.get("extensions", (".wav", ".mp3")),
    ),
    "map": lambda cfg: _MappingInternal._audio_map(cfg.get("clips", {})),
    "derive": lambda cfg: _MappingInternal._audio_derive(
        cfg.get("words", 3),
        cfg.get("separator", "_"),
        cfg.get("directory", ""),
        cfg.get("extensions", (".wav", ".mp3")),
    ),
}


# ---------------------------------------------------------------------------
# Default-behaviors applicator  (JSON config → callable)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Internal combinators
# ---------------------------------------------------------------------------
