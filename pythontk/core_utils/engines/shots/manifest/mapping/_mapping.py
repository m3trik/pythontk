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
            "scene": ["show"]
        }
    }

Supported ``audio_resolve`` methods:

- ``"prefix"``: Match files starting with ``{step_id}_``.
- ``"regex"``: Match files against a regex pattern with
  ``{step_id}`` placeholder.
- ``"map"``: Explicit ``step_id → clip_stem`` lookup table.
- ``"derive"``: Build clip name from ``step_id`` + first N words
  of the audio text (PascalCase).
- ``"cell"``: The file the step's own cell names (``field``, default
  ``audio_file``: the Audio Files column), in ``directory``.
- ``"narrate"``: The narration :class:`pythontk.Narrator` rendered for the
  step's audio text into ``directory``, found by its words (looked up only:
  a sheet is never spoken while it is read); the clip named as ``"derive"``
  names it.

``audio_resolve`` may be a list of methods: the first to find a file wins,
else the first clip with no file stands, so a recording (a cell, a prefix)
replaces narration with no bookkeeping. A step that already holds an audio
file is left as it is. A relative ``directory`` (or ``pronounce`` file)
resolves against the *base* :func:`resolve` is given -- the project's
settings folder -- else stays as written.

API
---
:func:`discover` — list available mapping names in a directory.
:func:`load_mapping` — read a JSON mapping and return a callable.
:func:`resolve` — ``(csv_path, mapping_name) → List[BuilderStep]``.
"""

from __future__ import annotations

import dataclasses as _dataclasses
import functools
import json
import logging
import os
import re as _re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
        base: Optional[str] = None,
    ) -> tuple:
        """Convert a mapping dict into ``(ColumnMap, post_process)``; relative audio
        folders resolve against *base* (none: as written)."""
        col_map = _MappingInternal._build_column_map(mapping.get("columns", {}))

        processors: List[Callable[[BuilderStep], None]] = []

        audio_cfg = mapping.get("audio_resolve")
        if audio_cfg:
            processors.append(_MappingInternal._build_audio_resolver(audio_cfg, base))

        behav_cfg = mapping.get("default_behaviors")
        if behav_cfg:
            processors.append(_MappingInternal._build_default_behaviors(behav_cfg))

        post = _MappingInternal._chain(processors) if processors else None
        return col_map, post

    @staticmethod
    def _resolved(data: Dict[str, Any]) -> Dict[str, Any]:
        """*data* with each word's settings declared whole (``ManifestActions.resolve``), so
        a reader of the parsed mapping reads one declaration per setting; *data* is not changed."""
        from pythontk.core_utils.engines.shots.manifest.manifest_actions import ManifestActions

        columns = (data or {}).get("columns") or {}
        if not columns.get("actions"):
            return data
        resolved = ManifestActions.resolve(columns["actions"], columns.get("settings") or {})
        return dict(data, columns=dict(columns, actions=resolved))

    @staticmethod
    def _build_column_map(columns: Dict[str, Any]) -> ColumnMap:
        """Construct a :class:`ColumnMap` from the ``columns`` section."""
        if not columns:
            return ColumnMap()
        return ColumnMap.from_dict(columns)

    @staticmethod
    def _build_audio_resolver(
        cfg: Union[Dict[str, Any], List[Dict[str, Any]]], base: Optional[str] = None
    ) -> Callable[[BuilderStep], None]:
        """Dispatch to the resolver builder for ``cfg["method"]`` (default ``prefix``); a
        list of methods is tried in order, the first to find a file winning
        (:meth:`_audio_chain`)."""
        if isinstance(cfg, (list, tuple)):
            return _MappingInternal._audio_chain(
                [_MappingInternal._build_audio_resolver(each, base) for each in cfg]
            )
        method = cfg.get("method", "prefix")
        try:
            builder = _AUDIO_BUILDERS[method]
        except KeyError:
            raise ValueError(f"Unknown audio_resolve method: {method!r}") from None
        return builder(_MappingInternal._based(cfg, base))

    @staticmethod
    def _based(cfg: Dict[str, Any], base: Optional[str]) -> Dict[str, Any]:
        """*cfg* with its relative ``directory`` -- and a ``pronounce`` given as a file -- under
        *base* (none: as written); a ``pronounce`` file is read into its word map."""
        cfg = dict(cfg)
        for key in ("directory", "pronounce"):
            value = cfg.get(key)
            if base and isinstance(value, str) and value and not os.path.isabs(value):
                cfg[key] = os.path.join(base, value)
        if isinstance(cfg.get("pronounce"), str):
            # A project's list, where it keeps one: none written yet says every word as written.
            path = cfg["pronounce"]
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as fh:
                    cfg["pronounce"] = json.load(fh)
            else:
                cfg["pronounce"] = {}
        return cfg

    @staticmethod
    def _audio_chain(resolvers: List[Callable[[BuilderStep], None]]) -> Callable[[BuilderStep], None]:
        """Try each resolver on the step in turn: the first whose clip has a file wins; when
        none finds one, the first clip without a file stands (so the step still gets its
        clip, placeholder length and all). A step that already holds an audio file is left."""

        def _resolve(step: BuilderStep) -> None:
            if any(o.kind == "audio" and o.source_path for o in step.objects):
                return
            fallback = None
            for each in resolvers:
                trial = _dataclasses.replace(step, objects=[])
                each(trial)
                clips = [o for o in trial.objects if o.kind == "audio"]
                found = [o for o in clips if o.source_path]
                if found:
                    step.objects.append(found[0])
                    return
                if fallback is None and clips:
                    fallback = clips[0]
            if fallback is not None:
                step.objects.append(fallback)

        return _resolve

    @staticmethod
    def _audio_cell(cfg: Dict[str, Any]) -> Callable[[BuilderStep], None]:
        """The file the step's own cell names (``field``, default ``audio_file``: the Audio
        Files column carried by ``metadata_pass``), in ``directory`` where it is not absolute;
        a named file that is missing is a clip with no file, so it shows as missing."""
        field_name = cfg.get("field", "audio_file")
        directory = cfg.get("directory", "")

        def _resolve(step: BuilderStep) -> None:
            cell = (step.pass_through.get(field_name) or "").strip()
            named = next((line.strip() for line in cell.splitlines() if line.strip()), "")
            if not named or named.upper() == "N/A":
                return
            path = named if os.path.isabs(named) or not directory else os.path.join(directory, named)
            step.objects.append(
                BuilderObject(
                    name=Path(named).stem,
                    kind="audio",
                    source_path=path if os.path.isfile(path) else "",
                )
            )

        return _resolve

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

        def _resolve(step: BuilderStep) -> None:
            clip_name = _MappingInternal._clip_name(step, words, separator)
            if not clip_name:
                return
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
    def _clip_name(step: BuilderStep, words: int = 3, separator: str = "_") -> str:
        """The clip a step's voice text names: ``step_id`` + its first *words* words in PascalCase
        (``A01_WelcomeToThe``); empty where the step has nothing to say."""
        if not step.audio or step.audio.upper() == "N/A":
            return ""
        tokens = step.audio.split(None, words)[:words]
        if not tokens:
            return ""
        return step.step_id + separator + "".join(_re.sub(r"[^\w]+", "", w).capitalize() for w in tokens)

    @staticmethod
    def _audio_narrate(cfg: Dict[str, Any]) -> Callable[[BuilderStep], None]:
        """The narration of each step's voice text in ``cfg["directory"]``, found by its words
        (:meth:`pythontk.Narrator.key`, with the ``voice``, ``lang``, ``speed`` and ``pronounce``
        the render used); its clip named as :meth:`_audio_derive` names it (``words``,
        ``separator``). Looked up only: a line not rendered yet is a clip with no file, never
        spoken here (``narration.render`` speaks them, asked).
        """
        from pythontk.audio_utils.narrator import Narrator

        narrator = Narrator.from_settings(cfg)
        words, separator, directory = cfg.get("words", 3), cfg.get("separator", "_"), cfg["directory"]

        def _resolve(step: BuilderStep) -> None:
            clip_name = _MappingInternal._clip_name(step, words, separator)
            if not clip_name:
                return
            path = os.path.join(directory, narrator.key(step.audio) + ".wav")
            step.objects.append(
                BuilderObject(name=clip_name, kind="audio", source_path=path if os.path.isfile(path) else "")
            )

        return _resolve

    @staticmethod
    def _build_default_behaviors(
        cfg: Dict[str, List[str]],
    ) -> Callable[[BuilderStep], None]:
        """Assign default behaviors to objects by *kind*.

        ``cfg`` maps object kinds to behavior name lists::

            {"audio": ["set_clip"], "scene": ["show", "hide"]}

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
        return _MappingInternal._resolved(data)

    @staticmethod
    def option_specs(mapping: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """The settings *mapping* exposes, one plain dict each, in file order.

        Shaped for a UI widget factory (uitk's ``AttributeSpec`` reads these
        names): ``key``, ``kind`` (``bool`` | ``choice``), ``label``,
        ``tooltip``, ``section`` (the title a UI groups it under; ``""``:
        none), ``default``, and for a choice ``choices`` as
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
                "section": opt.get("section") or "",
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
    def option_sections(
        mapping: Optional[Dict[str, Any]],
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """*mapping*'s option specs (:meth:`option_specs`) grouped by their
        ``section``, as ``[(title, [spec, ...]), ...]``: the sections in the
        order their first option appears, each section's options in file
        order -- a UI lists them under one title each, wherever the file puts
        them.  Options naming no section come first, under ``""``.

        Parameters:
            mapping: A mapping template (its ``options`` block is read); ``None``
                or one without options has none.

        Returns:
            ``[(title, [spec, ...]), ...]``, each spec as :meth:`option_specs`
            shapes it; ``[]`` when the template has no options.
        """
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for spec in Mapping.option_specs(mapping):
            groups.setdefault(spec["section"], []).append(spec)
        untitled = groups.pop("", None)
        sections = list(groups.items())
        return ([("", untitled)] if untitled else []) + sections

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
                choices = opt.get("choices", {})
                choice = choices.get(value)
                if choice is None:
                    choice = choices.get(spec["default"], {})
                if "try" in choice:
                    # Each named choice's column patch is a layout the parse
                    # tries on the sheet (``ColumnMap.layouts``).
                    layouts = [
                        choices[name].get("set", {}).get("columns", {})
                        for name in choice["try"]
                    ]
                    patch = {"columns": {"layouts": layouts}}
                else:
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
        base: Optional[str] = None,
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
            base: The folder a relative ``audio_resolve`` folder is read against (the
                project's settings folder; a sheet's own folder moves with each export);
                ``None`` leaves it as written.
        """
        if mapping is None and name is not None:
            mapping = Mapping.load_mapping(name, directory)

        mapping = Mapping.apply_options(mapping, options)
        col_map, post = _MappingInternal._build_pipeline(mapping, base)
        return ManifestModel.parse_csv(csv_path, columns=col_map, post_process=post)

    @staticmethod
    def audio_method(
        mapping: Optional[Dict[str, Any]], method: str, base: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """The settings of *mapping*'s audio method named *method* (``"narrate"``),
        as its resolver reads them: a relative ``directory`` and a ``pronounce``
        file under *base*. ``None`` when the mapping, its options applied
        (:meth:`apply_options`), uses no such method."""
        methods = (mapping or {}).get("audio_resolve") or []
        methods = methods if isinstance(methods, list) else [methods]
        found = next((m for m in methods if m.get("method") == method), None)
        return None if found is None else _MappingInternal._based(found, base)

    @staticmethod
    def narrate(
        sheet: str,
        mapping: str = "default",
        options: Optional[Dict[str, Any]] = None,
        base: Optional[str] = None,
        accept: Optional[List[str]] = None,
        prune: bool = False,
        workers: int = 0,
        prompt: bool = True,
        confirm: Optional[Callable[[str], bool]] = None,
        ctx=None,
    ) -> Dict[str, Any]:
        """Narrate a sheet's voice lines (``narration.render``): every step's voice text that no
        recording covers, into the folder its mapping's ``narrate`` method names, so reading the
        sheet finds them. The sheet is never written; a step whose voice text is ``N/A`` says nothing.

        Parameters:
            sheet: The sheet (a CSV path).
            mapping: Its template's name (``default``, ``linear``, ``interactive``), or a mapping file's path.
            options: Values for the template's options (``{"audio": "narrate"}``).
            base: The project folder relative audio folders are read against (where its settings live).
            accept: Step IDs whose recordings are accepted for their current lines.
            prune: Remove narration files no line uses (asked; a person's file never).
            workers: Engine runs at once (0: chosen from the cores).
            prompt: Ask before the narration engine's first install (about 170 MB); False installs
                it unasked (the caller consents). Where nobody can be asked (no console, no
                *confirm*), nothing is downloaded and the run stops.
            confirm: What asks, ``(question) -> bool`` (a panel's dialog); none: the console.
            ctx: The run's context (an ``OpContext``).

        Returns:
            ``Narrator.render``'s result, plus ``"steps"`` (the voiced steps counted) and ``"directory"``.

        Raises:
            ValueError: The mapping, with these options, has no ``narrate`` method.
            FileNotFoundError: The engine is not installed and may not be: declined, or nobody
                to ask (no console: pass ``prompt=False`` to install it).
        """
        from pythontk.audio_utils.narrator import Narrator

        loaded = Mapping.apply_options(Mapping.load_mapping(mapping), options)
        settings = Mapping.audio_method(loaded, "narrate", base)
        if settings is None:
            raise ValueError(f"the {mapping!r} mapping, with options {options or {}}, does not narrate: "
                             "choose its narration option (e.g. options={'audio': 'narrate'})")
        consent = (confirm if confirm is not None else True) if prompt else False
        narrator = Narrator.from_settings(settings, prompt=consent)
        directory = settings["directory"]
        steps = Mapping.resolve(sheet, mapping=loaded, base=base)
        lines, recordings = {}, {}
        inside = os.path.normcase(os.path.abspath(directory))

        def narrated(path: str) -> bool:
            """Whether *path* is in the narration folder (``narration_takes/`` is not in ``narration/``)."""
            path = os.path.normcase(os.path.abspath(path))
            try:
                return os.path.commonpath([path, inside]) == inside
            except ValueError:  # another drive
                return False

        for step in steps:
            if not step.audio or step.audio.strip().upper() == "N/A":
                continue
            lines[step.step_id] = step.audio
            for clip in step.objects:
                if clip.kind == "audio" and clip.source_path and not narrated(clip.source_path):
                    recordings[step.step_id] = clip.source_path
        result = narrator.render(lines, directory, recordings=recordings, accept=accept or (), prune=prune,
                                 workers=workers, ctx=ctx)
        result.update(steps=len(lines), directory=directory)
        return result

    @staticmethod
    def starter_sheet(name: str) -> Optional[Path]:
        """The built-in starter sheet of template *name*: a small task in its
        columns and phrasing to copy and fill (``starters/<name>.csv``), or None
        for a template that ships none.

        Parameters:
            name: A built-in template's name (``linear``, ``interactive``).
        """
        path = Path(__file__).parent / "starters" / f"{name}.csv"
        return path if path.is_file() else None

    @staticmethod
    def template_for(csv_path: str, directory: Optional[str] = None) -> Optional[str]:
        """The template a sheet written in cells is written for, or None for a sheet of
        sentences (any template may read one: a person's choice).

        A template in the cell format declares its actions and the columns that mark a
        sheet as its own (``cell_marks``). Of those whose marks the sheet's header carries,
        the one is chosen that leaves the fewest of its columns unread, then the fewest of
        its Action words undeclared, then declares the fewest words: a Linear sheet reads as
        ``linear`` though ``interactive`` holds its words too, and a sheet with a mistyped
        word still reads as the template the rest of it is written for.

        Parameters:
            csv_path: Path to the CSV file, or an ``http(s)`` URL.
            directory: A folder of templates in place of the built-in and user ones.
        """
        from pythontk.core_utils.engines.shots.manifest.manifest_actions import ManifestActions

        rows = ManifestModel._read_csv_rows(csv_path)
        best: Optional[Tuple[Tuple[int, int, int], str]] = None
        for name in Mapping.discover(directory):
            try:
                col_map, _ = _MappingInternal._build_pipeline(Mapping.load_mapping(name, directory))
            except Exception as exc:  # a broken template of a user's reads no sheet
                log.debug("template %r not read: %s", name, exc)
                continue
            marks = {m.strip().lower() for m in col_map.cell_marks}
            if not col_map.actions or not marks:
                continue
            aliases = {a.strip().lower() for a in col_map.step_id}
            header = next((r for r in rows if any(c.strip().lower() in aliases for c in r)), None)
            if header is None or not any(c.strip().lower() in marks for c in header):
                continue
            titles = [c.strip().lower() for c in header]
            at = next((titles.index(a.lower()) for a in col_map.row_pass.get("action", ()) if a.lower() in titles), None)
            words = {ManifestActions.word(r[at]) for r in rows if r is not header and at is not None and len(r) > at}
            score = (len(col_map.unmapped(header)), len(words - set(col_map.actions) - {""}), len(col_map.actions))
            if best is None or score < best[0]:
                best = (score, name)
        return best[1] if best else None

    @staticmethod
    def new_sheet(out: str, template: str = "interactive", ctx=None) -> Dict[str, Any]:
        """Start a new sheet (``sheet.new``): *template*'s starter sheet written to *out*, to
        rename and fill. A starter is a small task that uses every word its template declares
        and reads with nothing to report, so a sheet made from it starts correct.

        Parameters:
            out: The CSV to write. A file that exists is never replaced.
            template: A template that ships a starter (``interactive``, ``linear``).
            ctx: The run's context (an ``OpContext``), asked before the file is written.

        Returns:
            ``{"sheet": the file, "template": its template, "columns": its header}``.

        Raises:
            ValueError: *template* ships no starter (the tolerant ``default`` reads any sheet).
            FileExistsError: *out* exists.
        """
        import csv
        import shutil

        source = Mapping.starter_sheet(template)
        if source is None:
            shipped = ", ".join(sorted(p.stem for p in (Path(__file__).parent / "starters").glob("*.csv")))
            raise ValueError(f"the {template!r} template ships no starter sheet: name one of {shipped}")
        target = Path(out)
        if target.exists():
            raise FileExistsError(f"{target} exists, and a new sheet never replaces a file: name another")
        if ctx is None or ctx.should_write(str(target), f"write the {template} starter sheet"):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        with open(source, encoding="utf-8-sig", newline="") as handle:
            columns = next(csv.reader(handle))
        return {"sheet": str(target), "template": template, "columns": columns}

    @staticmethod
    def unmapped_columns(
        csv_path: str,
        mapping: Optional[Dict[str, Any]] = None,
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """The columns of the sheet's header rows no field of *mapping* takes, in
        first-seen order: a column the sheet calls by a name the template does not
        know, read off every header row (a sheet repeating its header per section
        may differ between them).

        Parameters:
            csv_path: Path to the CSV file, or an ``http(s)`` URL.
            mapping: Pre-loaded mapping dict (the default ``ColumnMap`` if None).
            options: Values for the template's options (:meth:`apply_options`).
        """
        col_map, _ = _MappingInternal._build_pipeline(Mapping.apply_options(mapping or {}, options))
        step_aliases = {a.strip().lower() for a in col_map.step_id}
        found: Dict[str, None] = {}
        for row in ManifestModel._read_csv_rows(csv_path):
            if not any(c.strip().lower() in step_aliases for c in row):
                continue
            try:  # a header only where the parser would read one (a data cell may hold an alias)
                ManifestModel._resolve_columns(row, col_map)
            except ValueError:
                continue
            found.update(dict.fromkeys(col_map.unmapped(row)))
        return list(found)

    @staticmethod
    def infer_options(
        csv_path: str,
        mapping: Dict[str, Any],
        built: Dict[str, Iterable[str]],
        values: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """The option values *csv_path* reads as a build from it recorded.

        For a scene that read the sheet before its options were kept with it
        (``ManifestEdits.sheet_options``): its shots recorded what each step
        listed (``csv_objects``), so the options are the ones that read that
        back.  A production scene was built with Object Case = upper: read
        As written, 1 of its 20 built steps lists what the build recorded;
        upper reads 16.  A coordinate search from *values*: each option tries
        each of its values and keeps one that reads more of *built*, until a
        pass gains nothing -- so a tie keeps *values*.

        Parameters:
            csv_path: The sheet (a path or URL).
            mapping: The template (``load_mapping``), options block included.
            built: ``{step_id: [object name, ...]}`` -- what the build recorded.
            values: Where the search starts (the user's options); each option
                left out starts at its default.

        Returns:
            ``{option: value}`` for every option of *mapping*; *values* as they
            were when *built* is empty or nothing reads it better.
        """
        specs = Mapping.option_specs(mapping)
        best = {s["key"]: (values or {}).get(s["key"], s["default"]) for s in specs}
        target = {(str(sid), str(n)) for sid, names in built.items() for n in names}
        if not target:
            return best
        scores: Dict[str, int] = {}

        # A trial that reads nothing is expected here: the parser's warning
        # for it would read as a problem with the sheet.
        parser_log = logging.getLogger(ManifestModel.__module__)

        def score(vals: Dict[str, Any]) -> int:
            key = json.dumps(vals, sort_keys=True, default=str)
            if key not in scores:
                was = parser_log.disabled
                parser_log.disabled = True
                try:
                    steps = Mapping.resolve(csv_path, mapping=mapping, options=vals)
                except Exception:  # a value the sheet cannot be read with
                    scores[key] = -1
                else:
                    read = {(s.step_id, o.name) for s in steps for o in s.objects}
                    scores[key] = len(target & read)
                finally:
                    parser_log.disabled = was
            return scores[key]

        best_score = score(best)
        improved = True
        while improved:
            improved = False
            for spec in specs:
                if spec["kind"] == "choice":
                    candidates = [value for _label, value, _tip in spec["choices"]]
                else:
                    candidates = [False, True]
                for value in candidates:
                    if value == best[spec["key"]]:
                        continue
                    trial = dict(best, **{spec["key"]: value})
                    trial_score = score(trial)
                    if trial_score > best_score:
                        best, best_score, improved = trial, trial_score, True
        return best


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
    "cell": lambda cfg: _MappingInternal._audio_cell(cfg),
    "narrate": lambda cfg: _MappingInternal._audio_narrate(cfg),
}


# ---------------------------------------------------------------------------
# Default-behaviors applicator  (JSON config → callable)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Internal combinators
# ---------------------------------------------------------------------------
