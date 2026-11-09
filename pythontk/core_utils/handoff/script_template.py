# !/usr/bin/python
# coding=utf-8
"""Generic on-disk script-template discovery + ``__KEY__`` rendering.

A small, reusable kit for "a folder of ``*.ext`` templates, each declaring which
*modes* it supports via a top-level tuple, with ``__KEY__`` placeholders substituted
at render time". It backs the app hand-off bridges (Maya / Blender / RizomUV render a
launch script per template) but knows nothing about FBX, DCCs, or Qt -- it is usable
for any "pick a template, fill in placeholders, hand the text to something" flow.

The mode-declaration field name is a parameter (default ``"BRIDGE_MODES"``) so the
on-disk template contract is the caller's to choose.

Qt-free and DCC-free.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pythontk.str_utils._str_utils import StrUtils


# The three hand-off shapes, distinguished by *where the result lands*. THE mode
# vocabulary for the whole ecosystem -- every bridge (Blender / Maya / RizomUV /
# Marmoset / Substance) imports these rather than spelling the strings itself. They
# are an ON-DISK contract as much as a Python one: a template's ``BRIDGE_MODES`` tuple
# names them, so a second definition anywhere is a second dialect of a file format.
#
# Canonical one-way "send to the target app" mode: the payload goes out, the target
# opens it interactively, and nothing comes back.
SEND_TO = "send_to"

# "Run the target app headlessly and keep what it wrote" mode -- the blocking
# counterpart of :data:`SEND_TO`. A ``save_as`` template ends by writing ONE artifact
# (a native scene file) and the caller waits for it, so the deliverer judges the run by
# that file rather than by having launched something.
SAVE_AS = "save_as"

# "Run the target app headlessly and bring the result back into the HOST" mode. Like
# :data:`SAVE_AS` it blocks, but the artifact is not the deliverable -- it is an
# intermediate the bridge re-ingests (transfer the result onto the original scene
# objects, discard the scaffolding). What the artist gets is CHANGED HOST STATE, which
# is why this is a mode of its own rather than a ``save_as`` whose file happens to be
# read afterwards: naming the mechanism (``save_as``) in a panel tells the artist to go
# looking for a deliverable that does not exist.
#
# The mode says where the result lands, NOT how the target writes it -- two artifact
# shapes ride it, and the deliverer registered for the mode decides which:
#
# * **The payload, edited in place** (RizomUV's ``ZomLoad``/``ZomSave``) -- one path in
#   and out, so success is judged by that file having CHANGED rather than by a new one
#   appearing (:data:`pythontk.core_utils.handoff.script_run.REWRITTEN`); see
#   :class:`~pythontk.core_utils.handoff.app_handoff.ScriptRoundTripDeliverer`.
# * **A new artifact** (mayatk's Blender lightmap bake returns a manifest of maps + UV
#   layouts) -- mechanically a ``save_as`` run, judged by the file being CREATED; see
#   :class:`~pythontk.core_utils.handoff.app_handoff.ScriptRunDeliverer`.
ROUND_TRIP = "round_trip"

# Legacy on-disk spellings, folded to the canon when a template is PARSED. The mode
# tuple lives in template files -- including ones a user wrote against an older
# contract -- and an unrecognized mode does not raise: :meth:`template_modes` filters
# it out and silently falls back to the primary mode, so a stale spelling would turn a
# headless round trip into an interactive send with nothing pointing at the cause.
# Normalizing here keeps ONE value in code while still reading the old files.
_MODE_ALIASES: Dict[str, str] = {"roundtrip": ROUND_TRIP}

# Cache of compiled ``<FIELD> = (...)`` matchers, keyed by the declaration field name.
_MODE_FIELD_RE: Dict[str, "re.Pattern[str]"] = {}


class _ScriptTemplateInternal(object):
    """Internal helpers for ScriptTemplate."""

    @staticmethod
    def _mode_field_re(field: str) -> "re.Pattern[str]":
        """Return (memoized) the regex that matches ``<field> = ( ... )``.

        Parsed WITHOUT importing the template -- the file still carries raw ``__KEY__``
        placeholders that aren't valid Python pre-substitution.
        """
        rx = _MODE_FIELD_RE.get(field)
        if rx is None:
            rx = re.compile(rf"^\s*{re.escape(field)}\s*=\s*\(([^)]*)\)", re.MULTILINE)
            _MODE_FIELD_RE[field] = rx
        return rx


class ScriptTemplate(_ScriptTemplateInternal):
    """Discover on-disk script templates and render their ``__KEY__`` slots.

    A folder of ``*.ext`` templates, each declaring the modes it supports
    via a top-level tuple, with ``__KEY__`` placeholders substituted at
    render time. Reusable across every app the toolkit scripts.
    """

    #: The folder under the per-user config root that holds every app's own
    #: scripts, one sub-folder per app (:meth:`user_dir`).
    USER_SCRIPTS = "scripts"

    @staticmethod
    def user_dir(app: str) -> Path:
        """Where a person's own scripts for *app* live: one folder per app,
        unnamespaced, so a script written once shows in every host and in the
        standalone app (``<user config root>/scripts/<app>``, created on demand
        by whoever writes to it).

        Parameters:
            app: The target app's script collection, e.g. ``"marmoset"``.
        """
        from pythontk.core_utils.user_config import UserConfig

        return UserConfig.user_config_root() / ScriptTemplate.USER_SCRIPTS / app

    @staticmethod
    def search_dirs(builtin_dir, app: Optional[str] = None) -> Tuple[Path, ...]:
        """The folders a script is looked up in, first match winning: the
        person's own (:meth:`user_dir`, when *app* is given), then the shipped
        *builtin_dir*. So a user script shadows the built-in one of its name."""
        builtin = (Path(builtin_dir),) if builtin_dir else ()
        return ((ScriptTemplate.user_dir(app),) if app else ()) + builtin

    @staticmethod
    def _dirs(template_dirs) -> List[Path]:
        if isinstance(template_dirs, (str, Path)):
            return [Path(template_dirs)]
        return [Path(d) for d in template_dirs or ()]

    @staticmethod
    def list_templates(template_dirs, extension: str = ".py") -> List[Path]:
        """Return user-visible templates (skips ``_``-prefixed stems), by stem.

        *template_dirs* is one folder or an ordered sequence of them
        (:meth:`search_dirs`): a stem found in an earlier folder shadows the same
        stem in a later one, and a missing folder holds nothing.
        """
        found: Dict[str, Path] = {}
        for folder in ScriptTemplate._dirs(template_dirs):
            if not folder.is_dir():
                continue
            for p in folder.glob(f"*{extension}"):
                if not p.stem.startswith("_"):
                    found.setdefault(p.stem, p)
        return [found[stem] for stem in sorted(found)]

    @staticmethod
    def find(template_dirs, name: str, extension: str = ".py") -> Optional[Path]:
        """The template *name* resolves to in *template_dirs* (first folder that
        holds it), or ``None``."""
        for folder in ScriptTemplate._dirs(template_dirs):
            path = folder / f"{name}{extension}"
            if path.is_file():
                return path
        return None

    @staticmethod
    def tier(template_path, template_dirs) -> Optional[str]:
        """``"user"`` when *template_path* sits in the first of *template_dirs*
        and more than one is given (the person's own folder, by
        :meth:`search_dirs`' order), ``"builtin"`` for any other of them, else
        ``None``."""
        dirs = ScriptTemplate._dirs(template_dirs)
        parent = Path(template_path).parent.resolve()
        for i, folder in enumerate(dirs):
            if parent == folder.resolve():
                return "user" if i == 0 and len(dirs) > 1 else "builtin"
        return None

    @staticmethod
    def normalize_modes(modes: Optional[Sequence[str]]) -> Tuple[str, ...]:
        """Fold legacy on-disk spellings in *modes* to the canonical values.

        The one place :data:`_MODE_ALIASES` is applied, so a bridge that parses
        ``BRIDGE_MODES`` its own way (the Substance engine reads the file with
        :mod:`ast` rather than through :meth:`declared_modes`) folds the same
        spellings as everyone else instead of silently dropping a stale one to its
        primary mode. Non-strings are discarded; unknown strings pass through, since
        judging them belongs to the caller's ``allowed`` tuple.
        """
        return tuple(
            _MODE_ALIASES.get(mode, mode)
            for mode in (modes or ())
            if isinstance(mode, str)
        )

    @staticmethod
    def declared_values(template_path, field: str) -> Optional[Tuple[str, ...]]:
        """Return the strings a template declares via ``<field> = (...)``, VERBATIM.

        The generic reader: templates declare more than modes through this same
        ``<FIELD> = (...)`` convention (mayatk's bake template also carries
        ``BRIDGE_OUTPUT_EXT`` / ``BRIDGE_OUTPUT`` / ``BRIDGE_TIMEOUT``), so a template
        stays the one place its own contract is written down. Nothing is interpreted
        here -- in particular NOT the mode aliasing, which would corrupt any other
        field whose value happened to collide with a legacy mode spelling.

        ``None`` means the file declares nothing (or is unreadable) -- distinct from an
        empty tuple, and the distinction is the point: :meth:`template_modes` may assume
        a mode for an UNANNOTATED template, but must never override an explicit one.
        """
        try:
            text = Path(template_path).read_text(encoding="utf-8")
        except OSError:
            return None
        m = _ScriptTemplateInternal._mode_field_re(field).search(text)
        if not m:
            return None
        return tuple(
            item.strip().strip("'\"") for item in m.group(1).split(",") if item.strip()
        )

    #: The kinds a script's ``Row:`` line may declare (a parameter row's kinds).
    ROW_KINDS = ("bool", "int", "float", "str", "choice", "path", "file")
    #: ``Row:`` options and the row field each sets.
    _ROW_OPTIONS = {
        "min": "minimum",
        "max": "maximum",
        "step": "step",
        "decimals": "decimals",
        "choices": "choices",
        "tooltip": "tooltip",
        "section": "section",
        "placeholder": "placeholder",
    }
    _ROW_LINE = re.compile(r"^[ \t]*(?:#|//|--|;)[ \t]*Row:[ \t]*(.*)$", re.MULTILINE)
    _ROW_KEY = re.compile(r"^__([A-Z][A-Z0-9_]*)__$")
    #: One ``Row:`` token: an option is an UNQUOTED ``name=`` before its
    #: value; a value is double- or single-quoted (kept whole, backslashes
    #: and the other quote included) or bare. A token opening a quote it
    #: never closes matches nothing, so the line does not parse.
    _ROW_TOKEN = re.compile(
        r"""[ \t]*(?:(?P<name>[A-Za-z_][A-Za-z0-9_]*)=)?"""
        r"""(?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)'|(?P<bare>[^\s"'][^\s]*))"""
    )

    @staticmethod
    def declared_rows(template_path) -> List[Dict[str, Any]]:
        """The settings a script declares for itself, one header line each::

            # Row: __ITERATIONS__ int 4 "Iterations" min=1 max=64 tooltip="Passes."
            // Row: __SOFTEN__ bool on
            # Row: __MODE__ choice Fast "Mode" choices=Fast|Best

        A line is a comment (``#``, ``//``, ``--`` or ``;``) starting ``Row:``,
        then the row's ``__KEY__`` token, its kind (:attr:`ROW_KINDS`), an
        optional default and label, and ``option=value`` pairs (``min``,
        ``max``, ``step``, ``decimals``, ``choices`` split on ``|``,
        ``tooltip``, ``section``, ``placeholder``). So a script a person wrote
        shows its own rows in the panel that runs it, where a panel's registry
        knows only the built-in ones; the panel reads them without running the
        script. A line that does not parse is skipped (one typo never hides the
        script), as is a key's second declaration. An option is an unquoted
        ``name=value``; a quoted default or label holding ``=`` is the
        default or the label.

        Weighed: ``shlex`` (the header's first reader) -- it drops the quotes
        that tell a quoted ``"a=b"`` default from an ``a=b`` option, and its
        escapes eat a Windows path's backslashes; ``argparse`` / ``getopt``
        -- they parse ``--flag`` lists, not positional values followed by
        ``name=value`` pairs; a script's own docstring or a sidecar JSON --
        the header is the one place a person's script states its rows
        without running it. So one token pattern reads the line
        (:attr:`_ROW_TOKEN`).

        Returns:
            ``[{"key", "kind", "default", ...}, ...]`` in file order: a
            parameter row's fields (``AttributeSpec``'s), typed by the kind.
        """
        try:
            text = Path(template_path).read_text(encoding="utf-8")
        except OSError:
            return []
        rows: List[Dict[str, Any]] = []
        for line in ScriptTemplate._ROW_LINE.findall(text):
            row = ScriptTemplate._parse_row(line)
            if row is not None and all(r["key"] != row["key"] for r in rows):
                rows.append(row)
        return rows

    @staticmethod
    def _parse_row(line: str) -> Optional[Dict[str, Any]]:
        """One ``Row:`` line's fields, or ``None`` when it does not parse."""
        tokens: List[Tuple[Optional[str], str]] = []  # (option name, value)
        line, pos = line.rstrip(), 0
        while pos < len(line):
            match = ScriptTemplate._ROW_TOKEN.match(line, pos)
            if match is None or match.end() == pos:
                return None
            value = next(v for v in match.group("dq", "sq", "bare") if v is not None)
            tokens.append((match.group("name"), value))
            pos = match.end()
            if pos < len(line) and not line[pos].isspace():
                return None  # a quote closed mid-token: no token of the line
        if len(tokens) < 2 or tokens[0][0] or tokens[1][0]:
            return None
        key = ScriptTemplate._ROW_KEY.match(tokens[0][1])
        kind = tokens[1][1]
        if not key or kind not in ScriptTemplate.ROW_KINDS:
            return None
        row: Dict[str, Any] = {"key": key.group(1), "kind": kind}
        rest = tokens[2:]
        default = rest.pop(0)[1] if rest and rest[0][0] is None else None
        try:
            for name, value in rest:
                if name is None:
                    row.setdefault("label", value)
                    continue
                field = ScriptTemplate._ROW_OPTIONS.get(name)
                if field is None:
                    return None
                if field == "choices":
                    row[field] = [c for c in value.split("|") if c]
                elif field in ("minimum", "maximum", "step"):
                    row[field] = int(value) if kind == "int" else float(value)
                elif field == "decimals":
                    row[field] = int(value)
                else:
                    row[field] = value
            row["default"] = ScriptTemplate._row_value(kind, default, row.get("choices"))
        except ValueError:
            return None
        ordered = {"key": row.pop("key"), "kind": row.pop("kind"), "default": row.pop("default")}
        ordered.update(row)
        return ordered

    @staticmethod
    def _row_value(kind: str, text: Optional[str], choices=None) -> Any:
        """*text* as a *kind* value; ``None`` gives the kind's empty value."""
        if kind == "bool":
            if text is None:
                return False
            word = text.lower()
            if word in ("1", "on", "true", "yes"):
                return True
            if word in ("0", "off", "false", "no"):
                return False
            raise ValueError(text)
        if kind == "int":
            return 0 if text is None else int(text)
        if kind == "float":
            return 0.0 if text is None else float(text)
        if kind == "choice" and text is None:
            return choices[0] if choices else ""
        return "" if text is None else text

    @staticmethod
    def declared_modes(
        template_path, field: str = "BRIDGE_MODES"
    ) -> Optional[Tuple[str, ...]]:
        """Return the MODES a template declares, legacy spellings folded to the canon.

        :meth:`declared_values` plus :meth:`normalize_modes` -- the strict read, used
        where a template's contract is non-negotiable (a script that has to write a
        specific artifact). ``None`` still means "declares nothing"; anything the alias
        table does not name passes through untouched, since deciding what is acceptable
        belongs to the bridge's ``allowed`` tuple, not to the reader.
        """
        declared = ScriptTemplate.declared_values(template_path, field)
        return None if declared is None else ScriptTemplate.normalize_modes(declared)

    @staticmethod
    def template_modes(
        template_path,
        allowed: Sequence[str] = (SEND_TO,),
        field: str = "BRIDGE_MODES",
    ) -> Tuple[str, ...]:
        """Return the modes a template declares via its ``<field> = (...)`` tuple.

        Falls back to ``(allowed[0],)`` when the file is unreadable, declares no such
        field, or declares only values outside *allowed* -- so a custom template a user
        drops in unannotated still works with the bridge it was dropped into.
        """
        fallback = (allowed[0],) if allowed else (SEND_TO,)
        declared = ScriptTemplate.declared_modes(template_path, field)
        if declared is None:
            return fallback
        valid = tuple(mode for mode in declared if mode in allowed)
        return valid or fallback

    @staticmethod
    def list_template_modes(
        template_dirs,
        extension: str = ".py",
        allowed: Sequence[str] = (SEND_TO,),
        field: str = "BRIDGE_MODES",
    ) -> List[Tuple[str, str]]:
        """Return ``[(stem, mode), ...]`` for every (template, mode) pairing in
        *template_dirs* (one folder or an ordered sequence, :meth:`list_templates`)."""
        out: List[Tuple[str, str]] = []
        for path in ScriptTemplate.list_templates(template_dirs, extension):
            for mode in ScriptTemplate.template_modes(path, allowed, field):
                out.append((path.stem, mode))
        return out

    @staticmethod
    def child_path(path) -> str:
        """*path* as a rendered template hands it to the app it launches.

        Forward slashes (safe inside any string literal), and spelled so an app
        that opens files through the ANSI code page can open it: Maya said "File
        not found" for a payload under a folder that code page cannot hold (a
        Cyrillic name on cp1252), so such folders become their 8.3 names
        (:meth:`pythontk.AppLauncher.ansi_safe_path`; a no-op for every other path
        and off Windows).

        Parameters:
            path: A file or folder path, or ``None`` / ``""``.

        Returns:
            str: The path to substitute, or ``""`` for no path.
        """
        if not path:
            return ""
        from pythontk.core_utils.app_launcher import AppLauncher

        return str(AppLauncher.ansi_safe_path(str(path))).replace("\\", "/")

    @staticmethod
    def render_template(template_path, context: Dict[str, str]) -> str:
        """Substitute ``__KEY__`` placeholders in *template_path* using *context*.

        Thin wrapper over :meth:`pythontk.StrUtils.replace_delimited` (``__``/``__``
        delimiters) so the substitution rule lives in one place.
        """
        text = Path(template_path).read_text(encoding="utf-8")
        return StrUtils.replace_delimited(text, context)
