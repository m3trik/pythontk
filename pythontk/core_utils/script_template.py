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
from typing import Dict, List, Optional, Sequence, Tuple

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
#   appearing (:data:`pythontk.core_utils.script_run.REWRITTEN`); see
#   :class:`~pythontk.core_utils.app_handoff.ScriptRoundTripDeliverer`.
# * **A new artifact** (mayatk's Blender lightmap bake returns a manifest of maps + UV
#   layouts) -- mechanically a ``save_as`` run, judged by the file being CREATED; see
#   :class:`~pythontk.core_utils.app_handoff.ScriptRunDeliverer`.
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
    """ScriptTemplate — module namespace."""

    @staticmethod
    def list_templates(template_dir, extension: str = ".py") -> List[Path]:
        """Return user-visible templates in *template_dir* (skips ``_``-prefixed stems)."""
        return sorted(
            p
            for p in Path(template_dir).glob(f"*{extension}")
            if not p.stem.startswith("_")
        )

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
        template_dir,
        extension: str = ".py",
        allowed: Sequence[str] = (SEND_TO,),
        field: str = "BRIDGE_MODES",
    ) -> List[Tuple[str, str]]:
        """Return ``[(stem, mode), ...]`` for every (template, mode) pairing."""
        out: List[Tuple[str, str]] = []
        for path in ScriptTemplate.list_templates(template_dir, extension):
            for mode in ScriptTemplate.template_modes(path, allowed, field):
                out.append((path.stem, mode))
        return out

    @staticmethod
    def render_template(template_path, context: Dict[str, str]) -> str:
        """Substitute ``__KEY__`` placeholders in *template_path* using *context*.

        Thin wrapper over :meth:`pythontk.StrUtils.replace_delimited` (``__``/``__``
        delimiters) so the substitution rule lives in one place.
        """
        text = Path(template_path).read_text(encoding="utf-8")
        return StrUtils.replace_delimited(text, context)
