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
from typing import Dict, List, Sequence, Tuple

from pythontk.str_utils._str_utils import StrUtils


# Canonical one-way "send to the target app" mode. Callers are send-only today; the
# machinery already parses a tuple so a future round-trip mode slots in without a
# restructure.
SEND_TO = "send_to"

# Cache of compiled ``<FIELD> = (...)`` matchers, keyed by the declaration field name.
_MODE_FIELD_RE: Dict[str, "re.Pattern[str]"] = {}


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


def list_templates(template_dir, extension: str = ".py") -> List[Path]:
    """Return user-visible templates in *template_dir* (skips ``_``-prefixed stems)."""
    return sorted(
        p
        for p in Path(template_dir).glob(f"*{extension}")
        if not p.stem.startswith("_")
    )


def template_modes(
    template_path,
    allowed: Sequence[str] = (SEND_TO,),
    field: str = "BRIDGE_MODES",
) -> Tuple[str, ...]:
    """Return the modes a template declares via its ``<field> = (...)`` tuple.

    Falls back to ``(allowed[0],)`` when the file is unreadable, declares no such
    field, or declares only values outside *allowed*.
    """
    fallback = (allowed[0],) if allowed else (SEND_TO,)
    try:
        text = Path(template_path).read_text(encoding="utf-8")
    except OSError:
        return fallback
    m = _mode_field_re(field).search(text)
    if not m:
        return fallback
    modes = tuple(
        item.strip().strip("'\"") for item in m.group(1).split(",") if item.strip()
    )
    valid = tuple(mode for mode in modes if mode in allowed)
    return valid or fallback


def list_template_modes(
    template_dir,
    extension: str = ".py",
    allowed: Sequence[str] = (SEND_TO,),
    field: str = "BRIDGE_MODES",
) -> List[Tuple[str, str]]:
    """Return ``[(stem, mode), ...]`` for every (template, mode) pairing."""
    out: List[Tuple[str, str]] = []
    for path in list_templates(template_dir, extension):
        for mode in template_modes(path, allowed, field):
            out.append((path.stem, mode))
    return out


def render_template(template_path, context: Dict[str, str]) -> str:
    """Substitute ``__KEY__`` placeholders in *template_path* using *context*.

    Thin wrapper over :meth:`pythontk.StrUtils.replace_delimited` (``__``/``__``
    delimiters) so the substitution rule lives in one place.
    """
    text = Path(template_path).read_text(encoding="utf-8")
    return StrUtils.replace_delimited(text, context)
