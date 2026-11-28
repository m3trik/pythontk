# !/usr/bin/python
# coding=utf-8
"""A sheet's actions: the words its Action cells may hold, as its template declares them.

A template in the cell format declares each word as data (``ColumnMap.actions``): the row
cells it takes and needs, its settings, whether its row plays a take, and the sentence a row
of it reads as.  This module reads one row against that declaration -- what its cells say,
what is wrong with them and how to fix it, and the row said back as a sentence.  Nothing
here knows any word: the words are the template's.

A declaration::

    "use": {
        "family": "animation",
        "cells": ["target", "with", "clip", "members"],
        "needs": ["target", "clip"],
        "take": true,
        "when": "user",
        "settings": {"trigger": {"choices": ["select", "hold", "near"], "default": "select"},
                     "hold": {"type": "number", "min": 0}},
        "says": {"": "Selecting {target}[ with {with} in hand] plays {clip}",
                 "trigger=hold": "Holding {target}[ with {with} in hand] plays {clip}"}
    }

A sheet in cells has two kinds of row.  A step's row (the one with its ID) is the step
alone: its name, its voice line, where it goes by itself (To) and its own settings; it
holds no action.  Each row under it is one action.  A step, a row and a setting are each
held to the declaration by one check (``check_step_row``, ``check_row``, ``check_step``),
every problem said as an issue at its cell with its fix.

The empty word (``""``) declares what a row with no Action is, where a template gives it a
meaning (a row that only lists a shot's members); a template that declares none refuses it.

A setting is declared once, for every word, in the template's ``settings``: its type, the
primitive it belongs to (:attr:`ManifestActions.PRIMITIVES`) and its bounds, so a name means
one thing in every word. A word lists the settings it takes and adds only what is its own:
its ``choices``, its ``default`` and whether it is ``required`` (:meth:`resolve` joins the two)::

    "settings": {"when": {"type": "choice", "primitive": "trigger"}},
    "actions": {"button": {..., "settings": {"when": {"choices": ["end", "start"], "default": "end"}}}}
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = ["ManifestActions"]

#: One problem with a row: the header of the cell it is in, and what is wrong with its fix.
Issue = Tuple[str, str]


class _ManifestActionsInternal(object):
    """Internal helpers for ManifestActions."""

    _SEGMENT = re.compile(r"\[([^\[\]]*)\]")
    _FIELD = re.compile(r"\{(\w+)\}")
    _COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
    _OBJECT = re.compile(
        r"^[^\s;=,]+$"
    )  # one name, or a path (TABLE|KEY_GRP): no spaces, separators or pairs
    _FLAGS = {
        "on": True,
        "yes": True,
        "true": True,
        "off": False,
        "no": False,
        "false": False,
    }
    _SECOND = re.compile(
        r",\s*\w+\s*="
    )  # a second pair after a comma: settings are split by ; or a line

    @staticmethod
    def _joined(names: Sequence[str]) -> str:
        """*names* as words: ``A``, ``A and B``, ``A, B and C``."""
        names = [n for n in names if n]
        if len(names) < 2:
            return "".join(names)
        return "%s and %s" % (", ".join(names[:-1]), names[-1])

    @staticmethod
    def _fill(template: str, values: Mapping[str, str]) -> str:
        """*template* with its ``{fields}`` filled; a ``[bracketed part]`` whose field is empty is dropped."""

        def part(match: "re.Match") -> str:
            inner = match.group(1)
            fields = _ManifestActionsInternal._FIELD.findall(inner)
            return inner if all(values.get(f) for f in fields) else ""

        text = _ManifestActionsInternal._SEGMENT.sub(part, template)
        text = _ManifestActionsInternal._FIELD.sub(
            lambda m: values.get(m.group(1), ""), text
        )
        return re.sub(r"\s{2,}", " ", text).strip()

    @staticmethod
    def _value_error(
        name: str, value: str, declared: Mapping[str, Any]
    ) -> Optional[str]:
        """Why *value* is no value of the setting *name* as *declared*, with its fix; None where it is."""
        choices = declared.get("choices")
        kind = declared.get("type") or ("choice" if choices else "text")
        if kind != "text" and _ManifestActionsInternal._SECOND.search(value):
            return "'%s=%s' holds two settings: write a ; between them, not a comma" % (
                name,
                value,
            )
        if choices and value.lower() not in [str(c).lower() for c in choices]:
            return "'%s' is no %s: write one of %s" % (
                value,
                name,
                ", ".join(str(c) for c in choices),
            )
        if kind == "number":
            try:
                number = float(value)
            except ValueError:
                return "%s takes a number, not '%s': write a number" % (name, value)
            low, high = declared.get("min"), declared.get("max")
            if low is not None and number < low:
                return "%s is at least %s, not %s: write %s or more" % (
                    name,
                    low,
                    value,
                    low,
                )
            if high is not None and number > high:
                return "%s is at most %s, not %s: write %s or less" % (
                    name,
                    high,
                    value,
                    high,
                )
        elif kind == "flag" and value.lower() not in _ManifestActionsInternal._FLAGS:
            return "%s is on or off, not '%s': write on or off" % (name, value)
        elif (
            kind == "color"
            and not choices
            and not _ManifestActionsInternal._COLOR.match(value)
        ):
            return (
                "%s takes a colour as #RRGGBB, not '%s': write its six hex digits (#ffcc00)"
                % (name, value)
            )
        elif kind == "object" and not _ManifestActionsInternal._OBJECT.match(value):
            return (
                "%s names one object, as the scene spells it, not '%s': write its name alone, or the end of "
                "its path that tells it apart where two share it" % (name, value)
            )
        return None


class ManifestActions(_ManifestActionsInternal):
    """Rows read against a template's declared actions (``ColumnMap.actions``)."""

    #: The row cells a word may take or need; Settings and the notes are every word's.
    CELLS = ("target", "with", "clip", "members", "label", "to")
    #: What each cell holds, as the fix of a needed one left empty says it.
    HOLDS = {
        "target": "the part it is about, by its scene name",
        "with": "the tool in hand",
        "clip": "the take it plays (its step's ID; <ID>_<n> where the step plays several)",
        "members": "what its take animates",
        "label": "the words it shows",
        "to": "where it goes (next, a step's ID, restart, exit or retry)",
    }
    #: When a word happens: as its step starts, on the user, as it ends, or never (a note).
    WHEN = ("start", "user", "end", "none")
    #: Where a choice may go besides a step: the next step, the first, out, or nowhere (a wrong answer).
    DESTINATIONS = ("next", "restart", "exit", "retry")
    #: The types a setting may declare (``choices`` alone makes one a choice); ``object`` is one
    #: object's exact name (a node's, or the end of its path that tells it apart where two share it);
    #: ``state`` one of its targets' indicator-light states, as the delivery's emissive groups name
    #: them (``off``, ``amber``), held to them by a reader that knows the delivery.
    SETTING_TYPES = (
        "text",
        "number",
        "flag",
        "choice",
        "color",
        "clip",
        "step",
        "span",
        "object",
        "state",
    )
    #: The parts a word is made of, each setting belonging to one: what starts it, what must be
    #: true first, what it does, what a shown thing looks like and where a placed thing sits.
    PRIMITIVES = ("trigger", "requirement", "effect", "look", "place")
    #: What a template's ``settings`` declares of each setting, once for every word.
    SETTING_KEYS = ("type", "primitive", "min", "max", "help")
    #: What a word adds of its own to a setting it takes.
    WORD_SETTING_KEYS = ("choices", "default", "required")
    #: What a word that places something places it for (its ``places``), read by Placement: where
    #: the user stands, where a pointer points from, where a picture is taken from, where a
    #: held part is brought. Its object is its ``where`` setting, else the helper made for it.
    PURPOSES = ("stand", "point", "view", "hold")
    #: The setting a word that places something names its object in.
    WHERE = "where"
    #: The span that is its own step's: a shown thing lasts until its step ends.
    SPAN_STEP = "step"

    @staticmethod
    def validate_actions(value: Any) -> List[str]:
        """What is wrong with a template's ``actions`` (``ColumnMap``'s field validator)."""
        if not isinstance(value, dict):
            return ["expected {word: declaration}"]
        errors: List[str] = []
        for word, declared in value.items():
            if not isinstance(declared, dict):
                errors.append("%r: expected a declaration (a mapping)" % word)
                continue
            if word != ManifestActions.word(word):
                errors.append(
                    "%r: a word is lower case with underscores (%r)"
                    % (word, ManifestActions.word(word))
                )
            for key in ("cells", "needs", "distinct"):
                unknown = [
                    c for c in declared.get(key) or () if c not in ManifestActions.CELLS
                ]
                if unknown:
                    errors.append(
                        "%r: %s names %s, not one of %s"
                        % (word, key, unknown, list(ManifestActions.CELLS))
                    )
            for group in declared.get("needs_any") or ():
                own = list(declared.get("cells") or ()) + list(
                    declared.get("settings") or {}
                )
                unknown = [c for c in group if c not in own]
                if unknown or len(group) < 2:
                    errors.append(
                        "%r: needs_any lists two or more of its cells or settings, not %s"
                        % (word, list(group))
                    )
            places = declared.get("places")
            if places is not None:
                if places not in ManifestActions.PURPOSES:
                    errors.append(
                        "%r: places one of %s, not %r"
                        % (word, list(ManifestActions.PURPOSES), places)
                    )
                if ManifestActions.WHERE not in (declared.get("settings") or {}):
                    errors.append(
                        "%r: a word that places something takes a '%s' setting"
                        % (word, ManifestActions.WHERE)
                    )
            for key in ("needs", "distinct"):
                missing = [
                    c
                    for c in declared.get(key) or ()
                    if c not in (declared.get("cells") or ())
                ]
                if missing:
                    errors.append(
                        "%r: %s %s, which its cells do not list" % (word, key, missing)
                    )
            if declared.get("when") not in (None,) + ManifestActions.WHEN:
                errors.append(
                    "%r: when is one of %s" % (word, list(ManifestActions.WHEN))
                )
            for name, setting in (declared.get("settings") or {}).items():
                keys = ManifestActions.WORD_SETTING_KEYS + ManifestActions.SETTING_KEYS
                kind = setting.get("type") if isinstance(setting, dict) else None
                if (
                    not isinstance(setting, dict)
                    or set(setting) - set(keys)
                    or kind not in (None,) + ManifestActions.SETTING_TYPES
                ):
                    errors.append(
                        "%r: setting %r is {%s}: its type (one of %s), primitive and bounds are the "
                        "template's settings'"
                        % (word, name, ", ".join(ManifestActions.WORD_SETTING_KEYS), list(ManifestActions.SETTING_TYPES))
                    )
            says = declared.get("says")
            if says is not None and not isinstance(says, (str, dict)):
                errors.append("%r: says is a sentence, or {condition: sentence}" % word)
        return errors

    @staticmethod
    def validate_settings(value: Any) -> List[str]:
        """What is wrong with a template's ``settings`` (``ColumnMap``'s field validator): each
        declared once, ``{type, primitive, min, max, help}``, its type one of
        :attr:`SETTING_TYPES` and its primitive one of :attr:`PRIMITIVES`."""
        if not isinstance(value, dict):
            return ["expected {setting: declaration}"]
        errors: List[str] = []
        for name, declared in value.items():
            if name != ManifestActions.word(name):
                errors.append(
                    "%r: a setting is lower case with underscores (%r)"
                    % (name, ManifestActions.word(name))
                )
            if not isinstance(declared, dict):
                errors.append("%r: expected a declaration (a mapping)" % name)
                continue
            unknown = set(declared) - set(ManifestActions.SETTING_KEYS)
            if unknown:
                errors.append(
                    "%r: declares %s, not one of %s: a word gives its own %s"
                    % (
                        name,
                        sorted(unknown),
                        list(ManifestActions.SETTING_KEYS),
                        ", ".join(ManifestActions.WORD_SETTING_KEYS),
                    )
                )
            if declared.get("type") not in ManifestActions.SETTING_TYPES:
                errors.append(
                    "%r: its type is one of %s" % (name, list(ManifestActions.SETTING_TYPES))
                )
            if declared.get("primitive") not in ManifestActions.PRIMITIVES:
                errors.append(
                    "%r: it belongs to one of %s (its primitive)"
                    % (name, list(ManifestActions.PRIMITIVES))
                )
        return errors

    @staticmethod
    def check_declared(actions: Mapping[str, Any], settings: Mapping[str, Any]) -> List[str]:
        """What is wrong between a template's words and its ``settings``: a setting a word takes
        that the template does not declare, a word restating a declared type, primitive or bound
        otherwise, a choice no word gives its choices, and a placing word whose ``where`` is no
        object. Empty where they agree."""
        errors: List[str] = []
        if not isinstance(actions, dict) or not isinstance(settings, dict):
            return errors
        for word, declared in actions.items():
            if not isinstance(declared, dict):
                continue
            for name, own in (declared.get("settings") or {}).items():
                table = settings.get(name)
                if not isinstance(table, dict):
                    errors.append(
                        "%r: takes the setting %r, which the template's settings do not declare: "
                        "declare it there once (type and primitive)" % (word, name)
                    )
                    continue
                for key in ManifestActions.SETTING_KEYS:
                    if isinstance(own, dict) and key in own and own[key] != table.get(key):
                        errors.append(
                            "%r: restates %r's %s as %r, which the template's settings declare %r: "
                            "a setting means one thing in every word" % (word, name, key, own[key], table.get(key))
                        )
                if table.get("type") == "choice" and not (own or {}).get("choices"):
                    errors.append("%r: gives the choice %r no choices" % (word, name))
            if declared.get("places") is not None and (settings.get(ManifestActions.WHERE) or {}).get("type") != "object":
                errors.append(
                    "%r: a word that places something takes '%s' declared of type object"
                    % (word, ManifestActions.WHERE)
                )
        return errors

    @staticmethod
    def resolve(actions: Mapping[str, Any], settings: Mapping[str, Any]) -> Dict[str, Any]:
        """*actions* with each setting a word takes declared whole: the template's declaration
        (*settings*) under what the word gives of its own (its choices, default, required), so
        every reader of a word reads one declaration. A word's own key wins; a name the template
        does not declare is left as the word gives it (:meth:`check_declared` says so)."""
        out: Dict[str, Any] = {}
        for word, declared in (actions or {}).items():
            if not isinstance(declared, dict) or not declared.get("settings"):
                out[word] = declared
                continue
            joined = {
                name: dict(settings.get(name) or {}, **(own or {}))
                for name, own in declared["settings"].items()
            }
            out[word] = dict(declared, settings=joined)
        return out

    @staticmethod
    def word(cell: str) -> str:
        """An Action cell as its word: lower case, anything but letters and digits an underscore
        (``"Pick Up"`` -> ``pick_up``). Nothing else about a cell is normalised."""
        return re.sub(r"[^0-9a-z]+", "_", (cell or "").strip().lower()).strip("_")

    @staticmethod
    def names(cell: str) -> List[str]:
        """The names a cell lists, one per line, in order."""
        return [n.strip() for n in (cell or "").splitlines() if n.strip()]

    @staticmethod
    def read_settings(cell: str) -> Tuple[Dict[str, str], List[str]]:
        """``(settings, errors)`` for a Settings cell: ``key=value`` pairs split by ``;`` or a line.

        A part that is no pair, or a name given twice, is an error; the names are lower-cased.
        """
        settings: Dict[str, str] = {}
        errors: List[str] = []
        for part in re.split(r"[;\n]", cell or ""):
            part = part.strip()
            if not part:
                continue
            name, equals, value = part.partition("=")
            name, value = name.strip().lower(), value.strip()
            if not equals or not name or not value:
                errors.append("'%s' is no setting: write name=value" % part)
            elif name in settings:
                errors.append("%s is set twice: keep one" % name)
            else:
                settings[name] = value
        return settings, errors

    @staticmethod
    def setting(
        row: Mapping[str, Any], declared: Optional[Mapping[str, Any]], name: str
    ) -> Optional[str]:
        """The value *row* gives the setting *name*, else its declared default (None: neither)."""
        value = (row.get("settings") or {}).get(name)
        if value is not None:
            return value
        default = ((declared or {}).get("settings") or {}).get(name, {}).get("default")
        return None if default is None else str(default)

    @staticmethod
    def check_row(
        row: Mapping[str, Any], actions: Mapping[str, Any], headers: Mapping[str, str]
    ) -> List[Issue]:
        """What is wrong with one row's own cells, each with its fix.

        Parameters:
            row: The row as the reader made it: ``word``, ``assets`` (its Target's names), ``with``
                and ``members`` (names), ``clip``, ``label``, ``to``, ``settings`` and
                ``setting_errors``.
            actions: The template's declarations.
            headers: ``{cell key: the sheet's own header}`` for the messages (``action``,
                ``target``, ``settings`` and each of :attr:`CELLS`).
        """
        issues: List[Issue] = []

        def head(key: str) -> str:
            return headers.get(key) or key.title()

        word = row.get("word", "")
        filled = {
            "target": bool(row.get("assets")),
            "with": bool(row.get("with")),
            "clip": bool(row.get("clip")),
            "members": bool(row.get("members")),
            "label": bool(row.get("label")),
            "to": bool(row.get("to")),
        }
        declared = actions.get(word)
        if declared is None:
            if word:
                known = ", ".join(sorted(w for w in actions if w))
                issues.append(
                    (
                        head("action"),
                        "'%s' is no action of this template: write one of %s"
                        % (word, known),
                    )
                )
            else:
                issues.append(
                    (
                        head("action"),
                        "the row fills %s and names no action: write one in %s"
                        % (
                            ManifestActions._joined(
                                [head(k) for k, on in filled.items() if on]
                                or [head("settings")]
                            ),
                            head("action"),
                        ),
                    )
                )
            return issues
        shown = word or "a row with no %s" % head("action")
        cells = declared.get("cells") or ()
        for key, on in filled.items():
            if on and key not in cells:
                issues.append(
                    (head(key), "%s takes no %s: clear it" % (shown, head(key)))
                )
        for key in declared.get("needs") or ():
            if not filled.get(key) and headers and key not in headers:
                continue  # the sheet has no such column: said once for the sheet (missing_columns)
            if not filled.get(key):
                issues.append(
                    (
                        head(key),
                        "%s needs its %s: write %s"
                        % (shown, head(key), ManifestActions.HOLDS[key]),
                    )
                )
        given = row.get("settings") or {}
        for group in declared.get("needs_any") or ():
            if not any(filled.get(key) or given.get(key) for key in group):
                named = [head(k) if k in ManifestActions.CELLS else k for k in group]
                issues.append(
                    (
                        head(group[0]),
                        "%s needs its %s: fill one of them"
                        % (shown, " or its ".join(named)),
                    )
                )
        to = row.get("to") or ""
        for error in row.get("setting_errors") or ():
            issues.append((head("settings"), error))
        settings = declared.get("settings") or {}
        for name, value in (row.get("settings") or {}).items():
            if name not in settings:
                issues.append(
                    (
                        head("settings"),
                        "%s has no setting '%s': %s"
                        % (shown, name, ManifestActions._instead(settings)),
                    )
                )
                continue
            error = ManifestActions._value_error(name, value, settings[name])
            if error:
                issues.append((head("settings"), error))
        for name, setting in settings.items():
            if setting.get("required") and name not in (row.get("settings") or {}):
                issues.append(
                    (
                        head("settings"),
                        "%s needs its %s: write %s=<value>" % (shown, name, name),
                    )
                )
        if (
            to
            and "to" in cells
            and to.lower() in ManifestActions.DESTINATIONS
            and to != to.lower()
        ):
            issues.append(
                (
                    head("to"),
                    "'%s' is written in lower case: write %s" % (to, to.lower()),
                )
            )
        return issues

    @staticmethod
    def check_step(
        rows: Sequence[Mapping[str, Any]],
        actions: Mapping[str, Any],
        headers: Mapping[str, str],
        step_id: str,
        step_ids: Sequence[str],
        legal=None,
    ) -> List[Tuple[int, Issue, bool]]:
        """What is wrong between one step's rows and the sheet: ``[(row index, issue, is a warning)]``.

        A take two rows play, an effect row or an ``after`` naming a take its step lacks, a choice
        going to a step the sheet lacks, two rows of a word sharing a cell the word declares
        ``distinct`` (two buttons of one Label), two rows whose own helpers would share a name
        (``ManifestPlacement.helper_names``); and, as a warning, a take not named for its step (its
        ID, or its ID and ``_<n>``) and a span that names a step the sheet has passed (what it
        shows would stay).

        Parameters:
            rows: The step's rows, read.
            actions: The template's declarations.
            headers: ``{cell key: header}`` for the messages.
            step_id: The step's ID as the reader spells it.
            step_ids: Every step's ID, as the reader spells them.
            legal: How a step ID written in a cell is spelled as the reader spells it (``B03.5`` -> ``B03_5``).
        """
        legal = legal or (lambda text: text)

        def head(key: str) -> str:
            return headers.get(key) or key.title()

        out: List[Tuple[int, Issue, bool]] = []
        # Two rows naming no helper that would be given one name (ManifestPlacement.helper_names: one purpose, one
        # target): the second must say which object it happens at.
        from pythontk.core_utils.engines.shots.manifest.manifest_placement import (
            ManifestPlacement,
        )

        placing = [
            dict(row, places=(actions.get(row.get("word", "")) or {}).get("places"))
            for row in rows
        ]
        helpers: Dict[str, int] = {}
        for n, helper in enumerate(
            ManifestPlacement.helper_names(step_id, placing) if step_id else ()
        ):
            if helper is None:
                continue
            if helper in helpers:
                out.append(
                    (
                        n,
                        (
                            head("settings"),
                            "an earlier row of the step names its own helper %s too: give this one "
                            "its own object (where=<name>), or both would be the one helper"
                            % helper,
                        ),
                        False,
                    )
                )
            else:
                helpers[helper] = n
        takes: Dict[str, int] = {}
        for n, row in enumerate(rows):
            if row.get("take") and row.get("clip"):
                if row["clip"] in takes:
                    out.append(
                        (
                            n,
                            (
                                head("clip"),
                                "the take %s is played by an earlier row of the step: "
                                "name this row's own take" % row["clip"],
                            ),
                            False,
                        )
                    )
                else:
                    takes[row["clip"]] = n
        for n, row in enumerate(rows):
            declared = actions.get(row.get("word", "")) or {}
            clip = row.get("clip") or ""
            if (
                clip
                and row.get("take")
                and not re.fullmatch(r"%s(?:_\d+)?" % re.escape(step_id), legal(clip))
            ):
                out.append(
                    (
                        n,
                        (
                            head("clip"),
                            "the take %s is not named for its step: %s, or %s_<n>, "
                            "says which step it is everywhere"
                            % (clip, step_id, step_id),
                        ),
                        True,
                    )
                )
            if (
                clip
                and not row.get("take")
                and "clip" in (declared.get("cells") or ())
                and clip not in takes
            ):
                out.append(
                    (
                        n,
                        (
                            head("clip"),
                            "no row of the step plays a take %s: name one of %s, or clear it"
                            % (clip, ", ".join(takes) or "its takes (it has none)"),
                        ),
                        False,
                    )
                )
            for name, setting in (declared.get("settings") or {}).items():
                value = (row.get("settings") or {}).get(name)
                if (
                    value
                    and setting.get("type") == "clip"
                    and (value not in takes or takes[value] == n)
                ):
                    others = ", ".join(t for t, at in takes.items() if at != n)
                    out.append(
                        (
                            n,
                            (
                                head("settings"),
                                "%s names the take %s, which no other row of the step plays: %s"
                                % (
                                    name,
                                    value,
                                    "name one of %s, or clear it" % others
                                    if others
                                    else "clear it",
                                ),
                            ),
                            False,
                        )
                    )
                if (
                    value
                    and setting.get("type") == "step"
                    and legal(value) not in step_ids
                ):
                    out.append(
                        (
                            n,
                            (
                                head("settings"),
                                "%s names the step %s, which the sheet lacks: write a step's ID"
                                % (name, value),
                            ),
                            False,
                        )
                    )
                if (
                    value
                    and setting.get("type") == "span"
                    and value.lower() != ManifestActions.SPAN_STEP
                ):
                    if legal(value) not in step_ids:
                        out.append(
                            (
                                n,
                                (
                                    head("settings"),
                                    "the sheet has no step %s for %s: write a step's ID, or %s "
                                    "(while its own step runs)"
                                    % (value, name, ManifestActions.SPAN_STEP),
                                ),
                                False,
                            )
                        )
                    elif step_id in step_ids and step_ids.index(
                        legal(value)
                    ) <= step_ids.index(step_id):
                        # It ends as that step starts, and the sheet is past it: only a jump back ends it.
                        out.append(
                            (
                                n,
                                (
                                    head("settings"),
                                    "%s names %s, which the sheet is past by this step, so what "
                                    "the row shows stays: name a later step, or %s"
                                    % (name, value, ManifestActions.SPAN_STEP),
                                ),
                                True,
                            )
                        )
            to = row.get("to") or ""
            if (
                to
                and "to" in (declared.get("cells") or ())
                and to.lower() not in ManifestActions.DESTINATIONS
                and legal(to) not in step_ids
            ):
                out.append(
                    (
                        n,
                        (
                            head("to"),
                            "the sheet has no step %s: write a step's ID, or one of %s"
                            % (to, ", ".join(ManifestActions.DESTINATIONS)),
                        ),
                        False,
                    )
                )
        seen: Dict[Tuple[str, str, str], int] = {}
        for n, row in enumerate(rows):
            word = row.get("word", "")
            for key in (actions.get(word) or {}).get("distinct") or ():
                said = " ".join(str(row.get(key) or "").split()).lower()
                if said and seen.setdefault((word, key, said), n) != n:
                    out.append(
                        (
                            n,
                            (
                                head(key),
                                "another %s row of the step has the %s '%s': give each its own"
                                % (word, head(key), row.get(key)),
                            ),
                            False,
                        )
                    )
        return out

    @staticmethod
    def _instead(settings: Mapping[str, Any]) -> str:
        """The fix for a setting a word or a step does not have: the ones it has, else to clear it."""
        return (
            "clear it, or write one of %s" % ", ".join(sorted(settings))
            if settings
            else "clear it (it has none)"
        )

    @staticmethod
    def missing_columns(
        rows: Sequence[Mapping[str, Any]], actions: Mapping[str, Any]
    ) -> List[Issue]:
        """The columns *rows* need and their sheet's header lacks, each once: ``[(the Action
        column's header, what is missing and its fix)]``. A row says its own empty cell
        (:meth:`check_row`); a column the sheet does not have is the sheet's, said once.

        Parameters:
            rows: Rows as the reader made them, each with its ``word`` and ``headers``
                (``{cell key: the sheet's own header}`` for the columns its header carries).
            actions: The template's declarations.
        """
        lacking: Dict[str, List[str]] = {}
        action = "Action"
        for row in rows:
            headers = row.get("headers") or {}
            action = headers.get("action") or action
            for key in (actions.get(row.get("word", "")) or {}).get("needs") or ():
                if (
                    headers
                    and key not in headers
                    and row["word"] not in lacking.setdefault(key, [])
                ):
                    lacking[key].append(row["word"])
        return [
            (
                action,
                "the sheet has no %s column, which %s need%s: add the column"
                % (
                    key.title(),
                    ManifestActions._joined(sorted(words)),
                    "" if len(words) > 1 else "s",
                ),
            )
            for key, words in lacking.items()
            if words
        ]

    @staticmethod
    def check_step_row(
        to: str,
        settings: Mapping[str, str],
        errors: Sequence[str],
        declared: Mapping[str, Any],
        headers: Mapping[str, str],
        step_ids: Sequence[str],
        legal=None,
    ) -> List[Issue]:
        """What is wrong with a step row's own To and Settings, each with its fix.

        Parameters:
            to: Its To cell: where the step goes by itself once done.
            settings: Its Settings cell, read (:meth:`read_settings`).
            errors: What that reading could not read.
            declared: The settings a step may have (the template's ``step_settings``).
            headers: ``{cell key: header}`` for the messages.
            step_ids: Every step's ID, as the reader spells them.
            legal: How a step ID written in a cell is spelled as the reader spells it.
        """
        legal = legal or (lambda text: text)
        issues: List[Issue] = [
            (headers.get("settings") or "Settings", error) for error in errors
        ]
        if (
            to
            and to.lower() not in ("next", "restart", "exit")
            and legal(to) not in step_ids
        ):
            issues.append(
                (
                    headers.get("to") or "To",
                    "the sheet has no step %s: write a step's ID, next, "
                    "restart or exit" % to,
                )
            )
        for name, value in settings.items():
            if name not in declared:
                issues.append(
                    (
                        headers.get("settings") or "Settings",
                        "a step has no setting '%s': %s"
                        % (name, ManifestActions._instead(declared)),
                    )
                )
                continue
            error = ManifestActions._value_error(name, value, declared[name])
            if error:
                issues.append((headers.get("settings") or "Settings", error))
        return issues

    @staticmethod
    def sentence(row: Mapping[str, Any], actions: Mapping[str, Any]) -> str:
        """The row said as a sentence, from its word's ``says`` (empty: the word declares none).

        ``says`` is one sentence, or ``{condition: sentence}``: the first whose condition
        (``name=value``, of a cell or a setting, its default counted) holds, else the one under ``""``.
        """
        declared = actions.get(row.get("word", ""))
        if not declared:
            return ""
        says = declared.get("says") or ""
        values = {
            "target": ManifestActions._joined(row.get("assets") or ()),
            "with": ManifestActions._joined(row.get("with") or ()),
            "members": ManifestActions._joined(row.get("members") or ()),
            "clip": row.get("clip") or "",
            "to": row.get("to") or "",
            "label": ", ".join(ManifestActions.names(row.get("label") or "")),
        }
        for name in declared.get("settings") or {}:
            values.setdefault(name, ManifestActions.setting(row, declared, name) or "")
        if isinstance(says, dict):
            chosen = says.get("", "")
            for condition, text in says.items():
                name, equals, value = condition.partition("=")
                if (
                    equals
                    and str(values.get(name.strip(), "")).lower()
                    == value.strip().lower()
                ):
                    chosen = text
                    break
            says = chosen
        return ManifestActions._fill(says, values)

    @staticmethod
    def objects(row: Mapping[str, Any], actions: Mapping[str, Any]) -> List[str]:
        """Every object a row names, each once: its Target, With and Members, and each setting its
        word declares an object (``where``) -- what a sheet read alone stands for its delivery by."""
        out: List[str] = []
        named = (
            list(row.get("assets") or ())
            + list(row.get("with") or ())
            + list(row.get("members") or ())
        )
        declared = (actions.get(row.get("word", "")) or {}).get("settings") or {}
        named += [
            str(value)
            for name, value in (row.get("settings") or {}).items()
            if (declared.get(name) or {}).get("type") == "object" and value
        ]
        for name in named:
            if name not in out:
                out.append(name)
        return out

    @staticmethod
    def members(row: Mapping[str, Any]) -> List[str]:
        """What the take a row plays animates: its Members, else its Target's and its With's names."""
        listed = list(row.get("members") or ())
        if listed:
            return listed
        out: List[str] = []
        for name in list(row.get("assets") or ()) + list(row.get("with") or ()):
            if name not in out:
                out.append(name)
        return out
