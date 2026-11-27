# !/usr/bin/python
# coding=utf-8
"""Pure Shot Manifest data model + CSV parser.

DCC-agnostic — no ``maya`` / ``bpy`` / Qt.  Holds the step/object graph a
structured production CSV parses into, the column-mapping schema, behavior
detection, and the assessment/plan result dataclasses.  Duration resolution and
key emission (which reach a scene) live in the hooked engine layer, not here.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

from pythontk import SchemaSpec
from pythontk.str_utils._str_utils import StrUtils
from pythontk.core_utils.engines.shots.shot_model import ShotStore
from pythontk.net_utils.remote_file import RemoteFile

log = logging.getLogger(__name__)


__all__ = [
    "BuilderObject",
    "BuilderStep",
    "PlannedShot",
    "PlannedInsert",
    "ObjectStatus",
    "StepStatus",
    "ColumnMap",
    "ShotPairing",
    "Action",
    "FitMode",
    "DEFAULT_INITIAL_SHOT_LENGTH",
    "DEFAULT_FIT_MODE",
    "AUDIO_PLACEHOLDER_DURATION",
    "ManifestModel",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class _ManifestModelInternal(object):
    """Internal helpers for ManifestModel."""

    @staticmethod
    def _resolve_columns(header: List[str], col_map: ColumnMap) -> _ResolvedColumns:
        """Match header cell text to column indices via *col_map* aliases.

        A header written in cells (one of ``col_map.cell_marks`` among its columns, under a
        template that declares its actions) needs no description column: a cell row is read
        by its cells, and its notes column is for people.

        Raises:
            ValueError: If a required column cannot be found.
        """
        normalized = [c.strip().lower() for c in header]
        marks = {m.strip().lower() for m in col_map.cell_marks}
        in_cells = bool(col_map.actions) and bool(marks) and any(c in marks for c in normalized)

        def _find_optional(aliases: Tuple[str, ...]) -> Optional[int]:
            for alias in aliases:
                try:
                    return normalized.index(alias.lower())
                except ValueError:
                    continue
            return None

        def _find(aliases: Tuple[str, ...], field_name: str) -> int:
            idx = _find_optional(aliases)
            if idx is None:
                raise ValueError(
                    f"Column '{field_name}' not found in header row. "
                    f"Expected one of {aliases!r}, got {header}"
                )
            return idx

        resolved = _ResolvedColumns(
            step_id=_find(col_map.step_id, "step_id"),
            description=(
                _find_optional(col_map.description) if in_cells else _find(col_map.description, "description")
            ),
            assets=_find(col_map.assets, "assets"),
            audio=_find_optional(col_map.audio) if col_map.audio else None,
            behaviors=_find_optional(col_map.behaviors) if col_map.behaviors else None,
        )
        for key, aliases in col_map.metadata_pass.items():
            idx = _find_optional(aliases)
            if idx is not None:
                resolved.metadata_pass[key] = idx
        for key, aliases in col_map.row_pass.items():
            idx = _find_optional(aliases)
            if idx is not None:
                resolved.row_pass[key] = idx
        return resolved

    @staticmethod
    def _header_names(header: List[str], cols: "_ResolvedColumns") -> Dict[str, str]:
        """``{column key: the sheet's own header}`` for an issue to name its cell by: ``target``
        (the assets column), ``step``, and each other column the header carries (``action``,
        ``with``, ``clip``..., ``description``, ``audio``, a pass-through column's key). A
        column the header lacks has no key."""

        def at(index: Optional[int]) -> str:
            return header[index].strip() if index is not None and index < len(header) else ""

        names = {"target": at(cols.assets), "step": at(cols.step_id)}
        names.update({key: at(index) for key, index in cols.metadata_pass.items()})
        names.update({key: at(index) for key, index in cols.row_pass.items()})
        for key in ("description", "audio", "behaviors"):
            if getattr(cols, key) is not None:
                names[key] = at(getattr(cols, key))
        return names

    @staticmethod
    def _reads_by_cells(row: List[str], cols: "_ResolvedColumns", actions: Dict[str, Any]) -> bool:
        """Whether *row*, of a sheet written in cells (``ColumnMap.cell_marks``), is read by
        its cells: its Action cell is filled, the template declares what a row with none is
        (the empty word), or it fills a cell only the cell format has (which then asks for
        its action)."""

        def filled(key: str) -> bool:
            index = cols.row_pass.get(key)
            return index is not None and index < len(row) and bool(row[index].strip())

        return filled("action") or "" in actions or any(
            filled(key) for key in ("with", "members", "label", "to", "settings")
        )

    @staticmethod
    def _read_cells(
        record: Dict[str, Any], actions: Dict[str, Any], headers: Dict[str, str], keyed: set, shots: str,
        step_looks: str = "key",
    ) -> None:
        """Read *record*, a row's cells, against the template's *actions*, in place.

        Adds ``cells`` (True), ``word``, ``with`` and ``members`` as names, ``settings`` as
        a mapping, ``take`` (its word plays the take its Clip names), ``places`` (what its
        word places a helper for, as the template declares it; empty: nothing), ``says`` (the row as a
        sentence), ``issues`` and ``warnings`` (``[(header, what is wrong and its fix)]``),
        ``headers`` (``{cell key: the sheet's own header}``, for a reader's own issues)
        and ``shot``: the ``[name, behaviors]`` it puts in its shot. A row that plays a take
        lists what the take animates; a row of a word this host keys (*keyed*) lists its
        Target with that word where it names a take; one that names none belongs to its step,
        and lists its Target where every step is a shot (*shots* is ``every``: a step with no
        shot has nothing to key it on), with that word where the step's looks are this host's
        to key (*step_looks* is ``key``) and with none where they are left to the tool that
        runs the steps (``leave``); a row with no word lists its Target; any other row lists
        nothing.
        """
        from pythontk.core_utils.engines.shots.manifest.manifest_actions import ManifestActions

        word = ManifestActions.word(record.get("action", ""))
        declared = actions.get(word) or {}
        settings, errors = ManifestActions.read_settings(record.get("settings", ""))
        record.update(
            {
                "cells": True,
                "word": word,
                "with": ManifestActions.names(record.get("with", "")),
                "members": ManifestActions.names(record.get("members", "")),
                "settings": settings,
                "setting_errors": errors,
                "warnings": [],
                "headers": headers,
            }
        )
        record["take"] = bool(declared.get("take")) and bool(record.get("clip"))
        record["places"] = declared.get("places") or ""
        record["issues"] = ManifestActions.check_row(record, actions, headers)
        record["says"] = ManifestActions.sentence(record, actions)
        if record["take"]:
            shot = [[name, []] for name in ManifestActions.members(record)]
        elif not word and word in actions:
            shot = [[name, []] for name in record["assets"]]
        elif word in keyed and word in actions and record.get("clip"):
            shot = [[name, [word]] for name in record["assets"]]  # keyed into the take it names
        elif word in keyed and word in actions and shots == "every":
            shot = [[name, [word] if step_looks == "key" else []] for name in record["assets"]]
        else:
            shot = []
        record["shot"] = shot

    @staticmethod
    def _parse_rows(
        rows: List[List[str]], col_map: "ColumnMap", source: str
    ) -> List["BuilderStep"]:
        """Assemble steps from already-read CSV *rows* (see :meth:`parse_csv`).

        Exclusions and post-processing are the caller's; each step records
        its asset cell's ``(row, column)`` so a column can be written back.
        With ``col_map.layouts`` the rows are read once per layout and the
        one reading the most steps is kept (the first on a tie); only its
        warnings are logged.
        """
        candidates = [col_map]
        if col_map.layouts:
            base = {k: v for k, v in col_map.to_dict().items() if k != "layouts"}
            candidates = [
                ColumnMap.from_dict({**base, **layout}) for layout in col_map.layouts
            ]
        best, best_notes = None, []
        for candidate in candidates:
            notes: List[str] = []
            steps = _ManifestModelInternal._parse_layout(rows, candidate, source, notes)
            if best is None or len(steps) > len(best):
                best, best_notes = steps, notes
        for note in best_notes:
            log.warning(note)
        return best

    @staticmethod
    def _parse_layout(
        rows: List[List[str]], col_map: "ColumnMap", source: str, notes: List[str]
    ) -> List["BuilderStep"]:
        """:meth:`_parse_rows` for one layout (``col_map.layouts`` unread),
        its warnings appended to *notes* rather than logged."""
        cols: Optional[_ResolvedColumns] = None
        steps: List[BuilderStep] = []
        seen_ids: set = set()
        unread: List[str] = []  # step-ID cells no pattern took, a few
        current_section = ""
        current_section_title = ""
        current_step: Optional[BuilderStep] = None
        step_id_aliases = {a.lower() for a in col_map.step_id}
        step_res = [re.compile(p) for p in col_map.step_pattern]
        section_res = [re.compile(p) for p in col_map.section_pattern]
        _first = _ManifestModelInternal._first_match
        _group = _ManifestModelInternal._group
        asset_excludes = {v.upper() for v in col_map.exclude_values.get("assets", ())}
        # A row cell the template treats as empty (``exclude_values``) reads "".
        pass_excludes = {
            key: {v.upper() for v in col_map.exclude_values.get(key, ())}
            for key in col_map.row_pass
        }
        # Per step: whether the header above it carried the clip column (a
        # sheet repeats its header per section, with different columns).
        clip_header: Dict[str, bool] = {}

        behavior_source = col_map.behavior_source
        from pythontk.core_utils.engines.shots.manifest.behaviors import Behaviors
        from pythontk.core_utils.engines.shots.manifest.manifest_actions import ManifestActions

        # The cell format: a template that declares its actions reads a row with an Action
        # by its cells (and one with none, where it declares the empty word), never by its
        # sentence.  A row it does not read so is read as before.
        actions = col_map.actions
        keyed = set(Behaviors.list_behaviors()) if actions else set()
        headers: Dict[str, str] = {}
        in_cells = False  # whether the header above says the sheet is written in cells
        marks = {m.strip().lower() for m in col_map.cell_marks}
        cell_rows: Dict[str, List[Dict[str, Any]]] = {}
        celled: set = set()  # the steps under a header that says the sheet is in cells
        own_cells: Dict[str, Tuple[List[str], Dict[str, str]]] = {}  # a step row's Settings errors and headers
        # A row no step reads is said with its row in the sheet, at the step before it (else the
        # next one read): never dropped. *unheld*: under a step row that was not read, whose
        # issue speaks for the rows under it.
        stray: List[Tuple[str, str]] = []
        unheld = False

        def _unread(key: str, text: str) -> None:
            issue = (headers.get(key) or key.title(), text)
            (steps[-1].issues if steps else stray).append(issue)

        def _marked(header: List[str]) -> bool:
            return not marks or any(c.strip().lower() in marks for c in header)

        phrases = Behaviors.phrase_table()  # once per parse, not once per row
        # Objects the prose names, per step -- listed only where every asset
        # cell of the step is blank (``object_source``): a cell that says
        # anything, an excluded "N/A" included, is the sheet speaking for
        # the step's objects itself.
        prose_objects: Dict[str, Dict[str, BuilderObject]] = {}
        assets_spoken: set = set()
        read_prose = col_map.object_source == "column_else_description"

        def _note_subjects(step_id: str, description: str) -> None:
            listed = prose_objects.setdefault(step_id, {})
            for subject, behavior in Behaviors.subjects(description, phrases):
                name = _ManifestModelInternal.prose_name(
                    subject, col_map.object_case, col_map.object_name_rule
                )
                obj = listed.setdefault(
                    name.casefold(), BuilderObject(name=name, origin="description")
                )
                if behavior not in obj.behaviors:
                    obj.behaviors.append(behavior)

        def _row_behaviors(description: str, cell: str):
            """``(behaviors, speaks)`` for one row from the template's source;
            *speaks* is whether the row said anything about behaviors itself
            (else it inherits its step's)."""
            if behavior_source == "column" or (
                behavior_source == "column_else_description" and cell
            ):
                return Behaviors.from_cell(cell, phrases), bool(cell)
            return Behaviors.detect(description, phrases), bool(description)

        def _asset_names(cell: str) -> List[str]:
            # One object per line: a name can't hold a newline, so a cell
            # listing several (as an export's Objects cell does) splits safely.
            names = (n.strip() for n in cell.splitlines())
            return [n for n in names if n and n.upper() not in asset_excludes]

        def _add_object(step: BuilderStep, name: str, behaviors: List[str]) -> None:
            # One object however many rows name it: its behaviors join the
            # earlier row's, in row order (repeats kept), as the build keys
            # them -- two objects were anchored row by row by Assess and Apply.
            for obj in step.objects:
                if obj.name == name and obj.kind == "scene":
                    obj.behaviors.extend(behaviors)
                    return
            step.objects.append(BuilderObject(name=name, behaviors=list(behaviors)))

        # Per-step accumulators (first-row-wins for metadata_pass; behaviors
        # are the inherit source for continuation rows without a description)
        step_pass: Dict[str, str] = {}
        step_behaviors: List[str] = []

        for row_idx, row in enumerate(rows):
            if not row:
                continue

            first = _ManifestModelInternal._strip_cell(row[0])

            # --- section header ---
            sec_match = _first(section_res, first)
            if sec_match:
                current_section = _group(sec_match, 1)
                current_section_title = _group(sec_match, 2)
                current_step = None
                unheld = False
                continue

            # --- column header row (resolve indices) ---
            if first.lower() in step_id_aliases:
                cols = _ManifestModelInternal._resolve_columns(row, col_map)
                headers = _ManifestModelInternal._header_names(row, cols)
                in_cells = _marked(row)
                continue
            # The step-ID header may not sit in the first column.  Accept a
            # row with a matching cell anywhere — but only if it actually
            # resolves (which also demands the description/assets headers),
            # so a data row containing a bare alias can't hijack the column
            # map.  Not gated on ``cols is None``: layouts repeat the header
            # per section, and an unrecognized repeat would be misread as a
            # continuation row of the previous step.
            if any(
                _ManifestModelInternal._strip_cell(c).lower() in step_id_aliases
                for c in row
            ):
                try:
                    cols = _ManifestModelInternal._resolve_columns(row, col_map)
                    headers = _ManifestModelInternal._header_names(row, cols)
                    in_cells = _marked(row)
                    continue
                except ValueError:
                    pass

            # Skip data rows before we've seen a header
            if cols is None:
                continue

            # --- step row ---
            # Read the step ID from the resolved step column (column 0 for
            # the default layouts) rather than assuming it sits first.
            step_cell = (
                _ManifestModelInternal._strip_cell(row[cols.step_id])
                if len(row) > cols.step_id
                else ""
            )
            step_match = _first(step_res, step_cell) if step_cell else None
            if step_cell and not step_match and len(unread) < 3:
                unread.append(step_cell)
            if step_cell and not step_match and actions and in_cells:
                # In the cell format a filled Step cell is a step's row or a section's: any other
                # is a mistake, and its rows are no part of the step above.
                _unread("step", "sheet row %d: '%s' is no step ID and no section, so it and the rows under it are "
                                "not read: write a step's ID (A02, IN_01, B03A), or a section as SECTION C, a colon, its title"
                        % (row_idx + 1, step_cell))
                current_step, unheld = None, True
                continue
            description = (
                _ManifestModelInternal._strip_cell(row[cols.description])
                if cols.description is not None and len(row) > cols.description
                else ""
            )
            asset = (
                _ManifestModelInternal._strip_cell(row[cols.assets])
                if len(row) > cols.assets
                else ""
            )
            audio = (
                _ManifestModelInternal._strip_cell(row[cols.audio])
                if cols.audio is not None and len(row) > cols.audio
                else ""
            )
            behavior_cell = (
                _ManifestModelInternal._strip_cell(row[cols.behaviors])
                if cols.behaviors is not None and len(row) > cols.behaviors
                else ""
            )
            cells = bool(actions) and in_cells and _ManifestModelInternal._reads_by_cells(row, cols, actions)
            if cells:
                row_behaviors, speaks = [], True  # its cells say what it does; its notes are never read
            else:
                row_behaviors, speaks = _row_behaviors(description, behavior_cell)
            row_cells = {
                key: (_ManifestModelInternal._strip_cell(row[idx]) if len(row) > idx else "")
                for key, idx in cols.row_pass.items()
            }
            row_cells = {
                key: "" if cell.upper() in pass_excludes.get(key, ()) else cell
                for key, cell in row_cells.items()
            }
            row_record = (
                dict({"text": description, "assets": _asset_names(asset)}, **row_cells)
                if description or asset or any(row_cells.values())
                else None
            )
            # A step row is the step alone: its To and Settings are the step's own, and it holds no action.
            own = None
            if cells and step_match and row_record is not None:
                acts = bool(row_record["assets"]) or any(
                    row_record.get(key) for key in ("action", "with", "clip", "members", "label")
                )
                if not acts:
                    own = (row_record.get("to", ""), row_record.get("settings", ""))
                    row_record = None
                    description = ""
            if cells and row_record is not None:
                _ManifestModelInternal._read_cells(row_record, actions, headers, keyed, col_map.shots, col_map.step_looks)
                if step_match:
                    row_record["issues"].append(
                        (headers.get("action") or "Action",
                         "a step's row holds no action: write this one on the row below it")
                    )
                # The step's words are its rows said back, never its notes.
                description = row_record["says"]

            if step_match:
                # The step becomes a shot whose name is its exported clip
                # name: spell it legally here (``B03.5`` -> ``B03_5``) so the
                # store accepts it and every later lookup agrees.
                step_id = StrUtils.to_legal_name(_group(step_match, 1))
                if step_id in seen_ids:
                    first = next(s for s in steps if s.step_id == step_id)
                    first.issues.append(
                        (headers.get("step") or "Step",
                         "sheet row %d is step %s too, so it and the rows under it are not read: give that step "
                         "its own ID" % (row_idx + 1, step_id))
                    )
                    notes.append(f"Duplicate step_id '{step_id}' (sheet row {row_idx + 1}): not read.")
                    current_step, unheld = None, True
                    continue
                seen_ids.add(step_id)
                unheld = False
                step_behaviors = row_behaviors
                # Every metadata_pass column the doc has, a blank cell as "":
                # the doc speaks for the key, so a cleared cell clears it.
                step_pass = {}
                for key, idx in cols.metadata_pass.items():
                    step_pass[key] = (
                        _ManifestModelInternal._strip_cell(row[idx])
                        if len(row) > idx
                        else ""
                    )
                current_step = BuilderStep(
                    step_id=step_id,
                    section=current_section,
                    section_title=current_section_title,
                    description=description,
                    audio=audio,
                    lines=[description] if description else [],
                )
                current_step._pass_through = dict(step_pass)
                current_step.headers = headers
                current_step.issues.extend(stray)
                del stray[:]
                if own is not None:
                    current_step.to = own[0]
                    current_step.settings, errors = ManifestActions.read_settings(own[1])
                    own_cells[step_id] = (errors, dict(headers))
                clip_header[step_id] = "clip" in cols.row_pass
                if actions and in_cells:
                    celled.add(step_id)
                    current_step.celled = True
                if row_record:
                    current_step.rows.append(row_record)
                steps.append(current_step)

                if cells:
                    assets_spoken.add(step_id)
                    cell_rows.setdefault(step_id, [])
                    if row_record:
                        cell_rows[step_id].append(row_record)
                        for name, behaviors in row_record["shot"]:
                            _add_object(current_step, name, behaviors)
                    step_behaviors = []
                    continue
                if asset:
                    assets_spoken.add(step_id)
                for name in _asset_names(asset):
                    _add_object(current_step, name, step_behaviors)
                if read_prose:
                    _note_subjects(step_id, description)
                continue

            # --- continuation row (belongs to previous step) ---
            if current_step is not None:
                if row_record:
                    current_step.rows.append(row_record)
                # Merge continuation description into the parent step
                if description:
                    joined = "; " if cells and current_step.description else " "
                    current_step.description = (current_step.description + joined + description).strip()
                    current_step.lines.append(description)
                    if read_prose and not cells:
                        _note_subjects(current_step.step_id, description)
                if cells:
                    assets_spoken.add(current_step.step_id)
                    if row_record:
                        cell_rows.setdefault(current_step.step_id, []).append(row_record)
                        for name, behaviors in row_record["shot"]:
                            _add_object(current_step, name, behaviors)
                    continue

                if asset:
                    assets_spoken.add(current_step.step_id)
                names = _asset_names(asset)
                if names:
                    # A row carrying its own description speaks for itself --
                    # including when that description implies NO behavior, which
                    # is how a static prop beside a fading one is expressed.
                    # Gating on the description's presence (not on whether it
                    # detected anything) is what makes that sayable.  A row
                    # without one inherits the parent STEP's behaviors, not
                    # those of ``current_step.description``, which by this point
                    # has absorbed earlier continuation rows' text too.
                    behaviors = row_behaviors if speaks else step_behaviors
                    for name in names:
                        _add_object(current_step, name, behaviors)
            elif cells and row_record is not None and not unheld:
                stray.append(
                    (headers.get("action") or "Action",
                     "sheet row %d: the row stands above its section's first step, so no step reads it: move it "
                     "under its step" % (row_idx + 1))
                )

        if stray and steps:  # rows after the last step's section header
            steps[-1].issues.extend(stray)
        elif stray:
            notes.extend("'%s' (%s): %s" % (source, header, text) for header, text in stray)
        ids = [step.step_id for step in steps]
        for step in steps:
            if step.step_id in own_cells:
                errors, heads = own_cells[step.step_id]
                step.issues = ManifestActions.check_step_row(
                    step.to, step.settings, errors, col_map.step_settings, heads, ids, StrUtils.to_legal_name
                ) + step.issues
        if cell_rows:
            celled_rows = [row for step in steps for row in cell_rows.get(step.step_id) or ()]
            lacking = ManifestActions.missing_columns(celled_rows, actions)
            if lacking:  # the sheet's own, said once: at the first step whose rows are read by cells
                next(s for s in steps if cell_rows.get(s.step_id)).issues.extend(lacking)
            for step in steps:
                rows_read = cell_rows.get(step.step_id) or []
                found = ManifestActions.check_step(
                    rows_read, actions, headers, step.step_id, ids, StrUtils.to_legal_name
                )
                for n, issue, warning in found:
                    rows_read[n]["warnings" if warning else "issues"].append(issue)
            wrong = sum(len(r["issues"]) for rows_read in cell_rows.values() for r in rows_read)
            wrong += sum(len(step.issues) for step in steps)
            if wrong:
                notes.append(
                    f"'{source}': {wrong} cell{'s' if wrong != 1 else ''} to fix "
                    "(each row's issues name its cell and the fix)."
                )

        if any(spec.get("take") for spec in actions.values()):
            # A template with a word that plays a take: each step names the takes its rows
            # play, in row order (none for a step that only speaks). A sheet of sentences
            # names none this way: its shot is its own take.
            for step in steps:
                if step.step_id not in celled:
                    continue
                step.takes = list(dict.fromkeys(
                    StrUtils.to_legal_name(row["clip"]) for row in cell_rows.get(step.step_id, ()) if row.get("take")
                ))

        if col_map.shots == "clip":
            # Only a step one of whose rows names a clip plays animation, read
            # under its own section's header; a sheet with no clip column
            # names none (the option says so).
            has_clip = any(clip_header.values())
            for step in steps:
                step.animated = clip_header.get(step.step_id, False) and any(
                    row.get("clip") for row in step.rows
                )
            if steps and not has_clip:
                notes.append(
                    f"'{source}' has no animation clip column (one of "
                    f"{list(col_map.row_pass.get('clip', ()))}): the template "
                    "gives a shot only to a step whose rows name a clip, so no "
                    "step gets one."
                )
        # A missing header row used to fail silently: every data row was
        # skipped and the caller saw 0 steps with no explanation.
        if cols is None and rows:
            notes.append(
                f"No header row found in '{source}' — 0 steps parsed. Expected a "
                f"column named one of {sorted(step_id_aliases)}."
            )
        elif cols is not None and not steps and unread:
            # A header but no step: the sheet writes its IDs another way.
            notes.append(
                f"No step read from '{source}': no step-ID cell matches "
                f"{list(col_map.step_pattern)} (e.g. "
                f"{', '.join(repr(c) for c in unread)}) -- the template reads "
                "another layout of step IDs."
            )
        for step in steps:
            if step.step_id not in assets_spoken:
                step.objects.extend(prose_objects.get(step.step_id, {}).values())
        return steps

    @staticmethod
    def validate_actions(value: Any) -> List[str]:
        """What is wrong with a template's ``actions`` (``ManifestActions.validate_actions``)."""
        from pythontk.core_utils.engines.shots.manifest.manifest_actions import ManifestActions

        return ManifestActions.validate_actions(value)

    #: Values ``ColumnMap.behavior_source`` accepts.
    BEHAVIOR_SOURCES = ("description", "column", "column_else_description")
    #: Values ``ColumnMap.object_source`` accepts.
    OBJECT_SOURCES = ("column", "column_else_description")
    #: Values ``ColumnMap.shots`` accepts.
    SHOT_RULES = ("every", "clip")
    #: Values ``ColumnMap.step_looks`` accepts.
    STEP_LOOKS = ("key", "leave")
    #: ``ColumnMap.object_case`` / ``object_name_rule`` value that leaves a
    #: prose name as the doc wrote it.
    KEEP = "keep"

    @staticmethod
    def prose_name(subject: str, case: str, rule: str) -> str:
        """The object name a doc's prose *subject* stands for: *case*
        (``StrUtils.set_case``), then the legal-name *rule*
        (``StrUtils.apply_name_rule``); :attr:`KEEP` skips either step."""
        keep = _ManifestModelInternal.KEEP
        name = subject if case == keep else StrUtils.set_case(subject, case)
        return name if rule == keep else StrUtils.apply_name_rule(name, rule)

    @staticmethod
    def validate_patterns(value: Any) -> List[str]:
        """Validate a row-grammar field: a list of regexes that compile."""
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            return ["expected a list of regular expressions"]
        errs: List[str] = []
        for pattern in value:
            try:
                re.compile(pattern)
            except (re.error, TypeError) as exc:
                errs.append(f"{pattern!r} is not a valid regular expression: {exc}")
        return errs

    @staticmethod
    def _first_match(regexes: List["re.Pattern"], text: str) -> Optional["re.Match"]:
        """The first of *regexes* to match *text*, or ``None``."""
        for rx in regexes:
            m = rx.match(text)
            if m:
                return m
        return None

    @staticmethod
    def _group(match: "re.Match", n: int) -> str:
        """Capture group *n* of *match* ("" when absent); group 1 falls back
        to the whole match for a pattern without groups."""
        if match.re.groups >= n:
            return (match.group(n) or "").strip()
        return match.group(0).strip() if n == 1 else ""

    @staticmethod
    def _strip_cell(cell: str) -> str:
        """Strip whitespace from a CSV cell."""
        return (cell or "").strip()

    @staticmethod
    def _decode_csv_bytes(raw: bytes) -> List[List[str]]:
        """Decode raw CSV bytes into rows, tolerating non-UTF-8 encodings.

        Production manifests frequently come out of Excel as cp1252; a
        UTF-8-only read used to fail the entire load on the first accented
        character.  The UTF-8 BOM is stripped from the raw bytes up front —
        otherwise a BOM'd file that falls back to cp1252 (or the lossy
        read) leaks it into the first header cell and the header never
        resolves.  Tries UTF-8 first, then cp1252, then a lossy UTF-8 read
        as a last resort so a stray byte can't kill the manifest.
        """
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        for encoding in ("utf-8", "cp1252"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("utf-8", errors="replace")
        return list(csv.reader(io.StringIO(text, newline="")))

    @staticmethod
    def _read_csv_rows(source: str) -> List[List[str]]:
        """Read all CSV rows from a local path or an ``http(s)`` URL.

        A URL's bytes come from :class:`~pythontk.RemoteFile` (a Google
        Sheets share link is rewritten to its CSV export); a path is read
        from disk.  Decoding is shared in :meth:`_decode_csv_bytes`, so the
        BOM/encoding tolerance is identical for both.
        """
        if RemoteFile.is_url(source):
            raw = RemoteFile.read_bytes(source)
        else:
            with open(source, "rb") as fh:
                raw = fh.read()
        return _ManifestModelInternal._decode_csv_bytes(raw)


class ManifestModel(_ManifestModelInternal):
    """The shot manifest's data model and CSV parser.

    Parses a structured production CSV into the step/object graph the rest
    of the shot engine plans against, applying the column-mapping schema
    and detecting each step's behaviors. DCC-agnostic.
    """

    #: Below this much free space, :meth:`describe_read_failure` states the
    #: figure beside the causes: low disk is then a plausible culprit and the
    #: number helps spot a (nearly) full volume.  A fact, never asserted as THE
    #: cause; above it the figure is noise.
    LOW_DISK_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB

    @classmethod
    def describe_read_failure(cls, path: str, exc: OSError) -> str:
        """Explain a CSV that exists but cannot be read, without over-committing.

        ``isfile()`` passed but the bytes would not read.  The tempting
        diagnosis -- "it's a cloud file that hasn't downloaded, make it
        available offline" -- is usually wrong: cloud placeholders hydrate on
        demand fine, and a genuine failure is far more often a full volume or a
        stopped sync client.  So the likely causes are *enumerated* (the
        cloud-client clause only for a cloud-managed file) rather than one
        asserted, the raw error is always included, and the actual free space
        is appended when it is below :attr:`LOW_DISK_BYTES`.
        """
        import os

        from pythontk.file_utils._file_utils import FileUtils

        causes = ["the disk may be full"]
        if FileUtils.is_cloud_placeholder(path):
            causes.append("your cloud sync client may not be running")
        causes.append("the file may be locked by another program")
        causes.append("the drive may be disconnected")
        msg = (
            f"Can't read CSV: {exc}. The file exists but its contents can't be "
            f"read - {', '.join(causes)}. Check, then reload."
        )
        free = FileUtils.free_space(path)
        if free is not None and free < cls.LOW_DISK_BYTES:
            drive = os.path.splitdrive(os.path.abspath(path))[0] or "the drive"
            msg += f" ({drive} has only {free // (1024 * 1024)} MB free.)"
        return msg

    @staticmethod
    def detect_behaviors(text: str) -> List[str]:
        """Return behavior names inferred from descriptive *text*.

        Each pattern is tested independently so text mentioning both
        "fades in" and "fades out" yields ``["fade_in", "fade_out"]``.
        """
        from pythontk.core_utils.engines.shots.manifest.behaviors import Behaviors

        return Behaviors.detect(text)

    @staticmethod
    def parse_csv(
        filepath: str,
        columns: Optional[ColumnMap] = None,
        post_process: Optional[Callable[[BuilderStep], None]] = None,
    ) -> List[BuilderStep]:
        """Parse a structured CSV into a list of :class:`BuilderStep`.

        Parameters:
            filepath: Path to the CSV file, or an ``http(s)`` URL (a Google
                Sheets share link is accepted; see :class:`~pythontk.RemoteFile`).
            columns: Optional header-name mapping.  Defaults cover the
                common sequence-document layouts.
            post_process: Optional callable invoked on each step after
                assembly.  Use to compute derived fields (e.g.
                audio objects) from the parsed data.

        Returns:
            Ordered list of steps, each carrying its objects and detected
            behaviors.
        """
        col_map = columns or ColumnMap()
        rows = _ManifestModelInternal._read_csv_rows(filepath)
        steps = _ManifestModelInternal._parse_rows(rows, col_map, filepath)

        # Apply exclude list
        if col_map.exclude_steps:
            excluded = {
                StrUtils.to_legal_name(s).upper() for s in col_map.exclude_steps
            }
            steps = [s for s in steps if s.step_id.upper() not in excluded]

        # Apply post-processing hook (e.g. derive audio objects from step fields)
        if post_process:
            for step in steps:
                post_process(step)
        return steps


@dataclass
class BuilderObject:
    """One asset within a step."""

    name: str
    behaviors: List[str] = field(default_factory=list)  # e.g. ["fade_in", "fade_out"]
    kind: str = "scene"  # "scene" | "audio"
    source_path: str = ""  # file path for audio creation (transient)
    # Where the name came from: "column" (the sheet's asset cell),
    # "description" (named by a behavior phrase in the step's prose),
    # "shot" (auto-filled from the paired shot) or "member" (a member of the
    # step's shot the step does not list -- shown by the panel, never edited).
    origin: str = "column"
    #: The sheet's own value of each field an edit changed (``name``,
    #: ``behaviors``; ``step`` -- the step the sheet lists it under, when it
    #: was moved) -- filled by ``ManifestEdits.apply``, empty when unedited.
    sheet: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuilderStep:
    """One step (= one future sequencer shot).

    ``step_id`` is the step's identity -- what its shot is bound to
    (``metadata["step"]``), what ranges and assessments are keyed by -- and
    never changes; ``name`` renames the step's shot without moving that
    binding (:attr:`shot_name`).
    """

    step_id: str  # e.g. "A04"
    section: str  # e.g. "A"
    section_title: str  # e.g. "AILERON RIGGING"
    description: str  # merged step-contents text (used for behavior detection)
    objects: List[BuilderObject] = field(default_factory=list)
    audio: str = ""  # narration/voice-over text from CSV
    # Extra columns copied verbatim into shot metadata (first-row-wins); filled
    # by :func:`parse_csv` from a ColumnMap's ``metadata_pass``.
    _pass_through: Dict[str, str] = field(default_factory=dict)
    #: The shot's name when it is not the step ID (a rename); empty = the ID.
    #: A build writes it onto the paired shot, which stays bound to the ID.
    name: str = ""
    #: The sheet's own value of each field an edit changed (``name``,
    #: ``description``) -- filled by ``ManifestEdits.apply``.
    sheet: Dict[str, Any] = field(default_factory=dict)
    #: Each row's step-contents cell as the sheet writes it, in order (the
    #: step row's own first, then each continuation row's; a blank cell left
    #: out): what :attr:`description` joins, kept apart for a reader that acts
    #: on one row at a time. The sheet's text: an edit changes only
    #: :attr:`description`.
    lines: List[str] = field(default_factory=list)
    #: Each row of the step as the sheet writes it, in order (a row with nothing
    #: in it left out): ``{"text", "assets": [names], <row_pass key>: cell}``, so
    #: a reader binds a row's words to that row's own names.
    rows: List[Dict[str, Any]] = field(default_factory=list)
    #: Where the step goes by itself once done, as its row's To cell writes it (a sheet in
    #: cells): empty, the next step; a step's ID, ``next``, ``restart`` or ``exit``.
    to: str = ""
    #: The step's own settings, from its row's Settings cell (a sheet in cells).
    settings: Dict[str, str] = field(default_factory=dict)
    #: What is wrong with the step row's own cells: ``[(header, what and its fix)]``.
    issues: List[Tuple[str, str]] = field(default_factory=list)
    #: The step stands under a header written in cells, read by a template that declares its
    #: actions: its rows say all it holds, so one that names no object says so on purpose.
    celled: bool = False
    #: ``{column key: the sheet's own header}`` of the header row above the step: what a
    #: reader names a cell by (``target``, ``step``, ``audio`` and each column it carries).
    headers: Dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    #: Whether the step plays animation, so a build gives it a shot.  Every
    #: step does unless its template's ``shots`` is ``clip`` and none of its
    #: rows names an animation clip (``ColumnMap.shots``); a Local Edit marks
    #: one either way (``ManifestEdits``, field ``animated``).  A shot already
    #: bound to the step stays its shot.
    animated: bool = True
    #: The takes the step's rows play, by name, in row order: each a span inside the
    #: step's shot and a clip of its own in an export (``ShotStore.TAKES``). Empty: the
    #: step only speaks. ``None``: the sheet names no takes (a sheet of sentences, a
    #: template with no word that plays one), and the step's shot is its own take.
    takes: Optional[List[str]] = None

    def sheet_issues(self) -> List[Tuple[str, str]]:
        """What the sheet's own check found in the step, ``[(status, text)]`` in sheet order:
        ``sheet`` an error, ``sheet_warning`` a row read anyway. A text names its cell as
        every reader says it: ``row 2 (Clip): <reason>: <fix>`` (the row counted in the
        step), or ``(To): ...`` for the step's own row."""
        found = [("sheet", "(%s): %s" % (header, text)) for header, text in self.issues]
        for n, row in enumerate(self.rows, 1):
            found += [("sheet", "row %d (%s): %s" % (n, h, t)) for h, t in row.get("issues") or ()]
            found += [("sheet_warning", "row %d (%s): %s" % (n, h, t)) for h, t in row.get("warnings") or ()]
        return found

    @property
    def pass_through(self) -> Dict[str, str]:
        """The step's pass-through columns (its mapping's ``metadata_pass``):
        ``{key: cell}`` as the step's first row writes them, a blank cell as
        ``""``; a copy, for a reader outside the manifest."""
        return dict(self._pass_through)

    @property
    def shot_name(self) -> str:
        """The name this step's shot carries: :attr:`name`, else the step ID."""
        return self.name or self.step_id

    @property
    def display_text(self) -> str:
        """Text shown in the tree Description column."""
        return self.description

    @classmethod
    def from_detection(
        cls,
        candidates: List[Dict],
    ) -> Tuple[List["BuilderStep"], Dict[str, Tuple[float, float]]]:
        """Convert detection candidates to BuilderSteps + pre-filled ranges.

        Parameters:
            candidates: List of dicts with keys: name, start, end, objects.

        Returns:
            ``(steps, ranges)`` — steps list and dict mapping
            ``step_id`` → ``(start, end)``.
        """
        steps: List["BuilderStep"] = []
        ranges: Dict[str, Tuple[float, float]] = {}
        for i, cand in enumerate(candidates):
            step_id = cand.get("name")
            start = cand.get("start")
            end = cand.get("end")
            if step_id is None or start is None or end is None:
                log.warning(
                    "Skipping detection candidate %d: missing required "
                    "key(s) (name=%r, start=%r, end=%r)",
                    i,
                    step_id,
                    start,
                    end,
                )
                continue
            obj_names = cand.get("objects", [])
            objects = [BuilderObject(name=n) for n in obj_names]
            step = cls(
                step_id=step_id,
                section="",
                section_title="",
                description="",
                objects=objects,
            )
            steps.append(step)
            ranges[step_id] = (start, end)
        return steps, ranges

    @classmethod
    def from_shots(
        cls,
        shots: List[Any],
    ) -> Tuple[List["BuilderStep"], Dict[str, Tuple[float, float]]]:
        """Convert existing store shots back into BuilderSteps + their ranges.

        The inverse of ``ShotManifest._step_metadata``: a shot the manifest
        built carries its CSV objects, behaviors, section and voice text in
        ``metadata``, and gets that step back -- so members discovered after
        the build stay *additional*, exactly as against the CSV.  A shot built
        any other way (sequencer, detection, by hand) has no ``csv_objects``;
        its own members become the step's objects.

        Parameters:
            shots: ``ShotBlock``-like records (``name``, ``start``, ``end``,
                ``objects``, ``metadata``, ``description``), in step order.

        Returns:
            ``(steps, ranges)`` — steps list and dict mapping
            ``step_id`` → ``(start, end)``.
        """
        steps: List["BuilderStep"] = []
        ranges: Dict[str, Tuple[float, float]] = {}
        for shot in shots:
            meta = shot.metadata or {}
            behaviors: Dict[Tuple[str, str], List[str]] = {}
            sources: Dict[Tuple[str, str], str] = {}
            for entry in meta.get("behaviors", []):
                if not isinstance(entry, dict) or not entry.get("behavior"):
                    continue
                key = (entry.get("name", ""), entry.get("kind", "scene"))
                behaviors.setdefault(key, []).append(entry["behavior"])
                if entry.get("source_path"):
                    sources[key] = entry["source_path"]
            raw = meta["csv_objects"] if "csv_objects" in meta else shot.objects
            objects = []
            for entry in raw:
                if isinstance(entry, dict):
                    key = (entry.get("name", ""), entry.get("kind", "scene"))
                else:
                    key = (entry, "scene")
                if any((o.name, o.kind) == key for o in objects):
                    # One object, as the parse makes it: a build before it
                    # merged two rows listed the object once per row.
                    continue
                objects.append(
                    BuilderObject(
                        name=key[0],
                        behaviors=behaviors.get(key, []),
                        kind=key[1],
                        source_path=sources.get(key, ""),
                    )
                )
            # A shot bound to a doc step IS that step (``ShotManifest.pair``),
            # whatever it has been renamed since.
            step_id = meta.get("step") or shot.name
            steps.append(
                cls(
                    step_id=step_id,
                    section=meta.get("section", ""),
                    section_title=meta.get("section_title", ""),
                    description=shot.description,
                    objects=objects,
                    audio=meta.get("voice_text", ""),
                    name=shot.name if shot.name != step_id else "",
                )
            )
            ranges[step_id] = (shot.start, shot.end)
        return steps, ranges


# ---------------------------------------------------------------------------
# Build plan (compute-then-commit)
# ---------------------------------------------------------------------------

Action = Literal[
    "created", "patched", "skipped", "locked", "kept", "removed", "refused"
]


@dataclass
class PlannedShot:
    """Immutable build instruction computed before any store mutation.

    Produced by the manifest planner and consumed by the DCC commit step.
    Fields capture the *final* position each shot will occupy, so downstream
    consumers (behavior keying, ripple bookkeeping) never read stale ranges.
    """

    step: BuilderStep
    action: Action
    start: float = 0.0
    end: float = 0.0
    objects: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    existing_shot_id: Optional[int] = None
    ripple_delta: float = 0.0  # shift applied to later shots


@dataclass
class PlannedInsert:
    """Where a step with no shot goes on a built timeline
    (``ShotManifest.plan_insertions``).

    *start* / *end* are its bounds once every insert of the plan is in.  It
    moves the shots after it *push* frames later; *pivot* is the current
    start of the first existing shot -- or typed range -- it moves (``None``:
    it moves none): a range typed in the timeline's current frames from there
    on moves with them (``ShotManifest.ride_inserts``).

    *follows* is what comes before it on the timeline, nearest first, back to
    the existing shot it goes after: ``("insert", step_id)`` -- another
    insert of the plan --, ``("typed", step_id)`` -- a step whose shot a
    Build makes at its typed range -- and ``("shot", shot_id)`` last, absent
    when it goes before every existing shot.  ``insert_planned`` reads it to
    put the shot where the plan does with any of the plan's inserts made.
    """

    step: BuilderStep
    start: float
    end: float
    duration: float
    follows: Tuple[Tuple[str, Any], ...] = ()
    pivot: Optional[float] = None
    push: float = 0.0

    @property
    def after_shot_id(self) -> Optional[int]:
        """The existing shot it goes after (``None``: before every one)."""
        kind, key = self.follows[-1] if self.follows else (None, None)
        return key if kind == "shot" else None

    @property
    def after_step_id(self) -> Optional[str]:
        """The step whose new shot it directly follows -- another insert, or
        a step with a typed range -- or ``None`` when it directly follows
        :attr:`after_shot_id` (or the head)."""
        kind, key = self.follows[0] if self.follows else (None, None)
        return key if kind in ("insert", "typed") else None


FitMode = Literal["extend_only", "fit_contents"]

# Defaults live on ShotStore (single source of truth for shot-construction
# policy).  Re-exported here so the pure-python API retains keyword defaults.
DEFAULT_INITIAL_SHOT_LENGTH: float = ShotStore.DEFAULT_INITIAL_SHOT_LENGTH
DEFAULT_FIT_MODE: FitMode = ShotStore.DEFAULT_FIT_MODE  # type: ignore[assignment]

# Placeholder length for audio steps whose source is not yet resolvable
# (track not loaded, file missing).  Kept small so the shot visibly grows to
# clip length once the source materialises — matches the default fallback of
# the engine's duration resolution so both code paths agree.
AUDIO_PLACEHOLDER_DURATION: float = 30.0


# ---------------------------------------------------------------------------
# Assessment data structures
# ---------------------------------------------------------------------------


@dataclass
class ObjectStatus:
    """Assessment result for one object within a step."""

    name: str
    exists: bool
    # "valid" | "missing_object" | "ambiguous_object" | "not_in_shot" |
    # "unknown_behavior" | "behavior_conflict" | "missing_behavior" |
    # "stale_behavior" | "user_animated" -- colours in ``SHOT_PALETTE``.
    status: str
    behaviors: List[str] = field(
        default_factory=list
    )  # expected behaviors (empty = user-animated)
    broken_behaviors: List[str] = field(
        default_factory=list
    )  # subset of *behaviors* that failed verification, or name no template
    key_range: Optional[Tuple[float, float]] = None  # actual keyframe extent
    #: Verified behaviors whose keys an older effect recipe made -- Build
    #: re-keys them (``ShotManifest.is_stale``).
    stale_behaviors: List[str] = field(default_factory=list)
    #: Of *behaviors*, those no behavior template names (in
    #: *broken_behaviors* too): never verified.
    unknown_behaviors: List[str] = field(default_factory=list)
    #: Of *behaviors*, those whose channels hold keys the animator owns
    #: (``ShotManifest.unowned_keys``): Build leaves them, unsatisfied (in
    #: *broken_behaviors*) or stale (in neither list) alike.
    conflict_behaviors: List[str] = field(default_factory=list)


@dataclass
class StepStatus:
    """Assessment result for one step."""

    step_id: str
    built: bool  # shot exists in sequencer
    objects: List[ObjectStatus] = field(default_factory=list)
    additional_objects: List[str] = field(default_factory=list)  # in shot but not CSV
    shrinkable_frames: float = 0.0  # frames of unused range at step tail
    locked: bool = False  # shot is user-finalized; skip automated flags
    #: ``[obj, behavior]`` the shot holds keys for that the doc no longer
    #: lists -- the manifest's own output, which Build removes.
    dropped_behaviors: List[List[str]] = field(default_factory=list)
    #: The shot was edited by hand since its last build, and Build keeps it
    #: as it is (``ShotManifest.edited_by_hand``; off when the build was
    #: asked to rebuild edited shots).
    edited: bool = False
    #: What a Build rewrites on the shot that no object status shows, in
    #: :attr:`UPDATES` order: ``"name"`` (the step was renamed),
    #: ``"description"``, ``"objects"`` (one the last build listed, the step
    #: no longer does), ``"takes"`` (the takes its rows play, where the shot
    #: holds others) -- a Local Edit's, or the doc's.
    updates: List[str] = field(default_factory=list)
    #: The step plays animation (``BuilderStep.animated``): off, a step with
    #: no shot needs none, and Build makes none.
    animated: bool = True
    #: What the sheet's own check found in the step (``BuilderStep.sheet_issues``):
    #: ``[(status, text)]``, ``sheet`` an error no Build resolves, ``sheet_warning``
    #: a row read anyway.
    sheet: List[Tuple[str, str]] = field(default_factory=list)
    #: The step is written in cells (``BuilderStep.celled``): one that names no object
    #: (a question, a line of narration) is in order, never ``no_objects``.
    celled: bool = False

    #: What a Build does for each of :attr:`updates`, as a panel says it.
    UPDATES = {
        "name": "renames its shot",
        "description": "rewrites its description",
        "objects": "takes out the objects the step no longer lists",
        "takes": "records the takes its rows play",
    }

    @property
    def status(self) -> str:
        """Worst-of-children rollup.

        Priority: ``"locked"`` (user-finalized) > ``"edited"`` (kept as the
        hand left it) > ``"sheet"`` (the sheet's check found an error in its
        rows: a person's to fix, whatever a Build does) > ``"missing_shot"`` >
        the object statuses in
        :attr:`ROLLUP` order > ``"no_objects"`` (a sheet of sentences lists
        none -- informational, never a silent "valid"; a step written in
        cells says all it holds, so none is in order) > ``"valid"``.  An edited
        step's objects keep their own statuses.  A step with no shot that
        plays no animation is ``"no_shot"`` unless an object of it is wrong.
        """
        if self.locked:
            return "locked"
        if self.edited:
            return "edited"
        if self.sheet_errors:
            return "sheet"
        if not self.built and self.animated:
            return "missing_shot"
        found = {o.status for o in self.objects}
        for status in self.ROLLUP:
            if status in found:
                return status
        if not self.built:
            return "no_shot"
        return "valid" if self.objects or self.celled else "no_objects"

    @property
    def sheet_errors(self) -> int:
        """How many errors the sheet's check found in the step (:attr:`sheet`)."""
        return sum(1 for status, _text in self.sheet if status == "sheet")

    #: What each status means and what to do about it -- the one wording every
    #: panel shows (step rows, object rows, "not in doc" shot rows).
    HELP = {
        "sheet": "The sheet's own check found a problem in this step's rows -- its fix is the sheet's; no Build resolves it.",
        "sheet_warning": "The sheet's own check has a warning for this step's rows -- read anyway.",
        "missing_shot": "No shot yet -- Build creates it.",
        "missing_object": "Not found in the scene -- an artist's to add; Build never creates objects.",
        "ambiguous_object": "Several scene objects answer to this name -- rename one, or write its path in the doc.",
        "not_in_shot": "In the scene but not a member of this step's shot -- Build adds it.",
        "behavior_conflict": "Animator keys sit on this behavior's channels -- Build leaves them; resolve by hand.",
        "unknown_behavior": "No behavior template has this name -- check the doc, or add a template.",
        "missing_behavior": "Behavior keys not found in the shot's range -- Build applies them.",
        "stale_behavior": "Keyed under an older effect recipe (Render Effects / Audio Clips) -- Build re-keys it.",
        "no_objects": "The doc lists no objects for this step.",
        "not_in_doc": "No doc step pairs with this shot -- Build never removes it; right-click to remove it yourself.",
        "locked": "Locked: user-finalized, never modified by Build.",
        "edited": "Edited by hand since its last Build -- Build leaves it as it is. Right-click > Rebuild from Sheet takes the doc back for this step; Edits > Rebuild Hand-Edited Shots (header menu) for every step.",
        "dropped_behavior": "Keys of behaviors the doc dropped -- Build removes them.",
        "shot_update": "The step changed since its shot was built -- Build updates the shot.",
        "no_shot": "No shot: the step plays no animation (no row names an animation clip) -- Build makes none. Right-click > Mark as a Shot to give it one.",
    }

    #: Each status in three words at most: what a row says without a hover,
    #: beside the sentence :attr:`HELP` gives it.
    LABELS = {
        "missing_shot": "not built",
        "missing_object": "not in scene",
        "ambiguous_object": "ambiguous name",
        "not_in_shot": "not in shot",
        "behavior_conflict": "keys conflict",
        "unknown_behavior": "unknown behavior",
        "missing_behavior": "keys missing",
        "stale_behavior": "old recipe",
        "no_objects": "no objects",
        "not_in_doc": "not in sheet",
        "locked": "locked",
        "edited": "edited by hand",
        "user_animated": "hand keyed",
        "additional": "not listed",
        "scene_discovered": "found in scene",
        "collision": "overlaps",
        "dropped_behavior": "keys to remove",
        "shot_update": "to update",
        "no_shot": "no shot",
        "sheet": "sheet issue",
        "sheet_warning": "sheet warning",
    }

    #: Object statuses no Build resolves: the step waits on a person
    #: (:meth:`plan`'s ``needs_you``).
    NEEDS_YOU = (
        "missing_object",
        "ambiguous_object",
        "behavior_conflict",
        "unknown_behavior",
    )

    #: Object statuses that name the step, worst first: by the severity a
    #: panel shows each at (``SHOT_STATUS_LEVELS``: errors, warnings, then
    #: notes), so the status a step is named by is the one its row's icon
    #: shows.
    ROLLUP = (
        "missing_object",
        "behavior_conflict",
        "ambiguous_object",
        "not_in_shot",
        "unknown_behavior",
        "missing_behavior",
        "stale_behavior",
    )

    #: Object statuses a Build fixes -- with an unbuilt step and the keys of
    #: dropped behaviors, what keeps Build worth pressing once every step is
    #: built (:attr:`needs_build`).
    BUILD_FIXES = ("not_in_shot", "missing_behavior", "stale_behavior")

    @property
    def needs_build(self) -> bool:
        """Whether a Build would change this step: its shot is not built, an
        object is one a Build fixes (:attr:`BUILD_FIXES`), the shot holds
        keys of behaviors the doc dropped, or it has :attr:`updates`.  Never
        for a locked or a kept (edited) step: Build leaves both as they are."""
        if self.locked or self.edited:
            return False
        if not self.built:
            return self.animated
        if self.dropped_behaviors or self.updates:
            return True
        return any(o.status in self.BUILD_FIXES for o in self.objects)

    @property
    def quiet(self) -> bool:
        """Whether the assessment has nothing to report of the step: in order
        (``valid``, or ``no_shot``: it plays no animation and needs none),
        nothing for a Build to do, every object ``valid`` and nothing in its
        shot it does not list -- what a view of the steps that need something
        leaves out.  An unused tail (:attr:`shrinkable_frames`) is no issue."""
        return (
            self.status in ("valid", "no_shot")
            and not self.needs_build
            and not self.additional_objects
            and not self.sheet
            and all(o.status == "valid" for o in self.objects)
        )

    #: Each state :meth:`behavior_states` gives a behavior of an object row:
    #: the status whose colour and severity a panel shows it at (a key of
    #: ``SHOT_PALETTE``; ``None``: no status's, a Local Edit's mark or none)
    #: and the state in a few words.  ``""``: not assessed.
    BEHAVIOR_STATES = {
        "": (None, ""),
        "taken": ("locked", "taken away here (a Local Edit)"),
        "off": ("locked", "off for every row (its mode: Off)"),
        "locked": ("locked", "locked -- Build leaves its step as it is"),
        "object_missing": ("missing_object", "its object is not in the scene"),
        "object_ambiguous": (
            "ambiguous_object",
            "several scene objects answer to its object's name -- not checked",
        ),
        "not_in_shot": (
            "not_in_shot",
            "its object is not in the step's shot -- Build adds it, then this",
        ),
        "unknown": ("unknown_behavior", "no behavior template has this name"),
        "conflict": (
            "behavior_conflict",
            "animator keys sit on its channels -- Build leaves them",
        ),
        "missing": (
            "missing_behavior",
            "keys not found in the shot's range -- Build applies them",
        ),
        "marker_missing": (
            "missing_behavior",
            "no marker where its keys would go -- Build places it",
        ),
        "stale": (
            "stale_behavior",
            "keyed under an older effect recipe -- Build re-keys it",
        ),
        "added": (None, "added here (a Local Edit)"),
        "to_build": ("missing_shot", "to build -- its step has no shot yet"),
        "no_shot": (
            "no_shot",
            "its step plays no animation -- Build makes no shot for it",
        ),
        "built": ("valid", "built"),
        "marked": ("valid", "marked on the timeline"),
    }

    #: An object's own status that says what each of its behaviors is: Assess
    #: checked none of them (:meth:`behavior_states`).
    _OBJECT_BEHAVIOR_STATES = {
        "missing_object": "object_missing",
        "ambiguous_object": "object_ambiguous",
        "not_in_shot": "not_in_shot",
    }

    @staticmethod
    def behavior_states(
        step: Optional["StepStatus"],
        name: str,
        behaviors: Iterable[str],
        sheet: Optional[Iterable[str]] = None,
        off: Iterable[str] = (),
        marked: Iterable[str] = (),
    ) -> List[Tuple[str, str, str]]:
        """Each behavior of the object *name* in *step*'s assessment, as an
        object row shows it: ``(behavior, state, words)``, a
        :attr:`BEHAVIOR_STATES` key and its words -- the object's own
        *behaviors*, then those *sheet* gives it that it no longer has.

        One judgment for every panel: a behavior is built only where its
        step is (an unbuilt step's object assesses ``valid``: its behaviors
        are ``to_build``), and a state an object's own status decides (not in
        the scene, ambiguous, not in the shot) says that.

        Parameters:
            step: The step's assessment, or ``None`` before one.
            name: The object, as the step lists it.
            behaviors: Its behaviors, its Local Edits applied.
            sheet: What the sheet gives it (``None``: *behaviors*); one only
                there was taken away here, one only in *behaviors* added.
            off: The behavior types set to Off for every row.
            marked: The behavior types Build marks on the timeline instead
                of keying.
        """
        behaviors = list(behaviors)
        given = behaviors if sheet is None else list(sheet)
        off, marked = set(off), set(marked)
        # Last wins within a step, as a name -> status map reads it.
        found = {o.name: o for o in step.objects}.get(name) if step else None

        def judge(b: str) -> str:
            if b not in behaviors:
                return "taken"
            if b in off:
                return "off"
            if found is None:  # not assessed (or not this step's object)
                return "added" if b not in given else ""
            if step.locked:
                return "locked"
            if found.status in StepStatus._OBJECT_BEHAVIOR_STATES:
                return StepStatus._OBJECT_BEHAVIOR_STATES[found.status]
            if b in found.unknown_behaviors:
                return "unknown"
            if b in found.conflict_behaviors:
                return "conflict"
            if b in found.broken_behaviors:
                return "marker_missing" if b in marked else "missing"
            if b in found.stale_behaviors:
                return "stale"
            if b not in given:
                return "added"
            if not step.built:
                return "to_build" if step.animated else "no_shot"
            return "marked" if b in marked else "built"

        shown = behaviors + [b for b in dict.fromkeys(given) if b not in behaviors]
        return [
            (b, state, StepStatus.BEHAVIOR_STATES[state][1])
            for b in shown
            for state in (judge(b),)
        ]

    @property
    def label(self) -> str:
        """The step's cause in a few words (:attr:`LABELS`): its status's,
        led by how many of its objects share it ("2 not in scene"); for a
        step with nothing wrong, what a Build still changes on its shot;
        ``""`` when nothing does."""
        status = self.status
        if status == "no_shot":
            return ""  # nothing to do: a panel's frames cells say it (LABELS)
        if status in ("valid", "no_objects"):
            if self.dropped_behaviors:
                return self.LABELS["dropped_behavior"]
            if self.updates:
                return self.LABELS["shot_update"]
            return self.LABELS.get(status, "")
        label = self.LABELS.get(status, "")
        if status == "sheet":
            return f"{self.sheet_errors} {label}s" if self.sheet_errors > 1 else label
        count = sum(1 for o in self.objects if o.status == status)
        return f"{count} {label}" if count > 1 else label

    @staticmethod
    def plan(results: Optional[List["StepStatus"]]) -> Dict[str, int]:
        """What a Build of the assessed *results* does, counted by step,
        before it is pressed: ``create`` (no shot yet), ``update`` (a built
        step a Build changes; with ``create``, the steps :attr:`needs_build`
        names), ``keep`` (edited by hand, left as it is), ``skip`` (locked),
        ``needs_you`` (an object no Build resolves, :attr:`NEEDS_YOU`, or an
        error of the sheet's own check, :attr:`sheet_errors`; such a
        step may be created or updated too), ``in_sync`` (built, nothing to
        do) and ``no_shot`` (plays no animation: no shot to make)."""
        plan = dict.fromkeys(
            ("create", "update", "keep", "skip", "needs_you", "in_sync", "no_shot"), 0
        )
        for r in results or ():
            if r.locked:
                plan["skip"] += 1
                continue
            if r.edited:
                plan["keep"] += 1
                continue
            waiting = bool(r.sheet_errors) or any(o.status in StepStatus.NEEDS_YOU for o in r.objects)
            if not r.built and not r.animated:
                plan["no_shot"] += 1
            elif not r.built:
                plan["create"] += 1
            elif r.needs_build:
                plan["update"] += 1
            elif not waiting:
                plan["in_sync"] += 1
            if waiting:
                plan["needs_you"] += 1
        return plan

    @staticmethod
    def plan_text(plan: Dict[str, int]) -> str:
        """*plan* (:meth:`plan`) as one sentence: what Build does, what it
        leaves alone, what waits on a person."""

        def counted(pairs):
            return [f"{plan[key]} {word}" for key, word in pairs if plan.get(key)]

        does = counted((("create", "to create"), ("update", "to update")))
        keeps = counted(
            (("keep", "kept (edited by hand)"), ("skip", "locked"), ("no_shot", "with no shot"))
        )
        waiting = plan.get("needs_you", 0)
        waits = [f"{waiting} need{'s' if waiting == 1 else ''} you"] if waiting else []
        if not does:
            rest = [f"{plan.get('in_sync', 0)} in sync", *keeps, *waits]
            return "Nothing to build: " + ", ".join(rest) + "."
        return "; ".join(", ".join(part) for part in (does, keeps, waits) if part) + "."

    @property
    def missing_count(self) -> int:
        return sum(1 for o in self.objects if o.status == "missing_object")

    @property
    def total_count(self) -> int:
        return len(self.objects)

    @staticmethod
    def find_object(
        results: List["StepStatus"], name: str, step_id: Optional[str] = None
    ) -> Optional[ObjectStatus]:
        """The first assessed :class:`ObjectStatus` named *name* in *results*.

        *step_id* restricts the search to that step's result: one object can
        appear in several steps with different statuses (a fade verified in
        one, broken in another), so a first-match scan across steps would
        report another step's verdict.
        """
        for result in results or ():
            if step_id is not None and result.step_id != step_id:
                continue
            # Last wins within a step, as a name -> status map reads it.
            found = {o.name: o for o in result.objects}.get(name)
            if found is not None:
                return found
        return None


@dataclass
class ShotPairing:
    """Which shot each step is -- :meth:`ShotManifest.pair`'s answer.

    Attributes:
        shots: ``{step_id: ShotBlock}`` for every paired step.
        how: ``{step_id: "binding" | "name" | "take" | "order"}`` -- how it paired.
        orphans: Shots no step pairs with ("not in the doc"), timeline order.
        timeline: Every shot the pairing read, in timeline order.
        links: ``{shot_id: step_id}`` per orphan that belongs with a step
            without being its shot -- one split off it
            (``ShotStore.step_links``); never a pairing.
    """

    shots: Dict[str, Any] = field(default_factory=dict)
    how: Dict[str, str] = field(default_factory=dict)
    orphans: List[Any] = field(default_factory=list)
    timeline: List[Any] = field(default_factory=list)
    links: Dict[int, str] = field(default_factory=dict)

    def rows(
        self,
        steps: List["BuilderStep"],
        hide_missing: bool = False,
        orphans: bool = True,
        keep: Iterable[str] = (),
    ) -> List[Tuple[Optional["BuilderStep"], Optional[Any]]]:
        """The manifest's rows: ``(step, shot)`` per step in doc order --
        ``shot`` ``None`` for a step with no shot yet -- and ``(None, shot)``
        per orphan, where the timeline has it.

        An orphan follows the row of the shot before it on the timeline
        (after any orphan already there), so a shot split off or added in the
        Shot Sequencer sits beside the shots it sits between; one with no
        paired shot before it leads the row of the paired shot after it, and
        with none at all the orphans close the list.

        Parameters:
            steps: The steps that were paired, in doc order.
            hide_missing: Leave out the steps with no shot ("missing shot")
                -- every shot keeps its row.
            orphans: List the orphans (only a sheet can leave a shot out).
            keep: Step IDs shown whatever *hide_missing* says (a step the
                user just added).
        """
        keep = set(keep)
        rows = [
            (step, self.shots.get(step.step_id))
            for step in steps
            if not hide_missing or step.step_id in self.shots or step.step_id in keep
        ]
        if not orphans or not self.orphans:
            return rows
        orphan_ids = {shot.shot_id for shot in self.orphans}
        after: Dict[Optional[int], List[Any]] = {}  # paired shot -> orphans
        before: Dict[Optional[int], List[Any]] = {}
        anchor: Optional[int] = None
        leading: List[Any] = []
        # No timeline (a pairing made by hand): nothing to place the orphans
        # by, so they close the list rather than vanish.
        for shot in self.timeline or self.orphans:
            if shot.shot_id not in orphan_ids:
                if anchor is None and leading:
                    before[shot.shot_id] = leading
                anchor = shot.shot_id
            elif anchor is None:
                leading.append(shot)
            else:
                after.setdefault(anchor, []).append(shot)
        if anchor is None:  # no shot pairs: the orphans close the list
            return rows + [(None, shot) for shot in leading]
        out: List[Tuple[Optional["BuilderStep"], Optional[Any]]] = []
        for step, shot in rows:
            key = shot.shot_id if shot is not None else None
            out.extend((None, s) for s in before.pop(key, ()))
            out.append((step, shot))
            out.extend((None, s) for s in after.pop(key, ()))
        # Anchored on a shot whose step is not among *steps*: never dropped.
        for rest in (*before.values(), *after.values()):
            out.extend((None, s) for s in rest)
        return out


# ---------------------------------------------------------------------------
# CSV column mapping
# ---------------------------------------------------------------------------


@dataclass
class ColumnMap(SchemaSpec):
    """Maps logical fields to CSV header names (case-insensitive), plus the
    row grammar (which cells start a step or a section).

    Each column field is a tuple of acceptable header aliases. The parser reads
    the header row and resolves names to column indices automatically.

    A :class:`~pythontk.SchemaSpec`, so the ``columns`` block of a mapping
    file is self-validating and self-documenting; serialisable via
    :meth:`to_dict` / :meth:`from_dict` (tuple ⇄ list) so instances round-trip
    through JSON.
    """

    step_id: Tuple[str, ...] = SchemaSpec.spec_field(
        help="Header alias(es) for the step-ID column.",
        example=["Step"],
        default=("Step",),
    )
    description: Tuple[str, ...] = SchemaSpec.spec_field(
        help="Header alias(es) for the step description / contents column.",
        example=["Step Contents", "Contents"],
        default=("Step Contents", "Contents"),
    )
    assets: Tuple[str, ...] = SchemaSpec.spec_field(
        help="Header alias(es) for the assets / object-names column.",
        example=["Asset Names", "Asset"],
        default=("Asset Names", "Asset"),
    )
    audio: Tuple[str, ...] = SchemaSpec.spec_field(
        help="Header alias(es) for the audio / voice-over column (optional).",
        example=["Voice Support", "Voice"],
        default=("Voice Support", "Voice"),
    )
    exclude_steps: Tuple[str, ...] = SchemaSpec.spec_field(
        help="Step IDs to skip entirely (e.g. setup rows).",
        example=["SETUP"],
        default=("SETUP",),
    )
    exclude_values: Dict[str, Tuple[str, ...]] = SchemaSpec.spec_field(
        help='Per-field cell values to treat as empty, e.g. {"assets": ["N/A"]}.',
        example={"assets": ["N/A"]},
        default_factory=lambda: {"assets": ("N/A",)},
    )
    behaviors: Tuple[str, ...] = SchemaSpec.spec_field(
        help="Header alias(es) for a behaviors column (optional): names or phrases per cell.",
        example=["Behaviors"],
        default=(),
    )
    behavior_source: str = SchemaSpec.spec_field(
        help=(
            "Where a row's behaviors come from: `description` (phrases in its "
            "text), `column` (the behaviors column), or "
            "`column_else_description`."
        ),
        choices=_ManifestModelInternal.BEHAVIOR_SOURCES,
        example="column_else_description",
        default="description",
    )
    object_source: str = SchemaSpec.spec_field(
        help=(
            "Where a step's objects come from: `column` (its asset cells) or "
            "`column_else_description` -- a step whose asset cells are empty "
            "lists the objects its prose names by a behavior phrase "
            '("Highlight Red Door" -> `Red_Door`, with `highlight`).'
        ),
        choices=_ManifestModelInternal.OBJECT_SOURCES,
        example="column_else_description",
        default="column",
    )
    object_case: str = SchemaSpec.spec_field(
        help=(
            "Case of the object names read from the description: a "
            "`StrUtils.set_case` case, or `keep` (as written)."
        ),
        choices=(_ManifestModelInternal.KEEP,) + StrUtils.CASES,
        example="lower",
        default=_ManifestModelInternal.KEEP,
    )
    object_name_rule: str = SchemaSpec.spec_field(
        help=(
            "Legal-name rule for the object names read from the description: "
            "a `StrUtils.apply_name_rule` rule (`legal`: every non-alphanumeric "
            "becomes `_`), or `keep` (spaces and all)."
        ),
        choices=(_ManifestModelInternal.KEEP,) + StrUtils.NAME_RULES,
        example="legal",
        default="legal",
    )
    metadata_pass: Dict[str, Tuple[str, ...]] = SchemaSpec.spec_field(
        help="Extra columns copied into shot metadata: {key: [header aliases]}.",
        example={"priority": ["Priority"]},
        default_factory=dict,
    )
    row_pass: Dict[str, Tuple[str, ...]] = SchemaSpec.spec_field(
        help=(
            "Columns read on every row, each row keeping its own cell "
            "(`BuilderStep.rows`): {key: [header aliases]}."
        ),
        example={"clip": ["Animation Clip Names"]},
        default_factory=dict,
    )
    actions: Dict[str, Dict[str, Any]] = SchemaSpec.spec_field(
        help=(
            "The words an Action cell (the `action` column of `row_pass`) may hold, "
            "each declared: `{word: {family, cells, needs, needs_any, distinct, take, when, places, settings, says}}` "
            "(`ManifestActions`): `needs_any` groups cells or settings of which one is filled; `distinct` lists "
            "the cells no two rows of the word share in one step (two buttons of one Label); `places` says "
            "what a word that places something places it for (`stand`, `point`, `view`, `hold`), its object "
            "named by its `where` setting (type `object`). A template that declares them is in the cell format: a "
            "row with an Action is read by its cells (Target, and the `with`, `clip`, "
            "`members`, `label`, `to` and `settings` columns of `row_pass`), never by its "
            "sentence. The empty word declares what a row with no Action is."
        ),
        example={"highlight": {"cells": ["target"], "says": "{target} is highlighted"}},
        default_factory=dict,
        validate=_ManifestModelInternal.validate_actions,
    )
    step_settings: Dict[str, Dict[str, Any]] = SchemaSpec.spec_field(
        help=(
            "The settings a step may have, written in its own row's Settings cell (a sheet in cells): "
            "`{name: {type, choices, default, min, max}}`, as an action's."
        ),
        example={"dwell": {"type": "number", "min": 0}},
        default_factory=dict,
    )
    cell_marks: Tuple[str, ...] = SchemaSpec.spec_field(
        help=(
            "The headers that say a sheet is written in cells: one of them in its header row, "
            "and its rows are read by `actions`; none, and it is a sheet of sentences, its "
            "Action cell the fallback it was. Empty: every sheet of this template is in cells."
        ),
        example=["Target", "With", "Label", "To", "Settings"],
        default=(),
    )
    shots: str = SchemaSpec.spec_field(
        help=(
            "Which steps a build gives a shot: `every` step (a video, where a "
            "step's narration needs its time), or only one of whose rows names "
            "an animation clip in the `clip` column of `row_pass` (`clip`: an "
            "interactive task, where a step that moves nothing is a page, not "
            "a take; a sheet without that column names no clip). Either way a shot "
            "is its step's, and holds the takes its rows play."
        ),
        choices=_ManifestModelInternal.SHOT_RULES,
        example="clip",
        default="every",
    )
    step_looks: str = SchemaSpec.spec_field(
        help=(
            "What a build does with a look (`highlight`, `fade_in`, `fade_out`) whose row names no "
            "take in Clip, in a sheet written in cells: such a look belongs to its step. `key`: it "
            "is keyed into the step's shot (a sheet played straight through: a video, a "
            "click-through). `leave`: it is left to the tool that runs the steps (an interactive "
            "task, whose takes wait on the user: keyed into one, a look meant for the step's start "
            "would play only once its take does), and its Target is a member of the shot with "
            "nothing keyed on it. A look that names a take is keyed into it either way."
        ),
        choices=_ManifestModelInternal.STEP_LOOKS,
        example="leave",
        default="key",
    )
    step_pattern: Tuple[str, ...] = SchemaSpec.spec_field(
        help=(
            "Regexes a step-ID cell must match to start a step (first match "
            "wins); group 1 is the ID, else the whole match.  The ID is spelled "
            "as a legal shot name (non-alphanumerics -> `_`)."
        ),
        example=[r"^([A-Z]+_?\d+(?:\.\d+)?)$"],
        default=(r"^([A-Z]\d+)\.\)", r"^([A-Z]{2,})$"),
        validate=_ManifestModelInternal.validate_patterns,
    )
    section_pattern: Tuple[str, ...] = SchemaSpec.spec_field(
        help=(
            "Regexes a row's first cell must match to start a section; group 1 "
            "is the section ID, group 2 (optional) its title."
        ),
        example=[r"(?i)^SECTION\s+([A-Z0-9]+)\s*:\s*(.*)", "^(INTRO)$"],
        default=(r"(?i)^SECTION\s+([A-Z0-9]+)\s*:\s*(.*)",),
        validate=_ManifestModelInternal.validate_patterns,
    )
    layouts: Tuple[Dict[str, Any], ...] = SchemaSpec.spec_field(
        help=(
            "Alternative layouts tried on the same sheet, each a dict of these "
            "settings over the rest (`{}` is the rest as they are): the sheet "
            "is read with each and the one reading the most steps is kept, the "
            "first on a tie -- for sheets that write their step IDs "
            "differently (`A01.)` or `A01`)."
        ),
        example=[{}, {"step_pattern": [r"^([A-Z]+_?\d+)$"]}],
        default=(),
    )

    def unmapped(self, header: List[str]) -> List[str]:
        """The cells of *header* no field takes, in order (a blank cell aside):
        a column the sheet calls by a name the template does not know."""
        taken = {a.strip().lower() for a in self.step_id + self.description + self.assets + self.audio + self.behaviors}
        for aliases in list(self.metadata_pass.values()) + list(self.row_pass.values()):
            taken |= {a.strip().lower() for a in aliases}
        return [c.strip() for c in header if c.strip() and c.strip().lower() not in taken]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict (tuples → lists)."""
        result: Dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, dict):
                result[f.name] = {
                    k: list(v) if isinstance(v, tuple) else v for k, v in val.items()
                }
            elif isinstance(val, tuple):
                result[f.name] = list(val)
            else:
                result[f.name] = val
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnMap":
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        known = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        for k, v in data.items():
            if k not in known:
                continue
            if isinstance(v, dict):
                kwargs[k] = {
                    dk: tuple(dv) if isinstance(dv, list) else dv
                    for dk, dv in v.items()
                }
            elif isinstance(v, list):
                kwargs[k] = tuple(v)
            else:
                kwargs[k] = v
        return cls(**kwargs)


@dataclass
class _ResolvedColumns:
    """Integer column indices resolved from a header row."""

    step_id: int
    description: Optional[int]  # None: a sheet in cells with no notes column
    assets: int
    audio: Optional[int] = None
    behaviors: Optional[int] = None
    metadata_pass: Dict[str, int] = field(default_factory=dict)
    row_pass: Dict[str, int] = field(default_factory=dict)
