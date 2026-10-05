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
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from pythontk import SchemaSpec
from pythontk.str_utils._str_utils import StrUtils
from pythontk.core_utils.engines.shots.shot_model import ShotStore
from pythontk.net_utils.remote_file import RemoteFile

log = logging.getLogger(__name__)


__all__ = [
    "BuilderObject",
    "BuilderStep",
    "PlannedShot",
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

        Raises:
            ValueError: If a required column cannot be found.
        """
        normalized = [c.strip().lower() for c in header]

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
            description=_find(col_map.description, "description"),
            assets=_find(col_map.assets, "assets"),
            audio=_find_optional(col_map.audio) if col_map.audio else None,
            behaviors=_find_optional(col_map.behaviors) if col_map.behaviors else None,
        )
        for key, aliases in col_map.metadata_pass.items():
            idx = _find_optional(aliases)
            if idx is not None:
                resolved.metadata_pass[key] = idx
        return resolved

    @staticmethod
    def _parse_rows(
        rows: List[List[str]], col_map: "ColumnMap", source: str
    ) -> List["BuilderStep"]:
        """Assemble steps from already-read CSV *rows* (see :meth:`parse_csv`).

        Exclusions and post-processing are the caller's; each step records
        its asset cell's ``(row, column)`` so a column can be written back.
        """
        cols: Optional[_ResolvedColumns] = None
        steps: List[BuilderStep] = []
        seen_ids: set = set()
        current_section = ""
        current_section_title = ""
        current_step: Optional[BuilderStep] = None
        step_id_aliases = {a.lower() for a in col_map.step_id}
        step_res = [re.compile(p) for p in col_map.step_pattern]
        section_res = [re.compile(p) for p in col_map.section_pattern]
        _first = _ManifestModelInternal._first_match
        _group = _ManifestModelInternal._group
        asset_excludes = {v.upper() for v in col_map.exclude_values.get("assets", ())}

        behavior_source = col_map.behavior_source
        from pythontk.core_utils.engines.shots.manifest.behaviors import Behaviors

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
            # listing several (as ``asset_column`` writes them) splits safely.
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
                continue

            # --- column header row (resolve indices) ---
            if first.lower() in step_id_aliases:
                cols = _ManifestModelInternal._resolve_columns(row, col_map)
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
            description = (
                _ManifestModelInternal._strip_cell(row[cols.description])
                if len(row) > cols.description
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
            row_behaviors, speaks = _row_behaviors(description, behavior_cell)

            if step_match:
                # The step becomes a shot whose name is its exported clip
                # name: spell it legally here (``B03.5`` -> ``B03_5``) so the
                # store accepts it and every later lookup agrees.
                step_id = StrUtils.to_legal_name(_group(step_match, 1))
                if step_id in seen_ids:
                    log.warning("Duplicate step_id '%s' — skipping.", step_id)
                    current_step = None
                    continue
                seen_ids.add(step_id)
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
                )
                current_step._pass_through = dict(step_pass)
                steps.append(current_step)

                current_step.asset_cell = (row_idx, cols.assets)
                if asset:
                    assets_spoken.add(step_id)
                for name in _asset_names(asset):
                    _add_object(current_step, name, step_behaviors)
                if read_prose:
                    _note_subjects(step_id, description)
                continue

            # --- continuation row (belongs to previous step) ---
            if current_step is not None:
                # Merge continuation description into the parent step
                if description:
                    current_step.description += " " + description
                    if read_prose:
                        _note_subjects(current_step.step_id, description)

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

        # A missing header row used to fail silently: every data row was
        # skipped and the caller saw 0 steps with no explanation.
        if cols is None and rows:
            log.warning(
                "No header row found in '%s' — 0 steps parsed. Expected a "
                "column named one of %s.",
                source,
                sorted(step_id_aliases),
            )
        for step in steps:
            if step.step_id not in assets_spoken:
                step.objects.extend(prose_objects.get(step.step_id, {}).values())
        return steps

    #: Values ``ColumnMap.behavior_source`` accepts.
    BEHAVIOR_SOURCES = ("description", "column", "column_else_description")
    #: Values ``ColumnMap.object_source`` accepts.
    OBJECT_SOURCES = ("column", "column_else_description")
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
    def asset_column(
        source: str,
        fills: Dict[str, List[str]],
        columns: Optional["ColumnMap"] = None,
    ) -> List[str]:
        """The source sheet's asset column, row for row, with *fills* written
        into the empty asset cells of their steps.

        Pasted over the sheet's asset column from its first row, it lands
        every name in its step's row and rewrites every other cell with the
        value it already holds -- a paste covers the whole range, so the
        column must carry what is there.  The source is re-read and re-parsed
        (with the same *columns*), so rows added since the load still line up;
        a step's names share one cell, one per line, which :meth:`parse_csv`
        reads back as one object each.

        Parameters:
            source: The CSV path or URL the steps were parsed from.
            fills: ``{step_id: [asset names]}`` (``ShotManifest.fill_missing_assets``).
            columns: The column map the steps were parsed with.

        Returns:
            One cell per row, from the first row through the last step's row.

        Raises:
            ValueError: No step row was found, or the asset column moves
                between sections (no single column can be pasted).
        """
        col_map = columns or ColumnMap()
        rows = _ManifestModelInternal._read_csv_rows(source)
        steps = _ManifestModelInternal._parse_rows(rows, col_map, source)
        cells = [s.asset_cell for s in steps if s.asset_cell is not None]
        if not cells:
            raise ValueError(f"No step rows found in {source!r}.")
        cols = {c for _r, c in cells}
        if len(cols) > 1:
            raise ValueError(
                "The asset column is not the same in every section, so no "
                "single column can be pasted back."
            )
        col = cols.pop()
        last = max(r for r, _c in cells)
        column = [
            (row[col] if len(row) > col else "").strip() for row in rows[: last + 1]
        ]
        for step in steps:
            names = fills.get(step.step_id)
            row = step.asset_cell[0] if step.asset_cell else None
            if names and row is not None and not column[row]:
                column[row] = "\n".join(names)
        return column

    @staticmethod
    def column_clipboard(cells: List[str]) -> Tuple[str, str]:
        """``(tsv, html)`` clipboard forms of one column of *cells*.

        Spreadsheets paste the HTML table (a ``<br>`` stays a line break in
        its cell); the TSV, with multi-line cells quoted, is the plain-text
        fallback.
        """
        import html

        def _tsv(cell: str) -> str:
            if any(ch in cell for ch in '\t\n"'):
                return '"' + cell.replace('"', '""') + '"'
            return cell

        tsv = "\n".join(_tsv(c) for c in cells)
        body = "".join(
            "<tr><td>" + html.escape(c).replace("\n", "<br>") + "</td></tr>"
            for c in cells
        )
        return tsv, f"<table>{body}</table>"

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
    # "description" (named by a behavior phrase in the step's prose) or
    # "shot" (auto-filled from the paired shot).
    origin: str = "column"


@dataclass
class BuilderStep:
    """One step (= one future sequencer shot)."""

    step_id: str  # e.g. "A04"
    section: str  # e.g. "A"
    section_title: str  # e.g. "AILERON RIGGING"
    description: str  # merged step-contents text (used for behavior detection)
    objects: List[BuilderObject] = field(default_factory=list)
    audio: str = ""  # narration/voice-over text from CSV
    # Extra columns copied verbatim into shot metadata (first-row-wins); filled
    # by :func:`parse_csv` from a ColumnMap's ``metadata_pass``.
    _pass_through: Dict[str, str] = field(default_factory=dict)
    # (row, column) of this step's asset cell in the source CSV, so a
    # generated asset list can be written back into the right cell.
    asset_cell: Optional[Tuple[int, int]] = None

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
            # A shot bound to a doc step IS that step (``ShotManifest.pair``).
            step_id = meta.get("step") or shot.name
            steps.append(
                cls(
                    step_id=step_id,
                    section=meta.get("section", ""),
                    section_title=meta.get("section_title", ""),
                    description=shot.description,
                    objects=objects,
                    audio=meta.get("voice_text", ""),
                )
            )
            ranges[step_id] = (shot.start, shot.end)
        return steps, ranges


# ---------------------------------------------------------------------------
# Build plan (compute-then-commit)
# ---------------------------------------------------------------------------

Action = Literal["created", "patched", "skipped", "locked", "removed", "refused"]


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

    @property
    def status(self) -> str:
        """Worst-of-children rollup.

        Priority: ``"locked"`` (user-finalized) > ``"missing_shot"`` >
        the object statuses in :attr:`ROLLUP` order > ``"no_objects"`` (the
        doc lists none -- informational, never a silent "valid") > ``"valid"``.
        """
        if self.locked:
            return "locked"
        if not self.built:
            return "missing_shot"
        found = {o.status for o in self.objects}
        for status in self.ROLLUP:
            if status in found:
                return status
        return "valid" if self.objects else "no_objects"

    #: What each status means and what to do about it -- the one wording every
    #: panel shows (step rows, object rows, "not in doc" shot rows).
    HELP = {
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
    }

    #: Object statuses that name the step, worst first.
    ROLLUP = (
        "missing_object",
        "ambiguous_object",
        "not_in_shot",
        "behavior_conflict",
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
        object is one a Build fixes (:attr:`BUILD_FIXES`), or the shot holds
        keys of behaviors the doc dropped.  Never for a locked step."""
        if self.locked:
            return False
        if not self.built or self.dropped_behaviors:
            return True
        return any(o.status in self.BUILD_FIXES for o in self.objects)

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
        how: ``{step_id: "binding" | "name" | "order"}`` -- how it paired.
        orphans: Shots no step pairs with ("not in the doc"), timeline order.
    """

    shots: Dict[str, Any] = field(default_factory=dict)
    how: Dict[str, str] = field(default_factory=dict)
    orphans: List[Any] = field(default_factory=list)


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
    description: int
    assets: int
    audio: Optional[int] = None
    behaviors: Optional[int] = None
    metadata_pass: Dict[str, int] = field(default_factory=dict)
