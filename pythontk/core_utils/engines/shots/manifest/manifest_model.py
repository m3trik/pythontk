# !/usr/bin/python
# coding=utf-8
"""Pure Shot Manifest data model + CSV parser.

DCC-agnostic — no ``maya`` / ``bpy`` / Qt.  Holds the step/object graph a
structured production CSV parses into, the column-mapping schema, behavior
detection, and the assessment/plan result dataclasses.  Duration resolution and
key emission (which reach a scene) live in the hooked engine layer, not here.
"""
import csv
import io
import logging
import re
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from pythontk import SchemaSpec, spec_field
from pythontk.core_utils.engines.shots.shot_model import ShotStore

log = logging.getLogger(__name__)


__all__ = [
    "BuilderObject",
    "BuilderStep",
    "PlannedShot",
    "ObjectStatus",
    "StepStatus",
    "ColumnMap",
    "Action",
    "FitMode",
    "DEFAULT_INITIAL_SHOT_LENGTH",
    "DEFAULT_FIT_MODE",
    "AUDIO_PLACEHOLDER_DURATION",
    "detect_behaviors",
    "parse_csv",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BuilderObject:
    """One asset within a step."""

    name: str
    behaviors: List[str] = field(default_factory=list)  # e.g. ["fade_in", "fade_out"]
    kind: str = "scene"  # "scene" | "audio"
    source_path: str = ""  # file path for audio creation (transient)


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


# ---------------------------------------------------------------------------
# Build plan (compute-then-commit)
# ---------------------------------------------------------------------------

Action = Literal["created", "patched", "skipped", "locked", "removed"]


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
    status: str  # "valid" | "missing_object" | "missing_behavior" | "user_animated"
    behaviors: List[str] = field(
        default_factory=list
    )  # expected behaviors (empty = user-animated)
    broken_behaviors: List[str] = field(
        default_factory=list
    )  # subset of *behaviors* that failed verification
    key_range: Optional[Tuple[float, float]] = None  # actual keyframe extent


@dataclass
class StepStatus:
    """Assessment result for one step."""

    step_id: str
    built: bool  # shot exists in sequencer
    objects: List[ObjectStatus] = field(default_factory=list)
    additional_objects: List[str] = field(default_factory=list)  # in shot but not CSV
    shrinkable_frames: float = 0.0  # frames of unused range at step tail
    locked: bool = False  # shot is user-finalized; skip automated flags

    @property
    def status(self) -> str:
        """Worst-of-children rollup.

        Priority: ``"locked"`` (user-finalized) > ``"missing_shot"``
        > ``"missing_object"`` > ``"missing_behavior"`` > ``"valid"``.
        """
        if self.locked:
            return "locked"
        if not self.built:
            return "missing_shot"
        if any(o.status == "missing_object" for o in self.objects):
            return "missing_object"
        if any(o.status == "missing_behavior" for o in self.objects):
            return "missing_behavior"
        return "valid"

    @property
    def missing_count(self) -> int:
        return sum(1 for o in self.objects if o.status == "missing_object")

    @property
    def total_count(self) -> int:
        return len(self.objects)


# ---------------------------------------------------------------------------
# Behavior detection
# ---------------------------------------------------------------------------

_BEHAVIOR_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bfades?\s+in\b", re.I), "fade_in"),
    (re.compile(r"\bfades?\s+out\b", re.I), "fade_out"),
]


def detect_behaviors(text: str) -> List[str]:
    """Return behavior names inferred from descriptive *text*.

    Each pattern is tested independently so text mentioning both
    "fades in" and "fades out" yields ``["fade_in", "fade_out"]``.
    """
    found = []
    for pattern, name in _BEHAVIOR_PATTERNS:
        if pattern.search(text):
            found.append(name)
    return found


# ---------------------------------------------------------------------------
# CSV column mapping
# ---------------------------------------------------------------------------


@dataclass
class ColumnMap(SchemaSpec):
    """Maps logical fields to CSV header names (case-insensitive).

    Each field is a tuple of acceptable header aliases. The parser reads
    the header row and resolves names to column indices automatically.

    A :class:`~pythontk.SchemaSpec`, so the ``columns`` block of a mapping
    file is self-validating and self-documenting; serialisable via
    :meth:`to_dict` / :meth:`from_dict` (tuple ⇄ list) so instances round-trip
    through JSON.
    """

    step_id: Tuple[str, ...] = spec_field(
        help="Header alias(es) for the step-ID column.",
        example=["Step"],
        default=("Step",),
    )
    description: Tuple[str, ...] = spec_field(
        help="Header alias(es) for the step description / contents column.",
        example=["Step Contents", "Contents"],
        default=("Step Contents", "Contents"),
    )
    assets: Tuple[str, ...] = spec_field(
        help="Header alias(es) for the assets / object-names column.",
        example=["Asset Names", "Asset"],
        default=("Asset Names", "Asset"),
    )
    audio: Tuple[str, ...] = spec_field(
        help="Header alias(es) for the audio / voice-over column (optional).",
        example=["Voice Support", "Voice"],
        default=("Voice Support", "Voice"),
    )
    exclude_steps: Tuple[str, ...] = spec_field(
        help="Step IDs to skip entirely (e.g. setup rows).",
        example=["SETUP"],
        default=("SETUP",),
    )
    exclude_values: Dict[str, Tuple[str, ...]] = spec_field(
        help='Per-field cell values to treat as empty, e.g. {"assets": ["N/A"]}.',
        example={"assets": ["N/A"]},
        default_factory=lambda: {"assets": ("N/A",)},
    )
    metadata_pass: Dict[str, Tuple[str, ...]] = spec_field(
        help='Extra columns copied into shot metadata: {key: [header aliases]}.',
        example={"priority": ["Priority"]},
        default_factory=dict,
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
            else:
                result[f.name] = list(val)
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
    metadata_pass: Dict[str, int] = field(default_factory=dict)


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
    )
    for key, aliases in col_map.metadata_pass.items():
        idx = _find_optional(aliases)
        if idx is not None:
            resolved.metadata_pass[key] = idx
    return resolved


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^SECTION\s+([A-Z0-9]+)\s*:\s*(.*)", re.I)
_STEP_RE = re.compile(r"^([A-Z]\d+)\.\)")
_ALT_STEP_RE = re.compile(r"^([A-Z]{2,})$")  # non-numbered IDs: SETUP, INTRO …


def _strip_cell(cell: str) -> str:
    """Strip whitespace from a CSV cell."""
    return (cell or "").strip()


def _read_csv_rows(filepath: str) -> List[List[str]]:
    """Read all CSV rows, tolerating non-UTF-8 encodings.

    Production manifests frequently come out of Excel as cp1252; a
    UTF-8-only read used to fail the entire load on the first accented
    character.  The UTF-8 BOM is stripped from the raw bytes up front —
    otherwise a BOM'd file that falls back to cp1252 (or the lossy
    read) leaks it into the first header cell and the header never
    resolves.  Tries UTF-8 first, then cp1252, then a lossy UTF-8 read
    as a last resort so a stray byte can't kill the manifest.
    """
    with open(filepath, "rb") as fh:
        raw = fh.read()
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


def parse_csv(
    filepath: str,
    columns: Optional[ColumnMap] = None,
    post_process: Optional[Callable[[BuilderStep], None]] = None,
) -> List[BuilderStep]:
    """Parse a structured CSV into a list of :class:`BuilderStep`.

    Parameters:
        filepath: Path to the CSV file.
        columns: Optional header-name mapping.  Defaults cover
            common layouts (C-5M, C-130H, C-17A).
        post_process: Optional callable invoked on each step after
            assembly.  Use to compute derived fields (e.g.
            audio objects) from the parsed data.

    Returns:
        Ordered list of steps, each carrying its objects and detected
        behaviors.
    """
    col_map = columns or ColumnMap()
    cols: Optional[_ResolvedColumns] = None
    steps: List[BuilderStep] = []
    seen_ids: set = set()
    current_section = ""
    current_section_title = ""
    current_step: Optional[BuilderStep] = None
    step_id_aliases = {a.lower() for a in col_map.step_id}
    asset_excludes = {v.upper() for v in col_map.exclude_values.get("assets", ())}
    # Per-step accumulator for metadata_pass values (first-row-wins)
    step_pass: Dict[str, str] = {}

    rows = _read_csv_rows(filepath)
    for row in rows:
        if not row:
            continue

        first = _strip_cell(row[0])

        # --- section header ---
        sec_match = _SECTION_RE.match(first)
        if sec_match:
            current_section = sec_match.group(1)
            current_section_title = sec_match.group(2).strip()
            current_step = None
            continue

        # --- column header row (resolve indices) ---
        if first.lower() in step_id_aliases:
            cols = _resolve_columns(row, col_map)
            continue
        # The step-ID header may not sit in the first column.  Accept a
        # row with a matching cell anywhere — but only if it actually
        # resolves (which also demands the description/assets headers),
        # so a data row containing a bare alias can't hijack the column
        # map.  Not gated on ``cols is None``: layouts repeat the header
        # per section, and an unrecognized repeat would be misread as a
        # continuation row of the previous step.
        if any(_strip_cell(c).lower() in step_id_aliases for c in row):
            try:
                cols = _resolve_columns(row, col_map)
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
            _strip_cell(row[cols.step_id]) if len(row) > cols.step_id else ""
        )
        step_match = _STEP_RE.match(step_cell) or _ALT_STEP_RE.match(step_cell)
        description = (
            _strip_cell(row[cols.description])
            if len(row) > cols.description
            else ""
        )
        asset = _strip_cell(row[cols.assets]) if len(row) > cols.assets else ""
        audio = (
            _strip_cell(row[cols.audio])
            if cols.audio is not None and len(row) > cols.audio
            else ""
        )

        if step_match:
            step_id = step_match.group(1)
            if step_id in seen_ids:
                log.warning("Duplicate step_id '%s' — skipping.", step_id)
                current_step = None
                continue
            seen_ids.add(step_id)
            step_behaviors = detect_behaviors(description)
            # Collect metadata_pass values for this step row
            step_pass = {}
            for key, idx in cols.metadata_pass.items():
                val = _strip_cell(row[idx]) if len(row) > idx else ""
                if val:
                    step_pass[key] = val
            current_step = BuilderStep(
                step_id=step_id,
                section=current_section,
                section_title=current_section_title,
                description=description,
                audio=audio,
            )
            current_step._pass_through = dict(step_pass)
            steps.append(current_step)

            if asset and asset.upper() not in asset_excludes:
                obj = BuilderObject(
                    name=asset,
                    behaviors=list(step_behaviors),
                )
                current_step.objects.append(obj)
            continue

        # --- continuation row (belongs to previous step) ---
        if current_step is not None:
            # Merge continuation description into the parent step
            if description:
                current_step.description += " " + description

            if asset and asset.upper() not in asset_excludes:
                # Own description overrides, otherwise inherit from parent step
                row_behaviors = detect_behaviors(description) if description else []
                behaviors = row_behaviors or detect_behaviors(
                    current_step.description
                )
                obj = BuilderObject(
                    name=asset,
                    behaviors=list(behaviors),
                )
                current_step.objects.append(obj)

    # A missing header row used to fail silently: every data row was
    # skipped and the caller saw 0 steps with no explanation.
    if cols is None and rows:
        log.warning(
            "No header row found in '%s' — 0 steps parsed. Expected a "
            "column named one of %s.",
            filepath,
            sorted(step_id_aliases),
        )

    # Apply exclude list
    if col_map.exclude_steps:
        excluded = {s.upper() for s in col_map.exclude_steps}
        steps = [s for s in steps if s.step_id.upper() not in excluded]

    # Apply post-processing hook (e.g. derive audio objects from step fields)
    if post_process:
        for step in steps:
            post_process(step)

    return steps
