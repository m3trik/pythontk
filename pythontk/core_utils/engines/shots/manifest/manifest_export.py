# !/usr/bin/python
# coding=utf-8
"""The Shot Manifest as a table to hand on: a spreadsheet for the sheet's
owner, a file a pipeline reads, a table for a review.

One record per manifest row (:meth:`ManifestExport.records`, from the
``(step, shot)`` rows of ``ShotPairing.rows``), the steps as the manifest has
them -- Local Edits applied, the names read from a step's text or filled from
the scene -- with their shots' frames and status.  Then rendered as CSV, JSON
or Markdown, for a file (:meth:`ManifestExport.write`) or the clipboard
(:meth:`ManifestExport.render`, :meth:`ManifestExport.clipboard`).  Pure data;
the panel picks the rows, the columns and where it goes.
"""

from __future__ import annotations

import csv
import html
import io
import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["ManifestExport"]


class _ManifestExportInternal(object):
    """Internal helpers for ManifestExport."""

    @staticmethod
    def _frame(value: Optional[float]) -> str:
        """A frame as a cell shows it: ``201``, ``201.5``; ``""`` for none."""
        if value is None:
            return ""
        value = float(value)
        return str(int(value)) if value.is_integer() else str(round(value, 3))

    @staticmethod
    def _section(record: Dict[str, Any]) -> str:
        """``"A: Machine Access"``, or the section's ID alone."""
        section, title = record.get("section", ""), record.get("section_title", "")
        return f"{section}: {title}" if section and title else section or title

    @staticmethod
    def _cells(record: Dict[str, Any]) -> Dict[str, str]:
        """*record*'s step-level cells by column key (objects aside)."""
        frame = _ManifestExportInternal._frame
        return {
            "step": record.get("step") or "",
            "name": record.get("name", ""),
            "shot": record.get("shot", ""),
            "section": _ManifestExportInternal._section(record),
            "description": record.get("description", ""),
            "voice": record.get("voice", ""),
            "start": frame(record.get("start")),
            "end": frame(record.get("end")),
            "status": record.get("status", "").replace("_", " "),
        }

    @staticmethod
    def _object_cells(obj: Dict[str, Any]) -> Dict[str, str]:
        """One object's cells: its name and its behaviors, comma-separated."""
        return {"objects": obj["name"], "behaviors": ", ".join(obj["behaviors"])}


class ManifestExport(_ManifestExportInternal):
    """The Shot Manifest's rows as a table: CSV, JSON or Markdown.

    Example::

        rows = ShotManifest(store).pair(steps).rows(steps)
        records = ManifestExport.records(rows, statuses={"A01": "valid"})
        ManifestExport.write("C:/out/manifest.csv", records, objects="cell")
    """

    #: Each column's key and header, in table order.
    COLUMNS: Tuple[Tuple[str, str], ...] = (
        ("step", "Step"),
        ("name", "Name"),
        ("shot", "Shot"),
        ("section", "Section"),
        ("description", "Description"),
        ("objects", "Objects"),
        ("behaviors", "Behaviors"),
        ("voice", "Voice"),
        ("start", "Start"),
        ("end", "End"),
        ("status", "Status"),
    )
    #: Each format's file extension.
    FORMATS: Dict[str, str] = {"csv": ".csv", "json": ".json", "markdown": ".md"}
    #: How a table lays out a step's objects: ``"rows"`` -- a row each, the
    #: step's own cells on the first (a sheet's continuation rows) -- or
    #: ``"cell"`` -- one row per step, a line per object.
    OBJECT_LAYOUTS: Tuple[str, ...] = ("rows", "cell")

    @classmethod
    def records(
        cls,
        rows: Iterable[Tuple[Optional[Any], Optional[Any]]],
        ranges: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        statuses: Optional[Dict[str, str]] = None,
        object_name: Optional[Callable[[str], str]] = None,
    ) -> List[Dict[str, Any]]:
        """One record per manifest row.

        Parameters:
            rows: ``(step, shot)`` pairs (``ShotPairing.rows``): a
                ``BuilderStep`` and its shot (``None`` before it is built), or
                ``None`` and a shot no step pairs with.
            ranges: ``{step_id: (start, end)}`` for a step with no shot yet --
                where it would go.  A shot's own frames win.
            statuses: ``{step_id: status}`` from an assessment
                (``StepStatus.status``).  Without one, a step's status is
                ``missing_shot``, ``locked`` or ``built``; a shot no step
                pairs with is ``not_in_doc``.
            object_name: How an object's name is written (a host's leaf
                name, as a table shows it); as the step or shot has it by
                default.

        Returns:
            Plain dicts: ``step`` (``None`` for a shot no step pairs with),
            ``name`` (the step's, Local Edits applied -- what Build names its
            shot; ``""`` for a shot no step pairs with), ``shot`` (the name
            its shot has in the scene, ``""`` before it is built: it differs
            from ``name`` until a rename is built, or after a hand renamed
            it), ``section``,
            ``section_title``, ``description``, ``objects``
            (``[{"name", "behaviors", "kind"}]``), ``voice``, ``start``,
            ``end`` and ``status``.
        """
        ranges = ranges or {}
        statuses = statuses or {}
        name = object_name or str
        out: List[Dict[str, Any]] = []
        for step, shot in rows:
            start, end = (shot.start, shot.end) if shot is not None else (None, None)
            if step is None:
                out.append(
                    {
                        "step": None,
                        "name": "",
                        "shot": shot.name,
                        "section": "",
                        "section_title": "",
                        "description": shot.description or "",
                        "objects": [
                            {"name": name(member), "behaviors": [], "kind": "scene"}
                            for member in shot.objects
                        ],
                        "voice": "",
                        "start": start,
                        "end": end,
                        "status": "not_in_doc",
                    }
                )
                continue
            if shot is None:
                start, end = ranges.get(step.step_id, (None, None))
                status = "missing_shot"
            else:
                status = "locked" if shot.locked else "built"
            out.append(
                {
                    "step": step.step_id,
                    "name": step.shot_name,
                    "shot": shot.name if shot is not None else "",
                    "section": step.section,
                    "section_title": step.section_title,
                    "description": step.description,
                    "objects": [
                        {
                            "name": name(o.name),
                            "behaviors": list(o.behaviors),
                            "kind": o.kind,
                        }
                        for o in step.objects
                    ],
                    "voice": step.audio,
                    "start": start,
                    "end": end,
                    "status": statuses.get(step.step_id, status),
                }
            )
        return out

    @classmethod
    def columns(cls, keys: Optional[Iterable[str]] = None) -> List[str]:
        """*keys* in table order, unknown ones left out; every column for
        ``None``."""
        if keys is None:
            return [key for key, _header in cls.COLUMNS]
        wanted = set(keys)
        return [key for key, _header in cls.COLUMNS if key in wanted]

    @classmethod
    def table(
        cls,
        records: List[Dict[str, Any]],
        columns: Optional[Iterable[str]] = None,
        objects: str = "rows",
    ) -> List[List[str]]:
        """*records* as text cells, the header row first.

        Parameters:
            records: From :meth:`records`.
            columns: Column keys (:attr:`COLUMNS`); every column for ``None``.
            objects: :attr:`OBJECT_LAYOUTS` -- ``"rows"`` lays a step's
                objects out a row each when the table has an Objects column.

        Raises:
            ValueError: An unknown *objects* layout.
        """
        if objects not in cls.OBJECT_LAYOUTS:
            raise ValueError(f"Not an object layout: {objects!r}")
        keys = cls.columns(columns)
        headers = dict(cls.COLUMNS)
        table = [[headers[key] for key in keys]]
        a_row_each = objects == "rows" and "objects" in keys
        for record in records:
            cells = cls._cells(record)
            listed = [cls._object_cells(o) for o in record["objects"]]
            if a_row_each:
                blank = dict.fromkeys(cells, "")
                for i, obj in enumerate(listed or [{}]):
                    row = {**(cells if i == 0 else blank), **obj}
                    table.append([row.get(key, "") for key in keys])
                continue
            for key in ("objects", "behaviors"):
                cells[key] = "\n".join(obj[key] for obj in listed)
            table.append([cells[key] for key in keys])
        return table

    @classmethod
    def render(
        cls,
        records: List[Dict[str, Any]],
        fmt: str = "csv",
        columns: Optional[Iterable[str]] = None,
        objects: str = "rows",
    ) -> str:
        """*records* as the text of *fmt* (:attr:`FORMATS`).

        JSON keeps each record whole, less the columns left out (``section``
        carries ``section_title``; an object's ``behaviors`` go with the
        Behaviors column); CSV and Markdown are :meth:`table`'s.

        Raises:
            ValueError: An unknown *fmt* or *objects* layout.
        """
        if fmt not in cls.FORMATS:
            raise ValueError(f"Not an export format: {fmt!r}")
        keys = cls.columns(columns)
        if fmt == "json":
            drop = {key for key, _h in cls.COLUMNS if key not in keys}
            if "section" in drop:
                drop.add("section_title")
            out = []
            for record in records:
                record = {k: v for k, v in record.items() if k not in drop}
                if "objects" in record and "behaviors" in drop:
                    record["objects"] = [
                        {k: v for k, v in o.items() if k != "behaviors"}
                        for o in record["objects"]
                    ]
                out.append(record)
            return json.dumps(out, indent=2, ensure_ascii=False) + "\n"
        table = cls.table(records, keys, objects)
        if fmt == "markdown":
            return cls.markdown(table)
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerows(table)
        return buf.getvalue()

    @staticmethod
    def markdown(table: Sequence[Sequence[str]]) -> str:
        """*table* (header row first) as a Markdown pipe table: a ``|`` in a
        cell escaped, a line break a ``<br>``."""

        def _cell(text: str) -> str:
            return text.replace("|", "\\|").replace("\n", "<br>")

        if not table:
            return ""
        lines = ["| " + " | ".join(_cell(c) for c in table[0]) + " |"]
        lines.append("|" + "|".join(" --- " for _ in table[0]) + "|")
        lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in table[1:]]
        return "\n".join(lines) + "\n"

    @staticmethod
    def clipboard(table: Sequence[Sequence[str]]) -> Tuple[str, str]:
        """``(tsv, html)`` clipboard forms of *table* for a spreadsheet paste.

        Spreadsheets paste the HTML table (a ``<br>`` stays a line break in
        its cell); the TSV, a multi-line cell quoted, is the plain-text
        fallback.
        """

        def _tsv(cell: str) -> str:
            if any(ch in cell for ch in '\t\n"'):
                return '"' + cell.replace('"', '""') + '"'
            return cell

        tsv = "\n".join("\t".join(_tsv(c) for c in row) for row in table)
        body = "".join(
            "<tr>"
            + "".join(
                "<td>" + html.escape(c).replace("\n", "<br>") + "</td>" for c in row
            )
            + "</tr>"
            for row in table
        )
        return tsv, f"<table>{body}</table>"

    @classmethod
    def format_of(cls, path: str) -> Optional[str]:
        """The format *path*'s extension names (:attr:`FORMATS`), or ``None``."""
        lowered = path.lower()
        return next(
            (f for f, ext in cls.FORMATS.items() if lowered.endswith(ext)), None
        )

    @classmethod
    def write(
        cls,
        path: str,
        records: List[Dict[str, Any]],
        columns: Optional[Iterable[str]] = None,
        objects: str = "rows",
    ) -> str:
        """Write *records* to *path*, in the format its extension names, and
        return *path*.

        Written whole or not at all (``FileUtils.atomic_write_text``); a CSV
        carries a UTF-8 byte-order mark, which Excel needs to read anything
        but ASCII.

        Raises:
            ValueError: *path* names no format of :attr:`FORMATS`.
            OSError: The file could not be written.
        """
        from pythontk.file_utils._file_utils import FileUtils

        fmt = cls.format_of(path)
        if fmt is None:
            raise ValueError(f"Not a {', '.join(cls.FORMATS.values())} file: {path!r}")
        text = cls.render(records, fmt, columns, objects)
        FileUtils.atomic_write_text(
            path, text, encoding="utf-8-sig" if fmt == "csv" else "utf-8"
        )
        return path
