# !/usr/bin/python
# coding=utf-8
"""The Shot Manifest's edit layer: a user's changes to a build sheet, kept
apart from the sheet and composed over it.

The sheet is never written.  Every edit is recorded against the sheet's own
identities -- a step by its step ID, an object by the step the sheet lists it
under and the name the sheet spells -- so re-parsing the sheet and applying the
same edits always gives the same steps, showing the sheet as-is is a matter of
not applying them, and applying them again restores every one.  Pure data:
plain values in and out (:meth:`ManifestEdits.to_dict`); the panel keeps one
layer per source on the scene's ``ShotStore.manifest_edits``, found by
:meth:`ManifestEdits.layer_key` -- the sheet's path, else its file name.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pythontk.core_utils.engines.shots.manifest.manifest_model import (
    BuilderObject,
    BuilderStep,
)
from pythontk.net_utils.remote_file import RemoteFile

__all__ = ["ManifestEdits"]

#: Marks a ``sheet`` argument the caller left out (``None`` is a value).
_UNSET = object()


class _ManifestEditsInternal(object):
    """Internal helpers for ManifestEdits."""

    @staticmethod
    def _insert_index(steps: List[BuilderStep], after: str) -> int:
        """Where a step added after *after* goes: right behind that step, at
        the top for ``""``, at the end once the sheet no longer has it."""
        if not after:
            return 0
        for i, step in enumerate(steps):
            if step.step_id == after:
                return i + 1
        return len(steps)

    @staticmethod
    def _find(step: BuilderStep, name: str) -> Optional[BuilderObject]:
        """*step*'s object named *name* (the first), or ``None``."""
        return next((o for o in step.objects if o.name == name), None)

    #: The type each edited field holds: what :meth:`ManifestEdits.from_dict`
    #: keeps (``name`` / ``step`` non-empty, ``behaviors`` all names).
    _FIELD_TYPES = {
        "name": str,
        "description": str,
        "step": str,
        "hidden": bool,
        "behaviors": list,
    }

    @classmethod
    def _kept(cls, record: Any, fields: Tuple[str, ...]) -> Dict[str, Any]:
        """The fields of saved *record* among *fields* that hold their type."""
        if not isinstance(record, dict):
            return {}
        kept = {}
        for key, value in record.items():
            if key not in fields or not isinstance(value, cls._FIELD_TYPES[key]):
                continue
            if key in ("name", "step") and not value.strip():
                continue
            if key == "behaviors" and not all(isinstance(b, str) for b in value):
                continue
            kept[key] = value
        return kept

    @staticmethod
    def _same(value: Any, sheet: Any) -> bool:
        """Whether an edit to *value* leaves the sheet's *sheet* as it was."""
        if isinstance(value, (list, tuple)) and isinstance(sheet, (list, tuple)):
            return list(value) == list(sheet)
        return value == sheet

    @staticmethod
    def _path_form(path: str) -> str:
        """*path* as two spellings of one file compare: either separator, and
        this platform's case rule (``os.path.normcase``)."""
        path = str(path).replace("\\", "/")
        return os.path.normcase(os.path.normpath(path)) if path else ""

    @staticmethod
    def _file_name(path: str) -> str:
        """*path*'s file name, whichever separator it was written with, as
        :meth:`_path_form` compares it."""
        return os.path.normcase(str(path).replace("\\", "/").rsplit("/", 1)[-1])

    #: The store's per-sheet layers, keyed alike and re-keyed together.
    _LAYERS = ("manifest_edits", "manifest_tags", "manifest_options")

    @classmethod
    def _layer_keys(cls, store: Any) -> List[str]:
        """Every source *store* keeps a Shot Manifest layer for -- edits,
        colour tags or the options it is read with -- in the order first
        saved."""
        keys: List[str] = []
        for layers in cls._LAYERS:
            keys += [k for k in getattr(store, layers, None) or {} if k not in keys]
        return keys


class ManifestEdits(_ManifestEditsInternal):
    """A build sheet's edits: renamed, re-described, hidden and added steps;
    renamed, re-behaviored, hidden and moved objects.

    Attributes:
        steps: ``{step_id: {field: value}}`` for :attr:`STEP_FIELDS` -- a sheet
            step's or an added step's.
        objects: ``{home_step_id: {sheet_name: {field: value}}}`` for
            :attr:`OBJECT_FIELDS`, keyed where the sheet lists the object
            (a moved object's ``step`` is where it went).
        added: ``[{"step_id", "after"}]`` -- steps the sheet does not have, in
            the order they were added; each goes right after ``after``.

    Example::

        edits = ManifestEdits()
        edits.set_step("A04", "name", "A04_Intro")
        edits.set_object("A04", "bolt01", "name", "bolt_01")
        steps = edits.apply(ManifestModel.parse_csv(path))
    """

    VERSION = 1
    #: What an edit may change on a step.
    STEP_FIELDS = ("name", "description", "hidden")
    #: What an edit may change on an object.
    OBJECT_FIELDS = ("name", "behaviors", "step", "hidden")
    #: The ID a new step starts from (:meth:`add_step` numbers it unique).
    NEW_STEP_ID = "NEW_STEP"

    def __init__(self):
        self.steps: Dict[str, Dict[str, Any]] = {}
        self.objects: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.added: List[Dict[str, str]] = []

    # ---- persistence -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """The edits as plain JSON-able data (:meth:`from_dict` reads it)."""
        return {
            "version": self.VERSION,
            "steps": copy.deepcopy(self.steps),
            "objects": copy.deepcopy(self.objects),
            "added": copy.deepcopy(self.added),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ManifestEdits":
        """Read :meth:`to_dict` data; anything malformed is left out.

        A tolerant reader: saved scenes outlive this code, so an unknown field
        or a value of the wrong shape is dropped, never raised on.
        """
        edits = cls()
        if not isinstance(data, dict):
            return edits

        def _section(key: str, kind: type):
            value = data.get(key)
            return value if isinstance(value, kind) else kind()

        for step_id, record in _section("steps", dict).items():
            kept = cls._kept(record, cls.STEP_FIELDS)
            if kept:
                edits.steps[str(step_id)] = kept
        for home, records in _section("objects", dict).items():
            if not isinstance(records, dict):
                continue
            for name, record in records.items():
                kept = cls._kept(record, cls.OBJECT_FIELDS)
                if kept:
                    edits.objects.setdefault(str(home), {})[str(name)] = kept
        for record in _section("added", list):
            if isinstance(record, dict) and record.get("step_id"):
                edits.added.append(
                    {
                        "step_id": str(record["step_id"]),
                        "after": str(record.get("after") or ""),
                    }
                )
        return edits

    # ---- the scene's layers --------------------------------------------------

    @classmethod
    def layer_key(cls, store: Any, source: str) -> Optional[str]:
        """The key *source*'s Shot Manifest layers -- these edits and its
        colour tags (``ManifestTags``) -- are saved under on *store*, or
        ``None`` when it keeps none.

        The sheet's path first (spelled either way: separators, and this
        platform's case rule); else the ONE layer saved under a sheet of the
        same file name -- a sheet moved, or opened through another mount,
        keeps its edits (maintainer 2026-10-11).  Two layers of that name are
        two sheets: neither is taken, so same-named sheets never share edits.
        The scene's own shots (``""``) and a sheet read from a URL match only
        themselves: a URL names its sheet whole, and two shared sheets'
        export links end alike (``export?format=csv``).

        Parameters:
            store: A ``ShotStore`` (``manifest_edits`` / ``manifest_tags``),
                or ``None`` (no store: no layer).
            source: The sheet's path, or ``""`` for the scene's own shots.

        Returns:
            The saved key, as saved; ``None`` when nothing matches.
        """
        if store is None:
            return None
        keys = cls._layer_keys(store)
        source = str(source or "")
        if source in keys:
            return source
        if not source or RemoteFile.is_url(source):
            return None
        files = [k for k in keys if k and not RemoteFile.is_url(k)]
        form = cls._path_form(source)
        same = [k for k in files if cls._path_form(k) == form]
        if same:
            return same[0]
        name = cls._file_name(source)
        named = [k for k in files if cls._file_name(k) == name]
        return named[0] if len(named) == 1 else None

    @classmethod
    def same_sheet(cls, a: str, b: str) -> bool:
        """Whether paths *a* and *b* name one sheet, as :meth:`layer_key`
        finds a lone layer: the same path in either spelling, else -- two
        files -- the same file name.  A URL is one sheet only with itself."""
        a, b = str(a or ""), str(b or "")
        if not a or not b or RemoteFile.is_url(a) or RemoteFile.is_url(b):
            return a == b
        if cls._path_form(a) == cls._path_form(b):
            return True
        return cls._file_name(a) == cls._file_name(b)

    @classmethod
    def adopt_layers(cls, store: Any, source: str) -> bool:
        """Move the layers :meth:`layer_key` finds for *source* under
        *source* itself -- edits and tags together, so a source keeps one
        key.  What a save does first: a moved sheet's layer is re-keyed to
        where it is now, and is found there by its exact path from then on.

        Returns:
            Whether anything moved (*store* is then marked dirty).
        """
        key = cls.layer_key(store, source)
        if key is None or key == source:
            return False
        for name in cls._LAYERS:
            layers = getattr(store, name, None)
            if layers is not None and key in layers:
                layers[source] = layers.pop(key)
        store.mark_dirty()
        return True

    @classmethod
    def sheet_options(cls, store: Any, source: str) -> Optional[Dict[str, Any]]:
        """How the scene reads sheet *source*: ``{"template": name or None,
        "values": {option: value}}`` -- the mapping template and its option
        values, kept with the scene beside the sheet's edits (found as they
        are, :meth:`layer_key`); ``None`` for a sheet the scene keeps none
        for, which the user's own options then seed (maintainer 2026-10-12).

        Returns:
            A copy of the record, or ``None``.
        """
        key = cls.layer_key(store, source)
        record = (getattr(store, "manifest_options", None) or {}).get(key)
        if key is None or not isinstance(record, dict):
            return None
        return {
            "template": record.get("template"),
            "values": dict(record.get("values") or {}),
        }

    @classmethod
    def set_sheet_options(
        cls,
        store: Any,
        source: str,
        template: Optional[str],
        values: Optional[Dict[str, Any]],
    ) -> bool:
        """Keep *template* and its option *values* as how the scene reads
        sheet *source* -- re-keyed to *source* first (:meth:`adopt_layers`).

        What a Build, an option change and a saved Local Edit record: a
        sheet's object names, and so its edits' keys, depend on how it was
        read (SOL_REPLACE_ASSEMBLY was built with Object Case = upper; read
        As written, every built object and all 5 edits went missing).

        Returns:
            Whether *store* changed (it is then marked dirty).
        """
        moved = cls.adopt_layers(store, source)
        record = {"template": template or None, "values": dict(values or {})}
        if store.manifest_options.get(source) == record:
            return moved
        store.manifest_options[source] = record
        store.mark_dirty()
        return True

    @classmethod
    def read(cls, store: Any, source: str) -> "ManifestEdits":
        """*source*'s edits saved on *store* (:meth:`layer_key`); empty when
        none are, or *store* is ``None``."""
        key = cls.layer_key(store, source)
        return cls.from_dict(store.manifest_edits.get(key) if key is not None else None)

    def write(self, store: Any, source: str) -> bool:
        """Save these edits on *store* as *source*'s layer -- re-keyed there
        first (:meth:`adopt_layers`) -- or drop the layer once nothing is
        edited.

        Returns:
            Whether *store* changed (it is then marked dirty).
        """
        moved = self.adopt_layers(store, source)
        if self.has_edits:
            data = self.to_dict()
            if store.manifest_edits.get(source) == data:
                return moved
            store.manifest_edits[source] = data
        elif store.manifest_edits.pop(source, None) is None:
            return moved  # nothing saved, nothing to drop
        store.mark_dirty()
        return True

    # ---- queries -----------------------------------------------------------

    @property
    def has_edits(self) -> bool:
        """Whether there is anything to apply (or to reset)."""
        return bool(self.steps or self.objects or self.added)

    def count(self) -> int:
        """How many edits there are: one per edited field, one per added step."""
        n = len(self.added)
        n += sum(len(record) for record in self.steps.values())
        n += sum(
            len(record)
            for records in self.objects.values()
            for record in records.values()
        )
        return n

    def step_record(self, step_id: str) -> Dict[str, Any]:
        """*step_id*'s edited fields (a copy; empty when unedited)."""
        return dict(self.steps.get(step_id, {}))

    def object_record(self, home: str, sheet_name: str) -> Dict[str, Any]:
        """The edited fields of the object the sheet lists as *sheet_name*
        under *home* (a copy; empty when unedited)."""
        return dict(self.objects.get(home, {}).get(sheet_name, {}))

    def is_added(self, step_id: str) -> bool:
        """Whether *step_id* is a step this layer added (not the sheet's)."""
        return any(record["step_id"] == step_id for record in self.added)

    def hidden_steps(self) -> List[str]:
        """The IDs of the hidden steps, in the order they were hidden."""
        return [sid for sid, record in self.steps.items() if record.get("hidden")]

    def hidden_objects(self, step_id: Optional[str] = None) -> List[Tuple[str, str]]:
        """The hidden objects that sit in *step_id* -- listed there, or moved
        there -- or, with ``None``, every hidden object.

        Parameters:
            step_id: The step to list them of; ``None`` lists them all.

        Returns:
            ``[(home, sheet_name)]``: the key each is recorded under (the step
            the sheet lists it in, by the name the sheet spells -- :meth:`anchor`).
        """
        return [
            (home, name)
            for home, records in self.objects.items()
            for name, record in records.items()
            if record.get("hidden")
            and (step_id is None or record.get("step", home) == step_id)
        ]

    @staticmethod
    def anchor(step: BuilderStep, obj: BuilderObject) -> Tuple[str, str]:
        """``(home, sheet_name)`` -- the key an edit to *obj*, shown in
        *step*, is recorded under: where the sheet lists it, by the name the
        sheet spells.  *obj* must come from :meth:`apply` (or the sheet)."""
        return obj.sheet.get("step", step.step_id), obj.sheet.get("name", obj.name)

    @staticmethod
    def name_taken(
        steps: Iterable[BuilderStep], name: str, step_id: Optional[str] = None
    ) -> Optional[str]:
        """The ID of the step other than *step_id* whose shot is already
        named *name* (ignoring case, as the store does), or ``None``."""
        folded = name.casefold()
        for step in steps:
            if step.step_id != step_id and step.shot_name.casefold() == folded:
                return step.step_id
        return None

    def stale(self, steps: Iterable[BuilderStep]) -> List[str]:
        """Each edit the sheet's *steps* no longer give a place to, described.

        An edit of a step or object the sheet dropped (or renamed in the
        sheet itself), a move to a step that is gone or now lists the object
        itself, an added step whose ID the sheet now uses.  :meth:`apply` skips them; they stay recorded so
        a sheet that brings the name back gets its edits back too.
        """
        steps = list(steps)
        ids = {step.step_id for step in steps}
        added = [record["step_id"] for record in self.added]
        out = [f"{sid}: the sheet has a step of that ID" for sid in added if sid in ids]
        known = ids | set(added)
        out += [f"{sid}: no such step" for sid in self.steps if sid not in known]
        by_id = {step.step_id: step for step in steps}
        for home, records in self.objects.items():
            step = by_id.get(home)
            for name, record in records.items():
                if step is None or self._find(step, name) is None:
                    out.append(f"{home} / {name}: no such object")
                elif record.get("step") and record["step"] not in known:
                    out.append(f"{home} / {name}: no step {record['step']} to move to")
                elif record.get("step") in by_id and self._find(
                    by_id[record["step"]], record.get("name", name)
                ):
                    out.append(f"{home} / {name}: {record['step']} lists it already")
        return out

    # ---- edits -------------------------------------------------------------

    def set_step(
        self, step_id: str, field: str, value: Any, sheet: Any = _UNSET
    ) -> None:
        """Edit *field* of step *step_id*; a value equal to the sheet's own
        (*sheet*) drops the edit instead.

        Parameters:
            field: One of :attr:`STEP_FIELDS`.
            value: ``name`` / ``description``: a string; ``hidden``: a bool.
            sheet: The sheet's value of *field* -- default the step ID for
                ``name``, ``False`` for ``hidden``.  Required for
                ``description``: only the sheet knows it.

        Raises:
            ValueError: An unknown field, an empty name, or no *sheet* given
                for a field that has no default.
        """
        if field not in self.STEP_FIELDS:
            raise ValueError(f"Not a step field: {field!r}")
        if sheet is _UNSET:
            defaults = {"name": step_id, "hidden": False}
            if field not in defaults:
                raise ValueError(f"The sheet's {field} is needed to edit it")
            sheet = defaults[field]
        if field == "name":
            value = str(value).strip()
            if not value:
                raise ValueError("A step needs a name")
        elif field == "hidden":
            value = bool(value)
        else:
            value = str(value)
        record = self.steps.setdefault(step_id, {})
        if self._same(value, sheet):
            record.pop(field, None)
        else:
            record[field] = value
        if not record:
            del self.steps[step_id]

    def set_object(
        self, home: str, sheet_name: str, field: str, value: Any, sheet: Any = _UNSET
    ) -> None:
        """Edit *field* of the object the sheet lists as *sheet_name* under
        *home*; a value equal to the sheet's own drops the edit instead.

        Parameters:
            field: One of :attr:`OBJECT_FIELDS` -- ``name`` (a string),
                ``behaviors`` (a list of names), ``step`` (the step it moves
                to; *home* moves it back), ``hidden`` (a bool).
            sheet: The sheet's value of *field* -- defaults to *sheet_name*,
                *home* and ``False``; required for ``behaviors``.

        Raises:
            ValueError: An unknown field, an empty name or step, or no *sheet*
                given for ``behaviors``.
        """
        if field not in self.OBJECT_FIELDS:
            raise ValueError(f"Not an object field: {field!r}")
        if sheet is _UNSET:
            defaults = {"name": sheet_name, "step": home, "hidden": False}
            if field not in defaults:
                raise ValueError(f"The sheet's {field} is needed to edit it")
            sheet = defaults[field]
        if field in ("name", "step"):
            value = str(value).strip()
            if not value:
                raise ValueError(f"An object needs a {field}")
        elif field == "hidden":
            value = bool(value)
        else:
            value = [str(b) for b in value]
        records = self.objects.setdefault(home, {})
        record = records.setdefault(sheet_name, {})
        if self._same(value, sheet):
            record.pop(field, None)
        else:
            record[field] = value
        if not record:
            del records[sheet_name]
        if not records:
            del self.objects[home]

    def revert_step(self, step_id: str, field: Optional[str] = None) -> bool:
        """Drop *step_id*'s edit of *field* (every field when ``None``).
        Returns whether anything was dropped.  An added step stays added:
        :meth:`remove_step` removes it."""
        record = self.steps.get(step_id)
        if not record or (field is not None and field not in record):
            return False
        if field is None:
            record.clear()
        else:
            del record[field]
        if not record:
            del self.steps[step_id]
        return True

    def revert_object(
        self, home: str, sheet_name: str, field: Optional[str] = None
    ) -> bool:
        """Drop the object's edit of *field* (every field when ``None``).
        Returns whether anything was dropped."""
        records = self.objects.get(home) or {}
        record = records.get(sheet_name)
        if not record or (field is not None and field not in record):
            return False
        if field is None:
            record.clear()
        else:
            del record[field]
        if not record:
            del records[sheet_name]
        if not records:
            del self.objects[home]
        return True

    def add_step(
        self, after: str, taken: Iterable[str] = (), step_id: Optional[str] = None
    ) -> str:
        """Add a step right after step *after* (``""``: at the top) and
        return its ID: *step_id* -- a shot added to the manifest keeps its
        name -- else :attr:`NEW_STEP_ID`, numbered past every name in *taken*
        (the steps' IDs and names) and every step already added.

        The ID is the step's identity, as a sheet step's is: it never
        changes, and a rename (:meth:`set_step` ``name``) names its shot.
        """
        used = {str(n).casefold() for n in taken}
        used |= {record["step_id"].casefold() for record in self.added}
        used |= {sid.casefold() for sid in self.steps}
        base = (step_id or "").strip() or self.NEW_STEP_ID
        step_id, n = base, 1
        while step_id.casefold() in used:
            n += 1
            step_id = f"{base}_{n}"
        self.added.append({"step_id": step_id, "after": after or ""})
        return step_id

    def remove_step(self, step_id: str) -> bool:
        """Remove an added step -- its edits, and the moves into it, which
        put those objects back where the sheet lists them.  Returns whether
        *step_id* was an added step."""
        removed = next((r for r in self.added if r["step_id"] == step_id), None)
        if removed is None:
            return False
        self.added.remove(removed)
        self.steps.pop(step_id, None)
        for record in self.added:
            if record["after"] == step_id:
                record["after"] = removed["after"]  # keeps its place
        for home, records in list(self.objects.items()):
            for name, record in list(records.items()):
                if record.get("step") == step_id:
                    self.revert_object(home, name, "step")
        return True

    def rename_all(self, steps: Iterable[BuilderStep], name: str, new: str) -> int:
        """Rename every object named *name* in *steps* (from :meth:`apply`)
        to *new*, each in its own step.  Returns how many were renamed."""
        renamed = 0
        for step in steps:
            for obj in step.objects:
                if obj.name == name:
                    home, sheet_name = self.anchor(step, obj)
                    self.set_object(home, sheet_name, "name", new)
                    renamed += 1
        return renamed

    def clear(self) -> None:
        """Drop every edit: the sheet as-is."""
        self.steps.clear()
        self.objects.clear()
        self.added.clear()

    # ---- composing ---------------------------------------------------------

    def apply(self, steps: List[BuilderStep]) -> List[BuilderStep]:
        """The sheet's *steps* with every edit applied, as new steps.

        *steps* are never changed.  Added steps go in after their anchor and
        take its section; a step's edits set its ``name`` / ``description``;
        an object's set its ``name`` / ``behaviors``, move it to the end of the
        step it went to, or (hidden) leave it out; hidden steps are left out.
        Each changed field's sheet value lands in the copy's ``sheet`` dict.
        An edit the sheet gives no place to (:meth:`stale`) is skipped.
        """
        out = copy.deepcopy(steps)
        ids = {step.step_id for step in out}
        for record in self.added:
            step_id = record["step_id"]
            if step_id in ids:
                continue  # the sheet took the ID: stale
            i = self._insert_index(out, record["after"])
            near = out[i - 1] if i else (out[0] if out else None)
            out.insert(
                i,
                BuilderStep(
                    step_id=step_id,
                    section=near.section if near else "",
                    section_title=near.section_title if near else "",
                    description="",
                ),
            )
            ids.add(step_id)
        by_id = {step.step_id: step for step in out}

        for step_id, record in self.steps.items():
            step = by_id.get(step_id)
            if step is None:
                continue
            if "name" in record:
                step.sheet["name"] = step.shot_name
                step.name = record["name"]
            if "description" in record:
                step.sheet["description"] = step.description
                step.description = record["description"]

        moves: List[Tuple[BuilderObject, BuilderStep, BuilderStep]] = []
        for home, records in self.objects.items():
            step = by_id.get(home)
            if step is None:
                continue
            # Found by the sheet's names before any is renamed: a rename onto
            # another object's sheet name must not capture that one's edits.
            by_name: Dict[str, BuilderObject] = {}
            for obj in step.objects:
                by_name.setdefault(obj.name, obj)
            for sheet_name, record in records.items():
                obj = by_name.get(sheet_name)
                if obj is None:
                    continue
                if "behaviors" in record:
                    obj.sheet["behaviors"] = list(obj.behaviors)
                    obj.behaviors = list(record["behaviors"])
                if "name" in record:
                    obj.sheet["name"] = obj.name
                    obj.name = record["name"]
                target = by_id.get(record.get("step", home))
                if record.get("hidden") or (target is not None and target is not step):
                    step.objects.remove(obj)
                    if target is not None and target is not step:
                        obj.sheet["step"] = home
                        if not record.get("hidden"):
                            moves.append((obj, step, target))
        for obj, home_step, target in moves:
            if self._find(target, obj.name) is None:
                target.objects.append(obj)
            else:  # the step lists that name already (the sheet's since): stale
                del obj.sheet["step"]
                home_step.objects.append(obj)

        hidden = set(self.hidden_steps())
        return [step for step in out if step.step_id not in hidden]
