# !/usr/bin/python
# coding=utf-8
"""Per-scene notes kept with the project -- one file per project, any host.

A Reference Manager's Notes column, as a store: ONE
``<project>/<data rule>/scene_notes.json`` (Maya's ``data`` rule, else
``data/``), keyed by each scene's project-relative path. The project is the
one :meth:`Workspace.for_scene` names, so the mayatk and blendertk panels read
and write the same file and a note travels with the project. Hoisted from the
two DCC packages, which carried identical copies (2026-10-13).
"""

import os
from typing import Dict, Mapping, Optional, Tuple

from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.workspace import Workspace

#: ``read_json``'s default when an existing file cannot be read or parsed.
_UNREADABLE = object()


class SceneNotes(LoggingMixin):
    """Per-scene notes, kept with the PROJECT: one
    ``<project>/<data rule>/scene_notes.json`` (Maya's ``data`` rule, else
    ``data/``), keyed by each scene's project-relative path.

    One file per project instead of a ``<scene>.metadata.json`` beside every
    listed scene, and the same file every host's panel reads and writes, so a
    note travels with the project and reads the same in Maya and Blender. A
    note in a legacy sidecar (``ptk.Metadata``'s, where notes were kept
    before 2026-10-13) is imported on its first read and the sidecar removed
    once the store holds it; notes a host kept elsewhere, keyed by path, come
    in through :meth:`import_notes`.

    Parameters:
        session_root: The panel's current workspace root: the project of a
            scene no ``workspace.mel`` claims (:meth:`Workspace.for_scene`).
        log_level: The store's logging level.
    """

    FILE_NAME = "scene_notes.json"
    #: The store's format version, written into the file.
    VERSION = 1
    #: ``ptk.Metadata``'s sidecar, where notes were kept before 2026-10-13.
    LEGACY_SIDECAR = ".metadata.json"
    LEGACY_KEY = "Comments"

    def __init__(self, session_root: Optional[str] = None, log_level="WARNING"):
        super().__init__()
        self.logger.setLevel(log_level)
        self.session_root = session_root
        self._cache = {}  # store path -> (mtime_ns, notes dict)

    def store_path(self, scene: str) -> Optional[str]:
        """The notes file of *scene*'s project, there or not; None without one."""
        located = self._locate(scene)
        return located[0] if located else None

    def get(self, scene: str) -> str:
        """*scene*'s note, ``""`` without one (a legacy sidecar's is imported).

        A notes file that exists but cannot be read reads as ``""`` (logged).
        """
        located = self._locate(scene)
        if located is None:
            return ""
        store, key = located
        try:
            notes = self._read(store)
        except OSError as e:
            self.logger.warning(str(e))
            return ""
        found = self._find(notes, key)
        if found is not None:
            return notes[found]
        return self._migrate(scene, store, key)

    def set(self, scene: str, text: str) -> None:
        """Store *text* as *scene*'s note; an empty one removes it.

        Raises:
            OSError: No project holds *scene*, or its notes file exists but
                could not be read (a sharing lock, corrupt JSON) -- nothing is
                written over it, so the project's other notes survive -- or
                could not be written.
        """
        located = self._locate(scene)
        if located is None:
            raise OSError(f"No project holds {scene}: its note has nowhere to go.")
        store, key = located
        notes = dict(self._read(store))
        found = self._find(notes, key)
        if found is not None:
            del notes[found]
        if text:
            notes[key] = text
        self._write(store, notes)
        self._drop_legacy(scene)

    def move(self, old: str, new: str) -> None:
        """Carry *old*'s note to *new* (a renamed scene). Best-effort: logged.

        The new note is written first and the old one dropped after, so a
        failed write leaves the note where it was.
        """
        try:
            text = self.get(old)
            if not text:
                return
            self.set(new, text)
            self.remove(old)
        except OSError as e:
            self.logger.warning(f"Could not carry the note of {old} to {new}: {e}")

    def remove(self, scene: str) -> None:
        """Forget *scene*'s note (a deleted scene). Best-effort: logged; an
        unreadable notes file is left untouched."""
        located = self._locate(scene)
        if located is None:
            return
        store, key = located
        try:
            notes = dict(self._read(store))
            found = self._find(notes, key)
            if found is not None:
                del notes[found]
                self._write(store, notes)
            self._drop_legacy(scene)
        except OSError as e:
            self.logger.warning(f"Could not remove the note of {scene}: {e}")

    def import_notes(self, notes: Mapping[str, str]) -> Dict[str, str]:
        """Move notes a host kept elsewhere -- ``{scene path: text}``, its panel
        settings -- into each scene's project store.

        A note the store already holds for the scene wins (it is the newer
        one); an empty one is dropped.

        Parameters:
            notes: The notes, keyed by each scene's absolute path.

        Returns:
            The notes that stay where they were: their scene is gone, or its
            project's file could not be written.
        """
        kept: Dict[str, str] = {}
        for scene, text in notes.items():
            if not text:
                continue
            if not os.path.isfile(scene):
                kept[scene] = text
                continue
            try:
                if not self.get(scene):
                    self.set(scene, text)
            except OSError:
                kept[scene] = text
        return kept

    # ------------------------------------------------------------------ internals
    def _locate(self, scene: str) -> Optional[Tuple[str, str]]:
        """``(store path, key)`` for *scene*, or None when no project holds it."""
        if not scene:
            return None
        ws = Workspace.for_scene(scene, session_root=self.session_root)
        if ws is None:
            return None
        folder = ws.resolve_dir(("data",), default="data")
        return os.path.join(folder, self.FILE_NAME), ws.relative(scene)

    def _read(self, store: str) -> dict:
        """The notes *store* holds: ``{}`` when it does not exist.

        Raises:
            OSError: *store* exists but could not be read or is not a notes
                document. Never an empty store: writing ``{}`` plus one note
                over it would erase every other scene's note. A failed read
                is never cached.
        """
        try:
            stamp = os.stat(store).st_mtime_ns
        except FileNotFoundError:
            return {}
        cached = self._cache.get(store)
        if cached and cached[0] == stamp:
            return cached[1]
        data = FileUtils.read_json(store, default=_UNREADABLE)
        if data is _UNREADABLE and not os.path.exists(store):
            return {}  # removed between the stat and the read
        notes = data.get("notes", {}) if isinstance(data, dict) else None
        if not isinstance(notes, dict):
            raise OSError(
                f"Could not read the notes file {store} (locked, or not a notes "
                "document): left untouched."
            )
        notes = dict(notes)
        self._cache[store] = (stamp, notes)
        return notes

    def _write(self, store: str, notes: dict) -> None:
        os.makedirs(os.path.dirname(store), exist_ok=True)
        FileUtils.write_json(store, {"version": self.VERSION, "notes": notes}, indent=2)
        self._cache.pop(store, None)

    @staticmethod
    def _find(notes: dict, key: str) -> Optional[str]:
        """The stored spelling of *key* (paths compare as the filesystem does)."""
        if key in notes:
            return key
        folded = os.path.normcase(key)
        return next((k for k in notes if os.path.normcase(k) == folded), None)

    def _migrate(self, scene: str, store: str, key: str) -> str:
        """Import *scene*'s legacy sidecar note into the store; the sidecar goes
        only after the store holds it."""
        sidecar = scene + self.LEGACY_SIDECAR
        if not os.path.isfile(sidecar):
            return ""
        data = FileUtils.read_json(sidecar, default={}) or {}
        text = data.get(self.LEGACY_KEY) if isinstance(data, dict) else None
        if not text:
            return ""
        try:
            notes = dict(self._read(store))
            notes[key] = text
            self._write(store, notes)
        except OSError as e:
            self.logger.debug(f"Kept the legacy note of {scene}: {e}")
            return text
        self._drop_legacy(scene)
        return text

    def _drop_legacy(self, scene: str) -> None:
        """Remove the note from *scene*'s legacy sidecar (the whole sidecar when
        that was all it held)."""
        sidecar = scene + self.LEGACY_SIDECAR
        if not os.path.isfile(sidecar):
            return
        data = FileUtils.read_json(sidecar)
        if not isinstance(data, dict):  # unreadable or not ours: never removed
            self.logger.debug(f"Left the unreadable legacy sidecar {sidecar}.")
            return
        rest = {k: v for k, v in data.items() if k != self.LEGACY_KEY}
        try:
            if rest:
                FileUtils.write_json(sidecar, rest, indent=4)
            else:
                os.remove(sidecar)
        except OSError as e:
            self.logger.debug(f"Could not tidy the legacy sidecar {sidecar}: {e}")
