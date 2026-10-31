# !/usr/bin/python
# coding=utf-8
"""A project's scene files as a scene browser lists and saves them -- pure, host-free.

:class:`SceneListing` holds the rules both DCCs' Reference Manager panel applies:
how a scene file reads as a row (:meth:`~SceneListing.display_name`), which rows a
name or note filter keeps, and the Naming conventions -- a case style, a suffix
and a folder-structure pattern (``{scenes}`` / ``{name}`` / ``{workspace}`` /
``{suffix}``) -- that place a saved scene in its workspace
(:meth:`~SceneListing.save_target`) and filter the listing by that structure. A
script given the same values reproduces the panel's save path. The workspace is
read through :class:`~pythontk.file_utils.workspace.Workspace` (its ``scene``
rule names the ``{scenes}`` folder); nothing here touches a scene.
"""

from __future__ import annotations

import os
from fnmatch import fnmatchcase
from typing import Iterable, List, Optional, Sequence

from pythontk.file_utils.workspace import Workspace
from pythontk.str_utils._str_utils import StrUtils


class SceneListing:
    """Row labels, filter matches and the naming conventions' path rules."""

    #: The delimiters a filter (and a note) splits on.
    DELIMITERS = (",", ";")

    @staticmethod
    def display_name(path: str, hide_extension: bool, hide_suffix: str) -> str:
        """The table label for *path* -- its basename, less the extension and/or a
        TRAILING *hide_suffix*.

        End-anchored on the stem, never a substring pass: the suffix field holds a
        naming token (``_LOC``), and removing it wherever it appears rewrites the
        middle of a name -- ``ITA_LOCKHANDLE.ma`` listed as ``ITAKHANDLE``. The
        extension is split off first so the token is still at the end when the
        column is showing it, and ``endswith`` matches the suffix filter, so a row
        shown for its suffix hides that same one.
        """
        stem, ext = os.path.splitext(os.path.basename(path))
        if hide_suffix and stem.endswith(hide_suffix):
            stem = stem[: -len(hide_suffix)]
        return stem if hide_extension else f"{stem}{ext}"

    @classmethod
    def split_patterns(cls, text: str) -> List[str]:
        """*text* split on every :attr:`DELIMITERS` entry, each part trimmed."""
        parts = [text or ""]
        for delim in cls.DELIMITERS:
            expanded = []
            for part in parts:
                expanded.extend(s.strip() for s in part.split(delim) if s.strip())
            parts = expanded
        return parts

    @staticmethod
    def text_matches_filter(
        text: str, filter_patterns: Sequence[str], ignore_case: bool = True
    ) -> bool:
        """True if *text* (a file name) matches ANY of the already-split glob
        *filter_patterns*."""
        if not text or not filter_patterns:
            return False
        t = text.lower() if ignore_case else text
        for pattern in filter_patterns:
            p = pattern.lower() if ignore_case else pattern
            if fnmatchcase(t, p):
                return True
        return False

    @classmethod
    def matches_notes_filter(
        cls, notes_text: str, filter_text: str, ignore_case: bool = True
    ) -> bool:
        """True if the note *notes_text* matches the filter.

        Notes are often comma or semicolon-delimited ("Layout, Speedrun"), so each
        filter pattern is checked against the whole note as well as each segment.
        """
        if not notes_text or not filter_text:
            return False
        candidates = [notes_text] + cls.split_patterns(notes_text)
        for pattern in cls.split_patterns(filter_text):
            p = pattern.lower() if ignore_case else pattern
            for candidate in candidates:
                c = candidate.lower() if ignore_case else candidate
                if fnmatchcase(c, p):
                    return True
        return False

    @staticmethod
    def scenes_folder(workspace: str) -> str:
        """The ``{scenes}`` placeholder for *workspace*: its ``scene`` file rule when
        that is a relative folder, else the literal ``"scenes"`` -- the folder Save
        writes under, the folder-structure filter matches and both previews show."""
        try:
            rule = Workspace.load(workspace).rules.get("scene") if workspace else None
        except Exception:  # noqa: BLE001 - an unreadable workspace.mel
            rule = None
        return rule if rule and not os.path.isabs(rule) else "scenes"

    @staticmethod
    def format_name(name: str, case_style: str = "None", suffix: str = "") -> str:
        """*name* in the *case_style* convention (``StrUtils.set_case``; ``"None"``
        leaves it), with *suffix* appended -- excluded from the case."""
        if case_style and case_style.startswith("Case: "):
            case_style = case_style[len("Case: ") :]
        if case_style and case_style != "None":
            try:
                name = StrUtils.set_case(name, case_style)
            except Exception:  # noqa: BLE001 - an unknown style leaves the name
                pass
        return name + suffix if suffix else name

    @staticmethod
    def strip_all_suffixes(base: str, suffix: str) -> str:
        """*base* less EVERY trailing *suffix*: a prefill still ending in one (a
        ``_v01_v01`` an old rename doubled) would save under another name than the
        one both previews show."""
        stripped = StrUtils.strip_suffix(base, [suffix])
        while stripped != base:
            base, stripped = stripped, StrUtils.strip_suffix(stripped, [suffix])
        return base

    @staticmethod
    def normalize_structure_pattern(pattern: Optional[str]) -> str:
        """The folder-structure *pattern* trimmed, with the common ``{scene}`` typo
        read as ``{scenes}``: left as typed, ``replace_placeholders`` keeps the
        unknown placeholder verbatim and a save makes a literal folder of it. A
        caller that warns about the typo checks for ``"{scene}"`` first."""
        return (pattern or "").strip().replace("{scene}", "{scenes}")

    @classmethod
    def default_save_name(
        cls, scene_path: Optional[str], case_style: str = "None", suffix: str = ""
    ) -> str:
        """The name a save of the scene at *scene_path* prepopulates: its base, less
        every trailing *suffix* (so it isn't double-appended), case-formatted; ''
        with no scene. Only the extension comes off (``hero.rig.ma`` gives
        ``hero.rig``)."""
        if not scene_path:
            return ""
        base = os.path.splitext(os.path.basename(scene_path))[0]
        return cls.format_name(cls.strip_all_suffixes(base, suffix), case_style)

    @classmethod
    def save_target(
        cls,
        workspace: str,
        name: str,
        case_style: str = "None",
        suffix: str = "",
        pattern: str = "",
        extension: str = "",
    ) -> str:
        """The absolute path a save of *name* into *workspace* writes under the
        Naming conventions: the folder *pattern* resolves to (normalized as
        :meth:`normalize_structure_pattern` reads it; none: the workspace root),
        and the name in *case_style* with *suffix* and *extension* appended.

        Parameters:
            workspace (str): The workspace (project) directory; it must exist.
            name (str): The scene's name, without suffix or extension.
            case_style (str): A ``StrUtils.set_case`` style (``"Case: "`` prefix
                allowed); ``"None"`` leaves the name.
            suffix (str): Appended to the file name, outside the case.
            pattern (str): The folder structure under the workspace (``{scenes}``,
                ``{name}`` -- the cased name --, ``{workspace}``, ``{suffix}``).
            extension (str): The scene file's extension, with its dot (``".ma"``).

        Returns:
            str: The normalized absolute file path.

        Raises:
            ValueError: No valid workspace, or an invalid placeholder pattern.
        """
        if not workspace or not os.path.isdir(workspace):
            raise ValueError("Current workspace directory is invalid.")
        pattern = cls.normalize_structure_pattern(pattern)
        target_dir = workspace
        if pattern:
            try:
                rel = StrUtils.replace_placeholders(
                    pattern,
                    scenes=cls.scenes_folder(workspace),
                    name=cls.format_name(name, case_style),
                    workspace=os.path.basename(os.path.normpath(workspace)),
                    suffix=suffix,
                )
            except ValueError as e:
                raise ValueError(f"Invalid folder structure pattern: {e}") from e
            target_dir = os.path.join(workspace, rel)
        file_name = cls.format_name(name, case_style, suffix)
        return os.path.normpath(os.path.join(target_dir, file_name + extension))

    @staticmethod
    def filter_by_folder_structure(
        files: Iterable[str],
        workspace: str,
        pattern: str,
        suffix: str,
        scenes: str,
    ) -> List[str]:
        """The *files* whose folder ends with *pattern* resolved for each
        (``{scenes}`` / ``{name}`` -- the stem less *suffix* -- / ``{workspace}`` /
        ``{suffix}``); a file may sit deeper than the structure. A file on another
        drive, or a malformed pattern, is left out."""
        workspace_name = os.path.basename(os.path.normpath(workspace))
        kept = []
        for f in files:
            try:
                rel_dir = os.path.relpath(os.path.dirname(f), workspace)
            except ValueError:  # another drive (Windows)
                continue
            base = os.path.splitext(os.path.basename(f))[0]
            name = base[: -len(suffix)] if suffix and base.endswith(suffix) else base
            try:
                expected = StrUtils.replace_placeholders(
                    pattern,
                    scenes=scenes,
                    name=name,
                    workspace=workspace_name,
                    suffix=suffix,
                )
            except ValueError:  # an unbalanced brace: no file can match it
                continue
            rel_parts = os.path.normcase(os.path.normpath(rel_dir)).split(os.sep)
            exp_parts = os.path.normcase(os.path.normpath(expected)).split(os.sep)
            if (
                len(rel_parts) >= len(exp_parts)
                and rel_parts[-len(exp_parts) :] == exp_parts
            ):
                kept.append(f)
        return kept
