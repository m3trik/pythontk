# !/usr/bin/python
# coding=utf-8
"""Shared project-workspace model + ``workspace.mel`` codec.

A *workspace* is the ecosystem's project concept: a root directory plus named
file rules (``scene`` → ``scenes``, ``sourceImages`` → ``sourceimages``, …)
that say where each kind of file lives. Maya's ``workspace.mel`` marker file is
the on-disk serialization — despite the extension it is a flat, order-tolerant
rule store (``workspace -fr "rule" "path";`` lines) that Maya round-trips
losslessly *including rule names it does not recognize*, which makes it a
legitimate shared format rather than a Maya-only one: a single project folder
serves Maya (natively) and Blender (via ``blendertk``) at the same time.

Pure-Python mechanism only — **no** DCC import, per the pythontk charter
(mirrors ``file_utils.usd``: the dependency-free floor beneath the DCC
adapters). Three composable primitives:

- :func:`parse_workspace_mel` / :func:`write_workspace_mel` — the codec. The
  writer is merge-preserving: managed rules update their existing lines in
  place, unrecognized MEL (comments, variables, hand-written lines) survives
  verbatim, and an unchanged file is never rewritten.
- :class:`Workspace` — the model: root + rules, semantic directory resolution
  (rule → existing conventional folder → default), load/save/create, and
  discovery (:meth:`Workspace.find`, :meth:`Workspace.find_containing`).

Downstream: ``mayatk`` keeps ``cmds.workspace`` as the in-Maya authority for
the *active* project (Maya parses the marker natively); ``blendertk`` — which
has no native project system — builds its current-workspace resolver and
rule-fed folder accessors on this module.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence

#: The marker file that makes a directory a workspace (Maya's project file).
WORKSPACE_MARKER = "workspace.mel"

#: Maya-standard file-rule template used when creating a workspace from
#: scratch. Deliberately a curated subset — Maya fills in any rule it misses
#: with its own defaults on project load; consumers resolve semantically via
#: :meth:`Workspace.resolve_dir`, never by hardcoding folder names.
DEFAULT_FILE_RULES: Dict[str, str] = {
    "scene": "scenes",
    "mayaAscii": "scenes",
    "mayaBinary": "scenes",
    "sourceImages": "sourceimages",
    "images": "images",
    "movie": "movies",
    "sound": "sound",
    "audio": "sound",
    "scripts": "scripts",
    "autoSave": "autosave",
    "fileCache": "cache/nCache",
    "alembicCache": "cache/alembic",
    "OBJ": "data",
    "FBX": "data",
}

#: Display labels for well-known file rules — Maya's own Project Window
#: vocabulary, in its Primary → Secondary display order. Rule editors show
#: these "nice names" (falling back to the raw key for custom rules) and use
#: the dict's order to sort rows, so a shared project reads the same way in
#: both DCCs and in Maya's native window.
RULE_NICE_NAMES: Dict[str, str] = {
    # Primary Project Locations
    "scene": "Scenes",
    "templates": "Templates",
    "images": "Images",
    "sourceImages": "Source Images",
    "renderData": "Render Data",
    "clips": "Clips",
    "sound": "Sound",
    "scripts": "Scripts",
    "diskCache": "Disk Cache",
    "movie": "Movies",
    "translatorData": "Translator Data",
    "timeEditor": "Time Editor",
    "autoSave": "AutoSave",
    "sceneAssembly": "Scene Assembly",
    # Secondary Project Locations (the commonly shared subset)
    "offlineEdit": "Offline Edits",
    "3dPaintTextures": "3D Paint Textures",
    "depth": "Depth",
    "iprImages": "IPR Images",
    "shaders": "Shaders",
    "particles": "Particle Cache",
    "fluidCache": "Fluid Cache",
    "fileCache": "File Cache",
    "mayaAscii": "Maya Ascii",
    "mayaBinary": "Maya Binary",
    "mel": "mel",
    "OBJ": "OBJ",
    "audio": "audio",
    "move": "move",
    # Translator / Custom Data Locations
    "FBX": "FBX",
    "FBX export": "FBX export",
    "OBJexport": "OBJ export",
    "alembicCache": "Alembic Cache",
}

_HEADER = "//Maya Project Definition"

# One `workspace -fr "name" "value";` (or long-form -fileRule) line. Quoted
# fields may contain MEL escapes (\" and \\) and spaces ("FBX export").
_RULE_LINE_RE = re.compile(
    r'^\s*workspace\s+(?:-fr|-fileRule)\s+'
    r'"((?:[^"\\]|\\.)*)"\s+"((?:[^"\\]|\\.)*)"\s*;\s*$'
)


def _unescape(s: str) -> str:
    return re.sub(r"\\(.)", r"\1", s)


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def parse_workspace_mel(source: str) -> Dict[str, str]:
    """File rules from a ``workspace.mel`` — *source* is a path or the file's text.

    Later duplicates of a rule win (MEL executes top to bottom). Lines that are
    not file rules (comments, ``workspace -v`` variables, hand-written MEL) are
    ignored here — :func:`write_workspace_mel` preserves them verbatim instead.
    """
    if "\n" not in source and os.path.isfile(source):
        # utf-8-sig: tolerate a BOM (e.g. PowerShell-written files) — no-op otherwise.
        with open(source, "r", encoding="utf-8-sig", errors="replace") as f:
            text = f.read()
    else:
        text = source.lstrip("﻿")
    rules: Dict[str, str] = {}
    for line in text.splitlines():
        m = _RULE_LINE_RE.match(line)
        if m:
            rules[_unescape(m.group(1))] = _unescape(m.group(2))
    return rules


def write_workspace_mel(
    path: str,
    rules: Dict[str, str],
    preserve: bool = True,
    remove: Sequence[str] = (),
) -> bool:
    """Write / merge file *rules* into the ``workspace.mel`` at *path*.

    ``preserve=True`` (default) never destroys what it can't parse: each managed
    rule updates its first existing line in place (later duplicates of that rule
    collapse into it), new rules are appended after the last rule line, and every
    other line — comments, variables, hand-authored MEL — survives verbatim.
    ``preserve=False`` rewrites the file canonically (header + rules only).

    *remove* names rules whose lines should be **dropped** — the only way a rule
    ever leaves the file, since the merge deliberately never deletes on its own
    (an editor computes ``set(old) - set(new)`` and passes it here). A name in
    both *rules* and *remove* is written (rules win).

    Rule values are normalized to forward slashes (Maya's own convention). A
    content-identical result is not rewritten. Returns True when the file changed.
    """
    rules_out = {str(k): str(v).replace("\\", "/") for k, v in rules.items()}
    remove_set = {str(r) for r in remove} - set(rules_out)
    existing_text = None
    if os.path.isfile(path):
        # utf-8-sig: a BOM (e.g. PowerShell-written files) would otherwise glue
        # to the first rule line and defeat the in-place merge.
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            existing_text = f.read()

    def rule_line(name: str) -> str:
        return f'workspace -fr "{_escape(name)}" "{_escape(rules_out[name])}";'

    if preserve and existing_text is not None:
        out: List[str] = []
        handled: set = set()
        last_rule_idx = -1
        for line in existing_text.splitlines():
            m = _RULE_LINE_RE.match(line)
            if m:
                name = _unescape(m.group(1))
                if name in remove_set:
                    continue  # explicit removal — drop every line of this rule
                if name in rules_out:
                    if name in handled:
                        continue  # collapse a duplicate managed line
                    handled.add(name)
                    out.append(rule_line(name))
                else:
                    out.append(line)
                last_rule_idx = len(out) - 1
            else:
                out.append(line)
        new = [rule_line(n) for n in rules_out if n not in handled]
        if new:
            pos = last_rule_idx + 1 if last_rule_idx >= 0 else len(out)
            out[pos:pos] = new
        text = "\n".join(out)
    else:
        text = "\n".join([_HEADER, ""] + [rule_line(n) for n in rules_out])

    if not text.endswith("\n"):
        text += "\n"
    if existing_text is not None and existing_text.replace("\r\n", "\n") == text:
        return False
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return True


class Workspace:
    """A project workspace: root directory + named file rules.

    Rules map a semantic name to a directory (workspace-relative unless
    absolute). An *unmarked* workspace (no ``workspace.mel``) is the same model
    with an empty rule set — resolution falls through to existing conventional
    folders, then defaults — so casual folders-of-scenes work with zero
    ceremony and :meth:`save`/:meth:`create` *promote* them when asked.
    """

    def __init__(self, root: str, rules: Optional[Dict[str, str]] = None):
        self.root = os.path.normpath(root)
        self.rules: Dict[str, str] = dict(rules) if rules else {}

    def __repr__(self) -> str:
        return f"Workspace({self.root!r}, rules={len(self.rules)})"

    def __fspath__(self) -> str:
        return self.root

    def __eq__(self, other):
        if isinstance(other, Workspace):
            return os.path.normcase(self.root) == os.path.normcase(other.root)
        return NotImplemented

    def __hash__(self):
        return hash(os.path.normcase(self.root))

    # ------------------------------------------------------------------ marker
    @property
    def marker_path(self) -> str:
        return os.path.join(self.root, WORKSPACE_MARKER)

    @property
    def is_marked(self) -> bool:
        return os.path.isfile(self.marker_path)

    # ------------------------------------------------------------- persistence
    @classmethod
    def load(cls, root: str) -> "Workspace":
        """The workspace at *root* — rules parsed from its marker when present,
        empty (unmarked) otherwise."""
        ws = cls(root)
        if ws.is_marked:
            ws.rules = parse_workspace_mel(ws.marker_path)
        return ws

    def save(self, create_dirs: bool = False, remove: Sequence[str] = ()) -> "Workspace":
        """Persist the rules to the marker file (merge-preserving — see
        :func:`write_workspace_mel`). ``create_dirs`` also makes each
        workspace-relative rule directory; *remove* drops the named rules from
        the file (an editor passes the rules the user deleted)."""
        os.makedirs(self.root, exist_ok=True)
        write_workspace_mel(self.marker_path, self.rules, remove=remove)
        if create_dirs:
            for rel in set(self.rules.values()):
                if rel and rel != "." and not os.path.isabs(rel):
                    os.makedirs(os.path.normpath(os.path.join(self.root, rel)), exist_ok=True)
        return self

    @classmethod
    def create(
        cls,
        root: str,
        rules: Optional[Dict[str, str]] = None,
        create_dirs: bool = True,
    ) -> "Workspace":
        """Create (or promote) *root* as a marked workspace.

        *rules* defaults to :data:`DEFAULT_FILE_RULES`. Idempotent on an
        already-marked workspace: its existing rules win — the template only
        fills in missing keys — so promoting never clobbers a project.
        """
        ws = cls.load(root)
        for k, v in (dict(DEFAULT_FILE_RULES) if rules is None else dict(rules)).items():
            ws.rules.setdefault(k, v)
        return ws.save(create_dirs=create_dirs)

    # -------------------------------------------------------------- resolution
    def _abs(self, value: str) -> str:
        v = str(value)
        return os.path.normpath(v if os.path.isabs(v) else os.path.join(self.root, v))

    def resolve(self, *rule_names: str, default: Optional[str] = None) -> Optional[str]:
        """Absolute directory for the first present rule; *default*
        (workspace-relative unless absolute) when none is; None when neither."""
        for name in rule_names:
            if name in self.rules:
                return self._abs(self.rules[name])
        return self._abs(default) if default is not None else None

    def resolve_dir(
        self,
        rule_names: Sequence[str],
        conventions: Sequence[str] = (),
        default: Optional[str] = None,
    ) -> Optional[str]:
        """Semantic directory lookup: rule → first *existing* conventional
        subdir → *default*. A present rule is authoritative even when its folder
        doesn't exist yet (rules say where files *should* go); conventions only
        match folders that actually exist."""
        hit = self.resolve(*rule_names)
        if hit is not None:
            return hit
        for c in conventions:
            p = self._abs(c)
            if os.path.isdir(p):
                return p
        return self._abs(default) if default is not None else None

    @property
    def scene_dir(self) -> str:
        """Where scene files live (rule ``scene``/``mayaAscii``/``mayaBinary``,
        an existing ``scenes/``, else the root)."""
        return self.resolve_dir(("scene", "mayaAscii", "mayaBinary"), ("scenes",), default=".")

    @property
    def source_images_dir(self) -> str:
        """Where textures live (rule ``sourceImages``, an existing
        ``sourceimages/`` or ``textures/``, else ``sourceimages/``)."""
        return self.resolve_dir(("sourceImages",), ("sourceimages", "textures"), default="sourceimages")

    # --------------------------------------------------------------- discovery
    @staticmethod
    def _holds_scene(directory: str, exts: tuple) -> bool:
        try:
            return any(
                f.lower().endswith(exts) and os.path.isfile(os.path.join(directory, f))
                for f in os.listdir(directory)
            )
        except OSError:
            return False

    @classmethod
    def find(
        cls,
        root_dir: str,
        recursive: bool = False,
        scene_exts: Sequence[str] = (),
        require_marker: bool = False,
    ) -> List["Workspace"]:
        """Workspaces under *root_dir* (the root itself included).

        Marked directories (containing :data:`WORKSPACE_MARKER`) always count.
        With *scene_exts* (e.g. ``(".blend",)``), a directory directly holding
        such files counts as an unmarked workspace — unless it sits inside a
        marked one (a project's ``scenes/`` folder is part of that project, not
        its own workspace) or ``require_marker`` is set. ``recursive=False``
        looks at *root_dir* and its immediate children only. Sorted root first,
        then alphabetically.
        """
        if not (root_dir and os.path.isdir(root_dir)):
            return []
        exts = tuple(e.lower() for e in scene_exts)
        root_norm = os.path.normpath(root_dir)

        def classify(d: str) -> Optional[bool]:
            """True = marked, False = heuristic candidate, None = not a workspace."""
            if os.path.isfile(os.path.join(d, WORKSPACE_MARKER)):
                return True
            if exts and not require_marker and cls._holds_scene(d, exts):
                return False
            return None

        candidates: Dict[str, bool] = {}
        if recursive:
            for dirpath, _dirnames, _files in os.walk(root_norm):
                flag = classify(dirpath)
                if flag is not None:
                    candidates[os.path.normpath(dirpath)] = flag
        else:
            flag = classify(root_norm)
            if flag is not None:
                candidates[root_norm] = flag
            try:
                names = sorted(os.listdir(root_norm))
            except OSError:
                names = []
            for name in names:
                sub = os.path.join(root_norm, name)
                if os.path.isdir(sub):
                    flag = classify(sub)
                    if flag is not None:
                        candidates[os.path.normpath(sub)] = flag

        stop = os.path.normcase(os.path.dirname(root_norm))

        def under_marked(d: str) -> bool:
            cur = os.path.dirname(d)
            while cur and os.path.normcase(cur) != stop:
                if os.path.isfile(os.path.join(cur, WORKSPACE_MARKER)):
                    return True
                nxt = os.path.dirname(cur)
                if nxt == cur:
                    break
                cur = nxt
            return False

        keep = [d for d, marked in candidates.items() if marked or not under_marked(d)]
        keep.sort(key=lambda p: (p != root_norm, p.lower()))
        return [cls.load(d) for d in keep]

    @classmethod
    def find_containing(cls, path: str) -> Optional["Workspace"]:
        """The nearest marked workspace containing *path* (file or directory),
        walking up — the shared analogue of mayatk's
        ``EnvUtils.find_workspace_using_path``. None when no marker is found."""
        if not path:
            return None
        cur = os.path.abspath(path)
        if not os.path.isdir(cur):
            cur = os.path.dirname(cur)
        while cur:
            if os.path.isfile(os.path.join(cur, WORKSPACE_MARKER)):
                return cls.load(cur)
            nxt = os.path.dirname(cur)
            if nxt == cur:
                return None
            cur = nxt
        return None


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------
