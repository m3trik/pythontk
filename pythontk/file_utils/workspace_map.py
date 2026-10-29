# !/usr/bin/python
# coding=utf-8
"""The Workspace Map: the project workspaces under a root folder, measured and
shaped as a folder -> workspace tree for a browser (tentacle's Workspaces panel).

Host-free over :class:`~pythontk.file_utils.workspace.Workspace`: the scan, its
cache, the per-workspace measure (scene count, size, recent scenes) and the
tree are written once here. What differs per host is a short adapter,
:attr:`WorkspaceMapBase.HOST` (a :class:`WorkspaceMapHost`; mayatk's and
blendertk's ``WorkspaceMap`` each bind theirs):

=========================  =======================================================
``current_workspace``      the session's workspace folder (the scan's default root)
``find_workspaces``        the workspace folders a scan under a root lists
``scene_files``            the scene files a workspace holds
``create_workspace``       build a project at a folder from the active template
``promote_workspace``      mark a folder as a project by the layout it has
=========================  =======================================================

With the pure host (:class:`PureWorkspaceMapHost`) a workspace is a marked
folder and its scenes are the files with the host's scene extensions.
"""

import os
from typing import Dict, List, Optional, Protocol, Sequence

from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.file_utils.workspace import Workspace, WorkspaceTemplates


class WorkspaceMapHost(Protocol):
    """What the Workspace Map asks of the host whose projects it browses.

    Each DCC implements it (mayatk's ``MayaWorkspaceMapHost``, blendertk's
    ``BlenderWorkspaceMapHost``); :class:`PureWorkspaceMapHost` holds it with no
    host behind it, and a fake stands in for either in a plain-Python test.
    """

    def current_workspace(self) -> str:
        """The session's workspace folder ('' when the host has none)."""

    def find_workspaces(self, root_dir: str, recursive: bool) -> List[str]:
        """The workspace folders under *root_dir* (itself included) a browser
        lists: ``recursive=False`` looks at *root_dir* and its children only."""

    def scene_files(self, workspace: str) -> List[str]:
        """The scene files *workspace* holds, as full paths."""

    def create_workspace(self, root: str) -> Optional[Workspace]:
        """Build a project at *root* from the active workspace template."""

    def promote_workspace(self, root: str) -> Optional[Workspace]:
        """Mark *root* as a project by the layout it already has; None when it is
        no folder."""


class PureWorkspaceMapHost:
    """The :class:`WorkspaceMapHost` with no host: no session workspace, a
    workspace is a marked folder holding a scene (by *scene_exts*), and projects
    build from the shared template store.

    Parameters:
        scene_exts: The scene file extensions (``(".ma", ".mb")``); none counts
            every marked folder and lists no scenes.
    """

    def __init__(self, scene_exts: Sequence[str] = ()):
        self.scene_exts = tuple(ext.lower() for ext in scene_exts)

    @staticmethod
    def current_workspace() -> str:
        return ""

    def find_workspaces(self, root_dir: str, recursive: bool) -> List[str]:
        found = Workspace.find(root_dir, recursive=recursive, require_marker=True)
        return [
            ws.root
            for ws in found
            if not self.scene_exts or self.scene_files(ws.scene_dir)
        ]

    def scene_files(self, workspace: str) -> List[str]:
        if not (self.scene_exts and os.path.isdir(workspace)):
            return []
        return sorted(
            os.path.join(folder, name)
            for folder, _dirs, names in os.walk(workspace)
            for name in names
            if name.lower().endswith(self.scene_exts)
        )

    @staticmethod
    def create_workspace(root: str) -> Optional[Workspace]:
        return (
            Workspace.create(root, rules=WorkspaceTemplates.rules()) if root else None
        )

    def promote_workspace(self, root: str) -> Optional[Workspace]:
        return Workspace.promote(root, scene_exts=self.scene_exts)


class WorkspaceMapBase(LoggingMixin):
    """The workspaces under a root folder, host-agnostic (see the module docstring).

    Every host read goes through :attr:`HOST`. The scan runs on demand and is
    cached (:attr:`workspace_data`); retargeting the root or the recursion
    rescans.

    Parameters:
        log_level: The logger's level.
    """

    HOST: WorkspaceMapHost = PureWorkspaceMapHost()

    def __init__(self, log_level="WARNING"):
        super().__init__()
        self.set_log_level(log_level)
        self._current_working_dir = None
        self._recursive_search = True
        self._workspace_data = {}

    # ------------------------------------------------------------------ the root
    @property
    def current_workspace(self) -> str:
        """The session's workspace folder (the host's), '' with none."""
        return self.HOST.current_workspace()

    @property
    def current_working_dir(self) -> str:
        """The root folder being scanned: the session's workspace until one is set."""
        if self._current_working_dir is None or not os.path.isdir(
            self._current_working_dir
        ):
            self._current_working_dir = self.current_workspace
        return self._current_working_dir

    @current_working_dir.setter
    def current_working_dir(self, value):
        """Retarget the scan at *value* (an existing folder, by absolute path;
        anything else is ignored: '.', an empty field normalized, is the launch
        folder, never a root) and rescan."""
        if value and os.path.isabs(value) and os.path.isdir(value):
            self._current_working_dir = value
            self.invalidate_workspace_data()

    @property
    def recursive_search(self) -> bool:
        """Whether discovery looks below the root's immediate children."""
        return self._recursive_search

    @recursive_search.setter
    def recursive_search(self, value):
        """Set the discovery depth and rescan."""
        self._recursive_search = value
        self.invalidate_workspace_data()

    # ------------------------------------------------------------------ the scan
    @property
    def workspace_data(self) -> Dict[str, Dict]:
        """``{workspace path: its measure}`` from the last scan (scanning if none)."""
        if self._workspace_data is None:
            self.invalidate_workspace_data()
        return self._workspace_data

    def find_available_workspaces(self, root_dir: Optional[str] = None) -> list:
        """``[(name, path), ...]`` for the workspaces under *root_dir* (default:
        the root being scanned)."""
        if root_dir is None:
            root_dir = self.current_working_dir
        if not root_dir or not os.path.isdir(root_dir):
            return []
        return [
            (os.path.basename(os.path.normpath(path)), path)
            for path in self.HOST.find_workspaces(root_dir, self.recursive_search)
        ]

    def invalidate_workspace_data(self) -> None:
        """Scan for workspaces and rebuild the data cache."""
        self.logger.debug(f"Scanning for workspaces under: {self.current_working_dir}")
        self._workspace_data = {}
        workspaces = self.find_available_workspaces()
        if not workspaces:
            self.logger.warning("No valid workspaces found.")
        for name, path in workspaces:
            if os.path.isdir(path):
                info = self._analyze_workspace(path)
                info["name"] = name
                info["path"] = path
                self._workspace_data[path] = info

    def _analyze_workspace(self, workspace_path: str) -> Dict:
        """Measure a workspace: its scenes, the five most recent, their total size
        and the folders at its root.

        Deliberately cheap: one ``stat`` per scene file (reused for both the
        recent-files sort and the reported size) plus one ``listdir``. Walking
        the workspace's whole tree to total every file, for every workspace on
        each refresh, stalls the UI for seconds on a real project root, so
        ``size_mb`` is the size of the *scene files*, which is what a browser
        reports on.
        """
        info = {
            "scene_count": 0,
            "scenes": [],
            "recent_files": [],
            "subdirectories": [],
            "size_mb": 0,
            "last_modified": None,
        }
        try:
            scenes = self.HOST.scene_files(workspace_path)
            info["scenes"] = scenes
            info["scene_count"] = len(scenes)

            scene_data = []
            total_size = 0
            for scene in scenes:
                try:
                    stat = os.stat(scene)
                except OSError:
                    continue
                scene_data.append((scene, stat.st_mtime))
                total_size += stat.st_size
            if scene_data:
                scene_data.sort(key=lambda x: x[1], reverse=True)
                info["recent_files"] = [s[0] for s in scene_data[:5]]
                info["last_modified"] = scene_data[0][1]
            info["size_mb"] = total_size / (1024 * 1024)

            try:
                info["subdirectories"] = [
                    d
                    for d in os.listdir(workspace_path)
                    if os.path.isdir(os.path.join(workspace_path, d))
                ]
            except OSError:
                pass
        except Exception as e:  # noqa: BLE001 - one bad workspace never stops a scan
            self.logger.error(f"Error analyzing workspace {workspace_path}: {e}")
        return info

    # ------------------------------------------------------------------ the tree
    def get_workspace_tree_data(self, filter_text: Optional[str] = None) -> Dict:
        """The scan as a tree: ``{parent folder name: {"path", "type":
        "directory", "workspaces": {name: measure + "type": "workspace"}}}``,
        narrowed to the workspace names holding *filter_text* (any case)."""
        tree_data = {}
        for workspace_path, workspace_info in self.workspace_data.items():
            workspace_name = workspace_info["name"]
            if filter_text and filter_text.strip():
                if filter_text.lower() not in workspace_name.lower():
                    continue
            parent_dir = os.path.dirname(workspace_path)
            parent_name = os.path.basename(parent_dir) if parent_dir else "Root"
            if parent_name not in tree_data:
                tree_data[parent_name] = {
                    "path": parent_dir,
                    "workspaces": {},
                    "type": "directory",
                }
            tree_data[parent_name]["workspaces"][workspace_name] = {
                **workspace_info,
                "type": "workspace",
            }
        return tree_data

    def get_filtered_workspaces(self, filter_text: Optional[str] = None) -> List[Dict]:
        """The measured workspaces whose names hold *filter_text* (any case)."""
        return [
            info
            for info in self.workspace_data.values()
            if not (filter_text and filter_text.strip())
            or filter_text.lower() in info["name"].lower()
        ]

    # ------------------------------------------------------------------ projects
    # Named create_/mark_PROJECT, not *_workspace: the hosts' EnvUtils own
    # ``create_workspace(root, ...)`` / ``promote_workspace(root)`` with other
    # arguments; one name with two contracts is a trap through a subclass.
    def create_project(self, name: str) -> Optional[Workspace]:
        """Create a project named *name* under the root, built from the ACTIVE
        workspace template (the store both hosts read), and rescan."""
        root = self.current_working_dir
        if not (root and os.path.isdir(root)):
            raise OSError(f"Invalid root directory: {root}")
        ws = self.HOST.create_workspace(os.path.join(root, name))
        self.invalidate_workspace_data()
        return ws

    def mark_root_as_project(self) -> Optional[Workspace]:
        """Mark the ROOT folder as a shared Maya/Blender project (a
        ``workspace.mel`` describing the layout it already has; no file moves),
        and rescan; None when there is no root or it is a project already.

        The root, not a tree row: the folder that needs marking is the unmarked
        one the user browsed to, which is what the root holds.
        """
        root = self.current_working_dir
        if not (root and os.path.isdir(root)) or Workspace(root).is_marked:
            return None
        ws = self.HOST.promote_workspace(root)
        self.invalidate_workspace_data()
        return ws
