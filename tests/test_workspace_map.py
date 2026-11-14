# !/usr/bin/python
# coding=utf-8
"""Tests for ``pythontk.file_utils.workspace_map`` -- the Workspace Map's scan,
measure and tree (``WorkspaceMapBase``) over a host adapter.

A fake host records what the map asks of it (the session's workspace, the
discovery, a workspace's scenes, the project builders); the pure host
(``PureWorkspaceMapHost``) runs over real folders. Zero-dep, venv-runnable; the
DCC adapters are mayatk's ``test_workspace.py`` (``TestWorkspaceMap``) and
blendertk's ``test_workspace_map.py``.
"""

import os
import shutil
import tempfile
import unittest

from pythontk.file_utils.workspace import WORKSPACE_MARKER, Workspace
from pythontk.file_utils.workspace_map import PureWorkspaceMapHost, WorkspaceMapBase


def _write(path: str, data: bytes = b"x") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _project(parent: str, name: str, scenes=("scenes/a.ma",)) -> str:
    """A marked project folder holding *scenes* (relative paths)."""
    root = os.path.join(parent, name)
    _write(os.path.join(root, WORKSPACE_MARKER), b'workspace -fr "scene" "scenes";\n')
    for rel in scenes:
        _write(os.path.join(root, rel))
    return root


class _FakeHost:
    """A host whose discovery is every marked folder and whose scenes are the
    files it was told about; it records each call."""

    def __init__(self, session="", scenes=None):
        self.session = session
        self.scenes = scenes or {}
        self.calls = []

    def current_workspace(self):
        self.calls.append(("current_workspace",))
        return self.session

    def find_workspaces(self, root_dir, recursive):
        self.calls.append(("find_workspaces", root_dir, recursive))
        return [ws.root for ws in Workspace.find(root_dir, recursive=recursive)]

    def scene_files(self, workspace):
        return list(self.scenes.get(workspace, []))

    def create_workspace(self, root):
        self.calls.append(("create_workspace", root))
        return Workspace.create(root, create_dirs=False)

    def promote_workspace(self, root):
        self.calls.append(("promote_workspace", root))
        return Workspace.promote(root, scene_exts=(".ma",))


class _Map(WorkspaceMapBase):
    HOST = None  # set per test


class WorkspaceMapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ptk_ws_map_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.alpha = _project(self.tmp, "alpha", ("scenes/a.ma", "scenes/b.ma"))
        self.beta = _project(self.tmp, "beta")
        self.host = _FakeHost(
            session=self.alpha,
            scenes={
                self.alpha: [
                    os.path.join(self.alpha, "scenes", "a.ma"),
                    os.path.join(self.alpha, "scenes", "b.ma"),
                ],
                self.beta: [os.path.join(self.beta, "scenes", "a.ma")],
            },
        )
        _Map.HOST = self.host

    def test_the_root_defaults_to_the_hosts_session_workspace(self):
        wm = _Map()
        self.assertEqual(wm.current_workspace, self.alpha)
        self.assertEqual(wm.current_working_dir, self.alpha)
        wm.current_working_dir = os.path.join(self.tmp, "missing")  # ignored
        self.assertEqual(wm.current_working_dir, self.alpha)

    def test_a_relative_root_is_refused(self):
        """'.' (an empty field normalized) is the launch folder, never a root."""
        wm = _Map()
        wm.current_working_dir = self.tmp
        calls = len(self.host.calls)
        wm.current_working_dir = "."
        self.assertEqual(wm.current_working_dir, self.tmp)
        self.assertEqual(len(self.host.calls), calls)  # no rescan

    def test_no_root_marks_nothing(self):
        """A host with no session workspace and no root set: nothing to mark, and
        never the launch folder."""
        _Map.HOST = _FakeHost(session="")
        self.assertIsNone(_Map().mark_root_as_project())
        self.assertNotIn("promote_workspace", [c[0] for c in _Map.HOST.calls])

    def test_retargeting_the_root_scans_through_the_host(self):
        wm = _Map()
        wm.current_working_dir = self.tmp
        self.assertIn(("find_workspaces", self.tmp, True), self.host.calls)
        self.assertEqual(
            {info["name"] for info in wm.workspace_data.values()}, {"alpha", "beta"}
        )
        wm.recursive_search = False
        self.assertEqual(self.host.calls[-1], ("find_workspaces", self.tmp, False))

    def test_each_workspace_is_measured_by_its_scenes(self):
        wm = _Map()
        info = wm._analyze_workspace(self.alpha)
        self.assertEqual(info["scene_count"], 2)
        self.assertEqual(len(info["recent_files"]), 2)
        self.assertEqual(info["subdirectories"], ["scenes"])
        self.assertAlmostEqual(info["size_mb"], 2 / (1024 * 1024))
        self.assertIsNotNone(info["last_modified"])

    def test_the_tree_groups_by_parent_folder_and_filters_by_name(self):
        wm = _Map()
        wm.current_working_dir = self.tmp
        tree = wm.get_workspace_tree_data()
        parent = os.path.basename(self.tmp)
        self.assertEqual(list(tree), [parent])
        self.assertEqual(tree[parent]["type"], "directory")
        self.assertEqual(sorted(tree[parent]["workspaces"]), ["alpha", "beta"])
        self.assertEqual(tree[parent]["workspaces"]["alpha"]["type"], "workspace")
        narrowed = wm.get_workspace_tree_data("ALP")
        self.assertEqual(list(narrowed[parent]["workspaces"]), ["alpha"])
        self.assertEqual(
            [w["name"] for w in wm.get_filtered_workspaces("bet")], ["beta"]
        )

    def test_a_project_builds_under_the_root_through_the_host(self):
        wm = _Map()
        wm.current_working_dir = self.tmp
        ws = wm.create_project("gamma")
        built = ("create_workspace", os.path.join(self.tmp, "gamma"))
        self.assertEqual(self.host.calls[-2], built)  # then the rescan
        self.assertTrue(ws.is_marked)

    def test_no_root_builds_nothing(self):
        """A host with no session workspace and no root set: nowhere to build."""
        _Map.HOST = _FakeHost(session="")
        with self.assertRaises(OSError):
            _Map().create_project("delta")

    def test_marking_the_root_promotes_it_once(self):
        loose = os.path.join(self.tmp, "loose")
        _write(os.path.join(loose, "a.ma"))
        wm = _Map()
        wm.current_working_dir = loose
        ws = wm.mark_root_as_project()
        self.assertIn(("promote_workspace", loose), self.host.calls)
        self.assertEqual(ws.rules["scene"], ".")
        self.assertIsNone(wm.mark_root_as_project())  # marked now


class PureWorkspaceMapHostTest(unittest.TestCase):
    """The pure host over real folders: a workspace is a marked folder holding
    a scene; with no extensions every marked folder counts and none has scenes."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ptk_ws_pure_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.full = _project(self.tmp, "full", ("scenes/a.ma", "scenes/sub/b.mb"))
        self.empty = _project(self.tmp, "empty", ())
        _write(os.path.join(self.tmp, "unmarked", "c.ma"))

    def test_discovery_needs_a_marker_and_a_scene(self):
        host = PureWorkspaceMapHost((".ma", ".mb"))
        self.assertEqual(host.current_workspace(), "")
        self.assertEqual(host.find_workspaces(self.tmp, recursive=True), [self.full])
        self.assertEqual(
            sorted(PureWorkspaceMapHost().find_workspaces(self.tmp, recursive=True)),
            sorted([self.empty, self.full]),
        )

    def test_a_workspaces_scenes_are_its_scene_files(self):
        host = PureWorkspaceMapHost((".MA", ".mb"))
        self.assertEqual(
            [os.path.relpath(p, self.full) for p in host.scene_files(self.full)],
            [os.path.join("scenes", "a.ma"), os.path.join("scenes", "sub", "b.mb")],
        )
        self.assertEqual(PureWorkspaceMapHost().scene_files(self.full), [])

    def test_the_default_map_runs_on_the_pure_host(self):
        class _PureMap(WorkspaceMapBase):
            HOST = PureWorkspaceMapHost((".ma", ".mb"))

        wm = _PureMap()
        wm.current_working_dir = self.tmp
        self.assertEqual(
            [info["name"] for info in wm.workspace_data.values()], ["full"]
        )
        ws = wm.create_project("built")
        self.assertTrue(ws.is_marked)


if __name__ == "__main__":
    unittest.main()
