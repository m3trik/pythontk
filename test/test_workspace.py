# !/usr/bin/python
# coding=utf-8
"""Tests for ``pythontk.file_utils.workspace`` — the workspace.mel codec
(parse / merge-preserving write), the Workspace model (rules, semantic
directory resolution, create/promote), and discovery (find / find_containing,
including the marked-ancestor suppression rule).

All zero-dep and venv-runnable; the Maya-side contract (a real Maya opening a
workspace this module wrote, and preserving foreign rules on its own rewrite)
is covered downstream in ``mayatk/test/test_workspace_mel.py``.
"""

import os
import shutil
import tempfile
import unittest

from pythontk.file_utils.workspace import (
    DEFAULT_FILE_RULES,
    RULE_NICE_NAMES,
    WORKSPACE_MARKER,
    Workspace,
    WorkspaceTemplates,
)

# A realistic Maya-authored file: header comment, variables, spaced rule names,
# escaped quotes, a duplicate rule (later wins), and a hand-written MEL line.
MAYA_SAMPLE = """\
//Maya 2025 Project Definition

workspace -fr "scene" "scenes";
workspace -fr "sourceImages" "sourceimages";
workspace -fr "FBX export" "data";
workspace -fr "odd\\"name" "we\\\\ird";
workspace -v "customVariable" "value";
// a hand-written comment
workspace -fr "scene" "shots/scenes";
if (`about -mac`) { workspace -fr "images" "img"; }
"""


class WorkspaceMelCodecTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ptk_ws_codec_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _marker(self, name="workspace.mel"):
        return os.path.join(self.tmp, name)

    def test_parse_text_and_path(self):
        rules = Workspace.parse_workspace_mel(MAYA_SAMPLE)
        path = self._marker()
        with open(path, "w", encoding="utf-8") as f:
            f.write(MAYA_SAMPLE)
        self.assertEqual(Workspace.parse_workspace_mel(path), rules)

    def test_parse_semantics(self):
        rules = Workspace.parse_workspace_mel(MAYA_SAMPLE)
        self.assertEqual(rules["scene"], "shots/scenes")  # later duplicate wins
        self.assertEqual(rules["FBX export"], "data")  # spaced rule name
        self.assertEqual(rules['odd"name'], "we\\ird")  # MEL escapes unescaped
        self.assertNotIn("customVariable", rules)  # -v is not a file rule
        self.assertNotIn("images", rules)  # non-rule MEL line ignored

    def test_write_fresh_canonical(self):
        path = self._marker()
        self.assertTrue(
            Workspace.write_workspace_mel(
                path, {"scene": "scenes", "sourceImages": "src\\images"}
            )
        )
        with open(path, encoding="utf-8") as f:
            text = f.read()
        self.assertIn('workspace -fr "scene" "scenes";', text)
        # backslashes normalized to Maya's forward-slash convention
        self.assertIn('workspace -fr "sourceImages" "src/images";', text)
        self.assertTrue(text.startswith("//"))
        self.assertTrue(text.endswith("\n"))

    def test_merge_preserves_unknown_lines(self):
        path = self._marker()
        with open(path, "w", encoding="utf-8") as f:
            f.write(MAYA_SAMPLE)
        Workspace.write_workspace_mel(
            path, {"scene": "scenes", "blenderRule": "blender"}
        )
        with open(path, encoding="utf-8") as f:
            text = f.read()
        # unknown lines survive verbatim
        self.assertIn('workspace -v "customVariable" "value";', text)
        self.assertIn("// a hand-written comment", text)
        self.assertIn("if (`about -mac`)", text)
        self.assertIn('workspace -fr "FBX export" "data";', text)
        # managed rule updated in place, duplicate collapsed
        self.assertEqual(text.count('workspace -fr "scene"'), 1)
        self.assertIn('workspace -fr "scene" "scenes";', text)
        # new rule appended among the rules
        self.assertIn('workspace -fr "blenderRule" "blender";', text)
        # round-trip: the merged file parses back to the merged values
        merged = Workspace.parse_workspace_mel(path)
        self.assertEqual(merged["scene"], "scenes")
        self.assertEqual(merged["blenderRule"], "blender")

    def test_unchanged_content_not_rewritten(self):
        path = self._marker()
        Workspace.write_workspace_mel(path, {"scene": "scenes"})
        before = os.stat(path).st_mtime_ns
        self.assertFalse(Workspace.write_workspace_mel(path, {"scene": "scenes"}))
        self.assertEqual(os.stat(path).st_mtime_ns, before)

    def test_remove_drops_only_named_rules(self):
        path = self._marker()
        with open(path, "w", encoding="utf-8") as f:
            f.write(MAYA_SAMPLE)
        Workspace.write_workspace_mel(
            path, {"sourceImages": "tex"}, remove=["scene", "notThere"]
        )
        rules = Workspace.parse_workspace_mel(path)
        self.assertNotIn("scene", rules)  # both duplicate scene lines dropped
        self.assertEqual(rules["sourceImages"], "tex")
        self.assertEqual(rules["FBX export"], "data")  # untouched rule survives
        with open(path, encoding="utf-8") as f:
            text = f.read()
        self.assertIn(
            'workspace -v "customVariable" "value";', text
        )  # non-rules survive

    def test_remove_loses_to_rules(self):
        path = self._marker()
        Workspace.write_workspace_mel(path, {"scene": "scenes"})
        Workspace.write_workspace_mel(path, {"scene": "shots"}, remove=["scene"])
        self.assertEqual(Workspace.parse_workspace_mel(path), {"scene": "shots"})

    def test_bom_file_parses_and_merges(self):
        """A BOM'd file (e.g. PowerShell-written) still parses, and merge updates in place."""
        path = self._marker()
        with open(path, "w", encoding="utf-8-sig") as f:
            f.write('workspace -fr "scene" "scenes";\n')
        self.assertEqual(Workspace.parse_workspace_mel(path), {"scene": "scenes"})
        Workspace.write_workspace_mel(path, {"scene": "shots"})
        with open(path, encoding="utf-8-sig") as f:
            text = f.read()
        self.assertEqual(
            text.count('workspace -fr "scene"'), 1
        )  # updated, not appended
        self.assertEqual(Workspace.parse_workspace_mel(path), {"scene": "shots"})

    def test_crlf_input_is_equivalent(self):
        path = self._marker()
        with open(path, "w", encoding="utf-8", newline="\r\n") as f:
            f.write('//x\nworkspace -fr "scene" "scenes";\n')
        # same rules → content-identical modulo EOLs → no rewrite
        self.assertFalse(Workspace.write_workspace_mel(path, {"scene": "scenes"}))


class WorkspaceModelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ptk_ws_model_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_create_writes_marker_and_dirs(self):
        ws = Workspace.create(os.path.join(self.tmp, "proj"))
        self.assertTrue(ws.is_marked)
        self.assertEqual(
            Workspace.parse_workspace_mel(ws.marker_path), DEFAULT_FILE_RULES
        )
        for rel in ("scenes", "sourceimages", "cache/alembic"):
            self.assertTrue(os.path.isdir(os.path.join(ws.root, rel)), rel)

    def test_create_is_idempotent_promotion(self):
        root = os.path.join(self.tmp, "proj")
        Workspace.create(root, rules={"scene": ".", "myRule": "x"}, create_dirs=False)
        # re-create with the default template: existing rules win, missing fill in
        ws = Workspace.create(root, create_dirs=False)
        self.assertEqual(ws.rules["scene"], ".")
        self.assertEqual(ws.rules["myRule"], "x")
        self.assertEqual(ws.rules["sourceImages"], "sourceimages")

    def test_resolve_rule_is_authoritative(self):
        ws = Workspace(self.tmp, {"sourceImages": "tex/maps"})
        # rule wins even though the folder does not exist yet
        self.assertEqual(
            ws.resolve_dir(("sourceImages",), ("textures",)),
            os.path.normpath(os.path.join(self.tmp, "tex/maps")),
        )
        # absolute rule values pass through
        abs_dir = os.path.join(self.tmp, "elsewhere")
        ws2 = Workspace(self.tmp, {"scene": abs_dir})
        self.assertEqual(ws2.resolve("scene"), os.path.normpath(abs_dir))

    def test_resolve_dir_convention_needs_existing(self):
        ws = Workspace(self.tmp)  # unmarked, no rules
        self.assertEqual(
            ws.resolve_dir(("sourceImages",), ("textures",), default="sourceimages"),
            os.path.join(self.tmp, "sourceimages"),  # convention missing → default
        )
        os.makedirs(os.path.join(self.tmp, "textures"))
        self.assertEqual(
            ws.resolve_dir(("sourceImages",), ("textures",), default="sourceimages"),
            os.path.join(self.tmp, "textures"),  # existing convention wins
        )

    def test_semantic_dirs(self):
        ws = Workspace(self.tmp)  # unmarked, nothing on disk
        self.assertEqual(ws.scene_dir, os.path.normpath(self.tmp))  # falls to root
        marked = Workspace.create(os.path.join(self.tmp, "proj"))
        self.assertEqual(marked.scene_dir, os.path.join(marked.root, "scenes"))
        self.assertEqual(
            marked.source_images_dir, os.path.join(marked.root, "sourceimages")
        )

    def test_promote_describes_a_flat_folder(self):
        root = os.path.join(self.tmp, "flat")
        os.makedirs(os.path.join(root, "textures"))
        open(os.path.join(root, "a.blend"), "w").close()
        ws = Workspace.promote(root, scene_exts=(".blend",))
        self.assertTrue(ws.is_marked)
        self.assertEqual(ws.rules["scene"], ".")
        self.assertEqual(ws.rules["sourceImages"], "textures")
        self.assertFalse(os.path.isdir(os.path.join(root, "scenes")))  # creates nothing

    def test_promote_keeps_conventional_layout_alone(self):
        root = os.path.join(self.tmp, "tidy")
        os.makedirs(os.path.join(root, "scenes"))
        os.makedirs(os.path.join(root, "sourceimages"))
        os.makedirs(os.path.join(root, "textures"))
        open(os.path.join(root, "scenes", "a.ma"), "w").close()
        ws = Workspace.promote(root, scene_exts=(".ma", ".mb"))
        self.assertEqual(ws.rules["scene"], "scenes")
        # sourceimages/ exists, so textures/ is NOT what the rule should point at
        self.assertEqual(ws.rules["sourceImages"], "sourceimages")

    def test_promote_scene_exts_are_per_dcc(self):
        """A .blend at the root is not a scene to a Maya-flavored promotion."""
        root = os.path.join(self.tmp, "blendonly")
        os.makedirs(root)
        open(os.path.join(root, "a.blend"), "w").close()
        self.assertEqual(
            Workspace.promote(root, scene_exts=(".ma", ".mb")).rules["scene"], "scenes"
        )

    def test_promote_does_not_grow_alias_rules_a_caller_omitted(self):
        root = os.path.join(self.tmp, "custom")
        os.makedirs(root)
        open(os.path.join(root, "a.ma"), "w").close()
        ws = Workspace.promote(root, scene_exts=(".ma",), rules={"scene": "scenes"})
        self.assertEqual(ws.rules["scene"], ".")
        self.assertNotIn("mayaAscii", ws.rules)

    def test_promote_is_non_destructive(self):
        root = os.path.join(self.tmp, "keep")
        os.makedirs(root)
        Workspace.promote(root)
        with open(os.path.join(root, WORKSPACE_MARKER), "a", encoding="utf-8") as f:
            f.write('//note\nworkspace -fr "customRule" "custom";\n')
        again = Workspace.promote(root)
        self.assertEqual(again.rules["customRule"], "custom")
        with open(again.marker_path, encoding="utf-8") as f:
            self.assertIn("//note", f.read())

    def test_promote_rejects_a_missing_root(self):
        self.assertIsNone(Workspace.promote(os.path.join(self.tmp, "nope")))
        self.assertIsNone(Workspace.promote(""))

    def test_nice_names_cover_the_default_template(self):
        """Every default rule renders with a Project Window label (rule editors fall back
        to the raw key only for custom rules)."""
        self.assertTrue(set(DEFAULT_FILE_RULES) <= set(RULE_NICE_NAMES))

    def test_fspath_and_equality(self):
        ws = Workspace(self.tmp)
        self.assertEqual(
            os.path.join(ws, "x"), os.path.join(os.path.normpath(self.tmp), "x")
        )
        self.assertEqual(
            ws, Workspace(self.tmp.upper() if os.name == "nt" else self.tmp)
        )


class WorkspaceDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ptk_ws_find_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _touch(self, *parts):
        path = os.path.join(self.tmp, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("")
        return path

    def test_find_marked_and_heuristic(self):
        self._touch("marked", WORKSPACE_MARKER)
        self._touch("loose", "shot.blend")
        os.makedirs(os.path.join(self.tmp, "empty"))
        found = Workspace.find(self.tmp, scene_exts=(".blend",))
        names = [os.path.basename(w.root) for w in found]
        self.assertEqual(sorted(names), ["loose", "marked"])
        by_name = {os.path.basename(w.root): w for w in found}
        self.assertTrue(by_name["marked"].is_marked)
        self.assertFalse(by_name["loose"].is_marked)

    def test_find_suppresses_scenes_under_marked_root(self):
        self._touch("proj", WORKSPACE_MARKER)
        self._touch("proj", "scenes", "shot.blend")
        self._touch("standalone", "thing.blend")
        found = Workspace.find(self.tmp, recursive=True, scene_exts=(".blend",))
        names = sorted(os.path.basename(w.root) for w in found)
        # proj/scenes is part of proj, not its own workspace
        self.assertEqual(names, ["proj", "standalone"])

    def test_find_root_first_ordering(self):
        self._touch("root.blend")
        self._touch("alpha", "a.blend")
        found = Workspace.find(self.tmp, scene_exts=(".blend",))
        self.assertEqual(found[0].root, os.path.normpath(self.tmp))

    def test_find_require_marker(self):
        self._touch("marked", WORKSPACE_MARKER)
        self._touch("loose", "shot.blend")
        found = Workspace.find(self.tmp, scene_exts=(".blend",), require_marker=True)
        self.assertEqual([os.path.basename(w.root) for w in found], ["marked"])

    def test_for_path_marked_ancestor(self):
        self._touch("proj", WORKSPACE_MARKER)
        scene = self._touch("proj", "scenes", "shot.blend")
        ws = Workspace.for_path(scene)
        self.assertEqual(os.path.basename(ws.root), "proj")
        self.assertTrue(ws.is_marked)

    def test_for_path_falls_back_to_own_folder_unmarked(self):
        loose = self._touch("loose", "thing.blend")
        ws = Workspace.for_path(loose)
        self.assertEqual(os.path.basename(ws.root), "loose")
        self.assertFalse(ws.is_marked)

    def test_for_path_edge_cases(self):
        self.assertIsNone(Workspace.for_path(""))
        self.assertIsNone(
            Workspace.for_path(os.path.join(self.tmp, "nope", "deeper", "x.ma"))
        )

    def test_find_containing(self):
        self._touch("proj", WORKSPACE_MARKER)
        blend = self._touch("proj", "scenes", "shot.blend")
        ws = Workspace.find_containing(blend)
        self.assertIsNotNone(ws)
        self.assertEqual(os.path.basename(ws.root), "proj")
        self.assertIsNone(Workspace.find_containing(self._touch("outside", "x.blend")))

    def test_find_invalid_root(self):
        self.assertEqual(Workspace.find(""), [])
        self.assertEqual(Workspace.find(os.path.join(self.tmp, "nope")), [])


class WorkspaceTemplatesTest(unittest.TestCase):
    """The shared, unnamespaced named-rule-set store both DCC adapters read."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ptk_ws_tpl_")
        # Sandbox the preset root — never touch the user's live templates.
        self._prev_root = os.environ.get("UITK_PRESETS_ROOT")
        os.environ["UITK_PRESETS_ROOT"] = os.path.join(self.tmp, "presets")

    def tearDown(self):
        if self._prev_root is None:
            os.environ.pop("UITK_PRESETS_ROOT", None)
        else:
            os.environ["UITK_PRESETS_ROOT"] = self._prev_root
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_store_is_unnamespaced(self):
        """No owning package: a workspace.mel is a SHARED project, so mayatk and
        blendertk must resolve one and the same template directory."""
        store = WorkspaceTemplates.store()
        self.assertEqual(store.package, "")
        # Directly under the config root — no "<package>/" segment in between.
        self.assertEqual(store.user_dir.name, WorkspaceTemplates.STORE_NAME)
        self.assertEqual(
            os.path.normcase(str(store.user_dir.parent)),
            os.path.normcase(os.path.join(self.tmp, "presets")),
        )

    def test_rules_default_to_the_standard_set(self):
        self.assertEqual(WorkspaceTemplates.rules(), DEFAULT_FILE_RULES)

    def test_save_makes_it_the_active_default(self):
        WorkspaceTemplates.save("studio", {"scene": "shots", "sourceImages": "tex"})
        self.assertEqual(
            WorkspaceTemplates.rules(), {"scene": "shots", "sourceImages": "tex"}
        )
        self.assertIn("studio", WorkspaceTemplates.list())

    def test_named_lookup_and_meta_strip(self):
        WorkspaceTemplates.save("gui", {"_meta": {"version": 1}, "scene": "shots"})
        # "_meta" is uitk PresetManager bookkeeping, never a file rule.
        self.assertEqual(WorkspaceTemplates.rules("gui"), {"scene": "shots"})

    def test_delete_falls_back_to_the_standard_set(self):
        WorkspaceTemplates.save("studio", {"scene": "shots"})
        self.assertTrue(WorkspaceTemplates.delete("studio"))
        self.assertEqual(WorkspaceTemplates.rules(), DEFAULT_FILE_RULES)

    def test_unknown_name_falls_back(self):
        self.assertEqual(WorkspaceTemplates.rules("nope"), DEFAULT_FILE_RULES)

    def test_empty_template_never_yields_empty_rules(self):
        """A rule set that saved empty must not build ruleless projects."""
        WorkspaceTemplates.save("blank", {})
        self.assertEqual(WorkspaceTemplates.rules("blank"), DEFAULT_FILE_RULES)

    def test_legacy_per_package_dir_is_migrated(self):
        """Templates saved before the store was shared (blendertk-namespaced)
        survive the move — they are user data."""
        root = WorkspaceTemplates.store().user_dir.parent
        legacy = root / "blendertk" / WorkspaceTemplates.STORE_NAME
        legacy.mkdir(parents=True, exist_ok=True)
        (legacy / "old.json").write_text('{"scene": "legacy"}', encoding="utf-8")
        self.assertIn("old", WorkspaceTemplates.list())
        self.assertEqual(WorkspaceTemplates.rules("old"), {"scene": "legacy"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
