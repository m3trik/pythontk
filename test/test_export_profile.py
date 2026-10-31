# !/usr/bin/python
# coding=utf-8
"""ExportProfile -- the Scene Exporter panels' export-button contract, once."""

import logging
import os
import unittest

from pythontk.core_utils.engines.scene_export.export_profile import (
    ExportProfile,
    ExportRun,
)
from pythontk.str_utils._str_utils import StrUtils


TASKS = {
    "export_visible_objects": {
        "widget_type": "ComboBox",
        "add": {"All": "all", "Visible": "visible", "Selected": "selected"},
    },
    "export_data_node": {"widget_type": "QCheckBox", "setChecked": True},
    "convert_textures": {
        "widget_type": "ComboBox",
        "object_name": "texture_output",
        "add": {"OFF": None, "Unity": "unity", "glTF": "gltf"},
    },
    "optimize_textures": {
        "widget_type": "ComboBox",
        "object_name": "texture_optimize",
        "add": {"OFF": None, "Native": True, "2048": 2048, "1024": 1024},
    },
    "ignore_groups": {
        "widget_type": "QLineEdit",
        "value_method": "text",
        "setText": "",
    },
    "smart_bake": {"widget_type": "QCheckBox"},
}
CHECKS = {
    "check_default_materials": {"widget_type": "QCheckBox", "setChecked": True},
    "check_path_length": {
        "widget_type": "SpinBox",
        "value_method": "value",
        "setValue": 240,
    },
    "check-with dash": {"widget_type": "QCheckBox"},
}


class TestNaming(unittest.TestCase):
    def test_widget_key_is_the_object_name_or_the_legal_name(self):
        self.assertEqual(ExportProfile.widget_key("a-b c", {}), "a_b_c")
        self.assertEqual(
            ExportProfile.widget_key("convert_textures", TASKS["convert_textures"]),
            "texture_output",
        )

    def test_the_widget_reads_are_the_panels(self):
        """pythontk knows no Qt: the panel reads its rows (tentacle's
        SceneExporterSlots, through uitk) and hands ``{objectName: value}`` in.
        The reading rule and its widget vocabulary left with it (2026-10-29)."""
        for name in ("read_values", "value_method", "CHECKBOX"):
            self.assertFalse(hasattr(ExportProfile, name), name)


class TestLegacyRegexFold(unittest.TestCase):
    """The retired RegEx field folds into the token system's modifier grammar,
    so ONE regex implementation serves the field and the inline form."""

    def test_an_explicit_delimiter_passes_through(self):
        for spec in ("_bar.*->", "a=>b", "(foo|bar)->baz"):
            with self.subTest(spec=spec):
                self.assertEqual(ExportProfile.fold_legacy_regex(spec), spec)

    def test_the_legacy_pipe_shorthand_becomes_an_arrow(self):
        """`A|B` meant "replace A with B" in the retired field. It cannot stay a
        delimiter -- that is regex alternation -- so it folds."""
        self.assertEqual(ExportProfile.fold_legacy_regex("scene|asset"), "scene->asset")

    def test_a_bare_pattern_deletes_its_match(self):
        self.assertEqual(ExportProfile.fold_legacy_regex("_bar.*"), "_bar.*->")

    def test_blank_is_no_modifier(self):
        for value in (None, "", "   "):
            with self.subTest(value=value):
                self.assertIsNone(ExportProfile.fold_legacy_regex(value))

    def test_the_fold_round_trips_through_the_shared_primitive(self):
        """What the old field did, the new primitive must still do."""
        for regex, expected in (
            ("test_->prod_", "prod_scene"),
            ("scene|asset", "test_asset"),
            ("_scene.*", "test"),
        ):
            with self.subTest(regex=regex):
                spec = ExportProfile.fold_legacy_regex(regex)
                got, error = StrUtils.apply_regex_modifier("test_scene", spec)
                self.assertIsNone(error)
                self.assertEqual(got, expected)


class TestLegacyRegexFoldsIntoThePattern(unittest.TestCase):
    """Retiring the RegEx field must not drop the rule it held: it folds onto
    the name token so the pattern states the whole thing."""

    def test_the_wildcard_carries_the_folded_regex(self):
        self.assertEqual(
            ExportProfile.fold_legacy_naming("WIP_*", name_regex="test_->prod_"),
            "WIP_{scene:test_->prod_}",
        )

    def test_a_retired_token_spelling_still_gets_it(self):
        """A saved `{name}_x` has always had the RegEx applied; the alias is
        deprecated, not broken."""
        self.assertEqual(
            ExportProfile.fold_legacy_naming("{name}_x", name_regex="test_->prod_"),
            "{name:test_->prod_}_x",
        )

    def test_no_regex_leaves_the_pattern_untouched(self):
        for value in (None, "", "   "):
            with self.subTest(value=value):
                self.assertEqual(
                    ExportProfile.fold_legacy_naming("WIP_*", name_regex=value), "WIP_*"
                )

    def test_the_folded_pattern_resolves_to_what_the_field_produced(self):
        """The whole point: same file, whether the rule came from the retired
        field or from an inline modifier."""
        ctx = {"scene": "test_scene", "name": "test_scene"}
        # A saved RegEx field folds first, as both exporters do at their entry.
        from_field = ExportProfile.resolve_output_path(
            ExportProfile.fold_legacy_naming("WIP_*", name_regex="test_->prod_"), ctx
        )
        inline = ExportProfile.resolve_output_path("WIP_{scene:test_->prod_}", ctx)
        self.assertEqual(from_field["stem"], "WIP_prod_scene")
        self.assertEqual(from_field["stem"], inline["stem"])


class TestRunConfig(unittest.TestCase):
    def _values(self, **overrides):
        """An untouched panel's widget values, as ``read_values`` would return them."""
        values = {
            "export_visible_objects": "all",
            "export_data_node": True,
            "texture_output": None,
            "texture_optimize": None,
            "ignore_groups": "",
            "smart_bake": False,
            "check_default_materials": True,
            "check_path_length": 240,
            "check_with_dash": False,
        }
        values.update(overrides)
        return values

    def test_off_entries_are_dropped_and_the_scope_is_lifted_out(self):
        config = ExportProfile.run_config(
            self._values(export_visible_objects="selected"), TASKS, CHECKS
        )
        self.assertEqual(config["export_mode"], "selected")
        self.assertFalse(config["export_visible"])
        self.assertNotIn("export_visible_objects", config["tasks"])
        self.assertNotIn("smart_bake", config["tasks"])
        self.assertTrue(config["tasks"]["export_data_node"])
        self.assertTrue(config["tasks"]["check_default_materials"])
        self.assertEqual(config["tasks"]["check_path_length"], 240)

    def test_an_optimize_choice_sets_the_cap_and_arms_its_check(self):
        config = ExportProfile.run_config(
            self._values(texture_optimize=1024, texture_output="gltf"), TASKS, CHECKS
        )
        tasks = config["tasks"]
        self.assertEqual(tasks["texture_max_size"], 1024)
        self.assertEqual(tasks["optimize_textures"], "gltf")
        self.assertEqual(tasks["check_texture_optimization"], "gltf")
        self.assertEqual(tasks["check_material_compatibility"], "gltf")

    def test_a_native_optimize_choice_arms_the_check_without_a_cap(self):
        tasks = ExportProfile.run_config(
            self._values(texture_optimize=True), TASKS, CHECKS
        )["tasks"]
        self.assertNotIn("texture_max_size", tasks)
        self.assertIs(tasks["optimize_textures"], True)
        self.assertIs(tasks["check_texture_optimization"], True)

    def test_override_drops_every_check_but_the_preconditions(self):
        config = ExportProfile.run_config(
            self._values(texture_optimize=True), TASKS, CHECKS, override_checks=True
        )
        self.assertNotIn("check_default_materials", config["tasks"])
        self.assertNotIn("check_path_length", config["tasks"])
        self.assertNotIn("check_texture_optimization", config["tasks"])
        self.assertTrue(config["tasks"]["export_data_node"])
        for check in ExportProfile.ALWAYS_CHECKS:
            self.assertIs(config["tasks"][check], True)

    def test_the_writability_check_runs_with_no_row(self):
        """A destination held open fails the write whatever the panel says, and
        finding out first costs milliseconds, so no row asks."""
        self.assertIn("check_output_writable", ExportProfile.ALWAYS_CHECKS)
        tasks = ExportProfile.run_config(self._values(), TASKS, CHECKS)["tasks"]
        self.assertIs(tasks["check_output_writable"], True)

    # The fix rows the DCC panels offer, each standing in for its check.
    FIXES = {
        "conform_shape_names": {"widget_type": "QCheckBox"},
        "reassign_duplicate_materials": {"widget_type": "QCheckBox"},
        "flatten_sheared_chains": {"widget_type": "QCheckBox"},
        "snap_keys_to_frame": {"widget_type": "QCheckBox"},
        "tie_all_keyframes": {"widget_type": "QCheckBox"},
    }

    def test_a_fix_arms_the_check_that_verifies_it(self):
        """What the fix could not repair still stops the export."""
        values = dict.fromkeys(self.FIXES, True)
        tasks = ExportProfile.run_config(values, self.FIXES, {})["tasks"]
        for fix in self.FIXES:
            with self.subTest(fix=fix):
                self.assertIs(tasks[ExportProfile.VERIFYING_CHECKS[fix]], True)

    def test_a_fix_left_off_takes_its_concern_with_it(self):
        values = dict.fromkeys(self.FIXES, False)
        tasks = ExportProfile.run_config(values, self.FIXES, {})["tasks"]
        for fix in self.FIXES:
            with self.subTest(fix=fix):
                self.assertNotIn(ExportProfile.VERIFYING_CHECKS[fix], tasks)

    def test_a_panel_still_offering_the_check_keeps_its_say(self):
        """An older DCC package builds the check row too; its unticked box wins."""
        checks = {"check_untied_keyframes": {"widget_type": "QCheckBox"}}
        values = {"tie_all_keyframes": True, "check_untied_keyframes": False}
        tasks = ExportProfile.run_config(values, self.FIXES, checks)["tasks"]
        self.assertTrue(tasks["tie_all_keyframes"])
        self.assertNotIn("check_untied_keyframes", tasks)

    def test_every_verifying_check_is_scheduled_after_its_fix(self):
        for fix, check in ExportProfile.VERIFYING_CHECKS.items():
            with self.subTest(fix=fix):
                self.assertIn(fix, ExportProfile.CHECK_DEPENDENCIES[check])

    def test_ignore_groups_is_wrapped_with_its_match_mode(self):
        tasks = ExportProfile.run_config(
            self._values(ignore_groups="temp*"),
            TASKS,
            CHECKS,
            ignore_groups_case_sensitive=True,
        )["tasks"]
        self.assertEqual(
            tasks["ignore_groups"], {"names": "temp*", "case_sensitive": True}
        )

    def test_a_panels_values_become_the_run(self):
        """What the panel read off its rows -- every row, by objectName -- is
        the run (the reading itself is the panel's)."""
        values = {
            "export_visible_objects": "visible",
            "export_data_node": True,
            "texture_output": None,
            "texture_optimize": 2048,
            "ignore_groups": "temp*",
            "smart_bake": False,
            "check_default_materials": True,
            "check_path_length": 260,
            "check_with_dash": False,
        }
        config = ExportProfile.run_config(values, TASKS, CHECKS)
        self.assertEqual(config["export_mode"], "visible")
        self.assertEqual(config["tasks"]["texture_max_size"], 2048)
        self.assertNotIn("smart_bake", config["tasks"])


class TestOutputPath(unittest.TestCase):
    """The Output Filename rule both Scene Exporter panels resolve through."""

    CONTEXT = {
        "name": "hero",
        "scene": "hero",
        "date": "2026-09-13",
        "time": "10-00-00",
        "user": "u",
    }

    def setUp(self):
        import shutil
        import tempfile

        self.tmp = tempfile.mkdtemp(prefix="export_profile_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def touch(self, name):
        open(os.path.join(self.tmp, name), "w").close()

    def resolve(self, pattern, output_format="fbx", **kwargs):
        return ExportProfile.resolve_output_path(
            pattern, self.CONTEXT, self.tmp, output_format=output_format, **kwargs
        )

    def test_a_counter_takes_the_next_version_among_the_files_the_format_ships(self):
        """A GLB-only export leaves no .fbx behind; FBX + GLB versions as one."""
        self.touch("hero_v002.glb")
        self.touch("hero_v005.fbx")
        glb = self.resolve("*_v{n:03d}", "glb")
        self.assertEqual([os.path.basename(p) for p in glb["paths"]], ["hero_v003.glb"])
        # The export path stays .fbx: a GLB-only run names its temp FBX after it.
        self.assertTrue(glb["path"].endswith("hero_v003.fbx"))
        pair = self.resolve("*_v{n:03d}", "fbx_glb")
        self.assertEqual(
            [os.path.basename(p) for p in pair["paths"]],
            ["hero_v006.fbx", "hero_v006.glb"],
        )
        self.assertEqual(pair["n"], 6)

    def test_without_a_counter_nothing_is_scanned_and_the_extension_is_the_format_s(
        self,
    ):
        resolved = self.resolve("asset.fbx", "usd")
        self.assertIsNone(resolved["n"])
        self.assertEqual(os.path.basename(resolved["path"]), "asset.usd")

    def test_a_counter_spec_an_int_cannot_take_numbers_plainly(self):
        resolved = self.resolve("*_v{n:s}")
        self.assertEqual(resolved["stem"], "hero_v1")
        self.assertTrue(resolved["counter_error"])
        levels = [level for level, _ in ExportProfile.naming_report(resolved, {})]
        self.assertIn("error", levels)

    def test_retired_version_and_timestamp_inputs_fold_into_the_same_file(self):
        """A saved Version pattern / Timestamp flag folds into the pattern once,
        at the exporter's entry point (:meth:`ExportProfile.fold_legacy_naming`),
        and the pattern alone resolves to the file they used to make."""
        folded = ExportProfile.fold_legacy_naming(
            "WIP_*", version_format="{stem}_v{n:03d}", timestamp=True
        )
        # the wildcard expands to the CANONICAL name token (ExportProfile.NAME_KEY)
        self.assertEqual(folded, "WIP_{scene}_{date}_{time}_v{n:03d}")
        self.assertEqual(
            self.resolve(folded)["stem"], "WIP_hero_2026-09-13_10-00-00_v001"
        )
        self.assertEqual(ExportProfile.fold_legacy_naming("WIP_*"), "WIP_*")

    def test_the_report_words_each_diagnostic_at_its_level(self):
        resolved = self.resolve("{nope}_a?b_{n}")
        report = ExportProfile.naming_report(resolved, {"name": "", "n": ""})
        self.assertEqual([level for level, _ in report], ["warning"] * 2)
        text = " ".join(message for _, message in report)
        for fragment in ("{nope}", "?"):
            self.assertIn(fragment, text)

    def test_the_version_suffix_strips_what_the_counter_wrote_and_nothing_else(self):
        self.touch("hero_v002.fbx")
        stem = self.resolve("*_v{n:03d}")["stem"]
        strip = ExportProfile.VERSION_SUFFIX_RE.sub
        self.assertEqual((stem, strip("", stem)), ("hero_v003", "hero"))
        self.assertEqual(strip("", "hero_V12"), "hero")
        for kept in ("hero_v003_final", "hero_v", "herov003"):
            with self.subTest(name=kept):
                self.assertEqual(strip("", kept), kept)


class TestCheckRows(unittest.TestCase):
    """Check-row values both exporters read the same way."""

    def test_the_max_texture_size_row_is_megabytes_or_off(self):
        for off in (None, 0, 0.0, "", "OFF", " off ", "abc", -4, float("inf")):
            with self.subTest(value=off):
                self.assertIsNone(ExportProfile.texture_size_limit_bytes(off))
        mib = 1024 * 1024
        self.assertEqual(ExportProfile.texture_size_limit_bytes(16), 16 * mib)
        self.assertEqual(ExportProfile.texture_size_limit_bytes("0.5"), mib // 2)


class TestExportRun(unittest.TestCase):
    """``ExportRun.from_tasks`` -- the per-run modes popped out of the export
    button's dict, once for both DCC exporters. Added: 2026-09-13
    """

    def test_a_version_pattern_warns_and_still_reaches_the_run(self):
        """Retired 2026-09-23: a ``tasks["version"]`` warns (removed in 0.19.0:
        saved data, re-dated from 0.12.0) and is carried on the run for the
        exporter to fold into the name once; a blank one is no request and
        says nothing."""
        import warnings

        with self.assertWarns(DeprecationWarning) as retired:
            run = ExportRun.from_tasks({"version": "{stem}_v{n:03d}"})[0]
        self.assertIn("tasks['version']", str(retired.warning))
        self.assertIn("0.19.0", str(retired.warning))
        self.assertEqual(run.version_format, "{stem}_v{n:03d}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ExportRun.from_tasks({"version": ""})
        self.assertEqual([w for w in caught if w.category is DeprecationWarning], [])

    def test_the_self_contained_flag_is_retired_and_warns_only_when_set(self):
        """Retired 2026-10-05: every FBX embeds its media, so nothing reads
        ``fbx_media_selfcontained``; a caller setting it is told (removed in
        0.15.0) and still gets a run, and a run that never names it is quiet."""
        import warnings

        with self.assertWarns(DeprecationWarning) as retired:
            ExportRun(fbx_media_selfcontained=True)
        self.assertIn("fbx_media_selfcontained", str(retired.warning))
        self.assertIn("0.15.0", str(retired.warning))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ExportRun()
        self.assertEqual([w for w in caught if w.category is DeprecationWarning], [])

    def test_every_mode_key_is_popped_and_no_task_is(self):
        tasks = {key: True for key in ExportRun.MODE_KEYS}
        tasks.update({"smart_bake": True, "check_framerate": "ntsc"})
        with self.assertWarns(DeprecationWarning):  # "version", retired
            run, remaining, _notes = ExportRun.from_tasks(tasks)
        self.assertEqual(set(remaining), {"smart_bake", "check_framerate"})
        self.assertEqual(remaining["check_framerate"], "ntsc")
        self.assertTrue(run.texture_write_back)
        self.assertTrue(run.animation_write_back)
        self.assertTrue(run.verify_deliverables)
        self.assertTrue(run.drop_rig_apparatus)

    def test_the_rig_helper_row_is_an_fbx_mode(self):
        """Exclude Rig Helpers edits the written FBX after the write, so it is a
        mode like Verify The Written File: popped, off for a caller that never
        names it (``tasks=`` is exact), and inert -- with a note -- on a USD
        run, which writes no FBX. Added: 2026-09-19"""
        self.assertFalse(ExportRun.from_tasks({})[0].drop_rig_apparatus)
        run, remaining, notes = ExportRun.from_tasks(
            {"drop_rig_apparatus": True, "output_format": "fbx_glb"}
        )
        self.assertTrue(run.drop_rig_apparatus)
        self.assertEqual((remaining, notes), ({}, []))
        run, _t, notes = ExportRun.from_tasks(
            {"drop_rig_apparatus": True, "output_format": "usd"}
        )
        self.assertFalse(run.drop_rig_apparatus)
        self.assertEqual([level for level, _ in notes], ["info"])
        self.assertIn("Exclude Rig Helpers", notes[0][1])

    def test_the_output_format_and_its_legacy_flag(self):
        self.assertEqual(ExportRun.from_tasks({})[0].output_format, "fbx")
        legacy = ExportRun.from_tasks({"create_glb": True})[0]
        self.assertEqual(legacy.output_format, "fbx_glb")
        self.assertTrue(legacy.create_glb)
        self.assertFalse(legacy.glb_only)
        wins = ExportRun.from_tasks({"create_glb": True, "output_format": "GLB"})[0]
        self.assertTrue(wins.glb_only and wins.create_glb)
        self.assertTrue(ExportRun.from_tasks({"output_format": "usd"})[0].usd)
        run, _tasks, notes = ExportRun.from_tasks({"output_format": "obj"})
        self.assertEqual(run.output_format, "fbx")
        self.assertEqual([level for level, _ in notes], ["warning"])

    def test_the_glb_optimisation_dials(self):
        """Secondary map size, UASTC RDO and the key-reduction bound are modes:
        popped, coerced, None when off -- and every table's OFF is index 0 and
        falsy, so the run config drops it. Added: 2026-09-13"""
        run, remaining, _notes = ExportRun.from_tasks(
            {
                "secondary_max_size": 2048,
                "uastc_rdo": "1.0",
                "glb_key_tolerance": 1e-4,
                "optimize_keys": "extremes",  # the bound rides it (2026-09-14)
                "smart_bake": True,
            }
        )
        self.assertEqual(
            (run.secondary_max_size, run.uastc_rdo, run.glb_key_tolerance),
            (2048, 1.0, 1e-4),
        )
        self.assertEqual(set(remaining), {"smart_bake", "optimize_keys"})
        off = ExportRun.from_tasks(
            {"secondary_max_size": 0, "uastc_rdo": None, "glb_key_tolerance": "x"}
        )[0]
        self.assertEqual(
            (off.secondary_max_size, off.uastc_rdo, off.glb_key_tolerance),
            (None, None, None),
        )
        self.assertIsNone(ExportRun().uastc_rdo)
        for name in (
            "SECONDARY_MAX_SIZE_OPTIONS",
            "UASTC_RDO_OPTIONS",
            "GLB_KEY_REDUCTION_OPTIONS",
        ):
            table = getattr(ExportProfile, name)
            self.assertFalse(list(table.values())[0], f"{name}: OFF is index 0, falsy")

    def test_the_key_reduction_rides_optimize_keys(self):
        """Reduce GLB Keys is folded into Optimize Keys (2026-09-14): FBX2glTF
        resamples every frame, so the Maya-side optimisation never reaches
        the GLB and the tolerance row is the GLB half of ONE choice. With
        Optimize Keys off the bound is inert and says so."""
        run, _t, notes = ExportRun.from_tasks(
            {"optimize_keys": "extremes", "glb_key_tolerance": 1e-4}
        )
        self.assertEqual(run.glb_key_tolerance, 1e-4)
        self.assertEqual(notes, [])
        for tasks in (
            {"glb_key_tolerance": 1e-4},
            {"optimize_keys": None, "glb_key_tolerance": 1e-4},
        ):
            run, _t, notes = ExportRun.from_tasks(tasks)
            self.assertIsNone(run.glb_key_tolerance, tasks)
            self.assertEqual([lvl for lvl, _ in notes], ["info"], tasks)
            self.assertIn("Optimize Keys", notes[0][1])
        first = list(ExportProfile.GLB_KEY_REDUCTION_OPTIONS)[0]
        self.assertEqual(first, "Keep Every Key")
        self.assertEqual(
            list(ExportProfile.SECONDARY_MAX_SIZE_OPTIONS)[0], "Same As Other Maps"
        )

    def test_a_glb_only_run_skips_the_fbx_key_hygiene(self):
        """With no FBX deliverable the FBX is a temp intermediate the converter
        resamples per frame: Optimize Keys, snap, tie and the two key checks
        change nothing the GLB can see and cost minutes, so they are dropped
        with a note -- Optimize Keys still reaches the GLB as its reduction."""
        hygiene = {
            "optimize_keys": "extremes",
            "snap_keys_to_frame": True,
            "tie_all_keyframes": True,
            "check_floating_point_keys": True,
            "check_untied_keyframes": True,
        }
        run, remaining, notes = ExportRun.from_tasks(
            {
                "output_format": "glb",
                "glb_key_tolerance": 1e-4,
                "smart_bake": True,
                **hygiene,
            }
        )
        self.assertEqual(set(remaining), {"smart_bake"})
        self.assertEqual(run.glb_key_tolerance, 1e-4, "read before the pop")
        self.assertFalse(run.optimize_keys_level, "SmartBake's own pass rides it")
        self.assertEqual([lvl for lvl, _ in notes], ["info"])
        for key in hygiene:
            self.assertIn(key, notes[0][1])
        self.assertEqual(ExportRun.FBX_KEY_HYGIENE, tuple(hygiene))
        run, remaining, notes = ExportRun.from_tasks(
            {"output_format": "fbx_glb", "glb_key_tolerance": 1e-4, **hygiene}
        )
        self.assertEqual(set(remaining), set(hygiene), "the FBX ships: all kept")
        self.assertEqual(run.optimize_keys_level, "extremes")
        self.assertEqual(notes, [])
        _run, remaining, notes = ExportRun.from_tasks(
            {"output_format": "glb", "smart_bake": True}
        )
        self.assertEqual(
            (set(remaining), notes),
            ({"smart_bake"}, []),
            "nothing to skip, nothing said",
        )

    def test_the_texture_file_type_dial(self):
        known = ExportProfile.texture_file_type_options().values()
        run, _t, notes = ExportRun.from_tasks(
            {"glb_texture_format": ".PNG", "output_format": "glb"}, known
        )
        self.assertEqual(run.texture_file_type, "png")
        self.assertEqual([level for level, _ in notes], ["debug"])
        run = ExportRun.from_tasks(
            {"texture_file_type": "jpg", "glb_texture_format": "png"}, known
        )[0]
        self.assertEqual(run.texture_file_type, "jpg", "the new key wins")
        _run, _t, notes = ExportRun.from_tasks({"texture_file_type": "xyz"}, known)
        self.assertEqual([level for level, _ in notes], ["error"])
        self.assertIn("xyz", notes[0][1])
        ungated = ExportRun.from_tasks({"texture_file_type": "xyz"})[0]
        self.assertEqual(ungated.texture_file_type, "xyz", "no dial, no gate")

    def test_ktx2_needs_a_glb_to_ride_in(self):
        both = ExportRun.from_tasks(
            {
                "texture_file_type": ExportRun.KTX2_WITH_FALLBACK,
                "output_format": "fbx_glb",
            }
        )[0]
        self.assertEqual((both.texture_file_type, both.ktx2_fallback), ("ktx2", True))
        alone = ExportRun.from_tasks(
            {"texture_file_type": "KTX2", "output_format": "glb"}
        )[0]
        self.assertEqual(
            (alone.texture_file_type, alone.ktx2_fallback), ("ktx2", False)
        )
        run, _t, notes = ExportRun.from_tasks(
            {"texture_file_type": ExportRun.KTX2_WITH_FALLBACK, "output_format": "fbx"}
        )
        self.assertIsNone(run.texture_file_type)
        self.assertFalse(run.ktx2_fallback)
        self.assertEqual([level for level, _ in notes], ["info"])

    def test_the_write_back_flags_and_their_legacy_spelling(self):
        self.assertTrue(
            ExportRun.from_tasks({"optimize_textures_write_back": 1})[
                0
            ].texture_write_back
        )
        run, remaining, _n = ExportRun.from_tasks(
            {"texture_write_back": 0, "optimize_textures_write_back": 1}
        )
        self.assertFalse(run.texture_write_back, "the new key wins")
        self.assertNotIn("optimize_textures_write_back", remaining)
        self.assertFalse(ExportRun.from_tasks({})[0].animation_write_back)

    def test_the_texture_pass_modes_read_the_real_tasks(self):
        run, remaining, _n = ExportRun.from_tasks(
            {"optimize_textures": True, "texture_max_size": 2048}
        )
        self.assertTrue(run.optimize_textures)
        self.assertEqual(run.texture_max_size, 2048)
        self.assertIsNone(run.texture_template)
        self.assertIn("optimize_textures", remaining, "a real task is read, not popped")
        self.assertNotIn("texture_max_size", remaining)
        run = ExportRun.from_tasks({"optimize_textures": "glTF 2.0"})[0]
        self.assertEqual(run.texture_template, "glTF 2.0")
        run = ExportRun.from_tasks(
            {"convert_textures": "Unity", "optimize_textures": "glTF 2.0"}
        )[0]
        self.assertEqual(
            run.texture_template, "Unity", "the conversion's template wins"
        )

    def test_the_task_derived_modes_and_a_resumed_subset(self):
        full = {"optimize_keys": "extremes", "convert_to_relative_paths": True}
        run = ExportRun.from_tasks(full)[0]
        self.assertEqual(run.optimize_keys_level, "extremes")
        self.assertTrue(run.relative_paths)
        # A run built from the full dict keeps its modes; only with_tasks
        # re-derives, and an override's resume never calls it.
        self.assertEqual(
            run.with_tasks({"smart_bake": True}).optimize_keys_level, False
        )
        self.assertEqual(run.optimize_keys_level, "extremes")

    def test_the_clip_mode_resolver_is_one_copy_for_both_exporters(self):
        """A pre-combo preset's boolean still names the deliverable it meant;
        an unknown mode is refused, named."""
        self.assertEqual(ExportRun.clip_mode(True), "both")
        self.assertEqual(ExportRun.clip_mode(False), "full")
        self.assertEqual(ExportRun.clip_mode(None), "full")
        self.assertEqual(ExportRun.clip_mode(" Shots "), "shots")
        with self.assertRaises(ValueError):
            ExportRun.clip_mode("everything")

    def test_splits_takes_is_a_shot_bearing_clips_row(self):
        run = ExportRun.from_tasks({})[0]
        for value, expected in (
            ("shots", True),
            ("both", True),
            (True, True),
            ("full", False),
            (False, False),
            ("everything", False),  # the task raises on it, naming it
        ):
            self.assertIs(
                run.with_tasks({"apply_declared_takes": value}).splits_takes,
                expected,
                value,
            )
        self.assertFalse(run.with_tasks({}).splits_takes, "the row off splits nothing")

    def test_the_run_is_a_frozen_value_object(self):
        run = ExportRun.from_tasks({"output_format": "glb"})[0]
        with self.assertRaises(Exception):
            run.output_format = "fbx"  # type: ignore[misc]
        resolved = run.replace(export_path="C:/out/asset.fbx", versioned=True)
        self.assertEqual(resolved.export_path, "C:/out/asset.fbx")
        self.assertTrue(resolved.versioned and resolved.glb_only)
        self.assertEqual(run.export_path, "", "the original is untouched")


class TestGlbTextureParams(unittest.TestCase):
    """The Scene Exporter's texture rows, resolved to a GLB's texture pass ONCE.

    Both DCC exporters carried a private copy of this resolution, and the WebXR
    preview carried none: it downsized every push to the web ceiling whatever
    the export was set to. One method on the run is what both exporters and the
    preview now call, so the same rows make the same texture pass. Added: 2026-09-21
    """

    KNOWN = ExportProfile.texture_file_type_options().values()

    def _run(self, **tasks):
        return ExportRun.from_tasks({"output_format": "glb", **tasks}, self.KNOWN)[0]

    def test_untouched_rows_keep_every_pixel_in_the_web_container(self):
        """Optimize Textures OFF is the rows' default, and OFF resizes nothing:
        the GLB takes the web container and keeps each map's own resolution.
        Until 2026-09-21 an OFF row took the web ceiling (2048 px), a resize
        under a setting that reads as "no optimization"."""
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        self.assertEqual(
            self._run().glb_texture_params(),
            MeshConvert.web_delivery_texture_params(max_size=0),
        )

    def test_each_row_overrides_only_its_own_half(self):
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        policy = MeshConvert.web_delivery_texture_params()
        typed = self._run(texture_file_type="jpg").glb_texture_params()
        self.assertEqual(typed["image_format"], "JPEG", "an extension, as Pillow's id")
        self.assertEqual(typed["max_size"], 0, "Optimize Textures is still OFF")
        capped = self._run(
            optimize_textures=True, texture_max_size=4096
        ).glb_texture_params()
        self.assertEqual(capped["max_size"], 4096)
        self.assertEqual(capped["image_format"], policy["image_format"])
        dials = self._run(secondary_max_size=1024, uastc_rdo=1.0).glb_texture_params()
        self.assertEqual((dials["secondary_max_size"], dials["uastc_rdo"]), (1024, 1.0))

    def test_a_ceiling_only_counts_while_the_pass_is_on(self):
        """OFF keeps every pixel, whatever size the dial last held. With the pass
        on, a row naming no ceiling (a plain Optimize) takes the web policy's,
        and Optimize + Max N caps at N."""
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        for run, expected in (
            (self._run(), 0),
            (self._run(texture_max_size=4096), 0),  # a size with the pass off: inert
            (self._run(optimize_textures=True), MeshConvert.WEB_DELIVERY_MAX_SIZE),
            (self._run(optimize_textures=True, texture_max_size=4096), 4096),
        ):
            with self.subTest(run=run):
                self.assertEqual(run.glb_texture_params()["max_size"], expected)
        # The dial's own ceiling is still readable (the size check's remedy
        # names it); only the texture pass gates it on the pass being on.
        self.assertEqual(self._run(texture_max_size=4096).glb_max_size(), 4096)
        self.assertEqual(self._run(optimize_textures=True).glb_max_size(), 0)

    def test_the_template_budget_reads_the_templates_own_ceiling(self):
        from pythontk.core_utils.engines.textures.map_optimizer import MapOptimizer
        from pythontk.core_utils.engines.textures.output_template import (
            OutputTemplates,
        )

        budgeted = next(
            name
            for name, _description in OutputTemplates.profile_choices()
            if OutputTemplates.budget(name).max_size
        )
        run = self._run(
            optimize_textures=budgeted,
            texture_max_size=MapOptimizer.SIZE_CLAMP_TEMPLATE,
        )
        self.assertEqual(run.glb_max_size(), OutputTemplates.budget(budgeted).max_size)
        self.assertEqual(
            run.glb_texture_params()["max_size"],
            OutputTemplates.budget(budgeted).max_size,
        )

    def test_a_container_a_glb_cannot_carry_takes_the_policys_and_says_so(self):
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        log = logging.getLogger("test_glb_texture_params")
        with self.assertLogs(log, level="INFO") as logged:
            params = self._run(texture_file_type="tga").glb_texture_params(logger=log)
        self.assertEqual(params["image_format"], MeshConvert.WEB_DELIVERY_FORMAT)
        self.assertTrue(any("TGA" in line for line in logged.output), logged.output)

    def test_ktx2_with_a_core_readable_twin(self):
        params = self._run(
            texture_file_type=ExportRun.KTX2_WITH_FALLBACK
        ).glb_texture_params()
        self.assertEqual(
            (params["image_format"], params["ktx2_fallback"]), ("KTX2", True)
        )
        alone = self._run(texture_file_type="ktx2").glb_texture_params()
        self.assertEqual(
            (alone["image_format"], alone["ktx2_fallback"]), ("KTX2", False)
        )


class TestGlbOnlyProducer(unittest.TestCase):
    """A producer that makes the GLB and nothing else -- the WebXR preview --
    offers the Scene Exporter's GLB rows (the texture pass) and resolves them
    identically. Added: 2026-09-21
    """

    #: The Scene Exporter panel's GLB rows, as a task table.
    ROWS = {
        "texture_file_type": {
            "widget_type": "ComboBox",
            "add": ExportProfile.texture_file_type_options(),
        },
        "optimize_textures": {
            "widget_type": "ComboBox",
            "object_name": "texture_optimize",
            "add": ExportProfile.optimize_textures_options(),
        },
        "secondary_max_size": {
            "widget_type": "ComboBox",
            "add": ExportProfile.SECONDARY_MAX_SIZE_OPTIONS,
        },
        "uastc_rdo": {
            "widget_type": "ComboBox",
            "add": ExportProfile.UASTC_RDO_OPTIONS,
        },
    }

    def _exported(self, values):
        """What the Scene Exporter's GLB gets from *values*: the export
        button's read (``run_config``), the run's parse, the run's resolution."""
        widget_values = {
            ExportProfile.widget_key(name, spec): values[name]
            for name, spec in self.ROWS.items()
            if name in values
        }
        tasks = ExportProfile.run_config(widget_values, self.ROWS, {})["tasks"]
        tasks["output_format"] = "glb"
        run = ExportRun.from_tasks(
            tasks, ExportProfile.texture_file_type_options().values()
        )[0]
        return run.glb_texture_params(), run.rendering

    def _previewed(self, values):
        run, notes = ExportRun.for_glb(values)
        self.assertNotIn("error", [level for level, _ in notes], notes)
        return run.glb_texture_params(), run.rendering

    def test_every_offered_setting_resolves_as_the_export_does(self):
        """The whole point: the same rows make the same texture pass and publish
        the same lighting recipe (the recipe's own, from both). Every entry the
        preview offers, row by row, against the exporter's own chain."""
        options = ExportProfile.glb_options()
        for row, table in options.items():
            for label, value in table.items():
                with self.subTest(row=row, choice=label):
                    self.assertEqual(
                        self._previewed({row: value}), self._exported({row: value})
                    )
        combined = {
            "texture_file_type": ExportRun.KTX2_WITH_FALLBACK,
            "optimize_textures": 4096,
            "secondary_max_size": 1024,
            "uastc_rdo": 1.0,
        }
        self.assertEqual(self._previewed(combined), self._exported(combined))

    def test_the_rows_are_the_exporters_own_tables(self):
        """Same labels and values, from the same tables, minus what a GLB-only
        producer cannot honour: a container glTF cannot embed (it would only
        ever mean the web default), and the Template Budget entry (its budget
        comes from the export's Texture Template, whose material conversion a
        GLB-only producer does not run)."""
        from pythontk.core_utils.engines.textures.map_optimizer import MapOptimizer
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        options = ExportProfile.glb_options()
        self.assertEqual(list(options), list(ExportProfile.GLB_ROWS))
        types = options["texture_file_type"]
        self.assertEqual(list(types.items())[0], ("Original", ""))
        self.assertIn(ExportRun.KTX2_WITH_FALLBACK, types.values())
        for label, value in types.items():
            if value and value != ExportRun.KTX2_WITH_FALLBACK:
                self.assertIn(value, MeshConvert.GLB_IMAGE_FORMATS, label)
        self.assertNotIn("tga", types.values())
        full = ExportProfile.optimize_textures_options()
        self.assertEqual(
            options["optimize_textures"],
            {k: v for k, v in full.items() if v != MapOptimizer.SIZE_CLAMP_TEMPLATE},
        )
        self.assertEqual(
            options["secondary_max_size"], ExportProfile.SECONDARY_MAX_SIZE_OPTIONS
        )
        self.assertEqual(options["uastc_rdo"], ExportProfile.UASTC_RDO_OPTIONS)
        for row in ExportProfile.GLB_TEXTURE_ROWS:
            self.assertFalse(
                list(options[row].values())[0], f"{row}: index 0 is the off state"
            )

    def test_the_optimize_combo_is_taken_apart_in_one_place(self):
        self.assertEqual(ExportProfile.optimize_textures_tasks(0), {})
        self.assertEqual(ExportProfile.optimize_textures_tasks(None), {})
        self.assertEqual(
            ExportProfile.optimize_textures_tasks(True), {"optimize_textures": True}
        )
        self.assertEqual(
            ExportProfile.optimize_textures_tasks(4096),
            {"optimize_textures": True, "texture_max_size": 4096},
        )
        self.assertEqual(
            ExportProfile.optimize_textures_tasks(1024, "glTF 2.0"),
            {"optimize_textures": "glTF 2.0", "texture_max_size": 1024},
        )

    def test_an_absent_row_is_the_exporters_default(self):
        self.assertEqual(self._previewed({}), self._exported({}))

    def test_the_rows_start_where_the_exporter_starts_them(self):
        """What a mirroring panel defaults each row to: the off state -- so an
        untouched mirror builds the texture pass and publishes the recipe an
        absent row does."""
        from pythontk.core_utils.engines.scene_export.handoff_record import (
            HandoffRecord,
        )

        defaults = ExportProfile.glb_defaults()
        self.assertEqual(list(defaults), list(ExportProfile.GLB_ROWS))
        (untouched, chosen), (absent, none) = (
            self._previewed(defaults),
            self._previewed({}),
        )
        self.assertEqual(untouched, absent)
        self.assertEqual(
            HandoffRecord.rendering_policy(chosen),
            HandoffRecord.rendering_policy(none),
        )

    def test_an_unknown_row_is_said_rather_than_dropped(self):
        """A misspelt key would otherwise leave the images at the policy with
        nothing saying why the setting did nothing."""
        _run, notes = ExportRun.for_glb({"max_size": 4096})
        self.assertEqual([level for level, _ in notes], ["warning"])
        self.assertIn("max_size", notes[0][1])

    def test_an_unknown_container_is_refused(self):
        _run, notes = ExportRun.for_glb({"texture_file_type": "xyz"})
        self.assertIn("error", [level for level, _ in notes])


class TestBakedReflections(unittest.TestCase):
    """The Baked Reflections row, retired 2026-10-19 (removed in 0.17.0).

    It set how strongly a lightmapped material reflects the viewer's studio.
    Its default, the lighting recipe's own level, was already the right answer
    (each texel's reflection is scaled by its own bake, and a deliverable that
    carries its reflection probe reflects that whole), so no panel offers it:
    the deliverable publishes the recipe unchanged. Added: 2026-09-21
    """

    def test_no_row_decides_the_lighting_any_more(self):
        self.assertEqual(ExportProfile.GLB_LIGHTING_ROWS, {})
        self.assertNotIn("baked_reflections", ExportProfile.GLB_ROWS)
        self.assertNotIn("baked_reflections", ExportProfile.glb_options())
        self.assertNotIn("baked_reflections", ExportProfile.glb_defaults())

    def test_an_older_panels_row_is_dropped_with_a_note(self):
        """A DCC package from before the retirement still sends its row; the
        deliverable publishes the recipe whatever it says, and the log says so."""
        from pythontk.core_utils.engines.scene_export.handoff_record import (
            HandoffRecord,
        )

        for value in ("off", "half", 0.4):
            with self.subTest(value=value):
                run, tasks, notes = ExportRun.from_tasks({"baked_reflections": value})
                self.assertIsNone(run.baked_reflections)
                self.assertEqual(run.rendering, {})
                self.assertEqual(
                    HandoffRecord.rendering_policy(run.rendering),
                    HandoffRecord.RENDERING_POLICY,
                )
                self.assertNotIn("baked_reflections", tasks)
                self.assertEqual([level for level, _ in notes], ["info"])
                self.assertIn("Baked Reflections", notes[0][1])
        _run, _tasks, notes = ExportRun.from_tasks({})
        self.assertEqual(notes, [])

    def test_a_caller_setting_the_level_is_told_and_still_published(self):
        """The field warns until its removal and, until then, still reaches both
        carriers: the GLB's handoff block and the FBX's handoff record."""
        from pythontk.core_utils.engines.scene_export.handoff_record import (
            HandoffRecord,
        )
        from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords

        with self.assertWarns(DeprecationWarning) as retired:
            run = ExportRun(baked_reflections=0.5)
        self.assertIn("baked_reflections", str(retired.warning))
        self.assertIn("0.17.0", str(retired.warning))
        # Published where the file uses it: a lightmapped material's level,
        # in a file that carries lightmaps.
        glb = HandoffRecord.block(
            "glb",
            [SceneRecords.LIGHTMAPS.key],
            {"application": "test"},
            rendering=run.rendering,
        )
        block = HandoffRecord.block(
            "fbx",
            HandoffRecord.carried([SceneRecords.LIGHTMAPS.key]),
            rendering=run.rendering,
        )
        for recipe in (glb["rendering"], block["rendering"]):
            self.assertEqual(recipe["lightmappedMaterials"]["envMapIntensity"], 0.5)
            self.assertEqual(recipe["lightmappedMaterials"]["envMapTerms"], "specular")

    def test_the_retired_tables_still_answer_an_older_panel(self):
        """An older mayatk / blendertk builds the row from these at import."""
        from pythontk.core_utils.engines.scene_export.handoff_record import (
            HandoffRecord,
        )

        with self.assertWarns(DeprecationWarning):
            options = ExportProfile.BAKED_REFLECTIONS_OPTIONS
        with self.assertWarns(DeprecationWarning):
            levels = ExportRun.BAKED_REFLECTION_LEVELS
        self.assertEqual(list(options.values()), list(levels))
        with self.assertWarns(DeprecationWarning):
            default = ExportProfile.baked_reflections_default()
        self.assertEqual(
            levels[default],
            HandoffRecord.RENDERING_POLICY["lightmappedMaterials"]["envMapIntensity"],
        )
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(ExportRun.baked_reflection_level("half"), 0.5)

    def test_reading_a_retired_table_warns_and_returns_it_unchanged(self):
        """REGRESSION (2026-10-19): both tables said "removed in 0.17.0" in a
        comment and warned no one who read them, from the class or an
        instance."""
        tables = {
            "BAKED_REFLECTIONS_OPTIONS": (
                ExportProfile,
                {
                    "Off (Pure Bake)": "off",
                    "Quarter": "quarter",
                    "Half": "half",
                    "Full": "full",
                },
            ),
            "BAKED_REFLECTION_LEVELS": (
                ExportRun,
                {"off": 0.0, "quarter": 0.25, "half": 0.5, "full": 1.0},
            ),
        }
        for name, (owner, value) in tables.items():
            for reader in (owner, owner()):
                with self.subTest(name=name, reader=reader):
                    with self.assertWarns(DeprecationWarning) as caught:
                        read = getattr(reader, name)
                    self.assertEqual(read, value)
                    self.assertIn(f"{owner.__name__}.{name}", str(caught.warning))
                    self.assertIn("0.17.0", str(caught.warning))

    def test_the_rules_that_still_read_the_tables_do_not_warn(self):
        """The retired-row note reads them for an older preset; a notice the
        caller cannot act on is noise."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            notes = ExportProfile._retired_row_notes(
                {"baked_reflections": "Quarter"}, {}, {}
            )
        self.assertEqual([str(w.message) for w in caught], [])
        self.assertIn("Quarter", notes[0][1])


class TestRetiredPresetRows(unittest.TestCase):
    """A preset a released panel saved still names the rows retired
    2026-10-19, and a load skips a key no widget takes -- so a preset that
    asked for one of them would stop getting it with nothing said.
    REGRESSION (2026-10-19): the released default (Snap Keys To Frame off,
    Check For Floating Point Keys on) loaded with the check silently gone, and
    a user preset's Baked Reflections "Quarter" silently shipped Full.
    """

    #: The fix rows a panel offers, as their definitions spell them.
    FIXES = {
        "conform_shape_names": {"widget_type": "QCheckBox", "setText": "Fix Names"},
        "snap_keys_to_frame": {"widget_type": "QCheckBox", "setText": "Snap Keys"},
        "tie_all_keyframes": {"widget_type": "QCheckBox", "setText": "Tie Keys"},
    }

    def notes(self, stored, tasks=None, checks=None):
        return ExportProfile._retired_row_notes(
            stored, self.FIXES if tasks is None else tasks, checks or {}
        )

    def test_a_check_its_fix_leaves_off_is_said_lost(self):
        notes = self.notes(
            {"snap_keys_to_frame": False, "check_floating_point_keys": True}
        )
        self.assertEqual([level for level, _ in notes], ["warning"])
        self.assertIn("Floating Point Keys", notes[0][1])
        self.assertIn("Snap Keys", notes[0][1])

    def test_a_pair_that_agrees_says_nothing(self):
        for on in (True, False):
            with self.subTest(on=on):
                stored = {"tie_all_keyframes": on, "check_untied_keyframes": on}
                self.assertEqual(self.notes(stored), [])

    def test_a_fix_whose_check_was_off_is_now_checked(self):
        notes = self.notes({"tie_all_keyframes": True, "check_untied_keyframes": False})
        self.assertEqual([level for level, _ in notes], ["info"])
        self.assertIn("Untied Keyframes", notes[0][1])

    def test_a_row_the_panel_still_offers_or_never_had_is_not_retired(self):
        stored = {"tie_all_keyframes": False, "check_untied_keyframes": True}
        offered = {"check_untied_keyframes": {"widget_type": "QCheckBox"}}
        self.assertEqual(self.notes(stored, checks=offered), [])
        # blendertk has no Fix Mangled Names: its check was never armed there.
        stored = {"conform_shape_names": False, "check_mangled_names": True}
        self.assertEqual(self.notes(stored, tasks={}), [])

    def test_a_baked_reflections_level_other_than_the_recipes_is_said_dropped(self):
        notes = self.notes({"baked_reflections": "Quarter"})
        self.assertEqual([level for level, _ in notes], ["info"])
        self.assertIn("Quarter", notes[0][1])
        # The recipe's own level (Full) is what ships anyway: nothing to say.
        self.assertEqual(self.notes({"baked_reflections": "Full"}), [])
        self.assertEqual(self.notes({"baked_reflections": "full"}), [])

    def test_a_writability_check_turned_off_is_said_to_run(self):
        notes = self.notes({"check_output_writable": False})
        self.assertEqual([level for level, _ in notes], ["info"])
        self.assertEqual(self.notes({"check_output_writable": True}), [])

    def test_a_current_preset_says_nothing(self):
        self.assertEqual(self.notes({}), [])
        self.assertEqual(self.notes({"tie_all_keyframes": True}), [])

    def test_the_retired_keys_are_the_rows_the_notes_speak_for(self):
        """What a panel's saved state drops once said (2026-10-19): a check
        whose fix the panel offers and whose own row it does not, the Baked
        Reflections row, and the writability check -- each only while the
        panel does not build it."""
        self.assertEqual(
            ExportProfile._retired_row_keys(self.FIXES, {}),
            (
                "check_mangled_names",
                "check_floating_point_keys",
                "check_untied_keyframes",
                "baked_reflections",
                "check_output_writable",
            ),
        )
        offered = {"check_untied_keyframes": {}, "check_output_writable": {}}
        self.assertEqual(
            ExportProfile._retired_row_keys(
                {**self.FIXES, "baked_reflections": {}}, offered
            ),
            ("check_mangled_names", "check_floating_point_keys"),
        )


class TestRunTables(unittest.TestCase):
    """The shared task order / check map, scoped to what a manager implements.
    Added: 2026-09-13
    """

    class _Partial:
        """A manager missing three tasks and two checks of the canonical set."""

        def set_linear_unit(self):
            pass

        def ignore_groups(self):
            pass

        def smart_bake(self):
            pass

        def optimize_keys(self):
            pass

        def check_framerate(self):
            pass

        def check_root_default_transforms(self):
            pass

        def check_untied_keyframes(self):
            pass

        check_definitions = {}  # a property in the real managers: not a check

    def test_scoped_tables_keep_the_canonical_order_minus_the_absent(self):
        manager = ExportProfile.scoped_tables(type("M", (self._Partial,), {}))
        self.assertEqual(
            manager.TASK_ORDER,
            ["set_linear_unit", "ignore_groups", "smart_bake", "optimize_keys"],
        )
        self.assertEqual(
            manager.CHECK_DEPENDENCIES,
            {
                "check_framerate": ("ignore_groups", "smart_bake"),
                "check_root_default_transforms": ("set_linear_unit", "ignore_groups"),
                "check_untied_keyframes": (
                    "ignore_groups",
                    "smart_bake",
                    "optimize_keys",
                ),
            },
        )

    def test_unimplemented_names_the_gap_in_table_order(self):
        gaps = ExportProfile.unimplemented(self._Partial)
        self.assertEqual(
            gaps["tasks"][:3], ["set_workspace", "exclude_hdr", "conform_shape_names"]
        )
        self.assertNotIn("smart_bake", gaps["tasks"])
        self.assertIn("check_output_writable", gaps["checks"])
        self.assertNotIn("check_framerate", gaps["checks"])

    def test_every_dependency_names_a_task_in_the_order(self):
        known = set(ExportProfile.TASK_ORDER)
        for check, tasks in ExportProfile.CHECK_DEPENDENCIES.items():
            self.assertTrue(check.startswith("check_"), check)
            self.assertFalse(set(tasks) - known, (check, set(tasks) - known))

    def test_the_combo_tables_keep_their_index_contracts(self):
        optimize = list(ExportProfile.optimize_textures_options().items())
        self.assertEqual(optimize[0], ("OFF", 0))
        self.assertEqual(optimize[-1][0], "Optimize + Template Budget")
        types = list(ExportProfile.texture_file_type_options().items())
        self.assertEqual(types[0][0], "Original")
        self.assertFalse(types[0][1])
        self.assertEqual(types[-1][1], ExportRun.KTX2_WITH_FALLBACK)
        rates = list(ExportProfile.frame_rate_options().items())
        self.assertEqual(rates[0], ("OFF", None))
        self.assertTrue(all("fps" in label for label, _ in rates[1:]), rates)
        self.assertEqual(list(ExportProfile.BAKE_RANGE_OPTIONS.values())[0], None)
        self.assertEqual(
            list(ExportProfile.ANIMATION_CLIPS_OPTIONS.values())[-1], "both"
        )


if __name__ == "__main__":
    unittest.main()
