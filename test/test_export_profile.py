# !/usr/bin/python
# coding=utf-8
"""ExportProfile -- the Scene Exporter panels' export-button contract, once."""

import unittest

from pythontk.core_utils.export_profile import ExportProfile


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

    def test_value_method_follows_the_export_buttons_rule(self):
        self.assertEqual(ExportProfile.value_method({}), "isChecked")
        self.assertEqual(
            ExportProfile.value_method({"widget_type": "ComboBox"}), "currentData"
        )
        self.assertEqual(
            ExportProfile.value_method(CHECKS["check_path_length"]), "value"
        )


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

    def test_override_drops_every_check_but_keeps_the_tasks(self):
        config = ExportProfile.run_config(
            self._values(), TASKS, CHECKS, override_checks=True
        )
        self.assertNotIn("check_default_materials", config["tasks"])
        self.assertNotIn("check_path_length", config["tasks"])
        self.assertTrue(config["tasks"]["export_data_node"])

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

    def test_live_widgets_are_read_by_the_buttons_rule(self):
        """Checkbox -> isChecked, combo -> currentData, else the definition's method."""

        class _Combo:
            def __init__(self, data):
                self._data = data

            def currentData(self):
                return self._data

        class _Check:
            def __init__(self, on):
                self._on = on

            def isChecked(self):
                return self._on

        class _Spin:
            def value(self):
                return 260

        class _Text:
            def text(self):
                return "temp*"

        widgets = {
            "export_visible_objects": _Combo("visible"),
            "export_data_node": _Check(True),
            "texture_output": _Combo(None),
            "texture_optimize": _Combo(2048),
            "ignore_groups": _Text(),
            "smart_bake": _Check(False),
            "check_default_materials": _Check(True),
            "check_path_length": _Spin(),
            "check_with_dash": _Check(False),
        }
        values = ExportProfile.read_values(widgets, TASKS, CHECKS)
        self.assertEqual(
            values,
            {
                "export_visible_objects": "visible",
                "export_data_node": True,
                "texture_output": None,
                "texture_optimize": 2048,
                "ignore_groups": "temp*",
                "smart_bake": False,
                "check_default_materials": True,
                "check_path_length": 260,
                "check_with_dash": False,
            },
        )
        config = ExportProfile.run_config(values, TASKS, CHECKS)
        self.assertEqual(config["export_mode"], "visible")
        self.assertEqual(config["tasks"]["texture_max_size"], 2048)
        self.assertNotIn("smart_bake", config["tasks"])


if __name__ == "__main__":
    unittest.main()
