#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk HotkeyUtils.

Run with:
    python -m pytest test_hotkey_utils.py -v
    python test_hotkey_utils.py
"""
import unittest

from pythontk.str_utils.hotkey_utils import HotkeyUtils

from conftest import BaseTestCase


class HotkeyUtilsTest(BaseTestCase):
    """Tests for HotkeyUtils class."""

    def test_parse_key_plain(self):
        self.assertEqual(HotkeyUtils.parse_key("f"), (False, False, False, "f"))

    def test_parse_key_modifiers(self):
        self.assertEqual(HotkeyUtils.parse_key("ctl+sht+i"), (True, False, True, "i"))
        self.assertEqual(HotkeyUtils.parse_key("alt+F3"), (False, True, False, "F3"))

    def test_parse_key_case_insensitive_modifiers(self):
        self.assertEqual(HotkeyUtils.parse_key("CTL+ALT+q"), (True, True, False, "q"))

    def test_qt_sequence_to_key_round_trip(self):
        self.assertEqual(HotkeyUtils.qt_sequence_to_key("Ctrl+Shift+I"), "ctl+sht+i")
        self.assertEqual(HotkeyUtils.key_to_qt_sequence("ctl+sht+i"), "Ctrl+Shift+I")

    def test_qt_sequence_to_key_meta_and_cmd_fold_to_ctl(self):
        self.assertEqual(HotkeyUtils.qt_sequence_to_key("Meta+F3"), "ctl+F3")
        self.assertEqual(HotkeyUtils.qt_sequence_to_key("Cmd+F3"), "ctl+F3")

    def test_qt_sequence_to_key_no_non_modifier_key(self):
        self.assertEqual(HotkeyUtils.qt_sequence_to_key("Ctrl"), "")
        self.assertEqual(HotkeyUtils.qt_sequence_to_key(""), "")

    def test_key_to_qt_sequence_empty(self):
        self.assertEqual(HotkeyUtils.key_to_qt_sequence(""), "")
        self.assertEqual(HotkeyUtils.key_to_qt_sequence(None), "")

    def test_key_to_qt_sequence_multi_char_key_not_uppercased_oddly(self):
        # multi-char tokens (function keys) pass through as-is (already canonical case)
        self.assertEqual(HotkeyUtils.key_to_qt_sequence("F3"), "F3")

    def test_humanize_label_basic(self):
        self.assertEqual(HotkeyUtils.humanize_label("back_face_culling"), "Back Face Culling")

    def test_humanize_label_with_prefix(self):
        self.assertEqual(
            HotkeyUtils.humanize_label("m_back_face_culling", prefix="m_"),
            "Back Face Culling",
        )

    def test_humanize_label_acronyms(self):
        acronyms = {"uv": "UV", "id": "ID"}
        self.assertEqual(
            HotkeyUtils.humanize_label("m_toggle_uv_select_type", prefix="m_", acronyms=acronyms),
            "Toggle UV Select Type",
        )
        self.assertEqual(
            HotkeyUtils.humanize_label("m_object_id", prefix="m_", acronyms=acronyms),
            "Object ID",
        )

    def test_humanize_label_preserves_existing_uppercase_acronym(self):
        # A word already upper-case in the source name (len > 1) is preserved even
        # without an explicit acronyms entry for it.
        self.assertEqual(
            HotkeyUtils.humanize_label("m_toggle_UV_select_type", prefix="m_"),
            "Toggle UV Select Type",
        )

    def test_mod_order_and_qt_mod_map_are_stable(self):
        self.assertEqual(HotkeyUtils.MOD_ORDER, ("ctl", "alt", "sht"))
        self.assertEqual(HotkeyUtils.QT_MOD_MAP["meta"], "ctl")
        self.assertEqual(HotkeyUtils.QT_MOD_MAP["cmd"], "ctl")


if __name__ == "__main__":
    unittest.main()
