# !/usr/bin/python
# coding=utf-8
"""SceneRecords' host-facing rules: row resolution and stack-shaped records.

What each host wrote by hand until 2026-10-29 -- ``FbxUtils``' row
resolution (``_resolve_row``, ``producers``, ``stagers``; one rule now, logged
through the host's own logger) and ``BakeSessionStore``'s ``peek`` / ``pop``.
The rows themselves (``producer_rows``, ``stager_rows``) are pinned in
``test_scene_records.py``.
"""

import logging
import unittest
from unittest import mock

from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords as SR

#: A class every interpreter has, so a row resolves without a host.
ROW = ("json", "JSONDecoder", "decode", "raw_decode")


class TestResolveRow(unittest.TestCase):
    def setUp(self):
        self.log = logging.getLogger("test_scene_record_rows")

    def test_a_row_resolves_to_its_class_methods(self):
        import json

        self.assertEqual(
            SR.resolve_row("Stager 'x'", *ROW, log=self.log),
            (json.JSONDecoder.decode, json.JSONDecoder.raw_decode),
        )

    def test_a_column_left_none_is_a_quiet_none(self):
        with mock.patch.object(self.log, "warning") as warning:
            resolved = SR.resolve_row("Stager 'x'", "json", "JSONDecoder", "decode", None, log=self.log)
        self.assertIsNone(resolved[1])
        warning.assert_not_called()

    def test_a_misspelt_method_warns_through_the_hosts_logger(self):
        with self.assertLogs(self.log, level="WARNING") as seen:
            resolved = SR.resolve_row("Producer for 'x'", "json", "JSONDecoder", "decod", log=self.log)
        self.assertEqual(resolved, (None,))
        self.assertIn("json.JSONDecoder.decod, which does not exist", seen.output[0])

    def test_an_uninstalled_subsystem_is_none_and_only_debug_logged(self):
        with self.assertLogs(self.log, level="DEBUG") as seen:
            self.assertIsNone(SR.resolve_row("Hook 'x'", "no.such.module", "X", "y", log=self.log))
        self.assertTrue(all(line.startswith("DEBUG") for line in seen.output))


class TestResolvedTables(unittest.TestCase):
    def test_producers_resolve_and_narrow_and_skip_what_does_not_import(self):
        table = {
            SR.MATERIALS: ("json", "JSONDecoder", "decode"),
            SR.LIGHTMAPS: ("no.such.module", "X", "y"),
        }
        with mock.patch.object(SR, "registered_hooks", return_value={}):
            import json

            self.assertEqual(SR.resolved_producers(table, "maya"), {SR.MATERIALS: json.JSONDecoder.decode})
            self.assertEqual(SR.resolved_producers(table, "maya", only=["lightmaps"]), {})

    def test_stagers_keep_a_row_with_a_missing_method_and_drop_an_unimportable_one(self):
        table = {
            "fx": ROW,
            "typo": ("json", "JSONDecoder", "decode", "finsh"),
            "gone": ("no.such.module", "X", "prepare", None),
        }
        with mock.patch.object(SR, "registered_hooks", return_value={}):
            with self.assertLogs(level="WARNING"):
                stagers = SR.resolved_stagers(table, "maya")
        self.assertEqual(sorted(stagers), ["fx", "typo"])
        self.assertIsNone(stagers["typo"][1])
        with mock.patch.object(SR, "registered_hooks", return_value={}):
            self.assertEqual(sorted(SR.resolved_stagers(table, "maya", names=["fx"])), ["fx"])


class TestStackEntries(unittest.TestCase):
    """A record of ``{"id": ...}`` entries, oldest first (``SMART_BAKE_SESSIONS``)."""

    def stack(self):
        return [{"id": "a", "n": 1}, {"id": "b", "n": 2}, {"id": "a", "n": 3}]

    def test_peek_reads_the_latest_or_the_latest_with_an_id(self):
        stack = self.stack()
        self.assertEqual(SR.peek_entry(stack)["n"], 3)
        self.assertEqual(SR.peek_entry(stack, "b")["n"], 2)
        self.assertEqual(SR.peek_entry(stack, "a")["n"], 3)
        self.assertIsNone(SR.peek_entry(stack, "z"))
        self.assertIsNone(SR.peek_entry([]))
        self.assertEqual(stack, self.stack(), "a peek takes nothing")

    def test_pop_takes_what_peek_reads_and_nothing_when_none_matches(self):
        stack = self.stack()
        self.assertEqual(SR.pop_entry(stack, "a")["n"], 3)
        self.assertEqual([e["n"] for e in stack], [1, 2])
        self.assertIsNone(SR.pop_entry(stack, "z"))
        self.assertEqual([e["n"] for e in stack], [1, 2])
        self.assertEqual(SR.pop_entry(stack)["n"], 2)
        self.assertIsNone(SR.pop_entry([]))


if __name__ == "__main__":
    unittest.main()
