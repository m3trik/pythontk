# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.TemplateSet — schema-validated two-tier template files."""
import json
import os
import shutil
import tempfile
import unittest
from dataclasses import dataclass

from pythontk.core_utils.preset_store import Codec
from pythontk.core_utils.schema_spec import SchemaSpec, SchemaError, spec_field
from pythontk.core_utils.template_set import TemplateSet


@dataclass
class CfgSpec(SchemaSpec):
    title: str = spec_field(help="A title.", required=True, example="hi")
    count: int = spec_field(help="A count.", default=1, example=2)


class TemplateSetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.builtin = os.path.join(self.tmp, "builtin")
        self.user = os.path.join(self.tmp, "user")
        os.makedirs(self.builtin)
        with open(os.path.join(self.builtin, "base.json"), "w", encoding="utf-8") as fh:
            json.dump({"title": "base", "count": 5}, fh)
        self.ts = TemplateSet(
            "maps", CfgSpec, "mayatk", builtin_dir=self.builtin, user_dir=self.user
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_names_lists_builtin(self):
        self.assertEqual(self.ts.names(), ["base"])
        self.assertEqual(self.ts.source("base"), "builtin")

    def test_load_validates_and_deserialises(self):
        obj = self.ts.load("base")
        self.assertIsInstance(obj, CfgSpec)
        self.assertEqual((obj.title, obj.count), ("base", 5))

    def test_load_raises_on_invalid_stored_file(self):
        self.ts.store.save("broken", {"count": 1})  # missing required 'title'
        with self.assertRaises(SchemaError):
            self.ts.load("broken")

    def test_raw_skips_validation(self):
        self.ts.store.save("broken", {"count": 1})
        self.assertEqual(self.ts.raw("broken"), {"count": 1})  # no raise

    def test_save_writes_user_template(self):
        path = self.ts.save("custom", {"title": "z", "count": 9})
        self.assertTrue(path.is_file())
        self.assertEqual(self.ts.source("custom"), "user")
        self.assertEqual(self.ts.load("custom").count, 9)  # round-trips + validates

    def test_skeleton_and_write_skeleton(self):
        self.assertEqual(self.ts.skeleton(), {"title": "hi", "count": 2})
        path = self.ts.write_skeleton("mine")
        self.assertTrue(path.is_file())
        self.assertEqual(self.ts.source("mine"), "user")
        self.assertEqual(self.ts.load("mine").title, "hi")  # round-trips + validates

    def test_user_shadows_builtin(self):
        self.ts.store.save("base", {"title": "override", "count": 9})
        self.assertEqual(self.ts.load("base").title, "override")
        self.assertEqual(self.ts.source("base"), "user")

    def test_markdown_generated_from_schema(self):
        md = self.ts.markdown(title="Map Format")
        self.assertIn("Map Format", md)
        self.assertIn("`title`", md)

    def test_active_pointer_passthrough(self):
        self.assertIsNone(self.ts.active)
        self.ts.active = "base"
        self.assertEqual(self.ts.active, "base")

    def test_custom_codec_round_trips_with_its_extension(self):
        # A non-JSON codec (ext + load/dump) drives discovery and IO.
        codec = Codec(".txt", json.loads, json.dumps)
        ts = TemplateSet(
            "alt", CfgSpec, "mayatk", user_dir=self.user, codec=codec
        )
        path = ts.store.save("c", {"title": "z", "count": 3})
        self.assertEqual(path.suffix, ".txt")
        self.assertEqual(ts.names(), ["c"])
        self.assertEqual(ts.load("c").count, 3)


if __name__ == "__main__":
    unittest.main()
