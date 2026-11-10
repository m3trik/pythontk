# !/usr/bin/python
# coding=utf-8
"""Unit tests for CadRecord: the CAD import's record, declared once."""

import unittest

from pythontk.file_utils.cad_import.cad_record import CadRecord


class TestCadRecord(unittest.TestCase):
    def test_the_skeleton_validates(self):
        result = CadRecord.validate(CadRecord.skeleton())
        self.assertEqual(result.errors, [], result.errors)

    def test_a_root_node_states_a_null_parent(self):
        data = CadRecord.skeleton()
        data["nodes"][0]["parent"] = None
        self.assertEqual(CadRecord.validate(data).errors, [])

    def test_a_wrongly_typed_value_is_refused(self):
        data = CadRecord.skeleton()
        data["parts"][0]["area"] = "large"
        self.assertTrue(CadRecord.validate(data).errors)

    def test_an_unknown_format_is_refused(self):
        data = CadRecord.skeleton()
        data["source"]["format"] = "sldprt"
        self.assertTrue(CadRecord.validate(data).errors)

    def test_it_publishes_a_json_schema(self):
        schema = CadRecord.json_schema()
        self.assertIn("parts", schema["properties"])
        self.assertIn("nodes", schema["required"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
