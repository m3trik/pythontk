#!/usr/bin/python
# coding=utf-8
"""Unit tests for pythontk SymbolRecord.

Run with:
    python -m pytest test_symbol_record.py -v
    python test_symbol_record.py
"""
import json
import unittest

from pythontk.core_utils.symbol_record import SymbolRecord

from conftest import BaseTestCase

# The field order is a hard contract: the static registry sidecar
# (API_REGISTRY.json) is json.dumps(asdict(...)), so a reordering would churn
# every committed registry. Freeze it here.
EXPECTED_FIELDS = ["name", "qualname", "kind", "signature", "summary", "line", "deprecated"]


class SymbolRecordTest(BaseTestCase):
    """SymbolRecord test class."""

    def _rec(self, **overrides):
        base = dict(
            name="listify",
            qualname="CoreUtils.listify",
            kind="method",
            signature="(x, threshold=None)",
            summary="Return x as a list.",
            line=42,
            deprecated=False,
        )
        base.update(overrides)
        return SymbolRecord(**base)

    def test_field_order_is_frozen(self):
        """asdict() key order must match the registry sidecar contract."""
        self.assertEqual(list(self._rec().as_dict().keys()), EXPECTED_FIELDS)

    def test_as_dict_values(self):
        d = self._rec().as_dict()
        self.assertEqual(d["name"], "listify")
        self.assertEqual(d["qualname"], "CoreUtils.listify")
        self.assertEqual(d["kind"], "method")
        self.assertEqual(d["line"], 42)
        self.assertFalse(d["deprecated"])

    def test_as_json_round_trips(self):
        rec = self._rec()
        self.assertEqual(json.loads(rec.as_json()), rec.as_dict())

    def test_deprecated_defaults_false(self):
        rec = SymbolRecord("f", "C.f", "method", "()", "", 1)
        self.assertFalse(rec.deprecated)

    # -- to_registry_row: must reproduce the generator's member-row format -----

    def test_registry_row_plain_method(self):
        row = self._rec(summary="Return x as a list.").to_registry_row()
        self.assertEqual(
            row, "  - `CoreUtils.listify(x, threshold=None)` — Return x as a list."
        )

    def test_registry_row_no_summary(self):
        row = self._rec(summary="").to_registry_row()
        self.assertEqual(row, "  - `CoreUtils.listify(x, threshold=None)`")

    def test_registry_row_staticmethod(self):
        row = self._rec(kind="staticmethod", summary="").to_registry_row()
        self.assertIn(" *(static)*", row)

    def test_registry_row_classmethod(self):
        row = self._rec(kind="classmethod", summary="").to_registry_row()
        self.assertIn(" *(class)*", row)

    def test_registry_row_property(self):
        row = self._rec(kind="property", signature="", summary="").to_registry_row()
        self.assertIn(" *(property)*", row)

    def test_registry_row_deprecated(self):
        row = self._rec(deprecated=True, summary="").to_registry_row()
        self.assertIn(" **DEPRECATED**", row)

    def test_registry_row_deprecated_static_with_summary(self):
        """Decoration order: kind marker, then DEPRECATED, then summary."""
        row = self._rec(
            kind="staticmethod", deprecated=True, summary="Old helper."
        ).to_registry_row()
        self.assertEqual(
            row,
            "  - `CoreUtils.listify(x, threshold=None)` *(static)* "
            "**DEPRECATED** — Old helper.",
        )


if __name__ == "__main__":
    unittest.main(exit=False)
