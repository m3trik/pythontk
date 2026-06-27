# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.SchemaSpec — dataclass-defined template schemas."""
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List

from pythontk.core_utils.schema_spec import (
    SchemaSpec,
    SchemaError,
    spec_field,
)


@dataclass
class Inner(SchemaSpec):
    alias: list = spec_field(help="Header aliases.", example=["A"], default_factory=list)


def _validate_resolver(value: Any) -> List[str]:
    if not isinstance(value, dict):
        return ["expected a mapping"]
    if value.get("method") not in ("x", "y"):
        return [f"method {value.get('method')!r} is not one of ['x', 'y']"]
    return []


@dataclass
class Demo(SchemaSpec):
    title: str = spec_field(help="The title.", required=True, example="hello")
    count: int = spec_field(help="A number.", default=3)
    mode: str = spec_field(help="A choice.", choices=["a", "b"], default="a")
    inner: Inner = spec_field(help="Nested block.", nested=Inner, default_factory=Inner)
    resolver: Dict = spec_field(
        help="Polymorphic block.", validate=_validate_resolver, default=None
    )


class SchemaSpecValidateTest(unittest.TestCase):
    def test_valid_minimal(self):
        res = Demo.validate({"title": "x"})
        self.assertTrue(res.ok)
        self.assertEqual(res.warnings, [])

    def test_missing_required_is_error(self):
        res = Demo.validate({})
        self.assertFalse(res.ok)
        self.assertTrue(any("title" in e for e in res.errors))

    def test_unknown_key_is_warning_not_error(self):
        res = Demo.validate({"title": "x", "bogus": 1})
        self.assertTrue(res.ok)  # tolerated
        self.assertTrue(any("bogus" in w for w in res.warnings))

    def test_underscore_key_reserved_no_warning(self):
        res = Demo.validate({"title": "x", "_note": "hi"})
        self.assertTrue(res.ok)
        self.assertEqual(res.warnings, [])

    def test_bad_choice_is_error(self):
        res = Demo.validate({"title": "x", "mode": "z"})
        self.assertFalse(res.ok)
        self.assertTrue(any("mode" in e for e in res.errors))

    def test_custom_validator_runs_with_field_prefix(self):
        res = Demo.validate({"title": "x", "resolver": {"method": "q"}})
        self.assertFalse(res.ok)
        self.assertTrue(any(e.startswith("resolver:") for e in res.errors))

    def test_nested_errors_are_path_prefixed(self):
        res = Demo.validate({"title": "x", "inner": "notadict"})
        self.assertFalse(res.ok)
        self.assertTrue(any(e.startswith("inner:") for e in res.errors))

    def test_nested_warning_is_path_prefixed(self):
        res = Demo.validate({"title": "x", "inner": {"bogus": 1}})
        self.assertTrue(res.ok)
        self.assertTrue(any(w.startswith("inner.") for w in res.warnings))

    def test_not_a_mapping_is_error(self):
        res = Demo.validate(["not", "a", "dict"])
        self.assertFalse(res.ok)

    def test_raise_if_errors(self):
        with self.assertRaises(SchemaError):
            Demo.validate({}).raise_if_errors()
        Demo.validate({"title": "x"}).raise_if_errors()  # no raise


class SchemaSpecGenerateTest(unittest.TestCase):
    def test_skeleton_uses_examples_and_omits_unset_optionals(self):
        sk = Demo.skeleton()
        self.assertEqual(sk["title"], "hello")
        self.assertEqual(sk["count"], 3)
        self.assertEqual(sk["inner"], {"alias": ["A"]})  # nested skeleton
        self.assertNotIn("resolver", sk)  # unset optional omitted

    def test_skeleton_is_valid_against_its_own_schema(self):
        self.assertTrue(Demo.validate(Demo.skeleton()).ok)

    def test_describe_lists_fields(self):
        names = [d.name for d in Demo.describe()]
        self.assertEqual(names, ["title", "count", "mode", "inner", "resolver"])

    def test_markdown_contains_keys_and_nested_table(self):
        md = Demo.to_markdown(title="Demo Format")
        self.assertIn("Demo Format", md)
        self.assertIn("`title`", md)
        self.assertIn("One of: a, b.", md)  # choices surfaced
        self.assertIn("Inner", md)  # nested schema rendered


class RaiseOrWarnTest(unittest.TestCase):
    def test_raises_on_errors(self):
        with self.assertRaises(SchemaError):
            Demo.validate({}).raise_or_warn(prefix="demo: ")

    def test_warnings_tolerated_by_default(self):
        Demo.validate({"title": "x", "bogus": 1}).raise_or_warn()  # no raise

    def test_strict_raises_on_warnings(self):
        with self.assertRaises(SchemaError):
            Demo.validate({"title": "x", "bogus": 1}).raise_or_warn(strict=True)


class SchemaSpecRoundTripTest(unittest.TestCase):
    def test_from_dict_recurses_nested(self):
        obj = Demo.from_dict({"title": "x", "inner": {"alias": ["Q"]}})
        self.assertIsInstance(obj.inner, Inner)
        self.assertEqual(obj.inner.alias, ["Q"])

    def test_to_dict_omits_none_and_serialises_nested(self):
        obj = Demo.from_dict({"title": "x", "inner": {"alias": ["Q"]}})
        d = obj.to_dict()
        self.assertEqual(d["inner"], {"alias": ["Q"]})
        self.assertNotIn("resolver", d)  # None omitted

    def test_round_trip_dict_to_obj_to_dict(self):
        src = {"title": "x", "count": 9, "mode": "b", "inner": {"alias": ["Q"]}}
        out = Demo.from_dict(src).to_dict()
        for k, v in src.items():
            self.assertEqual(out[k], v)


class SchemaSpecRegressionTest(unittest.TestCase):
    """Regressions from the 2026-06-27 review: skeleton self-validity for *any*
    subclass, and no mutable aliasing across skeleton()/to_dict()/from_dict()."""

    def test_skeleton_self_valid_with_required_field_lacking_example(self):
        @dataclass
        class R(SchemaSpec):
            name: str = spec_field(help="Required, no example.", required=True)

        sk = R.skeleton()
        self.assertIn("name", sk)  # a required key must never be dropped
        self.assertTrue(R.validate(sk).ok)

    def test_skeleton_self_valid_with_choices_and_mismatched_example(self):
        @dataclass
        class C(SchemaSpec):
            mode: str = spec_field(
                help="Choice with a bad example.", choices=["p", "q"], example="zzz"
            )

        sk = C.skeleton()
        self.assertIn(sk["mode"], ["p", "q"])  # falls back to a valid choice
        self.assertTrue(C.validate(sk).ok)

    def test_skeleton_does_not_alias_mutable_example(self):
        @dataclass
        class M(SchemaSpec):
            tags: list = spec_field(example=["A", "B"], default_factory=list)

        sk1, sk2 = M.skeleton(), M.skeleton()
        self.assertIsNot(sk1["tags"], sk2["tags"])
        sk1["tags"].append("X")
        self.assertEqual(sk2["tags"], ["A", "B"])  # the other skeleton is intact
        self.assertEqual(M.skeleton()["tags"], ["A", "B"])  # class metadata intact

    def test_to_dict_output_is_independent_of_instance(self):
        @dataclass
        class MD(SchemaSpec):
            tags: list = spec_field(default_factory=list)

        obj = MD.from_dict({"tags": ["x"]})
        d = obj.to_dict()
        d["tags"].append("Z")
        self.assertEqual(obj.tags, ["x"])

    def test_from_dict_does_not_alias_source_dict(self):
        @dataclass
        class MD(SchemaSpec):
            tags: list = spec_field(default_factory=list)

        src = {"tags": ["x"]}
        MD.from_dict(src).tags.append("Z")
        self.assertEqual(src["tags"], ["x"])

    def test_markdown_skips_autogenerated_dataclass_docstring(self):
        @dataclass
        class NoDoc(SchemaSpec):
            a: str = spec_field(help="A field.", example="x")

        # @dataclass synthesises __doc__ = "NoDoc(a: str = None)" — that
        # constructor signature must not leak into the generated reference.
        self.assertNotIn("NoDoc(", NoDoc.to_markdown())


if __name__ == "__main__":
    unittest.main()
