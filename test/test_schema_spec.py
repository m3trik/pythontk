# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.SchemaSpec — dataclass-defined template schemas."""

import json
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from pythontk.core_utils.schema_spec import SchemaSpec, SchemaError


@dataclass
class Inner(SchemaSpec):
    alias: list = SchemaSpec.spec_field(
        help="Header aliases.", example=["A"], default_factory=list
    )


def _validate_resolver(value: Any) -> List[str]:
    if not isinstance(value, dict):
        return ["expected a mapping"]
    if value.get("method") not in ("x", "y"):
        return [f"method {value.get('method')!r} is not one of ['x', 'y']"]
    return []


@dataclass
class Demo(SchemaSpec):
    title: str = SchemaSpec.spec_field(
        help="The title.", required=True, example="hello"
    )
    count: int = SchemaSpec.spec_field(help="A number.", default=3)
    mode: str = SchemaSpec.spec_field(help="A choice.", choices=["a", "b"], default="a")
    inner: Inner = SchemaSpec.spec_field(
        help="Nested block.", nested=Inner, default_factory=Inner
    )
    resolver: Dict = SchemaSpec.spec_field(
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
            name: str = SchemaSpec.spec_field(
                help="Required, no example.", required=True
            )

        sk = R.skeleton()
        self.assertIn("name", sk)  # a required key must never be dropped
        self.assertTrue(R.validate(sk).ok)

    def test_skeleton_self_valid_with_choices_and_mismatched_example(self):
        @dataclass
        class C(SchemaSpec):
            mode: str = SchemaSpec.spec_field(
                help="Choice with a bad example.", choices=["p", "q"], example="zzz"
            )

        sk = C.skeleton()
        self.assertIn(sk["mode"], ["p", "q"])  # falls back to a valid choice
        self.assertTrue(C.validate(sk).ok)

    def test_skeleton_does_not_alias_mutable_example(self):
        @dataclass
        class M(SchemaSpec):
            tags: list = SchemaSpec.spec_field(example=["A", "B"], default_factory=list)

        sk1, sk2 = M.skeleton(), M.skeleton()
        self.assertIsNot(sk1["tags"], sk2["tags"])
        sk1["tags"].append("X")
        self.assertEqual(sk2["tags"], ["A", "B"])  # the other skeleton is intact
        self.assertEqual(M.skeleton()["tags"], ["A", "B"])  # class metadata intact

    def test_to_dict_output_is_independent_of_instance(self):
        @dataclass
        class MD(SchemaSpec):
            tags: list = SchemaSpec.spec_field(default_factory=list)

        obj = MD.from_dict({"tags": ["x"]})
        d = obj.to_dict()
        d["tags"].append("Z")
        self.assertEqual(obj.tags, ["x"])

    def test_from_dict_does_not_alias_source_dict(self):
        @dataclass
        class MD(SchemaSpec):
            tags: list = SchemaSpec.spec_field(default_factory=list)

        src = {"tags": ["x"]}
        MD.from_dict(src).tags.append("Z")
        self.assertEqual(src["tags"], ["x"])

    def test_markdown_skips_autogenerated_dataclass_docstring(self):
        @dataclass
        class NoDoc(SchemaSpec):
            a: str = SchemaSpec.spec_field(help="A field.", example="x")

        # @dataclass synthesises __doc__ = "NoDoc(a: str = None)" — that
        # constructor signature must not leak into the generated reference.
        self.assertNotIn("NoDoc(", NoDoc.to_markdown())


@dataclass
class Occupant(SchemaSpec):
    """One rack occupant."""

    label: str = SchemaSpec.spec_field(
        help="Front label.", required=True, example="DMM"
    )
    height_u: int = SchemaSpec.spec_field(help="Height in U.", default=1)


@dataclass
class Bay(SchemaSpec):
    """A rack bay holding an ordered list of occupants."""

    ru: int = SchemaSpec.spec_field(help="Rack units.", default=42)
    occupants: list = SchemaSpec.spec_field(
        help="Installed gear, bottom-up.",
        nested=[Occupant],
        default_factory=list,
    )


class SchemaSpecNestedListTest(unittest.TestCase):
    """List-of-nested support: ``nested=[Spec]`` for a key whose value is a
    list of sub-records (e.g. a rack bay's occupants)."""

    def test_from_dict_builds_list_of_instances(self):
        bay = Bay.from_dict(
            {
                "ru": 24,
                "occupants": [{"label": "Scope", "height_u": 8}, {"label": "DMM"}],
            }
        )
        self.assertEqual(len(bay.occupants), 2)
        self.assertTrue(all(isinstance(o, Occupant) for o in bay.occupants))
        self.assertEqual(bay.occupants[0].label, "Scope")
        self.assertEqual(bay.occupants[0].height_u, 8)

    def test_to_dict_serialises_list_of_instances(self):
        bay = Bay.from_dict({"occupants": [{"label": "Scope"}, {"label": "DMM"}]})
        d = bay.to_dict()
        self.assertEqual(
            d["occupants"],
            [{"label": "Scope", "height_u": 1}, {"label": "DMM", "height_u": 1}],
        )

    def test_round_trip(self):
        src = {"ru": 24, "occupants": [{"label": "Scope", "height_u": 8}]}
        out = Bay.from_dict(src).to_dict()
        self.assertEqual(out["occupants"], [{"label": "Scope", "height_u": 8}])

    def test_validate_recurses_each_element_with_indexed_path(self):
        res = Bay.validate({"occupants": [{"label": "ok"}, {"height_u": 2}]})
        self.assertFalse(res.ok)
        # the second element is missing its required 'label'
        self.assertTrue(
            any(e.startswith("occupants[1].") and "label" in e for e in res.errors)
        )

    def test_validate_rejects_non_list(self):
        res = Bay.validate({"occupants": {"label": "x"}})
        self.assertFalse(res.ok)
        self.assertTrue(any("expected a list" in e for e in res.errors))

    def test_validate_accepts_empty_list(self):
        self.assertTrue(Bay.validate({"occupants": []}).ok)

    def test_skeleton_emits_single_element_example_list(self):
        sk = Bay.skeleton()
        self.assertIsInstance(sk["occupants"], list)
        self.assertEqual(len(sk["occupants"]), 1)
        self.assertEqual(sk["occupants"][0]["label"], "DMM")

    def test_skeleton_is_valid_against_its_own_schema(self):
        self.assertTrue(Bay.validate(Bay.skeleton()).ok)

    def test_markdown_marks_list_and_renders_element_schema(self):
        md = Bay.to_markdown()
        self.assertIn("(list)", md)
        self.assertIn("Occupant", md)

    def test_malformed_nested_metadata_raises(self):
        @dataclass
        class Bad(SchemaSpec):
            items: list = SchemaSpec.spec_field(
                nested=[Occupant, Bay], default_factory=list
            )

        with self.assertRaises(SchemaError):
            Bad.validate({"items": []})


# -- payload shapes: typed fields and the JSON Schema -------------------------


@dataclass
class Leg(SchemaSpec):
    """One leg."""

    TYPED = True

    side: str = SchemaSpec.spec_field(
        help="Which side.", required=True, choices=("left", "right")
    )
    length: Optional[float] = SchemaSpec.spec_field(
        help="Metres, or null.", required=True
    )
    tip: Tuple[float, float, float] = SchemaSpec.spec_field(required=True)
    tags: Dict[str, int] = SchemaSpec.spec_field(default_factory=dict)
    mode: Literal["walk", "run"] = SchemaSpec.spec_field(default="walk")


@dataclass
class Body(SchemaSpec):
    """A body.

    Its legs, by their own schema.
    """

    TYPED = True

    count: int = SchemaSpec.spec_field(help="How many.", required=True)
    legs: List[Leg] = SchemaSpec.spec_field(
        help="The legs.", required=True, nested=[Leg], default_factory=list
    )
    lead: Optional[Leg] = SchemaSpec.spec_field(help="The leading leg.")


class TypedValidateTest(unittest.TestCase):
    """A TYPED schema holds every present value to its annotation."""

    VALID = {"count": 2, "legs": [{"side": "left", "length": 0.5, "tip": [0, 1, 2]}]}

    def test_a_valid_payload_passes(self):
        self.assertEqual(Body.validate(self.VALID).errors, [])

    def test_a_null_meets_its_optional(self):
        doc = {"count": 1, "legs": [{"side": "left", "length": None, "tip": [0, 0, 0]}]}
        self.assertEqual(Body.validate(doc).errors, [])

    def test_a_bool_is_never_a_number(self):
        res = Body.validate({"count": True, "legs": []})
        self.assertEqual(res.errors, ["count: expected integer, got boolean"])

    def test_an_int_is_a_float(self):
        doc = {"count": 1, "legs": [{"side": "right", "length": 3, "tip": [1, 2, 3]}]}
        self.assertTrue(Body.validate(doc).ok)

    def test_errors_are_located_through_nesting(self):
        doc = {
            "count": 1,
            "legs": [
                {
                    "side": "left",
                    "length": "far",
                    "tip": [0, "y", 2],
                    "tags": {"a": 1.5},
                }
            ],
        }
        self.assertEqual(
            Body.validate(doc).errors,
            [
                "legs[0].length: expected number, got string",
                "legs[0].tip[1]: expected number, got string",
                "legs[0].tags.a: expected integer, got number",
            ],
        )

    def test_a_fixed_tuple_counts_its_items(self):
        doc = {"count": 1, "legs": [{"side": "left", "length": 1, "tip": [0, 1]}]}
        self.assertIn("legs[0].tip: expected 3 items, got 2", Body.validate(doc).errors)

    def test_a_literal_admits_its_values_only(self):
        doc = dict(self.VALID, legs=[dict(self.VALID["legs"][0], mode="fly")])
        self.assertIn(
            "legs[0].mode: 'fly' is not one of ['walk', 'run']",
            Body.validate(doc).errors,
        )

    def test_an_annotated_schema_validates_without_nested_metadata(self):
        res = Body.validate(
            dict(self.VALID, lead={"side": "up", "length": 1, "tip": [0, 0, 0]})
        )
        self.assertEqual(
            res.errors, ["lead.side: 'up' is not one of ['left', 'right']"]
        )

    def test_unknown_keys_stay_warnings(self):
        res = Body.validate(dict(self.VALID, extra=1))
        self.assertTrue(res.ok)
        self.assertEqual(res.warnings, ["unknown key 'extra' (ignored)"])

    def test_an_untyped_schema_checks_no_types(self):
        self.assertTrue(Demo.validate({"title": 5, "count": "three"}).ok)


class JsonSchemaTest(unittest.TestCase):
    """``json_schema`` -- the declaration as the document other languages read."""

    def test_the_document_names_its_dialect_title_and_first_paragraph(self):
        doc = Body.json_schema()
        self.assertEqual(doc["$schema"], SchemaSpec.JSON_SCHEMA_DIALECT)
        self.assertEqual(doc["title"], "Body")
        self.assertEqual(doc["description"], "A body.")
        self.assertEqual(doc["required"], ["count", "legs"])

    def test_field_types_come_from_the_annotations(self):
        leg = Body.json_schema()["$defs"]["Leg"]["properties"]
        self.assertEqual(
            leg["length"],
            {
                "anyOf": [{"type": "number"}, {"type": "null"}],
                "description": "Metres, or null.",
            },
        )
        self.assertEqual(
            leg["tip"],
            {
                "type": "array",
                "prefixItems": [{"type": "number"}] * 3,
                "minItems": 3,
                "maxItems": 3,
            },
        )
        self.assertEqual(
            leg["tags"], {"type": "object", "additionalProperties": {"type": "integer"}}
        )
        self.assertEqual(leg["mode"], {"type": "string", "enum": ["walk", "run"]})
        self.assertEqual(
            leg["side"],
            {"type": "string", "enum": ["left", "right"], "description": "Which side."},
        )

    def test_nested_schemas_are_defined_once_and_referenced(self):
        doc = Body.json_schema()
        self.assertEqual(sorted(doc["$defs"]), ["Leg"])
        self.assertEqual(doc["properties"]["legs"]["items"], {"$ref": "#/$defs/Leg"})
        self.assertEqual(
            doc["properties"]["lead"]["anyOf"],
            [{"$ref": "#/$defs/Leg"}, {"type": "null"}],
        )

    def test_the_document_is_json(self):
        json.dumps(Body.json_schema())
        json.dumps(Demo.json_schema())  # a template schema too

    def test_a_typed_annotation_contradicting_nested_is_refused(self):
        @dataclass
        class Liar(SchemaSpec):
            """Says two things."""

            TYPED = True
            legs: List[Inner] = SchemaSpec.spec_field(
                nested=[Leg], default_factory=list
            )

        with self.assertRaises(SchemaError):
            Liar.json_schema()

    def test_two_schemas_of_one_name_are_refused(self):
        def make():
            @dataclass
            class Leg(SchemaSpec):
                """A different leg."""

                name: str = SchemaSpec.spec_field(required=True)

            return Leg

        other = make()

        @dataclass
        class Both(SchemaSpec):
            """Two legs of one name."""

            a: Leg = SchemaSpec.spec_field(nested=Leg)
            b: Any = SchemaSpec.spec_field(nested=other)

        with self.assertRaises(SchemaError):
            Both.json_schema()


if __name__ == "__main__":
    unittest.main()
