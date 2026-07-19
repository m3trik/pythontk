#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk NamedTupleContainer.

Run with:
    python -m pytest test_namedtuple_container.py -v
    python test_namedtuple_container.py
"""
import os
import tempfile
import unittest
from collections import namedtuple

from pythontk.core_utils.namedtuple_container import NamedTupleContainer

from conftest import BaseTestCase


class NamedTupleContainerTest(BaseTestCase):
    """Tests for NamedTupleContainer class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.Person = namedtuple("Person", ["name", "age", "city"])
        self.people = [
            self.Person("Alice", 30, "New York"),
            self.Person("Bob", 25, "London"),
            self.Person("Charlie", 35, "Tokyo"),
        ]

    def test_init_empty(self):
        """Test creating an empty container."""
        container = NamedTupleContainer()
        self.assertEqual(len(container), 0)
        self.assertEqual(container.fields, [])

    def test_init_with_tuples(self):
        """Test creating container with named tuples."""
        container = NamedTupleContainer(named_tuples=self.people)

        self.assertEqual(len(container), 3)
        self.assertEqual(container.fields, ["name", "age", "city"])

    def test_init_with_fields(self):
        """Test creating container with explicit fields."""
        container = NamedTupleContainer(fields=["a", "b", "c"])
        self.assertEqual(container.fields, ["a", "b", "c"])

    def test_iter(self):
        """Test iteration over container."""
        container = NamedTupleContainer(named_tuples=self.people)
        items = list(container)

        self.assertEqual(len(items), 3)
        self.assertEqual(items[0].name, "Alice")

    def test_repr(self):
        """Test string representation."""
        container = NamedTupleContainer(named_tuples=self.people)
        repr_str = repr(container)

        self.assertIn("3 items", repr_str)
        self.assertIn("fields=", repr_str)

    def test_len(self):
        """Test length of container."""
        container = NamedTupleContainer(named_tuples=self.people)
        self.assertEqual(len(container), 3)

    def test_getattr_for_field(self):
        """Test dynamic attribute access for fields."""
        container = NamedTupleContainer(named_tuples=self.people)

        names = container.name
        self.assertEqual(names, ["Alice", "Bob", "Charlie"])

        ages = container.age
        self.assertEqual(ages, [30, 25, 35])

    def test_getattr_invalid_field(self):
        """Test that invalid field raises AttributeError."""
        container = NamedTupleContainer(named_tuples=self.people)

        with self.assertRaises(AttributeError):
            _ = container.invalid_field

    def test_getitem_single(self):
        """Test getting single item by index."""
        container = NamedTupleContainer(named_tuples=self.people)

        item = container[0]
        self.assertEqual(item.name, "Alice")

    def test_getitem_slice(self):
        """Test getting items by slice."""
        container = NamedTupleContainer(named_tuples=self.people)

        items = container[1:]
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].name, "Bob")

    def test_setitem(self):
        """Test setting item by index."""
        container = NamedTupleContainer(named_tuples=self.people)
        new_person = self.Person("Diana", 28, "Paris")
        container[0] = new_person

        self.assertEqual(container[0].name, "Diana")

    def test_delitem(self):
        """Test deleting item by index."""
        container = NamedTupleContainer(named_tuples=self.people)
        del container[0]

        self.assertEqual(len(container), 2)
        self.assertEqual(container[0].name, "Bob")

    def test_extend_with_tuples(self):
        """Test extending with regular tuples."""
        container = NamedTupleContainer(fields=["name", "age", "city"])
        container.extend([("Alice", 30, "New York"), ("Bob", 25, "London")])

        self.assertEqual(len(container), 2)

    def test_extend_with_named_tuples(self):
        """Test extending with named tuples."""
        container = NamedTupleContainer(fields=["name", "age", "city"])
        container.extend(self.people)

        self.assertEqual(len(container), 3)

    def test_extend_with_extender_func(self):
        """Test extending with custom extender function."""

        def extender(container, objects, **metadata):
            return [(obj, len(obj)) for obj in objects]

        container = NamedTupleContainer(
            fields=["value", "length"],
            extender_func=extender,
        )
        container.extend(["hello", "world"])

        self.assertEqual(len(container), 2)
        self.assertEqual(container[0].length, 5)

    def test_extend_duplicates_allowed(self):
        """Test extending with duplicates allowed."""
        container = NamedTupleContainer(
            named_tuples=self.people[:1],
            metadata={"allow_duplicates": True},
        )
        container.extend(self.people[:1])

        self.assertEqual(len(container), 2)

    def test_extend_duplicates_not_allowed(self):
        """Test extending with duplicates not allowed."""
        container = NamedTupleContainer(
            named_tuples=self.people[:1],
            metadata={"allow_duplicates": False},
        )
        container.extend(self.people[:1])

        self.assertEqual(len(container), 1)

    def test_get_all(self):
        """Test get all items."""
        container = NamedTupleContainer(named_tuples=self.people)
        results = container.get()

        self.assertEqual(len(results), 3)

    def test_get_with_condition(self):
        """Test get with condition."""
        container = NamedTupleContainer(named_tuples=self.people)
        results = container.get(age=30)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Alice")

    def test_get_with_return_field(self):
        """Test get with return field."""
        container = NamedTupleContainer(named_tuples=self.people)
        result = container.get(return_field="name", age=30)

        self.assertEqual(result, "Alice")

    def test_filter(self):
        """Test filter method."""
        container = NamedTupleContainer(named_tuples=self.people)
        filtered = container.filter(lambda x: x.age > 25)

        self.assertEqual(len(filtered), 2)
        self.assertIn(filtered[0].name, ["Alice", "Charlie"])

    def test_map(self):
        """Test map method."""
        container = NamedTupleContainer(named_tuples=self.people)

        def increment_age(person):
            return person._replace(age=person.age + 1)

        mapped = container.map(increment_age)

        self.assertEqual(mapped[0].age, 31)  # Alice was 30
        self.assertEqual(mapped[1].age, 26)  # Bob was 25

    def test_modify(self):
        """Test modify method."""
        container = NamedTupleContainer(named_tuples=self.people)
        modified = container.modify(0, age=31)

        self.assertEqual(modified.age, 31)
        self.assertEqual(container[0].age, 31)

    def test_modify_invalid_index(self):
        """Test modify with invalid index."""
        container = NamedTupleContainer(named_tuples=self.people)

        with self.assertRaises(IndexError):
            container.modify(10, age=31)

    def test_modify_invalid_field(self):
        """Test modify with invalid field."""
        container = NamedTupleContainer(named_tuples=self.people)

        with self.assertRaises(AttributeError):
            container.modify(0, invalid_field="value")

    def test_remove(self):
        """Test remove method."""
        container = NamedTupleContainer(named_tuples=self.people)
        removed = container.remove(0)

        self.assertEqual(removed.name, "Alice")
        self.assertEqual(len(container), 2)

    def test_remove_invalid_index(self):
        """Test remove with invalid index."""
        container = NamedTupleContainer(named_tuples=self.people)

        with self.assertRaises(IndexError):
            container.remove(10)

    def test_clear(self):
        """Test clear method."""
        container = NamedTupleContainer(named_tuples=self.people)
        container.clear()

        self.assertEqual(len(container), 0)

    def test_to_dict_list(self):
        """Test to_dict_list method."""
        container = NamedTupleContainer(named_tuples=self.people)
        dict_list = container.to_dict_list()

        self.assertEqual(len(dict_list), 3)
        self.assertEqual(dict_list[0]["name"], "Alice")
        self.assertIsInstance(dict_list[0], dict)

    def test_to_csv_and_from_csv(self):
        """Test CSV export and import."""
        container = NamedTupleContainer(named_tuples=self.people)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            filepath = f.name

        try:
            container.to_csv(filepath)
            loaded = NamedTupleContainer.from_csv(filepath)

            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[0].name, "Alice")
        finally:
            os.unlink(filepath)


class NamedTupleContainerContractTest(BaseTestCase):
    """Pins for contract bugs found in the 2026-07 improvement pass.

    Each test was written red-first against the prior implementation.
    """

    def setUp(self):
        super().setUp()
        self.Person = namedtuple("Person", ["name", "age", "city"])
        self.people = [
            self.Person("Alice", 30, "New York"),
            self.Person("Bob", 25, "London"),
            self.Person("Charlie", 35, "Tokyo"),
        ]

    def test_init_normalizes_plain_tuples(self):
        """Plain tuples + explicit fields must yield working field access.

        This is the `Metadata._batch_get` call shape: it builds plain tuples
        and relies on the container to make them addressable by field.
        """
        container = NamedTupleContainer(
            named_tuples=[("a.txt", "/p/a.txt"), ("b.txt", "/p/b.txt")],
            fields=["filename", "filepath"],
        )
        self.assertEqual(container.filename, ["a.txt", "b.txt"])
        self.assertTrue(hasattr(container[0], "_fields"))
        self.assertEqual(container.get(filename="b.txt", return_field="filepath"), "/p/b.txt")

    def test_get_single_no_match_returns_none(self):
        """conditions + return_field with no match returns None (as documented)."""
        container = NamedTupleContainer(named_tuples=self.people)
        self.assertIsNone(container.get(return_field="name", age=99))

    def test_get_list_paths_return_lists(self):
        """List-returning paths stay lists even when empty."""
        container = NamedTupleContainer(named_tuples=self.people)
        self.assertEqual(container.get(age=99), [])
        self.assertEqual(container.get(return_field="name"), ["Alice", "Bob", "Charlie"])

    def test_modify_rejects_namedtuple_method_names(self):
        """Field validation must check _fields, not hasattr (count/index are methods)."""
        container = NamedTupleContainer(named_tuples=self.people)
        with self.assertRaises(AttributeError):
            container.modify(0, count=1)

    def test_partially_initialized_getattr_raises_attribute_error(self):
        """__getattr__ on an instance without `fields` must not infinitely recurse."""
        obj = NamedTupleContainer.__new__(NamedTupleContainer)
        with self.assertRaises(AttributeError):
            _ = obj.name

    def test_copy_preserves_contents(self):
        """Shallow copy round-trips (guards the __getattr__ special-method path)."""
        import copy

        container = NamedTupleContainer(named_tuples=self.people)
        dup = copy.copy(container)
        self.assertEqual(list(dup), list(container))
        self.assertEqual(dup.fields, container.fields)

    def test_extend_empty_list_is_noop(self):
        """extend([]) must be a no-op, not a mis-dispatch into the object path."""
        calls = []

        def extender(container, objects, **metadata):
            calls.append(objects)
            return []

        container = NamedTupleContainer(fields=["a", "b"], extender_func=extender)
        container.extend([])
        self.assertEqual(len(container), 0)
        self.assertEqual(calls, [])

    def test_extend_none_is_noop(self):
        container = NamedTupleContainer(fields=["a", "b"])
        container.extend(None)
        self.assertEqual(len(container), 0)

    def test_extend_adopts_fields_from_named_tuples(self):
        """A field-less container adopts fields from incoming named tuples."""
        container = NamedTupleContainer()
        container.extend(self.people)
        self.assertEqual(container.fields, ["name", "age", "city"])
        self.assertEqual(container.name, ["Alice", "Bob", "Charlie"])
        container.extend([("Diana", 28, "Paris")])  # tuple class adopted too
        self.assertEqual(container[3].name, "Diana")

    def test_extend_custom_signature_func(self):
        """A metadata-supplied signature_func drives duplicate detection."""
        container = NamedTupleContainer(
            named_tuples=self.people[:1],
            metadata={"signature_func": lambda nt: nt.name},
        )
        container.extend([self.Person("Alice", 99, "Nowhere")])  # same name = dup
        self.assertEqual(len(container), 1)
        container.extend([self.Person("Diana", 28, "Paris")])
        self.assertEqual(len(container), 2)

    def test_extend_custom_signature_func_unhashable(self):
        """Custom signatures may be unhashable — dedup must still work."""
        container = NamedTupleContainer(
            named_tuples=self.people[:1],
            metadata={"signature_func": lambda nt: [nt.name]},
        )
        container.extend([self.Person("Alice", 99, "Nowhere")])
        self.assertEqual(len(container), 1)

    def test_extend_dedup_with_unhashable_values(self):
        """Duplicate detection must tolerate unhashable field values (lists/dicts)."""
        Row = namedtuple("Row", ["name", "data"])
        container = NamedTupleContainer(named_tuples=[Row("a", [1, 2])])
        container.extend([Row("a", [1, 2]), Row("b", [3])])
        self.assertEqual(len(container), 2)
        self.assertEqual(container.name, ["a", "b"])

    def test_extend_aligns_named_tuples_by_field_name(self):
        """Same field set in a different order must align by name, not position."""
        Swapped = namedtuple("Swapped", ["city", "name", "age"])
        container = NamedTupleContainer(named_tuples=self.people[:1])
        container.extend([Swapped("Paris", "Diana", 28)])
        self.assertEqual(container[1].name, "Diana")
        self.assertEqual(container[1].city, "Paris")
        self.assertEqual(container[1].age, 28)

    def test_fields_from_string(self):
        """String fields split on commas/whitespace, mirroring namedtuple()."""
        container = NamedTupleContainer(fields="name, age city")
        self.assertEqual(container.fields, ["name", "age", "city"])
        container.extend([("A", 1, "X")])
        self.assertEqual(container[0].age, 1)

    def test_fields_from_metadata_string(self):
        container = NamedTupleContainer(metadata={"fields": "name age"})
        self.assertEqual(container.fields, ["name", "age"])

    def test_negative_index_modify_and_remove(self):
        """Negative indices follow list semantics."""
        container = NamedTupleContainer(named_tuples=list(self.people))
        container.modify(-1, age=99)
        self.assertEqual(container[2].age, 99)
        removed = container.remove(-1)
        self.assertEqual(removed.name, "Charlie")
        self.assertEqual(len(container), 2)

    def test_filter_and_map_preserve_subclass(self):
        """filter/map must return the subclass with its extra state intact."""

        class Tagged(NamedTupleContainer):
            def __init__(self, tag, **kwargs):
                super().__init__(**kwargs)
                self.tag = tag

        sub = Tagged("x", named_tuples=self.people)
        filtered = sub.filter(lambda nt: nt.age > 25)
        self.assertIsInstance(filtered, Tagged)
        self.assertEqual(filtered.tag, "x")
        self.assertEqual(len(filtered), 2)

        mapped = sub.map(lambda nt: nt._replace(age=nt.age + 1))
        self.assertIsInstance(mapped, Tagged)
        self.assertEqual(mapped.tag, "x")
        self.assertEqual(mapped[0].age, 31)

    def test_filter_does_not_mutate_source(self):
        container = NamedTupleContainer(named_tuples=self.people)
        filtered = container.filter(lambda nt: nt.age > 25)
        filtered.clear()
        self.assertEqual(len(container), 3)

    def test_repr_uses_subclass_name(self):
        class Tagged(NamedTupleContainer):
            pass

        self.assertIn("Tagged", repr(Tagged(named_tuples=self.people)))


if __name__ == "__main__":
    unittest.main(exit=False)
