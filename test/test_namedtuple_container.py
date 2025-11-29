#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk NamedTupleContainer.

Run with:
    python -m pytest test_namedtuple_container.py -v
    python test_namedtuple_container.py
"""
import csv
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


if __name__ == "__main__":
    unittest.main(exit=False)
