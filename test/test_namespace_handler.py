#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk NamespaceHandler and Placeholder.

Run with:
    python -m pytest test_namespace_handler.py -v
    python test_namespace_handler.py
"""
import unittest

from pythontk.core_utils.namespace_handler import NamespaceHandler, Placeholder

from conftest import BaseTestCase


class PlaceholderTest(BaseTestCase):
    """Tests for Placeholder class."""

    def test_init(self):
        """Test placeholder initialization."""
        placeholder = Placeholder(list)

        self.assertEqual(placeholder.class_type, list)
        self.assertIsNone(placeholder.factory)
        self.assertEqual(placeholder.args, ())
        self.assertEqual(placeholder.kwargs, {})
        self.assertEqual(placeholder.meta, {})

    def test_init_with_factory(self):
        """Test placeholder with factory."""

        def factory():
            return [1, 2, 3]

        placeholder = Placeholder(list, factory=factory)
        self.assertEqual(placeholder.factory, factory)

    def test_init_with_args_kwargs(self):
        """Test placeholder with args and kwargs."""
        placeholder = Placeholder(
            dict,
            args=({"a": 1},),
            kwargs={"name": "test"},
            meta={"info": "metadata"},
        )

        self.assertEqual(placeholder.args, ({"a": 1},))
        self.assertEqual(placeholder.kwargs, {"name": "test"})
        self.assertEqual(placeholder.meta, {"info": "metadata"})

    def test_info(self):
        """Test info method."""
        placeholder = Placeholder(list, args=(1, 2), meta={"key": "value"})
        info = placeholder.info()

        self.assertEqual(info["type"], "list")
        self.assertIsNone(info["factory"])
        self.assertEqual(info["args"], (1, 2))
        self.assertEqual(info["meta"], {"key": "value"})

    def test_create(self):
        """Test create method."""
        placeholder = Placeholder(list)
        result = placeholder.create([1, 2, 3])

        self.assertEqual(result, [1, 2, 3])

    def test_create_with_factory(self):
        """Test create with factory."""

        def factory():
            return {"created": True}

        placeholder = Placeholder(dict, factory=factory)
        result = placeholder.create()

        self.assertEqual(result, {"created": True})

    def test_repr(self):
        """Test repr."""
        placeholder = Placeholder(list)
        repr_str = repr(placeholder)

        self.assertIn("Placeholder", repr_str)
        self.assertIn("list", repr_str)


class NamespaceHandlerTest(BaseTestCase):
    """Tests for NamespaceHandler class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.owner = object()

    def test_init(self):
        """Test namespace handler initialization."""
        handler = NamespaceHandler(owner=self.owner, identifier="test")

        self.assertEqual(handler._identifier, "test")
        self.assertEqual(handler._owner, self.owner)

    def test_setattr_and_getattr(self):
        """Test setting and getting attributes."""
        handler = NamespaceHandler(owner=self.owner)
        handler.my_attr = "value"

        self.assertEqual(handler.my_attr, "value")

    def test_setitem_and_getitem(self):
        """Test setting and getting items."""
        handler = NamespaceHandler(owner=self.owner)
        handler["my_key"] = "value"

        self.assertEqual(handler["my_key"], "value")

    def test_contains(self):
        """Test contains operator."""
        handler = NamespaceHandler(owner=self.owner)
        handler.my_attr = "value"

        self.assertIn("my_attr", handler)
        self.assertNotIn("other_attr", handler)

    def test_delitem(self):
        """Test deleting items."""
        handler = NamespaceHandler(owner=self.owner)
        handler["my_key"] = "value"
        del handler["my_key"]

        self.assertNotIn("my_key", handler)

    def test_keys(self):
        """Test keys method."""
        handler = NamespaceHandler(owner=self.owner)
        handler.a = 1
        handler.b = 2

        keys = handler.keys()
        self.assertIn("a", keys)
        self.assertIn("b", keys)

    def test_items(self):
        """Test items method."""
        handler = NamespaceHandler(owner=self.owner)
        handler.a = 1
        handler.b = 2

        items = dict(handler.items())
        self.assertEqual(items, {"a": 1, "b": 2})

    def test_values(self):
        """Test values method."""
        handler = NamespaceHandler(owner=self.owner)
        handler.a = 1
        handler.b = 2

        values = handler.values()
        self.assertIn(1, values)
        self.assertIn(2, values)

    def test_setdefault_new_key(self):
        """Test setdefault with new key."""
        handler = NamespaceHandler(owner=self.owner)
        result = handler.setdefault("new_key", "default")

        self.assertEqual(result, "default")
        self.assertEqual(handler["new_key"], "default")

    def test_setdefault_existing_key(self):
        """Test setdefault with existing key."""
        handler = NamespaceHandler(owner=self.owner)
        handler["existing"] = "original"
        result = handler.setdefault("existing", "default")

        self.assertEqual(result, "original")

    def test_has(self):
        """Test has method."""
        handler = NamespaceHandler(owner=self.owner)
        handler.a = 1

        self.assertTrue(handler.has("a"))
        self.assertFalse(handler.has("b"))

    def test_raw(self):
        """Test raw method."""
        handler = NamespaceHandler(owner=self.owner)
        handler.a = 1

        self.assertEqual(handler.raw("a"), 1)
        self.assertIsNone(handler.raw("nonexistent"))

    def test_resolve_with_default(self):
        """Test resolve with default value."""
        handler = NamespaceHandler(owner=self.owner)
        result = handler.resolve("nonexistent", default="fallback")

        self.assertEqual(result, "fallback")

    def test_placeholder_set_and_resolve(self):
        """Test setting and resolving placeholder via get."""
        handler = NamespaceHandler(owner=self.owner)
        placeholder = Placeholder(list)
        handler.my_list = placeholder

        self.assertTrue(handler.has_placeholder("my_list"))

        # Access via get with resolve_placeholders=False returns the placeholder
        raw_value = handler.get("my_list", resolve_placeholders=False)
        self.assertIsInstance(raw_value, Placeholder)

        # Check via getattr returns the placeholder object
        attr_value = handler.my_list
        self.assertIsInstance(attr_value, Placeholder)

    def test_set_placeholder_method(self):
        """Test set_placeholder method."""
        handler = NamespaceHandler(owner=self.owner)
        placeholder = Placeholder(dict)
        handler.set_placeholder("my_dict", placeholder)

        self.assertTrue(handler.has_placeholder("my_dict"))
        self.assertEqual(handler.get_placeholder("my_dict"), placeholder)

    def test_is_placeholder(self):
        """Test is_placeholder method."""
        handler = NamespaceHandler(owner=self.owner)
        placeholder = Placeholder(list)

        self.assertTrue(handler.is_placeholder(placeholder))
        self.assertFalse(handler.is_placeholder([1, 2, 3]))

    def test_resolver_function(self):
        """Test custom resolver function."""

        def resolver(name):
            if name == "resolved_attr":
                return "resolved_value"
            return None

        handler = NamespaceHandler(
            owner=self.owner, identifier="test", resolver=resolver
        )
        result = handler.resolved_attr

        self.assertEqual(result, "resolved_value")

    def test_resolver_caches_result(self):
        """Test that resolver result is cached."""
        call_count = [0]

        def resolver(name):
            call_count[0] += 1
            return f"resolved_{name}"

        handler = NamespaceHandler(owner=self.owner, resolver=resolver)

        # First access
        _ = handler.test_attr
        # Second access should use cache
        _ = handler.test_attr

        self.assertEqual(call_count[0], 1)

    def test_getattr_invalid_raises(self):
        """Test that invalid attribute raises AttributeError."""
        handler = NamespaceHandler(owner=self.owner)

        with self.assertRaises(AttributeError):
            _ = handler.nonexistent

    def test_keys_with_placeholders(self):
        """Test keys method includes placeholders when requested."""
        handler = NamespaceHandler(owner=self.owner)
        handler.a = 1
        handler.b = Placeholder(list)

        keys_without = handler.keys(inc_placeholders=False)
        keys_with = handler.keys(inc_placeholders=True)

        self.assertIn("a", keys_without)
        self.assertNotIn("b", keys_without)
        self.assertIn("a", keys_with)
        self.assertIn("b", keys_with)

    def test_repr(self):
        """Test repr."""
        handler = NamespaceHandler(owner=self.owner, identifier="test_handler")
        handler.a = 1

        repr_str = repr(handler)
        self.assertIn("NamespaceHandler", repr_str)
        self.assertIn("test_handler", repr_str)


if __name__ == "__main__":
    unittest.main(exit=False)
