#!/usr/bin/python
# coding=utf-8
"""Regression tests for pythontk CoreUtils.get_derived_type.

These target confirmed defects in the MRO-walking base-type resolver:
- return_name=True with filter_by_base_type=True crashed with AttributeError
  because the base name (already a str) had ``.__name__`` called on it.
- The ``include`` filter was a no-op whenever ``exclude`` was empty (the
  default), because the two conditions were AND-ed together.
- filter_by_base_type walks reaching ``object`` (whose ``__base__`` is None)
  crashed with AttributeError instead of returning the documented None.

Written as unittest.TestCase so BOTH the project runner (test/run_tests.py,
unittest discovery) and pytest collect it — module-level pytest functions are
invisible to unittest discovery and silently never ran.

Run with:
    python -m pytest test_core_utils.py -q
"""
import unittest

from pythontk import CoreUtils


class _Base:
    pass


class _Mid(_Base):
    pass


class _Leaf(_Mid):
    pass


class TestGetDerivedType(unittest.TestCase):
    def test_return_name_with_filter_by_base_type(self):
        """return_name + filter_by_base_type must not crash on the string base name.

        Regression: ``derived_type`` is ``cls.__base__.__name__`` (a str) when
        ``filter_by_base_type=True``; calling ``.__name__`` on it raised
        ``AttributeError: 'str' object has no attribute '__name__'``.
        """
        result = CoreUtils.get_derived_type(
            _Leaf(), return_name=True, filter_by_base_type=True
        )
        # First MRO entry is _Leaf; its __base__ is _Mid.
        self.assertEqual(result, "_Mid")

    def test_include_filter_restricts_without_exclude(self):
        """``include`` must skip non-matching MRO classes even with empty exclude.

        Regression: the include check was gated behind ``derived_type in
        exclude``, so with the default empty exclude the leaf class was always
        returned and include never restricted anything.
        """
        # Leaf is NOT in include, so the walker must skip it and return _Mid.
        result = CoreUtils.get_derived_type(_Leaf(), include=[_Mid])
        self.assertIs(result, _Mid)

    def test_exclude_still_dominates_include(self):
        """A type in both include and exclude is excluded (exclude dominance)."""
        # _Mid excluded even though included -> next eligible base is _Base.
        result = CoreUtils.get_derived_type(
            _Leaf(), include=[_Mid, _Base], exclude=[_Mid]
        )
        self.assertIs(result, _Base)

    def test_default_returns_leaf_class(self):
        """No filters -> leaf class object, unchanged behavior."""
        self.assertIs(CoreUtils.get_derived_type(_Leaf()), _Leaf)

    def test_unmatched_filter_by_base_type_returns_none(self):
        """An unmatched filtered walk returns None instead of raising.

        Regression: with filter_by_base_type=True the walk reaches ``object``,
        whose ``__base__`` is None; ``cls.__base__.__name__`` raised
        AttributeError instead of honoring the documented None return.
        """
        result = CoreUtils.get_derived_type(
            _Leaf(), filter_by_base_type=True, include=["NoSuchBase"]
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
