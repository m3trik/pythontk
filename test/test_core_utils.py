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

import ast
import os
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


class TestTeardownGuard(unittest.TestCase):
    """The one teardown policy: log, never raise, never mask the body's error."""

    def test_restore_failure_is_logged_not_raised(self):
        import logging

        log = logging.getLogger("teardown_guard_test")
        with self.assertLogs(log, level="WARNING") as captured:
            with CoreUtils.teardown_guard(log, "widget state"):
                raise RuntimeError("cannot restore")
        self.assertIn("widget state not fully restored", captured.output[0])

    def test_body_error_survives_a_failing_restore(self):
        import logging

        log = logging.getLogger("teardown_guard_test")

        def scope():
            try:
                raise ValueError("the real error")
            finally:
                with CoreUtils.teardown_guard(log, "x"):
                    raise RuntimeError("restore also failed")

        with self.assertLogs(log, level="WARNING"):
            with self.assertRaises(ValueError):
                scope()

    def test_default_logger_when_none_given(self):
        with self.assertLogs("pythontk.core_utils._core_utils", level="WARNING"):
            with CoreUtils.teardown_guard(what="thing"):
                raise RuntimeError("boom")


class TestListifyThreadingPolicy(unittest.TestCase):
    """``listify(threading=True)`` builds a ThreadPoolExecutor per call.

    Measured on 2/200/2000-item lists, that is a LOSS for every pure-Python
    and stat-bound method it decorated -- 2.4x to 173x slower, worst at the
    small list sizes real callers pass:

        format_path                6.0x - 48.7x slower
        convert_to_relative_path   2.4x - 18.3x
        move_decimal_point         9.7x - 34.9x
        clamp                     33.8x - 98.1x
        set_case                  29.1x - 173.0x
        split_delimited_string    34.8x - 158.5x
        truncate                  25.6x - 151.9x
        time_stamp                 1.4x -  9.7x   (os.path.getmtime; still a loss)

    ``create_mask`` is the one real winner -- numpy/PIL release the GIL, and
    it measured 0.56x / 0.25x / 0.21x the serial time on 2/20/100 1024px
    images -- so it KEEPS the flag. This test pins both halves of that split,
    since the cost is invisible at every call site.
    """

    #: Decorated methods measured slower threaded; the flag must be gone.
    SERIAL = (
        ("FileUtils", "format_path", ("C:/a/b.txt", "C:/a/c.txt"), ()),
        (
            "FileUtils",
            "convert_to_relative_path",
            ("C:/a/b.txt", "C:/a/c.txt"),
            ("C:/a",),
        ),
        ("MathUtils", "move_decimal_point", (1.0, 2.0), (2,)),
        ("MathUtils", "clamp", (1.0, 2.0), ()),
        ("StrUtils", "set_case", ("ab", "cd"), ()),
        ("StrUtils", "split_delimited_string", ("a,b", "c,d"), ()),
        ("StrUtils", "truncate", ("x" * 200, "y" * 200), ()),
        ("StrUtils", "time_stamp", (__file__, __file__), ()),
    )

    def _count_pools(self, call):
        """Run *call* with the executor the decorator uses swapped for a
        counting stand-in, and report how many pools it built."""
        import pythontk.core_utils._core_utils as cu

        built = []
        real = cu.ThreadPoolExecutor

        class Counting(real):
            def __init__(self, *a, **kw):
                built.append(1)
                super().__init__(*a, **kw)

        cu.ThreadPoolExecutor = Counting
        try:
            call()
        finally:
            cu.ThreadPoolExecutor = real
        return len(built)

    def test_pure_python_methods_do_not_build_a_thread_pool(self):
        import pythontk as ptk

        for cls_name, meth, items, extra in self.SERIAL:
            with self.subTest(method=f"{cls_name}.{meth}"):
                fn = getattr(getattr(ptk, cls_name), meth)
                n = self._count_pools(lambda: fn(list(items), *extra))
                self.assertEqual(
                    n,
                    0,
                    f"{cls_name}.{meth} built {n} thread pool(s); it was "
                    "measured slower threaded and must run serially",
                )

    def test_create_mask_keeps_threading(self):
        """The deliberate hold-out: it is the only decoration that measured
        FASTER threaded, so a well-meaning sweep must not strip it too."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")
        import pythontk as ptk

        imgs = [Image.new("RGBA", (8, 8), (1, 2, 3, 255)) for _ in range(2)]
        n = self._count_pools(lambda: ptk.ImgUtils.create_mask(imgs, (1, 2, 3, 255)))
        self.assertEqual(n, 1, "create_mask must stay threaded -- it is 4.8x faster")

    def test_a_scalar_call_never_builds_a_pool(self):
        """Pre-existing guard, pinned here because the split above is only
        safe while it holds: a single item has nothing to parallelize."""
        import pythontk as ptk

        self.assertEqual(self._count_pools(lambda: ptk.StrUtils.set_case("solo")), 0)


class TestListifyAnnotations(unittest.TestCase):
    """``@listify`` changes the published return type, and the annotations
    described the undecorated body.

    Given a list the wrapper returns a list, so ``-> str`` on a listified
    function is simply false -- and ``inspect.signature``, ``help()`` and
    ``API_REGISTRY.md`` all republish it. ``format_path`` had the right shape
    (``Union[str, List[str]]``) the whole time, so this is drift, not an
    unsolved question.
    """

    @staticmethod
    def _listified():
        """Every ``@listify``-decorated function in the package, by AST."""
        found = []
        for root, _, files in os.walk(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "pythontk")
        ):
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(root, name)
                with open(path, encoding="utf-8") as fh:
                    tree = ast.parse(fh.read())
                for node in ast.walk(tree):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if any("listify" in ast.unparse(d) for d in node.decorator_list):
                        found.append((path, node))
        return found

    def test_there_are_listified_functions_to_check(self):
        """A zero-sample sweep is not a pass."""
        self.assertGreaterEqual(len(self._listified()), 8)

    def test_every_listified_function_is_annotated(self):
        missing = [
            f"{os.path.basename(p)}::{n.name}"
            for p, n in self._listified()
            if n.returns is None
        ]
        self.assertEqual(missing, [], f"listified but unannotated: {missing}")

    def test_no_listified_function_promises_a_scalar_only_return(self):
        lying = []
        for path, node in self._listified():
            if node.returns is None:
                continue
            annotation = ast.unparse(node.returns)
            if "List[" not in annotation and "list" not in annotation:
                lying.append(f"{os.path.basename(path)}::{node.name} -> {annotation}")
        self.assertEqual(
            lying, [], f"scalar-only annotation on a listified fn: {lying}"
        )

    def test_the_wrapper_really_does_return_a_list(self):
        """The premise, measured rather than assumed."""
        from pythontk import FileUtils

        one = FileUtils.convert_to_relative_path("C:/a/b/c.png", "C:/a")
        many = FileUtils.convert_to_relative_path(
            ["C:/a/b/c.png", "C:/a/b/d.png"], "C:/a"
        )
        self.assertIsInstance(one, str)
        self.assertIsInstance(many, list)
        self.assertEqual(len(many), 2)


if __name__ == "__main__":
    unittest.main()
