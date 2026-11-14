#!/usr/bin/python
# coding=utf-8
"""Unit tests for pythontk UpstreamPatch.

Run with:
    python -m pytest test_upstream_patch.py -v
    python test_upstream_patch.py
"""

import json
import unittest

import pythontk as ptk
from pythontk.core_utils.upstream_patch import UpstreamPatch

from conftest import BaseTestCase


class _Target:
    """Stands in for the third-party attribute a patch replaces."""

    @staticmethod
    def method(value):
        return f"stock:{value}"

    @classmethod
    def made_by(cls):
        return cls.__name__

    def greet(self, value):
        return f"hello:{value}"


class _SubTarget(_Target):
    """Inherits every attribute above; owns none of them."""


#: This module as ``sys.modules`` knows it. A target names the module the way an
#: import does, so a test that hard-codes its own basename would patch a SECOND
#: copy of this file under a runner that imports it as a package member.
_HERE = __name__


def _patch(name, target=None, **kwargs):
    """A patch registered under a name this test owns, cleaned up by the case."""
    kwargs.setdefault("reason", "a defect, described")
    return UpstreamPatch(
        name=name, target=target or f"{_HERE}:_Target.method", **kwargs
    )


class UpstreamPatchTest(BaseTestCase):
    """Declaring, applying and retiring a patch."""

    def tearDown(self):
        # The registry is process-wide by design (the sweep walks it), so a test
        # that leaves entries behind changes what the next one sees.
        for name in [n for n in UpstreamPatch._registry if n.startswith("test:")]:
            del UpstreamPatch._registry[name]
        super().tearDown()

    def test_the_replacement_runs_and_is_handed_the_original(self):
        patch = _patch("test:basic")

        @patch.replaces
        def replacement(original, value):
            return f"patched({original(value)})"

        self.assertEqual(_Target.method("x"), "stock:x")
        with patch.applied() as active:
            self.assertTrue(active)
            self.assertEqual(_Target.method("x"), "patched(stock:x)")
        self.assertEqual(_Target.method("x"), "stock:x")

    def test_a_staticmethod_target_stays_a_staticmethod(self):
        """Restored from ``getattr``, the target came back a PLAIN function:
        every instance call after the first block raised a TypeError, for the
        rest of the process."""
        patch = _patch("test:static")
        patch.replaces(lambda original, value: f"p({original(value)})")
        with patch.applied():
            self.assertEqual(_Target().method("x"), "p(stock:x)")
            self.assertEqual(_Target.method("x"), "p(stock:x)")
        self.assertIs(type(vars(_Target)["method"]), staticmethod)
        self.assertEqual(_Target().method("x"), "stock:x")

    def test_a_classmethod_target_binds_the_class_it_was_called_on(self):
        patch = _patch("test:classmethod", target=f"{_HERE}:_Target.made_by")
        patch.replaces(lambda original: f"p({original()})")
        with patch.applied():
            self.assertEqual(_SubTarget.made_by(), "p(_SubTarget)")
        self.assertIs(type(vars(_Target)["made_by"]), classmethod)
        self.assertEqual(_SubTarget.made_by(), "_SubTarget")

    def test_an_inherited_target_is_handed_back_to_inheritance(self):
        """Patched through a subclass that only inherits it, the attribute was
        restored ONTO the subclass -- where a later patch of the base class no
        longer reached it."""
        patch = _patch("test:inherited", target=f"{_HERE}:_SubTarget.greet")
        patch.replaces(lambda original, self, value: f"p({original(self, value)})")
        with patch.applied():
            self.assertEqual(_SubTarget().greet("x"), "p(hello:x)")
            self.assertEqual(_Target().greet("x"), "hello:x")
        self.assertNotIn("greet", vars(_SubTarget))
        self.assertEqual(_SubTarget().greet("x"), "hello:x")

    def test_the_original_is_restored_when_the_block_raises(self):
        patch = _patch("test:raising")
        patch.replaces(lambda original, value: "patched")

        with self.assertRaises(ValueError):
            with patch.applied():
                raise ValueError("the block failed")
        self.assertEqual(_Target.method("x"), "stock:x")

    def test_nesting_restores_to_what_each_block_found(self):
        patch = _patch("test:nested")
        patch.replaces(lambda original, value: f"p({original(value)})")

        with patch.applied():
            self.assertEqual(_Target.method("x"), "p(stock:x)")
            with patch.applied():
                self.assertEqual(_Target.method("x"), "p(p(stock:x))")
            self.assertEqual(_Target.method("x"), "p(stock:x)")
        self.assertEqual(_Target.method("x"), "stock:x")

    def test_a_target_this_host_does_not_ship_is_a_no_op(self):
        """The whole point of resolving late: a patch for a package the host does
        not have must not be an import error at declaration, and must not make
        every call site ask whether it applies."""
        patch = _patch("test:absent", target="no_such_module_anywhere:thing")
        patch.replaces(lambda original: None)

        self.assertFalse(patch.available)
        with patch.applied() as active:
            self.assertFalse(active)

    def test_an_attribute_the_owner_lost_is_a_no_op_too(self):
        patch = _patch("test:gone", target=f"{_HERE}:_Target.renamed_away")
        patch.replaces(lambda original: None)

        self.assertFalse(patch.available)
        with patch.applied() as active:
            self.assertFalse(active)

    def test_a_patch_with_no_replacement_declared_does_nothing(self):
        patch = _patch("test:declared-only")

        with patch.applied() as active:
            self.assertFalse(active)
            self.assertEqual(_Target.method("x"), "stock:x")

    def test_the_probe_answers_whether_upstream_still_has_the_defect(self):
        """What retires a patch. The sweep test asserts this is True; the release
        that fixes the defect upstream makes it False, which fails that sweep and
        names the patch to delete."""
        patch = _patch("test:probed")

        @patch.detects
        def still_broken():
            return _Target.method("x") == "stock:x"

        self.assertTrue(patch.still_needed())

    def test_a_patch_with_no_probe_cannot_claim_to_be_needed(self):
        patch = _patch("test:unprobed")
        with self.assertRaises(RuntimeError) as caught:
            patch.still_needed()
        self.assertIn("no probe", str(caught.exception))

    def test_a_probe_cannot_be_asked_about_a_target_that_is_not_here(self):
        patch = _patch("test:absent-probe", target="no_such_module_anywhere:thing")
        patch.detects(lambda: True)
        with self.assertRaises(RuntimeError) as caught:
            patch.still_needed()
        self.assertIn("does not resolve", str(caught.exception))

    def test_a_patch_must_say_what_it_corrects(self):
        with self.assertRaises(ValueError):
            UpstreamPatch(name="test:reasonless", target="json:dumps", reason="")

    def test_a_malformed_target_is_refused_at_declaration(self):
        with self.assertRaises(ValueError):
            _patch("test:bad-target", target="json.dumps")

    def test_redeclaring_the_same_patch_is_a_reload_not_a_collision(self):
        """Module reloads re-run declarations (``ptk.ModuleReloader``, and the DCC
        harnesses that purge a package between modules). Raising there would turn a
        reload into an ImportError, so same name AND same target replaces."""
        first = _patch("test:reloaded")
        second = _patch("test:reloaded")

        self.assertIsNot(first, second)
        self.assertIs(UpstreamPatch._registry["test:reloaded"], second)

    def test_the_same_name_against_a_different_target_is_a_collision(self):
        _patch("test:collide")
        with self.assertRaises(ValueError) as caught:
            _patch("test:collide", target="json:loads")
        self.assertIn("already declared", str(caught.exception))

    def test_declaring_two_replacements_or_probes_is_refused(self):
        patch = _patch("test:twice")
        patch.replaces(lambda original: None)
        with self.assertRaises(ValueError):
            patch.replaces(lambda original: None)
        patch.detects(lambda: True)
        with self.assertRaises(ValueError):
            patch.detects(lambda: True)

    def test_the_registry_reports_what_has_been_declared(self):
        patch = _patch("test:listed")
        self.assertIn(patch, UpstreamPatch.registry())

    def test_it_patches_a_plain_module_attribute_too(self):
        """Not just methods: the target is any dotted attribute of any module."""
        patch = _patch("test:module-level", target="json:dumps")
        patch.replaces(lambda original, obj, **kw: "[" + original(obj, **kw) + "]")

        with patch.applied():
            self.assertEqual(json.dumps({"a": 1}), '[{"a": 1}]')
        self.assertEqual(json.dumps({"a": 1}), '{"a": 1}')

    def test_it_is_reachable_from_the_package_root(self):
        self.assertIs(ptk.UpstreamPatch, UpstreamPatch)


if __name__ == "__main__":
    unittest.main()
