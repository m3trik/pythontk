#!/usr/bin/python
# coding=utf-8
"""Tests for pythontk PackageRequirement: a pip requirement read without ``packaging``.

The ecosystem states version ranges for packages it installs on demand
(``extapps>=0.8.0,<0.9``), and ``packaging`` is not guaranteed inside a DCC's
interpreter, so the parse and the comparison are pinned here against what pip
itself would answer.
"""

import unittest
from unittest import mock

from pythontk.core_utils.package_requirement import PackageRequirement

from conftest import BaseTestCase


class ParseTest(BaseTestCase):
    def test_a_name_and_a_range(self):
        req = PackageRequirement("extapps>=0.8.0,<0.9")
        self.assertEqual(req.name, "extapps")
        self.assertEqual(req.specifiers, ((">=", "0.8.0"), ("<", "0.9")))
        self.assertEqual(req.extras, ())
        self.assertEqual(req.marker, "")
        # The text pip gets is the text the caller wrote.
        self.assertEqual(str(req), "extapps>=0.8.0,<0.9")

    def test_spacing_extras_and_a_marker(self):
        req = PackageRequirement(' my_pkg [a, b] >= 1.2 , != 1.3 ; extra == "apps" ')
        self.assertEqual(req.name, "my_pkg")
        self.assertEqual(req.extras, ("a", "b"))
        self.assertEqual(req.specifiers, ((">=", "1.2"), ("!=", "1.3")))
        # Kept as text, never evaluated.
        self.assertEqual(req.marker, 'extra == "apps"')

    def test_the_parenthesized_form_metadata_can_carry(self):
        req = PackageRequirement("extapps (<0.9,>=0.8.0)")
        self.assertEqual(req.specifiers, (("<", "0.9"), (">=", "0.8.0")))

    def test_a_bare_name_has_no_specifiers(self):
        self.assertEqual(PackageRequirement("extapps").specifiers, ())

    def test_what_it_does_not_read_raises(self):
        for spec in (
            "pkg @ https://example.com/pkg.whl",  # a direct reference
            "git+https://example.com/pkg.git",
            "pkg===1.0",  # arbitrary equality
            "pkg==1.0.*",  # a wildcard
            "pkg~=1",  # ~= needs two release segments
            "pkg>=",
            "pkg>=1.0+local",  # a local label in a specifier
            "",
        ):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                PackageRequirement(spec)


class ContainsTest(BaseTestCase):
    def test_the_floor_and_next_minor_cap(self):
        req = PackageRequirement("extapps>=0.8.0,<0.9")
        self.assertTrue(req.contains("0.8.0"))
        self.assertTrue(req.contains("0.8.17"))
        self.assertFalse(req.contains("0.7.9"))
        self.assertFalse(req.contains("0.9.0"))
        self.assertFalse(req.contains("1.0"))

    def test_a_pre_release_of_the_cap_is_outside_it(self):
        """PEP 440: ``<V`` admits no pre-release of V itself. Plain ordering
        would let ``0.9.0rc1`` through a ``<0.9`` cap -- the very minor the cap
        exists to keep out."""
        req = PackageRequirement("extapps>=0.8.0,<0.9")
        self.assertFalse(req.contains("0.9.0rc1"))
        self.assertFalse(req.contains("0.9.0.dev3"))
        # A pre-release INSIDE the range satisfies it, as pip treats one installed.
        self.assertTrue(req.contains("0.8.5rc1"))
        # ... but the floor's own pre-release sorts below the floor.
        self.assertFalse(req.contains("0.8.0rc1"))

    def test_a_post_release_of_an_exclusive_floor_is_outside_it(self):
        req = PackageRequirement("pkg>1.7")
        self.assertFalse(req.contains("1.7.post1"))
        self.assertTrue(req.contains("1.7.1"))

    def test_equality_pads_the_release(self):
        req = PackageRequirement("pkg==1.0")
        self.assertTrue(req.contains("1.0.0"))
        self.assertFalse(req.contains("1.0.1"))
        self.assertFalse(PackageRequirement("pkg!=1.0").contains("1.0.0"))
        # A local label never orders.
        self.assertTrue(req.contains("1.0+build.7"))

    def test_compatible_release(self):
        req = PackageRequirement("pkg~=0.8.2")
        self.assertTrue(req.contains("0.8.2"))
        self.assertTrue(req.contains("0.8.9"))
        self.assertFalse(req.contains("0.8.1"))
        self.assertFalse(req.contains("0.9.0"))
        self.assertTrue(PackageRequirement("pkg~=0.8").contains("0.99"))
        self.assertFalse(PackageRequirement("pkg~=0.8").contains("1.0"))

    def test_inclusive_bounds(self):
        self.assertTrue(PackageRequirement("pkg<=1.0").contains("1.0.0"))
        self.assertFalse(PackageRequirement("pkg<=1.0").contains("1.0.post1"))
        self.assertTrue(PackageRequirement("pkg>=1.0").contains("1.0"))

    def test_an_unreadable_version_satisfies_no_range(self):
        req = PackageRequirement("pkg>=0.1")
        for version in (None, "", "unknown", "1.x"):
            with self.subTest(version=version):
                self.assertFalse(req.contains(version))

    def test_a_bare_name_takes_any_version(self):
        req = PackageRequirement("pkg")
        self.assertTrue(req.contains("0.0.1"))
        self.assertTrue(req.contains(None))


class VersionOrderTest(BaseTestCase):
    def test_pre_releases_order_below_their_release(self):
        key = PackageRequirement.version_key
        ordered = [
            "0.9.0.dev1",
            "0.9.0a1",
            "0.9.0b2",
            "0.9.0rc1",
            "0.9.0",
            "0.9.0.post1",
            "0.9.1",
            "1!0.1",  # an epoch outranks every release
        ]
        keys = [key(v) for v in ordered]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(key("1.0"), key("1.0.0"))
        self.assertEqual(key("1.0RC1"), key("1.0rc1"))
        self.assertEqual(key("1.0-1"), key("1.0.post1"))

    def test_a_non_version_raises(self):
        with self.assertRaises(ValueError):
            PackageRequirement.version_key("latest")

    def test_release_reads_the_leading_segment_and_never_raises(self):
        release = PackageRequirement.release
        self.assertEqual(release("0.0.8"), (0, 0, 8))
        self.assertEqual(release("1.2.0rc1"), (1, 2, 0))
        self.assertEqual(release("1.2.x"), (1, 2))
        self.assertEqual(release(""), ())
        self.assertEqual(release(None), ())


class DeclaredByTest(BaseTestCase):
    """What a distribution's own metadata says it needs: the one place a host's
    range lives, so the code reading it cannot drift from the pyproject a
    release ratchets."""

    ROWS = [
        "pythontk>=0.18.2",
        'mayatk>=0.29.2; extra == "maya"',
        'extapps<0.9,>=0.8.0; extra == "apps"',
        'extapps>=0.1; python_version < "3.0"',
    ]

    def _declared(self, *args, rows=ROWS, **kwargs):
        with mock.patch("importlib.metadata.requires", return_value=rows):
            return PackageRequirement.declared_by(*args, **kwargs)

    def test_an_extras_row_without_the_marker_that_selected_it(self):
        req = self._declared("tentacletk", "extapps", extra="apps")
        # pip evaluates a marker in ITS environment, where no extra is active:
        # handed the row as written, it would skip the install.
        self.assertEqual(str(req), "extapps<0.9,>=0.8.0")
        self.assertTrue(req.contains("0.8.3"))

    def test_without_an_extra_only_an_unconditional_row_counts(self):
        self.assertEqual(
            str(self._declared("tentacletk", "pythontk")), "pythontk>=0.18.2"
        )
        self.assertIsNone(self._declared("tentacletk", "mayatk"))

    def test_names_compare_by_pep_503(self):
        req = self._declared("tentacletk", "ExtApps", extra="APPS")
        self.assertEqual(req.name, "extapps")

    def test_an_absent_distribution_declares_nothing(self):
        from importlib.metadata import PackageNotFoundError

        with mock.patch(
            "importlib.metadata.requires", side_effect=PackageNotFoundError("x")
        ):
            self.assertIsNone(PackageRequirement.declared_by("nope", "extapps"))
        self.assertIsNone(self._declared("tentacletk", "extapps", rows=None))


if __name__ == "__main__":
    unittest.main()
