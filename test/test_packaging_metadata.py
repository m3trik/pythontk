#!/usr/bin/python
# coding=utf-8
"""Static guard on the *declared* dependency bounds in ``pyproject.toml``.

pythontk is installed into interpreters it does not own -- Maya's, Blender's,
Metashape's -- and a version bound here is resolved against whatever that host
already ships. That makes the bounds a compatibility contract rather than a
packaging detail, and the ``numpy`` one in particular is a decision that was
argued out and must not be re-made by accident:

* A **floor** is safe only while it sits at or below the oldest release any
  supported host ships (measured 2026-08-06: Maya 2025 = numpy 1.24.4, Blender
  5.1 = numpy 2.3.4). Raise it past that and ``pip install`` inside Maya starts
  *upgrading the host's* numpy, which is not ours to re-resolve.
* A **ceiling** cannot be expressed here at all. Environment markers see the
  interpreter, never the host application, and the two hosts want opposite
  bounds -- ``numpy<2`` would force a downgrade inside Blender. The
  no-host-numpy case (a ``--target`` install into an empty directory, where pip
  resolves 2.x and 1.x-compiled extensions then refuse to load) is handled at
  install time with ``--no-deps``, documented under Install in
  ``docs/README.md``.

Run with::

    python -m pytest test_packaging_metadata.py -v
"""
import os
import re
import unittest

try:  # stdlib from 3.11; the repo's own floor is 3.9
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - depends on the runner
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

PYPROJECT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pyproject.toml")

#: The oldest numpy any supported host ships. A floor above this would make
#: ``pip install pythontk`` upgrade the host's own numpy -- see the module
#: docstring. Bump only alongside a re-survey of the hosts.
OLDEST_HOST_NUMPY = (1, 24, 4)


@unittest.skipIf(tomllib is None, "no TOML parser available on this interpreter")
class PackagingMetadataTestCase(unittest.TestCase):
    """Pin the dependency-bound decisions so a future edit has to be deliberate."""

    @staticmethod
    def _dependencies():
        """The declared runtime requirement strings."""
        with open(PYPROJECT, "rb") as f:
            return tomllib.load(f)["project"]["dependencies"]

    @staticmethod
    def _specifier(requirements, name):
        """The version-specifier half of *name*'s requirement (``""`` when bare).

        Hand-parsed rather than via ``packaging``: this file guards the
        dependency list, so it must not need anything off that list to run.
        """
        for requirement in requirements:
            # name [extras] specifier ; marker
            head = requirement.split(";", 1)[0].strip()
            match = re.match(r"^([A-Za-z0-9._-]+)\s*(\[[^\]]*\])?\s*(.*)$", head)
            if match and match.group(1).lower() == name.lower():
                return match.group(3).strip()
        raise AssertionError(f"{name!r} is not declared in {requirements!r}")

    @staticmethod
    def _lower_bound(specifier):
        """The ``>=`` / ``>`` floor as a version tuple, or ``None`` if unbounded."""
        match = re.search(r">=?\s*([0-9]+(?:\.[0-9]+)*)", specifier)
        if not match:
            return None
        return tuple(int(part) for part in match.group(1).split("."))

    @staticmethod
    def _has_upper_bound(specifier):
        """Whether *specifier* caps the version (``<``, ``<=``, ``==``, ``~=``)."""
        return bool(re.search(r"(?<![!>])<|~=|===?", specifier))

    def test_numpy_is_declared_with_a_floor(self):
        """A bare ``numpy`` states no minimum API at all -- record the real one."""
        specifier = self._specifier(self._dependencies(), "numpy")
        self.assertIsNotNone(
            self._lower_bound(specifier),
            f"numpy requirement {specifier!r} declares no lower bound",
        )

    def test_pillow_is_declared_with_a_floor(self):
        """``Image.Resampling`` / ``Image.Palette`` are Pillow 9.1 enums.

        Declared bare, pip is free to resolve an older Pillow, and the failure
        is an AttributeError at the call site rather than at install -- the
        same class of break the numpy floor above exists to prevent.
        """
        specifier = self._specifier(self._dependencies(), "Pillow")
        floor = self._lower_bound(specifier)
        self.assertIsNotNone(
            floor,
            f"Pillow requirement {specifier!r} declares no lower bound",
        )
        self.assertGreaterEqual(
            floor,
            (9, 1),
            f"Pillow floor {floor} predates the Image.Resampling/Image.Palette "
            f"enums the image utils call unguarded",
        )

    def test_the_numpy_floor_does_not_outrank_the_oldest_host(self):
        """A floor above Maya 2025's 1.24.4 would upgrade the host's own numpy."""
        specifier = self._specifier(self._dependencies(), "numpy")
        floor = self._lower_bound(specifier)
        # Checked rather than left to blow up in the comparison: an absent floor
        # is the sibling test's failure, and this one should say so plainly
        # instead of raising a TypeError on `None <= tuple`.
        if floor is None:
            self.skipTest(f"numpy requirement {specifier!r} declares no floor")
        self.assertLessEqual(
            floor,
            OLDEST_HOST_NUMPY,
            f"numpy floor {floor} is newer than the oldest supported host ships "
            f"{OLDEST_HOST_NUMPY}; it would re-resolve numpy inside that DCC",
        )

    def test_numpy_carries_no_ceiling(self):
        """Deliberate, not an oversight: the hosts want *opposite* ceilings.

        Maya 2025 needs ``<2`` (its extensions are compiled against 1.x) and
        Blender 5.1 needs ``>=2`` (it ships 2.3.4) -- and no environment marker
        can tell them apart, because markers see the interpreter and not the
        application hosting it. Whoever adds a ceiling here breaks one host to
        fix the other; the install-time answer is ``--no-deps``.
        """
        specifier = self._specifier(self._dependencies(), "numpy")
        self.assertFalse(
            self._has_upper_bound(specifier),
            f"numpy requirement {specifier!r} caps the version; see this test's "
            f"docstring before removing it",
        )

    def test_the_bound_readers_reject_the_shapes_they_replaced(self):
        """The guard above only has teeth if its readers can see both defects.

        A bare ``numpy`` (what shipped until 2026-08-17) must read as unbounded,
        and every ceiling spelling must read as a ceiling -- otherwise the three
        assertions above would pass on exactly the declarations they exist to
        catch.
        """
        self.assertIsNone(self._lower_bound(self._specifier(["numpy"], "numpy")))
        self.assertIsNotNone(
            self._lower_bound(self._specifier(["numpy>=1.24"], "numpy"))
        )
        for capped in ("<2", ">=1.24,<2", "==1.24.4", "~=1.24"):
            self.assertTrue(self._has_upper_bound(capped), capped)
        for uncapped in ("", ">=1.24", ">1.23", ">=1.24,!=1.25.0"):
            self.assertFalse(self._has_upper_bound(uncapped), uncapped)

    def test_a_requirement_with_extras_or_a_marker_still_resolves(self):
        """The parser must not be fooled by the other legal requirement shapes."""
        requirements = ["Pillow", "numpy[all] >= 1.24 ; python_version >= '3.9'"]
        self.assertEqual(self._specifier(requirements, "numpy"), ">= 1.24")


if __name__ == "__main__":
    unittest.main()
