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

import fnmatch
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


class TestDataFileParity(unittest.TestCase):
    """The sdist and the wheel must ship the same non-``.py`` files.

    Two independent declarations decide that -- ``recursive-include`` in
    ``MANIFEST.in`` for the sdist, ``[tool.setuptools.package-data]`` in
    ``pyproject.toml`` for the wheel -- and nothing kept them in step. They
    had already drifted: the wheel listed ``*.md`` and ``*.txt`` while the
    sdist did not, so ``pythontk/core_utils/README.md`` and its two siblings
    shipped in one channel and not the other; the sdist listed ``*.png``,
    which the wheel did not.

    Four runtime data files resolve relative to ``__file__`` and two of them
    degrade silently when absent, so a file that misses a channel is not a
    packaging nicety -- it is a feature that quietly stops working for whoever
    installed from that channel.
    """

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @staticmethod
    def _sdist_patterns(text):
        return sorted(
            {
                p
                for line in re.findall(r"recursive-include\s+\S+\s+(.*)", text)
                for p in line.split()
            }
        )

    @staticmethod
    def _wheel_patterns(text):
        block = re.search(
            r"\[tool\.setuptools\.package-data\]\s*\n(.*?)(?=\n\[|\Z)", text, re.S
        )
        return sorted(set(re.findall(r'"(\*\.[A-Za-z0-9]+)"', block.group(1))))

    def _patterns(self):
        with open(os.path.join(self.ROOT, "MANIFEST.in"), encoding="utf-8") as f:
            sdist = self._sdist_patterns(f.read())
        with open(os.path.join(self.ROOT, "pyproject.toml"), encoding="utf-8") as f:
            wheel = self._wheel_patterns(f.read())
        return sdist, wheel

    def _data_files(self):
        pkg = os.path.join(self.ROOT, "pythontk")
        return [
            os.path.relpath(os.path.join(d, f), pkg).replace(os.sep, "/")
            for d, _, fs in os.walk(pkg)
            for f in fs
            if not f.endswith((".py", ".pyc")) and "__pycache__" not in d
        ]

    def test_the_two_channels_declare_the_same_patterns(self):
        sdist, wheel = self._patterns()
        self.assertTrue(sdist, "no recursive-include patterns found in MANIFEST.in")
        self.assertEqual(
            sdist,
            wheel,
            "MANIFEST.in (sdist) and package-data (wheel) disagree; a pattern "
            f"in only one ships to only one channel.\n  only sdist: "
            f"{sorted(set(sdist) - set(wheel))}\n  only wheel: "
            f"{sorted(set(wheel) - set(sdist))}",
        )

    def test_every_shipped_data_file_matches_a_pattern(self):
        """A new data-file extension has to be declared, not discovered in a
        bug report from whoever installed the release."""
        sdist, wheel = self._patterns()
        files = self._data_files()
        self.assertTrue(files, "no non-.py files found under pythontk/")
        for rel in sorted(files):
            base = os.path.basename(rel)
            with self.subTest(file=rel):
                self.assertTrue(
                    any(fnmatch.fnmatch(base, p) for p in sdist),
                    f"{rel} is not covered by any MANIFEST.in pattern; it "
                    "would be missing from the sdist",
                )
                self.assertTrue(
                    any(fnmatch.fnmatch(base, p) for p in wheel),
                    f"{rel} is not covered by any package-data pattern; it "
                    "would be missing from the wheel",
                )


class TestNoDccImports(unittest.TestCase):
    """pythontk is the bottom of the stack: DCC-agnostic and zero-dep.

    The only guard on that was a substring search over ONE file
    (``test_map_compositor.TestEnginePurity``), covering Qt and nothing else,
    across 1 of 142 modules -- and a substring match reads its own comments and
    docstrings as violations while missing ``import  maya`` with two spaces.
    This walks the AST of every module instead.

    Two different rules, because the two hazards differ:

    * A **DCC** module (``maya``, ``bpy``, ``nuke`` ...) must not appear at
      all. pythontk has no business talking to one, at any scope, and today
      not one module does.
    * A **Qt binding** must not be imported at MODULE level, because that is
      what makes the module unimportable without Qt. Resolving one lazily
      inside a function, tolerantly, is the sanctioned pattern -- exactly what
      ``net_utils/rpc/plugin_core._qtcore`` does so the RPC server stays
      importable in a plain interpreter.
    """

    PACKAGE = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pythontk"
    )
    DCC = frozenset(
        {
            "maya",
            "pymel",
            "bpy",
            "bmesh",
            "nuke",
            "MaxPlus",
            "pymxs",
            "unreal",
            "hou",
            "substance_painter",
            "sd",
            "krita",
        }
    )
    QT = frozenset(
        {"PySide2", "PySide6", "PyQt5", "PyQt6", "qtpy", "shiboken2", "shiboken6"}
    )

    def _modules(self):
        import ast

        for d, _, fs in os.walk(self.PACKAGE):
            if "__pycache__" in d:
                continue
            for f in sorted(fs):
                if not f.endswith(".py"):
                    continue
                path = os.path.join(d, f)
                with open(path, encoding="utf-8") as fh:
                    source = fh.read()
                rel = os.path.relpath(path, self.PACKAGE).replace(os.sep, "/")
                yield rel, ast.parse(source, filename=path)

    @staticmethod
    def _imports(tree, top_level_only):
        """``(lineno, root_module)`` for each import in *tree*.

        With *top_level_only*, only imports in the module body -- the ones that
        run on ``import pythontk`` -- counting a ``try:`` / ``if`` wrapper at
        module level as still module level, since it executes just the same.
        """
        import ast

        def walk(nodes):
            for node in nodes:
                if isinstance(node, ast.Import):
                    for a in node.names:
                        yield node.lineno, a.name.split(".")[0]
                elif isinstance(node, ast.ImportFrom):
                    # A relative import has no root package name to judge.
                    if node.level == 0 and node.module:
                        yield node.lineno, node.module.split(".")[0]
                elif top_level_only:
                    if isinstance(node, (ast.Try, ast.If, ast.With)):
                        yield from walk(ast.iter_child_nodes(node))
                else:
                    yield from walk(ast.iter_child_nodes(node))

        return list(walk(tree.body if top_level_only else [tree]))

    def test_no_module_imports_a_dcc_at_any_scope(self):
        offenders = []
        scanned = 0
        for rel, tree in self._modules():
            scanned += 1
            for lineno, root in self._imports(tree, top_level_only=False):
                if root in self.DCC:
                    offenders.append(f"{rel}:{lineno} imports {root}")
        self.assertGreater(scanned, 100, f"only {scanned} modules scanned; walk broke")
        self.assertEqual(
            offenders, [], "pythontk must not import a DCC:\n" + "\n".join(offenders)
        )

    def test_no_module_imports_qt_at_module_level(self):
        offenders = []
        scanned = 0
        for rel, tree in self._modules():
            scanned += 1
            for lineno, root in self._imports(tree, top_level_only=True):
                if root in self.QT:
                    offenders.append(f"{rel}:{lineno} imports {root}")
        self.assertGreater(scanned, 100, f"only {scanned} modules scanned; walk broke")
        self.assertEqual(
            offenders,
            [],
            "Qt at module level makes the module unimportable without Qt; "
            "resolve it lazily inside a function instead:\n" + "\n".join(offenders),
        )

    def test_the_walk_actually_sees_imports(self):
        """A guard that scans nothing passes forever. Prove the walker finds
        the deferred Qt import it is meant to tolerate."""
        import ast

        path = os.path.join(self.PACKAGE, "net_utils", "rpc", "plugin_core.py")
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        deep = {root for _, root in self._imports(tree, top_level_only=False)}
        top = {root for _, root in self._imports(tree, top_level_only=True)}
        self.assertTrue(self.QT & deep, "function-scoped Qt import not detected")
        self.assertFalse(self.QT & top, "that Qt import is not module level")


if __name__ == "__main__":
    unittest.main()
