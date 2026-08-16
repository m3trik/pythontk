# !/usr/bin/python
# coding=utf-8
"""Tests for ``core_utils/doc_audit.py`` — and the README rot gates built on it.

Two layers, deliberately in one file:

1. Unit tests for the ``DocAudit`` primitive itself.
2. The documentation gates: every fenced python block in the repo's
   hand-written READMEs must bind to the live API, and every literal output
   the front-door README claims must both be true and still appear in the
   document. Editing an example or renaming an API without updating the
   docs fails here, not in a user's session.
"""
import types
import unittest
from pathlib import Path

import pythontk as ptk
from pythontk import DocAudit

REPO_ROOT = Path(__file__).resolve().parents[1]

# Hand-written docs that carry live-API code examples (or may grow them).
AUDITED_DOCS = [
    REPO_ROOT / "docs" / "README.md",
    REPO_ROOT / "pythontk" / "core_utils" / "README.md",
    REPO_ROOT / "pythontk" / "file_utils" / "README.md",
    REPO_ROOT / "pythontk" / "net_utils" / "README.md",
]


class TestDocAuditPrimitive(unittest.TestCase):
    """The general markdown-example audit primitive."""

    def test_extracts_only_matching_language_blocks(self):
        md = (
            "intro\n"
            "```python\na = 1\n```\n"
            "```bash\nls -la\n```\n"
            "```python\nb = 2\n```\n"
        )
        blocks = DocAudit.extract_code_blocks(md)
        self.assertEqual(blocks, ["a = 1\n", "b = 2\n"])
        self.assertEqual(DocAudit.extract_code_blocks(md, lang="bash"), ["ls -la\n"])

    def test_flags_missing_attribute(self):
        problems = DocAudit.audit_code("import pythontk as ptk\nptk.NoSuchThing")
        self.assertEqual(len(problems), 1)
        self.assertIn("no attribute 'NoSuchThing'", problems[0])

    def test_flags_unknown_keyword_argument(self):
        problems = DocAudit.audit_code("ptk.filter_list([1], not_a_param=2)")
        self.assertEqual(len(problems), 1)
        self.assertIn("no parameter 'not_a_param'", problems[0])

    def test_flags_missing_from_import_name(self):
        problems = DocAudit.audit_code("from pythontk import DefinitelyMissingName")
        self.assertEqual(len(problems), 1)
        self.assertIn("DefinitelyMissingName", problems[0])

    def test_flags_syntax_error(self):
        problems = DocAudit.audit_code("def broken(:\n    pass")
        self.assertEqual(len(problems), 1)
        self.assertIn("syntax error", problems[0])

    def test_skips_unknown_roots_and_local_names(self):
        code = (
            "result = mystery_helper(anything=1)\n"  # unknown root: skipped
            "class Local:\n"
            "    pass\n"
            "Local.does_not_exist(whatever=2)\n"  # locally defined: skipped
        )
        self.assertEqual(DocAudit.audit_code(code), [])

    def test_var_keyword_callables_accept_any_kwarg(self):
        roots = {"m": types.SimpleNamespace(f=lambda **kw: None)}
        self.assertEqual(DocAudit.audit_code("m.f(anything=1)", roots), [])

    def test_instance_call_chain_resolves_through_the_class(self):
        # ptk.FrameExtractor().extract_frames_sharpest(window_sec=...) — the
        # instance stands in as its class for member + signature checks.
        good = "ptk.FrameExtractor().extract_frames_sharpest('v.mp4', 'o/', window_sec=1.0)"
        self.assertEqual(DocAudit.audit_code(good), [])
        bad = "ptk.FrameExtractor().extract_frames_sharpest('v.mp4', 'o/', not_real=1)"
        self.assertEqual(len(DocAudit.audit_code(bad)), 1)

    def test_valid_kwargs_pass(self):
        code = (
            "import pythontk as ptk\n"
            "ptk.filter_list(['a_x', 'b'], inc=['a_*'], exc=['*_y'])\n"
        )
        self.assertEqual(DocAudit.audit_code(code), [])

    def test_non_import_error_during_import_probing_does_not_crash_the_audit(self):
        """Only ImportError means "optional dep, stays unknown" -- any other
        import-time exception (RuntimeError/OSError from an optional-dep
        module) must not crash the whole audit; the name should just stay
        unknown, same as an ImportError."""
        import importlib
        from unittest import mock

        real_import_module = importlib.import_module

        def _boom(name, *a, **kw):
            if name == "definitely_not_a_real_module_boom":
                raise RuntimeError("optional dep blew up at import time")
            return real_import_module(name, *a, **kw)

        with mock.patch(
            "pythontk.core_utils.doc_audit.importlib.import_module",
            side_effect=_boom,
        ):
            problems = DocAudit.audit_code(
                "import definitely_not_a_real_module_boom\n"
                "definitely_not_a_real_module_boom.thing"
            )
        self.assertEqual(problems, [])

    def test_non_import_error_on_from_import_does_not_crash_the_audit(self):
        import importlib
        from unittest import mock

        real_import_module = importlib.import_module

        def _boom(name, *a, **kw):
            if name == "definitely_not_a_real_module_boom":
                raise OSError("optional dep blew up at import time")
            return real_import_module(name, *a, **kw)

        with mock.patch(
            "pythontk.core_utils.doc_audit.importlib.import_module",
            side_effect=_boom,
        ):
            problems = DocAudit.audit_code(
                "from definitely_not_a_real_module_boom import Thing\n"
                "Thing.member"
            )
        self.assertEqual(problems, [])

    def test_wildcard_import_is_not_reported_as_a_missing_name(self):
        """``from <mod> import *`` binds a name *set*, not a name: looking up
        the literal "*" on the module invented a problem for a snippet that
        executes cleanly."""
        self.assertEqual(DocAudit.audit_code("from pythontk import *\n"), [])

    def test_wildcard_import_binds_names_for_later_checks(self):
        # The star names must actually enter the namespace -- unexpanded, every
        # later use is skipped as an unknown root and the audit goes blind on
        # exactly the blocks that use a wildcard.
        clean = "from pythontk import *\nMapFactory.resolve_map_type('a.png')\n"
        self.assertEqual(DocAudit.audit_code(clean), [])
        problems = DocAudit.audit_code(
            "from pythontk import *\nMapFactory.definitely_not_a_method()\n"
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("no attribute 'definitely_not_a_method'", problems[0])

    def test_wildcard_import_respects_module_dunder_all(self):
        """Mirror the interpreter: ``__all__`` defines the star surface, so a
        public-but-unexported name stays unknown (skipped), not checked."""
        import importlib
        from unittest import mock

        fake = types.ModuleType("fake_star_all_mod")
        fake.Exported = types.SimpleNamespace(real=1)
        fake.NotExported = types.SimpleNamespace(real=1)
        fake.__all__ = ["Exported", "ListedButMissing"]  # broken __all__ entry
        real_import_module = importlib.import_module

        def _fake(name, *a, **kw):
            if name == "fake_star_all_mod":
                return fake
            return real_import_module(name, *a, **kw)

        with mock.patch(
            "pythontk.core_utils.doc_audit.importlib.import_module", side_effect=_fake
        ):
            problems = DocAudit.audit_code(
                "from fake_star_all_mod import *\n"
                "Exported.bogus\n"
                "NotExported.bogus\n",
                roots={},
            )
        self.assertEqual(len(problems), 1)
        self.assertIn("Exported has no attribute 'bogus'", problems[0])
        self.assertNotIn("NotExported", problems[0])

    def test_wildcard_import_falls_back_to_public_dir(self):
        """No ``__all__``: the module's public surface is the star surface,
        underscored names excluded -- again matching the interpreter."""
        import importlib
        from unittest import mock

        fake = types.ModuleType("fake_star_dir_mod")
        fake.Public = types.SimpleNamespace(real=1)
        fake._private = types.SimpleNamespace(real=1)
        real_import_module = importlib.import_module

        def _fake(name, *a, **kw):
            if name == "fake_star_dir_mod":
                return fake
            return real_import_module(name, *a, **kw)

        with mock.patch(
            "pythontk.core_utils.doc_audit.importlib.import_module", side_effect=_fake
        ):
            problems = DocAudit.audit_code(
                "from fake_star_dir_mod import *\n"
                "Public.bogus\n"
                "_private.bogus\n",
                roots={},
            )
        self.assertEqual(len(problems), 1)
        self.assertIn("Public has no attribute 'bogus'", problems[0])

    def test_wildcard_import_of_unimportable_module_is_swallowed(self):
        """A module that will not import stays unknown -- the star form must
        keep the same optional-dep behaviour as the named form."""
        import importlib
        from unittest import mock

        real_import_module = importlib.import_module

        def _boom(name, *a, **kw):
            if name == "definitely_not_a_real_module_boom":
                raise RuntimeError("optional dep blew up at import time")
            return real_import_module(name, *a, **kw)

        with mock.patch(
            "pythontk.core_utils.doc_audit.importlib.import_module", side_effect=_boom
        ):
            problems = DocAudit.audit_code(
                "from definitely_not_a_real_module_boom import *\nThing.member\n",
                roots={},
            )
        self.assertEqual(problems, [])

    def test_a_hostile_module_surface_is_reported_not_raised(self):
        """Expanding a wildcard has to read ``__all__``/``dir()`` off a live
        module, and either can misbehave: a non-iterable ``__all__``, a lazy
        ``__getattr__`` that raises for an uninstalled extra, or a custom
        ``__dir__`` that blows up. This module's contract is to REPORT a
        problem, so any of those taking the whole audit down with a traceback
        turns one bad optional dep into a dead docs gate."""
        import sys
        import types

        class _RaisingDir(types.ModuleType):
            def __dir__(self):
                raise RuntimeError("dir blew up")

        bad_all = types.ModuleType("probe_bad_all")
        bad_all.__all__ = 5  # not iterable
        raising_all = types.ModuleType("probe_raising_all")

        def _raise(name):
            raise ImportError("__all__ requires the [extras] install")

        raising_all.__getattr__ = _raise
        cases = {
            "probe_bad_all": bad_all,
            "probe_raising_all": raising_all,
            "probe_raising_dir": _RaisingDir("probe_raising_dir"),
        }
        for name, module in cases.items():
            sys.modules[name] = module
            self.addCleanup(sys.modules.pop, name, None)

        for name in cases:
            with self.subTest(module=name):
                # No raise: the surface is unusable, so nothing binds and the
                # block's later references simply stay unknown.
                self.assertEqual(DocAudit.audit_code(f"from {name} import *\n"), [])

    def test_audit_markdown_prefixes_block_numbers(self):
        md = "```python\na = 1\n```\n\n```python\nptk.NoSuchThing\n```\n"
        problems = DocAudit.audit_markdown(md)
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("block 2:"))


class TestReadmeExamplesBind(unittest.TestCase):
    """Gate: every README code example binds to the live API."""

    def test_all_audited_docs_exist(self):
        # A moved doc silently exits the gate; catch the move itself.
        for path in AUDITED_DOCS:
            self.assertTrue(path.is_file(), f"audited doc missing: {path}")

    def test_readme_examples_bind_to_live_api(self):
        for path in AUDITED_DOCS:
            problems = DocAudit.audit_markdown(path.read_text(encoding="utf-8"))
            self.assertEqual(
                problems, [], f"{path.name}: stale examples:\n" + "\n".join(problems)
            )


class TestReadmeClaimedOutputs(unittest.TestCase):
    """Gate: literal outputs the front-door README claims are real AND
    still present in the document. Behavior drift fails the first assert;
    doc drift fails the second."""

    @classmethod
    def setUpClass(cls):
        cls.readme = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    def check(self, actual, expected, fragment):
        self.assertEqual(actual, expected)
        if fragment is not None:
            self.assertIn(fragment, self.readme)

    def test_filter_list_example(self):
        actual = ptk.filter_list(
            ["mesh_main", "mesh_backup", "mesh_LOD0", "cube_old"],
            inc=["mesh_*", "cube_*"],
            exc=["*_backup", "*_old"],
        )
        self.check(actual, ["mesh_main", "mesh_LOD0"], "['mesh_main', 'mesh_LOD0']")

    def test_find_str_and_format_example(self):
        actual = ptk.find_str_and_format(["mesh_old", "cube_old"], to="*_new", fltr="*_old")
        self.check(actual, ["mesh_new", "cube_new"], "['mesh_new', 'cube_new']")

    def test_remap_example(self):
        self.check(ptk.remap(50, old_range=(0, 100), new_range=(0, 1)), 0.5, "# 0.5")

    def test_ease_in_out_example(self):
        self.check(ptk.ProgressionCurves.ease_in_out(0.5), 0.5, None)

    def test_collapse_integer_sequence_example(self):
        actual = ptk.collapse_integer_sequence([1, 2, 3, 5, 7, 8, 9, 15])
        self.check(actual, "1-3, 5, 7-9, 15", '"1-3, 5, 7-9, 15"')

    def test_resolve_map_type_examples(self):
        self.check(
            ptk.MapFactory.resolve_map_type("character_Normal_DirectX.png"),
            "Normal_DirectX",
            '"Normal_DirectX"',
        )
        self.check(
            ptk.MapFactory.resolve_map_type("material_BC.tga"),
            "Base_Color",
            '"Base_Color"',
        )


if __name__ == "__main__":
    unittest.main()
