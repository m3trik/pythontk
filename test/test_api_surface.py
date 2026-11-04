# !/usr/bin/python
# coding=utf-8
"""Tests for ``ApiSurface``: the one engine every reader of the public API goes through.

The static half reads a package's source (nothing imported) for the registry,
``api.search`` / ``api.describe`` and the candidate scan; the live half reads a
loaded class for ``HelpMixin`` and the drift gate; one summary rule serves both.
"""

import ast
import contextlib
import gc
import importlib
import json
import os
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pythontk import (
    ApiSurface,
    ClassSurface,
    ModuleSurface,
    PackageManager,
    PackageSurface,
    SymbolRecord,
    TempArtifacts,
)

PKG = "apisurf_pkg"

TOOLS = '''\
"""Tools a package publishes."""

import os
from apisurf_pkg._mixins import _Area

LIMIT = 3
_HIDDEN = 4
helper = 5


def add_one(x: int) -> int:
    """Add one to *x*.

    More detail.
    """
    return x + 1


def _private(x):
    """Private."""


class BoxUtils(_Area):
    """Box tools, e.g. a measure, and more."""

    @staticmethod
    def measure_box(width: float, height: float = 1.0) -> float:
        """The area of a box *width* by *height*."""
        return width * height

    @classmethod
    def rename_all(cls, names: list, ctx=None) -> list:
        """Rename every name, planning each."""
        return names

    @Deprecation.symbol(remove_in="0.20.0", since="2026-10-01")
    def old_way(self):
        """Retired: use ``measure_box``."""

    @property
    def size(self) -> int:
        """The size."""
        return 1

    @size.setter
    def size(self, value):
        pass

    def bound(self, x: int) -> int:
        """An instance method."""
'''

MIXINS = '''\
class _Area:
    """A capability mixin in its own file."""

    @staticmethod
    def area_of(w: float, h: float) -> float:
        """The area of *w* by *h* (:class:`~apisurf_pkg.tools.BoxUtils` uses it)."""
        return w * h
'''


class _PackageCase(unittest.TestCase):
    """A package on disk, on ``sys.path``, never imported by the static read."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.source = self.root / PKG
        self.write("__init__.py", '"""A test package."""\n')
        self.write("tools.py", TOOLS)
        self.write("_mixins.py", MIXINS)
        self.write("__main__.py", "def main():\n    pass\n")
        self.write("test/test_tools.py", "def test_x():\n    pass\n")
        self.write("plugin_src/hooks/__init__.py", "")
        self.write(
            "plugin_src/hooks/hook.py",
            'def get_thing(x: int) -> int:\n    """Elsewhere."""\n',
        )
        sys.path.insert(0, self._tmp.name)
        importlib.invalidate_caches()

    def tearDown(self):
        sys.path.remove(self._tmp.name)
        for name in [m for m in sys.modules if m.split(".")[0] == PKG]:
            del sys.modules[name]
        self._tmp.cleanup()

    def write(self, relpath: str, text: str):
        path = self.source / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")

    @contextlib.contextmanager
    def registered(self, hosts=()):
        """Entry points registering this package (and *hosts* as runners),
        and no package installed editable."""
        groups = {
            ApiSurface.PROVIDER_GROUP: [
                SimpleNamespace(name="x", value=f"{PKG}.operations")
            ],
            ApiSurface.HOST_GROUP: [
                SimpleNamespace(name=h, value=f"{PKG}.runner:Runner") for h in hosts
            ],
        }
        with (
            patch.object(
                PackageManager,
                "entry_points",
                side_effect=lambda group: groups.get(group, []),
            ),
            patch.object(PackageManager, "editable_packages", return_value=[]),
        ):
            yield


class TestSummary(unittest.TestCase):
    """One rule for every summary: the registry, the catalog, ``.help()``."""

    def test_the_first_sentence_of_the_first_paragraph_its_lines_joined(self):
        doc = "Mirror the meshes\n    across an axis. Then more.\n\nDetails."
        self.assertEqual(ApiSurface.summary(doc), "Mirror the meshes across an axis.")

    def test_an_abbreviation_reads_on_and_rest_reads_as_markdown(self):
        doc = "Read a material, e.g. :class:`~pkg.Mat`, i.e. the header. Next."
        self.assertEqual(
            ApiSurface.summary(doc), "Read a material, e.g. `pkg.Mat`, i.e. the header."
        )
        self.assertEqual(ApiSurface.markdown("``x`` and :meth:`y`"), "`x` and `y`")

    def test_a_limit_cuts_with_an_ellipsis_and_nothing_reads_empty(self):
        self.assertEqual(ApiSurface.summary("A" * 120 + ".", limit=10), "A" * 9 + "…")
        self.assertEqual(ApiSurface.summary("Short.", limit=10), "Short.")
        for empty in (None, "", "   \n  "):
            self.assertEqual(ApiSurface.summary(empty), "")


class TestRead(_PackageCase):
    def test_it_reads_the_public_surface_without_importing(self):
        surface = ApiSurface.read(PKG)
        self.assertNotIn(PKG, sys.modules, "the static read imported the package")
        self.assertIsInstance(surface, PackageSurface)
        self.assertEqual(surface.name, PKG)
        modules = {m.relpath: m for m in surface.modules}
        # No test folder, no script; a module with no public names (the bare
        # __init__, the private mixin) is absent; a plug-in tree is read (the
        # registry covers it), though search leaves it out.
        self.assertEqual(sorted(modules), ["plugin_src/hooks/hook.py", "tools.py"])
        tools = modules["tools.py"]
        self.assertIsInstance(tools, ModuleSurface)
        self.assertEqual([f.qualname for f in tools.functions], ["add_one"])
        self.assertEqual(tools.functions[0].signature, "(x: int) -> int")
        self.assertEqual(tools.functions[0].summary, "Add one to *x*.")
        self.assertEqual([c.name for c in tools.constants], ["LIMIT"])
        (box,) = tools.classes
        self.assertIsInstance(box, ClassSurface)
        self.assertEqual(box.bases, ["_Area"])
        self.assertEqual(box.summary, "Box tools, e.g. a measure, and more.")
        kinds = {m.name: m.kind for m in box.members}
        self.assertEqual(
            kinds,
            {
                "measure_box": "staticmethod",
                "rename_all": "classmethod",
                "old_way": "method",
                "size": "property",  # its setter is not a second member
                "bound": "method",
                "area_of": "staticmethod",  # pulled up from the private base in another file
            },
        )
        members = {m.name: m for m in box.members}
        self.assertEqual(members["area_of"].qualname, "BoxUtils.area_of")
        self.assertTrue(members["area_of"].path.endswith("_mixins.py"))
        self.assertEqual(
            (members["old_way"].deprecated, members["old_way"].remove_in),
            (True, "0.20.0"),
        )
        self.assertEqual(members["old_way"].since, "2026-10-01")
        self.assertIn("planning each", members["rename_all"].doc)

    def test_a_summary_is_held_to_one_line(self):
        self.write("long.py", f'def wordy():\n    """{"word " * 40}."""\n')
        (wordy,) = next(
            m for m in ApiSurface.read(PKG).modules if m.relpath == "long.py"
        ).functions
        self.assertEqual(len(wordy.summary), ApiSurface.SUMMARY_MAX)
        self.assertTrue(wordy.summary.endswith("…"))

    def test_it_reads_again_only_when_a_file_changes(self):
        first = ApiSurface.read(PKG)
        self.assertIs(ApiSurface.read(PKG), first)
        time.sleep(0.01)
        self.write("extra.py", 'def more() -> None:\n    """More."""\n')
        again = ApiSurface.read(PKG)
        self.assertIsNot(again, first)
        self.assertIn("extra.py", [m.relpath for m in again.modules])

    def test_a_root_writes_the_source_root_relative_and_the_sidecar_round_trips(self):
        surface = ApiSurface.read(self.source, root=self.root)
        self.assertEqual(surface.source_root, PKG)
        data = surface.to_dict()
        member = data["modules"][-1]["classes"][0]["members"][0]
        self.assertNotIn("remove_in", member)  # an empty one says nothing
        self.assertNotIn(
            "deprecated", data["modules"][-1]["classes"][0]
        )  # a live class
        rebuilt = PackageSurface.from_dict(json.loads(json.dumps(data)))
        self.assertEqual(rebuilt.to_dict(), data)
        self.assertIsInstance(rebuilt.modules[-1].classes[0].members[0], SymbolRecord)

    def test_retirements_smaller_than_a_symbol_are_read_with_their_deadline(self):
        self.write(
            "kw.py",
            """\
            _REMOVE_IN = "0.21.0"

            @Deprecation.parameter("old", remove_in=_REMOVE_IN, since="2026-10-02")
            def f(old=None):
                pass

            @Deprecation.parameter("other", remove_in=compute())
            def g(other=None):
                pass
            """,
        )
        rows = sorted(r for r in ApiSurface.retirements(PKG) if r[0] == "kw.py")
        self.assertEqual(
            rows,
            [
                ("kw.py", "Deprecation.parameter('old')", "0.21.0", "2026-10-02"),
                (
                    "kw.py",
                    "Deprecation.parameter('other')",
                    "",
                    "",
                ),  # unreadable: said, not guessed
            ],
        )


class TestReadCache(_PackageCase):
    """A package read once is read from disk by the next process, and a read
    in progress is shared, never repeated."""

    def setUp(self):
        super().setUp()
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        self.cache = Path(cache.name)
        env = patch.dict(os.environ, {"PYTHONTK_CACHE_ROOT": cache.name})
        env.start()
        self.addCleanup(env.stop)
        # The package sits in a temp folder, whose reads are kept in the
        # process only; these tests are about the cache, so it keeps them.
        keeps = patch.object(
            ApiSurface, "_keeps_on_disk", return_value=True, create=True
        )
        keeps.start()
        self.addCleanup(keeps.stop)
        self._forget()

    @staticmethod
    def _forget():
        """This process's reads gone, as in a new process."""
        ApiSurface._READS.clear()
        ApiSurface._WALKED.clear()

    def entries(self, pattern: str = "*.json"):
        return sorted(self.cache.rglob(pattern))

    def test_a_reread_after_an_edit_walks_only_the_modules_it_reaches(self):
        """What the per-read parse gives up, the kept module surfaces give
        back: no tree is kept, and an edit costs the modules it reaches."""
        self.write(
            "late.py",
            "from apisurf_pkg._base import _Late\n\nclass Late(_Late):\n    pass\n",
        )
        ApiSurface.read(str(self.source))
        walked, real = [], ApiSurface._walk_module

        def walk(path, root, *memo):
            walked.append(path.name)
            return real(path, root, *memo)

        with patch.object(ApiSurface, "_walk_module", side_effect=walk):
            for edit, reached in (
                (("extra.py", 'def more() -> None:\n    """More."""\n'), ["extra.py"]),
                (("_mixins.py", MIXINS + "\n"), ["_mixins.py", "tools.py"]),
                (
                    ("_base.py", "class _Late:\n    def hello(self):\n        pass\n"),
                    ["_base.py", "late.py"],
                ),
            ):
                walked.clear()
                time.sleep(0.05)
                self.write(*edit)
                surface = ApiSurface.read(str(self.source))
                self.assertEqual(sorted(walked), reached)
        late = next(c for m in surface.modules for c in m.classes if c.name == "Late")
        self.assertEqual(
            [m.name for m in late.members], ["hello"]
        )  # its base, written later

    def test_a_file_it_could_not_read_is_read_again_and_the_read_kept_nowhere(self):
        """A sync client's or a scanner's lock on one file is momentary: the
        read that met it is never kept, in the process or on disk."""
        tools, real = self.source / "tools.py", Path.read_text

        def locked(path, *args, **kwargs):
            if Path(path) == tools:
                raise PermissionError(13, "held by a sync client", str(path))
            return real(path, *args, **kwargs)

        with patch.object(Path, "read_text", locked):
            first = ApiSurface.read(str(self.source))
        self.assertNotIn("tools.py", [m.relpath for m in first.modules])
        self.assertEqual(self.entries(), [])
        again = ApiSurface.read(str(self.source))
        self.assertIn("tools.py", [m.relpath for m in again.modules])
        self.assertEqual(len(self.entries()), 1)

    def test_no_parsed_tree_outlives_its_read(self):
        for cache in (True, False):
            with self.subTest(cache=cache):
                ApiSurface.read(str(self.source), cache=cache)
                gc.collect()
                kept = [
                    o
                    for o in gc.get_objects()
                    if isinstance(o, ast.ClassDef)
                    and str(getattr(o, "ptk_path", "")).startswith(str(self.source))
                ]
                self.assertEqual(kept, [])

    def test_a_write_that_cannot_replace_its_entry_leaves_it_whole_and_no_part(self):
        ApiSurface.read(str(self.source))
        (entry,) = self.entries()
        before = entry.read_bytes()
        self._forget()
        time.sleep(0.05)
        self.write("extra.py", 'def more() -> None:\n    """More."""\n')
        with patch("os.replace", side_effect=PermissionError("held by a reader")):
            surface = ApiSurface.read(str(self.source))
        self.assertIn("extra.py", [m.relpath for m in surface.modules])
        self.assertEqual(entry.read_bytes(), before)
        self.assertEqual(self.entries("*.part"), [])

    def test_a_hit_refreshes_its_entry_and_one_it_cannot_refresh_is_still_a_hit(self):
        ApiSurface.read(str(self.source))
        (entry,) = self.entries()
        old = time.time() - 10 * 86400
        os.utime(entry, (old, old))
        self._forget()
        with patch.object(
            ApiSurface, "_walk_module", side_effect=AssertionError("parsed again")
        ):
            ApiSurface.read(str(self.source))
            self.assertGreater(entry.stat().st_mtime, old + 86400)
            self._forget()
            with patch("os.utime", side_effect=PermissionError("read-only")):
                ApiSurface.read(str(self.source))

    def test_a_write_sweeps_what_is_stale_while_another_process_sweeps_too(self):
        folder = self.cache / "api_surface"
        folder.mkdir(parents=True)

        def aged(name: str, days: float) -> Path:
            path = folder / name
            path.write_text("{}", encoding="utf-8")
            when = time.time() - days * 86400
            os.utime(path, (when, when))
            return path

        stale = [aged("a_stale.json", 15), aged("b_stale.json", 15)]
        fresh = aged("fresh.json", 13)
        stranded = aged("a.1.2.part", 1 / 24)  # a write a killed process left
        writing = aged("b.1.2.part", 0)  # a write in progress
        real, raced = Path.unlink, []

        def unlink(path, *args, **kwargs):
            if path.suffix == ".json" and not raced:  # another process got there first
                raced.append(path)
                real(path)
                raise FileNotFoundError(2, "swept by another process", str(path))
            return real(path, *args, **kwargs)

        with patch.object(Path, "unlink", unlink):
            ApiSurface.read(str(self.source))
        self.assertEqual([p.exists() for p in stale], [False, False])
        self.assertEqual(
            (fresh.exists(), stranded.exists(), writing.exists()), (True, False, True)
        )

    def test_a_new_process_reads_a_package_from_the_cache_until_its_source_changes(
        self,
    ):
        first = ApiSurface.read(str(self.source))
        described = ApiSurface.describe(f"{PKG}.tools:BoxUtils.measure_box")
        self._forget()
        with patch.object(
            ApiSurface, "_walk_module", side_effect=AssertionError("parsed again")
        ):
            second = ApiSurface.read(str(self.source))
            self.assertEqual(second.to_dict(), first.to_dict())
            # What the sidecar leaves out comes back too: the docstring, the file.
            self.assertEqual(
                ApiSurface.describe(f"{PKG}.tools:BoxUtils.measure_box"), described
            )
            retired = next(
                r for _, r in second.symbols() if getattr(r, "name", "") == "old_way"
            )
            self.assertEqual(getattr(retired, "since", None), "2026-10-01")
        self._forget()
        time.sleep(0.05)
        self.write("tools.py", TOOLS + '\n\ndef extra() -> None:\n    """Extra."""\n')
        third = ApiSurface.read(str(self.source))
        self.assertIn("extra", [r.name for m in third.modules for r in m.functions])

    def test_a_cache_another_version_of_the_reader_wrote_is_not_read(self):
        ApiSurface.read(str(self.source))
        self._forget()
        with (
            patch.object(ApiSurface, "_reader_key", return_value="another reader"),
            patch.object(
                ApiSurface, "_walk_module", side_effect=AssertionError("parsed")
            ),
            self.assertRaises(AssertionError),
        ):
            ApiSurface.read(str(self.source))

    def test_reads_of_one_package_at_once_parse_it_once(self):
        walked = []
        real = ApiSurface._walk_module

        def walk(path, root, *memo):
            walked.append(path)
            time.sleep(0.02)
            return real(path, root, *memo)

        with patch.object(ApiSurface, "_walk_module", side_effect=walk):
            threads = [
                threading.Thread(target=ApiSurface.read, args=(str(self.source),))
                for _ in range(3)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(30)
        self.assertEqual(len(walked), len(set(walked)))  # each file once

    def test_a_tree_read_once_is_kept_nowhere(self):
        """A release's source copied to temp is read once and deleted: keeping
        its read (MBs per package) would only fill the cache for the sweep."""
        surface = ApiSurface.read(str(self.source), cache=False)
        self.assertIn("BoxUtils", [c.name for m in surface.modules for c in m.classes])
        self.assertEqual(ApiSurface._READS, {})
        self.assertEqual(ApiSurface._PARSED, {})
        self.assertEqual(ApiSurface._WALKED, {})
        self.assertEqual(
            list(Path(os.environ["PYTHONTK_CACHE_ROOT"]).rglob("*.json")), []
        )
        self._forget()
        with patch.object(
            ApiSurface, "_cached", side_effect=AssertionError("cache read")
        ):
            ApiSurface.read(str(self.source), cache=False)


class TestReadKeptOnDisk(_PackageCase):
    """What the per-user cache keeps, and under which key."""

    def setUp(self):
        super().setUp()
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        self.cache = Path(cache.name)
        env = patch.dict(os.environ, {"PYTHONTK_CACHE_ROOT": cache.name})
        env.start()
        self.addCleanup(env.stop)
        ApiSurface._READS.clear()

    def test_a_tree_under_a_temp_folder_is_kept_in_the_process_only(self):
        """A test's sandbox or a release copied out is gone long before its
        entry would be swept."""
        first = ApiSurface.read(str(self.source))
        self.assertEqual(list(self.cache.rglob("*.json")), [])
        self.assertIs(ApiSurface.read(str(self.source)), first)
        with (
            patch.dict(os.environ, {"PYTHONTK_TEMP_ROOT": str(self.root)}),
            patch("tempfile.gettempdir", return_value=str(self.cache)),
        ):
            ApiSurface._READS.clear()
            ApiSurface.read(str(self.source))  # under the ecosystem temp root
        self.assertEqual(list(self.cache.rglob("*.json")), [])

    def test_each_python_keeps_its_own_entry(self):
        """Maya's, Blender's and the system's Python share one user cache, and
        a file one of them cannot parse another can."""
        names = set()
        for version in ((3, 11, 0), (3, 13, 0)):
            with patch.object(sys, "version_info", version):
                names.add(ApiSurface._cache_file(self.source, "x").name)
        self.assertEqual(len(names), 2)


class TestLocate(unittest.TestCase):
    def test_a_repo_folder_on_the_path_is_looked_through_to_its_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            inner = Path(tmp) / "surfrepo" / "surfrepo"
            inner.mkdir(parents=True)
            (inner / "__init__.py").write_text("", encoding="utf-8")
            sys.path.insert(0, tmp)  # the repo folder, not the package, is findable
            importlib.invalidate_caches()
            try:
                self.assertTrue(os.path.samefile(ApiSurface.locate("surfrepo"), inner))
            finally:
                sys.path.remove(tmp)
                sys.modules.pop("surfrepo", None)

    def test_a_dotted_or_unknown_name_is_refused(self):
        for name in ("pythontk.core_utils", "no_such_package_here", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                ApiSurface.locate(name)


class TestSearchAndDescribe(_PackageCase):
    BOX = f"{PKG}.tools:BoxUtils"

    def test_search_finds_every_public_symbol_a_caller_can_import(self):
        with self.registered():
            everything = [r["target"] for r in ApiSurface.search("", limit=None)]
        self.assertEqual(
            everything,
            [
                f"{PKG}.tools:BoxUtils",
                f"{self.BOX}.area_of",
                f"{self.BOX}.bound",
                f"{self.BOX}.measure_box",
                f"{self.BOX}.old_way",
                f"{self.BOX}.rename_all",
                f"{self.BOX}.size",
                f"{PKG}.tools:LIMIT",
                f"{PKG}.tools:add_one",
            ],
        )  # not the private names, not what tools only imported, not the plug-in tree
        self.assertNotIn(PKG, sys.modules, "search imported the package")

    def test_a_name_holding_every_word_ranks_first_and_a_row_says_how_to_call_it(self):
        with self.registered(hosts=("maya",)):
            first, *rest = ApiSurface.search("area")
            (box,) = ApiSurface.search("box area width")
            (old,) = ApiSurface.search("old_way")
        self.assertEqual(first["target"], f"{self.BOX}.area_of")  # in the name
        self.assertIn(
            f"{self.BOX}.measure_box", [r["target"] for r in rest]
        )  # in the summary only
        self.assertEqual(
            box,
            {
                "target": f"{self.BOX}.measure_box",
                "kind": "staticmethod",
                "signature": "BoxUtils.measure_box(width: float, height: float = 1.0) -> float",
                "summary": "The area of a box *width* by *height*.",
                "host": "maya",
                "exec": "maya.python.exec",  # how code calls it
            },
        )
        self.assertEqual((old["deprecated"], old["remove_in"]), (True, "0.20.0"))
        with self.registered():
            self.assertEqual(len(ApiSurface.search("", limit=2)), 2)
            self.assertEqual(ApiSurface.search("", limit=0), [])
            with self.assertRaises(ValueError):
                ApiSurface.search("x", packages=["no_such_package_here"])

    def test_a_default_search_skips_a_registered_package_it_cannot_locate(self):
        """One editable install that is no package here (a single module, a
        distribution named unlike its import) never costs the rest."""
        with (
            self.registered(),
            patch.object(
                PackageManager,
                "editable_packages",
                return_value=["no_such_package_here"],
            ),
            self.assertLogs(
                "pythontk.core_utils.api_surface.api_surface", "WARNING"
            ) as logs,
        ):
            found = [r["target"] for r in ApiSurface.search("measure")]
        self.assertIn(f"{self.BOX}.measure_box", found)
        self.assertIn("no_such_package_here", "\n".join(logs.output))

    def test_describe_reads_the_source_and_imports_nothing(self):
        with self.registered():
            rename = ApiSurface.describe(f"{self.BOX}.rename_all")
            box = ApiSurface.describe(f"{PKG}.tools:BoxUtils")
        self.assertNotIn(PKG, sys.modules, "describe imported the package")
        self.assertEqual(rename["kind"], "classmethod")
        self.assertEqual(
            rename["signature"],
            "BoxUtils.rename_all(cls, names: list, ctx=None) -> list",
        )
        self.assertEqual(rename["doc"], "Rename every name, planning each.")
        self.assertRegex(rename["file"], r"tools\.py:\d+$")
        self.assertNotIn("host", rename)
        self.assertEqual((box["kind"], box["bases"]), ("class", ["_Area"]))
        self.assertIn(
            {
                "name": "area_of",
                "kind": "staticmethod",
                "signature": "BoxUtils.area_of(w: float, h: float) -> float",
                "summary": "The area of *w* by *h* (`apisurf_pkg.tools.BoxUtils` uses it).",
            },
            box["members"],
        )

    def test_describe_refuses_what_is_no_public_symbol(self):
        for target, why in (
            (f"{self.BOX}._secret", "private"),
            (
                f"{PKG}.tools:os.getcwd",
                "no public symbol",
            ),  # a name the module only imported
            (f"{PKG}.tools", "module:qualified.name"),
            (f"{PKG}.tools:BoxUtils.measure", "did you mean"),
        ):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, why):
                ApiSurface.describe(target)


class TestRegistration(unittest.TestCase):
    def test_registered_lists_each_package_once_providers_first(self):
        groups = {
            ApiSurface.PROVIDER_GROUP: [
                SimpleNamespace(name="a", value="alpha.operations"),
                SimpleNamespace(
                    name="a", value="alpha.operations:OPERATIONS"
                ),  # an editable and an install
            ],
            ApiSurface.HOST_GROUP: [
                SimpleNamespace(name="maya", value="beta:Scene.operation_host"),
                SimpleNamespace(name="blender", value="alpha.hosts:Runner"),
            ],
        }
        with (
            patch.object(
                PackageManager, "entry_points", side_effect=lambda g: groups.get(g, [])
            ),
            patch.object(PackageManager, "editable_packages", return_value=[]),
        ):
            self.assertEqual(ApiSurface.registered(), ["alpha", "beta"])
            self.assertEqual(
                (ApiSurface.host("beta"), ApiSurface.host("alpha")), ("maya", "blender")
            )
            self.assertIsNone(ApiSurface.host("gamma"))

    def test_registered_adds_each_package_installed_editable_after_them(self):
        # The code being worked on here is searched too, registered or not.
        groups = {
            ApiSurface.HOST_GROUP: [SimpleNamespace(name="maya", value="beta:S.h")]
        }
        with (
            patch.object(
                PackageManager, "entry_points", side_effect=lambda g: groups.get(g, [])
            ),
            patch.object(
                PackageManager, "editable_packages", return_value=["gamma", "beta"]
            ),
        ):
            self.assertEqual(ApiSurface.registered(), ["beta", "gamma"])

    def test_editable_packages_reads_each_distribution_installed_editable(self):
        def dist(name, direct_url=None, top_level=None):
            files = {"direct_url.json": direct_url, "top_level.txt": top_level}
            return SimpleNamespace(metadata={"Name": name}, read_text=files.get)

        editable = '{"url": "file:///x", "dir_info": {"editable": true}}'
        installed = '{"url": "file:///x", "dir_info": {}}'
        dists = [
            dist("alpha-tools", editable, "alpha\n_alpha_native\n"),
            dist("wheel-only"),
            dist("built", installed, "built\n"),
            dist(
                "My-Pkg", editable
            ),  # no top_level.txt: the name, as an import spells it
            dist("broken", "{not json"),
            dist("single", editable, "single_mod\n"),  # one module, no package
            dist("hatch-thing", editable),  # imports by another name: none here
            dist(None, editable),  # no name in its metadata
            dist("listed", "[]"),  # a direct_url.json that is no object
            dist("odd", '{"dir_info": ["editable"]}'),
            SimpleNamespace(metadata=None, read_text={"direct_url.json": editable}.get),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("alpha", "my_pkg", "built"):
                (Path(tmp) / name).mkdir()
                (Path(tmp) / name / "__init__.py").write_text("", encoding="utf-8")
            (Path(tmp) / "single_mod.py").write_text("", encoding="utf-8")
            sys.path.insert(0, tmp)
            importlib.invalidate_caches()
            try:
                with patch("importlib.metadata.distributions", return_value=dists):
                    self.assertEqual(
                        PackageManager.editable_packages(), ["alpha", "my_pkg"]
                    )
            finally:
                sys.path.remove(tmp)
                importlib.invalidate_caches()


# -- C# ------------------------------------------------------------------------------
CS_PKG = "apisurf_cs"

#: A package whose C# lives beside it in a checkout (``../Unity/com.test.pkg``)
#: and inside it in a wheel (``templates``), the first that exists read.
CS_INIT = '''\
"""A package with a C# half."""

CSHARP_SOURCES = ("../Unity/com.test.pkg", "templates")
'''

CS_KERNEL = r"""// <auto-generated> a header comment, not a doc </auto-generated>
using System;
using System.Collections.Generic;
using Alias = System.Collections.Generic.List<int>;

namespace Test.Kernel
{
    /// <summary>
    /// One run's context: an operation asks it before each change
    /// (<see cref="ShouldChange"/>); <c>null</c> is a plain run. More detail.
    /// </summary>
    /// <remarks>Prior art: none.</remarks>
    [Serializable]
    public sealed class OpContext : IDisposable, IComparable<OpContext>
    {
        readonly Func<bool> _cancelled;   // private: not surface

        /// <param name="dryRun">Whether this run only plans.</param>
        /// <param name="cancelled">Whether the run was cancelled.</param>
        public OpContext(bool dryRun = false, Func<bool> cancelled = null)
        {
            DryRun = dryRun;
            _cancelled = cancelled;
            var text = $"a {{literal}} {(dryRun ? "x}" : "y{")} and {dryRun:F2}";
            var raw = @"C:\path ""quoted"" {not a block";
            var both = $@"{raw}"" }} {{";
            char brace = '{', quote = '"', tick = '\'';
            // a comment { with a brace
            /* and } another */
        }

        /// <summary>Whether this run only plans: every change is listed, none made.</summary>
        public bool DryRun { get; }

        /// <summary>The label, set by the caller.</summary>
        public string Label { get; private set; } = "none;{";

        /// <summary>Every change asked for.</summary>
        public List<string> Changes { get; } = new List<string> { "a", "b" };

        /// <summary>Ask before changing <paramref name="target"/>; true: make the change.</summary>
        /// <param name="target">What would change.</param>
        /// <param name="action">What would be done to it.</param>
        /// <returns>Whether to make it.</returns>
        /// <exception cref="OperationCanceledException">The run was cancelled.</exception>
        public bool ShouldChange(object target, string action = "a;b{") => Ask("change", target, action);

        /// <summary>The largest id.</summary>
        public const int IdMax = 34;

        /// <summary>The values kept between calls.</summary>
        public static readonly Dictionary<string, object> Vars = new Dictionary<string, object>();

        /// <summary>Raised on each change.</summary>
        public event Action<string> Changed;

        /// <summary>Retired.</summary>
        [Obsolete("Use ShouldChange; removed in 0.9.0 (deprecated 2026-10-01).")]
        public bool Allow(object target) => ShouldChange(target);

        internal void Hidden() { }
        protected void ForSubclasses() { }
        protected internal void Both() { }
        void Implicit() { }
        public void Dispose() { }
        public int CompareTo(OpContext other) => 0;

        bool Ask(string kind, object target, string action) { return !DryRun; }

        /// <summary>A snippet that raised.</summary>
        public sealed class SnippetException : Exception
        {
            internal SnippetException(string message) : base(message) { }

            /// <summary>Its line.</summary>
            public int Line;
        }

        sealed class Secret { public int Leak; }
    }

    /// <summary>JSON, read and written.</summary>
    public static class Json
    {
        /// <summary>The tree <paramref name="text"/> parses to.</summary>
        public static object Parse(string text) { return null; }

        /// <summary>The JSON text of <paramref name="value"/>.</summary>
        public static string Write(object value) => Write(value, false);

        /// <summary>The JSON text of <paramref name="value"/>, indented when asked.</summary>
        public static string Write(object value, bool indent) { return ""; }

        /// <summary>Run code.</summary>
        public static Dictionary<string, object> Run(
            [System.ComponentModel.Description("C# statements; `return x;` hands x back.")]
            string code)
        { return null; }

        /// <summary>A list of objects a spec reads.</summary>
        public static List<T> Objs<T>(IDictionary<string, object> o, string key, Func<object, T> read,
                                      int length = -1) where T : class
        { return null; }

        /// <summary>Its limits.</summary>
        public static void Limits(int slot, out double? min, ref int count, params string[] keys) { min = null; }

#if UNITY_EDITOR
        /// <summary>Whether a pointer is there.</summary>
        public static bool Available(int index) => true;
#else
        /// <summary>Whether a pointer is there.</summary>
        public static bool Available(int index) => false;
#endif
    }

    /// <summary>What an operation targets.</summary>
    public interface ITarget
    {
        /// <summary>Its name.</summary>
        string Name { get; }

        /// <summary>Open it.</summary>
        void Open(string path);
    }

    /// <summary>How a run ended.</summary>
    public enum Status
    {
        /// <summary>It finished.</summary>
        Done,
        /// <summary>It stopped part way.</summary>
        Partial = 2,
        [Obsolete("Use " +
                  "Partial; this value goes in test 0.9.0.")]
        Halted,
    }

    /// <summary>A point.</summary>
    public readonly struct Vec
    {
        /// <summary>A component by index.</summary>
        public double this[int i] => i == 0 ? 1 : 2;
    }

    /// <summary>Decides one change.</summary>
    public delegate bool Decide(object target, string action);

    /// <summary>Marks an operation (internal: never surface).</summary>
    [AttributeUsage(AttributeTargets.Method, Inherited = false)]
    internal sealed class OperationAttribute : Attribute
    {
        /// <summary>The id.</summary>
        public string Id { get; }
    }

    internal enum Effect { Read, Code }

    static class Registry { public static int Count; }
}
"""

CS_CORE = """\
namespace Test.Core;

/// <summary>What one run hands back.</summary>
public sealed class Outcome
{
    /// <summary>The operation id that ran.</summary>
    public string op;

    /// <summary>A word on how it went.</summary>
    public string status, error;
}
"""

CS_RECORD_G = """\
// <auto-generated> GENERATED: do not edit </auto-generated>
using System;

/// <summary>The record's declared shape.</summary>
public static partial class Record
{
    /// <summary>The schema version read.</summary>
    public const int CurrentVersion = 1;
}
"""

CS_RECORD = """\
public static partial class Record
{
    /// <summary>Read one off a parsed tree.</summary>
    public static Record.Shape Read(object value) => null;

    /// <summary>The shape.</summary>
    public sealed class Shape { }
}
"""

CS_TESTS = """\
namespace Test.Kernel.Tests { public class OpContextTests { public void ItWorks() { } } }
"""

#: Every public C# name the fixture declares, as search lists them.
CS_TARGETS = [
    "Record",
    "Record.CurrentVersion",
    "Record.Read",
    "Record.Shape",
    "Test.Core.Outcome",
    "Test.Core.Outcome.error",
    "Test.Core.Outcome.op",
    "Test.Core.Outcome.status",
    "Test.Kernel.Decide",
    "Test.Kernel.ITarget",
    "Test.Kernel.ITarget.Name",
    "Test.Kernel.ITarget.Open",
    "Test.Kernel.Json",
    "Test.Kernel.Json.Available",
    "Test.Kernel.Json.Limits",
    "Test.Kernel.Json.Objs",
    "Test.Kernel.Json.Parse",
    "Test.Kernel.Json.Run",
    "Test.Kernel.Json.Write(object)",
    "Test.Kernel.Json.Write(object, bool)",
    "Test.Kernel.OpContext",
    "Test.Kernel.OpContext.Allow",
    "Test.Kernel.OpContext.Changed",
    "Test.Kernel.OpContext.Changes",
    "Test.Kernel.OpContext.CompareTo",
    "Test.Kernel.OpContext.Dispose",
    "Test.Kernel.OpContext.DryRun",
    "Test.Kernel.OpContext.IdMax",
    "Test.Kernel.OpContext.Label",
    "Test.Kernel.OpContext.OpContext",
    "Test.Kernel.OpContext.ShouldChange",
    "Test.Kernel.OpContext.SnippetException",
    "Test.Kernel.OpContext.SnippetException.Line",
    "Test.Kernel.OpContext.Vars",
    "Test.Kernel.Status",
    "Test.Kernel.Status.Done",
    "Test.Kernel.Status.Halted",
    "Test.Kernel.Status.Partial",
    "Test.Kernel.Vec",
    "Test.Kernel.Vec.this",
]


class _CSharpCase(unittest.TestCase):
    """A package on ``sys.path`` whose C# sits beside it, as a checkout keeps
    it: ``<root>/apisurf_cs/`` and ``<root>/Unity/com.test.pkg/``."""

    def setUp(self):
        self.temp = TempArtifacts.scoped("api_surface_csharp")
        self.addCleanup(self.temp.cleanup, force=True)
        self.root = Path(self.temp.dir_path())
        self.source = self.root / CS_PKG
        self.csharp = self.root / "Unity" / "com.test.pkg"
        self.write(self.source / "__init__.py", CS_INIT)
        self.write(self.source / "tools.py", 'def tool() -> int:\n    """A tool."""\n')
        for relpath, text in (
            ("Kernel/OpContext.cs", CS_KERNEL),
            ("Core/Outcome.cs", CS_CORE),
            ("Feature/Record.g.cs", CS_RECORD_G),
            ("Feature/Record.cs", CS_RECORD),
            ("Kernel/Tests/OpContextTests.cs", CS_TESTS),
        ):
            self.write(self.csharp / relpath, text)
        sys.path.insert(0, str(self.root))
        importlib.invalidate_caches()
        self.addCleanup(importlib.invalidate_caches)
        self.addCleanup(sys.path.remove, str(self.root))

    @staticmethod
    def write(path: Path, text: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    @contextlib.contextmanager
    def registered(self, host="unity"):
        """This package registered, its runner serving *host*."""
        groups = {
            ApiSurface.PROVIDER_GROUP: [
                SimpleNamespace(name="x", value=f"{CS_PKG}.operations")
            ],
            ApiSurface.HOST_GROUP: (
                [SimpleNamespace(name=host, value=f"{CS_PKG}:Host.run")] if host else []
            ),
        }
        with (
            patch.object(
                PackageManager,
                "entry_points",
                side_effect=lambda group: groups.get(group, []),
            ),
            patch.object(PackageManager, "editable_packages", return_value=[]),
        ):
            yield

    def records(self):
        """``{target: symbol}`` of the package's C#."""
        return dict(ApiSurface._csharp_rows(CS_PKG))


class TestCSharpDeclaration(_CSharpCase):
    """Where a package's C# is: declared once in its root ``__init__.py``,
    read without importing it, the first declared folder that exists."""

    def test_a_checkout_reads_the_folder_beside_the_package(self):
        self.assertEqual(
            ApiSurface.csharp_sources(CS_PKG), ("../Unity/com.test.pkg", "templates")
        )
        self.assertTrue(os.path.samefile(ApiSurface.csharp_root(CS_PKG), self.csharp))
        self.assertNotIn(CS_PKG, sys.modules, "the declaration was imported")

    def test_a_wheel_reads_the_folder_inside_the_package(self):
        wheel = self.root / "site" / CS_PKG
        self.write(wheel / "__init__.py", CS_INIT)
        self.write(wheel / "templates" / "Kernel" / "OpContext.cs", CS_KERNEL)
        self.assertTrue(
            os.path.samefile(ApiSurface.csharp_root(wheel), wheel / "templates")
        )
        targets = [
            m.relpath for m in ApiSurface.read_csharp(wheel / "templates").modules
        ]
        self.assertEqual(targets, ["Test.Kernel"])

    def test_the_first_declared_folder_wins_and_none_declared_is_none(self):
        self.write(self.source / "templates" / "Old.cs", "public class Old { }\n")
        self.assertTrue(os.path.samefile(ApiSurface.csharp_root(CS_PKG), self.csharp))
        bare = self.root / "bare"
        self.write(bare / "__init__.py", '"""No C#."""\n')
        self.assertIsNone(ApiSurface.csharp_root(bare))
        self.assertEqual(ApiSurface.csharp_sources(bare), ())
        given = ApiSurface.csharp_root(bare, sources=("../Unity/com.test.pkg",))
        self.assertTrue(os.path.samefile(given, self.csharp))


class TestCSharpRead(_CSharpCase):
    """The public surface C# declares, read from the source as written."""

    def test_every_public_name_and_none_that_is_not(self):
        self.assertEqual(sorted(self.records()), CS_TARGETS)

    def test_a_namespace_is_a_module_and_a_type_a_class(self):
        surface = ApiSurface.read_csharp(CS_PKG)
        self.assertEqual(
            [m.relpath for m in surface.modules], ["Test.Core", "Test.Kernel", "global"]
        )
        rows = self.records()
        context = rows["Test.Kernel.OpContext"]
        self.assertIsInstance(context, ClassSurface)
        self.assertEqual(context.kind, "class")
        self.assertEqual(context.bases, ["IDisposable", "IComparable<OpContext>"])
        self.assertEqual(
            context.declaration,
            "public sealed class OpContext : IDisposable, IComparable<OpContext>",
        )
        self.assertEqual(
            context.summary,
            "One run's context: an operation asks it before each change "
            "(`ShouldChange`); `null` is a plain run.",
        )
        self.assertIn("Prior art: none.", context.doc)
        kinds = {
            t: rows[t].kind
            for t in (
                "Test.Kernel.Status",
                "Test.Kernel.Vec",
                "Test.Kernel.ITarget",
                "Test.Kernel.Decide",
                "Test.Kernel.OpContext.SnippetException",
            )
        }
        self.assertEqual(
            kinds,
            {
                "Test.Kernel.Status": "enum",
                "Test.Kernel.Vec": "struct",
                "Test.Kernel.ITarget": "interface",
                "Test.Kernel.Decide": "delegate",
                "Test.Kernel.OpContext.SnippetException": "class",
            },
        )

    def test_each_member_is_spelled_as_written_its_attributes_and_body_left_out(self):
        rows = self.records()
        spelled = {
            t[len("Test.Kernel.") :]: (rows[t].kind, rows[t].signature)
            for t in CS_TARGETS
            if t.startswith("Test.Kernel.") and isinstance(rows[t], SymbolRecord)
        }
        expected = {
            "OpContext.OpContext": (
                "constructor",
                "public OpContext(bool dryRun = false, Func<bool> cancelled = null)",
            ),
            "OpContext.DryRun": ("property", "public bool DryRun { get; }"),
            "OpContext.Label": (
                "property",
                "public string Label { get; private set; }",
            ),
            "OpContext.Changes": ("property", "public List<string> Changes { get; }"),
            "OpContext.ShouldChange": (
                "method",
                'public bool ShouldChange(object target, string action = "a;b{")',
            ),
            "OpContext.IdMax": ("constant", "public const int IdMax = 34"),
            "OpContext.Vars": (
                "field",
                "public static readonly Dictionary<string, object> Vars",
            ),
            "OpContext.Changed": ("event", "public event Action<string> Changed"),
            "OpContext.Allow": ("method", "public bool Allow(object target)"),
            "OpContext.Dispose": ("method", "public void Dispose()"),
            "OpContext.CompareTo": ("method", "public int CompareTo(OpContext other)"),
            "OpContext.SnippetException.Line": ("field", "public int Line"),
            "Json.Parse": ("staticmethod", "public static object Parse(string text)"),
            "Json.Write(object)": (
                "staticmethod",
                "public static string Write(object value)",
            ),
            "Json.Write(object, bool)": (
                "staticmethod",
                "public static string Write(object value, bool indent)",
            ),
            "Json.Run": (
                "staticmethod",
                "public static Dictionary<string, object> Run(string code)",
            ),
            "Json.Objs": (
                "staticmethod",
                "public static List<T> Objs<T>(IDictionary<string, object> o, "
                "string key, Func<object, T> read, int length = -1) where T : class",
            ),
            "Json.Limits": (
                "staticmethod",
                "public static void Limits(int slot, out double? min, ref int count, "
                "params string[] keys)",
            ),
            "Json.Available": (
                "staticmethod",
                "public static bool Available(int index)",
            ),
            "ITarget.Name": ("property", "string Name { get; }"),
            "ITarget.Open": ("method", "void Open(string path)"),
            "Status.Done": ("constant", "Done"),
            "Status.Partial": ("constant", "Partial = 2"),
            "Status.Halted": ("constant", "Halted"),
            "Vec.this": ("indexer", "public double this[int i] { get; }"),
            "Decide": (
                "delegate",
                "public delegate bool Decide(object target, string action)",
            ),
        }
        self.assertEqual(spelled, expected)
        outcome = {
            t: rows[t].signature
            for t in CS_TARGETS
            if t.startswith("Test.Core.Outcome.")
        }
        self.assertEqual(
            outcome,
            {
                "Test.Core.Outcome.op": "public string op",
                "Test.Core.Outcome.status": "public string status",
                "Test.Core.Outcome.error": "public string error",
            },
        )

    def test_the_doc_reads_as_a_docstring_and_the_record_knows_its_file_and_line(self):
        rows = self.records()
        should = rows["Test.Kernel.OpContext.ShouldChange"]
        self.assertEqual(
            should.summary, "Ask before changing `target`; true: make the change."
        )
        self.assertEqual(
            should.doc,
            "Ask before changing `target`; true: make the change.\n\n"
            "Parameters:\n"
            "    target: What would change.\n"
            "    action: What would be done to it.\n\n"
            "Returns:\n"
            "    Whether to make it.\n\n"
            "Raises:\n"
            "    OperationCanceledException: The run was cancelled.",
        )
        self.assertTrue(should.path.endswith("OpContext.cs"))
        lines = CS_KERNEL.splitlines()
        self.assertIn("public bool ShouldChange(", lines[should.line - 1])
        ctor = rows["Test.Kernel.OpContext.OpContext"]
        self.assertEqual(ctor.summary, "")
        self.assertIn("    dryRun: Whether this run only plans.", ctor.doc)
        self.assertEqual(
            rows["Test.Kernel.Status.Partial"].summary, "It stopped part way."
        )
        self.assertEqual(
            rows["Test.Core.Outcome.error"].summary, "A word on how it went."
        )

    def test_obsolete_is_deprecated_with_the_release_its_message_names(self):
        rows = self.records()
        allow = rows["Test.Kernel.OpContext.Allow"]
        self.assertEqual((allow.deprecated, allow.remove_in), (True, "0.9.0"))
        self.assertEqual(allow.since, "2026-10-01")
        halted = rows["Test.Kernel.Status.Halted"]
        self.assertEqual((halted.deprecated, halted.remove_in), (True, "0.9.0"))
        self.assertFalse(rows["Test.Kernel.OpContext.ShouldChange"].deprecated)

    def test_a_partial_type_is_one_type_and_a_generated_file_is_api(self):
        rows = self.records()
        record = rows["Record"]
        self.assertEqual(record.summary, "The record's declared shape.")
        self.assertTrue(record.path.endswith("Record.g.cs"))
        self.assertEqual([m.name for m in record.members], ["Read", "CurrentVersion"])

    def test_a_static_member_says_how_code_calls_it(self):
        rows = self.records()
        calls = {
            t: getattr(rows[t], "call", "")
            for t in (
                "Test.Kernel.Json.Run",
                "Test.Kernel.Json.Limits",
                "Test.Kernel.Json.Objs",
                "Test.Kernel.OpContext.OpContext",
                "Test.Kernel.OpContext.IdMax",
                "Test.Kernel.Status.Done",
                "Record.Read",
                "Test.Kernel.OpContext.ShouldChange",
            )
        }
        self.assertEqual(
            calls,
            {
                "Test.Kernel.Json.Run": "return Test.Kernel.Json.Run(code);",
                "Test.Kernel.Json.Limits": (
                    "Test.Kernel.Json.Limits(slot, out var min, ref count, keys);"
                ),
                "Test.Kernel.Json.Objs": "return Test.Kernel.Json.Objs<T>(o, key, read, length);",
                "Test.Kernel.OpContext.OpContext": (
                    "return new Test.Kernel.OpContext(dryRun, cancelled);"
                ),
                "Test.Kernel.OpContext.IdMax": "return Test.Kernel.OpContext.IdMax;",
                "Test.Kernel.Status.Done": "return Test.Kernel.Status.Done;",
                "Record.Read": "return Record.Read(value);",
                "Test.Kernel.OpContext.ShouldChange": "",  # an instance's
            },
        )


class TestCSharpSearchAndDescribe(_CSharpCase):
    """``api.search`` and ``api.describe`` answer for C# as for Python, and
    say where each runs and how code calls it."""

    def test_search_finds_csharp_beside_the_python_and_says_how_to_call_it(self):
        with self.registered():
            should, context = ApiSurface.search("OpContext ShouldChange")
            (tool,) = ApiSurface.search("tool", packages=[CS_PKG])
            found = [r["target"] for r in ApiSurface.search("", limit=None)]
        self.assertEqual(
            should,
            {
                "target": "Test.Kernel.OpContext.ShouldChange",
                "kind": "method",
                "signature": 'public bool ShouldChange(object target, string action = "a;b{")',
                "summary": "Ask before changing `target`; true: make the change.",
                "host": "unity",
                "exec": "unity.csharp.exec",
            },
        )
        self.assertEqual(context["target"], "Test.Kernel.OpContext")  # in its summary
        # Its host runs its C#; its Python runs in any Python.
        self.assertEqual(
            tool,
            {
                "target": f"{CS_PKG}.tools:tool",
                "kind": "function",
                "signature": "tool() -> int",
                "summary": "A tool.",
                "exec": "python.exec",
            },
        )
        self.assertEqual(sorted(t for t in found if ":" not in t), CS_TARGETS)
        self.assertNotIn(CS_PKG, sys.modules, "search imported the package")

    def test_overloads_are_told_apart_and_a_retired_one_says_so(self):
        with self.registered():
            writes = ApiSurface.search("Json Write")
            (allow,) = ApiSurface.search("OpContext Allow")
        self.assertEqual(
            [r["target"] for r in writes][:2],  # then the class, "written"
            ["Test.Kernel.Json.Write(object)", "Test.Kernel.Json.Write(object, bool)"],
        )
        self.assertEqual((allow["deprecated"], allow["remove_in"]), (True, "0.9.0"))

    def test_describe_reads_a_member_and_a_type(self):
        with self.registered():
            run = ApiSurface.describe("Test.Kernel.Json.Run")
            context = ApiSurface.describe("Test.Kernel.OpContext")
        self.assertEqual(run["kind"], "staticmethod")
        self.assertEqual(run["host"], "unity")
        self.assertEqual(run["exec"], "unity.csharp.exec")
        self.assertEqual(run["call"], "return Test.Kernel.Json.Run(code);")
        self.assertEqual(run["doc"], "Run code.")
        self.assertRegex(run["file"], r"OpContext\.cs:\d+$")
        self.assertEqual(context["kind"], "class")
        self.assertEqual(
            context["signature"],
            "public sealed class OpContext : IDisposable, IComparable<OpContext>",
        )
        self.assertEqual(context["bases"], ["IDisposable", "IComparable<OpContext>"])
        self.assertIn(
            {
                "name": "DryRun",
                "kind": "property",
                "signature": "public bool DryRun { get; }",
                "summary": "Whether this run only plans: every change is listed, none made.",
            },
            context["members"],
        )

    def test_describe_refuses_what_is_no_public_csharp_name(self):
        with self.registered():
            for target, why in (
                ("Test.Kernel.OperationAttribute", "no public C# name"),
                ("Test.Kernel.Json.Write", "did you mean"),
                ("Test.Kernel.OpContext.Hidden", "no public C# name"),
            ):
                with (
                    self.subTest(target=target),
                    self.assertRaisesRegex(ValueError, why),
                ):
                    ApiSurface.describe(target)


class TestCSharpReadCache(_CSharpCase):
    """The C# read rides the Python read's cache: one per source, read again
    when a file changes, kept on disk with what describe shows."""

    def setUp(self):
        super().setUp()
        cache = self.temp.dir_path("cache")
        env = patch.dict(os.environ, {"PYTHONTK_CACHE_ROOT": cache})
        env.start()
        self.addCleanup(env.stop)
        keeps = patch.object(ApiSurface, "_keeps_on_disk", return_value=True)
        keeps.start()
        self.addCleanup(keeps.stop)
        self._forget()

    @staticmethod
    def _forget():
        ApiSurface._READS.clear()
        ApiSurface._CS_PARSED.clear()

    def test_an_edited_file_is_read_again_and_an_unchanged_tree_is_not(self):
        first = ApiSurface.read_csharp(CS_PKG)
        self.assertIs(ApiSurface.read_csharp(CS_PKG), first)
        parsed, real = [], ApiSurface._cs_parse_file

        def parse(path, *args):
            parsed.append(Path(path).name)
            return real(path, *args)

        time.sleep(0.05)
        self.write(
            self.csharp / "Core" / "Outcome.cs",
            CS_CORE.replace(
                "public string op;", "public string op;\n    public int code;"
            ),
        )
        with patch.object(ApiSurface, "_cs_parse_file", side_effect=parse):
            again = ApiSurface.read_csharp(CS_PKG)
        self.assertEqual(parsed, ["Outcome.cs"])
        self.assertIn("Test.Core.Outcome.code", dict(ApiSurface._csharp_rows(CS_PKG)))
        self.assertIsNot(again, first)

    def test_a_new_process_reads_it_from_disk_with_what_describe_shows(self):
        with self.registered():
            described = ApiSurface.describe("Test.Kernel.Json.Objs")
            context = ApiSurface.describe("Test.Kernel.OpContext")
        self._forget()
        with (
            self.registered(),
            patch.object(
                ApiSurface, "_cs_parse_file", side_effect=AssertionError("parsed again")
            ),
        ):
            self.assertEqual(ApiSurface.describe("Test.Kernel.Json.Objs"), described)
            self.assertEqual(ApiSurface.describe("Test.Kernel.OpContext"), context)
        python = ApiSurface._cache_file(self.source, "x")
        csharp = ApiSurface._cache_file(self.source, "x", "csharp")
        self.assertNotEqual(python, csharp)  # a folder read both ways keeps two


class TestSearchSkipsWhatItCannotLocate(_PackageCase):
    def test_an_unlocatable_package_after_one_whose_last_row_matches_is_skipped(self):
        """An empty query matches every row, which once left the loop's flag
        saying "packages named": the next unlocatable one then raised."""
        with (
            self.registered(),
            patch.object(
                PackageManager,
                "editable_packages",
                return_value=["no_such_package_here"],
            ),
            self.assertLogs("pythontk.core_utils.api_surface.api_surface", "WARNING"),
        ):
            self.assertTrue(ApiSurface.search("", limit=None))


class _Mixin:
    @property
    def mixed(self) -> int:
        """A property from a mixin."""
        return 1

    @staticmethod
    def static_mixed() -> None:
        """A static method from a mixin."""


class Live(_Mixin):
    """A live class."""

    LABEL = "x"

    def own(self, x: int = 1) -> int:
        """Its own method."""
        return x

    @classmethod
    def make(cls) -> "Live":
        """Make one."""
        return cls()

    async def wait(self) -> None:
        """Wait."""

    def gen(self):
        """Yield."""
        yield 1

    def retired(self):
        """Retired."""


Live.retired.__deprecated__ = "use own"
Live.retired.__deprecated_record__ = SimpleNamespace(remove_in="0.22.0")


class TestLive(unittest.TestCase):
    def test_members_read_kinds_through_the_mro(self):
        kinds = {r.name: r.kind for r in ApiSurface.members(Live)}
        self.assertEqual(kinds["mixed"], "property")  # inherited: still a property
        self.assertEqual(kinds["static_mixed"], "staticmethod")
        self.assertEqual(kinds["make"], "classmethod")
        self.assertEqual(kinds["own"], "method")
        self.assertEqual(kinds["LABEL"], "attribute")
        self.assertNotIn(
            "mixed", {r.name for r in ApiSurface.members(Live, inherited=False)}
        )
        self.assertEqual(
            {r.name for r in ApiSurface.members(Live, "properties")}, {"mixed"}
        )

    def test_a_record_is_the_registry_shape_and_enrich_adds_the_live_facts(self):
        own = ApiSurface.record(Live, "own")
        self.assertEqual(
            (own.qualname, own.signature, own.summary),
            ("Live.own", "(self, x: int = 1) -> int", "Its own method."),
        )
        self.assertIsNone(ApiSurface.record(Live, "nope"))
        retired = ApiSurface.record(Live, "retired")
        self.assertEqual((retired.deprecated, retired.remove_in), (True, "0.22.0"))
        data = ApiSurface.enrich(Live, ApiSurface.record(Live, "static_mixed"))
        self.assertEqual(data["defined_in"], "_Mixin")
        self.assertIn("test_api_surface.py:", data["source"])
        self.assertEqual(ApiSurface.flags(Live.wait), ["async"])
        self.assertEqual(ApiSurface.flags(Live.gen), ["generator"])

    def test_exports_are_what_the_root_publishes_resolved(self):
        """The root namespace, live: what no static read can see (a name the
        root drops while its class stays); ``python -m pythontk --index`` and
        the surface snapshot read it."""
        rows = {r["name"]: r for r in ApiSurface.exports("pythontk")}
        self.assertEqual(
            (
                rows["ApiSurface"]["tier"],
                rows["ApiSurface"]["module"],
                rows["ApiSurface"]["kind"],
            ),
            ("root", "pythontk.core_utils.api_surface.api_surface", "class"),
        )
        self.assertEqual(
            (rows["filter_list"]["tier"], rows["filter_list"]["qualname"]),
            ("bare", "IterUtils.filter_list"),
        )

    def test_signature_location_and_source_read_through_anything(self):
        self.assertEqual(ApiSurface.signature_of(Live.own), "(self, x: int = 1) -> int")
        self.assertEqual(ApiSurface.signature_of(Live.mixed, "property"), "")
        self.assertEqual(ApiSurface.signature_of(len), "(obj, /)")
        self.assertIsNone(ApiSurface.location(len))
        self.assertIn("def own", ApiSurface.source_of(Live.own))
        self.assertEqual(
            os.path.basename(ApiSurface.location(Live).rsplit(":", 1)[0]),
            "test_api_surface.py",
        )


if __name__ == "__main__":
    unittest.main()
