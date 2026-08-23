#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk RenamePlan and FileNaming.

Covers (RenamePlan):
- live apply calls the strategy only for changed entries, parallel results
- dry run never calls the strategy, returns the plan
- a failing item is reported and keeps its old name without aborting the batch
- a host that uniquifies is reported and the actual name is returned
- the report is one log_group record (not one per item) + a summary line

Covers (FileNaming):
- expand(): files + non-recursive directory contents, missing paths dropped
- find(): stem matching (wildcards / regex / ignore_case), extension untouched
- rename(): pattern grammar on stems, extension preserved, dry run is a no-op
- rename(): same stem in two directories both renamed
- set_case() / strip_chars(): stems only; empty stems skipped
- collision never overwrites an existing file

Run with:
    python -m pytest test_file_naming.py -v
    python test_file_naming.py
"""

import logging
import os
import shutil
import tempfile
import unittest
from unittest import mock

from pythontk.file_utils.file_naming import FileNaming, RenamePlan


class _Sink(logging.Handler):
    """Collect emitted records (raw log_group records included)."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _logger_with_sink():
    logger = RenamePlan.logger
    sink = _Sink()
    logger.addHandler(sink)
    return logger, sink


class TestRenamePlan(unittest.TestCase):
    def setUp(self):
        self.logger, self.sink = _logger_with_sink()

    def tearDown(self):
        self.logger.removeHandler(self.sink)

    def test_live_apply_only_changed(self):
        calls = []

        def rename(key, new):
            calls.append((key, new))
            return new

        plan = [("k1", "a", "b"), ("k2", "c", "c"), ("k3", "d", "e")]
        result = RenamePlan.apply(plan, rename, logger=self.logger)
        self.assertEqual(calls, [("k1", "b"), ("k3", "e")])
        self.assertEqual(result, [("a", "b"), ("c", "c"), ("d", "e")])

    def test_dry_run_never_renames(self):
        rename = mock.Mock()
        plan = [("k1", "a", "b")]
        result = RenamePlan.apply(plan, rename, dry_run=True, logger=self.logger)
        rename.assert_not_called()
        self.assertEqual(result, [("a", "b")])
        texts = [r.getMessage() for r in self.sink.records]
        self.assertTrue(any("DRY RUN" in t for t in texts))
        self.assertTrue(any("nothing was changed" in t for t in texts))

    def test_failure_keeps_old_name_and_continues(self):
        def rename(key, new):
            if key == "bad":
                raise RuntimeError("locked")
            return new

        plan = [("bad", "a", "b"), ("ok", "c", "d")]
        result = RenamePlan.apply(plan, rename, logger=self.logger)
        self.assertEqual(result, [("a", "a"), ("c", "d")])
        warnings = [r for r in self.sink.records if r.levelno == logging.WARNING]
        self.assertEqual(len(warnings), 1)
        self.assertIn("locked", warnings[0].getMessage())

    def test_uniquified_name_is_reported_and_returned(self):
        result = RenamePlan.apply(
            [("k", "a", "b")], lambda k, n: n + "1", logger=self.logger
        )
        self.assertEqual(result, [("a", "b1")])
        texts = [r.getMessage() for r in self.sink.records]
        self.assertTrue(any("instead of 'b'" in t for t in texts))

    def test_report_is_one_group_record(self):
        plan = [(f"k{i}", f"old{i}", f"new{i}") for i in range(5)]
        RenamePlan.apply(plan, lambda k, n: n, title="Op", logger=self.logger)
        raw = [r for r in self.sink.records if getattr(r, "raw", False)]
        self.assertEqual(len(raw), 1)
        msg = raw[0].getMessage()
        self.assertIn("Op", msg)
        self.assertIn("5 of 5", msg)
        for i in range(5):
            self.assertIn(f"old{i} → <b>new{i}</b>", msg)
        summary = [r for r in self.sink.records if r.levelname == "RESULT"]
        self.assertEqual(len(summary), 1)

    def test_link_renders_item_name(self):
        RenamePlan.apply(
            [("k", "a", "b")],
            lambda k, n: n,
            logger=self.logger,
            link=lambda key, name: f"<a href='x://{key}'>{name}</a>",
        )
        raw = [r for r in self.sink.records if getattr(r, "raw", False)]
        self.assertIn("<a href='x://k'>a</a> → <b>b</b>", raw[0].getMessage())

    def test_empty_plan(self):
        result = RenamePlan.apply([], lambda k, n: n, logger=self.logger)
        self.assertEqual(result, [])
        self.assertTrue(
            any("nothing in scope" in r.getMessage() for r in self.sink.records)
        )

    def test_plain_stdlib_logger(self):
        """A caller-supplied plain logger (no notice/result/log_group) must work."""
        plain = logging.getLogger("file_naming_plain_test")
        plain.propagate = False
        plain.setLevel(logging.DEBUG)
        sink = _Sink()
        plain.addHandler(sink)
        try:
            RenamePlan.apply([("k", "a", "b")], lambda k, n: n, logger=plain)
            RenamePlan.apply(
                [("k", "a", "b")], lambda k, n: n, logger=plain, dry_run=True
            )
            RenamePlan.apply([], lambda k, n: n, logger=plain)
        finally:
            plain.removeHandler(sink)
        texts = [r.getMessage() for r in sink.records]
        self.assertTrue(any("renamed 1 item" in t for t in texts), texts)
        self.assertTrue(any("  a → b" == t for t in texts), texts)  # html stripped
        self.assertTrue(any("DRY RUN" in t for t in texts), texts)
        self.assertTrue(any("nothing in scope" in t for t in texts), texts)


class TestFileNaming(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="fn_test_")
        self.sub = os.path.join(self.dir, "sub")
        os.mkdir(self.sub)
        self.files = {}
        for name in ("pCube1.png", "pCube2.PNG", "sphere.txt", "ReadMe.md"):
            self.files[name] = self.touch(os.path.join(self.dir, name))
        self.touch(os.path.join(self.sub, "pCube1.png"))  # same stem, other dir
        self.touch(os.path.join(self.sub, "nested.txt"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def touch(self, path):
        with open(path, "wb") as f:
            f.write(b"x")
        return path

    def names(self, directory=None):
        return sorted(os.listdir(directory or self.dir))

    # -- expand / find -------------------------------------------------

    def test_expand_dir_is_non_recursive_and_sorted(self):
        files = FileNaming.expand(self.dir)
        self.assertEqual(
            [os.path.basename(f) for f in files],
            sorted(["pCube1.png", "pCube2.PNG", "sphere.txt", "ReadMe.md"]),
        )

    def test_expand_mixed_and_missing(self):
        missing = os.path.join(self.dir, "nope.txt")
        files = FileNaming.expand([self.files["sphere.txt"], missing, self.sub])
        self.assertEqual(
            [os.path.basename(f) for f in files],
            ["sphere.txt", "nested.txt", "pCube1.png"],
        )

    def test_find_matches_stem_not_extension(self):
        hits = FileNaming.find(self.dir, "pCube*")
        self.assertEqual(
            sorted(os.path.basename(h) for h in hits), ["pCube1.png", "pCube2.PNG"]
        )
        self.assertEqual(FileNaming.find(self.dir, "*png"), [])

    def test_find_regex_and_ignore_case(self):
        hits = FileNaming.find(self.dir, r"cube\d$", regex=True, ignore_case=True)
        self.assertEqual(len(hits), 2)
        self.assertEqual(FileNaming.find(self.dir, "readme"), [])
        self.assertEqual(len(FileNaming.find(self.dir, "readme", ignore_case=True)), 1)

    def test_find_empty_filter_returns_all(self):
        self.assertEqual(len(FileNaming.find(self.dir, "")), 4)

    # -- rename --------------------------------------------------------

    def test_rename_pattern_preserves_extension(self):
        result = FileNaming.rename(self.dir, "*box*", "*Cube*")
        renamed = {os.path.basename(o): os.path.basename(n) for o, n in result}
        self.assertEqual(
            renamed, {"pCube1.png": "pbox1.png", "pCube2.PNG": "pbox2.PNG"}
        )
        self.assertIn("pbox1.png", self.names())
        self.assertIn("pbox2.PNG", self.names())
        self.assertNotIn("pCube1.png", self.names())

    def test_rename_dry_run_touches_nothing(self):
        before = self.names()
        result = FileNaming.rename(self.dir, "**_v2", "pCube*", dry_run=True)
        self.assertEqual(self.names(), before)
        self.assertEqual(
            sorted(os.path.basename(n) for _, n in result),
            ["pCube1_v2.png", "pCube2_v2.PNG"],
        )

    def test_rename_same_stem_in_two_dirs(self):
        result = FileNaming.rename([self.dir, self.sub], "**_A", "pCube1")
        self.assertEqual(len(result), 2)
        self.assertIn("pCube1_A.png", self.names())
        self.assertIn("pCube1_A.png", self.names(self.sub))

    def test_rename_never_overwrites(self):
        # 'sphere' -> 'pCube1' would collide with the existing pCube1.png? No:
        # different extension. Build a true collision: two stems to one name.
        self.touch(os.path.join(self.dir, "pCube3.png"))
        result = FileNaming.rename(self.dir, "same", "pCube1|pCube3")
        new_names = [os.path.basename(n) for _, n in result]
        self.assertEqual(new_names.count("same.png"), 1)
        self.assertEqual(self.names().count("same.png"), 1)
        files = [n for n in self.names() if os.path.isfile(os.path.join(self.dir, n))]
        self.assertEqual(len(files), 5)  # nothing lost

    def test_rename_never_produces_an_unusable_name(self):
        """An emptied stem or one carrying a path separator is dropped, not written."""
        before = self.names()
        result = FileNaming.rename(self.dir, "", "sphere")  # strip the whole stem
        self.assertEqual(result, [])
        result = FileNaming.rename(self.dir, "sub/x", "sphere")  # would become a path
        self.assertEqual(result, [])
        self.assertEqual(self.names(), before)

    def test_rename_invalid_regex_is_reported_not_raised(self):
        result = FileNaming.rename(self.dir, "x", "(", regex=True)
        self.assertEqual(result, [])

    # -- set_case / strip_chars ------------------------------------------

    def test_set_case_stems_only(self):
        FileNaming.set_case(self.files["ReadMe.md"], "upper")
        self.assertIn("README.md", self.names())

    def test_strip_chars(self):
        FileNaming.strip_chars(self.files["sphere.txt"], num_chars=2, trailing=True)
        self.assertIn("sphe.txt", self.names())
        FileNaming.strip_chars(os.path.join(self.dir, "sphe.txt"), num_chars=1)
        self.assertIn("phe.txt", self.names())

    def test_strip_chars_skips_emptied_stem(self):
        before = self.names()
        result = FileNaming.strip_chars(self.files["sphere.txt"], num_chars=6)
        self.assertEqual(result, [])
        self.assertEqual(self.names(), before)


    def test_strip_chars_zero_is_a_no_op_in_both_directions(self):
        """A count of zero must change nothing -- and must not "skip" everything.

        ``s[:-0]`` is the EMPTY string, so the trailing branch proposed an empty
        stem for every file and each one came back as a "not a valid file name"
        skip, while the leading branch (``s[0:]``) was already a clean no-op. A
        spinbox cleared to 0 reaches this.
        """
        for trailing in (False, True):
            with self.subTest(trailing=trailing):
                before = self.names()
                result = FileNaming.strip_chars(
                    self.dir, num_chars=0, trailing=trailing
                )
                self.assertEqual(result, [])
                self.assertEqual(self.names(), before)

if __name__ == "__main__":
    unittest.main()
