# !/usr/bin/python
# coding=utf-8
"""Tests for the ``python -m pythontk`` CLI shell (``pythontk/__main__.py``)."""
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout

from pythontk.__main__ import _main


class TestIndexFlag(unittest.TestCase):
    """`--index` — the live twin of the generated API_INDEX.md."""

    def run_cli(self, argv):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = _main(argv)
        return rc, out.getvalue()

    def test_index_lists_public_surface(self):
        rc, out = self.run_cli(["--index"])
        self.assertEqual(rc, 0)
        for expected in ("CoreUtils", "MapFactory", "TempArtifacts", "filter_list"):
            self.assertIn(expected, out)

    def test_index_json_rows_carry_name_module_kind_tier(self):
        rc, out = self.run_cli(["--index", "--json"])
        self.assertEqual(rc, 0)
        rows = json.loads(out)
        by_name = {r["name"]: r for r in rows}
        self.assertEqual(
            by_name["CoreUtils"]["module"], "pythontk.core_utils._core_utils"
        )
        self.assertEqual(by_name["CoreUtils"]["kind"], "class")
        self.assertEqual(by_name["CoreUtils"]["tier"], "root")
        # Bare wildcard aliases are the second tier, resolved to their owner.
        self.assertEqual(by_name["filter_list"]["tier"], "bare")
        self.assertEqual(by_name["filter_list"]["qualname"], "IterUtils.filter_list")
        # A constant's location is where it is registered, not its type's home
        # (a plain dict's own __module__ would report "builtins").
        self.assertTrue(
            by_name["DEFAULT_FILE_RULES"]["module"].endswith("file_utils.workspace")
        )
        # Every row is fully populated — no silent unresolvables in a clean env.
        self.assertTrue(all(r["module"] for r in rows), "unresolvable symbols in index")

    def test_index_covers_both_tiers(self):
        import pythontk

        rc, out = self.run_cli(["--index", "--json"])
        self.assertEqual(rc, 0)
        names = {r["name"] for r in json.loads(out)}
        self.assertLessEqual(set(pythontk.__all__), names)
        self.assertLessEqual(
            set(pythontk.METHOD_TO_MODULE) - set(pythontk.__all__), names
        )

    def test_index_locations_are_valid_cli_targets(self):
        # The listing is only useful if what it prints can be fed back in.
        rc, out = self.run_cli(["--index", "--json"])
        rows = {r["name"]: r for r in json.loads(out)}
        row = rows["filter_list"]
        rc, out = self.run_cli([f"{row['module']}.{row['qualname']}", "--where"])
        self.assertEqual(rc, 0)
        self.assertIn("_iter_utils", out)

    def test_target_still_required_without_index(self):
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            self.run_cli([])

    def test_index_with_a_target_is_rejected_not_silently_ignored(self):
        """`python -m pythontk sometarget --index` must not silently drop the
        target -- the combination is meaningless (--index lists everything,
        a target asks about one thing), so it should error like argparse's
        other mutually-exclusive combinations, not run --index and pretend
        the target was never given."""
        with self.assertRaises(SystemExit) as ctx, redirect_stderr(io.StringIO()):
            self.run_cli(["pythontk.CoreUtils", "--index"])
        self.assertEqual(ctx.exception.code, 2)

    def test_member_introspection_still_works(self):
        rc, out = self.run_cli(["pythontk.CoreUtils", "listify", "--signature"])
        self.assertEqual(rc, 0)
        self.assertIn("listify", out)


class TestMissingMemberExitCode(unittest.TestCase):
    """A typo'd member must fail like every other user error in this CLI.

    ``HelpMixin.help/source/where/signature(..., returns=True)`` hand back the
    "has no member" text as an ordinary *value*, so the CLI used to print it to
    stdout and return 0 -- making a bad member the one mistake that reports
    success. `--source > file || fallback` then never took the fallback and
    wrote the error text into the file.
    """

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = _main(argv)
        return rc, out.getvalue(), err.getvalue()

    def test_missing_member_help_exits_nonzero_on_stderr(self):
        rc, out, err = self.run_cli(["pythontk.CoreUtils", "no_such_member"])
        self.assertNotEqual(rc, 0)
        self.assertIn("has no member 'no_such_member'", err)
        self.assertEqual(out, "")

    def test_missing_member_source_exits_nonzero_on_stderr(self):
        # The verified consequence: `--source > file || fallback` must be able
        # to tell "wrote source" from "wrote an error message".
        rc, out, err = self.run_cli(
            ["pythontk.CoreUtils", "no_such_member", "--source"]
        )
        self.assertNotEqual(rc, 0)
        self.assertIn("has no member 'no_such_member'", err)
        self.assertEqual(out, "")

    def test_missing_member_where_exits_nonzero_on_stderr(self):
        rc, out, err = self.run_cli(["pythontk.CoreUtils", "no_such_member", "--where"])
        self.assertNotEqual(rc, 0)
        self.assertIn("has no member 'no_such_member'", err)
        self.assertEqual(out, "")

    def test_missing_member_signature_exits_nonzero_on_stderr(self):
        rc, out, err = self.run_cli(
            ["pythontk.CoreUtils", "no_such_member", "--signature"]
        )
        self.assertNotEqual(rc, 0)
        self.assertIn("has no member 'no_such_member'", err)
        self.assertEqual(out, "")

    def test_missing_member_json_keeps_error_payload_on_stdout(self):
        # The {"error": ...} envelope is a machine contract - it stays on
        # stdout and keeps its shape; only the exit code changes.
        rc, out, err = self.run_cli(["pythontk.CoreUtils", "no_such_member", "--json"])
        self.assertNotEqual(rc, 0)
        payload = json.loads(out)
        self.assertIn("has no member 'no_such_member'", payload["error"])
        self.assertEqual(err, "")

    def test_missing_member_matches_the_non_helpmixin_target_exit_code(self):
        """The same mistake on a plain module already fails (AttributeError ->
        exit 1); the HelpMixin branch must not be the odd one out."""
        with self.assertRaises(AttributeError):
            self.run_cli(["os", "no_such_member"])

    def test_present_member_still_succeeds(self):
        # Guards the getattr probe against over-reach: a member that *does*
        # resolve must still take the normal path on every flag.
        cases = (
            ([], "listify"),
            (["--source"], "def listify"),
            (["--where"], "_core_utils.py"),
            (["--signature"], "listify"),
            (["--json"], "listify"),
        )
        for flags, needle in cases:
            with self.subTest(flags=flags):
                rc, out, err = self.run_cli(["pythontk.CoreUtils", "listify"] + flags)
                self.assertEqual(rc, 0)
                self.assertIn(needle, out)
                self.assertEqual(err, "")


if __name__ == "__main__":
    unittest.main()
