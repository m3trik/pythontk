# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.agent_instructions: AgentInstructions (check, sync,
measure, duplicates) and the pack hook it writes."""

import codecs
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pythontk import AgentInstructions, OpContext

A = AgentInstructions

#: The hook sync wrote before it ran without a shell and skipped what it cannot
#: read: a repo holding it is sync's own, and is brought up to date.
EARLIER_HOOK = r'''"""Claude Code SessionStart hook, written by pythontk's AgentInstructions.sync
(edit its template, not this copy): adds this codebase's agent packs to a session
rooted here, but none a CLAUDE.md above already imports (a workspace root's stub
imports its own, and Claude Code loads it in every session below it)."""
import json
from pathlib import Path

LIMIT = 10000
here = Path(__file__).resolve().parent
provided = set()
for parent in here.parent.parents:
    stub = parent / "CLAUDE.md"
    lines = stub.read_text(encoding="utf-8-sig", errors="replace").splitlines() if stub.is_file() else []
    for line in lines:
        ref = Path(line.strip()[1:])
        if line.startswith("@") and ref.parent.as_posix().endswith(".agents/packs"):
            provided.add(ref.name)
packs = sorted((here / "packs").glob("*.md"))
text = "\n\n".join(p.read_text(encoding="utf-8") for p in packs if p.name not in provided)
if text.strip():
    out = {"hookEventName": "SessionStart", "additionalContext": text[:LIMIT]}
    print(json.dumps({"hookSpecificOutput": out}))
'''

#: How sync registered the hook before: a shell string PowerShell cannot run.
SHELL_ENTRY = {
    "type": "command",
    "command": 'python "$CLAUDE_PROJECT_DIR/.agents/session_hook.py"'
    ' || python3 "$CLAUDE_PROJECT_DIR/.agents/session_hook.py"',
    "timeout": 10,
}


class _Tree(unittest.TestCase):
    """A temporary tree: a packs source and a repo whose CLAUDE.md holds rules."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # Resolved: the plan names every file resolved, and a TEMP spelled
        # with 8.3 short names (a Windows runner's RUNNER~1) differs.
        self.base = Path(self._tmp.name).resolve()
        self.source = self.base / "packs_source"
        self.source.mkdir()
        (self.source / "principles.md").write_text(
            "# Principles\n- Adopt before building.\n", encoding="utf-8"
        )
        (self.source / "python.md").write_text(
            "# Python\n- Lazy imports.\n", encoding="utf-8"
        )
        self.repo = self.base / "repo"
        (self.repo / "sub").mkdir(parents=True)
        self.write("repo/CLAUDE.md", "# repo\n\n- Keep the repo tidy (§3).\n")
        self.write("repo/sub/CLAUDE.md", "# sub\n\n- Sub rules.\n")

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, relative: str, text: str) -> Path:
        path = self.base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def read(self, relative: str) -> str:
        return (self.base / relative).read_text(encoding="utf-8")

    def codes(self, **kwargs):
        return sorted(
            {p["code"] for p in A.check(str(self.repo), **kwargs)["problems"]}
        )


class TestSync(_Tree):
    def test_a_dry_run_lists_every_write_and_writes_nothing(self):
        ctx = OpContext(dry_run=True)
        planned = A.sync(
            str(self.repo), str(self.source), packs=["principles"], ctx=ctx
        )["written"]
        self.assertIn(str(self.repo / "AGENTS.md"), planned)
        self.assertIn(str(self.repo / ".agents" / "packs" / "principles.md"), planned)
        self.assertTrue(all(c.kind == "write" for c in ctx.changes))
        self.assertFalse((self.repo / "AGENTS.md").exists())
        self.assertFalse((self.repo / ".agents").exists())

    def test_rules_move_to_agents_md_and_claude_md_becomes_the_stub(self):
        A.sync(str(self.repo))
        self.assertEqual(self.read("repo/CLAUDE.md"), "@AGENTS.md\n")
        self.assertEqual(self.read("repo/sub/CLAUDE.md"), "@AGENTS.md\n")
        self.assertIn("Keep the repo tidy", self.read("repo/AGENTS.md"))
        self.assertIn("Sub rules", self.read("repo/sub/AGENTS.md"))

    def test_a_repo_gets_its_packs_a_pointer_and_the_hook(self):
        A.sync(str(self.repo), str(self.source), packs=["python", "principles"])
        agents = self.read("repo/AGENTS.md")
        self.assertTrue(agents.startswith("# repo\n\n" + A.POINTER_BEGIN))
        self.assertIn("[principles](.agents/packs/principles.md) · [python]", agents)
        self.assertEqual(
            self.read("repo/.agents/packs/python.md"), "# Python\n- Lazy imports.\n"
        )
        self.assertEqual(self.read("repo/.agents/session_hook.py"), A.HOOK_SCRIPT)
        settings = json.loads(self.read("repo/.claude/settings.json"))
        (entry,) = settings["hooks"]["SessionStart"][0]["hooks"]
        self.assertEqual(entry["args"], [f"${{CLAUDE_PROJECT_DIR}}/{A.HOOK_FILE}"])
        self.assertEqual(self.read("repo/CLAUDE.md"), "@AGENTS.md\n")
        self.assertEqual(self.codes(source=str(self.source)), [])

    def test_a_second_sync_changes_nothing_and_keeps_other_settings(self):
        self.write(
            "repo/.claude/settings.json",
            json.dumps({"permissions": {"allow": ["Read"]}}),
        )
        A.sync(str(self.repo), str(self.source), packs=["principles"])
        first = {
            p: p.read_text(encoding="utf-8")
            for p in self.repo.rglob("*")
            if p.is_file()
        }
        self.assertEqual(A.sync(str(self.repo), str(self.source))["written"], [])
        second = {
            p: p.read_text(encoding="utf-8")
            for p in self.repo.rglob("*")
            if p.is_file()
        }
        self.assertEqual(first, second)
        settings = json.loads(self.read("repo/.claude/settings.json"))
        self.assertEqual(settings["permissions"], {"allow": ["Read"]})
        self.assertEqual(len(settings["hooks"]["SessionStart"]), 1)

    def test_a_path_scoped_pack_is_a_rule_and_skips_the_hook_and_the_imports(self):
        (self.source / "csharp.md").write_text(
            '---\npaths:\n  - "**/*.cs"\n---\n# C#\n- Seal it.\n', encoding="utf-8"
        )
        A.sync(str(self.repo), str(self.source), packs=["csharp", "principles"])
        self.assertTrue((self.repo / ".claude/rules/csharp.md").is_file())
        self.assertFalse((self.repo / ".agents/packs/csharp.md").exists())
        self.assertIn("[csharp](.claude/rules/csharp.md)", self.read("repo/AGENTS.md"))
        self.assertEqual(self.codes(source=str(self.source)), [])
        A.sync(
            str(self.base),
            str(self.source),
            packs=["csharp", "principles"],
            mode="import",
        )
        self.assertEqual(
            self.read("CLAUDE.md"), "@AGENTS.md\n@.agents/packs/principles.md\n"
        )
        launch = [Path(i["path"]).name for i in A.measure(str(self.repo))["launch"]]
        self.assertNotIn("csharp.md", launch)
        touched = A.measure(str(self.repo), [str(self.repo / "src" / "a.cs")])[
            "touched"
        ]
        self.assertIn("csharp.md", [Path(i["path"]).name for i in touched])

    def test_a_pack_that_becomes_path_scoped_moves(self):
        A.sync(str(self.repo), str(self.source), packs=["python"])
        (self.source / "python.md").write_text(
            '---\npaths: ["**/*.py"]\n---\n# Python\n', encoding="utf-8"
        )
        result = A.sync(str(self.repo), str(self.source))
        self.assertEqual(
            result["removed"], [str(self.repo / ".agents/packs/python.md")]
        )
        self.assertTrue((self.repo / ".claude/rules/python.md").is_file())
        self.assertEqual(self.codes(source=str(self.source)), [])

    def test_a_file_already_gone_is_not_listed_as_removed(self):
        A.sync(str(self.repo), str(self.source), packs=["python"])
        (self.repo / ".agents/packs/python.md").unlink()
        (self.source / "python.md").write_text(
            '---\npaths: ["**/*.py"]\n---\n# Python\n', encoding="utf-8"
        )
        self.assertEqual(A.sync(str(self.repo), str(self.source))["removed"], [])

    def test_a_claude_md_that_is_not_utf8_is_refused_and_left_as_it_was(self):
        for encoding in ("cp1252", "utf-16"):
            raw = "# repo\n\n- Keep it tidy — always.\n".encode(encoding)
            (self.repo / "CLAUDE.md").write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "not UTF-8"):
                A.sync(str(self.repo))
            self.assertEqual((self.repo / "CLAUDE.md").read_bytes(), raw)
            self.assertFalse((self.repo / "AGENTS.md").exists())
            self.assertIn("encoding", self.codes())

    def test_a_stub_keeps_its_other_imports(self):
        A.sync(str(self.repo), str(self.source), packs=["principles"], mode="import")
        stub = "@AGENTS.md\n@.agents/packs/principles.md\n@notes.md\n"
        self.write("repo/CLAUDE.md", stub)
        self.assertEqual(self.codes(source=str(self.source), mode="import"), [])
        A.sync(str(self.repo), str(self.source), mode="import")
        self.assertEqual(self.read("repo/CLAUDE.md"), stub)
        A.sync(str(self.repo), str(self.source), mode="hook")  # a hook imports no packs
        self.assertEqual(self.read("repo/CLAUDE.md"), "@AGENTS.md\n@notes.md\n")

    def test_the_pointer_block_goes_in_and_leaves_the_rest_as_written(self):
        self.write("repo/CLAUDE.md", "# repo\n- rule\n```\na\n\n\nb\n```\n")
        A.sync(str(self.repo), str(self.source), packs=["principles"])
        agents = self.read("repo/AGENTS.md")
        self.assertTrue(agents.startswith("# repo\n\n" + A.POINTER_BEGIN))
        self.assertIn(A.POINTER_END + "\n\n- rule\n```\na\n\n\nb\n```\n", agents)

    def test_a_workspace_root_imports_its_packs_and_carries_no_hook(self):
        A.sync(str(self.repo), str(self.source), packs=["principles"], mode="import")
        self.assertEqual(
            self.read("repo/CLAUDE.md"), "@AGENTS.md\n@.agents/packs/principles.md\n"
        )
        self.assertFalse((self.repo / A.HOOK_FILE).exists())
        self.assertEqual(self.codes(source=str(self.source), mode="import"), [])

    def test_rules_in_both_files_are_merged_by_hand(self):
        self.write("repo/AGENTS.md", "# repo\n- other\n")
        with self.assertRaisesRegex(ValueError, "merge them by hand"):
            A.sync(str(self.repo))

    def test_a_folder_with_no_rules_and_no_packs_is_left_alone(self):
        empty = self.base / "empty"
        empty.mkdir()
        self.assertEqual(A.sync(str(empty))["written"], [])


class TestSyncLeavesWhatItDoesNotOwn(_Tree):
    """sync removes and rewrites only its own files, and refuses what it cannot read."""

    def test_a_pointer_link_outside_the_pack_folders_removes_nothing(self):
        A.sync(str(self.repo), str(self.source), packs=["principles"])
        mine = self.write("repo/docs/principles.md", "# my notes\n")
        outside = self.write("outside/principles.md", "# not the repo's\n")
        agents = self.read("repo/AGENTS.md")
        links = (
            "docs/principles.md",
            str(outside),
            outside.as_posix(),
            "../outside/principles.md",
        )
        for link in links:
            with self.subTest(link=link):
                self.write(
                    "repo/AGENTS.md",
                    agents.replace(".agents/packs/principles.md", link),
                )
                self.assertEqual(
                    A.sync(str(self.repo), str(self.source))["removed"], []
                )
                self.assertTrue(mine.is_file())
                self.assertTrue(outside.is_file())
                self.assertIn(
                    "[principles](.agents/packs/principles.md)",
                    self.read("repo/AGENTS.md"),
                )
        self.assertEqual(
            A.vendored(str(self.repo)), {"principles": ".agents/packs/principles.md"}
        )

    def test_a_pack_name_is_a_plain_name(self):
        for name in ("../principles", "a/b", "", "principles.md"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "is not a pack name"),
            ):
                A.sync(str(self.repo), str(self.source), packs=[name])
        self.assertFalse((self.repo / "AGENTS.md").exists())

    def test_settings_or_a_pack_that_is_not_utf8_is_refused_and_left_as_it_was(self):
        A.sync(str(self.repo), str(self.source), packs=["principles"])
        settings = self.repo / A.SETTINGS
        vendored = self.repo / ".agents/packs/principles.md"
        origin = self.source / "principles.md"
        for path, raw in (
            (
                settings,
                json.dumps({"permissions": {"allow": ["Read"]}}).encode("utf-16"),
            ),
            (vendored, "# Principles — mine\n".encode("utf-16")),
            (origin, "# Principles — changed\n".encode("cp1252")),
        ):
            with self.subTest(path=path.name):
                before = {p: p.read_bytes() for p in (settings, vendored, origin)}
                path.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, "not UTF-8"):
                    A.sync(str(self.repo), str(self.source))
                self.assertEqual(path.read_bytes(), raw)
                self.assertIn("encoding", self.codes(source=str(self.source)))
                path.write_bytes(before[path])

    def test_a_rewritten_file_keeps_its_newlines_and_its_bom(self):
        (self.repo / "AGENTS.md").write_bytes(
            "﻿# repo\r\n\r\n- a rule\r\n".encode("utf-8")
        )
        (self.repo / "CLAUDE.md").write_bytes(b"@AGENTS.md\r\n")
        A.sync(str(self.repo), str(self.source), packs=["principles"])
        raw = (self.repo / "AGENTS.md").read_bytes()
        self.assertTrue(raw.startswith(codecs.BOM_UTF8))
        self.assertNotIn(b"\n", raw.replace(b"\r\n", b""))  # every line ends CRLF
        self.assertIn(A.POINTER_END.encode("utf-8") + b"\r\n\r\n- a rule\r\n", raw)
        self.assertEqual((self.repo / "CLAUDE.md").read_bytes(), b"@AGENTS.md\r\n")
        self.assertEqual(self.codes(source=str(self.source)), [])
        self.assertEqual(A.sync(str(self.repo), str(self.source))["written"], [])

    def test_a_write_that_cannot_finish_leaves_the_file_as_it_was(self):
        A.sync(str(self.repo))
        before = self.read("repo/AGENTS.md")
        with (
            patch("os.replace", side_effect=PermissionError("held")),
            self.assertRaises(PermissionError),
        ):
            A.sync(str(self.repo), str(self.source), packs=["principles"])
        self.assertEqual(self.read("repo/AGENTS.md"), before)
        self.assertEqual([p.name for p in self.base.rglob("*.part")], [])

    def test_a_hook_sync_did_not_write_is_refused_and_an_earlier_template_replaced(
        self,
    ):
        hook = self.write("repo/.agents/session_hook.py", "print('my own hook')\n")
        with self.assertRaisesRegex(ValueError, "not the hook sync writes"):
            A.sync(str(self.repo), str(self.source), packs=["principles"])
        self.assertEqual(hook.read_text(encoding="utf-8"), "print('my own hook')\n")
        self.assertFalse((self.repo / "AGENTS.md").exists())  # refused before any write
        hook.write_text(EARLIER_HOOK, encoding="utf-8")
        A.sync(str(self.repo), str(self.source), packs=["principles"])
        self.assertEqual(self.read("repo/.agents/session_hook.py"), A.HOOK_SCRIPT)

    def test_the_hook_runs_without_a_shell_and_the_shell_form_is_rewritten(self):
        A.sync(str(self.repo), str(self.source), packs=["principles"])
        settings = self.repo / A.SETTINGS
        (entry,) = json.loads(settings.read_text(encoding="utf-8"))["hooks"][
            "SessionStart"
        ][0]["hooks"]
        self.assertEqual(
            (entry["command"], entry["args"]),
            ("python", ["${CLAUDE_PROJECT_DIR}/.agents/session_hook.py"]),
        )
        self.write(
            "repo/.claude/settings.json",
            json.dumps({"hooks": {"SessionStart": [{"hooks": [SHELL_ENTRY]}]}}),
        )
        self.assertEqual(self.codes(source=str(self.source)), ["hook"])
        A.sync(str(self.repo), str(self.source))
        groups = json.loads(settings.read_text(encoding="utf-8"))["hooks"][
            "SessionStart"
        ]
        self.assertEqual([g["hooks"] for g in groups], [[entry]])
        self.assertEqual(self.codes(source=str(self.source)), [])


class TestCheck(_Tree):
    def test_each_break_is_reported_with_sync_as_its_fix(self):
        self.write(
            "repo/sub/AGENTS.md", "# sub\n@../other.md\n- Since 2026-01-02 we do x.\n"
        )
        self.write("repo/sub/CLAUDE.md", "# leftover rules\n")
        ctx = OpContext()
        A.check(
            str(self.repo),
            standard=str(self.write("std.md", "## 3. Naming\n")),
            ctx=ctx,
        )
        codes = sorted({i.code for i in ctx.issues})
        self.assertEqual(
            codes,
            [
                "instructions.agents_import",
                "instructions.churn",
                "instructions.convert",
                "instructions.stub",
            ],
        )
        fixes = {i.code: i.fix for i in ctx.issues}
        self.assertEqual(fixes["instructions.convert"].op, "instructions.sync")
        self.assertIsNone(fixes["instructions.agents_import"])

    def test_an_unresolved_section_citation_is_reported(self):
        A.sync(str(self.repo))
        standard = self.write("std.md", "## 1. Style\n")
        problems = A.check(str(self.repo), standard=str(standard))["problems"]
        self.assertEqual([p["code"] for p in problems], ["anchor"])
        self.assertIn("§3", problems[0]["message"])

    def test_drift_a_stale_pointer_and_a_missing_hook_are_reported(self):
        A.sync(str(self.repo), str(self.source), packs=["principles"])
        (self.source / "principles.md").write_text(
            "# Principles\n- Changed.\n", encoding="utf-8"
        )
        (self.repo / A.HOOK_FILE).unlink()
        self.write("repo/.agents/packs/python.md", "# Python\n")
        self.assertEqual(
            self.codes(source=str(self.source)), ["hook", "pack_drift", "pointer"]
        )
        A.sync(str(self.repo), str(self.source))
        self.assertEqual(self.codes(source=str(self.source)), [])

    def test_a_decorator_in_a_code_block_is_no_import(self):
        A.sync(str(self.repo))
        self.write(
            "repo/sub/AGENTS.md", "# sub\n```python\n@staticmethod\ndef f(): ...\n```\n"
        )
        self.assertEqual(self.codes(), [])


class TestHook(_Tree):
    def run_hook(self, repo: Path) -> str:
        proc = subprocess.run(
            [sys.executable, str(repo / A.HOOK_FILE)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout

    def test_a_standalone_repo_gets_its_packs_and_one_under_a_provider_does_not(self):
        A.sync(str(self.repo), str(self.source), packs=["principles", "python"])
        out = json.loads(self.run_hook(self.repo))["hookSpecificOutput"]
        self.assertEqual(out["hookEventName"], "SessionStart")
        self.assertIn("Adopt before building", out["additionalContext"])
        self.assertIn("Lazy imports", out["additionalContext"])

        A.sync(str(self.base), str(self.source), packs=["principles"], mode="import")
        out = json.loads(self.run_hook(self.repo))["hookSpecificOutput"][
            "additionalContext"
        ]
        self.assertNotIn("Adopt before building", out)  # the workspace root provides it
        self.assertIn("Lazy imports", out)

        A.sync(str(self.base), str(self.source), packs=["python"], mode="import")
        self.assertEqual(self.run_hook(self.repo), "")

    def test_a_pack_is_skipped_only_when_a_folder_above_imports_it(self):
        # A worktree inside a hook-mode repo: the repo vendors the packs but its
        # stub imports none, so the worktree's own hook must add them.
        A.sync(str(self.repo), str(self.source), packs=["principles", "python"])
        tree = self.repo / ".claude" / "worktrees" / "x"
        self.write("repo/.claude/worktrees/x/CLAUDE.md", "# x\n- Worktree rules.\n")
        A.sync(str(tree), str(self.source), packs=["principles", "python"])
        out = json.loads(self.run_hook(tree))["hookSpecificOutput"]["additionalContext"]
        self.assertIn("Adopt before building", out)
        self.assertIn("Lazy imports", out)
        names = [Path(i["path"]).name for i in A.measure(str(tree))["launch"]]
        self.assertEqual(names.count("principles.md"), 1)

    def test_a_pack_it_cannot_read_is_skipped_and_the_others_are_added(self):
        A.sync(str(self.repo), str(self.source), packs=["principles", "python"])
        (self.repo / ".agents/packs/python.md").write_bytes(
            "# Python — saved as cp1252\n".encode("cp1252")
        )
        out = json.loads(self.run_hook(self.repo))["hookSpecificOutput"][
            "additionalContext"
        ]
        self.assertIn("Adopt before building", out)
        self.assertNotIn("cp1252", out)

    def test_a_claude_md_above_it_cannot_read_costs_nothing(self):
        A.sync(str(self.repo), str(self.source), packs=["principles"])
        self.write("CLAUDE.md", "@AGENTS.md\n")
        hook = self.repo / A.HOOK_FILE
        code = compile(hook.read_text(encoding="utf-8"), str(hook), "exec")
        real = Path.read_text

        def read_text(path, *args, **kwargs):
            if path.name == "CLAUDE.md":
                raise PermissionError(13, "locked by another process", str(path))
            return real(path, *args, **kwargs)

        out = io.StringIO()
        with (
            patch.object(Path, "read_text", read_text),
            contextlib.redirect_stdout(out),
        ):
            exec(code, {"__file__": str(hook), "__name__": "__main__"})
        context = json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"]
        self.assertIn("Adopt before building", context)


class TestMeasure(_Tree):
    def test_launch_and_touched_loads_follow_the_measured_rules(self):
        A.sync(str(self.repo))
        self.write(
            "repo/.claude/rules/ps.md",
            '---\npaths:\n  - "**/*.ps1"\n---\n- PowerShell rule.\n',
        )
        self.write("repo/.claude/rules/all.md", "- Always.\n")
        result = A.measure(str(self.repo), [str(self.base / "repo/sub/x.ps1")])
        launch = [
            Path(i["path"]).relative_to(self.base).as_posix() for i in result["launch"]
        ]
        touched = [
            Path(i["path"]).relative_to(self.base).as_posix() for i in result["touched"]
        ]
        self.assertEqual(
            launch, ["repo/CLAUDE.md", "repo/AGENTS.md", "repo/.claude/rules/all.md"]
        )
        self.assertEqual(
            touched,
            ["repo/.claude/rules/ps.md", "repo/sub/CLAUDE.md", "repo/sub/AGENTS.md"],
        )
        self.assertEqual(result["tokens"], result["chars"] // 4)

    def test_agents_md_alone_loads_only_while_no_claude_md_is_above(self):
        self.write("solo/AGENTS.md", "# solo\n- rule\n")
        launch = [
            Path(i["path"]).name for i in A.measure(str(self.base / "solo"))["launch"]
        ]
        self.assertEqual(launch, ["AGENTS.md"])
        self.write("CLAUDE.md", "# above\n")
        launch = [
            Path(i["path"]).name for i in A.measure(str(self.base / "solo"))["launch"]
        ]
        self.assertEqual(launch, ["CLAUDE.md"])  # the parent's silences AGENTS.md

    def test_a_repo_hook_counts_the_packs_no_folder_above_provides(self):
        A.sync(str(self.repo), str(self.source), packs=["principles", "python"])
        names = [Path(i["path"]).name for i in A.measure(str(self.repo))["launch"]]
        self.assertEqual(
            names, ["CLAUDE.md", "AGENTS.md", "principles.md", "python.md"]
        )


class TestOperations(unittest.TestCase):
    def test_the_committed_catalog_is_current(self):
        import pythontk.operations as declarations
        from pythontk import OperationProvider

        self.assertEqual(OperationProvider.problems(declarations.__name__), [])


class TestDuplicates(_Tree):
    def test_a_rule_written_in_two_files_is_one_group(self):
        self.write(
            "repo/sub/CLAUDE.md",
            "- Keep the whole repo tidy always (§3).\n- Unrelated words entirely here.\n",
        )
        groups = A.duplicates(str(self.repo))
        self.assertEqual(len(groups), 1)
        self.assertEqual(
            {Path(m["path"]).parent.name for m in groups[0]}, {"repo", "sub"}
        )

    def test_the_vendored_copies_of_one_pack_are_read_once(self):
        (self.source / "principles.md").write_text(
            "# Principles\n- Adopt what already exists before building anything new.\n",
            encoding="utf-8",
        )
        A.sync(str(self.repo), str(self.source), packs=["principles"])
        A.sync(str(self.base), str(self.source), packs=["principles"], mode="import")
        self.assertEqual(
            A.duplicates(str(self.base)), []
        )  # one pack, two sanctioned copies


if __name__ == "__main__":
    unittest.main()
