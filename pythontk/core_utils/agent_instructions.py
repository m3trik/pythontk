# !/usr/bin/python
# coding=utf-8
"""AgentInstructions -- a codebase's agent instructions as a checked tree.

The layout it keeps, measured against Claude Code and readable by any agent:

* ``AGENTS.md`` holds a folder's rules, at every level that has any: the open
  convention other agents read. A nested one scopes its rules to its folder.
* ``CLAUDE.md`` beside it is a stub: ``@AGENTS.md``. Claude Code reads
  ``AGENTS.md`` only while no ``CLAUDE.md`` exists at or above the session root,
  so a one-line stub at each level is what makes it load in every layout.
  ``AGENTS.md`` itself never ``@imports`` (other agents do not expand them).
* A **pack** is one Markdown file of rules several codebases share (principles,
  a language's spellings), written once in a source folder and vendored, under
  a drift check, into each codebase that takes it: ``.agents/packs/<name>.md``.
  A pointer block at the top of ``AGENTS.md`` names them for every agent.
* How Claude Code gets a codebase's packs depends on where it sits. A
  workspace root (``mode="import"``) imports them in its stub, so they load
  once however many of its repos a session touches. A repo (``mode="hook"``)
  carries a tiny stdlib SessionStart hook in its own ``.claude/settings.json``
  that adds them only to a session rooted there, and only those no folder
  above provides: hooks run only from the session root (measured), so the hook
  serves a standalone clone, a cloud session or CI, and never doubles a pack in
  the workspace. It runs as ``python`` with no shell (PowerShell 5.1 runs no
  ``||``), so a clone with no ``python`` gets no packs, and its ``AGENTS.md``
  still links them.
* A pack for one kind of file (a language's spellings) says so with ``paths:``
  frontmatter, and is vendored as a Claude Code rule, ``.claude/rules/<name>.md``,
  instead: it loads only when a matching file is touched, in any layout, with
  no hook. A codebase's own rules for one kind of file are written the same way.

:meth:`check` reports every way a tree breaks that layout, as operation issues
whose fix is :meth:`sync`; :meth:`sync` brings it in line (and is the dry run's
plan first); :meth:`measure` says what a session rooted somewhere loads, at
launch and on touching given paths; :meth:`duplicates` finds the rules written
more than once. They are pythontk operations (``instructions.*``), so the
command line and MCP reach them, and they sit over any codebase without its
code knowing.

:meth:`sync` changes only what it owns: it removes a pack only from the pack
folders, under the pack's own name; it refuses a file it cannot read as UTF-8
and a hook it did not write; and it writes each file whole (a ``.part`` beside
it, renamed onto it), in the newlines and byte-order mark the file has.

Adopted, not built: the ``AGENTS.md`` convention, and Claude Code's own
loading (``CLAUDE.md`` imports, path-scoped ``.claude/rules``, SessionStart
hooks). A Claude Code plugin carries agents, skills, hooks and MCP servers,
but no other agent reads one and a clone does not carry it, so a pack is a
vendored file; what is built is only the check that keeps the copies in step
and the layout whole. ``FileUtils.atomic_write`` is the house's whole-file
write, but it sits above this module in pythontk's layer order, so a write
here is the same rename, without its retry.
"""

from __future__ import annotations

import codecs
import fnmatch
import hashlib
import json
import os
import re
import secrets
import stat
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple


class _AgentInstructionsInternal:
    """Internal helpers for :class:`AgentInstructions`."""

    #: Folders never walked into.
    SKIP_DIRS = frozenset(
        {
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            "build",
            "dist",
            ".archive",
        }
    )
    _IMPORT = re.compile(r"^@(\S+)\s*$")
    _ANCHOR = re.compile(r"§\s?(\d+)")
    _DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b")
    _WORD = re.compile(r"[a-z0-9_]+")
    #: A pack's name: what its pointer link and its file are named by.
    _NAME = re.compile(r"[\w-]+")
    _LINK = re.compile(r"\[([\w-]+)\]\(([^)]+)\)")
    #: The sha1 of each earlier :attr:`AgentInstructions.HOOK_SCRIPT`, as read:
    #: a hook holding one is sync's own and is brought up to date; sync refuses
    #: to replace any other.
    _HOOK_EARLIER = frozenset({"53bc19a1d35458949412a9a14753aa08204479bb"})

    @classmethod
    def _walk(cls, root: Path) -> Iterator[Path]:
        """Every folder under *root* (itself first), skipping tooling folders
        (environments, builds, caches, archives, Claude Code's worktrees)."""
        for folder, dirs, _files in os.walk(root):
            dirs[:] = sorted(
                d
                for d in dirs
                if d not in cls.SKIP_DIRS
                and not d.endswith(".egg-info")
                and not (d == "worktrees" and Path(folder).name == ".claude")
            )
            yield Path(folder)

    @staticmethod
    def _read(path: Path) -> str:
        """*path*'s text; ``""`` when it is missing or not UTF-8 (:meth:`_decodes`
        tells which, before anything moves it)."""
        try:
            return path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            return ""

    @staticmethod
    def _decodes(path: Path) -> bool:
        """Whether *path* reads as UTF-8 (a missing one does: it holds nothing)."""
        try:
            path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            return False
        except OSError:
            pass
        return True

    @classmethod
    def _refuse_unreadable(cls, path: Path) -> None:
        """Raise when *path* exists and is not UTF-8: sync would read it as
        empty and write over what it holds.

        Raises:
            ValueError: *path* is not UTF-8 text.
        """
        if path.is_file() and not cls._decodes(path):
            raise ValueError(f"{path} is not UTF-8 text: save it as UTF-8, then sync")

    @staticmethod
    def _write(path: Path, text: str, like: Optional[Path] = None) -> None:
        """Write *text* to *path* whole or not at all: to a ``.part`` beside it,
        renamed onto it, the ``.part`` removed if anything fails. The file keeps
        its newlines (CRLF when most of its lines end so) and its UTF-8
        byte-order mark; a new file takes *like*'s, else LF and none. On POSIX
        it keeps its mode.

        Raises:
            OSError: The write or the rename failed (*path* unchanged).
        """
        raw = b""
        for source in (path, like):
            if source is not None and not raw:
                try:
                    raw = source.read_bytes()
                except OSError:
                    raw = b""
        if raw.count(b"\r\n") * 2 > raw.count(b"\n"):
            text = text.replace("\n", "\r\n")
        bom = codecs.BOM_UTF8 if raw.startswith(codecs.BOM_UTF8) else b""
        data = bom + text.encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f".{path.name}.{secrets.token_hex(4)}.part")
        try:
            with open(part, "wb") as fh:
                fh.write(data)
            if os.name == "posix":
                try:
                    os.chmod(part, stat.S_IMODE(os.stat(path).st_mode))
                except FileNotFoundError:
                    pass  # a new file keeps the mode it was created with
            os.replace(part, path)
        finally:
            try:
                part.unlink()
            except FileNotFoundError:
                pass  # renamed onto the file: nothing left

    @classmethod
    def _imports(cls, text: str) -> List[str]:
        """*text*'s ``@path`` import lines, outside code fences (Claude Code
        expands none inside one)."""
        refs, fenced = [], False
        for line in text.splitlines():
            if line.strip().startswith("```"):
                fenced = not fenced
            elif not fenced:
                match = cls._IMPORT.match(line)
                if match:
                    refs.append(match.group(1))
        return refs

    @classmethod
    def _provided(cls, top: Path) -> set:
        """The pack files a ``CLAUDE.md`` above *top* imports: Claude Code loads
        them in a session rooted at *top*, so its hook skips them (the
        :attr:`HOOK_SCRIPT` reads them alike)."""
        names = set()
        for parent in top.parents:
            for ref in cls._imports(cls._read(parent / cls.STUB)):
                if Path(ref).parent.as_posix().endswith(cls.PACKS_DIR):
                    names.add(Path(ref).name)
        return names

    @classmethod
    def _is_stub(cls, text: str) -> bool:
        """Only ``@`` import lines, the first ``@AGENTS.md``."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return (
            bool(lines)
            and lines[0] == "@AGENTS.md"
            and all(cls._IMPORT.match(line) for line in lines)
        )

    @classmethod
    def _stub_text(cls, packs: Mapping[str, str], mode: str, current: str = "") -> str:
        """A ``CLAUDE.md`` stub; a workspace root's also imports its packs
        (all but the path-scoped ones, which load as rules). The other imports
        the *current* stub holds (a private notes file) are kept after them."""
        lines = ["@AGENTS.md"]
        if mode == "import":
            lines += [
                f"@{where}"
                for where in packs.values()
                if where.startswith(cls.PACKS_DIR)
            ]
        for ref in cls._imports(current):
            if (
                ref != cls.SOURCE
                and not ref.startswith(cls.PACKS_DIR)
                and f"@{ref}" not in lines
            ):
                lines.append(f"@{ref}")
        return "\n".join(lines) + "\n"

    @classmethod
    def _where(cls, name: str, text: str) -> str:
        """Where pack *name* is vendored: a pack with ``paths:`` frontmatter is
        a path-scoped rule, loaded only on touching a matching file."""
        if cls._rule_globs(text) is not None:
            return f"{cls.RULES_DIR}/{name}.md"
        return f"{cls.PACKS_DIR}/{name}.md"

    @staticmethod
    def _inside(where: str) -> bool:
        """Whether a pointer link names a path inside the codebase: relative,
        with no drive, root, scheme or ``..``. Another is no record of a pack."""
        return (
            bool(where)
            and not where.startswith(("/", "\\"))
            and ":" not in where
            and ".." not in re.split(r"[\\/]", where)
        )

    @classmethod
    def _owned(cls, top: Path, name: str, where: str) -> bool:
        """Whether *where* (under *top*) is a file sync vendors pack *name* as:
        its own file in :attr:`PACKS_DIR` or :attr:`RULES_DIR`, resolved. Sync
        removes nothing else."""
        path = (top / where).resolve()
        folders = {
            os.path.normcase(str((top / folder).resolve()))
            for folder in (cls.PACKS_DIR, cls.RULES_DIR)
        }
        own_name = os.path.normcase(path.name) == os.path.normcase(f"{name}.md")
        return own_name and os.path.normcase(str(path.parent)) in folders

    @classmethod
    def _vendored(cls, root: Path) -> Dict[str, str]:
        """``{pack: where}`` this codebase takes: the pointer block's links
        inside the codebase (the record), plus any file in :attr:`PACKS_DIR`
        it lacks."""
        found = {}
        block = re.search(
            re.escape(cls.POINTER_BEGIN) + r"(.*?)" + re.escape(cls.POINTER_END),
            cls._read(root / cls.SOURCE),
            re.S,
        )
        for name, where in cls._LINK.findall(block.group(1) if block else ""):
            if cls._inside(where):
                found[name] = where
        folder = root / cls.PACKS_DIR
        for path in sorted(folder.glob("*.md")) if folder.is_dir() else ():
            found.setdefault(path.stem, f"{cls.PACKS_DIR}/{path.name}")
        return dict(sorted(found.items()))

    @classmethod
    def _pointer(cls, packs: Mapping[str, str]) -> str:
        links = " · ".join(f"[{name}]({where})" for name, where in packs.items())
        return (
            f"{cls.POINTER_BEGIN}\n"
            f"Shared rules (packs; Claude Code loads them, other agents read them first): {links}\n"
            f"{cls.POINTER_END}"
        )

    @classmethod
    def _with_pointer(cls, text: str, packs: Mapping[str, str]) -> str:
        """*text* (an ``AGENTS.md``) with its pointer block written for *packs*:
        replaced where it is, else after the title line; dropped when no packs."""
        block = cls._pointer(packs) if packs else ""
        pattern = re.compile(
            re.escape(cls.POINTER_BEGIN) + r".*?" + re.escape(cls.POINTER_END) + r"\n?",
            re.S,
        )
        if pattern.search(text):
            return pattern.sub(lambda _m: block + "\n" if block else "", text, count=1)
        if not block:
            return text
        lines = text.splitlines(keepends=True)
        head = (
            [lines.pop(0).rstrip("\n") + "\n", "\n"]
            if lines and lines[0].startswith("#")
            else []
        )
        while lines and not lines[0].strip():
            lines.pop(0)  # one blank line after the block, whatever was there
        return "".join(head + [block, "\n\n" if lines else "\n"] + lines)

    @classmethod
    def _session_start(cls, text: str) -> Tuple[Dict[str, Any], List[Any]]:
        """``(settings, its SessionStart groups)`` read from ``settings.json``
        text, the groups list made where it is missing.

        Raises:
            ValueError: The text is not JSON, or holds no object whose
                ``hooks.SessionStart`` is a list.
        """
        try:
            settings = json.loads(text) if text.strip() else {}
        except ValueError as exc:
            raise ValueError(f"{cls.SETTINGS} is not JSON ({exc})") from None
        hooks = settings.setdefault("hooks", {}) if isinstance(settings, dict) else None
        groups = (
            hooks.setdefault("SessionStart", []) if isinstance(hooks, dict) else None
        )
        if not isinstance(groups, list):
            raise ValueError(f"{cls.SETTINGS} holds no hooks.SessionStart list")
        return settings, groups

    @classmethod
    def _pack_hooks(cls, groups: List[Any]) -> List[Dict[str, Any]]:
        """Each hook in *groups* that runs :attr:`HOOK_FILE`, in any form sync
        has registered it (the shell string before the exec form)."""
        found = []
        for group in groups:
            for hook in group.get("hooks") or () if isinstance(group, dict) else ():
                if not isinstance(hook, dict):
                    continue
                args = hook.get("args") if isinstance(hook.get("args"), list) else []
                words = [str(hook.get("command", ""))] + [str(a) for a in args]
                if any(cls.HOOK_FILE in word for word in words):
                    found.append(hook)
        return found

    @classmethod
    def _settings_with_hook(cls, text: str) -> str:
        """``.claude/settings.json`` text with the pack hook registered once,
        in its current form (an earlier one is rewritten in place).

        Raises:
            ValueError: As :meth:`_session_start`.
        """
        settings, groups = cls._session_start(text)
        registered = cls._pack_hooks(groups)
        if registered:
            registered[0].update(cls._hook_entry())
        else:
            groups.append({"hooks": [cls._hook_entry()]})
        return json.dumps(settings, indent=2) + "\n"

    @classmethod
    def _hook_entry(cls) -> Dict[str, Any]:
        """The hook's registration: ``python`` run with no shell (exec form),
        which Claude Code runs alike under bash and PowerShell, and whose
        ``${CLAUDE_PROJECT_DIR}`` it expands (measured)."""
        return {
            "type": "command",
            "command": "python",
            "args": [f"${{CLAUDE_PROJECT_DIR}}/{cls.HOOK_FILE}"],
            "timeout": 10,
        }

    @classmethod
    def _own_hook(cls, text: str) -> bool:
        """Whether a hook file holding *text* is sync's: the template, or an
        earlier one (:attr:`_HOOK_EARLIER`)."""
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return text == cls.HOOK_SCRIPT or digest in cls._HOOK_EARLIER

    @classmethod
    def _rules(cls, text: str) -> List[str]:
        """The rule lines of an instruction file: its bullets, outside code."""
        rules, fenced = [], False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                fenced = not fenced
            elif not fenced and re.match(r"^([-*]|\d+\.)\s+\S", stripped):
                rules.append(re.sub(r"^([-*]|\d+\.)\s+", "", stripped))
        return rules

    @classmethod
    def _words(cls, line: str) -> frozenset:
        return frozenset(w for w in cls._WORD.findall(line.lower()) if len(w) > 2)

    @staticmethod
    def _rule_globs(text: str) -> Optional[List[str]]:
        """A rules file's ``paths:`` globs (``None``: it always loads)."""
        if not text.startswith("---"):
            return None
        head = text.split("---", 2)[1] if text.count("---") >= 2 else ""
        globs, inside = [], False
        for line in head.splitlines():
            if line.strip().startswith("paths:"):
                inside = True
                rest = line.split(":", 1)[1].strip()
                if rest.startswith("["):
                    return [
                        g.strip().strip("\"'")
                        for g in rest.strip("[]").split(",")
                        if g.strip()
                    ]
            elif inside and line.strip().startswith("-"):
                globs.append(line.strip()[1:].strip().strip("\"'"))
            elif inside and line.strip():
                break
        return globs or None


class AgentInstructions(_AgentInstructionsInternal):
    """A codebase's agent instructions, kept as a checked tree (see the module docstring)."""

    #: A folder's rules.
    SOURCE = "AGENTS.md"
    #: Claude Code's file, a stub importing :attr:`SOURCE`.
    STUB = "CLAUDE.md"
    #: Where a codebase's vendored packs live.
    PACKS_DIR = ".agents/packs"
    #: Where a path-scoped pack (``paths:`` frontmatter) is vendored instead.
    RULES_DIR = ".claude/rules"
    #: The SessionStart hook a repo carries (``mode="hook"``).
    HOOK_FILE = ".agents/session_hook.py"
    #: Where Claude Code reads a project's hooks.
    SETTINGS = ".claude/settings.json"
    #: The most a SessionStart hook may add, so a codebase's packs together.
    PACK_LIMIT = 10000
    MODES = ("hook", "import")
    POINTER_BEGIN = "<!-- packs (generated: edit the packs, then sync) -->"
    POINTER_END = "<!-- /packs -->"
    #: The hook, stdlib only, written verbatim into each repo that takes packs.
    HOOK_SCRIPT = '''"""Claude Code SessionStart hook, written by pythontk's AgentInstructions.sync
(edit its template, not this copy): adds this codebase's agent packs to a session
rooted here, but none a CLAUDE.md above already imports (a workspace root's stub
imports its own, and Claude Code loads it in every session below it).

Claude Code runs it as `python`, with no shell: a virtual environment provides
one, as does python-is-python3 on Debian and Ubuntu. Without one the packs are
not added, and AGENTS.md still links them. A file it cannot read is skipped, and
it never fails the session."""
import json
from pathlib import Path

LIMIT = 10000


def read(path, errors="strict"):
    """*path*'s text, or None when it cannot be read (locked, gone, not UTF-8)."""
    try:
        return path.read_text(encoding="utf-8-sig", errors=errors)
    except (OSError, UnicodeError):
        return None


try:
    here = Path(__file__).resolve().parent
    provided = set()
    for parent in here.parent.parents:
        for line in (read(parent / "CLAUDE.md", "replace") or "").splitlines():
            ref = Path(line.strip()[1:])
            if line.startswith("@") and ref.parent.as_posix().endswith(".agents/packs"):
                provided.add(ref.name)
    packs = sorted((here / "packs").glob("*.md"))
    texts = [read(p) for p in packs if p.name not in provided]
    text = "\\n\\n".join(t for t in texts if t and t.strip())
    if text:
        out = {"hookEventName": "SessionStart", "additionalContext": text[:LIMIT]}
        print(json.dumps({"hookSpecificOutput": out}))
except Exception:  # a hook that fails costs the session its packs, nothing more
    pass
'''

    @classmethod
    def vendored(cls, root: str) -> Dict[str, str]:
        """The packs the codebase at *root* takes: ``{name: where}``, each
        path relative to *root* (its pointer block is the record)."""
        return cls._vendored(Path(root).resolve())

    @classmethod
    def check(
        cls,
        root: str,
        source: Optional[str] = None,
        standard: Optional[str] = None,
        mode: str = "hook",
        ctx=None,
    ) -> Dict[str, Any]:
        """Report every way the instruction tree under *root* breaks the layout.

        Each problem is an operation issue (``ctx.issue``) whose fix is
        ``instructions.sync`` where sync can make it.

        Parameters:
            root: The codebase's root folder.
            source: The packs' source folder, to check the vendored copies against.
            standard: A standards document whose ``## N.`` sections the
                instructions cite as ``§N``; each citation must resolve.
            mode: How the root's packs reach Claude Code: ``"hook"`` (a repo)
                or ``"import"`` (a workspace root).
            ctx: The operation context (issues are also returned).

        Returns:
            dict: ``files`` (instruction files read) and ``problems``
            (``{code, location, message}`` each).
        """
        top = Path(root).resolve()
        problems: List[Dict[str, str]] = []
        fix = {"op": "instructions.sync", "args": {"root": str(top), "mode": mode}}
        if source:
            fix["args"]["source"] = str(Path(source).resolve())

        def report(
            code: str, location: Path, message: str, fixable: bool = True
        ) -> None:
            problems.append(
                {"code": code, "location": str(location), "message": message}
            )
            if ctx is not None:
                ctx.issue(
                    f"instructions.{code}",
                    message,
                    "warning",
                    str(location),
                    fix if fixable else None,
                )

        def unreadable(path: Path) -> bool:
            if path.is_file() and not cls._decodes(path):
                report(
                    "encoding",
                    path,
                    f"{path.name} is not UTF-8 text; save it as UTF-8 (sync leaves it alone)",
                    False,
                )
                return True
            return False

        sections = None
        if standard:
            sections = {
                int(m)
                for m in re.findall(r"^##\s+(\d+)\.", cls._read(Path(standard)), re.M)
            }
        files = 0
        for folder in cls._walk(top):
            agents, stub = folder / cls.SOURCE, folder / cls.STUB
            if [p for p in (agents, stub) if unreadable(p)]:
                continue
            stub_text = cls._read(stub) if stub.is_file() else None
            if agents.is_file():
                files += 1
                text = cls._read(agents)
                if cls._imports(text):
                    report(
                        "agents_import",
                        agents,
                        f"{agents.name} @imports (other agents do not expand them); inline the rules",
                        False,
                    )
                if stub_text is None:
                    report(
                        "stub",
                        stub,
                        f"{stub.name} is missing beside {agents.name}: Claude Code would not read it",
                    )
                elif not cls._is_stub(stub_text):
                    report(
                        "stub",
                        stub,
                        f"{stub.name} holds rules of its own; they belong in {agents.name}",
                    )
                for line in text.splitlines():
                    if cls._DATE.search(line):
                        report(
                            "churn",
                            agents,
                            f"a dated line churns the prompt cache: {line.strip()[:80]}",
                            False,
                        )
                        break
                if sections is not None:
                    for n in sorted(
                        {int(m) for m in cls._ANCHOR.findall(text)} - sections
                    ):
                        report(
                            "anchor",
                            agents,
                            f"§{n} is not a section of {Path(standard).name}",
                            False,
                        )
            elif stub_text is not None:
                files += 1
                if not cls._is_stub(stub_text):
                    report(
                        "convert",
                        stub,
                        f"{stub.name} holds rules; move them to {cls.SOURCE} and leave the stub",
                    )
        vendored = cls._vendored(top)
        if vendored:
            agents_text = cls._read(top / cls.SOURCE)
            if cls._pointer(vendored) not in agents_text:
                report(
                    "pointer",
                    top / cls.SOURCE,
                    "the packs pointer block is missing or stale",
                )
            hooked = [w for w in vendored.values() if w.startswith(cls.PACKS_DIR)]
            size = sum(len(cls._read(top / w)) for w in hooked)
            if size > cls.PACK_LIMIT:
                report(
                    "pack_size",
                    top / cls.PACKS_DIR,
                    f"the packs total {size} characters; a hook adds at most {cls.PACK_LIMIT}",
                    False,
                )
            for name, where in vendored.items():
                if not (top / where).is_file():
                    report(
                        "pack_missing",
                        top / where,
                        f"pack {name!r} is not vendored where the pointer says",
                    )
                elif unreadable(top / where):
                    continue
                elif source:
                    origin = Path(source) / f"{name}.md"
                    if not origin.is_file():
                        report(
                            "pack_unknown",
                            top / where,
                            f"no pack {name!r} in {source}",
                            False,
                        )
                    elif not unreadable(origin) and cls._read(origin) != cls._read(
                        top / where
                    ):
                        report(
                            "pack_drift",
                            top / where,
                            f"pack {name!r} differs from its source",
                        )
            if mode == "hook" and hooked:
                hook, settings = top / cls.HOOK_FILE, top / cls.SETTINGS
                if hook.is_file() and not cls._own_hook(cls._read(hook)):
                    report(
                        "hook",
                        hook,
                        f"{hook.name} is not the hook sync writes; move it aside, then sync",
                        False,
                    )
                elif cls._read(hook) != cls.HOOK_SCRIPT:
                    report(
                        "hook", hook, "the pack hook is missing or not the template's"
                    )
                if not unreadable(settings):
                    try:
                        registered = cls._pack_hooks(
                            cls._session_start(cls._read(settings))[1]
                        )
                    except ValueError as exc:
                        report("settings", settings, f"{exc}; fix it, then sync", False)
                    else:
                        entry = cls._hook_entry()
                        if not registered:
                            report(
                                "hook",
                                settings,
                                "the pack hook is not registered for SessionStart",
                            )
                        elif any(registered[0].get(k) != v for k, v in entry.items()):
                            report(
                                "hook",
                                settings,
                                "the pack hook is registered in an earlier form (a shell string "
                                "PowerShell cannot run)",
                            )
            stub_now = cls._read(top / cls.STUB)
            if mode == "import" and stub_now != cls._stub_text(
                vendored, mode, stub_now
            ):
                report(
                    "stub", top / cls.STUB, "the root stub does not import its packs"
                )
            elif (
                mode == "hook"
                and cls._is_stub(stub_now)
                and stub_now != cls._stub_text(vendored, mode, stub_now)
            ):
                report(
                    "stub",
                    top / cls.STUB,
                    "the root stub imports packs its hook adds too",
                )
        return {"files": files, "problems": problems}

    @classmethod
    def sync(
        cls,
        root: str,
        source: Optional[str] = None,
        packs: Optional[List[str]] = None,
        mode: str = "hook",
        ctx=None,
    ) -> Dict[str, Any]:
        """Bring the instruction tree under *root* in line with the layout.

        Moves each ``CLAUDE.md`` that holds rules into ``AGENTS.md`` and leaves
        the stub; writes a missing stub; vendors the root's packs from *source*
        (those it already has, plus *packs*); writes the pointer block; and
        installs the pack hook (``mode="hook"``) or the root stub's imports
        (``mode="import"``). A dry run lists every write first.

        Every file is written whole, in the newlines and byte-order mark it
        has. A pack that moves is removed only from the pack folders, under
        its own name. Everything refused is refused before anything is written.

        Parameters:
            root: The codebase's root folder.
            source: The packs' source folder (needed to add or refresh packs).
            packs: Pack names to add to the ones already vendored.
            mode: ``"hook"`` for a repo, ``"import"`` for a workspace root.
            ctx: The operation context: each file is written only when it
                answers ``should_write``.

        Returns:
            dict: ``written`` and ``removed``, the files written and removed
            (or, in a dry run, to be).

        Raises:
            ValueError: An unknown *mode*, a pack name that is not a plain
                name or has no source, a CLAUDE.md with rules beside an
                AGENTS.md (merge them by hand), an instruction file,
                ``settings.json`` or pack that is not UTF-8, a
                ``settings.json`` that is not a JSON object, or a hook file
                sync did not write (nothing is written).
            OSError: A file could not be written (it is left as it was).
        """
        if mode not in cls.MODES:
            raise ValueError(f"mode {mode!r} is not one of {list(cls.MODES)}")
        for name in packs or ():
            if not isinstance(name, str) or not cls._NAME.fullmatch(name):
                raise ValueError(
                    f"{name!r} is not a pack name (letters, digits, _ and -)"
                )
        top = Path(root).resolve()
        writes: List[Tuple[Path, str, str]] = []
        likes: Dict[Path, Path] = {}
        removes: List[Tuple[Path, str]] = []

        def plan(
            path: Path, text: str, action: str, like: Optional[Path] = None
        ) -> None:
            if cls._read(path) != text or not path.is_file():
                writes.append((path, text, action))
                if like is not None:
                    likes.setdefault(path, like)

        for folder in cls._walk(top):
            agents, stub = folder / cls.SOURCE, folder / cls.STUB
            for path in (agents, stub):
                cls._refuse_unreadable(path)
            if not stub.is_file() or cls._is_stub(cls._read(stub)):
                if agents.is_file() and not stub.is_file():
                    plan(
                        stub,
                        cls._stub_text((), "hook"),
                        "the stub that makes Claude Code read AGENTS.md",
                    )
                continue
            if agents.is_file():
                raise ValueError(
                    f"{stub} holds rules beside {agents}: merge them by hand"
                )
            plan(agents, cls._read(stub), "the rules CLAUDE.md held", like=stub)
            plan(stub, cls._stub_text({}, "hook"), "a stub importing AGENTS.md")

        names = cls._vendored(top)
        added = [n for n in packs or () if n not in names]
        if added and not source:
            raise ValueError("adding a pack needs its source folder")
        for name in sorted(set(names) | set(added)):
            if source:
                origin = Path(source) / f"{name}.md"
                if not origin.is_file():
                    raise ValueError(f"no pack {name!r} in {source}")
                cls._refuse_unreadable(origin)
                text = cls._read(origin)
                where = cls._where(name, text)
                recorded = names.get(name, where)
                if recorded != where and cls._owned(top, name, recorded):
                    removes.append((top / recorded, f"pack {name!r} moved to {where}"))
                names[name] = where
                cls._refuse_unreadable(top / where)
                plan(top / where, text, f"vendor pack {name!r}")
        names = dict(sorted(names.items()))
        hooked = {n: w for n, w in names.items() if w.startswith(cls.PACKS_DIR)}
        agents = top / cls.SOURCE
        pending = {path: text for path, text, _ in writes}
        if names or agents.is_file() or agents in pending:
            current = pending.get(agents, cls._read(agents)) or f"# {top.name}\n"
            plan(agents, cls._with_pointer(current, names), "the packs pointer block")
            stub_now = pending.get(top / cls.STUB, cls._read(top / cls.STUB))
            plan(
                top / cls.STUB, cls._stub_text(hooked, mode, stub_now), "the root stub"
            )
        if hooked and mode == "hook":
            hook, settings = top / cls.HOOK_FILE, top / cls.SETTINGS
            if hook.is_file() and not cls._own_hook(cls._read(hook)):
                raise ValueError(
                    f"{hook} is not the hook sync writes: move it aside, then sync"
                )
            plan(hook, cls.HOOK_SCRIPT, "the SessionStart hook that adds the packs")
            cls._refuse_unreadable(settings)
            plan(
                settings,
                cls._settings_with_hook(cls._read(settings)),
                "register the pack hook",
            )

        written = []
        merged: Dict[Path, Tuple[str, str]] = {}
        for path, text, action in writes:
            merged[path] = (text, merged.get(path, ("", action))[1])
        for path, (text, action) in merged.items():
            if ctx is None or ctx.should_write(path, action):
                cls._write(path, text, likes.get(path))
            written.append(str(path))
        removed = []
        for path, action in removes:
            if path.is_file():
                if ctx is None or ctx.should_delete(path, action):
                    path.unlink()
                removed.append(str(path))
        return {"written": written, "removed": removed}

    @classmethod
    def measure(cls, root: str, touched: Optional[List[str]] = None) -> Dict[str, Any]:
        """What a Claude Code session rooted at *root* loads from instruction files.

        The model of Claude Code's loading, as measured: at launch, the
        ``CLAUDE.md`` (with its ``@imports``) and the unconditional
        ``.claude/rules`` of the root and every folder above it, ``AGENTS.md``
        in their place only when no ``CLAUDE.md`` exists at or above the root,
        and the root's pack hook; then, for each touched path, those of every
        folder between the root and it, plus the ``paths:`` rules it matches.

        Parameters:
            root: The session's root folder.
            touched: Paths the session reads or edits, in order.

        Returns:
            dict: ``launch`` and ``touched`` (each a list of ``{path, chars}``),
            ``chars`` and ``tokens`` (an estimate, a quarter of the characters).
        """
        top = Path(root).resolve()
        seen: set = set()
        claude_above = any((d / cls.STUB).is_file() for d in [top, *top.parents])
        name = cls.STUB if claude_above else cls.SOURCE

        def load(path: Path, into: List[Dict[str, Any]], depth: int = 0) -> None:
            if path in seen or not path.is_file() or depth > 5:
                return
            seen.add(path)
            text = cls._read(path)
            into.append({"path": str(path), "chars": len(text)})
            for ref in cls._imports(text):
                load((path.parent / ref).resolve(), into, depth + 1)

        def folder(
            d: Path, into: List[Dict[str, Any]], file: Optional[Path] = None
        ) -> None:
            load(d / name, into)
            rules = d / ".claude" / "rules"
            for rule in sorted(rules.glob("*.md")) if rules.is_dir() else ():
                globs = cls._rule_globs(cls._read(rule))
                if globs is None or (
                    file is not None
                    and any(
                        fnmatch.fnmatch(file.relative_to(d).as_posix(), g)
                        for g in globs
                    )
                ):
                    load(rule, into)

        launch: List[Dict[str, Any]] = []
        for d in reversed([top, *top.parents]):
            folder(d, launch)
        if cls.HOOK_FILE in cls._read(top / cls.SETTINGS):
            provided = cls._provided(top)
            for pack in sorted((top / cls.PACKS_DIR).glob("*.md")):
                if pack.name not in provided:
                    load(pack, launch)
        later: List[Dict[str, Any]] = []
        for path in touched or ():
            file = Path(path).resolve()
            chain = [p for p in reversed(file.parents) if top in p.parents]
            for d in [*reversed(top.parents), top, *chain]:
                folder(d, later, file)
        chars = sum(item["chars"] for item in launch + later)
        return {
            "launch": launch,
            "touched": later,
            "chars": chars,
            "tokens": chars // 4,
        }

    @classmethod
    def duplicates(
        cls, root: str, threshold: float = 0.6
    ) -> List[List[Dict[str, str]]]:
        """Rules written more than once across the instruction files under *root*.

        Parameters:
            root: The folder to read (its ``AGENTS.md``, ``CLAUDE.md`` with
                rules, vendored packs and ``.claude/rules``).
            threshold: The share of words two rules must have in common
                (Jaccard, words of three letters or more).

        Returns:
            list: Groups of two or more ``{path, rule}``, from more than one file.
        """
        top = Path(root).resolve()
        rules: List[Tuple[str, str, frozenset]] = []
        copies: set = set()
        for folder in cls._walk(top):
            candidates = [folder / cls.SOURCE, folder / cls.STUB]
            for sub in (cls.PACKS_DIR, cls.RULES_DIR):
                for path in (
                    sorted((folder / sub).glob("*.md"))
                    if (folder / sub).is_dir()
                    else ()
                ):
                    text = cls._read(path)
                    if (
                        text not in copies
                    ):  # a pack's other vendored copies are sanctioned
                        copies.add(text)
                        candidates.append(path)
            for path in candidates:
                if path.is_file():
                    for rule in cls._rules(cls._read(path)):
                        words = cls._words(rule)
                        if len(words) >= 4:
                            rules.append((str(path), rule, words))
        parent = list(range(len(rules)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(rules)):
            for j in range(i + 1, len(rules)):
                a, b = rules[i][2], rules[j][2]
                if rules[i][0] != rules[j][0] and len(a & b) / len(a | b) >= threshold:
                    parent[find(i)] = find(j)
        groups: Dict[int, List[int]] = {}
        for i in range(len(rules)):
            groups.setdefault(find(i), []).append(i)
        return [
            [{"path": rules[i][0], "rule": rules[i][1]} for i in members]
            for members in groups.values()
            if len({rules[i][0] for i in members}) > 1
        ]
