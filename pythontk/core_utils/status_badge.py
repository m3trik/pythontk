# !/usr/bin/python
# coding=utf-8
"""Shields.io status badges embedded in a markdown file.

Single source of truth for the test-status badge every package in the ecosystem
stamps into its README, so a badge always means the same thing everywhere:

    the number of *individual test cases* that passed.

Suites, modules and script categories are never the unit -- a "5 passed" badge
on a repo whose five test scripts assert two hundred things reads as far weaker
coverage than it is, and is not comparable with a sibling package's number.
Skipped tests are excluded from the count (they did not pass) but do not change
the colour, so an all-green run with environment-gated skips still reads green.

The generic half (:meth:`StatusBadge.render` / :meth:`StatusBadge.update`) draws
any label/message/colour badge; :meth:`StatusBadge.test_status` and
:meth:`StatusBadge.update_test_badge` layer the test-count convention on top.
"""
import os
import re
from pathlib import Path
from typing import Optional, Tuple, Union

PathLike = Union[str, os.PathLike]


class _StatusBadgeInternal:
    """Markdown/URL plumbing behind :class:`StatusBadge`."""

    SHIELDS_HOST = "https://img.shields.io/badge"

    # Any shields.io badge carrying the given label, linked ``[![Alt](url)](href)``
    # or bare ``![Alt](url)``. The alt text is deliberately unconstrained so a
    # legacy badge written under a different alt (or label casing) is migrated
    # in place rather than duplicated.
    @classmethod
    def _patterns(cls, label: str):
        # Match the ENCODED label -- that is what lands in the URL. Matching the
        # raw one silently fails for any label needing escaping (e.g. "Unit
        # Tests" -> "Unit%20Tests"), and a matcher that never matches appends a
        # second badge on every run instead of replacing the first.
        url = rf"https://img\.shields\.io/badge/{re.escape(cls._encode(label))}-[^)]*"
        return (
            re.compile(rf"\[!\[[^\]]*\]\({url}\)\]\([^)]*\)", re.IGNORECASE),
            re.compile(rf"!\[[^\]]*\]\({url}\)", re.IGNORECASE),
        )

    # A badge line already at the top of the file -- used to place a first-time
    # badge at the end of the existing badge block instead of above the title.
    _BADGE_LINE = re.compile(r"^\s*\[?!\[[^\]]*\]\(https://img\.shields\.io/")

    @staticmethod
    def _encode(text: str) -> str:
        """Escape a shields.io path segment (``-`` separates the segments)."""
        return (
            str(text)
            .replace("_", "__")
            .replace("-", "--")
            .replace(",", "%2C")
            .replace(" ", "%20")
        )

    @classmethod
    def _insert(cls, content: str, badge: str) -> str:
        """Place a first-time badge: after the leading badge block, else on top."""
        lines = content.splitlines()
        last = -1
        for i, line in enumerate(lines):
            if cls._BADGE_LINE.match(line):
                last = i
            elif line.strip() and last >= 0:
                break  # badge block ended
        if last >= 0:
            lines.insert(last + 1, badge)
            return "\n".join(lines) + ("\n" if content.endswith("\n") else "")
        return badge + "\n\n" + content


    # ---- Run-completeness gate -------------------------------------------
    # Shared by every package runner: a badge must never be stamped by a run
    # the environment scoped down, or the number published is a smaller green
    # than the suite really is. Lives here rather than in each runner because
    # six of them stamp badges and this file is already the documented single
    # writer (m3trik/docs/TEST_BADGE_STANDARD.md).

    @staticmethod
    def _unwrap(test):
        """Unwrap a ``unittest._SubTest`` to the case that owns it."""
        return getattr(test, "test_case", test)

    @classmethod
    def _is_import_standin(cls, test) -> bool:
        """True when *test* stands in for a module that never imported.

        ``TestLoader`` substitutes a case defined in ``unittest.loader``
        (``ModuleImportFailure`` / ``ModuleSkipped``) when a module raised on
        import or raised ``SkipTest`` at import time -- the module genuinely
        did not run, whatever its stand-in reports.

        Deliberately narrower than "defined somewhere in unittest": a
        ``setUpClass``/``setUpModule`` skip is reported through
        ``unittest.suite._ErrorHolder``, and that module DID import and run.
        Treating the two alike blocked the badge forever for any module whose
        cases are all setUpClass-gated, which contradicts this module's own
        docstring ("an all-green run with environment-gated skips still reads
        green") and made greenness depend on which skip idiom a test author
        happened to use.
        """
        test = cls._unwrap(test)
        return (getattr(type(test), "__module__", "") or "").startswith(
            "unittest.loader"
        )


class StatusBadge(_StatusBadgeInternal):
    """Render a shields.io badge and keep it up to date in a markdown file.

    Test-run usage (what every package's test runner calls)::

        ptk.StatusBadge.update_test_badge(
            readme_path, passed=1234, failed=0, test_dir=Path(__file__).parent
        )
    """

    # ---- Run-completeness gate (see _StatusBadgeInternal) -----------------

    @staticmethod
    def discover_module_names(test_dir: PathLike) -> set:
        """Return the module names ``TestLoader.discover`` will import.

        Mirrors discovery's top-level ``test_*.py`` match; nested directories
        are only recursed into when they are packages (``__init__.py``), which
        no test subdirectory in this ecosystem is.
        """
        return {p.stem for p in Path(test_dir).glob("test_*.py")}

    @classmethod
    def module_of(cls, test) -> str:
        """Return the ``test_*`` module *test* came from ('' if unknown).

        Handles the three shapes a runner sees: a real case (its class's
        ``__module__``), a loader stand-in for a module that never imported
        (its ``_testMethodName`` carries the dotted module name), and an
        ``_ErrorHolder`` from ``setUpClass``/``setUpModule`` (its description
        reads ``"setUpClass (test_mod.TestCase)"``).
        """
        test = cls._unwrap(test)
        name = getattr(type(test), "__module__", "") or ""

        if cls._is_import_standin(test):
            name = getattr(test, "_testMethodName", "") or ""
        elif name.startswith("unittest"):
            # _ErrorHolder: no __module__ of its own. Its description reads
            # "setUpClass (pkg.test_mod.TestCase)" or "setUpModule (pkg.test_mod)",
            # so the trailing segment is a CLASS in the first form and the module
            # itself in the second -- strip it only for setUpClass.
            description = getattr(test, "description", "") or getattr(
                test, "id", lambda: ""
            )()
            inner = description[description.find("(") + 1 : description.rfind(")")]
            if inner and description.startswith("setUpClass"):
                inner = inner.rsplit(".", 1)[0]
            name = inner or ""

        return name.rsplit(".", 1)[-1]

    @classmethod
    def is_import_standin(cls, test) -> bool:
        """True when *test* stands in for a module that never imported."""
        return cls._is_import_standin(test)

    @classmethod
    def gate(cls, expected, ran, passed: int, failed: int) -> Tuple[bool, str]:
        """Return ``(allowed, reason)`` for stamping the test badge.

        Parameters:
            expected: Module names discovered on disk.
            ran: Module names that actually executed.
            passed: Passing test cases (skips excluded).
            failed: Failures + errors.
        """
        missing = sorted(set(expected) - set(ran))
        if missing:
            shown = ", ".join(missing[:6])
            if len(missing) > 6:
                shown += f", +{len(missing) - 6} more"
            return False, (
                f"{len(missing)} of {len(expected)} module(s) did not run: {shown}"
            )
        if not passed and not failed:
            return False, "no test cases ran"
        return True, ""


    LABEL = "Tests"
    COLOR_PASSED = "brightgreen"
    COLOR_MIXED = "orange"
    COLOR_FAILED = "red"
    COLOR_UNKNOWN = "lightgrey"

    # ---------------------------------------------------------------- render

    @classmethod
    def url(
        cls,
        message: str,
        color: str,
        label: str = LABEL,
        style: Optional[str] = None,
    ) -> str:
        """Build the shields.io image URL.

        Parameters:
            message: Right-hand text, e.g. ``"1234 passed"``.
            color: Shields colour name.
            label: Left-hand text.
            style: Optional shields style (e.g. ``"flat-square"``). Omitted ->
                a plain ``.svg`` URL. Purely cosmetic; pass whatever the rest of
                the README's badge row uses.

        Returns:
            The badge image URL.
        """
        path = f"{cls.SHIELDS_HOST}/{cls._encode(label)}-{cls._encode(message)}-{color}"
        return f"{path}?style={style}" if style else f"{path}.svg"

    @classmethod
    def render(
        cls,
        message: str,
        color: str,
        label: str = LABEL,
        link: str = "",
        style: Optional[str] = None,
    ) -> str:
        """Return the badge as a markdown image, linked when ``link`` is given."""
        image = f"![{label}]({cls.url(message, color, label, style)})"
        return f"[{image}]({link})" if link else image

    # ------------------------------------------------------- test semantics

    @classmethod
    def test_status(cls, passed: int, failed: int) -> Tuple[str, str]:
        """Map a test run to ``(message, color)``.

        Parameters:
            passed: Individual test cases that passed. Skipped tests are NOT
                passes -- exclude them from this number.
            failed: Failures + errors.

        Returns:
            Tuple of the badge message and colour.
        """
        if passed == 0 and failed == 0:
            # Nothing ran at all -- discovery found no tests, or the runner
            # never reached them. That is UNKNOWN, not green: a green badge for
            # a run that produced no results is the worst reading of all.
            return "0 passed", cls.COLOR_UNKNOWN
        if failed == 0:
            return f"{passed} passed", cls.COLOR_PASSED
        if passed == 0:
            return f"{failed} failed", cls.COLOR_FAILED
        return f"{passed} passed, {failed} failed", cls.COLOR_MIXED

    # ---------------------------------------------------------------- write

    @classmethod
    def update(
        cls,
        readme_path: PathLike,
        message: str,
        color: str,
        label: str = LABEL,
        link: str = "",
        style: Optional[str] = None,
    ) -> bool:
        """Replace (or insert) the badge with this label in a markdown file.

        Parameters:
            readme_path: Markdown file to stamp.
            message: Right-hand badge text.
            color: Shields colour name.
            label: Left-hand badge text; also identifies the badge to replace.
            link: Optional href the badge points at.
            style: Optional shields style.

        Returns:
            True if the badge is in place; False if the file is missing or the
            read/write failed. Stamping a badge is a cosmetic side effect of a
            test run -- it must never turn a green run red, so I/O errors (a
            read-only or cloud-sync-locked README) are reported, not raised.
            Programming errors still propagate.
        """
        readme = Path(readme_path)
        if not readme.exists():
            return False

        badge = cls.render(message, color, label, link, style)
        try:
            content = readme.read_text(encoding="utf-8")
        except OSError:
            return False

        for pattern in cls._patterns(label):
            if pattern.search(content):
                # Escape the replacement -- badge URLs contain no backrefs, but
                # ``\`` handling in re.sub templates would still bite.
                updated = pattern.sub(lambda _m: badge, content, count=1)
                break
        else:
            updated = cls._insert(content, badge)

        if updated != content:
            try:
                readme.write_text(updated, encoding="utf-8")
            except OSError:
                return False
        return True

    @classmethod
    def update_test_badge(
        cls,
        readme_path: PathLike,
        passed: int,
        failed: int,
        test_dir: Optional[PathLike] = None,
        link: Optional[str] = None,
        style: Optional[str] = None,
    ) -> bool:
        """Stamp a test-status badge -- the ecosystem-standard entry point.

        Parameters:
            readme_path: Markdown file to stamp.
            passed: Individual test cases passed (skips excluded).
            failed: Failures + errors.
            test_dir: Test directory the badge links to. Resolved relative to
                the README's own location, so moving the README can't break it.
            link: Explicit href, overriding ``test_dir``.
            style: Optional shields style.

        Returns:
            True if the badge is in place; False if the README is missing or
            could not be read/written -- see :meth:`update`.
        """
        readme = Path(readme_path)
        if link is None:
            link = ""
            if test_dir is not None:
                rel = os.path.relpath(Path(test_dir).resolve(), readme.resolve().parent)
                link = Path(rel).as_posix() + "/"

        message, color = cls.test_status(passed, failed)
        return cls.update(readme, message, color, cls.LABEL, link, style)


# --------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------
# Repos whose badge is written by CI shell rather than Python (server, comfyui)
# reimplement `test_status`'s wording in bash; keep the two in step -- the
# standard they both follow is m3trik/docs/TEST_BADGE_STANDARD.md.
