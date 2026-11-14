#!/usr/bin/python
# coding=utf-8
"""
Main test runner for pythontk package.

This script discovers and runs the test modules, collecting results
and outputting them to both console and a log file. A full run also updates
the README.md badge with test results.

Run with:
    python run_tests.py                        # the full suite (review / release)
    python run_tests.py test_str core_utils    # named modules only
    python run_tests.py test_str -k split      # ...and only tests matching "split"
    python run_tests.py -v                     # Verbose output
    python run_tests.py --log                  # Enable log file output
    python run_tests.py --no-badge             # Skip README badge update

A scoped run (modules or ``-k``) never stamps the badge. For the modules a
change reaches, ``python m3trik/scripts/affected_tests.py <files> --run``.
"""

import argparse
import datetime
import io
import os
import sys
import unittest
from pathlib import Path

from pythontk import TeeStream
from pythontk.core_utils.status_badge import StatusBadge

# cp1252 consoles can't encode characters test docstrings legitimately use
# ("→"); unittest's printErrors then dies MID-REPORT, eating the failure list
# and the summary (bitten in uitk's runner). Degrade gracefully instead.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except (ValueError, OSError):
            pass

# Ensure Qt binding selection is explicit for qtpy-based modules
if "QT_API" not in os.environ:
    os.environ["QT_API"] = "pyside6"


class _TestRunnerInternal:
    """Internal helpers for :class:`TestRunner` -- module coverage + badge gating.

    A partial run must not stamp the README badge, whether it was scoped by
    argument or by environment (numpy/Pillow/Qt missing, so those modules never
    import). See m3trik/docs/TEST_BADGE_STANDARD.md.

    The expected module set is *derived* from the ``test_*.py`` files on disk
    rather than recorded anywhere, so it can never go stale, and a refused stamp
    always prints its reason -- mirroring mayatk's runner, which prints
    ``[INFO] Badge not updated (some modules did not run).``
    """

    # discover_module_names / module_of / is_import_standin / gate live on
    # ptk.StatusBadge -- the documented single writer for this badge
    # (m3trik/docs/TEST_BADGE_STANDARD.md). Six runners stamp badges; the
    # completeness rule has to be one implementation, not one per runner.

    @classmethod
    def stamp_badge(cls, test_dir, readme_path, result) -> bool:
        """Update the README badge for *result*, unless the run fell short.

        Returns True when the badge was written; otherwise prints the reason it
        was not (mirroring mayatk's runner) and returns False.
        """
        allowed, reason = StatusBadge.gate(
            StatusBadge.discover_module_names(test_dir),
            result.modules_ran,
            result.passed,
            result.failures + result.errors,
        )
        if not allowed:
            print(f"[INFO] Badge not updated ({reason}).")
            return False
        return update_readme_badge(
            result.passed, result.failures + result.errors, readme_path
        )


class TestResult:
    """Container for test result statistics."""

    def __init__(self, result: unittest.TestResult, duration: float):
        self.tests_run = result.testsRun
        self.failures = len(result.failures)
        self.errors = len(result.errors)
        self.skipped = len(result.skipped)
        self.passed = self.tests_run - self.failures - self.errors - self.skipped
        self.duration = duration
        # Module coverage, for the badge guard (see _TestRunnerInternal).
        self.modules_ran = set(getattr(result, "modules_ran", ()))
        self.modules_executed = set(getattr(result, "modules_executed", ()))
        self.failure_details = result.failures
        self.error_details = result.errors
        self.success = self.failures == 0 and self.errors == 0

    @property
    def summary(self) -> str:
        """Return a one-line summary of results."""
        status = "PASSED" if self.success else "FAILED"
        return (
            f"{status}: {self.tests_run} tests, "
            f"{self.passed} passed, "
            f"{self.failures} failed, "
            f"{self.errors} errors, "
            f"{self.skipped} skipped "
            f"({self.duration:.2f}s)"
        )


class TestRunner(_TestRunnerInternal):
    """Discovers and runs the test modules -- all of them, or a named subset."""

    def __init__(
        self,
        test_dir: Path,
        verbosity: int = 1,
        modules=None,
        patterns=None,
    ):
        """
        Parameters:
            test_dir: The directory holding ``test_*.py``.
            verbosity: unittest verbosity (0 quiet, 1 dots, 2 names).
            modules: Module names to run (bare or ``test_``-prefixed); None runs
                every discovered module.
            patterns: ``-k`` filters on test ids; a pattern without ``*`` matches
                as a substring (unittest's own ``-k`` semantics).
        """
        self.test_dir = test_dir
        self.verbosity = verbosity
        self.modules = list(modules or [])
        self.patterns = list(patterns or [])
        self.log_buffer = io.StringIO()

    @staticmethod
    def resolve_modules(test_dir: Path, names):
        """``(stems, unknown)``: each name as its ``test_*`` stem, and the names
        no module on disk answers to."""
        available = StatusBadge.discover_module_names(test_dir)
        stems, unknown = [], []
        for name in names:
            stem = name[:-3] if name.endswith(".py") else name
            stem = stem if stem.startswith("test_") else f"test_{stem}"
            if stem in available:
                stems.append(stem)
            else:
                unknown.append(name)
        return stems, unknown

    def discover_tests(self) -> unittest.TestSuite:
        """The suite to run: every module, or :attr:`modules`; :attr:`patterns`
        narrows either to the matching test ids."""
        loader = unittest.TestLoader()
        if self.patterns:
            loader.testNamePatterns = [
                p if "*" in p else f"*{p}*" for p in self.patterns
            ]
        if not self.modules:
            return loader.discover(
                start_dir=str(self.test_dir),
                pattern="test_*.py",
                top_level_dir=str(self.test_dir),
            )
        if str(self.test_dir) not in sys.path:
            sys.path.insert(0, str(self.test_dir))
        stems, _unknown = self.resolve_modules(self.test_dir, self.modules)
        # loadTestsFromName turns an import failure into a failing stand-in
        # case, exactly as discover() does -- a broken module stays visible.
        return unittest.TestSuite(loader.loadTestsFromName(stem) for stem in stems)

    def run(self, log_to_file: bool = False) -> TestResult:
        """Run all discovered tests and collect results.

        Parameters:
            log_to_file: If True, write results to a log file.

        Returns:
            TestResult object with statistics.
        """
        suite = self.discover_tests()

        # Create stream that writes to both console and buffer
        stream = TeeStream(sys.stdout, self.log_buffer)

        # Print header
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""
{"=" * 70}
pythontk Test Suite
{"=" * 70}
Started: {timestamp}
Test Directory: {self.test_dir}
{"=" * 70}
"""
        stream.write(header)

        # Run tests
        import time

        start_time = time.perf_counter()

        runner = unittest.TextTestRunner(
            stream=stream, verbosity=self.verbosity, resultclass=DetailedTestResult
        )
        # unittest writes through `stream`, but product code does not: a bare
        # print(), or the traceback.print_exc() of a swallowed exception, goes
        # to the REAL stdout/stderr and never reached the log. A failure whose
        # only explanation was printed therefore came out of the log as a bare
        # assertion diff -- which is how six instances of the full-suite file-IO
        # flake were logged as unexplained (2026-08-20). Tee both for the
        # duration of the run; `stream` holds the ORIGINAL stdout object, so
        # unittest's own output is not doubled by the redirect.
        real_out, real_err = sys.stdout, sys.stderr
        sys.stdout = TeeStream(real_out, self.log_buffer)
        sys.stderr = TeeStream(real_err, self.log_buffer)
        try:
            result = runner.run(suite)
        finally:
            # In a finally so a runner crash cannot leave the process writing
            # into a buffer nobody reads.
            sys.stdout, sys.stderr = real_out, real_err

        duration = time.perf_counter() - start_time

        # Create result object
        test_result = TestResult(result, duration)

        # Print footer
        footer = f"""
{"=" * 70}
{test_result.summary}
{"=" * 70}
"""
        stream.write(footer)

        # The tracebacks are above (unittest's own report); end on the ids, so
        # the last lines of a run say exactly what failed.
        if test_result.failure_details or test_result.error_details:
            stream.write("\nFAILED TESTS:\n")
            for test, _traceback in test_result.failure_details:
                stream.write(f"  FAIL:  {test.id()}\n")
            for test, _traceback in test_result.error_details:
                stream.write(f"  ERROR: {test.id()}\n")

        # Save log file if requested
        if log_to_file:
            self._save_log(timestamp)

        return test_result

    def _save_log(self, timestamp: str):
        """Save test results to a log file."""
        log_dir = self.test_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        # Create filename from timestamp
        safe_timestamp = timestamp.replace(":", "-").replace(" ", "_")
        log_file = log_dir / f"test_results_{safe_timestamp}.log"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(self.log_buffer.getvalue())

        print(f"\nLog saved to: {log_file}")


class DetailedTestResult(unittest.TextTestResult):
    """unittest's text result, recording which test modules actually ran, so a
    run the environment scoped down can be refused the README badge (see
    :class:`_TestRunnerInternal`).

    It adds no status marks of its own: ``TextTestResult`` already writes the
    dot / ``ok`` for every outcome, and writing them again here printed each one
    twice (``....ss`` for two passes and a skip).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modules_ran = set()  # imported, and its cases were run
        self.modules_executed = set()  # produced at least one non-skipped case

    def _note_module(self, test, executed: bool):
        """Credit *test*'s module; a loader stand-in means it never ran."""
        if StatusBadge.is_import_standin(test):
            return
        name = StatusBadge.module_of(test)
        if not name:
            return
        self.modules_ran.add(name)
        if executed:
            self.modules_executed.add(name)

    def addSuccess(self, test):
        super().addSuccess(test)
        self._note_module(test, True)

    def addError(self, test, err):
        super().addError(test, err)
        self._note_module(test, True)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._note_module(test, True)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._note_module(test, True)

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._note_module(test, True)

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._note_module(test, False)


def update_readme_badge(passed: int, failed: int, readme_path: Path) -> bool:
    """Update the README with a test status badge.

    Thin wrapper over the ecosystem-wide SSoT (``ptk.StatusBadge``) so every
    package's badge counts the same unit -- individual test cases, skips
    excluded. See m3trik/docs/TEST_BADGE_STANDARD.md.

    Parameters:
        passed: Number of passed tests (skips excluded).
        failed: Number of failed tests (failures + errors).
        readme_path: Path to the README.md file.

    Returns:
        True if README was updated successfully.
    """
    from pythontk.core_utils.status_badge import StatusBadge

    ok = StatusBadge.update_test_badge(
        readme_path, passed, failed, test_dir=Path(__file__).resolve().parent
    )
    if not ok:
        print(f"README badge not updated (missing or unwritable): {readme_path}")
        return False

    print(f"\nREADME badge updated: {StatusBadge.test_status(passed, failed)[0]}")
    return True


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Run pythontk test suite")
    parser.add_argument(
        "modules",
        nargs="*",
        help="Test modules to run (bare or test_-prefixed); none = the full suite",
    )
    parser.add_argument(
        "-k",
        dest="patterns",
        action="append",
        default=[],
        help="Only tests whose id matches (substring, or a * pattern); repeatable",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (can be used multiple times)",
    )
    parser.add_argument("--log", action="store_true", help="Save results to a log file")
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimal output")
    parser.add_argument(
        "--no-badge",
        action="store_true",
        help="Skip updating README badge",
    )

    args = parser.parse_args()

    # Determine verbosity
    verbosity = 0 if args.quiet else args.verbose

    # Get test directory (where this script lives)
    test_dir = Path(__file__).parent
    root_dir = test_dir.parent

    # Ensure root directory (package root) is in path for imports
    # This ensures 'import pythontk' resolves to the package inside root_dir
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    # Ensure test directory is in path for imports
    if str(test_dir) not in sys.path:
        sys.path.insert(0, str(test_dir))

    # Process-level isolation before discovery imports a single module: no
    # real browser launch, one throwaway temp root. Here as well as in
    # conftest.py because this runner discovers with ``unittest``, which never
    # loads a conftest -- so the canonical run would have been the unguarded one.
    from pythontk.core_utils.test_sandbox import TestSandbox

    TestSandbox.activate()

    # A name nothing answers to means the run would not cover what was asked
    # for; running the rest and exiting 0 would read as that coverage.
    _stems, unknown = TestRunner.resolve_modules(test_dir, args.modules)
    if unknown:
        print(f"[ERROR] No such test module(s): {', '.join(unknown)}")
        sys.exit(2)

    # Run tests
    runner = TestRunner(
        test_dir, verbosity=verbosity, modules=args.modules, patterns=args.patterns
    )
    result = runner.run(log_to_file=args.log)

    # A refused launch that a broad ``except`` swallowed leaves its test green;
    # this is what makes it fail anyway -- in production that call opened a
    # tab. Before the badge, which must not stamp such a run as passing.
    leaked = list(TestSandbox.launches)
    if leaked:
        print(
            f"[ERROR] {len(leaked)} browser launch(es) blocked by the sandbox and "
            "swallowed by the code under test: " + ", ".join(leaked)
        )

    # A module that imported but produced only skips still counts as run (the
    # standard keeps environment-gated skips green), but say so out loud.
    skipped_only = sorted(result.modules_ran - result.modules_executed)
    if skipped_only:
        print(
            f"[WARNING] {len(skipped_only)} module(s) contributed only skips: "
            + ", ".join(skipped_only)
        )

    # Update README badge unless --no-badge is specified -- and never on a run
    # scoped by argument (modules / -k) or by the environment (StatusBadge.gate).
    # The gate alone would pass a -k that happens to match a test in every module.
    if args.modules or args.patterns:
        if not args.no_badge:
            print("[INFO] Badge not updated (scoped run: modules or -k).")
    elif not args.no_badge and not leaked:
        TestRunner.stamp_badge(test_dir, root_dir / "docs" / "README.md", result)

    # Exit with appropriate code
    sys.exit(0 if result.success and not leaked else 1)


if __name__ == "__main__":
    main()
