# !/usr/bin/python
# coding=utf-8
"""Process-level test isolation -- keep a test run off the developer's machine.

Two things a test process does to the machine it runs on that no assertion
ever catches:

* **Launch a real browser.** ``webbrowser.open`` is how :class:`PreviewServer`
  shows a preview, and a bridge test that reaches it pops a localhost tab over
  whatever the developer is doing -- one per push, for the life of the run,
  with every test green.
* **Litter the system temp dir.** Every :class:`TempArtifacts` store, hand-off
  payload and scratch dir defaults to ``tempfile.gettempdir()``. A suite that
  is killed (routine under a DCC host) or that leans on the ``detached`` policy
  leaves thousands of ``<prefix>_<tag>`` entries behind, reclaimed only by the
  seven-day sweep -- and only if that prefix ever allocates again. Measured on
  one workstation: 24,500 entries, 1,371 of them from one preview suite.

This lives in the shipped package, at the bottom of the stack, because both
effects are process-wide and every downstream suite (uitk, mayatk, blendertk,
tentacle, extapps) needs the identical guard -- a second copy is a copy that
drifts. ``uitk.testing.TestSandbox`` extends it with the Qt-side stores.
Downstream use is one line, at import time of a conftest or a test runner --
before the first temp allocation, and in EVERY entry point (a ``unittest``
runner never loads a conftest)::

    import pythontk as ptk
    ptk.TestSandbox.activate()

A test that legitimately exercises a browser launch patches ``webbrowser.open``
itself: ``unittest.mock.patch`` layers over the guard and restores it after.
"""

import os
import tempfile
from typing import Callable, Dict, List


class _TestSandboxInternal:
    """Guard mechanics behind :class:`TestSandbox`.

    State lives in ONE shared dict rather than in per-class attributes: a
    downstream subclass (``uitk.testing.TestSandbox``) activating the guards
    must leave the base reading as active too, or a later ``activate()`` on
    the base would redirect the temp dir a second time.
    """

    #: The ``webbrowser`` entry points a launch can go through.
    _LAUNCHERS = ("open", "open_new", "open_new_tab")
    #: What a child process reads for its temp dir (``tempfile`` checks these first).
    _TEMP_ENV = ("TMPDIR", "TEMP", "TMP")

    _state: Dict[str, object] = {
        "guard": None,  # the one function standing in for every launcher
        "temp_dir": None,
        "temp_store": None,  # the TempArtifacts owning temp_dir; held so its exit cleanup fires
    }

    @classmethod
    def _make_guard(cls) -> Callable[..., bool]:
        """The launcher stand-in: record the URL, then refuse loudly.

        One function object for all three entry points, created once, so
        :meth:`TestSandbox.is_active` can test identity against it.
        """

        def blocked(url, *args, **kwargs):
            cls.launches.append(str(url))
            raise RuntimeError(
                f"TestSandbox blocked a real browser launch for {url!r}. A test "
                "that means to open a page patches `webbrowser.open`; one that "
                "reached this by accident has a deliverer or bridge left on its "
                "opening default (build it with open_browser=False)."
            )

        return blocked


class TestSandbox(_TestSandboxInternal):
    """Keep this process's side effects off the developer's machine. Idempotent."""

    #: URLs the browser guard refused, in order -- for a runner's end-of-run
    #: report, which is what surfaces a refusal that a broad ``except`` between
    #: the launch and the test swallowed.
    launches: List[str] = []

    @classmethod
    def browser(cls) -> None:
        """Refuse every ``webbrowser`` launch for the rest of the process.

        Raises rather than returning ``False``: a quiet refusal is logged as
        "no browser could be launched" and the test goes green, which is how a
        tab-per-push leak survives a suite in the first place. The URL is also
        recorded on :attr:`launches`.
        """
        state = cls._state
        if state["guard"] is not None:
            return
        import webbrowser

        guard = cls._make_guard()
        for name in cls._LAUNCHERS:
            setattr(webbrowser, name, guard)
        state["guard"] = guard

    @classmethod
    def temp(cls) -> str:
        """Route the process's temp dir into one throwaway root; returns it.

        ``tempfile.gettempdir()`` -- and so every :class:`TempArtifacts` store,
        hand-off payload and scratch dir that defaults to it -- resolves inside
        the root for the rest of the process, and ``TMPDIR``/``TEMP``/``TMP``
        point there so a child process (mayapy, Blender, FBX2glTF) inherits the
        same. The root goes at interpreter exit; a process that is killed
        instead leaves it for the age-gated sweep the next activation runs, so
        the worst case is one stale directory rather than thousands of loose
        files. Prefix sweeps inside a run still work: they scan whatever
        ``gettempdir()`` answers.
        """
        state = cls._state
        if state["temp_dir"] is not None:
            return state["temp_dir"]
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        # Allocated in the REAL temp dir, before the redirect, so the sweep of
        # prior killed runs looks where they landed.
        store = TempArtifacts("ptk_test_sandbox", policy="session")
        root = store.dir_path()
        tempfile.tempdir = root
        for name in cls._TEMP_ENV:
            os.environ[name] = root
        state["temp_store"] = store
        state["temp_dir"] = root
        return root

    @classmethod
    def activate(cls) -> str:
        """Both guards; returns the temp root. Safe to call more than once."""
        cls.browser()
        return cls.temp()

    @classmethod
    def is_active(cls) -> bool:
        """True while both guards are in place.

        For a suite that wants to *assert* its isolation rather than assume
        it: a guard that was activated and later undone reads False here.
        """
        state = cls._state
        if state["guard"] is None or state["temp_dir"] is None:
            return False
        import webbrowser

        return tempfile.gettempdir() == state["temp_dir"] and all(
            getattr(webbrowser, name) is state["guard"] for name in cls._LAUNCHERS
        )
