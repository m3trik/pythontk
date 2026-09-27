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
* **Fill the developer's Recycle Bin.** A lightmap re-bake sets the maps it
  superseded aside through :meth:`FileUtils.move_to_trash`, and a suite that
  re-bakes -- the bake tests, a bridge's wire-back -- sent its scratch maps
  into the machine's own bin, one per test, every run (measured 2026-09-27:
  11 in one mayatk run). Under the sandbox the call answers "no trash here",
  so the caller sets the file aside in its own folder, inside the temp root.

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
itself: ``unittest.mock.patch`` layers over the guard and restores it after. A
test of the trash itself runs inside :meth:`TestSandbox.real_trash`.

A third effect is opt-in per test, :meth:`TestSandbox.user_config`: reading
(or writing) the developer's own config docs -- the naming convention, preset
stores -- through the shared user-config root.
"""

import os
import tempfile
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List


class _TestSandboxInternal:
    """Guard mechanics behind :class:`TestSandbox`.

    State lives in ONE shared dict rather than in per-class attributes: a
    downstream subclass (``uitk.testing.TestSandbox``) activating the guards
    must leave the base reading as active too, or a later ``activate()`` on
    the base would redirect the temp dir a second time.
    """

    #: The ``webbrowser`` module functions a launch can go through.
    _LAUNCHERS = ("open", "open_new", "open_new_tab")
    #: Browser CLASSES whose ``open`` is a second, unguarded route to the same
    #: launch: a caller that needs a SPECIFIC browser cannot use the module
    #: functions, which only ever open the system default, so it builds one of
    #: these directly. Patching only the module functions left that route open.
    _LAUNCHER_TYPES = ("BackgroundBrowser", "GenericBrowser")
    #: What a child process reads for its temp dir (``tempfile`` checks these first).
    _TEMP_ENV = ("TMPDIR", "TEMP", "TMP")
    #: Names the root for a child, which nests its own inside it rather than
    #: re-deriving it from :attr:`_TEMP_ENV` (see :meth:`TestSandbox.temp`).
    _ROOT_ENV = "PYTHONTK_TEST_TEMP_ROOT"

    _state: Dict[str, object] = {
        "guard": None,  # the one function standing in for every launcher
        "temp_dir": None,
        "temp_store": None,  # the TempArtifacts owning temp_dir; held so its exit cleanup fires
        "trash": None,  # the real FileUtils.move_to_trash, while the guard stands in
        "can_trash": None,  # ...and the real FileUtils.can_trash
    }

    @classmethod
    def _make_guard(cls) -> Callable[..., bool]:
        """The launcher stand-in: record the URL, then refuse loudly.

        One function object for all three entry points, created once, so
        :meth:`TestSandbox.is_active` can test identity against it.
        """

        def blocked(url, *args, **kwargs):
            # Also serves as a method on a browser class, where the first
            # argument is the instance and the URL is the next one.
            if hasattr(url, "open") and args:
                url = args[0]
            cls.launches.append(str(url))
            raise RuntimeError(
                f"TestSandbox blocked a real browser launch for {url!r}. A test "
                "that means to open a page patches `webbrowser.open`; one that "
                "reached this by accident has a deliverer or bridge left on its "
                "opening default (build it with open_browser=False)."
            )

        return blocked

    @classmethod
    def _make_trash_guard(cls) -> Callable[[str], None]:
        """The ``move_to_trash`` stand-in: the real one's refusals (a missing
        path, a folder), else record the file and answer ``None`` -- a volume
        with no trash, a branch every caller already handles."""

        def no_trash(path):
            path = os.path.abspath(os.fspath(path))
            if not os.path.lexists(path):
                raise FileNotFoundError(2, "No such file", path)
            if os.path.isdir(path) and not os.path.islink(path):
                raise IsADirectoryError(21, "A folder, not a file", path)
            cls.trashed.append(path)
            return None

        return no_trash


class TestSandbox(_TestSandboxInternal):
    """Keep this process's side effects off the developer's machine. Idempotent."""

    #: URLs the browser guard refused, in order -- for a runner's end-of-run
    #: report, which is what surfaces a refusal that a broad ``except`` between
    #: the launch and the test swallowed.
    launches: List[str] = []
    #: Files a caller asked to move to the trash while the trash guard stood in.
    trashed: List[str] = []

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
        for name in cls._LAUNCHER_TYPES:
            browser_type = getattr(webbrowser, name, None)
            if browser_type is not None:
                browser_type.open = guard
        state["guard"] = guard

    @classmethod
    def temp(cls) -> str:
        """Route the process's temp dir into one throwaway root; returns it.

        ``tempfile.gettempdir()`` -- and so every :class:`TempArtifacts` store,
        hand-off payload and scratch dir that defaults to it -- resolves inside
        the root for the rest of the process, and ``TMPDIR``/``TEMP``/``TMP``
        point there so a child process (mayapy, Blender, FBX2glTF) inherits the
        same. The root goes at interpreter exit, or at
        :meth:`ProcessExit.hard_exit`, which skips it; a process that is killed
        instead leaves it for the age-gated sweep the next activation runs, so
        the worst case is one stale directory rather than thousands of loose
        files. Prefix sweeps inside a run still work: they scan whatever
        ``gettempdir()`` answers.

        Those variables are not enough for a Python child: a fresh
        interpreter derives its temp dir from them by probing each with a
        throwaway write and, on any failure but ``FileExistsError``, falls past
        all three -- they name the same root -- to the user's real temp dir,
        silently (measured: a root that had gone, an ``OSError`` on three
        probes in a row, or a refused create). So the root is also named in
        ``PYTHONTK_TEST_TEMP_ROOT``, and a child activating the sandbox nests
        its own root inside that one without probing. A child that must share
        the parent's root instead is handed it outright by whatever launches
        it (``tempfile.tempdir = root`` before its first line -- blendertk's
        runner does this).
        """
        state = cls._state
        if state["temp_dir"] is not None:
            return state["temp_dir"]
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        # Top level: allocated in the REAL temp dir, before the redirect, so
        # the sweep of prior killed runs looks where they landed. In a child:
        # inside the parent's root, by name.
        parent = os.environ.get(cls._ROOT_ENV) or None
        store = TempArtifacts("ptk_test_sandbox", policy="session", dir=parent)
        root = store.dir_path()
        tempfile.tempdir = root
        for name in cls._TEMP_ENV + (cls._ROOT_ENV,):
            os.environ[name] = root
        state["temp_store"] = store
        state["temp_dir"] = root
        return root

    @classmethod
    def trash(cls) -> None:
        """Keep :meth:`FileUtils.move_to_trash` off the machine's trash for the
        rest of the process: it answers ``None``, as on a volume with no trash,
        so a caller sets the file aside its own way
        (``FileDependencies.set_aside``: a ``_superseded`` folder beside it,
        inside the temp root), and :meth:`FileUtils.can_trash` answers False,
        so a prompt says what will really happen. Each file is recorded on
        :attr:`trashed`. A test patching either call layers over the guard,
        as for the browser."""
        state = cls._state
        if state["trash"] is not None:
            return
        from pythontk.file_utils._file_utils import FileUtils

        state["trash"] = FileUtils.__dict__["move_to_trash"]
        state["can_trash"] = FileUtils.__dict__["can_trash"]
        FileUtils.move_to_trash = staticmethod(cls._make_trash_guard())
        # A prompt asks first: told "no trash" too, it says what will happen.
        FileUtils.can_trash = staticmethod(lambda path: False)

    @classmethod
    @contextmanager
    def real_trash(cls) -> Iterator[None]:
        """The machine's own trash for the block -- for a test of
        :meth:`FileUtils.move_to_trash` itself, which cleans up what it
        moved. A no-op outside the guard."""
        from pythontk.file_utils._file_utils import FileUtils

        real = cls._state["trash"]
        if real is None:
            yield
            return
        guards = (FileUtils.__dict__["move_to_trash"], FileUtils.__dict__["can_trash"])
        FileUtils.move_to_trash = real
        FileUtils.can_trash = cls._state["can_trash"]
        try:
            yield
        finally:
            FileUtils.move_to_trash, FileUtils.can_trash = guards

    @classmethod
    def activate(cls) -> str:
        """Every guard -- the browser, the trash, the temp root; returns the
        temp root. Safe to call more than once."""
        cls.browser()
        cls.trash()
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

        patched = [getattr(webbrowser, name) for name in cls._LAUNCHERS]
        patched += [
            getattr(webbrowser, name).open
            for name in cls._LAUNCHER_TYPES
            if getattr(webbrowser, name, None) is not None
        ]
        return tempfile.gettempdir() == state["temp_dir"] and all(
            entry is state["guard"] for entry in patched
        )

    @classmethod
    @contextmanager
    def user_config(cls) -> Iterator[str]:
        """Point the ecosystem user-config root at a throwaway dir for the block.

        Every :class:`UserConfig` doc, every :class:`PresetStore` user tier and
        the :class:`NamingConvention` resolve under ``$UITK_PRESETS_ROOT``, so a
        test that reads one reads the DEVELOPER's settings -- a convention with
        Lightmap set to ``_LM`` fails a test written against ``_Lightmap`` -- and
        a test that writes one edits them. Opt-in per test rather than part of
        :meth:`activate`: a suite asserting the default root must still see it.

        Also hides a studio convention doc (``$PYTHONTK_NAMING_CONVENTION``) and
        reloads the convention on entry and exit, so neither side reads the
        other's cached table. Yields the root. Pre-3.11 ``unittest`` setUp form::

            sandbox = TestSandbox.user_config()
            self.config_root = sandbox.__enter__()
            self.addCleanup(sandbox.__exit__, None, None, None)
        """
        from pythontk.core_utils.naming_convention import (
            CONFIG_ENV_VAR,
            NamingConvention,
        )
        from pythontk.core_utils.user_config import CONFIG_ROOT_ENV_VAR
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        saved = {
            var: os.environ.get(var) for var in (CONFIG_ROOT_ENV_VAR, CONFIG_ENV_VAR)
        }
        with TempArtifacts("ptk_user_config", policy="scoped") as store:
            os.environ[CONFIG_ROOT_ENV_VAR] = store.dir_path()
            os.environ.pop(CONFIG_ENV_VAR, None)
            NamingConvention.reload()
            try:
                yield os.environ[CONFIG_ROOT_ENV_VAR]
            finally:
                for var, value in saved.items():
                    if value is None:
                        os.environ.pop(var, None)
                    else:
                        os.environ[var] = value
                NamingConvention.reload()
