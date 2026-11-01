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

A host that goes on after the run -- the user's own Maya, where mayatk's
in-session harness runs a suite -- takes the guards for the run only:
``with ptk.TestSandbox.activated():`` puts back everything it put up, an
exception included.

A test that legitimately exercises a browser launch patches ``webbrowser.open``
itself: ``unittest.mock.patch`` layers over the guard and restores it after. A
test of the trash itself runs inside :meth:`TestSandbox.real_trash`.

A third effect is opt-in per test, :meth:`TestSandbox.user_config`: reading
(or writing) the developer's own config docs -- the naming convention, preset
stores -- through the shared user-config root.
"""

import os
import tempfile
from contextlib import ExitStack, contextmanager
from typing import Callable, Dict, Iterator, List, Optional, Tuple


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
    #: The ecosystem temp root (``TempArtifacts.ROOT_ENV_VAR``) and per-user
    #: cache root (``UserConfig.CACHE_ROOT_ENV_VAR``), both pointed into the root.
    _STORE_ENV = ("PYTHONTK_TEMP_ROOT", "PYTHONTK_CACHE_ROOT")
    #: The per-user cache's folder in the root.
    _CACHE_FOLDER = "user_cache"
    #: The root's ``TempArtifacts`` prefix; a session store, so its tag names its owner.
    _ROOT_PREFIX = "ptk_test_sandbox"

    _state: Dict[str, object] = {
        "guard": None,  # the one function standing in for every launcher
        "launchers": None,  # the (owner, name, real) entries the guard stands in for
        "temp_dir": None,
        "temp_store": None,  # the TempArtifacts owning temp_dir; held so its exit cleanup fires
        "trash": None,  # the real FileUtils.move_to_trash, while the guard stands in
        "can_trash": None,  # ...and the real FileUtils.can_trash
        "trash_owners": None,  # (FileUtils class, its move, its can) per class guarded
        "host_env": {},  # {variable: its value before the first redirect, None = unset}
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

    @classmethod
    def _lift_browser(cls) -> None:
        """Put back every launcher :meth:`TestSandbox.browser` stands in for."""
        state = cls._state
        for owner, name, real in state["launchers"] or ():
            if real is not None:
                setattr(owner, name, real)
            elif name in vars(owner):  # inherited before the guard took the name
                delattr(owner, name)
        state["guard"] = state["launchers"] = None

    @classmethod
    def _guard_trash(cls, owner) -> None:
        """Stand the trash guard in on *owner*, a ``FileUtils`` class, keeping
        its own calls to put back (:meth:`_lift_trash`)."""
        cls._state["trash_owners"].append(
            (owner, owner.__dict__["move_to_trash"], owner.__dict__["can_trash"])
        )
        owner.move_to_trash = staticmethod(cls._make_trash_guard())
        # A prompt asks first: told "no trash" too, it says what will happen.
        owner.can_trash = staticmethod(lambda path: False)

    @classmethod
    def _lift_trash(cls) -> None:
        """Put back each guarded class's own :meth:`FileUtils.move_to_trash`
        and ``can_trash`` -- a reload inside the block guards a second one."""
        state = cls._state
        if state["trash"] is None:
            return
        for owner, move, can in state["trash_owners"] or ():
            owner.move_to_trash = move
            owner.can_trash = can
        state["trash"] = state["can_trash"] = state["trash_owners"] = None

    @classmethod
    def _redirect_env(cls, name: str, value: Optional[str]) -> None:
        """Set *name* to *value* (unset for ``None``), recording what it was
        before the first redirect -- what :meth:`TestSandbox.host_environ`
        hands back. Every guard that changes a variable goes through here
        (uitk's presets root too), or a detached app would inherit that one.

        The record is replaced, never edited in place: a test that snapshots
        :attr:`_state` with ``mock.patch.dict`` (a shallow copy) gets the
        record back with everything else.
        """
        ledger = cls._state["host_env"]
        if name not in ledger:
            cls._state["host_env"] = {**ledger, name: os.environ.get(name)}
        cls._set_env(name, value)

    @classmethod
    def _restore_env(cls, name: str, value: Optional[str]) -> None:
        """Put *name* back to *value* (unset for ``None``). Forgotten once
        that is the host's own value; a nested guard restoring to an outer
        guard's value leaves it recorded."""
        ledger = cls._state["host_env"]
        if name in ledger and ledger[name] == value:
            cls._state["host_env"] = {k: v for k, v in ledger.items() if k != name}
        cls._set_env(name, value)

    @staticmethod
    def _set_env(name: str, value: Optional[str]) -> None:
        """``os.environ[name] = value``, where ``None`` unsets it."""
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    @classmethod
    def _temp_snapshot(cls) -> Tuple[Optional[str], Dict[str, Optional[str]]]:
        """What :meth:`TestSandbox.temp` changes, read before it does."""
        names = cls._TEMP_ENV + (cls._ROOT_ENV,) + cls._STORE_ENV
        return tempfile.tempdir, {name: os.environ.get(name) for name in names}

    @classmethod
    def _lift_temp(
        cls, snapshot: Tuple[Optional[str], Dict[str, Optional[str]]]
    ) -> None:
        """Point the temp dir and its variables back where *snapshot* found
        them, then remove the root (best-effort: what is still held goes at
        exit, as before)."""
        state = cls._state
        tempdir, env = snapshot
        tempfile.tempdir = tempdir
        for name, value in env.items():
            cls._restore_env(name, value)
        store = state["temp_store"]
        state["temp_dir"] = state["temp_store"] = None
        if store is not None:
            store.cleanup(force=True)


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
        # The real ones, kept so :meth:`activated` can put them back.
        real = [
            (webbrowser, name, getattr(webbrowser, name)) for name in cls._LAUNCHERS
        ]
        for name in cls._LAUNCHER_TYPES:
            browser_type = getattr(webbrowser, name, None)
            if browser_type is not None:
                real.append((browser_type, "open", vars(browser_type).get("open")))
        for owner, name, _ in real:
            setattr(owner, name, guard)
        state["launchers"] = real
        state["guard"] = guard

    @classmethod
    def temp(cls) -> str:
        """Route the process's temp dir into one throwaway root; returns it.

        ``tempfile.gettempdir()`` -- and so every :class:`TempArtifacts` store,
        hand-off payload and scratch dir that defaults to it -- resolves inside
        the root for the rest of the process, and ``TMPDIR``/``TEMP``/``TMP``
        point there so a child process (mayapy, Blender, FBX2glTF) inherits the
        same -- a waited one: a detached app outlives the root, so it is
        launched with the host's (:meth:`host_environ`). The root IS the
        ecosystem temp root (``$PYTHONTK_TEMP_ROOT``, :meth:`TempArtifacts.root`),
        so a store in this process or in a Python child lands in it however
        that child's ``tempfile`` resolves, and the per-user cache root
        (``$PYTHONTK_CACHE_ROOT``, :meth:`UserConfig.cache_dir`) is a folder in
        it, so a test's regenerable cache never reaches the developer's. A test
        that steers stores into a folder of its own patches
        ``$PYTHONTK_TEMP_ROOT``, not ``tempfile.tempdir``. The root goes at
        interpreter exit, or at
        :meth:`ProcessExit.hard_exit`, which skips it. A root that outlives its
        owner -- the process was killed, or a file in it was still held when
        it exited -- goes at the next activation, which reclaims every root
        whose owner has exited (a session store's tag names its owner, see
        :class:`TempArtifacts`), so the worst case is one stale directory
        until the next run.

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

        # Top level: allocated in the REAL temp root, before the redirect, so
        # the sweep of prior killed runs looks where they landed -- FLAT in it,
        # a root of roots rather than a store folder: every level of nesting
        # is paid again by every path a suite builds inside it, and a mayatk
        # chunk's records ran past Windows' 260 one level deeper. In a child:
        # inside the parent's root, by name.
        parent = os.environ.get(cls._ROOT_ENV) or None
        if parent is None:
            parent = TempArtifacts.root()
            legacy = tempfile.gettempdir()
            if os.path.normcase(os.path.abspath(legacy)) != os.path.normcase(parent):
                # For one release (TempArtifacts._LEGACY_SWEEP_REMOVE_IN): the
                # roots the pre-root layout left flat in the system temp dir.
                TempArtifacts(
                    cls._ROOT_PREFIX, policy="session", dir=legacy
                ).sweep_stale()
        store = TempArtifacts(cls._ROOT_PREFIX, policy="session", dir=parent)
        root = store.dir_path()
        tempfile.tempdir = root
        for name in cls._TEMP_ENV + (cls._ROOT_ENV, cls._STORE_ENV[0]):
            cls._redirect_env(name, root)
        cls._redirect_env(cls._STORE_ENV[1], os.path.join(root, cls._CACHE_FOLDER))
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
        state["trash_owners"] = []
        cls._guard_trash(FileUtils)

    @classmethod
    def rearm(cls) -> None:
        """Carry the standing guards over a reload of pythontk.

        Reloading re-executes ``FileUtils`` and this module: the new
        ``FileUtils`` trashes for real, and the new ``TestSandbox`` reads as
        off. mayatk's in-session harness reloads pythontk inside its sandbox
        (the user's long-lived Maya may hold stale modules), so a lightmap
        re-bake there sent scratch maps to the user's Recycle Bin. Called after
        such a reload, this shares the standing state with the reloaded
        sandbox and stands the trash guard in on the reloaded ``FileUtils``,
        whose own calls the way out puts back. A no-op where nothing was
        reloaded and no guard stands.
        """
        import sys

        state = cls._state
        module = sys.modules.get(__name__)
        reloaded = getattr(module, "_TestSandboxInternal", None)
        if reloaded is not None and reloaded._state is not state:
            reloaded._state = state
            module.TestSandbox.launches = cls.launches
            module.TestSandbox.trashed = cls.trashed
        if state["trash"] is None:
            return
        from pythontk.file_utils._file_utils import FileUtils

        if not any(owner is FileUtils for owner, _, _ in state["trash_owners"]):
            cls._guard_trash(FileUtils)

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
    @contextmanager
    def activated(cls) -> Iterator[str]:
        """Every guard for the ``with`` block only; yields the temp root.

        :meth:`activate` suits a process that is all test run -- a runner, or
        a DCC it launched -- since nothing undoes it. A host that goes on after
        the run needs the guards gone again: mayatk's in-session harness runs a
        suite inside the user's own Maya, and under :meth:`activate` that
        session went on refusing browser launches, keeping nothing from the
        trash and writing temp files into a root deleted when Maya exits.

        On the way out, an exception included, the launchers, the trash and the
        temp dir (``TMPDIR``/``TEMP``/``TMP`` with it) are put back as the block
        found them, and the root is removed. A guard already standing on entry
        (an :meth:`activate` before the block) stays standing: the block takes
        down only what it put up.
        """
        state = cls._state
        with ExitStack() as lift:
            if state["guard"] is None:
                lift.callback(cls._lift_browser)
                cls.browser()
            if state["trash"] is None:
                lift.callback(cls._lift_trash)
                cls.trash()
            if state["temp_dir"] is None:
                lift.callback(cls._lift_temp, cls._temp_snapshot())
                cls.temp()
            yield state["temp_dir"]

    @classmethod
    def host_environ(
        cls, env: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, str]]:
        """*env* (default: the live process environment) as the host had it,
        before any guard redirected a variable in it -- or ``None`` while
        nothing is redirected.

        For a child that outlives the run: the root goes when its owner does,
        at exit or at the next activation's sweep of a root whose owner has
        exited, so a child still writing into it loses its files.
        :meth:`AppLauncher.launch <pythontk.AppLauncher.launch>` hands this to
        every detached launch -- measured: a GUI Maya the mayatk runner
        launched kept the root as its TEMP, the user went on working in it,
        and the WebXR preview's FBX was swept between export and conversion.

        Only a variable that still holds the sandbox's value goes back: one
        the caller set itself in *env* is its own choice.

        Parameters:
            env: The child's environment; ``None`` reads this process's live
                block (:meth:`AppLauncher.process_environ
                <pythontk.AppLauncher.process_environ>`) -- what an inheriting
                child gets, a DCC host's own variables included, which the
                ``os.environ`` snapshot misses.

        Returns:
            A new dict, or ``None`` when no guard has a variable redirected
            (the child can inherit as is).
        """
        ledger = cls._state["host_env"]
        if not ledger:
            return None
        if env is None:
            from pythontk.core_utils.app_launcher import AppLauncher

            env = AppLauncher.process_environ()
        out = dict(env)
        # Windows names are case-insensitive, and the live block keeps its
        # own spelling: match the key as it is there, or a second spelling
        # would ride beside the redirected one.
        keys = {k.upper(): k for k in out} if os.name == "nt" else {}
        for name, value in ledger.items():
            key = keys.get(name.upper(), name)
            if out.get(key) != os.environ.get(name):
                continue
            if value is None:
                out.pop(key, None)
            else:
                out[key] = value
        return out

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
        from pythontk.core_utils.user_config import UserConfig
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        CONFIG_ROOT_ENV_VAR = UserConfig.CONFIG_ROOT_ENV_VAR
        saved = {
            var: os.environ.get(var) for var in (CONFIG_ROOT_ENV_VAR, CONFIG_ENV_VAR)
        }
        with TempArtifacts("ptk_user_config", policy="scoped") as store:
            cls._redirect_env(CONFIG_ROOT_ENV_VAR, store.dir_path())
            cls._redirect_env(CONFIG_ENV_VAR, None)
            NamingConvention.reload()
            try:
                yield os.environ[CONFIG_ROOT_ENV_VAR]
            finally:
                for var, value in saved.items():
                    cls._restore_env(var, value)
                NamingConvention.reload()
