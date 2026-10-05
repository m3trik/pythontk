# !/usr/bin/python
# coding=utf-8
"""What the preview's serve root holds, and how each file gets there.

The web runtime, served as it is laid out in this package: the viewer page
(``viewer.html``, materialized as ``index.html``) and its kernel
(``kernel/*.js``, the modules the page imports); the viewer-script registry --
the page's extension seam: :attr:`SCRIPTS` / :meth:`add_script` -- whose
active packaged features are served at their own paths (``features/...``,
each with the modules beside it) and a caller's own scripts under
``scripts/``; and the atomic write every served file lands through, so a page
polling the root never reads a partial one. A managed (temp) root is re-synced
from the package and swept of whatever is no longer active; a caller-supplied
root is a working directory, never overwritten or deleted from.

One job of :class:`~pythontk.PreviewServer`, composed in ``server.py``; its
methods reach the rest of the server through ``self``.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.file_utils._file_utils import FileUtils

if TYPE_CHECKING:
    from pythontk.net_utils.preview.server import PreviewServer


class _ServeRootMixin:
    """Viewer page, viewer scripts, atomic asset writes.

    A private part of :class:`~pythontk.PreviewServer`; call it through the facade.
    """

    #: The web runtime's root, which :attr:`SCRIPTS`' paths resolve under: the
    #: page (``viewer.html``), its kernel (``kernel/``) and the packaged
    #: features (``features/``). Served as laid out here, so a module's
    #: relative imports resolve the same on disk and in the page.
    SCRIPTS_DIR = Path(__file__).parent

    #: The kernel's folder, under :attr:`SCRIPTS_DIR` and the serve root alike.
    KERNEL_ROUTE = "kernel"

    #: The packaged features' folder, under :attr:`SCRIPTS_DIR` and the serve
    #: root alike.
    FEATURES_ROUTE = "features"

    #: Serve-root subdirectory (and URL prefix) a caller's own scripts live
    #: under (:meth:`add_script` with a path).
    SCRIPTS_ROUTE = "scripts"

    #: Built-in viewer scripts -- the packaged features -- registered name ->
    #: the entry module's path under :attr:`SCRIPTS_DIR`. A feature that is one
    #: module is a file in ``features/``; one with modules of its own (a
    #: host-free model beside its three.js adapter) is a folder, every module
    #: in which is served with its entry. This is the *extension registry*:
    #: the viewer page itself stays the stable path and gains behaviour by a
    #: module being activated, never by being edited (OCP). A script is an ES
    #: module whose default export is called once with the page's viewer API
    #: -- see ``docs/webxr_preview.md`` for that surface and :meth:`add_script`
    #: for registering one from outside this package.
    SCRIPTS: Dict[str, str] = {
        "turntable": "features/turntable.js",
        "inspect": "features/inspect.js",
        "shadow_rig": "features/shadow_rig/shadow_rig.js",
        "articulated_rig": "features/articulated_rig/articulated_rig.js",
        "playblast": "features/playblast.js",
        "snapshot": "features/snapshot.js",
    }

    #: Packaged scripts a deliverable turns on by itself: registered name ->
    #: the root-extras key whose presence in a published GLB activates it.
    #: The default set is otherwise empty (the page must not pay for a seam
    #: nobody asked to use), but a script that exists to read a manifest the
    #: conversion wrote is not optional in any useful sense -- the shadow rigs
    #: ship as still planes without it, which reads as a broken export rather
    #: than a missing checkbox. :meth:`publish` probes the GLB's JSON chunk
    #: (never its geometry) and activates through :meth:`add_script`, so the
    #: script joins whatever set the push named, in the same load order. Opt
    #: out by removing the entry, or with :meth:`remove_script` after the push.
    #: The keys are the records' declared web projections, never spelled here.
    AUTO_SCRIPTS: Dict[str, str] = {
        "shadow_rig": SceneRecords.SHADOWS.web.key,
        # An articulated rig is a prop a hand is meant to move; without its
        # script the page shows it frozen at whatever the clip says, which
        # reads as "the rig did not export" rather than as a missing option.
        "articulated_rig": SceneRecords.ARTICULATION.web.key,
        # A deliverable that ships clips grows a picker and a transport in the
        # page; recording what that transport plays is the same kind of "not
        # optional in any useful sense" as the shadow rigs. There is no
        # checkbox for it because the alternative is worse: the button would be
        # missing on exactly the push a reviewer just watched and wants to
        # send on, and getting it would mean re-exporting the animation.
        "playblast": SceneRecords.SHOTS.web.key,
    }

    #: Viewer scripts that write through the owner's routes -- a recording, a
    #: still -- and are therefore withheld from a guest (:meth:`start_guest`).
    #: The guest listener refuses every write, so served to a guest each would
    #: be a button that fails; withheld, the guest never downloads it either. A
    #: script registered under another name is served to guests: an overlay
    #: that only reads needs nothing here.
    OWNER_SCRIPTS = frozenset({"playblast", "snapshot"})

    @property
    def scripts(self) -> tuple:
        """Registered names of the viewer scripts currently active, in load order."""
        with self._lock:
            return tuple(self._scripts)

    def add_script(
        self, name: str, path: Optional[Union[str, Path]] = None
    ) -> "PreviewServer":
        """Activate a viewer script, on disk and in the manifest at once.

        Every mutator here materializes immediately rather than deferring to
        the next publish, and that is a correctness requirement rather than a
        convenience: the manifest starts naming a script the moment it is
        registered, and the page claims a script URL *before* awaiting its
        import, so a single 404 in the gap would retire that module for the
        life of the page.

        Parameters:
            name: A key of :attr:`SCRIPTS` (a packaged script, imported from
                its own path, ``features/turntable.js``), or any name at all
                when *path* is given.
            path: An ES module outside this package to serve under *name*
                (``scripts/<name>.js``) -- the seam a consumer extends through
                without vendoring anything into pythontk. It may import the
                kernel's pure modules by their served path
                (``../kernel/math.js``).

        Re-registering a name replaces its source and keeps its position, so a
        caller can override a packaged script with its own file.
        """
        source = self._resolve_script(name, path)
        with self._lock:
            self._scripts[name] = source
        self._ensure_scripts()
        return self

    def remove_script(self, name: str) -> "PreviewServer":
        """Deactivate a viewer script (unknown names are ignored), and sweep it.

        Materialized at once, for the reason on :meth:`add_script`.
        """
        with self._lock:
            self._scripts.pop(name, None)
        self._ensure_scripts()
        return self

    def set_scripts(
        self, scripts: Optional[Union[Dict[str, Any], List[str], tuple]]
    ) -> "PreviewServer":
        """Replace the whole active set (``None`` or empty clears it).

        Accepts an iterable of names (packaged scripts) or a ``{name: path}``
        mapping (external modules). Replacing rather than merging is what makes
        a per-push script set possible -- see :meth:`PreviewBridge.push`.

        Materialized at once, for the reason on :meth:`add_script`.
        """
        if isinstance(scripts, str):
            # A str is iterable, so this would otherwise walk its CHARACTERS and
            # fail with KeyError('t') -- a message that names nothing the caller
            # wrote. The singular form has its own method.
            raise TypeError(
                f"set_scripts expects a list or mapping, not a string: {scripts!r}. "
                f"Use add_script({scripts!r}) or set_scripts([{scripts!r}])."
            )
        items = (
            scripts.items()
            if isinstance(scripts, dict)
            else [(name, None) for name in (scripts or ())]
        )
        # Resolved BEFORE the swap so a bad name leaves the active set intact
        # rather than half-applied -- the page is already running against it.
        resolved = {name: self._resolve_script(name, path) for name, path in items}
        with self._lock:
            self._scripts = resolved
        self._ensure_scripts()
        return self

    def _resolve_script(
        self, name: str, path: Optional[Union[str, Path]] = None
    ) -> Path:
        """The file backing *name*, raising rather than serving a 404 later.

        An unknown packaged name is a *caller* error that is otherwise silent:
        the manifest would name a module, the page's import would 404, and the
        only trace is a console warning nobody is watching inside a headset.
        """
        if path is not None:
            source = Path(path)
            if not source.is_file():
                raise FileNotFoundError(
                    f"Viewer script {name!r}: no such file: {source}"
                )
            return source
        filename = self.SCRIPTS.get(name)
        if filename is None:
            raise KeyError(
                f"Unknown viewer script {name!r}. Packaged: "
                f"{', '.join(sorted(self.SCRIPTS)) or 'none'}; "
                f"pass path= to register one of your own."
            )
        source = self.SCRIPTS_DIR / filename
        if not source.is_file():
            # A broken install rather than a caller error, but left unchecked it
            # surfaces the same way the unknown name would: `_ensure_scripts`
            # raises out of a copy deep inside `deliver()`, AFTER the active set
            # was swapped and after the conversion passes have already run.
            raise FileNotFoundError(
                f"Packaged viewer script {name!r} is missing from "
                f"{self.SCRIPTS_DIR} -- the install did not carry its package data."
            )
        return source

    def _active_scripts(self) -> Dict[str, Path]:
        """Snapshot of the active name -> source map."""
        with self._lock:
            return dict(self._scripts)

    def _ensure_viewer(self) -> None:
        """Materialize the packaged viewer into the serve root.

        A **caller-supplied** root is treated as a working directory: an
        existing page is left alone, so edits made there survive a restart.

        A **managed** (temp) root is re-synced from the package whenever this
        runs -- which is every :meth:`start` *and* every :meth:`publish`, not
        just the first. The deliverer holding a server lives for the whole DCC
        session, and :meth:`start` is idempotent, so a once-only copy meant an
        edited viewer could not reach an already-running session at all: the
        page is where the preview is actually rendered, so that presents as
        "the fix did nothing" with nothing wrong in the pipeline. Re-syncing on
        publish makes an edit land on the next push instead of the next restart.

        The comparison is by content, so the common case is a read and no write.
        """
        if not self._viewer:
            return
        src = self.SCRIPTS_DIR / "viewer.html"
        if not src.is_file():  # pragma: no cover - packaging failure
            self.logger.warning("Viewer page missing from the package: %s", src)
            return
        served = [self.root / "index.html"]
        self._sync_file(src, served[0])
        # The page is markup; the viewer is its kernel, which travels with it.
        # Swept like the scripts: a module an update retired must not linger
        # beside the ones the new page imports.
        kernel = self._kernel_files()
        if not kernel:
            # A packaging failure: the page alone never starts, since it
            # imports kernel/main.js. Said here, where it is a fact about the
            # install, rather than left to a blank tab.
            self.logger.warning(
                "Viewer kernel missing from the package: %s",
                self.SCRIPTS_DIR / self.KERNEL_ROUTE,
            )
        for route, module in kernel.items():
            self._sync_file(module, self.root / route)
        self._sweep(self.KERNEL_ROUTE, kernel)
        served += [self.root / route for route in sorted(kernel)]
        # Fingerprint of the page actually SERVED -- the page and every kernel
        # module it imports (a caller-owned root keeps its own copies, which may
        # differ from the package) -- published in the manifest so an OPEN tab
        # can tell its script has been superseded and reload itself: the poll
        # swaps the model but never the JavaScript running it, so a viewer edit
        # otherwise reached a running session's page only via F5 (measured
        # 2026-09-05: a highlight channel the GLB carried, and a page from
        # before the binding existed).
        digest = hashlib.sha1()
        try:
            for path in served:
                digest.update(path.relative_to(self.root).as_posix().encode("utf-8"))
                digest.update(b"\0")
                digest.update(path.read_bytes())
            stamp = digest.hexdigest()[:12]
        except OSError:
            stamp = ""
        with self._lock:
            self._viewer_stamp = stamp

    def _kernel_files(self) -> Dict[str, Path]:
        """The page's kernel modules: served path -> packaged file."""
        directory = self.SCRIPTS_DIR / self.KERNEL_ROUTE
        return {
            f"{self.KERNEL_ROUTE}/{module.name}": module
            for module in sorted(directory.glob("*.js"))
        }

    def _served_files(self, name: str, source: Path) -> Dict[str, Path]:
        """Where the active script *name* (backed by *source*) is served:
        served path -> source file, the entry module first.

        A packaged feature is served at its own path, with every module of its
        folder (its model beside its adapter), so its relative imports resolve
        in the page as they do on disk; anything else -- a caller's module, or
        a caller's file registered over a packaged name -- at
        ``scripts/<name>.js``, a function of the *name* alone: two callers
        registering different files under one name is a collision the registry
        resolves, not one the URL space has to.
        """
        entry = self.SCRIPTS.get(name)
        if entry is None or source != self.SCRIPTS_DIR / entry:
            return {f"{self.SCRIPTS_ROUTE}/{name}.js": source}
        files = {entry: source}
        folder = Path(entry).parent
        if folder.as_posix() != self.FEATURES_ROUTE:
            for module in sorted(source.parent.glob("*.js")):
                files.setdefault((folder / module.name).as_posix(), module)
        return files

    def _script_url(self, name: str, source: Path) -> str:
        """The URL the page imports the active script *name* from."""
        return next(iter(self._served_files(name, source)))

    def _sync_file(self, src: Path, dst: Path) -> None:
        """Place *src* at *dst* unless the caller owns it or it is already current.

        The one materialization policy, shared by the viewer page and every
        viewer script because both answer the same two questions the same way:
        a **caller-supplied** root is a working directory, so a file already
        there is never overwritten (edits made in place survive a restart); a
        **managed** root is ours, so it is compared by content -- making the
        common case a read and no write -- and rewritten when it differs.

        The write is atomic (via :meth:`_write_asset`) rather than a plain
        copy: this runs on every publish, against a directory a browser is
        actively fetching from, and a reader landing mid-copy gets a truncated
        file. For a script that is terminal -- the page claims a script URL
        before awaiting its import, so a parse failure retires it for the life
        of the page.
        """
        if dst.exists() and (
            self._temp is None or dst.read_bytes() == src.read_bytes()
        ):
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        self._write_asset(src, dst, move=False)

    def _ensure_scripts(self) -> None:
        """Materialize the active viewer scripts into the serve root.

        The extension seam's disk half: each active packaged feature
        (:attr:`PreviewServer.SCRIPTS`) and caller-registered module is copied
        under the serve root, where :meth:`_served_files` says, so the page can
        ``import()`` what :meth:`PreviewServer.manifest` names.

        Per-file placement is :meth:`_sync_file`'s policy, shared with the
        viewer page. What is specific here is the **sweep**: in a managed root
        the directories are entirely ours, so a module no longer active is
        removed -- otherwise a script switched off for this push stays on disk
        and the next reader of the serve root sees one the manifest does not
        name. A caller-supplied root is never deleted from.
        """
        served: Dict[str, Path] = {}
        for name, src in self._active_scripts().items():
            served.update(self._served_files(name, src))
        for route, src in served.items():
            self._sync_file(src, self.root / route)
        for route in (self.FEATURES_ROUTE, self.SCRIPTS_ROUTE):
            self._sweep(route, served)

    def _sweep(self, route: str, keep: Dict[str, Path]) -> None:
        """In a managed root, delete every module under *route* that *keep*
        (served path -> source) does not serve, and the folders that empties."""
        directory = self.root / route
        if self._temp is None or not directory.is_dir():
            return  # never sweep a directory the caller owns
        for stale in directory.rglob("*.js"):
            if stale.relative_to(self.root).as_posix() not in keep:
                stale.unlink()
        for folder in sorted(directory.rglob("*"), reverse=True):
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()

    def _write_asset(self, src: Path, dst: Path, move: bool) -> None:
        """Place ``src`` at ``dst`` so a concurrent poll never sees a partial file.

        The write lands on a sibling ``.part`` first and is then renamed:
        ``os.replace`` is atomic on the same filesystem, so a request either
        gets the whole previous asset or the whole new one. Copying straight
        onto the served path would hand a mid-copy GLB to any poll that lands
        during the write -- routine, since the viewer polls once a second.
        """
        part = dst.with_name(dst.name + ".part")
        if move:
            shutil.move(str(src), str(part))
        else:
            shutil.copyfile(src, part)
        FileUtils.replace_file(part, dst)
