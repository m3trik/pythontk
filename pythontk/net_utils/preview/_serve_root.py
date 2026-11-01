# !/usr/bin/python
# coding=utf-8
"""What the preview's serve root holds, and how each file gets there.

The web runtime, served as it is laid out in this package: the viewer page
(``viewer.html``, materialized as ``index.html``) and its kernel
(``kernel/*.js``, the modules the page imports); the viewer-script registry --
the page's extension seam: :attr:`PreviewServer.SCRIPTS` /
:meth:`PreviewServer.add_script` -- whose active packaged features are served
at their own paths (``features/...``, each with the modules beside it) and a
caller's own scripts under ``scripts/``; and the atomic write every served
file lands through, so a page polling the root never reads a partial one. A
managed (temp) root is re-synced from the package and swept of whatever is no
longer active; a caller-supplied root is a working directory, never
overwritten or deleted from.

:class:`_ServeRoot` is the server's collaborator that owns the active script
set and the served page's fingerprint. It reads the server's tables
(:attr:`PreviewServer.SCRIPTS`, its routes, its root) through it at call time,
so a class or instance override of one reaches it; the server's state lock
guards the set a manifest reads.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union, TYPE_CHECKING

from pythontk.file_utils._file_utils import FileUtils

if TYPE_CHECKING:
    from pythontk.net_utils.preview.server import PreviewServer


class _ServeRoot:
    """Viewer page, viewer scripts, atomic asset writes.

    A private collaborator of :class:`~pythontk.PreviewServer`; call it through
    the facade, whose methods carry the documentation.
    """

    def __init__(self, server: "PreviewServer"):
        self._server = server
        #: Active viewer scripts, registered name -> source file. Ordered, and
        #: the page imports them in this order.
        self.scripts: Dict[str, Path] = {}
        #: Fingerprint of the page actually served (:meth:`ensure_viewer`).
        self.viewer_stamp = ""

    # ------------------------------------------------------------------
    # The active script set
    # ------------------------------------------------------------------
    def names(self) -> tuple:
        with self._server._lock:
            return tuple(self.scripts)

    def add(self, name: str, path: Optional[Union[str, Path]] = None) -> None:
        source = self.resolve(name, path)
        with self._server._lock:
            self.scripts[name] = source
        self.ensure_scripts()

    def remove(self, name: str) -> None:
        with self._server._lock:
            self.scripts.pop(name, None)
        self.ensure_scripts()

    def replace(
        self, scripts: Optional[Union[Dict[str, Any], List[str], tuple]]
    ) -> None:
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
        resolved = {name: self.resolve(name, path) for name, path in items}
        with self._server._lock:
            self.scripts = resolved
        self.ensure_scripts()

    def resolve(self, name: str, path: Optional[Union[str, Path]] = None) -> Path:
        """The file backing *name*, raising rather than serving a 404 later.

        An unknown packaged name is a *caller* error that is otherwise silent:
        the manifest would name a module, the page's import would 404, and the
        only trace is a console warning nobody is watching inside a headset.
        """
        server = self._server
        if path is not None:
            source = Path(path)
            if not source.is_file():
                raise FileNotFoundError(
                    f"Viewer script {name!r}: no such file: {source}"
                )
            return source
        filename = server.SCRIPTS.get(name)
        if filename is None:
            raise KeyError(
                f"Unknown viewer script {name!r}. Packaged: "
                f"{', '.join(sorted(server.SCRIPTS)) or 'none'}; "
                f"pass path= to register one of your own."
            )
        source = server.SCRIPTS_DIR / filename
        if not source.is_file():
            # A broken install rather than a caller error, but left unchecked it
            # surfaces the same way the unknown name would: `ensure_scripts`
            # raises out of a copy deep inside `deliver()`, AFTER the active set
            # was swapped and after the conversion passes have already run.
            raise FileNotFoundError(
                f"Packaged viewer script {name!r} is missing from "
                f"{server.SCRIPTS_DIR} -- the install did not carry its package data."
            )
        return source

    def active(self) -> Dict[str, Path]:
        """Snapshot of the active name -> source map."""
        with self._server._lock:
            return dict(self.scripts)

    def script_urls(self, exclude: Iterable[str] = ()) -> List[str]:
        """The URL the page imports each active script from, in load order,
        but for the names in *exclude*. Call under the server's lock."""
        return [
            self.script_url(name, source)
            for name, source in self.scripts.items()
            if name not in exclude
        ]

    def script_files(self, exclude: Iterable[str] = ()) -> set:
        """Every served path of the active scripts (each with the modules it
        imports beside it), but for the names in *exclude*. Call under the
        server's lock."""
        files: set = set()
        for name, source in self.scripts.items():
            if name not in exclude:
                files.update(self.served_files(name, source))
        return files

    # ------------------------------------------------------------------
    # The page
    # ------------------------------------------------------------------
    def ensure_viewer(self) -> None:
        """Materialize the packaged viewer into the serve root.

        A **caller-supplied** root is treated as a working directory: an
        existing page is left alone, so edits made there survive a restart.

        A **managed** (temp) root is re-synced from the package whenever this
        runs -- which is every :meth:`PreviewServer.start` *and* every
        :meth:`PreviewServer.publish`, not just the first. The deliverer holding
        a server lives for the whole DCC session, and ``start`` is idempotent,
        so a once-only copy meant an edited viewer could not reach an
        already-running session at all: the page is where the preview is
        actually rendered, so that presents as "the fix did nothing" with
        nothing wrong in the pipeline. Re-syncing on publish makes an edit land
        on the next push instead of the next restart.

        The comparison is by content, so the common case is a read and no write.
        """
        server = self._server
        if not server._viewer:
            return
        src = server.SCRIPTS_DIR / "viewer.html"
        if not src.is_file():  # pragma: no cover - packaging failure
            server.logger.warning("Viewer page missing from the package: %s", src)
            return
        served = [server.root / "index.html"]
        self.sync_file(src, served[0])
        # The page is markup; the viewer is its kernel, which travels with it.
        # Swept like the scripts: a module an update retired must not linger
        # beside the ones the new page imports.
        kernel = self.kernel_files()
        if not kernel:
            # A packaging failure: the page alone never starts, since it
            # imports kernel/main.js. Said here, where it is a fact about the
            # install, rather than left to a blank tab.
            server.logger.warning(
                "Viewer kernel missing from the package: %s",
                server.SCRIPTS_DIR / server.KERNEL_ROUTE,
            )
        for route, module in kernel.items():
            self.sync_file(module, server.root / route)
        self.sweep(server.KERNEL_ROUTE, kernel)
        served += [server.root / route for route in sorted(kernel)]
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
                digest.update(path.relative_to(server.root).as_posix().encode("utf-8"))
                digest.update(b"\0")
                digest.update(path.read_bytes())
            stamp = digest.hexdigest()[:12]
        except OSError:
            stamp = ""
        with server._lock:
            self.viewer_stamp = stamp

    def kernel_files(self) -> Dict[str, Path]:
        """The page's kernel modules: served path -> packaged file."""
        server = self._server
        directory = server.SCRIPTS_DIR / server.KERNEL_ROUTE
        return {
            f"{server.KERNEL_ROUTE}/{module.name}": module
            for module in sorted(directory.glob("*.js"))
        }

    def served_files(self, name: str, source: Path) -> Dict[str, Path]:
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
        server = self._server
        entry = server.SCRIPTS.get(name)
        if entry is None or source != server.SCRIPTS_DIR / entry:
            return {f"{server.SCRIPTS_ROUTE}/{name}.js": source}
        files = {entry: source}
        folder = Path(entry).parent
        if folder.as_posix() != server.FEATURES_ROUTE:
            for module in sorted(source.parent.glob("*.js")):
                files.setdefault((folder / module.name).as_posix(), module)
        return files

    def script_url(self, name: str, source: Path) -> str:
        """The URL the page imports the active script *name* from."""
        return next(iter(self.served_files(name, source)))

    # ------------------------------------------------------------------
    # Placing files
    # ------------------------------------------------------------------
    def sync_file(self, src: Path, dst: Path) -> None:
        """Place *src* at *dst* unless the caller owns it or it is already current.

        The one materialization policy, shared by the viewer page and every
        viewer script because both answer the same two questions the same way:
        a **caller-supplied** root is a working directory, so a file already
        there is never overwritten (edits made in place survive a restart); a
        **managed** root is ours, so it is compared by content -- making the
        common case a read and no write -- and rewritten when it differs.

        The write is atomic (via :meth:`write_asset`) rather than a plain
        copy: this runs on every publish, against a directory a browser is
        actively fetching from, and a reader landing mid-copy gets a truncated
        file. For a script that is terminal -- the page claims a script URL
        before awaiting its import, so a parse failure retires it for the life
        of the page.
        """
        if dst.exists() and (
            self._server._temp is None or dst.read_bytes() == src.read_bytes()
        ):
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        self.write_asset(src, dst, move=False)

    def ensure_scripts(self) -> None:
        """Materialize the active viewer scripts into the serve root.

        The extension seam's disk half: each active packaged feature
        (:attr:`PreviewServer.SCRIPTS`) and caller-registered module is copied
        under the serve root, where :meth:`served_files` says, so the page can
        ``import()`` what :meth:`PreviewServer.manifest` names.

        Per-file placement is :meth:`sync_file`'s policy, shared with the
        viewer page. What is specific here is the **sweep**: in a managed root
        the directories are entirely ours, so a module no longer active is
        removed -- otherwise a script switched off for this push stays on disk
        and the next reader of the serve root sees one the manifest does not
        name. A caller-supplied root is never deleted from.
        """
        server = self._server
        served: Dict[str, Path] = {}
        for name, src in self.active().items():
            served.update(self.served_files(name, src))
        for route, src in served.items():
            self.sync_file(src, server.root / route)
        for route in (server.FEATURES_ROUTE, server.SCRIPTS_ROUTE):
            self.sweep(route, served)

    def sweep(self, route: str, keep: Dict[str, Path]) -> None:
        """In a managed root, delete every module under *route* that *keep*
        (served path -> source) does not serve, and the folders that empties."""
        server = self._server
        directory = server.root / route
        if server._temp is None or not directory.is_dir():
            return  # never sweep a directory the caller owns
        for stale in directory.rglob("*.js"):
            if stale.relative_to(server.root).as_posix() not in keep:
                stale.unlink()
        for folder in sorted(directory.rglob("*"), reverse=True):
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()

    @staticmethod
    def write_asset(src: Path, dst: Path, move: bool) -> None:
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
