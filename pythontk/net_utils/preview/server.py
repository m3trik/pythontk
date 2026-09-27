# !/usr/bin/python
# coding=utf-8
"""Localhost static-file server for live browser / WebXR previews.

The transport half of the "push the current selection to a headset" loop: a
producer (a DCC bridge, an exporter, a test) calls :meth:`PreviewServer.publish`
with a freshly written asset; the page already open in a browser notices the
version bump on its next poll and swaps the model in without a reload.

Localhost by design
-------------------
``navigator.xr`` -- the whole WebXR entry point -- is only exposed on a *secure
context*. ``http://localhost`` is one by definition, so binding the loopback
interface buys a full ``immersive-vr`` session with no TLS certificate, no
reverse proxy and no tunnel. That covers every PC-tethered headset (Quest over
Link / Air Link, Index, WMR), because there the browser runs on *this* machine
and the headset is just the display.

Serving the same page to anyone else -- a standalone headset, a reviewer in
another city -- needs real HTTPS: a plain LAN IP is not a secure context and
silently yields no VR button at all. :meth:`PreviewServer.share` provides it
without giving up the loopback bind. It opens a SECOND listener that can only
read (:meth:`PreviewServer.start_guest`) and fronts that one with a
:class:`pythontk.ShareTunnel`, whose provider terminates TLS on a name it owns.
Who may write is decided by which socket a request arrived on, never by a
header: the owner's listener answers no public name, whatever fronts it.

Example (publish a GLB and open it):
    >>> with PreviewServer(title="Selection") as server:
    ...     server.publish("C:/tmp/scene.glb")
    ...     server.open_in_browser()
    ...     server.publish("C:/tmp/scene_v2.glb")   # the open page swaps to it

Served surface:
    ``GET /``                -> the viewer page (materialized into the serve root)
    ``GET /manifest.json``   -> ``{"version", "viewer", "asset", "updated",
                                "title", "userPos", "locomotion", "scripts",
                                "xrRuntime"}``;
                                also the heartbeat behind :meth:`PreviewServer.has_viewer`
    ``GET /scripts/<name>.js`` -> an active viewer script (see :attr:`PreviewServer.SCRIPTS`)
    ``GET /<name>``          -> any published asset, by name
    ``POST /viewer-closed``  -> the viewer's unload beacon, so a closed tab is
                                known at once rather than after a timeout
    ``POST /settings``       -> a delivery dial the page writes into the GLB
    ``POST /playblast/*``    -> ``begin`` / ``frame`` / ``finish`` / ``cancel``:
                                the page recording a clip to a movie file
                                (see :mod:`pythontk.net_utils.preview.playblast`)
    ``POST /snapshot``       -> a still of the page's view, as a PNG body
                                (see :meth:`PreviewServer.save_snapshot`)

The guest listener serves ``GET /``, ``/manifest.json`` (marked
``"guest": true``, without the owner-only scripts) and exactly the files that
manifest names; it takes the close beacon and refuses every other write.

Layout: this module holds the :class:`PreviewServer` facade -- lifecycle, the
live manifest and viewer liveness, publish and its delivery dials, and the
browser it opens. Its other jobs are private mixins beside it (``_serve_root``:
the viewer page, the viewer scripts and the atomic writes; ``_sharing``: the
guest listener and its tunnel; ``_page_outputs``: recordings and stills), and
the HTTP handler with the route names is :mod:`.routes`, re-exported here.
"""

from __future__ import annotations

import os
import threading
import time
import webbrowser
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING,
)

from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.file_utils.temp_artifacts import TempArtifacts
from pythontk.net_utils._net_utils import NetUtils
from pythontk.net_utils.preview._page_outputs import _PageOutputsMixin
from pythontk.net_utils.preview._serve_root import _ServeRootMixin
from pythontk.net_utils.preview._sharing import _SharingMixin

# The route names are the handler's; re-exported here, where callers and the
# page's tests have always imported them from.
from pythontk.net_utils.preview.routes import (
    PLAYBLAST_ACTIONS as PLAYBLAST_ACTIONS,
    PLAYBLAST_PATH as PLAYBLAST_PATH,
    SETTINGS_PATH as SETTINGS_PATH,
    SNAPSHOT_PATH as SNAPSHOT_PATH,
    VIEWER_CLOSED_PATH as VIEWER_CLOSED_PATH,
    _VIEWER_ID,
    _PreviewHandler,
    _PreviewHTTPServer,
)

if TYPE_CHECKING:  # the recorder is imported on use -- see PreviewServer.playblast
    from pythontk.net_utils.preview.playblast import PreviewPlayblast


def _mesh_convert():
    """The GLB converter, imported on use rather than at module scope.

    Every consumer here is deferred for one reason -- the converter module
    pulls in the managed-binary installer, which no other ``PreviewServer``
    user needs -- and the note was previously repeated at each call site.
    """
    from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

    return MeshConvert


class PreviewServer(LoggingMixin, _ServeRootMixin, _SharingMixin, _PageOutputsMixin):
    """Serve a directory of preview assets on loopback, with a live manifest.

    Parameters:
        root: Directory to serve. Defaults to a session-scoped temp directory
            (removed at interpreter exit) allocated through
            :class:`pythontk.TempArtifacts`.
        host: Interface to bind. Loopback by default -- see the module docstring
            for why anything else needs TLS to stay useful.
        port: ``None`` (default) prefers :attr:`DEFAULT_PORT` and falls back to
            an ephemeral port; ``0`` always takes an ephemeral port; an explicit
            port must bind or :meth:`start` raises ``OSError``.
        viewer: Materialize the packaged WebXR viewer as ``index.html``.
        title: Label shown in the viewer's status line.
        user_pos: Name of the node a view starts at -- a camera named so in
            the published scene: a headset session stands on the floor under
            it, facing where it looks, and the desktop view opens through it.
            ``None`` starts every view on the whole model. Live: read into the
            manifest on each poll, so a new name reaches the page's next
            session without a publish. See ``docs/webxr_preview.md``.
        locomotion: Whether a headset gets around on its thumbsticks (walk,
            snap-turn, teleport). ``False`` keeps a session where it started --
            under the *user_pos* camera when there is one, free to look round
            and step about the room but not to go anywhere. Live like
            *user_pos*, and at once: an open page, a guest's included, stops
            on its next poll.
    """

    DEFAULT_PORT = 8118

    #: Delivery dials the served page may write into the published GLB:
    #: name -> (:class:`MeshConvert` writer, coercion). An allow-list, because
    #: this is the only route by which the page reaches a file on disk.
    WRITABLE_SETTINGS: Dict[str, Tuple[str, Callable]] = {
        "normal_scale": ("set_glb_normal_scale", float),
    }

    #: Seconds a manifest poll counts as proof that a viewer is still open.
    #:
    #: It has to clear 60s, and by a margin. Chrome and Edge throttle
    #: ``setInterval`` in a hidden tab to roughly once per minute, and *hidden*
    #: is the normal state here: the user is working in the DCC with the
    #: preview minimised or on another tab. A window sized to the viewer's own
    #: 1s cadence would read a throttled-but-perfectly-live page as gone and
    #: pop a second tab over the DCC on every push -- worse than the missing
    #: relaunch it set out to fix. Prompt detection of a genuinely closed tab
    #: comes from the unload beacon instead, not from shortening this.
    VIEWER_TIMEOUT = 90.0

    #: Seconds a page's poll is ignored after that page beaconed that it
    #: closed. Every request is answered on its own thread, so a poll the page
    #: sent just before closing can be answered AFTER the beacon -- and brought
    #: the page back from the dead for a whole :attr:`VIEWER_TIMEOUT` (caught
    #: 2026-09-23 by the live guest test, under load). A page restored from the
    #: back/forward cache polls again once this has passed.
    CLOSED_LINGER = 5.0

    #: How often the serving thread wakes to check for a stop, in seconds.
    #: ``shutdown`` returns only after the next wake, so this bounds how long
    #: :meth:`stop` blocks. The stdlib default (0.5) was most of a suite that
    #: starts and stops a server per case: 154 cases, 74s of which ~60s was
    #: waiting on this. Twenty idle wakes a second on a daemon thread is not
    #: a cost a DCC session notices.
    POLL_INTERVAL = 0.05

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        viewer: bool = True,
        title: str = "Preview",
        user_pos: Optional[str] = "user_pos",
        locomotion: bool = True,
    ):
        self.host = host
        self.title = title
        self.user_pos = user_pos
        self.locomotion = locomotion
        self._requested_port = port
        self._viewer = viewer
        self._lock = threading.Lock()
        self._version = 0
        self._viewer_stamp = ""
        self._asset: Optional[str] = None
        #: The file the current asset was published FROM, so a setting the page
        #: writes can reach the caller's own deliverable and not only the copy.
        self._source: Optional[Path] = None
        #: Delivery dials the page has set, re-applied to each later publish.
        self._settings: Dict[str, Any] = {}
        self._updated: Optional[float] = None
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._port: Optional[int] = None
        self._viewer_seen: Optional[float] = None
        #: Active viewer scripts, registered name -> source file. Ordered, and
        #: the page imports them in this order.
        self._scripts: Dict[str, Path] = {}
        #: The page's clip recorder, allocated on first use, and a lock of its
        #: own for that one allocation -- see :meth:`playblast`.
        self._playblast: Optional["PreviewPlayblast"] = None
        self._playblast_lock = threading.Lock()
        #: Finished recordings, token -> file, so a download can follow.
        self._recordings: Dict[str, Path] = {}
        #: Held from choosing a still's number to writing it: two stills posted
        #: at once (a desktop tab and a headset) would otherwise both pick the
        #: same free number. Its own lock, so a write never holds up a poll.
        self._snapshot_lock = threading.Lock()
        #: The read-only listener a share fronts (see :meth:`start_guest`).
        self._guest_httpd: Optional[ThreadingHTTPServer] = None
        self._guest_thread: Optional[threading.Thread] = None
        #: Public names the guest listener answers to (:meth:`admit_host`).
        self._admitted: set = set()
        #: Guest tabs, page id -> last poll; oldest first (see _touch_guest).
        self._guests: Dict[str, float] = {}
        #: Pages that beaconed they closed, page id -> when (CLOSED_LINGER).
        self._closed: Dict[str, float] = {}
        #: The tunnel in front of the guest listener while sharing, and the
        #: public address of its alias, when there is one.
        self._tunnel = None
        self._alias_url: Optional[str] = None
        #: A share's tunnel while it still waits on its provider, so an
        #: unshare can end that wait (:meth:`ShareTunnel.cancel`).
        self._starting = None
        #: Held while a share sets up or installs its tunnel and while
        #: :meth:`unshare` runs -- never across a provider's wait -- and the
        #: count of unshares, so a share whose wait an unshare interrupted can
        #: tell (see :meth:`share`).
        self._share_lock = threading.Lock()
        self._share_generation = 0

        if root is None:
            # "session": a detached consumer (the browser) reads these while
            # this process lives, and there is no completion signal to delete
            # on -- so tie the lifetime to interpreter exit.
            self._temp = TempArtifacts("webxr_preview", policy="session")
            root = self._temp.dir_path()
        else:
            self._temp = None
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def port(self) -> Optional[int]:
        """The bound port, or ``None`` before :meth:`start`."""
        return self._port

    @property
    def url(self) -> Optional[str]:
        """The viewer URL, or ``None`` before :meth:`start`."""
        return None if self._port is None else f"http://{self.host}:{self._port}/"

    @property
    def version(self) -> int:
        """Number of published revisions; the viewer reloads when this changes."""
        return self._version

    @property
    def is_running(self) -> bool:
        return self._httpd is not None

    def has_viewer(self) -> bool:
        """Whether a page is currently watching this server.

        True while manifest polls keep arriving, and false once they stop for
        :attr:`VIEWER_TIMEOUT` seconds or the viewer beacons that it is
        unloading.

        This exists because "has anyone published yet" is *not* the same
        question, and using it as a stand-in is what made the preview feel
        broken: the server outlives every push, so once a tab had been opened
        and closed, nothing reopened it for the rest of the DCC session and
        each subsequent push landed on a page that no longer existed.
        """
        with self._lock:
            seen = self._viewer_seen
        return seen is not None and (time.time() - seen) <= self.VIEWER_TIMEOUT

    def _touch_viewer(self, viewer_id: str = "") -> None:
        """Record that a viewer is known to exist as of now -- unless *viewer_id*
        is a page that just said it closed (:attr:`CLOSED_LINGER`)."""
        with self._lock:
            if self._closed_recently(viewer_id):
                return
            self._viewer_seen = time.time()

    def _clear_viewer(self, viewer_id: str = "") -> None:
        """Record that no viewer is attached (it beaconed on unload)."""
        with self._lock:
            self._viewer_seen = None
            self._mark_closed(viewer_id)

    def _mark_closed(self, viewer_id: str) -> None:
        """Remember that page *viewer_id* closed; call under the lock."""
        if not _VIEWER_ID.fullmatch(viewer_id or ""):
            return
        now = time.time()
        for stale in [
            k for k, t in self._closed.items() if now - t >= self.CLOSED_LINGER
        ]:
            del self._closed[stale]
        self._closed[viewer_id] = now
        while len(self._closed) > self.MAX_GUESTS:  # ids are the pages' own
            self._closed.pop(next(iter(self._closed)))

    def _closed_recently(self, viewer_id: str) -> bool:
        """Whether *viewer_id* closed within :attr:`CLOSED_LINGER`; under the lock."""
        when = self._closed.get(viewer_id) if viewer_id else None
        return when is not None and time.time() - when < self.CLOSED_LINGER

    def manifest(self, guest: bool = False) -> Dict[str, Any]:
        """The payload served at ``/manifest.json``.

        Parameters:
            guest: The guest listener's version (see :meth:`start_guest`):
                marked ``"guest": true`` so the page hides what only the owner
                can do, without :attr:`OWNER_SCRIPTS`, and without
                ``xrRuntime`` -- a fact about THIS machine, which would only
                mislead a page open on someone else's.
        """
        # Read OUTSIDE the lock: a registry lookup has no business holding up a
        # publish, and this is not part of the published state.
        xr_runtime = None if guest else self._xr_runtime() is not None
        with self._lock:
            manifest = {
                "version": self._version,
                # Page fingerprint; the viewer reloads when it changes.
                "viewer": self._viewer_stamp,
                "asset": self._asset,
                "updated": self._updated,
                "title": self.title,
                # For guests too: where a view starts, and whether it can go
                # anywhere from there, are how the scene is meant to be seen.
                "userPos": self.user_pos,
                "locomotion": bool(self.locomotion),
                # URLs rather than names: the page imports these directly, and
                # the route is this module's business, not the viewer's.
                "scripts": [
                    f"{self.SCRIPTS_ROUTE}/{name}.js"
                    for name in self._scripts
                    if not (guest and name in self.OWNER_SCRIPTS)
                ],
            }
        if guest:
            manifest["guest"] = True
        else:
            # Whether an XR runtime is INSTALLED, which the page cannot see
            # and the machine can. It is the difference between "no headset
            # here" and "a headset that is not presenting", and the page
            # says something useful for each -- see the viewer's XR block.
            manifest["xrRuntime"] = xr_runtime
        return manifest

    def _resolve_port(self) -> int:
        """Return the port to bind: preferred if free, else an ephemeral one.

        ``None`` means "a stable port if you can get it" -- a preview tab stays
        valid across restarts only if the port does, which matters more here
        than anywhere else because reopening a browser inside a headset is
        tedious. An explicit port is taken literally so a caller pinning one for
        a tunnel gets a hard failure rather than a silent move.
        """
        requested = self._requested_port
        if requested is None:
            if NetUtils.is_port_bindable(self.DEFAULT_PORT, host=self.host):
                return self.DEFAULT_PORT
            self.logger.debug(
                "Port %s unavailable; falling back to an ephemeral port.",
                self.DEFAULT_PORT,
            )
            return 0
        return int(requested)

    def start(self) -> "PreviewServer":
        """Bind the port and serve on a daemon thread. Idempotent."""
        if self._httpd is not None:
            return self
        self._ensure_viewer()
        self._ensure_scripts()
        handler = partial(_PreviewHandler, directory=str(self.root), owner=self)
        try:
            self._httpd = _PreviewHTTPServer((self.host, self._resolve_port()), handler)
        except OSError:
            # The bindability probe in `_resolve_port` releases the port before
            # the real bind, so another process can take it in the gap. A
            # pinned port must still fail loudly, but ``port=None`` asked for
            # "stable if you can, ephemeral otherwise" -- honour that against
            # the race too, not just against a port that was already taken.
            if self._requested_port is not None:
                raise
            self._httpd = _PreviewHTTPServer((self.host, 0), handler)
        self._port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=partial(self._httpd.serve_forever, poll_interval=self.POLL_INTERVAL),
            name="ptk-preview-server",
            daemon=True,
        )
        self._thread.start()
        self.logger.info(
            "Preview server listening on %s (serving %s)", self.url, self.root
        )
        return self

    def stop(self) -> None:
        """Stop serving and release the port, ending any share. Idempotent."""
        # First, and ahead of the early return: a share cannot outlive the
        # server it fronts -- its tunnel would keep a public link alive in
        # front of a closed port, for whatever binds that port next.
        self.unshare()
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._httpd = None
        self._thread = None
        self._port = None
        # Any page that was watching is now watching a closed socket, and a
        # restart usually lands on a different ephemeral port -- so carrying
        # the flag over would suppress the tab a restarted server most needs.
        self._clear_viewer()
        self.logger.debug("Preview server stopped.")

    def publish(
        self,
        src: Union[str, Path],
        name: Optional[str] = None,
        move: bool = False,
    ) -> int:
        """Place an asset in the serve root and bump the manifest version.

        Parameters:
            src: The file to publish (typically a GLB).
            name: Served filename. Defaults to ``scene<ext>`` so republishing
                reuses one URL rather than growing the directory.
            move: Move instead of copy -- for a temp export with no other reader.

        Returns:
            The new version number.
        """
        src = Path(src)
        if not src.is_file():
            raise FileNotFoundError(f"PreviewServer.publish: no such file: {src}")
        # Re-sync the managed viewer here, not only in `start`: the server
        # outlives every individual push, so this is the only hook an edited
        # page has to reach a session that is already running.
        self._ensure_viewer()
        # Same reasoning for the scripts: the active set is per *push* (a
        # bridge can name one for a single delivery), so it has to be
        # materialized here, not only at start.
        self._ensure_scripts()
        name = name or f"scene{src.suffix}"
        # Before the copy, which may MOVE the source away.
        self._activate_auto_scripts(src)
        self._write_asset(src, self.root / name, move)
        with self._lock:
            settings = dict(self._settings)
        if settings:
            # A dial set in the page is a property of the DELIVERY, not of the
            # file that happened to be open when it was set, so a re-push
            # carries it instead of silently resetting to the file's defaults.
            # The SERVED copy only: re-stamping the source would edit a file
            # the caller just handed us and may still be writing.
            #
            # BEFORE the version bump, which is the page's signal to fetch:
            # advertising a version while the asset is still being rewritten
            # hands a poller a torn GLB.
            #
            # A failure here is logged, not raised: the push itself succeeded
            # and the model must still reach the page. Raising would leave the
            # asset written but never advertised, and report the PUSH as the
            # failure to whoever called -- a dial that could not be re-applied
            # is the lesser fact, and the page shows the file's own value.
            try:
                self._stamp(settings, [self.root / name])
            except (OSError, ValueError) as error:
                self.logger.warning(
                    "Published %s without re-applying %s: %s",
                    name,
                    ", ".join(settings),
                    error,
                )
        with self._lock:
            self._version += 1
            self._asset = name
            self._source = src
            self._updated = time.time()
            version = self._version
        self.logger.info("Published %s as %r (v%s)", src.name, name, version)
        return version

    def _activate_auto_scripts(self, src: Path) -> List[str]:
        """Activate each :attr:`AUTO_SCRIPTS` entry whose extras key *src* carries.

        Only a ``.glb`` is probed, and only its JSON chunk (the converter's
        lazy session never reads the geometry). Anything that is not a
        readable GLB -- a stub, a foreign container, a torn file -- simply
        activates nothing: the file is what the caller asked to serve, and
        this must never turn a publish into a failure.

        Returns:
            The names activated by this call (already-active ones are not
            repeated).
        """
        if not self.AUTO_SCRIPTS or src.suffix.lower() != ".glb":
            return []
        try:
            with _mesh_convert().open_glb(src) as edit:
                extras = edit.gltf.get("extras") or {}
        except Exception as error:  # noqa: BLE001 -- see the docstring
            self.logger.debug(
                "Auto scripts: %s is not a readable GLB (%s).", src.name, error
            )
            return []
        if not isinstance(extras, dict):
            return []
        with self._lock:
            active = set(self._scripts)
        activated = [
            name
            for name, key in self.AUTO_SCRIPTS.items()
            if key in extras and name not in active
        ]
        for name in activated:
            self.add_script(name)
        if activated:
            self.logger.info(
                "Activated viewer script(s) %s: %s carries extras %s.",
                ", ".join(activated),
                src.name,
                ", ".join(self.AUTO_SCRIPTS[name] for name in activated),
            )
        return activated

    def apply_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Write delivery dials into the published GLB, and remember them.

        The page tunes a value live against the loaded model; this is what
        makes that value part of the DELIVERABLE rather than of the session
        looking at it. Each dial is written into the glTF's own field (see
        :meth:`MeshConvert.set_glb_normal_scale`), so the file states it once,
        every runtime honours it, and the next load reads it back with nothing
        to re-apply.

        Both the served copy and -- when it still exists and is not the served
        copy itself -- the file it was published FROM are written, so previewing
        an exporter's GLB and saving from the page lands the value in the
        artifact the exporter produced. A moved temp export is neither, and is
        simply skipped.

        Parameters:
            settings: ``{name: value}``, names from :attr:`WRITABLE_SETTINGS`.

        Returns:
            ``{"applied": {name: coerced value}, "materials": {name: count}}``.

        Raises:
            KeyError: A name outside :attr:`WRITABLE_SETTINGS`.
            ValueError: A value that will not coerce, or a GLB the writer
                refuses.
            OSError: The write itself failed. The value is then NOT
                remembered -- see below.
        """
        unknown = sorted(set(settings) - set(self.WRITABLE_SETTINGS))
        if unknown:
            raise KeyError(f"no such setting: {', '.join(unknown)}")
        resolved = {
            name: self.WRITABLE_SETTINGS[name][1](value)
            for name, value in settings.items()
        }
        with self._lock:
            targets = self._setting_targets()
        # Written BEFORE it is remembered: a value that could not reach the file
        # must not ride along on the next publish either, which is what makes a
        # failed save simply a failed save rather than a dial silently set.
        counts = self._stamp(resolved, targets)
        with self._lock:
            self._settings.update(resolved)
        self.logger.info(
            "Wrote %s into %s.",
            ", ".join(f"{k}={v:g}" for k, v in resolved.items()),
            ", ".join(t.name for t in targets) or "nothing (no asset published)",
        )
        return {"applied": resolved, "materials": counts}

    def _setting_targets(self) -> List[Path]:
        """Files a setting write lands in: the served copy, then its source."""
        targets: List[Path] = []
        served = (self.root / self._asset) if self._asset else None
        if served is not None and served.is_file():
            targets.append(served)
        source = self._source
        if source is not None and source.is_file() and source != served:
            targets.append(source)
        return targets

    def _stamp(self, settings: Dict[str, Any], paths: Sequence[Path]) -> Dict[str, int]:
        """Apply *settings* to each of *paths*; materials changed, by name.

        The count reported is the FIRST path's -- the served copy, which is
        what the page is looking at. A later path is the same content and
        answers the same, except when it has already been stamped, where it
        rightly answers zero; reporting that would tell the page nothing
        changed when the model in front of it just did.
        """
        convert = _mesh_convert()
        counts: Dict[str, int] = {}
        for path in paths:
            for name, value in settings.items():
                writer = getattr(convert, self.WRITABLE_SETTINGS[name][0])
                counts.setdefault(name, writer(str(path), value))
        return counts

    #: Browsers known to ship Chromium's OpenXR backend, most preferred first.
    #: The list is short ON PURPOSE: it names the two builds verified to carry
    #: `openxr_loader`, rather than guessing from "is it Chromium?" -- Vivaldi,
    #: Opera, Brave and Qt's WebEngine are all Chromium and all measured
    #: WITHOUT it, so a family test would confidently pick a browser that can
    #: never show the VR button.
    WEBXR_BROWSERS = (
        r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
        r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe",
        r"%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe",
        r"%PROGRAMFILES(X86)%\Microsoft\Edge\Application\msedge.exe",
        r"%PROGRAMFILES%\Microsoft\Edge\Application\msedge.exe",
    )

    #: Where every OpenXR application -- Chromium's own backend included --
    #: looks for the installed runtime.
    _OPENXR_RUNTIME_KEY = r"SOFTWARE\Khronos\OpenXR\1"

    @classmethod
    def _xr_runtime(cls) -> Optional[str]:
        """Path to the installed OpenXR runtime manifest, or ``None``.

        Read from the same registry value Chromium reads, so it answers the
        one question worth asking before taking a tab off the user's default
        browser: could ANY browser enter an immersive session on this machine?
        """
        try:
            # Off Windows there is no `winreg` -- and no build that can enter
            # VR either. The 64-bit view is named explicitly, or a 32-bit
            # interpreter would be redirected to `WOW6432Node` and miss the
            # value the (64-bit) browser will actually read.
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                cls._OPENXR_RUNTIME_KEY,
                0,
                winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
            ) as key:
                manifest = winreg.QueryValueEx(key, "ActiveRuntime")[0]
        except (ImportError, OSError):  # No winreg, no key, or no value.
            return None
        # This sits in the path of every push and must never raise, and a
        # registry someone else owns is not a thing to trust the shape of.
        # The key also outlives an uninstall, and a manifest that is gone runs
        # nothing.
        if not isinstance(manifest, str) or not os.path.isfile(manifest):
            return None
        return manifest

    @classmethod
    def webxr_browser(cls) -> Optional[str]:
        """Which browser to open so an immersive session is POSSIBLE.

        ``None`` means no browser switch would help, and the caller should
        leave the user on the one they chose: either no OpenXR runtime is
        installed -- so nothing on this machine can enter VR and taking the
        tab would buy the user nothing -- or none of the capable builds is.
        """
        if not cls._xr_runtime():
            return None
        for candidate in cls.WEBXR_BROWSERS:
            # An unset variable leaves its `%NAME%` in place, which no more
            # names a file than a missing install does.
            path = os.path.expandvars(candidate)
            if os.path.isfile(path):
                return path
        return None

    def open_in_browser(self) -> bool:
        """Open the viewer. Starts the server if needed.

        A machine with an XR runtime gets a browser that can actually enter
        VR, because the default is frequently a Chromium fork WITHOUT the
        OpenXR backend: it loads the page perfectly and simply never grows a
        VR button, with nothing on screen to say why. A machine with no
        runtime keeps whatever browser the user chose -- see
        :meth:`webxr_browser`.
        """
        self.start()
        preferred = self.webxr_browser()
        opened = False
        if preferred:
            # A specific browser cannot go through `webbrowser.open`, which
            # only ever opens the system default.
            opened = webbrowser.BackgroundBrowser(preferred).open(self.url)
            if not opened:
                # The preference is an UPGRADE, never a new way to end up with
                # nothing: an exe that is present but will not start (a broken
                # or half-removed install) must not cost the user the browser
                # they would have got before this existed.
                self.logger.warning(
                    "%s would not launch; using the default browser, which may "
                    "not enter VR.",
                    preferred,
                )
        if not opened:
            opened = webbrowser.open(self.url)
        if opened:
            # Count the launch itself as a viewer, ahead of its first poll. A
            # cold browser start takes seconds, and without this a second push
            # arriving inside that gap sees no polls yet and opens a duplicate
            # tab. A real page then keeps the timestamp refreshed; one that
            # never appears simply lapses after VIEWER_TIMEOUT.
            self._touch_viewer()
        else:
            self.logger.warning(
                "No browser could be launched for %s - open it manually.", self.url
            )
        return opened

    def __enter__(self) -> "PreviewServer":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()
