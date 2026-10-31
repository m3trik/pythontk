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
    ...     server.publish_parts(["C:/tmp/room.glb", "C:/tmp/props.glb"])

Served surface:
    ``GET /``                -> the viewer page (materialized into the serve root)
    ``GET /manifest.json``   -> ``{"version", "viewer", "asset", "parts",
                                "updated", "title", "userPos", "locomotion",
                                "scripts", "xrRuntime"}``;
                                also the heartbeat behind :meth:`PreviewServer.has_viewer`
    ``GET /kernel/<module>.js`` -> the viewer itself: the modules the page imports
    ``GET /features/...``    -> an active packaged viewer script, and the modules
                                beside it (see :attr:`PreviewServer.SCRIPTS`)
    ``GET /scripts/<name>.js`` -> a caller's own active viewer script
                                (see :meth:`PreviewServer.add_script`)
    ``GET /<name>``          -> any published asset, by name
    ``POST /viewer-closed``  -> the viewer's unload beacon, so a closed tab is
                                known at once rather than after a timeout
    ``POST /settings``       -> a delivery dial the page writes into the GLB
    ``POST /playblast/*``    -> ``begin`` / ``frame`` / ``finish`` / ``cancel``:
                                the page recording a clip to a movie file
                                (see :mod:`pythontk.net_utils.preview.playblast`)
    ``POST /snapshot``       -> a still of the page's view, as a PNG body
                                (see :meth:`PreviewServer.save_snapshot`)
    ``GET /scene.json``      -> the published scene as data: its glTF JSON and
                                the lighting recipe it renders with, paged by
                                ``?section=&start=`` (see
                                :meth:`PreviewServer.describe_scene`)

The guest listener serves ``GET /`` and the kernel it imports,
``/manifest.json`` (marked ``"guest": true``, without the owner-only scripts),
``/scene.json`` and exactly the files that manifest names (each script with
the modules beside it); it takes the close beacon and refuses every other
write.

Layout: this module holds the :class:`PreviewServer` facade -- lifecycle, the
live manifest and viewer liveness, publish and its delivery dials, and the
browser it opens -- and every public method, its signature and its docstring.
Its other jobs are private collaborators beside it, each owning its own state:
``_serve_root`` (:class:`_ServeRoot`: the viewer page, the viewer scripts and
the atomic writes), ``_sharing`` (:class:`_PreviewShare`: the guest listener,
its admitted names and guests, the tunnel and the private segment -- the
share's whole security state), ``_page_outputs`` (:class:`_PageOutputs`:
recordings and stills) and ``_scene_description`` (:class:`_SceneDescription`:
the scene as data). Each holds the server and reads its tables and its
published state through it at call time, under the server's one state lock --
so a manifest is one instant's snapshot -- and calls its public methods through
it, so an override of one reaches every caller. The HTTP handler with the route
names is :mod:`.routes`, re-exported here.
"""

from __future__ import annotations

import os
import re
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

from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.temp_artifacts import TempArtifacts
from pythontk.net_utils._net_utils import NetUtils
from pythontk.net_utils.preview._page_outputs import _PageOutputs
from pythontk.net_utils.preview._scene_description import _SceneDescription
from pythontk.net_utils.preview._serve_root import _ServeRoot
from pythontk.net_utils.preview._sharing import _PreviewShare

# The route names are the handler's; re-exported here, where callers and the
# page's tests have always imported them from.
from pythontk.net_utils.preview.routes import (
    PLAYBLAST_ACTIONS as PLAYBLAST_ACTIONS,
    PLAYBLAST_PATH as PLAYBLAST_PATH,
    SCENE_PATH as SCENE_PATH,
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


class PreviewServer(LoggingMixin):
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

    # -- the serve root (_ServeRoot reads these through the server) ------------
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
        # Headless lighting work: a driver (LookCapture) swaps the bake and reads frames.
        "look_capture": "features/look_capture.js",
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
    #: The keys are the records' own, never spelled here: a GLB carries each
    #: record's block under its key.
    AUTO_SCRIPTS: Dict[str, str] = {
        "shadow_rig": SceneRecords.SHADOWS.key,
        # An articulated rig is a prop a hand is meant to move; without its
        # script the page shows it frozen at whatever the clip says, which
        # reads as "the rig did not export" rather than as a missing option.
        "articulated_rig": SceneRecords.ARTICULATION.key,
        # A deliverable that ships clips grows a picker and a transport in the
        # page; recording what that transport plays is the same kind of "not
        # optional in any useful sense" as the shadow rigs. There is no
        # checkbox for it because the alternative is worse: the button would be
        # missing on exactly the push a reviewer just watched and wants to
        # send on, and getting it would mean re-exporting the animation.
        "playblast": SceneRecords.SHOTS.key,
    }

    #: Viewer scripts that write through the owner's routes -- a recording, a
    #: still -- and are therefore withheld from a guest (:meth:`start_guest`).
    #: The guest listener refuses every write, so served to a guest each would
    #: be a button that fails; withheld, the guest never downloads it either. A
    #: script registered under another name is served to guests: an overlay
    #: that only reads needs nothing here.
    OWNER_SCRIPTS = frozenset({"playblast", "snapshot"})

    # -- sharing (_PreviewShare reads these through the server) ----------------
    #: Guest tabs a share tracks at once (:meth:`guest_count`); past it the one
    #: seen longest ago is forgotten. The ids are the guests' own choosing, so
    #: the table has to be bounded by this side.
    MAX_GUESTS = 256

    #: Environment variables a share reads when its caller names no alias: where
    #: to keep the stable redirect to the link (``ShareTunnel``'s *alias*: a
    #: path or ``user@host:/path``), and the public address that is served at
    #: -- the link to hand out instead of the tunnel's own. Machine config, so a
    #: web root's hostname never lands in a repo.
    ALIAS_ENV = "PYTHONTK_PREVIEW_ALIAS"
    ALIAS_URL_ENV = "PYTHONTK_PREVIEW_ALIAS_URL"

    #: Bytes of randomness in a private share's path segment: 96 bits, as
    #: 16 URL-safe characters -- unguessable, and short enough to read out.
    SHARE_TOKEN_BYTES = 12

    # -- page outputs (_PageOutputs reads these through the server) ------------
    #: Finished recordings kept addressable for download at once.
    MAX_RECORDINGS: int = 8

    #: Image types the page may post a still as: MIME -> (extension, the
    #: signature its bytes must open with). PNG only -- the page's still is
    #: lossless, and the table is the one place another type would be added.
    #: The signature is checked because this route writes a file under a name
    #: the page does not choose, and a body that is not the image it claims to
    #: be should be refused rather than saved as one.
    SNAPSHOT_TYPES: Dict[str, Tuple[str, bytes]] = {
        "image/png": (".png", b"\x89PNG\r\n\x1a\n"),
    }

    #: Refuse a still larger than this. A 4K PNG of a production scene is tens
    #: of megabytes at most; this is a guard against a runaway body, the same
    #: ceiling a recording's frame gets.
    MAX_SNAPSHOT_BYTES: int = 64 * 1024 * 1024

    # -- the scene as data (_SceneDescription reads these through the server) --
    #: Bytes of JSON the overview inlines, smallest section first, and the most
    #: one page of a section carries -- past one item, which goes whatever its
    #: size.
    SCENE_INLINE_BYTES = 128 * 1024

    #: The shape :meth:`describe_scene` answers in, for a reader that keys on it.
    SCENE_SCHEMA = "pythontk.preview.scene/1"

    #: What the overview tells a reader arriving with nothing else -- an agent
    #: handed the link.
    SCENE_ABOUT = (
        "How this preview's scene is built, as data. 'gltf' is the published "
        "asset's own glTF 2.0 JSON, verbatim, for each top-level section that "
        "fit; 'sections' indexes every section with its size, and its 'url' "
        "reads it a page at a time ('next' is the following page). "
        "'rendering' is the lighting recipe the viewer renders this asset with "
        "(three.js 'viewer.three'): tone mapping, environment, key light, and "
        "what a lightmapped material takes from each -- 'renderingSource' "
        "says whether the asset published it or the viewer's default applies. "
        "The asset itself downloads from 'asset'; the page is 'viewer.page'."
    )

    #: The three.js release the served page imports, as its import map pins it.
    _THREE_RELEASE = re.compile(r"three@(\d+\.\d+\.\d+)/build/")

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
        #: The one state lock: what is published and what each collaborator
        #: holds for a request to read, so a manifest is one instant's state.
        self._lock = threading.Lock()
        #: The jobs beside the facade's own, each owning its state (built
        #: first: nothing below, and no request, may find one missing).
        self._serve_root = _ServeRoot(self)
        self._share = _PreviewShare(self)
        self._page_outputs = _PageOutputs(self)
        self._scene = _SceneDescription(self)
        self._version = 0
        self._asset: Optional[str] = None
        #: The file the current asset was published FROM, so a setting the page
        #: writes can reach the caller's own deliverable and not only the copy.
        self._source: Optional[Path] = None
        #: Every part of the current publish, ``{"url", "name"}`` in order --
        #: the served file and the name the page offers it under -- the first
        #: being :attr:`_asset`; and the file each was published from.
        self._parts: List[Dict[str, str]] = []
        self._sources: List[Path] = []
        #: Delivery dials the page has set, re-applied to each later publish.
        self._settings: Dict[str, Any] = {}
        self._updated: Optional[float] = None
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._port: Optional[int] = None
        self._viewer_seen: Optional[float] = None
        #: Pages that beaconed they closed, page id -> when (CLOSED_LINGER):
        #: the owner's tab and a guest's alike.
        self._closed: Dict[str, float] = {}

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
        xr_runtime = None if guest else self._xr_runtime_name() or False
        with self._lock:
            manifest = {
                "version": self._version,
                # Page fingerprint; the viewer reloads when it changes.
                "viewer": self._serve_root.viewer_stamp,
                "asset": self._asset,
                # Every part, the asset first: the page loads them together.
                "parts": [dict(part) for part in self._parts],
                "updated": self._updated,
                "title": self.title,
                # For guests too: where a view starts, and whether it can go
                # anywhere from there, are how the scene is meant to be seen.
                "userPos": self.user_pos,
                "locomotion": bool(self.locomotion),
                # URLs rather than names: the page imports these directly, and
                # the route is this module's business, not the viewer's.
                "scripts": self._serve_root.script_urls(
                    exclude=self.OWNER_SCRIPTS if guest else ()
                ),
            }
        if guest:
            manifest["guest"] = True
        else:
            # Which XR runtime is ACTIVE (its name; false for none), which the
            # page cannot see and the machine can. It is the difference
            # between "no headset here" and "a headset that is not
            # presenting" -- and, as the name changes under an open page, "a
            # runtime this browser has not loaded" -- and the page says
            # something useful for each: see the viewer's XR block.
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
        self._serve_root.ensure_viewer()
        self._serve_root.ensure_scripts()
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
        # restart may land on a different port (an ephemeral one whenever
        # DEFAULT_PORT is taken) -- so carrying the flag over would suppress
        # the tab a restarted server may need.
        self._clear_viewer()
        self.logger.debug("Preview server stopped.")

    def publish(
        self,
        src: Union[str, Path],
        name: Optional[str] = None,
        move: bool = False,
        label: Optional[str] = None,
    ) -> int:
        """Place an asset in the serve root and bump the manifest version.

        Parameters:
            src: The file to publish (typically a GLB).
            name: Served filename. Defaults to ``scene<ext>`` so republishing
                reuses one URL rather than growing the directory.
            move: Move instead of copy -- for a temp export with no other reader.
            label: The name the page offers the file under (Export ▸ GLB
                saves it so). ``None`` keeps the served name: a scratch
                export's own is a random tag, never a name to hand out.

        Returns:
            The new version number.
        """
        src = Path(src)
        name = name or f"scene{src.suffix}"
        return self._publish([(src, name, label or name)], move)

    def publish_parts(
        self,
        srcs: Sequence[Union[str, Path]],
        labels: Optional[Sequence[str]] = None,
        move: bool = False,
    ) -> int:
        """Publish the PARTS of one scene together, as one version.

        Several files the page shows at once, each at its authored place --
        a room, the props in it, a character, each exported on its own -- so
        they line up the way they do in the scene they came from, and each
        can be hidden on the page (View) and saved (Export). The first is the
        primary: :meth:`describe_scene`, a still's and a recording's name,
        and a script reading the deliverable's extras by node index all read
        it. A single file is :meth:`publish`.

        Served as ``scene<ext>``, ``scene_1<ext>`` ... in order. A part a
        previous publish had beyond this one's count is removed once the
        version naming this set is up.

        Parameters:
            srcs: The files, primary first.
            labels: The name the page offers each under, in order; ``None``
                takes each file's own name. A repeated name is numbered
                (``model (2).glb``): the page tells parts apart by name.
            move: Move instead of copy -- for temp exports with no other reader.

        Returns:
            The new version number.

        Raises:
            ValueError: No file, or *labels* not one per file.
        """
        sources = [Path(src) for src in srcs]
        if not sources:
            raise ValueError("PreviewServer.publish_parts: no file to publish")
        if labels is None:
            labels = [source.name for source in sources]
        if len(labels) != len(sources):
            raise ValueError(
                f"PreviewServer.publish_parts: {len(labels)} labels for "
                f"{len(sources)} files"
            )
        labels = self._distinct(labels)
        entries = [
            (source, f"scene{'_' + str(index) if index else ''}{source.suffix}", label)
            for index, (source, label) in enumerate(zip(sources, labels))
        ]
        return self._publish(entries, move)

    @staticmethod
    def _distinct(labels: Sequence[str]) -> List[str]:
        """*labels* with each repeat numbered -- ``model.glb``, ``model
        (2).glb`` -- since the page names a part by its label: two parts
        called alike would be one switch in View, one remembered hidden name
        and one file name in a download."""
        seen: Dict[str, int] = {}
        distinct: List[str] = []
        for label in labels:
            stem, dot, suffix = str(label).rpartition(".")
            if not dot:
                stem, suffix = str(label), ""
            name, count = str(label), 1
            while name.lower() in seen:
                count += 1
                name = f"{stem} ({count}){dot}{suffix}"
            seen[name.lower()] = count
            distinct.append(name)
        return distinct

    def _publish(self, entries: List[Tuple[Path, str, str]], move: bool) -> int:
        """Write each ``(src, served name, label)`` and advertise them as one
        version; the version."""
        for src, _name, _label in entries:
            if not src.is_file():
                raise FileNotFoundError(f"PreviewServer.publish: no such file: {src}")
        # Re-sync the managed viewer here, not only in `start`: the server
        # outlives every individual push, so this is the only hook an edited
        # page has to reach a session that is already running.
        self._serve_root.ensure_viewer()
        # Same reasoning for the scripts: the active set is per *push* (a
        # bridge can name one for a single delivery), so it has to be
        # materialized here, not only at start.
        self._serve_root.ensure_scripts()
        with self._lock:
            settings = dict(self._settings)
        for src, name, _label in entries:
            # Before the copy, which may MOVE the source away.
            self._activate_auto_scripts(src)
            self._serve_root.write_asset(src, self.root / name, move)
            if settings:
                # A dial set in the page is a property of the DELIVERY, not of
                # the file that happened to be open when it was set, so a
                # re-push carries it instead of silently resetting to the
                # file's defaults. The SERVED copy only: re-stamping the source
                # would edit a file the caller just handed us and may still be
                # writing.
                #
                # BEFORE the version bump, which is the page's signal to fetch:
                # advertising a version while the asset is still being
                # rewritten hands a poller a torn GLB.
                #
                # A failure here is logged, not raised: the push itself
                # succeeded and the model must still reach the page. Raising
                # would leave the asset written but never advertised, and
                # report the PUSH as the failure to whoever called -- a dial
                # that could not be re-applied is the lesser fact, and the page
                # shows the file's own value.
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
            dropped = [
                part["url"]
                for part in self._parts[1:]
                if part["url"] not in {name for _src, name, _label in entries}
            ]
            self._version += 1
            self._asset = entries[0][1]
            self._source = entries[0][0]
            self._parts = [
                {"url": name, "name": label} for _src, name, label in entries
            ]
            self._sources = [src for src, _name, _label in entries]
            self._updated = time.time()
            version = self._version
        # After the version that stops naming them: a page polls the new set
        # before it asks for any of its files.
        for name in dropped:
            try:
                (self.root / name).unlink()
            except OSError:
                pass  # held open by a reader that finishes with it; harmless
        for src, name, _label in entries:
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
        active = set(self.scripts)
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
        """Files a setting write lands in: each part's served copy, then the
        file it was published from."""
        targets: List[Path] = []
        for part, source in zip(self._parts, self._sources):
            served = self.root / part["url"]
            if served.is_file():
                targets.append(served)
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
    def _xr_runtime_name(cls) -> Optional[str]:
        """The active OpenXR runtime's name, or ``None`` when there is none.

        A name rather than a yes/no, because Chromium loads a runtime ONCE per
        XR service and never re-reads the registry: a runtime made active
        later -- Virtual Desktop's, as it connects -- is invisible to that
        browser until it restarts, and only a name lets the page see the
        change under it. The manifest's own ``runtime.name``, else the file's
        stem: another vendor's file is read for a label, never trusted for a
        shape.
        """
        manifest = cls._xr_runtime()
        if manifest is None:
            return None
        # utf-8-sig: a vendor's file written with a BOM is still JSON.
        document = FileUtils.read_json(manifest, encoding="utf-8-sig")
        try:
            name = document["runtime"]["name"]
        except (KeyError, TypeError):  # Unread, or not a runtime manifest's shape.
            name = None
        if isinstance(name, str) and name.strip():
            return name.strip()
        return Path(manifest).stem

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

    # ------------------------------------------------------------------
    # Viewer scripts (_ServeRoot)
    # ------------------------------------------------------------------
    @property
    def scripts(self) -> tuple:
        """Registered names of the viewer scripts currently active, in load order."""
        return self._serve_root.names()

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
        self._serve_root.add(name, path)
        return self

    def remove_script(self, name: str) -> "PreviewServer":
        """Deactivate a viewer script (unknown names are ignored), and sweep it.

        Materialized at once, for the reason on :meth:`add_script`.
        """
        self._serve_root.remove(name)
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
        self._serve_root.replace(scripts)
        return self

    # ------------------------------------------------------------------
    # Sharing (_PreviewShare)
    # ------------------------------------------------------------------
    @property
    def guest_port(self) -> Optional[int]:
        """The guest listener's port, or ``None`` while it is closed."""
        return self._share.guest_port

    @property
    def guest_url(self) -> Optional[str]:
        """The guest listener on loopback: what a guest sees, from this machine
        -- under a private share's path segment, the only place it answers."""
        return self._share.guest_url

    def start_guest(self, port: int = 0) -> int:
        """Open the read-only listener a share points at; its port. Idempotent.

        A second door onto the same serve root, and the only one a share ever
        fronts. Its role is fixed by the socket rather than read from a
        request, so nothing a guest sends can make it the owner's:

        * reads are an allow-list -- the page, its manifest, and exactly what
          that names (the asset, the scripts outside :attr:`OWNER_SCRIPTS`),
          plus :meth:`describe_scene`, that same asset's JSON read for them;
        * every write is refused, and a guest's close beacon retires only
          that guest's tab;
        * its polls count toward :meth:`guest_count`, never toward
          :meth:`has_viewer` -- a guest watching must not stop the next push
          from reopening the owner's closed tab;
        * it answers the loopback spellings plus the names :meth:`admit_host`
          lets in.

        Parameters:
            port: ``0`` (the default) takes an ephemeral port: a guest arrives
                through a tunnel, so the number is never seen. Pin one for a
                transport configured ahead of time (a reverse proxy of your
                own).

        Returns:
            The bound port.
        """
        return self._share.start_guest(port)

    def stop_guest(self) -> None:
        """Close the read-only listener, forgetting its admitted names and its
        guests. Idempotent."""
        self._share.stop_guest()

    def admit_host(self, netloc: str) -> None:
        """Let the guest listener answer requests addressed to *netloc*.

        *netloc* is the public name a proxy forwards (``name``, or
        ``name:port`` off the default port -- as a ``Host`` header spells it).
        The Host check is the guest listener's DNS-rebinding defence too, so a
        proxy's name is named rather than the check dropped. :meth:`share`
        admits its tunnel's name itself; this is for a transport of your own.
        The owner's listener never admits one.
        """
        self._share.admit_host(netloc)

    def guest_count(self) -> int:
        """How many guest tabs are watching: polled within :attr:`VIEWER_TIMEOUT`."""
        return self._share.guest_count()

    def share(
        self,
        provider: Optional[str] = None,
        alias: Union[
            str, os.PathLike, Callable[[Optional[str]], Any], bool, None
        ] = None,
        alias_url: Optional[str] = None,
        timeout: Optional[float] = None,
        on_step: Optional[Callable[[Any], Any]] = None,
        private: bool = False,
    ) -> Dict[str, Any]:
        """Give this preview a link anyone can open: view-only, and live.

        Every later publish reaches the guests too -- the share is a second,
        read-only door onto the same serve root (:meth:`start_guest`), fronted
        by a :class:`pythontk.ShareTunnel`. The owner's page, its writes and its
        tab are untouched. Starts serving if nothing is yet; a share already
        running on the same provider and alias is returned as it stands.

        Each guest's browser downloads the deliverable and renders it on its
        own device -- a standalone headset included, since the link is HTTPS --
        so this machine serves files and renders nothing for anyone. The link
        is therefore the deliverable itself: a guest can save the GLB.

        Parameters:
            provider: A :attr:`ShareTunnel.PROVIDERS` name; ``None`` or
                ``"auto"`` takes this machine's default
                (:meth:`ShareTunnel.resolve_provider`).
            alias: Where to keep a stable redirect to the link -- a path,
                ``user@host:/path`` or a callable (see :class:`ShareTunnel`).
                ``None`` reads :attr:`ALIAS_ENV`; ``False`` keeps none.
            alias_url: The public address the alias is served at, handed out
                while the alias is current. ``None`` reads :attr:`ALIAS_URL_ENV`.
            timeout: Seconds to wait for the provider's link; ``None`` keeps
                :class:`ShareTunnel`'s default.
            on_step: Handed the :class:`ShareTunnel.StepRequired` of a one-time
                step the provider stops on (Tailscale before the tailnet
                enables Funnel), on the thread running this call -- which then
                waits for the user to take it and returns the link. ``None``:
                such a share fails at once as that ``StepRequired``.
            private: Add a fresh random path segment to the link
                (``https://<name>/<token>/``), answering nothing outside it:
                each share's link is its own, dead once the share ends, and
                the provider's name alone -- Tailscale's is the machine's,
                for good -- opens nothing. Keeps no alias: a stable redirect
                would be a lasting door to every private link. A share
                running with the other choice is replaced.

        Returns:
            :meth:`share_info`.

        Raises:
            FileNotFoundError: The provider's CLI is not installed; the message
                names the install (:meth:`ShareTunnel.settle` offers it).
            ShareTunnel.StepRequired: The provider waits on a one-time step
                and there is no *on_step* -- or it was not taken in time.
            RuntimeError: The provider exited without a link; the message
                carries what it printed. Also when :meth:`unshare` or
                :meth:`stop` ran while the provider was starting: its tunnel
                is stopped rather than kept, and a start still waiting --
                on a step above all -- ends at once.
            TimeoutError: The provider printed no link in time.
        """
        return self._share.share(
            provider=provider,
            alias=alias,
            alias_url=alias_url,
            timeout=timeout,
            on_step=on_step,
            private=private,
        )

    def unshare(self) -> None:
        """Stop sharing: the tunnel stops (an alias then says the share ended)
        and the guest listener closes. The owner's page is untouched.
        Idempotent; a share still waiting on its provider stops as its link
        arrives."""
        self._share.unshare()

    def share_info(self) -> Optional[Dict[str, Any]]:
        """What is being shared, or ``None`` -- also once the provider exited
        on its own, which is how a dropped share shows.

        Returns:
            ``{"url", "tunnel_url", "scene_url", "provider", "label",
            "public", "private", "guests", "guest_url", "alias_error"}``.
            ``url`` is the link to hand out: the alias's address when there
            is an alias AND its last update landed, else the tunnel's own --
            under a private share's segment -- since a stale alias must never
            be what gets sent. ``tunnel_url`` is the link the tunnel serves,
            the segment included. ``scene_url`` is where the share serves
            :meth:`describe_scene` -- the link for someone's tools or agent --
            always on the tunnel, since an alias is one redirect page rather
            than a site.
        """
        return self._share.share_info()

    def ping_share(
        self, timeout: float = 10.0, attempts: int = 3, wait: float = 2.0
    ) -> Dict[str, Any]:
        """Ask the share's link for its manifest the way a guest's browser
        does: whether someone ELSE can reach the share now.

        End to end -- public DNS, the provider's TLS, the tunnel, the guest
        listener -- and it is this share that answers: the reply must be the
        guest manifest at this server's version, so a provider's error page
        or a stale tunnel answering 200 does not pass. A ping is no guest: it
        sends no page id, so :meth:`guest_count` never counts it. The tunnel's
        own link is asked, never an alias, which is a page on another server.

        A guest's route, not this machine's: for a public link the name is
        resolved by public DNS (:meth:`NetUtils.public_addresses`) and the
        request goes to that address -- this machine's resolver can send it
        elsewhere (Tailscale's MagicDNS answers a ``ts.net`` name with the
        tailnet address, which proves nothing about the public ingress). A
        name public DNS does not have fails the ping: no stranger can open
        it. The certificate is checked with the trust a browser has
        (:meth:`NetUtils.tls_context`); one this machine still cannot verify
        is asked again unverified, and the share answering then passes with
        ``verified`` False -- the provider's certificate is a browser's to
        judge, and a root missing here is no guest's problem.

        A brand-new name can take seconds to reach every resolver, so a miss
        is asked again, *attempts* times in all, *wait* seconds apart.

        Parameters:
            timeout: Seconds one request may take.
            attempts: Requests at most; the first that passes ends it.
            wait: Seconds between attempts.

        Returns:
            ``{"ok", "url", "status", "ms", "attempts", "error", "route",
            "verified"}`` -- ``ms`` the round trip of the request that
            answered (``None`` without one), ``status`` its HTTP status,
            ``error`` why it did not pass (``None`` when it did), ``route``
            ``"public"`` (public DNS, as a stranger resolves it) or
            ``"local"`` (this machine's resolver: a tailnet-only link, plain
            HTTP, or no public resolver could be asked), ``verified`` whether
            the certificate was verified (``None`` for plain HTTP). Never
            raises: nothing shared is a failed ping, said in ``error``.
        """
        return self._share.ping_share(timeout=timeout, attempts=attempts, wait=wait)

    @property
    def share_url(self) -> Optional[str]:
        """The link to hand out while sharing, else ``None``."""
        info = self.share_info()
        return info["url"] if info else None

    # ------------------------------------------------------------------
    # Page outputs (_PageOutputs)
    # ------------------------------------------------------------------
    @property
    def playblast(self) -> "PreviewPlayblast":
        """The page's recorder, created on first use.

        Deferred rather than built in ``__init__`` because it allocates a
        scratch store, and the overwhelming majority of preview servers never
        record anything.
        """
        return self._page_outputs.playblast()

    def begin_playblast(self, **kwargs: Any) -> Dict[str, Any]:
        """Open a recording (see :meth:`PreviewPlayblast.begin`)."""
        return self.playblast.begin(**kwargs)

    def finish_playblast(
        self, token: str, target: Optional[str] = None
    ) -> Dict[str, Any]:
        """Encode a recording and report where it went.

        The movie lands beside the file that was published, when that file is
        still on disk -- an exporter's GLB, or one chosen with External GLB.
        A scene push has no such file (its GLB is the bridge's own scratch,
        released the moment it is published), so the movie goes to the serve
        root instead and the page's ``url`` is how it is collected.

        Returns:
            :meth:`PreviewPlayblast.finish`'s report plus ``"url"`` -- always
            present, and always the way to fetch the file from the page.
        """
        return self._page_outputs.finish_playblast(token, target)

    def recording_path(self, token: str) -> Optional[Path]:
        """The file a finished recording produced, or None."""
        return self._page_outputs.recording_path(token)

    def save_snapshot(
        self, data: bytes, content_type: str = "image/png"
    ) -> Dict[str, Any]:
        """Write a still of the page's view, and report where it went.

        The page renders its current view and posts the image; this decides
        where it lands, by the rule a recording follows (see
        :meth:`finish_playblast`): beside the published file when that file is
        still on disk, else in the serve root, from which the returned ``url``
        fetches it -- the only copy a scene push's still has.

        Numbered, never overwritten, as ``<deliverable>_view_<NNN>.png``: a
        still is taken again from another angle, and the second must not
        replace the first. The name is composed here and not accepted from the
        page, for the reason a recording's is.

        Parameters:
            data: The encoded image.
            content_type: Its MIME type, one of :attr:`SNAPSHOT_TYPES`.

        Returns:
            ``{"output", "name", "url", "in_serve_root", "bytes"}``. ``url`` is
            the still's served path RELATIVE to the page, or None when it
            landed beside the deliverable, off the serve root -- where it is
            already where its owner keeps it. Relative because the page is the
            one that follows it: a browser honours ``download`` only on the
            page's own origin, and the page is as validly open at
            ``localhost`` as at the ``127.0.0.1`` this server's :attr:`url`
            spells -- an absolute link navigated a ``localhost`` tab to the
            bare PNG instead of saving it.

        Raises:
            ValueError: An unsupported type, an empty or oversized body, or
                bytes that are not the image they claim to be.
            OSError: The write itself failed.
        """
        return self._page_outputs.save_snapshot(data, content_type)

    # ------------------------------------------------------------------
    # The scene as data (_SceneDescription)
    # ------------------------------------------------------------------
    @property
    def scene_url(self) -> Optional[str]:
        """Where :meth:`describe_scene` is served on this machine, or ``None``
        before :meth:`start`. A share's own is ``share_info()["scene_url"]``."""
        return self._scene.scene_url()

    def describe_scene(
        self, section: Optional[str] = None, start: int = 0
    ) -> Dict[str, Any]:
        """The published scene as data: the payload served at ``/scene.json``.

        Parameters:
            section: ``None`` for the overview. Else a top-level glTF key
                (``"nodes"``, ``"materials"`` ...): one page of it, from
                *start*, within :attr:`SCENE_INLINE_BYTES` -- ``next`` names
                the following page, ``None`` on the last. A key that is not a
                list comes back whole, as ``value``.
            start: The first item of the page.

        Returns:
            Always ``schema``, ``asset`` (the served name, or ``None`` before a
            publish) and ``version`` -- a page read across a publish shows it
            -- and ``parts`` (``[{"url", "name"}]``) when the scene was
            published as several (:meth:`publish_parts`): ``asset`` is the
            first, the one described.
            The overview adds ``about``, ``title``, ``updated``, ``viewer``
            (``{"page", "three"}``, or ``None`` when no page is served),
            ``rendering`` and ``renderingSource`` (``"asset"`` or
            ``"viewer default"``), ``gltf`` (the inlined sections) and
            ``sections`` (``key -> {"items", "bytes", "inline", "url"}``;
            ``items`` is ``None`` for a key that is not a list). Before a
            publish, or for an asset that is not a GLB, a ``note`` says why
            there is nothing more.

        Raises:
            KeyError: *section* is not a key of the asset's JSON.
            ValueError: The asset is not a readable GLB, or *start* is negative.
            OSError: The asset could not be read.
        """
        return self._scene.describe(section, start)

    # ------------------------------------------------------------------
    # Private names other code reads, forwarded to their owner
    # ------------------------------------------------------------------
    # The request handler (routes.py) asks the share about a guest's request.
    def _guest_route(self, path: str) -> Optional[str]:
        return self._share.guest_route(path)

    def _guest_files(self) -> set:
        """Served paths a guest may fetch besides the page: the kernel modules
        it imports, and what its manifest names -- each script with the
        modules it imports beside it, and every part of the asset. Composed
        here, from the serve root and what is published."""
        serve_root = self._serve_root
        files = set(serve_root.kernel_files()) if self._viewer else set()
        with self._lock:
            files.update(serve_root.script_files(exclude=self.OWNER_SCRIPTS))
            if self._asset:
                files.add(self._asset)
            files.update(part["url"] for part in self._parts)
        return files

    def _admitted_hosts(self) -> tuple:
        return self._share.admitted_hosts()

    def _touch_guest(self, viewer_id: str) -> None:
        self._share.touch_guest(viewer_id)

    def _drop_guest(self, viewer_id: str) -> None:
        self._share.drop_guest(viewer_id)

    @property
    def _tunnel(self):
        """The share's running tunnel, or ``None``."""
        return self._share.tunnel

    @property
    def _guests(self) -> Dict[str, float]:
        """The share's guest tabs, page id -> last poll, oldest first."""
        return self._share.guests

    @property
    def _viewer_stamp(self) -> str:
        """The served page's fingerprint (``manifest()["viewer"]``)."""
        return self._serve_root.viewer_stamp

    @_viewer_stamp.setter
    def _viewer_stamp(self, value: str) -> None:
        self._serve_root.viewer_stamp = value

    # A ping's request and its empty result, read through the class so a
    # replacement of either (a test's, a caller's) reaches ping_share.
    _ask_share = staticmethod(_PreviewShare.ask_share)
    _failed_ping = staticmethod(_PreviewShare.failed_ping)

    def __enter__(self) -> "PreviewServer":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()
