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

Serving the same page to a *standalone* headset browsing over the LAN is the
only case that needs more, and it needs real HTTPS -- a plain LAN IP is not a
secure context and silently yields no VR button at all. That is deliberately
out of scope here: ``host`` is the single seam, so fronting the port with a
tunnel that terminates TLS is a configuration change rather than a redesign.

Example (publish a GLB and open it):
    >>> with PreviewServer(title="Selection") as server:
    ...     server.publish("C:/tmp/scene.glb")
    ...     server.open_in_browser()
    ...     server.publish("C:/tmp/scene_v2.glb")   # the open page swaps to it

Served surface:
    ``GET /``                -> the viewer page (materialized into the serve root)
    ``GET /manifest.json``   -> ``{"version", "asset", "updated", "title"}``;
                                also the heartbeat behind :meth:`PreviewServer.has_viewer`
    ``GET /<name>``          -> any published asset, by name
    ``POST /viewer-closed``  -> the viewer's unload beacon, so a closed tab is
                                known at once rather than after a timeout
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pythontk.core_utils.app_handoff import (
    Deliverer,
    HandoffBridge,
    HandoffRequest,
    Payload,
)
from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.file_utils.temp_artifacts import TempArtifacts
from pythontk.net_utils._net_utils import NetUtils


#: Path the viewer beacons on unload, so a closed tab is known immediately
#: rather than after :attr:`PreviewServer.VIEWER_TIMEOUT`. Shared by the
#: handler and the served page (which is checked against it by test).
VIEWER_CLOSED_PATH = "viewer-closed"


class _PreviewHTTPServer(ThreadingHTTPServer):
    """Threading server that refuses to bind a port another listener holds.

    ``HTTPServer`` sets ``allow_reuse_address``, whose meaning is not portable.
    On POSIX it reuses a ``TIME_WAIT`` port -- exactly what you want when
    restarting a DCC-hosted server on a pinned port. On Windows ``SO_REUSEADDR``
    additionally permits binding *over a live listener*: two servers both start,
    and requests are delivered to whichever socket the stack picks. For a
    preview server that means silently pushing to a viewer owned by a stale
    process, so the flag is dropped there and ``bind()`` fails loudly instead.
    """

    daemon_threads = True
    allow_reuse_address = os.name != "nt"


class _PreviewHandler(SimpleHTTPRequestHandler):
    """Static handler with a live ``/manifest.json`` and caching disabled.

    It also owns both halves of the viewer-liveness signal the ``"auto"``
    open-a-tab decision reads: each manifest poll marks a viewer present, and
    the page's unload beacon on ``POST /viewer-closed`` marks it gone.

    Caching is off for the whole tree rather than just the manifest: every file
    under the serve root is republished in place, so a cached response is always
    the stale one. The viewer additionally appends ``?v=<version>`` to the asset
    URL, which alone would be enough for well-behaved caches -- this is the
    belt-and-braces half, and costs nothing on loopback.
    """

    server_version = "pythontk-preview"

    def __init__(self, *args, owner: "PreviewServer" = None, **kwargs):
        self._owner = owner
        super().__init__(*args, **kwargs)

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path.split("?", 1)[0] == "/manifest.json":
            # Only the manifest counts as proof of life: it is fetched on a
            # timer for as long as a page is open, whereas an asset GET happens
            # once per publish and a stray favicon request proves nothing.
            if self._owner is not None:
                self._owner._touch_viewer()
            self._send_json(self._owner.manifest() if self._owner else {})
            return
        super().do_GET()

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        """Accept the viewer's ``sendBeacon`` close notice; 404 anything else."""
        if self.path.split("?", 1)[0] != f"/{VIEWER_CLOSED_PATH}":
            self.send_error(404)
            return
        # Drain the body: sendBeacon always sends one, and leaving it in the
        # socket desynchronises the next request on a keep-alive connection.
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        # This is the one request that *changes* server state, and a beacon is
        # a CORS-simple request -- no preflight -- so without this any page the
        # user happens to be browsing could tell us the viewer had closed and
        # make the next push pop a tab over the DCC. A cross-origin POST always
        # carries its own Origin; a missing one (curl, a test) is allowed, as
        # nothing off-machine can reach a loopback bind unaided.
        #
        # Compared against the request's *own* Host header rather than against
        # the server's `url`: that URL is always spelled 127.0.0.1, while the
        # page is just as validly reached at http://localhost:<port> (the whole
        # secure-context guarantee this module is built on names it that way).
        # Matching on the server's spelling would 403 the real viewer's beacon
        # whenever it was opened by name -- and silently, since the fallback is
        # simply waiting out VIEWER_TIMEOUT.
        origin, host = self.headers.get("Origin"), self.headers.get("Host")
        if origin and host and origin.split("//", 1)[-1] != host:
            self.send_error(403, "Cross-origin post rejected")
            return
        if self._owner is not None:
            self._owner._clear_viewer()
        self.send_response(204)
        self.end_headers()

    def _send_json(self, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format, *args):  # noqa: A002 (BaseHTTPRequestHandler API)
        """Route request logging to the owner's logger instead of stderr.

        The default implementation writes straight to ``sys.stderr``, which
        inside a DCC means every poll -- one a second, forever -- lands in the
        script editor.
        """
        if self._owner is not None:
            self._owner.logger.debug("%s %s", self.address_string(), format % args)


class _PreviewServerInternal:
    """Helpers for :class:`PreviewServer` (port choice, viewer + asset writes)."""

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
        dst = self.root / "index.html"
        if dst.exists() and self._temp is None:
            return  # caller's working copy; never clobber it
        src = Path(__file__).with_name("preview_viewer.html")
        if not src.is_file():  # pragma: no cover - packaging failure
            self.logger.warning("Viewer page missing from the package: %s", src)
            return
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            return  # already current
        shutil.copyfile(src, dst)

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
        os.replace(part, dst)


class PreviewServer(LoggingMixin, _PreviewServerInternal):
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
    """

    DEFAULT_PORT = 8118

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

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        viewer: bool = True,
        title: str = "Preview",
    ):
        self.host = host
        self.title = title
        self._requested_port = port
        self._viewer = viewer
        self._lock = threading.Lock()
        self._version = 0
        self._asset: Optional[str] = None
        self._updated: Optional[float] = None
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._port: Optional[int] = None
        self._viewer_seen: Optional[float] = None

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

    def _touch_viewer(self) -> None:
        """Record that a viewer is known to exist as of now."""
        with self._lock:
            self._viewer_seen = time.time()

    def _clear_viewer(self) -> None:
        """Record that no viewer is attached (it beaconed on unload)."""
        with self._lock:
            self._viewer_seen = None

    def manifest(self) -> Dict[str, Any]:
        """The payload served at ``/manifest.json``."""
        with self._lock:
            return {
                "version": self._version,
                "asset": self._asset,
                "updated": self._updated,
                "title": self.title,
            }

    def start(self) -> "PreviewServer":
        """Bind the port and serve on a daemon thread. Idempotent."""
        if self._httpd is not None:
            return self
        self._ensure_viewer()
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
            target=self._httpd.serve_forever,
            name="ptk-preview-server",
            daemon=True,
        )
        self._thread.start()
        self.logger.info("Preview server listening on %s (serving %s)", self.url, self.root)
        return self

    def stop(self) -> None:
        """Stop serving and release the port. Idempotent."""
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
        name = name or f"scene{src.suffix}"
        self._write_asset(src, self.root / name, move)
        with self._lock:
            self._version += 1
            self._asset = name
            self._updated = time.time()
            version = self._version
        self.logger.info("Published %s as %r (v%s)", src.name, name, version)
        return version

    def open_in_browser(self) -> bool:
        """Open the viewer in the default browser. Starts the server if needed."""
        self.start()
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


class PreviewDeliverer(Deliverer):
    """Hand-off strategy: convert the produced FBX to GLB and publish it.

    The delivery half of a preview bridge, so a DCC only has to supply the two
    hooks it already has (``_resolve_objects`` / ``_produce``, both provided by
    the Maya and Blender export mixins) to gain a live preview:

        >>> class WebXrPreview(MayaExportMixin, ptk.HandoffBridge):
        ...     deliverer = ptk.PreviewDeliverer(title="Maya")

    Unlike the script-launch deliverers this sits beside, there is no
    application to discover or launch and no script to render: the "target app"
    is a browser the user already has open, so delivery is a format conversion
    plus a version bump. Keeping it a :class:`Deliverer` rather than a bespoke
    per-DCC chain is what stops the FBX -> GLB -> publish sequence from being
    written twice, once in each engine.

    The server is created on first delivery and reused, because the whole point
    is that a page left open in a headset keeps receiving pushes.

    Parameters:
        server: An existing :class:`PreviewServer`; one is created on demand
            otherwise.
        open_browser: ``"auto"`` (default) opens a tab only when no page is
            currently watching the server -- so the first push opens one, a
            push while the page is still open does not (it picks the new
            version up itself, and a fresh tab every time would both pile up
            and steal focus from the DCC), and a push after the tab was closed
            opens one again. ``True`` always opens, ``False`` never does.
        title: Label shown in the viewer, when creating the server.
    """

    #: Sidecar section -> the :class:`MeshConvert` method that applies it, in
    #: application order. Held as names so the converter module (which pulls in
    #: the managed-binary installer) stays unimported until delivery. This is
    #: the extension point the sidecar design promises: a new kind of extended
    #: setup is one more section in the bridge plus one more row here -- no
    #: method edits. Both current rows exist because FBX loses the channel for
    #: every shader that is not the host's own legacy model (measured on Maya
    #: 2025: an aiStandardSurface arrives with no emissive at all and a flat
    #: white base colour). Base colour runs first only for tidiness; the two
    #: touch disjoint fields.
    SIDECAR_APPLIERS: Dict[str, str] = {
        "base_color": "set_glb_base_color",
        "emissive": "set_glb_emissive",
    }

    def __init__(
        self,
        server: Optional[PreviewServer] = None,
        open_browser: Union[bool, str] = "auto",
        title: str = "Preview",
    ):
        self.server = server
        self.open_browser = open_browser
        self.title = title

    def ensure_server(self) -> PreviewServer:
        """The bridge's server, started, creating it on first use."""
        if self.server is None:
            self.server = PreviewServer(title=self.title)
        return self.server.start()

    def _apply_sidecar(
        self, bridge, glb: str, sidecar: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Apply each section of the scene sidecar to the converted GLB.

        The sidecar carries scene state the FBX cannot express, section by
        section. Most *material* data does survive the round trip, so this is
        not a material channel -- it is the general seam for everything that
        does not, and a new kind of extended setup (lights, environment, custom
        attributes) plugs in as one more section plus one more applier rather
        than as another parallel mechanism.

        A section the bridge did not emit is simply absent, and a failure is
        logged rather than raised: a preview missing one section still beats no
        preview at all.

        Every section shares **one** open GLB. Each applier can open a path for
        itself, and did -- which meant a two-section sidecar parsed and rewrote
        the whole file twice, on top of the pass the conversion itself makes.
        They now all edit one in-memory glTF that is written once when the block
        closes, so the cost stops scaling with the number of sections.

        Returns a ``{section: outcome}`` summary that rides back on the
        deliverer's result so the calling panel can *say* what happened.
        Silence was the original sin here: a section that read nothing, matched
        nothing, or failed all looked identical from the UI -- indistinguishable
        from the channel simply not being supported.
        """
        if not sidecar:
            return {}

        # Deferred for the same reason as in `deliver`: the converter module
        # pulls in the managed-binary installer, which no other user needs.
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        summary: Dict[str, str] = {}
        try:
            with MeshConvert.open_glb(glb) as target:
                for section, method in self.SIDECAR_APPLIERS.items():
                    apply = getattr(MeshConvert, method)
                    data = sidecar.get(section)
                    if not data:
                        continue
                    try:
                        applied = apply(target, data)
                    except (OSError, ValueError) as error:
                        # Still per-section: one broken section must not cost
                        # the whole preview. The sections do share one glTF now,
                        # so an applier that raised *after* mutating would leave
                        # that much behind -- which is why every failure an
                        # applier can actually reach (an unreadable texture, an
                        # image no decoder wants) is handled inside it and
                        # reported as a skipped image instead. This stays as the
                        # backstop for anything that is not.
                        bridge.logger.warning(
                            "Sidecar %r not applied: %s", section, error
                        )
                        summary[section] = f"failed ({error})"
                        continue
                    if not applied:
                        # The section was read but nothing landed -- almost
                        # always a name mismatch, which the applier has just
                        # logged in full.
                        bridge.logger.warning(
                            "Sidecar %r matched none of its %s entr(ies) in the GLB.",
                            section,
                            len(data),
                        )
                        summary[section] = f"0 of {len(data)} matched"
                        continue
                    bridge.logger.info(
                        "Sidecar %r applied to %s.", section, len(applied)
                    )
                    summary[section] = f"{len(applied)} of {len(data)}"
        except (OSError, ValueError) as error:
            # Opening or writing the container, which the sections no longer do
            # for themselves. That failure takes all of them with it, so report
            # every one as failed rather than leaving the sections that
            # "applied" claiming a success that never reached disk.
            bridge.logger.warning("Sidecar not applied to %s: %s", glb, error)
            return {
                section: f"failed ({error})"
                for section in self.SIDECAR_APPLIERS
                if sidecar.get(section)
            }
        return summary

    def deliver(
        self, bridge, payload: Payload, request: HandoffRequest
    ) -> Optional[Dict[str, Any]]:
        # Imported here rather than at module scope: the converter pulls in the
        # managed-binary installer, which no other PreviewServer user needs.
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        if not payload.primary:
            bridge.logger.error("Preview delivery got no exported file to convert.")
            return None

        server = self.ensure_server()
        # Allocate the GLB through the bridge's own payload artifacts rather
        # than deriving a path from the FBX: that keeps it inside the prefix
        # namespace every other bridge artifact joins, so the age-gated sweep
        # reclaims it after a hard DCC crash -- the case no `finally` survives.
        glb = bridge._make_payload_path(extension=".glb")
        try:
            # prompt=False is required, not a convenience: the confirm-download
            # path reads stdin, and a DCC has no tty -- left prompting, the very
            # first push inside Maya raises instead of installing.
            MeshConvert.fbx_to_glb(
                payload.primary, dst=glb, overwrite=True, prompt=False
            )
        except (OSError, RuntimeError, ValueError) as error:
            bridge.logger.error("Preview conversion to GLB failed: %s", error)
            return None

        extras = payload.extras or {}
        sidecar = extras.get("scene_sidecar")
        applied = self._apply_sidecar(bridge, glb, sidecar)
        if not sidecar:
            # Distinguish "switched off" from "on, but the scene had nothing":
            # both produce a bare-FBX preview, and only one is a surprise.
            bridge.logger.info(
                "No scene sidecar %s.",
                "requested" if "scene_sidecar" in extras else "produced for this export",
            )

        version = server.publish(glb, move=True)

        # Asked after publishing, so the freshest possible poll counts.
        open_browser = request.get("open_browser", self.open_browser)
        should_open = open_browser is True or (
            open_browser == "auto" and not server.has_viewer()
        )
        # The decision and the outcome are reported separately on purpose:
        # `open_in_browser` says whether a browser actually launched, and
        # returning the decision would claim a tab exists on the one machine
        # where none does.
        opened = should_open and server.open_in_browser()

        return {
            "url": server.url,
            "version": version,
            "asset": server.manifest()["asset"],
            "opened_browser": opened,
            "sidecar": applied,
            # Whether one was *offered* -- the caller cannot infer it from an
            # empty summary, which also means "switched off".
            "sidecar_requested": "scene_sidecar" in extras,
        }


class PreviewBridge(HandoffBridge):
    """Hand-off bridge whose target is a live preview page rather than an application.

    Sibling of :class:`pythontk.ScriptLaunchBridge`: both specialise the
    hand-off skeleton for one delivery shape. Everything about pushing geometry
    to a browser is host-independent -- the glTF-appropriate export defaults,
    the publish call, the URL -- so a DCC package supplies only what pythontk
    cannot know, which is the mixin that reads its selection:

        >>> class WebXrPreview(MayaExportMixin, ptk.PreviewBridge):
        ...     payload_prefix = "maya_webxr_preview"
        ...     deliverer = ptk.PreviewDeliverer(title="Maya")

    It lives here rather than being mirrored into each engine for the usual
    reason: mayatk and blendertk cannot import each other, so anything written
    in both drifts in both.
    """

    deliverer: Optional[PreviewDeliverer] = None

    def params_defaults(self) -> Dict[str, Any]:
        """glTF-appropriate export defaults, read by both DCC export mixins.

        ``EMBED_TEXTURES`` because the GLB is served standalone: an FBX
        referencing textures by path previews with every map resolving to
        nothing, since the browser can only fetch what the server hosts.
        Animation stays off -- it multiplies payload size, and a preview is
        aimed at look and scale.

        ``TRIANGULATE`` is deliberately **off**, even though glTF has no
        polygon primitive. Maya's FBX exporter rejects triangulation combined
        with smoothing groups outright -- *"Exporting a mesh with triangulation
        and Smoothing Groups enabled is not supported. The resulting FBX file
        may be invalid."* -- and the export mixins turn smoothing groups on for
        every hand-off, because that is what carries the hard/soft edge
        distinction across. Triangulating in the host would therefore trade a
        correct shading normal for an FBX the exporter itself calls invalid.
        The converter triangulates on the way to glTF regardless, so nothing is
        lost by leaving it to the one step that cannot avoid it.

        ``SCENE_SIDECAR`` carries extended scene setup the FBX cannot express,
        read from the live scene and applied to the GLB after conversion (see
        :meth:`PreviewDeliverer._apply_sidecar`). Turning it off is a
        deliberate probe -- the preview then shows exactly what the FBX itself
        carried, which is the only way to tell something the exporter dropped
        from something it mistranslated.
        """
        return {
            "INCLUDE_MATERIALS": True,
            "EMBED_TEXTURES": True,
            "TRIANGULATE": False,
            "INCLUDE_ANIMATION": False,
            "SCENE_SIDECAR": True,
        }

    @property
    def url(self) -> Optional[str]:
        """The preview URL, or ``None`` before the first push."""
        server = getattr(self.deliverer, "server", None)
        return server.url if server is not None else None

    def push(
        self,
        objects: Optional[List[Any]] = None,
        whole_scene: bool = False,
        open_browser: Union[bool, str] = "auto",
        **params: Any,
    ) -> Optional[Dict[str, Any]]:
        """Export and publish, returning the deliverer's result (``None`` on failure).

        Parameters:
            objects: What to preview; ``None`` uses the host's current selection.
            whole_scene: Preview the whole scene instead of the selection.
            open_browser: ``"auto"`` (default) opens a tab only when no page is
                already watching -- so the first push and any push after the
                tab was closed, but not one that an open page will pick up.
                ``True`` every push, ``False`` never.
            **params: Export param overrides (see :meth:`params_defaults`).
        """
        if whole_scene and objects is None:
            # ``None`` here means the host cannot enumerate itself, which the
            # skeleton defines as "fall back to the selection" -- so pass it
            # through rather than treating it as an empty scene.
            objects = self._scene_objects()
        return self.send(objects, params=params, open_browser=open_browser)

    @staticmethod
    def sidecar_summary(result: Optional[Dict[str, Any]]) -> str:
        """One plain-text line describing what the scene sidecar did.

        Lives here rather than in each host's panel because it reads only the
        deliverer's result -- nothing about it is Maya- or Blender-specific,
        and written per panel it would be the same paragraph twice.

        The three outcomes it separates all render as the *same* unlit preview,
        which is what made the feature undebuggable from the UI: switched off,
        on but the scene had nothing to carry, and on but nothing matched.
        """
        if not result:
            return ""
        if not result.get("sidecar_requested"):
            return "Scene sidecar off - showing what the FBX carried."
        applied = result.get("sidecar") or {}
        if not applied:
            # Section-agnostic on purpose: the sections are a registry
            # (PreviewDeliverer.SIDECAR_APPLIERS), so naming one here would go
            # stale with every addition -- it already had, when base colour
            # joined emissive and this line still said "no emissive found".
            return "Scene sidecar: nothing to carry (the scene has nothing the FBX drops)."
        return "Scene sidecar: " + ", ".join(
            f"{name} {outcome}" for name, outcome in sorted(applied.items())
        )

    def stop(self) -> None:
        """Stop serving and release the port."""
        server = getattr(self.deliverer, "server", None)
        if server is not None:
            server.stop()
