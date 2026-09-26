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
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
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
from urllib.parse import parse_qs, quote, unquote, urlparse

from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.temp_artifacts import TempArtifacts
from pythontk.net_utils._net_utils import NetUtils
from pythontk.str_utils._str_utils import StrUtils

if TYPE_CHECKING:  # the recorder is imported on use -- see PreviewServer.playblast
    from pythontk.net_utils.preview.playblast import PreviewPlayblast


#: Path the viewer beacons on unload, so a closed tab is known immediately
#: rather than after :attr:`PreviewServer.VIEWER_TIMEOUT`. Shared by the
#: handler and the served page (which is checked against it by test).
VIEWER_CLOSED_PATH = "viewer-closed"

#: Path the viewer posts a delivery dial to, to have it written into the GLB
#: itself. Only :attr:`PreviewServer.WRITABLE_SETTINGS` may be posted, and each
#: one names the ``MeshConvert`` writer that applies it.
SETTINGS_PATH = "settings"

#: Route prefix the viewer records a clip through: ``<prefix>/begin``,
#: ``/frame``, ``/finish``, ``/cancel``. One recording is many requests -- a
#: canvas readback is a Blob and holding a clip's worth of them in page memory
#: is how a headset tab dies -- so the page streams frames as it steps them.
PLAYBLAST_PATH = "playblast"

#: Sub-routes under :data:`PLAYBLAST_PATH`. An allow-list for the same reason
#: the settings dials are one: these are the routes by which a page reaches a
#: file on disk.
PLAYBLAST_ACTIONS = ("begin", "frame", "finish", "cancel")

#: Path the viewer posts a still of its view to. One request, raw image bytes:
#: a still is a single frame, so it needs none of a recording's token dance.
SNAPSHOT_PATH = "snapshot"

#: What a page's id may look like -- the ``?id=`` on its manifest polls and its
#: close beacon. Anything else is ignored rather than stored: a share counts its
#: guests by id, and the id is the one value a guest chooses.
_VIEWER_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _mesh_convert():
    """The GLB converter, imported on use rather than at module scope.

    Every consumer here is deferred for one reason -- the converter module
    pulls in the managed-binary installer, which no other ``PreviewServer``
    user needs -- and the note was previously repeated at each call site.
    """
    from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

    return MeshConvert


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

    #: Pending connections the kernel will hold while the accept loop is busy.
    #: ``socketserver`` defaults this to 5, which is a burst of five requests --
    #: and the page recording a clip sends one request per FRAME. Measured on a
    #: 151-frame recording: the browser intermittently got
    #: ``ERR_CONNECTION_TIMED_OUT`` part way through (a SYN the full queue
    #: dropped, not a refusal), failing the recording after most of it had been
    #: captured. Cheap to raise: an entry is a pending socket, not a thread.
    request_queue_size = 128

    def handle_error(self, request, client_address):
        """A peer that vanished mid-connection is routine here, not an error.

        A tunnel client holds kept-alive connections to the guest listener,
        and stopping a share kills it, resetting every one of them. The stock
        handler printed a traceback per connection to stderr -- a DCC's script
        editor -- for a stop that went exactly as asked. Anything else still
        reports as before.
        """
        if isinstance(
            sys.exc_info()[1],
            (ConnectionResetError, ConnectionAbortedError, BrokenPipeError),
        ):
            return
        super().handle_error(request, client_address)


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

    #: Keep-alive, which needs an accurate ``Content-Length`` on every response
    #: -- every path here sends one, and the 204s have no body by definition.
    #:
    #: ``SimpleHTTPRequestHandler`` defaults to HTTP/1.0, i.e. a NEW TCP
    #: connection per request. That is invisible for a page that polls a
    #: manifest once a second, and it is the difference between working and not
    #: for a page that posts one request per frame of a recording: 151 frames
    #: became 151 connections, which is what overran the accept queue above.
    #: Reusing one connection also removes 151 handshakes from the wire, and the
    #: whole point of the loopback bind is that this stays fast.
    protocol_version = "HTTP/1.1"

    #: Reap an idle kept-alive connection rather than holding its thread for the
    #: life of the server -- the cost keep-alive brings with it.
    timeout = 30

    def __init__(self, *args, owner: "PreviewServer" = None, guest=False, **kwargs):
        self._owner = owner
        #: Serving the read-only guest listener (see PreviewServer.start_guest).
        #: Fixed per listener, never read from the request: a role taken from
        #: a header is a role a client can claim.
        self._guest = guest
        super().__init__(*args, **kwargs)

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        if not self._discard_body() or self._foreign_host():
            return
        route = self.path.split("?", 1)[0].lstrip("/")
        if self._guest:
            self._guest_get(route)
            return
        if route.startswith(f"{PLAYBLAST_PATH}/"):
            # A finished recording, by token. It exists so the download works
            # wherever the movie landed: a recording of a real deliverable is
            # written BESIDE that file, which is off the serve root and so
            # unreachable by the static handler -- and the one viewer that
            # cannot simply open the containing folder is the headset.
            self._send_recording(route.split("/", 1)[1])
            return
        if self.path.split("?", 1)[0] == "/manifest.json":
            # Only the manifest counts as proof of life: it is fetched on a
            # timer for as long as a page is open, whereas an asset GET happens
            # once per publish and a stray favicon request proves nothing.
            if self._owner is not None:
                self._owner._touch_viewer(self._viewer_id())
            self._send_json(self._owner.manifest() if self._owner else {})
            return
        super().do_GET()

    def do_HEAD(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        """The static handler's HEAD, behind the same gates as GET.

        HEAD answers a path's existence, size and date, and it had no Host
        check at all: a rebound page could probe the serve root through it.
        """
        if not self._discard_body() or self._foreign_host():
            return
        if self._guest and not self._guest_may_read(self._route()):
            self.send_error(404)
            return
        super().do_HEAD()

    def _discard_body(self) -> bool:
        """Take a read's body off the socket; False once it has been refused.

        A body means nothing on a GET or HEAD, but one left unread is the next
        request's first bytes on a kept-alive connection, and on a refusal's
        close Windows answers it with a reset that discards the response
        (measured: 13 of 400 host-refused GETs carrying two bytes lost their
        403). Read -- or refused and drained -- as a POST's is
        (:meth:`_read_body`), within the same ceiling.
        """
        if not (
            self.headers.get("Content-Length") or self.headers.get("Transfer-Encoding")
        ):
            return True
        return self._read_body(self._route()) is not None

    def _route(self) -> str:
        """The request path without its query or leading slash, decoded."""
        return unquote(self.path.split("?", 1)[0].lstrip("/"))

    def _viewer_id(self) -> str:
        """The page's id, from its ``?id=``; empty when it sent none."""
        return (parse_qs(urlparse(self.path).query).get("id") or [""])[0]

    def _guest_may_read(self, route: str) -> bool:
        """Whether a guest may fetch *route*: the page and what its manifest names."""
        return route in ("", "index.html") or route in self._owner._guest_files()

    def _guest_get(self, route: str) -> None:
        """Answer a guest: the page, its manifest, and the files that names.

        An allow-list rather than the static handler's run of the serve root,
        because the root holds more than the share: a scene push's stills and
        recordings land there, a publish passes through a ``.part`` file, and
        ``scripts/`` would list itself. A guest sees exactly the files the
        manifest it was handed names.
        """
        owner = self._owner
        if route == "manifest.json":
            owner._touch_guest(self._viewer_id())
            self._send_json(owner.manifest(guest=True))
            return
        if self._guest_may_read(self._route()):
            super().do_GET()
            return
        self.send_error(404)

    def list_directory(self, path):
        """The static handler's folder listing -- never a guest's.

        A guest's allowed ``/`` is the page; on a root holding none (a server
        started without the viewer) the static handler answered it with a
        listing of the whole serve root instead, every still and recording
        named -- the files :meth:`_guest_get` exists to keep out of a share.
        """
        if self._guest:
            self.send_error(404)
            return None
        return super().list_directory(path)

    def _allowed_hosts(self) -> tuple:
        """Host header spellings this bind answers to.

        The loopback spellings are read off the live socket rather than the
        owner, so they are correct before the owner is attached and on an
        ephemeral port. The guest listener adds the public names the owner
        admitted (:meth:`PreviewServer.admit_host`); the owner's never does.
        """
        port = self.server.server_address[1]
        hosts = (f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}")
        if self._guest and self._owner is not None:
            hosts += self._owner._admitted_hosts()
        return hosts

    def _foreign_host(self) -> bool:
        """Refuse, and say so, when Host is not one this server answers to.

        The only defence against DNS rebinding: under it `Origin` and `Host`
        both name the attacker's domain and therefore AGREE, so comparing them
        proves nothing. Reads are guarded as well as writes -- the manifest and
        the published asset ARE the user's deliverable, and a rebound page that
        can GET them has read it.

        A missing Host is allowed, matching the Origin rule below: nothing off
        this machine reaches a loopback bind unaided, and HTTP/1.0 clients and
        hand-rolled probes legitimately omit it.
        """
        host = self.headers.get("Host")
        # Lowercased: a host name is case-insensitive, and a proxy may forward
        # the one a person typed.
        if host and host.lower() not in self._allowed_hosts():
            self.send_error(403, "Unrecognized Host")
            return True
        return False

    #: The largest body a JSON route takes -- the close beacon, a settings
    #: write, a recording's begin / finish / cancel -- and an unknown route
    #: gets before its 404. Each is a few hundred bytes from the page.
    MAX_JSON_BODY: int = 1024 * 1024

    #: The largest body the GUEST listener takes, whatever the route. Its one
    #: write, the close beacon, carries none, and it is the listener a tunnel
    #: exposes: a body is allocated in full before a byte of it arrives (see
    #: :meth:`_read_body`), so each idle connection a stranger opens holds
    #: this much of the host's memory until it times out.
    MAX_GUEST_BODY: int = 4 * 1024

    def _body_ceiling(self, route: str) -> int:
        """The most bytes a POST to *route* may carry (see :meth:`_read_body`)."""
        if self._guest:
            # A guest writes nothing, so it gets no route's allowance -- a
            # still's 64 MB is the owner's.
            return self.MAX_GUEST_BODY
        owner = self._owner
        if owner is not None and route == SNAPSHOT_PATH:
            return int(owner.MAX_SNAPSHOT_BYTES)
        if owner is not None and route == f"{PLAYBLAST_PATH}/frame":
            return int(owner.playblast.max_frame_bytes)
        return self.MAX_JSON_BODY

    def _read_body(self, route: str) -> Optional[bytes]:
        """The request body, or None once it has been refused.

        Measured against the route's ceiling BEFORE a byte is read, because a
        check on the buffered body bounds nothing: ``rfile.read(n)`` allocates
        *n* up front, so ``Content-Length: 8000000000`` committed 8 GB of the
        host DCC's memory while the read waited for bytes that never came. A
        body refused unread goes with its connection (every ``send_error``
        closes it), so it cannot desynchronise a next request -- once what the
        client still sends of it is read and dropped (:meth:`_drain_refused`),
        so the refusal is what the client reads rather than a reset.
        """
        if self.headers.get("Transfer-Encoding"):
            # The page always sends a length; a chunked body has no bound.
            self.send_error(411, "A Content-Length is required")
            return None
        raw = self.headers.get("Content-Length")
        try:
            length = int(raw) if raw is not None else 0
        except ValueError:
            length = -1
        if length < 0:
            self.send_error(400, "Bad Content-Length")
            return None
        ceiling = self._body_ceiling(route)
        if length > ceiling:
            self.send_error(
                413, f"A body of {length} bytes is over this route's {ceiling}"
            )
            self._drain_refused(length)
            return None
        return self.rfile.read(length)

    #: The most of a refused body :meth:`_drain_refused` reads and drops: the
    #: owner's page on loopback, where a still a little over its 64 MB ceiling
    #: drains in a fraction of a second -- and a guest, owed nothing past a
    #: beacon. Either way within DRAIN_SECONDS.
    DRAIN_OWNER_BYTES: int = 256 * 1024 * 1024
    DRAIN_GUEST_BYTES: int = 64 * 1024
    DRAIN_SECONDS: float = 2.0

    def _drain_refused(self, length: int) -> None:
        """Read and drop what the client still sends of a body just refused.

        The refusal has been sent; closing now, with the rest of the body in
        the socket, makes Windows answer with a reset that discards the
        response the client has not read yet -- a client still sending met
        ConnectionAbortedError instead of the 413 (measured: a third of 64 KiB -
        1 MiB posts against a 16-byte ceiling), which in the page reads "Failed
        to fetch" where the status line should say why. Read in chunks and
        dropped, never buffered, so a false Content-Length still commits no
        memory; bounded in bytes and time, and over at the client's own close.

        Parameters:
            length: The body's declared Content-Length.
        """
        budget = min(
            length, self.DRAIN_GUEST_BYTES if self._guest else self.DRAIN_OWNER_BYTES
        )
        deadline = time.monotonic() + self.DRAIN_SECONDS
        try:
            self.connection.settimeout(self.DRAIN_SECONDS)
            while budget > 0 and time.monotonic() < deadline:
                chunk = self.rfile.read1(min(budget, 64 * 1024))
                if not chunk:
                    break
                budget -= len(chunk)
        except OSError:
            pass  # the client gave up first, or went quiet: close as before

    def send_response_only(self, code, message=None):
        """The status line, with a reason it can carry.

        ``http.server`` encodes the line as strict Latin-1, so a reason naming
        a path in a Cyrillic or CJK folder -- an ``OSError`` from a write --
        raised inside the error path itself: the connection dropped and the
        page showed "Failed to fetch" instead of the reason it exists to show
        (``viewer.refusal``). A CR or LF would split the line. The error
        page's UTF-8 body still carries the full text.
        """
        if message is not None:
            message = (
                str(message)
                .replace("\r", " ")
                .replace("\n", " ")
                .encode("latin-1", "replace")
                .decode("latin-1")
            )
        super().send_response_only(code, message)

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        """Accept the viewer's close notice and its setting writes; 404 else."""
        route = self.path.split("?", 1)[0].lstrip("/")
        # Drained BEFORE the route is judged, not after: sendBeacon always sends
        # a body, so does a rejected recording request, and a body left in the
        # socket desynchronises the next request on a keep-alive connection --
        # which surfaces as the connection dropping on the request AFTER the one
        # that was refused, rather than as the refusal itself.
        body = self._read_body(route)
        if body is None:
            return
        if self._guest and route != VIEWER_CLOSED_PATH:
            # The guest listener has no write routes at all, whatever the route
            # would be on the owner's -- after the drain, like every refusal
            # here, and with the guest's small body allowance (_body_ceiling).
            self.send_error(403, "This is a view-only share")
            return
        playblast_action = (
            route.split("/", 1)[1] if route.startswith(f"{PLAYBLAST_PATH}/") else None
        )
        if playblast_action is not None and playblast_action not in PLAYBLAST_ACTIONS:
            self.send_error(404)
            return
        if playblast_action is None and route not in (
            VIEWER_CLOSED_PATH,
            SETTINGS_PATH,
            SNAPSHOT_PATH,
        ):
            self.send_error(404)
            return
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
        #
        # The settings route is held to the same check for a stronger reason:
        # it is the one request that writes to a FILE, so a page the user
        # happens to have open must not be able to restyle the deliverable.
        origin, host = self.headers.get("Origin"), self.headers.get("Host")
        # Host FIRST -- see `_foreign_host`. Deliberately AFTER the body drain
        # above, not at the top of the method as in `do_GET`: a refused POST
        # still arrived with a body, and leaving it in the socket desynchronises
        # the next request on a keep-alive connection.
        if self._foreign_host():
            return
        if origin and host and origin.split("//", 1)[-1] != host:
            self.send_error(403, "Cross-origin post rejected")
            return
        if self._owner is None:
            self.send_response(204)
            self.end_headers()
            return
        if route == VIEWER_CLOSED_PATH:
            if self._guest:
                # A guest's tab says it closed; the owner's viewer is untouched
                # -- a guest leaving must never make the next push pop a tab.
                self._owner._drop_guest(self._viewer_id())
            else:
                self._owner._clear_viewer(self._viewer_id())
            self.send_response(204)
            self.end_headers()
            return
        if playblast_action is not None:
            self._do_playblast(playblast_action, body)
            return
        if route == SNAPSHOT_PATH:
            self._do_snapshot(body)
            return
        try:
            payload = json.loads(body or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
            self._send_json(self._owner.apply_settings(payload))
        except (ValueError, TypeError, KeyError) as error:
            # 400, not 500: every one of these is the page saying something
            # this server does not accept, and the page shows the reason.
            self.send_error(400, f"Settings rejected: {error}")
        except OSError as error:
            self.send_error(500, f"Settings write failed: {error}")

    def _send_recording(self, token: str) -> None:
        """Stream a finished recording as an attachment, by token."""
        path = self._owner.recording_path(token) if self._owner else None
        if path is None:
            self.send_error(404, "No such recording")
            return
        try:
            size = path.stat().st_size
            handle = path.open("rb")
        except OSError as error:
            self.send_error(500, f"Recording unreadable: {error}")
            return
        with handle:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            # The page opens this in a hidden anchor; without the disposition a
            # browser plays the mp4 in place instead of saving it, which on a
            # headset means the preview is replaced by a video player.
            self.send_header(
                "Content-Disposition", f'attachment; filename="{path.name}"'
            )
            self.end_headers()
            shutil.copyfileobj(handle, self.wfile)

    def _do_playblast(self, action: str, body: bytes) -> None:
        """Answer one recording request. Errors are the page's to display.

        ``frame`` carries raw image bytes and its parameters in the query
        string; the other three carry JSON. Splitting on the body type rather
        than base64-ing every frame into JSON is worth the asymmetry: a
        1920x1080 PNG grows by a third in base64, on the link least able to
        afford it.
        """
        query = parse_qs(urlparse(self.path).query)
        first = lambda key, default=None: (query.get(key) or [default])[0]  # noqa: E731
        try:
            if action == "frame":
                result = self._owner.playblast.add_frame(
                    token=first("token", ""),
                    index=int(first("index", -1)),
                    data=body,
                )
            else:
                payload = json.loads(body or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("expected a JSON object")
                if action == "begin":
                    result = self._owner.begin_playblast(**payload)
                elif action == "finish":
                    result = self._owner.finish_playblast(**payload)
                else:  # cancel
                    result = {
                        "cancelled": self._owner.playblast.cancel(
                            payload.get("token", "")
                        )
                    }
        except KeyError as error:
            # A recording this server has no record of: finished, cancelled, or
            # a page that outlived a server restart. 409 rather than 404 -- the
            # ROUTE exists, the recording does not, and the page's remedy is to
            # start a new one rather than to retry this request.
            self.send_error(409, f"No such recording: {error}")
        except (TypeError, ValueError) as error:
            self.send_error(400, f"Recording rejected: {error}")
        except (OSError, RuntimeError) as error:
            self.send_error(500, f"Recording failed: {error}")
        else:
            self._send_json(result)

    def _do_snapshot(self, body: bytes) -> None:
        """Answer the page's still. Raw bytes, typed by ``Content-Type`` -- the
        same split a recording's frames make, for the same base64 reason."""
        try:
            result = self._owner.save_snapshot(
                body, content_type=self.headers.get("Content-Type", "")
            )
        except (TypeError, ValueError) as error:
            self.send_error(400, f"Image rejected: {error}")
        except OSError as error:
            self.send_error(500, f"Image write failed: {error}")
        else:
            self._send_json(result)

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
        src = Path(__file__).with_name("viewer.html")
        if not src.is_file():  # pragma: no cover - packaging failure
            self.logger.warning("Viewer page missing from the package: %s", src)
            return
        served = self.root / "index.html"
        self._sync_file(src, served)
        # Fingerprint of the page actually SERVED (a caller-owned root keeps its
        # own copy, which may differ from the package), published in the
        # manifest so an OPEN tab can tell its script has been superseded and
        # reload itself: the poll swaps the model but never the JavaScript
        # running it, so a viewer edit otherwise reached a running session's
        # page only via F5 (measured 2026-09-05: a highlight channel the GLB
        # carried, and a page from before the binding existed).
        try:
            stamp = hashlib.sha1(served.read_bytes()).hexdigest()[:12]
        except OSError:
            stamp = ""
        with self._lock:
            self._viewer_stamp = stamp

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
        self._write_asset(src, dst, move=False)

    def _ensure_scripts(self) -> None:
        """Materialize the active viewer scripts into ``<root>/scripts/``.

        The extension seam's disk half: :attr:`PreviewServer.SCRIPTS` (and any
        caller-registered module) is copied under the serve root so the page
        can ``import()`` what :meth:`PreviewServer.manifest` names. Each module
        is written as ``<registered name>.js`` rather than under its source
        filename, so the served URL is a function of the *name* alone -- two
        callers registering different files under one name is a collision the
        registry resolves, not one the URL space has to.

        Per-file placement is :meth:`_sync_file`'s policy, shared with the
        viewer page. What is specific here is the **sweep**: in a managed root
        the directory is entirely ours, so a module no longer active is removed
        -- otherwise a script switched off for this push stays on disk and the
        next reader of the serve root sees one the manifest does not name. A
        caller-supplied root is never deleted from.
        """
        directory = self.root / self.SCRIPTS_ROUTE
        active = self._active_scripts()
        if not active and not directory.is_dir():
            return  # nothing active and nothing to sweep
        directory.mkdir(parents=True, exist_ok=True)
        for name, src in active.items():
            self._sync_file(src, directory / f"{name}.js")
        if self._temp is None:
            return  # never sweep a directory the caller owns
        keep = {f"{name}.js" for name in active}
        for stale in directory.glob("*.js"):
            if stale.name not in keep:
                stale.unlink()

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

    def _page_output(self) -> Tuple[Path, str]:
        """``(directory, base)`` for a file the page asks this server to write.

        The one placement rule every page output follows -- a recording and a
        still alike -- so the two cannot disagree about where "beside the
        deliverable" is: the directory is
        :meth:`PreviewPlayblast.resolve_output_dir`'s answer, and *base* the
        name a file there is labelled with.

        *base* is the SOURCE's stem, not the served asset's: the asset is
        republished under a stable ``scene.glb`` so a page can keep one URL
        across pushes, which is exactly the property that makes it useless as a
        label -- every output of every deliverable would be called ``scene_*``.
        Sanitized, because the page's outputs are the one place a filename is
        built without the caller naming it.
        """
        from pythontk.net_utils.preview.playblast import PreviewPlayblast

        with self._lock:
            source = self._source
        directory = PreviewPlayblast.resolve_output_dir(source, self.root)
        # A label only when there IS a deliverable on disk. A scene push's
        # source is the bridge's scratch payload, moved into the serve root on
        # publish: its stem is a random tag, and outputs named after it
        # restarted their numbering under a new unreadable prefix every push.
        stem = source.stem if source is not None and directory != self.root else ""
        if directory != self.root and not self._writable(directory):
            # A deliverable on a read-only share: its outputs still belong to
            # the page, and the serve root is where the page can collect them.
            self.logger.warning(
                "%s is not writable; page output goes to the preview's own "
                "folder instead, from which the page downloads it.",
                directory,
            )
            directory = self.root
        return directory, StrUtils.to_legal_name(stem)

    @staticmethod
    def _writable(directory: Path) -> bool:
        """Whether a file can be created in *directory* -- found by trying.

        ``os.access`` answers from mode bits alone, and on Windows says yes to
        a read-only share or a denying ACL; the probe is self-cleaning.
        """
        try:
            with tempfile.NamedTemporaryFile(dir=directory):
                return True
        except OSError:
            return False


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

    #: Directory holding the packaged optional viewer scripts.
    SCRIPTS_DIR = Path(__file__).with_name("scripts")

    #: Serve-root subdirectory (and URL prefix) the active scripts live under.
    SCRIPTS_ROUTE = "scripts"

    #: Built-in viewer scripts, registered name -> filename in
    #: :attr:`SCRIPTS_DIR`. This is the *extension registry*: the viewer page
    #: itself stays the stable path and gains behaviour by a module being
    #: activated, never by being edited (OCP). A script is an ES module whose
    #: default export is called once with the page's viewer API -- see
    #: ``docs/webxr_preview.md`` for that surface and
    #: :meth:`add_script` for registering one from outside this package.
    SCRIPTS: Dict[str, str] = {
        "turntable": "turntable.js",
        "inspect": "inspect.js",
        "shadow_rig": "shadow_rig.js",
        "playblast": "playblast.js",
        "snapshot": "snapshot.js",
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
    AUTO_SCRIPTS: Dict[str, str] = {
        "shadow_rig": "shadow_web",
        # A deliverable that ships clips grows a picker and a transport in the
        # page; recording what that transport plays is the same kind of "not
        # optional in any useful sense" as the shadow rigs. There is no
        # checkbox for it because the alternative is worse: the button would be
        # missing on exactly the push a reviewer just watched and wants to
        # send on, and getting it would mean re-exporting the animation.
        "playblast": "animation_web",
    }

    #: Viewer scripts that write through the owner's routes -- a recording, a
    #: still -- and are therefore withheld from a guest (:meth:`start_guest`).
    #: The guest listener refuses every write, so served to a guest each would
    #: be a button that fails; withheld, the guest never downloads it either. A
    #: script registered under another name is served to guests: an overlay
    #: that only reads needs nothing here.
    OWNER_SCRIPTS = frozenset({"playblast", "snapshot"})

    #: Delivery dials the served page may write into the published GLB:
    #: name -> (:class:`MeshConvert` writer, coercion). An allow-list, because
    #: this is the only route by which the page reaches a file on disk.
    WRITABLE_SETTINGS: Dict[str, Tuple[str, Callable]] = {
        "normal_scale": ("set_glb_normal_scale", float),
    }

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

    #: Guest tabs a share tracks at once (:meth:`guest_count`); past it the one
    #: seen longest ago is forgotten. The ids are the guests' own choosing, so
    #: the table has to be bounded by this side.
    MAX_GUESTS = 256

    #: Seconds a page's poll is ignored after that page beaconed that it
    #: closed. Every request is answered on its own thread, so a poll the page
    #: sent just before closing can be answered AFTER the beacon -- and brought
    #: the page back from the dead for a whole :attr:`VIEWER_TIMEOUT` (caught
    #: 2026-09-23 by the live guest test, under load). A page restored from the
    #: back/forward cache polls again once this has passed.
    CLOSED_LINGER = 5.0

    #: Environment variables a share reads when its caller names no alias: where
    #: to keep the stable redirect to the link (``ShareTunnel``'s *alias*: a
    #: path or ``user@host:/path``), and the public address that is served at
    #: -- the link to hand out instead of the tunnel's own. Machine config, so a
    #: web root's hostname never lands in a repo.
    ALIAS_ENV = "PYTHONTK_PREVIEW_ALIAS"
    ALIAS_URL_ENV = "PYTHONTK_PREVIEW_ALIAS_URL"

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
            name: A key of :attr:`SCRIPTS` (a packaged script), or any name at
                all when *path* is given. It is also the served basename, so
                ``add_script("turntable")`` is imported from
                ``scripts/turntable.js``.
            path: An ES module outside this package to serve under *name* --
                the seam a consumer extends through without vendoring anything
                into pythontk.

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

    def _active_scripts(self) -> Dict[str, Path]:
        """Snapshot of the active name -> source map."""
        with self._lock:
            return dict(self._scripts)

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

    # ------------------------------------------------------------------
    # Sharing
    # ------------------------------------------------------------------
    @property
    def guest_port(self) -> Optional[int]:
        """The guest listener's port, or ``None`` while it is closed."""
        httpd = self._guest_httpd
        return None if httpd is None else httpd.server_address[1]

    @property
    def guest_url(self) -> Optional[str]:
        """The guest listener on loopback: what a guest sees, from this machine."""
        port = self.guest_port
        return None if port is None else f"http://{self.host}:{port}/"

    def start_guest(self, port: int = 0) -> int:
        """Open the read-only listener a share points at; its port. Idempotent.

        A second door onto the same serve root, and the only one a share ever
        fronts. Its role is fixed by the socket rather than read from a
        request, so nothing a guest sends can make it the owner's:

        * reads are an allow-list -- the page, its manifest, and exactly what
          that names (the asset, the scripts outside :attr:`OWNER_SCRIPTS`);
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
        self.start()
        if self._guest_httpd is None:
            handler = partial(
                _PreviewHandler, directory=str(self.root), owner=self, guest=True
            )
            self._guest_httpd = _PreviewHTTPServer((self.host, int(port)), handler)
            self._guest_thread = threading.Thread(
                target=partial(
                    self._guest_httpd.serve_forever, poll_interval=self.POLL_INTERVAL
                ),
                name="ptk-preview-guest",
                daemon=True,
            )
            self._guest_thread.start()
            self.logger.info("View-only listener on %s", self.guest_url)
        return self.guest_port

    def stop_guest(self) -> None:
        """Close the read-only listener, forgetting its admitted names and its
        guests. Idempotent."""
        httpd, self._guest_httpd = self._guest_httpd, None
        thread, self._guest_thread = self._guest_thread, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None:
            thread.join(timeout=5)
        with self._lock:
            self._admitted.clear()
            self._guests.clear()

    def admit_host(self, netloc: str) -> None:
        """Let the guest listener answer requests addressed to *netloc*.

        *netloc* is the public name a proxy forwards (``name``, or
        ``name:port`` off the default port -- as a ``Host`` header spells it).
        The Host check is the guest listener's DNS-rebinding defence too, so a
        proxy's name is named rather than the check dropped. :meth:`share`
        admits its tunnel's name itself; this is for a transport of your own.
        The owner's listener never admits one.
        """
        with self._lock:
            self._admitted.add(str(netloc).strip().lower())

    def _admitted_hosts(self) -> tuple:
        with self._lock:
            return tuple(self._admitted)

    def guest_count(self) -> int:
        """How many guest tabs are watching: polled within :attr:`VIEWER_TIMEOUT`."""
        cutoff = time.time() - self.VIEWER_TIMEOUT
        with self._lock:
            return sum(1 for seen in self._guests.values() if seen >= cutoff)

    def _touch_guest(self, viewer_id: str) -> None:
        """Record a poll from guest tab *viewer_id*; a malformed id is ignored."""
        if not _VIEWER_ID.fullmatch(viewer_id or ""):
            return
        with self._lock:
            if self._closed_recently(viewer_id):
                return  # a poll that crossed its own close beacon
            # Re-inserted, so the dict runs oldest-seen first and the cap below
            # forgets the stalest tab rather than an arbitrary one.
            self._guests.pop(viewer_id, None)
            self._guests[viewer_id] = time.time()
            while len(self._guests) > self.MAX_GUESTS:
                self._guests.pop(next(iter(self._guests)))

    def _drop_guest(self, viewer_id: str) -> None:
        with self._lock:
            self._guests.pop(viewer_id or "", None)
            self._mark_closed(viewer_id)

    def _guest_files(self) -> set:
        """Served paths a guest may fetch besides the page: what its manifest names."""
        with self._lock:
            files = {
                f"{self.SCRIPTS_ROUTE}/{name}.js"
                for name in self._scripts
                if name not in self.OWNER_SCRIPTS
            }
            if self._asset:
                files.add(self._asset)
        return files

    def share(
        self,
        provider: Optional[str] = None,
        alias: Union[
            str, os.PathLike, Callable[[Optional[str]], Any], bool, None
        ] = None,
        alias_url: Optional[str] = None,
        timeout: Optional[float] = None,
        on_step: Optional[Callable[[Any], Any]] = None,
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
        from pythontk.net_utils.share_tunnel import ShareTunnel

        if alias is None:
            alias = os.environ.get(self.ALIAS_ENV, "").strip() or None
        elif alias is False:
            alias = None
        if alias_url is None:
            alias_url = os.environ.get(self.ALIAS_URL_ENV, "").strip() or None
        name = ShareTunnel.resolve_provider(provider)
        with self._share_lock:
            with self._lock:
                current = self._tunnel
            if (
                current is not None
                and current.is_running
                and current.provider == name
                and current.alias == alias
            ):
                with self._lock:
                    self._alias_url = alias_url
                return self.share_info()
            generation = self._end_share()
            port = self.start_guest()
        options = {} if timeout is None else {"timeout": timeout}
        # The provider's wait (seconds -- a quick tunnel's name takes 6-18 s
        # to resolve; minutes, when a user has a step to take) is the one step
        # taken outside the share lock, so an unshare() or stop() on another
        # thread lands at once. It cancels the tunnel still starting and closes
        # the listener it fronts; the check below is how the share learns of it
        # when the link was already in.
        tunnel = None
        try:
            tunnel = ShareTunnel(
                port,
                provider=name,
                host=self.host,
                alias=alias,
                on_url=self._on_share_url,
                on_step=on_step,
                **options,
            )
            with self._share_lock:
                if self._share_generation == generation:
                    with self._lock:
                        self._starting = tunnel
                else:  # stopped already: the start ends before it spawns
                    tunnel.cancel()
            tunnel.start()
        except BaseException:
            with self._share_lock:
                if self._share_generation == generation:
                    self.stop_guest()
            raise
        finally:
            with self._lock:
                if self._starting is tunnel:
                    self._starting = None
        with self._share_lock:
            superseded = self._share_generation != generation
            if superseded:
                # Kept, the link would stay public in front of a closed port,
                # for whatever binds that port next.
                tunnel.stop()
            else:
                with self._lock:
                    self._tunnel = tunnel
                    self._alias_url = alias_url
        if superseded:
            raise RuntimeError(
                f"{tunnel.label}: the share was stopped before its link was ready."
            )
        info = self.share_info()
        if info is None:  # the client exited in the moment after its link
            output = "\n".join(tunnel.output(12))
            self.unshare()
            raise RuntimeError(f"{tunnel.label} stopped as the share began:\n{output}")
        return info

    def _on_share_url(self, url: Optional[str]) -> None:
        """The tunnel's link hook: admit its public name BEFORE the alias
        announces it, so the first guest through never meets a 403."""
        if url:
            self.admit_host(urlparse(url).netloc)

    def unshare(self) -> None:
        """Stop sharing: the tunnel stops (an alias then says the share ended)
        and the guest listener closes. The owner's page is untouched.
        Idempotent; a share still waiting on its provider stops as its link
        arrives."""
        with self._share_lock:
            self._end_share()

    def _end_share(self) -> int:
        """:meth:`unshare`'s work, under the share lock; the new generation."""
        with self._lock:
            tunnel, self._tunnel = self._tunnel, None
            starting, self._starting = self._starting, None
            self._alias_url = None
        self._share_generation += 1
        if starting is not None:
            starting.cancel()  # never blocks: that start holds its own lock
        if tunnel is not None:
            tunnel.stop()
        self.stop_guest()
        return self._share_generation

    def share_info(self) -> Optional[Dict[str, Any]]:
        """What is being shared, or ``None`` -- also once the provider exited
        on its own, which is how a dropped share shows.

        Returns:
            ``{"url", "tunnel_url", "provider", "label", "public", "guests",
            "guest_url", "alias_error"}``. ``url`` is the link to hand out:
            the alias's address when there is an alias AND its last update
            landed, else the tunnel's own -- a stale alias must never be what
            gets sent.
        """
        with self._lock:
            tunnel, alias_url = self._tunnel, self._alias_url
        if tunnel is None or not tunnel.is_running:
            return None
        tunnel_url = tunnel.url
        alias_live = (
            bool(alias_url) and tunnel.alias is not None and not tunnel.alias_error
        )
        return {
            "url": alias_url if alias_live else tunnel_url,
            "tunnel_url": tunnel_url,
            "provider": tunnel.provider,
            "label": tunnel.label,
            "public": tunnel.public,
            "guests": self.guest_count(),
            "guest_url": self.guest_url,
            "alias_error": tunnel.alias_error,
        }

    @property
    def share_url(self) -> Optional[str]:
        """The link to hand out while sharing, else ``None``."""
        info = self.share_info()
        return info["url"] if info else None

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

    # ------------------------------------------------------------------
    # Page recordings
    # ------------------------------------------------------------------
    @property
    def playblast(self) -> "PreviewPlayblast":
        """The page's recorder, created on first use.

        Deferred rather than built in ``__init__`` because it allocates a
        scratch store, and the overwhelming majority of preview servers never
        record anything.
        """
        # Under the lock: this server answers each request on its OWN thread, so
        # two frame POSTs racing the first access would each build a recorder --
        # and a frame would then be filed against one the `begin` never reached.
        # Double-checked, on a lock of its OWN. The race being closed is two
        # first-accesses building two recorders (this server answers each
        # request on its own thread), which lasts exactly as long as the first
        # request -- but this property is read once per FRAME, and taking the
        # server's lock here put every frame POST of a recording behind the same
        # lock the manifest poll uses. Measured: a 151-frame recording went from
        # 5s to 22s and then died with the accept queue full
        # (ERR_CONNECTION_TIMED_OUT part way through). The common path must not
        # lock at all; the attribute read and write are each atomic.
        if self._playblast is None:
            with self._playblast_lock:
                if self._playblast is None:
                    from pythontk.net_utils.preview.playblast import PreviewPlayblast

                    self._playblast = PreviewPlayblast()
        return self._playblast

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
        # Named for the deliverable rather than for the clip alone: a folder of
        # shot movies from three different pushes is otherwise unreadable.
        #
        # Composed here and NOT accepted from the caller: this method's caller
        # is the served page, and a name it chose would reach ``os.path.join``.
        # (``finish`` sanitizes a stem as well -- this is the half that keeps
        # the page from naming the file at all.)
        output_dir, base = self._page_output()
        name = self.playblast.clip_name(token)
        stem = f"{base}_{name}" if base and base not in name else name
        report = self.playblast.finish(
            token, output_dir=str(output_dir), target=target, stem=stem
        )
        path = Path(report["output"])
        with self._lock:
            self._recordings[token] = path
            # Bounded: the map only exists so a download can follow a finish,
            # and a preview server lives for a whole DCC session.
            for stale in list(self._recordings)[: -self.MAX_RECORDINGS]:
                self._recordings.pop(stale, None)
        report["url"] = f"{self.url}{PLAYBLAST_PATH}/{token}"
        report["in_serve_root"] = path.parent == self.root
        self.logger.info("Recording written to %s", path)
        return report

    def recording_path(self, token: str) -> Optional[Path]:
        """The file a finished recording produced, or None."""
        with self._lock:
            path = self._recordings.get(str(token))
        return path if path is not None and path.is_file() else None

    # ------------------------------------------------------------------
    # Page stills
    # ------------------------------------------------------------------
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
        mime = str(content_type or "").split(";", 1)[0].strip().lower()
        spec = self.SNAPSHOT_TYPES.get(mime)
        if spec is None:
            raise ValueError(
                f"unsupported image type {content_type!r}; expected one of "
                f"{', '.join(sorted(self.SNAPSHOT_TYPES))}"
            )
        extension, signature = spec
        if not data:
            raise ValueError("the image arrived empty")
        if len(data) > self.MAX_SNAPSHOT_BYTES:
            raise ValueError(
                f"the image is {len(data)} bytes; the ceiling is "
                f"{self.MAX_SNAPSHOT_BYTES}"
            )
        if not data.startswith(signature):
            raise ValueError(f"the body is not a {mime} image")

        directory, base = self._page_output()
        stem = f"{base}_view" if base else "view"
        with self._snapshot_lock:
            path = Path(
                FileUtils.next_version_path(
                    str(directory / f"{stem}{extension}"),
                    format="{stem}_{n:03d}{ext}",
                )
            )
            # Whole, then moved into place: the serve root is being read by a
            # page, and beside a deliverable the folder is the user's -- which
            # is also why a failed write takes its partial file with it.
            part = path.with_name(path.name + ".part")
            try:
                part.write_bytes(data)
                os.replace(part, path)
            except OSError:
                part.unlink(missing_ok=True)
                raise
        in_serve_root = directory == self.root
        self.logger.info("Image written to %s", path)
        return {
            "output": str(path),
            "name": path.name,
            "url": quote(path.name) if in_serve_root else None,
            "in_serve_root": in_serve_root,
            "bytes": len(data),
        }

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
