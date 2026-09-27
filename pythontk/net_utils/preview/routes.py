# !/usr/bin/python
# coding=utf-8
"""The preview server's served surface: its URL vocabulary and the HTTP handler.

The route names are shared with the served page (``viewer.html`` posts to
them, and a test holds the two together); :mod:`.server` re-exports them, the
path callers import them from. ``_PreviewHandler`` answers every request --
the static tree with caching off, the live ``/manifest.json``, the page's
write routes -- behind the gates the loopback bind relies on (the Host check
against DNS rebinding, the cross-origin check on writes, a body ceiling per
route measured before a byte is read); ``_PreviewHTTPServer`` is the listener
it runs on. Both serve a :class:`~pythontk.PreviewServer` (``owner``), which
holds every piece of state; a listener's role -- the owner's, or a share's
read-only guest -- is fixed when it is built, never read from a request.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, TYPE_CHECKING
from urllib.parse import parse_qs, unquote, urlparse

if TYPE_CHECKING:
    from pythontk.net_utils.preview.server import PreviewServer

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
