# !/usr/bin/python
# coding=utf-8
"""Sharing a preview by link: the read-only guest listener and its tunnel.

:meth:`PreviewServer.share` opens a second listener onto the same serve root
(:meth:`PreviewServer.start_guest`) -- reads an allow-list of what the guest
manifest names, refuses every write, counts its tabs apart from the owner's
viewer -- and fronts it with a :class:`~pythontk.ShareTunnel`, whose provider
terminates TLS on a public name. The owner's listener stays on loopback and
answers no public name; who may write is decided by the socket a request
arrived on.

A PRIVATE share adds a fresh random path segment to its link
(``https://<name>/<token>/``) and the guest listener answers nothing outside
it, so a link is good for one share only -- the provider's own name, which
Tailscale keeps for good, opens nothing by itself.
:meth:`PreviewServer.ping_share` asks the link from this machine whether a
guest can reach it.

:class:`_PreviewShare` is the server's collaborator that owns every piece of
that security state -- the guest listener and its thread, the names it
admits, the guests it counts, the tunnel, the private segment, and the lock
and generation a share sets up under -- so nothing else touches it. It holds
the :class:`~pythontk.PreviewServer` and calls its public methods through it
(an override or a ``mock.patch.object`` of one reaches every caller); the
server's state lock guards what a request reads.
"""

from __future__ import annotations

import http.client
import json
import os
import secrets
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union, TYPE_CHECKING
from urllib.parse import urlparse

from pythontk.net_utils.preview.routes import (
    SCENE_PATH,
    _VIEWER_ID,
    _PreviewHandler,
    _PreviewHTTPServer,
)

if TYPE_CHECKING:
    from pythontk.net_utils.preview.server import PreviewServer


class _PreviewShare:
    """The guest listener, its admitted names and guests, the share's tunnel.

    A private collaborator of :class:`~pythontk.PreviewServer`; call it through
    the facade, whose methods carry the documentation.
    """

    def __init__(self, server: "PreviewServer"):
        self._server = server
        #: The read-only listener a share fronts, and its serving thread.
        self.guest_httpd = None
        self.guest_thread: Optional[threading.Thread] = None
        #: Public names the guest listener answers to (``admit_host``).
        self.admitted: set = set()
        #: Guest tabs, page id -> last poll; oldest first (:meth:`touch_guest`).
        self.guests: Dict[str, float] = {}
        #: The tunnel in front of the guest listener while sharing, and the
        #: public address of its alias, when there is one.
        self.tunnel = None
        self.alias_url: Optional[str] = None
        #: A private share's secret path segment, or ``None``: the guest
        #: listener then answers only under ``/<token>/``.
        self.token: Optional[str] = None
        #: A share's tunnel while it still waits on its provider, so an
        #: unshare can end that wait (:meth:`ShareTunnel.cancel`).
        self.starting = None
        #: Held while a share sets up or installs its tunnel and while
        #: :meth:`unshare` runs -- never across a provider's wait -- and the
        #: count of unshares, so a share whose wait an unshare interrupted can
        #: tell (see :meth:`share`).
        self.lock = threading.Lock()
        self.generation = 0

    # ------------------------------------------------------------------
    # The guest listener
    # ------------------------------------------------------------------
    @property
    def guest_port(self) -> Optional[int]:
        httpd = self.guest_httpd
        return None if httpd is None else httpd.server_address[1]

    @property
    def guest_url(self) -> Optional[str]:
        port = self._server.guest_port
        if port is None:
            return None
        return f"http://{self._server.host}:{port}{self.share_path()}"

    def share_path(self) -> str:
        """The path a guest's page is served under: ``/``, or a private
        share's ``/<token>/``."""
        token = self.token
        return f"/{token}/" if token else "/"

    def guest_route(self, path: str) -> Optional[str]:
        """*path* as the guest listener routes it -- a private share's segment
        taken off -- or ``None`` when it lies outside the share.

        The segment is compared in constant time: it is the share's only
        secret, and a comparison that stops at the first wrong character
        would time its way to it a character at a time.

        A listener that is closing, or closed, answers nothing: a connection a
        guest or the tunnel kept alive outlives the listening socket, and its
        next request would otherwise read a share whose segment is already gone.
        """
        if self.guest_httpd is None:
            return None
        token = self.token
        if not token:
            return path
        head, _, rest = path.lstrip("/").partition("/")
        if not secrets.compare_digest(head.encode(), token.encode()):
            return None
        return "/" + rest

    def start_guest(self, port: int = 0) -> int:
        server = self._server
        server.start()
        if self.guest_httpd is None:
            handler = partial(
                _PreviewHandler, directory=str(server.root), owner=server, guest=True
            )
            self.guest_httpd = _PreviewHTTPServer((server.host, int(port)), handler)
            self.guest_thread = threading.Thread(
                target=partial(
                    self.guest_httpd.serve_forever, poll_interval=server.POLL_INTERVAL
                ),
                name="ptk-preview-guest",
                daemon=True,
            )
            self.guest_thread.start()
            server.logger.info("View-only listener on %s", server.guest_url)
        return server.guest_port

    def stop_guest(self) -> None:
        httpd, self.guest_httpd = self.guest_httpd, None
        thread, self.guest_thread = self.guest_thread, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None:
            thread.join(timeout=5)
        with self._server._lock:
            self.admitted.clear()
            self.guests.clear()

    def admit_host(self, netloc: str) -> None:
        with self._server._lock:
            self.admitted.add(str(netloc).strip().lower())

    def admitted_hosts(self) -> tuple:
        with self._server._lock:
            return tuple(self.admitted)

    # ------------------------------------------------------------------
    # Guests
    # ------------------------------------------------------------------
    def guest_count(self) -> int:
        server = self._server
        cutoff = time.time() - server.VIEWER_TIMEOUT
        with server._lock:
            return sum(1 for seen in self.guests.values() if seen >= cutoff)

    def touch_guest(self, viewer_id: str) -> None:
        """Record a poll from guest tab *viewer_id*; a malformed id is ignored."""
        if not _VIEWER_ID.fullmatch(viewer_id or ""):
            return
        server = self._server
        with server._lock:
            if server._closed_recently(viewer_id):
                return  # a poll that crossed its own close beacon
            # Re-inserted, so the dict runs oldest-seen first and the cap below
            # forgets the stalest tab rather than an arbitrary one.
            self.guests.pop(viewer_id, None)
            self.guests[viewer_id] = time.time()
            while len(self.guests) > server.MAX_GUESTS:
                self.guests.pop(next(iter(self.guests)))

    def drop_guest(self, viewer_id: str) -> None:
        server = self._server
        with server._lock:
            self.guests.pop(viewer_id or "", None)
            server._mark_closed(viewer_id)

    # ------------------------------------------------------------------
    # The share
    # ------------------------------------------------------------------
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
        from pythontk.net_utils.share_tunnel import ShareTunnel

        server = self._server
        if private or alias is False:
            alias = alias_url = None
        elif alias is None:
            alias = os.environ.get(server.ALIAS_ENV, "").strip() or None
        if alias_url is None and alias is not None:
            alias_url = os.environ.get(server.ALIAS_URL_ENV, "").strip() or None
        name = ShareTunnel.resolve_provider(provider)
        with self.lock:
            with server._lock:
                current = self.tunnel
            if (
                current is not None
                and current.is_running
                and current.provider == name
                and current.alias == alias
                and bool(self.token) == bool(private)
            ):
                with server._lock:
                    self.alias_url = alias_url
                return server.share_info()
            generation = self.end_share()
            # Before the listener opens: from its first request, a private
            # share's guest listener answers under the segment only.
            with server._lock:
                self.token = (
                    secrets.token_urlsafe(server.SHARE_TOKEN_BYTES) if private else None
                )
            port = server.start_guest()
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
                host=server.host,
                alias=alias,
                on_url=self.on_share_url,
                on_step=on_step,
                **options,
            )
            with self.lock:
                if self.generation == generation:
                    with server._lock:
                        self.starting = tunnel
                else:  # stopped already: the start ends before it spawns
                    tunnel.cancel()
            tunnel.start()
        except BaseException:
            with self.lock:
                if self.generation == generation:
                    server.stop_guest()
            raise
        finally:
            with server._lock:
                if self.starting is tunnel:
                    self.starting = None
        with self.lock:
            superseded = self.generation != generation
            if superseded:
                # Kept, the link would stay public in front of a closed port,
                # for whatever binds that port next.
                tunnel.stop()
            else:
                with server._lock:
                    self.tunnel = tunnel
                    self.alias_url = alias_url
        if superseded:
            raise RuntimeError(
                f"{tunnel.label}: the share was stopped before its link was ready."
            )
        info = server.share_info()
        if info is None:  # the client exited in the moment after its link
            output = "\n".join(tunnel.output(12))
            server.unshare()
            raise RuntimeError(f"{tunnel.label} stopped as the share began:\n{output}")
        return info

    def on_share_url(self, url: Optional[str]) -> None:
        """The tunnel's link hook: admit its public name BEFORE the alias
        announces it, so the first guest through never meets a 403."""
        if url:
            self._server.admit_host(urlparse(url).netloc)

    def unshare(self) -> None:
        with self.lock:
            self.end_share()

    def end_share(self) -> int:
        """:meth:`unshare`'s work, under the share lock; the new generation.

        The guest listener closes FIRST, its admitted names with it, while the
        private segment still stands: cleared with the listener still open, a
        guest request in that window -- as long as the tunnel took to stop --
        was answered outside the segment, under the public name.
        """
        server = self._server
        server.stop_guest()
        with server._lock:
            tunnel, self.tunnel = self.tunnel, None
            starting, self.starting = self.starting, None
            self.alias_url = None
            self.token = None
        self.generation += 1
        if starting is not None:
            starting.cancel()  # never blocks: that start holds its own lock
        if tunnel is not None:
            tunnel.stop()
        return self.generation

    def share_info(self) -> Optional[Dict[str, Any]]:
        server = self._server
        with server._lock:
            tunnel, alias_url = self.tunnel, self.alias_url
            path = self.share_path()
        if tunnel is None or not tunnel.is_running:
            return None
        # None with the tunnel's link: a client that exits between the check
        # above and the read reads as a link that is gone, not a crash. The
        # provider's own spelling stands unless a private segment is added.
        tunnel_url = tunnel.url
        if tunnel_url and path != "/":
            tunnel_url = f"{tunnel_url.rstrip('/')}{path}"
        alias_live = (
            bool(alias_url) and tunnel.alias is not None and not tunnel.alias_error
        )
        return {
            "url": alias_url if alias_live else tunnel_url,
            "tunnel_url": tunnel_url,
            "scene_url": (
                f"{tunnel_url.rstrip('/')}/{SCENE_PATH}" if tunnel_url else None
            ),
            "provider": tunnel.provider,
            "label": tunnel.label,
            "public": tunnel.public,
            "private": path != "/",
            "guests": server.guest_count(),
            "guest_url": server.guest_url,
            "alias_error": tunnel.alias_error,
        }

    # ------------------------------------------------------------------
    # Asking the share's own link
    # ------------------------------------------------------------------
    def ping_share(
        self, timeout: float = 10.0, attempts: int = 3, wait: float = 2.0
    ) -> Dict[str, Any]:
        server = self._server
        info = server.share_info()
        if info is None:
            return server._failed_ping("nothing is shared")
        if not info["tunnel_url"]:
            return server._failed_ping("the share's link is gone")
        result = server._failed_ping(None)
        result["url"] = info["tunnel_url"]
        url = f"{info['tunnel_url'].rstrip('/')}/manifest.json"
        secure = url.startswith("https:")
        public = secure and bool(info.get("public"))
        for attempt in range(max(1, int(attempts))):
            if attempt:
                time.sleep(wait)
            # Each attempt's own: a miss after an answer must not report the
            # earlier answer's status, round trip or route as its own.
            result.update(
                attempts=attempt + 1, status=None, ms=None, route="local", verified=None
            )
            address = None
            if public:
                from pythontk.net_utils._net_utils import NetUtils

                found = NetUtils.public_addresses(urlparse(url).hostname)
                if found == []:
                    result["error"] = (
                        "public DNS does not have the link's name, so no one "
                        "else's browser can find it"
                    )
                    continue
                if found:
                    address, result["route"] = found[0], "public"
            started = time.perf_counter()
            try:
                try:
                    status, body = server._ask_share(url, timeout, address, verify=True)
                    result["verified"] = True if secure else None
                except urllib.error.URLError as error:
                    if not isinstance(
                        getattr(error, "reason", None), ssl.SSLCertVerificationError
                    ):
                        raise
                    status, body = server._ask_share(
                        url, timeout, address, verify=False
                    )
                    result["verified"] = False
                result["ms"] = round((time.perf_counter() - started) * 1000, 1)
                result["status"] = status
                try:
                    payload = json.loads(body.decode("utf-8"))
                except ValueError:  # a provider's own page, answering 200
                    payload = None
                if not (isinstance(payload, dict) and payload.get("guest") is True):
                    result["error"] = "the link answered, but not with this share"
                    continue
                if payload.get("version") != server.version:
                    result["error"] = (
                        f"the link answered with version {payload.get('version')}, "
                        f"not this server's {server.version}"
                    )
                    continue
                result.update(ok=True, error=None)
                return result
            except urllib.error.HTTPError as error:
                result["ms"] = round((time.perf_counter() - started) * 1000, 1)
                result["status"] = error.code
                result["error"] = f"HTTP {error.code} {error.reason}".strip()
            except (urllib.error.URLError, OSError, ValueError) as error:
                reason = getattr(error, "reason", None) or error
                result["error"] = str(reason) or type(error).__name__
        return result

    @staticmethod
    def ask_share(
        url: str, timeout: float, address: Optional[str], verify: bool
    ) -> Tuple[int, bytes]:
        """GET *url* -- at *address* when given, the name still the one TLS
        and the ``Host`` header carry -- as ``(status, body)``. Raises as
        :func:`urllib.request.urlopen` does."""
        from pythontk.net_utils._net_utils import NetUtils

        if verify:
            context = NetUtils.tls_context()
        else:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "pythontk-preview-ping",
                "Cache-Control": "no-cache",
            },
        )
        if address is None:
            opened = urllib.request.urlopen(request, timeout=timeout, context=context)
        else:
            # No proxy: the request goes to *address* itself, as a guest's does.
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}), _PinnedHTTPSHandler(address, context)
            )
            opened = opener.open(request, timeout=timeout)
        with opened as response:
            return response.status, response.read(1 << 20)

    @staticmethod
    def failed_ping(error: Optional[str]) -> Dict[str, Any]:
        """:meth:`PreviewServer.ping_share`'s result before an answer: nothing
        passed -- *error* says why, ``None`` while the asking is still to come."""
        return {
            "ok": False,
            "url": None,
            "status": None,
            "ms": None,
            "attempts": 0,
            "error": error,
            "route": None,
            "verified": None,
        }


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """An HTTPS connection to a given *address*: TLS and ``Host`` still name
    the URL's host -- how a request reaches the address public DNS gives a
    name, whatever this machine's resolver says."""

    def __init__(self, *args: Any, address: str, context: ssl.SSLContext, **kwargs):
        super().__init__(*args, context=context, **kwargs)
        self._address, self._pinned_context = address, context

    def connect(self) -> None:
        sock = socket.create_connection((self._address, self.port), self.timeout)
        self.sock = self._pinned_context.wrap_socket(sock, server_hostname=self.host)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    """``urllib``'s HTTPS handler, connecting through :class:`_PinnedHTTPSConnection`."""

    def __init__(self, address: str, context: ssl.SSLContext):
        super().__init__(context=context)
        self._address, self._pinned_context = address, context

    def https_open(self, req):
        return self.do_open(
            partial(
                _PinnedHTTPSConnection,
                address=self._address,
                context=self._pinned_context,
            ),
            req,
        )
