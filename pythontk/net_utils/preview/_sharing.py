# !/usr/bin/python
# coding=utf-8
"""Sharing a preview by link: the read-only guest listener and its tunnel.

:meth:`share` opens a second listener onto the same serve root
(:meth:`start_guest`) -- reads an allow-list of what the guest manifest names,
refuses every write, counts its tabs apart from the owner's viewer -- and fronts
it with a :class:`~pythontk.ShareTunnel`, whose provider terminates TLS on a
public name. The owner's listener stays on loopback and answers no public name;
who may write is decided by the socket a request arrived on.

A PRIVATE share adds a fresh random path segment to its link
(``https://<name>/<token>/``) and the guest listener answers nothing outside
it, so a link is good for one share only -- the provider's own name, which
Tailscale keeps for good, opens nothing by itself. :meth:`ping_share` asks the
link from this machine whether a guest can reach it.

One job of :class:`~pythontk.PreviewServer`, composed in ``server.py``; its
methods reach the rest of the server through ``self``.
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
from typing import Any, Callable, Dict, Optional, Tuple, Union
from urllib.parse import urlparse

from pythontk.net_utils.preview.routes import (
    SCENE_PATH,
    _VIEWER_ID,
    _PreviewHandler,
    _PreviewHTTPServer,
)


class _SharingMixin:
    """Guest listener, admitted hosts, guest tabs, the share's tunnel.

    A private part of :class:`~pythontk.PreviewServer`; call it through the facade.
    """

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
        """The guest listener on loopback: what a guest sees, from this machine
        -- under a private share's path segment, the only place it answers."""
        port = self.guest_port
        if port is None:
            return None
        return f"http://{self.host}:{port}{self._share_path()}"

    def _share_path(self) -> str:
        """The path a guest's page is served under: ``/``, or a private
        share's ``/<token>/``."""
        token = self._share_token
        return f"/{token}/" if token else "/"

    def _guest_route(self, path: str) -> Optional[str]:
        """*path* as the guest listener routes it -- a private share's segment
        taken off -- or ``None`` when it lies outside the share.

        The segment is compared in constant time: it is the share's only
        secret, and a comparison that stops at the first wrong character
        would time its way to it a character at a time.
        """
        token = self._share_token
        if not token:
            return path
        head, _, rest = path.lstrip("/").partition("/")
        if not secrets.compare_digest(head.encode(), token.encode()):
            return None
        return "/" + rest

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
        """Served paths a guest may fetch besides the page: the kernel modules
        it imports, and what its manifest names -- each script with the
        modules it imports beside it, and every part of the asset."""
        files = set(self._kernel_files()) if self._viewer else set()
        with self._lock:
            for name, source in self._scripts.items():
                if name not in self.OWNER_SCRIPTS:
                    files.update(self._served_files(name, source))
            if self._asset:
                files.add(self._asset)
            files.update(part["url"] for part in self._parts)
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
        from pythontk.net_utils.share_tunnel import ShareTunnel

        if private or alias is False:
            alias = alias_url = None
        elif alias is None:
            alias = os.environ.get(self.ALIAS_ENV, "").strip() or None
        if alias_url is None and alias is not None:
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
                and bool(self._share_token) == bool(private)
            ):
                with self._lock:
                    self._alias_url = alias_url
                return self.share_info()
            generation = self._end_share()
            # Before the listener opens: from its first request, a private
            # share's guest listener answers under the segment only.
            with self._lock:
                self._share_token = (
                    secrets.token_urlsafe(self.SHARE_TOKEN_BYTES) if private else None
                )
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
            self._share_token = None
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
        with self._lock:
            tunnel, alias_url = self._tunnel, self._alias_url
            path = self._share_path()
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
            "guests": self.guest_count(),
            "guest_url": self.guest_url,
            "alias_error": tunnel.alias_error,
        }

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
        info = self.share_info()
        if info is None:
            return self._failed_ping("nothing is shared")
        if not info["tunnel_url"]:
            return self._failed_ping("the share's link is gone")
        result = self._failed_ping(None)
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
                    status, body = self._ask_share(url, timeout, address, verify=True)
                    result["verified"] = True if secure else None
                except urllib.error.URLError as error:
                    if not isinstance(
                        getattr(error, "reason", None), ssl.SSLCertVerificationError
                    ):
                        raise
                    status, body = self._ask_share(url, timeout, address, verify=False)
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
                if payload.get("version") != self.version:
                    result["error"] = (
                        f"the link answered with version {payload.get('version')}, "
                        f"not this server's {self.version}"
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
    def _ask_share(
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
    def _failed_ping(error: Optional[str]) -> Dict[str, Any]:
        """:meth:`ping_share`'s result before an answer: nothing passed --
        *error* says why, ``None`` while the asking is still to come."""
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

    @property
    def share_url(self) -> Optional[str]:
        """The link to hand out while sharing, else ``None``."""
        info = self.share_info()
        return info["url"] if info else None


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
