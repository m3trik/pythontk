# !/usr/bin/python
# coding=utf-8
"""Sharing a preview by link: the read-only guest listener and its tunnel.

:meth:`share` opens a second listener onto the same serve root
(:meth:`start_guest`) -- reads an allow-list of what the guest manifest names,
refuses every write, counts its tabs apart from the owner's viewer -- and fronts
it with a :class:`~pythontk.ShareTunnel`, whose provider terminates TLS on a
public name. The owner's listener stays on loopback and answers no public name;
who may write is decided by the socket a request arrived on.

One job of :class:`~pythontk.PreviewServer`, composed in ``server.py``; its
methods reach the rest of the server through ``self``.
"""

from __future__ import annotations

import os
import threading
import time
from functools import partial
from typing import Any, Callable, Dict, Optional, Union
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
        modules it imports beside it, and the asset."""
        files = set(self._kernel_files()) if self._viewer else set()
        with self._lock:
            for name, source in self._scripts.items():
                if name not in self.OWNER_SCRIPTS:
                    files.update(self._served_files(name, source))
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
            ``{"url", "tunnel_url", "scene_url", "provider", "label",
            "public", "guests", "guest_url", "alias_error"}``. ``url`` is the
            link to hand out: the alias's address when there is an alias AND
            its last update landed, else the tunnel's own -- a stale alias
            must never be what gets sent. ``scene_url`` is where the share
            serves :meth:`describe_scene` -- the link for someone's tools or
            agent -- always on the tunnel, since an alias is one redirect page
            rather than a site.
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
            # None with tunnel_url: a client that exits between the check
            # above and the read reads as a link that is gone, not a crash.
            "scene_url": (
                f"{tunnel_url.rstrip('/')}/{SCENE_PATH}" if tunnel_url else None
            ),
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
