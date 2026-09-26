# !/usr/bin/python
# coding=utf-8
"""Expose a local HTTP port at a public HTTPS link, through a tunnel CLI.

:class:`ShareTunnel` is the generic half of "send someone a link to what is
running on this machine": it starts a tunnel client in front of a loopback
port, learns the link the client prints, and keeps the client alive exactly as
long as the share -- never longer. It knows nothing about what is served; the
WebXR preview is one consumer (:meth:`pythontk.PreviewServer.share`).

Why a tunnel rather than a LAN bind
-----------------------------------
A LAN address reaches nobody outside the building, and -- the deciding reason
-- a browser grants the APIs a 3D viewer needs most, ``navigator.xr`` above
all, only to a *secure context*: localhost, or HTTPS with a real certificate.
A tunnel provides the second for free. The provider terminates TLS on a name it
owns (``*.trycloudflare.com``, ``*.ts.net``) and forwards plain HTTP to the
loopback port, so a guest needs nothing but a browser -- desktop, phone or a
standalone headset -- and this machine needs no open port, no certificate and
no router change.

Providers
---------
A registry of plain values (:attr:`ShareTunnel.PROVIDERS`), one entry per CLI:

* ``cloudflared`` -- a Cloudflare *quick tunnel*: no account, and a random
  ``https://<words>.trycloudflare.com`` link per start. TLS ends at
  Cloudflare's edge, which therefore sees the traffic.
* ``tailscale_funnel`` -- Tailscale Funnel: a STABLE public link
  (``https://<machine>.<tailnet>.ts.net``, bookmarkable) with TLS ending on
  this machine. Funnel must be enabled for the tailnet; relayed, so slower.
* ``tailscale_serve`` -- the same name, reachable only inside the tailnet.

An entry is data -- the arguments, the pattern the link is printed in, what
"ready" looks like, where to install it -- so a provider is added by an entry,
never by a branch. The link patterns are deliberately narrow: cloudflared's own
banner names ``developers.cloudflare.com`` and its errors name
``api.trycloudflare.com``, and "the first https URL printed" is neither.

Lifetime
--------
The client runs through :meth:`pythontk.AppLauncher.spawn`, so on Windows it
dies with this process however this process ends. That matters more here than
for most children: an orphaned tunnel keeps a public link pointing at a
loopback port, and whatever binds that port next would be published through
it. Off Windows a crash can still orphan it; a normal exit is covered by
:meth:`ShareTunnel.stop` running at ``atexit``.

The stable alias ("tiny link")
------------------------------
A quick tunnel's link changes on every start, and typing forty characters into
a headset is where sharing dies in practice. *alias* names where to keep a
one-line redirect to the current link -- a path on a web root you already
serve, ``user@host:/path`` over the system ``scp``, or a callable -- so
``https://you.example/vr`` is bookmarked once and always lands on the live
share. On stop the alias is rewritten to a page that says the share ended and
checks back, so a guest who opens it early lands on the next share by itself.

Example:
    >>> with ShareTunnel(port=8118) as tunnel:
    ...     print(tunnel.url)  # https://<words>.trycloudflare.com
"""

from __future__ import annotations

import atexit
import html
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urlparse

from pythontk.core_utils.app_launcher import AppLauncher
from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.core_utils.process_stream import OutputStream, ProcessReader

#: Where an alias can be kept: a local path, ``[user@]host:/path``, or a
#: callable handed the link (``None`` when the share stops).
AliasTarget = Union[str, "os.PathLike[str]", Callable[[Optional[str]], Any]]

_CLOUDFLARED_RELEASES = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download"
)


class _ShareTunnelInternal:
    """Helpers for :class:`ShareTunnel`: the provider table, the wait, the alias."""

    #: Seconds :meth:`ShareTunnel.stop` waits for the client to exit before
    #: killing it.
    _STOP_GRACE = 5.0

    #: Output lines kept for diagnosis (:meth:`ShareTunnel.output`).
    _HISTORY = 200

    #: Seconds a start waits for a brand-new link to resolve in public DNS,
    #: for a provider whose ``propagates`` says it mints one. Measured
    #: 2026-09-23: a quick tunnel's name answered NXDOMAIN for 6 s after its
    #: connection registered on one run, 18 s on the next.
    _DNS_WAIT = 30.0

    #: Seconds a start that handed its provider's one-time step to ``on_step``
    #: waits for the user to take it. The step is a person's -- a sign-in, a
    #: switch in an admin console -- so the link's own ``timeout`` does not
    #: bound it.
    _STEP_WAIT = 600.0

    #: ``[user@]host:/path`` -- scp's own spelling. The host needs two
    #: characters, so a Windows drive (``C:\\``) is never read as one, and a
    #: ``scheme://`` is not a host either.
    _REMOTE = re.compile(r"^(?:[\w.-]+@)?[\w.-]{2,}:(?![\\/]{2})")

    #: What a failed client's output says when the MACHINE is in the way rather
    #: than the provider -> what to do about it. Checked on a failed start and
    #: led with, because the raw line (a Go socket error) names neither.
    _DIAGNOSES = (
        (
            # WinError 10013, as Go spells it. Measured 2026-09-23 on a machine
            # whose firewall denies outbound by default: a freshly installed
            # cloudflared exited at once with "dial tcp ...:443: connectex: An
            # attempt was made to access a socket in a way forbidden by its
            # access permissions."
            re.compile(r"forbidden by its access permissions", re.IGNORECASE),
            "The firewall blocked {executable} from the network (WinError "
            "10013 -- outbound is denied unless a program is allowed). Allow "
            "that program outbound, then share again.",
        ),
    )

    @classmethod
    def _spec(cls, provider: str) -> Dict[str, Any]:
        try:
            return cls.PROVIDERS[provider]
        except KeyError:
            raise KeyError(
                f"Unknown tunnel provider {provider!r}; known: "
                f"{', '.join(sorted(cls.PROVIDERS))}."
            ) from None

    def _check_stopped(self, stops: int) -> None:
        """Raise when :meth:`stop` ran since the start that read *stops* began,
        or :meth:`cancel` ever did."""
        if self._cancelled or self._stops != stops:
            raise RuntimeError(
                f"{self.label}: the share was stopped before its link was ready."
            )

    def _step_required(
        self, spec: Dict[str, Any], url: str, waited: bool = False
    ) -> "ShareTunnel.StepRequired":
        """The :class:`StepRequired` for the page *url*: a step still to take,
        or -- *waited* -- one a start waited on in vain."""
        what = (
            "did not get past its one-time step"
            if waited
            else "needs a one-time step first"
        )
        return self.StepRequired(
            self._failed(f"{spec['label']} {what}: open {url} , then share again."),
            url=url,
            label=spec["label"],
        )

    def _await_link(self, spec: Dict[str, Any], stops: int) -> str:
        """Block until the client prints its link (and, where the provider
        says what "ready" looks like, that too); the link.

        Watches the process, the provider's own "do this first" line and
        :meth:`stop` as well as the clock. A client that exits -- not logged
        in, a flag it does not know -- fails at once with what it printed. One
        that stops to wait on the user (Tailscale, when Serve or Funnel is not
        yet enabled for the tailnet, prints an enable link and waits until
        someone uses it) carries on by itself once the step is taken, so the
        step goes to ``on_step`` and the start waits on -- up to
        :attr:`_STEP_WAIT`, the step being a person's. With no ``on_step`` it
        fails at once instead, naming the page: whoever pressed Share is
        looking at a busy indicator, not at the client's output, so waiting
        out the timeout would only hide the one step that unblocks it.
        """
        link_pattern = re.compile(spec["url"], re.IGNORECASE)
        ready_pattern = re.compile(spec["ready"]) if spec.get("ready") else None
        action_pattern = (
            re.compile(spec["action"], re.IGNORECASE) if spec.get("action") else None
        )
        found: Dict[str, str] = {}
        link, ready, blocked = threading.Event(), threading.Event(), threading.Event()

        def watch(_source: str, line: str) -> None:
            if not link.is_set():
                match = link_pattern.search(line)
                if match:
                    found["url"] = (
                        match.group(1) if match.groups() else match.group(0)
                    ).rstrip("/")
                    link.set()
            if ready_pattern is not None and ready_pattern.search(line):
                ready.set()
            if action_pattern is not None and not blocked.is_set():
                match = action_pattern.search(line)
                if match:
                    found["action"] = match.group(0)
                    blocked.set()

        unsubscribe = self._stream.subscribe(watch, replay_history=True)
        deadline = time.monotonic() + self.timeout
        told = False  # on_step has the step: the wait is the user's now
        try:
            for event, what in ((link, "link"), (ready, "connection")):
                if event is ready and ready_pattern is None:
                    continue
                while not event.wait(0.1):
                    self._check_stopped(stops)
                    if blocked.is_set() and not told:
                        time.sleep(0.2)  # the rest of the provider's message
                        step = self._step_required(spec, found["action"])
                        if self.on_step is None:
                            raise step
                        self.on_step(step)
                        self._check_stopped(stops)  # a hook that stopped it
                        told = True
                        deadline = time.monotonic() + self._STEP_WAIT
                    if self._process.poll() is not None:
                        # Its last lines may still be in the pipe; the reader
                        # ends at the pipe's end, which the exit brings.
                        self._reader.join(timeout=2)
                        if event.is_set():
                            break
                        if blocked.is_set():  # named the step, then gave up on it
                            raise self._step_required(
                                spec, found["action"], waited=True
                            )
                        raise RuntimeError(
                            self._failed(
                                f"{spec['label']} exited (code "
                                f"{self._process.returncode}) before it was ready."
                            )
                        )
                    if time.monotonic() > deadline:
                        if told:
                            raise self._step_required(
                                spec, found["action"], waited=True
                            )
                        raise TimeoutError(
                            self._failed(
                                f"{spec['label']} reported no {what} within "
                                f"{self.timeout:g}s."
                            )
                        )
        finally:
            unsubscribe()
        return found["url"]

    def _await_dns(self, url: str, stops: int) -> None:
        """Hold a brand-new link until public DNS answers it, or until a stop.

        A guest who opens the link sooner gets "site can't be reached" -- and
        keeps getting it after the name appears, because the miss is cached by
        their OS and browser. So the link is announced only once a public
        resolver answers it, asked directly (:meth:`NetUtils.resolves_publicly`)
        so this machine caches no miss either. Best effort by design: when no
        resolver can be asked, or the wait runs out, the link is announced
        anyway -- the tunnel is up, and the name is only moments away.
        """
        from pythontk.net_utils._net_utils import NetUtils

        host = urlparse(url).hostname
        deadline = time.monotonic() + self._DNS_WAIT
        while True:
            answer = NetUtils.resolves_publicly(host)
            if answer is True:
                return
            if answer is None:
                self.logger.debug("Public DNS unreachable; not waiting on %s.", host)
                return
            if time.monotonic() > deadline:
                self.logger.warning(
                    "%s does not resolve yet; the link may need a few more "
                    "seconds before it opens.",
                    host,
                )
                return
            time.sleep(1.0)
            self._check_stopped(stops)

    def _failed(self, message: str) -> str:
        """*message* and the client's last lines, led by the fix when its
        output names a machine-side cause (:attr:`_DIAGNOSES`)."""
        printed = "\n".join(self.output(self._HISTORY))
        for pattern, hint in self._DIAGNOSES:
            if pattern.search(printed):
                message = f"{hint.format(executable=self._executable)}\n{message}"
                break
        return f"{message}\n{self._tail()}"

    def _tail(self, lines: int = 12) -> str:
        return "\n".join(self.output(lines)) or "(it printed nothing)"

    def _teardown(self) -> None:
        """Stop the client if it is still running, and forget the link."""
        process, self._process = self._process, None
        self._url = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=self._STOP_GRACE)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning("Tunnel client %s did not exit.", process.pid)

    def _announce(self, url: Optional[str]) -> None:
        """Point the alias at *url* (``None``: the share ended).

        Never raises: the share works without its alias, so a web root that is
        offline must not cost the link -- the failure is logged and kept on
        :attr:`ShareTunnel.alias_error` for the caller to show.
        """
        if url is not None and self._url != url:
            return  # stopped by its own on_url: the stop announced the end
        self._alias_error = None
        if self.alias is None:
            return
        try:
            if callable(self.alias):
                self.alias(url)
            else:
                self._write_alias(self.alias, url)
        except Exception as error:  # noqa: BLE001 -- see the docstring
            self._alias_error = str(error) or type(error).__name__
            self.logger.warning(
                "The share's alias was not updated (%s): %s", self.alias, error
            )

    @classmethod
    def _write_alias(
        cls, target: Union[str, "os.PathLike[str]"], url: Optional[str]
    ) -> None:
        """Write the redirect page for *url* to *target* (a path or ``user@host:/path``)."""
        page = cls._redirect_page(url)
        target = os.fspath(target)
        if cls._REMOTE.match(target):
            cls._scp(page, target)
            return
        path = Path(target)
        if path.is_dir():
            path = path / "index.html"
        if not path.parent.is_dir():
            raise FileNotFoundError(f"The alias folder does not exist: {path.parent}")
        from pythontk.file_utils._file_utils import FileUtils

        # Atomic: a web server may be reading it.
        FileUtils.atomic_write_text(str(path), page)

    @staticmethod
    def _scp(page: str, target: str) -> None:
        """Copy *page* to *target* with the system ``scp``, never prompting.

        The system client rather than a Python SSH library: it already has the
        user's keys, agent and ``~/.ssh/config``, and ``-B`` makes a missing
        key a prompt-free failure instead of a hung DCC.
        """
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        artifacts = TempArtifacts("share_alias", policy="scoped")
        local = artifacts.path(extension=".html")
        try:
            with open(local, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(page)
            result = AppLauncher.run(
                "scp",
                ["-q", "-B", "-o", "ConnectTimeout=8", local, target],
                timeout=30,
                hide_window=True,
            )
        finally:
            artifacts.cleanup()
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"scp to {target} failed ({result.returncode}): {detail}"
            )

    @staticmethod
    def _redirect_page(url: Optional[str]) -> str:
        """A page that forwards to *url*; for ``None``, one that says the share
        ended and reloads itself until the alias names a live one again."""
        style = (
            "body{margin:0;min-height:100vh;display:flex;align-items:center;"
            "justify-content:center;padding:24px;text-align:center;"
            "background:#15171a;color:#e8eaed;font:16px/1.5 system-ui,sans-serif}"
            "a{color:#7cc3ff}"
        )
        head = (
            '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<meta name="robots" content="noindex">\n'
        )
        if url is None:
            # The reload carries a fresh query so a cached copy of THIS page is
            # never what answers it: the next share rewrites the file in place.
            return (
                f"{head}<title>Share ended</title>\n<style>{style}</style>\n</head>\n"
                "<body><p>Nothing is being shared right now.<br>"
                "This page checks again every 15 seconds.</p>\n<script>"
                "setTimeout(() => location.replace(location.pathname + '?t=' + "
                "Date.now()), 15000);</script>\n</body>\n</html>\n"
            )
        attribute = html.escape(url, quote=True)
        script = json.dumps(url).replace("<", "\\u003c")
        return (
            f'{head}<meta http-equiv="refresh" content="0; url={attribute}">\n'
            f"<title>Opening the shared view</title>\n<style>{style}</style>\n"
            f'</head>\n<body><p>Opening <a href="{attribute}">{attribute}</a>'
            f"&hellip;</p>\n<script>location.replace({script});</script>\n"
            "</body>\n</html>\n"
        )


class ShareTunnel(LoggingMixin, _ShareTunnelInternal):
    """Expose ``http://<host>:<port>`` at a public HTTPS link while running.

    Parameters:
        port: The local port to share.
        provider: A :attr:`PROVIDERS` name. ``None`` or ``"auto"`` takes the
            machine's default (:meth:`resolve_provider`).
        host: The loopback address the port is bound on.
        timeout: Seconds :meth:`start` waits for the provider's link.
        alias: Where to keep a stable redirect to the link -- a local path (a
            folder gets ``index.html``), ``[user@]host:/path`` (the system
            ``scp``), or a callable handed the link and, on stop, ``None``.
            See the module docstring.
        on_url: Called with the link as soon as it is known -- BEFORE the alias
            announces it, so a server can admit the name a guest will arrive
            under first -- and with ``None`` after the client stops. A raise
            on start fails the start. It and a callable *alias* run under the
            tunnel's lock, so a stop from another thread cannot interleave
            with them; neither may wait on a thread that stops this tunnel.
        on_step: Handed a :class:`StepRequired` when the provider stops on a
            one-time step in a browser (Tailscale, before the tailnet enables
            Funnel), so a caller can show its ``url``; the start then waits on,
            up to :attr:`_STEP_WAIT`, and returns the link once the step is
            taken. ``None``: such a start fails at once as that
            :class:`StepRequired`. Runs on the thread running :meth:`start`,
            under the same lock as *on_url*; a raise fails the start.
    """

    #: The tunnel CLIs this class drives: name -> plain values.
    #:
    #: ``label`` names it to a person; ``executable`` is the command, found on
    #: PATH, then at ``paths`` (``%VAR%`` expanded), then in
    #: :class:`AppInstaller`'s catalog; ``args`` are formatted with ``{host}``
    #: and ``{port}``; ``url`` is the pattern the link is printed in (group 1
    #: when it has one); ``ready`` is the line that says guests can connect, or
    #: None when the link itself says so; ``action``, optional, is the link a
    #: provider prints when it stops to wait on the user (handed to
    #: ``on_step`` and waited on, else a start fails at once naming it);
    #: ``propagates`` marks a provider that mints a new DNS
    #: name per start, which a start waits to see resolve in public DNS before
    #: announcing it; ``public`` is whether anyone can open
    #: the link or only members of a private network; ``install`` is where to
    #: get it; ``managed`` is an :meth:`AppInstaller.ensure` platform table when
    #: a download can install it for the user.
    PROVIDERS: Dict[str, Dict[str, Any]] = {
        "cloudflared": {
            "label": "Cloudflare quick tunnel",
            "executable": "cloudflared",
            # A DCC's child has no business replacing its own binary mid-session.
            "args": ["tunnel", "--no-autoupdate", "--url", "http://{host}:{port}"],
            "url": r"(https://(?!api\.)[a-z0-9]+(?:-[a-z0-9]+)+\.trycloudflare\.com)",
            # The link is printed BEFORE the edge accepts a connection; until
            # this line a guest following it gets Cloudflare's error page.
            "ready": r"Registered tunnel connection",
            # Every start mints a new DNS name, which resolves only seconds
            # later (see _DNS_WAIT).
            "propagates": True,
            "public": True,
            "install": "https://developers.cloudflare.com/cloudflare-one/"
            "connections/connect-networks/downloads/",
            "managed": {
                "windows": {
                    "url": f"{_CLOUDFLARED_RELEASES}/cloudflared-windows-amd64.exe",
                    "type": "binary",
                },
                "linux": {
                    "url": f"{_CLOUDFLARED_RELEASES}/cloudflared-linux-amd64",
                    "type": "binary",
                },
                "darwin": {
                    "url": f"{_CLOUDFLARED_RELEASES}/cloudflared-darwin-amd64.tgz",
                    "type": "tar.gz",
                },
            },
        },
        "tailscale_funnel": {
            "label": "Tailscale Funnel",
            "executable": "tailscale",
            "paths": (
                r"%ProgramFiles%\Tailscale\tailscale.exe",
                "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
            ),
            # Foreground (no --bg): the funnel lives exactly as long as this
            # client, and tailscaled removes it when the client goes -- killed
            # included -- so nothing outlives the share in the tailnet's config.
            "args": ["funnel", "http://{host}:{port}"],
            "url": r"^\s*(https://[a-z0-9.-]+\.ts\.net(?::\d+)?)/?\s*$",
            "ready": None,
            # Printed, then waited on, when the tailnet has not enabled the
            # feature yet (measured 2026-09-23: "Serve is not enabled on your
            # tailnet. To enable, visit: https://login.tailscale.com/f/serve?...").
            "action": r"https://login\.tailscale\.com/f/\S+",
            "public": True,
            "install": "https://tailscale.com/download",
        },
        "tailscale_serve": {
            "label": "Tailscale (tailnet only)",
            "executable": "tailscale",
            "paths": (
                r"%ProgramFiles%\Tailscale\tailscale.exe",
                "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
            ),
            "args": ["serve", "http://{host}:{port}"],
            "url": r"^\s*(https://[a-z0-9.-]+\.ts\.net(?::\d+)?)/?\s*$",
            "ready": None,
            "action": r"https://login\.tailscale\.com/f/\S+",
            "public": False,
            "install": "https://tailscale.com/download",
        },
    }

    #: Tried in order when nothing names a provider; the first INSTALLED wins.
    #: Public providers only -- a default has to produce a link anyone can open.
    PREFERENCE = ("cloudflared", "tailscale_funnel")

    #: The environment variable naming this machine's default provider.
    PROVIDER_ENV = "PYTHONTK_SHARE_PROVIDER"

    class StepRequired(RuntimeError):
        """A share waiting on a one-time step in a browser: Tailscale, when the
        tailnet has not enabled Serve or Funnel yet, prints the page that does
        it and waits. ``url`` is that page and ``label`` the provider's; the
        message, its first line the step, is user-facing -- so a panel can
        offer to open the page rather than only report it.

        What ``on_step`` is handed; raised by a start with no ``on_step``, and
        by one whose step was not taken in time."""

        def __init__(self, message: str, url: str, label: str):
            super().__init__(message)
            self.url = url
            self.label = label

    def __init__(
        self,
        port: int,
        provider: Optional[str] = None,
        host: str = "127.0.0.1",
        timeout: float = 45.0,
        alias: Optional[AliasTarget] = None,
        on_url: Optional[Callable[[Optional[str]], Any]] = None,
        on_step: Optional[Callable[["ShareTunnel.StepRequired"], Any]] = None,
    ):
        self.port = int(port)
        self.host = host
        self.provider = self.resolve_provider(provider)
        self.timeout = float(timeout)
        self.alias = alias
        self.on_url = on_url
        self.on_step = on_step
        #: Counts stops, read without the lock: a start compares it to the
        #: count it began with, so a stop from another thread ends a start
        #: that is still waiting -- which holds the lock -- at its next check.
        self._stops = 0
        #: Set by :meth:`cancel`, for good: this tunnel starts no more.
        self._cancelled = False
        #: Re-entrant: :meth:`start` and :meth:`stop` call *on_url* and the
        #: alias under it, and either may stop this tunnel.
        self._lock = threading.RLock()
        self._process: Optional[subprocess.Popen] = None
        self._stream: Optional[OutputStream] = None
        self._reader: Optional[ProcessReader] = None
        self._url: Optional[str] = None
        self._alias_error: Optional[str] = None
        #: The CLI the last start ran -- named when the machine blocked it.
        self._executable: Optional[str] = None

    # ------------------------------------------------------------------
    # Providers
    # ------------------------------------------------------------------
    @classmethod
    def resolve_provider(cls, provider: Optional[str] = None) -> str:
        """The provider a share uses.

        *provider* when named; else :attr:`PROVIDER_ENV`; else the first
        installed entry of :attr:`PREFERENCE`. With none installed the answer
        is the first preference, so the refusal that follows names the install
        that is easiest to make.

        Raises:
            KeyError: A name (from either source) that is not a provider.
        """
        name = provider if provider not in (None, "", "auto") else None
        name = name or os.environ.get(cls.PROVIDER_ENV, "").strip() or None
        if name and name != "auto":
            cls._spec(name)
            return name
        for candidate in cls.PREFERENCE:
            if cls.executable(candidate):
                return candidate
        return cls.PREFERENCE[0]

    @classmethod
    def executable(cls, provider: str) -> Optional[str]:
        """Path to *provider*'s CLI, or ``None`` when it is not installed."""
        spec = cls._spec(provider)
        command = spec["executable"]
        found = AppLauncher.find_app(command)
        if found:
            return found
        for candidate in spec.get("paths", ()):
            path = os.path.expandvars(candidate)
            if os.path.isfile(path):
                return path
        from pythontk.core_utils.app_installer import AppInstaller

        return AppInstaller.get_path(command)

    @classmethod
    def not_installed_error(cls, provider: str, detail: str = "") -> FileNotFoundError:
        """The fix-shaped error for a provider whose CLI is missing."""
        spec = cls._spec(provider)
        return FileNotFoundError(
            f"{spec['label']} needs the '{spec['executable']}' command{detail}. "
            f"Install it from {spec['install']}"
            + (
                " -- or let a panel download it for you."
                if spec.get("managed")
                else "."
            )
        )

    @classmethod
    def settle(
        cls,
        provider: Optional[str] = None,
        prompt: Union[bool, Callable[[str], bool]] = True,
        refused: Optional[Callable[[str], Any]] = None,
        installed: Optional[Callable[[str], Any]] = None,
    ) -> Optional[str]:
        """The provider a share may go ahead with, offering the download when
        its CLI is missing.

        The panel-side step, shaped as :meth:`ImgUtils.settle_ktx2_encoder` is:
        a missing tool is OFFERED, never dead-ended in a URL -- where a managed
        download exists (cloudflared: one executable). A provider that is a
        system service (Tailscale) is never installed from here; the refusal
        names where to get it.

        Parameters:
            provider: As :meth:`resolve_provider`.
            prompt: Consent for the download -- ``True`` asks on the console,
                ``False`` needs none, a callable ``(question) -> bool`` is asked
                instead (a panel passes its dialog). See
                :meth:`AppInstaller.consent`.
            refused: Receives the fix-shaped message when the CLI is still
                missing (declined, failed, or not installable here).
            installed: Receives the path of a CLI this call installed.

        Returns:
            The provider's name when its CLI is available now, else ``None`` --
            also for a name no provider has (a mistyped
            :attr:`PROVIDER_ENV`), which is a refusal, not a crash.
        """
        try:
            name = cls.resolve_provider(provider)
        except KeyError as error:
            if refused is not None:
                refused(error.args[0])
            return None
        if cls.executable(name):
            return name
        try:
            path = cls._install(name, prompt)
        except FileNotFoundError as error:
            if refused is not None:
                refused(str(error))
            return None
        if installed is not None:
            installed(path)
        return name

    @classmethod
    def _install(cls, provider: str, prompt: Union[bool, Callable[[str], bool]]) -> str:
        """Download *provider*'s CLI with consent; its path.

        Raises:
            FileNotFoundError: No managed download, declined, or it failed.
        """
        from pythontk.core_utils.app_installer import AppInstaller

        spec = cls._spec(provider)
        if not spec.get("managed"):
            raise cls.not_installed_error(provider, ", which is not installed")
        answer = AppInstaller.consent(
            prompt,
            f"Sharing needs {spec['label']} ('{spec['executable']}'), which is "
            "not installed. Download it into the pythontk tools folder now?",
        )
        if answer is None:
            raise cls.not_installed_error(
                provider,
                ", and no interactive console is available to confirm the "
                "download (pass prompt=False to install non-interactively)",
            )
        if not answer:
            raise cls.not_installed_error(provider, " and the download was declined")
        try:
            return AppInstaller.ensure(spec["executable"], platforms=spec["managed"])
        except (LookupError, RuntimeError, OSError) as error:
            raise cls.not_installed_error(
                provider, f", and the download failed ({error})"
            ) from error

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def label(self) -> str:
        """The provider's name for a person."""
        return self._spec(self.provider)["label"]

    @property
    def public(self) -> bool:
        """Whether anyone can open the link, or only a private network's members."""
        return bool(self._spec(self.provider).get("public"))

    @property
    def is_running(self) -> bool:
        """True while the client runs and has a link -- False once it exits,
        on its own or not."""
        process = self._process
        return self._url is not None and process is not None and process.poll() is None

    @property
    def url(self) -> Optional[str]:
        """The public link, or ``None`` when not running."""
        return self._url if self.is_running else None

    @property
    def netloc(self) -> Optional[str]:
        """The ``Host`` a request through the tunnel carries, lowercase."""
        url = self.url
        return urlparse(url).netloc.lower() if url else None

    @property
    def alias_error(self) -> Optional[str]:
        """Why the last alias update failed, or ``None``."""
        return self._alias_error

    def output(self, lines: int = 40) -> List[str]:
        """The client's most recent output -- where a provider says what is wrong."""
        if self._stream is None:
            return []
        return [line for _source, line in self._stream.history()[-lines:]]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> str:
        """Start the client; the link, once guests can reach it. Idempotent.

        Raises:
            FileNotFoundError: The provider's CLI is not installed; the message
                names the install.
            ShareTunnel.StepRequired: The provider waits on a one-time step in
                a browser (Funnel not enabled for the tailnet) and there is no
                :attr:`on_step` to hand it to -- or it was not taken within
                :attr:`_STEP_WAIT`. ``url`` is the page that takes it. A
                ``RuntimeError``, so a caller catching those still does.
            RuntimeError: The client exited before it was ready; the message
                carries its last lines, which is where a provider says why
                (not logged in, a blocked network). Also when :meth:`stop` or
                :meth:`cancel` ran while it waited.
            TimeoutError: No link within :attr:`timeout`; the client is stopped.
        """
        stops = self._stops  # a stop() from here on ends this start
        with self._lock:
            if self.is_running:
                return self._url
            self._teardown()  # a client that exited on its own
            self._check_stopped(stops)
            executable = self.executable(self.provider)
            if not executable:
                raise self.not_installed_error(
                    self.provider, ", which is not installed"
                )
            self._executable = executable
            spec = self._spec(self.provider)
            args = [str(a).format(host=self.host, port=self.port) for a in spec["args"]]
            self._stream = OutputStream(history=self._HISTORY)
            self._process = AppLauncher.spawn(executable, args)
            self._reader = ProcessReader(
                self._process.stdout, self._stream, self.provider
            )
            self._reader.start()
            try:
                url = self._await_link(spec, stops)
                if spec.get("propagates"):
                    self._await_dns(url, stops)
                self._check_stopped(stops)  # one that landed as the link did
            except BaseException:
                self._teardown()
                raise
            self._url = url
            # Announced under the lock, so a stop() from another thread lands
            # wholly before this or wholly after it -- never between, where
            # the stop's "ended" would be overwritten by a link it just killed.
            if self.on_url is not None:
                try:
                    self.on_url(url)
                except BaseException:
                    self._teardown()
                    raise
            atexit.register(self.stop)
            self.logger.info(
                "Sharing http://%s:%s at %s (%s).",
                self.host,
                self.port,
                url,
                spec["label"],
            )
            self._announce(url)
        return url

    def stop(self) -> None:
        """Stop the client and retire the link; an alias then says the share
        ended. Idempotent. A :meth:`start` still waiting on another thread
        ends first, rather than holding this until its wait runs out; a later
        start starts afresh."""
        self._stops += 1
        with self._lock:
            was_live = self._url is not None
            self._teardown()
            atexit.unregister(self.stop)
            if not was_live:
                return
            self.logger.info("Stopped sharing http://%s:%s.", self.host, self.port)
            if self.on_url is not None:
                try:
                    self.on_url(None)
                except Exception as error:  # noqa: BLE001 -- stopping must finish
                    self.logger.warning("on_url raised while stopping: %s", error)
            self._announce(None)

    def cancel(self) -> None:
        """Retire this tunnel before its link is up, from any thread, without
        waiting: a :meth:`start` still waiting stops its client and raises,
        and one that has not begun raises before it spawns anything. Never
        blocks, so a caller holding a lock of its own can use it; a link
        already up is :meth:`stop`'s to take down."""
        self._cancelled = True

    def __enter__(self) -> "ShareTunnel":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()
