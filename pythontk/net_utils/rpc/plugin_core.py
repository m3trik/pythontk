# !/usr/bin/python
# coding=utf-8
"""The in-application half of the RPC pair: registry + marshaller + server.

:mod:`.client` drives a plugin-hosted RPC server from the *outside*. This module
is what runs on the *inside* -- inside Marmoset Toolbag, inside Substance
Painter, inside any host that can import a Python package and keep it alive.
Together they are one protocol with two ends, and keeping both ends in one
package is what makes the wire format impossible to drift.

Three collaborators, composed by one facade:

* :class:`OpRegistry` -- decorator-based op table with signature introspection.
* :class:`MainThreadMarshaller` -- hop a call onto the host's Qt main thread.
* :class:`RpcPlugin` -- the facade a host plugin instantiates: owns a registry
  and a marshaller, serves the three routes, and gates auto-start on actually
  being hosted.

Everything that differs between hosts is **data** on :class:`RpcPlugin`
(``label`` / ``host_module`` / ``env_prefix`` / ``default_port``), so one core
serves every host rather than one near-copy per host.

Deployment -- READ THIS BEFORE EDITING
--------------------------------------
This file is consumed two ways, and the second one is why it must stay
**standard-library only** and free of ``pythontk`` imports:

1. **Imported** -- by a process of ours that hosts a service
   (``look_capture serve``), and by this package's tests.
2. **Staged** -- by a plugin that runs inside a host app (the ``marmoset_rpc``
   and ``substance_rpc`` payloads of mayatk / blendertk, and extapps' vendored
   ``substance_rpc``). Those payloads are installed into, or loaded by, Toolbag /
   Painter and must be self-contained: there is no ``pythontk`` on the host's
   ``sys.path``. So this file is copied verbatim into each payload as
   ``_rpc_core.py``.

The staged copies are byte-identical artifacts, not forks. Regenerate with
``python m3trik/scripts/sync_rpc_core.py``; ``--check`` fails on drift (CI gate).
**Never hand-edit a staged ``_rpc_core.py``** -- edit this file and re-stage.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import queue
import selectors
import socket
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

__all__ = ["OpRegistry", "MainThreadMarshaller", "RpcInstances", "RpcPlugin"]


# --------------------------------------------------------------------- registry
class _OpRegistryInternal(object):
    """Internal helpers for :class:`OpRegistry`."""

    @staticmethod
    def _describe_one(name, fn):
        """Inspect *fn* and return its JSON-friendly description dict."""
        try:
            params = [
                {
                    "name": p.name,
                    "default": (
                        "<required>"
                        if p.default is inspect.Parameter.empty
                        else repr(p.default)
                    ),
                }
                for p in inspect.signature(fn).parameters.values()
            ]
        except (TypeError, ValueError):
            params = []
        return {
            "name": name,
            "doc": (inspect.getdoc(fn) or "").strip(),
            "params": params,
        }


class OpRegistry(_OpRegistryInternal):
    """The callable surface a host plugin exposes over RPC.

    An **instance**, not a module of globals: a process may host more than one
    plugin (an agent inspecting both cores, a test suite exercising two hosts in
    one interpreter), and a shared module-level table would silently merge them.
    Per-instance state is also what lets tests build a throwaway registry instead
    of reaching for a ``clear()`` hook.
    """

    def __init__(self):
        self._ops = {}
        #: Ops answered on the request's own thread (``main_thread=False``).
        self._off_thread = set()

    def register(self, name, main_thread=True):
        """Decorator registering the wrapped function under *name*.

        Names are dot-namespaced (``"system.ping"``, ``"scene.list_materials"``)
        so the client can group related calls. A duplicate name raises rather
        than overwriting -- a typo that silently shadows a real op would surface
        much later as the feature quietly not happening.

        Parameters:
            name: The op's name on the wire.
            main_thread: False for an op that touches nothing of the host's
                (it reads the plugin's own bookkeeping: a job's status, a
                cancel flag): it is answered on the request's thread, at once,
                even while the host's main thread is busy. Every other op waits
                for that thread, so a client polling one through a long
                synchronous step there would time out and take the host for
                gone.
        """

        def decorator(fn):
            if name in self._ops:
                raise ValueError(f"Op {name!r} is already registered.")
            self._ops[name] = fn
            if not main_thread:
                self._off_thread.add(name)
            return fn

        return decorator

    def needs_main_thread(self, name):
        """False for an op registered ``main_thread=False``."""
        return name not in self._off_thread

    def get(self, name):
        """Return the op callable registered under *name*, or ``None``."""
        return self._ops.get(name)

    def all_ops(self):
        """Every registered op name, sorted."""
        return sorted(self._ops)

    def describe(self, name=None):
        """Describe one op (``None`` for all) as ``{name, doc, params}``.

        Enough for an agent or a human to discover the surface without reading
        the source. Defaults are stringified so the result round-trips through
        JSON even when a default is not itself serialisable.
        """
        if name is not None:
            fn = self._ops.get(name)
            return None if fn is None else self._describe_one(name, fn)
        return [self._describe_one(n, self._ops[n]) for n in sorted(self._ops)]


# -------------------------------------------------------------------- marshaller
class _MainThreadMarshallerInternal(object):
    """Internal helpers for :class:`MainThreadMarshaller`."""

    @staticmethod
    def _qtcore():
        """The active Qt binding's ``QtCore``, or ``None`` when Qt is absent.

        Resolved lazily and tolerantly (PySide6 for current hosts, PySide2 for
        older builds) so the registry and server stay importable in environments
        with no Qt at all -- tests, tooling, agent inspection.
        """
        try:
            from PySide6 import QtCore  # type: ignore  # noqa: PLC0415

            return QtCore
        except ImportError:
            pass
        try:
            from PySide2 import QtCore  # type: ignore  # noqa: PLC0415

            return QtCore
        except ImportError:
            pass
        return None

    #: The relay class, built once per process against whichever binding won.
    #: A ``QObject`` subclass can only be declared with Qt in hand, and this
    #: module must import without it, so the declaration is deferred to here.
    _relay_type = None

    @classmethod
    def _relay_class(cls, qtcore):
        """The ``QObject`` whose queued signal delivers a call to its own thread.

        A queued signal is the one hop that is portable across PySide2/PySide6
        *and* honours thread affinity. The obvious-looking alternatives do not:
        ``QTimer.singleShot(0, functor)`` builds its helper object in the
        **calling** thread (so the functor is queued to the server's daemon
        thread, which runs no event loop, and never fires at all), and the
        3-argument context overload takes a slot name, not a Python callable.
        """
        if cls._relay_type is not None:
            return cls._relay_type

        class _MainThreadRelay(qtcore.QObject):
            """Emits on any thread; runs on the thread the object belongs to."""

            _dispatch = qtcore.Signal(object)

            def __init__(self):
                super().__init__()
                self._dispatch.connect(self._invoke, qtcore.Qt.QueuedConnection)

            def _invoke(self, fn):
                fn()

            def post(self, fn):
                """Queue *fn* for this object's thread. Returns immediately."""
                self._dispatch.emit(fn)

        cls._relay_type = _MainThreadRelay
        return _MainThreadRelay


class MainThreadMarshaller(_MainThreadMarshallerInternal):
    """Run a callable on the host's Qt main thread and block for its result.

    The HTTP server answers on daemon threads, but a host's scene API
    (``mset.*``, ``substance_painter.*``) is main-thread-only -- calling it
    off-thread is undefined and tends to take the host down. Every dispatched op
    goes through here, so a pure-Python op pays a negligible cost for the
    uniformity and no op can forget.

    Degrades to a direct call whenever marshalling is unnecessary *or*
    impossible: no Qt, no ``QCoreApplication``, or already on the main thread.
    That is what lets the identical code path run under a plain test runner.
    """

    def __init__(self, disable_env=None, timeout=60.0):
        #: Env var that forces the direct-call path even when Qt looks live.
        #: Tests importing the plugin alongside other Qt users have no pumped
        #: event loop, so marshalling would deadlock; a named opt-out keeps that
        #: bypass explicit and small. Production hosts never set it. ``None``:
        #: no opt-out.
        self.disable_env = disable_env
        self.timeout = timeout
        #: Lazily built relay + the ``QCoreApplication`` it was bound to, so a
        #: host that tears its application down and stands a new one up gets a
        #: fresh relay instead of one with dead thread affinity.
        self._relay = None
        self._relay_app = None
        self._relay_lock = threading.Lock()

    def _main_thread_relay(self, qtcore, app):
        """Return the relay object living on *app*'s thread, building it once.

        Built under a lock because the HTTP server answers concurrent requests
        on separate threads, and two relays would be two objects with the same
        job -- harmless but wasteful, and only one would be reused.
        """
        with self._relay_lock:
            if self._relay is None or self._relay_app is not app:
                relay = self._relay_class(qtcore)()
                # Constructed on whichever thread got here first; affinity has
                # to be the main thread for the queued connection to land there.
                relay.moveToThread(app.thread())
                self._relay = relay
                self._relay_app = app
            return self._relay

    def is_active(self):
        """True when :meth:`run` will marshal rather than call direct.

        Both the decision :meth:`run` acts on and the diagnostic a caller can log
        at server start ("ops will trampoline onto the main thread" vs "run in
        place"). One predicate for both, so the diagnostic cannot claim a mode the
        dispatcher won't take.
        """
        if self._opted_out():
            return False
        qtcore = self._qtcore()
        if qtcore is None:
            return False
        app = qtcore.QCoreApplication.instance()
        if app is None:
            return False
        return qtcore.QThread.currentThread() != app.thread()

    def _opted_out(self):
        return bool(self.disable_env) and os.environ.get(self.disable_env) == "1"

    def post(self, fn, *args, **kwargs):
        """Queue *fn* for the main thread and return at once -- never blocks.

        For a caller that must not wait on the main thread: a host callback on
        one of the host's own worker threads (Maya's Evaluation Manager fires
        ``MDagMessage`` matrix callbacks there) would deadlock in :meth:`run`
        while the main thread waits on that evaluation. Queued even from the
        main thread, so *fn* always runs on a later event-loop pass, outside
        the caller's stack. With no Qt, no ``QCoreApplication`` or the opt-out
        set, *fn* runs in place. Its result is discarded, and an exception it
        raises reaches Qt's event loop, so *fn* catches what must not surface.
        Hold the marshaller until the call lands: its relay delivers it, and a
        collected relay drops it.
        """
        qtcore = self._qtcore()
        app = qtcore.QCoreApplication.instance() if qtcore is not None else None
        if app is None or self._opted_out():
            fn(*args, **kwargs)
            return
        self._main_thread_relay(qtcore, app).post(lambda: fn(*args, **kwargs))

    def run(self, fn, *args, timeout=None, **kwargs):
        """Call *fn*, on the main thread when one is reachable.

        The original exception propagates verbatim in every mode. *timeout*
        applies only to the marshalled path: a blocked main thread (host
        mid-bake, mid-render) raises :class:`TimeoutError` so the HTTP request
        fails visibly instead of hanging forever.
        """
        if not self.is_active():
            # No Qt, no QApplication, already on the main thread, or the explicit
            # opt-out -- every case where marshalling is unnecessary or impossible.
            # Sharing the predicate with :meth:`is_active` is what keeps the
            # diagnostic honest: it can never report a mode this method won't take.
            return fn(*args, **kwargs)

        qtcore = self._qtcore()
        app = qtcore.QCoreApplication.instance()
        if app is None:
            # Torn down between the gate above and here (host shutting down).
            # Same answer as every other "cannot marshal" case.
            return fn(*args, **kwargs)

        deadline = self.timeout if timeout is None else timeout
        result_q = queue.Queue(maxsize=1)

        def _runner():
            try:
                result_q.put(("ok", fn(*args, **kwargs)))
            except BaseException as exc:  # noqa: BLE001 - relayed to the caller
                result_q.put(("err", exc))

        # Hand the call to an object that *lives* on the main thread; its queued
        # signal is what crosses the boundary (see :meth:`_relay_class` for why
        # the shorter-looking QTimer spellings do not).
        self._main_thread_relay(qtcore, app).post(_runner)

        try:
            kind, payload = result_q.get(timeout=deadline)
        except queue.Empty:
            raise TimeoutError(
                f"Main-thread call did not complete within {deadline}s. "
                f"The host's event loop is probably blocked."
            )

        if kind == "err":
            raise payload
        return payload


# ------------------------------------------------------------------ http server
class _ReusableServer(HTTPServer):
    """Reuse a ``TIME_WAIT`` port on POSIX; refuse a LIVE listener everywhere.

    ``SO_REUSEADDR`` does not mean the same thing on both platforms. On POSIX
    it reuses a ``TIME_WAIT`` port -- exactly what a host relaunch on a pinned
    port needs, and the reason the flag was set here. On Windows it
    *additionally* permits binding over a live listener: two servers both
    start, and the stack decides which socket receives each request. For a
    plugin that pins its port inside Toolbag or Painter that means a client
    can be driving a stale server from a previous session while the new one
    looks healthy, so the flag is dropped there and ``bind()`` fails loudly
    instead. Same reasoning, same wording, as ``preview/server.py``'s
    ``_PreviewHTTPServer``; ``test_plugin_core.TestBindPolicy`` asserts the
    two together so the answer cannot drift apart again.
    """

    allow_reuse_address = os.name != "nt"

    def handle_error(self, request, client_address):
        """A peer that vanished mid-connection is routine here, not an error.

        Clients probe and abandon connections (a timed-out ping, a port check,
        a DCC closing mid-request), and the stock handler printed a full
        traceback per reset into the host's log -- noise the user can't act
        on. Same filter as ``preview/routes.py``'s ``_PreviewHTTPServer``;
        anything else still reports as before.
        """
        if isinstance(
            sys.exc_info()[1],
            (ConnectionResetError, ConnectionAbortedError, BrokenPipeError),
        ):
            return
        super().handle_error(request, client_address)


class _Abandoned(Exception):
    """An op's caller hung up before the op's turn came: it is not run."""


class _ThreadedServer(ThreadingMixIn, _ReusableServer):
    """A request per thread, for a server whose ops run on ANOTHER thread.

    :meth:`RpcPlugin.start`'s server: an op there waits for the host's main
    thread, for as long as a bake or an export holds it, and a serial server
    answered nothing meanwhile -- not ``/health``, not a job's status -- so a
    client took a busy host for a dead one. The ops themselves still run one at
    a time (:meth:`RpcPlugin._turn`). Daemon threads: a request waiting on
    the host never keeps its process alive.
    """

    daemon_threads = True


#: ``Host`` names a local client reaches this server by (beside its bound address).
_LOOPBACK_NAMES = ("127.0.0.1", "localhost", "::1")


def _make_handler(plugin):
    """Build the request handler class bound to *plugin*.

    :class:`BaseHTTPRequestHandler` is instantiated per request by the server,
    so the plugin is closed over here rather than passed in -- that is the seam
    that lets two plugins live in one process without a global.
    """

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            if self.path == "/health":
                self._respond(200, {"ok": True, "value": "alive"})
            else:
                self._respond(404, {"ok": False, "error": f"GET {self.path!r}"})

        def _refusal(self):
            """``(status, why)`` when a web page sent this request, else ``None``.

            A page in the person's browser may POST to loopback without asking
            (a "simple" request: a text/plain body, no CORS preflight), so an op
            -- one that evaluates code above all -- would run for any site open
            while the host app is. A browser marks a page's request with
            ``Origin``; a JSON body is one a page sends only after a preflight
            this server never answers; and a ``Host`` that is not this server's
            address is a DNS-rebound name. :class:`RpcClient` sends none of them.
            """
            if self.headers.get("Origin") is not None:
                return 403, "a web page's request (it carries Origin)"
            kind = (self.headers.get("Content-Type") or "").split(";")[0].strip()
            if kind.lower() != "application/json":
                return 415, f"Content-Type {kind or '(none)'!r}: send application/json"
            name = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            if name.lower() not in _LOOPBACK_NAMES + (str(self.server.server_address[0]),):
                return 403, f"Host {name!r} is not this server"
            # Fails closed: an announced plugin between servings has no token,
            # and then nothing passes.
            if plugin.announce and (
                not plugin.token or self.headers.get("Authorization") != f"Bearer {plugin.token}"
            ):
                return 401, "no token, or not this instance's (read its record)"
            return None

        def do_POST(self):  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length).decode("utf-8")
            except Exception as exc:  # noqa: BLE001
                self._respond(400, {"ok": False, "error": f"Bad request: {exc}"})
                return
            refused = self._refusal()  # the body is read first: unread, a reset hides the answer
            if refused is not None:
                self._respond(refused[0], {"ok": False, "error": f"Refused: {refused[1]}."})
                return
            try:
                req = json.loads(raw) if raw else {}
            except Exception as exc:  # noqa: BLE001
                self._respond(400, {"ok": False, "error": f"Bad JSON: {exc}"})
                return

            if self.path == "/describe":
                self._respond(
                    200,
                    {
                        "ok": True,
                        "value": plugin.registry.describe(req.get("op") or None),
                    },
                )
                return

            self._dispatch(req)

        def _dispatch(self, req):
            op_name = req.get("op")
            handler = plugin.registry.get(op_name)
            if handler is None:
                self._respond(
                    404,
                    {
                        "ok": False,
                        "error": f"Unknown op: {op_name!r}",
                        "available": plugin.registry.all_ops(),
                    },
                )
                return
            kwargs = req.get("kwargs") or {}

            def at_its_turn():
                # Asked where the op runs, when it runs: a caller that stopped
                # waiting was told the host is not reachable and the person ran
                # it again, so its first request must not run too, minutes
                # later, when the host frees up.
                if self._abandoned():
                    raise _Abandoned()
                return handler(**kwargs)

            try:
                if plugin.registry.needs_main_thread(op_name):
                    with plugin._turn(self.server, self._abandoned):  # one host op at a time
                        # A served plugin runs every op on the thread that
                        # serves (``serve``'s contract), whatever Qt the
                        # process holds; every other op goes through the
                        # marshaller, which calls in place where there is no
                        # other thread to reach.
                        value = at_its_turn() if plugin._served else plugin.marshaller.run(at_its_turn)
                else:
                    value = handler(**kwargs)  # the plugin's own bookkeeping: answered at once
            except _Abandoned:
                return  # no one is there to answer
            except Exception as exc:  # noqa: BLE001 - reported over the wire
                self._respond(
                    500,
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                )
                return
            self._respond(200, {"ok": True, "value": value})

        def _abandoned(self):
            """True once this request's caller hung up (it stopped waiting for
            the answer). A caller still waiting has sent its whole request and
            nothing more, so its socket is not readable; one that closed reads
            as an empty peek or a reset. (A caller that half-closes after
            sending reads the same: :class:`RpcClient` and the HTTP libraries
            do not.) A probe that cannot tell answers False: dropping a live
            caller's op is the worse mistake (``select()`` refuses a descriptor
            past 1023 on Linux, so a host holding many files read every caller
            as gone)."""
            sock = self.connection
            if sock.fileno() < 0:
                return True  # closed under the request
            try:
                with selectors.DefaultSelector() as probe:  # epoll / kqueue: no fd ceiling
                    probe.register(sock, selectors.EVENT_READ)
                    if not probe.select(0):
                        return False
                return sock.recv(1, socket.MSG_PEEK) == b""
            except (ConnectionResetError, ConnectionAbortedError):
                return True
            except (OSError, ValueError):
                return False

        def log_message(self, *_a, **_kw):
            """Silence access logs so they don't drown the host's own log."""

        def _respond(self, status, payload):
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return _Handler


# -------------------------------------------------------------------- instances
class RpcInstances(object):
    """Instance records: how a client finds a running plugin by what it is.

    A plugin started with ``announce=True`` writes one JSON record per process,
    ``<folder>/<label>/<id>.json``, and removes it when it stops::

        {"id", "label", "pid", "host", "port", "token", "started", "state": {...}}

    ``state`` is the host's to fill (:meth:`RpcPlugin.set_state`): what it has
    open, whether it is busy. A port is then whatever the OS gave (any number of
    instances, none fighting for one port), and the record says which is which.
    The folder is the person's own, so the token in it proves a caller is one of
    their processes (loopback is shared by every user of a machine): every op
    call to an announced plugin must carry it. A record whose process is gone
    (a crash leaves one) is pruned by the next reader. Prior art: Jupyter's
    runtime connection files (``jpserver-<pid>.json``).
    """

    #: The records' folder under the per-user cache root.
    SUBDIR = "rpc"
    #: Overrides the cache root, as ``UserConfig.cache_dir`` reads it.
    CACHE_ROOT_ENV_VAR = "PYTHONTK_CACHE_ROOT"
    _WRAPPER = "pythontk"

    @classmethod
    def folder(cls, label=None):
        """The records folder (*label*'s, when given), not created.

        Resolved as ``UserConfig.cache_dir(SUBDIR)`` is -- this file runs where
        pythontk is not importable, so it repeats that resolution, and a test
        pins the two to one answer.
        """
        from pathlib import Path  # noqa: PLC0415 -- only this path needs it

        override = os.environ.get(cls.CACHE_ROOT_ENV_VAR)
        if override:
            root = Path(os.path.expandvars(os.path.expanduser(override)))
        elif sys.platform == "win32":
            base = os.environ.get("LOCALAPPDATA") or os.path.join(
                os.path.expanduser("~"), "AppData", "Local"
            )
            root = Path(base) / cls._WRAPPER / "cache"
        elif sys.platform == "darwin":
            root = Path(os.path.expanduser("~"), "Library", "Caches", cls._WRAPPER)
        else:
            xdg = os.environ.get("XDG_CACHE_HOME", "")
            base = xdg if os.path.isabs(xdg) else os.path.join(os.path.expanduser("~"), ".cache")
            root = Path(base) / cls._WRAPPER
        if not root.is_absolute():
            root = root.absolute()
        folder = root / cls.SUBDIR
        return folder / label if label else folder

    @staticmethod
    def pid_alive(pid):
        """True while *pid* runs (a process another user owns counts as running)."""
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        if sys.platform == "win32":
            import ctypes  # noqa: PLC0415

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
            if not handle:
                return ctypes.get_last_error() == 5  # access denied: it is there
            try:  # a handle outlives its process: wait on it instead
                return kernel32.WaitForSingleObject(handle, 0) == 0x102
            finally:
                kernel32.CloseHandle(handle)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @classmethod
    def write(cls, record):
        """Write *record* (its ``label`` and ``id`` name the file), atomically."""
        folder = cls.folder(record["label"])
        folder.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":  # the token is the person's: theirs alone to read
            os.chmod(str(folder), 0o700)
        path = folder / f"{record['id']}.json"
        part = path.with_suffix(".json.part")
        part.write_text(json.dumps(record, default=str), encoding="utf-8")
        if os.name != "nt":
            os.chmod(str(part), 0o600)
        os.replace(str(part), str(path))
        return path

    @classmethod
    def remove(cls, label, instance_id):
        """Delete one record; one already gone is fine."""
        try:
            (cls.folder(label) / f"{instance_id}.json").unlink()
        except OSError:
            pass

    @classmethod
    def read(cls, label):
        """Every live record of *label*, oldest first; dead ones are deleted."""
        folder = cls.folder(label)
        records = []
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else ():
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue  # mid-replace, or not ours
            if not cls.pid_alive(record.get("pid")):
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            records.append(record)
        return sorted(records, key=lambda r: r.get("started", 0))


# ------------------------------------------------------------------- the facade
class RpcPlugin(object):
    """One host plugin: a registry, a marshaller, and the server that joins them.

    Serves the wire contract :class:`pythontk.net_utils.rpc.RpcClient` speaks:

    * ``GET  /health``   -> ``{"ok": true, "value": "alive"}``
    * ``POST /``         -> ``{"op": "<name>", "kwargs": {...}}``
    * ``POST /describe`` -> ``{"op": "<name>" | ""}``

    A POST a web page could send (an ``Origin`` header, a body that is not
    ``application/json``, a ``Host`` that is not this server) is refused before
    any op runs: a page open in the person's browser never reaches an op.

    Started with ``announce=True``, it also writes its instance record
    (:class:`RpcInstances`): a client finds it by what it is and has open
    (:meth:`RpcClient.instances`), and every op call must carry the record's
    token (:meth:`RpcClient.for_instance`).

    A host plugin's ``__init__.py`` builds one of these and re-exports what its
    op modules need::

        from ._rpc_core import RpcPlugin

        PLUGIN = RpcPlugin(
            label="marmoset_rpc", host_module="mset",
            env_prefix="MARMOSET_RPC", default_port=0, announce=True,
        )
        register = PLUGIN.registry.register
        run_on_main_thread = PLUGIN.marshaller.run

        from . import ops  # noqa: E402,F401 -- @register side effects
    """

    def __init__(
        self,
        label,
        host_module,
        env_prefix,
        default_port,
        host="127.0.0.1",
        main_thread_timeout=60.0,
        announce=False,
        pump=None,
    ):
        """
        Parameters:
            label: Log prefix and thread-name stem (e.g. ``"marmoset_rpc"``).
            host_module: The module only the host provides (``"mset"``,
                ``"substance_painter"``) -- the auto-start gate probes for it.
            env_prefix: Stem for this plugin's env vars: ``<PREFIX>_PORT``,
                ``<PREFIX>_AUTOSTART``, ``<PREFIX>_DISABLE_MAIN_THREAD``.
            default_port: Port used when ``<PREFIX>_PORT`` is unset.
            host: Bind address. Loopback by default -- this is a local control
                channel, not a service.
            main_thread_timeout: Seconds a marshalled op may take.
            announce: Write an instance record while serving, and require its
                token on every op call (:class:`RpcInstances`).
            pump: For a host whose Python cannot keep a thread running (one
                that ends when a plugin's script returns with a thread alive): ``pump(tick)``
                arranges for ``tick()`` to run on the host's own loop (an update
                callback), and :meth:`start` serves through it, no thread: each
                tick handles the requests waiting, on the host's thread.
        """
        self.label = label
        self.host_module = host_module
        self.env_prefix = env_prefix
        self.default_port = default_port
        self.host = host
        self.registry = OpRegistry()
        self.marshaller = MainThreadMarshaller(
            f"{env_prefix}_DISABLE_MAIN_THREAD", timeout=main_thread_timeout
        )
        self._server = None
        self._thread = None
        #: Held by the host op under way (:meth:`_turn`).
        self._op_lock = threading.Lock()
        self.pump = pump
        self._pumped = False
        #: True while :meth:`serve` serves: its ops are never marshalled.
        self._served = False
        self.announce = announce
        #: This serving's record id and token (``None`` until it serves announced).
        self.instance_id = None
        self.token = None
        self._state = {}
        self._started = None
        self._register_builtins()

    @staticmethod
    def import_ops(package):
        """Import *package* (dotted name), forcing its ``@register`` side effects.

        The op modules register themselves at **import time**, onto whichever
        registry the plugin package had bound when they last ran. That makes a
        plain ``from . import ops`` correct exactly once per interpreter: when a
        host reloads its plugins (Painter's *Python ▸ Reload Plugins Folder*, a
        disable/re-enable, ``importlib.reload``) the package body re-runs and
        builds a **new** :class:`RpcPlugin` with an empty registry, but the ops
        submodules are still in ``sys.modules`` -- so the import is a cache hit,
        no decorator fires, and the server comes back up serving only the
        built-in ``system.*`` trio while answering ``Unknown op`` for every
        feature it exists to provide. Silent, and indistinguishable from the
        host simply ignoring the call.

        Dropping the subtree first makes the import unconditional, so the
        registration is tied to the registry that is live *now*.

        Parameters:
            package: Dotted name of the ops package (e.g. ``"substance_rpc.ops"``).

        Returns:
            The freshly imported module.
        """
        import importlib  # noqa: PLC0415 -- only needed on this path

        prefix = package + "."
        for name in [
            m for m in list(sys.modules) if m == package or m.startswith(prefix)
        ]:
            del sys.modules[name]
        return importlib.import_module(package)

    #: Seconds between a waiting host op's looks at its caller (:meth:`_turn`).
    TURN_POLL = 0.5

    @contextlib.contextmanager
    def _turn(self, server, gone=lambda: False):
        """A host op's turn: one op at a time where requests are answered on
        threads of their own (*server*, the request's own, a
        :class:`_ThreadedServer`; read from the request, since :meth:`stop`
        clears this plugin's before that server stops). A pumped or served
        plugin answers on one thread, a request at a time, and an op there may
        pump the host's loop itself (a nested tick, whose op a lock held by the
        outer one would stop for good): no lock. A waiter asks *gone* every
        :attr:`TURN_POLL` seconds and leaves the queue (:class:`_Abandoned`)
        once its caller hung up."""
        if not isinstance(server, _ThreadedServer):
            yield
            return
        while not self._op_lock.acquire(timeout=self.TURN_POLL):
            if gone():
                raise _Abandoned()
        try:
            yield
        finally:
            self._op_lock.release()

    def _register_builtins(self):
        """Register the ``system.*`` ops the client contract assumes exist.

        :meth:`pythontk.RpcClient.list_ops` invokes ``system.list_ops`` as an *op*, not
        a route -- so a plugin that forgets to define it breaks a documented client
        method. Owning the transport-level trio here makes that impossible: every
        plugin answers ping / list_ops / describe by construction, and no per-plugin
        copy can drift (they already had, before this moved).

        Host-specific introspection (``system.version``, script evaluation) stays with
        the plugin -- only what the client itself calls belongs here. They read
        the plugin's own tables, so they answer off the host's main thread.
        """

        @self.registry.register("system.ping", main_thread=False)
        def _ping():
            """Liveness probe. Returns ``"pong"``."""
            return "pong"

        @self.registry.register("system.list_ops", main_thread=False)
        def _list_ops():
            """Sorted list of every registered op name."""
            return self.registry.all_ops()

        @self.registry.register("system.describe", main_thread=False)
        def _describe(op=""):
            """Describe *op* (or every op when empty) as ``{name, doc, params}``."""
            return self.registry.describe(op or None)

        @self.registry.register("system.identity", main_thread=False)
        def _identity():
            """This instance: ``{id, label, pid, state}`` (its record, without the token)."""
            return {
                "id": self.instance_id,
                "label": self.label,
                "pid": os.getpid(),
                "state": dict(self._state),
            }

    # ------------------------------------------------------------ environment
    def _env(self, suffix, default=None):
        """Read this plugin's ``<PREFIX>_<SUFFIX>`` environment variable."""
        return os.environ.get(f"{self.env_prefix}_{suffix}", default)

    @property
    def port(self):
        """Configured port: ``<PREFIX>_PORT`` if set and numeric, else the default."""
        try:
            return int(self._env("PORT", "") or self.default_port)
        except (TypeError, ValueError):
            return self.default_port

    def is_hosted(self):
        """True only inside the real host application.

        The host provides :attr:`host_module` and nothing else does, so its
        availability is the reliable signal that the *host's* plugin loader
        imported this package -- as opposed to an incidental import elsewhere
        (DCC slot discovery imports every ``*.py`` in a package to introspect
        classes, and must not bind a port as a side effect).

        Checks ``sys.modules`` first (hosts pre-inject their module), then falls
        back to ``find_spec``. Neither path executes the module.
        """
        if self.host_module in sys.modules:
            return True
        import importlib.util

        try:
            return importlib.util.find_spec(self.host_module) is not None
        except (ImportError, ValueError):
            return False

    # ----------------------------------------------------------------- server
    def is_running(self):
        """True while the HTTP server is bound."""
        return self._server is not None

    @property
    def address(self):
        """The bound ``(host, port)``, or ``None`` when not running."""
        return None if self._server is None else self._server.server_address

    #: Requests a pumped server handles per tick, at most (the rest wait a tick).
    PUMP_BATCH = 8

    def start(self, port=None, host=None):
        """Bind and serve on a daemon thread (with :attr:`pump`: from the host's
        loop, no thread). Idempotent.

        The thread is a daemon so the server dies with the host process; no
        explicit shutdown is required for normal use.
        """
        if self._server is not None:
            return self._server.server_address

        bind_host = self.host if host is None else host
        bind_port = self.port if port is None else port

        # A pumped server answers on the host's own thread, a request at a time;
        # a served one on threads of its own, so a busy host still answers.
        kind = _ReusableServer if self.pump is not None else _ThreadedServer
        server = kind((bind_host, bind_port), _make_handler(self))
        address = server.server_address  # port 0: the one the OS picked
        try:
            # Before the first request is served: its token is in force from it.
            self._open_record(address)
        except Exception:
            server.server_close()  # unannounced, nobody could reach it: no server either
            raise
        self._server = server
        if self.pump is not None:
            server.timeout = 0  # handle_request returns at once when nothing waits
            self._pumped = True
            self.pump(self._tick)
            print(f"[{self.label}] listening on http://{address[0]}:{address[1]} (from the host's loop)")
            return address
        self._thread = threading.Thread(
            target=server.serve_forever,
            daemon=True,
            name=f"{self.label}-server",
        )
        self._thread.start()
        print(f"[{self.label}] listening on http://{address[0]}:{address[1]}")
        return address

    def _tick(self):
        """One turn of a pumped server: the requests waiting now, on this thread."""
        server = self._server
        if server is None or not self._pumped:
            return
        with selectors.DefaultSelector() as selector:
            selector.register(server.socket, selectors.EVENT_READ)
            for _ in range(self.PUMP_BATCH):
                if self._server is not server or not selector.select(0):
                    return
                server.handle_request()

    def serve(self, port=None, host=None):
        """Bind and serve on the CALLING thread until :meth:`stop`; blocks.

        For a process whose job is the service: every op then runs on this
        thread (no Qt to marshal onto), so an op may drive something bound to
        the thread that created it -- a sync browser automation, a renderer --
        which :meth:`start`'s daemon thread could not. An op calling
        :meth:`stop` ends it once that op has answered.
        """
        if self._server is not None:
            raise RuntimeError(
                f"[{self.label}] already serving on {self._server.server_address}"
            )
        bind_host = self.host if host is None else host
        bind_port = self.port if port is None else port
        server = _ReusableServer((bind_host, bind_port), _make_handler(self))
        self._server, self._thread = server, threading.current_thread()
        address = server.server_address  # port 0: the one the OS picked
        try:
            self._open_record(address)
        except Exception:
            self._server = self._thread = None
            server.server_close()
            raise
        self._served = True
        print(f"[{self.label}] serving on http://{address[0]}:{address[1]}", flush=True)
        try:
            server.serve_forever()
        finally:
            self._served = False
            server.server_close()
            if self._server is server:
                self._server = self._thread = None
                self._close_record()

    # ---------------------------------------------------------------- record
    def set_state(self, **fields):
        """Merge *fields* into this instance's ``state`` (what the host has open,
        whether it is busy) and rewrite its record when it serves announced."""
        self._state.update(fields)
        if self.instance_id is not None:
            self._write_record()

    def _open_record(self, address):
        """Start this serving's record and token (``announce`` only)."""
        if not self.announce:
            return
        import secrets  # noqa: PLC0415
        import uuid  # noqa: PLC0415

        self.instance_id = uuid.uuid4().hex
        self.token = secrets.token_urlsafe(24)
        self._address = address
        self._started = time.time()
        self._write_record()

    def _write_record(self):
        RpcInstances.write(
            {
                "id": self.instance_id,
                "label": self.label,
                "pid": os.getpid(),
                "host": self._address[0],
                "port": self._address[1],
                "token": self.token,
                "started": self._started,
                "state": dict(self._state),
            }
        )

    def _close_record(self):
        """Withdraw the record; the token goes with it."""
        if self.instance_id is not None:
            RpcInstances.remove(self.label, self.instance_id)
        self.instance_id = self.token = None

    def stop(self):
        """Shut the server down (host teardown hook, tests, hot-reload, an op).

        From an op on the serving thread itself (:meth:`serve`) the shutdown
        is handed to another thread: ``shutdown`` waits for the serve loop,
        which is waiting for this op to return.
        """
        if self._server is None:
            return
        server = self._server
        self._server = None
        self._close_record()
        if self._pumped:
            # No serve loop to end (``shutdown`` would wait for one forever).
            self._pumped = False
            server.server_close()
            return

        def finish():
            server.shutdown()
            server.server_close()

        if threading.current_thread() is self._thread:
            self._thread = None
            threading.Thread(
                target=finish, daemon=True, name=f"{self.label}-stop"
            ).start()
            return
        self._thread = None
        finish()

    def autostart(self):
        """Start on plugin load, but only when actually hosted.

        Returns the bound address, or ``None`` when the gate declined. The
        no-op return is the whole point: importing a plugin package outside its
        host must stay free of side effects. ``<PREFIX>_AUTOSTART=0`` is an
        explicit opt-out honoured even inside the host (tests rely on it).
        """
        if self._env("AUTOSTART", "1") != "1":
            return None
        if not self.is_hosted():
            return None
        return self.start()

    def autostart_safely(self):
        """:meth:`autostart`, but a failure is logged instead of raised.

        The host lifecycle hook calls this: a port squabble must never take the
        host application down. Returns the address or ``None``.
        """
        try:
            return self.autostart()
        except Exception as exc:  # noqa: BLE001 - reported, never raised at the host
            print(f"[{self.label}] server failed to start: {exc}")
            return None
