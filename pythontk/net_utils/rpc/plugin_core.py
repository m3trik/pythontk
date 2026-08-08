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

1. **Imported** -- by a plugin that loads in place from the checkout and
   bootstraps ``sys.path`` back to it (``extapps.substance_workflow``'s Painter
   plugin). It just imports this module.
2. **Staged** -- by a plugin *installed* into the host's own plugin folder
   (mayatk / blendertk's ``marmoset_rpc`` + ``substance_rpc``). Those payloads
   are symlinked or copied into Toolbag / Painter and must be self-contained:
   there is no ``pythontk`` on the host's ``sys.path``. So this file is copied
   verbatim into each payload as ``_rpc_core.py``.

The staged copies are byte-identical artifacts, not forks. Regenerate with
``python m3trik/scripts/sync_rpc_core.py``; ``--check`` fails on drift (CI gate).
**Never hand-edit a staged ``_rpc_core.py``** -- edit this file and re-stage.
"""

from __future__ import annotations

import inspect
import json
import os
import queue
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

__all__ = ["OpRegistry", "MainThreadMarshaller", "RpcPlugin"]


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

    def register(self, name):
        """Decorator registering the wrapped function under *name*.

        Names are dot-namespaced (``"system.ping"``, ``"scene.list_materials"``)
        so the client can group related calls. A duplicate name raises rather
        than overwriting -- a typo that silently shadows a real op would surface
        much later as the feature quietly not happening.
        """

        def decorator(fn):
            if name in self._ops:
                raise ValueError(f"Op {name!r} is already registered.")
            self._ops[name] = fn
            return fn

        return decorator

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

    The HTTP server answers on a daemon thread, but a host's scene API
    (``mset.*``, ``substance_painter.*``) is main-thread-only -- calling it
    off-thread is undefined and tends to take the host down. Every dispatched op
    goes through here, so a pure-Python op pays a negligible cost for the
    uniformity and no op can forget.

    Degrades to a direct call whenever marshalling is unnecessary *or*
    impossible: no Qt, no ``QCoreApplication``, or already on the main thread.
    That is what lets the identical code path run under a plain test runner.
    """

    def __init__(self, disable_env, timeout=60.0):
        #: Env var that forces the direct-call path even when Qt looks live.
        #: Tests importing the plugin alongside other Qt users have no pumped
        #: event loop, so marshalling would deadlock; a named opt-out keeps that
        #: bypass explicit and small. Production hosts never set it.
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
        if os.environ.get(self.disable_env) == "1":
            return False
        qtcore = self._qtcore()
        if qtcore is None:
            return False
        app = qtcore.QCoreApplication.instance()
        if app is None:
            return False
        return qtcore.QThread.currentThread() != app.thread()

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
    """``SO_REUSEADDR`` so a host relaunch isn't blocked by a ``TIME_WAIT`` socket."""

    allow_reuse_address = True


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

        def do_POST(self):  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length).decode("utf-8")
                req = json.loads(raw) if raw else {}
            except Exception as exc:  # noqa: BLE001
                self._respond(400, {"ok": False, "error": f"Bad JSON: {exc}"})
                return

            if self.path == "/describe":
                self._respond(
                    200,
                    {"ok": True, "value": plugin.registry.describe(req.get("op") or None)},
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
            try:
                value = plugin.marshaller.run(handler, **(req.get("kwargs") or {}))
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


# ------------------------------------------------------------------- the facade
class RpcPlugin(object):
    """One host plugin: a registry, a marshaller, and the server that joins them.

    Serves the wire contract :class:`pythontk.net_utils.rpc.RpcClient` speaks:

    * ``GET  /health``   -> ``{"ok": true, "value": "alive"}``
    * ``POST /``         -> ``{"op": "<name>", "kwargs": {...}}``
    * ``POST /describe`` -> ``{"op": "<name>" | ""}``

    A host plugin's ``__init__.py`` builds one of these and re-exports what its
    op modules need::

        from ._rpc_core import RpcPlugin

        PLUGIN = RpcPlugin(
            label="marmoset_rpc", host_module="mset",
            env_prefix="MARMOSET_RPC", default_port=8765,
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

    def _register_builtins(self):
        """Register the ``system.*`` ops the client contract assumes exist.

        :meth:`pythontk.RpcClient.list_ops` invokes ``system.list_ops`` as an *op*, not
        a route -- so a plugin that forgets to define it breaks a documented client
        method. Owning the transport-level trio here makes that impossible: every
        plugin answers ping / list_ops / describe by construction, and no per-plugin
        copy can drift (they already had, before this moved).

        Host-specific introspection (``system.version``, script evaluation) stays with
        the plugin -- only what the client itself calls belongs here.
        """

        @self.registry.register("system.ping")
        def _ping():
            """Liveness probe. Returns ``"pong"``."""
            return "pong"

        @self.registry.register("system.list_ops")
        def _list_ops():
            """Sorted list of every registered op name."""
            return self.registry.all_ops()

        @self.registry.register("system.describe")
        def _describe(op=""):
            """Describe *op* (or every op when empty) as ``{name, doc, params}``."""
            return self.registry.describe(op or None)

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

    def start(self, port=None, host=None):
        """Bind and serve on a daemon thread. Idempotent.

        The thread is a daemon so the server dies with the host process; no
        explicit shutdown is required for normal use.
        """
        if self._server is not None:
            return self._server.server_address

        bind_host = self.host if host is None else host
        bind_port = self.port if port is None else port

        self._server = _ReusableServer((bind_host, bind_port), _make_handler(self))
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name=f"{self.label}-server",
        )
        self._thread.start()
        print(f"[{self.label}] listening on http://{bind_host}:{bind_port}")
        return self._server.server_address

    def stop(self):
        """Shut the server down (host teardown hook, tests, hot-reload)."""
        if self._server is None:
            return
        try:
            self._server.shutdown()
            self._server.server_close()
        finally:
            self._server = None
            self._thread = None

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
