#!/usr/bin/python
# coding=utf-8
"""Unit tests for pythontk's in-application RPC core (net_utils.rpc.plugin_core).

The core is what runs *inside* a host application (Marmoset Toolbag, Substance
Painter). These tests exercise it with no host at all: the host gate is probed
with a module name that is or isn't importable, and the marshaller's no-Qt path
is the same one production takes when a host has no event loop yet.

The load-bearing test is :class:`TestWireContract` — it drives a live server
through the real :class:`pythontk.RpcClient`. Client and server ship in one
package precisely so the wire format cannot drift; that test is the assertion.

Run with:
    python -m pytest test_plugin_core.py -v
"""

import os
import shutil
import sys
import tempfile
import threading
import unittest

from pythontk.net_utils.rpc.client import RpcClient
from pythontk.net_utils.rpc.plugin_core import (
    MainThreadMarshaller,
    OpRegistry,
    RpcPlugin,
)

try:  # Qt is optional here exactly as it is in the core itself.
    from PySide6 import QtCore as _QTCORE
except ImportError:  # pragma: no cover - binding-dependent
    try:
        from PySide2 import QtCore as _QTCORE  # type: ignore
    except ImportError:
        _QTCORE = None


def _make_plugin(**kw):
    """A plugin bound to a host module that is never importable (so it stays inert)."""
    kw.setdefault("label", "test_rpc")
    kw.setdefault("host_module", "a_host_module_that_does_not_exist")
    kw.setdefault("env_prefix", "TEST_RPC")
    kw.setdefault("default_port", 0)  # 0 = let the OS pick a free port
    return RpcPlugin(**kw)


class _EnvGuard(unittest.TestCase):
    """Restores every ``TEST_RPC_*`` var the case touched."""

    def setUp(self):
        self._env = {k: v for k, v in os.environ.items() if k.startswith("TEST_RPC")}

    def tearDown(self):
        for key in [k for k in os.environ if k.startswith("TEST_RPC")]:
            del os.environ[key]
        os.environ.update(self._env)


class TestOpRegistry(unittest.TestCase):
    def test_register_and_get(self):
        reg = OpRegistry()

        @reg.register("system.ping")
        def _ping():
            return "pong"

        self.assertIs(reg.get("system.ping"), _ping)
        self.assertEqual(reg.all_ops(), ["system.ping"])
        self.assertIsNone(reg.get("nope"))

    def test_duplicate_name_raises(self):
        """A typo that silently shadowed a real op would surface much later."""
        reg = OpRegistry()
        reg.register("a")(lambda: 1)
        with self.assertRaises(ValueError):
            reg.register("a")(lambda: 2)

    def test_instances_do_not_share_state(self):
        """Module-global op tables merge two plugins hosted in one process."""
        a, b = OpRegistry(), OpRegistry()
        a.register("only.in.a")(lambda: 1)
        self.assertEqual(b.all_ops(), [])

    def test_a_bare_registry_has_no_builtins(self):
        """The system.* trio belongs to RpcPlugin, not to every registry."""
        self.assertEqual(OpRegistry().all_ops(), [])

    def test_describe_reports_params_and_doc(self):
        reg = OpRegistry()

        @reg.register("scene.export")
        def _export(path, overwrite=False):
            """Export the scene."""

        described = reg.describe("scene.export")
        self.assertEqual(described["name"], "scene.export")
        self.assertEqual(described["doc"], "Export the scene.")
        self.assertEqual(
            described["params"],
            [
                {"name": "path", "default": "<required>"},
                {"name": "overwrite", "default": "False"},
            ],
        )

    def test_describe_all_is_sorted_and_unknown_is_none(self):
        reg = OpRegistry()
        reg.register("b.op")(lambda: None)
        reg.register("a.op")(lambda: None)
        self.assertEqual([d["name"] for d in reg.describe()], ["a.op", "b.op"])
        self.assertIsNone(reg.describe("missing"))

    def test_describe_survives_an_unintrospectable_callable(self):
        """Defaults are stringified so the result always round-trips through JSON."""
        reg = OpRegistry()
        reg.register("built.in")(len)  # C builtins can refuse signature inspection
        self.assertIsInstance(reg.describe("built.in")["params"], list)


class TestMainThreadMarshaller(_EnvGuard):
    def test_calls_directly_without_qt(self):
        """The no-Qt path is what lets tests run the production code path."""
        m = MainThreadMarshaller("TEST_RPC_DISABLE_MAIN_THREAD")
        self.assertEqual(m.run(lambda a, b: a + b, 1, b=2), 3)

    def test_exceptions_propagate_verbatim(self):
        m = MainThreadMarshaller("TEST_RPC_DISABLE_MAIN_THREAD")

        def _boom():
            raise KeyError("original")

        with self.assertRaises(KeyError):
            m.run(_boom)

    def test_env_opt_out_forces_the_direct_path(self):
        os.environ["TEST_RPC_DISABLE_MAIN_THREAD"] = "1"
        m = MainThreadMarshaller("TEST_RPC_DISABLE_MAIN_THREAD")
        self.assertFalse(m.is_active())
        self.assertEqual(m.run(lambda: "ran"), "ran")

    def test_plugin_derives_the_marshaller_env_var_from_its_prefix(self):
        self.assertEqual(
            _make_plugin().marshaller.disable_env, "TEST_RPC_DISABLE_MAIN_THREAD"
        )


@unittest.skipIf(_QTCORE is None, "no Qt binding available")
class TestMainThreadMarshallerWithQt(_EnvGuard):
    """The path production actually takes: a worker thread inside a live host.

    Every regression here is invisible to the no-Qt tests above -- those take
    the direct-call branch, which is the one case that never needed
    marshalling. A host (Painter, Toolbag) has a ``QCoreApplication``, so
    ``is_active()`` is True and the hop is real.
    """

    def setUp(self):
        super().setUp()
        self.app = _QTCORE.QCoreApplication.instance() or _QTCORE.QCoreApplication([])
        self.main_thread = _QTCORE.QThread.currentThread()

    def _run_off_thread(self, marshaller, fn, pump_ms=5000):
        """Call ``marshaller.run(fn)`` from a worker thread, pumping the main loop.

        Returns ``("ok", value)`` or ``("err", exception)`` -- mirroring what the
        RPC server's daemon thread sees.
        """
        outcome = []

        def worker():
            try:
                outcome.append(("ok", marshaller.run(fn)))
            except BaseException as exc:  # noqa: BLE001 -- relayed to the assert
                outcome.append(("err", exc))

        thread = threading.Thread(target=worker, name="test-rpc-server")
        thread.start()
        elapsed = _QTCORE.QElapsedTimer()
        elapsed.start()
        while thread.is_alive() and elapsed.elapsed() < pump_ms:
            self.app.processEvents(_QTCORE.QEventLoop.AllEvents, 20)
        thread.join(timeout=2)
        self.assertTrue(outcome, "worker never finished")
        return outcome[0]

    def test_is_active_off_the_main_thread(self):
        marshaller = MainThreadMarshaller("TEST_RPC_DISABLE_MAIN_THREAD", timeout=5.0)
        seen = []
        thread = threading.Thread(target=lambda: seen.append(marshaller.is_active()))
        thread.start()
        thread.join(timeout=5)
        self.assertEqual(seen, [True])

    def test_a_marshalled_call_actually_reaches_the_main_thread(self):
        """The whole point of the class -- and what silently stopped happening.

        A 0-delay ``QTimer.singleShot`` with no context object builds its helper
        object in the *calling* thread; queued from the server's daemon thread
        (no event loop) it never fires, so every op timed out at 60s and the
        host-side feature just didn't happen.
        """
        marshaller = MainThreadMarshaller("TEST_RPC_DISABLE_MAIN_THREAD", timeout=5.0)
        kind, value = self._run_off_thread(
            marshaller, lambda: _QTCORE.QThread.currentThread()
        )
        self.assertEqual(kind, "ok", f"marshalled call failed: {value!r}")
        self.assertIs(value, self.main_thread)

    def test_a_marshalled_exception_propagates_verbatim(self):
        marshaller = MainThreadMarshaller("TEST_RPC_DISABLE_MAIN_THREAD", timeout=5.0)

        def _boom():
            raise KeyError("original")

        kind, value = self._run_off_thread(marshaller, _boom)
        self.assertEqual(kind, "err")
        self.assertIsInstance(value, KeyError)

    def test_repeated_calls_reuse_the_relay(self):
        """Two hops in a row must both land; a one-shot relay would strand the second."""
        marshaller = MainThreadMarshaller("TEST_RPC_DISABLE_MAIN_THREAD", timeout=5.0)
        for expected in ("first", "second"):
            kind, value = self._run_off_thread(marshaller, lambda v=expected: v)
            self.assertEqual((kind, value), ("ok", expected))

    def test_a_blocked_main_thread_still_times_out(self):
        """The timeout must survive the fix -- a wedged host may never answer."""
        marshaller = MainThreadMarshaller("TEST_RPC_DISABLE_MAIN_THREAD", timeout=0.5)
        outcome = []

        def worker():
            try:
                outcome.append(("ok", marshaller.run(lambda: "never")))
            except BaseException as exc:  # noqa: BLE001
                outcome.append(("err", exc))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)  # main thread deliberately does NOT pump
        self.assertEqual(outcome[0][0], "err")
        self.assertIsInstance(outcome[0][1], TimeoutError)


class TestImportOps(unittest.TestCase):
    """A host that reloads its plugins must not end up with an empty op table.

    Painter's *Python ▸ Reload Plugins Folder* (and the disable/re-enable the
    bridge tells users to do after an install refresh) re-executes the plugin
    package: a fresh ``RpcPlugin`` with a fresh registry. A plain
    ``from . import ops`` then hits the still-cached submodule, the
    ``@register`` decorators never re-run, and the server comes back up
    answering ``Unknown op`` for everything it is supposed to serve.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rpc_ops_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        pkg = os.path.join(self.tmp, "fake_rpc_plugin")
        os.makedirs(os.path.join(pkg, "ops"))
        with open(os.path.join(pkg, "__init__.py"), "w", encoding="utf-8") as fh:
            fh.write("REGISTERED = []\n")
        with open(os.path.join(pkg, "ops", "__init__.py"), "w", encoding="utf-8") as fh:
            fh.write("from . import demo_ops  # noqa: F401\n")
        with open(os.path.join(pkg, "ops", "demo_ops.py"), "w", encoding="utf-8") as fh:
            fh.write("from .. import REGISTERED\nREGISTERED.append('demo.op')\n")
        sys.path.insert(0, self.tmp)
        self.addCleanup(sys.path.remove, self.tmp)
        self.addCleanup(self._purge)

    def _purge(self):
        for name in [
            m
            for m in sys.modules
            if m == "fake_rpc_plugin" or m.startswith("fake_rpc_plugin.")
        ]:
            del sys.modules[name]

    def test_import_ops_re_runs_registration_after_a_reload(self):
        import fake_rpc_plugin

        RpcPlugin.import_ops("fake_rpc_plugin.ops")
        self.assertEqual(fake_rpc_plugin.REGISTERED, ["demo.op"])

        # What a host reload does: forget the package, import it again. The
        # fresh module object starts with an empty table.
        del sys.modules["fake_rpc_plugin"]
        import fake_rpc_plugin as reloaded

        self.assertEqual(reloaded.REGISTERED, [])
        RpcPlugin.import_ops("fake_rpc_plugin.ops")
        self.assertEqual(reloaded.REGISTERED, ["demo.op"])

    def test_plain_import_is_the_broken_baseline(self):
        """Documents *why* import_ops exists, so nobody 'simplifies' it away."""
        import fake_rpc_plugin  # noqa: F401
        import fake_rpc_plugin.ops  # noqa: F401

        del sys.modules["fake_rpc_plugin"]
        import fake_rpc_plugin as reloaded

        # The re-import IS the assertion: F811 flags it as a redefinition, which
        # is exactly the baseline being documented -- cached, side effect skipped.
        import fake_rpc_plugin.ops  # noqa: F401, F811

        self.assertEqual(reloaded.REGISTERED, [])

    def test_returns_the_imported_module(self):
        import fake_rpc_plugin  # noqa: F401

        module = RpcPlugin.import_ops("fake_rpc_plugin.ops")
        self.assertEqual(module.__name__, "fake_rpc_plugin.ops")


class TestHostGate(_EnvGuard):
    """Importing a plugin package outside its host must bind no port."""

    def test_not_hosted_when_the_host_module_is_absent(self):
        plugin = _make_plugin()
        self.assertFalse(plugin.is_hosted())
        self.assertIsNone(plugin.autostart())
        self.assertFalse(plugin.is_running())

    def test_hosted_when_the_host_module_resolves(self):
        # `json` stands in for `mset` / `substance_painter`: always importable here.
        plugin = _make_plugin(host_module="json")
        self.assertTrue(plugin.is_hosted())

    def test_autostart_env_opt_out_wins_even_inside_the_host(self):
        os.environ["TEST_RPC_AUTOSTART"] = "0"
        plugin = _make_plugin(host_module="json")
        self.assertTrue(plugin.is_hosted())
        self.assertIsNone(plugin.autostart())

    def test_autostart_safely_reports_instead_of_raising(self):
        """A port squabble must never take the host application down."""
        plugin = _make_plugin(host_module="json")
        plugin.start = lambda *a, **kw: (_ for _ in ()).throw(OSError("port in use"))
        self.assertIsNone(plugin.autostart_safely())


class TestPortResolution(_EnvGuard):
    def test_default_port_when_unset(self):
        self.assertEqual(_make_plugin(default_port=8765).port, 8765)

    def test_env_overrides_the_default(self):
        os.environ["TEST_RPC_PORT"] = "9111"
        self.assertEqual(_make_plugin(default_port=8765).port, 9111)

    def test_a_junk_env_value_falls_back_rather_than_crashing_the_host(self):
        os.environ["TEST_RPC_PORT"] = "not-a-port"
        self.assertEqual(_make_plugin(default_port=8765).port, 8765)


class TestWireContract(_EnvGuard):
    """A live server driven by the real client — one protocol, both ends.

    Takes the marshaller's documented opt-out: the client call blocks *this*
    thread, so if anything earlier in the session stood up a ``QCoreApplication``
    (``TestMainThreadMarshallerWithQt`` does) the server's hop back onto the main
    thread would wait on a loop that is waiting on the server. Production hosts
    pump their loop and never hit it.
    """

    def setUp(self):
        super().setUp()
        os.environ["TEST_RPC_DISABLE_MAIN_THREAD"] = "1"
        os.environ["TEST_RPC_OTHER_DISABLE_MAIN_THREAD"] = "1"  # second plugin below
        self.plugin = _make_plugin()

        @self.plugin.registry.register("math.add")
        def _add(a, b=1):
            return a + b

        @self.plugin.registry.register("boom")
        def _boom():
            raise ValueError("op failed")

        host, port = self.plugin.start()
        self.client = RpcClient(port=port, host=host)

    def tearDown(self):
        self.plugin.stop()
        super().tearDown()

    def test_start_is_idempotent_and_reports_its_address(self):
        self.assertTrue(self.plugin.is_running())
        self.assertEqual(self.plugin.start(), self.plugin.address)

    def test_health(self):
        self.assertTrue(self.client.ping())

    def test_invoke_passes_kwargs_and_returns_the_value(self):
        self.assertEqual(self.client.invoke("math.add", a=2, b=3), 5)
        self.assertEqual(self.client.invoke("math.add", a=2), 3)  # default applies

    def test_list_ops_matches_the_registry(self):
        self.assertEqual(sorted(self.client.list_ops()), self.plugin.registry.all_ops())

    def test_the_client_contract_ops_exist_without_the_plugin_defining_them(self):
        """`RpcClient.list_ops` invokes an OP; a plugin must not have to remember it."""
        for op in ("system.ping", "system.list_ops", "system.describe"):
            self.assertIn(op, self.client.list_ops())

    def test_describe_round_trips(self):
        described = self.client.describe("math.add")
        self.assertEqual(described["name"], "math.add")
        self.assertEqual(
            described["params"],
            [{"name": "a", "default": "<required>"}, {"name": "b", "default": "1"}],
        )

    def test_an_unknown_op_is_reported_with_the_available_set(self):
        with self.assertRaises(Exception) as ctx:
            self.client.invoke("does.not.exist")
        self.assertIn("does.not.exist", str(ctx.exception))

    def test_an_op_that_raises_surfaces_its_error(self):
        with self.assertRaises(Exception) as ctx:
            self.client.invoke("boom")
        self.assertIn("op failed", str(ctx.exception))

    def test_stop_releases_the_port(self):
        self.plugin.stop()
        self.assertFalse(self.plugin.is_running())
        self.assertIsNone(self.plugin.address)
        self.assertFalse(self.client.ping(timeout=0.5))

    def test_two_plugins_coexist_in_one_process(self):
        """Per-instance state is what makes this possible at all."""
        other = _make_plugin(label="other_rpc", env_prefix="TEST_RPC_OTHER")
        other.registry.register("only.other")(lambda: "yes")
        host, port = other.start()
        try:
            self.assertEqual(
                RpcClient(port=port, host=host).invoke("only.other"), "yes"
            )
            # Neither registry leaked into the other.
            self.assertNotIn("only.other", self.client.list_ops())
            self.assertNotIn("math.add", other.registry.all_ops())
        finally:
            other.stop()


class TestRootExport(unittest.TestCase):
    def test_registered_on_package_root(self):
        import pythontk as ptk

        self.assertTrue(hasattr(ptk, "RpcPlugin"))
        self.assertTrue(hasattr(ptk, "OpRegistry"))
        self.assertTrue(hasattr(ptk, "MainThreadMarshaller"))


class TestBindPolicy(unittest.TestCase):
    """Both pinned-port servers must refuse to bind over a LIVE listener.

    ``SO_REUSEADDR`` does not mean the same thing on both platforms. On POSIX
    it reuses a ``TIME_WAIT`` port -- what you want when a DCC restarts a
    server on a pinned port. On Windows it *additionally* permits binding over
    a live listener: two servers both start and the stack picks which socket
    gets each request, so a client can be talking to a stale process from a
    previous session while the new one looks healthy.

    ``preview/server.py`` reasoned this out and set
    ``allow_reuse_address = os.name != "nt"``. ``plugin_core``'s
    ``_ReusableServer`` kept an unconditional ``True`` with a docstring giving
    only the POSIX rationale -- and it is the one that actually runs on a
    pinned port inside Toolbag and Painter. The two are asserted together
    because the hazard is identical and the answer must not drift again.
    """

    @staticmethod
    def _expected():
        return os.name != "nt"

    def test_the_rpc_plugin_server_refuses_a_live_listener(self):
        from pythontk.net_utils.rpc import plugin_core

        self.assertEqual(
            plugin_core._ReusableServer.allow_reuse_address,
            self._expected(),
            "on Windows SO_REUSEADDR lets a second server bind over a live "
            "one, so requests can reach a stale process",
        )

    def test_the_preview_server_agrees(self):
        from pythontk.net_utils.preview import server

        self.assertEqual(
            server._PreviewHTTPServer.allow_reuse_address, self._expected()
        )

    def test_the_two_servers_share_one_policy(self):
        from pythontk.net_utils.preview import server
        from pythontk.net_utils.rpc import plugin_core

        self.assertEqual(
            plugin_core._ReusableServer.allow_reuse_address,
            server._PreviewHTTPServer.allow_reuse_address,
            "same hazard, same pinned-port-inside-a-DCC shape: one answer",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
