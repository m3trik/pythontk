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
import unittest

from pythontk.net_utils.rpc.client import RpcClient
from pythontk.net_utils.rpc.plugin_core import (
    MainThreadMarshaller,
    OpRegistry,
    RpcPlugin,
)


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


class TestWireContract(unittest.TestCase):
    """A live server driven by the real client — one protocol, both ends."""

    def setUp(self):
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
            self.assertEqual(RpcClient(port=port, host=host).invoke("only.other"), "yes")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
