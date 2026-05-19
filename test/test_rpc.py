# !/usr/bin/python
# coding=utf-8
"""Tests for the generic RPC plumbing in pythontk.net_utils.

Covers the wire-protocol contract (ping/invoke/describe over HTTP),
the install/uninstall strategy (symlink-first, copytree fallback,
``__pycache__`` filter), and the batch pipeline (Call/Result/run_batch).

DCC-specific bindings (Toolbag's port, version-aware path resolver) are
tested in :mod:`mayatk.test.mock_tests.test_marmoset_rpc`.
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import unittest.mock
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythontk.net_utils.rpc.client import RpcClient
from pythontk.net_utils.rpc.installer import (
    install_plugin,
    uninstall_plugin,
    is_plugin_installed,
)
from pythontk.net_utils.rpc.job import Call, Result, run_batch


# ----------------------------------------------------------------------
# Test fixture: a tiny in-process server that speaks the wire format
# ----------------------------------------------------------------------


class _StubHandler(BaseHTTPRequestHandler):
    """Matches the marmoset_rpc plugin's wire format closely enough for
    client tests. Routes:
      GET  /health   -> 200
      POST /         -> dispatch from class.OPS table
      POST /describe -> echo back the op name (or a synthetic listing).
    """

    OPS = {
        "system.ping": lambda **kw: "pong",
        "system.list_ops": lambda **kw: ["system.ping", "echo"],
        "echo": lambda **kw: kw,
        "explode": lambda **kw: (_ for _ in ()).throw(ValueError("boom")),
    }

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"ok": True})
        else:
            self._respond(404, {"ok": False, "error": "no"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        req = json.loads(raw) if raw else {}

        if self.path == "/describe":
            op = req.get("op")
            if op == "":
                self._respond(200, {"ok": True, "value": [
                    {"name": "system.ping", "doc": "Heartbeat.", "params": []}
                ]})
            elif op == "system.ping":
                self._respond(200, {"ok": True, "value": {
                    "name": "system.ping", "doc": "Heartbeat.", "params": []
                }})
            else:
                self._respond(200, {"ok": True, "value": None})
            return

        op = req.get("op")
        kwargs = req.get("kwargs") or {}
        fn = self.OPS.get(op)
        if fn is None:
            self._respond(404, {
                "ok": False, "error": f"Unknown op: {op!r}",
            })
            return
        try:
            value = fn(**kwargs)
        except Exception as exc:
            self._respond(500, {
                "ok": False, "error": f"{type(exc).__name__}: {exc}",
            })
            return
        self._respond(200, {"ok": True, "value": value})

    def log_message(self, *_a, **_kw):
        pass  # silence

    def _respond(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _StubServer:
    """Context-manager wrapper around an in-process HTTPServer."""

    def __init__(self):
        self.port = _free_port()
        self._server = HTTPServer(("127.0.0.1", self.port), _StubHandler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_a):
        self._server.shutdown()
        self._server.server_close()


# ----------------------------------------------------------------------
# RpcClient: wire protocol contract
# ----------------------------------------------------------------------


class TestRpcClientWireProtocol(unittest.TestCase):
    def test_ping_succeeds_against_running_server(self):
        with _StubServer() as srv:
            client = RpcClient(port=srv.port, app_label="stub")
            self.assertTrue(client.ping(timeout=2.0))

    def test_ping_fails_against_dead_port(self):
        client = RpcClient(port=_free_port(), app_label="stub")
        self.assertFalse(client.ping(timeout=0.5))

    def test_invoke_returns_value(self):
        with _StubServer() as srv:
            client = RpcClient(port=srv.port, app_label="stub")
            self.assertEqual(client.invoke("system.ping"), "pong")

    def test_invoke_forwards_kwargs(self):
        with _StubServer() as srv:
            client = RpcClient(port=srv.port, app_label="stub")
            self.assertEqual(
                client.invoke("echo", a=1, b="x"),
                {"a": 1, "b": "x"},
            )

    def test_invoke_unknown_op_raises_runtimeerror(self):
        with _StubServer() as srv:
            client = RpcClient(port=srv.port, app_label="stub")
            with self.assertRaises(RuntimeError) as ctx:
                client.invoke("does.not.exist")
            self.assertIn("Unknown op", str(ctx.exception))

    def test_invoke_op_failure_propagates_exception_text(self):
        with _StubServer() as srv:
            client = RpcClient(port=srv.port, app_label="stub")
            with self.assertRaises(RuntimeError) as ctx:
                client.invoke("explode")
            self.assertIn("ValueError", str(ctx.exception))
            self.assertIn("boom", str(ctx.exception))

    def test_invoke_raises_connectionerror_when_unreachable(self):
        client = RpcClient(port=_free_port(), app_label="stub")
        with self.assertRaises(ConnectionError):
            client.invoke("system.ping", timeout=0.5)

    def test_list_ops_convenience(self):
        with _StubServer() as srv:
            client = RpcClient(port=srv.port, app_label="stub")
            ops = client.list_ops()
            self.assertIn("system.ping", ops)

    def test_describe_single_op(self):
        with _StubServer() as srv:
            client = RpcClient(port=srv.port, app_label="stub")
            d = client.describe("system.ping")
            self.assertEqual(d["name"], "system.ping")

    def test_describe_all_ops(self):
        with _StubServer() as srv:
            client = RpcClient(port=srv.port, app_label="stub")
            listing = client.describe("")
            self.assertIsInstance(listing, list)

    def test_app_label_surfaces_in_unreachable_error(self):
        client = RpcClient(port=_free_port(), app_label="MyDcc")
        with self.assertRaises(ConnectionError) as ctx:
            client.invoke("foo")
        self.assertIn("MyDcc", str(ctx.exception))


# ----------------------------------------------------------------------
# RpcClient: connect() launch path
# ----------------------------------------------------------------------


class TestRpcClientConnect(unittest.TestCase):
    def test_connect_no_finder_and_no_exe_raises(self):
        client = RpcClient(port=_free_port(), app_label="MyDcc")
        with self.assertRaises(FileNotFoundError) as ctx:
            client.connect(timeout=0.5)
        self.assertIn("MyDcc", str(ctx.exception))

    def test_connect_finder_used_when_no_exe_passed(self):
        called = []

        def _finder():
            called.append(True)
            return None  # report not found

        client = RpcClient(
            port=_free_port(), app_label="MyDcc", find_exe=_finder,
        )
        with self.assertRaises(FileNotFoundError):
            client.connect(timeout=0.5)
        self.assertEqual(called, [True])

    def test_connect_reuses_existing_when_reachable(self):
        with _StubServer() as srv:
            client = RpcClient(port=srv.port, app_label="stub")
            self.assertTrue(client.connect(timeout=1.0))
            # The finder must not have been called -- ping succeeded.
            self.assertIsNone(client._launched_process)

    def test_connect_force_new_bypasses_reuse(self):
        """force_new=True must launch even when a server is reachable."""
        with _StubServer() as srv:
            called = []

            def _finder():
                called.append(True)
                return None  # report not found -> raises FileNotFoundError

            client = RpcClient(
                port=srv.port, app_label="stub", find_exe=_finder,
            )
            with self.assertRaises(FileNotFoundError):
                client.connect(timeout=0.5, force_new=True)
            self.assertEqual(called, [True],
                             "force_new must skip the reuse path.")

    def test_connect_auto_cleanup_registers_atexit(self):
        """auto_cleanup=True registers an idempotent atexit hook."""
        with _StubServer() as srv:
            with unittest.mock.patch("atexit.register") as mock_register:
                client = RpcClient(port=srv.port, app_label="stub")
                client.connect(timeout=1.0, auto_cleanup=True)
                client.connect(timeout=1.0, auto_cleanup=True)  # idempotent
                self.assertEqual(mock_register.call_count, 1)


# ----------------------------------------------------------------------
# RpcClient: shutdown + context-manager lifecycle
# ----------------------------------------------------------------------


class TestRpcClientLifecycle(unittest.TestCase):
    def test_shutdown_noop_when_nothing_launched(self):
        """shutdown() on a never-connected client is a no-op."""
        client = RpcClient(port=_free_port(), app_label="stub")
        # Patch AppLauncher.close_process so a stray import can't fire.
        with unittest.mock.patch(
            "pythontk.AppLauncher.close_process"
        ) as mock_close:
            client.shutdown(force=True)
            mock_close.assert_not_called()

    def test_shutdown_terminates_launched_process_only(self):
        """The session-safety contract: only act on what we launched."""
        client = RpcClient(port=_free_port(), app_label="stub")
        proc = unittest.mock.MagicMock(pid=9999)
        client._launched_process = proc
        with unittest.mock.patch(
            "pythontk.AppLauncher.close_process"
        ) as mock_close:
            client.shutdown(force=True)
            mock_close.assert_called_once_with(9999, force=True)
        # And clears the handle so a double-shutdown is a no-op.
        self.assertIsNone(client._launched_process)

    def test_context_manager_pings_and_shuts_down(self):
        """``with RpcClient() as c:`` connects on enter, shuts down on exit."""
        with _StubServer() as srv:
            client = RpcClient(port=srv.port, app_label="stub")
            with unittest.mock.patch(
                "pythontk.AppLauncher.close_process"
            ) as mock_close:
                with client as c:
                    self.assertIs(c, client)
                    self.assertTrue(client.ping(timeout=1.0))
                # Nothing was launched -> close_process must not fire.
                mock_close.assert_not_called()


# ----------------------------------------------------------------------
# Installer: symlink-first + copytree fallback
# ----------------------------------------------------------------------


class TestRpcInstaller(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="rpc_install_test_")
        # Build a fake plugin source dir with the marker file.
        self.src = Path(self._tmp) / "plugin_src" / "my_plugin"
        self.src.mkdir(parents=True)
        (self.src / "__init__.py").write_text("# plugin", encoding="utf-8")
        (self.src / "extra.py").write_text("# extra", encoding="utf-8")
        # Populate __pycache__ so we can prove the filter works on copy.
        (self.src / "__pycache__").mkdir()
        (self.src / "__pycache__" / "extra.cpython-311.pyc").write_bytes(b"\x00")
        self.dest = Path(self._tmp) / "installed" / "my_plugin"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_install_creates_plugin_at_dest(self):
        result = install_plugin(self.src, self.dest)
        self.assertIsNotNone(result)
        self.assertTrue((self.dest / "__init__.py").is_file())
        self.assertTrue(is_plugin_installed(self.dest))

    def test_install_idempotent_without_force(self):
        first = install_plugin(self.src, self.dest)
        marker = self.dest / "_marker.txt"
        marker.write_text("untouched", encoding="utf-8")
        second = install_plugin(self.src, self.dest)
        self.assertEqual(first, second)
        self.assertTrue(marker.is_file(), "Idempotent install wiped the dir.")

    def test_install_force_rebuilds(self):
        install_plugin(self.src, self.dest)
        marker = self.dest / "_marker.txt"
        marker.write_text("delete me", encoding="utf-8")
        install_plugin(self.src, self.dest, force=True)
        self.assertFalse(marker.is_file(), "force=True should rebuild.")

    def test_install_returns_none_for_missing_source(self):
        missing = Path(self._tmp) / "does_not_exist"
        self.assertIsNone(install_plugin(missing, self.dest))

    def test_copytree_fallback_filters_pycache(self):
        """Force the copytree path by patching os.symlink to raise."""
        with unittest.mock.patch("os.symlink", side_effect=OSError("denied")):
            install_plugin(self.src, self.dest)
        # Real files made it across...
        self.assertTrue((self.dest / "__init__.py").is_file())
        self.assertTrue((self.dest / "extra.py").is_file())
        # ...but __pycache__ was filtered.
        self.assertFalse((self.dest / "__pycache__").exists(),
                         "copytree fallback must filter __pycache__")

    def test_uninstall_removes_plugin(self):
        install_plugin(self.src, self.dest)
        self.assertTrue(is_plugin_installed(self.dest))
        self.assertTrue(uninstall_plugin(self.dest))
        self.assertFalse(is_plugin_installed(self.dest))

    def test_uninstall_missing_is_safe(self):
        # Nothing installed yet.
        self.assertFalse(uninstall_plugin(self.dest))


# ----------------------------------------------------------------------
# run_batch: pipeline behaviour
# ----------------------------------------------------------------------


class TestRunBatch(unittest.TestCase):
    def test_run_batch_returns_result_per_call(self):
        with _StubServer() as srv:
            client = RpcClient(port=srv.port, app_label="stub")
            results = run_batch(
                [Call("system.ping"), Call("system.list_ops")],
                client=client,
            )
            self.assertEqual(len(results), 2)
            self.assertTrue(results[0].ok)
            self.assertEqual(results[0].value, "pong")
            self.assertTrue(results[1].ok)
            self.assertIn("system.ping", results[1].value)

    def test_run_batch_records_failures_without_aborting(self):
        with _StubServer() as srv:
            client = RpcClient(port=srv.port, app_label="stub")
            results = run_batch(
                [
                    Call("system.ping"),
                    Call("does.not.exist"),  # fails
                    Call("system.list_ops"),  # still runs
                ],
                client=client,
            )
            self.assertEqual(len(results), 3)
            self.assertTrue(results[0].ok)
            self.assertFalse(results[1].ok)
            self.assertIn("Unknown op", results[1].error)
            self.assertTrue(results[2].ok)

    def test_run_batch_stops_on_error_when_requested(self):
        with _StubServer() as srv:
            client = RpcClient(port=srv.port, app_label="stub")
            results = run_batch(
                [
                    Call("system.ping"),
                    Call("does.not.exist"),  # fails -> abort
                    Call("system.list_ops"),  # must NOT run
                ],
                client=client,
                stop_on_error=True,
            )
            self.assertEqual(len(results), 2)
            self.assertTrue(results[0].ok)
            self.assertFalse(results[1].ok)

    def test_run_batch_raises_when_plugin_unreachable(self):
        client = RpcClient(port=_free_port(), app_label="stub")
        with self.assertRaises(ConnectionError):
            run_batch([Call("system.ping")], client=client)


if __name__ == "__main__":
    unittest.main()
