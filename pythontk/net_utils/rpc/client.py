# !/usr/bin/python
# coding=utf-8
"""Generic HTTP JSON-RPC client for plugin-hosted RPC servers.

Drives the "Python client talks to a long-lived process over loopback"
pattern. The remote end is any HTTP server that speaks the wire format
below; nothing here is host-application-specific.

Wire format:

* **Health**:    ``GET  /health``    -> 200 OK if reachable
* **Invoke**:    ``POST /``          ``{"op": "<name>", "kwargs": {...}}``
                                       -> ``{"ok": true, "value": ...}`` or
                                          ``{"ok": false, "error": "..."}``
* **Describe**:  ``POST /describe``  ``{"op": "<name>" | ""}``
                                       -> ``{"value": {...} or [...]}``

Adapters subclass :class:`RpcClient` to bind defaults (port, app finder,
label) for a given host application.

Session-safety note: :meth:`shutdown` only acts on the process that
:meth:`connect` launched. A host app the user opened manually is never
touched -- that's the whole guarantee of the bridge.
"""
import atexit
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

__all__ = ["RpcClient"]


class RpcClient:
    """Generic HTTP JSON-RPC client for a DCC plugin server.

    Subclass to bind defaults for a specific DCC. The base class is
    intentionally usable on its own for tests and bespoke pipelines.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        app_label: str = "DCC plugin",
        find_exe: Optional[Callable[[], Optional[str]]] = None,
    ):
        self.host = host
        self.port = port
        self.app_label = app_label
        self._find_exe = find_exe
        self._launched_process = None
        self._atexit_registered = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    # ------------------------------------------------------------------
    # Probes
    # ------------------------------------------------------------------

    def ping(self, timeout: float = 1.0) -> bool:
        """Return True if the plugin's HTTP server is reachable."""
        try:
            with urllib.request.urlopen(
                f"{self.url}health", timeout=timeout
            ) as resp:
                return resp.status == 200
        except (urllib.error.URLError, ConnectionError, OSError):
            return False

    # ------------------------------------------------------------------
    # RPC surface
    # ------------------------------------------------------------------

    def invoke(self, op: str, timeout: float = 60.0, **kwargs: Any) -> Any:
        """Call *op* with *kwargs* and return its value.

        Raises:
            ConnectionError: the plugin didn't answer (DCC closed or
                plugin not loaded). Use :meth:`ping` to pre-check.
            RuntimeError: the op ran but failed; the message includes
                the exception type the DCC raised.
        """
        payload = json.dumps({"op": op, "kwargs": kwargs}).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Server responded non-2xx. The body may not be our JSON
            # envelope (framework default error pages are HTML/empty).
            raw = e.read().decode("utf-8", "replace")
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"Op {op!r} failed: HTTP {e.code} with non-JSON body: {raw!r}"
                ) from e
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            raise ConnectionError(
                f"{self.app_label} plugin not reachable at {self.url!r}: {e}"
            ) from e

        if not body.get("ok"):
            err = body.get("error", "unknown")
            raise RuntimeError(f"Op {op!r} failed: {err}")
        return body.get("value")

    def list_ops(self) -> list:
        """Convenience: ``self.invoke('system.list_ops')``.

        Raises :class:`RuntimeError` if the plugin doesn't register a
        ``system.list_ops`` op. Adapters that target a plugin without
        one should override this.
        """
        return self.invoke("system.list_ops")

    def describe(self, op: str = "", timeout: float = 5.0) -> Any:
        """Return one op's description, or all ops if *op* is empty.

        Goes through the dedicated ``/describe`` route so a buggy
        registry can't break introspection.
        """
        payload = json.dumps({"op": op}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}describe",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            raise ConnectionError(
                f"{self.app_label} plugin not reachable at {self.url!r}: {e}"
            ) from e
        return body.get("value")

    # ------------------------------------------------------------------
    # Lifecycle: connect / shutdown
    # ------------------------------------------------------------------

    def connect(
        self,
        exe: Optional[str] = None,
        timeout: float = 30.0,
        force_new: bool = False,
        poll_interval: float = 0.5,
        auto_cleanup: bool = False,
    ) -> bool:
        """Ensure the plugin is reachable; launch the DCC if it isn't.

        Args:
            exe: Explicit DCC executable path. If None, uses the
                ``find_exe`` callable passed at construction.
            timeout: Seconds to wait for the plugin's HTTP server to
                come up after launch.
            force_new: When True, skip the reuse-existing check and
                launch a fresh DCC unconditionally. *Caveat*: two DCC
                processes cannot both bind the same RPC port; if an
                existing instance is already listening, the new one's
                plugin will fail to start its server. ``ping`` still hits
                the *original* process. Close the existing DCC first (or
                pass a different ``port=``) for true isolation.
            poll_interval: Seconds between ``ping`` retries while
                waiting for the plugin.
            auto_cleanup: If True, register an ``atexit`` handler that
                calls :meth:`shutdown` on interpreter exit. Idempotent
                across repeated connect() calls.

        Returns:
            True if the plugin is reachable (existing or newly launched),
            False if the launch timed out.

        Raises:
            FileNotFoundError: no exe path and no find_exe callable.
            RuntimeError: AppLauncher.launch returned None.
        """
        if not force_new and self.ping(timeout=1.0):
            if auto_cleanup:
                self._register_atexit_cleanup()
            return True

        from pythontk import AppLauncher

        resolved_exe = exe or (self._find_exe() if self._find_exe else None)
        if not resolved_exe:
            raise FileNotFoundError(
                f"{self.app_label} not found. Pass exe=... or add the "
                "executable to PATH."
            )

        self._launched_process = AppLauncher.launch(resolved_exe, detached=True)
        if self._launched_process is None:
            raise RuntimeError(f"AppLauncher could not launch {resolved_exe!r}")

        if auto_cleanup:
            self._register_atexit_cleanup()

        start = time.time()
        while time.time() - start < timeout:
            if self._launched_process.poll() is not None:
                # DCC died before the plugin came up.
                return False
            if self.ping(timeout=1.0):
                return True
            time.sleep(poll_interval)
        return False

    def shutdown(self, force: bool = False) -> None:
        """Terminate the DCC process this connection launched.

        Only acts on the process started by :meth:`connect`; a DCC the
        user opened manually is never touched -- that's the safety
        guarantee of the RPC bridge.
        """
        proc = self._launched_process
        if proc is None:
            return
        try:
            from pythontk import AppLauncher
            AppLauncher.close_process(proc.pid, force=force)
        except Exception as exc:  # noqa: BLE001
            # Don't re-raise -- best-effort path. But warn so a leaked
            # process doesn't quietly eat RAM until the user reboots.
            print(
                f"[{self.app_label}] shutdown: close_process(pid={proc.pid}) "
                f"failed: {exc!r}. The host process may still be running.",
                file=sys.stderr,
            )
        self._launched_process = None

    def _register_atexit_cleanup(self) -> None:
        """Register an idempotent ``atexit`` handler.

        Captures :class:`AppLauncher` *now* (at register time) so the
        cleanup path doesn't trip over CPython nulling module globals
        during interpreter shutdown.
        """
        if self._atexit_registered:
            return
        try:
            from pythontk import AppLauncher
        except Exception as exc:  # noqa: BLE001
            print(
                f"[{self.app_label}] auto_cleanup: pythontk unavailable, "
                f"cleanup disabled ({exc!r}). Launched host process "
                "will not be terminated.",
                file=sys.stderr,
            )
            self._atexit_registered = True
            return

        def _cleanup():
            proc = self._launched_process
            if proc is None:
                return
            try:
                AppLauncher.close_process(proc.pid, force=True)
            except Exception as exc:  # noqa: BLE001
                # Never let atexit raise -- it would mask the original
                # interpreter-exit reason.
                print(
                    f"[{self.app_label}] atexit cleanup failed: {exc!r}. "
                    f"Launched host process (PID {getattr(proc, 'pid', '?')}) "
                    "may still be running.",
                    file=sys.stderr,
                )
            self._launched_process = None

        atexit.register(_cleanup)
        self._atexit_registered = True

    # ------------------------------------------------------------------
    # Context-manager sugar
    # ------------------------------------------------------------------

    def __enter__(self) -> "RpcClient":
        if not self.ping(timeout=0.5):
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Only shut down what we launched. Leaves user-launched DCCs alone.
        self.shutdown(force=True)
