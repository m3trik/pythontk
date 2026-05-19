# !/usr/bin/python
# coding=utf-8
"""One-shot batch pipeline over :class:`RpcClient`.

Build a list of :class:`Call`\\ s, hand them to :func:`run_batch`, get
back a list of :class:`Result`\\ s. The common case for scripted
pipelines that don't want to manage the connection lifecycle themselves.

Example::

    from pythontk.net_utils.rpc.client import RpcClient
    from pythontk.net_utils.rpc.job import Call, run_batch

    results = run_batch(
        [Call("system.version"), Call("scene.list_materials")],
        client=RpcClient(port=8765, app_label="Marmoset"),
    )
    for r in results:
        print(r.op, r.ok, r.value if r.ok else r.error)
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .client import RpcClient

__all__ = ["Call", "Result", "run_batch"]


@dataclass(frozen=True)
class Call:
    """One queued op invocation.

    *op* is the registry name; *kwargs* are forwarded verbatim to the
    registered function. *timeout* applies per-call.
    """
    op: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 60.0


@dataclass(frozen=True)
class Result:
    """Outcome of a single :class:`Call`.

    ``ok`` is True on success; ``value`` holds the op's return on
    success, ``error`` holds the exception message on failure. ``op``
    echoes :class:`Call.op` for correlation when iterating.
    """
    op: str
    ok: bool
    value: Any = None
    error: Optional[str] = None


def run_batch(
    calls: List[Call],
    client: RpcClient,
    stop_on_error: bool = False,
) -> List[Result]:
    """Connect, fire every call in *calls*, return a Result per call.

    Args:
        calls: Sequence of :class:`Call` objects to execute in order.
        client: Pre-configured :class:`RpcClient` (or subclass). Pinged
            once up-front; raises ``ConnectionError`` if unreachable.
        stop_on_error: Short-circuit on the first failure. Default is to
            run every call regardless -- useful when each call is
            independent and you want a complete report.

    The plugin must already be loaded inside a running DCC; this helper
    does NOT auto-launch. Use ``client.connect(...)`` upstream if you
    want the launch-on-miss behaviour.
    """
    if not client.ping(timeout=2.0):
        raise ConnectionError(
            f"{client.app_label} plugin not reachable at {client.url!r}"
        )

    results: List[Result] = []
    for c in calls:
        try:
            value = client.invoke(c.op, timeout=c.timeout, **c.kwargs)
            results.append(Result(op=c.op, ok=True, value=value))
        except Exception as exc:
            results.append(Result(op=c.op, ok=False, error=str(exc)))
            if stop_on_error:
                break
    return results
