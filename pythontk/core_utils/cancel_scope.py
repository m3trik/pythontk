# !/usr/bin/python
# coding=utf-8
"""Cooperative cancellation — one scope shared by every cancel affordance.

The ecosystem historically had two unrelated ways to stop a long operation:
``ExecutionMonitor``'s background thread (which polled the *global* key state
and delivered ``KeyboardInterrupt`` into the main thread) and uitk's progress
bar (whose ``update()`` returned ``False`` after an Esc-hold). The first was
unreliable by construction — an async exception lands at an arbitrary bytecode
boundary, cannot be revoked once armed, and never preempts a native call at all
— while the second could only see Esc as often as the caller ticked.

:class:`CancelScope` replaces both mechanisms' *state* with a single object.
Cancellation is always **cooperative**: something sets a flag, and the operation
notices at a point it chose. Nothing here can preempt a running native call;
that is a property of the runtime, not a limitation to be engineered around.

Two directions feed one scope:

* **push** — :meth:`CancelScope.cancel` from anywhere (a dialog button, a
  monitor thread, a Qt shortcut). Thread-safe and idempotent.
* **pull** — a source callable polled *by the operation's own thread* inside
  :meth:`poll` / :meth:`tick` / :meth:`checkpoint`. This is the shape a DCC
  needs: Maya's ``MComputation.isInterruptRequested()`` peeks the OS input
  queue for Esc without pumping the event loop, so it works during the long
  synchronous stretches where a Qt shortcut cannot be delivered.

Two consumption styles, both first-class — a scope does not care which the
caller uses, so a bool-style loop and an exception-style helper can cancel the
same operation:

    if not scope.tick():        # bool style (progress-bar contract)
        break

    ptk.CancelScope.check()     # exception style, ambient — deep helpers use
                                # this without taking a scope parameter

Ambient scopes use a :class:`contextvars.ContextVar`, so library code far below
the caller can honour cancellation with no signature churn. Note that a new
thread starts with an empty context: :meth:`current` returns ``None`` off the
activating thread, by design — pull sources must never be polled from a
non-owner thread (a DCC API call from a worker thread crashes the host).
"""
from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from typing import Callable, Iterable, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class OperationCancelled(BaseException):
    """Raised at a checkpoint when the governing scope has been cancelled.

    Derives from :class:`BaseException`, not :class:`Exception`, for the same
    reason :class:`asyncio.CancelledError` does: a cancel must not be swallowed
    by the broad ``except Exception`` blocks that long-running tool code is full
    of. Callers that genuinely need to clean up should use ``try/finally`` or
    catch this class by name.
    """

    def __init__(self, message: str = "Operation cancelled", scope=None, reason=None):
        super().__init__(message)
        self.scope = scope
        self.reason = reason


class CancelScope:
    """A cancellation flag with pull sources, ambient activation, and metrics.

    Parameters:
        name (str): Human-readable label used in logs and dialogs.
        sources (Iterable[Callable[[], bool]]): Pull sources — callables that
            return ``True`` when cancellation has been requested. Polled only
            on the owner thread (see module docstring).
        poll_min_interval (float): Minimum seconds between *source* polls, so a
            tight ``check()`` loop stays cheap. The flag itself is always read
            without throttling; ``0`` polls sources on every call.

    Example:
        scope = CancelScope("Bake textures")
        with scope.activate():
            for i, item in enumerate(items):
                if not scope.tick():
                    break
                process(item)
    """

    _ambient: contextvars.ContextVar = contextvars.ContextVar(
        "pythontk_cancel_scope", default=None
    )

    def __init__(
        self,
        name: str = "",
        sources: Iterable[Callable[[], bool]] = (),
        poll_min_interval: float = 0.05,
    ):
        self.name = name or "operation"
        self._lock = threading.RLock()
        self._event = threading.Event()
        self._reason: Optional[str] = None
        self._sources: List[Callable[[], bool]] = list(sources)
        self._listeners: List[Callable[["CancelScope"], None]] = []
        self._poll_min_interval = max(0.0, float(poll_min_interval))

        self._parent: Optional["CancelScope"] = None
        self._owner_ident: int = threading.get_ident()
        self._activations: List[contextlib.AbstractContextManager] = []

        self._started_at = time.monotonic()
        self._last_activity: Optional[float] = None
        self._last_source_poll: Optional[float] = None
        self._cancel_requested_at: Optional[float] = None
        self._consumed = False
        self._tick_count = 0

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def cancelled(self) -> bool:
        """True when this scope (or an enclosing one) has been cancelled.

        Reads flags only — never polls sources, so it is safe from any thread
        and free to call in a hot loop.
        """
        if self._event.is_set():
            return True
        parent = self._parent
        return bool(parent is not None and parent.cancelled)

    @property
    def reason(self) -> Optional[str]:
        """Why the scope was cancelled (``None`` while it is still live)."""
        if self._reason is not None:
            return self._reason
        parent = self._parent
        return parent.reason if parent is not None else None

    @property
    def tick_count(self) -> int:
        """Number of cooperative checkpoints reached so far."""
        return self._tick_count

    @property
    def consumed(self) -> bool:
        """True once a checkpoint has actually *reported* the cancellation.

        :attr:`cancelled` says a stop was **requested**; this says the operation
        reached a checkpoint and was told about it, which is the only evidence
        that it abandoned its work. The two diverge in both directions and the
        gap is not academic:

        * a slot with no cooperative checkpoints runs to completion with the
          flag set behind it, so ``cancelled`` is True for work that fully
          happened;
        * a progress bar that dropped its scope reference mid-task never polls
          again, so the loop finishes even though Esc was pressed.

        A caller deciding whether to **undo** must use this one. Acting on
        ``cancelled`` there rolls back completed work, which is worse than not
        offering rollback at all.
        """
        return self._consumed

    @property
    def has_ticked(self) -> bool:
        """True once the operation has reached at least one checkpoint.

        ``False`` is the tell that an operation is *structurally* uncancellable
        — one monolithic native call with no cooperative points — which lets
        callers tell the user the truth instead of implying Esc will work.
        """
        return self._tick_count > 0

    @property
    def elapsed(self) -> float:
        """Seconds since the scope was created."""
        return time.monotonic() - self._started_at

    @property
    def elapsed_since_tick(self) -> Optional[float]:
        """Seconds since the last checkpoint, or ``None`` if never ticked."""
        if self._last_activity is None:
            return None
        return time.monotonic() - self._last_activity

    @property
    def elapsed_since_request(self) -> Optional[float]:
        """Seconds since cancellation was requested, or ``None`` if not.

        A large value means the request has not been *consumed* — the operation
        is stuck in something that cannot be interrupted cooperatively.
        """
        if self._cancel_requested_at is None:
            return None
        return time.monotonic() - self._cancel_requested_at

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def add_source(self, source: Callable[[], bool]) -> "CancelScope":
        """Register a pull source; returns self for chaining.

        The source is polled only from the owner thread — the thread that
        created or last activated the scope.
        """
        if callable(source):
            with self._lock:
                if source not in self._sources:
                    self._sources.append(source)
        return self

    def remove_source(self, source: Callable[[], bool]) -> "CancelScope":
        """Unregister a pull source (no-op when absent)."""
        with self._lock:
            with contextlib.suppress(ValueError):
                self._sources.remove(source)
        return self

    def add_listener(self, listener: Callable[["CancelScope"], None]) -> "CancelScope":
        """Register a callback fired once, when the scope is cancelled.

        Listeners run on the *cancelling* thread, which may not be the owner
        thread — a listener that touches UI must marshal the call itself.
        Exceptions from listeners are suppressed: a broken observer must not
        break cancellation.
        """
        if callable(listener):
            with self._lock:
                if listener not in self._listeners:
                    self._listeners.append(listener)
        return self

    def remove_listener(self, listener: Callable[["CancelScope"], None]) -> "CancelScope":
        """Unregister a cancel listener (no-op when absent)."""
        with self._lock:
            with contextlib.suppress(ValueError):
                self._listeners.remove(listener)
        return self

    # ------------------------------------------------------------------
    # Cancel / reset
    # ------------------------------------------------------------------
    def cancel(self, reason: Optional[str] = None) -> None:
        """Request cancellation. Thread-safe, idempotent, never raises."""
        with self._lock:
            if self._event.is_set():
                return
            self._reason = reason or "cancelled"
            self._cancel_requested_at = time.monotonic()
            self._event.set()
            listeners = list(self._listeners)

        for listener in listeners:
            try:
                listener(self)
            except Exception:
                pass

    def reset(self) -> None:
        """Clear cancellation and metrics so the scope can be reused.

        Intended for long-lived owners (a progress bar reused across tasks),
        not for resurrecting a scope mid-operation.
        """
        with self._lock:
            self._event.clear()
            self._reason = None
            self._cancel_requested_at = None
            self._consumed = False
            self._last_activity = None
            self._last_source_poll = None
            self._tick_count = 0
            self._started_at = time.monotonic()

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def poll(self) -> bool:
        """Reach a checkpoint; return ``True`` when cancelled.

        Polls registered sources when called on the owner thread (throttled by
        *poll_min_interval*), then reports the flag. Counts as activity.
        """
        self._tick_count += 1
        now = time.monotonic()
        self._last_activity = now

        if not self._event.is_set() and threading.get_ident() == self._owner_ident:
            due = (
                self._last_source_poll is None
                or (now - self._last_source_poll) >= self._poll_min_interval
            )
            if due:
                self._last_source_poll = now
                for source in list(self._sources):
                    try:
                        if source():
                            self.cancel("source")
                            break
                    except Exception:
                        # A failing source must not take down the operation it
                        # was meant to protect; treat it as "no request".
                        pass

        # Reporting True *is* the consumption: this is the moment the operation
        # is told to stop, and the only proof it could have acted on it.
        if self._event.is_set():
            self._consumed = True
            return True
        parent = self._parent
        if parent is not None and parent.poll():
            self._consumed = True
            return True
        return False

    def tick(self) -> bool:
        """Checkpoint in bool style: ``True`` to continue, ``False`` if cancelled.

        Matches the progress-bar ``update()`` contract, so the two can be wired
        together without an adapter.
        """
        return not self.poll()

    def checkpoint(self) -> None:
        """Checkpoint in exception style: raise :class:`OperationCancelled`."""
        if self.poll():
            raise OperationCancelled(
                f"'{self.name}' cancelled", scope=self, reason=self.reason
            )

    def iterate(self, iterable: Iterable[T]) -> Iterator[T]:
        """Yield from *iterable*, checkpointing before each item.

        Convenience for the overwhelmingly common per-item loop::

            for item in scope.iterate(items):
                process(item)
        """
        for item in iterable:
            self.checkpoint()
            yield item

    # ------------------------------------------------------------------
    # Ambient activation
    # ------------------------------------------------------------------
    def _would_cycle(self, candidate: Optional["CancelScope"]) -> bool:
        """True when linking *candidate* as parent would form a loop.

        Re-activating an already-active outer scope from inside an inner one
        (``A → B → A``) would otherwise make :attr:`cancelled` recurse until the
        interpreter gives up. Rare, but this is a public primitive and the
        failure mode is a hang, so the link is simply skipped instead.
        """
        node = candidate
        while node is not None:
            if node is self:
                return True
            node = node._parent
        return False

    @contextlib.contextmanager
    def activate(self):
        """Install this scope as the ambient one for the calling thread.

        Nesting links the enclosing scope as parent, so a cancelled outer scope
        cancels inner checkpoints too. Also claims ownership for source polling
        — the activating thread is the one that runs the operation.

        Re-entrant: activating a scope that is already active restores the
        previous parent/owner on exit rather than clearing them, so an inner
        block can't strip the links an outer one depends on.
        """
        previous = CancelScope._ambient.get()
        with self._lock:
            prior_parent = self._parent
            prior_owner = self._owner_ident
            self._parent = None if self._would_cycle(previous) else previous
            self._owner_ident = threading.get_ident()
        token = CancelScope._ambient.set(self)
        try:
            yield self
        finally:
            CancelScope._ambient.reset(token)
            with self._lock:
                self._parent = prior_parent
                self._owner_ident = prior_owner

    def __enter__(self) -> "CancelScope":
        # A stack, not a single slot. Holding one activation meant ``with
        # scope:`` nested inside another ``with scope:`` overwrote the outer
        # record — and dropping that reference finalized the outer generator
        # *immediately*, running its ``finally`` (ContextVar reset, parent
        # unlink) while the outer block was still executing. The scope stopped
        # being ambient part-way through itself, so a checkpoint after the
        # inner block silently found nothing and never cancelled.
        activation = self.activate()
        self._activations.append(activation)
        return activation.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._activations:
            return False
        return self._activations.pop().__exit__(exc_type, exc_val, exc_tb)

    # ------------------------------------------------------------------
    # Ambient accessors
    # ------------------------------------------------------------------
    # NOTE: these resolve the *ambient* scope regardless of the receiver —
    # ``some_scope.check()`` checks whatever is active, not ``some_scope``.
    # Call them on the class (``CancelScope.check()``) to keep that obvious.
    @classmethod
    def current(cls) -> Optional["CancelScope"]:
        """The scope active on this thread, or ``None``."""
        return cls._ambient.get()

    @classmethod
    def check(cls) -> None:
        """Ambient checkpoint (exception style). No-op when no scope is active.

        The entry point for deep helpers: a mayatk/blendertk loop can call
        ``ptk.CancelScope.check()`` without accepting a scope parameter, and
        stays a plain function when nothing is monitoring it.
        """
        scope = cls._ambient.get()
        if scope is not None:
            scope.checkpoint()

    @classmethod
    def proceed(cls) -> bool:
        """Ambient checkpoint (bool style). ``True`` when no scope is active."""
        scope = cls._ambient.get()
        return True if scope is None else scope.tick()

    @classmethod
    def is_cancelled(cls) -> bool:
        """Flag-only read of the ambient scope; ``False`` when none is active."""
        scope = cls._ambient.get()
        return bool(scope is not None and scope.cancelled)

    def __repr__(self) -> str:
        state = f"cancelled({self.reason})" if self.cancelled else "live"
        return f"<CancelScope {self.name!r} {state} ticks={self._tick_count}>"
