# !/usr/bin/python
# coding=utf-8
"""Timed multi-step press toggles.

The stateful half of a "press the same key again to go further" hotkey. Where
:meth:`pythontk.CoreUtils.cycle` rotates a fixed sequence forever, a
:class:`StepToggle` models a *home* state plus ``N`` steps away from it, and —
crucially — forgets the deeper cycle once the user pauses. What a stale press
does instead is the toggle's ``stale`` policy:

    press, press, press   (rapid)             ->  step 1, step 2, home
    press ... pause ... press  (``"home"``)    ->  step 1, home
    press ... pause ... press  (``"restart"``) ->  step 1, step 1

``"home"`` suits an on/off toggle (isolate, a display mode): a multi-step
toggle degrades to a plain there-and-back one whenever the user acted once
and that was all they needed. ``"restart"`` suits an *action* whose home is
only an undo (framing a selection): after a pause the key must simply act
again — Maya's ``F`` re-frames after you've tumbled around, it never
teleports you back to a view you left minutes ago — so home stays reachable
only by running the whole cycle within the timeout. Zero-dep and
DCC-agnostic; the clock is injected so the timing is testable without sleeping.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

DEFAULT_TIMEOUT = 2.0  # seconds a deeper step stays reachable
STALE_POLICIES = ("home", "restart")


class StepToggle:
    """A press stepper: ``0`` (home) -> ``1`` -> ... -> ``steps`` -> ``0``.

    Parameters:
        steps (int): Number of states away from home. ``1`` gives a bi-state
            toggle (act / undo), ``2`` a tri-state (act / act harder / undo).
        timeout (float): Seconds after which the cycle goes stale. A stale press
            never steps deeper; what it does instead is ``stale``. ``None``
            disables the timer.
        stale (str): What a stale press does when the toggle is away from home:
            ``"home"`` (default) returns home — the toggle collapses to a plain
            on/off; ``"restart"`` begins a fresh cycle at step 1 instead, so the
            key always *acts* after a pause and home is reachable only by
            completing the cycle within the timeout. A stale press at home
            starts a fresh cycle under either policy.
        clock (callable): Monotonic seconds source (injected for testing).
        name (str): Optional identifier, set by :meth:`get`.

    Attributes:
        payload: Caller-owned slot for whatever must be restored on the way home
            (a camera snapshot, a display mode, ...). Never touched by the class.

    Example:
        toggle = StepToggle.get("frame")
        state = toggle.advance(steps=2, context=selection_signature, timeout=2.0)
        if not state:
            restore(toggle.payload)          # back home
        else:
            if toggle.began_cycle:
                toggle.payload = snapshot()  # entering the cycle
            act(state)
    """

    _instances: Dict[str, "StepToggle"] = {}

    def __init__(
        self,
        steps: int = 2,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        stale: str = "home",
        clock: Callable[[], float] = time.monotonic,
        name: Optional[str] = None,
    ):
        self.steps = max(1, int(steps))
        self.timeout = timeout
        self.stale = self._check_stale(stale)
        self.name = name
        self.payload: Any = None
        self._clock = clock
        self._state = 0
        self._last: Optional[float] = None
        self._context: Any = None
        self._began_cycle = False

    @staticmethod
    def _check_stale(policy: str) -> str:
        if policy not in STALE_POLICIES:
            raise ValueError(f"stale must be one of {STALE_POLICIES}, got {policy!r}")
        return policy

    # ------------------------------------------------------------------ shared instances
    @classmethod
    def get(cls, name: str, **kwargs) -> "StepToggle":
        """The shared toggle registered under *name*, created on first call.

        Hotkey macros are re-entered as free functions with no instance to hang
        state on, so the identifier is the state. ``kwargs`` are passed to the
        constructor on creation and ignored afterwards (an existing toggle keeps
        its live state; per-press overrides belong on :meth:`advance`).
        """
        toggle = cls._instances.get(name)
        if toggle is None:
            toggle = cls._instances[name] = cls(name=name, **kwargs)
        return toggle

    @classmethod
    def clear(cls, name: Optional[str] = None) -> None:
        """Drop the shared toggle *name* (or every one when ``None``)."""
        if name is None:
            cls._instances.clear()
        else:
            cls._instances.pop(name, None)

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> int:
        """The current step: ``0`` at home, else ``1..steps``."""
        return self._state

    @property
    def at_home(self) -> bool:
        """True while the toggle sits at its home (un-stepped) state."""
        return self._state == 0

    @property
    def began_cycle(self) -> bool:
        """True when the last :meth:`advance` *started* a cycle — the moment to
        capture whatever :attr:`payload` must restore on the way home.

        A cycle begins from home, and also on any stale press that lands on
        step 1 — a retarget after a pause, or every stale press under the
        ``"restart"`` policy: after a pause, acting again is a new session, so
        its home is the view you are leaving now — not the one from the cycle
        you abandoned. Retargeting *within* the timeout keeps the original
        home, so a quick re-aim still unwinds to where you began.
        """
        return self._began_cycle

    def reset(self) -> None:
        """Return to home without acting — state, timing, context and payload."""
        self._state = 0
        self._last = None
        self._context = None
        self._began_cycle = False
        self.payload = None

    def advance(
        self,
        steps: Optional[int] = None,
        context: Any = None,
        timeout: Optional[float] = None,
        stale: Optional[str] = None,
    ) -> int:
        """Register a press and return the new state.

        Parameters:
            steps (int): Override the step count for this press (a caller whose
                step count is a per-call argument). Clamped to ``>= 1``.
            context (any): What the cycle is *about* (e.g. a selection
                signature). When it differs from the previous press the cycle
                restarts at step 1 rather than stepping deeper or going home —
                acting on something new is never an "undo". ``None`` opts out.
            timeout (float): Override :attr:`timeout` for this press.
            stale (str): Override :attr:`stale` for this press.

        Returns:
            int: ``0`` (home) or the step reached, ``1..steps``.
        """
        steps = self.steps if steps is None else max(1, int(steps))
        timeout = self.timeout if timeout is None else timeout
        stale_policy = self.stale if stale is None else self._check_stale(stale)

        now = self._clock()
        lapsed = self._last is None or (
            timeout is not None and (now - self._last) > timeout
        )
        self._last = now

        was_home = self._state == 0
        retargeted = context is not None and context != self._context
        if retargeted:
            self._context = context
            self._state = 1
        elif lapsed:  # a paused cycle never steps deeper: it collapses or restarts
            self._state = 0 if (self._state and stale_policy == "home") else 1
        elif self._state >= steps:
            self._state = 0
        else:
            self._state += 1

        # A cycle begins from home, or on any stale press that lands on step 1
        # (a stale retarget, or a "restart" press) — see began_cycle.
        self._began_cycle = self._state == 1 and (was_home or lapsed)
        return self._state

    # ------------------------------------------------------------------ step magnitudes
    @staticmethod
    def scales(steps: int, spread: float = 0.0, gain: float = 1.45) -> List[float]:
        """Multiplier ramp for an ``N``-step toggle — one factor per step.

        The first step is the caller's unscaled ideal (``1.0``) and each further
        step is *gain* times stronger than the last, so the first press always
        lands exactly where the caller wants it and the extra presses go
        *beyond* it. ``spread > 0`` opts into pulling the whole ramp back as
        ``steps`` grows (a longer cycle starting gentler than the ideal) — off
        by default, because a first press that undershoots reads as the key not
        doing its job.

        Parameters:
            steps (int): Number of steps in the cycle.
            spread (float): How much each extra step softens the starting factor
                (``0.0``: the first step is always exactly the ideal).
            gain (float): Per-step multiplier.

        Returns:
            list: ``steps`` floats, ascending. e.g. ``scales(2) ->
            [1.0, 1.45]``, ``scales(3) -> [1.0, 1.45, 2.1025]``.
        """
        n = max(1, int(steps))
        start = 1.0 / (1.0 + spread * (n - 1))
        return [start * (gain**i) for i in range(n)]
