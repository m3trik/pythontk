# !/usr/bin/python
# coding=utf-8
"""Ordered, gated replay of the steps a hand-off manifest asks for.

A conversion payload (an FBX or USD intermediate) lands in the receiving app and
the :class:`~pythontk.HandoffManifest` beside it says what else has to be rebuilt
-- materials, instances, lights, the clock, the shots, the rig.  Each consumer
used to sequence those appliers by hand, so the *order*, the *gate* ("only when
the producer sent that section"), the *error policy* ("a fidelity step logs, the
instance replay raises") and the *progress/cancel* protocol were re-decided per
call site and drifted between them.

A plan is that sequencing made declarative.  The consumer names its steps once,
in order; this class decides which ones survive their gate, runs them, reports
progress, honours a stop request and applies the per-step error policy::

    plan = manifest.plan(on_error=self._log_failure)
    plan.add(None, "Importing the FBX", import_file)
    plan.add(manifest.MATERIALS, "Rebuilding materials", rebuild_materials,
             best_effort=True)
    plan.add(manifest.INSTANCES, "Rebuilding instances", replay_instances)
    plan.add(manifest.SHOTS, "Rebuilding shots", rebuild_shots,
             when=shots, best_effort=True)
    plan.run(progress=progress, done_label="Imported")

Adding a section to the hand-off is then one ``add`` line at the right position,
rather than an edit to a hand-written chain in every consumer.

**Error policy is opt-in, not default.**  A step raises unless it is declared
``best_effort``, because the failure that must never be silent -- a guaranteed
instance replay that quietly leaves the scene flattened -- is exactly the one an
author would forget to mark.  A loud cosmetic failure is recoverable; a silent
structural one is not.

**A stop request is not an error.**  :class:`~pythontk.OperationCancelled`
derives from ``BaseException``, so it passes straight through the ``except
Exception`` that implements ``best_effort`` and can never be swallowed by a
fidelity step's error policy.
"""

from __future__ import annotations

from typing import Any, Callable, List, Mapping, NamedTuple, Optional

from pythontk.core_utils.cancel_scope import OperationCancelled

__all__ = ["ManifestPlan"]

#: ``apply()`` -- a step's work, already bound to its arguments by the consumer.
Apply = Callable[[], Any]
#: ``progress(done, total, text) -> bool`` -- ``False`` stops the run.
Progress = Callable[[int, int, str], Any]
#: ``on_error(label, exception)`` -- a ``best_effort`` step's failure.
OnError = Callable[[str, BaseException], Any]


class _Step(NamedTuple):
    """One admitted step: what to run, what to call it, and what a failure costs.

    It keeps no section name: the section decided whether the step is here at
    all, and nothing downstream asks again.
    """

    label: str
    apply: Apply
    best_effort: bool


class _ManifestPlanInternal(object):
    """Internal helpers for :class:`ManifestPlan`."""

    #: Default lead of the :class:`~pythontk.OperationCancelled` message.
    _CANCEL_PREFIX = "Stopped before"

    _cancel_prefix: str

    @staticmethod
    def _carries(manifest: Optional[Mapping], section: Optional[str]) -> bool:
        """Whether *manifest* carries *section* with something worth replaying.

        ``None`` names a step that is not section-driven (importing the file,
        reducing keys) and always passes.  Otherwise the test is truthiness, not
        presence: every producer writes "absent means nothing to say", and a
        section that arrived empty asks for the same no-op as one that never
        arrived.  A step needing the subtler distinction passes it as *when*.

        The same rule as :meth:`~pythontk.HandoffManifest.carries`, spelled again
        here because a plan gates on any ``Mapping`` and must not import the
        manifest type; change one and change the other.
        """
        if section is None:
            return True
        if manifest is None:
            return False
        try:
            return bool(manifest.get(section))
        except AttributeError:  # not a mapping -- a malformed sidecar
            return False

    def _tick(
        self, progress: Optional[Progress], done: int, total: int, text: str
    ) -> None:
        """Report one step boundary; raise when the caller asks to stop.

        Raises:
            pythontk.OperationCancelled: *progress* returned ``False``.
        """
        if progress is not None and progress(done, total, text) is False:
            raise OperationCancelled(f"{self._cancel_prefix}: {text}")


class ManifestPlan(_ManifestPlanInternal):
    """An ordered list of gated manifest steps, run with one progress protocol.

    Built by :meth:`~pythontk.HandoffManifest.plan` (or directly, against any
    mapping).  :meth:`add` admits a step only when its gates pass, so the plan's
    length is the number of steps that will really run -- which is what the
    progress bar should count.

    Parameters:
        manifest: The manifest whose sections gate the steps.  ``None`` admits
            only steps that are not section-driven.
        on_error: ``on_error(label, exception)``, called when a ``best_effort``
            step fails.  Without it a failure propagates whatever the step's
            policy says -- a plan never swallows an exception it cannot report.
        cancel_prefix: Lead of the :class:`~pythontk.OperationCancelled`
            message, so a consumer keeps its own wording.
    """

    def __init__(
        self,
        manifest: Optional[Mapping] = None,
        *,
        on_error: Optional[OnError] = None,
        cancel_prefix: Optional[str] = None,
    ) -> None:
        self._manifest = manifest
        self._on_error = on_error
        self._steps: List[_Step] = []
        self._cancel_prefix = str(cancel_prefix or self._CANCEL_PREFIX)

    def __len__(self) -> int:
        """The number of admitted steps -- what :meth:`run` will execute."""
        return len(self._steps)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({[s.label for s in self._steps]!r})"

    @property
    def labels(self) -> List[str]:
        """The admitted steps' labels, in run order (handy in tests and logs)."""
        return [step.label for step in self._steps]

    def add(
        self,
        section: Optional[str],
        label: str,
        apply: Apply,
        *,
        when: bool = True,
        best_effort: bool = False,
    ) -> "ManifestPlan":
        """Admit a step when its gates pass; return the plan, so calls chain.

        Parameters:
            section: The manifest section this step replays, or ``None`` for a
                step that always runs (importing the payload, reducing keys).
                The step is dropped when the manifest does not carry it.
            label: Progress text, and the name a failure is reported under.
            apply: ``apply()`` -- the work, already bound to its arguments.
            when: An extra gate the caller computes (a user option, the carrier,
                a version rule).  ``False`` drops the step whatever the manifest
                carries.
            best_effort: A failure is reported to *on_error* and the run
                continues.  The default lets it propagate: a step whose failure
                would leave the scene structurally wrong must never be silent.

        Returns:
            This plan.
        """
        if when and self._carries(self._manifest, section):
            self._steps.append(_Step(str(label), apply, bool(best_effort)))
        return self

    def run(
        self,
        *,
        progress: Optional[Progress] = None,
        done_label: Optional[str] = None,
    ) -> List[Any]:
        """Run every admitted step in order; return their results, in order.

        *progress* is called before each step with ``(done, total, text)`` and,
        when *done_label* is given, once more at the end -- the protocol both
        bridge consumers already speak.  Returning ``False`` from it stops the
        run *between* steps: what has already been applied stays.

        The final report is informational and cannot stop anything: every step
        has already run, so turning a stop request there into an exception would
        fail a run that actually succeeded.  Only the per-step reports cancel.

        Parameters:
            progress: ``progress(done, total, text) -> bool``.
            done_label: Text of the final report; omitted, no final report is made.

        Returns:
            One entry per step, in run order (``None`` where a ``best_effort``
            step failed).

        Raises:
            pythontk.OperationCancelled: *progress* returned ``False`` before a step.
            Exception: Whatever a step raised, unless it is ``best_effort`` and
                an *on_error* was given.
        """
        results: List[Any] = []
        total = len(self._steps)
        for done, step in enumerate(self._steps):
            self._tick(progress, done, total, step.label)
            try:
                results.append(step.apply())
            # OperationCancelled derives from BaseException and so never lands
            # here: a stop request outranks any step's error policy.
            except Exception as e:  # noqa: BLE001 -- policy is the step's
                if not step.best_effort or self._on_error is None:
                    raise
                self._on_error(step.label, e)
                results.append(None)
        if done_label is not None and progress is not None:
            # Reported directly, NOT through _tick: there is nothing left to
            # stop, and a caller that cancels on this last call would otherwise
            # get an exception out of a run that completed every step.
            progress(total, total, done_label)
        return results
