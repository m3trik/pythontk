# pythontk — API Changes

_Diff vs the last release (origin/main @ 0dc3d78). Generated 2026-08-17._

## Added (2)

- `core_utils/_core_utils.py::CoreUtils.teardown_guard(logger=None, what: str = 'state')`
- `core_utils/task_factory.py::TaskFactory.stage_deferred_context(self, key: str, cm) -> bool`

## Signature changed (2)

- `core_utils/step_toggle.py::StepToggle.advance`
  - was: `(self, steps: Optional[int] = None, context: Any = None, timeout: Optional[float] = None) -> int`
  - now: `(self, steps: Optional[int] = None, context: Any = None, timeout: Optional[float] = None, stale: Optional[str] = None) -> int`
- `core_utils/step_toggle.py::StepToggle.scales`
  - was: `(steps: int, spread: float = 0.15, gain: float = 1.45) -> List[float]`
  - now: `(steps: int, spread: float = 0.0, gain: float = 1.45) -> List[float]`
