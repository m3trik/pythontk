# pythontk — API Changes

_Diff vs prior baseline. Generated 2026-06-07._

## Signature changed (1)

- `math_utils/_math_utils.py::MathUtils.catenary_sag`
  - was: `(cls, t: float, tension: float, round_amount: float = 0.0) -> float`
  - now: `(cls, t: float, tension: float, round_amount: float = 0.0, gather: float = 0.0) -> float`
