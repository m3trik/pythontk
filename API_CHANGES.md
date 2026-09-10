# pythontk — API Changes

_Diff vs the last release (origin/main @ 42710a1)._

## Added (20)

- `geo_utils/uv_budget.py::BudgetItem(class)`
- `geo_utils/uv_budget.py::BudgetItem.demand(self, tpu: float, pad: float) -> float`
- `geo_utils/uv_budget.py::BudgetPage(class)`
- `geo_utils/uv_budget.py::BudgetPage.fill(self) -> float`
- `geo_utils/uv_budget.py::BudgetPlan(class)`
- `geo_utils/uv_budget.py::BudgetPlan.density_ratio(self) -> Optional[float]`
- `geo_utils/uv_budget.py::BudgetPlan.rows(self) -> List[BudgetRow]`
- `geo_utils/uv_budget.py::BudgetRow(class)`
- `geo_utils/uv_budget.py::BudgetRow.assignment(self) -> Dict[str, int]`
- `geo_utils/uv_budget.py::BudgetRow.texels(self) -> int`
- `geo_utils/uv_budget.py::BudgetRow.underfilled(self, threshold: float = 0.5) -> List[int]`
- `geo_utils/uv_budget.py::BudgetRow.utilization(self) -> float`
- `geo_utils/uv_budget.py::BudgetRow.worst_fill(self) -> float`
- `geo_utils/uv_budget.py::UvBudget(class)`
- `geo_utils/uv_budget.py::UvBudget.density_at_pages(cls, items: Sequence[BudgetItem], pages: int, map_size: int, *, factor: int = 256, mip_levels: int = 0, fill: Optional[float] = None, level: bool = True) -> BudgetRow`
- `geo_utils/uv_budget.py::UvBudget.first_fit_decreasing(sizes: Sequence[Tuple[str, float]], capacity: float) -> Optional[List[List[str]]]`
- `geo_utils/uv_budget.py::UvBudget.padding_for(map_size: int, factor: int = 256, mip_levels: int = 0) -> Tuple[float, bool]`
- `geo_utils/uv_budget.py::UvBudget.pages_at_density(cls, items: Sequence[BudgetItem], density: float, map_size: int, *, factor: int = 256, mip_levels: int = 0, fill: Optional[float] = None, level: bool = False) -> BudgetRow`
- `geo_utils/uv_budget.py::UvBudget.partition_lpt(sizes: Sequence[Tuple[str, float]], pages: int) -> List[List[str]]`
- `geo_utils/uv_budget.py::UvBudget.plan(cls, items: Sequence[BudgetItem], *, map_size: int = 4096, density: Optional[float] = None, pages: Optional[int] = None, factor: int = 256, mip_levels: int = 0, fill: Optional[float] = None, level: bool = False, alternates: bool = True, map_sizes: Optional[Iterable[int]] = None) -> BudgetPlan`

## Signature changed (1)

- `core_utils/engines/shots/shot_plan.py::ShotPlanner.plan_respace`
  - was: `(store: ShotStore, gap: float, start_frame: float) -> MovePlan`
  - now: `(store: ShotStore, gap: float, start_frame: float, respect_locks: bool = True) -> MovePlan`
