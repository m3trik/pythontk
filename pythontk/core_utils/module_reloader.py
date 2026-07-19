# !/usr/bin/python
# coding=utf-8
"""Helpers for hot-reloading packages and their submodules."""
from __future__ import annotations

import fnmatch
import importlib
import pkgutil
import sys
import types
from types import ModuleType
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

__all__ = ["ModuleReloader", "ReloadReport", "reload_package"]

ModuleRef = Union[str, ModuleType]


class ReloadReport(List[ModuleType]):
    """List of successfully reloaded modules, with failure/skip details attached.

    Behaves exactly like the plain ``List[ModuleType]`` previously returned
    (``len()``, iteration, indexing), plus:

    Attributes:
        failed: ``(module_name, exception)`` pairs for modules whose reload
            raised. The exception is from the last pass that attempted them.
        skipped: Names of modules that were skipped (not importable, not
            present in ``sys.modules`` with ``import_missing=False``, or an
            unresolvable dependency reference).
    """

    def __init__(self, modules: Iterable[ModuleType] = ()) -> None:
        super().__init__(modules)
        self.failed: List[Tuple[str, BaseException]] = []
        self.skipped: List[str] = []

    @property
    def ok(self) -> bool:
        """True when no reload attempt raised."""
        return not self.failed


class ModuleReloader:
    """Flexible controller for reloading packages and related modules."""

    def __init__(
        self,
        *,
        include_submodules: bool = True,
        dependencies_first: Optional[Iterable[ModuleRef]] = None,
        dependencies_last: Optional[Iterable[ModuleRef]] = None,
        predicate: Optional[Callable[[ModuleType], bool]] = None,
        before_reload: Optional[Callable[[ModuleType], None]] = None,
        after_reload: Optional[Callable[[ModuleType], None]] = None,
        import_missing: bool = True,
        verbose: Union[bool, int] = False,
        max_passes: int = 2,
        exclude_modules: Optional[Iterable[str]] = None,
    ) -> None:
        self.include_submodules = include_submodules
        self.dependencies_first = list(dependencies_first or [])
        self.dependencies_last = list(dependencies_last or [])
        self.predicate = predicate
        self.before_reload = before_reload
        self.after_reload = after_reload
        self.import_missing = import_missing
        # Convert bool to int: False->0, True->1
        self.verbose = int(verbose) if isinstance(verbose, bool) else verbose
        self.max_passes = max_passes
        self.exclude_modules = list(exclude_modules or [])

    # ------------------------------------------------------------------
    def reload(
        self,
        package: ModuleRef,
        *,
        include_submodules: Optional[bool] = None,
        dependencies_first: Optional[Iterable[ModuleRef]] = None,
        dependencies_last: Optional[Iterable[ModuleRef]] = None,
        predicate: Optional[Callable[[ModuleType], bool]] = None,
        before_reload: Optional[Callable[[ModuleType], None]] = None,
        after_reload: Optional[Callable[[ModuleType], None]] = None,
        import_missing: Optional[bool] = None,
        verbose: Optional[Union[bool, int]] = None,
        max_passes: Optional[int] = None,
        exclude_modules: Optional[Iterable[str]] = None,
    ) -> ReloadReport:
        """Reload a package and return the modules processed.

        Modules are ordered by a topological sort of their observed import
        relationships (falling back to deepest-first for cycles), grouped as:
        ``dependencies_first`` → target package (+ submodules) →
        ``dependencies_last``. Package references in the dependency lists are
        expanded to their submodules when ``include_submodules`` is True.

        A module whose reload raises does not abort the run: the error is
        printed, remaining modules still reload, and the failure is recorded
        on the returned :class:`ReloadReport` (a ``List[ModuleType]``).

        Args:
            verbose: Verbosity level:
                - 0 or False: Silent (reload failures still print)
                - 1 or True: Basic (module names only)
                - 2: Detailed (include skip reasons and errors)
            max_passes: Number of reload passes to perform (default: 2).
                       Multiple passes help resolve circular dependencies the
                       topological sort cannot fully order.
            exclude_modules: List of glob patterns to exclude from reload (e.g. ["*_ui", "test_*"]).
        """
        include_submodules = (
            self.include_submodules
            if include_submodules is None
            else include_submodules
        )
        dependencies_first = self._merge_dependencies(
            self.dependencies_first, dependencies_first
        )
        dependencies_last = self._merge_dependencies(
            self.dependencies_last, dependencies_last
        )
        predicate = predicate or self.predicate
        before_reload = before_reload or self.before_reload
        after_reload = after_reload or self.after_reload
        import_missing = (
            self.import_missing if import_missing is None else import_missing
        )
        verbose_level = self.verbose if verbose is None else verbose
        # Convert bool to int: False->0, True->1
        verbose_level = (
            int(verbose_level) if isinstance(verbose_level, bool) else verbose_level
        )
        max_passes = self.max_passes if max_passes is None else max_passes
        exclude_modules = self._merge_dependencies(
            self.exclude_modules, exclude_modules
        )

        report = ReloadReport()

        target_package = self._resolve_module(package, import_missing=import_missing)

        collect_kwargs = dict(
            include_submodules=include_submodules,
            import_missing=import_missing,
            predicate=predicate,
            exclude_modules=exclude_modules,
            report=report,
            verbose_level=verbose_level,
        )

        first_block = self._collect_references(dependencies_first, **collect_kwargs)
        middle_block = self._collect_references([target_package], **collect_kwargs)
        last_block = self._collect_references(dependencies_last, **collect_kwargs)

        # Dedup across blocks without disturbing block boundaries: a module
        # listed in dependencies_first keeps its early slot; one listed in
        # dependencies_last is deferred out of the middle block.
        first_names = {self._canonical_module_name(m) for m in first_block}
        last_names = {self._canonical_module_name(m) for m in last_block}
        dependency_names = first_names | last_names
        middle_block = [
            m
            for m in middle_block
            if self._canonical_module_name(m) not in dependency_names
        ]
        last_block = [
            m
            for m in last_block
            if self._canonical_module_name(m) not in first_names
        ]

        ordered_modules: List[ModuleType] = []
        for block in (first_block, middle_block, last_block):
            ordered_modules.extend(self._sort_dependencies_first(block))

        failures: Dict[str, BaseException] = {}
        printed_failures: set = set()
        appended: set = set()

        for pass_num in range(1, max_passes + 1):
            if verbose_level >= 1 and max_passes > 1:
                print(f"Reload pass {pass_num}/{max_passes}...")

            for module in ordered_modules:
                canonical_name = self._canonical_module_name(module)
                target_module = module

                if canonical_name not in sys.modules:
                    if import_missing:
                        try:
                            target_module = importlib.import_module(canonical_name)
                        except ImportError as exc:
                            self._record_skip(
                                report,
                                canonical_name,
                                f"import failed ({exc})",
                                verbose_level,
                            )
                            continue
                    else:
                        self._record_skip(
                            report,
                            canonical_name,
                            "not present in sys.modules",
                            verbose_level,
                        )
                        continue
                else:
                    target_module = sys.modules[canonical_name]

                # Only run hooks on the first pass to avoid double-execution side effects
                if pass_num == 1 and before_reload:
                    before_reload(target_module)

                if verbose_level >= 1:
                    print(f"Reloading {canonical_name}")

                try:
                    refreshed = importlib.reload(target_module)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as exc:  # noqa: BLE001 — one broken module must not abort the run
                    failures[canonical_name] = exc
                    message = f"[ModuleReloader] FAILED {canonical_name}: {type(exc).__name__}: {exc}"
                    if message not in printed_failures:  # same error can recur per pass
                        printed_failures.add(message)
                        print(message)
                    continue

                failures.pop(canonical_name, None)

                if canonical_name not in appended:
                    appended.add(canonical_name)
                    report.append(refreshed)

                # Only run hooks on the last pass
                if pass_num == max_passes and after_reload:
                    after_reload(refreshed)

        report.failed = list(failures.items())
        return report

    # ------------------------------------------------------------------
    def _collect_references(
        self,
        references: Iterable[ModuleRef],
        *,
        include_submodules: bool,
        import_missing: bool,
        predicate: Optional[Callable[[ModuleType], bool]],
        exclude_modules: List[str],
        report: ReloadReport,
        verbose_level: int,
    ) -> List[ModuleType]:
        """Resolve refs to modules, expanding packages to their submodules.

        Submodules come before their package so dependents reload after their
        parts. Unresolvable references are skipped (recorded on the report)
        rather than aborting the run.
        """
        collected: List[ModuleType] = []
        for reference in references:
            try:
                module = self._resolve_module(reference, import_missing=import_missing)
            except ImportError as exc:
                name = reference if isinstance(reference, str) else str(reference)
                self._record_skip(report, name, str(exc), verbose_level)
                continue

            if include_submodules:
                collected.extend(
                    self._iter_package_modules(
                        module,
                        import_missing=import_missing,
                        predicate=predicate,
                        exclude_modules=exclude_modules,
                        report=report,
                        verbose_level=verbose_level,
                    )
                )
            collected.append(module)

        return self._unique_modules(collected)

    @staticmethod
    def _record_skip(
        report: ReloadReport, name: str, reason: str, verbose_level: int
    ) -> None:
        if name not in report.skipped:  # a skip can recur across passes
            report.skipped.append(name)
        if verbose_level >= 2:
            print(f"Skipping reload for {name}: {reason}.")

    @staticmethod
    def _merge_dependencies(
        defaults: Sequence[ModuleRef], overrides: Optional[Iterable[ModuleRef]]
    ) -> List[ModuleRef]:
        if overrides is None:
            return list(defaults)
        return list(overrides)

    @staticmethod
    def _resolve_module(reference: ModuleRef, *, import_missing: bool) -> ModuleType:
        if isinstance(reference, ModuleType):
            return reference
        if not isinstance(reference, str):
            raise TypeError("Module reference must be a module object or dotted string")
        module = sys.modules.get(reference)
        if module is None:
            if not import_missing:
                raise ImportError(f"Module '{reference}' is not imported")
            module = importlib.import_module(reference)
        return module

    @staticmethod
    def _iter_package_modules(
        package: ModuleType,
        *,
        import_missing: bool,
        predicate: Optional[Callable[[ModuleType], bool]],
        exclude_modules: Optional[List[str]] = None,
        report: Optional[ReloadReport] = None,
        verbose_level: int = 0,
    ) -> Iterable[ModuleType]:
        if not hasattr(package, "__path__"):
            return

        for module_info in pkgutil.walk_packages(
            package.__path__, package.__name__ + ".", onerror=lambda name: None
        ):
            # Check exclusions before importing
            if exclude_modules:
                if any(
                    fnmatch.fnmatch(module_info.name, pattern)
                    for pattern in exclude_modules
                ):
                    continue

            module = sys.modules.get(module_info.name)
            if module is None and import_missing:
                try:
                    module = importlib.import_module(module_info.name)
                except Exception as exc:  # e.g. a DCC-only module in another host
                    if report is not None:
                        ModuleReloader._record_skip(
                            report,
                            module_info.name,
                            f"import failed ({exc})",
                            verbose_level,
                        )
                    continue
            if module is None:
                continue
            if predicate and not predicate(module):
                continue
            yield module

    @staticmethod
    def _sort_dependencies_first(modules: Sequence[ModuleType]) -> List[ModuleType]:
        """Topologically sort so each module's dependencies reload before it.

        Edges come from what a module's namespace actually references:
        submodules bound by ``from . import x`` and the ``__module__`` of
        imported classes/functions. Cycles fall back to deepest-first, then
        input order — the old heuristic — and are what ``max_passes`` exists
        to mop up.
        """
        by_name: Dict[str, ModuleType] = {}
        for module in modules:
            by_name[ModuleReloader._canonical_module_name(module)] = module

        order_index = {name: i for i, name in enumerate(by_name)}
        remaining: Dict[str, set] = {}
        for name, module in by_name.items():
            deps = set()
            for value in list(vars(module).values()):
                dep = ModuleReloader._owning_module_name(value)
                if dep and dep != name and dep in by_name:
                    deps.add(dep)
            remaining[name] = deps

        ordered: List[ModuleType] = []
        while remaining:
            ready = [name for name, deps in remaining.items() if not deps]
            if not ready:  # cycle — break it with the old deepest-first heuristic
                ready = [
                    max(
                        remaining,
                        key=lambda n: (n.count("."), -order_index[n]),
                    )
                ]
            ready.sort(key=lambda n: (-n.count("."), order_index[n]))
            for name in ready:
                ordered.append(by_name[name])
                del remaining[name]
            for deps in remaining.values():
                deps.difference_update(ready)
        return ordered

    @staticmethod
    def _owning_module_name(value: object) -> Optional[str]:
        """Name of the module a namespace entry came from, if determinable.

        Restricted to modules, classes and functions so arbitrary objects'
        ``__getattr__`` is never triggered.
        """
        if isinstance(value, ModuleType):
            return ModuleReloader._canonical_module_name(value)
        if isinstance(value, (type, types.FunctionType)):
            try:
                owner = value.__module__
            except Exception:
                return None
            return owner if isinstance(owner, str) else None
        return None

    @staticmethod
    def _unique_modules(modules: Sequence[ModuleType]) -> List[ModuleType]:
        seen = set()
        unique: List[ModuleType] = []
        for module in modules:
            if module is None:
                continue
            name = ModuleReloader._canonical_module_name(module) or module.__name__
            if name in seen:
                continue
            seen.add(name)
            unique.append(module)
        return unique

    @staticmethod
    def _canonical_module_name(module: ModuleType) -> str:
        if module is None:
            return ""

        spec = getattr(module, "__spec__", None)
        if spec and getattr(spec, "name", None):
            name = spec.name
        else:
            name = module.__name__

        if name == "__init__" and getattr(module, "__package__", None):
            return module.__package__

        if name.endswith(".__init__"):
            return name[: -len(".__init__")]

        return name


def reload_package(
    package: ModuleRef,
    **kwargs,
) -> ReloadReport:
    """Convenience wrapper around :class:`ModuleReloader`."""

    reloader = ModuleReloader()
    return reloader.reload(package, **kwargs)
