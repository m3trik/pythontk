# !/usr/bin/python
# coding=utf-8
"""Helpers for hot-reloading packages and their submodules."""
from __future__ import annotations

import importlib
import pkgutil
import sys
from types import ModuleType
from typing import Callable, Iterable, List, Optional, Sequence, Union

__all__ = ["ModuleReloader", "reload_package"]

ModuleRef = Union[str, ModuleType]


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
    ) -> List[ModuleType]:
        """Reload a package and return the modules processed.

        Args:
            verbose: Verbosity level:
                - 0 or False: Silent (no output)
                - 1 or True: Basic (module names only)
                - 2: Detailed (include skip reasons and errors)
            max_passes: Number of reload passes to perform (default: 2).
                       Multiple passes help resolve circular and sibling dependencies
                       where module A imports B, but A is reloaded before B.
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

        target_package = self._resolve_module(package, import_missing=import_missing)

        module_order: List[ModuleType] = []

        if dependencies_first:
            module_order.extend(
                self._resolve_module(dep, import_missing=import_missing)
                for dep in dependencies_first
            )

        if include_submodules:
            module_order.extend(
                self._iter_package_modules(
                    target_package,
                    import_missing=import_missing,
                    predicate=predicate,
                    exclude_modules=exclude_modules,
                )
            )

        module_order.append(target_package)

        if dependencies_last:
            module_order.extend(
                self._resolve_module(dep, import_missing=import_missing)
                for dep in dependencies_last
            )

        ordered_modules = self._unique_modules(module_order)
        ordered_modules.sort(key=lambda mod: mod.__name__.count("."), reverse=True)

        reloaded: List[ModuleType] = []

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
                            if verbose_level >= 2:
                                print(
                                    f"Skipping reload for {canonical_name}: import failed ({exc})."
                                )
                            continue
                    else:
                        if verbose_level >= 2:
                            print(
                                f"Skipping reload for {canonical_name}: not present in sys.modules."
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
                except ImportError as exc:
                    if verbose_level >= 2:
                        print(f"Skipping reload for {canonical_name}: {exc}")
                    continue

                if pass_num == 1:
                    reloaded.append(refreshed)

                # Only run hooks on the last pass
                if pass_num == max_passes and after_reload:
                    after_reload(refreshed)

        return reloaded

    # ------------------------------------------------------------------
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
    ) -> Iterable[ModuleType]:
        if not hasattr(package, "__path__"):
            return []

        import fnmatch

        for module_info in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
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
                module = importlib.import_module(module_info.name)
            if module is None:
                continue
            if predicate and not predicate(module):
                continue
            yield module

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
) -> List[ModuleType]:
    """Convenience wrapper around :class:`ModuleReloader`."""

    reloader = ModuleReloader()
    return reloader.reload(package, **kwargs)
