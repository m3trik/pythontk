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
        verbose: bool = False,
    ) -> None:
        self.include_submodules = include_submodules
        self.dependencies_first = list(dependencies_first or [])
        self.dependencies_last = list(dependencies_last or [])
        self.predicate = predicate
        self.before_reload = before_reload
        self.after_reload = after_reload
        self.import_missing = import_missing
        self.verbose = verbose

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
        verbose: Optional[bool] = None,
    ) -> List[ModuleType]:
        """Reload a package and return the modules processed."""

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
        verbose = self.verbose if verbose is None else verbose

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
                    target_package, import_missing=import_missing, predicate=predicate
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
        for module in ordered_modules:
            canonical_name = self._canonical_module_name(module)
            target_module = module

            if canonical_name not in sys.modules:
                if import_missing:
                    try:
                        target_module = importlib.import_module(canonical_name)
                    except ImportError as exc:
                        if verbose:
                            print(
                                f"Skipping reload for {canonical_name}: import failed ({exc})."
                            )
                        continue
                else:
                    if verbose:
                        print(
                            f"Skipping reload for {canonical_name}: not present in sys.modules."
                        )
                    continue
            else:
                target_module = sys.modules[canonical_name]

            if before_reload:
                before_reload(target_module)

            if verbose:
                print(f"Reloading {canonical_name}")

            try:
                refreshed = importlib.reload(target_module)
            except ImportError as exc:
                if verbose:
                    print(f"Skipping reload for {canonical_name}: {exc}")
                continue

            reloaded.append(refreshed)

            if after_reload:
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
    ) -> Iterable[ModuleType]:
        if not hasattr(package, "__path__"):
            return []

        for module_info in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
        ):
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
