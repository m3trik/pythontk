# !/usr/bin/python
# coding=utf-8
"""Reusable module attribute resolver for package-style imports."""
import importlib
import inspect
import os
import pkgutil
import sys
import ast
from types import ModuleType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

__all__ = ["ModuleAttributeResolver", "PackageResolverHandle", "bootstrap_package"]

IncludeMapping = Mapping[str, Union[Sequence[str], str]]


class ModuleAttributeResolver:
    """Discover and resolve attributes exposed from package submodules lazily."""

    def __init__(
        self,
        module: Union[ModuleType, str],
        *,
        include: Optional[Mapping[str, Union[Sequence[str], str]]] = None,
        fallbacks: Optional[Mapping[str, str]] = None,
        module_to_parent: Optional[Mapping[str, str]] = None,
        on_import_error: Optional[Callable[[str, Exception], None]] = None,
        method_predicate: Optional[Callable[[str], bool]] = None,
        lazy_import: Optional[bool] = None,
    ) -> None:
        if isinstance(module, str):
            module = sys.modules[module]
        if not isinstance(module, ModuleType):
            raise TypeError("module must be a module object or module name")
        if not hasattr(module, "__path__"):
            raise ValueError(
                "resolver requires a package module with a __path__ attribute"
            )

        self._module = module
        self.package_name = module.__name__
        self._package_path = module.__path__
        self._include_spec = include
        self._direct_include: Optional[Dict[str, Tuple[str, ...]]] = None
        self._absolute_include: Optional[Dict[str, Tuple[str, ...]]] = None
        self._set_include(include)

        self.fallbacks: Dict[str, str] = dict(fallbacks or {})
        self.module_to_parent: Dict[str, str] = dict(module_to_parent or {})
        self.on_import_error = on_import_error
        self.method_predicate = method_predicate or (
            lambda name: not name.startswith("_")
        )
        self.lazy_import = lazy_import

        self.imported_modules: Dict[str, ModuleType] = {}
        self.class_to_module: Dict[str, str] = {}
        self.method_to_module: Dict[str, Tuple[str, str]] = {}
        self.class_method_to_module: Dict[str, Tuple[str, str]] = {}
        self.submodules: set[str] = set()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def build(self) -> "ModuleAttributeResolver":
        """Populate resolver dictionaries based on current include spec."""
        self.class_to_module.clear()
        self.method_to_module.clear()
        self.class_method_to_module.clear()
        self.submodules.clear()

        for _, modname, _ in pkgutil.walk_packages(
            self._package_path, prefix=f"{self.package_name}."
        ):
            # Register direct submodules for lazy resolution
            if modname.startswith(f"{self.package_name}."):
                rel_name = modname[len(self.package_name) + 1 :]
                if "." not in rel_name:
                    self.submodules.add(rel_name)

            classes = self._classes_for_module(modname)
            if classes is None:
                continue

            # Determine if we should lazy load this module
            should_lazy = self.lazy_import
            if should_lazy is None:
                # Auto-detect: check if module is safe to lazy load
                should_lazy = self._is_safe_to_lazy_load(modname)

            if should_lazy and classes and "*" not in classes:
                for class_name in classes:
                    self.class_to_module[class_name] = modname
                continue

            if should_lazy and (not classes or "*" in classes):
                try:
                    scanned_classes, scanned_methods = self._scan_module_attributes(
                        modname
                    )
                    if scanned_classes or scanned_methods:
                        for class_name in scanned_classes:
                            self.class_to_module[class_name] = modname
                        for method_name, class_name in scanned_methods.items():
                            self.method_to_module[method_name] = (modname, class_name)
                            self.class_method_to_module[method_name] = (
                                modname,
                                class_name,
                            )
                        continue
                except Exception as exc:
                    # Fallback to eager import if scanning fails
                    self._handle_import_error(modname, exc)

            try:
                module = importlib.import_module(modname)
            except ImportError as exc:
                self._handle_import_error(modname, exc)
                continue

            if not classes or "*" in classes:
                self._register_all_classes(module)
            else:
                self._register_selected_classes(module, classes)

        return self

    def rebuild(
        self, include: Optional[Mapping[str, Union[Sequence[str], str]]] = None
    ) -> "ModuleAttributeResolver":
        """Reset include spec (optional) and rebuild dictionaries."""
        if include is not None:
            self._include_spec = include
        self._set_include(self._include_spec)
        return self.build()

    def resolve(self, name: str):
        """Resolve an attribute using the registered dictionaries."""
        if name in self.class_to_module:
            module = self._import(self.class_to_module[name])
            return getattr(module, name)

        if name in self.method_to_module:
            module_name, class_name = self.method_to_module[name]
            module = self._import(module_name)
            class_obj = getattr(module, class_name)
            return getattr(class_obj, name)

        if name in self.class_method_to_module:
            module_name, class_name = self.class_method_to_module[name]
            module = self._import(module_name)
            class_obj = getattr(module, class_name)
            return getattr(class_obj, name)

        if name in self.module_to_parent:
            parent_module = self._import(self.module_to_parent[name])
            return getattr(parent_module, name)

        if name in self.submodules:
            return self._import(f"{self.package_name}.{name}")

        if name in self.fallbacks:
            module = self._import(self.fallbacks[name])
            attr = getattr(module, name)
            self.class_to_module[name] = self.fallbacks[name]
            return attr

        # Check if the attribute is available in the module itself (e.g. static methods on the package)
        # Use __dict__ lookup to avoid recursion if self._module has a __getattr__ that calls us
        if name in self._module.__dict__:
            return self._module.__dict__[name]

        raise AttributeError(f"module {self.package_name} has no attribute '{name}'")

    def get_module(self, module_name: str) -> ModuleType:
        """Import a module managed by the resolver, using the local cache."""
        return self._import(module_name)

    def bind_to(
        self,
        module_globals: MutableMapping[str, object],
        *,
        install_getattr: bool = True,
        eager: bool = False,
        names: Optional[Iterable[str]] = None,
    ) -> None:
        """Bind resolver helpers into a module's globals dictionary."""

        if install_getattr:

            def _module_getattr(target_name: str):
                return self.resolve(target_name)

            module_globals["__getattr__"] = _module_getattr

        if eager:
            export_names = list(names or self.iter_registered_names())
            for export_name in export_names:
                module_globals[export_name] = self.resolve(export_name)

    def iter_registered_names(self) -> Iterable[str]:
        """Return an iterable of attribute names known to the resolver."""
        yielded = set()
        for container in (
            self.class_to_module,
            self.method_to_module,
            self.class_method_to_module,
            self.module_to_parent,
            self.fallbacks,
            self.submodules,
        ):
            for key in container:
                if key not in yielded:
                    yielded.add(key)
                    yield key

    def clear_module_cache(self) -> None:
        """Drop cached module imports managed by the resolver."""
        self.imported_modules.clear()

    def _scan_module_attributes(
        self, module_name: str
    ) -> Tuple[Sequence[str], Dict[str, str]]:
        """Statically scan a module for classes and their methods."""
        tree = self._parse_module(module_name)
        if not tree:
            return [], {}

        top_level = []
        methods = {}

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if self.method_predicate(node.name):
                    top_level.append(node.name)
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            if self.method_predicate(item.name):
                                methods[item.name] = node.name
            elif isinstance(node, ast.FunctionDef):
                if self.method_predicate(node.name):
                    top_level.append(node.name)
        return top_level, methods

    def _is_safe_to_lazy_load(self, module_name: str) -> bool:
        """Check if a module is safe to lazy load (no top-level side effects)."""
        tree = self._parse_module(module_name)
        if not tree:
            return False  # If we can't parse, assume unsafe

        # Allowed top-level nodes
        SAFE_NODES = (
            ast.Import,
            ast.ImportFrom,
            ast.ClassDef,
            ast.FunctionDef,
            ast.Assign,
            ast.AnnAssign,
            ast.If,  # Assume if/try are for flow control/imports
            ast.Try,
        )

        for node in tree.body:
            if not isinstance(node, SAFE_NODES):
                # Found something suspicious (like a top-level function call)
                return False
        return True

    def _parse_module(self, module_name: str) -> Optional[ast.AST]:
        """Helper to parse a module file into an AST."""
        try:
            loader = pkgutil.get_loader(module_name)
            if not loader or not hasattr(loader, "get_filename"):
                return None

            filename = loader.get_filename(module_name)
            if not filename or not os.path.exists(filename):
                return None

            with open(filename, "r", encoding="utf-8") as f:
                return ast.parse(f.read(), filename=filename)
        except (ImportError, SyntaxError, OSError):
            return None

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _set_include(
        self, include: Optional[Mapping[str, Union[Sequence[str], str]]]
    ) -> None:
        if include is None:
            self._direct_include = None
            self._absolute_include = None
            return

        direct: Dict[str, Tuple[str, ...]] = {}
        absolute: Dict[str, Tuple[str, ...]] = {}
        for key, value in include.items():
            normalized = self._normalize_include_value(value)
            if "." in key:
                absolute[f"{self.package_name}.{key}"] = normalized
            else:
                direct[key] = normalized

        self._direct_include = direct
        self._absolute_include = absolute

    def _classes_for_module(self, module_name: str) -> Optional[Tuple[str, ...]]:
        if self._direct_include is None and self._absolute_include is None:
            return ("*",)

        if self._absolute_include and module_name in self._absolute_include:
            return self._absolute_include[module_name]

        if self._direct_include:
            component = module_name.split(".")[-1]
            if component in self._direct_include:
                return self._direct_include[component]

        return None

    def _register_all_classes(self, module: ModuleType) -> None:
        for name, obj in module.__dict__.items():
            if not inspect.isclass(obj):
                continue
            if obj.__module__ != module.__name__:
                continue
            self._register_class(name, obj, module.__name__, register_methods=True)

    def _register_selected_classes(
        self, module: ModuleType, classes: Sequence[str]
    ) -> None:
        for class_name in classes:
            obj = getattr(module, class_name, None)
            if obj and inspect.isclass(obj) and obj.__module__ == module.__name__:
                self._register_class(
                    class_name, obj, module.__name__, register_methods=False
                )

    def _register_class(
        self,
        class_name: str,
        class_obj,
        module_name: str,
        register_methods: bool = True,
    ) -> None:
        self.class_to_module[class_name] = module_name

        if not register_methods:
            return

        for method_name, attribute in class_obj.__dict__.items():
            if not self.method_predicate(method_name):
                continue
            if isinstance(attribute, (staticmethod, classmethod)):
                member = attribute.__func__
            else:
                member = attribute
            if callable(member):
                self.method_to_module[method_name] = (module_name, class_name)
                self.class_method_to_module[method_name] = (module_name, class_name)

    def _handle_import_error(self, modname: str, exc: Exception) -> None:
        if self.on_import_error:
            self.on_import_error(modname, exc)
        else:
            print(f"Failed to import module {modname}: {exc}")

    def _import(self, module_name: str) -> ModuleType:
        if module_name not in self.imported_modules:
            self.imported_modules[module_name] = importlib.import_module(module_name)
        return self.imported_modules[module_name]

    @staticmethod
    def _normalize_include_value(
        value: Union[Sequence[str], str, None],
    ) -> Tuple[str, ...]:
        if value is None:
            return ("*",)
        if isinstance(value, str):
            return ("*",) if value == "*" else (value,)
        return tuple(value)


class PackageResolverHandle:
    """Facade that wires a :class:`ModuleAttributeResolver` into a package module."""

    def __init__(
        self,
        resolver: ModuleAttributeResolver,
        module_globals: MutableMapping[str, Any],
        *,
        default_include: Optional[IncludeMapping] = None,
        default_fallbacks: Optional[Mapping[str, str]] = None,
        default_module_to_parent: Optional[Mapping[str, str]] = None,
        allow_getattr: bool = True,
        default_eager: bool = False,
        custom_getattr: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.resolver = resolver
        self.module_globals = module_globals
        self.allow_getattr = allow_getattr
        self.default_eager = default_eager
        self.custom_getattr = custom_getattr

        self._default_include = (
            dict(default_include) if default_include is not None else None
        )
        self._default_fallbacks = dict(default_fallbacks or {})
        self._default_module_to_parent = dict(default_module_to_parent or {})

        self.include_spec: Optional[Dict[str, Union[Sequence[str], str]]] = (
            dict(self._default_include) if self._default_include is not None else None
        )
        self.fallbacks_spec: Dict[str, str] = dict(self._default_fallbacks)
        self.module_parent_spec: Dict[str, str] = dict(self._default_module_to_parent)

    # ------------------------------------------------------------------
    # installation helpers
    # ------------------------------------------------------------------
    def install(
        self,
        *,
        expose_maps: bool = True,
        install_helpers: bool = True,
        allow_getattr: Optional[bool] = None,
        eager: Optional[bool] = None,
        custom_getattr: Optional[Callable[[str], Any]] = None,
    ) -> None:
        """Publish resolver artifacts into the target module globals."""

        allow_getattr_flag = (
            self.allow_getattr if allow_getattr is None else allow_getattr
        )
        eager_flag = self.default_eager if eager is None else eager
        self.allow_getattr = allow_getattr_flag
        self.default_eager = eager_flag
        if custom_getattr is not None:
            self.custom_getattr = custom_getattr

        if expose_maps:
            self.module_globals["CLASS_TO_MODULE"] = self.resolver.class_to_module
            self.module_globals["METHOD_TO_MODULE"] = self.resolver.method_to_module
            self.module_globals["CLASS_METHOD_TO_MODULE"] = (
                self.resolver.class_method_to_module
            )
            self.module_globals["IMPORTED_MODULES"] = self.resolver.imported_modules
            self.module_globals["MODULE_TO_PARENT"] = self.resolver.module_to_parent
            self.module_globals["FALLBACKS"] = self.resolver.fallbacks

        if install_helpers:
            self.module_globals["configure_resolver"] = self.configure
            self.module_globals["build_dictionaries"] = self.build_dictionaries
            self.module_globals["import_module"] = self.import_module
            self.module_globals["get_attribute_from_module"] = (
                self.get_attribute_from_module
            )
            self.module_globals["export_all"] = self.export_all

        if allow_getattr_flag:
            getattr_impl = self.custom_getattr or self.__getattr__
            self.module_globals["__getattr__"] = getattr_impl

        self.module_globals["_RESOLVER"] = self.resolver
        self.module_globals["PACKAGE_RESOLVER"] = self

        if eager_flag:
            self.export_all()

    # ------------------------------------------------------------------
    # public facade methods (mirrors legacy API)
    # ------------------------------------------------------------------
    def configure(
        self,
        *,
        include: Optional[IncludeMapping] = None,
        fallbacks: Optional[Mapping[str, str]] = None,
        module_to_parent: Optional[Mapping[str, str]] = None,
        merge: bool = True,
        eager: Optional[bool] = None,
        custom_getattr: Optional[Callable[[str], Any]] = None,
    ) -> ModuleAttributeResolver:
        """Reconfigure the underlying resolver and optionally re-export symbols."""

        self.include_spec = self._merge_include(
            self.include_spec, self._default_include, include, merge
        )
        self.fallbacks_spec = self._merge_map(
            self.fallbacks_spec, self._default_fallbacks, fallbacks, merge
        )
        self.module_parent_spec = self._merge_map(
            self.module_parent_spec,
            self._default_module_to_parent,
            module_to_parent,
            merge,
        )

        eager_flag = self.default_eager if eager is None else eager
        if custom_getattr is not None:
            self.custom_getattr = custom_getattr
            if self.allow_getattr:
                self.module_globals["__getattr__"] = self.custom_getattr
        self._apply_configuration(eager_flag)
        return self.resolver

    configure_resolver = configure

    def build_dictionaries(
        self,
        include_override: Optional[IncludeMapping] = None,
        *,
        eager: bool = False,
        custom_getattr: Optional[Callable[[str], Any]] = None,
    ) -> ModuleAttributeResolver:
        """Compatibility wrapper mirroring the legacy ``build_dictionaries`` helper."""

        eager_flag = eager or self.default_eager
        return self.configure(
            include=include_override,
            merge=False,
            eager=eager_flag,
            custom_getattr=custom_getattr,
        )

    def import_module(self, module_name: str) -> ModuleType:
        return self.resolver.get_module(module_name)

    def get_attribute_from_module(
        self, module: ModuleType, attribute_name: str, class_name: Optional[str] = None
    ):
        if class_name:
            class_obj = getattr(module, class_name)
            return getattr(class_obj, attribute_name)
        return getattr(module, attribute_name)

    def export_all(self) -> None:
        """Eagerly publish resolver-managed attributes into the module globals."""

        self.resolver.bind_to(self.module_globals, install_getattr=False, eager=True)

    def __getattr__(
        self, name: str
    ):  # pragma: no cover - exercised through module-level lookups
        return self.resolver.resolve(name)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _apply_configuration(self, eager: bool) -> None:
        include_payload: Optional[IncludeMapping]
        include_payload = self.include_spec if self.include_spec is not None else None

        self.resolver.fallbacks.clear()
        self.resolver.fallbacks.update(self.fallbacks_spec)

        self.resolver.module_to_parent.clear()
        self.resolver.module_to_parent.update(self.module_parent_spec)

        self.resolver.rebuild(include_payload)
        self.default_eager = eager

        if eager:
            self.export_all()

    @staticmethod
    def _merge_include(
        current: Optional[Dict[str, Union[Sequence[str], str]]],
        defaults: Optional[Mapping[str, Union[Sequence[str], str]]],
        override: Optional[IncludeMapping],
        merge: bool,
    ) -> Optional[Dict[str, Union[Sequence[str], str]]]:
        if merge:
            if current is None:
                base = dict(defaults) if defaults is not None else {}
            else:
                base = dict(current)
            if override:
                base.update(override)
            if not base and current is None and defaults is None and not override:
                return None
            return (
                base
                if base
                else (
                    None
                    if current is None and defaults is None and not override
                    else {}
                )
            )
        if override is None:
            return dict(defaults) if defaults is not None else None
        return dict(override)

    @staticmethod
    def _merge_map(
        current: Mapping[str, str],
        defaults: Mapping[str, str],
        override: Optional[Mapping[str, str]],
        merge: bool,
    ) -> Dict[str, str]:
        if not merge:
            return dict(defaults) if override is None else dict(override)

        base = dict(current)
        if override:
            base.update(override)
        return base


def bootstrap_package(
    module_globals: MutableMapping[str, Any],
    *,
    include: Optional[IncludeMapping] = None,
    fallbacks: Optional[Mapping[str, str]] = None,
    module_to_parent: Optional[Mapping[str, str]] = None,
    eager: bool = False,
    allow_getattr: bool = True,
    install_legacy_helpers: bool = True,
    on_import_error: Optional[Callable[[str, Exception], None]] = None,
    method_predicate: Optional[Callable[[str], bool]] = None,
    custom_getattr: Optional[Callable[[str], Any]] = None,
    lazy_import: Optional[bool] = None,
) -> PackageResolverHandle:
    """Bootstrap a package's ``__init__`` module with dynamic attribute resolution."""

    module_name = module_globals.get("__name__")
    if not module_name:
        raise ValueError("module globals must define '__name__'")

    module_obj = sys.modules.get(module_name)
    if module_obj is None:
        raise ValueError(f"module '{module_name}' is not loaded")

    package_name = module_globals.get("__package__") or getattr(
        module_obj, "__package__", None
    )
    module_file = module_globals.get("__file__") or getattr(
        module_obj, "__file__", None
    )

    if not hasattr(module_obj, "__path__"):
        if module_file and module_file.endswith("__init__.py"):
            package_path = os.path.abspath(os.path.dirname(module_file))
            module_obj.__path__ = [package_path]
            module_globals.setdefault("__path__", module_obj.__path__)
        else:
            raise ValueError(
                "resolver requires a package module with a __path__ attribute"
            )

        if package_name:
            if module_obj.__name__ != package_name:
                sys.modules[package_name] = module_obj
                module_obj.__name__ = package_name
            module_obj.__package__ = package_name
            module_globals["__name__"] = package_name
            module_globals["__package__"] = package_name
        else:
            package_name = module_obj.__name__

    resolver = ModuleAttributeResolver(
        module=module_obj,
        include=include,
        fallbacks=fallbacks,
        module_to_parent=module_to_parent,
        on_import_error=on_import_error,
        method_predicate=method_predicate,
        lazy_import=lazy_import,
    )
    resolver.build()

    handle = PackageResolverHandle(
        resolver,
        module_globals,
        default_include=include,
        default_fallbacks=fallbacks,
        default_module_to_parent=module_to_parent,
        allow_getattr=allow_getattr,
        default_eager=eager,
        custom_getattr=custom_getattr,
    )
    handle.install(
        expose_maps=True,
        install_helpers=install_legacy_helpers,
        allow_getattr=allow_getattr,
        eager=eager,
        custom_getattr=custom_getattr,
    )
    return handle
