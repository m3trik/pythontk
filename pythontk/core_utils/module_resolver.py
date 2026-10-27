# !/usr/bin/python
# coding=utf-8
"""Reusable module attribute resolver for package-style imports."""

import importlib
import importlib.util
import inspect
import os
import pkgutil
import sys
import ast
from types import FunctionType, ModuleType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

__all__ = [
    "ModuleAttributeResolver",
    "MovedNameError",
    "PackageResolverHandle",
    "bootstrap_package",
    "lazy_exports",
    "create_namespace_aliases",
    "import_footprint",
]

IncludeMapping = Mapping[str, Union[Sequence[str], str]]
#: A package's moved-names table: public name -> ``(new home, version)``, the
#: dotted path the name now lives at and the release it left in.
MovedMapping = Mapping[str, Tuple[str, str]]


class MovedNameError(AttributeError):
    """A public name that left its package, read at its old home.

    A move gets no alias (no library imports the product it moved to), so the
    old home answers with where the name went and from which release instead.
    An ``AttributeError``, so ``hasattr`` and ``getattr(..., default)`` behave
    as for any missing name.
    """

    def __init__(self, package: str, name: str, new_home: str, version: str):
        super().__init__(f"{package}.{name} moved to {new_home} in {package} {version}")
        self.name = name
        self.new_home = new_home
        self.version = version


class ModuleAttributeResolver:
    """Discover and resolve attributes exposed from package submodules lazily."""

    def __init__(
        self,
        module: Union[ModuleType, str],
        *,
        include: Optional[Mapping[str, Union[Sequence[str], str]]] = None,
        module_to_parent: Optional[Mapping[str, str]] = None,
        on_import_error: Optional[Callable[[str, Exception], None]] = None,
        method_predicate: Optional[Callable[[str], bool]] = None,
        lazy_import: Optional[bool] = None,
        moved: Optional[MovedMapping] = None,
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
        self.namespace_aliases: Dict[
            str, Tuple[str, Tuple[str, ...]]
        ] = {}  # alias_name -> (module_key, class_list)
        self._set_include(include)

        self.module_to_parent: Dict[str, str] = dict(module_to_parent or {})
        self.on_import_error = on_import_error
        self.method_predicate = method_predicate or (
            lambda name: not name.startswith("_")
        )
        self.lazy_import = lazy_import
        #: The moved-names table (:class:`MovedNameError`).
        self.moved: Dict[str, Tuple[str, str]] = dict(moved or {})

        self.imported_modules: Dict[str, ModuleType] = {}
        #: Parsed trees for the CURRENT build only -- see :meth:`_parse_module`.
        self._ast_cache: Dict[str, Optional[ast.AST]] = {}
        self.class_to_module: Dict[str, str] = {}
        self.method_to_module: Dict[str, Tuple[str, str]] = {}
        self.submodules: set[str] = set()
        #: Declared ``"module->Alias"`` entries not built yet -- see
        #: :meth:`_register_aliases`.
        self._pending_aliases: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
        #: What alias builds published into the package globals (the alias and
        #: each class it fuses), so the next registration can drop it.
        self._alias_published: Dict[str, Any] = {}
        #: Each flat name that two live classes define, with its owners
        #: (``"module.Class"``): one is reached by module order alone, so a
        #: package's tests pin this empty -- see :meth:`_register_flat`.
        self.flat_collisions: Dict[str, List[str]] = {}
        #: Flat names only a ``Deprecation.symbol`` alias defines (so far).
        self._deprecated_flat: Dict[str, Tuple[str, str]] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def build(self) -> "ModuleAttributeResolver":
        """Populate resolver dictionaries based on current include spec."""
        self.class_to_module.clear()
        self.method_to_module.clear()
        self.submodules.clear()
        self.flat_collisions.clear()
        self._deprecated_flat.clear()
        # Fresh per build: a rebuild is how ModuleReloader picks up an edited
        # file, so trees must never survive into the next one.
        self._ast_cache.clear()
        try:
            self._walk_and_register()
        finally:
            self._ast_cache.clear()
        # A retired door keeps its flat name only while nothing live claims it.
        for name, owner in self._deprecated_flat.items():
            self.method_to_module.setdefault(name, owner)
        return self

    def _walk_and_register(self) -> None:
        """The body of :meth:`build`, run with the AST cache live."""
        for _, modname, _ in pkgutil.walk_packages(
            self._package_path, prefix=f"{self.package_name}."
        ):
            # Register direct submodules for lazy resolution
            rel_name = modname[len(self.package_name) + 1 :]
            if "." not in rel_name:
                self.submodules.add(rel_name)

            classes = self._classes_for_module(modname)
            if classes is None:
                continue

            # Explicit class lists can always be lazily registered — names
            # are known statically and the module will be imported on first
            # attribute access, executing any side-effects at that point.
            if classes and "*" not in classes:
                for class_name in classes:
                    self.class_to_module[class_name] = modname
                continue

            # Determine if we should lazy load this module
            should_lazy = self.lazy_import
            if should_lazy is None:
                # Auto-detect: check if module is safe to lazy load
                should_lazy = self._is_safe_to_lazy_load(modname)

            if should_lazy and (not classes or "*" in classes):
                try:
                    scanned_classes, scanned_methods = self._scan_module_attributes(
                        modname
                    )
                    if scanned_classes or scanned_methods:
                        for class_name in scanned_classes:
                            self.class_to_module[class_name] = modname
                        for method_name, class_name, retired in scanned_methods:
                            self._register_flat(method_name, modname, class_name, retired)
                        continue
                except Exception as exc:
                    # If AST scanning fails, import eagerly below
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
        # Check package module itself first for namespace aliases and static attributes
        module_globals = self._module.__dict__
        if name in module_globals:
            return module_globals[name]

        if name in self._pending_aliases:
            alias = self._build_alias(name)
            if alias is None:
                raise AttributeError(
                    f"module {self.package_name} has no attribute '{name}'"
                )
            return alias

        if (
            self._pending_aliases
            and not name.startswith("_")
            and not self._is_registered(name)
        ):
            # A build also publishes each class an alias fuses
            # (``mtk.UvDiagnostics``), so a public name nothing registers may
            # be one of them.
            self._build_pending_aliases()
            if name in module_globals:
                return module_globals[name]

        return self._resolve_registered(name)

    def _is_registered(self, name: str) -> bool:
        return (
            name in self.class_to_module
            or name in self.method_to_module
            or name in self.module_to_parent
            or name in self.submodules
        )

    def _resolve_registered(self, name: str):
        """Resolve *name* through the registry alone, without building an alias."""
        # Check classes
        if name in self.class_to_module:
            if self.class_to_module[name] == self._alias_module:
                # A declared alias that is not pending: mid-build (a module it
                # fuses read it at import) or withdrawn. The placeholder names
                # no real module, so importing it would mask the reason.
                raise AttributeError(
                    f"module {self.package_name} has no attribute '{name}'"
                )
            module = self._import(self.class_to_module[name])
            return getattr(module, name)

        # Check methods (includes class methods)
        if name in self.method_to_module:
            module_name, class_name = self.method_to_module[name]
            module = self._import(module_name)
            class_obj = getattr(module, class_name)
            return getattr(class_obj, name)

        # Check module parent mappings
        if name in self.module_to_parent:
            parent_module = self._import(self.module_to_parent[name])
            return getattr(parent_module, name)

        # Check direct submodules
        if name in self.submodules:
            return self._import(f"{self.package_name}.{name}")

        if name in self.moved:
            new_home, version = self.moved[name]
            raise MovedNameError(self.package_name, name, new_home, version)
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
            for export_name in names or self.iter_registered_names():
                module_globals[export_name] = self.resolve(export_name)

    def iter_registered_names(self) -> Iterable[str]:
        """Return an iterable of attribute names known to the resolver."""
        yielded = set()
        for container in (
            self.class_to_module,
            self.method_to_module,
            self.module_to_parent,
            self.submodules,
        ):
            for key in container:
                if key not in yielded:
                    yielded.add(key)
                    yield key

    def clear_module_cache(self) -> None:
        """Drop cached module imports managed by the resolver."""
        self.imported_modules.clear()

    # ------------------------------------------------------------------
    # namespace aliases ("module->Alias" include keys)
    # ------------------------------------------------------------------
    def _register_aliases(self, stale: Optional[Mapping[str, Any]] = None) -> None:
        """Declare each namespace alias; :meth:`_build_alias` fuses it on first use.

        An alias fuses every class of the modules it names, so building one
        imports them all: building at bootstrap put mayatk's diagnostics
        package and its MASH module in every ``import mayatk`` (about 100 ms).
        Registering only declares the name, which the registry and ``__all__``
        list as before. An alias whose module cannot be found is skipped, as a
        build that failed always was.

        Parameters:
            stale (Mapping[str, Any]): What an earlier run's resolver published
                into these globals (``importlib.reload`` re-runs the bootstrap
                in the same globals). A name still bound to the object that run
                published is dropped, as is this resolver's own, so each alias
                is rebuilt from the modules as they are now.
        """
        module_globals = self._module.__dict__
        for published in (stale or {}, self._alias_published):
            for name, obj in published.items():
                if module_globals.get(name) is obj:
                    del module_globals[name]
        self._alias_published.clear()
        self._pending_aliases.clear()
        for alias_name, (module_key, class_names) in self.namespace_aliases.items():
            try:
                spec = importlib.util.find_spec(f"{self.package_name}.{module_key}")
            except (ImportError, ValueError):
                spec = None
            if spec is None:
                continue
            self._pending_aliases[alias_name] = (module_key, class_names)
            self.class_to_module[alias_name] = self._alias_module

    @property
    def _alias_module(self) -> str:
        """The registry entry of a declared alias, which names no real module."""
        return f"{self.package_name}._namespace_aliases"

    def _build_alias(self, alias_name: str) -> Optional[type]:
        """Fuse a declared namespace alias's classes into one class, now.

        Publishes the alias, and each class it fuses, in the package globals.
        One that cannot be built (its module fails to import, it has no class
        to fuse, or a class it lists is missing) is withdrawn from the registry
        and ``__all__``, as it was never published before.

        Parameters:
            alias_name (str): A name :meth:`_register_aliases` declared.

        Returns:
            (type) The alias, or ``None`` when it could not be built.
        """
        module_key, class_names = self._pending_aliases.pop(alias_name)
        try:
            classes = self._collect_alias_classes(module_key, class_names)
        except Exception:
            # Not an ImportError (that withdraws the alias below): a fused
            # module raised mid-import. Keep the alias pending, so the next
            # read retries and raises the real error again.
            self._pending_aliases[alias_name] = (module_key, class_names)
            raise
        if not classes:
            self._withdraw_alias(alias_name)
            return None
        try:
            alias = type(
                alias_name,
                tuple(classes.values()),
                {
                    "__module__": self.package_name,
                    "__doc__": "Unified namespace combining: "
                    + ", ".join(cls.__name__ for cls in classes.values()),
                },
            )
        except TypeError:  # not co-inheritable: an authoring error, so raised
            self._withdraw_alias(alias_name)
            raise
        for name, obj in (*classes.items(), (alias_name, alias)):
            self._module.__dict__[name] = obj
            self._alias_published[name] = obj
        return alias

    def _collect_alias_classes(
        self, module_key: str, class_names: Tuple[str, ...]
    ) -> Dict[str, type]:
        """The classes a declared alias fuses; empty when it cannot be built."""
        try:
            module = self._import(f"{self.package_name}.{module_key}")
        except ImportError:
            # An explicit class list may still resolve through the registry.
            module = None
        if class_names == ("*",):
            return self._alias_classes(module) if module is not None else {}
        classes: Dict[str, type] = {}
        for class_name in class_names:
            cls = getattr(module, class_name, None)
            if cls is None:
                try:
                    cls = self._resolve_registered(class_name)
                except (AttributeError, ImportError):
                    return {}
            classes[class_name] = cls
        return classes

    def _build_pending_aliases(self) -> None:
        """Build every declared alias not built yet."""
        for alias_name in list(self._pending_aliases):
            self._build_alias(alias_name)

    def _alias_classes(self, module: ModuleType) -> Dict[str, type]:
        """The public classes a ``"*"`` alias fuses: those *module* defines or,
        for a package, those each of its direct non-package submodules defines
        -- never one a module merely imports (``typing.Any``, a sibling's
        utils class, a ``Doc = ReportDoc`` alias)."""
        sources = [module]
        if hasattr(module, "__path__"):
            sources = []
            for _, modname, ispkg in pkgutil.iter_modules(module.__path__):
                if ispkg:
                    continue
                try:
                    sources.append(self._import(f"{module.__name__}.{modname}"))
                except (ImportError, AttributeError):
                    continue
        classes: Dict[str, type] = {}
        for source in sources:
            for name in dir(source):
                obj = getattr(source, name)
                if (
                    isinstance(obj, type)
                    and not name.startswith("_")
                    and obj.__module__ == source.__name__
                ):
                    classes.setdefault(name, obj)
        return classes

    def _withdraw_alias(self, alias_name: str) -> None:
        self.class_to_module.pop(alias_name, None)
        published_all = self._module.__dict__.get("__all__")
        if isinstance(published_all, list) and alias_name in published_all:
            published_all.remove(alias_name)

    def _scan_module_attributes(
        self, module_name: str
    ) -> Tuple[Sequence[str], List[Tuple[str, str, bool]]]:
        """Statically scan a module for classes and their methods.

        Returns:
            (tuple) The public top-level names, and each public method as
            ``(name, class, retired)`` -- *retired* for a ``Deprecation.symbol``
            alias -- in source order, so two classes defining one name both show.
        """
        tree = self._parse_module(module_name)
        if not tree:
            return [], []

        top_level = []
        methods = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if self.method_predicate(node.name):
                    top_level.append(node.name)
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            if self.method_predicate(item.name):
                                methods.append(
                                    (item.name, node.name, self._is_retired(item))
                                )
            elif isinstance(node, ast.FunctionDef):
                if self.method_predicate(node.name):
                    top_level.append(node.name)
        return top_level, methods

    @staticmethod
    def _is_retired(func: ast.FunctionDef) -> bool:
        """True for a ``@Deprecation.symbol(...)`` method: a retired door to a
        method another class owns, not a second owner of its name."""
        for decorator in func.decorator_list:
            call = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(call, ast.Attribute) and call.attr == "symbol":
                owner = call.value
                name = owner.attr if isinstance(owner, ast.Attribute) else ""
                if name == "Deprecation" or getattr(owner, "id", "") == "Deprecation":
                    return True
        return False

    def _register_flat(
        self, name: str, module_name: str, class_name: str, retired: bool = False
    ) -> None:
        """Publish *name* as the flat ``pkg.<name>`` of *class_name*'s method.

        A flat name has one owner. A second live owner is recorded in
        :attr:`flat_collisions` (and, as before, wins by module order). A
        retired alias owns nothing: it takes the name only when no live method
        does (:meth:`build`), so it never shadows the method it points at.
        """
        owner = (module_name, class_name)
        if retired:
            self._deprecated_flat.setdefault(name, owner)
            return
        previous = self.method_to_module.get(name)
        if previous is not None and previous != owner:
            owners = self.flat_collisions.setdefault(name, [".".join(previous)])
            if ".".join(owner) not in owners:
                owners.append(".".join(owner))
        self.method_to_module[name] = owner

    def _is_safe_to_lazy_load(self, module_name: str) -> bool:
        """Check if a module is safe to lazy load (no top-level side effects)."""
        tree = self._parse_module(module_name)
        if not tree:
            return False  # If we can't parse, assume unsafe
        return self._statements_are_side_effect_free(tree.body)

    #: Top-level statements that run nothing a caller relies on at import.
    _SIDE_EFFECT_FREE = (
        ast.Import,
        ast.ImportFrom,
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Assign,
        ast.AnnAssign,
        ast.Pass,
    )

    @classmethod
    def _statements_are_side_effect_free(cls, statements: Sequence[ast.stmt]) -> bool:
        """Whether *statements* (a module's top level) run nothing a caller
        relies on at import. A top-level ``if`` or ``try`` is judged by what it
        runs, not passed whole: its branches, body, ``else`` and ``finally``.
        Two parts are skipped: an exception handler, the failure path of a
        guarded import (``except ImportError: print(error)``), and an
        ``if __name__ == "__main__"`` block, which never runs on import."""
        for node in statements:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue  # Module docstrings / standalone string literals
            if isinstance(node, ast.If):
                if cls._is_main_guard(node):
                    # Its body runs only as a script; its ``else`` on every import.
                    if not cls._statements_are_side_effect_free(node.orelse):
                        return False
                    continue
                if not (
                    cls._statements_are_side_effect_free(node.body)
                    and cls._statements_are_side_effect_free(node.orelse)
                ):
                    return False
                continue
            if isinstance(node, ast.Try):
                if not all(
                    cls._statements_are_side_effect_free(block)
                    for block in (node.body, node.orelse, node.finalbody)
                ):
                    return False
                continue
            if not isinstance(node, cls._SIDE_EFFECT_FREE):
                return False  # e.g. a top-level call
        return True

    @staticmethod
    def _is_main_guard(node: ast.If) -> bool:
        """``if __name__ == "__main__":``, either way round."""
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
        ):
            return False
        sides = (test.left, test.comparators[0])
        return any(
            isinstance(a, ast.Name)
            and a.id == "__name__"
            and isinstance(b, ast.Constant)
            and b.value == "__main__"
            for a, b in (sides, sides[::-1])
        )

    def _parse_module(self, module_name: str) -> Optional[ast.AST]:
        """Parse a module file into an AST, once per build.

        Both ``_is_safe_to_lazy_load`` and ``_scan_module_attributes`` need the
        tree of any module on the lazy path, so an un-memoized parse read and
        parsed each of them twice -- measured at 13 calls for 7 modules on
        pythontk's own bootstrap, which every package here boots through.

        The cache is cleared by :meth:`build` on entry and exit, so a rebuild
        always re-reads from disk (what ``ModuleReloader`` relies on) and the
        trees are not retained afterwards. Neither consumer mutates the tree.
        """
        if module_name in self._ast_cache:
            return self._ast_cache[module_name]
        tree = self._read_module_ast(module_name)
        self._ast_cache[module_name] = tree
        return tree

    @staticmethod
    def _read_module_ast(module_name: str) -> Optional[ast.AST]:
        """Locate *module_name* on disk and parse it. ``None`` when it cannot
        be found or read."""
        try:
            loader = pkgutil.get_loader(module_name)
            if not loader or not hasattr(loader, "get_filename"):
                return None

            filename = loader.get_filename(module_name)
            if not filename or not os.path.exists(filename):
                return None

            with open(filename, "r", encoding="utf-8-sig") as f:
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
            self.namespace_aliases.clear()
            return

        direct: Dict[str, Tuple[str, ...]] = {}
        absolute: Dict[str, Tuple[str, ...]] = {}
        self.namespace_aliases.clear()

        for key, value in include.items():
            # Check for namespace alias using arrow delimiter
            alias_name = None
            module_key = key

            if "->" in key:
                # Split on arrow: "module.path->AliasName"
                module_key, alias_name = key.split("->", 1)
                module_key = module_key.strip()
                alias_name = alias_name.strip()

            normalized = self._normalize_include_value(value)

            # Store namespace alias if detected
            if alias_name:
                if isinstance(normalized, tuple):
                    self.namespace_aliases[alias_name] = (module_key, normalized)
                else:
                    # Convert to tuple for storage
                    self.namespace_aliases[alias_name] = (
                        module_key,
                        (
                            (normalized,)
                            if isinstance(normalized, str)
                            else tuple(normalized)
                        ),
                    )

            # Register module with original (non-arrow) key
            if "." in module_key:
                absolute[f"{self.package_name}.{module_key}"] = normalized
            else:
                direct[module_key] = normalized

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
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                self._register_class(name, obj, module.__name__, register_methods=True)

    def _register_selected_classes(
        self, module: ModuleType, classes: Sequence[str]
    ) -> None:
        for class_name in classes:
            obj = getattr(module, class_name, None)
            if obj is not None:
                # For classes, ensure they are defined in this module (not imported)
                if inspect.isclass(obj) and obj.__module__ != module.__name__:
                    continue

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
                retired = bool(getattr(member, "__deprecated__", None))
                self._register_flat(method_name, module_name, class_name, retired)

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
    ) -> Union[Tuple[str, ...], Tuple[str, Tuple[str, ...]]]:
        """Normalize include value, detecting namespace alias tuples.

        Returns:
            Tuple of class names, OR
            (alias_name, tuple_of_class_names) for namespace alias syntax
        """
        if value is None:
            return ("*",)
        if isinstance(value, str):
            return ("*",) if value == "*" else (value,)
        # Check if it's a tuple with (alias_name, [class_list]) pattern
        if isinstance(value, tuple) and len(value) == 2:
            alias_name, class_list = value
            if isinstance(alias_name, str) and isinstance(class_list, (list, tuple)):
                # This is a namespace alias definition
                return (alias_name, tuple(class_list))
        return tuple(value)


class PackageResolverHandle:
    """Facade that wires a :class:`ModuleAttributeResolver` into a package module."""

    def __init__(
        self,
        resolver: ModuleAttributeResolver,
        module_globals: MutableMapping[str, Any],
        *,
        default_include: Optional[IncludeMapping] = None,
        default_module_to_parent: Optional[Mapping[str, str]] = None,
        allow_getattr: bool = True,
        default_eager: bool = False,
        custom_getattr: Optional[Callable[[str], Any]] = None,
        set_all: bool = True,
    ) -> None:
        self.resolver = resolver
        self.module_globals = module_globals
        self.allow_getattr = allow_getattr
        self.default_eager = default_eager
        self.custom_getattr = custom_getattr
        self.set_all = set_all
        # Snapshot any hand-declared ``__all__`` so it is preserved (unioned)
        # with the resolver-registered names on every (re)build, never letting a
        # runtime reconfigure accumulate stale derived names. The list a handle
        # published is derived, not declared: a reload re-runs the bootstrap in
        # the same globals, where the last run's handle left its ``__all__``.
        declared = module_globals.get("__all__")
        previous = module_globals.get("PACKAGE_RESOLVER")
        if declared is not None and declared is getattr(
            previous, "_published_all", None
        ):
            declared = None
        self._declared_all: Tuple[str, ...] = tuple(declared or ())
        self._published_all: Optional[list] = None
        # What the last run's alias builds published into these same globals.
        stale = getattr(getattr(previous, "resolver", None), "_alias_published", None)
        self._stale_aliases: Dict[str, Any] = (
            dict(stale) if isinstance(stale, dict) else {}
        )

        self._default_include = (
            dict(default_include) if default_include is not None else None
        )
        self._default_module_to_parent = dict(default_module_to_parent or {})

        self.include_spec: Optional[Dict[str, Union[Sequence[str], str]]] = (
            dict(self._default_include) if self._default_include is not None else None
        )
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
            self.module_globals["IMPORTED_MODULES"] = self.resolver.imported_modules
            self.module_globals["MODULE_TO_PARENT"] = self.resolver.module_to_parent

        if install_helpers:
            self.module_globals["configure"] = self.configure
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

        # Declare the namespace aliases: each is built on first use. Without a
        # module ``__getattr__`` nothing would build one on first read, so
        # those are built now.
        self.resolver._register_aliases(self._stale_aliases)
        if eager_flag or not allow_getattr_flag:
            self.resolver._build_pending_aliases()
        if eager_flag:
            self.export_all()

        # Publish the resolved public surface as ``__all__`` (after the aliases
        # are declared, so their names are included).
        self._populate_all()

    # ------------------------------------------------------------------
    # public facade methods (mirrors legacy API)
    # ------------------------------------------------------------------
    def configure(
        self,
        *,
        include: Optional[IncludeMapping] = None,
        module_to_parent: Optional[Mapping[str, str]] = None,
        merge: bool = True,
        eager: Optional[bool] = None,
        custom_getattr: Optional[Callable[[str], Any]] = None,
    ) -> ModuleAttributeResolver:
        """Reconfigure the underlying resolver and optionally re-export symbols."""

        self.include_spec = self._merge_include(
            self.include_spec, self._default_include, include, merge
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
    def _populate_all(self) -> None:
        """Assign ``__all__`` from the resolver's registered top-level names.

        The exported surface is the union of any ``__all__`` the package
        hand-declared before bootstrap and the resolver's ``class_to_module``
        keys -- i.e. every module-level class, function, constant, and declared
        namespace alias reachable as ``pkg.Name``. Flat method aliases
        (``method_to_module``) and bare submodule names are intentionally
        excluded: they resolve lazily via ``__getattr__`` but do not belong in
        the advertised public surface. No-op when ``set_all`` is disabled.
        """
        if not self.set_all:
            return
        names = set(self._declared_all)
        names.update(self.resolver.class_to_module)
        self._published_all = sorted(names)
        self.module_globals["__all__"] = self._published_all

    def _apply_configuration(self, eager: bool) -> None:
        self.resolver.module_to_parent.clear()
        self.resolver.module_to_parent.update(self.module_parent_spec)

        self.resolver.rebuild(self.include_spec)
        self.default_eager = eager

        # Re-declare the namespace aliases: what the last builds published is
        # dropped, so each is rebuilt from the modules as they are now.
        self.resolver._register_aliases()
        if eager or not self.allow_getattr:
            self.resolver._build_pending_aliases()
        if eager:
            self.export_all()

        # Keep ``__all__`` in sync with the rebuilt resolver state.
        self._populate_all()

    @staticmethod
    def _merge_include(
        current: Optional[Dict[str, Union[Sequence[str], str]]],
        defaults: Optional[Mapping[str, Union[Sequence[str], str]]],
        override: Optional[IncludeMapping],
        merge: bool,
    ) -> Optional[Dict[str, Union[Sequence[str], str]]]:
        if not merge:
            # Replace mode: use override, or fall back to defaults
            if override is not None:
                return dict(override)
            return dict(defaults) if defaults is not None else None

        # Merge mode: combine current + override
        base = dict(current or defaults or {})
        if override:
            base.update(override)
        return base if base else None

    @staticmethod
    def _merge_map(
        current: Mapping[str, str],
        defaults: Mapping[str, str],
        override: Optional[Mapping[str, str]],
        merge: bool,
    ) -> Dict[str, str]:
        if not merge:
            return dict(override) if override is not None else dict(defaults)

        base = dict(current)
        if override:
            base.update(override)
        return base


def bootstrap_package(
    module_globals: MutableMapping[str, Any],
    *,
    include: Optional[IncludeMapping] = None,
    module_to_parent: Optional[Mapping[str, str]] = None,
    eager: bool = False,
    allow_getattr: bool = True,
    install_legacy_helpers: bool = True,
    on_import_error: Optional[Callable[[str, Exception], None]] = None,
    method_predicate: Optional[Callable[[str], bool]] = None,
    custom_getattr: Optional[Callable[[str], Any]] = None,
    lazy_import: Optional[bool] = None,
    set_all: bool = True,
    moved: Optional[MovedMapping] = None,
) -> PackageResolverHandle:
    """Bootstrap a package's ``__init__`` module with dynamic attribute resolution.

    When ``set_all`` is true (the default) the package's ``__all__`` is derived
    automatically from the resolved public surface, so it never has to be listed
    by hand or kept in sync with ``DEFAULT_INCLUDE``. Any ``__all__`` the module
    declared before this call is preserved and unioned in. Pass ``set_all=False``
    to manage ``__all__`` manually.

    *moved* is the package's moved-names table, ``{name: (new home, version)}``
    (the ``MOVED_NAMES`` literal each library's ``__init__`` declares): a read
    of a listed name raises :class:`MovedNameError` naming where it went.
    """

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
        module_to_parent=module_to_parent,
        on_import_error=on_import_error,
        method_predicate=method_predicate,
        lazy_import=lazy_import,
        moved=moved,
    )
    resolver.build()

    handle = PackageResolverHandle(
        resolver,
        module_globals,
        default_include=include,
        default_module_to_parent=module_to_parent,
        allow_getattr=allow_getattr,
        default_eager=eager,
        custom_getattr=custom_getattr,
        set_all=set_all,
    )
    handle.install(
        expose_maps=True,
        install_helpers=install_legacy_helpers,
        allow_getattr=allow_getattr,
        eager=eager,
        custom_getattr=custom_getattr,
    )
    return handle


class _ModuleHook:
    """Install a module ``__getattr__``/``__dir__`` that chains onto the one before
    it, and survives ``importlib.reload``.

    A reload re-runs the module body in the SAME globals, so an installer finds
    the hooks the previous run left there. Chaining onto those stacked one more
    layer per reload, each still serving what the old body declared. Every hook
    is therefore tagged with the ``__spec__`` it was installed under -- a reload
    binds a fresh one -- and :meth:`previous` walks past a stale hook to the one
    it chained onto, while two installers in one run still compose. Shared by
    :func:`lazy_exports` and ``Deprecation.attributes``.
    """

    _TAG = "_module_hook"

    @classmethod
    def previous(
        cls, module_globals: Mapping[str, Any], key: str
    ) -> Tuple[Optional[Callable], list]:
        """The hook a new ``module_globals[key]`` should chain onto.

        Parameters:
            module_globals (Mapping[str, Any]): The module's ``globals()``.
            key (str): ``"__getattr__"`` or ``"__dir__"``.

        Returns:
            (tuple) The live hook (``None`` if there is none), and the ``state``
            of each hook the previous run installed under *key* -- skipped here,
            or already by an earlier installer of this run (whichever runs first
            cuts them out of the chain).
        """
        token = module_globals.get("__spec__")
        hook = module_globals.get(key)
        stale = []
        while isinstance(hook, FunctionType):
            tag = hook.__dict__.get(cls._TAG)
            if tag is None:
                break
            spec, chained, state, cut = tag
            if spec is token:
                stale.extend(cut)
                break
            stale.append(state)
            hook = chained
        return hook, stale

    @classmethod
    def install(
        cls,
        module_globals: MutableMapping[str, Any],
        key: str,
        hook: FunctionType,
        chained: Optional[Callable] = None,
        state: Optional[Mapping[str, Any]] = None,
        stale: Sequence[Any] = (),
    ) -> None:
        """Publish *hook* as ``module_globals[key]``, tagged for :meth:`previous`.

        Parameters:
            module_globals (MutableMapping[str, Any]): The module's ``globals()``.
            key (str): ``"__getattr__"`` or ``"__dir__"``.
            hook (FunctionType): The new hook.
            chained (Callable): The hook it falls back to, from :meth:`previous`.
            state (Mapping[str, Any]): What a later run of the same installer
                needs back from this one.
            stale (Sequence): The stale states :meth:`previous` returned, kept
                for this run's later installers.
        """
        spec = module_globals.get("__spec__")
        hook.__dict__[cls._TAG] = (spec, chained, state, tuple(stale))
        module_globals[key] = hook


def lazy_exports(
    module_globals: MutableMapping[str, Any],
    sources: IncludeMapping,
) -> None:
    """Publish a subpackage's names from its submodules, each loaded on first use.

    The subpackage ``__init__`` form of :func:`bootstrap_package` (CODE_STANDARD
    section 4): the ``__init__`` stays a declaration, and importing the package
    loads none of the listed submodules. A parent's ``bootstrap_package`` scan
    (``pkgutil.walk_packages``) imports every subpackage ``__init__``, so an eager
    re-export there is paid by every ``import <root>`` in the ecosystem.

    Resolution is PEP 562: a name is imported on first attribute access and cached
    in the package globals, so ``from pkg import Name`` and
    ``mock.patch("pkg.Name.attr")`` behave as they would with an eager re-export.
    ``__dir__`` lists every name; ``__all__`` is the public (non-``_``) ones,
    unioned with any ``__all__`` declared before the call. A ``__getattr__``
    already in the globals (``Deprecation.attributes``) answers the names this
    one does not own.

    Reload-safe: ``importlib.reload`` re-runs the call in the same globals, and
    the re-run drops the names the last run cached, chains onto what preceded the
    last run's hooks rather than onto them, and unions only the hand-declared
    ``__all__`` -- so the package serves the reloaded submodules' objects.

    Parameters:
        module_globals (MutableMapping[str, Any]): The package's ``globals()``.
        sources (Mapping[str, Sequence[str] | str]): Submodule (relative to the
            package, dotted when nested) -> the names it defines. Explicit names
            only: a ``"*"`` would need the scan this exists to avoid.

    Raises:
        ValueError: A name is listed under two submodules, or a ``"*"`` is given.
        ImportError: (on first use of a name) Its submodule's import raised an
            AttributeError, or the submodule does not define the name; the
            AttributeError is the ``__cause__``. Any other error the import
            raises propagates as is.
    """
    package = module_globals["__name__"]
    owner: Dict[str, str] = {}
    for submodule, names in sources.items():
        for name in (names,) if isinstance(names, str) else names:
            if name == "*":
                raise ValueError(f"{package}: list {submodule!r}'s names, not '*'")
            if name in owner:
                raise ValueError(
                    f"{package}: {name!r} listed under both {owner[name]!r} "
                    f"and {submodule!r}"
                )
            owner[name] = submodule
    # A value the last run cached would shadow the reloaded submodule's.
    for name in owner:
        module_globals.pop(name, None)
    fallback, stale = _ModuleHook.previous(module_globals, "__getattr__")
    # The list a stale run published is derived, not declared by this run.
    declared = module_globals.get("__all__")
    if any(state and declared is state["__all__"] for state in stale):
        declared = None

    def __getattr__(name: str) -> Any:
        submodule = owner.get(name)
        if submodule is None:
            if fallback is not None:
                return fallback(name)
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        try:
            value = getattr(importlib.import_module(f"{package}.{submodule}"), name)
        except AttributeError as exc:
            # Raised as is, ``from pkg import name`` reports a bare "cannot
            # import name" without the cause, and ``hasattr`` reads "absent".
            raise ImportError(
                f"{package}.{name}: loading {package}.{submodule} failed: {exc}"
            ) from exc
        module_globals[name] = value
        return value

    def __dir__() -> list:
        return sorted(set(module_globals) | set(owner))

    exported = sorted(
        set(declared or ()) | {name for name in owner if not name.startswith("_")}
    )
    _ModuleHook.install(
        module_globals,
        "__getattr__",
        __getattr__,
        fallback,
        {"__all__": exported},
        stale,
    )
    _ModuleHook.install(module_globals, "__dir__", __dir__)
    module_globals["__all__"] = exported


def create_namespace_aliases(
    module_globals: MutableMapping[str, Any],
    aliases: Mapping[str, Union[str, Sequence[str]]],
    *,
    include_spec: Optional[IncludeMapping] = None,
) -> None:
    """Create multi-inheritance namespace classes from groups of related classes.

    This helper allows you to group multiple classes under a single namespace alias,
    creating a unified interface through multi-inheritance. Call this AFTER
    bootstrap_package() to ensure all classes are available.

    Args:
        module_globals: The package's globals() dict (same as passed to bootstrap_package)
        aliases: Dict mapping alias name -> module key or explicit class list.
            The grouped classes are fused by multiple inheritance, so they must be
            co-inheritable (mixins, or plain classes with no conflicting instance
            layout — value types with ``__slots__``/namedtuple bases will raise a
            "layout conflict"). Examples:
                {"MyMixins": "some_pkg.mixins"}         # class list from include_spec[key]
                {"MyMixins": ["FooMixin", "BarMixin"]}  # explicit class list
        include_spec: Optional DEFAULT_INCLUDE dict to lookup module keys.
                     If not provided, will try to use module_globals.get('_INCLUDE_SPEC')

    Example:
        # After bootstrap_package(), fuse related mixins into one namespace class:
        create_namespace_aliases(globals(), {
            "MyMixins": ["FooMixin", "BarMixin"],
        })

        obj = pkg.MyMixins()  # one instance exposing FooMixin's and BarMixin's methods

    Raises:
        ValueError: If module key not found in include_spec
        KeyError: If class name not found in module globals
    """
    # Try to get include_spec from module globals if not provided
    if include_spec is None:
        include_spec = module_globals.get("DEFAULT_INCLUDE") or module_globals.get(
            "_INCLUDE_SPEC"
        )

    for alias_name, source in aliases.items():
        # Determine class list
        if isinstance(source, str):
            # Source is a module key - lookup in include_spec
            if include_spec is None:
                raise ValueError(
                    f"Cannot resolve module key '{source}' without include_spec. "
                    "Pass include_spec parameter or define DEFAULT_INCLUDE in module."
                )

            if source not in include_spec:
                raise ValueError(
                    f"Module key '{source}' not found in include_spec. "
                    f"Available keys: {list(include_spec.keys())}"
                )

            class_spec = include_spec[source]

            # Normalize to list
            if isinstance(class_spec, str):
                if class_spec == "*":
                    raise ValueError(
                        f"Cannot create namespace alias for wildcard module '{source}'. "
                        "Specify explicit class list in include_spec or use explicit class list in aliases."
                    )
                class_names = [class_spec]
            else:
                class_names = list(class_spec)
        else:
            # Source is already a class list
            class_names = list(source)

        # Resolve class objects from module globals
        missing_classes = []
        base_classes = []

        # Get the resolver for lazy loading
        resolver = module_globals.get("PACKAGE_RESOLVER")

        for class_name in class_names:
            if class_name in module_globals:
                # Already loaded
                base_classes.append(module_globals[class_name])
            elif resolver:
                # Try lazy loading via resolver
                try:
                    cls = resolver.resolver.resolve(class_name)
                    base_classes.append(cls)
                    # Also store in globals for future access
                    module_globals[class_name] = cls
                except AttributeError:
                    missing_classes.append(class_name)
            else:
                missing_classes.append(class_name)

        if missing_classes:
            raise KeyError(
                f"Cannot create alias '{alias_name}': classes not found in module globals: "
                f"{missing_classes}. Ensure bootstrap_package() was called first."
            )

        # Create multi-inheritance namespace class
        namespace_class = type(
            alias_name,
            tuple(base_classes),
            {
                "__module__": module_globals.get("__name__", "unknown"),
                "__doc__": f"Unified namespace combining: {', '.join(class_names)}",
            },
        )

        # Register in module globals
        module_globals[alias_name] = namespace_class


# What a declaring ``__init__`` may load: this module, and the deprecation hook
# that declares a package's moved names (``Deprecation.attributes``, section 5).
_DECLARATION_MODULES = frozenset({__name__, "pythontk.core_utils.deprecation"})

# Run in a fresh interpreter by ``import_footprint``: every module the import
# loaded, whether it is a package, and whether its root boots through this module.
_FOOTPRINT_PROBE = """\
import json, sys
before = set(sys.modules)
__import__(sys.argv[1])
rows = []
for name in sorted(set(sys.modules) - before):
    module = sys.modules[name]
    if module is not None:
        root = sys.modules.get(name.partition(".")[0])
        booted = "PACKAGE_RESOLVER" in getattr(root, "__dict__", {})
        rows.append([name, hasattr(module, "__path__"), booted])
stdlib = sorted(getattr(sys, "stdlib_module_names", ()))
print(json.dumps({"rows": rows, "stdlib": stdlib}))
"""


def import_footprint(package: str) -> Dict[str, List[str]]:
    """What importing *package* loads beyond declarations, in a fresh interpreter.

    CODE_STANDARD section 4: a package ``__init__`` declares its names and
    imports none of them. Importing a package that keeps to it loads package
    ``__init__`` files, the declaration machinery (this module, and the
    ``Deprecation.attributes`` hook for moved names) and the standard library
    -- nothing else.
    The import runs in a child process (``sys.executable``, given this
    process's ``sys.path``), so nothing already loaded here can hide a load. In
    a DCC GUI, where ``sys.executable`` is the host binary, call it from a
    headless interpreter.

    Parameters:
        package (str): The package to import, e.g. ``"pythontk"``.

    Returns:
        (dict) ``"modules"``: the non-package modules that loaded from
        *package* or from any package that boots through this module, the
        declaration machinery excepted. ``"third_party"``: the top-level
        names of everything else that loaded outside the standard library
        (named by ``sys.stdlib_module_names``, Python 3.10+; always empty
        before). Both are empty for an import that only declares.

    Raises:
        RuntimeError: The child process could not import *package*.
    """
    import json
    import subprocess

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    proc = subprocess.run(
        [sys.executable, "-c", _FOOTPRINT_PROBE, package],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if proc.returncode:
        raise RuntimeError(f"importing {package} failed:\n{proc.stderr[-2000:]}")
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    own = package.partition(".")[0]
    stdlib = set(report["stdlib"])
    modules, third_party = [], set()
    for name, is_package, booted in report["rows"]:
        top = name.partition(".")[0]
        if booted or top == own:
            if not is_package and name not in _DECLARATION_MODULES:
                modules.append(name)
        elif stdlib and top not in stdlib:
            third_party.add(top)
    return {"modules": modules, "third_party": sorted(third_party)}
