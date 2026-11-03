# !/usr/bin/python
# coding=utf-8
"""The static half of :class:`~pythontk.ApiSurface`: a package's public API read from its source.

Walks a package's source with ``ast``, importing nothing, so it reads any
package anywhere -- one whose host application is absent, a checkout on a CI
box with nothing installed. What it records is the rule of the public
surface:

* top-level functions and classes whose names do not start with ``_``, the
  classes' public methods, properties, static and class methods, and the
  module's public ALL_CAPS constants;
* the members a public class inherits from a PRIVATE base declared beside it
  or imported from a sibling module of the same package, followed through
  further private bases (the capability-mixin shape the ecosystem builds its
  classes from: ``Matrices(_MatrixMath, ...)``); a public base is documented
  under its own entry;
* each retirement's ``remove_in`` and ``since``, read off its ``Deprecation``
  decorator, a shared string constant (``_REMOVE_IN = "0.18.0"``) resolved.

One job of :class:`~pythontk.ApiSurface`, composed in ``api_surface.py``;
every method is reached through the facade (``cls`` is ``ApiSurface``).

Each file is parsed once per read (:class:`_Parses`) and its tree dropped with
the read: the trees of the workspace's packages held for a process's life cost
about 900 MB (measured), and what a read keeps is its surface. A re-read after
an edit walks only the modules whose files changed (``_WALKED``), which keeps
the module surfaces, never a tree.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from pythontk.core_utils.api_surface.package_surface import (
    ClassSurface,
    ModuleSurface,
    PackageSurface,
)
from pythontk.core_utils.symbol_record import SymbolRecord


class _Parses(dict):
    """One read's parsed files, ``{path: tree}`` (``None``: no module): each
    file parsed once per read -- a mixin module several modules take a base
    from -- and dropped with it. :attr:`unread` lists the files that could not
    be read at all (a sync client's or a scanner's lock), so the read that met
    one is kept nowhere; :attr:`asked` the files the module being walked read
    or looked for, each as it was then (:attr:`stamps`)."""

    def __init__(self) -> None:
        super().__init__()
        self.unread: List[Path] = []
        self.asked: set = set()
        self.stamps: Dict[Path, Optional[Tuple[int, int]]] = {}

    @staticmethod
    def stamp(path: Path) -> Optional[Tuple[int, int]]:
        """*path*'s ``(mtime_ns, size)``; ``None`` when there is no such file."""
        try:
            stat = os.stat(path)
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def ask(self, path: Path) -> None:
        """Note that the module being walked reads (or looks for) *path*."""
        self.asked.add(path)
        if path not in self.stamps:
            self.stamps[path] = self.stamp(path)


class _SourceReader:
    """Reads a package's public surface from its source (see the module docstring)."""

    #: Folders a walk never enters.
    SKIP_DIR_NAMES = frozenset(
        {
            "build",
            "dist",
            "test",
            "tests",
            "docs",
            ".venv",
            "venv",
            "__pycache__",
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            "site-packages",
        }
    )
    #: The longest summary a static record keeps: one terminal line.
    SUMMARY_MAX = 100
    #: Class-name prefixes of generated code (Qt Designer's ``Ui_`` classes),
    #: which is no hand-written API.
    GENERATED_CLASS_PREFIXES = ("Ui_",)
    #: Decorator spellings that retire a WHOLE symbol. ``Deprecation.parameter``
    #: is absent on purpose: it retires one keyword of a function that stays.
    #: PEP 702's spellings read like ``pythontk``'s.
    DEPRECATION_DECORATORS = (
        "deprecated",
        "Deprecation.symbol",
        "warnings.deprecated",
        "typing_extensions.deprecated",
    )
    #: The ``Deprecation`` calls that retire something smaller than a symbol (a
    #: keyword, a value, a code path, a moved attribute, a class constant):
    #: never a symbol's mark, but the same one-release promise.
    RETIREMENT_CALLS = ("parameter", "values", "warn", "attributes", "class_attribute")

    #: Never filled: a read's parses live as long as the read (:class:`_Parses`).
    #: Kept, empty, for a caller that clears it between reads (m3trik's
    #: ``test_check_api_surface``), where clearing it is now a no-op.
    _PARSED: Dict[Path, Optional[ast.Module]] = {}
    #: ``{(file, package folder): (each file its walk read or looked for, as
    #: it was then; the module's surface)}``: a module whose files are all as
    #: they were is not walked again, so a re-read after an edit parses only
    #: what the edit reaches. It holds no tree; its surfaces are the ones the
    #: kept reads hold.
    _WALKED: Dict[
        Tuple[str, str],
        Tuple[
            Tuple[Tuple[Path, Optional[Tuple[int, int]]], ...], Optional[ModuleSurface]
        ],
    ] = {}

    # -- files ---------------------------------------------------------------------
    @classmethod
    def _iter_py_files(cls, root: Path) -> Iterable[Path]:
        """Every ``.py`` under *root* a walk reads, in POSIX-string order: the
        order is case-sensitive on every platform (a ``Path`` sort is not on
        Windows), so a read on Windows matches CI's byte for byte.
        ``__main__.py`` is a script, never API."""
        for path in sorted(root.rglob("*.py"), key=lambda p: p.as_posix()):
            if set(path.relative_to(root).parts[:-1]) & cls.SKIP_DIR_NAMES:
                continue
            if path.name == "__main__.py":
                continue
            yield path

    @classmethod
    def _parse(cls, path: Path, memo: Optional[_Parses] = None) -> Optional[ast.Module]:
        """*path*'s AST, parsed once per read (*memo*, the read's own; none:
        parsed each time asked); ``None`` when it does not decode or parse,
        which holds until the file changes, and when it cannot be read at all,
        which may not: that file is listed in ``memo.unread`` and remembered
        nowhere. Each top-level class is stamped with the file it is in
        (``ptk_path``), so a member pulled up from a private base in another
        file knows where it is defined."""
        if memo is not None:
            memo.ask(path)
            if path in memo:
                return memo[path]
        tree: Optional[ast.Module] = None
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except OSError:
            if memo is not None:
                memo.unread.append(path)
            return None
        except (UnicodeDecodeError, SyntaxError, ValueError):
            tree = None
        if tree is not None:
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    node.ptk_path = str(path)
        if memo is not None:
            memo[path] = tree
        return tree

    # -- string constants (a shared remove_in) -----------------------------------
    @staticmethod
    def _string_constants(tree: ast.AST) -> Dict[str, Optional[str]]:
        """``{name: value}`` for every ``NAME = "literal"`` assignment in
        *tree*, at module or class scope -- the shared ``_REMOVE_IN`` a module
        hands each of its retirements. A name bound to two DIFFERENT strings
        maps to ``None``: which one a call means is not readable statically."""
        found: Dict[str, Optional[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    prior = found.get(target.id, value.value)
                    found[target.id] = value.value if prior == value.value else None
        return found

    @staticmethod
    def _merge_string_constants(
        dst: Dict[str, Optional[str]], src: Dict[str, Optional[str]]
    ) -> None:
        """Fold *src* into *dst* under :meth:`_string_constants`' rule: a name
        the two bind to different strings becomes unreadable, so a module's
        constants merged with its bases' never resolve a ``remove_in`` to a
        version its own file never said."""
        for name, value in src.items():
            if name in dst and dst[name] != value:
                dst[name] = None
            else:
                dst.setdefault(name, value)

    @staticmethod
    def _string_value(node: ast.AST, constants: Dict[str, Optional[str]]) -> str:
        """The string *node* stands for: a literal, or a name or attribute
        (``_REMOVE_IN``, ``cls._REMOVE_IN``) bound to one; ``""`` when unreadable."""
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, str) else ""
        if isinstance(node, ast.Name):
            return constants.get(node.id) or ""
        if isinstance(node, ast.Attribute):
            return constants.get(node.attr) or ""
        return ""

    # -- decorators ----------------------------------------------------------------
    @staticmethod
    def _decorator_path(dec: ast.expr) -> str:
        """The dotted source spelling of a decorator (``ptk.Deprecation.symbol``):
        every retirement here is written through a class namespace, so only the
        dotted form tells ``@Deprecation.symbol`` from any other ``symbol``."""
        if isinstance(dec, ast.Call):
            dec = dec.func
        parts: List[str] = []
        while isinstance(dec, ast.Attribute):
            parts.append(dec.attr)
            dec = dec.value
        if isinstance(dec, ast.Name):
            parts.append(dec.id)
        return ".".join(reversed(parts))

    @classmethod
    def _retiring(cls, dec: ast.expr) -> bool:
        path = cls._decorator_path(dec)
        return any(
            path == name or path.endswith(f".{name}")
            for name in cls.DEPRECATION_DECORATORS
        )

    @classmethod
    def _deprecation_of(
        cls, decorators: list, constants: Optional[Dict[str, Optional[str]]] = None
    ) -> Tuple[bool, str]:
        """``(deprecated, remove_in)`` read off a decorator list; *constants*
        resolve a ``remove_in`` spelled as a shared name."""
        for dec in decorators:
            if not cls._retiring(dec):
                continue
            remove_in = ""
            if isinstance(dec, ast.Call):
                for keyword in dec.keywords:
                    if keyword.arg == "remove_in":
                        remove_in = cls._string_value(keyword.value, constants or {})
            return True, remove_in
        return False, ""

    @classmethod
    def _since_of(
        cls, decorators: list, constants: Optional[Dict[str, Optional[str]]] = None
    ) -> str:
        """The ``since`` date on the deprecation decorator in *decorators*
        (``""`` when there is none): the calendar half of the window."""
        for dec in decorators:
            if not cls._retiring(dec):
                continue
            if isinstance(dec, ast.Call):
                for keyword in dec.keywords:
                    if keyword.arg == "since":
                        return cls._string_value(keyword.value, constants or {})
            return ""
        return ""

    @classmethod
    def _stamp_since(cls, record, decorators: list, constants=None):
        """*record* with its retirement's ``since`` as a plain attribute (never a
        field: the mapping is ``asdict`` of the fields). Returns *record*."""
        if getattr(record, "deprecated", False):
            record.since = cls._since_of(decorators, constants)
        return record

    @staticmethod
    def _stamp_source(record, node: ast.AST, path: Union[str, Path]):
        """*record* with its whole docstring and defining file as plain
        attributes (``doc``, ``path``), which ``describe`` reads. Returns *record*."""
        record.doc = ast.get_docstring(node) or ""
        record.path = str(path)
        return record

    @classmethod
    def _decorator_kinds(
        cls, node: ast.FunctionDef, constants: Optional[Dict[str, Optional[str]]] = None
    ) -> Tuple[str, bool, str]:
        """``(kind, deprecated, remove_in)`` of a method: its kind is
        ``method``, ``staticmethod``, ``classmethod`` or ``property``."""
        kind = "method"
        for dec in node.decorator_list:
            name = (
                dec.id
                if isinstance(dec, ast.Name)
                else dec.attr
                if isinstance(dec, ast.Attribute)
                else (
                    dec.func.id
                    if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name)
                    else ""
                )
            )
            if name in ("staticmethod", "classmethod", "property"):
                kind = name
        deprecated, remove_in = cls._deprecation_of(node.decorator_list, constants)
        return kind, deprecated, remove_in

    @staticmethod
    def _is_public(name: str) -> bool:
        return not name.startswith("_")

    @staticmethod
    def _is_property_accessor(node: ast.FunctionDef) -> bool:
        """Whether *node* is a property's setter or deleter, which redefines a
        property its getter already records."""
        return any(
            isinstance(dec, ast.Attribute) and dec.attr in ("setter", "deleter")
            for dec in node.decorator_list
        )

    @staticmethod
    def _format_signature(node: ast.FunctionDef) -> str:
        """*node*'s signature as ``ast.unparse`` spells its annotations: an
        annotated default reads ``x: int = 1``, a bare one ``x=1``."""
        args = node.args

        def render(a: ast.arg, default: Optional[ast.expr] = None) -> str:
            text = a.arg
            if a.annotation is not None:
                text += f": {ast.unparse(a.annotation)}"
            if default is not None:
                rendered = ast.unparse(default)
                text += f" = {rendered}" if a.annotation else f"={rendered}"
            return text

        parts: List[str] = []
        posonly, regular = list(args.posonlyargs), list(args.args)
        positional = posonly + regular
        pad = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
        rendered = [render(a, d) for a, d in zip(positional, pad)]
        if posonly:
            parts.extend(rendered[: len(posonly)])
            parts.append("/")
            parts.extend(rendered[len(posonly) :])
        else:
            parts.extend(rendered)
        if args.vararg:
            parts.append("*" + render(args.vararg))
        elif args.kwonlyargs:
            parts.append("*")
        for a, d in zip(args.kwonlyargs, args.kw_defaults):
            parts.append(render(a, d))
        if args.kwarg:
            parts.append("**" + render(args.kwarg))
        sig = "(" + ", ".join(parts) + ")"
        if node.returns is not None:
            sig += f" -> {ast.unparse(node.returns)}"
        return sig

    # -- classes ---------------------------------------------------------------------
    @classmethod
    def _own_members(
        cls,
        node: ast.ClassDef,
        owner: str,
        constants: Optional[Dict[str, Optional[str]]] = None,
    ) -> List[SymbolRecord]:
        """The public methods declared directly in *node*'s body."""
        out: List[SymbolRecord] = []
        path = getattr(node, "ptk_path", "")
        for member in node.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not cls._is_public(member.name) or cls._is_property_accessor(member):
                continue
            kind, deprecated, remove_in = cls._decorator_kinds(member, constants)
            doc = ast.get_docstring(member)
            record = SymbolRecord(
                name=member.name,
                qualname=f"{owner}.{member.name}",
                kind=kind,
                signature=cls._format_signature(member),
                summary=cls.summary(doc, cls.SUMMARY_MAX),
                line=member.lineno,
                deprecated=deprecated,
                remove_in=remove_in,
            )
            cls._stamp_source(record, member, path)
            out.append(cls._stamp_since(record, member.decorator_list, constants))
        return out

    @classmethod
    def _class_members(
        cls,
        node: ast.ClassDef,
        owner: str,
        local_classes: Dict[str, ast.ClassDef],
        _seen: Optional[set] = None,
        _visited: Optional[set] = None,
        constants: Optional[Dict[str, Optional[str]]] = None,
    ) -> List[SymbolRecord]:
        """The public members of *node*, those it inherits from PRIVATE bases
        declared in its module or imported from a sibling module of its
        package included (:meth:`_imported_private_classes`).

        A public base is documented under its own entry, and a base from
        another package cannot be resolved from this tree. Bases are walked
        left to right, depth first, the first definition winning (Python's MRO
        for the one-base-per-capability shape used here). ``_visited`` guards
        a base graph that loops (a half-written file), which Python could
        never run but a walk of the disk can meet.
        """
        seen = set() if _seen is None else _seen
        visited = set() if _visited is None else _visited
        visited.add(node.name)
        out: List[SymbolRecord] = []
        for rec in cls._own_members(node, owner, constants):
            if rec.name not in seen:
                seen.add(rec.name)
                out.append(rec)
        for base in node.bases:
            name = base.id if isinstance(base, ast.Name) else None
            if (
                name is None
                or cls._is_public(name)
                or name not in local_classes
                or name in visited
            ):
                continue
            out.extend(
                cls._class_members(
                    local_classes[name], owner, local_classes, seen, visited, constants
                )
            )
        return out

    @classmethod
    def _import_target(
        cls,
        node: ast.ImportFrom,
        path: Path,
        pkg_source_root: Path,
        memo: Optional[_Parses] = None,
    ) -> Optional[Path]:
        """The ``.py`` file an ``ImportFrom`` names, when it is a module of the
        package being walked (absolute ``pkg.a.b`` or relative ``.b``). The
        file looked for is noted in *memo*, found or not: one written later
        changes what the module inherits."""
        parts = node.module.split(".") if node.module else []
        if node.level:
            base = path.parent
            for _ in range(node.level - 1):
                base = base.parent
        else:
            if not parts or parts[0] != pkg_source_root.name:
                return None
            base, parts = pkg_source_root, parts[1:]
        if not parts:
            return None
        target = base.joinpath(*parts).with_suffix(".py")
        if memo is not None:
            memo.ask(target)
        return target if target.is_file() else None

    @classmethod
    def _imported_private_classes(
        cls,
        tree: ast.Module,
        path: Path,
        pkg_source_root: Path,
        _visiting: Optional[set] = None,
        _constants: Optional[Dict[str, Optional[str]]] = None,
        memo: Optional[_Parses] = None,
    ) -> Dict[str, ast.ClassDef]:
        """The private classes *tree* imports from sibling modules of its own
        package, keyed by the name bound here: what lets a public class resolve
        the members it inherits from a private mixin in another file.

        Transitive (a mixin's own imported private bases come along), same
        package only. ``_visiting`` is the import PATH down to *path*, so only
        a module already on it is a cycle; a module two routes reach is walked
        on each (parsed once per read, *memo*). ``_constants`` collects the
        string constants of every module that CONTRIBUTES a base, so an
        inherited retirement resolves the ``_REMOVE_IN`` its own file declares.
        """
        visiting = {path} if _visiting is None else _visiting
        found: Dict[str, ast.ClassDef] = {}
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            wanted = {
                (a.asname or a.name): a.name
                for a in node.names
                if not cls._is_public(a.name)
            }
            if not wanted:
                continue
            target = cls._import_target(node, path, pkg_source_root, memo)
            if target is None or target in visiting:
                continue
            module = cls._parse(target, memo)
            if module is None:
                continue
            classes = {n.name: n for n in module.body if isinstance(n, ast.ClassDef)}
            adopted = {b: classes[s] for b, s in wanted.items() if s in classes}
            if adopted and _constants is not None:
                cls._merge_string_constants(_constants, cls._string_constants(module))
            transitive = cls._imported_private_classes(
                module, target, pkg_source_root, visiting | {target}, _constants, memo
            )
            found.update(adopted)
            for name, klass in transitive.items():
                found.setdefault(name, klass)
        return found

    @staticmethod
    def _qualify_attribute_base(base: str, module_aliases: Dict[str, str]) -> str:
        """``ptk.Base`` -> ``pythontk.Base`` when ``ptk`` is an ``import ... as`` alias."""
        head, sep, rest = base.partition(".")
        if sep and head in module_aliases:
            return f"{module_aliases[head]}{sep}{rest}"
        return base

    # -- modules and packages ------------------------------------------------------
    @classmethod
    def _walk_module(
        cls, path: Path, pkg_source_root: Path, memo: Optional[_Parses] = None
    ) -> Optional[ModuleSurface]:
        """One module's public surface; ``None`` when it has none (or does not
        parse). *memo* is the read's parses (none: this walk's own)."""
        memo = _Parses() if memo is None else memo
        tree = cls._parse(path, memo)
        if tree is None:
            return None
        relpath = path.relative_to(pkg_source_root).as_posix()
        # ``from pythontk import ShotSequencer as _Core`` + ``class ShotSequencer(_Core)``:
        # the base is recorded under its real name. Absolute imports only -- a
        # relative alias names this package's own private class.
        aliased_bases = {
            alias.asname: f"{imp.module}.{alias.name}"
            for imp in tree.body
            if isinstance(imp, ast.ImportFrom) and imp.level == 0 and imp.module
            for alias in imp.names
            if alias.asname
        }
        # ``import pythontk as ptk`` + ``class X(ptk.Base)``: recorded as ``pythontk.Base``.
        module_aliases = {
            alias.asname: alias.name
            for imp in tree.body
            if isinstance(imp, ast.Import)
            for alias in imp.names
            if alias.asname
        }
        funcs: List[SymbolRecord] = []
        classes: List[ClassSurface] = []
        constants: List[SymbolRecord] = []
        # The string constants a retirement may name, from this module and
        # from every module it takes a private base from; a name two of them
        # bind differently stays unreadable rather than resolving to a guess.
        string_constants: Dict[str, Optional[str]] = {}
        local_classes = cls._imported_private_classes(
            tree, path, pkg_source_root, _constants=string_constants, memo=memo
        )
        local_classes.update(
            {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        )
        cls._merge_string_constants(string_constants, cls._string_constants(tree))

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not cls._is_public(node.name):
                    continue
                deprecated, remove_in = cls._deprecation_of(
                    node.decorator_list, string_constants
                )
                record = SymbolRecord(
                    name=node.name,
                    qualname=node.name,
                    kind="function",
                    signature=cls._format_signature(node),
                    summary=cls.summary(ast.get_docstring(node), cls.SUMMARY_MAX),
                    line=node.lineno,
                    deprecated=deprecated,
                    remove_in=remove_in,
                )
                cls._stamp_source(record, node, path)
                funcs.append(
                    cls._stamp_since(record, node.decorator_list, string_constants)
                )
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                # ALL_CAPS only: a lowercase module binding is a runtime detail
                # (a logger, a compiled regex), and recording those would bury
                # the constants that ARE contract.
                if isinstance(node, ast.Assign):
                    targets = [t for t in node.targets if isinstance(t, ast.Name)]
                else:
                    targets = [node.target] if isinstance(node.target, ast.Name) else []
                for target in targets:
                    if not target.id.isupper() or not cls._is_public(target.id):
                        continue
                    record = SymbolRecord(
                        name=target.id,
                        qualname=target.id,
                        kind="constant",
                        signature="",
                        summary="",
                        line=node.lineno,
                    )
                    record.doc, record.path = "", str(path)
                    constants.append(record)
            elif isinstance(node, ast.ClassDef):
                if not cls._is_public(node.name) or node.name.startswith(
                    cls.GENERATED_CLASS_PREFIXES
                ):
                    continue
                members = cls._class_members(
                    node, node.name, local_classes, constants=string_constants
                )
                bases = [
                    aliased_bases.get(b.id, b.id)
                    if isinstance(b, ast.Name)
                    else cls._qualify_attribute_base(ast.unparse(b), module_aliases)
                    for b in node.bases
                ]
                deprecated, remove_in = cls._deprecation_of(
                    node.decorator_list, string_constants
                )
                entry = ClassSurface(
                    name=node.name,
                    summary=cls.summary(ast.get_docstring(node), cls.SUMMARY_MAX),
                    line=node.lineno,
                    bases=bases,
                    members=members,
                    deprecated=deprecated,
                    remove_in=remove_in,
                )
                cls._stamp_source(entry, node, path)
                classes.append(
                    cls._stamp_since(entry, node.decorator_list, string_constants)
                )

        if not funcs and not classes and not constants:
            return None
        doc = ast.get_docstring(tree)
        return ModuleSurface(
            relpath=relpath,
            summary=cls.summary(doc, cls.SUMMARY_MAX),
            functions=funcs,
            classes=classes,
            constants=constants,
        )

    @classmethod
    def _walk_kept(
        cls, path: Path, root: Path, memo: _Parses
    ) -> Optional[ModuleSurface]:
        """:meth:`_walk_module`, or the surface an earlier walk of *path* left
        while every file that walk read or looked for is as it was
        (:attr:`_WALKED`). A walk that met a file it could not read is not kept."""
        key = (str(path), str(root))
        kept = cls._WALKED.get(key)
        if kept is not None and all(_Parses.stamp(p) == s for p, s in kept[0]):
            return kept[1]
        memo.asked = set()
        unread = len(memo.unread)
        module = cls._walk_module(path, root, memo)
        if len(memo.unread) == unread:
            asked = tuple((p, memo.stamps.get(p)) for p in sorted(memo.asked, key=str))
            cls._WALKED[key] = (asked, module)
        return module

    @classmethod
    def _walk_package(
        cls,
        source_root: Path,
        label: str,
        files: Optional[List[Path]] = None,
        keep: bool = True,
    ) -> Tuple[PackageSurface, List[Path]]:
        """The surface of the package whose source folder is *source_root*,
        recorded under *label* (its ``source_root``), from *files* (default:
        :meth:`_iter_py_files`), and the files that could not be read: a read
        that met one is no surface to keep. *keep* walks again only the
        modules an edit reached (:meth:`_walk_kept`); False walks every one
        and remembers none."""
        memo = _Parses()
        walk = cls._walk_kept if keep else cls._walk_module
        paths = cls._iter_py_files(source_root) if files is None else files
        modules = [m for m in (walk(p, source_root, memo) for p in paths) if m]
        surface = PackageSurface(
            name=source_root.name, source_root=label, modules=modules
        )
        return surface, memo.unread

    @classmethod
    def _retirements(cls, source_root: Path) -> List[Tuple[str, str, str, str]]:
        """Every sub-symbol retirement in the package source, ``(relpath, what,
        remove_in, since)``: a ``Deprecation.<parameter|values|warn|attributes|
        class_attribute>(...)`` call that names ``remove_in``, anywhere in a
        module. One whose ``remove_in`` cannot be read (a helper's result, a
        name bound to two strings) carries ``""``. One row per distinct
        retirement."""
        out: List[Tuple[str, str, str, str]] = []
        for path in cls._iter_py_files(source_root):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except (SyntaxError, ValueError):
                continue
            relpath = path.relative_to(source_root).as_posix()
            constants = cls._string_constants(tree)
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in cls.RETIREMENT_CALLS
                ):
                    continue
                owner = cls._decorator_path(node.func.value)
                if owner != "Deprecation" and not owner.endswith(".Deprecation"):
                    continue
                given = next(
                    (kw.value for kw in node.keywords if kw.arg == "remove_in"), None
                )
                if given is None:
                    continue
                what = ast.unparse(node.args[0]) if node.args else ""
                since = next(
                    (
                        cls._string_value(kw.value, constants)
                        for kw in node.keywords
                        if kw.arg == "since"
                    ),
                    "",
                )
                row = (
                    relpath,
                    f"Deprecation.{node.func.attr}({what[:60]})",
                    cls._string_value(given, constants),
                    since,
                )
                if row not in out:
                    out.append(row)
        return out

    @classmethod
    def _importable(cls, source_root: Path, relpath: str) -> bool:
        """Whether the module at *relpath* imports by its dotted path: every
        folder above it is a package (``__init__.py``), so a plug-in another
        application loads (a ``plugin_src`` tree) is not."""
        parts = relpath.split("/")[:-1]
        return all(
            (source_root.joinpath(*parts[: i + 1]) / "__init__.py").is_file()
            for i in range(len(parts))
        )

    @staticmethod
    def _source_stamp(
        source_root: Path, files: Iterable[Path]
    ) -> Tuple[Tuple[str, int, int], ...]:
        """What says a package's source changed: each file's path, time and size."""
        stamp = []
        for path in files:
            try:
                stat = os.stat(path)
            except OSError:
                continue
            stamp.append(
                (
                    path.relative_to(source_root).as_posix(),
                    stat.st_mtime_ns,
                    stat.st_size,
                )
            )
        return tuple(stamp)
