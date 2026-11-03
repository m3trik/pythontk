# !/usr/bin/python
# coding=utf-8
"""OperationProvider -- one package's operations: its declarations, its committed catalog, their upkeep.

A provider is a Python package that declares operations in one module (a tuple
named ``OPERATIONS``, by default ``<package>/operations.py``) and commits the
catalog built from them beside it (``operations.json``), which every front end
reads without importing the package. This class owns that cycle for one
package, so a codebase is added and kept in line by operations, not by hand:

* :meth:`OperationProvider.init` (``operations.init``) onboards a package: its
  declaration module, its ``pythontk.operations`` entry point in
  ``pyproject.toml`` (registered at the package's next install) and its first
  catalog.
* :meth:`OperationProvider.write_catalog` (``operations.catalog``) rewrites a
  catalog from its declarations.
* :meth:`OperationProvider.check` (``operations.check``) holds every provider:
  its catalog current, and its operations within the rules a machine can test
  (a change that can plan, typed arguments, twins that agree), each problem an
  issue whose fix, where there is one, is an operation call.

Discovering and running operations stays with :class:`~pythontk.OperationCatalog`,
which reads what this class writes.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from pythontk.core_utils.api_surface.api_surface import ApiSurface
from pythontk.core_utils.operation_catalog.operation import Operation, OperationCatalogDocument
from pythontk.core_utils.package_manager import PackageManager


class _OperationProviderInternal:
    """Internal helpers for :class:`OperationProvider`: references and the rules."""

    #: The attribute a declaration module lists its operations in.
    DECLARATIONS = "OPERATIONS"
    #: A package's declaration module, unless a reference names another.
    MODULE = "operations"
    #: The keys that type a schema; an argument with none of them is untyped.
    _TYPING_KEYS = ("type", "anyOf", "oneOf", "allOf", "$ref", "enum", "const")
    #: The schema keys that describe an argument without typing it.
    _PROSE_KEYS = ("description", "default", "title", "examples")
    #: The pyproject.toml table a provider registers its module in.
    _TABLE = re.compile(r"^\[project\.entry-points\.[\"']pythontk\.operations[\"']\][ \t]*$", re.M)
    #: The declaration module a package starts with.
    _TEMPLATE = '''"""{package}'s operations: the actions agents and front ends run by id
(``python -m pythontk ops``).

Each entry declares one of {package}'s functions under an id and an effect
(see pythontk's ``Operation``); add ``from pythontk import Operation`` with the
first one. After changing one, or its function's signature or docstring,
rewrite the catalog beside this file:
``python -m pythontk ops apply operations.catalog module={package}.operations``.
"""

OPERATIONS = ()
'''

    # -- references --------------------------------------------------------------
    @classmethod
    def _split_reference(cls, value: str) -> str:
        """The module of a provider reference: ``"pkg.ops"`` or
        ``"pkg.ops:OPERATIONS"`` (the one attribute a declaration module
        lists its operations in, so one module has one catalog).

        Raises:
            ValueError: It names another attribute.
        """
        module, _, attr = value.partition(":")
        if attr.strip() and attr.strip() != cls.DECLARATIONS:
            raise ValueError(
                f"provider {value!r}: a declaration module lists its operations in "
                f"{cls.DECLARATIONS!r}, not {attr.strip()!r}"
            )
        return module.strip()

    @staticmethod
    def _module_location(module: str) -> Path:
        """The file or folder *module* is, found without importing it (in a
        namespace package, the search location that holds it; a repo folder
        on ``sys.path`` above its own package, which imports as an empty
        namespace, is looked through to the package inside it)."""
        parts = module.split(".")
        spec = importlib.util.find_spec(parts[0])
        if spec is None:
            raise ModuleNotFoundError(f"no module named {parts[0]!r}", name=parts[0])
        if not spec.submodule_search_locations:
            return Path(spec.origin)
        locations = [Path(p) for p in spec.submodule_search_locations]
        locations += [
            base / parts[0] for base in locations if (base / parts[0] / "__init__.py").is_file()
        ]
        for base in locations:
            candidate = base.joinpath(*parts[1:])
            if candidate.is_dir() or candidate.with_suffix(".py").is_file():
                return candidate
        for base in locations:
            if base.joinpath(*parts[1:]).with_suffix(".json").is_file():
                return base.joinpath(*parts[1:])  # a catalog with no module (another language's)
        return locations[0].joinpath(*parts[1:])

    @classmethod
    def _package_dir(cls, package: str) -> Path:
        """The folder of *package* (a top-level package), found without importing it.

        Raises:
            ValueError: No package of that name imports here (the reason
                named when a folder on ``sys.path`` hides it).
        """
        hidden = cls.import_problem(package)
        if hidden:
            raise ValueError(hidden)
        return ApiSurface.locate(package)

    @classmethod
    def _fresh_text(cls, module: str, provider: Optional[str]) -> str:
        """*module*'s catalog as its declarations make it now (imports them)."""
        document = cls.document(cls.declarations(module), provider or module.split(".")[0])
        return cls.render(document)

    # -- the rules ---------------------------------------------------------------
    @classmethod
    def _untyped(cls, schema: Mapping[str, Any]) -> bool:
        return not any(key in schema for key in cls._TYPING_KEYS)

    @classmethod
    def _shape(cls, schema: Mapping[str, Any]) -> Dict[str, Any]:
        """*schema* without the words that describe it: what a value must be."""
        return {k: v for k, v in schema.items() if k not in cls._PROSE_KEYS}

    @classmethod
    def _entry_findings(cls, entry: Mapping[str, Any]) -> List[Tuple[str, str]]:
        """``(code, message)`` for each rule one catalog entry breaks."""
        findings = []
        if entry["effect"] == Operation.CODE:
            if entry.get("dry_run"):
                findings.append(
                    (
                        "operations.code_plans",
                        f"{entry['id']} runs code, which has no plan: its function takes no "
                        "ctx (a plan would only echo the code)",
                    )
                )
        elif entry["effect"] != Operation.READ and not entry.get("dry_run"):
            how = (
                "give its function a ctx and guard each change with `if ctx is None or "
                "ctx.should_...(...)`"
                if entry.get("target")  # a Python function
                else "its method must take the operation context and ask it before each change"
            )
            findings.append(
                ("operations.no_dry_run", f"{entry['id']} changes {entry['effect']} but cannot plan: {how}")
            )
        for name, schema in (entry.get("args") or {}).get("properties", {}).items():
            if cls._untyped(schema):
                findings.append(
                    (
                        "operations.untyped",
                        f"{entry['id']}: argument {name!r} has no type, so nothing checks "
                        "what a caller passes; type it with a plain value (a name, a path, "
                        "an option)",
                    )
                )
        return findings

    @classmethod
    def _twin_findings(cls, entries: Iterable[Mapping[str, Any]]) -> List[Tuple[str, str]]:
        """``(code, message)`` for each pair of twins that disagree: operations
        two hosts run under one action (``maya.x.y``, ``blender.x.y``) share
        their effect, and the arguments they share keep one name and type; a
        host may add arguments of its own (a ``.blend`` holds several scenes)."""
        by_action: Dict[str, List[Mapping[str, Any]]] = {}
        for entry in entries:
            if entry.get("host"):
                by_action.setdefault(entry["id"].split(".", 1)[1], []).append(entry)
        findings = []
        for _, twins in sorted(by_action.items()):
            first, *others = sorted(twins, key=lambda e: e["id"])
            for other in others:
                diffs = []
                if first["effect"] != other["effect"]:
                    diffs.append(f"effect {first['effect']} vs {other['effect']}")
                ours = (first.get("args") or {}).get("properties", {})
                theirs = (other.get("args") or {}).get("properties", {})
                if not (set(ours) <= set(theirs) or set(theirs) <= set(ours)):
                    diffs.append(
                        f"arguments {', '.join(sorted(set(ours) - set(theirs)))} ({first['host']}) "
                        f"vs {', '.join(sorted(set(theirs) - set(ours)))} ({other['host']})"
                    )
                diffs += [
                    f"argument {name!r} takes another type"
                    for name in sorted(set(ours) & set(theirs))
                    if cls._shape(ours[name]) != cls._shape(theirs[name])
                ]
                if diffs:
                    findings.append(
                        (
                            "operations.twins",
                            f"twins {first['id']} and {other['id']} differ: {'; '.join(diffs)} "
                            "(a host may add arguments, never rename or retype a shared one)",
                        )
                    )
        return findings

    @classmethod
    def _document_findings(cls, path: Path) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
        """The findings and entries of a catalog another language writes: its
        shape first (nothing builds it here to compare), then the rules."""
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        errors = OperationCatalogDocument.validate(document).errors
        if errors:
            return [("operations.invalid", f"{path} is not a valid catalog: {'; '.join(errors)}")], []
        entries = list(document["operations"])
        return [f for entry in entries for f in cls._entry_findings(entry)], entries

    @classmethod
    def _audit(
        cls, reference: str, provider: Optional[str] = None
    ) -> Tuple[str, List[Tuple[str, str]], List[Dict[str, Any]]]:
        """``(where, findings, entries)`` for one provider: a declaration
        module's catalog against its declarations, or the document of a
        provider another language writes (a catalog file, or a registered
        module path holding only its ``.json``); then the rules."""
        if cls.is_catalog_file(reference):
            return (str(reference),) + cls._document_findings(Path(reference))
        module = cls._split_reference(reference)
        hidden = cls.import_problem(module.split(".")[0])
        if hidden:
            return module, [("operations.hidden", hidden)], []
        location = cls._module_location(module)
        if not (location.is_dir() or location.with_suffix(".py").is_file()) and cls.catalog_path(module).is_file():
            return (module,) + cls._document_findings(cls.catalog_path(module))
        text = cls._fresh_text(module, provider)
        fresh = json.loads(text)["operations"]
        path = cls.catalog_path(module)
        fix = f"run `python -m pythontk ops apply operations.catalog module={module}`"
        findings: List[Tuple[str, str]] = []
        if not path.is_file():
            findings.append(("operations.stale", f"{path} is missing: {fix}"))
        else:
            committed_text = path.read_text(encoding="utf-8")
            if committed_text != text:
                committed = {e["id"]: e for e in json.loads(committed_text).get("operations", [])}
                now = {e["id"]: e for e in fresh}
                changed = sorted(i for i in set(committed) | set(now) if committed.get(i) != now.get(i))
                findings.append(
                    ("operations.stale", f"{path} is stale ({', '.join(changed) or 'formatting'}): {fix}")
                )
        for entry in fresh:
            findings += cls._entry_findings(entry)
        return module, findings, fresh

    @classmethod
    def _with_entry_point(cls, text: str, package: str, module: str) -> str:
        """*text* (a ``pyproject.toml``) registering *module* under the
        ``pythontk.operations`` entry points, joining that table if it is there."""
        line = f'{package} = "{module}"'
        table = cls._TABLE.search(text)
        if table is None:
            return text.rstrip("\n") + f'\n\n[project.entry-points."pythontk.operations"]\n{line}\n'
        start = table.end()
        following = re.compile(r"^\[", re.M).search(text, start)
        body = text[start : following.start() if following else len(text)]
        if re.search(rf"^[ \t]*{re.escape(package)}[ \t]*=", body, re.M):
            return text  # registered already
        end = start + len(body.rstrip("\n"))
        return text[:end] + "\n" + line + text[end:]


class OperationProvider(_OperationProviderInternal):
    """One package's operations: build, onboard and check (see the module docstring).

    Every method takes the package, or a declaration module as ``"pkg.ops"``
    (``"pkg.ops:OPERATIONS"`` too), so each one is also an operation.
    """

    #: The entry-point group a provider registers its declaration module under.
    ENTRY_POINT_GROUP = ApiSurface.PROVIDER_GROUP

    # -- building: a provider's declarations -> its committed document --------
    @classmethod
    def module_of(cls, reference: str) -> str:
        """The declaration module a provider reference names.

        Raises:
            ValueError: It names an attribute other than ``OPERATIONS``.
        """
        return cls._split_reference(reference)

    @classmethod
    def declarations(cls, module: str) -> Tuple[Operation, ...]:
        """The :class:`Operation` records a declaration module lists (imports it).

        Raises:
            TypeError: Its ``OPERATIONS`` is not a sequence of :class:`Operation`.
        """
        module = cls._split_reference(module)
        ops = getattr(importlib.import_module(module), cls.DECLARATIONS)
        if not all(isinstance(op, Operation) for op in ops):
            raise TypeError(f"{module}.{cls.DECLARATIONS} must hold Operation records")
        return tuple(ops)

    @classmethod
    def document(cls, operations: Iterable[Operation], provider: str) -> Dict[str, Any]:
        """The catalog document for *operations* (each described; imports targets).

        Raises:
            ValueError: An id is declared twice.
        """
        ordered = sorted(operations, key=lambda op: op.id)
        ids = [op.id for op in ordered]
        repeated = sorted({i for i in ids if ids.count(i) > 1})
        if repeated:
            raise ValueError(f"{provider} declares {repeated} more than once")
        return OperationCatalogDocument(
            format=OperationCatalogDocument.FORMAT,
            provider=provider,
            operations=[op.entry() for op in ordered],
        ).to_dict()

    @classmethod
    def catalog_path(cls, module: str) -> Path:
        """Where *module*'s committed catalog lives, found without importing it:
        beside its source, same stem, ``.json`` (``mayatk/operations.json``)."""
        location = cls._module_location(cls._split_reference(module))
        if location.is_dir():
            return location / "operations.json"
        return location.with_suffix(".json")

    @staticmethod
    def render(document: Mapping[str, Any]) -> str:
        """*document* as its committed text: stable (a diff is the change) and
        strict JSON (no NaN or Infinity, which other languages refuse)."""
        return json.dumps(document, indent=2, allow_nan=False) + "\n"

    @classmethod
    def write_catalog(cls, module: str, provider: Optional[str] = None, ctx=None) -> str:
        """Rewrite *module*'s committed catalog from its declarations, when it
        changed; returns its path.

        Parameters:
            module: The declaration module (``mayatk.operations``).
            provider: The provider's name in the catalog; default: the module's
                top package.
            ctx: The operation context: the catalog is written only when it
                allows (a dry run lists the write).
        """
        text = cls._fresh_text(cls._split_reference(module), provider)
        path = cls.catalog_path(module)
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current != text and (ctx is None or ctx.should_write(path, "rewrite the catalog from its declarations")):
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        return str(path)

    @staticmethod
    def is_catalog_file(reference: Any) -> bool:
        """Whether a provider reference names a catalog file (a ``Path``, or
        text ending ``.json``), not a declaration module."""
        return isinstance(reference, Path) or str(reference).lower().endswith(".json")

    @classmethod
    def import_problem(cls, package: str) -> Optional[str]:
        """Why *package* does not import as itself here, or ``None``: a folder
        named after it on ``sys.path`` (its repository, above the package
        folder) makes it an empty namespace, so every name in it is missing."""
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            return None
        if spec is None or spec.origin not in (None, "namespace") or not spec.submodule_search_locations:
            return None
        for base in spec.submodule_search_locations:
            if (Path(base) / package / "__init__.py").is_file():
                return (
                    f"{package} imports as an empty namespace here: {base}, a folder of that "
                    f"name under a sys.path entry, hides the package inside it. Put {base} "
                    "itself on sys.path: install it editable in compat mode (pip install "
                    f"--config-settings editable_mode=compat -e {base}) or add it to PYTHONPATH"
                )
        return None

    # -- checking -------------------------------------------------------------------
    @classmethod
    def problems(cls, module: str, provider: Optional[str] = None) -> List[str]:
        """Every way *module*'s provider breaks the rules: a missing or stale
        catalog (with the command that fixes it), a change that cannot plan,
        an untyped argument. Empty when it keeps them."""
        return [message for _, message in cls._audit(module, provider)[1]]

    @classmethod
    def check(cls, modules: Optional[List[str]] = None, ctx=None) -> List[str]:
        """Check providers: each catalog current and within the rules, and
        twins across them agreeing.

        Parameters:
            modules: The providers to check: declaration modules, or catalog
                files another language writes; default: every provider
                registered under the ``pythontk.operations`` entry points.
            ctx: The operation context: each problem is also an error issue,
                a stale catalog's fixed by ``operations.catalog``.

        Returns:
            One line per problem; empty when every provider keeps the rules.
        """
        if modules is None:
            modules = [ep.value for ep in PackageManager.entry_points(cls.ENTRY_POINT_GROUP)]
        located: List[Tuple[str, str, str]] = []  # (module, code, message)
        entries: List[Dict[str, Any]] = []
        for reference in dict.fromkeys(modules):
            try:
                where, findings, fresh = cls._audit(reference)
            except Exception as exc:
                where, fresh = str(reference), []
                findings = [("operations.unloadable", f"{reference}: {type(exc).__name__}: {exc}")]
            located += [(where, code, message) for code, message in findings]
            entries += fresh
        located += [("", code, message) for code, message in cls._twin_findings(entries)]
        if ctx is not None:
            for module, code, message in located:
                fix = {"op": "operations.catalog", "args": {"module": module}} if code == "operations.stale" else None
                ctx.issue(code, message, "error", module or None, fix)
        return [message for _, _, message in located]

    # -- onboarding -----------------------------------------------------------------
    @classmethod
    def init(cls, package: str, ctx=None) -> Dict[str, str]:
        """Onboard *package*: its declaration module, its entry point and its
        first catalog, each made only where it is missing (running it again
        changes nothing).

        The entry point registers when the package is next installed
        (``pip install -e <project>``): until then ``--provider
        <package>.operations`` reaches it.

        Parameters:
            package: The top-level package (``mypkg``).
            ctx: The operation context: in a dry run each step is listed, not made.

        Returns:
            The paths of its declaration module, ``pyproject.toml`` and
            catalog, and the install command that registers it (``reinstall``).

        Raises:
            ValueError: *package* is not importable here, or its project
                folder holds no ``pyproject.toml``.
        """
        location = cls._package_dir(package)
        # Its own project: the folder above it, or above a src/ layout; never
        # an unrelated project file further up.
        project = next((d for d in list(location.parents)[:2] if (d / "pyproject.toml").is_file()), None)
        if project is None:
            raise ValueError(
                f"no pyproject.toml beside {location} (in its project folder, or the one above "
                f"a src/ layout): {package}'s operations are registered there"
            )
        module = f"{package}.{cls.MODULE}"
        declarations = location / f"{cls.MODULE}.py"
        if not declarations.is_file() and (
            ctx is None or ctx.should_create(declarations, "the declaration module")
        ):
            with open(declarations, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(cls._TEMPLATE.format(package=package))
        toml = project / "pyproject.toml"
        raw = toml.read_bytes().decode("utf-8")
        newline = "\r\n" if "\r\n" in raw else "\n"
        text = raw.replace("\r\n", "\n")
        joined = cls._with_entry_point(text, package, module)
        if joined != text and (
            ctx is None or ctx.should_change(toml, f"register {module} under the pythontk.operations entry points")
        ):
            toml.write_bytes(joined.replace("\n", newline).encode("utf-8"))
        catalog = cls.catalog_path(module)
        if declarations.is_file():
            importlib.invalidate_caches()
            cls.write_catalog(module, package, ctx)
        elif ctx is not None:  # a dry run: the module it would build from is not there yet
            ctx.should_write(catalog, "the first catalog")
        return {
            "operations": str(declarations),
            "pyproject": str(toml),
            "catalog": str(catalog),
            "reinstall": f"pip install -e {project}",
        }

