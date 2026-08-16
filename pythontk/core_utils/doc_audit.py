# !/usr/bin/python
# coding=utf-8
"""Audit markdown code examples against the live package surface.

Documentation examples rot silently: a rename or kwarg change ships, every
test stays green, and the README keeps teaching the old API. This module is
the rot gate. Like ``python -m pythontk`` it reads the *live* objects, so
what it validates is exactly what an installed user gets.

Per fenced code block it checks that:

- the block parses (``ast.parse``);
- every attribute chain rooted at a known namespace resolves on the live
  object (``ptk.MapFactory.resolve_map_type`` -- a rename breaks this);
- every keyword argument on a resolved call binds to the callable's
  signature (a kwarg rename breaks this);
- ``from <module> import <name>`` exposes the names it claims, whenever the
  module itself imports (a wildcard form instead binds the module's exported
  names, so the rest of the block stays checkable).

What it deliberately skips -- examples are illustrations, not programs:
names the block itself defines or assigns, unknown roots (placeholder
helpers like ``expensive_operation``), call results of unknown type, and
callables whose signature is not introspectable. The audit is a tripwire
for API drift, not an executor; literal *outputs* a document claims should
be pinned by an ordinary test beside it (see ``test/test_doc_audit.py``).
"""
import ast
import importlib
import inspect
import re
from typing import Any, List, Mapping, Optional, Tuple

from pythontk.core_utils import help_mixin


class _MissingAttribute(Exception):
    """A dotted chain rooted in a known namespace failed to resolve."""


class DocAudit(help_mixin.HelpMixin):
    """Validate markdown code examples against live objects.

    ``roots`` maps the names examples use for entry-point namespaces to the
    live objects they stand for. The default covers this package's own
    convention (``import pythontk as ptk``); any repo can pass its own, so
    downstream packages can gate their READMEs with the same primitive.
    """

    @classmethod
    def default_roots(cls) -> dict:
        pkg = importlib.import_module("pythontk")
        return {"pythontk": pkg, "ptk": pkg}

    @classmethod
    def extract_code_blocks(cls, markdown: str, lang: str = "python") -> List[str]:
        """Return the contents of every fenced ``lang`` code block.

        Parameters:
            markdown: The markdown document text.
            lang: The fence language tag to match (exact, at line start).
        """
        fence = re.compile(
            rf"^```{re.escape(lang)}[ \t]*\r?\n(.*?)^```",
            re.MULTILINE | re.DOTALL,
        )
        return [m.group(1) for m in fence.finditer(markdown)]

    @classmethod
    def audit_markdown(
        cls,
        markdown: str,
        roots: Optional[Mapping[str, Any]] = None,
        lang: str = "python",
    ) -> List[str]:
        """Audit every fenced ``lang`` block; return problems, empty if clean.

        Each problem string is prefixed ``block <n>`` (1-based, in document
        order) so it can be found in the source markdown.
        """
        problems = []
        for i, code in enumerate(cls.extract_code_blocks(markdown, lang=lang), 1):
            problems.extend(f"block {i}: {p}" for p in cls.audit_code(code, roots))
        return problems

    @classmethod
    def audit_code(
        cls, code: str, roots: Optional[Mapping[str, Any]] = None
    ) -> List[str]:
        """Audit one code snippet; return problems, empty if clean."""
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return [f"syntax error: {exc}"]

        ns = dict(cls.default_roots() if roots is None else roots)
        shadowed = cls._collect_local_names(tree)
        problems = []

        # Imports extend the known world -- and importing a name that no
        # longer exists is itself the rot being hunted. A module that does
        # not import at all (optional dep) just stays unknown. Catches
        # Exception, not just ImportError: an optional-dep module can fail
        # at import time with RuntimeError/OSError/etc (a missing native
        # library, a misconfigured environment) -- that is the same "stays
        # unknown" case as a missing package, not a reason to crash the
        # whole audit over one unrelated example.
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    try:
                        module = importlib.import_module(alias.name)
                        # `import a.b` binds `a`; `import a.b as c` binds a.b.
                        ns[bound] = (
                            module
                            if alias.asname
                            else importlib.import_module(bound)
                        )
                    except Exception:
                        shadowed.add(bound)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                try:
                    mod = importlib.import_module(node.module)
                except Exception:
                    shadowed.update(a.asname or a.name for a in node.names)
                    continue
                for alias in node.names:
                    # `import *` binds a name *set*, not a name: getattr on the
                    # literal "*" can only ever invent a problem. Expand it the
                    # way the interpreter does -- `__all__` when the module
                    # defines one, else its public surface -- so the names land
                    # in `ns` and later references are actually checked instead
                    # of being skipped as unknown roots (the audit was blind on
                    # exactly the blocks that use a wildcard).
                    if alias.name == "*":
                        # Reading a live module's star surface runs the
                        # module's own code -- a lazy ``__getattr__`` raising
                        # for an uninstalled extra, a custom ``__dir__``, or an
                        # ``__all__`` that is not even iterable. This module's
                        # contract is to REPORT problems, so any of those must
                        # leave the surface unknown rather than take the whole
                        # docs gate down with a traceback.
                        try:
                            exported = getattr(mod, "__all__", None)
                            if exported is None:
                                exported = [
                                    n for n in dir(mod) if not n.startswith("_")
                                ]
                            exported = list(exported)
                        except Exception:
                            continue
                        for name in exported:
                            # A name the module advertises but cannot produce is
                            # the module's own bug, not the example's -- and on
                            # a lazily-resolved surface it usually means an
                            # optional dep, so it just stays unknown.
                            try:
                                ns[name] = getattr(mod, name)
                            except Exception:
                                continue
                        continue
                    bound = alias.asname or alias.name
                    try:
                        ns[bound] = getattr(mod, alias.name)
                    except AttributeError:
                        problems.append(
                            f"line {node.lineno}: from {node.module} "
                            f"import {alias.name} - name not found"
                        )
                        shadowed.add(bound)

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                try:
                    cls._resolve_chain(node, ns, shadowed)
                except _MissingAttribute as miss:
                    problems.append(f"line {node.lineno}: {miss}")
            elif isinstance(node, ast.Call):
                problems.extend(cls._audit_call(node, ns, shadowed))

        # Nested chains re-raise what their parent chain already reported.
        return list(dict.fromkeys(problems))

    # ------------------------------------------------------------------
    @classmethod
    def _collect_local_names(cls, tree: ast.AST) -> set:
        """Names the snippet itself binds -- never resolved as live API."""
        local = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local.add(node.name)
            elif isinstance(node, (ast.Name, ast.arg)):
                if isinstance(node, ast.arg):
                    local.add(node.arg)
                elif isinstance(node.ctx, ast.Store):
                    local.add(node.id)
        return local

    @classmethod
    def _resolve_chain(
        cls, node: ast.AST, ns: Mapping[str, Any], shadowed: set
    ) -> Optional[Tuple[Any, str]]:
        """Resolve an attribute/call chain to ``(live_object, dotted_name)``.

        Returns None when the chain leaves the known world (unknown or
        locally bound root, non-class call result). Raises
        :class:`_MissingAttribute` when a known object lacks the attribute.
        """
        if isinstance(node, ast.Name):
            if node.id in shadowed or node.id not in ns:
                return None
            return ns[node.id], node.id
        if isinstance(node, ast.Attribute):
            base = cls._resolve_chain(node.value, ns, shadowed)
            if base is None:
                return None
            obj, dotted = base
            try:
                return getattr(obj, node.attr), f"{dotted}.{node.attr}"
            except AttributeError:
                raise _MissingAttribute(f"{dotted} has no attribute '{node.attr}'")
        if isinstance(node, ast.Call):
            base = cls._resolve_chain(node.func, ns, shadowed)
            if base is None:
                return None
            obj, dotted = base
            # An instance stands in as its class: attribute lookups and
            # signature checks read the same members either way.
            if isinstance(obj, type):
                return obj, f"{dotted}()"
            return None
        return None

    @classmethod
    def _audit_call(
        cls, node: ast.Call, ns: Mapping[str, Any], shadowed: set
    ) -> List[str]:
        """Check a call's keyword arguments against the live signature."""
        try:
            base = cls._resolve_chain(node.func, ns, shadowed)
        except _MissingAttribute as miss:
            return [f"line {node.lineno}: {miss}"]
        if base is None:
            return []
        func, dotted = base
        kw_names = [k.arg for k in node.keywords if k.arg]  # skip **expansions
        if not kw_names or not callable(func):
            return []
        try:
            params = inspect.signature(func).parameters
        except (ValueError, TypeError):
            return []
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            return []
        return [
            f"line {node.lineno}: {dotted}() has no parameter '{kw}'"
            for kw in kw_names
            if kw not in params
        ]
