#!/usr/bin/python
# coding=utf-8
"""Static guard: every pythontk module must stay importable on Python 3.9.

pyproject declares ``requires-python = ">=3.9"`` and the package is consumed by
hosts that ship their own interpreter -- notably Agisoft Metashape, whose
bundled Python is 3.9. PEP 604 unions (``int | str``) and ``match`` statements
are 3.10+ *at runtime*, so they silently break those hosts.

This test parses every module's AST (no import needed, so it runs anywhere) and
enforces the 3.9 contract:

* A PEP 604 union in any *annotation* is allowed only if the module also has
  ``from __future__ import annotations`` (which makes annotations lazy strings).
* PEP 604 unions in *runtime* positions (``isinstance`` / ``issubclass`` second
  arg, or a module-level type alias) are never allowed -- the future import does
  not help there.
* ``match`` statements are never allowed.

Run with::

    python -m pytest test_py39_compat.py -v
    python test_py39_compat.py
"""
import ast
import os
import unittest

PKG_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pythontk")
MATCH_NODE = getattr(ast, "Match", ())  # ast.Match exists only on 3.10+ hosts


def _iter_py_files(root):
    for dirpath, _dirs, files in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def _contains_bitor(node):
    """True if any descendant is a ``X | Y`` BinOp (PEP 604 union shape)."""
    for n in ast.walk(node):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr):
            return True
    return False


def _has_future_annotations(tree):
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            if any(a.name == "annotations" for a in node.names):
                return True
    return False


def _annotation_nodes(tree):
    """Yield every annotation expression in the module."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            for arg in (
                list(getattr(a, "posonlyargs", []))
                + list(a.args)
                + list(a.kwonlyargs)
                + [a.vararg, a.kwarg]
            ):
                if arg is not None and arg.annotation is not None:
                    yield arg.annotation
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign) and node.annotation is not None:
            yield node.annotation


class Py39CompatTest(unittest.TestCase):
    def setUp(self):
        self.files = list(_iter_py_files(PKG_ROOT))
        self.assertTrue(self.files, f"no .py files found under {PKG_ROOT}")

    def _trees(self):
        for path in self.files:
            with open(path, "r", encoding="utf-8") as fh:
                yield path, ast.parse(fh.read(), filename=path)

    def test_pep604_annotations_require_future_import(self):
        offenders = []
        for path, tree in self._trees():
            if _has_future_annotations(tree):
                continue
            if any(_contains_bitor(a) for a in _annotation_nodes(tree)):
                offenders.append(os.path.relpath(path, PKG_ROOT))
        self.assertFalse(
            offenders,
            "PEP 604 union annotations without `from __future__ import "
            "annotations` (breaks Python 3.9):\n  " + "\n  ".join(offenders),
        )

    def test_no_match_statements(self):
        offenders = []
        for path, tree in self._trees():
            if MATCH_NODE and any(
                isinstance(n, MATCH_NODE) for n in ast.walk(tree)
            ):
                offenders.append(os.path.relpath(path, PKG_ROOT))
        self.assertFalse(
            offenders,
            "`match` statements are 3.10+ and break Python 3.9:\n  "
            + "\n  ".join(offenders),
        )

    def test_no_runtime_pep604_unions(self):
        offenders = []
        for path, tree in self._trees():
            rel = os.path.relpath(path, PKG_ROOT)
            # isinstance / issubclass second argument
            for n in ast.walk(tree):
                if (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id in ("isinstance", "issubclass")
                    and len(n.args) >= 2
                    and _contains_bitor(n.args[1])
                ):
                    offenders.append(f"{rel} (isinstance/issubclass)")
            # module-level type alias:  Name = X | Y
            for n in tree.body:
                if (
                    isinstance(n, ast.Assign)
                    and isinstance(n.value, ast.BinOp)
                    and isinstance(n.value.op, ast.BitOr)
                ):
                    offenders.append(f"{rel} (module-level union alias)")
        self.assertFalse(
            offenders,
            "runtime PEP 604 unions break Python 3.9 even with the future "
            "import:\n  " + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
