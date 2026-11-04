# !/usr/bin/python
# coding=utf-8
"""PackageSurface -- one package's public API as its source declares it.

The static shape :meth:`~pythontk.ApiSurface.read` produces: a package's
modules, each with its public functions, classes (their members, those
inherited from private bases included) and ALL_CAPS constants, every symbol a
:class:`~pythontk.SymbolRecord`. Its mapping (:meth:`PackageSurface.to_dict`,
``asdict`` of these classes) is what a read keeps in the per-user cache and
what m3trik's release checks diff, so the field ORDER below is a contract.

Two values ride a record or a class as plain attributes, never fields, so the
mapping never carries them: ``since`` (the date a retirement's notice first
shipped, read by the expiry gate) and ``doc`` / ``path`` (the whole docstring
and the defining file, read by :meth:`~pythontk.ApiSurface.describe`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Tuple

from pythontk.core_utils.symbol_record import SymbolRecord


@dataclass
class ClassSurface:
    """One public class: its bases as written and its public members."""

    name: str
    summary: str
    line: int
    bases: List[str] = field(default_factory=list)
    members: List[SymbolRecord] = field(default_factory=list)
    #: A retired class (``ptk.Git``): read off its own decorator.
    deprecated: bool = False
    remove_in: str = ""


@dataclass
class ModuleSurface:
    """One module's public surface."""

    #: The module's path inside the package source folder, POSIX style.
    relpath: str
    summary: str
    functions: List[SymbolRecord] = field(default_factory=list)
    classes: List[ClassSurface] = field(default_factory=list)
    #: Public module-level ALL_CAPS names: importable, so their removal is a
    #: breaking change like any other.
    constants: List[SymbolRecord] = field(default_factory=list)

    @property
    def module(self) -> str:
        """The dotted module path below the package (``core_utils._core_utils``;
        ``""`` for the package's own ``__init__``)."""
        parts = (
            self.relpath[: -len(".py")].split("/")
            if self.relpath.endswith(".py")
            else [self.relpath]
        )
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)


@dataclass
class PackageSurface:
    """One package's public surface, module by module (see the module docstring)."""

    # Deliberately no timestamp: a date in the mapping would make two reads of
    # an unchanged surface differ.
    name: str
    #: The package's source folder, POSIX style: relative to the root the read
    #: named (a monorepo root), else absolute.
    source_root: str
    modules: List[ModuleSurface] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """The mapping: :func:`dataclasses.asdict`, with the
        ``remove_in`` / class ``deprecated`` keys that say nothing left out
        (absent reads back as the default, so the mapping grows only where a
        retirement was recorded)."""
        return self._prune(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PackageSurface":
        """A surface from its mapping (:meth:`to_dict`)."""
        modules = []
        for m in data.get("modules", []):
            classes = [
                ClassSurface(
                    name=c["name"],
                    summary=c.get("summary", ""),
                    line=c.get("line", 0),
                    bases=list(c.get("bases", [])),
                    members=[SymbolRecord(**r) for r in c.get("members", [])],
                    deprecated=c.get("deprecated", False),
                    remove_in=c.get("remove_in", ""),
                )
                for c in m.get("classes", [])
            ]
            modules.append(
                ModuleSurface(
                    relpath=m["relpath"],
                    summary=m.get("summary", ""),
                    functions=[SymbolRecord(**f) for f in m.get("functions", [])],
                    classes=classes,
                    constants=[SymbolRecord(**c) for c in m.get("constants", [])],
                )
            )
        return cls(
            name=data["name"],
            source_root=data.get("source_root", data["name"]),
            modules=modules,
        )

    def symbols(self) -> Iterator[Tuple[ModuleSurface, Any]]:
        """Every symbol with its module, in source order: each function, each
        class (a :class:`ClassSurface`) followed by its members, then the
        constants."""
        for module in self.modules:
            for record in module.functions:
                yield module, record
            for entry in module.classes:
                yield module, entry
                for record in entry.members:
                    yield module, record
            for record in module.constants:
                yield module, record

    @classmethod
    def _prune(cls, node: Any) -> Any:
        """*node* without ``remove_in`` keys that are empty, nor ``deprecated``
        keys on classes that are false: ``asdict`` would write them against
        every symbol in the ecosystem, saying nothing. ``deprecated`` stays on
        every symbol row, which has carried it since the mapping was first
        written."""
        if isinstance(node, dict):
            if not node.get("remove_in", None):
                node.pop("remove_in", None)
            if "members" in node and not node.get("deprecated", False):
                node.pop("deprecated", None)
            for value in node.values():
                cls._prune(value)
        elif isinstance(node, list):
            for value in node:
                cls._prune(value)
        return node
