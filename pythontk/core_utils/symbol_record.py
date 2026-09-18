# !/usr/bin/python
# coding=utf-8
"""SymbolRecord - the shared public-API symbol shape.

A single, minimal data record describing one public symbol (a top-level function
or a class member). It is produced by two independent introspection engines and
consumed by several tools, so it lives here at the bottom of the dependency
stack (zero non-stdlib imports) as the single source of truth for that shape.

Producers
    * ``m3trik/scripts/generate_api_registry.py`` - *static*, via ``ast``. Runs
      anywhere (no package import, no DCC/Qt) and owns the committed
      ``API_REGISTRY`` artifacts.
    * ``pythontk.HelpMixin`` - *dynamic*, via ``inspect`` on a live class. Sees
      the resolved MRO, metaclass/mixin-injected members, and unwrapped
      signatures the static walker cannot.

Consumers
    * the ``API_INDEX`` / ``API_REGISTRY`` markdown (static producer),
    * ``HelpMixin.help(as_dict=...)`` structured output / live-DCC RPC,
    * the runtime-vs-static drift gate, which compares the two producers'
      records by :meth:`key` - ``(qualname, kind)``, deliberately NOT the
      signature string, which the two engines render differently.

The field set is deliberately minimal and its ORDER is a hard contract: the
static registry sidecar (``API_REGISTRY.json``) is ``json.dumps(asdict(...))``,
so reordering churns every committed registry in the ecosystem. Runtime-only
enrichments (async/abstract flags, the defining class, a resolved source
location) are layered on by ``HelpMixin`` at the dict level, not added here.

``remove_in`` is the one field added since the set was first frozen, and it
pays for its churn: ``deprecated`` alone records THAT a name is retired, which
is what let ``flip_uvs`` ship in 50 releases after its notice. Both producers
now carry the release it stops working in -- the static one reads it off the
``Deprecation`` decorator with ``ast``, the dynamic one off the record the
decorator stamped -- so an alias that outlived its window is a comparison
rather than an act of memory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class SymbolRecord:
    """One public symbol: a top-level function or a class member.

    Field names and order mirror the static registry sidecar so this type is a
    drop-in for the generator's serialisation (see the module docstring).
    """

    name: str
    qualname: str
    kind: str  # function|method|staticmethod|classmethod|property|class|attribute
    signature: str
    summary: str
    line: int
    deprecated: bool = False
    #: Release the symbol stops working in, e.g. ``"0.11.0"``. Empty when the
    #: symbol is live, or when it is deprecated by a bare marker that named no
    #: removal version.
    remove_in: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """Plain ``dict`` of the fields (matches the ``hierarchy_diff`` convention)."""
        return asdict(self)

    def as_json(self, indent: int = 2) -> str:
        """JSON string of :meth:`as_dict`."""
        return json.dumps(self.as_dict(), indent=indent)

    def to_registry_row(self) -> str:
        """Render the full-registry class-member bullet.

        Single source of the member-row markdown that
        ``generate_api_registry.py`` emits for class members. No source link is
        included - top-level functions and class headers carry module context
        this record does not, so their links are added by the generator.
        """
        decoration = {
            "staticmethod": " *(static)*",
            "classmethod": " *(class)*",
            "property": " *(property)*",
        }.get(self.kind, "")
        if self.deprecated:
            # The removal version rides in the row itself so a reviewer reading
            # the registry sees the deadline without opening the source.
            due = f" (remove in {self.remove_in})" if self.remove_in else ""
            decoration += f" **DEPRECATED{due}**"
        summary = f" — {self.summary}" if self.summary else ""
        return f"  - `{self.qualname}{self.signature}`{decoration}{summary}"
