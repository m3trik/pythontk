# !/usr/bin/python
# coding=utf-8
"""SymbolRecord - the shared public-API symbol shape.

A single, minimal data record describing one public symbol (a top-level function
or a class member). Both halves of :class:`~pythontk.ApiSurface` produce it and
several tools consume it, so it lives here at the bottom of the dependency
stack (zero non-stdlib imports) as the single source of truth for that shape.

Producers (both :class:`~pythontk.ApiSurface`)
    * *static* -- :meth:`~pythontk.ApiSurface.read`, via ``ast``. Runs
      anywhere (no package import, no DCC/Qt); a read is kept in the
      per-user cache.
    * *live* -- :meth:`~pythontk.ApiSurface.members`, via ``inspect`` on a
      loaded class. Sees the resolved MRO, metaclass/mixin-injected members,
      and unwrapped signatures the static read cannot.

Consumers
    * the ``api.search`` / ``api.describe`` operations and m3trik's release
      checks (static),
    * ``HelpMixin.help(as_dict=...)`` structured output (live),
    * the runtime-vs-static drift gate, which compares the two producers'
      records by ``(qualname, kind)``, deliberately NOT the
      signature string, which the two engines render differently.

The field set is deliberately minimal and its ORDER is a hard contract: the
static read's mapping (``PackageSurface.to_dict``) is ``asdict`` of it, kept
in the per-user cache and diffed against a release's. Runtime-only
enrichments (async/abstract flags, the defining class, a resolved source
location) are layered on by :meth:`~pythontk.ApiSurface.enrich` at the dict
level, not added here.

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

    Field names and order are the static read's mapping (see the module
    docstring).
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
        """The member as a markdown bullet: ```Owner.name(sig)``` with its kind,
        a **DEPRECATED** marker (and its removal version) and its summary. No
        source link: the record carries no module context.
        """
        decoration = {
            "staticmethod": " *(static)*",
            "classmethod": " *(class)*",
            "property": " *(property)*",
        }.get(self.kind, "")
        if self.deprecated:
            # The removal version rides in the row itself so a reviewer sees
            # the deadline without opening the source.
            due = f" (remove in {self.remove_in})" if self.remove_in else ""
            decoration += f" **DEPRECATED{due}**"
        summary = f" — {self.summary}" if self.summary else ""
        return f"  - `{self.qualname}{self.signature}`{decoration}{summary}"
