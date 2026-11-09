# !/usr/bin/python
# coding=utf-8
"""Held until 0.22.0: the GLB block names v0.20.0 noticed for removal.

Their notices (``remove_in="0.18.0"``, ``since="2026-10-24"``) hold them to
2026-11-23, after 0.21.0 ships; the GLB block machinery itself retired whole
on 2026-11-08 (a GLB carries each record as it stands,
:attr:`RecordSpec.glb_verbatim`). Restored from v0.20.0 so a caller of the old
names keeps working through the window. ``scene_records`` re-exports them;
delete this module, those re-exports, ``RecordSpec.web`` and
``SceneRecords.web_shape`` at 0.22.0 (``pythontk/__init__.py``, "Held until
0.22.0").
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from pythontk.core_utils.deprecation import Deprecation


@dataclass(frozen=True)
class GlbBlock:
    """Retired 2026-11-08 (removed after 0.21.0): a GLB carries each record as
    it stands (:attr:`RecordSpec.glb_verbatim`), its version the record's.

    As v0.20.0 declared it: how a record rode a GLB as a block in its root
    ``extras``.

    Parameters:
        version: The block's schema version.
        shape: ``(module, class)`` of the block's ``SchemaSpec``.
        assembly: What the block made of the approved scene.
        node_fields: Where the block bound by glTF node index.
        clauses: The assembly's clauses that only some blocks need.
    """

    version: int = 1
    shape: Optional[Tuple[str, str]] = None
    assembly: str = ""
    node_fields: Tuple[Tuple[str, ...], ...] = ()
    clauses: Tuple[Any, ...] = ()

    def assembly_for(self, payload: Any = None) -> str:
        """The assembly a block *payload* needs (``Clause.join``)."""
        from pythontk.core_utils.engines.scene_export.scene_records import Clause

        return Clause.join(self.assembly, self.clauses, payload)

    @property
    @Deprecation.symbol(
        "RecordSpec.key -- a block takes its record's key",
        remove_in="0.18.0",
        since="2026-10-24",
    )
    def key(self) -> str:
        """The record's key, which the block is written under; empty for a
        block no record holds."""
        return self.__dict__.get("_record_key", "")


@Deprecation.symbol(
    "RecordSpec.glb_verbatim -- a GLB carries each record as it stands",
    remove_in="0.18.0",
    since="2026-10-24",
)
class WebProjection(GlbBlock):
    """Retired 2026-10-20 (removed after 0.21.0): :class:`GlbBlock`.

    Builds as it did in 0.16.1 -- ``WebProjection(key, version=1,
    shape=None)`` -- its *key* what :attr:`GlbBlock.key` reads."""

    def __init__(
        self, key: str, version: int = 1, shape: Optional[Tuple[str, str]] = None
    ) -> None:
        super().__init__(version=version, shape=shape)
        object.__setattr__(self, "_record_key", key)


def record_block(spec: Any) -> Optional[GlbBlock]:
    """The block *spec* (a ``RecordSpec``) rides a GLB as, for the retired
    ``RecordSpec.web``: one keyed by the record, in the record's own version
    and shape, where a GLB carries it (``glb_verbatim``); ``None`` where it
    does not."""
    if not spec.glb_verbatim:
        return None
    block = GlbBlock(version=spec.version, shape=spec.shape)
    object.__setattr__(block, "_record_key", spec.key)
    return block
