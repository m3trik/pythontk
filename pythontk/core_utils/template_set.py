# !/usr/bin/python
# coding=utf-8
"""A discoverable, user-extensible collection of schema-validated template files.

:class:`TemplateSet` is the small piece of glue that binds the two halves of the
template system:

* a :class:`~pythontk.core_utils.preset_store.PresetStore` — the *storage* SSoT
  (two-tier built-in + user discovery, shadowing, last-used pointer, any codec),
* a :class:`~pythontk.core_utils.schema_spec.SchemaSpec` — the *shape* SSoT
  (validation, skeleton generation, reference docs).

Two consumers in the mayatk ``shot_manifest`` tool share it — the JSON CSV
*mapping* files and the YAML *behavior* templates — so neither has to re-invent
"find templates across built-in + user dirs, load, validate, and offer a
documented skeleton to copy."  A tool wires one of these and gets:

* ``names()`` / ``source()`` — what's available and where it came from,
* ``load()`` — parse + validate + deserialise to a spec instance,
* ``skeleton()`` / ``write_skeleton()`` — a model file for users to copy,
* ``markdown()`` — generated format reference,

while file *placement* (built-in read-only vs. user-writable) and *naming* are
handled by the underlying store, so "extend it with your own file" just means
dropping one in :attr:`user_dir` (or clicking *New from template*).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from pythontk.core_utils.preset_store import JSON_CODEC, Codec, PresetStore
from pythontk.core_utils.schema_spec import SchemaSpec, ValidationResult

logger = logging.getLogger(__name__)


class TemplateSet:
    """Schema-aware, two-tier collection of template files.

    Parameters:
        name: Collection name (also the default user sub-directory).
        spec: A :class:`SchemaSpec` subclass describing one template file.
        package: Owning package, for the default user-dir location.
        builtin_dir: Directory of shipped, read-only templates.
        user_dir: Explicit writable directory (else derived under
            :func:`~pythontk.core_utils.user_config.user_config_root`).
        codec: On-disk format (defaults to JSON; pass YAML for ``*.yaml``).
    """

    def __init__(
        self,
        name: str,
        spec: Type[SchemaSpec],
        package: str = "",
        *,
        builtin_dir: Optional[Union[str, os.PathLike]] = None,
        user_dir: Optional[Union[str, os.PathLike]] = None,
        codec: Codec = JSON_CODEC,
    ):
        self.spec = spec
        self.store = PresetStore(
            name,
            package,
            builtin_dir=builtin_dir,
            user_dir=user_dir,
            codec=codec,
        )

    # -- discovery / tiering (delegated to the store) ----------------------
    def names(self, tier: Optional[str] = None) -> List[str]:
        """Sorted template names (``tier`` = ``None`` | ``"user"`` | ``"builtin"``)."""
        return self.store.list(tier)

    def source(self, name: str) -> Optional[str]:
        """Which tier *name* resolves from: ``"user"``, ``"builtin"``, or ``None``."""
        return self.store.source(name)

    def exists(self, name: str) -> bool:
        return self.store.exists(name)

    @property
    def user_dir(self) -> Path:
        """Writable directory users drop their own templates into."""
        return self.store.user_dir

    @property
    def builtin_dir(self) -> Optional[Path]:
        return self.store.builtin_dir

    @property
    def active(self) -> Optional[str]:
        """Last-selected template name (persisted across sessions)."""
        return self.store.active

    @active.setter
    def active(self, name: Optional[str]) -> None:
        self.store.active = name

    def delete(self, name: str) -> bool:
        """Delete a *user* template (built-ins are read-only)."""
        return self.store.delete(name)

    def rename(self, old: str, new: str) -> bool:
        return self.store.rename(old, new)

    def path(self, name: str, tier: str = "user") -> Path:
        return self.store.path(name, tier)

    # -- schema-aware operations ------------------------------------------
    def raw(self, name: str) -> Dict[str, Any]:
        """The parsed file as a plain ``dict`` (no validation)."""
        return self.store.load(name)

    def validate(self, name: str) -> ValidationResult:
        """Validate a stored template against the schema."""
        return self.spec.validate(self.store.load(name))

    def load(self, name: str, *, strict: bool = False) -> SchemaSpec:
        """Parse, validate, and deserialise *name* to a :class:`SchemaSpec`.

        Errors always raise (:class:`~pythontk.core_utils.schema_spec.SchemaError`).
        Warnings are logged; pass ``strict=True`` to raise on them too.
        """
        data = self.store.load(name)
        self.spec.validate(data).raise_or_warn(
            prefix=f"template {name!r}: ", logger=logger, strict=strict
        )
        return self.spec.from_dict(data)

    def skeleton(self) -> Dict[str, Any]:
        """A fully-populated example dict to model a new template after."""
        return self.spec.skeleton()

    def save(self, name: str, data: Dict[str, Any]) -> Path:
        """Write *data* as a user template *name* (built-ins stay read-only)."""
        return self.store.save(name, data)

    def write_skeleton(self, name: str) -> Path:
        """Write :meth:`skeleton` as a user template and return its path.

        Like :meth:`save`, this overwrites an existing *name* — guard the call
        site (e.g. seed only an empty folder) when clobbering a user's edited
        copy would be wrong.
        """
        return self.save(name, self.spec.skeleton())

    def markdown(self, title: Optional[str] = None) -> str:
        """Generated Markdown reference for this template's format."""
        return self.spec.to_markdown(title)
