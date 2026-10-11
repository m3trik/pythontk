# !/usr/bin/python
# coding=utf-8
"""The Shot Manifest's colour tags: a colour per step, the user's and the automatic.

A tag is a palette *slot* key (``"tag3"``, :meth:`Palette.tags`), never a colour:
the UI owns the palette, so a recoloured slot recolours every step holding it,
and nothing saved in a scene records a hue.  Two layers, the user's shown over
the automatic:

* **the user's** -- ``{step_id: slot}`` per source on the scene's
  ``ShotStore.manifest_tags``, keyed as the Local Edits layers are (the sheet's
  path, ``""`` for the scene's own shots).  Not an edit: a tag organises the
  table, changes nothing a Build reads, and shows whether Local Edits is on or
  off.
* **the automatic** -- the mapping template's ``color_by`` rule
  (:attr:`ManifestTags.RULES`): steps sharing a value share a slot.

Pure data: plain values in and out; the panel paints them.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Sequence

from pythontk.core_utils.color import Palette

__all__ = ["ManifestTags"]


class _ManifestTagsInternal(object):
    """Internal helpers for ManifestTags."""

    @staticmethod
    def _section(step: Any) -> str:
        """The value ``color_by: section`` groups *step* by."""
        return str(getattr(step, "section", "") or "")


class ManifestTags(_ManifestTagsInternal):
    """Colour tags on the Shot Manifest's steps (see the module docstring)."""

    #: ``color_by`` rule -> the value that groups steps (``None``: no automatic
    #: colour).  The mapping schema's ``color_by`` choices are these names, so
    #: a new rule is one entry here.
    RULES: Dict[str, Optional[Callable[[Any], str]]] = {
        "none": None,
        "section": _ManifestTagsInternal._section,
    }

    @classmethod
    def auto(
        cls, steps: Iterable[Any], rule: str, slots: Sequence[str]
    ) -> Dict[str, str]:
        """``{step_id: slot}``: the automatic colours *rule* gives *steps*.

        Steps sharing a group value share a slot; groups take *slots* in the
        order they first appear, wrapping after the last, so a sheet's first
        sections all get distinct colours -- walked in contrast order
        (``Palette.contrast_order``), so neighbouring groups never get
        neighbouring hues: taken in turn, a hue-ordered palette coloured a
        sheet's sections as one gradient and hid where each began and ended
        (2026-10-10).  A step with no value (a detected step has no section)
        gets none.  Pass the sheet's steps ahead of the shown ones
        (``[*sheet, *shown]``): the order then comes from the sheet, so hiding
        steps never reshuffles the colours, and a step the sheet lacks (one
        Local Edits added) still gets its group's.

        Parameters:
            steps: Objects with a ``step_id`` and what *rule* reads
                (``BuilderStep``).
            rule: A :attr:`RULES` name; ``"none"`` or an unknown name colours
                nothing.
            slots: The palette's slot keys, in order.

        Returns:
            ``{step_id: slot}`` for every step *rule* groups.
        """
        group_of = cls.RULES.get(rule)
        if group_of is None or not slots:
            return {}
        slots = Palette.contrast_order(slots)
        group_slots: Dict[str, str] = {}
        tags: Dict[str, str] = {}
        for step in steps:
            group = group_of(step)
            if not group:
                continue
            slot = group_slots.get(group)
            if slot is None:
                slot = group_slots[group] = slots[len(group_slots) % len(slots)]
            tags.setdefault(step.step_id, slot)
        return tags

    @staticmethod
    def read(store: Any, source: str) -> Dict[str, str]:
        """*source*'s user tags on *store*, ``{step_id: slot}`` (a copy).

        Parameters:
            store: A ``ShotStore``, or ``None`` (no scene store: no tags).
            source: The layer's key (the sheet's path, ``""`` for the scene).
        """
        if store is None:
            return {}
        return dict(store.manifest_tags.get(source) or {})

    @staticmethod
    def assign(
        store: Any, source: str, step_ids: Iterable[str], slot: Optional[str]
    ) -> bool:
        """Tag *step_ids* of *source* with *slot*, or clear their tags (``None``).

        A layer left empty is dropped.  The store is marked dirty only when a
        tag actually changed.

        Parameters:
            store: The ``ShotStore`` to write.
            source: The layer's key (the sheet's path, ``""`` for the scene).
            step_ids: The steps to tag.
            slot: A palette slot key, or ``None`` / ``""`` to clear.

        Returns:
            Whether any tag changed.
        """
        before = dict(store.manifest_tags.get(source) or {})
        tags = dict(before)
        for step_id in step_ids:
            if slot:
                tags[str(step_id)] = str(slot)
            else:
                tags.pop(str(step_id), None)
        if tags == before:
            return False
        if tags:
            store.manifest_tags[source] = tags
        else:
            store.manifest_tags.pop(source, None)
        store.mark_dirty()
        return True
