# !/usr/bin/python
# coding=utf-8
"""The change-detection baseline an exporter diffs a scene's hierarchy against.

ONE baseline per scene, not one per delivered file.  The set algebra lives here
so both DCC exporters run the same rule; each supplies its own storage and its
own path-normalisation, which is where they legitimately differ (Maya's
``exportSelected`` ships a node's parent chain, Blender's ``use_selection``
does not).

The record is a flat path set plus a hash, stamped with the scene file that
recorded it (:meth:`HierarchyBaseline.recorded_by`): a Save As copy carries its
source's record verbatim, so the scene data alone cannot tell the copy from the
source, and a copy made into another module must not be diffed against what the
source exported.  Scoping is derived at compare time from the roots being
exported rather than stored, so nothing has to be keyed, migrated or kept in
step when an export is renamed, re-scoped or re-pointed:

- :meth:`HierarchyBaseline.compare` diffs only the part of the baseline this
  export is answerable for (:meth:`HierarchyBaseline.relevant_roots`), so
  exporting asset B never reports asset A as missing -- while a root that was
  merely MOVED stays in scope, because a wrapper group rewrites every path and
  scoping off the current roots alone would call that a clean first export.
- :meth:`HierarchyBaseline.merge` replaces only that part, so one record
  accumulates every scope the scene has exported without them overwriting each
  other.
- A scope with nothing recorded under it is NEW, not wholly "extra" -- the
  first export of an asset has nothing to be diffed against, exactly as a
  missing manifest had nothing to be diffed against before.
- :meth:`HierarchyBaseline.adopt` fills such a scope from what the deliverable
  last shipped, and only such a scope, so a record never loses one
  deliverable's history to another's.
"""

import hashlib
import json
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


class HierarchyBaseline:
    """Pure set algebra over ``|``-delimited hierarchy paths.

    Every method is a function of its arguments -- no scene, no file, no clock
    -- so a consumer can preview a diff, test one, or replay one without a DCC.
    """

    #: Record schema version, stamped into every :meth:`encode`.  A reader that
    #: does not recognise it treats the record as absent (no baseline) rather
    #: than guessing: a wrong baseline reports false structural changes, which
    #: is worse than reporting none.
    FORMAT = 1

    #: The path separator both DCCs normalise to (see ``build_clean_path_set``).
    SEPARATOR = "|"

    @classmethod
    def top_level(cls, paths: Iterable[str]) -> List[str]:
        """The shallowest paths in *paths* -- those with no ancestor also present.

        These are an export's *roots*, and therefore its scope.

        Shallowest first, then alphabetically -- the depth alone is not a total
        order, and the callers pass a SET, so ordering by depth only made the
        result depend on set iteration order. That reached the user: the diff
        report rolls missing/extra up to these roots and shows the first 20, so
        an unchanged scene could name different nodes from run to run.

        Example:
            >>> HierarchyBaseline.top_level({"a|b", "a", "c"})
            ['a', 'c']
        """
        result: List[str] = []
        for path in sorted(paths, key=lambda p: (p.count(cls.SEPARATOR), p)):
            if not any(path.startswith(root + cls.SEPARATOR) for root in result):
                result.append(path)
        return result

    @classmethod
    def in_scope(cls, paths: Iterable[str], roots: Sequence[str]) -> Set[str]:
        """The subset of *paths* at or under any of *roots*.

        Prefix matching is on a whole component (``root`` or ``root|...``), never
        a bare ``startswith``: ``assetA_grp`` must not claim ``assetA_grp_old``.
        """
        roots = list(roots)
        return {
            p
            for p in paths
            if any(p == r or p.startswith(r + cls.SEPARATOR) for r in roots)
        }

    @classmethod
    def relevant_roots(
        cls,
        baseline: Iterable[str],
        current: Iterable[str],
        roots: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """The baseline roots THIS export is answerable for.

        Scope cannot be read off the current roots alone.  Wrapping an export in
        a new group rewrites every path, so the old root lives on as an inner
        component and a roots-only test finds nothing to diff against -- which
        would report a whole scene wrapped in a group as a clean first export,
        the exact accident the check exists to catch.

        A baseline root is in play when it is at or under one of *roots* (the
        ordinary case), when a current root sits under IT (the export narrowed),
        or when its own name still appears as a component of *current* (it moved
        -- it was reparented, wrapped, or regrouped). Matching on the root's name
        rather than on its leaves is deliberate: root names are asset-group
        names and rarely collide, while leaf names (``body``, ``Mesh``) collide
        constantly and would drag an unrelated asset into scope.
        """
        baseline, current = set(baseline), set(current)
        # Nothing exporting: there is no scope to derive, and "no scope" must
        # not read as "nothing to check". An export set that collapsed to empty
        # is the loudest structural change there is, so the WHOLE baseline is in
        # play and every path it holds is reported missing.
        if not current:
            return cls.top_level(baseline)
        if roots is None:
            roots = cls.top_level(current)
        components = {c for p in current for c in p.split(cls.SEPARATOR)}
        in_play = []
        for root in cls.top_level(baseline):
            touches = any(
                root == r
                or root.startswith(r + cls.SEPARATOR)
                or r.startswith(root + cls.SEPARATOR)
                for r in roots
            )
            if touches or root.split(cls.SEPARATOR)[-1] in components:
                in_play.append(root)
        return in_play

    @classmethod
    def compare(
        cls,
        baseline: Iterable[str],
        current: Iterable[str],
        roots: Optional[Sequence[str]] = None,
    ) -> Tuple[bool, List[str], List[str], bool]:
        """Diff *current* against the part of *baseline* in the same scope.

        Parameters:
            baseline: Every path the scene has recorded, across all scopes.
            current: The paths this export ships (already closed under
                ancestors by the caller if its DCC ships them).
            roots: The current export's roots; ``None`` derives them from
                *current* (:meth:`top_level`). The scope actually diffed is
                :meth:`relevant_roots`, which is broader.

        Returns:
            (tuple) ``(match, missing, extra, is_new_scope)``. *is_new_scope* is
            True when the baseline holds nothing in scope -- there was
            nothing to diff, so *match* is True and both lists are empty.  A
            caller reports that differently from a clean diff: one says "checked
            and unchanged", the other "first time, now recorded".
        """
        baseline, current = set(baseline), set(current)
        prior = cls.in_scope(baseline, cls.relevant_roots(baseline, current, roots))
        if not prior:
            return True, [], [], True
        missing = sorted(prior - current)
        extra = sorted(current - prior)
        return (not missing and not extra), missing, extra, False

    @classmethod
    def merge(
        cls,
        baseline: Iterable[str],
        current: Iterable[str],
        roots: Optional[Sequence[str]] = None,
    ) -> Set[str]:
        """*baseline* with this export's scope replaced by *current*.

        Only the exported scope rolls forward; every other scope the scene has
        recorded is left exactly as it was.  That is what lets one record serve
        every export a scene makes.
        """
        baseline, current = set(baseline), set(current)
        # An export that shipped NOTHING records nothing. :meth:`relevant_roots`
        # puts the whole baseline in play for an empty set -- which is right for
        # the diff (everything is missing) and catastrophic for the merge, which
        # would replace all of it with nothing and erase every scope's history.
        # Worse, the next export would then see a clean slate, hiding the very
        # collapse that just happened.
        if not current:
            return baseline
        # The SAME scope compare used, or the merge would strand paths it just
        # reported as missing.
        scope = cls.relevant_roots(baseline, current, roots)
        return (baseline - cls.in_scope(baseline, scope)) | current

    @classmethod
    def adopt(
        cls, baseline: Iterable[str], shipped: Optional[Iterable[str]]
    ) -> Optional[Set[str]]:
        """*baseline* with a deliverable's last-*shipped* hierarchy adopted,
        or ``None`` when there is nothing to adopt.

        A deliverable's record of what it last shipped (its sidecar) is the
        history of a scope the scene's own baseline may not hold: one set
        aside as another scene's, or recorded before this deliverable was
        exported from here.  Adopted per SCOPE, never all-or-nothing -- a
        scene exporting two deliverables after its baseline was set aside
        records the first one's scope at its first export, and a rule that
        adopted only into an empty baseline then refused the second one's
        history, leaving its next export nothing to diff against.

        ``None`` when *shipped* holds nothing, or when *baseline* already
        records any of its scope (:meth:`relevant_roots`, the scope
        :meth:`compare` diffs): the scene's own record is newer than what the
        deliverable last shipped, and merging the older one over it would
        resurrect paths the scene has since dropped.

        Example:
            >>> sorted(HierarchyBaseline.adopt({"a", "a|x"}, {"b", "b|y"}))
            ['a', 'a|x', 'b', 'b|y']
            >>> HierarchyBaseline.adopt({"a", "a|x"}, {"a", "a|z"}) is None
            True
        """
        shipped, baseline = set(shipped or ()), set(baseline)
        if not shipped or cls.relevant_roots(baseline, shipped):
            return None
        return baseline | shipped

    @staticmethod
    def paths_hash(paths: Iterable[str]) -> str:
        """SHA-256 over the sorted paths -- the fast-path equality check.

        Covers the paths and nothing else, so neither metadata churn nor a
        recorded diff can make an unchanged hierarchy look changed.
        """
        digest = hashlib.sha256()
        for path in sorted(paths):
            digest.update(path.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    @classmethod
    def encode(cls, paths: Iterable[str], scene: Optional[str] = None) -> Dict:
        """The stored record for *paths*.

        Parameters:
            paths: The hierarchy paths to record.
            scene: The scene file recording it, as the DCC's scene records
                spell a file (relative to the scene's own project), ``""``
                while the scene is unsaved. ``None`` leaves the record
                unstamped. Read back by :meth:`recorded_by`; neither a path
                nor hashed, so it changes no diff.
        """
        ordered = sorted(set(paths))
        record = {
            "format": cls.FORMAT,
            "paths": ordered,
            "hash": cls.paths_hash(ordered),
            "object_count": len(ordered),
        }
        if scene is not None:
            record["scene"] = scene
        return record

    @classmethod
    def _parse(cls, record) -> Optional[Dict]:
        """*record* as a recognised record mapping, else ``None``.

        The one reading of the schema :meth:`is_record`, :meth:`decode` and
        :meth:`recorded_by` share; a JSON string is parsed first.
        """
        if isinstance(record, str):
            try:
                record = json.loads(record)
            except ValueError:
                return None
        if (
            isinstance(record, dict)
            and record.get("format") == cls.FORMAT
            and isinstance(record.get("paths"), list)
        ):
            return record
        return None

    @classmethod
    def is_record(cls, record) -> bool:
        """*record* is a recognisable baseline, even an empty one.

        Separates "no paths recorded" from "nothing readable here", which
        :meth:`decode` alone cannot: both come back as an empty set, but only
        the second means a baseline was LOST. A consumer warns about one and
        not the other.
        """
        return cls._parse(record) is not None

    @classmethod
    def decode(cls, record) -> Set[str]:
        """The path set in *record*; empty for anything unreadable.

        Tolerant by design and silent about it -- the caller decides whether an
        absent baseline is worth a word.  A JSON string is accepted so a
        consumer can hand over a raw channel value without parsing it first.
        """
        parsed = cls._parse(record)
        if parsed is None:
            return set()
        return {p for p in parsed["paths"] if isinstance(p, str)}

    @classmethod
    def recorded_by(cls, record) -> Optional[str]:
        """The scene file *record* was recorded by, as :meth:`encode` stored it.

        ``""`` for a record made while its scene was unsaved; ``None`` for one
        that names no scene -- recorded before records were stamped -- and for
        anything unreadable.  Whether that scene is the one open now is the
        DCC's question (``pythontk.FileDependencies.written_here``): a Save As
        copy holds its source's stamp, and the source is still on disk.
        """
        parsed = cls._parse(record)
        scene = parsed.get("scene") if parsed is not None else None
        return scene if isinstance(scene, str) else None
