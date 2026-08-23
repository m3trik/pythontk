# !/usr/bin/python
# coding=utf-8
"""Batch renaming: a dry-run-aware plan executor and a file-system engine.

:class:`RenamePlan` is the one place a batch of "old name → new name" changes is
applied and reported. Every engine that renames named things (files here;
scene nodes in mayatk / blendertk) plans its new names with the ``str_utils``
pattern primitives, then hands the plan to :meth:`RenamePlan.apply` together
with the one strategy that differs per host — the callable that performs a
single rename. The executor owns dry-run (plan only, nothing touched), the
per-item error policy, and the panel-ready report (one
``log_group`` record, not one paragraph per item), so the three engines stay
identical in behavior and the tools hosting them identical in output.

:class:`FileNaming` is the file-system tenant: the same find / rename /
convert-case / strip-chars operations the DCC naming tools offer, applied to
file name *stems* (the extension is never touched) or, with ``base_names``, to
the map-suffix-free *base* of a texture file name.
"""

from __future__ import annotations

import os
import re
from typing import Callable, List, Optional, Sequence, Tuple

from pythontk.core_utils.help_mixin import HelpMixin
from pythontk.core_utils.logging_mixin import LoggerExt, LoggingMixin
from pythontk.iter_utils._iter_utils import IterUtils
from pythontk.str_utils._str_utils import StrUtils

# (key, old_name, new_name) — ``key`` is whatever the host's rename strategy
# needs to locate the item (a path, a node UUID, an object reference).
PlanEntry = Tuple[object, str, str]


class RenamePlan(LoggingMixin):
    """Apply a batch of planned renames with dry-run support and a report.

    Example:
        plan = [("/tmp/a.png", "a", "a_v2"), ("/tmp/b.png", "b", "b")]
        RenamePlan.apply(plan, rename_fn, title="Rename", dry_run=True)
        # -> [("a", "a_v2"), ("b", "b")]; nothing renamed, report labelled DRY RUN
    """

    MAX_REPORT_ITEMS = 60

    @classmethod
    def apply(
        cls,
        plan: Sequence[PlanEntry],
        rename: Callable[[object, str], str],
        title: str = "Rename",
        dry_run: bool = False,
        logger=None,
        link: Optional[Callable[[object, str], str]] = None,
        unit: str = "item",
    ) -> List[Tuple[str, str]]:
        """Execute *plan* (unless ``dry_run``) and emit one report for it.

        Parameters:
            plan: ``(key, old_name, new_name)`` triples. Entries whose names are
                equal are reported as unchanged and never passed to ``rename``.
            rename: ``rename(key, new_name) -> actual_name``. Performs ONE rename
                and returns the name the host actually assigned (a host may
                uniquify on collision). Raise to report the item as failed; the
                item then keeps its old name in the result.
            title: Operation name shown in the report ("Rename", "Convert Case").
            dry_run: Report what would change without calling ``rename``.
            logger: Logger receiving the report; defaults to this class's logger.
                Pass the host tool's logger so the report lands in its panel.
            link: Optional ``link(key, name) -> html`` used to render an item's
                name in the report (e.g. a ``log_link`` that selects the node).
            unit: Noun for the summary line ("object", "file").

        Returns:
            ``(old_name, new_name)`` pairs parallel to *plan*. On a live run
            ``new_name`` is the name actually assigned (the old name when the
            rename failed); on a dry run it is the planned name.
        """
        log = logger if logger is not None else cls.logger
        plan = list(plan)
        results: List[Tuple[str, str]] = []
        changed: List[Tuple[object, str, str]] = []
        unchanged: List[str] = []
        failed = 0

        for key, old, new in plan:
            if new == old:
                unchanged.append(old)
                results.append((old, old))
                continue
            actual = new
            if not dry_run:
                try:
                    actual = rename(key, new)
                except Exception as e:  # one bad item must not abort the batch
                    failed += 1
                    log.warning(f"Could not rename '{old}' → '{new}': {e}")
                    results.append((old, old))
                    continue
                if actual != new:
                    log.warning(f"'{old}' renamed to '{actual}' instead of '{new}'")
            changed.append((key, old, actual))
            results.append((old, actual))

        cls._report(log, title, plan, changed, unchanged, failed, dry_run, link, unit)
        return results

    @staticmethod
    def _emit(log, level: str, message: str) -> None:
        """Log at a LoggerExt level (``notice`` / ``result``), or ``info`` on a plain logger."""
        (getattr(log, level, None) or log.info)(message)

    @classmethod
    def _report(cls, log, title, plan, changed, unchanged, failed, dry_run, link, unit):
        """Emit the report: one group of ``old → new`` lines plus a summary."""
        if not plan:
            cls._emit(log, "notice", f"{title}: nothing in scope.")
            return

        def name(key, text):
            return link(key, text) if link else text

        mode = " (DRY RUN)" if dry_run else ""

        def plural(n):
            return unit if n == 1 else f"{unit}s"

        if changed:
            items = [
                f"{name(key, old)} → <b>{new}</b>"
                for key, old, new in changed[: cls.MAX_REPORT_ITEMS]
            ]
            hidden = len(changed) - len(items)
            if hidden > 0:
                items.append(f"… +{hidden} more")
            group_title = (
                f"{title}{mode} — {len(changed)} of {len(plan)} {plural(len(plan))} "
                f"{'would change' if dry_run else 'renamed'}"
            )
            if hasattr(log, "log_group"):
                log.log_group(
                    group_title, items, level="NOTICE" if dry_run else "SUCCESS"
                )
            else:  # plain stdlib logger
                log.info(group_title)
                for item in items:
                    log.info(f"  {LoggerExt.strip_html(item)}")

        if unchanged:
            shown = ", ".join(unchanged[:10])
            more = f", … +{len(unchanged) - 10} more" if len(unchanged) > 10 else ""
            log.info(f"Unchanged ({len(unchanged)}): {shown}{more}")

        if dry_run:
            cls._emit(
                log,
                "notice",
                f"Dry run — {len(changed)} {plural(len(changed))} would be renamed; "
                "nothing was changed.",
            )
        elif changed:
            tail = f" ({failed} failed)" if failed else ""
            cls._emit(
                log,
                "result",
                f"{title}: renamed {len(changed)} {plural(len(changed))}{tail}.",
            )
        elif failed:
            log.error(f"{title}: all {failed} renames failed.")
        else:
            cls._emit(
                log,
                "result",
                f"{title}: all {len(plan)} {plural(len(plan))} already conform.",
            )


class FileNaming(HelpMixin, LoggingMixin):
    """Batch find / rename files by name pattern — the DCC naming tools' file tenant.

    Every operation works on the file name **stem** (``report`` of
    ``report.pdf``): the extension is preserved verbatim, so a rename can never
    change a file's type. ``paths`` may be file paths or directories; a
    directory contributes its direct (non-recursive) files.

    ``base_names=True`` narrows that further, to the **base name** — the stem
    with its texture-map suffix and UDIM tile held back (the split is
    ``MapRegistry.split_map_suffix``, which owns the map-type taxonomy) — so one
    operation carries a whole texture set without touching the tokens that
    identify each map.

    Example:
        FileNaming.rename(["/shots/sh010"], "**_v2", "*_diffuse*", dry_run=True)
        FileNaming.set_case("/shots/sh010", "lower")
        FileNaming.strip_chars(files, num_chars=3, trailing=True)
        FileNaming.rename(files, "stone", "rock", base_names=True)
    """

    @staticmethod
    def expand(paths) -> List[str]:
        """Resolve *paths* (files and/or directories) to a list of file paths.

        Directories contribute their direct files only — a recursive sweep is
        an easy way to rename a whole tree by accident. Missing paths are
        dropped. Order is preserved; a directory's files are sorted.
        """
        files: List[str] = []
        for p in IterUtils.make_iterable(paths):
            if not p:
                continue
            p = os.path.normpath(os.path.expandvars(str(p)))
            if os.path.isdir(p):
                files.extend(
                    os.path.join(p, f)
                    for f in sorted(os.listdir(p))
                    if os.path.isfile(os.path.join(p, f))
                )
            elif os.path.isfile(p):
                files.append(p)
        return files

    @staticmethod
    def stem(path: str) -> str:
        """The file name without directory or extension."""
        return os.path.splitext(os.path.basename(path))[0]

    @classmethod
    def _name_parts(cls, files, base_names: bool) -> List[Tuple[str, str]]:
        """``(subject, tail)`` per file — what an operation acts on, and what it preserves.

        The subject is the whole stem by default and, under ``base_names``, the
        base name that ``MapRegistry.split_map_suffix`` peels a texture-map
        suffix and UDIM tile off (the registry owns that taxonomy; re-deriving
        it from the raw strip pattern is how two copies of this rule drifted
        once already). ``subject + tail`` is always the stem, so every operation
        plans ``new_subject + tail`` and stays extension- (and, in base-name
        mode, map-type-) safe by construction.
        """
        if not base_names:
            return [(cls.stem(f), "") for f in files]
        from pythontk.core_utils.engines.textures.map_registry import MapRegistry

        split = MapRegistry().split_map_suffix
        return [split(cls.stem(f)) for f in files]

    @classmethod
    def find(
        cls,
        paths,
        fltr: str,
        regex: bool = False,
        ignore_case: bool = False,
        base_names: bool = False,
    ) -> List[str]:
        """Files (from *paths*) whose stem matches *fltr*.

        Parameters:
            paths: File and/or directory paths (see :meth:`expand`).
            fltr: Wildcard pattern (``*chars*`` / ``chars*`` / ``*chars`` /
                ``a|b``) or, with ``regex``, a regular expression. Matched
                against the stem only.
            regex: Treat *fltr* as a regular expression.
            ignore_case: Case-insensitive match.
            base_names: Match the base name rather than the whole stem, so one
                pattern collects every map of a texture set (see
                ``MapRegistry.split_map_suffix``).

        Returns:
            The matching file paths, in *paths* order.
        """
        files = cls.expand(paths)
        if not fltr:
            return files
        subjects = [s for s, _tail in cls._name_parts(files, base_names)]
        hits = set(
            StrUtils.find_str(fltr, subjects, regex=regex, ignore_case=ignore_case)
        )
        return [f for f, s in zip(files, subjects) if s in hits]

    @classmethod
    def rename(
        cls,
        paths,
        to: str,
        fltr: str = "",
        regex: bool = False,
        ignore_case: bool = False,
        retain_suffix: bool = False,
        valid_suffixes: Optional[List[str]] = None,
        base_names: bool = False,
        dry_run: bool = False,
        logger=None,
    ) -> List[Tuple[str, str]]:
        """Rename files by pattern (same grammar as the DCC naming tools).

        Parameters:
            paths: File and/or directory paths (see :meth:`expand`).
            to: Replacement with the asterisk marking the kept part — see
                :meth:`StrUtils.find_str_and_format` (``*chars*`` replace the
                match, ``**chars`` append a suffix, ``chars**`` a prefix, ...).
            fltr: Which stems to rename and the text the modes replace. Empty
                matches every file.
            regex: Use regular expressions for *fltr* (capture groups are
                available in *to* as ``\\1`` / ``\\g<name>``).
            ignore_case: Case-insensitive filter.
            retain_suffix: Carry each stem's trailing ``_TYPE`` suffix over to
                its new name (see :meth:`StrUtils.retain_suffix`).
            valid_suffixes: The suffixes ``retain_suffix`` recognizes; None
                accepts any.
            base_names: Rename the base name rather than the whole stem — the
                texture-map suffix and UDIM tile are held back and re-attached
                (``MapRegistry.split_map_suffix``), so one rename carries a whole
                texture set and no map keeps a name that no longer identifies it.
            dry_run: Report the plan without renaming anything.
            logger: Report sink (defaults to this class's logger).

        Returns:
            ``(old_path, new_path)`` per matched file; equal when unchanged or
            when the rename failed. On a dry run ``new_path`` is the plan.
        """
        files = cls.expand(paths)
        parts = cls._name_parts(files, base_names)
        subjects = [s for s, _tail in parts]
        try:
            pairs = StrUtils.find_str_and_format(
                subjects,
                to,
                fltr,
                regex=regex,
                ignore_case=ignore_case,
                return_orig_strings=True,
            )
        except Exception as e:
            (logger or cls.logger).error(
                f"Invalid pattern — filter '{fltr}', rename '{to}': {e}"
            )
            return []
        # Consume matches positionally so same-subject files each get their own
        # entry (find_str_and_format returns one pair per matched input, in
        # input order). Same-stemmed files in different directories collide here,
        # and so does every map of one texture set under `base_names`.
        by_subject: dict = {}
        for f, (subject, tail) in zip(files, parts):
            by_subject.setdefault(subject, []).append((f, tail))
        plan = []
        for old, new in pairs:
            bucket = by_subject.get(old)
            if not bucket:
                continue
            path, tail = bucket.pop(0)
            if retain_suffix:
                new = StrUtils.retain_suffix(old, new, valid_suffixes)
            plan.append((path, old + tail, new + tail))
        return cls._apply(plan, "Rename", dry_run, logger)

    @classmethod
    def set_case(
        cls,
        paths,
        case: str = "capitalize",
        base_names: bool = False,
        dry_run: bool = False,
        logger=None,
    ) -> List[Tuple[str, str]]:
        """Re-case file stems: ``upper`` / ``lower`` / ``capitalize`` / ``swapcase`` / ``title``.

        ``base_names`` re-cases the base name only, leaving each map-type suffix
        in the spelling the pipeline that wrote it chose.

        Returns:
            ``(old_path, new_path)`` per file (see :meth:`rename`).
        """
        files = cls.expand(paths)
        plan = [
            (f, subject + tail, StrUtils.set_case(subject, case) + tail)
            for f, (subject, tail) in zip(files, cls._name_parts(files, base_names))
        ]
        return cls._apply(plan, f"Convert Case ({case})", dry_run, logger)

    @classmethod
    def strip_chars(
        cls,
        paths,
        num_chars: int = 1,
        trailing: bool = False,
        base_names: bool = False,
        dry_run: bool = False,
        logger=None,
    ) -> List[Tuple[str, str]]:
        """Delete *num_chars* leading (or ``trailing``) characters from each stem.

        A stem that would be emptied is skipped (and reported), and a count of zero
        or less is a no-op rather than a batch of skips: ``s[:-0]`` is the EMPTY
        string, so the trailing branch used to propose an empty stem for every file
        while the leading branch (``s[0:]``) correctly changed nothing. A spinbox
        cleared to 0 reaches this.

        ``base_names`` counts (and cuts) within the base name, so a trailing trim
        eats the end of the name rather than the map-type suffix behind it.

        Returns:
            ``(old_path, new_path)`` per file (see :meth:`rename`).
        """
        log = logger or cls.logger
        if num_chars <= 0:
            return []
        files = cls.expand(paths)
        plan = []
        for f, (subject, tail) in zip(files, cls._name_parts(files, base_names)):
            if num_chars >= len(subject):
                log.warning(
                    f"Skipped '{os.path.basename(f)}': cannot remove {num_chars} "
                    f"characters from a {len(subject)}-character name."
                )
                continue
            cut = subject[:-num_chars] if trailing else subject[num_chars:]
            plan.append((f, subject + tail, cut + tail))
        return cls._apply(plan, "Strip Chars", dry_run, logger)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    # Characters no portable file name may contain (Windows' reserved set plus
    # NUL) — a stem carrying one would create a path, not a name.
    _ILLEGAL_STEM_CHARS = re.compile(r'[<>:"/\\|?*\x00]')

    @classmethod
    def _apply(cls, plan, title, dry_run, logger) -> List[Tuple[str, str]]:
        """Run *plan* (``(path, old_stem, new_stem)``) through :class:`RenamePlan`.

        Entries whose new stem is not a usable file name (empty, ``.``/``..``,
        or carrying a path / reserved character) are dropped with a warning
        before the plan runs, so neither a dry run nor a live one shows them.
        """
        log = logger or cls.logger
        valid = []
        for path, old, new in plan:
            if new in ("", ".", "..") or cls._ILLEGAL_STEM_CHARS.search(new):
                log.warning(
                    f"Skipped '{os.path.basename(path)}': '{new}' is not a valid file name."
                )
                continue
            valid.append((path, old, new))
        plan = valid
        results = RenamePlan.apply(
            plan,
            cls._rename_file,
            title=title,
            dry_run=dry_run,
            logger=log,
            unit="file",
        )
        # Map stem results back to full paths for the caller.
        out = []
        for (path, _old, _new), (_o, actual) in zip(plan, results):
            out.append((path, cls._with_stem(path, actual)))
        return out

    @staticmethod
    def _with_stem(path: str, stem: str) -> str:
        directory, filename = os.path.split(path)
        return os.path.join(directory, stem + os.path.splitext(filename)[1])

    @classmethod
    def _rename_file(cls, path: str, new_stem: str) -> str:
        """The :class:`RenamePlan` strategy: rename one file on disk, never overwrite."""
        target = cls._with_stem(path, new_stem)
        if os.path.exists(target) and not (
            os.name == "nt" and target.lower() == path.lower()
        ):
            raise FileExistsError(f"'{os.path.basename(target)}' already exists")
        os.rename(path, target)
        return new_stem


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
