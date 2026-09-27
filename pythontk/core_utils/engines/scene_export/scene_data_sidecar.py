# !/usr/bin/python
# coding=utf-8
"""The scene-data sidecar -- what an export shipped, beside the deliverable.

An export ships with ONE sidecar -- ``.{stem}.scene_data.json`` -- an
externalized copy of what shipped in the FBX: the exported hierarchy (which
the FBX carries natively as its node tree) and the ``data_export`` carrier
channels (which ride into the FBX as user properties).  Consumers get the
export's metadata without parsing FBX -- and for GLB-only deliverables the
sidecar is the one guaranteed-readable copy.

Format v3::

    {
      "format": 3,
      "hierarchy": {
        "paths": [...], "object_count": N, "hash": "...",
        "last_diff": {"missing": [...], "extra": [...], "reparented": [...]}
      },
      "data_export": {"shot_metadata": {...}, "lightmap_metadata": {...}, ...}
    }

- ``hierarchy`` -- change-detection baseline: sorted, cleaned hierarchy paths
  from the last successful export plus a SHA-256 hash.  Subsequent exports
  compare against it to detect accidental structural changes (missing/extra
  nodes, reparenting).  :meth:`SceneDataSidecarBase.compare` reads ONLY
  ``paths``/``hash``, and the hash covers only the paths -- so neither
  metadata churn nor a recorded ``last_diff`` can false-positive the hierarchy
  check.  ``last_diff`` appears only when the hierarchy check flagged a
  structural change on the export that wrote this manifest (the user saw the
  diff and proceeded): it records that accepted diff, and the next clean
  export drops it.  The human-readable report of the same diff goes to a temp
  artifact, never to the export folder.
- ``data_export`` -- snapshot of the ``data_export`` carrier at export time
  (``DataNodes.dump``), decoded to nested JSON.  Never includes
  ``data_internal``: that node's contract is scene-private state that must not
  export, and a sidecar next to the deliverable is a form of export.  Omitted
  entirely when the carrier shipped nothing.

One file per stem is the contract (2026-08): the previous trio (manifest +
``.prev`` backup + ``.hierarchy_diff.txt`` report) cluttered shared delivery
folders, and a stale ``.prev`` reads as a deliverable.  The v3 write never
displaces the live manifest (tmp + atomic replace only), every successful
write sweeps the superseded companion files, and on Windows the manifest gets
``FILE_ATTRIBUTE_HIDDEN`` (re-applied after every write: ``os.replace`` hands
the result the tmp file's attributes).

Legacy content still reads: format v1 (a flat ``{"paths", "object_count",
"hash"}`` under ``.{stem}.hierarchy.json``) and v2 (nested sections with
external companions) load through the same accessors, a surviving ``.prev``
serves as a read fallback until it is swept, and :meth:`migrate_legacy`
promotes old-named files to the current name.

Everything here is file I/O and set algebra.  What reaches a scene is four
hooks a DCC adapter overrides (mayatk / blendertk ``SceneDataSidecar``):

=============================  ===============================================
``expand_to_descendants``      the export objects plus their descendants
``drop_intermediate``          nodes that are construction data, not shipped
``build_clean_path_set``       host paths -> the recorded spelling
``_close``                     what a recorded set means on comparison
=============================  ===============================================

With the defaults the class takes path strings and records exactly them.
"""

import hashlib
import json
import logging
import os
import re
from typing import Optional, Set, Tuple

import pythontk as ptk


class SceneDataSidecarBase:
    """The scene-data sidecar stored alongside export files, host-agnostic.

    One sidecar per export stem: ``.{stem}.scene_data.json`` -- hierarchy
    baseline (+ optional ``last_diff`` record) and the ``data_export``
    channel snapshot (see module docstring).  The v2-era companions
    (``.prev`` backup, ``.hierarchy_diff.txt`` report) are no longer
    written; existing ones still read/migrate and are swept on write.

    When ``base_stem=True`` is passed to the path helpers, a trailing
    ``_v\\d+`` suffix is stripped so that all versioned exports of a
    series share a single sidecar (the record rolls forward to the most
    recent export).  Used when the Scene Exporter's Output Filename carries a
    ``{n}`` version counter (``run.versioned``).

    Subclass per host and override the scene hooks (module docstring); every
    other method is the one copy both DCC packages share.
    """

    # The one shared copy (ExportProfile's): anchored to end-of-stem so it
    # only matches genuine version suffixes, not mid-name occurrences like
    # 'arch_v2_proxy'.
    VERSION_SUFFIX_RE = ptk.ExportProfile.VERSION_SUFFIX_RE

    FORMAT_VERSION = 3

    MANIFEST_SUFFIX = "scene_data.json"
    # Format-v1 manifest name, still recognized for read/migration.
    LEGACY_MANIFEST_SUFFIX = "hierarchy.json"

    # ------------------------------------------------------------------
    # Path derivation
    # ------------------------------------------------------------------

    @classmethod
    def base_stem(cls, export_path: str) -> str:
        """Return the export stem with any trailing ``_vNN`` suffix stripped."""
        stem = os.path.splitext(os.path.basename(export_path))[0]
        return cls.VERSION_SUFFIX_RE.sub("", stem)

    @classmethod
    def _stem_for(cls, export_path: str, base_stem: bool) -> str:
        return (
            cls.base_stem(export_path)
            if base_stem
            else os.path.splitext(os.path.basename(export_path))[0]
        )

    @classmethod
    def manifest_path_for(cls, export_path: str, *, base_stem: bool = False) -> str:
        """Return the sidecar manifest path for an export file.

        Parameters:
            export_path: The export file path (e.g. ``shot_v003.fbx``).
            base_stem: If True, strip the version suffix so all versions of a
                series share one manifest. Opt-in: callers without an active
                versioning context should leave this False to preserve
                per-file sidecar behavior.
        """
        directory = os.path.dirname(export_path)
        stem = cls._stem_for(export_path, base_stem)
        return os.path.join(directory, f".{stem}.{cls.MANIFEST_SUFFIX}")

    @classmethod
    def _legacy_manifest_path_for(
        cls, export_path: str, *, base_stem: bool = False
    ) -> str:
        """Return the format-v1 (``.hierarchy.json``) manifest path."""
        directory = os.path.dirname(export_path)
        stem = cls._stem_for(export_path, base_stem)
        return os.path.join(directory, f".{stem}.{cls.LEGACY_MANIFEST_SUFFIX}")

    @classmethod
    def diff_report_path_for(cls, export_path: str, *, base_stem: bool = False) -> str:
        """Return the v2-era on-disk diff report path for an export file.

        v3 writes the report to the system temp dir instead (see
        :meth:`format_diff_report`); this derivation survives so leftover
        v2 reports can be swept and carried through renames.
        """
        directory = os.path.dirname(export_path)
        stem = cls._stem_for(export_path, base_stem)
        return os.path.join(directory, f".{stem}.hierarchy_diff.txt")

    # ------------------------------------------------------------------
    # Legacy sidecar migration
    # ------------------------------------------------------------------

    @classmethod
    def find_legacy_manifest(cls, export_path: str) -> Optional[str]:
        """Return the path of a legacy per-version sidecar to migrate from.

        Scans the export directory for ``.{base}_v<N>.scene_data.json`` and
        ``.{base}_v<N>.hierarchy.json`` files and returns the one with the
        highest version number (compared as integers, so unpadded ``_v10``
        ranks above ``_v2``; on a version tie the current naming wins).
        Returns None if no legacy sidecars exist.
        """
        directory = os.path.dirname(export_path)
        base = cls.base_stem(export_path)
        if not directory or not os.path.isdir(directory):
            return None
        # Group 2 ranks the current naming above the v1 naming on a tie.
        legacy_re = re.compile(
            rf"^\.{re.escape(base)}_v(\d+)\."
            rf"(?:({re.escape(cls.MANIFEST_SUFFIX)})|{re.escape(cls.LEGACY_MANIFEST_SUFFIX)})$",
            re.IGNORECASE,
        )
        matches = []
        for f in os.listdir(directory):
            m = legacy_re.match(f)
            if m:
                matches.append((int(m.group(1)), bool(m.group(2)), f))
        if not matches:
            return None
        _, _, name = max(matches, key=lambda t: (t[0], t[1]))
        return os.path.join(directory, name)

    @staticmethod
    def _safe_replace(src: str, dst: str) -> bool:
        """``os.replace`` that survives transient locks (cloud-sync, AV scans).

        A failed sidecar rename must never abort the export that triggered
        the migration — a locked file here used to propagate up and mislabel
        a *successful* FBX write as "Failed to export objects".  The caller
        keeps working with whichever file still exists; returns True when
        the rename actually happened.
        """
        try:
            ptk.FileUtils.replace_file(src, dst)
            return True
        except OSError as e:
            logging.getLogger(__name__).warning(
                "Sidecar rename failed (%s -> %s): %s", src, dst, e
            )
            return False

    @classmethod
    def _promote_stem(cls, export_path: str, *, base_stem: bool) -> Optional[str]:
        """Rename a same-stem v1-named manifest (and its ``.prev``) to the
        current name.  Returns the current-name path if a manifest exists
        after promotion, else None.  Idempotent.
        """
        new_path = cls.manifest_path_for(export_path, base_stem=base_stem)
        old_path = cls._legacy_manifest_path_for(export_path, base_stem=base_stem)
        if not os.path.exists(new_path) and os.path.exists(old_path):
            cls._safe_replace(old_path, new_path)
        # Carry a surviving .prev backup too — even when orphaned (manifest
        # deleted, backup intact): reads fall back to .prev until the next
        # write sweeps it, and that protection must survive the migration.
        if os.path.exists(old_path + ".prev") and not os.path.exists(
            new_path + ".prev"
        ):
            cls._safe_replace(old_path + ".prev", new_path + ".prev")
        return new_path if os.path.exists(new_path) else None

    @classmethod
    def ensure_base_name(cls, export_path: str) -> Optional[str]:
        """Migrate a legacy per-version manifest to the base-stem name.

        Called by callers that opt into ``base_stem=True``.  If a base-stem
        manifest already exists (either naming — a v1-named one is renamed),
        returns its path.  Otherwise, if a legacy ``_v\\d+`` manifest exists,
        renames it (and its ``.prev``) to the base-stem path and returns the
        new path.  Returns None if nothing was found.

        Idempotent: callers can invoke this on every check without harm.
        """
        promoted = cls._promote_stem(export_path, base_stem=True)
        if promoted:
            return promoted
        legacy = cls.find_legacy_manifest(export_path)
        if not legacy:
            return None
        new_path = cls.manifest_path_for(export_path, base_stem=True)
        if not cls._safe_replace(legacy, new_path):
            # Locked mid-migration — behave as if the legacy manifest wasn't
            # found (the check reports "no manifest" and a fresh one is
            # written after export) rather than raising into the exporter.
            return None
        if os.path.exists(legacy + ".prev"):
            cls._safe_replace(legacy + ".prev", new_path + ".prev")
        return new_path

    @classmethod
    def migrate_legacy(
        cls, export_path: str, *, base_stem: bool = False
    ) -> Optional[str]:
        """Idempotently bring on-disk sidecars up to the current naming.

        Promotes a same-stem v1-named ``.hierarchy.json`` manifest to
        ``.scene_data.json``; with ``base_stem=True`` additionally promotes
        the newest per-version manifest of either naming to the shared
        base-stem name (the version-series promotion).

        Returns the manifest path if one exists after migration, else None.
        The single migration entry point for exporter callers — invoke before
        every manifest read/write.
        """
        if base_stem:
            return cls.ensure_base_name(export_path)
        return cls._promote_stem(export_path, base_stem=False)

    # ------------------------------------------------------------------
    # Scene hooks (a DCC adapter overrides these) and path algebra
    # ------------------------------------------------------------------

    @staticmethod
    def expand_to_descendants(objects) -> list:
        """Scene hook: *objects* plus all their descendants, as host paths.

        The manifest has to capture the scope the host's export writes.  The
        default has no scene: *objects* are already paths, and they are all
        there is.
        """
        return list(objects)

    @staticmethod
    def drop_intermediate(nodes) -> list:
        """Scene hook: *nodes* minus construction data the export never ships.

        Maya parks intermediate shapes (``...ShapeOrig``) beside a deformed
        mesh; FBX writes the evaluated mesh only, so recording one made the
        hierarchy check fail after an innocuous edit.  The default has no such
        nodes.
        """
        return list(nodes)

    @staticmethod
    def build_clean_path_set(paths) -> set:
        """Scene hook: host paths -> the recorded spelling, deduplicated.

        The default records the paths as given; a host whose paths carry
        decoration the deliverable does not (Maya's leading ``|`` and
        namespaces) cleans it here.
        """
        return set(paths)

    @classmethod
    def _close(cls, paths) -> set:
        """What a recorded path set means when compared: *paths* by default.

        A host whose export ships more than the selected paths overrides this
        -- Maya's ``exportSelected`` writes every node's parent chain, so
        mayatk closes both sides under ancestors (:meth:`with_ancestors`); a
        host that ships exactly the set (Blender's ``use_selection``) keeps it.
        """
        return set(paths)

    @staticmethod
    def with_ancestors(paths) -> set:
        """*paths* closed under ancestors: ``A|B|C`` also yields ``A`` and ``A|B``.

        Idempotent.  The closure a host whose export writes every node's
        parent chain applies (Maya's ``exportSelected``, probe-verified for
        FBX and mayaAscii 2026-08-17: a group ships whether the group itself
        or only its leaves are in the export set).  Blender's
        ``use_selection`` drops unselected parents, so blendertk does not
        close (see :meth:`_close`).
        """
        closed = set()
        for path in paths:
            parts = path.split("|")
            for i in range(1, len(parts) + 1):
                closed.add("|".join(parts[:i]))
        return closed

    @staticmethod
    def get_top_level(paths) -> list:
        """Return only paths whose ancestor is *not* also in the set.

        Given ``|``-delimited hierarchy paths, keeps only the shallowest
        entries.
        The one implementation is ``ptk.HierarchyBaseline.top_level`` -- the
        same roots the per-scene hierarchy baseline scopes by, so a rollup in
        a report and a scope in a diff cannot disagree about what a root is.
        """
        return ptk.HierarchyBaseline.top_level(paths)

    @staticmethod
    def detect_reparenting(missing: list, extra: list) -> list:
        """Detect nodes that were reparented rather than added/removed.

        When a subtree is moved under a new parent every original path
        appears in *missing* and the same paths prefixed with the new
        parent appear in *extra*.  This method finds those patterns and
        returns a list of ``(root_missing, new_parent, count)`` tuples
        describing each reparenting.  Unmatched paths are ignored.
        """
        if not missing or not extra:
            return []

        extra_set = set(extra)
        missing_by_root = {}
        for p in missing:
            root = p.split("|")[0]
            missing_by_root.setdefault(root, []).append(p)

        # Unique top-level roots present in extra (avoids redundant checks)
        extra_roots = sorted({e.split("|")[0] for e in extra})

        results = []
        for root, paths in missing_by_root.items():
            for candidate in extra_roots:
                if all(f"{candidate}|{p}" in extra_set for p in paths):
                    results.append((root, candidate, len(paths)))
                    break

        return results

    # ------------------------------------------------------------------
    # Manifest I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _paths_hash(sorted_paths: list) -> str:
        """Return a stable SHA-256 hex digest for a sorted path list."""
        payload = "\n".join(sorted_paths).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _hierarchy_section(manifest: dict) -> dict:
        """Return the hierarchy section of a loaded manifest.

        Formats v2+ nest it under ``"hierarchy"``; a v1 manifest IS the
        hierarchy section (flat ``paths``/``object_count``/``hash``).
        """
        section = manifest.get("hierarchy")
        return section if isinstance(section, dict) else manifest

    @staticmethod
    def _set_hidden(path: str) -> None:
        """Best-effort ``FILE_ATTRIBUTE_HIDDEN`` on Windows.

        The dot-prefix convention hides nothing on Windows, and the sidecar
        sits in shared delivery folders — so the manifest carries the real
        attribute there (elsewhere the dot-prefix already does the job).
        ``os.replace`` hands the result the tmp file's attributes, so this
        must re-run after every write.  Never raises: visibility is
        cosmetic and must not break the export being recorded.
        """
        if os.name != "nt":
            return
        try:
            import ctypes

            FILE_ATTRIBUTE_HIDDEN = 0x2
            attrs = ctypes.windll.kernel32.GetFileAttributesW(path)
            if attrs != -1:
                ctypes.windll.kernel32.SetFileAttributesW(
                    path, attrs | FILE_ATTRIBUTE_HIDDEN
                )
        except Exception:
            pass

    @classmethod
    def _sweep_superseded(cls, export_path: str, *, base_stem: bool) -> None:
        """Remove companion files a v3 manifest write supersedes.

        The v2-era trio left ``.prev`` backups, v1-named manifests, and
        on-disk diff reports beside the deliverable; a fresh manifest is
        the current record of all of them, so folders self-clean as assets
        re-export.  Best-effort: a locked leftover just survives to the
        next sweep.
        """
        manifest_path = cls.manifest_path_for(export_path, base_stem=base_stem)
        legacy_path = cls._legacy_manifest_path_for(export_path, base_stem=base_stem)
        for stale in (
            manifest_path + ".prev",
            legacy_path,
            legacy_path + ".prev",
            cls.diff_report_path_for(export_path, base_stem=base_stem),
        ):
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                except OSError:
                    pass

    @classmethod
    def write_manifest(
        cls,
        export_path: str,
        paths,
        *,
        data: Optional[dict] = None,
        last_diff: Optional[dict] = None,
        base_stem: bool = False,
    ) -> Optional[str]:
        """Write the sidecar manifest for *export_path*.

        The write never displaces the live manifest: the new payload lands
        in a temp file first and atomically replaces the old one, so there
        is no window with no (or a partial) manifest on disk.  A successful
        write then sweeps superseded v2-era companions (``.prev`` backups,
        v1-named manifests, on-disk diff reports) and hides the manifest on
        Windows.

        Parameters:
            export_path: The export file the manifest accompanies.
            paths: Iterable of cleaned hierarchy path strings.
            data: Decoded ``data_export`` channel snapshot to record
                (``DataNodes.dump(decode=True)`` shape).  Omitted from the
                manifest when empty.
            last_diff: The hierarchy check's mismatch record for THIS export
                (``{"missing": [...], "extra": [...], "reparented": [...]}``),
                stored as ``hierarchy.last_diff``.  Pass None when the check
                matched or didn't run — the payload is rebuilt every write,
                so a clean export drops the previous record.
            base_stem: If True, strip the version suffix from the path
                derivation so all versions share one manifest.

        Returns:
            The manifest file path on success, ``None`` on failure.
        """
        manifest_path = cls.manifest_path_for(export_path, base_stem=base_stem)
        sorted_paths = sorted(paths)

        payload = {
            "format": cls.FORMAT_VERSION,
            "hierarchy": {
                "paths": sorted_paths,
                "object_count": len(sorted_paths),
                "hash": cls._paths_hash(sorted_paths),
            },
        }
        if last_diff:
            payload["hierarchy"]["last_diff"] = last_diff
        if data:
            # A sidecar next to the deliverable is a form of export, so it
            # records no authoring-machine paths (MeshConvert scrubs the GLB
            # the same way).
            payload["data_export"] = ptk.MeshConvert.without_locate_hints(data)

        # tmp-then-replace is not just atomicity: a hidden file rejects
        # open('w') outright on Windows, so the manifest can never be written
        # in place once _set_hidden has run (atomic_write_text replaces it).
        # Serialised here rather than by FileUtils.write_json, which takes
        # only indent/encoding/sort_keys: the manifest is indented so it can
        # be read by eye, so it keeps non-ASCII names unescaped
        # (ensure_ascii=False), and a rare non-JSON-native channel value is
        # written as a string (default=str, as DataNodes.format_dump does)
        # rather than failing the record.
        text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        try:
            ptk.FileUtils.atomic_write_text(manifest_path, text)
        except OSError:
            return None

        cls._set_hidden(manifest_path)
        cls._sweep_superseded(export_path, base_stem=base_stem)
        return manifest_path

    @classmethod
    def _load_manifest(cls, manifest_path: str) -> Optional[dict]:
        """Load manifest JSON, falling back to a surviving ``.prev`` backup.

        v3 no longer writes ``.prev`` files, but one left behind by a v2
        writer is still the last-known-good baseline — comparing against it
        beats silently passing until the next write sweeps it.  Returns
        None when neither file is readable (no baseline yet).
        """
        for path in (manifest_path, manifest_path + ".prev"):
            # ptk.FileUtils.read_json owns the tolerant catch set for the
            # ecosystem -- absent/unreadable (OSError) AND not-JSON
            # (ValueError, which JSONDecodeError subclasses). The local
            # try/except was a seventh independent re-derivation of it.
            record = ptk.FileUtils.read_json(path)
            if record is not None:
                return record
        return None

    @classmethod
    def read_manifest(
        cls, export_path: str, *, base_stem: bool = False
    ) -> Optional[Set[str]]:
        """Read the hierarchy paths from the manifest for *export_path*.

        Falls back to a surviving v2-era ``.prev`` backup when the manifest
        itself is missing or unreadable.  Reads all manifest formats.

        Returns:
            A set of hierarchy path strings, or ``None`` if neither the
            manifest nor a backup can be read.
        """
        data = cls._load_manifest(
            cls.manifest_path_for(export_path, base_stem=base_stem)
        )
        if data is None:
            return None
        return set(cls._hierarchy_section(data).get("paths", []))

    @classmethod
    def read_data(cls, export_path: str, *, base_stem: bool = False) -> Optional[dict]:
        """Read the ``data_export`` snapshot from the manifest for *export_path*.

        Returns:
            The decoded channel dict, or ``None`` when no manifest exists or
            it carries no data section (v1 manifests never do).
        """
        data = cls._load_manifest(
            cls.manifest_path_for(export_path, base_stem=base_stem)
        )
        if data is None:
            return None
        section = data.get("data_export")
        return section if isinstance(section, dict) else None

    # ------------------------------------------------------------------
    # Diff report
    # ------------------------------------------------------------------

    @staticmethod
    def count_descendants(top_path: str, all_paths) -> int:
        """Count *top_path* plus its descendants in *all_paths*."""
        path_set = all_paths if isinstance(all_paths, set) else set(all_paths)
        return sum(1 for p in path_set if p == top_path or p.startswith(top_path + "|"))

    @classmethod
    def _format_top_level_section(cls, prefix: str, all_paths: list) -> list:
        """Build lines showing top-level parents with descendant counts.

        Returns a list of formatted strings for a single report section.
        """
        top = cls.get_top_level(all_paths)
        path_set = set(all_paths)
        lines = [f"{prefix} ({len(all_paths)} nodes, {len(top)} top-level):\n"]
        for t in top:
            count = cls.count_descendants(t, path_set)
            if count > 1:
                lines.append(f"  {t}  ({count} nodes)\n")
            else:
                lines.append(f"  {t}\n")
        lines.append("\n")
        return lines

    @classmethod
    def format_diff_report(
        cls, missing: list, extra: list, reparented: list = None
    ) -> str:
        """Return the human-readable hierarchy diff report as text.

        The report contains a summary with top-level rollups followed by
        a full path listing.  Reparenting patterns are called out at the
        top.  Callers decide where it goes — the exporter writes it to a
        temp artifact and links it from the log, never into the export
        folder (the structured record lives in the manifest's
        ``hierarchy.last_diff``).

        Parameters:
            missing: Paths present in manifest but absent in current scene.
            extra: Paths in current scene but absent from manifest.
            reparented: Pre-computed reparenting tuples from
                ``detect_reparenting``.  Computed on-demand if ``None``.
        """
        lines = []
        lines.append("Hierarchy Diff Report\n")
        lines.append("=" * 60 + "\n\n")

        # Summary
        lines.append("Summary\n")
        lines.append("-" * 40 + "\n")
        lines.append(f"  Missing:  {len(missing)}\n")
        lines.append(f"  Extra:    {len(extra)}\n")
        lines.append(f"  Total:    {len(missing) + len(extra)}\n")
        lines.append("\n")

        if reparented is None:
            reparented = cls.detect_reparenting(missing, extra)
        if reparented:
            for root, parent, count in reparented:
                lines.append(
                    f"Reparented: '{root}' moved under '{parent}' ({count} nodes)\n"
                )
            lines.append("\n")

        # Top-level rollup
        if missing:
            lines.extend(cls._format_top_level_section("Missing", missing))
        if extra:
            lines.extend(cls._format_top_level_section("Extra", extra))

        # Full path listing
        if missing or extra:
            lines.append("-" * 60 + "\n")
            lines.append("Full Path Listing\n")
            lines.append("-" * 60 + "\n\n")
        if missing:
            lines.append(f"All missing ({len(missing)}):\n")
            for p in missing:
                lines.append(f"  - {p}\n")
            lines.append("\n")
        if extra:
            lines.append(f"All extra ({len(extra)}):\n")
            for p in extra:
                lines.append(f"  + {p}\n")
        return "".join(lines)

    @classmethod
    def clean_stale_diff(cls, export_path: str, *, base_stem: bool = False) -> None:
        """Remove a leftover v2-era on-disk diff report.

        v3 never writes the report into the export folder, but one left by
        a v2 writer (or by an export aborted before its write could sweep)
        should disappear the moment a check passes.
        """
        diff_path = cls.diff_report_path_for(export_path, base_stem=base_stem)
        if os.path.exists(diff_path):
            try:
                os.remove(diff_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # High-level: full build + compare
    # ------------------------------------------------------------------

    @classmethod
    def build_full_path_set(cls, objects) -> set:
        """Expand *objects* to descendants, drop intermediates, clean, dedupe.

        The result is the hierarchy the export ships, through the scene hooks
        (:meth:`expand_to_descendants`, :meth:`drop_intermediate`,
        :meth:`build_clean_path_set`).
        """
        return cls.build_clean_path_set(
            cls.drop_intermediate(cls.expand_to_descendants(objects))
        )

    @classmethod
    def compare(
        cls,
        export_path: str,
        current_paths: set,
        *,
        base_stem: bool = False,
    ) -> Tuple[bool, list, list]:
        """Compare *current_paths* against the stored hierarchy baseline.

        Uses the stored hash for a fast-path equality check before
        falling back to a full set diff.  When the manifest is missing or
        unreadable a surviving v2-era ``.prev`` backup is used as the
        baseline; with neither present the check passes (no baseline yet —
        the exporter's check surfaces the unreadable-manifest case as a
        warning).  Reads all manifest formats; the data and ``last_diff``
        sections are ignored.

        Parameters:
            export_path: The export file whose manifest to compare against.
            current_paths: Set of cleaned hierarchy paths from the current
                scene.
            base_stem: If True, route manifest lookup through the version
                base-stem so the diff baseline rolls forward across versions.

        Returns:
            ``(match, missing, extra)`` where *match* is ``True`` when
            the hierarchies are identical.
        """
        manifest_path = cls.manifest_path_for(export_path, base_stem=base_stem)
        data = cls._load_manifest(manifest_path)
        if data is None:
            return True, [], []
        section = cls._hierarchy_section(data)

        # Fast-path: compare hashes before doing the full set diff
        stored_hash = section.get("hash")
        if stored_hash:
            current_hash = cls._paths_hash(sorted(current_paths))
            if stored_hash == current_hash:
                return True, [], []

        # Both sides mean what the host ships (_close): a manifest written
        # before a host's closure rule carries only what was selected, and a
        # caller may hand over a raw set.
        previous = cls._close(section.get("paths", []))
        current = cls._close(current_paths)
        missing = sorted(previous - current)
        extra = sorted(current - previous)
        return (not missing and not extra), missing, extra
