# !/usr/bin/python
# coding=utf-8
"""What a texture file IS and which set it belongs to: map-type and colour-space
resolution, base names, UDIM/UV tiles, and grouping files into texture sets
(the bodies behind the :class:`MapFactory` facade).
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from pythontk.img_utils._img_utils import ImgUtils
from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.tiled_path import TiledPath
from pythontk.iter_utils._iter_utils import IterUtils
from pythontk.str_utils._str_utils import StrUtils
from pythontk.core_utils.naming_convention import NamingConvention
from pythontk.core_utils.engines.textures.map_registry import MapRegistry


class _TextureSetInternal:
    """Bodies of :class:`MapFactory`'s file classification and set grouping.

    The public signatures and docstrings stay on :class:`MapFactory`; each
    delegates here. A ``MapFactory`` base, so ``cls`` is the factory (or a
    subclass) and its registries and helpers resolve through the MRO.
    """

    @classmethod
    def _get_aliases_by_len_desc(cls) -> List[str]:
        """Every alias across all map types, longest-first.

        Used by `resolve_map_type(key=False)`. The cache lives on the registry
        (the alias owner) so registration invalidates it with the other views.
        """
        return cls._map_registry.get_aliases_by_len_desc()

    @classmethod
    def resolve_map_type(cls, file: str, key: bool = True, validate: str = None) -> str:
        """Body of :meth:`MapFactory.resolve_map_type`."""
        ImgUtils.assert_pathlike(file, "file")
        filename = FileUtils.format_path(file, "name")

        if key:
            result = cls._map_registry.resolve_type_from_path(file)
        else:
            filename_lower = filename.lower()
            separators = cls._map_registry.SEPARATORS
            result = None
            for alias in cls._get_aliases_by_len_desc():
                alias_lower = alias.lower()
                if filename_lower == alias_lower:
                    result = filename
                    break
                # Any registry separator counts, not `_` alone: this path fed
                # `resolve_texture_filename`, so a `-`/`.`/space-delimited suffix
                # was invisible here while the other two suffix implementations
                # accepted it, and the round-trip renamed the file.
                start = len(filename) - len(alias)
                if (
                    start > 0
                    and filename_lower.endswith(alias_lower)
                    and filename[start - 1] in separators
                ):
                    # Slice the alias out of the original filename to preserve case
                    result = filename[start:]
                    break

        if validate:
            # Case-insensitive: `result` may carry filename casing (key=False)
            # which won't match the canonical-cased registry entries verbatim.
            valid_types_lower = {validate.lower()} | {
                a.lower() for a in cls.map_types[validate]
            }
            if (result or "").lower() not in valid_types_lower:
                raise ValueError(
                    f"Invalid map type '{result}'. Expected type is one of: "
                    f"{[validate] + list(cls.map_types[validate])}"
                )

        return result

    @classmethod
    def resolve_color_space(cls, file: str, default: str = "Linear") -> str:
        """Body of :meth:`MapFactory.resolve_color_space`."""
        map_type = cls._map_registry.resolve_type_from_path(file)
        entry = cls._map_registry.get(map_type) if map_type else None
        return entry.color_space if entry else default

    @classmethod
    def resolve_texture_filename(
        cls,
        texture_path: str,
        map_type: str,
        prefix: str = None,
        suffix: str = None,
        ext: str = None,
    ) -> str:
        """Body of :meth:`MapFactory.resolve_texture_filename`."""
        ImgUtils.assert_pathlike(texture_path, "texture_path")

        # If no map type was resolved, we can't safely synthesize a "<base>_<type>"
        # filename without dropping naming detail. Preserve the original path
        # (changing extension if explicitly requested via `ext`).
        if not map_type:
            directory = FileUtils.format_path(texture_path, "path")
            stem, original_ext = os.path.splitext(os.path.basename(texture_path))
            ext_out = f".{ext.lower().lstrip('.')}" if ext else original_ext
            # Idempotent affix application: strip the configured prefix/suffix from
            # the existing stem before re-applying, so "Optimized_foo" + prefix
            # "Optimized_" stays "Optimized_foo" (not "Optimized_Optimized_foo").
            stem_core = StrUtils.strip_known_affix(
                stem, prefix=prefix or "", suffix=suffix or ""
            )
            prefix_str = prefix or ""
            suffix_str = f"_{suffix.lstrip('_')}" if suffix else ""
            return os.path.join(
                directory, f"{prefix_str}{stem_core}{suffix_str}{ext_out}"
            )

        # Extract sections from the given path
        directory = FileUtils.format_path(texture_path, "path")
        # Strip the configured prefix/suffix from the base so we can re-apply
        # them idempotently below.
        base_name = cls.get_base_texture_name(
            texture_path, prefix=prefix or "", suffix=suffix or ""
        )
        original_ext = FileUtils.format_path(texture_path, "ext")

        # Ensure map_type does not start with an underscore
        map_type = map_type.lstrip("_")

        # Ensure suffix formatting (prevents double underscores)
        suffix = f"_{suffix.lstrip('_')}" if suffix else ""

        # Determine output file extension (preserve original unless explicitly changed)
        ext = f".{ext.lower().lstrip('.')}" if ext else f".{original_ext}"

        # A tile token trails the map type (matching TextureProcessor.output_path_for)
        # so the output is still a readable tile sequence AND two tiles of the same
        # material resolve to different paths instead of overwriting each other.
        tile_token = cls.get_tile_token(texture_path)

        # Construct the final filename correctly
        new_name = StrUtils.replace_placeholders(
            "{prefix}{base_name}_{map_type}{suffix}{tile}{ext}",
            prefix=prefix or "",
            base_name=base_name,
            map_type=map_type,
            suffix=suffix,
            tile=tile_token,
            ext=ext,
        )

        return os.path.join(directory, new_name)

    @classmethod
    def get_base_texture_name(
        cls,
        filepath_or_filename: str,
        prefix: str = "",
        suffix: str = "",
    ) -> str:
        """Body of :meth:`MapFactory.get_base_texture_name`."""
        ImgUtils.assert_pathlike(filepath_or_filename, "filepath_or_filename")

        filename = os.path.basename(str(filepath_or_filename))
        base_name, _ = os.path.splitext(filename)

        registry = MapRegistry()

        # The registry owns the taxonomy AND the order the two tokens come off in
        # (tile first, or the suffix is unreachable). The tail it returns is
        # dropped rather than restored: this is the MATERIAL's name (and the
        # texture-set key's stem), which every tile of a set shares.
        # `get_tile_token` reads the token back for output naming;
        # `group_textures_by_set` re-appends it to keep tiles separable.
        base_name, _tail = registry.split_map_suffix(base_name)

        # Strip any configured user prefix/suffix so callers can re-apply them
        # idempotently, then collapse a trailing delimiter (preserves the
        # original behavior for filenames like 'foo_.png' even when no affix
        # was supplied). Every delimiter, not just `_`: the suffix rules accept
        # all of `SEPARATORS`, so collapsing one of them and leaving the rest
        # put 'rock-' and 'rock' in different texture sets.
        return StrUtils.strip_known_affix(
            base_name, prefix=prefix, suffix=suffix
        ).rstrip(registry.SEPARATORS)

    @classmethod
    def _split_reading_types(cls) -> set:
        """The map types a compound spelling can be misread as: the lobe types,
        and Emissive, whose ``Emission_Color`` / ``Emissive_Color`` (2026-10-10)
        read a set named ``<x>_Emission``'s base colour as an emission colour."""
        return set(cls._map_registry.LOBE_TYPE_CHANNELS) | {"Emissive"}

    @classmethod
    def _base_map_reading(
        cls, path: str, prefix: str = "", suffix: str = ""
    ) -> Optional[Tuple[str, str]]:
        """The base-map reading of a lobe-named file, or None when it has none.

        ``Hero_Coat_Roughness`` reads per file as a coat roughness of set
        ``Hero``; its base-map reading is the roughness of a set named
        ``Hero_Coat``. The matched map suffix is split after each separator
        inside it, shortest lobe prefix first, until the remainder is WHOLLY a
        map-type suffix of a non-lobe type: ``Clear_Coat_Roughness`` passes
        ``Coat_Roughness`` (still a lobe) and stops at ``Roughness``. The lobe
        word(s) join the base exactly as spelled; the configured affixes and a
        trailing separator come off as :meth:`get_base_texture_name` takes them.

        Parameters:
            path: A texture path or file name.
            prefix: User prefix stripped from the base.
            suffix: User suffix stripped from the base.

        Returns:
            ``(map_type, base)`` for the base-map reading, or ``None`` when the
            suffix has no lobe-word + base-map split.
        """
        registry = cls._map_registry
        name_only = os.path.splitext(os.path.basename(str(path)))[0]
        # The tile token never joins the base: off before the suffix is split.
        stem, _tile = registry.split_tile_token(name_only)
        base, _tail = registry.split_map_suffix(stem)
        split_types = cls._split_reading_types()
        for cut in range(len(base) + 1, len(stem)):
            if stem[cut] not in registry.SEPARATORS:
                continue
            rest = stem[cut:]  # the separator + the remainder
            if registry.split_map_suffix(rest)[0]:
                continue  # the remainder is not wholly a map-type suffix
            # A dummy extension: `resolve_type_from_path` splits one off, and a
            # remainder must never lose a dotted token to it.
            map_type = registry.resolve_type_from_path(f"{rest}.png")
            if map_type and map_type not in split_types:
                return map_type, StrUtils.strip_known_affix(
                    stem[:cut], prefix=prefix, suffix=suffix
                ).rstrip(registry.SEPARATORS)
        return None

    @classmethod
    def classify_textures(
        cls,
        paths: Iterable[str],
        prefix: str = "",
        suffix: str = "",
    ) -> Dict[str, Tuple[Optional[str], str]]:
        """Body of :meth:`MapFactory.classify_textures`."""
        readings: Dict[str, Tuple[Optional[str], str]] = {}
        alternatives: Dict[str, Tuple[str, str]] = {}
        split_types = cls._split_reading_types()
        for path in paths:
            if path in readings:
                continue
            map_type = cls.resolve_map_type(path)
            readings[path] = (
                map_type,
                cls.get_base_texture_name(path, prefix=prefix, suffix=suffix),
            )
            if map_type in split_types:
                alternative = cls._base_map_reading(path, prefix, suffix)
                if alternative:
                    alternatives[path] = alternative
        if not alternatives:
            return readings

        # The sets the unambiguous maps name. A lobe reading needs its set to
        # exist apart from the ambiguous files (an OpenPBR set has a base map);
        # a base-map reading wins whenever its own set exists, or the lobe's
        # does not (lobe maps with no base map are not a material).
        anchors = {
            base
            for path, (map_type, base) in readings.items()
            if map_type and path not in alternatives
        }
        for path, alternative in alternatives.items():
            if alternative[1] in anchors or readings[path][1] not in anchors:
                readings[path] = alternative
        return readings

    @classmethod
    def resolve_map_types(cls, paths: Iterable[str]) -> Dict[str, Optional[str]]:
        """Body of :meth:`MapFactory.resolve_map_types`."""
        return {
            path: map_type
            for path, (map_type, _base) in cls.classify_textures(paths).items()
        }

    @classmethod
    def _set_base_name(cls, path: str, neighbours: Iterable[Any] = ()) -> str:
        """*path*'s base name read against the other paths of its set.

        For a writer that names one output after a source it holds together
        with that source's set-mates (a channel pack): non-path entries (an
        in-memory image) are skipped.

        Parameters:
            path: The source the output is named after.
            neighbours: The other sources, in any form.

        Returns:
            str: The base name :meth:`classify_textures` gives *path*.
        """
        if not isinstance(path, str):
            return cls.get_base_texture_name(path)  # its own pathlike check
        batch = [path] + [n for n in neighbours if isinstance(n, str) and n != path]
        return cls.classify_textures([n for n in batch if n])[path][1]

    @classmethod
    def get_tile_token(cls, filepath_or_filename: str) -> str:
        """Body of :meth:`MapFactory.get_tile_token`."""
        ImgUtils.assert_pathlike(filepath_or_filename, "filepath_or_filename")

        filename = os.path.basename(str(filepath_or_filename))
        name_only, _ = os.path.splitext(filename)
        return cls._map_registry.split_tile_token(name_only)[1]

    @classmethod
    def get_tile_paths(cls, filepath: str) -> List[str]:
        """Body of :meth:`MapFactory.get_tile_paths`."""
        ImgUtils.assert_pathlike(filepath, "filepath")

        filepath = str(filepath)
        filename = os.path.basename(filepath)
        folder = filepath[: len(filepath) - len(filename)]
        stem, ext = os.path.splitext(filename)
        head, token = cls._map_registry.split_tile_token(stem)
        if not token:
            return []
        tile = (
            "u[0-9]+_v[0-9]+"
            if TiledPath.scheme(token[1:]) == "uvtile"
            else "1[0-9]{3}"
        )
        # Names compare the way the filesystem compares them: case-blind on Windows.
        fold = os.path.normcase
        sibling = re.compile(
            re.escape(fold(head + token[0])) + tile + re.escape(fold(ext)) + r"\Z"
        )
        try:
            names = os.listdir(folder or ".")
        except OSError:
            return []
        return sorted(folder + name for name in names if sibling.match(fold(name)))

    @classmethod
    def group_textures_by_set(
        cls,
        image_paths: List[str],
        prefix: str = "",
        suffix: str = "",
    ) -> Dict[str, List[str]]:
        """Body of :meth:`MapFactory.group_textures_by_set`."""
        paths = list(image_paths)
        return cls._group_by_reading(
            paths, cls.classify_textures(paths, prefix=prefix, suffix=suffix)
        )

    @classmethod
    def _group_by_reading(
        cls, paths: List[str], readings: Dict[str, Tuple[Optional[str], str]]
    ) -> Dict[str, List[str]]:
        """``{<base><tile token>: paths}`` from a :meth:`classify_textures` result."""
        texture_sets: Dict[str, List[str]] = {}
        for path in paths:
            key = f"{readings[path][1]}{cls.get_tile_token(path)}"
            texture_sets.setdefault(key, []).append(path)
        return texture_sets

    @classmethod
    def collapse_tile_sets(
        cls, texture_sets: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Body of :meth:`MapFactory.collapse_tile_sets`."""
        merged: Dict[str, Dict[str, str]] = {}
        for key, paths in texture_sets.items():
            maps = merged.setdefault(cls._map_registry.split_tile_token(key)[0], {})
            for path in paths:
                stem, ext = os.path.splitext(os.path.basename(str(path)))
                map_name = cls._map_registry.split_tile_token(stem)[0] + ext.lower()
                if map_name not in maps or str(path) < maps[map_name]:
                    maps[map_name] = str(path)
        return {base: sorted(maps.values()) for base, maps in merged.items()}

    @classmethod
    def dominant_texture_set(cls, paths: Iterable[str]) -> Optional[Tuple[str, str]]:
        """Body of :meth:`MapFactory.dominant_texture_set`."""
        texts = [text for text in (str(path or "") for path in paths or ()) if text]
        readings = cls.classify_textures(texts)
        votes: List[Tuple[str, str]] = []
        for text in texts:
            map_type, base = readings[text]
            if not map_type:
                continue  # not a material map
            if base:
                folder = os.path.dirname(text)
                votes.append((base, os.path.normpath(folder) if folder else ""))
        if not votes:
            return None

        def majority(values: List[str]) -> str:
            counts: Dict[str, int] = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            return max(sorted(counts), key=counts.get)

        base = majority([b for b, _folder in votes])
        return base, majority([f for b, f in votes if b == base])

    @staticmethod
    def lightmap_for_base(map_name: str, old_base: str, new_base: str) -> Optional[str]:
        """Body of :meth:`MapFactory.lightmap_for_base`."""
        stem, ext = os.path.splitext(os.path.basename(str(map_name or "")))
        prefix, suffix = NamingConvention.affix_parts("lightmap")
        match = re.fullmatch(
            rf"({re.escape(prefix)}){re.escape(old_base)}({re.escape(suffix)}[_-]*\d*)",
            stem,
            flags=re.IGNORECASE,
        )
        return f"{match[1]}{new_base}{match[2]}{ext}" if match else None

    @classmethod
    def _supplement_sets_from_dir(
        cls,
        texture_sets: Dict[str, List[str]],
        directory: str,
        prefix: str = "",
        suffix: str = "",
        logger=None,
    ) -> Dict[str, List[str]]:
        """Gap-fill each texture set with same-base-name siblings from ``directory``.

        For every set, scans ``directory`` **and its subdirectories** for files
        that resolve to the same base
        name (honoring ``prefix``/``suffix``) and a recognized map type, then appends
        any whose map type is missing from the set. Provided files always win — an
        existing map slot is never replaced and files already present are not
        duplicated. Lets callers pull in required maps that live next to the inputs
        but weren't part of the supplied list (e.g. a Normal sitting in a project's
        ``sourceimages`` that was never wired into the material).

        Parameters:
            texture_sets: Mapping of base name -> file paths (mutated in place).
            directory: Directory to scan for sibling textures.
            prefix: Prefix stripped during base-name resolution (must match set keys).
            suffix: Suffix stripped during base-name resolution.
            logger: Optional logger for reporting discovered files.

        Returns:
            Dict[str, List[str]]: The same ``texture_sets`` mapping, supplemented.
        """
        log = logger or cls.logger
        if not (directory and os.path.isdir(directory)):
            return texture_sets

        # Recursive: the caller hands over a texture ROOT (a Maya project's
        # ``sourceImages`` rule, say), and that root is routinely one folder
        # per asset rather than a flat pile. A root-only scan finds nothing in
        # that layout -- the maps sitting beside the ones already wired are one
        # level down -- so discovery silently no-ops exactly where it is needed.
        # Matching stays base-name + map-type, so depth widens the search
        # without widening what can match.
        dir_files = FileUtils.get_dir_contents(
            directory,
            "filepath",
            recursive=True,
            inc_files=[f"*.{ext}" for ext in ImgUtils.texture_file_types],
        )
        if not dir_files:
            return texture_sets

        # The directory is one batch: its files read against each other, so a
        # set named for a lobe (``Hero_Coat``) finds its maps under that name.
        dir_readings = cls.classify_textures(dir_files, prefix=prefix, suffix=suffix)
        dir_by_set = cls._group_by_reading(dir_files, dir_readings)

        for base_name, files in texture_sets.items():
            siblings = dir_by_set.get(base_name)
            if not siblings:
                continue

            present_types = set(cls.resolve_map_types(files).values())
            present_paths = {os.path.normcase(os.path.abspath(f)) for f in files}

            for sib in siblings:
                key = os.path.normcase(os.path.abspath(sib))
                if key in present_paths:
                    continue
                map_type = dir_readings[sib][0]
                if not map_type or map_type in present_types:
                    continue

                files.append(sib)
                present_types.add(map_type)
                present_paths.add(key)
                if log:
                    log.info(
                        f"Discovered {os.path.basename(sib)} ({map_type}) "
                        f"for set '{base_name}'"
                    )

        return texture_sets

    @classmethod
    def filter_images_by_type(cls, files, types=""):
        """Body of :meth:`MapFactory.filter_images_by_type`."""
        types = IterUtils.make_iterable(types)
        files = list(files)
        by_path = cls.resolve_map_types(files)
        return [f for f in files if by_path[f] in types]

    @classmethod
    def sort_images_by_type(
        cls, files: Union[List[Union[str, Tuple[str, Any]]], Dict[str, Any]]
    ) -> Dict[str, List[Union[str, Tuple[str, Any]]]]:
        """Body of :meth:`MapFactory.sort_images_by_type`."""
        if isinstance(files, dict):
            # Convert dictionary to list of tuples
            files = list(files.items())

        files = list(files)
        # Read as one batch, so a set named for a lobe keeps its base maps.
        by_path = cls.resolve_map_types(
            file[0] if isinstance(file, tuple) else file for file in files
        )
        sorted_images = {}
        for file in files:
            # Determine if the input is a path or a tuple of (path, file data)
            is_tuple = isinstance(file, tuple)

            file_path = file[0] if is_tuple else file
            map_type = by_path[file_path]
            if not map_type:
                continue

            if map_type not in sorted_images:
                sorted_images[map_type] = []

            # Add the file to the sorted list according to its input format
            sorted_images[map_type].append(file if is_tuple else file_path)

        return sorted_images

    @classmethod
    def contains_map_types(cls, files, map_types):
        """Body of :meth:`MapFactory.contains_map_types`."""
        if isinstance(files, (list, set, tuple)):
            # convert list to dict of the correct format.
            files = cls.sort_images_by_type(files)

        map_types = IterUtils.make_iterable(map_types)

        result = next(
            (True for i in files.keys() if cls.resolve_map_type(i) in map_types),
            False,
        )

        return True if result else False

    @classmethod
    def is_normal_map(cls, file):
        """Body of :meth:`MapFactory.is_normal_map`."""
        typ = cls.resolve_map_type(file)
        return any(
            (
                typ in cls.map_types["Normal_DirectX"],
                typ in cls.map_types["Normal_OpenGL"],
                typ in cls.map_types["Normal"],
            )
        )
