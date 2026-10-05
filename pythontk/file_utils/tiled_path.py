# !/usr/bin/python
# coding=utf-8
"""Tile / frame tokens in a file path -- the one vocabulary every host reads.

A texture path can name a SET of files rather than one: ``rock.<UDIM>.png`` is
every UDIM tile beside it, ``seq.<f>.exr`` every frame of a sequence.  Hosts
spell the set with a placeholder token (Maya writes ``<UDIM>``, Mudbox
``<u>_<v>``, Blender stores a ``u#_v#`` set as ``<UVTILE>``), and every tool
that touches such a path asks the same four questions of it: does it name a
set (:meth:`TiledPath.has_token`), is that set frames or tiles
(:meth:`TiledPath.is_frame_sequence`), which one real file stands for it
(:meth:`TiledPath.representative`), and which files are in it
(:meth:`TiledPath.tiles`).

One table answers all four (:attr:`TiledPath.TOKENS`).  Before it existed the
answer depended on which copy asked: four vocabularies had grown across the
DCC packages, one knew six tokens and another two, and a ``<u>_<v>`` or
``<frame>`` path skipped a scene exporter's representative collapse because
its private regex listed three.
"""

import glob
import os
import re
from typing import Dict, List, Optional, Tuple


class TiledPath:
    """The tile / frame token vocabulary of a file path, as one table.

    Every method is a classmethod over plain path strings.  The pure tests
    (:meth:`has_token`, :meth:`is_frame_sequence`, :meth:`scheme`,
    :meth:`wildcard`, :meth:`spell`) never touch the disk; :meth:`tiles` and
    :meth:`representative` glob it.
    """

    #: Token (lower case) -> ``(stand-in, glob)``.
    #:
    #: The stand-in is the one concrete value a FIXED token collapses to --
    #: the set's first tile in its OWN numbering: ``<udim>`` is ``1001`` but
    #: ``<uvtile>`` is ``u1_v1``, since UDIM and UV-tile numbering are not
    #: interchangeable and a probe must name the very file a representative
    #: collapse names.  A frame counter has no fixed first value (numbering,
    #: padding and start frame all vary per render), so its stand-in is
    #: ``None`` and it is GLOBBED instead.
    #:
    #: The glob is what the token expands to when a caller needs the whole
    #: set, as specific as the stand-in beside it: a blanket ``*`` would make
    #: ``tex.<UDIM>.png`` collect ``tex.u1_v1.png`` and conflate the two tile
    #: vocabularies.  UDIM tiles are always four digits; the UV-tile spellings
    #: compose from ``u`` / ``v``; frame padding varies, so nothing narrower
    #: than ``*`` is safe there.
    TOKENS: Dict[str, Tuple[Optional[str], str]] = {
        "<udim>": ("1001", "[0-9][0-9][0-9][0-9]"),  # Mari / Maya uvTilingMode 3
        "<uvtile>": ("u1_v1", "u*_v*"),  # ZBrush / Mudbox; Blender's TILED u#_v#
        "<u>": ("u1", "u*"),  # ``<u>_<v>`` composes from these two
        "<v>": ("v1", "v*"),
        "<f>": (None, "*"),  # frame sequence -- no fixed first frame
        "<frame>": (None, "*"),
    }

    #: Longest-first alternation, so a composite token is never shadowed by
    #: one of its prefixes (``<uvtile>`` is not ``<u>``).  Case-insensitive:
    #: Maya writes ``<UDIM>``, hand-edited and exchanged paths carry every
    #: casing.
    TOKEN_RE = re.compile(
        "|".join(sorted(TOKENS, key=len, reverse=True)), re.IGNORECASE
    )

    #: A concrete UV tile as a file name spells one (``u1_v1``).
    _UV_TILE_RE = re.compile(r"u[0-9]+_v[0-9]+\Z", re.IGNORECASE)
    #: A concrete UDIM tile (``1001``-``1999``).
    _UDIM_RE = re.compile(r"1[0-9]{3}\Z")

    @classmethod
    def has_token(cls, path: Optional[str]) -> bool:
        """Does *path* carry a tile / frame token -- does it name a SET?

        A pure string test, no disk access: a frame pattern with nothing on
        disk is still a token path, which is the answer a caller asking "can
        I treat this as ONE file?" needs.

        Parameters:
            path: A stored path, or just its basename.
        """
        return bool(path) and bool(cls.TOKEN_RE.search(path))

    @classmethod
    def is_frame_sequence(cls, path: Optional[str]) -> bool:
        """Does *path*'s token count FRAMES (``<f>`` / ``<frame>``), not tiles?

        The split a cost has to make: a tile set is loaded whole, a frame
        sequence one frame at a time.  A token with no fixed stand-in in
        :attr:`TOKENS` is a frame counter.  A pure string test.

        Parameters:
            path: A stored path, or just its basename.
        """
        return any(
            cls.TOKENS[match.group(0).lower()][0] is None
            for match in cls.TOKEN_RE.finditer(path or "")
        )

    @classmethod
    def scheme(cls, spelling: Optional[str]) -> Optional[str]:
        """The numbering a tile spelling uses: ``"udim"``, ``"uvtile"`` or ``"frame"``.

        Reads both a placeholder token (``<UDIM>``, ``<uvtile>``, ``<u>_<v>``,
        ``<f>``) and a concrete tile as a file name spells one (``1001``,
        ``u1_v1``), so a host deciding how to TILE an image asks one place.

        Parameters:
            spelling: A token or tile, without its leading separator.

        Returns:
            ``None`` when *spelling* is neither.
        """
        text = (spelling or "").strip().lower()
        if not text:
            return None
        if cls._UDIM_RE.match(text):
            return "udim"
        if cls._UV_TILE_RE.match(text):
            return "uvtile"
        tokens = [m.group(0).lower() for m in cls.TOKEN_RE.finditer(text)]
        if not tokens or cls.TOKEN_RE.sub("", text).strip("._-"):
            return None
        if any(cls.TOKENS[t][0] is None for t in tokens):
            return "frame"
        return "udim" if tokens == ["<udim>"] else "uvtile"

    @classmethod
    def tile_token_pattern(cls) -> str:
        """A regex alternation matching a whole TILE placeholder token.

        ``<udim>`` / ``<uvtile>`` / ``<u>_<v>`` -- the placeholders that stand
        for one tile set, for a caller splitting a trailing tile token off a
        file name (``MapRegistry.split_tile_token``).  Frame tokens are not
        tiles and are left out.  Compile it case-insensitively.
        """
        fixed = [t for t, (stand_in, _) in cls.TOKENS.items() if stand_in]
        composite = [t for t in fixed if t not in ("<u>", "<v>")] + ["<u>_<v>"]
        return "|".join(re.escape(t) for t in sorted(composite, key=len, reverse=True))

    @classmethod
    def wildcard(cls, path: Optional[str], wildcard: Optional[str] = "*") -> str:
        """*path* with every token replaced by *wildcard*.

        The name-matching companion to :meth:`tiles`: a plain pattern for
        matching NAMES already in hand (an ``fnmatch`` over an index, a
        display string), with no directory to escape and no disk to touch.

        Parameters:
            path: A stored path, or just its basename.
            wildcard: What each token becomes (``"*"`` suits ``fnmatch``).
                ``None`` substitutes each token's OWN glob from
                :attr:`TOKENS` and glob-escapes the literal segments, so a
                match is exactly as strict as the disk glob :meth:`tiles` runs
                -- ``rock.<UDIM>.png`` never collects ``rock.thumb.png``.
        """
        path = path or ""
        if wildcard is not None:
            return cls.TOKEN_RE.sub(wildcard, path)
        pattern: List[str] = []
        cursor = 0
        for match in cls.TOKEN_RE.finditer(path):
            pattern.append(
                glob.escape(path[cursor : match.start()])
                + cls.TOKENS[match.group(0).lower()][1]
            )
            cursor = match.end()
        pattern.append(glob.escape(path[cursor:]))
        return "".join(pattern)

    @classmethod
    def spell(cls, path: Optional[str], tile: int = 1001) -> str:
        """*path* with each TILE token spelled for UDIM tile number *tile*.

        ``<UDIM>`` becomes the number, ``<uvtile>`` its ``u#_v#`` (1-based:
        1001 is ``u1_v1``, 1012 is ``u2_v2``), ``<u>`` / ``<v>`` their halves.
        A frame token has no tile and is left as it is.  No disk access: for
        a host that knows which tile it wants (an image's first declared
        tile); :meth:`representative` finds one on disk instead.

        Parameters:
            path: A stored path.
            tile: A UDIM tile number.
        """
        offset = int(tile) - 1001
        u, v = offset % 10 + 1, offset // 10 + 1
        spelled = {
            "<udim>": str(int(tile)),
            "<uvtile>": "u%d_v%d" % (u, v),
            "<u>": "u%d" % u,
            "<v>": "v%d" % v,
        }
        return cls.TOKEN_RE.sub(
            lambda m: spelled.get(m.group(0).lower(), m.group(0)), path or ""
        )

    @classmethod
    def tiles(cls, path: Optional[str]) -> List[str]:
        """Every file on disk *path*'s token denotes, sorted (the SET).

        Each token globs by its own vocabulary (:attr:`TOKENS`), so a
        ``<UDIM>`` pattern never collects ``u#_v#`` tiles beside it; literal
        segments are glob-escaped, so a folder named ``sh[ot]_01`` cannot
        swallow the match.  A token-free path yields itself when it is on disk
        and nothing when it is not, so the list doubles as the existence
        verdict.

        Parameters:
            path: A path, absolute or already resolved.

        Returns:
            Forward-slashed paths, sorted; empty when nothing matches.
        """
        if not path:
            return []
        if not cls.TOKEN_RE.search(path):
            return [path.replace("\\", "/")] if os.path.isfile(path) else []
        return sorted(
            hit.replace("\\", "/") for hit in glob.glob(cls.wildcard(path, None))
        )

    @classmethod
    def rename(
        cls, path: str, new_name: str, dry_run: bool = False
    ) -> List[Tuple[str, str]]:
        """Rename every file *path* denotes to *new_name*, in its own folder.

        *new_name* is a file NAME carrying the same tokens, in the same order,
        as *path*'s: each tile keeps its own values, spelled into the new name
        (``rock.<UDIM>.png`` -> ``stone.<UDIM>.png`` renames ``rock.1001.png``
        to ``stone.1001.png`` and so on). A token-free path renames its one
        file. Files only -- whatever references them is the caller's to repoint.

        Safe over the set: nothing is renamed unless every target is free (a
        case-only change of the same file is not a collision), and a rename that
        fails part-way puts the ones already done back before it raises.

        Parameters:
            path: The file or token pattern, absolute or already resolved.
            new_name: The new file name -- no folder.
            dry_run: Plan and check only; touch nothing.

        Returns:
            ``[(old, new)]`` forward-slashed, in tile order: what was (or, with
            *dry_run*, would be) renamed. ``[]`` when the name is unchanged.

        Raises:
            ValueError: *new_name* is empty or names a folder, carries other
                tokens than *path*, or *path* denotes nothing on disk.
            FileExistsError: A target is another existing file.
            OSError: A rename failed (after the rollback).
        """
        path = (path or "").replace("\\", "/")
        old_name = os.path.basename(path)
        new_name = str(new_name or "").strip()
        if not new_name or new_name in (".", "..") or re.search(r"[\\/]", new_name):
            raise ValueError(f"Not a file name: {new_name!r}.")

        def tokens(name):
            return [m.group(0).lower() for m in cls.TOKEN_RE.finditer(name)]

        if tokens(new_name) != tokens(old_name):
            raise ValueError(
                f"{new_name!r} must keep the tokens of {old_name!r} "
                f"({', '.join(tokens(old_name)) or 'none'}), in order: they "
                "name the tiles, which keep their numbers."
            )
        files = cls.tiles(path)
        if not files:
            raise ValueError(f"Nothing on disk to rename: {path}")
        if new_name == old_name:
            return []

        # Each token of the old name becomes a group; a tile's groups are the
        # values its new name is spelled with.
        parts, cursor = [], 0
        for match in cls.TOKEN_RE.finditer(old_name):
            parts.append(re.escape(old_name[cursor : match.start()]) + "(.+?)")
            cursor = match.end()
        parts.append(re.escape(old_name[cursor:]) + r"\Z")
        tile_re = re.compile("".join(parts), re.IGNORECASE)
        pairs: List[Tuple[str, str]] = []
        for old in files:
            found = tile_re.match(os.path.basename(old))
            if not found:
                raise ValueError(f"{old} does not spell {old_name!r}.")
            values = iter(found.groups())
            new = cls.TOKEN_RE.sub(lambda _m: next(values), new_name)
            pairs.append((old, f"{os.path.dirname(old)}/{new}"))

        targets = set()
        for old, new in pairs:
            key = os.path.normcase(os.path.abspath(new))
            if key in targets:
                raise FileExistsError(f"Two files would be renamed to {new}.")
            targets.add(key)
            same = key == os.path.normcase(os.path.abspath(old))
            if os.path.exists(new) and not same:
                raise FileExistsError(f"Already exists: {new}")
        if dry_run:
            return pairs

        done: List[Tuple[str, str]] = []
        try:
            for old, new in pairs:
                os.rename(old, new)
                done.append((old, new))
        except OSError:
            for old, new in reversed(done):
                try:
                    os.rename(new, old)
                except OSError:
                    pass  # the original error is the one worth raising
            raise
        return pairs

    @classmethod
    def representative(cls, path: Optional[str]) -> Optional[str]:
        """The one concrete file *path*'s token denotes (collapse to one).

        What a caller that has to TOUCH the file needs -- an existence probe,
        a size read, a content hash, a single-file optimizer -- with the token
        collapsed.  A FIXED token prefers its stand-in (``1001``) so a staged
        representative keeps naming the same file, but does not require it: a
        set running 1002-1005 returns its first real tile.  Nothing on disk at
        all returns the stand-in, the honest "where it was looked for".  A
        frame token globs every literal segment escaped and returns the first
        hit.

        Parameters:
            path: A path, absolute or already resolved.

        Returns:
            The probe path.  A token-free path comes back unchanged (existing
            or not), so ``representative(p) == p`` also tells "no token" from
            "token".  ``None`` for an empty input, or a frame pattern with
            nothing on disk.
        """
        if not path:
            return None
        matches = list(cls.TOKEN_RE.finditer(path))
        if not matches:
            return path

        fixed: List[str] = []
        pattern: List[str] = []
        needs_glob = False
        cursor = 0
        for match in matches:
            literal = path[cursor : match.start()]
            stand_in, token_glob = cls.TOKENS[match.group(0).lower()]
            cursor = match.end()
            if stand_in is None:  # a frame counter: no fixed stand-in
                needs_glob = True
                fixed.append(literal)
                pattern.append(glob.escape(literal) + token_glob)
            else:
                fixed.append(literal + stand_in)
                pattern.append(glob.escape(literal) + stand_in)
        tail = path[cursor:]

        if not needs_glob:
            probe = "".join(fixed) + tail
            if os.path.exists(probe):
                return probe
            tiles = cls.tiles(path)
            return tiles[0] if tiles else probe
        hits = sorted(glob.glob("".join(pattern) + glob.escape(tail)))
        return hits[0] if hits else None


# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
