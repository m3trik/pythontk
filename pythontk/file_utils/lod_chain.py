# !/usr/bin/python
# coding=utf-8
"""``LodChain``: the one level-of-detail convention every carrier writes.

A LOD group is a node ``<base>`` whose children ``<base>_LOD0`` ..
``<base>_LOD<n>`` are its levels, most detailed first, at most
:attr:`LodChain.MAX_LEVELS`. Its switch points are screen-relative heights,
one per level: the height (0..1 of the screen) below which that level stops
being drawn, the last one the cull height (0: never culled).

Adopted, not invented. The names are the ``_LOD<n>`` convention game
engines' model importers read, and LOD0 is the most detailed in every source
(the engines, glTF's ``MSFT_lod``, OpenUSD, the FBX SDK, RealityKit). Each
format then states the group in its own terms around the same names: FBX an
``FbxLODGroup`` node attribute on ``<base>`` (thresholds in percent, one fewer
than its children), glTF ``MSFT_lod`` on the LOD0 node, USD a variant set
``LOD`` with variants ``LOD0..n`` (the convention engines' USD importers
read). The heights read straight across: an engine's screen-relative
transition height and ``MSFT_screencoverage`` are per level with the cull last.

Reads every known form: the names in any case, and the ``LOD_<n>`` level
containers DCCs' LOD generators write (named in :attr:`notes`: engines'
importers read neither).

Plain values in and out, so the hosts, the converters and the importers share
one reading; nothing here touches a scene or a file.
"""

import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from pythontk.core_utils.help_mixin import HelpMixin


class LodChain(HelpMixin):
    """The LOD convention: level names, groups, their issues and switch heights.

    A *node* given to :meth:`groups` is a mapping with ``path`` (a ``|``
    separated path, the leaf the name), ``parent`` (its parent's path, ``None``
    at the scene root) and optionally ``triangles`` and ``bounds`` (its world
    box with its subtree, ``((xmin, ymin, zmin), (xmax, ymax, zmax))``), each
    ``None`` when unknown.
    """

    #: The level suffix written; read in any case.
    SUFFIX = "_LOD"
    #: The strictest reader's limit (a game engine's LOD group).
    MAX_LEVELS = 8
    #: How far a level's box corner may stray from LOD0's, as a share of
    #: LOD0's diagonal, before it is noted: a reduction moves the surface a
    #: little; a level that moved or lost a part pops at its switch.
    BOUNDS_TOLERANCE = 0.1

    _LEVEL = re.compile(r"^(?P<base>.+?)_lod(?P<index>\d+)$", re.IGNORECASE)
    _CONTAINER = re.compile(r"^lod_?(?P<index>\d+)$", re.IGNORECASE)

    # ------------------------------------------------------------ names
    @staticmethod
    def _leaf(path: str) -> str:
        return str(path).rsplit("|", 1)[-1]

    @classmethod
    def parse(cls, name: str) -> Optional[Tuple[str, int]]:
        """``(base, index)`` for a level name ``<base>_LOD<n>`` (any case), else ``None``.

        A path is read by its leaf; a namespace stays part of the base.
        """
        match = cls._LEVEL.match(cls._leaf(name))
        if not match:
            return None
        return match.group("base"), int(match.group("index"))

    @classmethod
    def parse_container(cls, name: str) -> Optional[int]:
        """The level a ``LOD_<n>`` / ``LOD<n>`` container holds (DCCs' LOD
        generators, USD's ``LOD`` variants), else ``None``."""
        match = cls._CONTAINER.match(cls._leaf(name))
        return int(match.group("index")) if match else None

    @classmethod
    def level_name(cls, base: str, index: int) -> str:
        """The name of level *index* of *base*: ``<base>_LOD<index>``."""
        if not 0 <= int(index) < cls.MAX_LEVELS:
            raise ValueError(f"A LOD level is 0..{cls.MAX_LEVELS - 1}, not {index}.")
        return f"{base}{cls.SUFFIX}{int(index)}"

    # ------------------------------------------------------------ groups
    @classmethod
    def groups(cls, nodes: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """The LOD groups among *nodes*, ordered by parent then base.

        Returns:
            One dict per group: ``parent`` (the group's node, ``None`` at the
            scene root), ``base``, ``form`` (``"names"`` or ``"containers"``),
            ``levels`` (paths by level, ``None`` where one is missing),
            ``issues`` (what keeps it from being written as a group: it then
            ships as plain nodes, as authored) and ``notes`` (what a reader
            should know but does not block it).
        """
        nodes = [dict(n) for n in nodes]
        children: Dict[Optional[str], Set[str]] = {}
        for node in nodes:
            children.setdefault(node.get("parent"), set()).add(node["path"])

        found: Dict[Tuple[Optional[str], str, str], Dict[str, Any]] = {}
        for node in nodes:
            parent = node.get("parent")
            parsed = cls.parse(node["path"])
            if parsed is not None:
                base, index = parsed
                key = (parent, "names", base.casefold())
                spelled = base
            else:
                index = cls.parse_container(node["path"])
                if index is None:
                    continue
                key = (parent, "containers", "")
                spelled = cls._leaf(parent) if parent else "LOD containers at the scene root"
            entry = found.setdefault(key, {"base": spelled, "members": []})
            entry["members"].append((index, node["path"], node.get("triangles"), node.get("bounds")))

        groups = []
        for (parent, form, _), entry in found.items():
            members = sorted(entry["members"], key=lambda m: (m[0], m[1]))
            if len(members) < 2:
                continue  # a lone level: see lone_levels
            groups.append(cls._group(parent, form, entry["base"], members, children.get(parent, set())))
        groups.sort(key=lambda g: (g["parent"] or "", g["base"].casefold(), g["form"]))
        return groups

    @classmethod
    def _group(
        cls,
        parent: Optional[str],
        form: str,
        base: str,
        members: Sequence[Tuple[int, str, Optional[int], Any]],
        siblings: Set[str],
    ) -> Dict[str, Any]:
        issues: List[str] = []
        notes: List[str] = []
        top = max(member[0] for member in members)
        if top >= cls.MAX_LEVELS:
            # Bounded before anything is sized by it: a name states the number
            # (a building model's Wall_LOD300 is a level of development).
            issues.append(
                f"{base}: LOD{top} is past the {cls.MAX_LEVELS}-level limit "
                f"(LOD0..LOD{cls.MAX_LEVELS - 1}); not read as a LOD group."
            )
            return {
                "parent": parent,
                "base": base,
                "form": form,
                "levels": [member[1] for member in members],
                "issues": issues,
                "notes": notes,
            }
        levels: List[Optional[str]] = [None] * (top + 1)
        triangles: List[Optional[int]] = [None] * (top + 1)
        bounds: List[Any] = [None] * (top + 1)
        for index, path, count, box in members:
            if levels[index] is not None:
                issues.append(f"{base}: LOD{index} is named twice ({levels[index]}, {path}).")
                continue
            levels[index], triangles[index], bounds[index] = path, count, box
        if levels[0] is None:
            issues.append(f"{base}: no LOD0, the most detailed level.")
        for index in range(1, top):
            if levels[index] is None:
                issues.append(f"{base}: LOD{index} is missing between LOD0 and LOD{top}.")
        known = [(i, t) for i, t in enumerate(triangles) if t is not None]
        for (_, before), (index, count) in zip(known, known[1:]):
            if count >= before:
                issues.append(f"{base}: LOD{index} has no fewer triangles than LOD{index - 1}.")
        if parent is None:
            issues.append(f"{base}: its levels sit at the scene root; a LOD group needs a node of its own.")
        elif len(siblings) > len({member[1] for member in members}):  # its levels are among them
            issues.append(f"{base}: its node {parent} holds other nodes beside its levels; a LOD group's node holds its levels alone.")
        strays = cls._stray_levels(bounds)
        if strays:
            notes.append(
                f"{base}: " + "; ".join(f"LOD{i}'s bounds stray {share:.0%} of LOD0's size from LOD0's" for i, share in strays)
                + ". A level that moved, or lost or gained a part, pops at its switch."
            )
        if form == "containers":
            notes.append(
                f"{base}: levels held in LOD_<n> containers (a DCC's LOD generator); "
                "game engines do not read that form: they read levels named <base>_LOD<n>."
            )
        return {"parent": parent, "base": base, "form": form, "levels": levels, "issues": issues, "notes": notes}

    @classmethod
    def _stray_levels(cls, bounds: Sequence[Any]) -> List[Tuple[int, float]]:
        """``(level, share)`` for each level whose box corner strays from
        LOD0's by more than :attr:`BOUNDS_TOLERANCE` of LOD0's diagonal; none
        when LOD0's box is unknown or flat. A level of unknown box is not measured."""
        try:
            (low0, high0) = bounds[0]
            diagonal = sum((float(b) - float(a)) ** 2 for a, b in zip(low0, high0)) ** 0.5
        except (TypeError, ValueError, IndexError):
            return []
        if not diagonal > 0.0:
            return []
        strays = []
        for index, box in enumerate(bounds[1:], 1):
            try:
                low, high = box
                share = max(
                    abs(float(v) - float(v0))
                    for corner, corner0 in ((low, low0), (high, high0))
                    for v, v0 in zip(corner, corner0)
                ) / diagonal
            except (TypeError, ValueError):
                continue
            if share > cls.BOUNDS_TOLERANCE:
                strays.append((index, share))
        return strays

    @classmethod
    def lone_levels(
        cls, nodes: Iterable[Mapping[str, Any]], groups: Optional[Sequence[Mapping[str, Any]]] = None
    ) -> List[str]:
        """Level-named nodes with no sibling level: plain nodes with a suffix.
        *groups*: :meth:`groups` of the same *nodes*, when the caller has them."""
        nodes = list(nodes)
        grouped = {p for g in (cls.groups(nodes) if groups is None else groups) for p in g["levels"] if p}
        return sorted(n["path"] for n in nodes if cls.parse(n["path"]) and n["path"] not in grouped)

    #: What an export may ship of the LOD groups (the Scene Exporter's LODs row).
    MODES = ("authored", "lod0", "generate")

    @classmethod
    def names_levels(cls, paths: Iterable[str]) -> bool:
        """Whether any of *paths* is named as a LOD level or level container:
        the cheap gate a host runs before it counts a triangle."""
        return any(cls.parse(p) or cls.parse_container(p) is not None for p in paths)

    @classmethod
    def plan(cls, nodes: Iterable[Mapping[str, Any]], mode: str = "authored") -> Dict[str, Any]:
        """What an export ships of the LOD groups among *nodes* -- the one
        decision every Scene Exporter applies; a host only lists its nodes.

        Parameters:
            nodes: As :meth:`groups` takes them.
            mode: ``"authored"`` ships each valid group as one; ``"lod0"``
                (Most Detailed Only) its most detailed level alone;
                ``"generate"`` (Generate Missing) ships the groups as authored
                (the host generates the missing ones).

        Returns:
            ``{"groups", "drop", "listing", "warnings", "notes", "lone"}``:
            the valid groups, the level paths the write leaves out (each with
            its subtree), one line per group for the export's report, why a
            group ships as plain nodes, what a reader should know, and the
            level-named nodes with no sibling level.
        """
        if mode not in cls.MODES:
            raise ValueError(f"A LOD mode is one of {', '.join(cls.MODES)}, not {mode!r}.")
        nodes = list(nodes)
        groups = cls.groups(nodes)
        valid = [g for g in groups if not g["issues"]]
        lone = cls.lone_levels(nodes, groups)
        notes = [note for g in groups for note in g["notes"]]
        if lone:
            notes.append(
                f"LOD-named node(s) with no sibling level, shipped as plain nodes: "
                f"{', '.join(cls._leaf(p) for p in lone)}."
            )
        return {
            "groups": valid,
            "drop": cls.lower_levels(valid) if mode == "lod0" else [],
            "listing": [f"{g['base']}: {sum(1 for p in g['levels'] if p)} levels" for g in valid],
            "warnings": [f"LOD group shipped as plain nodes -- {i}" for g in groups for i in g["issues"]],
            "notes": notes,
            "lone": lone,
        }

    @staticmethod
    def refine_selection(
        selected: Iterable[str],
        drop: Iterable[str],
        children: Mapping[str, Sequence[str]],
    ) -> List[str]:
        """*selected* with every *drop* path, and its subtree, left out.

        For a writer that carries each selected node's whole subtree (a DCC's
        FBX and USD export): a selected node above a dropped one is replaced by
        its children, recursively, so the write keeps everything else. The
        writer still writes each kept node's ancestors (measured on a DCC's FBX
        export, 2026-11-10: a selected leaf ships with its parents and their
        LOD group, never its siblings).

        Parameters:
            selected: The paths the write would select.
            drop: The paths to leave out (:meth:`plan`'s ``drop``).
            children: ``{path: child paths}`` for the nodes above a dropped one
                -- every child the writer would carry, shapes included.

        Returns:
            The refined paths, in order, each once.
        """
        dropped = {str(p) for p in drop}
        # Every proper ancestor of a dropped path, once: each visit is then a
        # walk up its own path, not a scan of every dropped one.
        above = {d[:i] for d in dropped for i in range(1, len(d)) if d[i] == "|" and d[:i]}
        kept: List[str] = []

        def under_dropped(path: str) -> bool:
            if path in dropped:
                return True
            cut = path.rfind("|")
            while cut > 0:
                if path[:cut] in dropped:
                    return True
                cut = path.rfind("|", 0, cut)
            return False

        def visit(path: str) -> None:
            if under_dropped(path):
                return
            if path in above:
                for child in children.get(path, ()):
                    visit(child)
                return
            kept.append(path)

        for path in selected:
            visit(str(path))
        return list(dict.fromkeys(kept))

    #: The variant set USD states a LOD group in (the convention engines'
    #: USD importers read: variants exactly ``LOD0..n``, one level each).
    USD_VARIANT_SET = "LOD"

    @classmethod
    def variant_plan(cls, groups: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """The USD variant set each group becomes, as plain data a host's USD
        writer authors (``pxr`` lives in the hosts, never here): ``{"group",
        "levels", "variants"}`` -- the group node's name, its level nodes'
        names (LOD0 first) and the variants ``LOD0..n`` that hold them, one
        each. Selected: LOD0, what a reader without LOD support composes."""
        plan = []
        for group in groups:
            levels = [p for p in group.get("levels") or [] if p]
            if group.get("issues") or not group.get("parent") or len(levels) < 2:
                continue
            plan.append(
                {
                    "group": cls._leaf(group["parent"]),
                    "levels": [cls._leaf(p) for p in levels],
                    "variants": [f"LOD{i}" for i in range(len(levels))],
                }
            )
        return plan

    @classmethod
    def record_groups(
        cls,
        groups: Iterable[Mapping[str, Any]],
        ships: Callable[[str], bool],
        heights_of: Callable[[str, int], Sequence[float]],
    ) -> List[Dict[str, Any]]:
        """The ``lods`` record's entries for the valid *groups* a write ships
        -- the one reading both hosts' producers apply.

        Parameters:
            groups: :meth:`groups`.
            ships: Whether the write ships a level, by its name.
            heights_of: The group's switch heights, by its node's path and
                level count (the host's own store).

        Returns:
            ``{"name", "levels", "heights"}`` per group with at least two levels
            shipped: its node's name, those levels' names, and their switches
            (:meth:`kept_heights`: a dropped level's range goes to the next).
        """
        entries = []
        for group in groups:
            if group.get("issues") or not group.get("parent"):
                continue
            levels = [cls._leaf(p) for p in group["levels"] if p]
            kept = [index for index, name in enumerate(levels) if ships(name)]
            if len(kept) < 2:
                continue
            heights = heights_of(group["parent"], len(levels))
            entries.append(
                {
                    "name": cls._leaf(group["parent"]),
                    "levels": [levels[k] for k in kept],
                    "heights": cls.kept_heights(heights, kept),
                }
            )
        return entries

    @staticmethod
    def lower_levels(groups: Iterable[Mapping[str, Any]]) -> List[str]:
        """Every level but each group's most detailed one, group by group."""
        lower: List[str] = []
        for group in groups:
            present = [p for p in group["levels"] if p]
            lower.extend(present[1:])
        return lower

    #: Generation's defaults: levels in all (LOD0 included), each level's
    #: share of the last's triangles (engines' generated mesh LODs halve too), and the
    #: fewest triangles a generated level may have.
    DEFAULT_LEVELS = 3
    DEFAULT_RATIO = 0.5
    MIN_TRIANGLES = 64

    @classmethod
    def ladder(
        cls,
        triangles: int,
        levels: int = DEFAULT_LEVELS,
        ratio: float = DEFAULT_RATIO,
        floor: int = MIN_TRIANGLES,
    ) -> List[int]:
        """The triangle target of each level to generate below a LOD0 of
        *triangles*: each *ratio* of the last, stopping before one would fall
        below *floor*; at most ``levels - 1`` of them (LOD0 is the source).
        Empty when the mesh is too light for a level."""
        if not 1 <= int(levels) <= cls.MAX_LEVELS:
            raise ValueError(f"A LOD group holds 1..{cls.MAX_LEVELS} levels, not {levels}.")
        if not 0.0 < float(ratio) < 1.0:
            raise ValueError(f"Each level keeps a share of the last's triangles, 0..1 exclusive, not {ratio}.")
        targets: List[int] = []
        count = float(triangles)
        for _ in range(int(levels) - 1):
            count *= float(ratio)
            if count < floor:
                break
            targets.append(int(count))
        return targets

    # ------------------------------------------------------------ heights
    @classmethod
    def default_heights(cls, count: int, first: float = 0.5, cull: float = 0.0) -> List[float]:
        """A ladder for *count* levels: each switch at half the last, then *cull*."""
        if count < 1:
            raise ValueError(f"A LOD group has at least one level, not {count}.")
        heights = [float(first) * 0.5**i for i in range(count - 1)] + [float(cull)]
        cls.check_heights(heights)
        return heights

    @staticmethod
    def check_heights(heights: Sequence[float]) -> None:
        """Raise ``ValueError`` unless *heights* fall level by level within 0..1."""
        if not heights:
            raise ValueError("A LOD group states one height per level; none given.")
        for height in heights:
            if not 0.0 <= float(height) <= 1.0:
                raise ValueError(f"A switch height is a fraction of the screen, 0..1, not {height}.")
        for before, after in zip(heights, heights[1:]):
            if not float(after) < float(before):
                raise ValueError(f"Switch heights must fall level by level: {list(heights)}.")

    @staticmethod
    def to_fbx_thresholds(heights: Sequence[float]) -> List[float]:
        """FbxLODGroup thresholds in percent: one fewer than the levels (no cull)."""
        return [round(float(h) * 100.0, 6) for h in heights[:-1]]

    @staticmethod
    def from_fbx_thresholds(thresholds: Sequence[float], cull: float = 0.0) -> List[float]:
        """Heights from FbxLODGroup percent thresholds, with *cull* for the last level."""
        return [float(t) / 100.0 for t in thresholds] + [float(cull)]

    @staticmethod
    def kept_heights(heights: Sequence[float], kept: Sequence[int]) -> List[float]:
        """The switch heights of a group a write ships only some levels of:
        each kept level (*kept*, indices into the group's levels, rising) but
        the last switches where it did, and the last kept draws down to the
        group's cull -- a dropped level's range goes to the next one kept."""
        kept = sorted(int(k) for k in kept)
        if not kept:
            return []
        return [float(heights[k]) for k in kept[:-1]] + [float(heights[-1])]
