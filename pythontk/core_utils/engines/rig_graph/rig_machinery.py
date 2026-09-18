# !/usr/bin/python
# coding=utf-8
"""Naming the rig apparatus a bake leaves inert -- the rule, with no DCC in it.

A carrier ships every node of a scene as an object, so a rig that could not
travel arrives TWICE over: its motion, as keys on whatever renders, and the
whole apparatus that used to produce that motion -- constraint nodes, IK
handles, control curves, up-vector locators, the groups that hold only those,
and joints nothing content is deformed by. In the target it selects, it draws
and it drives nothing.

Deciding WHICH nodes those are is three questions, and only the first is about a
DCC at all:

1. what is this node? -- a per-node fact (:data:`RigMachinery.KINDS`), answered
   by whoever can see the scene;
2. given those facts, what is apparatus? -- an algebra over the hierarchy, which
   is what this module is;
3. of what was delivered, what can safely be removed? -- the same algebra from
   the other end, against a carrier that kept names and lost paths.

Keeping 2 and 3 here is what lets one rule serve both directions of a hand-off
and both carriers, and be tested without launching anything.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


class RigMachinery:
    """The apparatus rule: what a baked rig leaves behind, and what may go.

    Both halves are pure functions over plain values. A producer answers "what
    is each node" and gets back the apparatus; a consumer answers "what did I
    receive" and gets back what it may delete. Neither half imports a DCC, and
    neither decides anything the caller can see better than it can.
    """

    #: What a node IS, once something that can see the scene has looked. Every
    #: kind but :attr:`CONTENT` is apparatus when the hierarchy agrees.
    KINDS: Tuple[str, ...] = (
        "constraint",
        "ik",
        "control",
        "locator",
        "joint",
        "group",
    )
    #: The kind that is never apparatus and protects every node above it: a
    #: mesh, a surface, a camera, a light -- whatever the scene is FOR. A node a
    #: DCC cannot classify belongs here, so an unrecognised shape survives.
    CONTENT: str = "content"

    # ---------------------------------------------------------------- producer
    @classmethod
    def classify(
        cls,
        nodes: Mapping[str, str],
        *,
        seeds: Iterable[str] = (),
        protected: Iterable[str] = (),
        separator: str = "|",
    ) -> Dict[str, str]:
        """The apparatus in *nodes*: ``{path: kind}``, empty when there is none.

        Apparatus means CONNECTED TO A RIG, not merely "draws nothing". The
        sweep starts at *seeds* -- the nodes a rig graph names, plus whatever the
        caller knows is rig by type -- and reaches only their unprotected
        descendants and the wrapper groups above them. That distinction is the
        whole rule: a scene's own empties and locators (a data-export marker, a
        user-position group) draw nothing either, and deleting them would be a
        loss nobody asked for.

        Protection propagates UP, never down: a node of kind :attr:`CONTENT`, and
        anything in *protected*, keeps every ancestor that holds it. So a control
        with geometry parented under it survives as the thing carrying that
        geometry, while its own constraint node does not.

        Parameters:
            nodes: ``{path: kind}`` for every node in the scene, paths spelled
                with *separator* (``"|a|b|c"``). A kind outside :attr:`KINDS` is
                treated as :attr:`CONTENT` -- unclassifiable is not disposable.
            seeds: Paths that ARE the rig: the graph's own nodes, constraint and
                IK nodes. A seed that is protected or unknown is ignored.
            protected: Paths that must survive whatever their kind -- deformer
                influences, and the nodes a plan still intends to build with,
                which the consumer has to resolve after the import.
            separator: The hierarchy separator of *nodes*' paths.

        Returns:
            dict: ``{path: kind}``, a subset of *nodes*.
        """
        if not nodes:
            return {}
        keep = set()

        def protect(path: str) -> None:
            """*path* survives, and so does every ancestor that holds it."""
            parts = path.split(separator)
            for i in range(2, len(parts) + 1):
                keep.add(separator.join(parts[:i]))

        for path, kind in nodes.items():
            if kind not in cls.KINDS:
                protect(path)
        for path in protected:
            protect(path)

        candidates = set()
        for seed in seeds:
            if seed in keep or seed not in nodes:
                continue
            candidates.add(seed)
            parts = seed.split(separator)  # the wrapper groups holding only rig
            for i in range(len(parts) - 1, 1, -1):
                ancestor = separator.join(parts[:i])
                if ancestor in keep:
                    break
                candidates.add(ancestor)
        # A group that holds only rig holds only rig all the way down -- this is
        # what reaches the up-vector locator parked beside a control, which no
        # record names and nothing is parented under. Pure prefix work: the paths
        # already state the hierarchy, so nothing needs asking again.
        prefixes = tuple(sorted(path + separator for path in candidates))
        for path in nodes:
            if path in keep or path in candidates:
                continue
            if any(path.startswith(prefix) for prefix in prefixes):
                candidates.add(path)
        return {path: nodes[path] for path in sorted(candidates)}

    @classmethod
    def unambiguous(
        cls,
        kinds: Mapping[str, str],
        names: Iterable[str],
        *,
        separator: str = "|",
    ) -> Tuple[Dict[str, str], Tuple[str, ...]]:
        """*kinds* less every entry a consumer could not address without risk.

        A carrier keeps names and loses paths, so the far side can only match a
        LEAF name -- and a mirrored rig repeats short names. A leaf that
        :meth:`classify` gives to apparatus AND to a node that must survive would
        take that node with it, so the ambiguity is resolved here, where the full
        paths still exist, and it resolves in favour of keeping: un-stripped
        apparatus is merely the old behaviour, a deleted null is a loss.

        Parameters:
            kinds: The apparatus, as :meth:`classify` returned it.
            names: EVERY path in the scene, not only the classified ones -- a
                carrier may land a shape as its own object too, which puts its
                name in the consumer's namespace. A node under named apparatus
                travels with it and is not a survivor.
            separator: The hierarchy separator of both path sets.

        Returns:
            tuple: ``(kept, dropped)`` -- the safe subset, and the paths given up
            on, for the caller to report.
        """
        survivors = {
            path.rsplit(separator, 1)[-1]
            for path in names
            if path not in kinds and path.rsplit(separator, 1)[0] not in kinds
        }
        dropped = tuple(
            sorted(p for p in kinds if p.rsplit(separator, 1)[-1] in survivors)
        )
        return {p: k for p, k in kinds.items() if p not in dropped}, dropped

    # ---------------------------------------------------------------- consumer
    @classmethod
    def select(
        cls,
        section: Mapping[str, str],
        subtrees: Mapping[str, Sequence[str]],
        *,
        protected: Iterable[str] = (),
        separator: str = "|",
        suffix: str = ".",
    ) -> Tuple[Dict[str, str], Tuple[str, ...]]:
        """What a delivered scene may drop for *section*, and what it may not.

        The producer proved that nothing content sits under a node it named. This
        is the net under that proof, because the two scenes are not the same
        scene: a carrier renames on collision, an importer re-parents, and the
        far side may have rebuilt a rig onto some of these very nodes. So every
        candidate's WHOLE subtree is checked before any of it goes, and a subtree
        holding something protected is refused -- by name, for the caller to say
        out loud.

        Parameters:
            section: The manifest's machinery section, ``{producer path: kind}``.
                Matched by leaf name, which is all a carrier keeps.
            subtrees: ``{object name: [that object and its descendants]}`` for
                everything the import delivered.
            protected: Names that must never be removed -- renderable objects,
                armatures something is skinned to, anything a rebuilt rig drives.
            separator: The hierarchy separator the PRODUCER spelled *section*'s
                paths with -- ``"|"`` from Maya, ``"/"`` from Blender. Getting it
                wrong takes the whole path for the leaf and matches nothing, so
                the strip silently does nothing.
            suffix: The separator a carrier's collision suffix uses
                (``"ctrl.001"``); the stem is tried when the full name misses.

        Returns:
            tuple: ``({name: kind} to remove, (name, ...) refused)``.
        """
        wanted = {
            str(path).rsplit(separator, 1)[-1]: str(kind)
            for path, kind in section.items()
        }
        if not wanted:
            return {}, ()
        guarded = set(protected)
        doomed: Dict[str, str] = {}
        refused: List[str] = []
        for name, family in subtrees.items():
            if name in doomed:
                continue
            stem = name if name in wanted else name.rsplit(suffix, 1)[0]
            if stem not in wanted:
                continue
            if any(member in guarded for member in family):
                refused.append(name)
                continue
            for member in family:
                # Its OWN kind when the section names it -- a shape object under
                # a control is not another "group" -- else the root's.
                own = member if member in wanted else member.rsplit(suffix, 1)[0]
                doomed[member] = wanted.get(own, wanted[stem])
        return doomed, tuple(sorted(refused))

    @staticmethod
    def tally(kinds: Mapping[str, str]) -> Dict[str, int]:
        """``{kind: count}`` over *kinds*, for a log line that says WHAT went."""
        counts: Dict[str, int] = {}
        for kind in kinds.values():
            counts[kind] = counts.get(kind, 0) + 1
        return dict(sorted(counts.items()))
