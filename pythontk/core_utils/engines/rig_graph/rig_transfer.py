# !/usr/bin/python
# coding=utf-8
"""Applying a hand-off manifest's ``rig`` section to a target scene -- the ONE
orchestration both bridges run.

Building a rig on the far side is four steps in a fixed order: plan-and-build,
verify against the source's samples, let the survivors take ownership of their
channels, and say what happened. None of that is DCC-specific -- only the
builder is -- yet mayatk's importer and blendertk's importer each grew their
own copy, 76% identical, which is exactly the drift this engine exists to
prevent: two importers that disagree about what "transferred" means.

A DCC contributes a BUILDER and nothing else. Everything about the order, what
counts as verified, which records survive, and what the log says lives here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from pythontk.core_utils.engines.rig_graph.rig_verify import RigVerify


class RigTransfer:
    """Apply a manifest's ``rig`` section (schema section 15.3) to a scene.

    The builder protocol -- duck-typed, and satisfied by both packages'
    ``RigGraphBuilder``:

    ==========================================  ====================================
    ``build(graph, imported, is_usd=False)``    plan and build; the result dict
    ``commit(record_id) -> int``                survivor owns its channels;
                                                DELETES the payload's keys there
    ``remove(record_id) -> int``                take a demotion's build back
    ``sample_world(node_id, frame)``            a world point, or None
    ``scope()``                                 context manager: the frame is
                                                restored and every touched node
                                                is evaluable for the duration
    ``linear_unit()`` / ``up_axis()``           what this scene measures in
    ==========================================  ====================================

    Example:
        >>> result = RigTransfer.apply(manifest.get("rig"), builder, imported)
        >>> sorted(result)
        ['baked', 'built', 'edits', 'plan', 'report', 'verify']
    """

    @staticmethod
    def capability_key(capability: Dict[str, Any]) -> str:
        """A short, stable digest of *capability*, for a conversion's identity.

        A cached payload was planned against the consumer's capability, so a
        builder that GAINS an op must not replay a payload planned without it.
        Sorted keys, so the digest tracks content rather than dict order.

        Parameters:
            capability: A capability manifest, as plain values.

        Returns:
            str: 12 hex characters.
        """
        import hashlib
        import json

        blob = json.dumps(capability, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    @classmethod
    def apply(
        cls,
        section: Optional[Dict[str, Any]],
        builder: Any,
        imported: Sequence[Any],
        *,
        is_usd: bool = False,
        frame_offset: float = 0.0,
        source_unit: str = "cm",
        source_up_axis: str = "y",
        logger: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """Build *section*'s graph with *builder*, verify it, and report.

        The builder re-plans the shipped graph against its OWN capability --
        the manifest the producer planned against, so the two agree -- builds
        what the plan allows, and every built record the plan wants measured is
        checked against the source's sampled world positions. A rig component
        is all-or-nothing (schema 9.3): a miss or a failed build takes every
        built record of its component with it, so what remains is either a
        complete, verified rig or exactly the bake.

        Parameters:
            section: The manifest's ``rig`` section, or None/empty to do
                nothing (the common case -- most conversions carry no rig).
            builder: The target's rig builder (protocol above).
            imported: What the payload import created; ids resolve against it.
            is_usd: Resolve ids by prim path (USD) rather than leaf name (FBX).
            frame_offset: Added to a source frame to reach the target's.
            source_unit: What a graph that does not STATE its linear unit is
                assumed to have been sampled in, and likewise *source_up_axis*.
                This is the PRODUCER's convention, so it differs by direction
                (a Maya graph is centimetres and Y-up, a Blender graph metres
                and Z-up) and cannot be one shared constant: guessing the wrong
                one scales every sample by 100 and swaps two axes, which
                diverges every record and looks exactly like a rig that simply
                would not transfer. Both extractors always write the keys, so
                this is the fallback for a malformed graph -- which is said out
                loud rather than absorbed.
            logger: Anything with ``info`` / ``warning``. The transfer EDITS
                the scene it is given -- it deletes the payload's keys where a
                rig took over, and a builder may restructure what it built on
                -- so every such edit is stated rather than left for the user
                to discover.

        Returns:
            dict: The builder's result, or None when there was no graph.
        """
        graph = (section or {}).get("graph")
        if not graph:
            return None
        result = builder.build(graph, imported, is_usd=is_usd)
        source = graph.get("source") or {}
        if logger is not None and not source.get("linear_unit"):
            logger.warning(
                "Rig: the graph states no linear unit; assuming "
                f"{source_unit!r} / {source_up_axis!r} up. A wrong guess here "
                "diverges every record."
            )
        # One scope for the whole measurement: the frame moves while sampling
        # and must come back, and a hidden node is not evaluated at all.
        with builder.scope():
            RigVerify.verify_plan(
                result,
                (section or {}).get("verify_samples") or {},
                builder.sample_world,
                source_unit=str(source.get("linear_unit") or source_unit),
                source_up_axis=str(source.get("up_axis") or source_up_axis),
                target_unit=str(builder.linear_unit()),
                target_up_axis=str(builder.up_axis()),
                frame_offset=frame_offset,
                remove=builder.remove,
                graph=graph,
            )
        # Only now: a record that survived verification OWNS its channels, so
        # the payload's baked keys there are deleted. Doing this before the
        # measurement would leave a demoted record with no motion at all.
        deleted = sum(builder.commit(record_id) for record_id in result["built"])
        cls._report(result, deleted, logger)
        return result

    @staticmethod
    def _report(result: Dict[str, Any], deleted: int, logger: Any) -> None:
        """Say what happened, including what was destroyed."""
        if logger is None:
            return
        logger.info("Rig: " + RigVerify.summary(result))
        if deleted:
            logger.info(
                f"Rig: {deleted} baked key curve(s) deleted where a verified "
                "record now drives the channel (its motion is the rig's)."
            )
        for edit in result.get("edits") or ():
            logger.info(f"Rig: {edit}")
        for entry in RigVerify.demoted(result):
            logger.warning(
                f"Rig {entry['kind']}: {entry['record']} {entry.get('detail')}"
            )
