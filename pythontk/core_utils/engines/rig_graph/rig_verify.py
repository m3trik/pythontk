# !/usr/bin/python
# coding=utf-8
"""Point-cloud verification for a rebuilt rig -- pure math, no DCC.

Section 9.4 of the schema: verification is a step, not a hope, and it is ONE
implementation. The exporter that sampled the source, the importer that built
the target and the tests that guard both all measure through this, so the
number a ``diverged`` report entry carries (section 10) is the same number
everywhere. A DCC contributes a sampler (world-space points of one object at
one frame); everything about *comparing* them lives here.
"""

from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

Point = Tuple[float, float, float]
#: Metres per unit for the linear units a source may sample in.
UNIT_METRES: Dict[str, float] = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "in": 0.0254,
    "ft": 0.3048,
}
#: The default verify tolerance as a PHYSICAL length: one centimetre. A graph
#: states its own in ``linear_unit`` (section 8); this applies when it says
#: nothing, and it is fixed in metres so a metre-scaled source is not held to
#: a bar 100x looser than a centimetre-scaled one -- measured: a return leg
#: passed 86 records at "1.0", and that 1.0 was one METRE.
VERIFY_TOLERANCE_M = 0.01


class RigVerify:
    """Compare sampled world points of one object against ground truth.

    The measurement is split, because a single "worst distance" hides WHICH
    thing went wrong: ``rigid`` is the length of the MEAN delta -- the whole
    object displaced as one (a placement / parent-space error) -- and
    ``residual`` is the worst per-point distance from that mean -- the shape
    itself deformed differently (a skinning / solver error). A return leg
    reading 0.0126 mm rigid and 0.0605 mm residual is a different diagnosis
    from the reverse, and only the split says so.

    Units are whatever the caller sampled in; a tolerance is in the same unit
    (the graph's ``linear_unit``).
    """

    @staticmethod
    def compare(want: Sequence[Point], got: Sequence[Point]) -> Dict[str, Any]:
        """Measure *got* against *want*, point by point, at one frame.

        Parameters:
            want: Ground-truth world points, in sample order.
            got: The rebuilt object's world points, same order.

        Returns:
            dict: ``worst`` (largest per-point distance), ``rigid`` (length of
                the mean delta), ``residual`` (largest distance from that
                mean) and ``count`` (points compared).

        Raises:
            ValueError: the point counts differ. A topology mismatch is not a
                distance, and reporting one as a large number would let it
                pass a generous tolerance.
        """
        if len(want) != len(got):
            raise ValueError(
                f"point count differs: {len(want)} expected, {len(got)} sampled"
            )
        n = len(want)
        if not n:
            return {"worst": 0.0, "rigid": 0.0, "residual": 0.0, "count": 0}
        deltas = [
            (gx - wx, gy - wy, gz - wz) for (wx, wy, wz), (gx, gy, gz) in zip(want, got)
        ]
        mean = tuple(sum(d[k] for d in deltas) / n for k in range(3))
        worst = max(sqrt(dx * dx + dy * dy + dz * dz) for dx, dy, dz in deltas)
        rigid = sqrt(sum(m * m for m in mean))
        residual = max(
            sqrt(sum((d[k] - mean[k]) ** 2 for k in range(3))) for d in deltas
        )
        return {"worst": worst, "rigid": rigid, "residual": residual, "count": n}

    @classmethod
    def compare_frames(
        cls,
        want: Mapping[Any, Sequence[Point]],
        got: Mapping[Any, Sequence[Point]],
    ) -> Dict[str, Any]:
        """:meth:`compare` over every frame both sides sampled; the worst wins.

        Parameters:
            want: ``{frame: points}`` ground truth.
            got: ``{frame: points}`` from the rebuilt object.

        Returns:
            dict: ``worst`` / ``rigid`` / ``residual`` as the maximum over the
                frames, plus ``per_frame`` (``{frame: worst}``) so a report can
                say WHEN it diverged, not just that it did.

        Raises:
            ValueError: a frame sampled on one side only, or a point-count
                mismatch on any frame (from :meth:`compare`).
        """
        frames = sorted(want)
        if frames != sorted(got):
            raise ValueError(
                f"frames differ: {sorted(want)} expected, {sorted(got)} sampled"
            )
        per_frame: Dict[Any, float] = {}
        worst = rigid = residual = 0.0
        for frame in frames:
            m = cls.compare(want[frame], got[frame])
            per_frame[frame] = m["worst"]
            worst, rigid = max(worst, m["worst"]), max(rigid, m["rigid"])
            residual = max(residual, m["residual"])
        return {
            "worst": worst,
            "rigid": rigid,
            "residual": residual,
            "per_frame": per_frame,
        }

    @staticmethod
    def default_tolerance(unit: str) -> float:
        """:data:`VERIFY_TOLERANCE_M` expressed in *unit* (``"cm"`` -> 1.0,
        ``"m"`` -> 0.01); an unknown unit is taken as centimetres."""
        return VERIFY_TOLERANCE_M / UNIT_METRES.get(str(unit).lower(), 0.01)

    #: Report kinds that took a BUILT record back (section 10).
    DEMOTED_KINDS = ("failed", "diverged", "cascaded")
    #: Report kinds the PLAN decided before anything was built.
    PLANNED_KINDS = ("baked", "dropped", "cyclic")

    @classmethod
    def demoted(cls, result: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """The report entries of *result* that took a built record back --
        the ones a consumer warns about, one vocabulary for every DCC."""
        return [
            e for e in result.get("report", []) if e.get("kind") in cls.DEMOTED_KINDS
        ]

    @classmethod
    def summary(cls, result: Mapping[str, Any]) -> str:
        """One log line for a build result: what was built, what the bake keeps,
        what was taken back, and -- by rule -- what the plan never tried (a
        ``component`` entry names its blocker in its detail)."""
        report = result.get("report", [])
        planned = Counter(
            str(e.get("reason")) for e in report if e.get("kind") in cls.PLANNED_KINDS
        )
        line = (
            f"{len(result.get('built', []))} record(s) built, "
            f"{len(result.get('baked', []))} node(s) left to the bake, "
            f"{len(cls.demoted(result))} demoted"
        )
        if planned:
            line += f"; planned to bake by reason: {dict(planned)}"
        return line + "."

    @staticmethod
    def verdict(measured: float, tolerance: float) -> Dict[str, Any]:
        """Pass or fail one measurement, carrying both numbers.

        A ``diverged`` report entry (section 10) is only actionable with its
        numbers, so this returns them beside the boolean rather than the
        boolean alone.

        Parameters:
            measured: The distance measured (typically ``worst``).
            tolerance: The record's ``policy.verify.tolerance``, same unit.

        Returns:
            dict: ``passed``, ``measured``, ``tolerance``.
        """
        return {
            "passed": measured <= tolerance,
            "measured": measured,
            "tolerance": tolerance,
        }

    @staticmethod
    def convert_point(
        point: Sequence[float],
        source_unit: str = "cm",
        source_up_axis: str = "y",
        target_unit: str = "m",
        target_up_axis: str = "z",
    ) -> Point:
        """A sampled point in the SOURCE's unit and up-axis, spelled in the
        target's -- the conversion both importers apply (a Y-up source's
        ``(x, y, z)`` is a Z-up target's ``(x, -z, y)``)."""
        scale = UNIT_METRES.get(str(source_unit).lower(), 0.01) / UNIT_METRES.get(
            str(target_unit).lower(), 1.0
        )
        x, y, z = (float(v) * scale for v in point)
        src, dst = str(source_up_axis).lower(), str(target_up_axis).lower()
        if src == dst:
            return (x, y, z)
        return (x, -z, y) if (src, dst) == ("y", "z") else (x, z, -y)

    @classmethod
    def verify_plan(
        cls,
        result: Dict[str, Any],
        samples: Mapping[str, Mapping[Any, Sequence[float]]],
        sample: Callable[[str, int], Optional[Sequence[float]]],
        *,
        source_unit: str = "cm",
        source_up_axis: str = "y",
        target_unit: str = "m",
        target_up_axis: str = "z",
        frame_offset: float = 0.0,
        remove: Callable[[str], Any],
        graph: Any = None,
    ) -> List[str]:
        """Section 9.4, once: measure every built record the plan wants verified
        and demote the misses IN PLACE -- and, given the graph, take a missed
        record's whole rig component back with it.

        The DCC contributes *sample*: the target's world position of a node id
        at a frame (``None`` when the id resolves to nothing it can measure).
        Everything else -- unit and axis conversion, the comparison, the
        demotion and its report entry -- lives here, so mayatk's importer and
        blendertk's importer cannot drift apart on what "verified" means.

        Parameters:
            result: A builder's ``build()`` result; ``built`` / ``baked`` /
                ``report`` are updated in place.
            samples: ``{node id: {frame: [x, y, z]}}`` as the SOURCE sampled
                them, in ``source_unit`` / ``source_up_axis``.
            sample: ``sample(node_id, frame) -> point | None`` on the target.
            frame_offset: Added to a source frame to reach the target's
                (an FBX import may shift the clock by one).
            remove: ``remove(record_id)`` -- takes a demoted record's build back.
            graph: The RigGraph the plan came from (the object or its plain
                form). With it, every built record sharing a component
                (:meth:`RigGraph.components`) with a diverged or failed one is
                taken back too (``cascaded``): a component is all-or-nothing,
                because a rig with one baked link is a bake with clutter.

        Returns:
            list: The record ids demoted (``diverged``, then ``cascaded``). A
                record that had samples but nothing measurable on the target
                is reported as
                ``unverified`` and left built: the CHECK did not happen, and
                saying it passed would be the one thing worse than a miss.
        """
        demoted: List[str] = []
        for record_id, policy in list(result.get("verify", {}).items()):
            tolerance = cls.convert_point(
                (
                    float(
                        (policy or {}).get(
                            "tolerance", cls.default_tolerance(source_unit)
                        )
                    ),
                    0.0,
                    0.0,
                ),
                source_unit,
                source_up_axis,
                target_unit,
                target_up_axis,
            )[0]
            nodes = next(
                (
                    e.get("nodes", [])
                    for e in result.get("report", [])
                    if e.get("record") == record_id
                ),
                [],
            )
            worst, compared = 0.0, 0
            for node_id in nodes:
                for frame, want in (samples.get(node_id) or {}).items():
                    got = sample(node_id, int(round(float(frame) + frame_offset)))
                    if got is None:
                        continue
                    expected = cls.convert_point(
                        want, source_unit, source_up_axis, target_unit, target_up_axis
                    )
                    worst = max(worst, cls.compare([expected], [tuple(got)])["worst"])
                    compared += 1
            if not compared:
                if any(samples.get(n) for n in nodes):
                    result.setdefault("report", []).append(
                        {
                            "kind": "unverified",
                            "severity": "warn",
                            "record": record_id,
                            "nodes": list(nodes),
                            "reason": "verify",
                            "detail": {
                                "error": "no measurable target for the sampled nodes"
                            },
                            "recoverable": True,
                        }
                    )
                continue
            verdict = cls.verdict(abs(worst), abs(tolerance))
            if verdict["passed"]:
                continue
            remove(record_id)
            demoted.append(record_id)
            result["built"] = [r for r in result.get("built", []) if r != record_id]
            for node_id in nodes:
                if node_id not in result.setdefault("baked", []):
                    result["baked"].append(node_id)
            result.setdefault("report", []).append(
                {
                    "kind": "diverged",
                    "severity": "warn",
                    "record": record_id,
                    "nodes": list(nodes),
                    "reason": "verify",
                    "detail": {**verdict, "unit": target_unit, "compared": compared},
                    "recoverable": True,
                }
            )
        if graph is not None:
            demoted.extend(cls._cascade(graph, result, remove, set(demoted)))
        return demoted

    @staticmethod
    def _cascade(
        graph: Any, result: Dict[str, Any], remove: Callable[[str], Any], dead: set
    ) -> List[str]:
        """Take back every built record whose rig component lost a link.

        *dead* is this pass's demotions; a ``failed`` build in the report
        counts too. Mirrors the planner's component rule (section 9.3) at
        verify time, so the consumer ends with either a complete, verified
        component or exactly what a bake would have produced.
        """
        from pythontk.core_utils.engines.rig_graph.rig_model import RigGraph

        rig = graph if isinstance(graph, RigGraph) else RigGraph.from_dict(graph)
        dead = set(dead) | {
            str(e["record"])
            for e in result.get("report", [])
            if e.get("kind") == "failed" and e.get("record")
        }
        built = list(result.get("built", []))
        cascaded: List[str] = []
        for component in rig.components(set(built) | dead):
            blockers = [rid for rid in component if rid in dead]
            if not blockers:
                continue
            for record_id in component:
                if record_id not in built:
                    continue
                remove(record_id)
                cascaded.append(record_id)
                record = rig.record(record_id)
                nodes = list(record.target_ids()) if record else []
                for node_id in nodes:
                    if node_id not in result.setdefault("baked", []):
                        result["baked"].append(node_id)
                result.setdefault("report", []).append(
                    {
                        "kind": "cascaded",
                        "severity": "warn",
                        "record": record_id,
                        "nodes": nodes,
                        "reason": "component",
                        "detail": {"blocker": blockers[0], "component": len(component)},
                        "recoverable": True,
                    }
                )
        result["built"] = [r for r in built if r not in cascaded]
        return cascaded
