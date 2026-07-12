# coding=utf-8
"""Pure shot-boundary detection math.

DCC-agnostic clustering / boundary algorithms extracted from the Maya
detection acquisition.  These functions take plain data — animation
*segments* or key *entries* already gathered from a scene — and turn them
into shot-region candidates.  Scene acquisition (querying anim curves,
resolving curves to transforms, filtering flat objects) is a DCC concern and
lives in the mayatk / blendertk detection layers; those layers gather the
inputs and call the functions here.

Candidate shape (returned by both functions):
    ``{"name": str, "start": float, "end": float, "objects": list[str]}``
"""
from typing import Any, Dict, List, Optional, Tuple


STANDARD_TRANSFORM_ATTRS: frozenset = frozenset(
    {
        "translateX",
        "translateY",
        "translateZ",
        "rotateX",
        "rotateY",
        "rotateZ",
        "scaleX",
        "scaleY",
        "scaleZ",
        "visibility",
    }
)
"""Per-axis transform + visibility attributes.

Used across the shots system to distinguish genuine scene-content animation
from custom trigger/marker attributes.
"""


__all__ = [
    "STANDARD_TRANSFORM_ATTRS",
    "cluster_segments_by_gap",
    "boundaries_from_key_entries",
]


def cluster_segments_by_gap(
    segments: List[Dict[str, Any]],
    gap_threshold: float = 5.0,
    min_duration: float = 2.0,
) -> List[Dict[str, Any]]:
    """Cluster per-object animation segments into shot-region candidates.

    Given *segments* (each a dict with ``"start"``, ``"end"``, ``"obj"`` —
    already gathered from a scene by the DCC layer), groups contiguous
    segments into regions separated by gaps of at least *gap_threshold*
    frames.  Each cluster becomes one candidate spanning ``[min start,
    max end]`` with the sorted set of contributing objects; clusters shorter
    than *min_duration* are discarded.

    Parameters:
        segments: Segment dicts with ``"start"``, ``"end"``, ``"obj"`` keys.
        gap_threshold: Minimum gap (frames) between clusters.
        min_duration: Minimum shot duration in frames.  Clusters shorter
            than this are discarded.

    Returns:
        List of candidate dicts with ``"name"``, ``"start"``, ``"end"``, and
        ``"objects"`` keys, ordered by cluster (ascending start time).
    """
    if not segments:
        return []

    # Do not mutate the caller's list — a pure function sorts a copy.
    segments = sorted(segments, key=lambda s: s["start"])

    clusters: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = [segments[0]]
    current_end = segments[0]["end"]

    for seg in segments[1:]:
        if seg["start"] - current_end > gap_threshold:
            clusters.append(current)
            current = [seg]
            current_end = seg["end"]
        else:
            current.append(seg)
            current_end = max(current_end, seg["end"])
    clusters.append(current)

    candidates: List[Dict[str, Any]] = []
    for cluster in clusters:
        start = min(s["start"] for s in cluster)
        end = max(s["end"] for s in cluster)
        if (end - start) < min_duration:
            continue
        objs = sorted({str(s["obj"]) for s in cluster})
        candidates.append(
            {
                "name": f"Shot {len(candidates) + 1}",
                "start": start,
                "end": end,
                "objects": objs,
            }
        )
    return candidates


def boundaries_from_key_entries(
    entries: List[Tuple[float, float, str]],
    gap_threshold: float = 5.0,
    key_filter: str = "all",
) -> List[Dict[str, Any]]:
    """Build shot-region candidates from ``(time, value, object)`` key entries.

    Each unique key time is treated as an explicit shot boundary; keys closer
    than *gap_threshold* frames merge into one boundary.  Designed for stepped
    / marker keys (e.g. audio triggers) where each key marks the start of a
    shot rather than continuous animation.

    Parameters:
        entries: ``(time, value, object)`` triples gathered from a scene's
            selected keys by the DCC layer.
        gap_threshold: Keys within this many frames merge into one boundary.
        key_filter: How to interpret key values:

            ``"all"``
                Every key is a boundary (contiguous shots).
            ``"skip_zero"``
                Keys with value 0 are ignored; only non-zero keys become
                boundaries.
            ``"zero_as_end"``
                Non-zero keys start shots; zero-value keys end the preceding
                shot (allows gaps between shots).

    Returns:
        List of candidate dicts with ``"name"``, ``"start"``, ``"end"``, and
        ``"objects"`` keys.  Unlike the DCC caller, no flat-object filtering
        is applied here — that needs scene queries and stays in the DCC layer.
    """
    if not entries:
        return []

    def _is_zero(v) -> bool:
        """Treat None and near-zero floats as 'zero'."""
        return v is None or abs(v) < 1e-9

    # Stable sort: same-time entries have zeros first so that in
    # ``zero_as_end`` mode a closing zero is processed before the
    # opening non-zero trigger at the same frame.  Sort a copy — pure.
    entries = sorted(entries, key=lambda e: (e[0], 0 if _is_zero(e[1]) else 1))

    # ---- "zero_as_end" mode: pair non-zero starts with zero ends ---------
    if key_filter == "zero_as_end":
        candidates: List[Dict[str, Any]] = []
        current_start: Optional[float] = None
        current_objs: set = set()
        for t, v, obj in entries:
            if not _is_zero(v):
                if current_start is None:
                    current_start = t
                    current_objs = {obj}
                else:
                    current_objs.add(obj)
            else:
                # Zero-value key ends the current shot
                if current_start is not None:
                    candidates.append(
                        {
                            "name": f"Shot {len(candidates) + 1}",
                            "start": current_start,
                            "end": t,
                            "objects": sorted(str(o) for o in current_objs),
                        }
                    )
                    current_start = None
                    current_objs = set()
        # Trailing shot with no closing zero key
        if current_start is not None:
            candidates.append(
                {
                    "name": f"Shot {len(candidates) + 1}",
                    "start": current_start,
                    "end": current_start + 1.0,
                    "objects": sorted(str(o) for o in current_objs),
                }
            )
        return candidates

    # ---- "skip_zero" mode: filter zeros, then use boundary logic below -----
    if key_filter == "skip_zero":
        entries = [(t, v, obj) for t, v, obj in entries if not _is_zero(v)]
        if not entries:
            return []
        # Fall through to "all" mode boundary logic.

    # ---- "all" mode: merge keys within gap_threshold into boundary points
    boundaries: List[Tuple[float, set]] = []  # (time, {objects})
    first_time = entries[0][0]
    cur_time = entries[0][0]
    cur_objs: set = {entries[0][2]}

    for t, _v, obj in entries[1:]:
        if t - cur_time <= gap_threshold:
            cur_objs.add(obj)
            cur_time = t
        else:
            boundaries.append((first_time, cur_objs))
            first_time = t
            cur_time = t
            cur_objs = {obj}
    boundaries.append((first_time, cur_objs))

    if not boundaries:
        return []

    # Build contiguous regions: each boundary starts a shot that ends
    # at the next boundary.  The last shot gets a nominal 1-frame end
    # (a DCC range resolver may compute the real end).
    candidates = []
    for i, (start, objs) in enumerate(boundaries):
        if i + 1 < len(boundaries):
            end = boundaries[i + 1][0]
        else:
            end = start + 1.0
        candidates.append(
            {
                "name": f"Shot {len(candidates) + 1}",
                "start": start,
                "end": end,
                "objects": sorted(str(o) for o in objs),
            }
        )
    return candidates
