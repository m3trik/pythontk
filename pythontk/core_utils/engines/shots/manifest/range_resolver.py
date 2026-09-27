# !/usr/bin/python
# coding=utf-8
"""Range resolution algorithm for the Shot Manifest.

Converts user-entered ranges + gap-detected boundaries into a fully resolved
``(step_id, start, end, is_user)`` list for every build step.  Pure logic — no
Qt or DCC imports.  Shot durations come from an injectable ``duration_fn`` so
the audio "from_source" probe (a DCC concern) stays out of this module; the
default is the pure :func:`~pythontk.core_utils.engines.shots.manifest.behaviors.compute_duration`.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from pythontk.core_utils.engines.shots.manifest.manifest_model import BuilderStep


class RangeResolver:
    """Resolve build-step ranges from user entries and gap boundaries.

    Turns partially specified ranges plus detected shot boundaries into a
    complete ``(step_id, start, end, is_user)`` list covering every step.
    Pure logic — no DCC, no IO.
    """

    @staticmethod
    def prune_to_top_boundaries(
        region_starts: List[float], n_steps: int
    ) -> List[float]:
        """Keep only *n_steps* region starts by selecting the largest gaps.

        Picks the *n_steps - 1* largest consecutive differences in
        *region_starts* as the primary shot boundaries, then returns
        the first region plus the region after each selected boundary.
        """
        if len(region_starts) <= n_steps:
            return region_starts
        diffs = [
            (region_starts[i + 1] - region_starts[i], i)
            for i in range(len(region_starts) - 1)
        ]
        diffs.sort(key=lambda x: -x[0])
        top_indices = sorted(d[1] for d in diffs[: n_steps - 1])
        selected = [region_starts[0]]
        for idx in top_indices:
            selected.append(region_starts[idx + 1])
        return selected

    @staticmethod
    def resolve_ranges(
        steps: List[BuilderStep],
        user_ranges: Dict[str, Tuple[Optional[float], Optional[float]]],
        gap_starts: List[float],
        gap_end_map: Dict[float, float],
        gap: float,
        use_selected_keys: bool,
        last_resolved: List[Tuple[str, float, Optional[float], bool]],
        from_step_idx: int = 0,
        default_duration: float = 0,
        duration_fn: Optional[Callable[..., float]] = None,
    ) -> List[Tuple[str, float, Optional[float], bool]]:
        """Compute a resolved ``(start, end)`` for every step.

        Merges user-entered ranges with gap-detected auto-fill boundaries.

        Parameters
        ----------
        steps
            Ordered list of build steps from CSV or detection.
        user_ranges
            Map of ``step_id → (start, end_or_None)`` for user-entered values.
        gap_starts
            Detected animation-region start frames (pre-sorted).
        gap_end_map
            Map of ``region_start → region_end`` for detected regions that
            have an explicit end (e.g. from ``zero_as_end`` mode).
        gap
            Inter-shot gap in frames (from ShotStore settings).
        use_selected_keys
            When ``True``, steps without a matching detected region are
            skipped rather than placed sequentially.
        last_resolved
            Previously resolved list — entries before *from_step_idx* are
            reused as a frozen prefix.
        from_step_idx
            Only re-resolve from this index onward.
        default_duration
            When positive and no animation regions are detected, each step
            is assigned this uniform duration instead of behavior-derived
            durations.  Set to ``0`` to use the old behavior.
        duration_fn
            Callable ``(objects, fallback=...) -> float`` returning a step's
            content-driven duration.  ``None`` (default) uses the pure
            :func:`behaviors.compute_duration`; a DCC caller passes one bound to
            an audio-measurement provider so audio steps size to their clip length.

        Returns
        -------
        list[tuple[str, float, float | None, bool]]
            ``(step_id, start, end_or_None, is_user)`` in step order.
        """
        if not steps:
            return []

        if duration_fn is None:
            from pythontk.core_utils.engines.shots.manifest.behaviors import Behaviors

            duration_fn = Behaviors.compute_duration

        # When use_selected_keys is active and no regions were found,
        # abort instead of falling through to sequential placement.
        if use_selected_keys and not gap_starts:
            return []

        # When more regions than steps, keep only the largest
        # boundaries so each step maps to a major animation section.
        if len(gap_starts) > len(steps):
            gap_starts = RangeResolver.prune_to_top_boundaries(gap_starts, len(steps))

        # Build the resolved list
        resolved: List[Tuple[str, float, Optional[float], bool]] = []
        gap_idx = 0
        # When no animation is detected and a default_duration is set, use
        # uniform placement so steps get sensible ranges (e.g. 200f each).
        use_default = default_duration > 0 and not gap_starts and not use_selected_keys

        cursor = 0.0 if use_default else 1.0  # start at 0 for default ranges
        cursor_forced = False  # True once a user range advances cursor

        def _step_duration(objs: List) -> float:
            """Content duration under the active sizing policy.

            In use_default mode audio steps still consult *duration_fn* so clips
            render at their real length; purely template-driven steps take the
            uniform default so authoring isn't penalised for declaring behaviors.
            """
            if use_default:
                if any(getattr(o, "kind", "") == "audio" for o in objs):
                    return duration_fn(objs, fallback=default_duration)
                return default_duration
            return duration_fn(objs)

        # Frozen prefix: reuse last-resolved values for steps before
        # from_step_idx.  last_resolved may be sparse (selected-keys mode
        # skips unresolved steps), so entries are matched by step_id —
        # positional copying would freeze the wrong steps' ranges and
        # duplicate the edited step.
        frozen_ids: set = set()
        if from_step_idx > 0 and last_resolved:
            by_id = {entry[0]: entry for entry in last_resolved}
            for step in steps[:from_step_idx]:
                entry = by_id.get(step.step_id)
                if entry is not None:
                    resolved.append(entry)
                    frozen_ids.add(step.step_id)
            # Advance cursor past the frozen prefix
            if resolved:
                _, _, prev_end, _ = resolved[-1]
                if prev_end is not None:
                    cursor = prev_end + gap
                    cursor_forced = True
            # Advance gap_idx past gaps consumed by the frozen prefix
            for gs in gap_starts:
                if gs < cursor:
                    gap_idx += 1
                else:
                    break

        for step in steps:
            if step.step_id in frozen_ids:
                continue  # already in frozen prefix

            user = user_ranges.get(step.step_id)
            if user is not None:
                start, end = user
                resolved.append((step.step_id, start, end, True))
                # Advance cursor past this user-defined range
                if end is not None:
                    cursor = end + gap
                else:
                    cursor = start + duration_fn(step.objects) + gap
                cursor_forced = True
            elif gap_starts and gap_idx < len(gap_starts):
                # If a prior user range pushed the cursor past this gap
                # boundary, place the step at the cursor instead so
                # downstream steps never overlap with earlier ranges.
                raw_start = gap_starts[gap_idx]
                start = max(raw_start, cursor) if cursor_forced else raw_start
                # Preserve the detected end (e.g. from zero_as_end mode)
                # so that gaps aren't collapsed to next_start - gap.
                # However, if the step was pushed past its detected end,
                # the original end is no longer valid for this position.
                detected_end = gap_end_map.get(raw_start)
                if detected_end is not None and start > detected_end:
                    detected_end = None
                gap_idx += 1
                resolved.append((step.step_id, start, detected_end, False))
                if detected_end is not None:
                    cursor = detected_end + gap
                else:
                    cursor = start + duration_fn(step.objects) + gap
            else:
                # In selected-keys mode, do not fabricate ranges for
                # steps that have no corresponding detected region.
                if use_selected_keys:
                    continue
                # Sequential placement from cursor.  In use_default mode the
                # caller opts into uniform sizing (see _step_duration).
                start = cursor
                resolved.append((step.step_id, start, None, False))
                cursor = start + _step_duration(step.objects) + gap

        # Second pass: resolve None ends as next_start - gap (or last key)
        step_by_id = {s.step_id: s for s in steps}
        for i in range(len(resolved)):
            step_id, start, end, is_user = resolved[i]
            if end is None:
                if i + 1 < len(resolved):
                    # Clamp: a next step pinned at or before start + gap
                    # would otherwise produce an inverted range (end < start).
                    end = max(start, resolved[i + 1][1] - gap)
                else:
                    step_obj = step_by_id.get(step_id)
                    objs = step_obj.objects if step_obj else []
                    end = start + _step_duration(objs)
            resolved[i] = (step_id, start, end, is_user)

        return resolved

    # ---------------------------------------------------- user-range bookkeeping
    # The rules a manifest panel applies to the ranges a user types, pure so
    # both DCC panels share them (they used to carry them as controller code).

    @staticmethod
    def step_index(steps: List[BuilderStep], step_id: str) -> int:
        """The index of *step_id* in *steps*, or ``-1`` when absent."""
        return next((i for i, s in enumerate(steps) if s.step_id == step_id), -1)

    @staticmethod
    def all_ranges_complete(
        steps: List[BuilderStep],
        user_ranges: Dict[str, Tuple[Optional[float], Optional[float]]],
    ) -> bool:
        """True when every step has a user-supplied ``(start, end)`` pair."""
        return bool(steps) and all(
            (r := user_ranges.get(s.step_id)) is not None
            and r[0] is not None
            and r[1] is not None
            for s in steps
        )

    @staticmethod
    def cascade_from(
        steps: List[BuilderStep],
        user_ranges: Dict[str, Tuple[Optional[float], Optional[float]]],
        step_idx: int,
    ) -> List[str]:
        """Drop the user ranges of every step after *step_idx*, in place.

        What an edit to one step's range does to the steps downstream of it:
        they re-flow from the new anchor instead of keeping positions that
        were chosen relative to the old one.

        Returns:
            The step ids whose user range was dropped.
        """
        dropped = []
        for s in steps[step_idx + 1 :]:
            if user_ranges.pop(s.step_id, None) is not None:
                dropped.append(s.step_id)
        return dropped

    @staticmethod
    def parse_range_edit(
        start_text: str, end_text: str
    ) -> Optional[Tuple[float, Optional[float]]]:
        """A user's Start/End cell text as a ``(start, end_or_None)`` range.

        Returns ``None`` when both cells are empty -- the edit CLEARS the step's
        user range.

        Raises:
            ValueError: the edit is not a range: a cell that is not a number,
                an End without a Start, a negative Start, or an End at or
                before the Start.
        """
        start_text, end_text = start_text.strip(), end_text.strip()
        if not start_text and not end_text:
            return None
        start = float(start_text) if start_text else None
        end = float(end_text) if end_text else None
        if start is None:
            raise ValueError("a range needs a start")
        if start < 0:
            raise ValueError(f"start {start:g} is negative")
        if end is not None and end <= start:
            raise ValueError(f"end {end:g} is not after start {start:g}")
        return start, end

    @staticmethod
    def previous_end(
        steps: List[BuilderStep],
        last_resolved: List[Tuple[str, float, Optional[float], bool]],
        step_idx: int,
    ) -> Optional[float]:
        """The resolved end of the nearest step before *step_idx* that has one.

        The floor a user's Start for step *step_idx* may not go below.
        *last_resolved* may be sparse (selected-keys mode skips unresolved
        steps), so predecessors are matched by ``step_id``: positional indexing
        would compare against the wrong step's end.  ``None`` for the first
        step, or when nothing before it resolved.
        """
        if step_idx <= 0 or not last_resolved:
            return None
        resolved_ends = {sid: e for sid, _, e, _ in last_resolved}
        return next(
            (
                resolved_ends[s.step_id]
                for s in reversed(steps[:step_idx])
                if resolved_ends.get(s.step_id) is not None
            ),
            None,
        )

    @staticmethod
    def find_collisions(
        resolved: List[Tuple[str, float, Optional[float], bool]],
    ) -> List[Tuple[str, str]]:
        """Adjacent resolved ranges that overlap, as ``(step_id, next_step_id)``.

        A range without an end occupies its start frame only.
        """
        pairs = []
        for (curr_id, curr_start, curr_end, _), (next_id, next_start, _, _) in zip(
            resolved, resolved[1:]
        ):
            effective_end = curr_end if curr_end is not None else curr_start
            if effective_end > next_start:
                pairs.append((curr_id, next_id))
        return pairs

    @staticmethod
    def gaps_from_regions(
        regions: Optional[List[Dict]],
    ) -> Tuple[List[float], Dict[float, float]]:
        """Detected regions as :meth:`resolve_ranges`' ``(gap_starts, gap_end_map)``.

        *regions* are ``{"start", "end"}`` dicts (``end`` optional); a region
        with an end maps its start to it.
        """
        if not regions:
            return [], {}
        starts = [r["start"] for r in regions]
        ends = {r["start"]: r["end"] for r in regions if r.get("end") is not None}
        return starts, ends
