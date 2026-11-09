# !/usr/bin/python
# coding=utf-8
"""Animation: say which clip is which, and ship only the keys it needs.

The file's clips are its takes -- one per shot, named after it, as the FBX
carries them -- so this carries the FBX's ``animation`` record into the GLB as
it stands (``extras.animation``: one layout, two containers), each shot's
take the animation of its clip's name; places each clip on the timeline for
the passes that key it (the take it came from, measured off the FBX); and
compacts, reduces and prunes the samplers.

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through the facade it holds (``self._mc``).
"""

import copy
import logging
import struct
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Eager: SceneRecords is the light declaration module every pass reads, and
# the signatures name GlbTarget.
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit, GlbTarget, _SamplerBatch

logger = logging.getLogger(__name__)


class _Animation:
    """Animation: the clips' places, the ``animation`` block, key compaction.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    def __init__(self, mc: type) -> None:
        self._mc = mc  # the facade class: every name it exposes is read through it

    @staticmethod
    def _animation_span(gltf: dict, animation: dict) -> Optional[Tuple[float, float]]:
        """``(first, last)`` keyframe time of *animation*, in seconds, or ``None``.

        Read off the sampler INPUT accessors' ``min``/``max``, which glTF
        requires an animation sampler input to carry -- so this costs no buffer
        decode, and a file that omits them (not a legal one, but this is public
        API pointed at whatever it is handed) reports no span rather than a
        wrong one.
        """
        accessors = gltf.get("accessors") or []
        lo: Optional[float] = None
        hi: Optional[float] = None
        for sampler in animation.get("samplers") or []:
            index = sampler.get("input")
            if not isinstance(index, int) or not 0 <= index < len(accessors):
                continue
            accessor = accessors[index] or {}
            low, high = accessor.get("min"), accessor.get("max")
            # Shape-checked, not just truthiness: this is public API pointed at
            # whatever GLB it is handed, and an accessor whose min is a bare
            # number rather than the spec's array would raise out of a pass
            # whose whole contract is that it degrades to "no span".
            if not (isinstance(low, list) and isinstance(high, list)):
                continue
            if not low or not high:
                continue
            if not isinstance(low[0], (int, float)) or not isinstance(
                high[0], (int, float)
            ):
                continue
            lo = low[0] if lo is None else min(lo, low[0])
            hi = high[0] if hi is None else max(hi, high[0])
        return None if lo is None or hi is None else (float(lo), float(hi))

    @staticmethod
    def _numeric_pairs(keys: Any) -> List[Sequence[Any]]:
        """The ``[frame, value]`` entries of *keys* that are actually numbers.

        The tracks arrive as JSON decoded out of a string channel on a node in
        the deliverable, so their shape is whatever a producer wrote -- and a
        non-numeric entry would otherwise raise out of a pass whose whole
        contract is that a malformed channel degrades to "no tracks". Same rule
        :meth:`_animation_span` applies to accessor bounds, for the same reason.
        """
        out: List[Sequence[Any]] = []
        for key in keys if isinstance(keys, (list, tuple)) else ():
            if not isinstance(key, (list, tuple)) or len(key) < 2:
                continue
            # bool passes the int check, which is correct: a JSON ``true`` is a
            # legitimate way to write a visibility value.
            if isinstance(key[0], (int, float)) and isinstance(key[1], (int, float)):
                out.append(key)
        return out

    def _shot_clips(self, gltf: Dict[str, Any]) -> List[str]:
        """The clip names *gltf*'s ``animation`` record declares, in its order:
        the takes the file is meant to carry, one per shot. Empty when it
        carries none, or -- warned, re-export to cure -- a record of another
        version (``MeshConvert._current_record``)."""
        record = self._mc._current_record(gltf, SceneRecords.SHOTS)
        return [
            str(s["clip"])
            for s in (record or {}).get("shots") or ()
            if isinstance(s, dict) and s.get("clip")
        ]

    #: What Maya (and so FBX2glTF) names the whole-timeline AnimStack -- the
    #: one take a scene with no shots ships. The clip :meth:`_synthesize_clips`
    #: makes for a file that carries none is named the same, so a consumer sees
    #: the clip it would have seen had the exporter armed animation.
    DEFAULT_CLIP_NAME = "Take 001"

    def _synthesize_clips(
        self, gltf: Dict[str, Any], names: Optional[Sequence[str]] = None
    ) -> List[Dict[str, Any]]:
        """Give a file the clips it is owed, empty, for its passes to fill.

        Every take a shot names (:meth:`_shot_takes`) -- FBX2glTF writes
        none for a take that holds still, glTF forbidding an empty clip --
        and every clip a build's overlay states
        (``MeshConvert.take_preview_curves``: an animation-off push names the
        takes an animated one would split): one empty clip per name in
        *names* the file lacks, each placed among the clips it has in
        *names*' order (after the clip of the name before it), so the page
        lists the takes in take order. ``None`` gives a clip-less file the one
        clip a file with no shots carries, for a preview's curves.
        :meth:`_hold_take_spans` holds a clip to its take's end;
        :meth:`prune_glb_animations` drops one that stays hollow.

        Returns:
            The clips of *names*, as the file now holds them; the one
            synthesized when *names* is ``None``.
        """
        if names is None:
            clip = {"name": self._mc.DEFAULT_CLIP_NAME, "samplers": [], "channels": []}
            gltf["animations"] = [clip]
            logger.info("No clip in the file: %r synthesized for the previewed effect.", clip["name"])
            return [clip]
        names = list(names)
        animations = gltf.setdefault("animations", [])
        held: Dict[str, Dict[str, Any]] = {}
        for animation in animations:
            held.setdefault(str(animation.get("name") or ""), animation)
        added = []
        for index, name in enumerate(names):
            if name in held:
                continue
            clip = {"name": name, "samplers": [], "channels": []}
            earlier = next((held[n] for n in reversed(names[:index]) if n in held), None)
            later = next((held[n] for n in names[index + 1 :] if n in held), None)
            if earlier is not None:
                at = next(i for i, a in enumerate(animations) if a is earlier) + 1
            elif later is not None:
                at = next(i for i, a in enumerate(animations) if a is later)
            else:
                at = len(animations)
            animations.insert(at, clip)
            held[name] = clip
            added.append(name)
        if not animations:
            del gltf["animations"]  # ``minItems: 1``: no clip, no list
        if added:
            logger.info(
                "%d clip(s) the file is owed and the converter wrote none of (%s).",
                len(added),
                ", ".join(added),
            )
        return [held[name] for name in names if name in held]

    def take_clip_spans(self, fbx: Any, bake_fps: Optional[float] = None) -> Dict[str, float]:
        """``{take: seconds}`` each take's clip must span from its t=0, in take
        order: from where FBX2glTF puts the clip's t=0 -- the take's first key
        over all its curves, floored to the bake grid (*bake_fps*), as
        ``MeshConvert.render_effect_curves`` places its curves; the window's
        start for a take that keys nothing -- to the end of the window the
        take declares (``FbxFile.take_windows``, the span
        ``FbxMedia.fit_take_windows`` writes). ``{}`` for a file that declares
        no window."""
        from pythontk.file_utils.mesh_convert.fbx_file import FbxFile

        windows = fbx.take_windows()
        if not windows:
            return {}
        rate = float(bake_fps or self._mc.CONVERSION_BAKE_FPS)
        ticks = fbx.take_ticks()
        spans: Dict[str, float] = {}
        for take, (start, end) in windows.items():
            zero = (
                self._mc._visibility._clip_zero(ticks[take][0], rate, FbxFile.TICKS_PER_SECOND)
                if take in ticks
                else start
            )
            if end - zero > 1e-9:
                spans[take] = round(end - zero, 9) + 0.0
        return spans

    def _shot_takes(self, gltf: Dict[str, Any], takes: Sequence[str]) -> List[str]:
        """The *takes* a shot of *gltf*'s ``animation`` record names (ignoring
        case, as :meth:`_name_takes_after_shots` joins them), in take order:
        the takes owed a clip where the converter wrote none -- a still
        shot's. A take no shot names is not: a static write's (Maya declares
        a take for every write) or one the converter lost the motion of is no
        shot to keep. Silent on a record of another version; the manifest
        pass warns of it."""
        record = self._mc.data_export_channel(gltf, SceneRecords.SHOTS.key)
        if not isinstance(record, dict) or record.get("version") != SceneRecords.SHOTS.version:
            return []
        clips = {
            str(s["clip"]).casefold()
            for s in record.get("shots") or ()
            if isinstance(s, dict) and s.get("clip")
        }
        return [take for take in takes if take.casefold() in clips]

    #: A hold's paths, in the order one is taken: the first the clip leaves
    #: free on its node, with glTF's rest for it. Translation first -- a
    #: presence gate is a scale.
    HOLD_PATHS = (
        ("translation", (0.0, 0.0, 0.0)),
        ("rotation", (0.0, 0.0, 0.0, 1.0)),
        ("scale", (1.0, 1.0, 1.0)),
    )

    def _hold_take_spans(
        self, edit: "GlbEdit", spans: Dict[str, float], shots: Sequence[str] = ()
    ) -> List[str]:
        """Hold each clip of *spans* to its seconds where its channels end
        before them -- or, for a clip of *shots*, where it has none (a still
        shot's; FBX2glTF writes a still take as a clip with no channel, which
        the prune drops, and a take no shot names keeps that fate: a static
        write's, or one whose motion the converter lost, is no shot to
        keep): one STEP channel on a scene root holding
        its rest on a path the clip leaves free, keyed at 0 and at the span's
        end -- how a glTF clip says "this take lasts this long" when nothing
        in it moves to the end. Never on the ``data_export`` carrier (the
        build takes it off the file) nor on a node with a matrix (glTF
        forbids animating one). One BIN append; a shared span or rest shares
        its accessor.

        Returns:
            The clips held.
        """
        gltf = edit.gltf
        by_name: Dict[str, Dict[str, Any]] = {}
        for animation in gltf.get("animations") or []:
            by_name.setdefault(str(animation.get("name") or ""), animation)
        roots = self._hold_roots(gltf)
        still = {str(name).casefold() for name in shots}
        batch = _SamplerBatch()
        planned = []
        unheld = []
        for name, seconds in spans.items():
            clip = by_name.get(name)
            if clip is None:
                continue
            if not clip.get("channels") and name.casefold() not in still:
                continue
            end = self._animation_time_span(edit, clip)[1]
            if end is not None and end >= seconds - 1e-6:
                continue
            taken = {
                ((c.get("target") or {}).get("node"), (c.get("target") or {}).get("path"))
                for c in clip.get("channels") or []
            }
            choice = next(
                (
                    (node, path, rest)
                    for node in roots
                    for path, rest in self.HOLD_PATHS
                    if (node, path) not in taken
                ),
                None,
            )
            if choice is None:
                unheld.append(name)
                continue
            node, path, default = choice
            own = gltf["nodes"][node].get(path)
            rest = (
                [float(v) for v in own]
                if isinstance(own, list) and len(own) == len(default)
                else list(default)
            )
            planned.append(
                (clip, node, path, batch.slot([0.0, float(seconds)]), batch.slot(rest + rest), len(rest))
            )
        if unheld:
            logger.warning(
                "Animation: %d take clip(s) could not be held to their take's end "
                "(%s) -- no node is free to carry the hold.",
                len(unheld),
                ", ".join(unheld),
            )
        if not planned:
            return []
        if not batch.commit(edit):
            logger.warning("Animation: this GLB's buffer is external -- no take clip held.")
            return []
        held = []
        for clip, node, path, times, values, width in planned:
            sampler = batch.sampler(clip, times, values, "VEC3" if width == 3 else "VEC4", "STEP")
            clip.setdefault("channels", []).append(
                {"sampler": sampler, "target": {"node": node, "path": path}}
            )
            held.append(str(clip.get("name") or ""))
        edit.dirty = True
        logger.info(
            "Animation: %d take clip(s) held to their take's end (%s).",
            len(held),
            ", ".join(held),
        )
        return held

    def _hold_roots(self, gltf: Dict[str, Any]) -> List[int]:
        """The nodes a hold may ride, best first: the scene's roots, then
        every other node -- never a ``data_export`` carrier (the scene's, a
        module's, a numbered copy: ``SceneStoreBase.is_carrier``) nor a node
        with a matrix."""
        from pythontk.core_utils.engines.scene_export.scene_store import SceneStoreBase

        nodes = gltf.get("nodes") or []
        scenes = gltf.get("scenes") or []
        index = gltf.get("scene", 0)
        scene = scenes[index] if isinstance(index, int) and 0 <= index < len(scenes) else {}
        roots = [i for i in scene.get("nodes") or [] if isinstance(i, int) and 0 <= i < len(nodes)]
        return [
            i
            for i in dict.fromkeys(roots + list(range(len(nodes))))
            if not nodes[i].get("matrix")
            and not SceneStoreBase.is_carrier(nodes[i].get("name"))
        ]

    def prune_glb_animations(self, glb: GlbTarget) -> List[str]:
        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            animations = gltf.get("animations")
            if not animations:
                return []
            kept: List[Dict[str, Any]] = []
            dropped: List[str] = []
            for index, animation in enumerate(animations):
                if animation.get("channels") and animation.get("samplers"):
                    kept.append(animation)
                else:
                    dropped.append(str(animation.get("name") or f"animations[{index}]"))
            if not dropped:
                return []
            if kept:
                gltf["animations"] = kept
            else:
                # ``animations`` has ``minItems: 1``: an empty array is as
                # invalid as the hollow entry was.
                del gltf["animations"]
            edit.dirty = True

            declared = set(self._shot_clips(gltf))
            shots = [name for name in dropped if name in declared]
            if shots:
                # Warned: the scene declared these and the file will not list
                # them, which the manifest pass reports next as takes it lacks.
                logger.warning(
                    "Animation: dropped %d declared take(s) with no keyframes "
                    "(%s) -- glTF cannot carry an empty clip.",
                    len(shots),
                    ", ".join(shots),
                )
            rest = [name for name in dropped if name not in declared]
            if rest:
                logger.info(
                    "Animation: dropped %d empty animation(s) (%s) -- nothing "
                    "in the exported set is keyed there.",
                    len(rest),
                    ", ".join(rest),
                )
            return dropped

    def compact_glb_animations(self, glb: GlbTarget) -> Dict[str, int]:
        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            animations = gltf.get("animations") or []
            accessors = gltf.get("accessors") or []
            if not animations or not accessors:
                return {"channels": 0, "accessors": 0, "bytes": 0}

            # Pass 1 -- classify. An output accessor is collapsible only when
            # EVERY channel reading it is constant: samplers are shared, and
            # shrinking one that something still animates through would freeze
            # that motion.
            constant: Dict[int, bytes] = {}
            refused: set = set()
            per_animation: List[List[tuple]] = []
            for animation in animations:
                samplers = animation.get("samplers") or []
                rows = []
                for channel in animation.get("channels") or []:
                    index = channel.get("sampler")
                    if not isinstance(index, int) or not 0 <= index < len(samplers):
                        continue
                    sampler = samplers[index] or {}
                    output = sampler.get("output")
                    if not isinstance(output, int):
                        continue
                    if (sampler.get("interpolation") or "LINEAR") == "CUBICSPLINE":
                        # Three elements per key -- in-tangent, value,
                        # out-tangent -- so two keys is SIX elements, not two.
                        # Collapsing one would write a sampler whose output
                        # count no longer matches its input, which is a
                        # malformed file rather than a smaller one.
                        refused.add(output)
                        continue
                    if output in refused:
                        continue
                    if output not in constant:
                        elements = self._mc._accessor_elements(edit, output)
                        if not elements or len(elements) < 3:
                            # Nothing to win on a channel already at two keys,
                            # and an unreadable layout is left alone.
                            refused.add(output)
                            continue
                        first = elements[0]
                        if any(e != first for e in elements):
                            refused.add(output)
                            continue
                        constant[output] = first
                    rows.append((index, output))
                per_animation.append(rows)

            # Constancy is a property of the ACCESSOR, so every channel reading
            # one agrees about it -- but a refusal is a property of the SAMPLER
            # (interpolation), so one clip can refuse an output another clip
            # already accepted. The survivors are what both agree on.
            collapsible = {k: v for k, v in constant.items() if k not in refused}
            if not collapsible:
                return {"channels": 0, "accessors": 0, "bytes": 0}

            # Pass 2 -- one shared two-key time input per animation, spanning
            # the clip's own range so its duration is unchanged, plus one
            # two-key value block per collapsible output.
            payloads: List[bytes] = []
            input_slot: Dict[int, int] = {}
            for position, animation in enumerate(animations):
                # Against `collapsible`, not against the rows: an output one
                # clip accepted can still be refused by a LATER clip's
                # interpolation, and a clip left with nothing to collapse must
                # not mint a time input no sampler will read -- an accessor
                # keeps its bufferView alive, so that orphan survives the
                # collector and the next run mints another.
                if not any(out in collapsible for _s, out in per_animation[position]):
                    continue
                lo, hi = self._mc._animation_time_span(edit, animation)
                if lo is None:
                    continue
                input_slot[position] = len(payloads)
                payloads.append(struct.pack("<2f", lo, hi))
            output_slot: Dict[int, int] = {}
            for output, element in collapsible.items():
                output_slot[output] = len(payloads)
                payloads.append(element + element)

            added_views = self._mc._append_bin_views(edit, payloads)
            if not added_views:  # external buffer: nothing this pass can do
                return {"channels": 0, "accessors": 0, "bytes": 0}

            # Pass 3 -- mint the accessors and rewire the samplers.
            new_input: Dict[int, int] = {}
            for position, slot in input_slot.items():
                # Read back OUT of the packed bytes, not from the float64 the
                # span was measured as: `min` must not exceed the value the
                # file actually stores, and packing to float32 can round it up.
                lo, hi = struct.unpack("<2f", payloads[slot])
                accessors.append(
                    {
                        "bufferView": added_views[slot],
                        "componentType": 5126,
                        "count": 2,
                        "type": "SCALAR",
                        # Required on an animation input by the spec, and the
                        # only place a reader learns the clip's length.
                        "min": [lo],
                        "max": [hi],
                    }
                )
                new_input[position] = len(accessors) - 1
            # The outputs are re-pointed IN PLACE rather than appended beside
            # the originals. An accessor keeps its bufferView alive whether or
            # not anything still reads the accessor, so minting replacements
            # and abandoning the old ones collects nothing -- measured: 13,187
            # channels rewired, 0.00 MB reclaimed. Editing them is also the
            # truthful shape: this is the same channel, shorter. Safe because
            # `collapsible` already excludes every output any channel refused,
            # and a mesh never reads an animation accessor.
            for output, slot in output_slot.items():
                accessor = accessors[output]
                accessor["bufferView"] = added_views[slot]
                # The new view starts at the data: an offset INTO the old view
                # (a packer sharing one view between accessors) would read
                # past the two keys it was just given.
                accessor.pop("byteOffset", None)
                accessor["count"] = 2
                accessor.pop("min", None)
                accessor.pop("max", None)

            channels = 0
            replaced_inputs: Set[int] = set()
            for position, animation in enumerate(animations):
                if position not in new_input:
                    continue
                samplers = animation.get("samplers") or []
                for sampler_index, output in per_animation[position]:
                    if output not in output_slot:
                        continue
                    sampler = samplers[sampler_index]
                    replaced_inputs.add(sampler.get("input"))
                    sampler["input"] = new_input[position]
                    channels += 1

            edit.dirty = True
            # A converter shares one time input across a clip, which its moving
            # channels keep alive; a clip rebuild writes one PER CHANNEL, and
            # each collapsed channel's would stay behind as a dead accessor.
            self._mc._drop_orphaned_accessors(edit, replaced_inputs)
            reclaimed = self._mc._compact_bin(edit)
            logger.info(
                "Animation: collapsed %d constant channel(s) across %d clip(s) "
                "to two keys, reclaiming %.2f MB -- a clip PINS the pose of "
                "what it does not animate, which two keys say as well as "
                "thousands.",
                channels,
                len(animations),
                reclaimed / 1048576.0,
            )
            return {
                "channels": channels,
                "accessors": len(output_slot),
                "bytes": reclaimed,
            }

    def reduce_glb_animations(
        self,
        glb: GlbTarget,
        tolerance: float,
        rotation_tolerance: Optional[float] = None,
    ) -> Dict[str, int]:
        from pythontk.file_utils.mesh_convert.glb.key_reduction import (
            GlbKeyReduction,
        )

        return GlbKeyReduction.reduce(glb, tolerance, rotation_tolerance)

    def _animation_time_span(self, edit: "GlbEdit", animation: Dict[str, Any]) -> tuple:
        """``(first, last)`` input time over every sampler, or ``(None, None)``.

        Read from each input accessor's ``min``/``max``, which the spec
        requires an animation input to carry, and decoded only when a writer
        left them off.
        """
        lo = hi = None
        accessors = edit.gltf.get("accessors") or []
        for sampler in animation.get("samplers") or []:
            index = (sampler or {}).get("input")
            if not isinstance(index, int) or not 0 <= index < len(accessors):
                continue
            accessor = accessors[index] or {}
            low, high = accessor.get("min"), accessor.get("max")
            if isinstance(low, list) and low and isinstance(high, list) and high:
                start, end = float(low[0]), float(high[0])
            else:
                elements = self._mc._accessor_elements(edit, index)
                if not elements:
                    continue
                start = struct.unpack("<f", elements[0])[0]
                end = struct.unpack("<f", elements[-1])[0]
            lo = start if lo is None else min(lo, start)
            hi = end if hi is None else max(hi, end)
        return lo, hi

    def apply_glb_animations(self, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            animations = gltf.get("animations") or []
            # The record as the FBX carries it, read in the schema it declares
            # (another version is refused, warned: re-export) -- and carried
            # as it stands: nothing of it is bound to this file.
            record = self._mc._current_record(gltf, SceneRecords.SHOTS)
            if not isinstance(record, dict):
                return None
            shots = [
                s
                for s in (record.get("shots") or [])
                if isinstance(s, dict) and s.get("clip") is not None
            ]
            by_clip = {str(s["clip"]) for s in shots}
            renamed = self._name_takes_after_shots(animations, by_clip)
            if renamed:
                logger.info(
                    "Animation: %d clip(s) renamed to their shot's clip name, "
                    "which a reader joins by (%s).",
                    len(renamed),
                    ", ".join(f"{old!r} -> {new!r}" for old, new in renamed),
                )
            names = [
                animation.get("name") or f"animation_{index}"
                for index, animation in enumerate(animations)
            ]
            #: Shot clips carrying no channels: reported.
            hollow = sorted(
                name
                for name, animation in zip(names, animations)
                if name in by_clip and not (animation.get("channels") or [])
            )
            if hollow:
                # Not an error -- a shot can legitimately hold still -- but the
                # one thing a reviewer needs told: the clip is in the list and
                # plays as nothing. In the conversion chain
                # :meth:`prune_glb_animations` has already removed these (glTF
                # forbids them); this is the path for a file handed in
                # directly, or a prune skipped.
                logger.info(
                    "Animation: %d shot clip(s) carry no keyframes (%s).",
                    len(hollow),
                    ", ".join(hollow),
                )
            uncut = sorted(by_clip - set(names))
            if uncut:
                # Loud, like every lightmap miss: the scene declares these
                # shots and the file carries no take of their name -- the
                # take split did not reach this file, the take held no
                # keyframes and was pruned, or the names disagree. A consumer
                # that plays a named shot's clip will not find one.
                logger.warning(
                    "Animation: %d shot(s) have no clip in the GLB (%s) -- the "
                    "file carries %d clip(s) where the scene declares %d shot(s), "
                    "one take each.",
                    len(uncut),
                    ", ".join(uncut),
                    len(names),
                    len(by_clip),
                )

            edit.dirty = True
            return SceneRecords.put_glb_block(
                gltf, SceneRecords.SHOTS, copy.deepcopy(record)
            )

    @staticmethod
    def _name_takes_after_shots(
        animations: List[Dict[str, Any]], clips: Set[str]
    ) -> List[Tuple[str, str]]:
        """Name each animation that is a shot's take EXACTLY the shot's clip.

        A shot's clip is unique ignoring case and is the name its take is
        written under, so an animation named after it in another case alone
        (a converter's or a hand's respelling) is that take; renamed, every
        reader joins it by name. One left that differs in more than case is
        no shot's: :meth:`apply_glb_animations` warns of the shot it lacks.
        An exact name already held stands -- two animations are never given
        one name.

        Returns:
            ``[(old name, new name)]`` for each animation renamed.
        """
        held = {a.get("name") for a in animations if isinstance(a.get("name"), str)}
        folded = {clip.casefold(): clip for clip in clips}
        renamed: List[Tuple[str, str]] = []
        for animation in animations:
            name = animation.get("name")
            if not isinstance(name, str) or name in clips:
                continue
            clip = folded.get(name.casefold())
            if clip is None or clip in held:
                continue
            animation["name"] = clip
            held.add(clip)
            renamed.append((name, clip))
        return renamed
