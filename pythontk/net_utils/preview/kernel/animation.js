import * as THREE from 'three';
import { el } from './dom.js';
import { materialsOf } from './model.js';
import { SHOTS } from './records.js';
import { readExtras } from './scene.js';

// A shot-split deliverable ships one clip per shot AND, on the Maya route, the
// whole-timeline take its FBX exporter keeps alongside them — so `animations[0]`
// is routinely the entire timeline rather than the first shot, and every clip's
// own keyframe times are rebased to zero, which loses where each shot sat.
// `extras.animation_web` (MeshConvert.apply_glb_animations) is the file's own
// answer: it marks the clips a shot declared, carries their authoring frame
// ranges, and names the one to open on. Without the block this still plays —
// the clips are listed in file order under their own names — it just cannot say
// which of them are shots.
export let mixer = null;
// The deliverable's `animation_web` block, kept past setupAnimation: a script
// recording a clip needs the AUTHORING frame rate and frame numbers, and those
// live in the manifest rather than on the three.js clip (which knows only
// seconds). Null on a file that ships no manifest.
let animationManifest = null;
export let clips = [];        // [{clip, meta}] in PICKER order — declared shots first
let action = null;
let clipIndex = -1;
// Autoplay, deliberately: in an immersive session the DOM overlay is not
// rendered at all, so a headset user has no transport to press. A preview that
// arrives paused would look like a preview with no animation in it.
export let playing = true;
let scrubbing = false;
let lastTick = -1;   // last playhead position written to the DOM (0-1000)
// Last shot label written beside the playhead, so a segment change redraws the
// readout even when the playhead has not moved a visible amount — scrubbing
// slowly across a shot boundary is exactly when the label matters most, and it
// is the one case where the tick can repeat while the answer changes.
let lastShot = null;
// The synthetic whole-timeline entry, built from the shots when the file ships
// no continuous clip of its own (Animation Clips: "Shots Only"). Null while a
// real clip is selected — `activeSequence` is what the transport branches on,
// so a file that HAS a whole-timeline clip keeps the plain single-action path.
let sequence = null;
let activeSequence = null;
let sequenceTime = 0;      // playhead in seconds along the authoring span
let sequenceSegment = null;

// The frame range is what makes two clips that both start at t=0 tellable
// apart. "(full range)" is only said when something else IS a shot — with no
// shots at all, every clip is the full range and the label would be noise.
//
// "empty" is said whenever the deliverable says so: a declared shot whose range
// holds still bakes no curve, so it arrives named and carrying nothing. Picking
// it plays 0.00s, which reads as broken animation unless the label admits the
// clip itself is empty.
function clipLabel(entry, anyDeclared) {
  if (entry.sequence) {
    const s = entry.sequence;
    return `${s.name}  (${s.startFrame}–${s.endFrame}, ${s.segments.length} shots)`;
  }
  const meta = entry.meta;
  const empty = meta?.empty ? ', empty' : '';
  if (meta?.declared && typeof meta.start_frame === 'number') {
    return `${entry.clip.name}  (${meta.start_frame}–${meta.end_frame}${empty})`;
  }
  if (anyDeclared && !meta?.declared) return `${entry.clip.name}  (full range)`;
  return meta?.empty ? `${entry.clip.name}  (empty)` : entry.clip.name;
}

// --- authored alpha ramps (KHR_animation_pointer) ---------------------------
// glTF animates translation, rotation, scale and morph weights -- and nothing
// else. A material fade rides in `animations` only through KHR_animation_pointer,
// which lets a channel target any property by JSON pointer; the deliverable
// writes each authored ramp that way (MeshConvert.apply_glb_fades), on a
// material cloned per faded subtree so nothing else sampling it fades too.
// three.js does not implement the extension (checked against 0.180, well past
// the 0.169 pinned above): GLTFLoader SKIPS a channel with no `target.node`, so
// the clips arrive playable but silently minus their fades.
//
// This is the shim, and it is deliberately small: it translates the file's own
// pointer channels into native keyframe tracks on the loaded clips, and from
// then on the mixer plays a fade exactly the way it plays a translation -- same
// playhead, same scrub, same pause, no per-frame code and no frame arithmetic.
// The GLB is the single statement of the fade; the day three.js ships support,
// this function deletes and nothing else changes.
// The material properties a pointer can reach, and how each lane reaches the
// three.js material. One row per channel the exporter's table writes
// (pythontk `glb_fades.CHANNELS`): alpha is the fourth lane of the base colour
// factor and lands on `material.opacity`; a highlight is the whole VEC3
// emissive factor and lands on `material.emissive` as a colour track. Adding a
// channel there is adding a row here; the wiring below is shared.
const POINTER_BINDINGS = [
  {
    pointer: /^\/materials\/(\d+)\/pbrMetallicRoughness\/baseColorFactor$/,
    blended: true,
    track(mesh, material, times, output, count) {
      // The whole VEC4 is animated (glTF has no pointer to one component); the
      // alpha is its fourth lane. RGB is the material's own and is left to it.
      const alphas = new Float32Array(count);
      for (let k = 0; k < count; k += 1) alphas[k] = output.getW(k);
      return new THREE.NumberKeyframeTrack(`${mesh.uuid}.material.opacity`, times, alphas);
    },
  },
  {
    pointer: /^\/materials\/(\d+)\/emissiveFactor$/,
    blended: false,
    track(mesh, material, times, output, count) {
      // glTF factors are linear, as is the material's working colour space.
      const rgb = new Float32Array(count * 3);
      for (let k = 0; k < count; k += 1) {
        rgb[k * 3] = output.getX(k);
        rgb[k * 3 + 1] = output.getY(k);
        rgb[k * 3 + 2] = output.getZ(k);
      }
      return new THREE.ColorKeyframeTrack(`${mesh.uuid}.material.emissive`, times, rgb);
    },
  },
];

// A blended surface writes depth only while it is OPAQUE.
//
// glTF says nothing about depth, so GLTFLoader derives it: `alphaMode: BLEND`
// becomes `transparent = true, depthWrite = false`. That is right for a window
// and wrong for a solid object that merely happens to fade, and alphaMode is a
// property of the MATERIAL, not of the clip -- so an object that fades in
// during one shot renders with depth writes off in all twelve. With no depth
// written, a closed mesh draws its far faces over its near ones in index
// order, which reads exactly like inverted normals (measured on the production
// assembly: 15 materials, every one of them depthWrite false at full alpha).
//
// Restoring it wholesale would be the opposite bug -- an object at alpha 0.2
// would punch a hole in whatever is drawn behind it -- so the state follows
// the alpha the mixer is currently driving. This is DERIVED state, not a
// second copy of the fade: it reads whatever opacity the mixer set, so
// scrubbing, pausing and clip switches are all covered without any of them
// being named here. `onBeforeRender` runs before the draw's GL state is set,
// so the change lands on the same frame; `this` is the mesh, and it is one
// shared function rather than a closure per mesh.
function depthFromOpacity() {
  this.material.depthWrite = this.material.opacity >= 0.999;
}

// The rule is per MATERIAL but has to be installed on every MESH wearing one.
// The track binding needs a single representative mesh per material instance
// (two would drive the same property twice a frame), and a faded subtree can
// hold several meshes sharing its cloned material -- so installing only on the
// representative leaves the rule stale the moment that one is frustum-culled
// and another is not.
function installDepthRule(scene, materials) {
  if (!materials.size) return;
  scene.traverse((node) => {
    if (node.isMesh && !Array.isArray(node.material) && materials.has(node.material)) {
      node.onBeforeRender = depthFromOpacity;
    }
  });
}

// Returns `{tracks, materials}`: the tracks wired (one per clip, channel and
// material instance) and the distinct glTF materials they animate -- the HUD's
// count, since a fade in twelve shots is one faded material, not twelve.
export async function wirePointerTracks(gltf) {
  const parser = gltf.parser;
  const defs = parser.json.animations || [];
  if (!defs.length) return { tracks: 0, materials: 0 };

  // glTF material index -> the three.js materials the loader built from it.
  // Through the loader's own association map, which it carries onto the
  // variants it clones (vertex colours, flat shading), rather than by name:
  // the cloned fade material keeps its source's NAME on purpose, so the
  // lightmap pass -- which binds by name -- still recognises it.
  const byIndex = new Map();
  gltf.scene.traverse((node) => {
    // A multi-material mesh has no `.material.opacity` to bind; the loader
    // never builds one from a GLB (each primitive becomes its own Mesh), so
    // this is a foreign-file guard rather than a path the pipeline takes.
    if (!node.isMesh || Array.isArray(node.material)) return;
    for (const material of materialsOf(node)) {
      const index = parser.associations.get(material)?.materials;
      if (index === undefined) continue;
      if (!byIndex.has(index)) byIndex.set(index, new Map());
      // One binding per distinct material INSTANCE: two meshes sharing an
      // instance would otherwise drive the same property twice a frame.
      if (!byIndex.get(index).has(material)) byIndex.get(index).set(material, node);
    }
  });

  let wired = 0;
  const blended = new Set();
  const animated = new Set();  // glTF material indices a track now drives
  for (let i = 0; i < defs.length; i += 1) {
    const clip = gltf.animations[i];
    if (!clip) continue;
    for (const channel of defs[i].channels || []) {
      const target = channel.target || {};
      if (target.path !== 'pointer') continue;
      const pointer = target.extensions?.KHR_animation_pointer?.pointer || '';
      let binding = null;
      let match = null;
      for (const candidate of POINTER_BINDINGS) {
        match = candidate.pointer.exec(pointer);
        if (match) { binding = candidate; break; }
      }
      if (!binding) continue;
      const owners = byIndex.get(Number(match[1]));
      if (!owners) continue;
      animated.add(Number(match[1]));
      const sampler = defs[i].samplers[channel.sampler];
      const [input, output] = await Promise.all([
        parser.getDependency('accessor', sampler.input),
        parser.getDependency('accessor', sampler.output),
      ]);
      const times = Array.from(input.array);
      for (const [material, mesh] of owners) {
        // Bound by uuid rather than name: glTF names are not unique (the
        // production assembly ships two `prop533`), and a name binding would
        // drive the first match and leave the rest at full alpha.
        clip.tracks.push(binding.track(mesh, material, times, output, input.count));
        if (binding.blended) {
          // alphaMode BLEND already set transparent + depthWrite off at load;
          // stated here too so a file whose clone lost it still fades.
          material.transparent = true;
          blended.add(material);
        }
        wired += 1;
      }
    }
    clip.resetDuration();
  }
  installDepthRule(gltf.scene, blended);
  return { tracks: wired, materials: animated.size };
}

// Shots laid back onto the timeline they were cut from. The clips carry their
// own frame range, so the sequence is reconstructible without the stack they
// came from -- which is the whole point: shipping both halves costs the file
// the same performance twice (66.5 MB on the production assembly), and the
// half a shot-switching player never reads is the continuous one.
//
// Placed at their AUTHORED frames rather than end to end. The shots on that
// assembly sit 15 frames apart, and concatenating them would run the sequence
// short and put every shot at the wrong time -- a different thing from what
// the whole-timeline clip played. A gap holds the previous shot's last pose,
// which is what the continuous clip did there too.
function buildSequence(entries, manifest) {
  const fps = Number(manifest?.fps) || 0;
  if (!fps) return null;   // frames cannot be placed in time without it
  // Only when the file ships no continuous clip of its own: with one present
  // the picker would offer two ways to watch the same thing, and the real one
  // is exact where this is a reconstruction.
  if (entries.some((entry) => !entry.meta?.declared)) return null;
  // The TIMELINE is every declared shot; the SEGMENTS are the ones that carry
  // a curve. A shot whose range holds still bakes nothing and arrives empty,
  // and dropping it from the span as well would end the sequence early -- the
  // whole-timeline clip played through those frames, holding the last pose,
  // which is what the gap rule below already does for them.
  // BOTH bounds, not just the start: the span is measured across every
  // declared shot, so one entry missing an end_frame makes `last` NaN and the
  // whole sequence NaN-long -- a transport that divides by it draws nothing and
  // never recovers. A shot the manifest describes only half of is not a shot
  // this can place.
  const declared = entries.filter((e) => (
    typeof e.meta?.start_frame === 'number' && typeof e.meta?.end_frame === 'number'
  ));
  if (declared.length < 2) return null;
  const shots = declared
    .filter((e) => e.clip.duration > 0)
    .sort((a, b) => a.meta.start_frame - b.meta.start_frame);
  if (!shots.length) return null;

  const first = Math.min(...declared.map((e) => e.meta.start_frame));
  const last = Math.max(...declared.map((e) => e.meta.end_frame));
  const segments = shots.map((entry) => ({
    entry,
    offset: (entry.meta.start_frame - first) / fps,
    duration: entry.clip.duration,
  }));
  return {
    segments,
    fps,
    startFrame: first,
    endFrame: last,
    duration: (last - first) / fps,
    name: 'FULL SEQUENCE',
  };
}

// The segment a global time lands in: the LAST one that has started. A time
// inside a gap therefore resolves to the shot before it, whose pose is then
// held -- rather than to nothing, which would leave the model at whatever the
// previous frame drew.
function segmentAt(seconds) {
  if (!sequence) return null;
  let found = sequence.segments[0];
  for (const segment of sequence.segments) {
    if (segment.offset <= seconds) found = segment; else break;
  }
  return found;
}

// Pose the model at `seconds` along the sequence. The segment's action is
// driven by ASSIGNING its time, never by letting the mixer advance it: two
// actions must never accumulate time at once, and the gap hold depends on the
// clamp below rather than on where the mixer happened to get to.
function applySequence(seconds) {
  if (!sequence || !mixer) return;
  const segment = segmentAt(seconds);
  if (!segment) return;
  if (segment !== sequenceSegment) {
    if (action) action.stop();
    sequenceSegment = segment;
    action = mixer.clipAction(segment.entry.clip);
    action.reset();
    action.play();
    // Paused throughout: `mixer.update(0)` still poses the model at the
    // action's current time (the same property `seek` relies on), so the
    // sequence owns the clock and the mixer only evaluates.
    action.paused = true;
  }
  action.time = THREE.MathUtils.clamp(seconds - segment.offset, 0, segment.duration);
  mixer.update(0);
}

export function setupAnimation(gltf) {
  // Reset FIRST: a model swap must not leave the previous model's mixer driving
  // the frame loop, nor its clip names in the picker.
  if (mixer) mixer.stopAllAction();
  mixer = null;
  action = null;
  clips = [];
  clipIndex = -1;
  // The sequence is the previous model's too: `sequenceSegment` holds one of
  // its clips, and a swap to a file that builds no sequence would otherwise
  // leave `activeSequence` pointing at it -- the frame loop branches on that.
  sequence = null;
  activeSequence = null;
  sequenceSegment = null;
  sequenceTime = 0;

  animationManifest = null;

  const animations = gltf.animations || [];
  if (!animations.length) {
    el.anim.hidden = true;
    return 0;
  }

  const manifest = readExtras(gltf, SHOTS.webKey);
  animationManifest = manifest;
  // Joined on the glTF animation INDEX, which is what the block carries it by —
  // names are what a user reads, not a key: nothing stops two AnimStacks from
  // sharing one, and a name join would then hand one clip the other's frame
  // range. The block is written from this same file, so the indices agree.
  const byIndex = new Map((manifest?.clips || []).map((entry) => [entry.animation, entry]));
  const entries = animations.map((clip, index) => ({ clip, meta: byIndex.get(index) || null }));
  const anyDeclared = entries.some((entry) => entry.meta?.declared);
  // Declared shots first in the PICKER only. The glTF's own `animations` order
  // is left exactly as written, so a consumer keying on animation index still
  // reads the file the way every other reader does.
  clips = anyDeclared
    ? [...entries.filter((e) => e.meta?.declared), ...entries.filter((e) => !e.meta?.declared)]
    : entries;

  // FIRST in the picker when it exists: a deliverable that ships only shots is
  // still a sequence, and "watch the whole thing" is what a reviewer opens.
  sequence = buildSequence(entries, manifest);
  if (sequence) clips = [{ clip: null, meta: null, sequence }, ...clips];

  mixer = new THREE.AnimationMixer(gltf.scene);
  el.clipSelect.replaceChildren(...clips.map((entry, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = clipLabel(entry, anyDeclared);
    return option;
  }));
  el.clipSelect.hidden = clips.length < 2;
  el.anim.hidden = false;

  // The file's own answer to "which one first", falling back to the picker's
  // first entry — which is already a declared shot whenever there is one.
  // Resolved through the manifest entry rather than by comparing the glTF
  // clip's name, so it lands on the same clip the index join above did.
  const named = (manifest?.clips || []).find((entry) => entry.name === manifest.default_clip);
  const wanted = named ? clips.findIndex((entry) => entry.meta === named) : -1;
  selectClip(wanted >= 0 ? wanted : 0);
  // The FILE's clip count, not the picker's row count: a script reading this
  // off the 'load' event is asking what the deliverable carries, and the
  // synthetic whole-timeline entry is the page's reconstruction rather than
  // something the file ships.
  return animations.length;
}

// What is selected, in the terms a recorder needs: seconds for the transport,
// frames and a rate for the FILE it writes. The two must agree — a movie of a
// shot is compared against a viewport playblast of the same frames, so its
// length and its numbering are not the page's to round.
//
// The frame rate is the deliverable's, never the display's: the page renders at
// whatever the device manages, and a recording sampled at that rate would be a
// screen capture rather than a playblast.
export function clipInfo() {
  const entry = clips[clipIndex];
  if (!entry) return null;
  const fps = Number(animationManifest?.fps) || 0;
  if (entry.sequence) {
    return {
      name: entry.sequence.name,
      duration: entry.sequence.duration,
      fps: entry.sequence.fps,
      startFrame: entry.sequence.startFrame,
      endFrame: entry.sequence.endFrame,
      sequence: true,
    };
  }
  const duration = entry.clip?.duration || 0;
  const start = typeof entry.meta?.start_frame === 'number' ? entry.meta.start_frame : 0;
  // The clip's OWN length decides the end, not the manifest's end_frame: a
  // whole-timeline clip carries no declared range, and a declared one whose
  // curve is shorter than its range would otherwise ask for frames that do not
  // exist. A duration is the span BETWEEN the outer frames, so the last frame
  // is start + duration*fps and the count that spans them is one more than
  // that -- the same arithmetic the sequence's own startFrame/endFrame satisfy,
  // which is what lets a recorder count frames one way for both.
  const spans = fps ? Math.round(duration * fps) : 0;
  return {
    name: entry.clip?.name || 'clip',
    duration,
    fps,
    startFrame: start,
    endFrame: start + spans,
    sequence: false,
  };
}

export function selectClip(index) {
  if (!mixer || !clips[index]) return;
  if (action) action.stop();
  clipIndex = index;
  el.clipSelect.value = String(index);
  if (clips[index].sequence) {
    activeSequence = clips[index].sequence;
    sequenceTime = 0;
    sequenceSegment = null;  // force the first segment to bind
    action = null;
    applySequence(0);
    lastTick = -1;
    updateTransport();
    return;
  }
  activeSequence = null;
  sequenceSegment = null;
  action = mixer.clipAction(clips[index].clip);
  action.reset();
  action.play();
  // Play then pause, rather than not playing: a paused action still poses the
  // model at its current time, so a preview opened paused shows frame one of
  // the clip instead of the model's authored rest pose.
  action.paused = !playing;
  mixer.update(0);
  lastTick = -1; // the readout beside the playhead changed with the clip
  lastShot = null;
  updateTransport();
}

export function stepClip(delta) {
  if (clips.length < 2) return;
  selectClip((clipIndex + delta + clips.length) % clips.length);
}

export function togglePlay() {
  playing = !playing;
  // The sequence's segment action stays paused whatever the transport says --
  // it is posed by assignment, and un-pausing it would let the mixer advance
  // it alongside the sequence clock.
  if (action && !activeSequence) action.paused = !playing;
  el.playToggle.textContent = playing ? 'Pause' : 'Play';
}

export function seek(fraction) {
  if (activeSequence) {
    sequenceTime = THREE.MathUtils.clamp(
      fraction * activeSequence.duration, 0, activeSequence.duration);
    applySequence(sequenceTime);
    updateTransport();
    return;
  }
  if (!action) return;
  const duration = action.getClip().duration;
  action.time = THREE.MathUtils.clamp(fraction * duration, 0, duration);
  mixer.update(0); // apply the new time without advancing it
  updateTransport();
}

function updateTransport() {
  // The sequence reports its OWN clock, not the segment's: the playhead spans
  // the whole timeline, and the segment's local time would send it jumping
  // back to zero at every shot boundary.
  if (!activeSequence && !action) return;
  const duration = activeSequence
    ? activeSequence.duration
    : action.getClip().duration || 0;
  const at = activeSequence ? sequenceTime : action.time;
  const tick = duration ? Math.round((at / duration) * 1000) : 0;
  const shot = shotLabel(at);
  // This runs once per rendered frame — 120 times a second in a headset — and
  // the playhead only moves a VISIBLE amount 1000 times per clip. Writing the
  // same two DOM values on every other frame is pure cost on the device least
  // able to afford it. `lastTick` is reset by selectClip, so a clip swap that
  // lands on the same tick still redraws (the duration beside it changed).
  if (tick === lastTick && shot === lastShot) return;
  lastTick = tick;
  lastShot = shot;
  // Not while the user owns the slider: writing the playhead back into the
  // control being dragged fights the drag.
  if (!scrubbing) el.scrub.value = String(tick);
  // The authoring frame beside the seconds, for the sequence only: it is the
  // number a reviewer reads back to the animator, and the shot ranges in the
  // picker are quoted in it.
  const frame = activeSequence
    ? `  f${Math.round(activeSequence.startFrame + at * activeSequence.fps)}`
    : '';
  el.clipTime.textContent =
    `${shot ? `${shot} · ` : ''}${at.toFixed(2)} / ${duration.toFixed(2)}s${frame}`;
}

// Which shot the sequence playhead is standing in, for the readout. Only the
// sequence has one: on a single clip the picker already names what is playing,
// and repeating it beside the playhead would be noise.
//
// A time inside a GAP resolves to the shot before it — that is the pose on
// screen, held, exactly as the whole-timeline clip played it — so it is labelled
// as a hold rather than named outright. Without that the readout claims a shot
// is playing through frames it does not cover, which is the one reading of this
// label that could send someone looking for a bug in the shot.
export function shotLabel(seconds) {
  if (!activeSequence || !sequenceSegment) return null;
  const entry = sequenceSegment.entry;
  const name = entry.meta?.name || entry.clip?.name || 'shot';
  const held = seconds > sequenceSegment.offset + sequenceSegment.duration + 1e-6;
  return held ? `${name} (hold)` : name;
}

// The note the DCC's Shots panel carries for whatever the page is posed at —
// the segment's on the sequence, the selected clip's on a single shot. Unlike
// `shotLabel` it answers on BOTH, because its one reader is a burn-in rather
// than the readout: the picker naming a clip does not put its description in
// the movie, and a description is the half of a shot record a reviewer
// watching the file cannot otherwise see.
//
// Empty is null, not '': `extras.animation_web` omits the key on a shot with
// no note (MeshConvert.apply_glb_animations writes only truthy fields), and a
// caller should not have to tell "no description" from "an empty one".
export function shotDescription() {
  const entry = activeSequence && sequenceSegment
    ? sequenceSegment.entry
    : clips[clipIndex];
  return entry?.meta?.description || null;
}

// Whether the user owns the scrubber (a drag, an arrow key) -- the transport
// then leaves the playhead alone (see updateTransport).
export function setScrubbing(on) {
  scrubbing = on;
}

// One frame of the transport, before the page's 'frame' hook, so a script
// sees the model posed for THIS frame rather than the previous one.
export function advance(delta) {
  if (!mixer) return;
  if (activeSequence) {
    // The sequence advances its own clock and POSES through applySequence;
    // handing the same delta to the mixer as well would advance the segment
    // action a second time and run it ahead of the playhead.
    if (playing) {
      sequenceTime += delta;
      if (sequenceTime >= activeSequence.duration) sequenceTime = 0;  // loop
    }
    applySequence(sequenceTime);
  } else {
    mixer.update(delta);
  }
  updateTransport();
}
