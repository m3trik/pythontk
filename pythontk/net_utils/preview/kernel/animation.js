import * as THREE from 'three';
import { el } from './dom.js';
import { fileMaterialKey } from './lightmaps.js';
import { materialsOf } from './model.js';
import { SHOTS } from './records.js';
import { readExtras } from './scene.js';

// A shot deliverable ships one clip per shot: the FBX's takes, one per shot,
// each named after it and spanning its range, converted one glTF animation
// each, every clip's own keyframe times rebased to zero. `extras.animation` is
// the scene's own `animation` record, carried as the FBX holds it
// (MeshConvert.apply_glb_animations): its shots in timeline order -- every
// one, whatever clips the file holds, each with its description and metadata
// -- and the scene's frame rate (`fps`), which glTF cannot hold. A shot plays
// the animation its `clip` NAMES, and the shots' clips played in turn are the
// whole sequence. Without the record this still plays -- the clips are listed
// in file order under their own names -- it just cannot say which of them are
// shots, nor the rate they were cut at.
export let mixer = null;
// The deliverable's `animation` record, kept past setupAnimation: the shots'
// descriptions are read off it by clip name, and a recorder reads its rate.
// Null on a file that ships none.
let animationManifest = null;
export let clips = [];        // [{clip, meta}] in PICKER order -- the sequence, then the shots
let action = null;
let clipIndex = -1;
// Autoplay, deliberately: in an immersive session the DOM overlay is not
// rendered at all, so a headset user has no transport to press. A preview that
// arrives paused would look like a preview with no animation in it.
export let playing = true;
let scrubbing = false;
let lastTick = -1;   // last playhead position written to the DOM (0-1000)
// Last shot label written beside the playhead, so a segment change redraws the
// readout even when the playhead has not moved a visible amount -- scrubbing
// slowly across a shot boundary is exactly when the label matters most, and it
// is the one case where the tick can repeat while the answer changes.
let lastShot = null;
// The whole sequence: the shot clips played in turn. Null while a single clip
// is selected -- `activeSequence` is what the transport branches on.
let sequence = null;
let activeSequence = null;
let sequenceTime = 0;      // playhead in seconds along the sequence
let sequenceSegment = null;

// "empty" is said of a clip whose glTF animation has no channels: a shot that
// holds still bakes no curve, so it arrives named and carrying nothing.
// Picking it plays 0.00s, which reads as broken animation unless the label
// admits the clip itself is empty. Told by the clip the loader built from
// those channels, which has no tracks -- so a clip of a set of parts
// (parts.js) is told just as a lone file's is.
function clipLabel(entry) {
  if (entry.sequence) return `${entry.sequence.name}  (${entry.sequence.shotCount} shots)`;
  return entry.clip.tracks.length ? entry.clip.name : `${entry.clip.name}  (empty)`;
}

// A clip joined to the shot it plays, BY NAME -- the shot's `clip` is its
// take's name, which the conversion guarantees is the animation's
// (MeshConvert.apply_glb_animations) -- and this is the one place the two
// meet: every reader below takes the joined meta. `name` is the clip's,
// `animation` its index in the file, `shot` whether a shot plays it.
function clipMeta(clip, index, shots) {
  const name = clip.name || `animation_${index}`;
  const shot = shots.get(name);
  if (!shot) return { name, animation: index, shot: false };
  return {
    name,
    animation: index,
    shot: true,
    ...(shot.description ? { description: shot.description } : {}),
    ...(shot.metadata ? { metadata: shot.metadata } : {}),
  };
}

// The scene's frame rate as the record states it, or 0 when it states none
// (an older or a foreign GLB): the page guesses none -- a rate read off a
// clip's key spacing is wrong wherever key reduction left a gap.
function recordFps() {
  const fps = Number(animationManifest?.fps);
  return Number.isFinite(fps) && fps > 0 ? fps : 0;
}

// --- animated render effects (KHR_animation_pointer) ------------------------
// glTF animates translation, rotation, scale and morph weights -- and nothing
// else. A material fade, highlight or animated emission rides in `animations`
// only through KHR_animation_pointer, which lets a channel target any property
// by JSON pointer; the deliverable writes each effect's curve that way
// (MeshConvert.apply_glb_fades), on a material copied per animated subtree so
// nothing else sampling it moves too. three.js does not implement the
// extension (checked against 0.180, well past the 0.169 pinned above):
// GLTFLoader SKIPS a channel with no `target.node`, so the clips arrive
// playable but silently minus their effects.
//
// This is the shim, and it is deliberately small: it translates the file's own
// pointer channels into native keyframe tracks on the loaded clips, and from
// then on the mixer plays an effect exactly the way it plays a translation --
// same playhead, same scrub, same pause, no frame arithmetic. The GLB is the
// single statement of each curve; the day three.js ships support, the track
// building deletes and only the render-effect rules below remain.
// The material properties a pointer can reach, and how each lane reaches the
// three.js material. One row per property the exporter's table writes
// (pythontk `glb.fades.CHANNELS`): alpha is the fourth lane of the base colour
// factor and lands on `material.opacity`; a highlight or an animated emission
// is the whole VEC3 emissive factor and lands on `material.emissive` as a
// colour track. Adding a channel there is adding a row here; the wiring below
// is shared.
const POINTER_BINDINGS = [
  {
    pointer: /^\/materials\/(\d+)\/pbrMetallicRoughness\/baseColorFactor$/,
    fades: true,
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
    fades: false,
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

// The property the shim keys a highlight's glow on: 1 while the highlight is
// above 0, else 0, stepped -- the spans the exporter states on the channel
// (`extras.render_effects.active`), since glTF cannot animate a texture's
// presence. A plain number on the material so the mixer can drive it, and
// restore it to 0 when no playing clip does.
const GLOWING = 'renderEffectsGlowing';

// The render-effect rules, per drawn MESH, on the state the mixer just set --
// derived, never a second copy of a curve, so scrubbing, pausing and clip
// switches are all covered without any of them being named here.
//
// Opaque at rest (the `render_effects` contract): a material keeps its
// authored alpha mode -- opaque, sorted with the opaque pass, writing depth --
// while its opacity is 1, and is blended only while its opacity is below 1.
// glTF says nothing about depth, so GLTFLoader derives it from alphaMode; a
// fading copy of an opaque material arrives opaque, and its blending is
// switched on here for the frames it fades and back off at full opacity. A
// solid object that wrote no depth would draw its far faces over its near
// ones (measured on the production assembly when every faded material shipped
// BLEND: it reads exactly like inverted normals), and one blended at full
// opacity would sort with the transparent pass for no reason. At 0 its node
// is hidden by the deliverable's own presence gate (a stepped scale).
//
// The highlight over a map: while the highlight is above 0 the glow is the
// highlight itself, and the material's mapped emission is set aside for those
// moments (glTF emission is factor x map, and a map hides an added glow:
// measured 2026-09-05 on a production board, the channel was written and
// nothing changed on screen); at rest the map, and its strength, are back.
//
// `onBeforeRender` runs before the draw's GL state is set, so a change lands
// on the same frame; `this` is the mesh, and it is one shared function rather
// than a closure per mesh. A program switch (`needsUpdate`) happens only on
// the frames a state flips.
function renderEffectRules() {
  const material = this.material;
  const rest = material.userData.renderEffectsRest;
  if (!rest) return;
  if (rest.fades) {
    const fading = material.opacity < 0.999;
    if (rest.legacy) {
      // A file built before `render_effects` v2 ships its fade copies BLEND
      // for good: depth follows the alpha, as the page has always drawn it.
      material.depthWrite = !fading;
    } else {
      const transparent = rest.transparent || fading;
      if (material.transparent !== transparent) {
        material.transparent = transparent;
        material.needsUpdate = true;
      }
      material.depthWrite = rest.depthWrite && !fading;
    }
  }
  if (rest.glows) {
    const glowing = material[GLOWING] > 0.5;
    if ((material.emissiveMap === null) !== glowing) {
      material.emissiveMap = glowing ? null : rest.emissiveMap;
      material.emissiveIntensity = glowing ? 1 : rest.emissiveIntensity;
      material.needsUpdate = true;
    }
  }
}

// What a material is at rest, read once before any rule touches it: what the
// rules switch back to.
function restOf(material) {
  if (!material.userData.renderEffectsRest) {
    material.userData.renderEffectsRest = {
      fades: false,
      glows: false,
      legacy: false,
      transparent: material.transparent,
      depthWrite: material.depthWrite,
      emissiveMap: material.emissiveMap || null,
      emissiveIntensity: material.emissiveIntensity ?? 1,
    };
  }
  return material.userData.renderEffectsRest;
}

// The rules are per MATERIAL but have to be installed on every MESH wearing
// one. The track binding needs a single representative mesh per material
// instance (two would drive the same property twice a frame), and an
// animated subtree can hold several meshes sharing its copied material -- so
// installing only on the representative leaves the rule stale the moment
// that one is frustum-culled and another is not.
function installRules(scene, materials) {
  if (!materials.size) return;
  scene.traverse((node) => {
    if (node.isMesh && !Array.isArray(node.material) && materials.has(node.material)) {
      node.onBeforeRender = renderEffectRules;
    }
  });
}

// The glow track of a highlight channel over a mapped material, or null: the
// channel's stated spans, stepped.
function glowTrack(mesh, material, channel) {
  const active = channel.extras?.render_effects?.active;
  if (!material.emissiveMap || !Array.isArray(active) || !active.length) return null;
  if (material[GLOWING] === undefined) material[GLOWING] = 0;
  return new THREE.NumberKeyframeTrack(
    `${mesh.uuid}.material.${GLOWING}`,
    active.map((run) => Number(run[0])),
    active.map((run) => (run[1] ? 1 : 0)),
    THREE.InterpolateDiscrete,
  );
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
  // the copied material keeps its source's NAME on purpose, so the lightmap
  // pass -- which binds by name -- still recognises it.
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
  const ruled = new Set();
  // The file's materials a track now drives: by glTF index, a copy the
  // lightmap bind made apart from its original (fileMaterialKey).
  const animated = new Set();
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
      for (const material of owners.keys()) animated.add(fileMaterialKey(material, parser.associations));
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
        const rest = restOf(material);
        if (binding.fades) {
          rest.fades = true;
          // A channel that does not say what it plays is an older file's.
          rest.legacy = !channel.extras?.render_effects;
          ruled.add(material);
        }
        const glow = glowTrack(mesh, material, channel);
        if (glow) {
          clip.tracks.push(glow);
          rest.glows = true;
          ruled.add(material);
        }
        wired += 1;
      }
    }
    clip.resetDuration();
  }
  installRules(gltf.scene, ruled);
  return { tracks: wired, materials: animated.size };
}

// The whole sequence: the shots' clips played in turn, in the record's order
// -- the takes' timeline order. Each clip is its take, which spans its shot,
// so the clips in turn are the sequence; no whole-timeline clip is needed (a
// file with shots carries none). A clip's duration is the span BETWEEN its
// first and last frames, so each cut HOLDS the clip's last frame for one
// frame at the record's rate before the next clip's first: laid end to end
// on its duration alone, every shot after the first would play one frame
// early, its last frame never shown. A file that states no rate cannot say
// how long a frame is, and plays them end to end. A shot whose clip holds no
// curve -- which this build never ships (each shot's take is held to its window,
// _Animation._hold_take_spans), a file built elsewhere may -- arrives empty
// or not at all: it has no length to play and is passed over.
function buildSequence(entries, manifest) {
  const byName = new Map(entries.map((entry) => [entry.meta.name, entry]));
  const ordered = (manifest?.shots || [])
    .map((shot) => byName.get(shot.clip))
    .filter((entry) => entry && entry.clip.duration > 0);
  if (ordered.length < 2) return null;
  const fps = recordFps();
  const hold = fps ? 1 / fps : 0;
  let offset = 0;
  const segments = ordered.map((entry, index) => {
    const segment = { entry, offset, duration: entry.clip.duration };
    offset += entry.clip.duration + (index < ordered.length - 1 ? hold : 0);
    return segment;
  });
  return {
    segments,
    shotCount: segments.length,
    fps,
    duration: offset,
    name: 'FULL SEQUENCE',
  };
}

// The segment a sequence time lands in: the LAST one that has started, so the
// frame a cut holds is the clip before it, and the sequence's end still poses
// the last shot's last frame. Within a billionth: a recorder posing frame i
// at i / fps meets a segment's offset summed the other way round.
function segmentAt(seconds) {
  if (!sequence) return null;
  let found = sequence.segments[0];
  for (const segment of sequence.segments) {
    if (segment.offset <= seconds + 1e-9) found = segment; else break;
  }
  return found;
}

// Pose the model at `seconds` along the sequence. The segment's action is
// driven by ASSIGNING its time, never by letting the mixer advance it: two
// actions must never accumulate time at once, and the hold at a cut depends on
// the clamp below rather than on where the mixer happened to get to.
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

// *manifest* is the deliverable's `animation` record -- read off *gltf* by
// default; a set of parts hands in theirs merged (parts.js), whose clips the
// merged `animations` hold by name.
/** @param {import('./records.js').AnimationRecord | null} [manifest] */
export function setupAnimation(gltf, manifest = readExtras(gltf, SHOTS)) {
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

  animationManifest = manifest;
  // Joined by NAME: a shot's clip is the animation of its name in this file,
  // or in any of a set of parts -- the record binds nothing to an index.
  const shots = new Map((manifest?.shots || []).map((shot) => [shot.clip, shot]));
  const entries = animations.map((clip, index) => ({
    clip,
    meta: clipMeta(clip, index, shots),
  }));
  // Shots first in the PICKER only, each group in the file's order -- the
  // takes' timeline order. The glTF's own `animations` order is left exactly
  // as written, so a consumer keying on animation index still reads the file
  // the way every other reader does.
  clips = [
    ...entries.filter((entry) => entry.meta?.shot),
    ...entries.filter((entry) => !entry.meta?.shot),
  ];

  // FIRST in the picker when it exists: a deliverable of shots is a
  // sequence, and "watch the whole thing" is one pick away.
  sequence = buildSequence(entries, manifest);
  if (sequence) clips = [{ clip: null, meta: null, sequence }, ...clips];

  mixer = new THREE.AnimationMixer(gltf.scene);
  el.clipSelect.replaceChildren(...clips.map((entry, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = clipLabel(entry);
    return option;
  }));
  el.clipSelect.hidden = clips.length < 2;
  el.anim.hidden = false;

  // Opened on the first shot's clip that plays, else the first clip that
  // plays, else the first shot's, else the first -- read off the record's
  // shots, as the retired block named the clip to open on. Not the sequence: a
  // reviewer opens on a shot, the whole thing one pick away. Never a clip
  // with no channels while one plays: that shows 0.00s of nothing, which
  // reads as broken animation rather than as a shot that holds (the build
  // drops such clips, prune_glb_animations; a file handed in directly may
  // still carry one).
  const isShot = (entry) => !entry.sequence && Boolean(entry.meta?.shot);
  const plays = (entry) => !entry.sequence && entry.clip.tracks.length > 0;
  const opening = [(entry) => isShot(entry) && plays(entry), plays, isShot]
    .map((test) => clips.findIndex(test))
    .find((index) => index >= 0);
  selectClip(opening ?? 0);
  // The FILE's clip count, not the picker's row count: a script reading this
  // off the 'load' event is asking what the deliverable carries, and the
  // sequence entry is the page's composition rather than something the file
  // ships.
  return animations.length;
}

// What is selected, in the terms a recorder needs: its length in seconds,
// glTF's unit, and the scene's frame rate (`fps`), which the record states
// for a file whose format holds none -- the rate a recording steps it at,
// never the display's: the page renders at whatever the device manages, and
// a recording sampled at that rate would be a screen capture rather than a
// playblast. 0 on a file that states none: the page guesses none (a rate read
// off a clip's key spacing is wrong wherever key reduction left a gap), and a
// reader that steps frames then brings its own (the playblast asks). No
// frame numbers: a shot is named, and need not start at any frame.
export function clipInfo() {
  const entry = clips[clipIndex];
  if (!entry) return null;
  return {
    name: entry.sequence ? entry.sequence.name : entry.clip?.name || 'clip',
    duration: entry.sequence ? entry.sequence.duration : entry.clip?.duration || 0,
    fps: recordFps(),
    sequence: Boolean(entry.sequence),
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
    lastShot = null;
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
  // the whole sequence, and the segment's local time would send it jumping
  // back to zero at every shot boundary.
  if (!activeSequence && !action) return;
  const duration = activeSequence
    ? activeSequence.duration
    : action.getClip().duration || 0;
  const at = activeSequence ? sequenceTime : action.time;
  const tick = duration ? Math.round((at / duration) * 1000) : 0;
  const shot = shotLabel(at);
  // This runs once per rendered frame -- 120 times a second in a headset -- and
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
  el.clipTime.textContent =
    `${shot ? `${shot} · ` : ''}${at.toFixed(2)} / ${duration.toFixed(2)}s`;
}

// Which shot the playhead is standing in, for the readout: on the sequence,
// the shot whose clip is playing. A single clip has none: the picker already
// names what is playing, and repeating it beside the playhead would be noise.
export function shotLabel(seconds) {
  if (!activeSequence) return null;
  const segment = segmentAt(seconds);
  if (!segment) return null;
  return segment.entry.meta?.name || segment.entry.clip?.name || 'shot';
}

// The note the DCC's Shots panel carries for whatever the page is posed at --
// the segment's shot on the sequence, the selected clip's on a single shot,
// read off the block's shots by clip name. Unlike `shotLabel` it answers on
// BOTH, because its one reader is a burn-in rather than the readout: the
// picker naming a clip does not put its description in the movie, and a
// description is the half of a shot record a reviewer watching the file
// cannot otherwise see.
//
// Empty is null, not '': `extras.animation` omits the key on a shot with
// no note (MeshConvert.apply_glb_animations writes only truthy fields), and a
// caller should not have to tell "no description" from "an empty one".
export function shotDescription() {
  const entry = activeSequence && sequenceSegment
    ? sequenceSegment.entry
    : clips[clipIndex];
  return entry?.meta?.description || null;
}

// Whether any shot of the model on screen carries a description -- what
// decides that there is a caption to offer (captions.js).
export function hasDescriptions() {
  return (animationManifest?.shots || []).some((shot) => Boolean(shot.description));
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
