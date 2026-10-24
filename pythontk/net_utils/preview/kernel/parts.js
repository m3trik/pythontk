// The PARTS on screen: the published files, one or several, and what the
// page offers per file.
//
// A push may publish several files at once -- the parts of one scene, each
// exported on its own (a room, the props in it, a character) -- and the page
// shows them TOGETHER at their authored places, so they line up the way they
// do in the scene they came from. Each part is a file the server serves as
// it stands; the load (load.js) parses each, runs the per-file passes on each
// (its fades, its lightmaps), and hangs every part's scene under one root,
// which is the model the rest of the page measures, frames and stands.
//
// What is the deliverable's rather than a file's is answered once: the bake's
// reflection probe and the lighting recipe come from the first part that
// carries one, and the clips of every part are one picker (see `composite`).
//
// Per part, the menus offer: View ▸ the part shown or hidden (only with more
// than one -- a single file has nothing to hide it from), and Export ▸ the
// GLB as served, under its own name. A part hidden here stays hidden across
// pushes, by name, so a part set aside for a look at the rest is not handed
// back by the next push.

import * as THREE from 'three';
import { download } from './dom.js';
import { menuFor } from './menus.js';
import { SHOTS } from './records.js';
import { readExtras } from './scene.js';

//: The parts on screen, `[{name, url, gltf, root}]`, in published order --
//: the first is the PRIMARY, whose glTF a script's 'load' event names.
export let parts = [];

//: Part names hidden in View, kept across loads.
const hidden = new Set();
//: The menu items the current parts filed, removed when they are replaced.
let items = [];

const viewMenu = menuFor('View');
const exportMenu = menuFor('Export');

// The file name a part is saved under: its published name, with the served
// file's extension (an FBX source is served as the GLB built from it).
function saveName(part) {
  const served = (part.url.split('?')[0].match(/\.[A-Za-z0-9]+$/) || ['.glb'])[0];
  return part.name.replace(/\.[A-Za-z0-9]+$/, '') + served;
}

/**
 * The parts on screen become *loaded* -- `[{name, url, gltf}]`, each parsed
 * -- under *root*. Their menu items replace the last load's; a part hidden
 * by name before stays hidden.
 */
export function setParts(loaded, root) {
  for (const item of items) item.remove();
  items = [];
  parts = loaded.map((part) => ({ ...part, root: part.gltf.scene }));
  for (const part of parts) {
    part.root.userData.part = part.name;
    // A lone file is never hidden: it has no View switch to bring it back.
    part.root.visible = parts.length < 2 || !hidden.has(part.name);
  }
  root.userData.parts = parts.map((part) => part.name);

  parts.forEach((part, index) => {
    if (parts.length > 1) {
      items.push(viewMenu.add('toggle', {
        label: part.name,
        title: `Show or hide ${part.name}; the rest stay where they are`,
        group: 'parts',
        order: 100 + index,
        value: part.root.visible,
        onChange: (on) => {
          part.root.visible = on;
          if (on) hidden.delete(part.name); else hidden.add(part.name);
        },
      }));
    }
    items.push(exportMenu.add('action', {
      label: parts.length > 1 ? `GLB: ${saveName(part)}` : 'GLB',
      title: `Save ${saveName(part)} as it is served here: the file the page is showing`,
      group: 'model',
      order: index,
      // Against the page, never the asset's absolute address: a browser
      // honours a download name on the page's own origin only.
      run: () => download(part.url.split('?')[0], saveName(part)),
    }));
  });
}

// A track bound by NAME, re-bound to its node's uuid. Every part's names are
// unique only within that part -- the loader makes them so per file -- and
// one mixer over the parts' shared root would drive the first match it
// found, in whichever part came first. Tracks already bound by uuid (the
// fades' pointer tracks) are left alone.
function retarget(clip, partRoot) {
  for (const track of clip.tracks) {
    const { nodeName } = THREE.PropertyBinding.parseTrackName(track.name);
    if (!nodeName || !track.name.startsWith(nodeName)) continue;
    const node = THREE.PropertyBinding.findNode(partRoot, nodeName);
    if (!node || node.uuid === nodeName) continue;
    track.name = node.uuid + track.name.slice(nodeName.length);
  }
}

// The parts' `animation` blocks as one: each clip's index moved past the
// clips of the parts before it, each shot kept once (the parts of one scene
// state the same shots), and the frame rate and the clip to open on taken
// from the first part that names them. Null when no part has one.
function mergedShots(loaded) {
  let merged = null;
  let offset = 0;
  for (const { gltf } of loaded) {
    /** @type {import('./records.js').AnimationBlock} */
    const block = readExtras(gltf, SHOTS);
    if (block) {
      if (!merged) merged = { ...block, clips: [], shots: [] };
      if (merged.fps === undefined && block.fps !== undefined) merged.fps = block.fps;
      if (!merged.default_clip && block.default_clip) merged.default_clip = block.default_clip;
      for (const entry of block.clips || []) {
        merged.clips.push({ ...entry, animation: entry.animation + offset });
      }
      for (const shot of block.shots || []) {
        if (!merged.shots.some((kept) => kept.clip === shot.clip)) merged.shots.push(shot);
      }
    }
    offset += (gltf.animations || []).length;
  }
  return merged;
}

/**
 * What the deliverable-wide passes read when there are several parts: the
 * shared *root* as one glTF -- every part's clips (re-bound, above), and the
 * loader's association tables merged, which is all the model's measure and
 * its transport ask of it. One part is passed through untouched.
 *
 * Returns `{gltf, shots}`: `shots` the `animation` block to hand the
 * transport -- one part's own, or the parts' merged -- null with none.
 */
export function composite(loaded, root) {
  if (loaded.length === 1) {
    return { gltf: loaded[0].gltf, shots: readExtras(loaded[0].gltf, SHOTS) };
  }
  const associations = new Map();
  const animations = [];
  for (const { gltf } of loaded) {
    for (const [key, value] of gltf.parser?.associations || []) associations.set(key, value);
    for (const clip of gltf.animations || []) {
      retarget(clip, gltf.scene);
      animations.push(clip);
    }
  }
  return {
    gltf: { scene: root, scenes: [root], animations, parser: { associations } },
    shots: mergedShots(loaded),
  };
}
