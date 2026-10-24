/*
  Relighting a loaded model in place -- the seam lighting work drives.

  A lighting tool (pythontk's look match) tries a lighting in seconds by
  swapping the bake itself: each lightmapped material's map, and the
  reflection probe, replaced by linear half-float data the tool sends -- no
  re-publish, no reload. `restoreLighting` puts the file's own bake back.
  `renderLinear` reads a view as linear light, before the tone map and
  without the background: the frame a layer of light is measured in. A swap
  that brings the bake's lights (`highlights`, layers drawn as the deliverable
  draws them: MeshConvert.highlight_probe_layers) builds the probe's highlight
  copies from them, as a load does from the file's -- the frame is what ships.

  Exposed as `viewer.lighting` (api.js).
*/

import * as THREE from 'three';
import { applyLighting, lightmappedMaterials } from './lightmaps.js';
import { current } from './model.js';
import { parts } from './parts.js';
import { addProbe, disposeProbe, loadProbe, prefilterEquirect, releaseProbe, useProbe } from './probe.js';
import { renderer, scene } from './scene.js';

//: material -> the map and intensity its file bound, kept at its first swap.
const originals = new Map();
//: Textures the current swap made; the next swap (or a restore) frees them.
let made = [];

/**
 * @typedef {{width: number, height: number, data: Uint16Array}} HalfImage
 *   Linear RGBA, half floats, rows top-down as an image's are.
 */

/** @param {HalfImage} image @returns {any} a three.js DataTexture */
function halfTexture(image) {
  const texture = new THREE.DataTexture(image.data, image.width, image.height, THREE.RGBAFormat, THREE.HalfFloatType);
  texture.colorSpace = THREE.LinearSRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.flipY = false;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Swap the bake: every lightmapped material whose map `maps` names takes that
 * image (keeping its own UV channel and transform), and `probe`, when given,
 * becomes the reflection probe.
 *
 * @param {{
 *   maps: Object<string, string>,
 *   images?: Object<string, HalfImage>,
 *   probe?: (HalfImage & {position: number[], box?: {min: number[], max: number[]} | null}) | null,
 *   highlights?: (HalfImage & {materials: string[]})[],
 * }} lighting `maps`: material name -> image name; `images`: image name -> data;
 *   `highlights`: the probe's highlight layers, each added to `probe` for its
 *   materials (a swap without a probe keeps the one up, copies and all).
 * @returns {number} How many materials took a map.
 */
export function relight({ maps, images = {}, probe = null, highlights = [] }) {
  for (const material of originals.keys()) {
    if (!lightmappedMaterials.includes(material)) originals.delete(material);  // a model since replaced
  }
  const sources = {};
  for (const [name, image] of Object.entries(images)) sources[name] = halfTexture(image);
  const previous = made;
  made = Object.values(sources);
  let bound = 0;
  for (const material of lightmappedMaterials) {
    const source = sources[maps[material.name]];
    if (!source || !material.lightMap) {
      // Left out of this swap: the file's own map again, never the texture of
      // an earlier swap, which is freed below.
      const own = originals.get(material);
      if (own) {
        material.lightMap = own.map;
        material.lightMapIntensity = own.intensity;
        originals.delete(material);
      }
      continue;
    }
    if (!originals.has(material)) {
      originals.set(material, { map: material.lightMap, intensity: material.lightMapIntensity });
    }
    const own = originals.get(material).map;
    const texture = source.clone();  // shares the image; the transform is this material's
    texture.channel = own.channel;
    texture.offset.copy(own.offset);
    texture.repeat.copy(own.repeat);
    texture.rotation = own.rotation;
    texture.center.copy(own.center);
    texture.matrixAutoUpdate = own.matrixAutoUpdate;
    if (!own.matrixAutoUpdate) texture.matrix.copy(own.matrix);
    texture.needsUpdate = true;
    material.lightMap = texture;
    material.lightMapIntensity = 1;  // the data is linear light: no encode scalar to undo
    made.push(texture);
    bound += 1;
  }
  for (const texture of previous) texture.dispose();
  if (probe && current) {
    const prefiltered = prefilterEquirect(probe.data, probe.width, probe.height, THREE.HalfFloatType);
    const copies = [];
    if (highlights.length) {
      const base = { width: probe.width, height: probe.height, data: floats(probe.data) };
      for (const layer of highlights) {
        const data = addProbe(base, { width: layer.width, height: layer.height, data: floats(layer.data) });
        if (!data) continue;
        copies.push({
          prefiltered: prefilterEquirect(data, layer.width, layer.height, THREE.FloatType),
          materials: layer.materials || [],
        });
      }
    }
    useProbe({ prefiltered, copies, position: probe.position, box: probe.box || null }, current);
  }
  applyLighting();
  return bound;
}

/** @param {Uint16Array} half RGBA half floats @returns {Float32Array} */
function floats(half) {
  const out = new Float32Array(half.length);
  for (let i = 0; i < half.length; i++) out[i] = THREE.DataUtils.fromHalfFloat(half[i]);
  return out;
}

/** Put the file's own bake back: its lightmaps and its reflection probe (copies and all). */
export async function restoreLighting() {
  for (const [material, own] of originals) {
    material.lightMap = own.map;
    material.lightMapIntensity = own.intensity;
  }
  originals.clear();
  for (const texture of made) texture.dispose();
  made = [];
  const model = current;
  let own = null;
  for (const part of parts) {
    own = await loadProbe(part.gltf);
    if (own) break;
  }
  if (current !== model) {
    // A newer load took the page while the probe decoded: its probe is up,
    // not this file's.
    if (own) disposeProbe(own);
    return;
  }
  if (own && model) useProbe(own, model);
  else {
    // A file with no probe of its own shows none -- not the one a swap put up.
    if (own) disposeProbe(own);
    releaseProbe();
  }
  applyLighting();
}

/**
 * A view as linear light: the scene through *camera*, no tone map, no background.
 *
 * @param {{camera: any, width: number, height: number}} view  `camera`: a three.js camera
 * @returns {Float32Array} RGBA, `width * height * 4`, rows bottom-up as WebGL reads them.
 */
export function renderLinear({ camera, width, height }) {
  const target = new THREE.WebGLRenderTarget(width, height, { type: THREE.FloatType });
  const toneMapping = renderer.toneMapping;
  const background = scene.background;
  renderer.toneMapping = THREE.NoToneMapping;
  scene.background = null;
  try {
    renderer.setRenderTarget(target);
    renderer.render(scene, camera);
    const pixels = new Float32Array(width * height * 4);
    renderer.readRenderTargetPixels(target, 0, 0, width, height, pixels);
    return pixels;
  } finally {
    renderer.setRenderTarget(null);
    renderer.toneMapping = toneMapping;
    scene.background = background;
    target.dispose();
  }
}
