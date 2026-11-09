// Baked lightmaps: binding the deliverable's maps to the objects its record
// names, the shader edits a baked material is drawn with, and the lighting
// policy that turns on how much of the model is baked.

import * as THREE from 'three';
import { mediaOf } from './extras.js';
import { plural } from './format.js';
import { materialsOf } from './model.js';
import { probeEnvironment, syncProbeCopies } from './probe.js';
import { LIGHTMAPS, LIGHTMAP_UV_SET } from './records.js';
import { resolveEntries } from './resolve.js';
import {
  DEFAULT_BAKED_ENV_INTENSITY, keyLight, policy, readExtras, releaseStudio, renderer, scene,
  studioEnvironment,
} from './scene.js';

// glTF 2.0 has no lightmap slot, and the deliverable binds none: it carries
// the `lightmaps` record as it stands in its root extras -- each baked
// object's map, read through its second UV set at its scale-offset -- and
// every map as the bufferView named by its file (records.js GLB_MEDIA), in a
// web encoding: 8-bit sRGB whose decoded texels times the view's
// `extras.scale` are the map. This puts each map on the objects the record
// names, resolved in this file's scene by the rule every reader resolves the
// record by (resolve.js), on materials of their own where objects that share a
// material light differently. Four details decide whether it looks right:
//
//   * colorSpace -- the map is sRGB-encoded to survive 8-bit; read linear it
//     renders far too dark.
//   * channel -- the map is read through UV set 1 (LightmapRecord.UV_SET).
//   * scale -- the encoder divided by the map's (capped) peak to fit its HDR
//     range into 8 bits; `lightMapIntensity` multiplies it back.
//   * level -- a texel is a white card's radiance, drawn as stored: texel x
//     albedo, as an engine's lightmap path and the baking renderer draw it
//     (HandoffRecord's probe clause). three.js
//     reads a lightMap as irradiance (its diffuse is texel x albedo / pi), so
//     `lightMapIntensity` also carries pi (LIGHTMAP_AS_STORED); the probe then
//     plays as captured, at 1 (scene.js DEFAULT_PROBE_INTENSITY).
//   * the rect -- an atlas holds many objects, each at its scale-offset, in
//     the record's V-up UVs; read in glTF's V-down space its V offset flips
//     (`1 - scaleY - offsetY`), and it is sampled clamped at the edges, since
//     a rect may run past [0, 1] where repeating reads another object's light.

// What `lightMapIntensity` carries beyond a map's own scale for three.js to
// draw the map as stored, texel x albedo: three.js divides a lightMap's
// irradiance by pi in its Lambert term. Every material bound here or by a
// relight (relight.js) takes it, so the whole page is in one unit.
export const LIGHTMAP_AS_STORED = Math.PI;

// What a BAKED material takes from the page's own lighting.
//
// A lightmap is irradiance: it already holds the surface's diffuse lighting,
// every light in the scene and the sky included. three.js adds it through
// BRDF_Lambert and then ALSO adds the environment's irradiance, so a baked
// surface was lit twice. Measured on a production room (2026-09-21): a
// quarter of the environment on top of the bake doubled every shadow and
// lifted every surface, which reads as a washed-out bake and gets reported
// as a baker regression. The same environment is the only thing in the
// render that can show a reflection, a roughness map or a normal map on a
// baked surface -- a lightmap carries no direction and no specular, and with
// the environment off entirely (as this once did) a room measured 51 of 57
// materials baked, 54 normal maps bound, no surface detail visible anywhere.
// So it is not withdrawn, it is confined to the SPECULAR term: the role a
// reflection probe plays for a lightmapped surface in a runtime. The diffuse
// comes from the bake alone. That is one line dropped from the chunk below,
// the one that adds the environment's irradiance -- a shader rule rather
// than a material setting, because `envMapIntensity` scales both terms.
//
// And at the LEVEL the export chose (`lightmappedMaterials.envMapIntensity`,
// read by readRenderingPolicy), scaled per texel by its bake: the studio is
// brighter than the room the bake lit. Unscaled, its reflections lifted the
// darkest baked surfaces from 0.06 to 0.22 of display; one flat quarter
// spared those but left a production table's lit plastic and metal with no
// visible reflection (measured 2026-10-01). So each texel's share is its
// bake's luma over the environment's own irradiance about that normal,
// clamped to 1 (`reflectionNormalization`): a shadow reflects next to
// nothing, a lit surface up to the level. A deliverable that carries its own
// reflection probe (probe.js) is reflecting the very room the bake lit, so
// the level is the studio's alone and its reflections play whole (see
// applyLighting); the normalization still holds each texel to its bake. With
// the irradiance line gone, everything the environment gives a baked material
// rides `radiance` / `clearcoatRadiance` -- `iblIrradiance` now carries the
// bake itself (see Energy below), so the multiscatter and sheen terms read
// the bake -- and scaling those two at the end of the chunk is the whole
// rule: the level one uniform shared by every baked material, so a new level
// needs no recompile.
//
// Relief. A lightmap never consults the normal, so a normal map on a baked
// surface reached the picture only through the environment's specular
// (measured: normalScale 1 -> 4 moved ~1% of pixels by ~0.4/255, reported as
// "no normal effect at all"). The fix modulates the bake by the map: the
// baked light is taken to arrive along the page's own key-light direction,
// and the lightmap texel is scaled by how the PERTURBED normal faces that
// direction relative to how the flat one does, half-Lambert wrapped.
//
// From the surface's OWN side of its plane, though: where it faces away from
// the key, the direction is mirrored across that plane first. A bake's light
// reached a surface from its front whichever side of it the key sits on, and
// taken as it stood, the key lit every surface facing away from it from
// behind -- the flat response the ratio divides by fell toward zero, so a
// FLAT map rendered a face turned straight away from the key black (and one
// 20 degrees off that at 134 where the bake puts 169), and one bump moved a
// ceiling 40 levels where it moved the floor 6 (measured 2026-09-23, reported
// as normals rendering wrong). Mirrored, the flat response never falls below
// 0.5, so the ratio stays within [0, 2] with no clamp, is exactly 1 wherever
// the map is flat -- normalScale 0 renders the pure bake -- and a surface
// wears a bump exactly as its mirror image across the key's horizon does. A
// surface already facing the key renders as it always did.
//
// Not a second bake and not physically derived: it is the single-direction
// reduction of the directional-lightmap trick, chosen because it is one
// uniform and a few lines of GLSL, and because it makes `normalTexture.scale`
// -- the value the file carries -- do in this preview what it does in a
// runtime that lights. Only installed when the material has a normal map:
// there is nothing to relieve by otherwise, and the pure bake must stay the
// pure bake.
//
// Energy. The bake enters as the ENVIRONMENT's irradiance does (`iblIrradiance`),
// not as the plain `irradiance` three.js adds a lightmap to. three.js runs
// `iblIrradiance` through its full indirect model (RE_IndirectSpecular_Physical):
// the diffuse it lights is weighted by one minus the specular's share -- the
// Fresnel the surface reflects instead, plus multiscatter -- and multiscatter's
// own energy comes from it. Added as `irradiance`, the bake lit the diffuse at
// full weight beside the reflection, so a surface was lit twice wherever it
// reflects most: at grazing angles, under a room's own reflections (probe.js).
// Measured against the bake's own renderer on a production room (2026-10-03): a
// baked wood table 1.20 -> 1.16 of its luminance, its walls and floor 1.02-1.04
// -> 1.00-1.02;
// a matte face seen square-on moves under 2%.
//
// All three edits land in three.js 0.169's `lights_fragment_maps` chunk, at
// `iblIrradiance += getIBLIrradiance( geometryNormal );` and at
// `irradiance += lightMapIrradiance;`; `normal` is the perturbed normal there
// and `nonPerturbedNormal` the geometry's own. The key light is a world
// direction and the shader works in view space, so it is turned per fragment
// with the `viewMatrix` three.js already provides.
//
// The hook receives the source with its `#include <...>` directives still
// UNEXPANDED (the renderer resolves them afterwards), so neither line can be
// matched in what the hook is handed: the directive is replaced with the
// chunk itself, edited. A replace aimed at a line directly is a silent no-op
// -- measured, the "patched" render moved the same ~1% of pixels as the
// unpatched one -- which is why the live suite asserts on the pixels.
const IBL_DIFFUSE_GLSL = 'iblIrradiance += getIBLIrradiance( geometryNormal );';
const LIGHTMAP_GLSL = 'irradiance += lightMapIrradiance;';
const LIGHTMAP_AS_IBL_GLSL = 'iblIrradiance += lightMapIrradiance;';
// Checked once, at boot, against the chunk this three.js ships: a line it no
// longer carries makes its replace below a silent no-op -- every baked surface
// washes out again, lit twice where it reflects, or the Normal Scale dial goes
// inert -- so a miss is an ERROR (the live suite fails on console errors), not
// a warning nobody in a headset reads.
for (const [edit, line] of [
  ['environment diffuse', IBL_DIFFUSE_GLSL],
  ['bake energy and normal relief', LIGHTMAP_GLSL],
]) {
  if (!THREE.ShaderChunk.lights_fragment_maps.includes(line)) {
    console.error(`three.js changed lights_fragment_maps: the baked shader's ${edit} edit no longer applies`);
  }
}
const bakeLightDir = { value: new THREE.Vector3() };
const RELIEF_GLSL = [
  '{',
  '  vec3 bakeL = normalize( ( viewMatrix * vec4( bakeLightDir, 0.0 ) ).xyz );',
  // Onto the surface's own side of its plane (see above) -- a reflection, so
  // still unit length.
  '  bakeL -= 2.0 * min( dot( nonPerturbedNormal, bakeL ), 0.0 ) * nonPerturbedNormal;',
  // `flat` is a reserved word in GLSL (an interpolation qualifier).
  '  float bumpLit = dot( normal, bakeL ) * 0.5 + 0.5;',
  '  float baseLit = dot( nonPerturbedNormal, bakeL ) * 0.5 + 0.5;',
  '  iblIrradiance += lightMapIrradiance * ( bumpLit / baseLit );',
  '}',
].join('\n');

// The baked-reflection level: ONE uniform object shared by every baked
// material's program, so applyLighting sets it once for all of them.
const bakedEnvUniform = { value: DEFAULT_BAKED_ENV_INTENSITY };
const BAKED_ENV_GLSL = [
  '#if defined( RE_IndirectSpecular )',
  '{',
  '  float bakedReflection = bakedEnvIntensity;',
  // Lightmap-normalized reflections: the environment is a studio, not the room
  // the bake lit, so its reflections are scaled per texel by how lit the BAKE
  // says that texel is against how lit the studio would make it -- a shadow
  // reflects next to nothing, a lit surface up to the full level.
  '  #if defined( USE_LIGHTMAP ) && defined( USE_ENVMAP ) && defined( ENVMAP_TYPE_CUBE_UV )',
  '    const vec3 LUMA = vec3( 0.2126, 0.7152, 0.0722 );',
  '    float studio = dot( getIBLIrradiance( geometryNormal ), LUMA );',
  '    float baked = dot( lightMapIrradiance, LUMA );',
  '    bakedReflection *= clamp( baked / max( studio, 1e-4 ), 0.0, 1.0 );',
  '  #endif',
  '  radiance *= bakedReflection;',
  '  clearcoatRadiance *= bakedReflection;',
  '}',
  '#endif',
].join('\n');

function patchBakedShader(material) {
  const relieve = Boolean(material.normalMap);
  if (relieve) bakeLightDir.value.copy(keyLight.position).normalize();
  material.onBeforeCompile = (shader) => {
    const chunk = THREE.ShaderChunk.lights_fragment_maps
      .replace(IBL_DIFFUSE_GLSL, '')
      .replace(LIGHTMAP_GLSL, relieve ? RELIEF_GLSL : LIGHTMAP_AS_IBL_GLSL);
    if (!shader.fragmentShader.includes('#include <lights_fragment_maps>')) {
      console.error('three.js changed its fragment shader: the baked shader edits no longer apply');
    }
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <lights_fragment_maps>', `${chunk}\n${BAKED_ENV_GLSL}`);
    shader.uniforms.bakedEnvIntensity = bakedEnvUniform;
    let declared = 'uniform float bakedEnvIntensity;';
    if (relieve) {
      shader.uniforms.bakeLightDir = bakeLightDir;
      declared += '\nuniform vec3 bakeLightDir;';
    }
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${declared}`);
  };
  // A distinct program from an unpatched material's with the same maps, or
  // the renderer would hand one of them the other's shader.
  material.customProgramCacheKey = () => (relieve ? 'baked-relief' : 'baked');
}

// A map's texture, decoded off the file's own buffer as the loader decodes an
// image (`parser.textureLoader`, or its KTX2 loader for `image/ktx2`), in
// glTF's texture space (no flip), sampled clamped; null when it cannot be read.
async function mapTexture(parser, media) {
  const mimeType = media.mimeType || 'image/png';
  const loader = mimeType === 'image/ktx2' ? parser.options?.ktx2Loader : parser.textureLoader;
  if (!loader) throw new Error(`no decoder for ${mimeType}`);
  const bytes = await parser.getDependency('bufferView', media.view);
  const url = URL.createObjectURL(new Blob([bytes], { type: mimeType }));
  try {
    const decoded = await loader.loadAsync(url);
    const texture = decoded?.isTexture ? decoded : new THREE.Texture(decoded);
    texture.flipY = false;
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.channel = LIGHTMAP_UV_SET;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    // Floors/walls fill the frame at grazing angles, where plain trilinear
    // collapses the atlas to its coarsest mips and rect gutters average into
    // the content as dark tile borders. Anisotropic taps keep fine-mip
    // detail along the view direction; 8 is plenty and headset-safe.
    texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
    texture.needsUpdate = true;
    return texture;
  } finally {
    URL.revokeObjectURL(url);
  }
}

// The three.js meshes of glTF node *object*: the node itself for a single
// primitive, else its primitives' meshes (the loader groups them under it,
// beside the child nodes', which carry a node association of their own).
function primitivesOf(object, associations) {
  if (object.isMesh) return [object];
  return object.children.filter((child) => child.isMesh && associations.get(child)?.nodes === undefined);
}

// What each material instance is to the bake, kept apart from its userData,
// which the loader fills from the file and three.js clones and serializes:
// `{map, rect, copy, key}` -- the record's file name of the map it binds (null
// for none), its rect, the index of the highlight copy it reads (null for the
// probe), and the binding they make together.
const bindings = new WeakMap();
// Per loaded file, `copy index -> Set<material>`: the instances that read each
// highlight copy the record lists (probe.js puts each copy on them).
const copyReadersOf = new WeakMap();
// Per loaded file with a record, its node objects by glTF node index.
const nodeObjectsOf = new WeakMap();

// *gltf*'s node objects by glTF node index (the loader's own), once its
// record is bound; empty for a file without one.
export function nodeObjects(gltf) {
  return nodeObjectsOf.get(gltf) || [];
}

// `issues` collects what the pass had to skip, for the load's report (see
// measureModel): each skip leaves an object rendering unlit, which on screen is
// indistinguishable from one that was never baked. *bake* gathers what it
// bound (bindLightmaps): the load commits it once it is the current one.
async function bindFile(gltf, issues, bake) {
  const parser = gltf.parser;
  const json = parser?.json;
  /** @type {import('./records.js').LightmapRecord} */
  const record = readExtras(gltf, LIGHTMAPS);
  const entries = Array.isArray(record?.objects) ? record.objects : [];
  if (!json || !entries.length) return 0;
  const associations = parser.associations;

  // A file the record names that several bufferViews hold is held by none
  // (records.js GLB_MEDIA): its objects render unlit, said as the build and
  // MeshConvert.verify_glb say it.
  const named = [
    ...entries.map((entry) => entry?.map),
    record.probe?.map,
    ...(Array.isArray(record.probe?.highlights) ? record.probe.highlights.map((copy) => copy?.map) : []),
  ].filter((name) => typeof name === 'string' && name);
  const held = new Map();  // folded name -> the record's spelling
  for (const name of named) {
    const tie = {};
    mediaOf(json, name, tie);
    if (tie.tie && !held.has(name.toLowerCase())) held.set(name.toLowerCase(), name);
  }
  if (held.size) {
    issues.push({
      level: 'warning',
      text: `${plural(held.size, 'file')} the record names ${held.size === 1 ? 'is' : 'are'} held by several bufferViews: none is read`,
      names: [...held.values()],
    });
  }

  const resolved = resolveEntries(json, entries);
  const tied = entries.filter((_entry, i) => resolved[i].status === 'ambiguous');
  if (tied.length) {
    issues.push({
      level: 'warning',
      text: `${plural(tied.length, 'baked object')} left unlit: nothing tells ${tied.length === 1 ? 'it' : 'them'} apart from a namesake`,
      names: tied.map((entry) => entry.name),
    });
  }

  // Node index -> what lights it: the first entry resolved to it whose map
  // rides the file. An entry whose node is elsewhere (a selection export of a
  // scene-wide record) or whose map did not ship lights nothing here.
  const lit = new Map();
  entries.forEach((entry, i) => {
    const { nodes, status } = resolved[i];
    if (!nodes || status === 'ambiguous' || status === 'absent') return;
    const media = mediaOf(json, entry.map);
    if (!media) return;
    const rect = Array.isArray(entry.scaleOffset) && entry.scaleOffset.length === 4
      ? entry.scaleOffset.map(Number)
      : [1, 1, 0, 0];
    for (const node of nodes) if (!lit.has(node)) lit.set(node, { map: entry.map, media, rect });
  });
  // Node index -> the highlight copy it reads in the probe's place (one at
  // most), for each copy the file carries.
  const copyOf = new Map();
  const copies = Array.isArray(record.probe?.highlights) ? record.probe.highlights : [];
  copies.forEach((copy, index) => {
    if (!mediaOf(json, copy?.map)) return;
    for (const { nodes } of resolveEntries(json, copy.objects || [])) {
      for (const node of nodes || []) if (!copyOf.has(node)) copyOf.set(node, index);
    }
  });
  if (!lit.size && !copyOf.size) return 0;

  // Every primitive of a node the record names, by the binding it takes: a
  // material whose wearers all take one binding takes it in place; else each
  // binding takes a copy of the material of its own, and the wearers taking
  // none keep the material as authored -- objects that light alike share one.
  const wearers = new Map();  // material -> Map<key, {binding, meshes}>
  // Each node's object through the loader's own cache, never its association
  // map: the instances of one mesh share one association there, which names
  // the last instance's node for all of them.
  const objects = await parser.getDependencies('node');
  nodeObjectsOf.set(gltf, objects);
  // Every mesh node: a material's unlit wearers decide whether it binds in place.
  for (const [node, def] of (json.nodes || []).entries()) {
    const object = def?.mesh === undefined ? null : objects[node];
    if (!object) continue;
    const light = lit.get(node) || null;
    const copy = copyOf.has(node) ? copyOf.get(node) : null;
    for (const mesh of primitivesOf(object, associations)) {
      if (Array.isArray(mesh.material) || !mesh.material) continue;
      // A primitive without the bake's UV set reads one texel of the map,
      // and measureModel names its object for it.
      const map = light;
      const key = JSON.stringify([map && map.map.toLowerCase(), map && map.rect, copy]);
      if (!wearers.has(mesh.material)) wearers.set(mesh.material, new Map());
      const byKey = wearers.get(mesh.material);
      if (!byKey.has(key)) byKey.set(key, { map, copy, key, meshes: [] });
      byKey.get(key).meshes.push(mesh);
    }
  }

  // One decode per map, however many objects and rects read it.
  const decoded = new Map();
  const unreadable = new Set();
  const textureFor = async (map) => {
    const name = map.map.toLowerCase();
    if (!decoded.has(name)) {
      decoded.set(name, mapTexture(parser, map.media).catch((error) => {
        unreadable.add(map.map);
        console.warn(`preview: lightmap ${map.map} not read: ${error.message || error}`);
        return null;
      }));
    }
    const base = await decoded.get(name);
    if (!base) return null;
    const [sx, sy, ox, oy] = map.rect;
    if (sx === 1 && sy === 1 && ox === 0 && oy === 0) return base;
    // The rect in glTF's V-down texture space (ImgUtils.flip_rect_v), on a
    // texture of its own that shares the map's one uploaded image.
    const texture = base.clone();
    texture.repeat.set(sx, sy);
    texture.offset.set(ox, 1 - sy - oy);
    texture.needsUpdate = true;
    return texture;
  };

  let bound = 0;
  const readers = new Map();
  const rects = new Map();  // `${map}|${rect}` -> its texture
  for (const [material, byKey] of wearers) {
    const takes = [...byKey.values()];
    const plain = takes.find((take) => !take.map && take.copy === null);
    for (const take of takes) {
      if (!take.map && take.copy === null) continue;  // as authored
      // In place when every wearer takes this one binding; a copy otherwise.
      let instance = material;
      if (takes.length > 1 && (plain || take !== takes[0])) {
        instance = material.clone();
        // The loader's own association rides the copy, so whatever binds by
        // the file's material -- the render effects' pointer tracks -- binds
        // it as the original.
        const own = associations.get(material);
        if (own) associations.set(instance, { ...own });
        for (const mesh of take.meshes) mesh.material = instance;
      }
      bindings.set(instance, {
        map: take.map?.map ?? null,
        rect: take.map?.rect ?? null,
        view: take.map?.media.view ?? null,
        file: gltf,
        copy: take.copy,
        key: take.key,
      });
      if (take.copy !== null) {
        if (!readers.has(take.copy)) readers.set(take.copy, new Set());
        readers.get(take.copy).add(instance);
      }
      if (!take.map) continue;
      bake.baked.add(instance);
      const rectKey = `${take.map.map.toLowerCase()}|${take.map.rect}`;
      if (!rects.has(rectKey)) rects.set(rectKey, textureFor(take.map));
      const texture = await rects.get(rectKey);
      if (!texture) continue;
      instance.lightMap = texture;
      instance.lightMapIntensity = (take.map.media.scale ?? 1) * LIGHTMAP_AS_STORED;
      patchBakedShader(instance);
      instance.needsUpdate = true;
      bake.materials.push(instance);
      bound += 1;
    }
  }
  copyReadersOf.set(gltf, readers);
  if (unreadable.size) {
    issues.push({
      level: 'warning',
      text: `${plural(unreadable.size, 'lightmap')} the file carries could not be read`,
      names: [...unreadable],
    });
  }
  return bound;
}

// The material instances that read highlight copy *index* of *gltf*'s record
// (bindLightmaps), for probe.js to put the copy on.
export function copyReaders(gltf, index) {
  return copyReadersOf.get(gltf)?.get(index) || new Set();
}

// The lightmap *material* binds, or null: `{map, rect, view, file}` -- the
// record's file name of its map (relight.js swaps by it), its rect, and the
// bufferView of the loaded file (`file`, its gltf) the map rides.
export function bindingOf(material) {
  const binding = bindings.get(material);
  return binding?.map ? { map: binding.map, rect: binding.rect, view: binding.view, file: binding.file } : null;
}

// What *material* is in the file: the loader's own material index joined to
// the binding the bake gave it, so a copy the bind made counts apart from its
// original as the copy of a file's material it stands for (specs.js, the
// render effects' count).
export function fileMaterialKey(material, associations) {
  const index = associations?.get(material)?.materials;
  if (index === undefined) return material;
  const binding = bindings.get(material);
  return binding ? `${index}|${binding.key}` : index;
}

// The whole lighting policy, in one place, so the load path and a later
// re-read cannot drift into disagreeing about it. Nothing here is per
// material: the environment plays at one level for the whole model, and what
// a BAKED material takes from it is decided in its shader (patchBakedShader)
// -- the specular term only, since its diffuse is in its bake, at the level
// the export published. An earlier form dimmed the environment per baked
// material, diffuse and specular together: any diffuse at all on top of the
// bake washed the room out, which is why the level now reaches the specular
// alone. `lightmappedMaterials` is rebuilt per load and holds only
// materials this model owns, so it dies with the model; the lookdev dials
// read it.
export let lightmapped = 0;
export let lightmappedMaterials = [];
// Every material instance a baked object of the file's record wears where
// its map rides the file and a UV set reads it, bound or not: what its bake
// lit, as the file states it, which measureModel holds against what bound
// (specs.js). A set of its own, never a userData flag: the loader copies a
// material's glTF extras into its userData, any key included.
export let bakedMaterials = new WeakSet();

// A load starting over: the materials it binds are its own (see above).
export function resetLightmaps() {
  lightmappedMaterials = [];
  bakedMaterials = new WeakSet();
}

// Bind a deliverable's lightmaps onto its parsed scene (see bindFile):
// `{count, materials, baked}` -- how many materials took one, those
// materials, and every material its bake lit. *gltfs* is one file, or the
// parts of one scene (parts.js): each binds the bake its own record names,
// and the count is the set's. Async -- each map is decoded off its file's own
// buffer -- and touching only the scene it is handed, so a load binds before
// its model goes up and a superseded one leaves the page as it was; the load
// keeps it (applyLightmaps) once it is the current one.
export async function bindLightmaps(gltfs, issues = []) {
  const bake = { count: 0, materials: [], baked: new WeakSet() };
  for (const gltf of [].concat(gltfs)) bake.count += await bindFile(gltf, issues, bake);
  return bake;
}

// *bake* (bindLightmaps) as the model on screen's -- the count applyLighting
// turns on, the materials the lookdev dials and relight read; returns the count.
export function applyLightmaps(bake) {
  lightmapped = bake.count;
  lightmappedMaterials = bake.materials;
  bakedMaterials = bake.baked;
  return lightmapped;
}

export function applyLighting() {
  // The deliverable's own reflection probe when it carries one and it is on
  // (probe.js) -- the room its bake lit -- at its published level (1: as
  // captured, the bake's units); the studio otherwise, at the policy's. The studio is freed
  // while the probe stands in for it and built again when it is next needed.
  const probe = probeEnvironment();
  scene.environment = probe || studioEnvironment();
  if (probe) releaseStudio();
  scene.environmentIntensity = probe ? policy.probeIntensity : policy.environmentIntensity;
  // The probe's highlight copies play at the probe's level (probe.js).
  syncProbeCopies();
  // What a baked material reflects of it: the export's level (see
  // patchBakedShader), on top of the one environment level above. That level
  // exists because the studio is brighter than the room the bake lit; a probe
  // IS that room, so its reflections play whole, each texel still scaled by
  // its bake (`reflectionNormalization`).
  bakedEnvUniform.value = probe ? 1 : policy.bakedEnvIntensity;
  // The key light cannot go per material without render layers, so it stays
  // scene-wide and goes off the moment ANYTHING is baked. Gating it on the
  // model being FULLY baked (as this briefly did) is what blows out a baked
  // room: a scene-wide term cannot be spared the geometry it contradicts, so
  // a 0.9 directional light lands on surfaces whose lighting is already in
  // their lightmap, and a room is routinely partly baked (measured: 51 of 57
  // materials) so the gate never opens. It reads as a baker regression --
  // re-baking with the scene's lights turned down changes nothing, because the
  // extra light is added here, downstream of the EXR.
  //
  // The un-baked props that gate was added for are not left dark: they keep
  // the FULL environment (they were on 0.25 when the key light mattered, which
  // is what made them look unlit), and the environment is normal- and
  // view-dependent -- getIBLIrradiance and getIBLRadiance take the normal --
  // so their normal maps and specular stay legible without it. What they give
  // up is a directional diffuse gradient, which is precisely the term that
  // must not reach the baked geometry standing beside them.
  //
  // Render layers would let the key light skip only the baked meshes, but
  // layers 1 and 2 are already spoken for: three.js 0.169.0 -- the version
  // pinned in the importmap above -- does `cameraL.layers.enable(1)` and
  // `cameraR.layers.enable(2)` in WebXRManager, so geometry parked on layer 1
  // renders to the LEFT EYE ONLY (read from that source, not assumed). A higher
  // layer dodges the collision, but the mask still has to survive into the XR
  // sub-cameras and the failure shows up in a headset rather than at the desk.
  // Layers are also per OBJECT while a bake is per MATERIAL, so a mesh with
  // mixed materials would need a conservative rule regardless.
  keyLight.intensity = lightmapped ? 0 : policy.keyIntensity;
}
