// Baked lightmaps: binding the deliverable's maps to its materials, the shader
// edits a baked material is drawn with, and the lighting policy that turns on
// how much of the model is baked.

import * as THREE from 'three';
import { plural } from './format.js';
import { materialsOf } from './model.js';
import { probeEnvironment } from './probe.js';
import { LIGHTMAPS } from './records.js';
import {
  DEFAULT_BAKED_ENV_INTENSITY, keyLight, policy, readExtras, releaseStudio, renderer, scene,
  studioEnvironment,
} from './scene.js';

// glTF 2.0 has no lightmap slot, so a baked lightmap reaches us disguised as a real
// texture slot on TEXCOORD_1 (see blendertk's LightmapWebExport). The scene extras carry
// a `lightmap_web` manifest naming which materials are wearing the disguise; this puts
// them back. Three details decide whether it looks right or subtly wrong:
//
//   * colorSpace — GLTFLoader treats occlusion as DATA and loads it linear, but the map
//     was written sRGB-encoded to survive 8-bit. Left linear it renders far too dark.
//   * channel — the map is packed against UV2, not UV0.
//   * intensity — the encoder divided by a percentile to fit a huge HDR range into a PNG;
//     multiplying it back is what restores the original exposure.
//
// Also clears the slot it arrived in, so the same texture is not applied twice under two
// different interpretations. Returns how many materials it bound, which is what decides
// the scene's lighting policy (see applyLighting).
//
// Which real slot each carrier arrives in — mirrors blendertk's LightmapWebExport.CARRIERS.
// A Map rather than an object literal because the lookup decides whether to REBIND a
// material: property access on an object literal walks the prototype chain, so a manifest
// naming 'constructor' as its carrier would hand back a truthy function and slip past the
// refusal below. Map.get answers only for keys actually in the table.
const CARRIER_SLOTS = new Map([['occlusion', 'aoMap'], ['emissive', 'emissiveMap']]);

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
// Measured against Arnold on a production room (2026-10-03): a baked wood table
// 1.20 -> 1.16 of Arnold's luminance, its walls and floor 1.02-1.04 -> 1.00-1.02;
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

// `issues` collects what the pass had to skip, for the load's report (see
// measureModel): each skip leaves a material rendering unlit, which on screen is
// indistinguishable from a material that was never baked.
function bindLightmaps(gltf, issues) {
  const manifest = readExtras(gltf, LIGHTMAPS.webKey);
  if (!manifest || !manifest.materials) return 0;

  // An unknown carrier is REFUSED rather than defaulted. The previous form special-cased
  // 'fused' (a bake level both DCC tools have since removed, so nothing can emit it) and
  // sent everything else to aoMap, so the only branch it had was dead while any future or
  // misspelled carrier would be rebound under the interpretation it is least likely to
  // want. Not binding leaves the map in the slot it arrived in, where it still reads as
  // plausible occlusion -- and is SAID, since that is all anyone would see.
  const carrier = manifest.carrier || 'occlusion';
  const slot = CARRIER_SLOTS.get(carrier);
  if (!slot) {
    issues.push({
      level: 'error',
      text: `no lightmap bound: the file carries them as '${carrier}', which this page does not read`,
    });
    return 0;
  }

  const uv = manifest.uv ?? 1;
  let bound = 0;
  const answered = new Set();  // manifest names some loaded material wears
  const emptied = new Set();   // ... whose map was not in the carrier slot

  gltf.scene.traverse((node) => {
    for (const material of materialsOf(node)) {
      // Own keys only: the manifest is parsed JSON, and a material named
      // 'constructor' would otherwise find Object's and bind whatever sits in
      // its occlusion slot as light -- the reason CARRIER_SLOTS is a Map.
      if (!Object.hasOwn(manifest.materials, material.name)) continue;
      const entry = manifest.materials[material.name];
      answered.add(material.name);
      if (!entry || material.userData.lightmapBound) continue;
      const texture = material[slot];
      if (!texture) {
        emptied.add(material.name);
        continue;
      }

      texture.colorSpace = manifest.encoding === 'srgb' ? THREE.SRGBColorSpace : THREE.NoColorSpace;
      texture.channel = uv;
      // Floors/walls fill the frame at grazing angles, where plain trilinear
      // collapses the atlas to its coarsest mips and rect gutters average into
      // the content as dark tile borders. Anisotropic taps keep fine-mip
      // detail along the view direction; 8 is plenty and headset-safe.
      texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
      texture.needsUpdate = true;

      material.lightMap = texture;
      material.lightMapIntensity = entry.intensity ?? 1;
      material[slot] = null;
      if (carrier === 'emissive') material.emissive = new THREE.Color(0x000000);
      patchBakedShader(material);
      material.userData.lightmapBound = true;
      material.needsUpdate = true;
      lightmappedMaterials.push(material);
      bound += 1;
    }
  });
  const unworn = Object.keys(manifest.materials).filter((name) => !answered.has(name));
  if (unworn.length) {
    issues.push({
      level: 'warning',
      text: `${plural(unworn.length, 'lightmapped material')} the file names ${unworn.length === 1 ? 'is' : 'are'} not in the model`,
      names: unworn,
    });
  }
  if (emptied.size) {
    issues.push({
      level: 'warning',
      text: `${plural(emptied.size, 'lightmapped material')} arrived without ${emptied.size === 1 ? 'its' : 'their'} map`,
      names: [...emptied],
    });
  }
  return bound;
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

// A load starting over: the materials it binds are its own (see above).
export function resetLightmaps() {
  lightmappedMaterials = [];
}

// Bind the deliverable's lightmaps (see bindLightmaps) and keep how many
// materials took one -- the count applyLighting turns on.
export function applyLightmaps(gltf, issues = []) {
  lightmapped = bindLightmaps(gltf, issues);
  return lightmapped;
}

export function applyLighting() {
  // The deliverable's own reflection probe when it carries one and it is on
  // (probe.js) -- the room its bake lit -- at the level that keeps it in the
  // bake's units; the studio otherwise, at the policy's. The studio is freed
  // while the probe stands in for it and built again when it is next needed.
  const probe = probeEnvironment();
  scene.environment = probe || studioEnvironment();
  if (probe) releaseStudio();
  scene.environmentIntensity = probe ? policy.probeIntensity : policy.environmentIntensity;
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
  // re-baking with the Maya lights turned down changes nothing, because the
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
