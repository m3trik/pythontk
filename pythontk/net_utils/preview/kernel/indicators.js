// An indicator light seen from where a person stands.
//
// An indicator is an emissive material with named states (the `render_effects`
// record's `states`: off, green, amber...). A real one is a few millimetres
// across -- at a metre and a half it covers a pixel or two of a headset's
// view, too few for its colour, let alone its blinking, to read, and the
// look keeps bloom off (it would halo every bright surface, not the lights).
// So each indicator mesh wears a glow card: a soft camera-facing disc in the
// emission's colour, never smaller on screen than MIN_PIXELS, drawn only
// while the emission is on. Close up it sits inside the light's own glow; far
// off it is what says the light is lit. It is the record's rule for every
// runtime that reads indicators; nothing in the file says to draw it, as an
// indicator always needs to be seen.

import * as THREE from 'three';
import { materialsOf } from './model.js';
import { VISIBILITY } from './records.js';
import { onBeforeDraw, readExtras } from './scene.js';

/** The least a lit indicator's glow covers on screen, in pixels across. */
export const MIN_PIXELS = 10;
/** The glow's width over its light's, close up: a halo just past the lens. */
const HALO = 2.5;
/** Below this emission (its largest channel, linear) the light reads as off. */
const LIT = 0.01;

/** @type {{mesh: any, material: any, card: any, radius: number, centre: any}[]} */
let cards = [];
let texture = null;

// A soft disc, bright at its centre, gone at its rim (made once, shared).
function glowTexture() {
  if (texture) return texture;
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, 'rgba(255,255,255,1)');
  gradient.addColorStop(0.25, 'rgba(255,255,255,0.6)');
  gradient.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  texture = new THREE.CanvasTexture(canvas);
  return texture;
}

/**
 * The materials *gltf*'s `render_effects` record names as indicators: those
 * with states. A name joins by the leaf after its last ':' too, as the
 * record says an import may drop the namespace.
 * @param {object} gltf
 * @returns {Set<string>}
 */
export function indicatorMaterials(gltf) {
  const names = new Set();
  for (const entry of readExtras(gltf, VISIBILITY)?.materials || []) {
    if (!Array.isArray(entry.states) || !entry.states.length || !entry.material) continue;
    names.add(entry.material);
    names.add(entry.material.split(':').pop());
  }
  return names;
}

/** Drop every card (a new model loads). */
export function clearIndicators() {
  for (const { card } of cards) {
    card.removeFromParent();
    card.material.dispose();
  }
  cards = [];
}

/**
 * Give each mesh of *parts* wearing an indicator material a glow card.
 * @param {{gltf: object}[]} parts The loaded parts, each its own record.
 * @param {any} holder Where the cards hang (the scene).
 * @returns {number} How many cards.
 */
export function setupIndicators(parts, holder) {
  clearIndicators();
  for (const { gltf } of parts) {
    const names = indicatorMaterials(gltf);
    if (!names.size) continue;
    gltf.scene.traverse((mesh) => {
      if (!mesh.isMesh) return;
      const material = materialsOf(mesh).find((m) => names.has(m.name) || names.has(m.name.split(':').pop()));
      if (!material) return;
      if (!mesh.geometry.boundingSphere) mesh.geometry.computeBoundingSphere();
      const card = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTexture(), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true,
        toneMapped: false, fog: false,
      }));
      card.name = `${mesh.name} glow`;
      card.visible = false;
      card.renderOrder = 1;  // after the opaque scene: added over it
      holder.add(card);
      cards.push({ mesh, material, card, radius: mesh.geometry.boundingSphere.radius, centre: mesh.geometry.boundingSphere.center });
    });
  }
  return cards.length;
}

const world = new THREE.Vector3();
const scale = new THREE.Vector3();
const buffer = new THREE.Vector2();

// Each draw: every card at its light, in its emission's colour, at least
// MIN_PIXELS across for this camera (a headset's eye, or the page's).
function place(renderer, drawn, camera) {
  if (!cards.length) return;
  const eye = camera.isArrayCamera && camera.cameras.length ? camera.cameras[0] : camera;
  // A headset's eye is its viewport, in pixels; the page's, what it draws into.
  const height = eye.viewport?.w || renderer.getRenderTarget()?.height || renderer.getDrawingBufferSize(buffer).y || 1;
  // The view's height over its distance, off the projection itself: an XR
  // eye's frustum is the headset's (and asymmetric), its `fov` never set.
  const perspective = eye.isPerspectiveCamera ? 2 / eye.projectionMatrix.elements[5] : 0;
  for (const { mesh, material, card, radius, centre } of cards) {
    const colour = material.emissive;
    const lit = colour ? Math.max(colour.r, colour.g, colour.b) * (material.emissiveIntensity ?? 1) : 0;
    // Shown while lit and in the scene drawn, every node up to it visible.
    let node = mesh;
    let shown = lit > LIT;
    for (; shown && node && node !== drawn; node = node.parent) shown = node.visible;
    shown = shown && node === drawn;
    card.visible = shown;
    if (!shown) continue;
    mesh.getWorldScale(scale);
    const reach = radius * Math.max(scale.x, scale.y, scale.z);
    const size = 2 * reach * HALO;
    mesh.localToWorld(world.copy(centre));
    // Just before the light, by its own radius, so the surface it sits on clips none of the card.
    const toEye = eye.getWorldPosition(scale).sub(world);
    const away = toEye.length();
    if (away > reach) world.addScaledVector(toEye, reach / away);
    card.position.copy(world);
    const distance = Math.max(away - reach, 0);
    // The world size MIN_PIXELS spans at that distance (an orthographic view: none).
    const least = perspective ? (MIN_PIXELS / height) * perspective * distance : 0;
    card.scale.setScalar(Math.max(size, least));
    card.material.color.setRGB(colour.r, colour.g, colour.b).multiplyScalar(material.emissiveIntensity ?? 1);
    card.material.opacity = Math.min(1, lit);
    card.updateMatrixWorld();  // the scene's matrices were updated before this hook
  }
}

onBeforeDraw(place);
