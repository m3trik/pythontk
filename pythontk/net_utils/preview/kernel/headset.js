import * as THREE from 'three';
import { XRControllerModelFactory } from 'three/addons/webxr/XRControllerModelFactory.js';
import { renderer, scene } from './scene.js';

// The session's own reference space. The rig always offsets THIS one, never an
// offset of an offset, so nothing accumulates across a session's moves.
export let headsetSpace = null;

const OVERLAY_DISTANCE = 0.1;  // m in front of each eye
const OVERLAY_ORDER = 1e6;     // drawn last: over everything, cards included

// Everything below draws only in a headset; on the desktop it stays hidden, so
// a still or a playblast never carries it.
export const headsetLayer = new THREE.Group();
headsetLayer.visible = false;
scene.add(headsetLayer);

// A card of text in the headset, where the page's own chrome is not drawn at
// all: a canvas in that chrome's colours on a plane in the headset layer --
// the controls card below, and any script's (the packaged `inspect` rides
// one). See `viewer.addHeadsetCard` for the contract.
const CARD_FILL = 'rgba(21, 23, 26, 0.88)';
const CARD_UP = new THREE.Vector3(0, 1, 0);

export function addHeadsetCard({ name = '', width = 720, height = 400, metres = 0.36, radius = 28, over = false } = {}) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(metres, (metres * height) / width),  // the canvas's aspect
    new THREE.MeshBasicMaterial({
      map: texture, transparent: true, depthTest: !over, depthWrite: false, toneMapped: false,
    }),
  );
  mesh.name = name;
  if (over) {
    // Over the model, under the blink, which then fades it too.
    mesh.renderOrder = OVERLAY_ORDER - 1;
    mesh.frustumCulled = false;
  }
  mesh.visible = false;
  headsetLayer.add(mesh);
  const facing = new THREE.Vector3();
  const target = new THREE.Vector3();
  return {
    mesh,
    draw(paint) {
      context.clearRect(0, 0, width, height);
      context.fillStyle = CARD_FILL;
      if (context.roundRect) {
        context.beginPath();
        context.roundRect(0, 0, width, height, radius);
        context.fill();
      } else {
        context.fillRect(0, 0, width, height);
      }
      paint(context, width, height);
      texture.needsUpdate = true;
    },
    place(head, look, { ahead, aside = 0, below = 0, ease = 0 }, delta = 0) {
      facing.copy(look).setY(0);
      if (facing.lengthSq() < 1e-6) facing.set(0, 0, -1);
      facing.normalize().applyAxisAngle(CARD_UP, aside);
      target.copy(head).addScaledVector(facing, ahead);
      target.y = head.y - below;
      if (ease && mesh.visible) mesh.position.lerp(target, 1 - Math.exp(-Math.max(delta, 0) / ease));
      else mesh.position.copy(target);
      mesh.lookAt(head);
    },
  };
}

// The controllers themselves, modelled after the real ones. The models load on
// connect from the WebXR input-profiles CDN; with it unreachable they are just
// not drawn, and everything else here still works.
const controllerModels = new XRControllerModelFactory();
for (const index of [0, 1]) {
  const grip = renderer.xr.getControllerGrip(index);
  grip.add(controllerModels.createControllerModel(grip));
  scene.add(grip);
}

// The rig's blink and locomotion's vignette: one quad drawn in front of EACH
// eye, placed in `onBeforeRender` from the eye camera being drawn -- a child of
// the camera would trail the head by a frame, which in a headset reads as the
// view smearing. Radial for the vignette (dark from 25 degrees off the eye's
// axis to black at 50), flat for the blink. Each writes its own uniform.
export const overlay = new THREE.Mesh(
  new THREE.PlaneGeometry(0.6, 0.6),  // +-0.3 m at 0.1 m: 71 degrees each way
  new THREE.ShaderMaterial({
    uniforms: { fade: { value: 0 }, vignette: { value: 0 } },
    vertexShader: `
      varying vec2 vPosition;
      void main() {
        vPosition = position.xy;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: `
      uniform float fade;
      uniform float vignette;
      varying vec2 vPosition;
      void main() {
        float offAxis = length(vPosition) / ${OVERLAY_DISTANCE.toFixed(3)};  // tan of the angle
        float alpha = max(fade, vignette * smoothstep(0.466, 1.192, offAxis));
        if (alpha < 0.002) discard;
        gl_FragColor = vec4(0.0, 0.0, 0.0, alpha);
      }`,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  }),
);
overlay.frustumCulled = false;
overlay.renderOrder = OVERLAY_ORDER;
overlay.visible = false;
const overlayOffset = new THREE.Matrix4().makeTranslation(0, 0, -OVERLAY_DISTANCE);
overlay.onBeforeRender = (_renderer, _scene, eye) => {
  overlay.matrixWorld.multiplyMatrices(eye.matrixWorld, overlayOffset);
};
headsetLayer.add(overlay);

const poseArrays = ({ position: p, orientation: q }) => ({
  position: [p.x, p.y, p.z],
  orientation: [q.x, q.y, q.z, q.w],
});

// The head, the sticks, both controllers' grips and aim rays and whether each
// is squeezed, all in the session's own space: the rig maps them into the
// scene, so they must not be read through the offset space the rig itself
// builds. Returns what `headset.step` takes -- `{head, left, right, aim,
// grips, aims, squeeze}`: each pose `{position, orientation}` as arrays, each
// stick `[x, y]`, `grips` / `aims` `{left, right}` poses, `squeeze` `{left,
// right}` booleans (the grip OR the trigger -- either takes hold -- or, for a
// tracked hand, which has no gamepad, a pinch), null where absent -- and
// `sources`, the controller behind each hand. `aim` is the right hand's ray,
// which locomotion aims with. A hand that reports no grip space holds from its
// aim pose.
export const pinching = { left: false, right: false };  // set by the session's select events

export function readHeadsetInput(frame) {
  const input = {
    head: null, left: null, right: null, aim: null,
    grips: {}, aims: {}, squeeze: {}, sources: {},
  };
  const viewerPose = frame.getViewerPose(headsetSpace);
  if (viewerPose) input.head = poseArrays(viewerPose.transform);
  for (const source of frame.session.inputSources) {
    const hand = source.handedness;
    if (hand !== 'left' && hand !== 'right') continue;
    input.sources[hand] = source;
    const axes = source.gamepad?.axes;
    // xr-standard puts the thumbstick on axes 2-3; a touchpad-only controller
    // reports on 0-1.
    if (axes && axes.length >= 2) input[hand] = axes.length >= 4 ? [axes[2], axes[3]] : [axes[0], axes[1]];
    const aim = frame.getPose(source.targetRaySpace, headsetSpace);
    if (aim) input.aims[hand] = poseArrays(aim.transform);
    if (aim && hand === 'right') input.aim = input.aims[hand];
    const grip = source.gripSpace ? frame.getPose(source.gripSpace, headsetSpace) : null;
    if (grip) input.grips[hand] = poseArrays(grip.transform);
    else if (input.aims[hand]) input.grips[hand] = input.aims[hand];
    // xr-standard: button 0 the trigger, 1 the grip.
    const buttons = source.gamepad?.buttons || [];
    input.squeeze[hand] = !!(buttons[0]?.pressed || buttons[1]?.pressed || pinching[hand]);
  }
  return input;
}

// The session's own space as it starts (null as it ends): what the rig
// offsets, and what every pose is read in.
export function setHeadsetSpace(space) {
  headsetSpace = space;
}
