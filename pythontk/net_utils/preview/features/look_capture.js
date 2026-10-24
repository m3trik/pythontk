/*
  Look capture -- the frames a look match measures, from the real viewer.

  Driven headless by pythontk's `LookCapture` (a browser automation session):
  the page calls the default export once with the viewer API, and the driver
  imports this module by its served path and calls its named exports. Nothing
  here adds a menu or a button; a person never sees it.

  - `status()` -- whether the model is up and its lighting has settled.
  - `setLighting(payload)` -- swap the bake (`viewer.lighting.relight`) from
    base64 half floats, and optionally preview a material's emissive level.
  - `restoreLighting()` -- the file's own bake back (what ships).
  - `setRendering(on)` -- the frame loop off between requests (an idle page
    still draws every frame) and on again for the next.
  - `shot(options)` -- one frame through a given camera: the screen's own image
    (`lit`, a PNG), or linear light (`linear`, half floats) -- with chosen
    objects reflecting nothing, so a frame stays the sum of its lights.

  Activate:  server.add_script("look_capture")
*/

/** @type {any} */
let viewer = null;

export default function lookCapture(api) {
  viewer = api;
}

/** @param {string} b64 @returns {Uint16Array} */
function halfFloats(b64) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return new Uint16Array(bytes.buffer);
}

/** @param {Uint16Array} half @returns {string} */
function base64(half) {
  const bytes = new Uint8Array(half.buffer);
  let text = '';
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(text);
}

/** The file's own bake and lights again, after a swap: the lighting as it ships. */
export async function restoreLighting() {
  unpreviewEmissive();
  // Awaited, so the driver's call returns once the file's probe is back up: a
  // shot taken sooner would still show the swap's.
  await viewer.lighting.restore();
}

/** @type {Map<any, number>} material -> its own emissive level, while a preview holds it. */
const emissiveOwn = new Map();

/** Every previewed material back at its own emissive level. */
function unpreviewEmissive() {
  for (const [material, level] of emissiveOwn) material.emissiveIntensity = level;
  emissiveOwn.clear();
}

/** Sleep (false) or wake (true) the page's frame loop -- between a driver's requests. @param {boolean} on */
export function setRendering(on) {
  viewer.setRendering(on);
}

/** Whether the model is up and its lighting has settled (key light off over a bake, probe level applied). */
export function status() {
  if (!viewer || !viewer.model) return { ready: false };
  /** @type {any} */
  let key = null;
  viewer.scene.traverse((/** @type {any} */ node) => { if (node.isDirectionalLight) key = node; });
  const lightmaps = viewer.specs && viewer.specs.lightmaps;
  const baked = Boolean(lightmaps && lightmaps.baked);
  const settled = Boolean(viewer.scene.environment) && (!baked || (key && key.intensity === 0));
  return { ready: settled, baked };
}

/**
 * @param {{
 *   maps: Object<string, string>,
 *   images?: Object<string, {width: number, height: number, b64: string}>,
 *   probe?: {width: number, height: number, b64: string, position: number[], box?: any} | null,
 *   emissive?: {pattern: string, intensity: number} | null,
 *   highlights?: {width: number, height: number, b64: string, materials: string[]}[] | null,
 * }} payload `highlights`: the bake's lights as the deliverable draws them into
 *   the probe's copies (`viewer.lighting.relight`); absent, none.
 * @returns {number} How many materials took a map.
 */
export function setLighting(payload) {
  const images = {};
  for (const [name, image] of Object.entries(payload.images || {})) {
    images[name] = { width: image.width, height: image.height, data: halfFloats(image.b64) };
  }
  const probe = payload.probe
    ? { ...payload.probe, data: halfFloats(payload.probe.b64) }
    : null;
  const highlights = (payload.highlights || []).map((layer) => ({ ...layer, data: halfFloats(layer.b64) }));
  const bound = viewer.lighting.relight({ maps: payload.maps, images, probe, highlights });
  // A preview of an exported emissive level (the material's own factor) is
  // this relight's alone: the next one starts from every material's own.
  unpreviewEmissive();
  if (payload.emissive) {
    const pattern = new RegExp(payload.emissive.pattern);
    viewer.model.traverse((/** @type {any} */ node) => {
      for (const material of [].concat(node.material || [])) {
        if (material && material.emissiveMap && pattern.test(material.name || '')) {
          if (!emissiveOwn.has(material)) emissiveOwn.set(material, material.emissiveIntensity);
          material.emissiveIntensity = payload.emissive.intensity;
        }
      }
    });
  }
  return bound;
}

/**
 * The standard materials *patterns* name: by a material's own name or by the
 * name of its object or any ancestor; a pattern starting with `!` keeps what
 * it matches out. Each material once, however many meshes share it.
 *
 * @param {string[]} patterns
 * @returns {Set<any>}
 */
function materialsMatching(patterns) {
  const take = patterns.filter((p) => !p.startsWith('!')).map((p) => new RegExp(p));
  const keep = patterns.filter((p) => p.startsWith('!')).map((p) => new RegExp(p.slice(1)));
  const found = new Set();
  if (!take.length) return found;
  const model = viewer.model;
  model.traverse((/** @type {any} */ node) => {
    const names = [];
    for (let n = node; n && n !== model; n = n.parent) names.push(n.name || '');
    for (const material of [].concat(node.material || [])) {
      if (!material || !material.isMeshStandardMaterial || found.has(material)) continue;
      const all = names.concat(material.name || '');
      if (!take.some((p) => all.some((name) => p.test(name)))) continue;
      if (keep.some((p) => all.some((name) => p.test(name)))) continue;
      found.add(material);
    }
  });
  return found;
}

/**
 * The camera a frame is shot from: *camera* (model space) or the page's own.
 *
 * @param {{pos: number[], target: number[], up: number[], vfov: number} | null} camera
 * @returns {any}
 */
function viewFor(camera) {
  const THREE = viewer.THREE;
  const renderer = viewer.renderer;
  const model = viewer.model;
  model.updateMatrixWorld(true);
  if (!camera) return viewer.camera;
  const view = new THREE.PerspectiveCamera(camera.vfov, renderer.domElement.width / renderer.domElement.height, 0.01, 200);
  view.position.copy(model.localToWorld(new THREE.Vector3(...camera.pos)));
  view.up.copy(new THREE.Vector3(...camera.up).transformDirection(model.matrixWorld));
  view.lookAt(model.localToWorld(new THREE.Vector3(...camera.target)));
  return view;
}

/**
 * Which object every pixel shows: each mesh drawn in a flat colour of its own
 * (its index, as 24-bit RGB; 0 is no object), unlit, unaliased (a float
 * target, read back exactly), with a legend naming each colour's object path
 * and material. What a region names objects by (``LookTarget.objects``): exact,
 * no background, and right for any camera.
 *
 * @param {{camera?: {pos: number[], target: number[], up: number[], vfov: number} | null, scale?: number}} options
 * @returns {{png: string, legend: {color: number[], names: string[], material: string}[]}}
 */
export function ids({ camera = null, scale = 1 } = {}) {
  const THREE = viewer.THREE;
  const renderer = viewer.renderer;
  const model = viewer.model;
  const view = viewFor(camera);
  /** @type {any[][]} */
  const swapped = [];
  /** @type {{color: number[], names: string[], material: string}[]} */
  const legend = [];
  model.traverse((/** @type {any} */ node) => {
    if (!node.isMesh || !node.visible) return;
    const code = legend.length + 1;
    const rgb = [(code >> 16) & 255, (code >> 8) & 255, code & 255];
    const names = [];
    for (let n = node; n && n !== model; n = n.parent) names.unshift(n.name || '');
    const own = [].concat(node.material || [])[0];
    legend.push({ color: rgb, names, material: (own && own.name) || '' });
    swapped.push([node, node.material]);
    node.material = new THREE.MeshBasicMaterial({
      color: new THREE.Color().setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, THREE.LinearSRGBColorSpace),
      side: own ? own.side : THREE.FrontSide,
    });
  });
  try {
    const width = Math.round(renderer.domElement.width * scale);
    const height = Math.round(renderer.domElement.height * scale);
    const pixels = viewer.lighting.renderLinear({ camera: view, width, height });
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = /** @type {CanvasRenderingContext2D} */ (canvas.getContext('2d'));
    const image = context.createImageData(width, height);
    for (let y = 0; y < height; y++) {  // flipped to top-down
      const src = (height - 1 - y) * width * 4;
      const dst = y * width * 4;
      for (let x = 0; x < width * 4; x += 4) {
        image.data[dst + x] = Math.round(pixels[src + x] * 255);
        image.data[dst + x + 1] = Math.round(pixels[src + x + 1] * 255);
        image.data[dst + x + 2] = Math.round(pixels[src + x + 2] * 255);
        image.data[dst + x + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
    return { png: canvas.toDataURL('image/png'), legend };
  } finally {
    for (const [node, material] of swapped) {
      node.material.dispose();
      node.material = material;
    }
  }
}

/**
 * One frame.
 *
 * @param {{
 *   mode?: 'lit' | 'linear',
 *   camera?: {pos: number[], target: number[], up: number[], vfov: number} | null,
 *   black?: string[],
 *   uncap?: string[],
 *   scale?: number,
 * }} options `camera` is in model space (default: the page's own view);
 *   `black` are patterns (materialsMatching) whose materials reflect nothing
 *   for the frame -- a material shared with an unmatched object is blacked
 *   there too; `uncap` are patterns whose baked materials reflect their
 *   environment whole for the frame, without the lightmap's per-texel
 *   normalization (a diagnostic: what the environment alone gives them).
 *   `scale` sizes a linear frame against the drawing buffer.
 * @returns {string} `png:<data url>` or `f16:<width>x<height>:<base64>` (rows top-down).
 */
export function shot({ mode = 'lit', camera = null, black = [], uncap = [], scale = 1 } = {}) {
  const THREE = viewer.THREE;
  const renderer = viewer.renderer;
  const view = viewFor(camera);
  // A material without an envMap of its own reflects `scene.environment` at
  // `scene.environmentIntensity` (three r163+): its `envMapIntensity` is never
  // read. A blacked material gets the scene's own environment as its envMap
  // (the same texture, so the same program) at level 0, for this frame only.
  /** @type {Map<any, [any, number]>} */
  const held = new Map();
  for (const material of materialsMatching(black)) {
    held.set(material, [material.envMap, material.envMapIntensity]);
    material.envMap = material.envMap || viewer.scene.environment;
    material.envMapIntensity = 0;
  }
  /** @type {Map<any, [any, any]>} */
  const uncapped = new Map();
  for (const material of materialsMatching(uncap)) {
    if (!material.userData.lightmapBound) continue;
    const compile = material.onBeforeCompile;
    const key = material.customProgramCacheKey;
    uncapped.set(material, [compile, key]);
    material.onBeforeCompile = (/** @type {any} */ shader, /** @type {any} */ r) => {
      compile.call(material, shader, r);
      shader.fragmentShader = shader.fragmentShader.replace(/bakedReflection \*= clamp\([^;]*;/, '');
    };
    material.customProgramCacheKey = () => `${key.call(material)}-uncapped`;
    material.needsUpdate = true;
  }
  try {
    if (mode === 'linear') {
      const width = Math.round(renderer.domElement.width * scale);
      const height = Math.round(renderer.domElement.height * scale);
      const pixels = viewer.lighting.renderLinear({ camera: view, width, height });
      const half = new Uint16Array(width * height * 3);
      for (let y = 0; y < height; y++) {  // RGB only, flipped to top-down
        const src = (height - 1 - y) * width * 4;
        const dst = y * width * 3;
        for (let x = 0; x < width; x++) {
          half[dst + x * 3] = THREE.DataUtils.toHalfFloat(pixels[src + x * 4]);
          half[dst + x * 3 + 1] = THREE.DataUtils.toHalfFloat(pixels[src + x * 4 + 1]);
          half[dst + x * 3 + 2] = THREE.DataUtils.toHalfFloat(pixels[src + x * 4 + 2]);
        }
      }
      return `f16:${width}x${height}:${base64(half)}`;
    }
    const background = viewer.scene.background;
    viewer.scene.background = null;
    renderer.render(viewer.scene, view);
    const url = renderer.domElement.toDataURL('image/png');
    viewer.scene.background = background;
    return `png:${url}`;
  } finally {
    for (const [material, [envMap, level]] of held) {
      material.envMap = envMap;
      material.envMapIntensity = level;
    }
    for (const [material, [compile, key]] of uncapped) {
      material.onBeforeCompile = compile;
      material.customProgramCacheKey = key;
      material.needsUpdate = true;
    }
  }
}
