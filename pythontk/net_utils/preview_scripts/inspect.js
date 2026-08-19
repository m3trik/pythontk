/*
  Inspect — what this deliverable actually costs, reported from the page that
  is rendering it rather than inferred from the file.

  The numbers that decide whether a scene previews well in a headset are not
  the ones a GLB's size suggests: draw calls follow mesh NODES (a room measured
  1219 of them against 57 materials), and GPU memory follows decoded texture
  dimensions, not the compressed bytes on the wire (5.97 MB of images decoding
  to ~555 MB of RGBA, ~740 MB with mipmaps). Both are read straight off the
  renderer here, so the answer comes from the device that has to pay it.

  Activate:  bridge.push(scripts=["inspect"])
*/

function report(viewer) {
  const { renderer, THREE } = viewer;
  const model = viewer.model;
  if (!model) return null;

  const materials = new Set();
  const lightmapped = new Set();  // keyed like `materials`, so the two compare
  const textures = new Map();     // uuid -> texture, so a shared map counts once
  let meshes = 0;

  model.traverse((node) => {
    if (!node.isMesh) return;
    meshes += 1;
    const list = Array.isArray(node.material) ? node.material : node.material ? [node.material] : [];
    for (const material of list) {
      materials.add(material);
      if (material.lightMap) lightmapped.add(material);
      for (const value of Object.values(material)) {
        if (value && value.isTexture && value.image) textures.set(value.uuid, value);
      }
    }
  });

  // Decoded footprint, which is the figure that actually constrains a headset:
  // 4 bytes per texel for RGBA, plus the ~1/3 a full mip chain adds.
  let texels = 0;
  for (const texture of textures.values()) {
    const { width = 0, height = 0 } = texture.image;
    texels += width * height;
  }
  const megabytes = (texels * 4 * (4 / 3)) / (1024 * 1024);

  return {
    meshNodes: meshes,
    drawCalls: renderer.info.render.calls,
    materials: materials.size,
    lightmappedMaterials: lightmapped.size,
    textures: textures.size,
    decodedTextureMB: Math.round(megabytes),
    geometries: renderer.info.memory.geometries,
    three: THREE.REVISION,
  };
}

export default function inspect(viewer) {
  const button = viewer.addButton('Inspect', () => {
    const stats = report(viewer);
    if (!stats) return viewer.setStatus('nothing loaded to inspect');
    // The console is the deliberate sink: these are a dozen numbers a reviewer
    // reads once and copies into a bug report, and the HUD is sized for a line
    // the user reads at a glance while wearing a headset.
    console.table(stats);
    viewer.setStatus(
      `${stats.drawCalls} draw calls · ${stats.meshNodes} mesh nodes · `
      + `${stats.materials} materials · ${stats.textures} textures ≈ ${stats.decodedTextureMB} MB decoded`
    );
  });
  button.title = 'Report draw calls, materials and decoded texture memory (i)';

  viewer.on('key', (event) => {
    if (event.key === 'i') button.click();
  });

  // Reported once per load as well, so a push that quietly doubles the cost
  // says so without anyone having to think to ask.
  viewer.on('load', () => {
    const stats = report(viewer);
    if (stats) console.table(stats);
  });
}
