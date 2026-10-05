import { el } from './dom.js';
import { lightmappedMaterials } from './lightmaps.js';

// Lookdev is the area for dials that tune how the model READS in the preview
// and then write the result back into the GLB, rather than into this page --
// the normals dial below is its first and, for now, only occupant.
//
// Held back behind one flag instead of deleted. The dial works and its file
// side ships (`normalTexture.scale`, written by MeshConvert.set_glb_normal_scale
// through POST /settings), but a lone slider does not earn a permanent seat in
// a control bar that has to survive a phone-sized viewport, and lookdev worth
// exposing is a SET -- exposure, environment rotation, tone mapping -- tuned
// together. So the area stays wired and untested-by-eye rather than rotting:
// flip this to true and it returns exactly as it was.
//
// Adding the next dial: markup inside #lookdev, its own sync called from
// syncLookdev, and its element in that function's `controls` list so the panel
// keeps following its contents. Nothing else here counts the dials.
const LOOKDEV_ENABLED = false;

// Each dial hides itself when the model gives it nothing to do (the normals
// dial on a model with no baked normal map), and the panel follows them: an
// empty chrome row over the model is worse than no row.
export function syncLookdev() {
  syncNormalScale();
  const controls = [el.normals];
  el.lookdev.hidden = !LOOKDEV_ENABLED || controls.every((control) => control.hidden);
}

// How strongly the normal maps read on the BAKED materials, and only those.
//
// A lightmap contributes irradiance with no direction in it -- three.js adds
// `lightMapTexel * intensity` to the diffuse term and never consults the
// normal -- so on a baked surface the normal map survives only through the
// environment's specular term (its diffuse is withheld from a baked surface,
// see patchBakedShader). Correct, and flat-looking on a matte surface: the
// detail is there and barely lit. This is the dial that brings it back without
// touching the bake or re-lighting the room, which is why it is hidden
// entirely when nothing is baked.
//
// Its home is the FILE, not this page: `normalTexture.scale` is core glTF, so
// GLTFLoader has already read the model's own value into `material.normalScale`
// and the save button writes the new one back the same way. Hence a sync FROM
// the model on every load rather than a value the page remembers -- open a GLB
// saved at 1.8 and the slider says 1.8, in this viewer and in any other.
// Keyed on a baked material that actually HAS a normal map, not merely on the
// model being baked: `normalScale` exists on every MeshStandardMaterial, so
// reading the first baked material would show 1.00 for a file saved at 1.8
// whenever that first material happens to carry no normal map -- and offering
// the dial at all on a bake with no normal maps anywhere is a control that
// cannot change the render.
function normalCarriers() {
  return lightmappedMaterials.filter((material) => material.normalMap);
}

function syncNormalScale() {
  const carriers = normalCarriers();
  const value = carriers.length ? carriers[0].normalScale.x : 1;
  el.normals.hidden = !carriers.length;
  el.normalScale.value = String(value);
  el.normalValue.textContent = value.toFixed(2);
  el.normalSave.disabled = true;
  el.normalSave.textContent = 'Save';
}

// The dial sets the MAGNITUDE and keeps each material's own sign on y.
// GLTFLoader negates `normalScale.y` on a material whose mesh carries no
// TANGENT -- every GLB this pipeline shipped before the DCC hand-offs pinned
// tangents, and any other producer's that omits them -- because the
// derivative frame it falls back to runs green the other way, so
// `set(value, value)` turned green over on every such normal map at the first
// touch (measured: the loader's (1, -1) became (1.5, 1.5)). `1 / y` rather
// than `y < 0`: the sign has to survive the dial passing through 0, and
// -0 < 0 is false.
export function applyNormalScale(value) {
  for (const material of normalCarriers()) {
    material.normalScale.set(value, Math.sign(1 / material.normalScale.y) * value);
  }
  el.normalValue.textContent = value.toFixed(2);
  el.normalSave.disabled = false;
  el.normalSave.textContent = 'Save';
}

// Hands the value to the server, which writes it into the GLB with
// MeshConvert.set_glb_normal_scale. No reload follows: the page is already
// showing the value, and the write deliberately leaves the manifest version
// alone so saving does not bounce the model out from under the user.
export async function saveNormalScale() {
  el.normalSave.disabled = true;
  el.normalSave.textContent = 'Saving…';
  try {
    const response = await fetch('settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ normal_scale: Number(el.normalScale.value) }),
    });
    if (!response.ok) throw new Error(response.statusText);
    const result = await response.json();
    el.normalSave.textContent = `Saved · ${result.materials?.normal_scale ?? 0}`;
  } catch (error) {
    // Left ENABLED on failure: the value is still only in the page, so the
    // one useful thing the user can do is try again.
    el.normalSave.textContent = 'Save failed';
    el.normalSave.disabled = false;
    console.warn('Normal scale not saved:', error);
  }
}
