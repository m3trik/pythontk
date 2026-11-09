// Reading the deliverable's blocks: each a JSON object in the glTF's root
// `extras`, under its record's key, in the schema this page reads -- both
// from records.js, generated from pythontk's declarations. Pure -- no
// three.js, no page -- so a feature imports it without the stage, which
// builds the renderer as it loads.

import { GLB_MEDIA, RETIRED_BLOCK_KEYS } from './records.js';

// A name as a media join compares it (GlbMedia.fold): its file name, no
// folder, lower-cased.
const fold = (name) => String(name ?? '').replace(/\\/g, '/').split('/').pop().toLowerCase();

// The file a record names, as a GLB carries it (records.js GLB_MEDIA): the
// bufferView whose glTF name is the file's name, joined by `fold` on both
// sides, and what its extras state of its bytes -- `{view, mimeType, scale}`
// (`scale` null but for an 8-bit web encoding of an HDR map) -- or null when
// the file rides no view, or several: a tie, which `tie.tie` says.
export function mediaOf(json, name, tie = {}) {
  const wanted = fold(name);
  const views = json?.bufferViews;
  tie.tie = false;
  if (!wanted || !Array.isArray(views)) return null;
  const found = [];
  views.forEach((entry, index) => {
    if (typeof entry?.[GLB_MEDIA.name] === 'string' && fold(entry[GLB_MEDIA.name]) === wanted) found.push(index);
  });
  tie.tie = found.length > 1;
  if (found.length !== 1) return null;
  const view = found[0];
  const extras = views[view].extras || {};
  const scale = extras[GLB_MEDIA.scale];
  return {
    view,
    mimeType: extras[GLB_MEDIA.mimeType] ?? null,
    scale: typeof scale === 'number' && isFinite(scale) ? scale : null,
  };
}

// *block*'s object in the root *extras*, or null when the file carries none
// or carries another schema: a file written before the current one is
// re-exported, never read through a branch for its shape. *block* is a
// records.js entry (`{key, version}`). A block is a root-extras entry, and
// every one a GLB carries is its record as it stands (pythontk
// `SceneRecords.glb_block`, the same word): what this returns is the record.
export function readBlock(extras, block) {
  const value = extras?.[block.key];
  return value && typeof value === 'object' && value.version === block.version ? value : null;
}

// The load issues for *blocks* the root *extras* carries in another schema,
// and for each *retired* block key it still carries (a file published before
// its record was renamed): readBlock passes over each, and an asset that
// renders as though it carried nothing says why, once per block, rather than
// leaving it to be misread as a bad bake.
export function staleBlocks(extras, blocks, retired = RETIRED_BLOCK_KEYS) {
  const stale = blocks
    .filter((block) => {
      const value = extras?.[block.key];
      return value && typeof value === 'object' && value.version !== block.version;
    })
    .map((block) => ({
      level: 'error',
      text: `'${block.key}' v${extras[block.key].version} is not the v${block.version} this page reads -- re-export the file`,
    }));
  const gone = retired
    .filter((key) => extras?.[key] != null)
    .map((key) => ({ level: 'error', text: `'${key}' is a retired block -- re-export the file` }));
  return [...stale, ...gone];
}
