// Reading the deliverable's blocks: each a JSON object in the glTF's root
// `extras`, under its record's key, in the schema this page reads -- both
// from records.js, generated from pythontk's declarations. Pure -- no
// three.js, no page -- so a feature imports it without the stage, which
// builds the renderer as it loads.

import { RETIRED_BLOCK_KEYS } from './records.js';

// *block*'s object in the root *extras*, or null when the file carries none
// or carries another schema: a file written before the current one is
// re-exported, never read through a branch for its shape. *block* is a
// records.js entry (`{key, version}`).
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
