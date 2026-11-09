// A record's nodes, resolved in the GLB's own scene: the one rule every
// reader of a record (`lightmaps`, `shadows`) resolves its entries by --
// pythontk's `_Media.resolve_entries` (mesh_convert/_media.py), which the
// live suite holds this to (test_preview_viewer_live). Pure -- the glTF JSON
// in, node indices out -- so it runs before the scene is built and without a
// page.
//
// Exact name first, then namespace-tolerant (the name after the last ':');
// objects that share a name are told apart by their `hierarchy` (the path
// from the scene root, its own name last) against where each node sits; every
// set of entries whose candidate nodes overlap is settled together, and an
// entry binds a node only when each is the other's unique best match. A tie
// is never guessed -- one object's lighting would land on another -- and an
// entry whose best candidates all went to better matches is absent: its
// object is not in this file. A record whose entries name nodes that are no
// mesh (the `shadows` record's sources and contacts, the `articulation`
// record's joints) resolves them over every node by the same rule
// (`{ meshes: false }`), and one whose entry is found under another's node (a
// grabbed part under its joint) over that node's subtree alone (`{ under }`).
//
// Two halves, so a reader resolving many names alone builds the file's index
// once: `indexNodes` (the file's nodes by name) and `resolveIndexed` (entries
// against an index); `resolveEntries` is the two together.

const leafOf = (name) => String(name).slice(String(name).lastIndexOf(':') + 1);

/**
 * The file's nodes indexed for resolution, built once and read by every
 * `resolveIndexed` call over the same file: its MESH nodes (every node with
 * `meshes` false) by exact name, each name's namespace-stripped leaf to the
 * full names carrying it, and each such node's names from its scene root
 * down to itself. pythontk's `_Media.node_index`.
 *
 * @param {any} json The parsed glTF document.
 * @param {{meshes?: boolean, under?: number | null}} [options] `meshes` false
 *   indexes every node, not the mesh nodes alone; `under`, a node index,
 *   indexes that node and the nodes below it alone (`subtreeOf`).
 * @returns {{byName: Map<string, number[]>, leaves: Map<string, string[]>, lineage: Map<number, string[]>}}
 */
export function indexNodes(json, { meshes = true, under = null } = {}) {
  const nodes = json?.nodes || [];
  const parents = new Map();
  nodes.forEach((node, index) => {
    for (const child of node?.children || []) {
      // glTF allows a node one parent: the first listed wins.
      if (Number.isInteger(child) && !parents.has(child)) parents.set(child, index);
    }
  });
  const inside = under === null || under === undefined ? null : subtreeOf(json, under);
  const byName = new Map();
  const lineage = new Map();
  nodes.forEach((node, index) => {
    if (!node || (meshes && node.mesh === undefined)) return;
    if (inside && !inside.has(index)) return;
    const name = node.name ?? '';
    if (!byName.has(name)) byName.set(name, []);
    byName.get(name).push(index);
    const chain = [index];
    const seen = new Set(chain);
    while (parents.has(chain[chain.length - 1]) && !seen.has(parents.get(chain[chain.length - 1]))) {
      const up = parents.get(chain[chain.length - 1]);
      chain.push(up);
      seen.add(up);
    }
    lineage.set(index, chain.reverse().map((i) => String(nodes[i]?.name || '')));
  });
  const leaves = new Map();
  for (const full of byName.keys()) {
    const leaf = leafOf(full);
    if (!leaves.has(leaf)) leaves.set(leaf, []);
    leaves.get(leaf).push(full);
  }
  return { byName, leaves, lineage };
}

/**
 * Node *index* and every node below it, through `children`; a cycle ends the
 * walk. pythontk's `_Media.node_subtree`.
 *
 * @param {any} json The parsed glTF document.
 * @param {number} index A node index.
 * @returns {Set<number>}
 */
export function subtreeOf(json, index) {
  const nodes = json?.nodes || [];
  const found = new Set();
  const todo = [index];
  while (todo.length) {
    const at = todo.pop();
    if (!Number.isInteger(at) || found.has(at) || at < 0 || at >= nodes.length) continue;
    found.add(at);
    for (const child of nodes[at]?.children || []) todo.push(child);
  }
  return found;
}

// How many trailing names *hierarchy* and *lineage* share, namespace-blind,
// stopped at the first disagreement (an export may root the file under a node
// the scene never had, but cannot rename the parents it keeps).
function hierarchyScore(hierarchy, lineage) {
  let score = 0;
  for (let a = hierarchy.length - 1, b = lineage.length - 1; a >= 0 && b >= 0; a -= 1, b -= 1) {
    if (leafOf(hierarchy[a]) !== leafOf(lineage[b])) break;
    score += 1;
  }
  return score;
}

const better = (a, b) => a[0] - b[0] || a[1] - b[1];  // (hierarchy score, exact name)

// The key with the strictly highest rank among *pairs* (`[key, rank]`), or
// null on a tie or none.
function uniqueBest(pairs) {
  if (!pairs.length) return null;
  let top = pairs[0][1];
  for (const [, rank] of pairs) if (better(rank, top) > 0) top = rank;
  const best = pairs.filter(([, rank]) => better(rank, top) === 0);
  return best.length === 1 ? best[0][0] : null;
}

/**
 * Resolve every entry of a record list (`objects`, or a highlight copy's) to
 * the glTF's mesh nodes, as ONE assignment.
 *
 * @param {any} json The parsed glTF document.
 * @param {any[]} entries `{name, hierarchy?}` each.
 * @param {{meshes?: boolean}} [options] `meshes` false resolves over every
 *   node, not the mesh nodes alone.
 * @returns {{nodes: number[] | null, status: 'exact'|'leaf'|'ambiguous'|'absent', leaves: string[]}[]}
 *   One per entry, aligned: the node indices it lights, how its name matched,
 *   and -- for a tie -- where each candidate sits.
 */
export function resolveEntries(json, entries, { meshes = true } = {}) {
  return resolveIndexed(indexNodes(json, { meshes }), entries);
}

/**
 * `resolveEntries` against an index `indexNodes` built: the entries as ONE
 * assignment over the nodes it holds. A reader that resolves names one at a
 * time (each its own assignment) indexes the file once and calls this per
 * name, never `resolveEntries`, which re-indexes the whole file each call.
 *
 * @param {ReturnType<typeof indexNodes>} index The file's nodes, indexed.
 * @param {any[]} entries `{name, hierarchy?}` each.
 * @returns {ReturnType<typeof resolveEntries>}
 */
export function resolveIndexed(index, entries) {
  const { byName, leaves, lineage } = index;
  /** @type {{nodes: number[] | null, status: 'exact'|'leaf'|'ambiguous'|'absent', leaves: string[]}[]} */
  const resolved = [];
  const pools = new Map();  // entry index -> its candidate nodes
  entries.forEach((entry, i) => {
    const name = entry && typeof entry === 'object' ? entry.name : null;
    // A malformed entry never reaches the lookup: '' indexes NAMELESS nodes.
    if (!name || typeof name !== 'string') {
      resolved.push({ nodes: null, status: 'absent', leaves: [] });
      return;
    }
    /** @type {typeof resolved[number]} */
    let result;
    if (byName.has(name)) {
      result = { nodes: byName.get(name), status: 'exact', leaves: [] };
    } else {
      const named = leaves.get(leafOf(name)) || [];
      if (named.length === 1) result = { nodes: byName.get(named[0]), status: 'leaf', leaves: named };
      else if (named.length > 1) result = { nodes: null, status: 'ambiguous', leaves: named };
      else result = { nodes: null, status: 'absent', leaves: [] };
    }
    resolved.push(result);
    const pool = result.nodes || result.leaves.flatMap((full) => byName.get(full) || []);
    if (pool.length) pools.set(i, pool);
  });

  // Union-find over node indices: entries sharing any candidate land together.
  const parent = new Map();
  const root = (x) => {
    if (!parent.has(x)) parent.set(x, x);
    while (parent.get(x) !== x) {
      parent.set(x, parent.get(parent.get(x)));
      x = parent.get(x);
    }
    return x;
  };
  for (const pool of pools.values()) {
    const first = root(pool[0]);
    for (const node of pool.slice(1)) parent.set(root(node), first);
  }
  const groups = new Map();
  for (const [i, pool] of pools) {
    const key = root(pool[0]);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(i);
  }

  for (const members of groups.values()) {
    const all = new Set(members.flatMap((i) => pools.get(i)));
    if (members.length === 1 && all.size === 1) continue;  // one entry, one node
    const ranks = new Map();  // `${i}:${node}` -> [score, exact]
    const takers = new Map();  // node -> entries it could be
    for (const i of members) {
      const hierarchy = entries[i].hierarchy;
      const usable = Array.isArray(hierarchy) && hierarchy.length > 0;
      const exact = resolved[i].status === 'exact' ? 1 : 0;
      for (const node of pools.get(i)) {
        ranks.set(`${i}:${node}`, [usable ? hierarchyScore(hierarchy, lineage.get(node) || []) : 0, exact]);
        if (!takers.has(node)) takers.set(node, []);
        takers.get(node).push(i);
      }
    }
    const rank = (i, node) => ranks.get(`${i}:${node}`);
    const bestNode = new Map(members.map((i) => [i, uniqueBest(pools.get(i).map((n) => [n, rank(i, n)]))]));
    const owner = new Map();
    for (const [node, candidates] of takers) {
      const taker = uniqueBest(candidates.map((i) => [i, rank(i, node)]));
      if (taker !== null && bestNode.get(taker) === node) owner.set(node, taker);
    }
    for (const i of members) {
      const node = bestNode.get(i);
      if (node !== null && owner.get(node) === i) {
        resolved[i] = { nodes: [node], status: resolved[i].status === 'exact' ? 'exact' : 'leaf', leaves: [] };
        continue;
      }
      const pool = pools.get(i);
      let top = rank(i, pool[0]);
      for (const n of pool) if (better(rank(i, n), top) > 0) top = rank(i, n);
      const tops = pool.filter((n) => better(rank(i, n), top) === 0);
      if (tops.every((n) => owner.has(n))) {
        // Every node this entry could best be went to a better match.
        resolved[i] = { nodes: null, status: 'absent', leaves: [] };
      } else {
        const paths = pool.map((n) => (lineage.get(n) || []).join('/')).sort();
        resolved[i] = { nodes: null, status: 'ambiguous', leaves: paths };
      }
    }
  }
  return resolved;
}
