import * as THREE from 'three';
import { renderer } from './scene.js';

// What can be stood on -- the model's visible meshes, each with its scene-space
// box -- and the one question asked of them: what does a segment strike first?
// Shared by the start, which stands a session on the floor under its node, and
// locomotion, which throws its arc at them and follows them underfoot.
const WALKABLE = Math.cos(THREE.MathUtils.degToRad(40));  // the steepest floor

// Triangles to a cell of a mesh's triangle grid, were they spread evenly: the
// grid's resolution (see `gridOf`).
const CELL_TRIANGLES = 8;
const MAX_GRID_CELLS = 1 << 21;

export const surfaces = (() => {
  const UP = new THREE.Vector3(0, 1, 0);
  const raycaster = new THREE.Raycaster();
  const normalMatrix = new THREE.Matrix3();
  const instanceMatrix = new THREE.Matrix4();
  let list = [];                           // [{mesh, box, gridded}]
  let root = null;                         // the model they were read from ...
  const rootMatrix = new THREE.Matrix4();  // ... and its world matrix then

  // Every headset frame casts at these -- a walk once, for the floor underfoot;
  // an aimed arc once per segment -- and a raycast tests EVERY triangle of a
  // mesh whose bounds its ray crosses: measured, 10 ms a frame on a
  // 320k-triangle mesh, the whole of a 90 Hz budget, for a question about the
  // metre around the viewer. So each mesh's triangles are bucketed, in the
  // mesh's own space, on a uniform grid over its box -- CELL_TRIANGLES to a
  // cell -- and a segment is tested against the triangles in the cells its own
  // box overlaps: a few dozen, however dense the mesh. Built once per geometry,
  // on first use or as a session starts (`warm`), and kept beside it; the
  // model's moves (a turned pivot) need no rebuild, the segment is taken into
  // the mesh's space instead. What the stock raycast knows and a grid of the
  // stored positions does not -- a skinned or morphed mesh draws positions the
  // buffer does not hold, a multi-material one gives each group its own side,
  // an instanced one is many meshes -- keeps the stock raycast.
  const grids = new WeakMap();  // geometry -> its grid

  function griddable(mesh) {
    const { geometry, material } = mesh;
    return !mesh.isSkinnedMesh && !mesh.isInstancedMesh && !mesh.isBatchedMesh
      && !!material && !Array.isArray(material)
      && !!geometry.attributes.position
      && !(geometry.morphAttributes.position && mesh.morphTargetInfluences);
  }

  function gridOf(geometry) {
    let grid = grids.get(geometry);
    if (grid) return grid;
    const position = geometry.attributes.position;
    const order = geometry.index;
    const first = Math.max(0, geometry.drawRange.start);
    const end = Math.min(order ? order.count : position.count,
      geometry.drawRange.start + geometry.drawRange.count);
    const count = Math.max(0, Math.floor((end - first) / 3));
    const corner = order ? (t, k) => order.getX(first + 3 * t + k) : (t, k) => first + 3 * t + k;
    if (!geometry.boundingBox) geometry.computeBoundingBox();
    const box = geometry.boundingBox.clone();
    const low = box.min.toArray();
    const extent = box.getSize(new THREE.Vector3()).toArray();
    // Cubic cells shared out by the box's proportions. An axis thinner than a
    // cell (a floor's height) gets one, and the edge is sized over the rest --
    // rounded up to a cell of its own first, it would multiply the others.
    const wanted = Math.min(Math.max(1, count / CELL_TRIANGLES), MAX_GRID_CELLS);
    let axes = [0, 1, 2].filter((axis) => extent[axis] > 1e-6);
    let edge = 1;
    for (;;) {
      const measure = axes.reduce((product, axis) => product * extent[axis], 1);
      edge = axes.length ? (measure / wanted) ** (1 / axes.length) : 1;
      const thick = axes.filter((axis) => extent[axis] >= edge);
      if (thick.length === axes.length) break;
      axes = thick;
    }
    let dims = extent.map((e, axis) => (axes.includes(axis) ? Math.max(1, Math.ceil(e / edge)) : 1));
    const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
    const span = new Int32Array(6);  // a triangle's cells: x0 y0 z0 x1 y1 z1
    let scale, cells, starts, total;

    const cellAt = (axis, value) =>
      Math.min(dims[axis] - 1, Math.max(0, Math.floor((value - low[axis]) * scale[axis])));
    const spanOf = (t) => {
      a.fromBufferAttribute(position, corner(t, 0));
      b.fromBufferAttribute(position, corner(t, 1));
      c.fromBufferAttribute(position, corner(t, 2));
      for (let axis = 0; axis < 3; axis++) {
        const p = a.getComponent(axis), q = b.getComponent(axis), r = c.getComponent(axis);
        span[axis] = cellAt(axis, Math.min(p, q, r));
        span[axis + 3] = cellAt(axis, Math.max(p, q, r));
      }
    };
    const eachCell = (visit) => {
      for (let z = span[2]; z <= span[5]; z++) {
        for (let y = span[1]; y <= span[4]; y++) {
          for (let x = span[0]; x <= span[3]; x++) visit(x + dims[0] * (y + dims[1] * z));
        }
      }
    };
    // Counted first, coarsened while the long thin triangles of a pathological
    // mesh would file one triangle in thousands of cells.
    for (;;) {
      scale = dims.map((d, axis) => (extent[axis] > 1e-6 ? d / extent[axis] : 0));
      cells = dims[0] * dims[1] * dims[2];
      starts = new Uint32Array(cells + 1);
      total = 0;
      for (let t = 0; t < count; t++) {
        spanOf(t);
        eachCell((cell) => { starts[cell + 1] += 1; total += 1; });
      }
      if (total <= 16 * count + cells || cells === 1) break;
      dims = dims.map((d) => Math.max(1, d >> 1));
    }
    for (let cell = 0; cell < cells; cell++) starts[cell + 1] += starts[cell];
    const refs = new Uint32Array(total);
    const filled = starts.slice(0, cells);
    for (let t = 0; t < count; t++) {
      spanOf(t);
      eachCell((cell) => { refs[filled[cell]++] = t; });
    }
    grid = { count, corner, position, box, low, dims, scale, starts, refs, seen: new Uint32Array(count), stamp: 0 };
    grids.set(geometry, grid);
    return grid;
  }

  const toLocal = new THREE.Matrix4();
  const localRay = new THREE.Ray();
  const localSpan = new THREE.Box3();
  const localTo = new THREE.Vector3();
  const va = new THREE.Vector3(), vb = new THREE.Vector3(), vc = new THREE.Vector3();
  const struck = new THREE.Vector3();
  const struckWorld = new THREE.Vector3();

  // The nearest triangle of `mesh` the scene segment `from` -> `to` (`length`
  // long) strikes, as a stock raycast reports it (`point`, `distance`, `face`
  // and `object`, the sides the material culls culled), or null.
  function strike(mesh, from, to, length) {
    const grid = gridOf(mesh.geometry);
    if (!grid.count) return null;
    toLocal.copy(mesh.matrixWorld).invert();
    localRay.origin.copy(from).applyMatrix4(toLocal);
    localTo.copy(to).applyMatrix4(toLocal);
    localSpan.setFromPoints([localRay.origin, localTo]);
    if (!localSpan.intersectsBox(grid.box)) return null;
    localRay.direction.subVectors(localTo, localRay.origin).normalize();
    const { dims, low, scale, starts, refs, seen, corner, position } = grid;
    const cellAt = (axis, value) =>
      Math.min(dims[axis] - 1, Math.max(0, Math.floor((value - low[axis]) * scale[axis])));
    const lo = localSpan.min.toArray().map((value, axis) => cellAt(axis, value));
    const hi = localSpan.max.toArray().map((value, axis) => cellAt(axis, value));
    const side = mesh.material.side;
    // A triangle filed in several of the cells visited is tested once.
    grid.stamp += 1;
    let best = null;
    for (let z = lo[2]; z <= hi[2]; z++) {
      for (let y = lo[1]; y <= hi[1]; y++) {
        for (let x = lo[0]; x <= hi[0]; x++) {
          const cell = x + dims[0] * (y + dims[1] * z);
          for (let k = starts[cell]; k < starts[cell + 1]; k++) {
            const t = refs[k];
            if (seen[t] === grid.stamp) continue;
            seen[t] = grid.stamp;
            va.fromBufferAttribute(position, corner(t, 0));
            vb.fromBufferAttribute(position, corner(t, 1));
            vc.fromBufferAttribute(position, corner(t, 2));
            const hit = side === THREE.BackSide
              ? localRay.intersectTriangle(vc, vb, va, true, struck)
              : localRay.intersectTriangle(va, vb, vc, side === THREE.FrontSide, struck);
            if (!hit) continue;
            struckWorld.copy(struck).applyMatrix4(mesh.matrixWorld);
            const distance = from.distanceTo(struckWorld);
            if (distance > length || (best && distance >= best.distance)) continue;
            best = {
              distance,
              point: struckWorld.clone(),
              face: { normal: THREE.Triangle.getNormal(va, vb, vc, new THREE.Vector3()) },
              object: mesh,
            };
          }
        }
      }
    }
    return best;
  }

  function read() {
    list = [];
    if (!root) return;
    root.updateWorldMatrix(true, true);
    rootMatrix.copy(root.matrixWorld);
    root.traverseVisible((node) => {
      if (!node.isMesh) return;
      list.push({ mesh: node, box: new THREE.Box3().setFromObject(node), gridded: griddable(node) });
    });
  }

  // Every grid the model's surfaces need, built now rather than on the frame
  // that first casts at them: as a session starts, and as a model lands in one.
  function warm() {
    for (const { mesh, gridded } of list) if (gridded) gridOf(mesh.geometry);
  }

  // Re-read whenever the model has moved as a whole since: a script turning the
  // pivot -- the turntable, before a session -- would otherwise leave every box
  // where its mesh used to be, and the arc would pass straight through a wall.
  // Cheap to redo (a box is the geometry's own, transformed), and asked once a
  // frame rather than per cast: an arc casts dozens of segments.
  function refresh() {
    if (!root) return;
    root.updateWorldMatrix(true, false);
    if (!root.matrixWorld.equals(rootMatrix)) read();
  }

  // The model to read them from; each load hands its own over.
  function setRoot(model) {
    root = model || null;
    read();
    if (renderer.xr.isPresenting) warm();
  }

  // The first surface the segment `from` -> `to` strikes, or null. `valid` when
  // it can be stood on: facing up within WALKABLE, and struck from above.
  function cast(from, to) {
    const direction = new THREE.Vector3().subVectors(to, from);
    const length = direction.length();
    if (!length) return null;
    raycaster.set(from, direction.divideScalar(length));
    raycaster.far = length;
    const span = new THREE.Box3().setFromPoints([from, to]);
    let best = null;
    for (const { mesh, box, gridded } of list) {
      // The segment's box first: a raycast tests each mesh's bounds against the
      // whole INFINITE ray, so every mesh behind the segment would otherwise pay
      // its triangle tests on every segment of an arc, every frame.
      if (!box.intersectsBox(span)) continue;
      const hit = gridded ? strike(mesh, from, to, length) : raycaster.intersectObject(mesh, false)[0];
      if (hit && (!best || hit.distance < best.distance)) best = hit;
    }
    let normal = null;
    if (best) {
      let world = best.object.matrixWorld;
      if (best.instanceId !== undefined) {
        best.object.getMatrixAt(best.instanceId, instanceMatrix);
        world = instanceMatrix.premultiply(world);
      }
      normal = best.face.normal.clone().applyMatrix3(normalMatrix.getNormalMatrix(world)).normalize();
    }
    // The page's own floor, which a model need not carry: a room shipped
    // without one still has somewhere to land, where the grid is drawn.
    if (from.y >= 0 && to.y < 0) {
      const along = from.y / (from.y - to.y);
      if (!best || along * length < best.distance) {
        best = { point: from.clone().lerp(to, along), distance: along * length };
        normal = UP.clone();
      }
    }
    if (!best) return null;
    return {
      point: best.point.clone(),
      normal,
      valid: normal.y >= WALKABLE && raycaster.ray.direction.dot(normal) < 0,
    };
  }

  return { setRoot, refresh, cast, warm };
})();
