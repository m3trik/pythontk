# !/usr/bin/python
# coding=utf-8
"""Band-based UV seam placement for cylinder / tube / turned meshes.

:class:`CylinderSeams` works on plain arrays -- vertex positions, face vertex
lists, the mesh's edge table and its hard-edge ids -- so a host reads its mesh
into those and gets back edge ids into its OWN edge table, ready to cut. It
touches no scene, which is also what lets it be exercised offline against
hand-seamed reference meshes.

The seams reproduce what a texture artist does by hand on a turned or swept
part -- *clean and minimal*:

- The mesh is read as **rings** (edge chains that run around the tube) and
  the quad **bands** between consecutive rings. Non-quad faces (n-gon caps,
  triangle fans) form **cap** regions.
- Every band is classified by how its lengthwise edges sit against the
  band's own axis (the line through its two ring centres, so a bent or
  mitered tube is read locally): a **wall** runs along the axis (a
  cylinder, a slight taper); a **cone** is tilted further (a chamfer, a
  flare, a funnel) but still develops exactly as a sector; a **flat** band
  is near-perpendicular (a washer, a step) or leads into a cap (a dome) and
  lays out as a closed annulus / disc. Walls and cones are *strips*; a
  *trim* band (a fillet or bevel a few percent of the radius tall) joins the
  wall it rounds off, so a rounded collar rides its strip and a bead ring
  stays a ring.
- Every strip run is opened with **one** lengthwise cut, taken from a single
  edge chain (a *column*) that is followed through the whole tube, so every
  strip's seam lines up (a straight column on a turned part; the loop that
  follows the surface on a bent hose). Flat bands never get a lengthwise
  cut: an annulus / disc unfolds as-is. A run with no open end at all (a
  torus) also gets one crossing ring so it can unroll.
- A ring is cut where a strip meets a flat band (a step rim, a cap rim) --
  regardless of how shallow the angle -- and between two strips wherever the
  profile turns more than the taper tolerance (a chamfer never merges into
  its cylinders, a bent hose stays one strip); two flat bands only split at
  a fold-back (a countersink meeting a step), so a chamfer running into a
  step, or a domed cap with its fillet, stays one annulus / disc. An
  authored hard ring is a seam whatever the angle. ``angle`` is the crease
  threshold for geometry the tube reading doesn't cover, and -- set below
  the taper tolerance -- splits strips at gentler kinks.
- The column is chosen to hide the seam: the column facing away from the
  viewer (``camera`` -- :data:`DEFAULT_VIEW_DIR` when none is given, so the
  pick is deterministic) is used; ``invert_seam`` takes the opposite side.

3D boundary edges (an open tube's rims) are already UV borders and are never
cut. Faces that don't fit the band structure (a non-tube appendage, a
region where the quad grid breaks down) fall back to plain crease cuts.

After :meth:`seams`, :meth:`seed_uvs` hands out the developed shape of every
shell (strips unrolled from their seam, rings and discs unrolled radially) --
the seed ``unwrap_cylinder`` gives Unfold3D so it has nothing to untangle.
"""

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

Vec = Tuple[float, float, float]

#: Default taper tolerance (degrees): a band whose lengthwise edges tilt from
#: its own axis by more than this is a cone (its own sector) rather than a
#: wall, and two strips split where the profile turns by more than this. A
#: slight taper rides its cylinder's strip; a ~28 degree chamfer or flare
#: between two cylinders (the shallowest in the references) is its own piece.
DEFAULT_TAPER_ANGLE = 20.0

#: A hard ring whose faces are (near) coplanar carries no crease -- an
#: authored-hard but flat ring must not split anything.
COPLANAR_EPS_DEG = 0.5

#: Default flat angle (degrees): a band tilted from its axis past this is
#: flat -- a closed annulus: forcing a full ring on it stretches it by at most
#: 1 / sin(60) ~ 15%. Below it a band is a cone: a strip cut open along the
#: column, whose sector is its exact development (a 45 degree flare left as a
#: ring would stretch 41%, a 28 degree chamfer 2x). An artist who prefers
#: bevels as rings lowers it (45 keeps a 45 degree chamfer closed).
DEFAULT_FLAT_ANGLE = 60.0

#: Default trim ratio: a band shorter than this fraction of its ring radius
#: is *trim* (a fillet, a bevel, a bead) and joins a neighbour instead of
#: counting on its own.
DEFAULT_TRIM_RATIO = 0.12

#: A fold this sharp (degrees) is a hard corner whatever the bands: a trim
#: band never joins across it (a bead between two steps stays a ring), a cap
#: interior splits at it, and a cone stops counting as part of a cap there.
SHARP_FOLD = 60.0

#: Viewing direction assumed when no camera is given: the eye sits far out
#: along (1, 1, 1), where a default perspective camera looks from (Maya's
#: looks down (-1, -1, -1)), so the seam lands on the (-X, -Y, -Z) side of
#: the mesh.
DEFAULT_VIEW_DIR = (1.0, 1.0, 1.0)

WALL, CONE, FLAT = 0, 1, 2
RING, LENGTH = 0, 1


class CylinderSeams:
    """Band decomposition + seam selection on plain arrays.

    Build it from the mesh arrays, call :meth:`seams` for the edge ids to cut
    (also kept on :attr:`cuts`), then optionally :meth:`seed_uvs` for each
    shell's developed layout -- the seed an unfold starts from.

    Parameters:
        points: Vertex positions ``[(x, y, z), ...]`` (world space).
        faces: Per-face vertex id lists, in winding order.
        edges: ``[(a, b), ...]`` -- the mesh's edge table. Seam ids returned
            index into this list, so passing the host mesh's own edge order
            makes the result directly usable as its edge components.
        hard: Edge ids flagged hard (authored shading). Only consulted when
            the mesh also has soft edges -- an all-hard import carries no
            authoring signal -- and only ring-wide: a ring most of whose
            edges are hard is a seam, a lone hard facet edge is not.
    """

    def __init__(
        self,
        points: Sequence[Vec],
        faces: Sequence[Sequence[int]],
        edges: Sequence[Tuple[int, int]],
        hard: Optional[Iterable[int]] = None,
    ):
        self.pts = np.asarray(points, dtype=float).reshape(-1, 3)
        self.faces = [list(f) for f in faces]
        self.edges = [tuple(e) for e in edges]
        self.hard = set(hard or ())
        n_edges = len(self.edges)
        self.authored = 0 < len(self.hard) < n_edges  # both hard and soft present

        # --- adjacency ------------------------------------------------------
        key = {}
        for i, (a, b) in enumerate(self.edges):
            key[(a, b) if a < b else (b, a)] = i
        self.edge_faces: Dict[int, List[int]] = defaultdict(list)
        self.face_edges: List[List[int]] = []
        for fi, f in enumerate(self.faces):
            fe = []
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                e = key[(a, b) if a < b else (b, a)]
                fe.append(e)
                self.edge_faces[e].append(fi)
            self.face_edges.append(fe)
        self.vert_edges: Dict[int, List[int]] = defaultdict(list)
        for i, (a, b) in enumerate(self.edges):
            self.vert_edges[a].append(i)
            self.vert_edges[b].append(i)
        self.boundary = {e for e in range(n_edges) if len(self.edge_faces[e]) < 2}

        # --- face normals (Newell) ---------------------------------------
        self.normals = np.zeros((len(self.faces), 3))
        for fi, f in enumerate(self.faces):
            p = self.pts[f]
            n = np.zeros(3)
            for k in range(len(f)):
                n += np.cross(p[k], p[(k + 1) % len(f)])
            norm = np.linalg.norm(n)
            self.normals[fi] = n / norm if norm > 1e-12 else n

        # Filled by seams().
        self.cuts: Set[int] = set()
        self.edge_class: Dict[int, int] = {}
        self.irregular: Set[int] = set()
        self.rings: List[List[int]] = []
        self.ring_of: Dict[int, int] = {}
        self.band_keys: List[Tuple[int, int]] = []
        self.bands: Dict[Tuple[int, int], List[int]] = {}
        self.band_of_face: Dict[int, int] = {}
        self.band_kind: List[Optional[int]] = []
        self.band_length_edges: List[Set[int]] = []
        self.band_gen: List[float] = []
        self.band_size: List[float] = []
        self.bands_at_ring: Dict[int, List[int]] = defaultdict(list)
        self.caps: List[List[int]] = []
        self.cap_of_face: Dict[int, int] = {}
        self.caps_at_ring: Dict[int, Set[int]] = defaultdict(set)
        self.joined: Set[Tuple[int, int]] = set()
        self.runs: List[List[int]] = []

    # ------------------------------------------------------------ geometry
    def dihedral(self, e: int) -> Optional[float]:
        """Angle (degrees) between the two face normals at edge ``e``; ``None``
        on a boundary / non-manifold edge."""
        fs = self.edge_faces[e]
        if len(fs) != 2:
            return None
        d = float(np.clip(np.dot(self.normals[fs[0]], self.normals[fs[1]]), -1, 1))
        return math.degrees(math.acos(d))

    def edge_vec(self, e: int) -> np.ndarray:
        a, b = self.edges[e]
        return self.pts[b] - self.pts[a]

    def edge_mid(self, e: int) -> np.ndarray:
        a, b = self.edges[e]
        return (self.pts[a] + self.pts[b]) * 0.5

    def _edge_is_crease(self, e: int, angle: float) -> bool:
        """Per-edge crease test (irregular regions, cap interiors): dihedral
        only. A single edge's hard flag proves nothing -- a DCC's own
        primitives (Maya's) flag every facet edge of a smooth cylinder hard -- so
        authored hardness is only trusted ring-wide (see the ring rules)."""
        d = self.dihedral(e)
        return d is not None and d >= angle

    def _chain_dihedral(self, ch: List[int]) -> Optional[float]:
        """Mean dihedral (degrees) along an edge chain; ``None`` when no edge
        of it has two faces."""
        dih = [d for d in (self.dihedral(e) for e in ch) if d is not None]
        return sum(dih) / len(dih) if dih else None

    def _ring_dihedral(self, ri: int) -> float:
        d = self._chain_dihedral(self.rings[ri])
        return 180.0 if d is None else d

    def _ring_is_crease(self, ch: List[int], min_angle: float) -> bool:
        """A ring is a crease when it is authored hard (most edges hard, and
        not coplanar) or bends past ``min_angle`` on average."""
        mean = self._chain_dihedral(ch)
        if mean is None:
            return False
        if self.authored:
            n_hard = sum(1 for e in ch if e in self.hard)
            if n_hard * 2 > len(ch) and mean > COPLANAR_EPS_DEG:
                return True
        return mean >= min_angle

    def _ring_cut(self, ri: int) -> bool:
        """Whether ring ``ri`` carries a seam (any of its edges -- a junction
        ring cut per edge counts as cut, so no run merges across it)."""
        return any(e in self.cuts for e in self.rings[ri])

    # ---------------------------------------------------- ring / column class
    # Each quad has two opposite-edge pairs; on a tube one pair runs around
    # (ring edges), the other along (lengthwise edges). The class is seeded
    # from edges that can only be ring edges -- a cap's rim (quad edges shared
    # with a non-quad face) and an open tube's rim (boundary quad edges) --
    # and propagated across the quad grid: opposite edges of a quad share a
    # class, adjacent edges differ.
    def _classify_edges(self) -> Tuple[Dict[int, int], Set[int]]:
        """Return ``(edge -> RING/LENGTH, irregular quad ids)``."""
        quads = [fi for fi, f in enumerate(self.faces) if len(f) == 4]
        quad_set = set(quads)
        seeds: List[int] = []
        for e, fs in self.edge_faces.items():
            if not any(f in quad_set for f in fs):
                continue
            if any(f not in quad_set for f in fs):  # cap rim
                seeds.append(e)
            elif e in self.boundary:  # open rim
                seeds.append(e)

        assign: Dict[int, int] = {}  # quad -> which pair (0 / 1) is RING
        conflict: Set[int] = set()
        edge_class: Dict[int, int] = {}

        def pair_of(fi: int, e: int) -> int:
            return self.face_edges[fi].index(e) % 2

        def set_quad(fi: int, ring_pair: int) -> None:
            assign[fi] = ring_pair
            fe = self.face_edges[fi]
            for k in range(4):
                cls = RING if k % 2 == ring_pair else LENGTH
                prev = edge_class.get(fe[k])
                if prev is not None and prev != cls:
                    conflict.add(fi)
                    conflict.update(g for g in self.edge_faces[fe[k]] if g in quad_set)
                edge_class[fe[k]] = cls

        def propagate(stack: List[int]) -> None:
            while stack:
                fi = stack.pop()
                for k, e in enumerate(self.face_edges[fi]):
                    cls = RING if k % 2 == assign[fi] else LENGTH
                    for g in self.edge_faces[e]:
                        if g not in quad_set or g == fi:
                            continue
                        want = pair_of(g, e) if cls == RING else (pair_of(g, e) + 1) % 2
                        if g in assign:
                            if assign[g] != want:
                                conflict.add(g)
                                conflict.add(fi)
                            continue
                        set_quad(g, want)
                        stack.append(g)

        def seed(edge_ids: Iterable[int]) -> None:
            stack: List[int] = []
            for e in edge_ids:
                if edge_class.get(e, RING) != RING:
                    continue
                edge_class[e] = RING
                for fi in self.edge_faces[e]:
                    if fi not in quad_set:
                        continue
                    if fi in assign:
                        if assign[fi] != pair_of(fi, e):
                            conflict.add(fi)
                        continue
                    set_quad(fi, pair_of(fi, e))
                    stack.append(fi)
            propagate(stack)

        seed(seeds)
        # Quads no cap / rim seed reaches (a closed all-quad island such as a
        # torus): seed geometrically, island by island.
        while True:
            unreached = [fi for fi in quads if fi not in assign]
            if not unreached:
                break
            seed(self._geometric_seeds(unreached))
        irregular = set(conflict) | {fi for fi in quads if fi not in assign}
        return edge_class, irregular

    def _geometric_seeds(self, quads: List[int]) -> List[int]:
        """Ring seeds for an all-quad closed island (a torus, a closed
        revolved shape with quad-only poles).

        Both quad-edge families are tried as the ring family; the one whose
        chains are geometrically shorter wins (a torus's minor circles), which
        makes such a shape open with one lengthwise loop plus one crossing
        ring rather than being sliced as a stack of annuli.
        """
        fi = quads[0]
        best: Optional[Tuple[float, int]] = None
        for pair in (0, 1):
            e0 = self.face_edges[fi][pair]
            length = self._loop_length(e0)
            if best is None or length < best[0]:
                best = (length, e0)
        return [best[1]] if best else []

    def _loop_length(self, e0: int) -> float:
        """Geometric length of the edge loop through ``e0`` (walking straight
        through 4-valent vertices)."""
        seen = {e0}
        total = float(np.linalg.norm(self.edge_vec(e0)))
        for start in self.edges[e0]:
            e, v = e0, start
            while True:
                nxt = self._straight_through(e, v)
                if nxt is None or nxt in seen:
                    break
                seen.add(nxt)
                total += float(np.linalg.norm(self.edge_vec(nxt)))
                a, b = self.edges[nxt]
                v = b if a == v else a
                e = nxt
        return total

    def _straight_through(self, e: int, v: int) -> Optional[int]:
        """The edge continuing ``e`` straight through vertex ``v`` (the one
        sharing no face with ``e``), or ``None`` off a 4-valent vertex."""
        cand = [x for x in self.vert_edges[v] if x != e]
        if len(cand) != 3:
            return None
        fs = set(self.edge_faces[e])
        nxt = [x for x in cand if not (set(self.edge_faces[x]) & fs)]
        return nxt[0] if len(nxt) == 1 else None

    # ------------------------------------------------------ chains (rings)
    def _chains(self, edge_ids: Iterable[int]) -> List[List[int]]:
        """Split a set of edges into vertex-connected chains, following each
        chain through vertices that carry exactly two of the set's edges.
        Vertices with more than two (a junction) end the chain there."""
        ids = set(edge_ids)
        at: Dict[int, List[int]] = defaultdict(list)
        for e in ids:
            for v in self.edges[e]:
                at[v].append(e)
        seen: Set[int] = set()
        chains: List[List[int]] = []
        for e0 in sorted(ids):
            if e0 in seen:
                continue
            chain = [e0]
            seen.add(e0)
            for start in self.edges[e0]:
                e, v = e0, start
                while True:
                    here = at[v]
                    if len(here) != 2:
                        break
                    nxt = here[0] if here[1] == e else here[1]
                    if nxt in seen:
                        break
                    seen.add(nxt)
                    chain.append(nxt)
                    a, b = self.edges[nxt]
                    v = b if a == v else a
                    e = nxt
            chains.append(chain)
        return chains

    def _chain_verts(self, chain: List[int]) -> List[int]:
        vs: List[int] = []
        for e in chain:
            for v in self.edges[e]:
                if v not in vs:
                    vs.append(v)
        return vs

    def _chain_closed(self, chain: List[int]) -> bool:
        deg: Dict[int, int] = defaultdict(int)
        for e in chain:
            for v in self.edges[e]:
                deg[v] += 1
        return len(chain) > 2 and all(d == 2 for d in deg.values())

    def _plane_normal(self, vids: List[int]) -> np.ndarray:
        """Best-fit plane normal of a vertex set (smallest covariance axis)."""
        p = self.pts[vids]
        if len(vids) < 3:
            return np.zeros(3)
        c = p - p.mean(axis=0)
        _, vecs = np.linalg.eigh(c.T @ c)
        return vecs[:, 0]

    # ------------------------------------------------------------ main
    def seams(
        self,
        angle: float = 45.0,
        taper_angle: float = DEFAULT_TAPER_ANGLE,
        invert_seam: bool = False,
        camera: Optional[Vec] = None,
        flat_angle: float = DEFAULT_FLAT_ANGLE,
        trim_ratio: float = DEFAULT_TRIM_RATIO,
    ) -> Set[int]:
        """Edge ids to cut. See the module docstring for the rules.

        Parameters:
            angle: Crease threshold (degrees) for edges of regions the tube
                reading doesn't cover; set below ``taper_angle`` it also
                becomes the kink at which two strips split.
            taper_angle: Taper tolerance (degrees): a band tilted from its
                axis by more than this is a cone rather than a wall, and two
                strips split where the profile turns by more than this.
            invert_seam: Put the lengthwise seam on the side facing the
                viewer instead of away from it.
            camera: Viewer eye position; ``None`` places the eye far out along
                :data:`DEFAULT_VIEW_DIR`.
            flat_angle: Bands tilted from their axis past this (degrees) are
                flat -- closed rings; steeper ones (cones) are cut open on the
                column. Lower it to keep more bevels as rings.
            trim_ratio: Bands shorter than this fraction of their ring radius
                are trim (fillets, bevels, beads) and ride a neighbour instead
                of becoming shells of their own.
        """
        self.edge_class, self.irregular = self._classify_edges()
        self._decompose()
        self._classify_bands(taper_angle, flat_angle, trim_ratio)
        self.cuts = self._ring_cuts(angle, taper_angle)
        self._open_strips(camera, invert_seam)
        return self.cuts

    def _decompose(self) -> None:
        """Rings, the quad bands between them, cap regions, and each band's
        generator angle and height-over-radius."""
        ring_edges = {e for e, c in self.edge_class.items() if c == RING}
        self.rings = self._chains(ring_edges)
        self.ring_of = {e: ri for ri, ch in enumerate(self.rings) for e in ch}

        # Bands: regular quads grouped by their two rings.
        bands: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for fi, f in enumerate(self.faces):
            if len(f) != 4 or fi in self.irregular:
                continue
            rs = sorted(
                {self.ring_of[e] for e in self.face_edges[fi] if e in self.ring_of}
            )
            if len(rs) != 2:
                self.irregular.add(fi)
                continue
            bands[(rs[0], rs[1])].append(fi)
        self.bands = bands
        self.band_keys = sorted(bands)
        self.band_of_face = {
            fi: bi for bi, k in enumerate(self.band_keys) for fi in bands[k]
        }
        self.bands_at_ring = defaultdict(list)
        for bi, (r1, r2) in enumerate(self.band_keys):
            self.bands_at_ring[r1].append(bi)
            self.bands_at_ring[r2].append(bi)

        # Cap regions: connected non-quad faces.
        self.caps, self.cap_of_face = [], {}
        seen: Set[int] = set()
        for f0 in (fi for fi, f in enumerate(self.faces) if len(f) != 4):
            if f0 in seen:
                continue
            comp, stack = [], [f0]
            seen.add(f0)
            while stack:
                fi = stack.pop()
                comp.append(fi)
                for e in self.face_edges[fi]:
                    for g in self.edge_faces[e]:
                        if g not in seen and len(self.faces[g]) != 4:
                            seen.add(g)
                            stack.append(g)
            for fi in comp:
                self.cap_of_face[fi] = len(self.caps)
            self.caps.append(comp)
        self.caps_at_ring = defaultdict(set)
        for ci, cap in enumerate(self.caps):
            for fi in cap:
                for e in self.face_edges[fi]:
                    if e in self.ring_of:
                        self.caps_at_ring[self.ring_of[e]].add(ci)

        # Band geometry: generator angle against the band's own axis, and
        # height over ring radius.
        ring_verts = [self._chain_verts(ch) for ch in self.rings]
        ring_center = [self.pts[vs].mean(axis=0) for vs in ring_verts]
        ring_radius = [
            float(np.linalg.norm(self.pts[vs] - c, axis=1).mean())
            for vs, c in zip(ring_verts, ring_center)
        ]
        self.band_length_edges, self.band_gen, self.band_size = [], [], []
        for r1, r2 in self.band_keys:
            ledges = {
                e
                for fi in bands[(r1, r2)]
                for e in self.face_edges[fi]
                if self.edge_class.get(e) == LENGTH
            }
            self.band_length_edges.append(ledges)
            vecs = [self.edge_vec(e) for e in ledges]
            height = float(np.mean([np.linalg.norm(v) for v in vecs])) if vecs else 0.0
            radius = 0.5 * (ring_radius[r1] + ring_radius[r2])
            self.band_size.append(height / radius if radius > 1e-9 else float("inf"))
            # Band axis: the line through the two ring centres; a washer's
            # centres coincide, so fall back to its (shared) ring plane normal.
            axis = ring_center[r2] - ring_center[r1]
            if np.linalg.norm(axis) < 0.2 * height:
                n1 = self._plane_normal(ring_verts[r1])
                n2 = self._plane_normal(ring_verts[r2])
                axis = n1 + (n2 if np.dot(n1, n2) >= 0 else -n2)
            norm = np.linalg.norm(axis)
            if norm < 1e-12 or not vecs:
                self.band_gen.append(90.0)
                continue
            axis = axis / norm
            cos_mean = float(
                np.mean(
                    [abs(np.dot(v, axis)) / (np.linalg.norm(v) or 1.0) for v in vecs]
                )
            )
            self.band_gen.append(math.degrees(math.acos(min(1.0, cos_mean))))

    def _classify_bands(
        self, taper_angle: float, flat_angle: float, trim_ratio: float
    ) -> None:
        """WALL: cylinder-like. CONE: tilted past the taper tolerance but
        short of flat -- a strip too, developed as a sector. FLAT: annulus /
        disc. Trim bands join a neighbour, taking its kind."""
        band_kind: List[Optional[int]] = []
        for gen, size in zip(self.band_gen, self.band_size):
            if size < trim_ratio:
                band_kind.append(None)
            elif gen <= taper_angle:
                band_kind.append(WALL)
            elif gen < flat_angle:
                band_kind.append(CONE)
            else:
                band_kind.append(FLAT)
        self.band_kind = band_kind

        # A cone that leads into a cap -- reached from a cap region through
        # flat / cone / trim bands across rings that aren't sharp folds -- is
        # part of that cap (a domed end): it lays out with the disc rather
        # than as a sector of its own.
        cap_side: Set[int] = set()
        frontier = list(self.caps_at_ring)
        seen_rings: Set[int] = set()
        while frontier:
            r = frontier.pop()
            if r in seen_rings or self._ring_is_crease(self.rings[r], SHARP_FOLD):
                continue
            seen_rings.add(r)
            for bi in self.bands_at_ring[r]:
                if band_kind[bi] == WALL or bi in cap_side:
                    continue
                cap_side.add(bi)
                frontier.extend(rr for rr in self.band_keys[bi] if rr != r)
        for bi in cap_side:
            if band_kind[bi] == CONE:
                band_kind[bi] = FLAT

        # Trim bands join a neighbour, taking its kind; the ring between them
        # is never cut. A wall neighbour is preferred (a fillet is the
        # rounding of the wall's edge -- it rides the strip and the seam hides
        # in the crease beyond it), the smoother one if both sides are walls;
        # otherwise the neighbour across the smoothest ring. A join never
        # crosses a sharp crease (a bead between two hard rings stays its own
        # ring), and a trim band waits for its trim neighbours to resolve
        # before settling for a flat side, so a fillet chain reaches its wall.
        self.joined = set()  # (ring, band) pairs never cut

        def options_for(bi: int, prefer_wall: bool):
            options, unresolved = [], False
            for r in self.band_keys[bi]:
                if self._ring_is_crease(self.rings[r], SHARP_FOLD):
                    continue
                for nb in self.bands_at_ring[r]:
                    if nb == bi:
                        continue
                    if band_kind[nb] is None:
                        unresolved = True
                    else:
                        rank = 0 if (prefer_wall and band_kind[nb] == WALL) else 1
                        options.append((rank, self._ring_dihedral(r), band_kind[nb], r))
                if self.caps_at_ring.get(r):
                    options.append((1, self._ring_dihedral(r), FLAT, r))
            return options, unresolved

        pending = [bi for bi, kind in enumerate(band_kind) if kind is None]
        while pending:
            progressed = False
            for bi in list(pending):
                options, unresolved = options_for(bi, prefer_wall=True)
                if not options:
                    continue
                best = min(options)
                if best[0] != 0 and unresolved:
                    continue  # a trim neighbour may still bring a wall
                _, _, kind, r = best
                band_kind[bi] = kind
                self.joined.add((r, bi))
                pending.remove(bi)
                progressed = True
            if not progressed:
                # Only trim (or crease-locked) neighbours left: settle for the
                # smoothest resolved side, else stand alone as a ring.
                for bi in list(pending):
                    options, _ = options_for(bi, prefer_wall=False)
                    if options:
                        _, _, kind, r = min(options)
                        band_kind[bi] = kind
                        self.joined.add((r, bi))
                    else:
                        band_kind[bi] = FLAT
                    pending.remove(bi)

    def _ring_cuts(self, angle: float, taper_angle: float) -> Set[int]:
        """The ring seams (plus the crease cuts of irregular regions and cap
        interiors)."""
        cuts: Set[int] = set()

        def region(fi: int):
            if fi in self.band_of_face:
                bi = self.band_of_face[fi]
                return ("b", bi), self.band_kind[bi]
            if fi in self.cap_of_face:
                return ("c", self.cap_of_face[fi]), FLAT
            return ("i", fi), None

        for ri, ch in enumerate(self.rings):
            sides, kinds = set(), set()
            for e in ch:
                for fi in self.edge_faces[e]:
                    rid, kind = region(fi)
                    sides.add(rid)
                    kinds.add(kind)
            if len(sides) < 2:
                continue  # rim of an open tube, or inside one region
            if None in kinds or len(sides) > 2:
                # Touches an irregular region, or a junction of 3+ regions:
                # plain crease rule per edge.
                cuts.update(e for e in ch if self._edge_is_crease(e, angle))
                continue
            if any((ri, bi) in self.joined for bi in self.bands_at_ring[ri]):
                continue  # a trim band and the neighbour it joined
            strips = {k in (WALL, CONE) for k in kinds}
            if len(strips) == 2:
                cuts.update(ch)  # a strip meets a flat band: always a seam
                continue
            if True in strips:
                # Two strips: split where the profile turns past the taper
                # tolerance (a chamfer off a cylinder), else one strip (a
                # bent hose, a gentle taper); ``angle`` can only tighten it.
                min_angle = min(angle, taper_angle)
            else:
                # Two flat bands: coaxial flats turn < 30 degrees or fold back
                # past 120, so this only ever splits a fold-back.
                min_angle = angle
            if self._ring_is_crease(ch, min_angle):
                cuts.update(ch)

        # Irregular faces: crease rule on every edge they touch.
        for fi in self.irregular:
            for e in self.face_edges[fi]:
                if e not in cuts and self._edge_is_crease(e, angle):
                    cuts.add(e)
        # Inside a cap (non-quad next to non-quad) only a folded profile splits.
        for cap in self.caps:
            cap_set = set(cap)
            for fi in cap:
                for e in self.face_edges[fi]:
                    fs = self.edge_faces[e]
                    if len(fs) == 2 and all(g in cap_set for g in fs):
                        if self._edge_is_crease(e, max(angle, SHARP_FOLD)):
                            cuts.add(e)
        return cuts

    def _open_strips(self, camera: Optional[Vec], invert_seam: bool) -> None:
        """Group strip bands (walls / cones) into runs across uncut rings and
        open each run with one lengthwise column (added to ``self.cuts``); a
        run closed along its length (a torus) also gets one crossing ring."""
        band_keys, band_kind = self.band_keys, self.band_kind
        parent = list(range(len(band_keys)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # A run is a set of strip bands (walls / cones) connected through
        # uncut rings.
        for ri, bis in self.bands_at_ring.items():
            if self._ring_cut(ri):
                continue
            strips = [b for b in bis if band_kind[b] in (WALL, CONE)]
            for b in strips[1:]:
                ra, rb = find(strips[0]), find(b)
                if ra != rb:
                    parent[ra] = rb
        runs: Dict[int, List[int]] = defaultdict(list)
        for bi in range(len(band_keys)):
            if band_kind[bi] in (WALL, CONE):
                runs[find(bi)].append(bi)
        self.runs = [sorted(r) for r in sorted(runs.values(), key=min)]

        length_edges = {e for e, c in self.edge_class.items() if c == LENGTH}
        columns = self._chains(length_edges)
        col_of = {e: ci for ci, ch in enumerate(columns) for e in ch}
        col_score = self._column_scores(columns, camera, invert_seam)

        for run_bands in self.runs:
            # Only bands closed on both rings need opening; a band with an
            # open ring (a half pipe, the slit end of a scooped tube) already
            # has a lengthwise border, and cutting on through it would split
            # it in two.
            remaining = {
                bi
                for bi in run_bands
                if all(self._chain_closed(self.rings[r]) for r in band_keys[bi])
            }
            if not remaining:
                continue
            # The best (most hidden) column through the run; if it doesn't
            # reach every band (a run through a junction), the next best
            # covers the rest.
            while remaining:
                edges_in = set().union(
                    *(self.band_length_edges[bi] for bi in remaining)
                )
                cols = {col_of[e] for e in edges_in if e in col_of}
                if not cols:
                    break
                best = min(cols, key=lambda c: (col_score[c], c))
                cut_edges = set(columns[best]) & edges_in
                self.cuts.update(cut_edges)
                covered = {
                    bi for bi in remaining if self.band_length_edges[bi] & cut_edges
                }
                if not covered:
                    break
                remaining -= covered
            # No cut ring and no rim anywhere along the run (a torus): one
            # crossing ring lets it unroll.
            run_rings = {r for bi in run_bands for r in band_keys[bi]}
            if not any(
                self._ring_cut(r) or len(self.bands_at_ring[r]) < 2 for r in run_rings
            ):
                self.cuts.update(self.rings[min(run_rings)])

    def _column_scores(self, columns, camera, invert_seam) -> List[float]:
        """Per column, how much it faces the viewer (lower = more hidden).

        The mean, over the column's edges, of the edge normal dotted with the
        direction to the eye. ``camera=None`` places the eye infinitely far
        along :data:`DEFAULT_VIEW_DIR`; ``invert_seam`` negates the score so
        the column facing the viewer wins instead.
        """
        if camera is None:
            far = 1e6 * (1.0 + float(np.abs(self.pts).max()))
            cam = self.pts.mean(axis=0) + np.asarray(DEFAULT_VIEW_DIR) * far
        else:
            cam = np.asarray(camera, dtype=float)
        scores: List[float] = []
        for ch in columns:
            s = 0.0
            for e in ch:
                n = self.normals[self.edge_faces[e]].mean(axis=0)
                to_cam = cam - self.edge_mid(e)
                dist = np.linalg.norm(to_cam)
                if dist > 1e-12:
                    s += float(np.dot(n, to_cam / dist))
            scores.append(s / max(1, len(ch)))
        return [-s for s in scores] if invert_seam else scores

    # ------------------------------------------------------------ seeds
    def seed_uvs(self) -> Dict[int, List[Tuple[float, float]]]:
        """Seed UVs for every face after :meth:`seams` -- ``{face: [(u, v)
        per face-vertex]}``, each shell scaled into a unit box.

        The seed only has to be non-degenerate and un-folded for the unfold
        that follows, but here it is already the developed shape:

        - a **strip** run (walls / cones) is unrolled from its lengthwise
          seam: u = distance around the ring, v = distance along the column
          (per column, so a bent tube's strip is already curved);
        - every other shell of bands / caps (annulus, disc, dome, bead) is
          unrolled radially about its own axis; an irregular region is
          projected onto its best-fit plane.

        Faces sharing a vertex across a cut naturally get separate values,
        so the caller assigns each face-vertex's UV id from this table.
        """
        cuts = self.cuts
        n_faces = len(self.faces)
        # UV shells: faces joined across uncut interior edges.
        parent = list(range(n_faces))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e, fs in self.edge_faces.items():
            if len(fs) == 2 and e not in cuts:
                a, b = find(fs[0]), find(fs[1])
                if a != b:
                    parent[a] = b
        shells: Dict[int, List[int]] = defaultdict(list)
        for f in range(n_faces):
            shells[find(f)].append(f)

        out: Dict[int, List[Tuple[float, float]]] = {}
        done: Set[int] = set()
        for run in self.runs:
            strip = self._unroll_run(run)
            if strip:
                out.update(strip)
                done.update(strip)
        for faces in shells.values():
            rest = [f for f in faces if f not in done]
            if rest:
                # Band / cap shells wrap an axis: unroll their profile radially.
                # Anything else (an irregular region) just gets its plane.
                polar = all(
                    f in self.band_of_face or len(self.faces[f]) != 4 for f in rest
                )
                out.update(self._project_planar(rest, polar=polar))
        # Positive winding, then scale each shell into a unit box (aspect kept).
        for faces in shells.values():
            signed = 0.0
            for f in faces:
                uvs = out[f]
                for i in range(len(uvs)):
                    (u1, v1), (u2, v2) = uvs[i], uvs[(i + 1) % len(uvs)]
                    signed += u1 * v2 - u2 * v1
            if signed < 0:
                for f in faces:
                    out[f] = [(-u, v) for u, v in out[f]]
            us = [uv[0] for f in faces for uv in out[f]]
            vs = [uv[1] for f in faces for uv in out[f]]
            u0, v0 = min(us), min(vs)
            span = max(max(us) - u0, max(vs) - v0) or 1.0
            for f in faces:
                out[f] = [((u - u0) / span, (v - v0) / span) for u, v in out[f]]
        return out

    def _project_planar(
        self, faces: List[int], polar: bool = False
    ) -> Dict[int, List[Tuple[float, float]]]:
        """Coordinates of the faces' vertices in their best-fit plane.

        With ``polar`` the shell is read as a turned profile about the plane
        normal and unrolled radially: each vertex keeps its angle, and its
        radius is its in-plane radius plus its axial distance from the
        shell's innermost point. A washer is unchanged, a bead ring or a
        chamfer opens into an annulus of its true width, and a domed cap
        spreads into a disc -- none of which a flat projection can do
        without collapsing or folding.
        """
        vids = sorted({v for f in faces for v in self.faces[f]})
        p = self.pts[vids]
        c = p.mean(axis=0)
        if len(vids) >= 3:
            _, vecs = np.linalg.eigh((p - c).T @ (p - c))
            u_axis, v_axis, n_axis = vecs[:, 2], vecs[:, 1], vecs[:, 0]
        else:
            u_axis, v_axis, n_axis = np.eye(3)
        rel = self.pts - c
        x = rel @ u_axis
        y = rel @ v_axis
        if not polar:
            return {
                f: [(float(x[v]), float(y[v])) for v in self.faces[f]] for f in faces
            }
        z = rel @ n_axis
        rho = np.hypot(x, y)
        inner = min(vids, key=lambda v: rho[v])
        radius = rho + np.abs(z - z[inner])
        theta = np.arctan2(y, x)
        return {
            f: [
                (
                    float(radius[v] * math.cos(theta[v])),
                    float(radius[v] * math.sin(theta[v])),
                )
                for v in self.faces[f]
            ]
            for f in faces
        }

    def _unroll_run(self, run: List[int]) -> Dict[int, List[Tuple[float, float]]]:
        """Developed (u = around, v = along) coordinates for a strip run.

        Walks the run band by band from one end, and each band face by face
        from the seam (or the open side), so faces on either side of the seam
        get 0 and the full circumference respectively. Returns ``{}`` when
        the run isn't a clean quad strip (the caller then projects it).
        """
        cuts = self.cuts
        band_keys = self.band_keys
        bands = self.bands
        # Order the run's bands along the tube: rings shared by two of its
        # bands link them into a path (or a cycle for a torus).
        at_ring: Dict[int, List[int]] = defaultdict(list)
        for bi in run:
            for r in band_keys[bi]:
                at_ring[r].append(bi)
        ends = [bi for bi in run if any(len(at_ring[r]) == 1 for r in band_keys[bi])]
        if ends:
            order = [ends[0]]
            # First ring: the one not shared with a run neighbour.
            r_prev = next(r for r in band_keys[order[0]] if len(at_ring[r]) == 1)
        else:
            # A cycle (torus): start at its crossing cut ring so the strip's
            # two ends land there.
            cut_rings = [r for r in at_ring if self._ring_cut(r)]
            r_prev = cut_rings[0] if cut_rings else band_keys[run[0]][0]
            order = [at_ring[r_prev][0]]
        seen = {order[0]}
        ring_seq = [r_prev]
        while True:
            bi = order[-1]
            r1, r2 = band_keys[bi]
            r_next = r2 if r1 == r_prev else r1
            ring_seq.append(r_next)
            nxt = [b for b in at_ring[r_next] if b != bi and b not in seen]
            if not nxt:
                break
            order.append(nxt[0])
            seen.add(nxt[0])
            r_prev = r_next
        if len(order) != len(run):
            return {}

        def face_edges_of(fi, ring):
            """(ring edge on `ring`, lengthwise edges) of a band face."""
            ring_set = set(self.rings[ring])
            r_edge = [e for e in self.face_edges[fi] if e in ring_set]
            l_edges = [
                e for e in self.face_edges[fi] if self.edge_class.get(e) == LENGTH
            ]
            return (r_edge[0] if r_edge else None), l_edges

        def walk_band(bi: int, first: Optional[int], ring_a: int):
            """Faces of band `bi` in order around the tube from `first`
            (or from a face at a cut / boundary lengthwise edge)."""
            faces = bands[band_keys[bi]]
            fset = set(faces)
            if first is None or first not in fset:
                for fi in faces:
                    _, ls = face_edges_of(fi, ring_a)
                    if any(e in cuts or e in self.boundary for e in ls):
                        first = fi
                        break
                else:
                    first = faces[0]
            walk = [first]
            prev_edge = None
            # Enter through the cut / boundary lengthwise edge if there is one,
            # so the walk goes the other way round.
            _, ls = face_edges_of(first, ring_a)
            blocked = [e for e in ls if e in cuts or e in self.boundary]
            if blocked:
                prev_edge = blocked[0]
            while True:
                fi = walk[-1]
                _, ls = face_edges_of(fi, ring_a)
                step = [e for e in ls if e != prev_edge]
                if not step:
                    break
                e = step[0]
                if e in cuts or e in self.boundary:
                    break
                nxt = [g for g in self.edge_faces[e] if g != fi and g in fset]
                if not nxt or nxt[0] in walk:
                    break
                walk.append(nxt[0])
                prev_edge = e
            return walk

        out: Dict[int, List[Tuple[float, float]]] = {}
        along: Dict[int, float] = {}  # column position (index in walk) -> v at ring_a
        first_face: Optional[int] = None
        for k, bi in enumerate(order):
            ring_a, ring_b = ring_seq[k], ring_seq[k + 1]
            walk = walk_band(bi, first_face, ring_a)
            if len(walk) != len(bands[band_keys[bi]]):
                return {}

            # Around: cumulative ring-edge length on each ring, per face index.
            def cum(ring):
                acc, tot = [0.0], 0.0
                for fi in walk:
                    r_e, _ = face_edges_of(fi, ring)
                    tot += (
                        float(np.linalg.norm(self.edge_vec(r_e)))
                        if r_e is not None
                        else 0.0
                    )
                    acc.append(tot)
                return acc

            cum_a, cum_b = cum(ring_a), cum(ring_b)
            ring_a_verts = set(self._chain_verts(self.rings[ring_a]))
            new_along: Dict[int, float] = {}
            for i, fi in enumerate(walk):
                _, ls = face_edges_of(fi, ring_a)
                # Which lengthwise edge is on the walk's start side (index i)?
                # The one shared with the previous face, else the entry edge.
                if i > 0:
                    shared = set(ls) & set(face_edges_of(walk[i - 1], ring_a)[1])
                    e_i = next(iter(shared))
                else:
                    e_i = next(
                        (e for e in ls if e in cuts or e in self.boundary), ls[0]
                    )
                e_i1 = ls[0] if ls[1] == e_i else ls[1]
                len_i = float(np.linalg.norm(self.edge_vec(e_i)))
                len_i1 = float(np.linalg.norm(self.edge_vec(e_i1)))
                v_i = along.get(i, 0.0)
                v_i1 = along.get(i + 1, 0.0)
                new_along[i] = v_i + len_i
                new_along[i + 1] = v_i1 + len_i1
                col_of_vert = {}
                for e, idx in ((e_i, i), (e_i1, i + 1)):
                    for v in self.edges[e]:
                        col_of_vert[v] = idx
                uvs = []
                for v in self.faces[fi]:
                    idx = col_of_vert.get(v)
                    if idx is None:
                        return {}
                    on_a = v in ring_a_verts
                    u = (cum_a if on_a else cum_b)[idx]
                    vv = (v_i if idx == i else v_i1) if on_a else new_along[idx]
                    uvs.append((u, vv))
                out[fi] = uvs
            along = new_along
            # Next band starts from the face across the ring edge of this
            # band's first face.
            r_e, _ = face_edges_of(walk[0], ring_b)
            nxt = (
                [g for g in self.edge_faces.get(r_e, []) if g != walk[0]]
                if r_e is not None
                else []
            )
            first_face = nxt[0] if nxt else None
        return out
