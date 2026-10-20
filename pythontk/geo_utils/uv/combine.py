# !/usr/bin/python
# coding=utf-8
"""Several texture sets' UV islands packed into ONE atlas (arrays in -> arrays out).

The planning half of a texture combine: each source texture set (a material)
brings its islands, laid out in that set's own 0-1 map; this decides how big
each island is in the shared atlas, packs them (:class:`UvPack`), and returns
every island's new coordinates. The texel remap that follows -- reading each
source map through the old layout into the new one -- is
:class:`UvTransfer`'s, and the host owns extraction and write-back.

Three decisions shape the atlas:

- **Stacks pack once.** Islands of one set that OVERLAP in its map share those
  texels on purpose -- a wheel used four times, mirrored halves, the copies a
  kit-bash reuses. They are grouped (:meth:`UvCombine.clusters`) and each group
  moves as one rigid piece, so the atlas never pays twice for texture space the
  source shared. Islands of different sets never group: their overlap is two
  maps, not one.
- **Sizing** (:attr:`UvCombine.SIZING`): ``"density"`` gives every group the
  same world-space texel density; ``"proportional"`` keeps each set's islands
  at the sizes they have relative to each other in their own map -- the
  texel-density decisions the artist made there survive -- and scales each set
  as a whole. How the sets are then sized against each other is
  :attr:`UvCombine.NORMALIZE`: by each set's average world texel density
  (``"density"``), or by source map resolution (``"resolution"``: a 4K set's
  islands twice the linear size of a 2K set's, one source texel per atlas
  texel throughout).
- **Scale overrides**: every island carries a multiplier
  (:attr:`CombineShell.scale`) applied on top -- the host resolves which of its
  rows (a set, a mesh, a hand-picked island) sets it.

Numbers come out per set (:class:`CombineGroup`): its share of the packed
area, and the linear scale from its source map into the atlas -- times the
atlas size, the resolution the source set effectively keeps
(:meth:`CombineGroup.resolution`). :meth:`UvCombine.plan` estimates them
without packing (cheap enough to follow every edit in a UI);
:meth:`UvCombine.pack` measures them.
"""

from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional, Sequence

from pythontk.core_utils.help_mixin import HelpMixin
from pythontk.geo_utils.uv.pack import UvPack

#: Max abs error (UV units) of a cluster's rigid move. A rigid move fits to
#: ~2e-4 at most (mayatk's measured pack write-back calibration); an island the
#: engine split into two charts misses by 1e-1 or more.
RESIDUAL_TOLERANCE = 1e-3


@dataclass
class CombineShell:
    """One UV island of one source texture set.

    Parameters:
        key: The host's identity for the island (a ``(mesh, shell id)``),
            carried through untouched.
        group: The source texture set it reads (a material name).
        uvs: ``(N, 2)`` coordinates in the set's own map.
        triangles: ``(M, 3)`` indices into *uvs*.
        area_3d: Its world-space surface area (any consistent unit).
        scale: The override multiplier on its linear size (1.0 = none).
    """

    key: Any
    group: Hashable
    uvs: Any
    triangles: Any
    area_3d: float
    scale: float = 1.0


@dataclass
class CombineGroup:
    """What one source texture set gets in the atlas.

    ``coverage`` is the area its islands cover in their own map (stacks counted
    once), ``share`` its fraction of the atlas's packed area, ``scale`` the
    linear factor from its source map into the atlas (area-weighted over its
    islands; one value throughout under ``"proportional"`` sizing).
    """

    group: Hashable
    shells: int = 0
    stacks: int = 0
    coverage: float = 0.0
    share: float = 0.0
    scale: float = 0.0

    def resolution(self, atlas_size: int) -> float:
        """The resolution the set effectively keeps at *atlas_size*: its source
        map would need this many texels across to hold the detail it keeps."""
        return float(atlas_size) * self.scale


@dataclass
class CombinePlan:
    """How the atlas is sized: the island groups and each one's scale.

    ``clusters`` lists, per rigid group, the input shell indices; ``scales``
    the relative linear scale of each (the pack applies one global factor
    on top); ``groups`` the per-set numbers, estimated at fill ``fill`` by
    :meth:`UvCombine.plan`, measured by :meth:`UvCombine.pack`.
    """

    clusters: List[List[int]]
    scales: Any
    groups: Dict[Hashable, CombineGroup]
    fill: float


@dataclass
class CombineLayout:
    """A packed atlas: per input shell its new coordinates (``uvs``, aligned
    with the input), per cluster its rigid move (``transforms``, a
    :meth:`UvPack.solve_similarity` tuple from the source map), the measured
    :class:`CombinePlan` and the engine's fill."""

    uvs: List[Any]
    transforms: List[tuple]
    plan: CombinePlan
    utilization: float = 0.0


class UvCombine(HelpMixin):
    """Plan and pack several texture sets' islands into one atlas.

    Mechanism only: :class:`CombineShell` lists in, plain arrays out. A DCC
    adapter reads each mesh's arrays (:meth:`island_shells` turns them into
    shells, one per UV island, tagged with the material it reads) and writes
    the result back.
    """

    #: How islands are sized: equal world texel density, or each set's own
    #: proportions kept (see the module docstring).
    SIZING = ("density", "proportional")
    #: How ``"proportional"`` sets are sized against each other.
    NORMALIZE = ("density", "resolution")
    #: The fill :meth:`plan` assumes for an estimate -- a typical tight pack of
    #: mixed hard-surface islands (measured 0.57-0.72 in :class:`UvPack`).
    ESTIMATE_FILL = 0.7

    # ---------------------------------------------------------------- geometry
    @staticmethod
    def _arrays(shell: CombineShell):
        import numpy as np

        uvs = np.asarray(shell.uvs, dtype=np.float64).reshape(-1, 2)
        tris = np.asarray(shell.triangles, dtype=np.int64).reshape(-1, 3)
        return uvs, tris

    #: Barycentric weights of the probe points per triangle: its centroid and
    #: one point near each corner, so an island laid a third of a triangle
    #: over another is seen, not only one stacked centroid-on-centroid.
    _PROBE_WEIGHTS = (
        (1 / 3, 1 / 3, 1 / 3),
        (2 / 3, 1 / 6, 1 / 6),
        (1 / 6, 2 / 3, 1 / 6),
        (1 / 6, 1 / 6, 2 / 3),
    )

    @classmethod
    def _samples(cls, uvs, tris):
        """Probe points strictly inside each triangle (:attr:`_PROBE_WEIGHTS`)."""
        import numpy as np

        corners = uvs[tris]  # (M, 3, 2)
        weights = np.asarray(cls._PROBE_WEIGHTS)  # (4, 3)
        return np.einsum("wk,mkd->mwd", weights, corners).reshape(-1, 2)

    @staticmethod
    def _inside(points, uvs, tris, eps: float = 1e-7):
        """Per point: strictly inside any of the triangles (chunked)."""
        import numpy as np

        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        out = np.zeros(len(points), dtype=bool)
        if not len(points) or not len(tris):
            return out
        tri = uvs[tris]
        v0 = tri[:, 0]
        e1 = tri[:, 1] - v0
        e2 = tri[:, 2] - v0
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        keep = np.abs(det) > 1e-18
        if not keep.any():
            return out
        v0, e1, e2, det = v0[keep], e1[keep], e2[keep], det[keep]
        step = max(1, int(2_000_000 // len(det)))
        for start in range(0, len(points), step):
            w = points[start : start + step, None, :] - v0[None]
            a = (w[..., 0] * e2[:, 1] - w[..., 1] * e2[:, 0]) / det
            b = (e1[:, 0] * w[..., 1] - e1[:, 1] * w[..., 0]) / det
            hit = (a > eps) & (b > eps) & (a + b < 1.0 - eps)
            out[start : start + step] = hit.any(axis=1)
        return out

    @staticmethod
    def _convex_hull(points):
        """``(H, 2)`` counter-clockwise hull of *points* (monotone chain)."""
        import numpy as np

        pts = np.unique(np.round(np.asarray(points, dtype=np.float64), 12), axis=0)
        if len(pts) < 3:
            return pts

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        rows = [tuple(p) for p in pts]
        lower: List[tuple] = []
        for p in rows:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper: List[tuple] = []
        for p in reversed(rows):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)

    @classmethod
    def _hull_proxy(cls, points):
        """A one-chart stand-in covering *points*: their hull, fan-triangulated."""
        import numpy as np

        hull = cls._convex_hull(points)
        if len(hull) < 3:
            raise ValueError("a cluster with no UV area cannot be packed")
        fan = [(0, i, i + 1) for i in range(1, len(hull) - 1)]
        return hull, np.asarray(fan, dtype=np.int64)

    # ---------------------------------------------------------------- islands
    @staticmethod
    def island_shells(
        key: Any,
        uvs,
        shell_ids,
        tri_uv,
        tri_face,
        tri_area,
        materials: Sequence[Hashable],
        face_material,
    ) -> tuple:
        """One mesh's UV islands as :class:`CombineShell` -- the host's gather.

        Each island reads the material covering most of its world surface (an
        area-weighted vote over its triangles; ``None`` when no face of it
        wears one).

        Parameters:
            key: The host's identity for the mesh; each island's key is
                ``(key, island id)``.
            uvs: ``(N, 2)`` the mesh's UV points.
            shell_ids: ``(N,)`` the island of each point.
            tri_uv: ``(M, 3)`` the points of each triangle.
            tri_face: ``(M,)`` the face each triangle lies in.
            tri_area: ``(M,)`` each triangle's world-space area.
            materials: The mesh's materials; *face_material* indexes them.
            face_material: Per face, an index into *materials* (``-1``: none).

        Returns:
            ``(shells, faces, points)`` -- aligned lists, ordered by island
            id: each island's :class:`CombineShell`, the sorted faces it
            covers, and the indices into *uvs* its own ``uvs`` were taken from
            (where its packed coordinates are written back).
        """
        import numpy as np

        uvs = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
        shell_ids = np.asarray(shell_ids, dtype=np.int64).reshape(-1)
        tri_uv = np.asarray(tri_uv, dtype=np.int64).reshape(-1, 3)
        tri_face = np.asarray(tri_face, dtype=np.int64).reshape(-1)
        tri_area = np.asarray(tri_area, dtype=np.float64).reshape(-1)
        face_material = np.asarray(face_material, dtype=np.int64).reshape(-1)
        shells: List[CombineShell] = []
        faces: List[Any] = []
        points: List[Any] = []
        tri_shell = shell_ids[tri_uv[:, 0]]
        # One stable sort groups every island's triangles in their own order:
        # a mask per island re-scanned the whole mesh once per island.
        order = np.argsort(tri_shell, kind="stable")
        ids, starts = np.unique(tri_shell[order], return_index=True)
        remap = np.full(len(uvs), -1, dtype=np.int64)
        for sid, rows in zip(ids.tolist(), np.split(order, starts[1:])):
            t_uv = tri_uv[rows]
            used = np.unique(t_uv)
            remap[used] = np.arange(len(used))
            fm = face_material[tri_face[rows]]
            shaded = fm >= 0
            group = None
            if shaded.any():
                weights = np.bincount(fm[shaded], weights=tri_area[rows][shaded])
                group = materials[int(np.argmax(weights))]
            shells.append(
                CombineShell(
                    key=(key, int(sid)),
                    group=group,
                    uvs=uvs[used],
                    triangles=remap[t_uv],
                    area_3d=float(tri_area[rows].sum()),
                )
            )
            faces.append(np.unique(tri_face[rows]))
            points.append(used)
        return shells, faces, points

    # ---------------------------------------------------------------- clusters
    @classmethod
    def clusters(cls, shells: Sequence[CombineShell]) -> List[List[int]]:
        """The rigid groups: islands of one set whose UVs overlap, transitively.

        Two islands overlap when a probe point of either (each triangle's
        centroid and a point near each corner) lies strictly inside the other
        -- so a stacked copy, a mirrored half on its twin, and an island laid
        partly over another all group, while islands that only touch along an
        edge do not. Islands of different sets never group. A probe can miss a
        sliver of overlap; that costs atlas space (the two pack apart, each
        with its own texels), never correctness.

        Returns:
            One list of input indices per group, every input in exactly one,
            ordered by first member.
        """
        import numpy as np

        arrays = [cls._arrays(s) for s in shells]
        parent = list(range(len(shells)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        by_group: Dict[Hashable, List[int]] = {}
        for i, shell in enumerate(shells):
            by_group.setdefault(shell.group, []).append(i)
        tol = 1e-6
        for members in by_group.values():
            if len(members) < 2:
                continue
            lo = np.array([arrays[i][0].min(axis=0) for i in members])
            hi = np.array([arrays[i][0].max(axis=0) for i in members])
            probes = {i: cls._samples(*arrays[i]) for i in members}
            for a_pos, a in enumerate(members):
                rest = np.arange(a_pos + 1, len(members))
                if not len(rest):
                    break
                near = rest[
                    np.all(lo[rest] < hi[a_pos] - tol, axis=1)
                    & np.all(lo[a_pos] < hi[rest] - tol, axis=1)
                ]
                for b_pos in near.tolist():
                    b = members[b_pos]
                    if find(a) == find(b):
                        continue
                    if (
                        cls._inside(probes[a], *arrays[b]).any()
                        or cls._inside(probes[b], *arrays[a]).any()
                    ):
                        parent[find(b)] = find(a)
        groups: Dict[int, List[int]] = {}
        for i in range(len(shells)):
            groups.setdefault(find(i), []).append(i)
        return sorted(groups.values(), key=lambda m: m[0])

    @classmethod
    def _proxy(cls, shells, members: Sequence[int], hull: bool = False):
        """``(uvs, triangles)`` the packer moves for a cluster.

        One island packs as itself; a stack whose copies all lie inside its
        largest member packs as that member (tight, concavities intact);
        anything else -- or a cluster whose own island the engine could not
        move as one piece (*hull*) -- packs as the hull of all of them.
        """
        import numpy as np

        arrays = [cls._arrays(shells[i]) for i in members]
        if not hull:
            if len(members) == 1:
                return arrays[0]
            areas = [UvPack._uv_area(u, t) for u, t in arrays]
            base = int(np.argmax(areas))
            b_uvs, b_tris = arrays[base]
            if all(
                cls._inside(cls._samples(u, t), b_uvs, b_tris).all()
                for k, (u, t) in enumerate(arrays)
                if k != base
            ):
                return b_uvs, b_tris
        return cls._hull_proxy(np.vstack([u for u, _ in arrays]))

    # ---------------------------------------------------------------- planning
    @classmethod
    def plan(
        cls,
        shells: Sequence[CombineShell],
        *,
        sizing: str = "density",
        normalize: str = "density",
        resolutions: Optional[Dict[Hashable, int]] = None,
        fill: Optional[float] = None,
        clusters: Optional[List[List[int]]] = None,
    ) -> CombinePlan:
        """Size every island group, and estimate what each set gets.

        Parameters:
            shells: The islands, every set's.
            sizing: One of :attr:`SIZING`.
            normalize: One of :attr:`NORMALIZE` -- read by ``"proportional"``
                sizing only.
            resolutions: ``{group: source map size}``, which
                ``normalize="resolution"`` sizes by (a set missing from it
                takes the largest given).
            fill: The packed fill the estimate assumes; default
                :attr:`ESTIMATE_FILL`.
            clusters: Precomputed :meth:`clusters` (they depend on the UVs
                alone, so a UI re-planning after a scale edit reuses them).

        Returns:
            :class:`CombinePlan` -- estimated numbers, nothing packed.

        Raises:
            ValueError: An unknown *sizing* / *normalize*, or no shells.
        """
        import numpy as np

        if sizing not in cls.SIZING:
            raise ValueError(f"sizing must be one of {cls.SIZING}, not {sizing!r}")
        if normalize not in cls.NORMALIZE:
            raise ValueError(
                f"normalize must be one of {cls.NORMALIZE}, not {normalize!r}"
            )
        if not shells:
            raise ValueError("no shells to combine")
        clusters = clusters if clusters is not None else cls.clusters(shells)
        arrays = [cls._arrays(s) for s in shells]
        uv_area = np.array([UvPack._uv_area(u, t) for u, t in arrays])
        area_3d = np.array([max(float(s.area_3d), 0.0) for s in shells])

        group_of = [shells[m[0]].group for m in clusters]
        c_uv = np.array([uv_area[m].sum() for m in clusters])
        c_3d = np.array([area_3d[m].sum() for m in clusters])
        # A stack's footprint is ONE copy: the largest member stands for it.
        foot = np.array([uv_area[m].max() for m in clusters])
        mult = np.array([max(float(shells[i].scale) for i in m) for m in clusters])

        def density(a3, auv):
            ok = (a3 > 0) & (auv > 0)
            ratio = np.full(len(a3), np.nan)
            ratio[ok] = np.sqrt(a3[ok] / auv[ok])
            valid = ratio[~np.isnan(ratio)]
            return np.where(
                np.isnan(ratio), np.median(valid) if len(valid) else 1.0, ratio
            )

        if sizing == "density":
            k = density(c_3d, c_uv)
        else:
            groups = list(dict.fromkeys(group_of))
            if normalize == "resolution":
                given = {g: float(r) for g, r in (resolutions or {}).items() if r}
                fallback = max(given.values()) if given else 1.0
                per_group = {g: given.get(g, fallback) for g in groups}
            else:
                g_3d = np.array(
                    [c_3d[[g == x for x in group_of]].sum() for g in groups]
                )
                g_uv = np.array(
                    [c_uv[[g == x for x in group_of]].sum() for g in groups]
                )
                per_group = dict(zip(groups, density(g_3d, g_uv)))
            k = np.array([per_group[g] for g in group_of], dtype=np.float64)
        k = k * mult

        fill = cls.ESTIMATE_FILL if fill is None else float(fill)
        weight = k * k * foot
        total = float(weight.sum()) or 1.0
        g_scale = (fill / total) ** 0.5
        return CombinePlan(
            clusters=clusters,
            scales=k,
            groups=cls._group_stats(shells, clusters, foot, k * g_scale),
            fill=fill,
        )

    @staticmethod
    def _group_stats(shells, clusters, foot, linear) -> Dict[Hashable, CombineGroup]:
        """Per-set numbers from each cluster's footprint and atlas scale."""
        stats: Dict[Hashable, CombineGroup] = {}
        packed = foot * linear * linear
        total = float(packed.sum()) or 1.0
        weighted: Dict[Hashable, float] = {}
        for c, members in enumerate(clusters):
            group = shells[members[0]].group
            entry = stats.setdefault(group, CombineGroup(group=group))
            entry.shells += len(members)
            entry.stacks += int(len(members) > 1)
            entry.coverage += float(foot[c])
            entry.share += float(packed[c]) / total
            weighted[group] = weighted.get(group, 0.0) + float(foot[c] * linear[c])
        for group, entry in stats.items():
            entry.scale = weighted[group] / entry.coverage if entry.coverage else 0.0
        return stats

    # ---------------------------------------------------------------- packing
    @classmethod
    def pack(
        cls,
        shells: Sequence[CombineShell],
        plan: Optional[CombinePlan] = None,
        *,
        map_size: int = 2048,
        padding: int = 8,
        margin: float = 0.0,
        rotate: bool = True,
        brute_force: bool = False,
        **plan_kwargs,
    ) -> CombineLayout:
        """Pack every island into one square atlas at the planned sizes.

        Each cluster goes to the engine as one island (:meth:`_proxy`),
        pre-scaled by its planned scale; the engine applies one global scale
        on top, so the planned proportions come out exactly. A cluster's move
        is solved from its stand-in (:meth:`UvPack.solve_similarity`) and
        applied to every member, so a stack stays stacked. A stand-in the
        engine could not move as one piece (an island pinched to itself at a
        single UV point is two charts to it) re-packs as its hull.

        Parameters:
            shells: The islands.
            plan: A :meth:`plan` result; default plans with *plan_kwargs*.
            map_size: The atlas size in texels (the engine's page).
            padding: Gutter between islands, in atlas texels.
            margin: Inset of the packed area from the atlas border, in UV
                units (``0.002`` = about 8 texels at 4K).
            rotate: Let the engine turn islands where that packs tighter.
            brute_force: See :meth:`UvPack.pack_islands`.

        Returns:
            :class:`CombineLayout` with measured numbers.

        Raises:
            RuntimeError: xatlas missing, or a cluster the engine split even
                as a hull.
        """
        import numpy as np

        plan = plan or cls.plan(shells, **plan_kwargs)
        hulls = set()
        for _attempt in range(3):
            proxies = [
                cls._proxy(shells, m, hull=c in hulls)
                for c, m in enumerate(plan.clusters)
            ]
            result = UvPack.pack_islands(
                [(u * float(k), t) for (u, t), k in zip(proxies, plan.scales)],
                padding=int(padding),
                rotate=rotate,
                brute_force=brute_force,
                resolution=int(map_size),
                pages=1,
            )
            fit = 1.0 - 2.0 * float(margin)
            transforms, failed = [], set()
            for c, ((p_uvs, _), packed, written) in enumerate(
                zip(proxies, result.uvs, result.written)
            ):
                final = np.asarray(packed) * fit + float(margin)
                rows = np.asarray(written)
                move = UvPack.solve_similarity(p_uvs[rows], final[rows])
                if move[5] > RESIDUAL_TOLERANCE:
                    failed.add(c)
                transforms.append(move)
            if not failed:
                break
            if failed <= hulls:
                raise RuntimeError(
                    f"{len(failed)} island group(s) could not be moved as one "
                    "piece even as their hull"
                )
            hulls |= failed
        else:  # pragma: no cover - three attempts always settle or raise
            raise RuntimeError("the pack did not settle")

        uvs: List[Any] = [None] * len(shells)
        for members, move in zip(plan.clusters, transforms):
            for i in members:
                uvs[i] = UvPack.apply_similarity(cls._arrays(shells[i])[0], move)
        foot = np.array(
            [
                max(UvPack._uv_area(*cls._arrays(shells[i])) for i in m)
                for m in plan.clusters
            ]
        )
        linear = np.array([move[0] for move in transforms])
        measured = CombinePlan(
            clusters=plan.clusters,
            scales=plan.scales,
            groups=cls._group_stats(shells, plan.clusters, foot, linear),
            fill=float(result.utilization),
        )
        return CombineLayout(
            uvs=uvs,
            transforms=transforms,
            plan=measured,
            utilization=float(result.utilization),
        )
