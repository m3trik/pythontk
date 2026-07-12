# !/usr/bin/python
# coding=utf-8
"""Auto-instancing engine — DCC-agnostic separated-part clustering core.

The pure half of the ecosystem's auto-instancer:
:class:`~pythontk.core_utils.engines.instancing.assembly_sorter.AssemblySorter`
consumes plain per-part feature dicts (bbox / topology / area / center / volume /
material — the contract is documented on the class) and returns a *plan* of
index-groups: which separated mesh parts are repeated copies of one assembly, so
a DCC can replace the duplicates with instances. It never touches a scene — the
DCC adapters (mayatk / blendertk ``auto_instancer``) extract the feature dicts and
apply the returned grouping — so both toolkits share one clustering source of
truth without importing each other.

**Why only the sorter lives here.** The auto-instancing domain was already
*decomposed* per the charter's first rule: the general point-set math (PCA
alignment, nearest-neighbour, proximity clustering, positional hashing) is a
data-type primitive and stays as :class:`~pythontk.geo_utils.pointcloud.PointCloud`;
what remains — the multi-pass separated-part clustering pipeline that is
domain-specific and cannot be genericised — is the irreducible core that belongs
here.

Lazy-loaded via the pythontk root package; import from pythontk directly
(``from pythontk import AssemblySorter``).
"""

# Lazy-loaded via parent package - no explicit imports needed
