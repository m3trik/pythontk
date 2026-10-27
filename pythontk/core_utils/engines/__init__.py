# !/usr/bin/python
# coding=utf-8
"""Host-agnostic *domain engines* — complete application cores.

This namespace holds pure, DCC-agnostic cores for whole application domains:
a data model, a planning/algorithm layer, and dependency-injection seams
(Protocols, injected writer callables, overridable hooks) that let downstream
adapter packages (mayatk, blendertk, …) drive them against a real scene.

**How this differs from the ``*_utils`` packages.** The ``*_utils`` packages
promise *generality* — a helper there should plausibly be useful to any Python
program, and is placed by the data type it operates on (strings, images, math).
An engine here makes no such promise: it is domain-specific by design (a shot
timeline, a manifest importer). What it shares with ``*_utils`` is the same set
of **hard rules** — no DCC imports (``maya``, ``bpy``, Qt), zero-dependency
preferred, no knowledge of any specific consumer. The generality requirement is
relaxed; the agnosticism requirement is not.

**Why it lived here.** Two downstream packages that cannot import each other
(mayatk, blendertk) needed one shared source of truth for a domain's model and
math. Under the split-by-feature layout (CODE_STANDARD section 14) that home is
tentacle's ``core/<engine>/``: this namespace takes no new tenant, and each
engine below leaves for tentacle once its last mayatk and blendertk users have
moved, keeping only its general pieces and the ``SceneRecords`` contract.

**Placement decision rule.** First **decompose** the logic into
data-type-general primitives and place them in the appropriate ``*_utils``
package (the photogrammetry-ingest precedent) whenever a piece is genuinely
reusable on its own. A domain core several tools share goes to tentacle's
``core/<engine>/``, never here.

**Tenants.** ``shots/`` (timeline model + planner + manifest), ``scene_export/``
(scene records, the export snapshot and profile, the hierarchy baseline, the
exporter and sidecar shells both DCC Scene Exporters subclass),
``instancing/`` (separated-part clustering), ``textures/`` (PBR map taxonomy +
prep + packaging), ``rig_graph/`` (rig-intent model + capability negotiation +
planner), ``key_stash/`` (parked-keyframe clip store).

Engines are lazy-loaded via the pythontk root package; import from pythontk
directly (e.g. ``from pythontk import ShotStore, ShotPlanner``).
"""
