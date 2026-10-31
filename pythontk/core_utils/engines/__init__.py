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

**Why it lives here.** Two downstream packages that cannot import each other
(mayatk, blendertk) need one source of truth for a domain's model and math.
CODE_STANDARD section 14 puts each piece of logic at the lowest layer whose
dependencies it needs, and host-free logic both hosts share is pythontk's: a
domain engine is a complete, pure core (model + planning/algorithm layer)
behind an adapter seam (a Protocol, injected callables) each host implements.
The host halves of an engine are mayatk's and blendertk's, mirrored; its panels
and Slots classes are tentacle's, which holds only the launcher, the UIs and the
Slots and is imported by no library, so no engine ever moves up there.

**Placement decision rule.** First **decompose** the logic into
data-type-general primitives and place them in the appropriate ``*_utils``
package (the photogrammetry-ingest precedent) whenever a piece is genuinely
reusable on its own. What remains, a domain core both hosts drive through an
adapter seam, is a tenant here. The bar is *model + algorithm/planner*, not
merely "used by both DCCs": a single tool's domain math goes to the ``*_utils``
home of its data type, beside the general primitive it composes
(``geo_utils.CurtainDrape`` over ``RailSurface``), and a general orchestration
base stays in ``core_utils`` (``handoff/``). One application's engine
(Marmoset, Painter, Unity vocabulary) never comes here.

**Tenants.** ``shots/`` (timeline model + planner + manifest), ``scene_export/``
(scene records, the export snapshot and profile, the hierarchy baseline, the
exporter and sidecar shells both DCC Scene Exporters bind to their host),
``instancing/`` (separated-part clustering), ``textures/`` (PBR map taxonomy +
prep + packaging, emissive groups), ``rig_graph/`` (rig-intent model +
capability negotiation + planner), ``key_stash/`` (parked-keyframe clip store),
``lighting/`` (baked light layers, mix prediction, look match, the light rig
payload).

Engines are lazy-loaded via the pythontk root package; import from pythontk
directly (e.g. ``from pythontk import ShotStore, ShotPlanner``).
"""
