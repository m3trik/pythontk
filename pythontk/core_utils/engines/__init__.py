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
(mayatk, blendertk) need one shared source of truth for a domain's model and
math. Duplicating it downstream is a DRY/SSoT violation; pushing a
non-general document type into ``*_utils`` erodes that layer's charter. A
charter'd engine namespace is the honest home: pure and agnostic, but openly
domain-shaped.

**Placement decision rule.**

1. First try to **decompose** the logic into data-type-general primitives and
   place them in the appropriate ``*_utils`` package (the photogrammetry-ingest
   precedent). Prefer this whenever a piece is genuinely reusable on its own.
2. An **irreducible shared domain core** — needed by two-or-more downstream
   packages that cannot import one another — goes in ``core_utils/engines/<system>/``.
3. If an engine ever **outgrows the package** (size, or it needs a real
   dependency), graduate it to its own distribution. The namespace already
   marks the extraction boundary, so that move is mechanical.

Engines are lazy-loaded via the pythontk root package; import from pythontk
directly (e.g. ``from pythontk import ShotStore, plan_respace``).
"""
