# !/usr/bin/python
# coding=utf-8
"""PBR texture engine — DCC-agnostic map taxonomy, preparation, and packaging.

A complete, pure application core for PBR texture-map work — the SSoT both DCC
material toolkits (mayatk, blendertk) and the standalone ``extapps`` texture tools
drive so they classify and process maps *identically* without importing each
other:

- ``map_registry`` — the **domain model**: the PBR map-type taxonomy, per-target
  workflow presets (Unity / UE / glTF / Godot), and fallback / precedence /
  alias-resolution rules. Depends on nothing but ``core_utils`` — the clean core.
- ``map_factory`` — the **planner + strategies**: a ``ConversionRegistry``
  (pluggable strategy registry), a ``TextureProcessor`` (injected processing
  context), ``WorkflowHandler`` strategies, and the ``MapFactory.prepare_maps``
  orchestrator.
- ``map_optimizer`` — a pure **plan → apply** optimizer (``MapOptimizer``) with a
  read-only ``assess`` twin, mirroring the shots engine's plan/apply split.
- ``output_template`` — the per-map export **config model** (container / bit
  depth / DDS compression, keyed by workflow profile).
- ``map_compositor`` — composites prepared maps into channel-packed atlases.
- ``mat_report`` — a pure **view** (text / HTML) over the material + optimization
  record schema.

The engine *composes* the general image/file primitives it sits on
(``ImgUtils``, ``FileUtils``, ``StrUtils`` …), which stay in their data-type
buckets; only the irreducible texture-domain model, algorithms, and config live
here. It is file-in / file-out (its injected writer is disk IO, never a DCC
scene) — the shared-core rationale is SSoT/DRY, not scene injection.

Lazy-loaded via the pythontk root package; import from pythontk directly
(``from pythontk import MapFactory, MapRegistry, MapOptimizer``).
"""

# Lazy-loaded via parent package - no explicit imports needed
