# pythontk

**Role**: Core Python utils (file IO, strings, math, audio). DCC-agnostic, zero-dependency where possible.

**Nav**: [← root](../CLAUDE.md) · [docs](docs/README.md) · **Used by**: [uitk](../uitk/CLAUDE.md) · [mayatk](../mayatk/CLAUDE.md) · [blendertk](../blendertk/CLAUDE.md) · [tentacle](../tentacle/CLAUDE.md) · [unitytk](../unitytk/CLAUDE.md) · [extapps](../extapps/CLAUDE.md)

## Hard rules

- **No DCC imports.** No `maya`, `nuke`, `bpy`, `PySide`, `PyQt` — this is the bottom of the stack; everything above imports it. Sole exception: `net_utils/rpc/plugin_core.py` resolves its host app's own PySide lazily, inside that app (Painter, Toolbag).
- **Host adapters, not host subclasses.** A shared class reaches its host through a small Protocol beside it with a `Pure*` default (`ShotsHost` / `PureShotsHost`), injected or resolved from the running host; a library class keeping its own name for it is a binding that configures and adds, never overrides ([CODE_STANDARD](../m3trik/docs/CODE_STANDARD.md) §0; `check_dcc_twins.py` `HOST_BINDINGS`).
- **Zero-dependency preferred.** If you must add a dep, justify it and keep it optional.

## API surface

[`API_INDEX.md`](API_INDEX.md) · [`API_REGISTRY.md`](API_REGISTRY.md) · [`API_CHANGES.md`](API_CHANGES.md) · shadows [`API_SHADOWS.md`](../m3trik/docs/API_SHADOWS.md) — registry rules in [root](../CLAUDE.md). Bottom of the stack: no upstream.

## Placing shared logic — `*_utils` vs `core_utils/engines/`

Two homes, two contracts (both bound by the Hard rules above — no DCC imports, zero-dep preferred, no consumer knowledge):

- **`*_utils`** promises *generality* — plausibly useful to any Python program, placed by data type (strings, images, math). Try this **first**: decompose the logic into general primitives (the photogrammetry-ingest precedent — `FrameExtractor`, `MaskGenerator`, `QcGate`).
- **`core_utils/engines/<system>/`** holds the host-free models mayatk and blendertk share: a *domain engine* is a complete, pure core (model + planning + algorithms) behind an adapter seam each host implements ([CODE_STANDARD](../m3trik/docs/CODE_STANDARD.md) §14); the host halves and the panels never come here. Tenants: `shots/` (timeline model + planner + manifest), `scene_export/` (scene records + export profile/snapshot + hierarchy baseline, and the exporter / sidecar shells both DCC Scene Exporters bind to their host), `instancing/` (separated-part clustering — the general PCA/NN math stays in `geo_utils.PointCloud`), `textures/` (PBR map taxonomy + prep + packaging), `rig_graph/` (rig-intent model + capability negotiation + planner), `key_stash/` (parked-keyframe clip store), `lighting/` (baked light layers + look match). A *single tool's* domain math is NOT an engine even when both DCCs need it: it goes to the `*_utils` home of its data type, beside the general primitive it composes, written once (`geo_utils.CurtainDrape` over `RailSurface`; `geo_utils.ProbePlacement`).
  - **Not every shared framework is an engine.** The bar is *model + algorithm/planner*, not merely "used by both DCCs." A *general* orchestration base with no domain-model/algorithm layer stays in `core_utils` beside the other infra (`app_launcher`, `presets/`): e.g. `handoff/` (the Template-Method/Strategy app-bridge kit — its own docstring calls it *"Generic"*) is shared by mayatk+blendertk yet is **not** an engine. So are `net_utils.rpc` (generic JSON-RPC transport) and the `*_utils` primitives both DCCs happen to share (`PointCloud`, `Polyline`, `Color`, `Metadata`).

## Notable modules

- `Deprecation` — the ONE way to retire public surface (`@Deprecation.symbol` / `.parameter` / `.attributes` / `.class_attribute` / `.values`). `remove_in` is a version, not a sentence: the registry `--check` fails once it ships. Never hand-roll a `warnings.warn`. Rules in [CODE_STANDARD.md §5](../m3trik/docs/CODE_STANDARD.md).
- `UpstreamPatch` — the ONE way to correct a defect in code we do not own (`with patch.applied():`). Mirror of `Deprecation`: its point is `detects`, the probe that reproduces the defect against the STOCK code, so a sweep test fails when upstream fixes it and names the patch to delete. Declare them per package in one module (blendertk: `env_utils/upstream_patches.py`); never `setattr` a third party by hand.
- `AudioUtils` — ffmpeg-backed audio conversion and WAV compositing (shared by mayatk audio events).
- `AppLauncher` — subprocess launcher (used by mayatk's MayaConnection; do not bypass with raw subprocess).
- `MapFactory` — PBR texture-map orchestrator, a `core_utils/engines/textures/` tenant (`map_factory/` package: `conversions` registry → `processor` context → `handlers` strategies → `_map_factory` orchestrator; siblings `map_registry`/`map_optimizer`/`map_compositor`). Root re-exports (`ptk.MapFactory`, `ptk.MapCompositor`) unchanged.
- **Photogrammetry/SfM ingest cluster** — general-purpose media primitives that compose (in `extapps`' metashape/realityscan workflows) into a capture pipeline; placed by data type, not domain, so each stays reusable on its own: `FrameExtractor` (vid_utils), `ExposureEqualizer`/`ImageCurator`/`MaskGenerator` (img_utils), `QcLog`/`QcGate` (core_utils).

## Run tests

```powershell
& python o:\Cloud\Code\_scripts\pythontk\test\run_tests.py test_str -k split  # pointed: modules, -k
& python o:\Cloud\Code\_scripts\pythontk\test\run_tests.py                   # full suite: review / release
```

What a change reaches: `python m3trik/scripts/affected_tests.py <files> --run` (root Testing).

See [CHANGELOG.md](CHANGELOG.md) for history.
