# pythontk

**Role**: Core Python utils (file IO, strings, math, audio). DCC-agnostic, zero-dependency where possible.

**Nav**: [← root](../CLAUDE.md) · [docs](docs/README.md) · **Used by**: [uitk](../uitk/CLAUDE.md) · [mayatk](../mayatk/CLAUDE.md) · [tentacle](../tentacle/CLAUDE.md) · [unitytk](../unitytk/CLAUDE.md)

## Hard rules

- **No DCC imports.** No `maya`, `nuke`, `bpy`, `PySide`, `PyQt` — this is the bottom of the stack; everything above imports it.
- **Zero-dependency preferred.** If you must add a dep, justify it and keep it optional.

## API surface

**Before adding a helper, check the registry** (navigation rules: [root](../CLAUDE.md)):

- [`API_INDEX.md`](API_INDEX.md) (compact — read first) · [`API_REGISTRY.md`](API_REGISTRY.md) (grep, don't Read whole) · [`API_CHANGES.md`](API_CHANGES.md)
- Cross-package shadows: [`m3trik/docs/API_SHADOWS.md`](../m3trik/docs/API_SHADOWS.md)

## Placing shared logic — `*_utils` vs `core_utils/engines/`

Two homes, two contracts (both bound by the Hard rules above — no DCC imports, zero-dep preferred, no consumer knowledge):

- **`*_utils`** promises *generality* — plausibly useful to any Python program, placed by data type (strings, images, math). Try this **first**: decompose the logic into general primitives (the photogrammetry-ingest precedent — `FrameExtractor`, `MaskGenerator`, `QcGate`).
- **`core_utils/engines/<system>/`** hosts *domain engines* — a complete, pure app core (model + planning + algorithms with DI seams) that is domain-specific by design and can't be genericised without losing its shape. Use it only for an **irreducible shared core needed by ≥2 downstream packages that can't import each other** (e.g. mayatk + blendertk). Pushing such a document type into `*_utils` erodes that layer's charter; duplicating it downstream breaks SSoT. Tenants: `shots/` (timeline model + planner + manifest), `instancing/` (separated-part clustering — the general PCA/NN math stays in `geo_utils.PointCloud`), `textures/` (PBR map taxonomy + prep + packaging). If an engine outgrows the package (size / needs a dep), the namespace is the extraction boundary — graduate it to its own distribution. A *single tool's* domain math is NOT an engine even when both DCCs need it: extract its general primitive to `*_utils` and vendor the small remainder with the consumers, drift-guarded (the curtain precedent — `geo_utils.RailSurface` stayed, `CurtainDrape` went to `mayatk`/`blendertk` `edit_utils/_curtain_drape.py`).
  - **Not every shared framework is an engine.** The bar is *model + algorithm/planner*, not merely "used by both DCCs." A *general* orchestration base with no domain-model/algorithm layer stays in `core_utils` beside the other infra (`app_launcher`, `preset_store`): e.g. `app_handoff.py` (the Template-Method/Strategy app-bridge kit — its own docstring calls it *"Generic"*) is shared by mayatk+blendertk yet is **not** an engine. So are `net_utils.rpc` (generic JSON-RPC transport) and the `*_utils` primitives both DCCs happen to share (`PointCloud`, `Polyline`, `Color`, `Metadata`).

## Notable modules

- `AudioUtils` — ffmpeg-backed audio conversion and WAV compositing (shared by mayatk audio events).
- `AppLauncher` — subprocess launcher (used by mayatk's MayaConnection; do not bypass with raw subprocess).
- `MapFactory` — PBR texture-map orchestrator, a `core_utils/engines/textures/` tenant (`map_factory/` package: `conversions` registry → `processor` context → `handlers` strategies → `_map_factory` orchestrator; siblings `map_registry`/`map_optimizer`/`map_compositor`). Root re-exports (`ptk.MapFactory`, `ptk.MapCompositor`) unchanged.
- **Photogrammetry/SfM ingest cluster** — general-purpose media primitives that compose (in `extapps`' metashape/realityscan workflows) into a capture pipeline; placed by data type, not domain, so each stays reusable on its own: `FrameExtractor` (vid_utils), `ExposureEqualizer`/`ImageCurator`/`MaskGenerator` (img_utils), `QcLog`/`QcGate` (core_utils).

## Run tests

```powershell
& python o:\Cloud\Code\_scripts\pythontk\test\run_tests.py
```

See [CHANGELOG.md](CHANGELOG.md) for history.
