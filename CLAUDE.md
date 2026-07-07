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

## Notable modules

- `AudioUtils` — ffmpeg-backed audio conversion and WAV compositing (shared by mayatk audio events).
- `AppLauncher` — subprocess launcher (used by mayatk's MayaConnection; do not bypass with raw subprocess).
- `MapFactory` — PBR texture-map orchestrator. Now a package (`img_utils/map_factory/`): `conversions` (registry) → `processor` (`TextureProcessor` context) → `handlers` (workflow strategies) → `_map_factory` (orchestrator). Public path unchanged.
- **Photogrammetry/SfM ingest cluster** — general-purpose media primitives that compose (in `extapps`' metashape/realityscan workflows) into a capture pipeline; placed by data type, not domain, so each stays reusable on its own: `FrameExtractor` (vid_utils), `ExposureEqualizer`/`ImageCurator`/`MaskGenerator` (img_utils), `QcLog`/`QcGate` (core_utils).

## Run tests

```powershell
& python o:\Cloud\Code\_scripts\pythontk\test\run_tests.py
```

See [CHANGELOG.md](CHANGELOG.md) for history.
