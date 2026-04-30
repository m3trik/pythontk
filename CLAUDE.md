# pythontk

**Role**: Core Python utils (file IO, strings, math, audio). DCC-agnostic, zero-dependency where possible.

**Nav**: [← root](../CLAUDE.md) · **Used by**: [uitk](../uitk/CLAUDE.md) · [mayatk](../mayatk/CLAUDE.md) · [tentacle](../tentacle/CLAUDE.md) · [unitytk](../unitytk/CLAUDE.md)

## Hard rules

- **No DCC imports.** No `maya`, `nuke`, `bpy`, `PySide`, `PyQt` — this is the bottom of the stack; everything above imports it.
- **Zero-dependency preferred.** If you must add a dep, justify it and keep it optional.

## API surface

Before writing a new helper, **check the registry first** — duplicates undermine the SSoT goal.

- This package: [`API_REGISTRY.md`](API_REGISTRY.md) · [`API_CHANGES.md`](API_CHANGES.md) (diff vs last refresh)
- Cross-package shadows: [`m3trik/docs/API_SHADOWS.md`](../m3trik/docs/API_SHADOWS.md)

Refresh manually: `python m3trik/scripts/generate_api_registry.py pythontk` — otherwise auto-refreshed bi-weekly.

## Notable modules

- `AudioUtils` — ffmpeg-backed audio conversion and WAV compositing (shared by mayatk audio events).
- `AppLauncher` — subprocess launcher (used by mayatk's MayaConnection; do not bypass with raw subprocess).

## Run tests

```powershell
& python o:\Cloud\Code\_scripts\pythontk\test\run_all_tests.py
```

See [CHANGELOG.md](CHANGELOG.md) for history.
