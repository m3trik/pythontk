# pythontk

**Role**: Core Python utils (file IO, strings, math, audio). DCC-agnostic, zero-dependency where possible.

**Nav**: [← root](../CLAUDE.md) · **Used by**: [uitk](../uitk/CLAUDE.md) · [mayatk](../mayatk/CLAUDE.md) · [tentacle](../tentacle/CLAUDE.md) · [unitytk](../unitytk/CLAUDE.md)

## Hard rules

- **No DCC imports.** No `maya`, `nuke`, `bpy`, `PySide`, `PyQt` — this is the bottom of the stack; everything above imports it.
- **Zero-dependency preferred.** If you must add a dep, justify it and keep it optional.

## Notable modules

- `AudioUtils` — ffmpeg-backed audio conversion and WAV compositing (shared by mayatk audio events).
- `AppLauncher` — subprocess launcher (used by mayatk's MayaConnection; do not bypass with raw subprocess).

## Run tests

```powershell
& python o:\Cloud\Code\_scripts\pythontk\test\run_all_tests.py
```

See [CHANGELOG.md](CHANGELOG.md) for history.
