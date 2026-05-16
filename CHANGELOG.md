# pythontk — Changelog

## 2026

- **Affix hardening** — `StrUtils.strip_known_affix` / `StrUtils.apply_affix` idempotently strip and reapply a configured prefix/suffix (case-insensitive, separator-tolerant including stray leading/trailing `_`, no false-positives like `Matte_door` for prefix `Mat_`). `strip_known_affix` is a conservative primitive — it touches only the affix and adjacent separators, never internal/remote underscores. `apply_affix` is a no-op when both affixes are empty (so user-typed names pass through untouched), and trims dangling underscores only on the side(s) where an affix is being applied. `ImgUtils.get_base_texture_name`, `MapFactory.get_base_texture_name`, `MapFactory.group_textures_by_set`, `MapFactory.resolve_texture_filename`, and `MapFactory.prepare_maps` now accept `prefix`/`suffix` and route them through, so callers can safely round-trip texture/material names without producing `Mat_Mat_<name>` duplicates or trailing-underscore artifacts.
- **AudioUtils module** — reusable ffmpeg-backed audio conversion helpers (`ensure_playable_path`, extension checks, ffmpeg resolution) for downstream DCC tools.
- **Audio Composite Builder** — DCC-agnostic WAV compositing helper (`build_composite_wav`) — centralizes timeline-audio mixing logic.
