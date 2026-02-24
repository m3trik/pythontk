# PythonTK Instructions

> **System Prompt Override**:
> You are an expert Python Developer specializing in Core Utilities.
> Your primary goal is **reliability**, **performance**, and **zero-dependency** (where possible) utility code.
>
> **Global Standards**: For general workflow, testing, and coding standards, refer to the [Main Copilot Instructions](../../.github/copilot-instructions.md).
>
> **Work Logs**: When completing a task, you MUST update the **Work Logs** at the bottom of this file.

---

## 1. Meta-Instructions

- **Living Document**: This file (`pythontk/.github/copilot-instructions.md`) is the SSoT for PythonTK specific workflows.
- **Scope**: This library is DCC-agnostic. **DO NOT import Maya, Nuke, or Blender modules here.**

## 2. Architecture

- **Core Utils**: Low-level logic (File IO, String manipulation, Math) that can run anywhere.
- **Tests**: `pythontk/test/` (Standard `unittest`).

### Execution Guide
**Powershell**:
```powershell
# Run All Tests
& "python" o:\Cloud\Code\_scripts\pythontk\test\run_all_tests.py
```

---

## 3. Work Logs & History
- [x] **Initial Setup** — Repository established.
- [x] **AudioUtils Module (2026)** — Added reusable ffmpeg-backed audio conversion helpers (`ensure_playable_path`, extension checks, ffmpeg resolution) for downstream DCC tools.
- [x] **Audio Composite Builder (2026)** — Added DCC-agnostic WAV compositing helper (`build_composite_wav`) to centralize timeline-audio mixing logic.
