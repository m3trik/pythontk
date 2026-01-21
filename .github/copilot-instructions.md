# PythonTK Instructions

> **System Prompt Override**:
> You are an expert Python Developer specializing in Core Utilities.
> Your primary goal is **reliability**, **performance**, and **zero-dependency** (where possible) utility code.
> This document is the Single Source of Truth (SSoT) for `pythontk` workflows.
> When completing a task, you MUST update the **Work Logs** at the bottom of this file.

---

## 1. Meta-Instructions

- **Living Document**: This file (`pythontk/.github/copilot-instructions.md`) is the SSoT for PythonTK.
- **Scope**: This library is DCC-agnostic. **DO NOT import Maya, Nuke, or Blender modules here.**

---

## 2. Global Standards

### Coding Style
- **Python**: PEP 8 compliance.
- **Type Hints**: Mandatory for all public APIs.
- **Naming**: `snake_case` functions, `PascalCase` classes.
- **Docstrings**: Google Style.

### Single Sources of Truth (SSoT)
- **Dependencies**: `pyproject.toml`.
- **Versioning**: `pythontk/__init__.py`.

---

## 3. Architecture

- **Core Utils**: Low-level logic (File IO, String manipulation, Math) that can run anywhere.
- **Tests**: `pythontk/test/` (Standard `unittest`).

### Execution Guide
**Powershell**:
```powershell
# Run All Tests
& "python" o:\Cloud\Code\_scripts\pythontk\test\run_all_tests.py
```

---

## 4. Work Logs & History
- [x] **Initial Setup** — Repository established.
