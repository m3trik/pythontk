# !/usr/bin/python
# coding=utf-8
"""Structured run logs and threshold-based acceptance gates for pipeline
workflows. Engine-agnostic primitives originally extracted from
:mod:`extapps.photogrammetry.metashape_workflow` so a second engine
(RealityCapture, etc.) can reuse them.
"""
import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List


class GateError(RuntimeError):
    """Raised by :meth:`QcGate.check` when a halt-mode gate fails."""


class QcLog:
    """Append-only structured run log. Flushed to JSON on close.

    Stage-timing usage::

        with qc.stage("align") as st:
            ...
            st["rms_reproj_px"] = 0.42
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.data: Dict[str, Any] = {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stages": {},
            "gates": {},
            "warnings": [],
            "success": None,
        }
        self._t0 = time.monotonic()

    @contextmanager
    def stage(self, name: str):
        bucket: Dict[str, Any] = {}
        self.data["stages"][name] = bucket
        t = time.monotonic()
        try:
            yield bucket
        finally:
            bucket["duration_sec"] = round(time.monotonic() - t, 3)

    def warn(self, message: str) -> None:
        self.data["warnings"].append(message)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def finalize(self, success: bool) -> None:
        self.data["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.data["total_duration_sec"] = round(time.monotonic() - self._t0, 3)
        self.data["success"] = bool(success)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)


class QcGate:
    """Threshold-based acceptance gate that logs into a bound :class:`QcLog`.

    Rule keys are prefixed ``min_`` / ``max_``; the suffix names the metric
    key to compare. Example rules dict::

        {
            "align":   {"min_aligned_pct": 75.0, "max_rms_reproj_px": 1.0},
            "model":   {"min_largest_component_pct": 85.0},
        }

    In ``"halt"`` mode the first failed gate raises :class:`GateError`;
    in ``"warn"`` mode the failure is recorded and execution continues.
    """

    MODES = ("warn", "halt")

    def __init__(
        self,
        rules: Dict[str, Dict[str, float]],
        qc_log: QcLog,
        mode: str = "warn",
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.rules = rules
        self.qc = qc_log
        self.mode = mode

    def check(self, gate_name: str, metrics: Dict[str, Any]) -> bool:
        """Compare ``metrics`` against ``self.rules[gate_name]``. Returns
        True when all measured thresholds passed."""
        rules = self.rules.get(gate_name) or {}
        warnings: List[str] = []
        for rule_key, threshold in rules.items():
            if threshold is None:
                continue
            metric_key = rule_key.replace("min_", "").replace("max_", "")
            value = metrics.get(metric_key)
            if value is None:
                warnings.append(f"{metric_key}: metric not measured (gate skipped)")
                continue
            if rule_key.startswith("min_") and value < threshold:
                warnings.append(f"{metric_key} {value:.2f} < {threshold:.2f}")
            elif rule_key.startswith("max_") and value > threshold:
                warnings.append(f"{metric_key} {value:.2f} > {threshold:.2f}")
        passed = not any(
            w for w in warnings if "metric not measured" not in w
        )
        self.qc.data["gates"][gate_name] = {"passed": passed, "warnings": warnings}
        if warnings:
            for w in warnings:
                self.qc.warn(f"[gate:{gate_name}] {w}")
                print(f"[gate:{gate_name}] {w}")
        if not passed and self.mode == "halt":
            raise GateError(f"Gate '{gate_name}' failed: {warnings}")
        return passed
