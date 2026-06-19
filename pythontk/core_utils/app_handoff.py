# !/usr/bin/python
# coding=utf-8
"""Generic, Qt-free / DCC-free engine for "export something and hand it to an app".

The reusable backbone the ecosystem's app hand-off bridges share. It is built around
**two orthogonal extension axes** so a single, stable skeleton supports every bridge
shape -- script-launch (Maya / Blender / RizomUV), copy-to-project (Unity), and
launch-or-attach + RPC round-trip (Substance Painter / Marmoset Toolbag):

* **Template-Method** -- :meth:`HandoffBridge.send` owns the invariant flow:
  ``resolve objects -> preflight -> produce a payload -> deliver``. It knows nothing
  about FBX, scripts, RPC, or any specific app.
* **Strategy** -- the *deliver* step is a pluggable :class:`Deliverer`. pythontk ships
  the generic :class:`ScriptLaunchDeliverer` (render a template, launch a **fresh**
  app on it); other deliverers (Unity copy-to-Assets, Painter/Toolbag RPC) live with
  their app glue and plug into the same seam.
* **Data, not subclass attrs** -- per-app discovery is an :class:`AppSpec` dataclass;
  the script-launch deliverer is configured by a :class:`ScriptLaunchSpec` dataclass.
  A bridge declares *usage* as data and contributes only what truly differs.

This is the bottom-of-stack rule: no ``maya`` / ``bpy`` / ``PySide`` here. DCC bridges
defer their ``import maya.cmds`` / ``import bpy`` into call bodies so the surface
resolves headlessly.
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pythontk.core_utils.app_launcher import AppLauncher
from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.core_utils import script_template

# Re-export so callers get the canonical mode constant from one place.
SEND_TO = script_template.SEND_TO


# --------------------------------------------------------------------------- specs
@dataclass(frozen=True)
class AppSpec:
    """Declarative target-application executable-discovery config (data, not code).

    A frozen dataclass a bridge attaches to declare *how to find* its target app,
    resolved through :meth:`pythontk.AppLauncher.resolve_app_path`. Replaces the pile
    of per-bridge ``EXE_ENV_VARS`` / ``APP_NAMES`` / ``SCAN_GLOBS`` class attributes.
    """

    name: str = "target app"
    env_vars: Tuple[str, ...] = ()
    location_env_vars: Tuple[Tuple[str, Any], ...] = ()  # ((env_var, suffix), ...)
    app_names: Tuple[str, ...] = ()
    scan_globs: Tuple[str, ...] = ()
    not_found_msg: str = ""

    def resolve(self) -> Optional[str]:
        """Resolve the executable, first hit wins (env -> find_app -> install scan)."""
        return AppLauncher.resolve_app_path(
            env_vars=self.env_vars,
            location_env_vars=self.location_env_vars,
            app_names=self.app_names,
            scan_globs=self.scan_globs,
        )

    @property
    def not_found_message(self) -> str:
        """A user-facing "couldn't find it" message (custom, or a sensible default)."""
        return self.not_found_msg or f"{self.name} executable not found."


@dataclass
class HandoffRequest:
    """The unit of work threaded through the skeleton.

    *template* / *mode* drive the deliverer; *params* are the merged tunable knob
    values; *extras* carries any per-bridge orchestration knobs (e.g. ``output_dir``,
    ``target``) a richer ``send()`` wrapper wants to pass its producer/deliverer
    without widening the generic signature.
    """

    template: str = "import"
    mode: str = SEND_TO
    params: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Read a per-bridge orchestration knob from :attr:`extras`."""
        return self.extras.get(key, default)


@dataclass
class Payload:
    """What :meth:`HandoffBridge._produce` hands to the deliverer.

    *primary* is the main artifact path (typically the exported FBX) and may be
    ``None`` for templates that operate on an already-loaded project and export
    nothing. *extras* carries any side artifacts (material manifest, bake-pairs
    sidecar, staged textures, output dir, ...). A producer returning ``None`` (not a
    ``Payload``) signals a *failed* produce and aborts the hand-off.
    """

    primary: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------- strategy
class Deliverer:
    """Strategy: hand a produced :class:`Payload` to the target app.

    :meth:`preflight` validates the request *before* the (possibly expensive) produce
    step so a bad mode / missing exe / missing project aborts early. :meth:`deliver`
    performs the hand-off and returns a result dict, or ``None`` on a handled
    (already-logged) failure.
    """

    def preflight(self, bridge: "HandoffBridge", request: HandoffRequest) -> bool:
        """Validate *request* before producing the payload. Default: always proceed."""
        return True

    def deliver(
        self, bridge: "HandoffBridge", payload: Payload, request: HandoffRequest
    ) -> Optional[Dict[str, Any]]:
        """Hand *payload* to the target app; return a result dict or ``None``."""
        raise NotImplementedError


# ---------------------------------------------------------------------- orchestrator
class HandoffBridge(LoggingMixin):
    """Template-Method base: ``resolve -> preflight -> produce -> deliver``.

    A subclass supplies the polymorphic steps and (optionally) declares its target
    app + deliverer as data:

    * :attr:`app_spec` -- an :class:`AppSpec` (discovery), or override :attr:`app_path`.
    * :attr:`deliverer` -- a :class:`Deliverer` strategy, or override :meth:`_deliver`.
    * :meth:`_resolve_objects` -- read the host selection.
    * :meth:`_produce` -- build the :class:`Payload` (export FBX, sidecars, ...).

    Bridges that expose tunable params override :meth:`params_defaults`. Bridges with a
    richer public ``send()`` (extra app-specific kwargs) override :meth:`send` to pack
    those into :attr:`HandoffRequest.extras` and call :meth:`_run`.
    """

    app_spec: Optional[AppSpec] = None
    deliverer: Optional[Deliverer] = None
    # When False, an empty selection is allowed (e.g. a template that targets an
    # already-loaded project and exports nothing).
    requires_objects: bool = True
    # Temp payload filename stem (``<prefix>_<tag>.fbx``).
    payload_prefix: str = "handoff"

    def __init__(self, app_path: Optional[str] = None):
        super().__init__()
        self._app_path = app_path

    # ------------------ Executable discovery (data-driven) ------------------
    @property
    def app_path(self) -> Optional[str]:
        """Resolved target executable (cached), or ``None``.

        Resolves from :attr:`app_spec` on first access; override the property (or set
        :attr:`app_path`) for targets whose discovery doesn't fit the env/scan model.
        """
        if not self._app_path and self.app_spec is not None:
            self._app_path = self.app_spec.resolve()
        return self._app_path

    @app_path.setter
    def app_path(self, value: Optional[str]) -> None:
        self._app_path = value

    # ------------------ Parameters ------------------------------------------
    def params_defaults(self) -> Dict[str, Any]:
        """Return ``{key: default}`` for the bridge's tunable params (default empty)."""
        return {}

    def merge_params(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge *params* over :meth:`params_defaults` (user values win)."""
        merged = self.params_defaults()
        merged.update(params or {})
        return merged

    # ------------------ Orchestration ---------------------------------------
    def send(
        self,
        objects: Optional[List[Any]] = None,
        *,
        template: str = "import",
        mode: str = SEND_TO,
        params: Optional[Dict[str, Any]] = None,
        **extras: Any,
    ) -> Optional[Dict[str, Any]]:
        """Export *objects* and hand them to the target app (one-way).

        Returns the deliverer's result dict on success, or ``None`` on a handled
        failure (always logged). ``objects=None`` uses the host's current selection.
        Extra keyword args ride along in :attr:`HandoffRequest.extras`.
        """
        request = HandoffRequest(
            template=template, mode=mode, params=self.merge_params(params), extras=extras
        )
        return self._run(objects, request)

    def _run(
        self, objects: Optional[List[Any]], request: HandoffRequest
    ) -> Optional[Dict[str, Any]]:
        """The invariant skeleton; the public ``send()`` wrappers funnel through here."""
        resolved = self._resolve_objects(objects)
        if self.requires_objects and not resolved:
            self.logger.error("No valid objects supplied for sending.")
            return None

        # Preflight lets the deliverer abort (bad mode, missing exe, missing project)
        # *before* the potentially expensive produce step.
        if not self._preflight(resolved, request):
            return None

        try:
            payload = self._produce(resolved, request)
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to produce the export payload: {e}")
            return None
        if payload is None:  # producer signalled a handled failure
            return None

        return self._deliver(payload, request)

    def _preflight(self, objects: List[Any], request: HandoffRequest) -> bool:
        """Validate the request before producing. Delegates to the deliverer."""
        if self.deliverer is not None:
            return self.deliverer.preflight(self, request)
        return True

    def _deliver(
        self, payload: Payload, request: HandoffRequest
    ) -> Optional[Dict[str, Any]]:
        """Hand the produced *payload* to the target app via the deliverer strategy."""
        if self.deliverer is None:
            raise NotImplementedError(
                "Set a `deliverer` Strategy or override `_deliver()`."
            )
        return self.deliverer.deliver(self, payload, request)

    def _make_payload_path(self, extension: str = ".fbx") -> str:
        """Return a unique temp payload path (``<payload_prefix>_<tag><extension>``)."""
        tag = f"{time.time_ns():x}"
        return os.path.join(
            tempfile.gettempdir(), f"{self.payload_prefix}_{tag}{extension}"
        )

    # ------------------ Subclass hooks --------------------------------------
    def _resolve_objects(self, objects):  # pragma: no cover - subclass contract
        """Return the list of objects to export; ``None`` -> host selection."""
        raise NotImplementedError

    def _produce(
        self, objects, request: HandoffRequest
    ) -> Optional[Payload]:  # pragma: no cover
        """Build and return the :class:`Payload` (``None`` aborts the hand-off)."""
        raise NotImplementedError


# ------------------------------------------------- script-launch deliverer + spec
@dataclass(frozen=True)
class ScriptLaunchSpec:
    """Declarative config for the render-a-script-then-launch-a-fresh-app deliverer.

    *launch_args* maps the rendered script's path to the argv that makes the target
    run it on startup (e.g. ``lambda s: ["--python", s]`` /
    ``lambda s: ["-command", mel_wrapper(s)]``).
    """

    app: AppSpec
    template_dir: Path
    launch_args: Callable[[str], Sequence[str]]
    template_extension: str = ".py"
    modes: Tuple[str, ...] = (SEND_TO,)
    payload_prefix: str = "handoff"


class ScriptLaunchDeliverer(Deliverer):
    """Render a template, write it next to the payload, launch a **fresh** app on it.

    Shared by Maya / Blender / RizomUV bridges: validate the requested mode against
    the template's declared modes, render the ``templates/<stem>`` file with the
    payload path + the bridge's :meth:`render_context` substituted, write it to a temp
    script beside the payload, and launch a **fresh** detached instance of the target
    app pointed at that script (never attach to a running session -- the ecosystem
    session-safety rule).
    """

    def __init__(self, spec: ScriptLaunchSpec):
        self.spec = spec

    def _template_path(self, template: str) -> Path:
        return Path(self.spec.template_dir) / f"{template}{self.spec.template_extension}"

    def preflight(self, bridge: HandoffBridge, request: HandoffRequest) -> bool:
        spec = self.spec
        template_path = self._template_path(request.template)
        allowed = (
            script_template.template_modes(template_path, spec.modes)
            if template_path.is_file()
            else ()
        )
        if request.mode not in allowed:
            bridge.logger.error(
                f"Template '{request.template}' does not support mode "
                f"'{request.mode}'. Declared: {allowed}"
            )
            return False
        if not bridge.app_path:
            bridge.logger.error(spec.app.not_found_message)
            return False
        return True

    def deliver(
        self, bridge: HandoffBridge, payload: Payload, request: HandoffRequest
    ) -> Optional[Dict[str, Any]]:
        script = self.render(bridge, payload, request)
        if script is None:
            return None

        script_path = str(Path(payload.primary).with_suffix(self.spec.template_extension))
        Path(script_path).write_text(script, encoding="utf-8")
        bridge.logger.info(
            f"Sending to {self.spec.app.name} ({request.template}) with script "
            f"{script_path}"
        )

        # FRESH instance every time -- never attach to a running session. Detached:
        # control returns immediately.
        proc = AppLauncher.launch(
            bridge.app_path, args=self.spec.launch_args(script_path), detached=True
        )
        if proc is None:
            bridge.logger.error(
                f"Failed to launch {self.spec.app.name}: {bridge.app_path}"
            )
            return None

        bridge.logger.info(
            f"Sent to {self.spec.app.name} (interactive session)."
        )
        return {
            "script": script_path,
            "template": request.template,
            "mode": request.mode,
            "payload": payload.primary,
        }

    def render(
        self, bridge: HandoffBridge, payload: Payload, request: HandoffRequest
    ) -> Optional[str]:
        """Return the rendered script body for *request*'s template, or ``None`` on miss."""
        template_path = self._template_path(request.template)
        if not template_path.is_file():
            available = sorted(
                p.stem
                for p in script_template.list_templates(
                    self.spec.template_dir, self.spec.template_extension
                )
            )
            bridge.logger.error(
                f"Template '{request.template}' not found at {template_path}. "
                f"Available: {available}"
            )
            return None
        context = {"FBX_PATH": str(payload.primary).replace("\\", "/")}
        context.update(bridge.render_context(request.params))
        return script_template.render_template(template_path, context)


class ScriptLaunchBridge(HandoffBridge):
    """A :class:`HandoffBridge` whose delivery is :class:`ScriptLaunchDeliverer`.

    Subclasses set the :attr:`spec` (:class:`ScriptLaunchSpec`) and implement
    :meth:`render_context` (+ the :meth:`_resolve_objects` / :meth:`_produce` hooks,
    typically via a DCC export mixin). The deliverer and discovery are wired from the
    spec, so a concrete bridge is just data + the DCC-specific export.
    """

    spec: Optional[ScriptLaunchSpec] = None

    def __init__(self, app_path: Optional[str] = None):
        super().__init__(app_path=app_path)
        if self.spec is None:
            raise TypeError(f"{type(self).__name__} must set a ScriptLaunchSpec `spec`.")
        self.app_spec = self.spec.app
        self.payload_prefix = self.spec.payload_prefix
        self.deliverer = ScriptLaunchDeliverer(self.spec)

    def render_context(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Format *params* into a ``__KEY__`` substitution context (subclass hook)."""
        raise NotImplementedError

    def render_template(
        self, template: str, payload_path: str, params: Dict[str, Any]
    ) -> Optional[str]:
        """Render *template*'s body with *payload_path* + *params* (no launch).

        A convenience for previewing/testing the rendered script. *params* is used
        as-is (already merged by the caller).
        """
        return self.deliverer.render(
            self,
            Payload(primary=payload_path),
            HandoffRequest(template=template, params=params),
        )

    # ------------------ Template helpers (for the slot/UI layer) ------------
    def list_template_modes(self) -> List[Tuple[str, str]]:
        """``[(stem, mode), ...]`` for the bridge's template directory."""
        return script_template.list_template_modes(
            self.spec.template_dir, self.spec.template_extension, self.spec.modes
        )

    def list_templates(self) -> List[Path]:
        """User-visible template paths for the bridge."""
        return script_template.list_templates(
            self.spec.template_dir, self.spec.template_extension
        )


__all__ = [
    "SEND_TO",
    "AppSpec",
    "HandoffRequest",
    "Payload",
    "Deliverer",
    "HandoffBridge",
    "ScriptLaunchSpec",
    "ScriptLaunchDeliverer",
    "ScriptLaunchBridge",
]
