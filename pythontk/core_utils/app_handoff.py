# !/usr/bin/python
# coding=utf-8
"""Generic, Qt-free / DCC-free engine for "export something and hand it to an app".

The reusable backbone the ecosystem's app hand-off bridges share. It is built around
**two orthogonal extension axes** so a single, stable skeleton supports every bridge
shape -- script-launch (Maya / Blender / RizomUV), copy-to-project (Unity), and
launch-or-attach + RPC round-trip (Substance Painter / Marmoset Toolbag):

* **Template-Method** -- :meth:`HandoffBridge.send` owns the invariant flow:
  ``resolve objects -> preflight -> produce a payload -> deliver -> ingest``. It knows
  nothing about FBX, scripts, RPC, or any specific app. The trailing *ingest* step is
  what makes a **round trip** a mode of the one skeleton rather than a second pipeline:
  every hand-off produces and delivers, and the ones that bring a result back add a
  step instead of forking. It is a plain hook, not a Strategy -- every inbound leg in
  the ecosystem is irreducibly host-specific (re-import and transfer UVs onto the
  originals; rebuild materials from a manifest), so there is no shared algorithm for a
  strategy object to hold.
* **Strategy** -- the *deliver* step is a pluggable :class:`Deliverer`. pythontk ships
  two: :class:`ScriptLaunchDeliverer` (render a template, launch a **fresh** app on it,
  return immediately) and its blocking sibling :class:`ScriptRunDeliverer` (run the
  target headlessly and keep the artifact it wrote -- what makes ``save_as`` a native
  file of the *target* app possible). Other deliverers (Unity copy-to-Assets,
  Painter/Toolbag RPC) live with their app glue and plug into the same seam.
* **Per-mode strategies** -- :attr:`HandoffBridge.deliverers` maps a request *mode* to
  a deliverer, so one bridge supports both shapes (``send_to`` -> launch, ``save_as``
  -> run) off one export pipeline, with no branch in the skeleton.
* **Data, not subclass attrs** -- per-app discovery is an :class:`AppSpec` dataclass;
  the script-launch deliverer is configured by a :class:`ScriptLaunchSpec` dataclass.
  A bridge declares *usage* as data and contributes only what truly differs.

This is the bottom-of-stack rule: no ``maya`` / ``bpy`` / ``PySide`` here. DCC bridges
defer their ``import maya.cmds`` / ``import bpy`` into call bodies so the surface
resolves headlessly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)

from pythontk.core_utils.app_launcher import AppLauncher
from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.core_utils import script_template
from pythontk.file_utils.usd import USD_EXTENSIONS

if TYPE_CHECKING:  # import-time only: the runtime import is deferred into the body
    from pythontk.file_utils.temp_artifacts import TempArtifacts

# Re-export so callers get the canonical mode constants from one place.
SEND_TO = script_template.SEND_TO
SAVE_AS = script_template.SAVE_AS
ROUND_TRIP = script_template.ROUND_TRIP

#: The request param that picks the payload's interchange format (a *carrier*).
CARRIER_PARAM = "CARRIER"
#: Every carrier a hand-off payload can travel as, ``{carrier: extension}``. Which
#: format the payload is written in is a property of hand-off bridges in general --
#: the same selection can leave as FBX or as USD -- so the vocabulary lives here,
#: once, and a bridge only declares the subset its TARGET can read
#: (:attr:`HandoffBridge.carriers`). The DCC-side mixins turn a carrier into the
#: matching export call; nothing here knows how either format is written.
CARRIER_EXTENSIONS: Dict[str, str] = {"fbx": ".fbx", "usd": ".usd"}
#: The inverse, widened to every spelling a carrier's files come in (a ``.usda``
#: / ``.usdc`` / ``.usdz`` payload is still the ``usd`` carrier) -- what a writer
#: table is keyed on once a path exists (:meth:`HandoffBridge.carrier_of`).
CARRIER_BY_EXTENSION: Dict[str, str] = {
    **{ext: carrier for carrier, ext in CARRIER_EXTENSIONS.items()},
    **{ext: "usd" for ext in USD_EXTENSIONS},
}


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

    #: Attribute holding the memoized :meth:`resolve` result (see :attr:`path`).
    #: Written through ``object.__setattr__`` -- the dataclass is frozen, and the
    #: cache is derived state, not a field: it stays out of ``__eq__``/``__hash__``
    #: so a spec that has probed still compares equal to one that hasn't.
    _CACHE_ATTR = "_resolved_path"

    def resolve(self) -> Optional[str]:
        """Resolve the executable, first hit wins (env -> find_app -> install scan).

        Uncached by design: this is the "look right now" call a launch path wants.
        Callers asking repeatedly (a visibility gate, a panel init) want :attr:`path`.
        """
        return AppLauncher.resolve_app_path(
            env_vars=self.env_vars,
            location_env_vars=self.location_env_vars,
            app_names=self.app_names,
            scan_globs=self.scan_globs,
        )

    @property
    def path(self) -> Optional[str]:
        """:meth:`resolve`, memoized for this spec instance.

        Discovery is not cheap -- ``find_app`` hits the Windows ``App Paths``
        registry and ``scan_globs`` walks both Program Files roots, measured at
        12-155ms per app (a MISS is the expensive case, since it exhausts every
        stage). Anything that asks repeatedly -- a panel's availability gate,
        which re-runs on every show -- must not pay that each time.

        Specs are module-level singletons attached to their bridge class, so
        per-instance memoization is per-app process-wide without a registry.
        A miss is cached too: "not installed" is the answer that would otherwise
        cost the most to re-derive. Use :meth:`refresh` after an install.
        """
        try:
            return object.__getattribute__(self, self._CACHE_ATTR)
        except AttributeError:
            pass
        found = self.resolve()
        object.__setattr__(self, self._CACHE_ATTR, found)
        return found

    @property
    def available(self) -> bool:
        """Whether the target app is installed (cached -- see :attr:`path`)."""
        return bool(self.path)

    def refresh(self) -> Optional[str]:
        """Discard the memoized :attr:`path` and re-probe. Returns the new value.

        For the case the cache is otherwise wrong about: the user installed the
        app during this session and wants the tool to light up without a restart.
        """
        try:
            object.__delattr__(self, self._CACHE_ATTR)
        except AttributeError:
            pass
        return self.path

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
    # Per-mode delivery strategies, ``{mode: Deliverer}``. A mode absent here falls back
    # to :attr:`deliverer`, so a single-strategy bridge ignores this entirely. It is the
    # seam a second delivery *shape* plugs into (e.g. the blocking
    # :class:`ScriptRunDeliverer` under ``SAVE_AS`` beside the detached launch under
    # ``SEND_TO``) without touching the skeleton or the existing strategy.
    #
    # ``None``, not ``{}``: a mutable class-level default is shared by every subclass
    # that never rebinds it, so one bridge doing ``self.deliverers[mode] = x`` would
    # silently register that strategy on every other bridge in the process.
    deliverers: Optional[Dict[str, Deliverer]] = None
    # When False, an empty selection is allowed (e.g. a template that targets an
    # already-loaded project and exports nothing).
    requires_objects: bool = True
    # Temp payload filename stem (``<prefix>_<tag>.fbx``). Also the namespace
    # the run scratch is allocated in -- see :meth:`_scratch_dir`.
    payload_prefix: str = "handoff"
    # Carriers (see :data:`CARRIER_EXTENSIONS`) this bridge can produce AND its
    # target can read, in preference order: the first is what a request that
    # doesn't choose gets, and a request asking for any other is refused in
    # :meth:`_run` before anything is exported. A single entry means the bridge
    # offers no choice (a target that only reads FBX). Declared as data so a
    # panel can render the choice from the same tuple the engine enforces.
    carriers: Tuple[str, ...] = ("fbx",)

    # Modes whose staged artifacts the run itself CONSUMES, so it may delete
    # them -- a blocking round trip that relocates its real output somewhere
    # durable. See :meth:`_scratch_policy`, which reads this; the panel-side
    # twin is ``uitk.BridgeSlotsBase.TRANSIENT_OUTPUT_MODES``, which stops the
    # same run inheriting a durable Output Dir it would then delete.
    scoped_scratch_modes: Tuple[str, ...] = ()

    # ``HandoffRequest.extras`` keys the run scratch rides on. Underscored and
    # documented as private: no deliverer reads them, and a bridge stamping its
    # own key here would collide with the produce/ingest pair's bookkeeping.
    _SCRATCH_STORE_KEY = "_scratch_store"
    _SCRATCH_ROOT_KEY = "_scratch_root"

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

    @property
    def headless_app_path(self) -> Optional[str]:
        """Executable for a BLOCKING/headless run; defaults to :attr:`app_path`.

        Most targets are one binary run two ways (``blender`` vs ``blender
        --background``). Maya is not: the GUI ``maya.exe`` is the interactive target
        while a headless run belongs in ``mayapy``, so that bridge overrides this and
        derives the interpreter from whatever :attr:`app_path` resolved to -- an
        explicit user path still drives both.
        """
        return self.app_path

    # ------------------ Parameters ------------------------------------------
    def params_defaults(self) -> Dict[str, Any]:
        """Return ``{key: default}`` for the bridge's tunable params (default empty)."""
        return {}

    def merge_params(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge *params* over :meth:`params_defaults` (user values win)."""
        merged = self.params_defaults()
        merged.update(params or {})
        return merged

    # ------------------ Carrier ---------------------------------------------
    def carrier(self, request: HandoffRequest) -> str:
        """The carrier *request* travels as: ``params["CARRIER"]``, else the first of
        :attr:`carriers`. Lower-cased; NOT validated -- :meth:`_run` refuses a
        carrier the bridge doesn't offer, so a producer never sees one."""
        chosen = request.params.get(CARRIER_PARAM) or self.carriers[0]
        return str(chosen).strip().lower()

    def payload_extension(self, request: HandoffRequest) -> str:
        """The file extension of *request*'s carrier (``".fbx"`` / ``".usd"``)."""
        return CARRIER_EXTENSIONS[self.carrier(request)]

    @staticmethod
    def carrier_of(path: str) -> str:
        """The carrier a payload *path* names, by extension (``"fbx"`` / ``"usd"``).

        The key a producer's writer table is looked up with, so a bridge that
        stages its own payload path only has to name it right and every writer
        stays a one-entry addition. Raises ``ValueError`` for an extension no
        carrier owns -- a path like that is a bug upstream, never a silent FBX.
        """
        ext = os.path.splitext(str(path))[1].lower()
        try:
            return CARRIER_BY_EXTENSION[ext]
        except KeyError:
            raise ValueError(
                f"No hand-off carrier writes {ext or '<no extension>'!r} "
                f"({', '.join(sorted(CARRIER_BY_EXTENSION))})."
            ) from None

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
            template=template,
            mode=mode,
            params=self.merge_params(params),
            extras=extras,
        )
        return self._run(objects, request)

    def _run(
        self, objects: Optional[List[Any]], request: HandoffRequest
    ) -> Optional[Dict[str, Any]]:
        """The invariant skeleton; the public ``send()`` wrappers funnel through here."""
        resolved = self._resolve_objects(objects)
        if self.requires_objects and not resolved:
            self.logger.error(f"No valid objects supplied for '{request.mode}'.")
            return None

        # A carrier the target can't read is a failed hand-off however well the
        # export goes, so it is refused here, before the export is paid for. An
        # unknown spelling is refused too rather than mapped to the default: the
        # caller asked for something specific and would otherwise get FBX silently.
        carrier = self.carrier(request)
        if carrier not in self.carriers or carrier not in CARRIER_EXTENSIONS:
            self.logger.error(
                f"Carrier '{carrier}' is not one this hand-off offers "
                f"({', '.join(self.carriers)})."
            )
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

        result = self._deliver(payload, request)
        if result is None:  # deliverer signalled a handled failure
            return None

        # The return leg. A one-way hand-off's default ingest is the identity, so
        # send-only bridges pay nothing for the step existing.
        ingested = self._ingest(result, resolved, payload, request)
        if ingested is not None:
            # The run completed end to end, so a scoped scratch has served its
            # purpose. HERE rather than in ``_ingest``: that hook is routinely
            # overridden (and legitimately without calling ``super()``), and an
            # invariant a subclass can silently skip is not an invariant. A
            # failure anywhere above -- including a return leg that reports one
            # by returning ``None`` -- leaves the scratch on disk to inspect.
            self._discard_scratch(request, ingested)
        return ingested

    def _deliverer_for(self, request: HandoffRequest) -> Optional[Deliverer]:
        """The strategy for *request*'s mode: :attr:`deliverers`, else :attr:`deliverer`."""
        return (self.deliverers or {}).get(request.mode) or self.deliverer

    def _preflight(self, objects: List[Any], request: HandoffRequest) -> bool:
        """Validate the request before producing. Delegates to the deliverer."""
        deliverer = self._deliverer_for(request)
        if deliverer is not None:
            return deliverer.preflight(self, request)
        return True

    def _deliver(
        self, payload: Payload, request: HandoffRequest
    ) -> Optional[Dict[str, Any]]:
        """Hand the produced *payload* to the target app via the deliverer strategy."""
        deliverer = self._deliverer_for(request)
        if deliverer is None:
            raise NotImplementedError(
                "Set a `deliverer` Strategy or override `_deliver()`."
            )
        return deliverer.deliver(self, payload, request)

    def _make_payload_path(self, extension: str = ".fbx") -> str:
        """Return a unique temp payload path (``<payload_prefix>_<tag><extension>``).

        Detached policy: the launched app reads the payload after we return, so
        there is no deterministic delete — allocation instead sweeps *stale*
        same-prefix payloads from prior sessions (see :class:`TempArtifacts`).
        """
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        cached = getattr(self, "_payload_artifacts", None)
        if cached is None or cached.prefix != self.payload_prefix:
            cached = TempArtifacts(self.payload_prefix, policy="detached")
            self._payload_artifacts = cached
        return cached.path(extension=extension)

    # ------------------ Run scratch -----------------------------------------
    def _scratch_policy(self, request: HandoffRequest) -> str:
        """Lifetime for what this run stages: a :class:`TempArtifacts` policy.

        ``"detached"`` (the default) is the only sound answer for a hand-off the
        target app reads AFTER we return: there is no completion signal, so
        nothing may delete, and allocation sweeps stale leftovers by age instead.

        Return ``"scoped"`` for a run that CONSUMES what it stages -- a blocking
        round trip whose real output is relocated somewhere durable. The
        skeleton (:meth:`_run`) then removes the scratch once the run completes
        end to end, and KEEPS it on any failure, so a broken run stays
        inspectable and the age-gated sweep reclaims it later (a host that dies
        mid-run would skip any ``finally``).

        Reads :attr:`scoped_scratch_modes`, so the ordinary case is one line of
        DATA on the bridge; override this only for a policy the mode alone
        cannot express.

        Opting into ``"scoped"`` for a run whose real output can land INSIDE the
        scratch means also overriding :meth:`_delivered_paths` -- otherwise the
        cleanup takes that output with it.
        """
        return "scoped" if request.mode in self.scoped_scratch_modes else "detached"

    def _run_scratch(self, request: HandoffRequest) -> "TempArtifacts":
        """The temp store this run allocates its scratch directories from.

        One store per run, memoized on the request, so every directory the run
        opens shares one lifetime and one cleanup.
        """
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        store = request.extras.get(self._SCRATCH_STORE_KEY)
        if store is None:
            store = TempArtifacts(
                self.payload_prefix, policy=self._scratch_policy(request)
            )
            request.extras[self._SCRATCH_STORE_KEY] = store
        return store

    def _scratch_dir(self, request: HandoffRequest, name: str) -> str:
        """Return this run's scratch directory called *name*, created.

        The ``mkdtemp`` replacement for staging a hand-off: what it adds over a
        raw allocation is an owner (see :meth:`_scratch_policy`) and a swept
        prefix namespace, so an abandoned directory is collected on a later run
        rather than leaking forever.
        """
        store = self._run_scratch(request)
        if store.policy != "scoped":
            # Nothing deletes these, so a FIXED name is what keeps the folder
            # from growing a generation per send -- each run overwrites its own.
            return store.dir_path(name=name)
        # A scoped run deletes its own, so it takes a unique ROOT instead: two
        # hosts running at once must not share -- or delete -- one directory.
        # The readable names live inside it, so a root kept after a failure
        # still says what the run staged.
        root = request.extras.get(self._SCRATCH_ROOT_KEY)
        if root is None:
            root = store.dir_path()
            request.extras[self._SCRATCH_ROOT_KEY] = root
        target = os.path.join(root, name)
        os.makedirs(target, exist_ok=True)
        return target

    def _delivered_paths(self, result: Optional[Dict[str, Any]]) -> List[str]:
        """Paths this run DELIVERED, for the scoped-scratch guard.

        Any one of these inside the run scratch keeps the whole scratch: a
        bridge whose durable destination can fall back to the run's own output
        directory would otherwise delete the very files it just produced.
        Default: none, which is correct for every ``detached`` bridge (nothing
        is ever deleted) and for a scoped run that always relocates its output.
        """
        return []

    def _discard_scratch(self, request: HandoffRequest, result=None) -> None:
        """Remove this run's scratch, unless something it delivered lives in it."""
        store = request.extras.pop(self._SCRATCH_STORE_KEY, None)
        root = request.extras.pop(self._SCRATCH_ROOT_KEY, None)
        if store is None or store.policy != "scoped" or not root:
            return
        anchor = os.path.normcase(os.path.abspath(root))
        for path in self._delivered_paths(result):
            if not path:
                continue
            resolved = os.path.normcase(os.path.abspath(str(path)))
            if resolved == anchor or resolved.startswith(anchor + os.sep):
                self.logger.warning(
                    f"Scratch folder kept: this run's output landed inside it, "
                    f"so it holds the only copy. Move what you need somewhere "
                    f"permanent -- a later run sweeps this folder: {root}"
                )
                return
        if store.cleanup():
            self.logger.info(f"Staged hand-off artifacts cleaned up: {root}")

    @staticmethod
    def import_roots(*packages: str) -> List[str]:
        """``sys.path`` entries that make *packages* importable in a launched child app.

        A launched app does NOT inherit the parent's importable set. Blender in
        particular ignores ``PYTHONPATH`` outright (verified on 5.1: the variable is
        set in the child's environment and never reaches ``sys.path``), so a template
        that imports a toolkit to do post-import work silently degrades to its
        "toolkit unavailable" branch unless the roots are threaded in explicitly.

        Returns ONLY the roots for the named packages -- never the parent's whole
        ``sys.path``. A cross-version child (Maya's 3.11 -> Blender's 3.13) would
        otherwise get the parent's ``site-packages`` prepended, shadowing the child's
        own stdlib with binary-incompatible modules.

        Namespace-package aware: with a monorepo root on ``sys.path``, ``<repo>/pkg/``
        resolves as an EMPTY namespace module (``__file__ is None``) and its naive
        "parent directory" is the unusable repo root. When the spec has no real
        ``__init__.py`` this looks one level in for the actual package and returns
        that directory instead.
        """
        import os
        import importlib.util

        roots: List[str] = []
        for name in packages:
            try:
                spec = importlib.util.find_spec(name)
            except (ImportError, ValueError, ModuleNotFoundError):
                continue
            if spec is None:
                continue

            root = ""
            if spec.origin and os.path.isfile(spec.origin):
                # Real package: <root>/<name>/__init__.py -> <root>
                root = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
            else:
                for loc in list(getattr(spec, "submodule_search_locations", []) or []):
                    if os.path.isfile(os.path.join(loc, name, "__init__.py")):
                        root = os.path.abspath(loc)
                        break
            if root and root not in roots:
                roots.append(root)
        return roots

    # ------------------ Subclass hooks --------------------------------------
    def _resolve_objects(self, objects):  # pragma: no cover - subclass contract
        """Return the list of objects to export; ``None`` -> host selection."""
        raise NotImplementedError

    def _scene_objects(self) -> Optional[List[Any]]:
        """Everything a WHOLE-SCENE hand-off should carry, or ``None`` if unsupported.

        Only :meth:`ScriptLaunchBridge.save_as` uses it: "save the scene as ..." means
        the scene, not whatever happens to be selected. ``None`` (the default) means the
        host can't enumerate itself, and the caller falls back to the selection.
        """
        return None

    def _produce(
        self, objects, request: HandoffRequest
    ) -> Optional[Payload]:  # pragma: no cover
        """Build and return the :class:`Payload` (``None`` aborts the hand-off)."""
        raise NotImplementedError

    def _ingest(
        self,
        result: Dict[str, Any],
        objects: List[Any],
        payload: Payload,
        request: HandoffRequest,
    ) -> Optional[Dict[str, Any]]:
        """Bring the delivered result back into the host. Default: pass it through.

        The return leg of a round trip, and the ONLY step that touches host state
        *after* the target app has run. Overriders re-import the edited
        :attr:`Payload.primary`, transfer what came back onto *objects* (the same list
        ``_produce`` exported, so pairing needs no extra bookkeeping), clean up their
        scaffolding, and return an enriched result dict -- or ``None`` to report a
        handled, already-logged failure.

        Deliberately a hook rather than a Strategy: every real inbound leg is
        irreducibly host-specific, so there is no shared algorithm to inject and a
        strategy object would be indirection with a single implementation each.
        """
        return result


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
    # Seconds a BLOCKING run (:class:`ScriptRunDeliverer`) may take before the child is
    # killed; ``None`` = no limit. Ignored by the detached launch deliverer.
    timeout: Optional[float] = 600
    # Optional launch-time child-env factory (``None`` = inherit this process's env).
    # A callable, not a dict: the env must reflect launch-time state, and the spec is
    # built at import time. Bridges use it to keep source-app-private vars (e.g. an
    # ``OCIO`` pointing inside the source install -- see ``AppLauncher.handoff_env``)
    # from leaking into the target app.
    launch_env: Optional[Callable[[], Optional[Dict[str, str]]]] = None


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
        return (
            Path(self.spec.template_dir) / f"{template}{self.spec.template_extension}"
        )

    def _exe(self, bridge: HandoffBridge) -> Optional[str]:
        """The executable this deliverer runs (the interactive one)."""
        return bridge.app_path

    def _env(self, bridge: HandoffBridge) -> Optional[Dict[str, str]]:
        """The child env from the spec hook; ``None`` = inherit this process's.

        Sanitizing the child env is best-effort by contract: a failing hook must never
        cost the user the hand-off -- degrade to the inherited env.
        """
        if self.spec.launch_env is None:
            return None
        try:
            return self.spec.launch_env()
        except Exception:
            bridge.logger.warning(
                "launch_env hook failed; running with the inherited environment.",
                exc_info=True,
            )
            return None

    # When False (the launch route), an UNANNOTATED template is assumed to support the
    # spec's primary mode, so a custom template a user drops in just works. The blocking
    # route flips it: its template has to write a specific artifact, which an
    # unannotated script demonstrably does not do -- see :class:`ScriptRunDeliverer`.
    strict_modes: bool = False

    def _declared_modes(self, template_path: Path) -> Tuple[str, ...]:
        """Modes *template_path* offers, read per :attr:`strict_modes`."""
        if not template_path.is_file():
            return ()
        if self.strict_modes:
            return script_template.ScriptTemplate.declared_modes(template_path) or ()
        return script_template.ScriptTemplate.template_modes(
            template_path, self.spec.modes
        )

    def preflight(self, bridge: HandoffBridge, request: HandoffRequest) -> bool:
        spec = self.spec
        template_path = self._template_path(request.template)
        allowed = self._declared_modes(template_path)
        if request.mode not in allowed:
            bridge.logger.error(
                f"Template '{request.template}' does not support mode "
                f"'{request.mode}'. Declared: {allowed}"
            )
            return False
        if not self._exe(bridge):
            bridge.logger.error(spec.app.not_found_message)
            return False
        return True

    def deliver(
        self, bridge: HandoffBridge, payload: Payload, request: HandoffRequest
    ) -> Optional[Dict[str, Any]]:
        script = self.render(bridge, payload, request)
        if script is None:
            return None

        script_path = str(
            Path(payload.primary).with_suffix(self.spec.template_extension)
        )
        Path(script_path).write_text(script, encoding="utf-8")
        bridge.logger.info(
            f"Sending to {self.spec.app.name} ({request.template}) with script "
            f"{script_path}"
        )

        env = self._env(bridge)

        # FRESH instance every time -- never attach to a running session. Detached:
        # control returns immediately.
        proc = AppLauncher.launch(
            self._exe(bridge),
            args=self.spec.launch_args(script_path),
            detached=True,
            env=env,
        )
        if proc is None:
            bridge.logger.error(
                f"Failed to launch {self.spec.app.name}: {self._exe(bridge)}"
            )
            return None

        bridge.logger.info(f"Sent to {self.spec.app.name} (interactive session).")
        return {
            "script": script_path,
            "template": request.template,
            "mode": request.mode,
            "payload": payload.primary,
        }

    def _context(
        self, bridge: HandoffBridge, payload: Payload, request: HandoffRequest
    ) -> Dict[str, str]:
        """The ``__KEY__`` substitution context: the payload path + the bridge's params.

        The payload path is exposed under two tokens: ``PAYLOAD_PATH``, the
        carrier-neutral name, and ``FBX_PATH``, the name every template was written
        against before a payload could be anything but FBX. Both always hold the
        same path -- a USD payload arrives under ``FBX_PATH`` too -- so a template
        that must know the format reads the path's extension, not the token name.
        """
        primary = str(payload.primary).replace("\\", "/")
        context = {"PAYLOAD_PATH": primary, "FBX_PATH": primary}
        context.update(bridge.render_context(request.params))
        return context

    def render(
        self, bridge: HandoffBridge, payload: Payload, request: HandoffRequest
    ) -> Optional[str]:
        """Return the rendered script body for *request*'s template, or ``None`` on miss."""
        template_path = self._template_path(request.template)
        if not template_path.is_file():
            available = sorted(
                p.stem
                for p in script_template.ScriptTemplate.list_templates(
                    self.spec.template_dir, self.spec.template_extension
                )
            )
            bridge.logger.error(
                f"Template '{request.template}' not found at {template_path}. "
                f"Available: {available}"
            )
            return None
        return script_template.ScriptTemplate.render_template(
            template_path, self._context(bridge, payload, request)
        )


class ScriptRunDeliverer(ScriptLaunchDeliverer):
    """Render a template, run a **fresh** app on it ATTACHED, and keep what it wrote.

    The blocking sibling of :class:`ScriptLaunchDeliverer`: same template discovery,
    mode validation, rendering and env sanitizing (inherited) -- only the delivery
    differs. Instead of launching a detached GUI and returning immediately, this runs
    the target headlessly through :meth:`pythontk.ScriptRunner.run_script_to_artifact`
    and judges the hand-off by the **artifact** the script was asked to write (the exit
    code is advisory: a DCC that crashes in teardown after saving still succeeded).

    The artifact path rides on :attr:`HandoffRequest.extras` under ``"output"``; the
    rendered template receives it as ``__OUT_FILE__``. That is what turns a one-way
    "send" into "write me a file" -- and the file's *destination* is what distinguishes
    the two modes this serves. ``save_as`` hands it to the user (the target app's native
    scene format); ``round_trip`` hands it to :meth:`HandoffBridge._ingest`, which folds
    it back into the host and leaves the artist with changed scene state rather than a
    deliverable. Identical mechanics, so one deliverer covers both: a bridge registers
    whichever modes apply via its spec's ``modes=``.

    Mode declarations are read STRICTLY here: the template must name the mode itself.
    The lenient reading the launch route uses would green-light any template -- including
    the interactive ``import`` recipe, which has no ``__OUT_FILE__`` and never saves, so
    the run would fail on the missing artifact minutes later instead of in preflight.
    """

    strict_modes = True

    def _exe(self, bridge: HandoffBridge) -> Optional[str]:
        """The HEADLESS executable (``mayapy`` where the GUI target is ``maya.exe``)."""
        return bridge.headless_app_path

    def _context(
        self, bridge: HandoffBridge, payload: Payload, request: HandoffRequest
    ) -> Dict[str, str]:
        context = super()._context(bridge, payload, request)
        # The template is pointed at the STAGING sibling, never the caller's path --
        # see :meth:`deliver`. Recomputed rather than threaded through the request so
        # there is one definition of where the child writes.
        output = request.get("output")
        staged = self._staging_path(output) if output else ""
        context["OUT_FILE"] = str(staged).replace("\\", "/")
        return context

    @staticmethod
    def _staging_path(artifact: str) -> str:
        """A hidden sibling of *artifact* with the SAME extension.

        Same directory so promoting it is an atomic same-filesystem
        :func:`os.replace`; same extension because templates branch on it (a ``.mb``
        target must still be written as mayaBinary). A leftover from a killed run is
        cleared by the runner before the next one.
        """
        path = Path(artifact)
        return str(path.with_name(f".{path.stem}.saving{path.suffix}"))

    # Seam for tests: stub the headless run without patching pythontk internals.
    @staticmethod
    def run(app_exe, script_text, *, artifact, launch_args, timeout, env=None, expect=None):
        from pythontk.core_utils import script_run

        return script_run.ScriptRunner.run_script_to_artifact(
            app_exe,
            script_text,
            artifact=artifact,
            launch_args=launch_args,
            timeout=timeout,
            env=env,
            script_prefix="handoff_run",
            expect=script_run.CREATED if expect is None else expect,
        )

    def _timeout(self, request: HandoffRequest) -> Optional[float]:
        """Per-request timeout override, else the spec's."""
        timeout = request.get("timeout")
        return self.spec.timeout if timeout is None else timeout

    def deliver(
        self, bridge: HandoffBridge, payload: Payload, request: HandoffRequest
    ) -> Optional[Dict[str, Any]]:
        artifact = request.get("output")
        if not artifact:
            bridge.logger.error(
                f"Mode '{request.mode}' needs an output path (request.extras['output'])."
            )
            return None

        script = self.render(bridge, payload, request)
        if script is None:
            return None

        os.makedirs(os.path.dirname(os.path.abspath(artifact)) or ".", exist_ok=True)
        timeout = self._timeout(request)
        bridge.logger.info(
            f"Running {self.spec.app.name} headlessly ({request.template}) -> {artifact}"
        )
        # The target app writes to a staging sibling, promoted only on success. The
        # runner CLEARS the artifact path before running (a leftover would fake
        # success), which for a save-over is destructive: a failed run would take the
        # user's existing scene with it. Same directory, so the promotion is an atomic
        # same-filesystem replace; the extension is preserved because templates branch
        # on it (``.mb`` -> mayaBinary).
        staging = self._staging_path(artifact)
        try:
            result = self.run(
                self._exe(bridge),
                script,
                artifact=staging,
                launch_args=self.spec.launch_args,
                timeout=timeout,
                env=self._env(bridge),
            )
        except Exception as error:  # noqa: BLE001 - reported, never raised at the caller
            # A killed (timeout) or half-finished child can leave a partial file, and
            # an EMPTY one is a failure the runner reports without removing. Neither
            # may be left sitting in the user's output folder.
            try:
                os.remove(staging)
            except OSError:
                pass
            bridge.logger.error(
                f"{self.spec.app.name} did not produce {artifact}: {error}"
            )
            return None

        try:
            os.replace(staging, artifact)
        except OSError as error:
            # The scene WAS written and only the promotion failed (target open in
            # another app, read-only, ...). The staged file is KEPT and named: the
            # result the user waited minutes for must not be discarded over a rename.
            bridge.logger.error(
                f"{self.spec.app.name} saved successfully but {artifact} could not be "
                f"replaced ({error}). The result is at {staging}."
            )
            return None

        bridge.logger.info(
            f"Saved {artifact} in {result.duration:.1f}s "
            f"({os.path.getsize(artifact) // 1024} KB)."
        )
        return {
            "output": artifact,
            "template": request.template,
            "mode": request.mode,
            "payload": payload.primary,
            "duration": result.duration,
            "returncode": result.returncode,
        }


class ScriptRoundTripDeliverer(ScriptRunDeliverer):
    """Run a **fresh** app headlessly on the payload and let it edit that file in place.

    The delivery half of a round trip. Same blocking run as
    :class:`ScriptRunDeliverer`, but the artifact *is* :attr:`Payload.primary` -- the
    target loads the exported file, edits it, and saves back over it (RizomUV's
    ``ZomLoad``/``ZomSave`` is the canonical shape). Two consequences the save-as route
    does not have:

    * **No staging, no promotion.** The input and the output are one path, so there is
      nothing to promote and a hidden sibling would just be a copy nobody reads. The
      file is a temp payload the bridge owns, so a partial write costs nothing -- unlike
      a save-over of the user's own scene, which is exactly what staging protects.
    * **Judged by CHANGE, not creation.** Clearing the path first would delete the app's
      own input. An app that exits 0 without reaching its save call leaves the file
      untouched, and calling that a success would hand the caller back its own
      unmodified export to re-ingest as though it had been processed -- a silent no-op
      that looks like a working round trip. So the runner is asked for
      :data:`~pythontk.core_utils.script_run.REWRITTEN`.

    The result dict is what :meth:`HandoffBridge._ingest` receives; ``artifact`` is the
    edited file to re-import.
    """

    def _context(
        self, bridge: HandoffBridge, payload: Payload, request: HandoffRequest
    ) -> Dict[str, str]:
        # Skip ScriptRunDeliverer's staging-sibling OUT_FILE: here the target reads and
        # writes the one payload path, which the base already exposes as
        # __PAYLOAD_PATH__ / __FBX_PATH__.
        context = ScriptLaunchDeliverer._context(self, bridge, payload, request)
        context["OUT_FILE"] = str(payload.primary or "").replace("\\", "/")
        return context

    def deliver(
        self, bridge: HandoffBridge, payload: Payload, request: HandoffRequest
    ) -> Optional[Dict[str, Any]]:
        from pythontk.core_utils.script_run import REWRITTEN

        artifact = payload.primary
        if not artifact:
            bridge.logger.error(
                f"Mode '{request.mode}' needs a payload for the target to edit in place."
            )
            return None

        script = self.render(bridge, payload, request)
        if script is None:
            return None

        bridge.logger.info(
            f"Running {self.spec.app.name} headlessly ({request.template}) on {artifact}"
        )
        try:
            result = self.run(
                self._exe(bridge),
                script,
                artifact=artifact,
                launch_args=self.spec.launch_args,
                timeout=self._timeout(request),
                env=self._env(bridge),
                expect=REWRITTEN,
            )
        except Exception as error:  # noqa: BLE001 - reported, never raised at the caller
            # The payload is deliberately KEPT on failure: the rendered script and the
            # file it choked on are the only evidence of what went wrong, and the temp
            # namespace's age-gated sweep reclaims them either way.
            bridge.logger.error(f"{self.spec.app.name} did not process {artifact}: {error}")
            return None

        bridge.logger.info(f"{self.spec.app.name} finished in {result.duration:.1f}s.")
        return {
            "artifact": artifact,
            "template": request.template,
            "mode": request.mode,
            "payload": payload.primary,
            "duration": result.duration,
            "returncode": result.returncode,
        }


class ScriptLaunchBridge(HandoffBridge):
    """A :class:`HandoffBridge` whose delivery is :class:`ScriptLaunchDeliverer`.

    Subclasses set the :attr:`spec` (:class:`ScriptLaunchSpec`) and implement
    :meth:`render_context` (+ the :meth:`_resolve_objects` / :meth:`_produce` hooks,
    typically via a DCC export mixin). The deliverer and discovery are wired from the
    spec, so a concrete bridge is just data + the DCC-specific export.
    """

    spec: Optional[ScriptLaunchSpec] = None
    # Optional second spec for the BLOCKING route (headless argv, and a different
    # executable where the target has one). Set it and the bridge gains
    # :meth:`save_as`; leave it None and the bridge stays send-only.
    run_spec: Optional[ScriptLaunchSpec] = None
    # Optional third spec for the ROUND-TRIP route: the target edits the exported
    # payload IN PLACE and the bridge re-ingests it. Set it (plus an :meth:`_ingest`
    # override) and the bridge gains :meth:`round_trip`.
    #
    # Not the only way to get that method: a round trip whose target writes a NEW
    # intermediate instead (mayatk's lightmap bake returns a manifest) is mechanically
    # a ``save_as`` run, so it declares ``modes=(SAVE_AS, ROUND_TRIP)`` on `run_spec`
    # and leaves this None. `round_trip` gates on the deliverer registry, not on this
    # attribute -- the mode says where the result lands, the spec says how it is written.
    round_trip_spec: Optional[ScriptLaunchSpec] = None
    # Template + accepted extensions for :meth:`save_as` (the first is the default
    # appended when the caller's path has none).
    save_template: str = "_save_scene"
    save_extensions: Tuple[str, ...] = ()

    def __init__(self, app_path: Optional[str] = None):
        super().__init__(app_path=app_path)
        if self.spec is None:
            raise TypeError(
                f"{type(self).__name__} must set a ScriptLaunchSpec `spec`."
            )
        self.app_spec = self.spec.app
        self.payload_prefix = self.spec.payload_prefix
        self.deliverer = ScriptLaunchDeliverer(self.spec)
        # Per-mode registry (instance-owned: a class-level dict would leak one
        # bridge's strategies into every other).
        self.deliverers = {mode: self.deliverer for mode in self.spec.modes}
        for attr, strategy in (
            ("run_spec", ScriptRunDeliverer),
            ("round_trip_spec", ScriptRoundTripDeliverer),
        ):
            spec = getattr(self, attr)
            if spec is None:
                continue
            # A secondary spec that forgets `modes=` inherits ScriptLaunchSpec's
            # default `(SEND_TO,)` and would REPLACE the interactive send deliverer
            # here -- send() would then run the target headlessly and report a
            # missing artifact, with nothing pointing at the one-word omission that
            # caused it. Registering a mode twice is always a declaration bug, so
            # refuse at construction rather than mis-dispatch at send time.
            clash = sorted(set(spec.modes) & set(self.deliverers))
            if clash:
                raise ValueError(
                    f"{type(self).__name__}.{attr} claims mode(s) {clash}, already "
                    f"served by another spec. Set `modes=` on it (e.g. "
                    f"`modes=({SAVE_AS!r},)`)."
                )
            deliverer = strategy(spec)
            self.deliverers.update({mode: deliverer for mode in spec.modes})

    def render_context(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Format *params* into a ``__KEY__`` substitution context.

        Plain Python source literals -- ``repr`` renders bool / str / number / tuple
        exactly as a ``.py`` template needs them, which is what every template in the
        ecosystem is. Subclasses override to reach a richer, spec-aware formatter (or a
        different target language), and can call back here as the dependency-free
        fallback when that formatter's UI toolkit is unavailable (a DCC running
        headless).
        """
        return {key: repr(value) for key, value in params.items()}

    # ------------------ save_as (write the TARGET app's native format) ------
    def save_as(
        self,
        out_path: str,
        objects: Optional[List[Any]] = None,
        *,
        template: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **extras: Any,
    ) -> Optional[Dict[str, Any]]:
        """Write *out_path* in the TARGET app's native scene format (blocking).

        The same hand-off the interactive ``send()`` performs -- identical FBX export,
        identical material sidecar -- delivered to a **headless** instance of the target
        that imports the payload and saves the result. So a Maya artist gets a real
        ``.blend`` (and a Blender artist a real ``.ma``) without either app being open
        for it, and without a second export pipeline to maintain.

        Unlike ``send()``, ``objects=None`` means the WHOLE SCENE (via
        :meth:`_scene_objects`) -- "save the scene as ..." is about the scene, not the
        selection. Pass an explicit list to save just those.

        Returns the deliverer's result dict (``output`` / ``duration`` / ``returncode``)
        or ``None`` on a handled, already-logged failure. Requires :attr:`run_spec`.
        """
        if self.run_spec is None:
            self.logger.error(
                f"{type(self).__name__} has no `run_spec`; save_as is unavailable."
            )
            return None
        if objects is None:
            objects = self._scene_objects()  # None -> the host selection

        request = HandoffRequest(
            template=template or self.save_template,
            mode=SAVE_AS,
            params=self.merge_params(params),
            extras={
                "output": self.resolve_save_path(out_path),
                "timeout": timeout,
                **extras,
            },
        )
        return self._run(objects, request)

    # ------------------ round_trip (send it out, bring the result back) -----
    def round_trip(
        self,
        objects: Optional[List[Any]] = None,
        *,
        template: str = "import",
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        out: Optional[str] = None,
        **extras: Any,
    ) -> Optional[Dict[str, Any]]:
        """Export *objects*, let the target app work on them, and re-ingest the result.

        The third hand-off shape. ``send()`` is one-way and ``save_as()`` writes a file
        for the user; this one closes the loop: the same export pipeline feeds a
        **headless** target, and :meth:`_ingest` brings the result back onto the host's
        own objects (transfer, clean up). That is what makes "unwrap these in RizomUV"
        or "bake these lightmaps in Blender" a mode of the shared skeleton rather than
        a parallel pipeline with its own discovery, preflight and logging.

        What the caller gets back is CHANGED HOST STATE, never a file to go looking for
        -- which is the whole reason this is not spelled ``save_as``. Two artifact
        shapes ride the one mode, chosen by which deliverer is registered for
        :data:`ROUND_TRIP`:

        * :class:`ScriptRoundTripDeliverer` -- the target edits the exported payload in
          place. Leave *out* unset; the payload path is the artifact.
        * :class:`ScriptRunDeliverer` -- the target writes a NEW intermediate (a
          manifest, a map set) that :meth:`_ingest` reads. Pass *out*, exactly as
          ``save_as`` does; the difference is purely what happens to the file after.

        Requires a deliverer registered for the mode (a ``modes=`` entry on
        :attr:`round_trip_spec` or :attr:`run_spec`) and an :meth:`_ingest` override --
        without the latter the result is delivered and then simply reported, which is a
        silent no-op rather than a round trip.

        Returns whatever :meth:`_ingest` returns, or ``None`` on a handled,
        already-logged failure.
        """
        if ROUND_TRIP not in (self.deliverers or {}):
            # Gated on the DELIVERER, not on `round_trip_spec`: a bridge whose round
            # trip returns a new artifact registers the mode on `run_spec` (same
            # blocking machinery as save_as), and has no round_trip_spec at all.
            self.logger.error(
                f"{type(self).__name__} has no deliverer for '{ROUND_TRIP}'; "
                "round_trip is unavailable. Declare it on `round_trip_spec` (the "
                "target edits the payload in place) or on `run_spec` (the target "
                "writes a new artifact), via that spec's `modes=`."
            )
            return None
        if type(self)._ingest is HandoffBridge._ingest:
            # The target WILL run and edit the payload; the default identity ingest
            # then throws that away and reports success. Warn rather than refuse --
            # the work is already worth reporting by the time this could be detected
            # on the far side, and the run itself is not wrong, only pointless.
            self.logger.warning(
                f"{type(self).__name__} does not override `_ingest`; the round trip "
                "will run and discard its result."
            )
        request = HandoffRequest(
            template=template,
            mode=ROUND_TRIP,
            params=self.merge_params(params),
            extras={
                "timeout": timeout,
                # Normalized the same way ``save_as`` normalizes its destination, so the
                # artifact-writing shape reaches the deliverer with an identical
                # contract. ``None`` for the in-place shape, which reads the payload.
                "output": self.resolve_save_path(out) if out else None,
                **extras,
            },
        )
        return self._run(objects, request)

    @classmethod
    def resolve_save_path(cls, out_path: str) -> str:
        """Absolute *out_path*, with :attr:`save_extensions`' default appended if bare."""
        path = os.path.abspath(os.path.expanduser(os.path.expandvars(str(out_path))))
        if cls.save_extensions and not path.lower().endswith(cls.save_extensions):
            path += cls.save_extensions[0]
        return path

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
    @property
    def modes(self) -> Tuple[str, ...]:
        """Every mode this bridge can deliver -- the registry, not one spec's slice.

        Derived rather than restated, because the discovery helpers filter a template's
        declaration against this and silently fall back to ``[0]`` for anything outside
        it. A list that omits a mode some spec DOES serve therefore relabels that
        template as the primary one, and the panel then routes a blocking recipe through
        the detached launch: no ``__OUT_FILE__``, an app that opens and fails minutes
        later, and nothing pointing at the omission. Insertion order puts
        :attr:`spec`'s modes first, so ``[0]`` stays the lenient fallback.
        """
        return tuple(self.deliverers or ()) or tuple(self.spec.modes)

    def list_template_modes(self) -> List[Tuple[str, str]]:
        """``[(stem, mode), ...]`` for the bridge's template directory."""
        return script_template.ScriptTemplate.list_template_modes(
            self.spec.template_dir, self.spec.template_extension, self.modes
        )

    def list_templates(self) -> List[Path]:
        """User-visible template paths for the bridge."""
        return script_template.ScriptTemplate.list_templates(
            self.spec.template_dir, self.spec.template_extension
        )


__all__ = [
    "SEND_TO",
    "SAVE_AS",
    "ROUND_TRIP",
    "CARRIER_PARAM",
    "CARRIER_EXTENSIONS",
    "CARRIER_BY_EXTENSION",
    "AppSpec",
    "HandoffRequest",
    "Payload",
    "Deliverer",
    "HandoffBridge",
    "ScriptLaunchSpec",
    "ScriptLaunchDeliverer",
    "ScriptRunDeliverer",
    "ScriptRoundTripDeliverer",
    "ScriptLaunchBridge",
]
