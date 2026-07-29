# !/usr/bin/python
# coding=utf-8
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

from pythontk.core_utils.app_launcher import AppLauncher
from pythontk.core_utils.help_mixin import HelpMixin

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600  # Dense production meshes; matches the RizomUV bridge ceiling.

# ---------------------------------------------------------------- Ministry of Flat
# Discovery-only: the free license permits local commercial use but forbids
# redistribution, so the executable is never downloaded on the user's behalf.
MOF_DOWNLOAD_URL = "https://www.quelsolaar.com/ministry_of_flat/"
MOF_EXE = "UnWrapConsole3"

# ------------------------------------------------------- Boundary First Flattening
# MIT licensed with versioned release assets -> managed install (FBX2glTF pattern).
BFF_VERSION = "1.6"
BFF_URL = (
    "https://github.com/GeometryCollective/boundary-first-flattening"
    f"/releases/download/v{BFF_VERSION}/windows-v{BFF_VERSION}.zip"
)
BFF_PLATFORMS = {
    "windows": {"url": BFF_URL, "type": "zip", "executable": "bff-command-line"},
}
BFF_SHA256 = {
    "windows": "8d78da327524e6f73fbf1a622f6cb13cf2ee4906a56655d977ed0b7ae4366942"
}
BFF_DOWNLOAD_URL = (
    "https://github.com/GeometryCollective/boundary-first-flattening/releases"
)


class _UvUnwrapInternal:
    """Internal base carrying the per-engine argv builders and scan paths.

    These are :data:`ENGINES` data -- each is referenced as an ``EngineSpec``
    field, so they must exist before the registry is built and cannot live on
    :class:`UvUnwrap` itself (which is defined after it). Keeping them here
    rather than as module functions holds the package's helpers-on-a-class
    rule; ``UvUnwrap`` inherits them alongside its own private staticmethods.
    """

    @staticmethod
    def _tf(value: Any) -> str:
        """Ministry of Flat's boolean token form."""
        return "TRUE" if value else "FALSE"

    @staticmethod
    def _build_mof_args(
        obj_in: str, obj_out: str, params: Dict[str, Any]
    ) -> List[str]:
        """Ministry of Flat argv: ``<in.obj> <out.obj> [-FLAG <VALUE>]...``.

        Only the documented non-debug settings are exposed. Ministry of Flat's own
        documentation warns that changing anything past that block "is likely to
        result in worse UV mapping and or longer processing time", and it has no
        hard-surface/organic switch at all -- classification is automatic.
        """
        _tf = _UvUnwrapInternal._tf
        args = [obj_in, obj_out]
        for key, flag, coerce in (
            ("resolution", "-RESOLUTION", lambda v: str(int(v))),
            ("aspect", "-ASPECT", lambda v: str(float(v))),
            ("udims", "-UDIMS", lambda v: str(int(v))),
            ("density", "-DENSITY", lambda v: str(int(v))),
            ("separate_hard_edges", "-SEPARATE", _tf),
            ("use_normals", "-NORMALS", _tf),
            ("overlap_identical", "-OVERLAP", _tf),
            ("overlap_mirrored", "-MIRROR", _tf),
            ("world_scale", "-WORLDSCALE", _tf),
        ):
            if key in params:
                args += [flag, coerce(params[key])]
        return args

    @staticmethod
    def _build_bff_args(
        obj_in: str, obj_out: str, params: Dict[str, Any]
    ) -> List[str]:
        """BFF argv: ``<in.obj> <out.obj> [--nCones=N] [--normalizeUVs]``."""
        args = [obj_in, obj_out]
        if "n_cones" in params:
            args.append(f"--nCones={int(params['n_cones'])}")
        if params.get("normalize_uvs", True):
            # Keeps UVs in [0,1] so the DCC-side layout pass starts from a sane range.
            args.append("--normalizeUVs")
        return args

    @staticmethod
    def _mof_scan_globs() -> Tuple[str, ...]:
        """Manual-install locations searched for Ministry of Flat.

        Pure path-string construction (no I/O), so evaluating at import is free of
        side effects. ``~/.pythontk/tools/mof/`` gives users a canonical drop spot
        that works in PATH-less GUI hosts.
        """
        tools = os.path.join(os.path.expanduser("~"), ".pythontk", "tools")
        exe = MOF_EXE + ".exe"
        return (
            os.path.join(tools, "mof", exe),
            os.path.join(tools, "mof", "**", exe),
            r"{program_files}\MinistryOfFlat*\%s" % exe,
            r"{program_files}\Ministry of Flat*\%s" % exe,
        )


@dataclass(frozen=True)
class EngineSpec:
    """Everything :class:`UvUnwrap` needs to drive one external unwrapper.

    Adding an engine is one entry in :data:`ENGINES` -- no edits to the
    resolve/run/validate logic.
    """

    name: str
    label: str
    exe_names: Tuple[str, ...]
    env_var: str
    build_args: Callable[[str, str, Dict[str, Any]], List[str]]
    allowed_params: FrozenSet[str]
    download_url: str
    scan_globs: Tuple[str, ...] = ()
    install: Optional[Dict[str, Any]] = None  # None -> never auto-downloaded
    failure_hint: str = ""
    install_note: str = ""
    # Whether the engine arranges its own islands. Consumers use this to decide
    # between re-packing the result and merely scaling the engine's layout.
    packs_own_layout: bool = False


ENGINES: Dict[str, EngineSpec] = {
    "mof": EngineSpec(
        name="mof",
        label="Ministry of Flat",
        exe_names=(MOF_EXE,),
        env_var="PYTHONTK_MOF_EXE",
        build_args=_UvUnwrapInternal._build_mof_args,
        allowed_params=frozenset(
            {
                "resolution",
                "aspect",
                "udims",
                "density",
                "separate_hard_edges",
                "use_normals",
                "overlap_identical",
                "overlap_mirrored",
                "world_scale",
            }
        ),
        download_url=MOF_DOWNLOAD_URL,
        scan_globs=_UvUnwrapInternal._mof_scan_globs(),
        install=None,
        failure_hint=(
            "Ministry of Flat requires good topology; degenerate or "
            "non-manifold geometry yields poor results."
        ),
        install_note=(
            "Ministry of Flat cannot be installed automatically -- its license "
            "forbids redistribution. Download it once, then either put "
            f"{MOF_EXE}.exe on PATH, set %s, or drop it in "
            "~/.pythontk/tools/mof/."
        ),
        packs_own_layout=True,  # returns a packed atlas (in a rectangle, not 0-1)
    ),
    "bff": EngineSpec(
        name="bff",
        label="Boundary First Flattening",
        exe_names=("bff-command-line",),
        env_var="PYTHONTK_BFF_EXE",
        build_args=_UvUnwrapInternal._build_bff_args,
        allowed_params=frozenset({"n_cones", "normalize_uvs"}),
        download_url=BFF_DOWNLOAD_URL,
        install={
            "tool_name": "bff",
            "version": BFF_VERSION,
            "platforms": BFF_PLATFORMS,
            "sha256": BFF_SHA256,
            "executable": "bff-command-line",
        },
        failure_hint=(
            "BFF flattens one connected surface at a time; check the mesh for "
            "degenerate faces."
        ),
        install_note="Pass auto_install=True to download it (MIT licensed).",
    ),
}


class UvUnwrap(HelpMixin, _UvUnwrapInternal):
    """Automatic UV unwrapping via external CLI engines (OBJ in -> OBJ out).

    Engines:
        ``"mof"`` -- Ministry of Flat, the hard-surface path. Topology-aware
            segmentation; output arrives already packed. Discovery-only: the
            free license forbids redistribution, so the user installs it.
        ``"bff"`` -- Boundary First Flattening, the organic path. Conformal
            flattening with automatic cone singularities (MIT). Auto-installable
            from a pinned GitHub release. Output is flattened but not packed --
            the caller lays it out.

    Both engines accept n-gons and return the input topology unchanged (same
    vertex count, face count and winding), so a caller can map UVs back by
    component index rather than by spatial sampling.

    Success is judged by the *output file*, not the exit code: Ministry of Flat
    returns 1 even on a completely successful run, so a returncode check would
    reject every result it produces.
    """

    ENGINES = ENGINES
    DEFAULT_TIMEOUT = DEFAULT_TIMEOUT
    TEMP_PREFIX = "ptk_uvunwrap"

    # Product-facing names for the two paths, mapped to engine keys. Consumers
    # resolve through :meth:`resolve_method` rather than keeping their own copy,
    # so adding an engine is still a single-table change.
    METHODS = {"hard": "mof", "organic": "bff"}

    @classmethod
    def resolve_method(cls, method: str) -> str:
        """Map ``"hard"`` / ``"organic"`` to an engine key (keys pass through).

        Raises:
            ValueError: *method* names neither a method nor an engine.
        """
        engine = cls.METHODS.get(method, method)
        if engine not in cls.ENGINES:
            raise ValueError(
                f"Unknown unwrap method {method!r}. Expected one of "
                f"{sorted(cls.METHODS)} (or an engine key: {sorted(cls.ENGINES)})."
            )
        return engine

    # ------------------------------------------------------------- discovery

    @classmethod
    def available_engines(cls) -> Dict[str, Optional[str]]:
        """Map each engine name to its resolved executable path, or None.

        Never installs, never prompts, never raises -- safe to poll for UI
        state.
        """
        found = {}
        for name in cls.ENGINES:
            try:
                found[name] = cls.resolve_engine(name, required=False)
            except Exception as exc:  # noqa: BLE001 - polling must never raise
                logger.debug("engine probe failed for %r: %s", name, exc)
                found[name] = None
        return found

    @classmethod
    def resolve_engine(
        cls,
        engine: str,
        required: bool = True,
        auto_install: bool = False,
        prompt: bool = True,
    ) -> Optional[str]:
        """Resolve one engine's executable.

        Order: the engine's env-var override, then PATH / Windows App Paths,
        then its install-dir globs, then the pythontk-managed tool catalog, and
        finally -- for installable engines only, with *auto_install* -- a
        download.

        Parameters:
            engine:       Engine key (``"mof"`` / ``"bff"``).
            required:     Raise FileNotFoundError when unresolved.
            auto_install: Permit downloading an installable engine.
            prompt:       Ask before downloading (TTY only; a GUI host refuses
                          rather than downloading silently).

        Returns:
            Absolute path to the executable, or None.
        """
        spec = cls._spec(engine)

        explicit = os.environ.get(spec.env_var)
        if explicit and os.path.isfile(explicit):
            return explicit

        found = AppLauncher.resolve_app_path(
            app_names=spec.exe_names, scan_globs=spec.scan_globs
        )
        if found:
            return found

        from pythontk.core_utils.app_installer import AppInstaller

        if spec.install:
            managed = AppInstaller.get_path(
                spec.install["tool_name"],
                executable=spec.install["executable"],
                add_to_path=True,
            )
            if managed:
                return managed

        if not spec.install or not auto_install:
            if required:
                raise FileNotFoundError(cls._not_found_message(spec))
            return None

        if prompt:
            if not (sys.stdin and sys.stdin.isatty()):
                # GUI host / CI: refuse rather than silently pulling a binary.
                if required:
                    raise FileNotFoundError(
                        f"{spec.label} is not installed and no interactive "
                        "console is available to confirm the download. Pass "
                        "prompt=False to install non-interactively."
                    )
                return None
            sys.stdout.write(
                f"\n{spec.label} v{spec.install['version']} is not installed. "
                "Download to ~/.pythontk/tools/ now? [y/N] "
            )
            sys.stdout.flush()
            if sys.stdin.readline().strip().lower() not in ("y", "yes"):
                if required:
                    raise FileNotFoundError(
                        f"User declined {spec.label} installation."
                    )
                return None

        try:
            return AppInstaller.ensure(
                spec.install["tool_name"],
                platforms=spec.install["platforms"],
                executable=spec.install["executable"],
                version=spec.install["version"],
                sha256=spec.install["sha256"],
            )
        except (RuntimeError, OSError, LookupError) as exc:
            if required:
                raise
            logger.warning("%s install failed: %s", spec.label, exc)
            return None

    # ------------------------------------------------------------- execution

    @classmethod
    def unwrap(
        cls,
        obj_in: str,
        obj_out: Optional[str] = None,
        *,
        engine: str = "mof",
        overwrite: bool = False,
        auto_install: bool = True,
        prompt: bool = True,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        **params,
    ) -> str:
        """Unwrap *obj_in* with *engine*; return the output OBJ path.

        Parameters:
            obj_in:       Input Wavefront OBJ path.
            obj_out:      Output path. None allocates a temp path that outlives
                          this call (the caller reads it afterwards).
            engine:       ``"mof"`` (hard surface) or ``"bff"`` (organic).
            overwrite:    Replace an existing *obj_out*.
            auto_install: Permit downloading an installable engine.
            prompt:       Ask before downloading.
            timeout:      Seconds before the engine is killed. None disables.
            params:       Engine-specific settings; see the engine's
                          ``allowed_params``. Unknown names raise TypeError.

        Returns:
            Absolute path to the written OBJ.
        """
        spec = cls._spec(engine)
        src = cls._preflight(obj_in)
        cls._check_params(spec, params)

        if obj_out is None:
            dst = cls._temp_output()
        else:
            dst = os.path.abspath(obj_out)
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

        if os.path.exists(dst):
            if not overwrite:
                raise FileExistsError(
                    f"UV unwrap output already exists: {dst}. "
                    "Pass overwrite=True to replace."
                )
            os.remove(dst)  # Delete-first: existence alone then proves the run.

        exe = cls.resolve_engine(
            engine, required=True, auto_install=auto_install, prompt=prompt
        )
        argv = spec.build_args(src, dst, params)
        logger.debug("%s: %s %s", spec.label, exe, " ".join(argv))

        try:
            result = AppLauncher.run(exe, args=argv, timeout=timeout, hide_window=True)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{spec.label} did not finish within {timeout}s and was killed. "
                "Pass a larger timeout= for dense meshes."
            ) from exc

        cls._postflight(spec, dst, result)
        return dst

    @classmethod
    def hard_surface(cls, obj_in: str, obj_out: Optional[str] = None, **kwargs) -> str:
        """Unwrap with Ministry of Flat -- the hard-surface path."""
        kwargs["engine"] = cls.METHODS["hard"]
        return cls.unwrap(obj_in, obj_out, **kwargs)

    @classmethod
    def organic(cls, obj_in: str, obj_out: Optional[str] = None, **kwargs) -> str:
        """Unwrap with Boundary First Flattening -- the organic path."""
        kwargs["engine"] = cls.METHODS["organic"]
        return cls.unwrap(obj_in, obj_out, **kwargs)

    # -------------------------------------------------------------- internal

    @classmethod
    def _spec(cls, engine: str) -> EngineSpec:
        try:
            return cls.ENGINES[engine]
        except KeyError:
            raise ValueError(
                f"Unknown unwrap engine {engine!r}. "
                f"Available: {sorted(cls.ENGINES)}"
            ) from None

    @staticmethod
    def _not_found_message(spec: EngineSpec) -> str:
        note = spec.install_note
        if "%s" in note:
            note = note % spec.env_var
        return (
            f"{spec.label} executable not found (searched "
            f"{', '.join(spec.exe_names)} on PATH and the usual install "
            f"locations).\n{note}\nDownload: {spec.download_url}"
        )

    @staticmethod
    def _preflight(obj_in: str) -> str:
        src = os.path.abspath(obj_in)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"OBJ source not found: {src}")
        if os.path.splitext(src)[1].lower() != ".obj":
            raise ValueError(f"Expected a .obj input, got: {src}")
        if os.path.getsize(src) == 0:
            raise RuntimeError(f"OBJ source is empty: {src}")
        return src

    @staticmethod
    def _check_params(spec: EngineSpec, params: Dict[str, Any]) -> None:
        unknown = set(params) - spec.allowed_params
        if unknown:
            raise TypeError(
                f"{spec.label}: unsupported parameter(s) {sorted(unknown)}. "
                f"Supported: {sorted(spec.allowed_params)}"
            )

    _temp_artifacts = None

    @classmethod
    def _temp_output(cls) -> str:
        """A temp OBJ path that outlives this call.

        ``"detached"`` is the only sound policy here: the caller reads the file
        after ``unwrap`` returns, so nothing can signal completion -- stale
        files are reclaimed by the prefix sweep on a later allocation instead.
        The allocator is kept because that sweep runs once per *instance*: a
        fresh one per call would rescan the temp directory on every unwrap.
        """
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        if cls._temp_artifacts is None:
            cls._temp_artifacts = TempArtifacts(cls.TEMP_PREFIX, policy="detached")
        return cls._temp_artifacts.path(extension=".obj")

    @staticmethod
    def _has_uvs(obj_path: str) -> bool:
        """True when the OBJ carries at least one texture coordinate."""
        with open(obj_path, "r", encoding="utf-8", errors="replace") as f:
            return any(line.startswith("vt ") for line in f)

    @staticmethod
    def _tail(text: Optional[str], limit: int = 2048) -> str:
        if not text:
            return ""
        text = text.strip()
        return text if len(text) <= limit else "..." + text[-limit:]

    @classmethod
    def _postflight(cls, spec: EngineSpec, dst: str, result) -> None:
        """Validate the engine's output; raise with diagnostics on failure.

        The output file -- not ``returncode`` -- is the success gate. Ministry
        of Flat exits 1 after writing a perfectly good result, so the exit code
        is only ever reported as context for a failure detected here.
        """
        if os.path.isfile(dst) and os.path.getsize(dst) > 0 and cls._has_uvs(dst):
            if getattr(result, "returncode", 0) != 0:
                logger.debug(
                    "%s exited %s but produced valid UVs.",
                    spec.label,
                    result.returncode,
                )
            return

        if not os.path.isfile(dst):
            problem = "wrote no output file"
        elif os.path.getsize(dst) == 0:
            problem = "wrote an empty output file"
        else:
            problem = "produced an output file containing no UVs"

        detail = [
            f"{spec.label} {problem}.",
            f"  exit code: {getattr(result, 'returncode', '?')}",
            f"  output:    {dst}",
        ]
        out, err = cls._tail(getattr(result, "stdout", "")), cls._tail(
            getattr(result, "stderr", "")
        )
        if out:
            detail.append(f"  stdout: {out}")
        if err:
            detail.append(f"  stderr: {err}")
        if not out and not err:
            detail.append("  (no captured output)")
        if spec.failure_hint:
            detail.append(f"  hint: {spec.failure_hint}")
        raise RuntimeError("\n".join(detail))
