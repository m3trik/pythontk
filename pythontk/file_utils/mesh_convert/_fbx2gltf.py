# !/usr/bin/python
# coding=utf-8
"""The FBX2glTF CLI driver behind :meth:`MeshConvert.fbx_to_glb`, and the
repairs its output needs.

Resolves (or installs) the pinned godotengine/FBX2glTF binary, sizes the
conversion's timeout from the FBX, runs it, and drives every post-conversion
pass on one edit session. The spec-validity repairs of what the converter
itself writes -- curve-proxy nodes, unbound skins, skin skeleton roots,
tangent handedness -- live here with it.

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through ``cls``.
"""

import json
import logging
import os
import platform as _platform
import shlex
import shutil
import subprocess
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
)

from pythontk.file_utils.mesh_convert.glb.edit import GlbTarget

if TYPE_CHECKING:
    from pythontk.file_utils.mesh_convert.fbx_file import FbxFile

logger = logging.getLogger(__name__)

# godotengine/FBX2glTF — single-binary FBX -> glTF/GLB converter, the same
# tool Godot 4 uses internally for FBX import. Pinned to v0.13.1.
FBX2GLTF_VERSION = "0.13.1"
FBX2GLTF_PLATFORMS = {
    "windows": {
        "url": f"https://github.com/godotengine/FBX2glTF/releases/download/v{FBX2GLTF_VERSION}/FBX2glTF-windows-x86_64.zip",
        "type": "zip",
        "executable": "FBX2glTF-windows-x86_64",
    },
    "linux": {
        "url": f"https://github.com/godotengine/FBX2glTF/releases/download/v{FBX2GLTF_VERSION}/FBX2glTF-linux-x86_64.zip",
        "type": "zip",
        "executable": "FBX2glTF-linux-x86_64",
        "arch": "x86_64",  # no arm64 Linux build is published
    },
    "darwin": {
        "url": f"https://github.com/godotengine/FBX2glTF/releases/download/v{FBX2GLTF_VERSION}/FBX2glTF-macos-x86_64.zip",
        "type": "zip",
        "executable": "FBX2glTF-macos-x86_64",
    },
}


class _Fbx2GltfMixin:
    """FBX2glTF driver: install, timeouts, :meth:`fbx_to_glb`, output repairs.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    TOOL_NAME = "fbx2gltf"
    #: The pinned godotengine/FBX2glTF release this class downloads and drives
    #: -- public here so a UI can name it without importing this module's path.
    FBX2GLTF_VERSION = FBX2GLTF_VERSION
    #: FLOOR for a conversion, in seconds. Kept at the historical value so no
    #: small file converts on a shorter leash than before; the effective budget
    #: for a large one is derived by :meth:`conversion_timeout`.
    DEFAULT_TIMEOUT = 300
    #: Seconds of conversion budget allowed per MB of input FBX. Measured on a
    #: production assembly (250 MB FBX, 757 meshes, 98k triangles, 30 embedded
    #: textures): ~0.8 s/MB with the machine otherwise idle. 3 s/MB was that
    #: with room for a workstation doing something else -- and a second
    #: incident (2026-08-31: a 173 MB assembly timed out at 495s while a test
    #: suite shared the machine) blew through it, discarding another finished
    #: export's GLB. The asymmetry only sharpens with margin: a generous
    #: budget merely delays the report of a genuinely hung process, a tight
    #: one discards a deliverable.
    TIMEOUT_SECONDS_PER_MB = 10.0
    #: Seconds of budget per node-frame FBX2glTF bakes. Size is a WEAK proxy
    #: for its cost: the converter evaluates every node at every frame of
    #: every take (``--anim-framerate bake24``) whether or not the node is
    #: animated, so a scene's bake cost is nodes x frames. Measured on a
    #: production assembly: 2485 nodes x 1591 frames (3.95M node-frames)
    #: took ~250 s of a ~400 s conversion -- and a 12 MB TEXTURELESS export
    #: of the same scene, budgeted 300 s by size alone, timed out and lost
    #: the push. ~63 us per node-frame measured; this is that with the same
    #: ~5x margin the per-MB term carries.
    TIMEOUT_SECONDS_PER_NODE_FRAME = 3e-4
    #: The converter's default bake rate (``bake24``).
    CONVERSION_BAKE_FPS = 24.0
    #: ``timeout=AUTO_TIMEOUT`` (the default) derives the budget from the input.
    #: Negative because no real timeout can be, so it cannot collide with a
    #: caller's value -- and unlike ``None`` it is not already meaningful to
    #: ``subprocess.run``, where None means "wait forever".
    AUTO_TIMEOUT = -1.0

    @classmethod
    def conversion_timeout(cls, src: str, fbx: Optional["FbxFile"] = None) -> float:
        """Seconds to allow FBX2glTF for *src* -- :attr:`DEFAULT_TIMEOUT` or more.

        A flat budget cannot fit both a prop and a production assembly, and the
        cost of getting it wrong is asymmetric: too generous only delays the
        report of a genuinely hung process, while too tight discards a finished
        export's whole deliverable ("produced no file") on a scene that was
        converting normally. It also fails by wall-clock rather than by content,
        so it passes on a quiet machine and fails mid-workday -- which is how it
        reached production unnoticed.

        Two terms, and the larger wins: the input's size (per MB) and the
        bake it implies (per node-frame -- see
        :attr:`TIMEOUT_SECONDS_PER_NODE_FRAME`; the census comes from
        :class:`FbxFile`, seconds on a production assembly, so a caller that
        already holds the parse passes it). The
        second exists because the first is wrong on its own: a textureless
        export of an animated assembly is small and slow, and a budget that
        reads only its size discards its finished deliverable.

        An unreadable size, or a file the census cannot parse, falls back to
        what CAN be read -- and to the floor when nothing can: a budget must
        never be the reason a conversion is not attempted.

        Parameters:
            src: The FBX to be converted.
            fbx: *src* already parsed for its curves, as :meth:`bake_node_frames`
                takes it; parsed here when not given.
        """
        try:
            megabytes = os.path.getsize(src) / (1024 * 1024)
        except OSError:
            megabytes = 0.0
        node_frames = cls.bake_node_frames(src, fbx=fbx)
        return float(
            max(
                cls.DEFAULT_TIMEOUT,
                megabytes * cls.TIMEOUT_SECONDS_PER_MB,
                node_frames * cls.TIMEOUT_SECONDS_PER_NODE_FRAME,
            )
        )

    @classmethod
    def bake_node_frames(cls, src: str, fbx: Optional["FbxFile"] = None) -> int:
        """Node-frames FBX2glTF will evaluate for *src*: nodes x baked frames.

        ``Model`` records are the nodes. Each take contributes the span the
        converter bakes, at :attr:`CONVERSION_BAKE_FPS`: the key extent of its
        curves (``FbxFile.take_spans``), which is what FBX2glTF sizes a take
        by -- a take declaring 1-120 baked 3-200 (2026-10-04) -- else, for a
        take whose keys cannot be read, the ``LocalTime`` span the ``Takes``
        section declares. ``0`` for a file with no takes -- and for one that
        is not a readable binary FBX, so a budget derived from this degrades
        rather than raises.

        Parameters:
            src: The FBX to be converted.
            fbx: *src* already parsed for its curves (``FbxFile.load`` with
                ``span_arrays=("KeyTime",)``), by a caller that reads them too
                -- :meth:`fbx_to_glb` places every clip by the same parse.
                Parsed here when not given.
        """
        from pythontk.file_utils.mesh_convert.fbx_file import FbxFile

        if fbx is None:
            fbx = cls._read_curves(src)
        nodes = fbx.objects_census().get("Model", 0)
        spans = {take: last - first for take, (first, last) in fbx.take_spans().items()}
        for take in (fbx.section("Takes") or {}).get("children", []):
            props = take.get("props") or [b""]
            name = (
                props[0].decode("utf-8", "replace")
                if isinstance(props[0], bytes)
                else ""
            )
            if take.get("name") != "Take" or name in spans:
                continue
            for child in take.get("children", []):
                if (
                    child.get("name") == "LocalTime"
                    and len(child.get("props", [])) >= 2
                ):
                    start, end = child["props"][:2]
                    if isinstance(start, int) and isinstance(end, int) and end > start:
                        spans[name] = (end - start) / FbxFile.TICKS_PER_SECOND
        return int(nodes * sum(spans.values()) * cls.CONVERSION_BAKE_FPS)

    @staticmethod
    def _read_curves(src: str) -> "FbxFile":
        """*src* parsed for what a conversion reads of it -- its census and
        each curve's key extent, never its media -- or an EMPTY file when it
        cannot be read, so each reader degrades to "nothing measured" rather
        than raising: the budget falls back to the size, and the published
        clip spans stand."""
        from pythontk.file_utils.mesh_convert.fbx_file import FbxFile

        try:
            return FbxFile.load(src, span_arrays=("KeyTime",), raw_payloads=False)
        except (OSError, ValueError) as error:
            logger.debug("FBX curves not read: %s", error)
            return FbxFile(src, 0, [])

    @classmethod
    def _platform_exe_name(cls) -> str:
        """Return the FBX2glTF binary name for the current platform."""
        plat = _platform.system().lower()
        info = FBX2GLTF_PLATFORMS.get(plat)
        if not info:
            raise LookupError(f"FBX2glTF: unsupported platform '{plat}'")
        return info["executable"]

    @classmethod
    def resolve_binary(
        cls,
        required: bool = True,
        auto_install: bool = False,
        prompt: Union[bool, Callable[[str], bool]] = True,
    ) -> Optional[str]:
        """Resolve the FBX2glTF executable from PATH or managed installs.

        Parameters:
            required:      Raise FileNotFoundError when missing.
            auto_install:  Download FBX2glTF if not found.
            prompt:        Consent policy for the download -- ``True`` asks on
                           the console (no console = refuse), ``False`` needs
                           none, a callable ``(question) -> bool`` is asked
                           instead (a GUI's dialog). See
                           :meth:`AppInstaller.consent`.

        Returns:
            Absolute path to FBX2glTF executable, or None.
        """
        platform_exe = cls._platform_exe_name()
        # Try platform-specific binary name first (matches release zip),
        # then plain "FBX2glTF" for users who renamed it.
        for candidate in (platform_exe, "FBX2glTF"):
            on_path = shutil.which(candidate)
            if on_path:
                return on_path

        from pythontk.core_utils.app_installer import AppInstaller

        managed = AppInstaller.get_path(
            cls.TOOL_NAME, executable=platform_exe, add_to_path=True
        )
        if managed:
            return managed

        if not auto_install:
            if required:
                raise FileNotFoundError(
                    f"FBX2glTF not found on PATH (looked for {platform_exe!r}). "
                    "Pass auto_install=True to download it."
                )
            return None

        answer = AppInstaller.consent(
            prompt,
            f"FBX2glTF v{FBX2GLTF_VERSION} is not installed. "
            "Download to ~/.pythontk/tools/ now?",
        )
        if answer is None:
            # No interactive console (CI, GUI host, pythonw.exe, etc.).
            # Refuse to silently download — caller must opt-in via prompt=False.
            if required:
                raise FileNotFoundError(
                    "FBX2glTF is not installed and no interactive console "
                    "is available to confirm the download. Pass "
                    "prompt=False to install non-interactively."
                )
            return None
        if not answer:
            if required:
                raise FileNotFoundError("User declined FBX2glTF installation.")
            return None

        try:
            return AppInstaller.ensure(
                cls.TOOL_NAME,
                platforms=FBX2GLTF_PLATFORMS,
                executable=platform_exe,
                version=FBX2GLTF_VERSION,
            )
        except (RuntimeError, OSError, LookupError) as exc:
            if required:
                raise
            logger.warning("FBX2glTF install failed: %s", exc)
            return None

    @classmethod
    def _expand_grayscale_embeds(cls, src: str, store: Any) -> str:
        """The FBX that FBX2glTF reads: *src*, or its copy in *store* with every
        grayscale embed expanded (:meth:`FbxMedia.expand_grayscale`).

        A failure costs a warning, never the conversion: the file converts as
        it did before. A source that is not a binary FBX (an ASCII export) is
        read as it is, since the payload writer cannot re-emit one.
        """
        from pythontk.file_utils.mesh_convert.fbx_file import FbxFile
        from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia

        if not FbxFile.is_fbx(src):
            return src
        copy = store.path(extension=".fbx")
        try:
            outcome = FbxMedia.expand_grayscale(src, copy)
        except Exception as exc:  # noqa: BLE001 -- the conversion still runs
            logger.warning(
                "Grayscale texture expansion skipped (%s): FBX2glTF packs a "
                "grayscale roughness or metallic map as white.",
                exc,
            )
            return src
        if not outcome["expanded"]:
            return src
        logger.info(
            "FBX2glTF input: %d of %d embedded image(s) grayscale, expanded to "
            "RGB so the packed ORM texture keeps their values.",
            outcome["expanded"],
            outcome["images"],
        )
        return copy

    @classmethod
    def fbx_to_glb(
        cls,
        src: str,
        dst: Optional[str] = None,
        *,
        overwrite: bool = False,
        auto_install: bool = True,
        prompt: Union[bool, Callable[[str], bool]] = True,
        timeout: Optional[float] = AUTO_TIMEOUT,
        extra_args: Optional[List[str]] = None,
        sidecar: Optional[Dict[str, Any]] = None,
        data_export: Optional[Dict[str, Any]] = None,
        lightmaps: bool = True,
        lightmap_dirs: Sequence[str] = (),
        shadow_dirs: Sequence[str] = (),
        clip_mode: str = "both",
        report: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Convert an FBX file to a binary glTF 2.0 (GLB) file.

        FBX2glTF reads a copy whose grayscale embedded textures carry every
        colour channel (:meth:`FbxMedia.expand_grayscale`): it packs roughness
        and metallic from their maps' green and blue, so a grayscale map read
        as it is ships as 1.0. A file with no grayscale embed is read as is.

        Parameters:
            src:           Input FBX path.
            dst:           Output GLB path. Defaults to src with .glb extension.
                           ``.glb`` is appended if absent.
            overwrite:     Replace existing destination -- once FBX2glTF has
                           succeeded; a failed conversion leaves it in place.
            auto_install:  Download FBX2glTF if missing.
            prompt:        Consent policy for that download (see
                           :meth:`resolve_binary`).
            timeout:       Subprocess timeout in seconds. The default derives
                it from the input's size (:meth:`conversion_timeout`), because
                a flat budget that suits a prop discards a production
                assembly's finished deliverable. An explicit number is used
                as given; ``None`` disables the limit.
            extra_args:    Extra CLI flags forwarded to FBX2glTF
                           (e.g. ``["--draco"]``, ``["-v"]``). The SDK's
                           embedded-media extraction goes to a temp store
                           removed when the call returns, unless these name
                           ``--fbx-temp-dir`` themselves.
            sidecar:       A scene-sidecar envelope (:meth:`build_scene_sidecar`)
                           to apply to and embed in the converted GLB — the one
                           parameter that turns a bare conversion into a
                           scene-faithful deliverable. Applied inside the same
                           post-conversion edit session as the alpha repair, so
                           it costs no extra file pass. Callers that need the
                           per-section outcome summary call
                           :meth:`apply_scene_sidecar` separately instead.
            data_export:   ``{channel key: value or None}`` overlaid on the
                           file's in-band channels (:meth:`overlay_data_export`)
                           BEFORE any pass reads them, so every pass builds from
                           the overlay as if the FBX had carried it. How a
                           preview shows an effect the scene does not carry
                           (:meth:`effect_preview_channels`); ``None`` builds
                           from the FBX alone.
            lightmaps:     Wire the host scene's committed lightmaps into the
                           GLB (:meth:`apply_glb_lightmaps`). Default on and
                           self-feeding -- the manifest travels inside the FBX,
                           so a scene with no committed bake is a clean no-op
                           and callers pass nothing.
            lightmap_dirs: Extra directories to resolve the manifest's EXR
                           basenames against, forwarded as that method's
                           ``search_dirs``. The manifest carries its own
                           authoring-directory hint, but that is recorded when
                           the bake is COMMITTED and goes stale the moment the
                           project is reorganised or handed to another machine
                           -- at which point the bind silently finds nothing.
                           A host that knows where its textures live now (a
                           DCC's workspace, the scene's own folder) passes them
                           here rather than relying on a historical hint.
            shadow_dirs:   Directories to resolve the shadow rigs' loose maps
                           against (:meth:`apply_glb_shadows`): the horizon
                           data maps, and a silhouette the FBX did not embed.
                           Always searched after these: the FBX's own
                           directory, a ``sourceimages`` folder inside or
                           beside it (the Maya project layout), and every
                           *lightmap_dirs* entry -- the host's live texture
                           folders, which is where a rig's maps are written.
            clip_mode:     Which animation clips the GLB ships, from
                           :attr:`ANIMATION_CLIP_MODES` -- ``both`` (the
                           declared shots AND the whole-timeline stack they
                           were cut from), ``shots``, or ``full``. The two
                           halves hold the SAME performance, so a consumer that
                           plays one never reads the other; on a production
                           assembly the stack alone was 66.5 MB. Validated
                           HERE, before the conversion: the rebuild itself runs
                           under a warn-and-continue guard, so a typo caught
                           only there would downgrade to shipping Maya's lossy
                           split takes with nothing but a log line.
            report:        A dict this fills with what the chain did, for a
                           caller that reports rather than logs: ``"sidecar"``
                           (the per-section outcome
                           :meth:`apply_scene_sidecar` returns) and
                           ``"lightmaps"`` (:meth:`lightmap_report`
                           -- what the manifest wanted of THIS file against
                           what bound) and ``"data_export"`` (the channel keys
                           the overlay replaced; set when one was given), and
                           ``"timings"`` -- seconds spent in ``"fbx2gltf"``
                           (staging the input, the converter, placing its
                           GLB) and in ``"passes"`` (the edit session every
                           repair above runs in).
                           The return value stays the path, so
                           every existing caller is unchanged.

        Returns:
            Absolute path to the written GLB file.

        Raises:
            FileNotFoundError: *src* does not exist.
            ValueError: *src* is not an ``.fbx``, or *clip_mode* is unknown.
            FileExistsError: *dst* exists and *overwrite* is False.
            PermissionError: *dst* is held open by another process (asked
                before converting).
            RuntimeError: FBX2glTF failed, timed out, or wrote no GLB.
        """
        if clip_mode not in cls.ANIMATION_CLIP_MODES:
            raise ValueError(
                f"Unknown animation clip mode {clip_mode!r}; expected one of "
                f"{', '.join(cls.ANIMATION_CLIP_MODES)}."
            )
        src_abs = os.path.abspath(src)
        if not os.path.isfile(src_abs):
            raise FileNotFoundError(f"FBX source not found: {src_abs}")
        if os.path.splitext(src_abs)[1].lower() != ".fbx":
            raise ValueError(f"Expected .fbx input, got: {src_abs}")

        if dst is None:
            dst = os.path.splitext(src_abs)[0] + ".glb"
        elif not dst.lower().endswith(".glb"):
            dst = dst + ".glb"
        dst_abs = os.path.abspath(dst)

        if os.path.exists(dst_abs) and not overwrite:
            raise FileExistsError(
                f"GLB output already exists: {dst_abs}. Pass overwrite=True to replace."
            )
        from pythontk.file_utils._file_utils import FileUtils

        if FileUtils.is_locked(dst_abs):
            # Asked BEFORE the conversion: Windows refuses to replace a file a
            # viewer or a sync client holds open, and the move that ends a
            # conversion -- minutes in -- is the costliest place to learn it.
            raise PermissionError(
                FileUtils.describe_lock(dst_abs)
                or f"GLB output is held open by another process: {dst_abs}"
            )

        os.makedirs(os.path.dirname(dst_abs) or ".", exist_ok=True)

        binary = cls.resolve_binary(
            required=True, auto_install=auto_install, prompt=prompt
        )

        # The input's curves, parsed ONCE for both readers of them -- the
        # budget and the spans every clip is placed by (``_stamp_clip_spans``)
        # -- because a production FBX takes seconds to parse (8 s, a 133 MB
        # Maya assembly, 2026-10-04) and each reader parsed it again. Released
        # before the converter runs. Neither reads it when the caller gave a
        # budget and overlaid the visibility channel: that overlay is the
        # caller's statement for this build (an effect preview places its ramp
        # over its own extent), so the measurement is not stamped over it.
        stamp = cls.VISIBILITY_TRACKS_KEY not in (data_export or {})
        auto = timeout is not None and timeout < 0
        fbx = cls._read_curves(src_abs) if stamp or auto else None
        measured = fbx.take_spans() if stamp else {}
        if auto:
            # The default. An explicit number wins outright (a caller that says
            # 60 means 60) and ``None`` still means no limit.
            timeout = cls.conversion_timeout(src_abs, fbx=fbx)
        del fbx

        # FBX2glTF wants the output base WITHOUT extension; --binary forces .glb.
        # --user-properties copies FBX user properties into per-node glTF
        # ``extras`` (measured against v0.13.1 + Maya 2025: the DataNodes
        # ``data_export`` channels arrive with it and are silently dropped
        # without). A carrier must not drop data another carrier deliberately
        # embedded, and the flag is a no-op on an FBX with no user properties.
        # FBX2glTF writes into a store of this call's own, and its GLB replaces
        # *dst* only once the converter has succeeded: deleting *dst* up front
        # meant a conversion that failed or timed out -- minutes in, on a
        # production assembly -- also cost the file it was to replace, and in a
        # synced export folder that deletion reached every copy.
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        converted = TempArtifacts("fbx2gltf_output", policy="scoped")
        output_base = os.path.join(
            converted.dir_path(), os.path.splitext(os.path.basename(dst_abs))[0]
        )
        cmd = [
            binary,
            "-i",
            src_abs,
            "-o",
            output_base,
            "--binary",
            "--user-properties",
        ]
        if extra_args:
            cmd.extend(extra_args)

        # The SDK's import EXTRACTS every embedded texture into `<stem>.fbm`
        # beside the FBX it reads -- payload-sized (311 MB on a production
        # assembly), never read again once the GLB holds the images, and left
        # in whatever folder the input sits in: a synced export folder, or the
        # temp dir beside a pipeline's scratch copy. A store of this call's own
        # keeps it here; a caller naming its own directory keeps that one.
        media = TempArtifacts("fbx2gltf_media", policy="scoped")
        # Either CLI spelling counts: a second option beside the caller's
        # `--fbx-temp-dir=DIR` is a duplicate the parser can refuse.
        if not any(arg.split("=", 1)[0] == "--fbx-temp-dir" for arg in cmd):
            cmd += ["--fbx-temp-dir", media.dir_path()]

        # The converter reads a copy with its grayscale embeds expanded, released
        # as soon as it is done reading. Expanded inside the try: an interrupted
        # expansion must not strand a full-size copy, or the stores above.
        staged = TempArtifacts("fbx2gltf_input", policy="scoped")
        # Timed in two halves, because they answer different questions: the
        # converter is the push on a production assembly (87% of it, measured
        # by hand before this was recorded), and what moves it lives on the
        # DCC side -- the take, the node count -- while the passes are ours.
        started = perf_counter()
        try:
            cmd[cmd.index("-i") + 1] = cls._expand_grayscale_embeds(src_abs, staged)
            logger.debug("FBX2glTF: %s", shlex.join(cmd))
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    check=False,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"FBX2glTF timed out after {timeout}s converting {src_abs}"
                ) from exc
            finally:
                media.cleanup()
                staged.cleanup()

            if result.returncode != 0:
                raise RuntimeError(
                    f"FBX2glTF failed converting {src_abs} (exit={result.returncode}):\n"
                    f"  cmd: {shlex.join(cmd)}\n"
                    f"  stdout: {result.stdout}\n"
                    f"  stderr: {result.stderr}"
                )
            produced = output_base + ".glb"
            if not os.path.isfile(produced):
                raise RuntimeError(
                    f"FBX2glTF exited 0 but its GLB was not created ({produced}); {dst_abs} is unchanged.\n"
                    f"  stdout: {result.stdout}"
                )
            # Copied to a sibling of dst and swapped in by one atomic replace. A
            # move is not that: the store is in the system temp dir, usually on
            # another volume, where a move copies straight over dst -- as it
            # does on any volume on Windows, whose rename refuses an existing
            # file -- so a disk-full or I/O error part way through a
            # multi-hundred-MB GLB left the deliverable truncated.
            FileUtils.atomic_write(
                dst_abs, lambda part: shutil.copyfile(produced, part)
            )
        finally:
            converted.cleanup()
            media.cleanup()
            staged.cleanup()
        converted_at = perf_counter()

        # One post-conversion edit session for everything that touches the
        # JSON chunk: the alpha repair and (when given) the scene sidecar.
        # The alpha repair keeps its own guard so its failure never costs the
        # sidecar, and neither ever costs the successful conversion.
        try:
            with cls.open_glb(dst_abs) as edit:
                try:
                    fixes = cls.fix_glb_phantom_opaque_alpha(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("fix_glb_phantom_opaque_alpha skipped: %s", exc)
                    fixes = []
                for fx in fixes:
                    logger.info(
                        "fix_glb_phantom_opaque_alpha: %s baseColorFactor[3] %.3f -> %.3f (image: %s)",
                        fx["material"],
                        fx["old_alpha"],
                        fx["new_alpha"],
                        fx["image"],
                    )
                # Before every pass that reads or rewrites images, because it
                # RENUMBERS them: the sidecar's texture map and the optimizer's
                # per-image work both describe indices, and both would then
                # describe a file that no longer exists. Also the cheapest
                # point at which to remove work -- an image collapsed here is
                # one the texture pass never decodes.
                try:
                    cls.dedupe_glb_images(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("dedupe_glb_images skipped: %s", exc)
                # Before the sidecar writes the GLB's OWN handoff, so the
                # file never holds both accounts at once.
                try:
                    dropped = cls.strip_fbx_handoff(edit.gltf)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("strip_fbx_handoff skipped: %s", exc)
                else:
                    if dropped:
                        edit.dirty = True
                        logger.debug(
                            "Dropped the FBX handoff block from %d node(s); the "
                            "GLB carries its own.",
                            dropped,
                        )
                # Ahead of EVERY pass that reads an in-band channel (sidecar,
                # lightmaps, clips, gate, fades): each reads "out of the
                # deliverable itself", so an overlay placed after any one of
                # them would build that pass from the scene and the rest from
                # the overlay.
                if data_export:
                    try:
                        overlaid = cls.overlay_data_export(edit.gltf, data_export)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("data_export overlay skipped: %s", exc)
                    else:
                        if overlaid:
                            edit.dirty = True
                            logger.info(
                                "data_export overlay: %s stated for this build "
                                "(not what the FBX carried).",
                                ", ".join(overlaid),
                            )
                        # Reported, not just logged: a caller that asked for
                        # an overlay can only tell it landed by reading the
                        # file back, and the one that asks -- a panel's
                        # preview -- has to say so in ITS result instead.
                        if report is not None:
                            report["data_export"] = list(overlaid)
                # Right after the overlay, ahead of the same passes: each
                # clip's t=0 is where FBX2glTF put it -- the first key of its
                # take in the FBX it read -- so the spans come from that file
                # (measured above, with the budget) rather than from the
                # producer's prediction of it. Not over a visibility channel
                # the caller overlaid.
                if stamp:
                    try:
                        if cls._stamp_clip_spans(edit.gltf, measured):
                            edit.dirty = True
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Measured clip spans skipped: %s", exc)
                if sidecar:
                    # Guarded like every other pass in this chain: the apply
                    # handles its own per-section and container failures, and
                    # anything past those must cost the repairs, never the
                    # conversion -- the preview routes its envelope through
                    # here now, and a push must not fail on a sidecar.
                    try:
                        applied = cls.apply_scene_sidecar(edit, sidecar)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Scene sidecar skipped: %s", exc)
                        applied = {}
                    if report is not None:
                        report["sidecar"] = applied
                # Sweep images no material samples, BEFORE anything pays to
                # re-encode them. An embedded-media export carries every wired
                # file texture, which for a StingrayPBS scene includes
                # Autodesk's own environment maps (``diffuse_cube``,
                # ``specular_cube``, ``ibl_brdf_lut`` -- ~2.6 MB a file);
                # FBX2glTF re-embeds them and glTF has no slot for them to
                # land in. ``apply_scene_sidecar`` sweeps at its own tail, so
                # this is the OTHER half: a conversion offered no envelope
                # shipped that dead payload and paid the texture pass to
                # compress it first. Ordered after the sidecar (whose
                # ``extras.textures`` map is recorded at its tail and would go
                # stale under a renumber) and before the lightmap and shadow
                # passes, which add images of their own.
                try:
                    cls.prune_glb_unreferenced_textures(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB texture prune skipped: %s", exc)
                if lightmaps:
                    # What the manifest asked for OF THIS GLB, read before the
                    # bind: a miss leaves no record, so only the manifest can
                    # say what was wanted. Only when someone will read it.
                    coverage = None
                    if report is not None:
                        try:
                            coverage = cls.lightmap_manifest_coverage(edit)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("GLB lightmap coverage skipped: %s", exc)
                    # Guarded like the alpha repair: a lightmap failure must
                    # never cost the sidecar or the conversion.
                    try:
                        bound = cls.apply_glb_lightmaps(edit, search_dirs=lightmap_dirs)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("GLB lightmaps skipped: %s", exc)
                        bound = []
                    else:
                        if bound:
                            logger.info(
                                "Lightmaps wired into %d material binding(s).",
                                len(bound),
                            )
                    if coverage is not None:
                        report["lightmaps"] = cls.lightmap_report(coverage, bound)
                # After the lightmaps, whose image plumbing it shares, and
                # before the animation passes: a plane's fade is a pointer
                # channel the viewer reads BESIDE this manifest, so neither
                # depends on the other's order -- but the manifest must not
                # follow a pass that could renumber images, and none below
                # does. Unconditional and self-feeding like the lightmap
                # pass: no channel, no-op.
                try:
                    fbx_dir = os.path.dirname(src_abs)
                    shadow_search = list(shadow_dirs) + [
                        fbx_dir,
                        os.path.join(fbx_dir, "sourceimages"),
                        os.path.join(os.path.dirname(fbx_dir), "sourceimages"),
                        *lightmap_dirs,
                    ]
                    shadows = cls.apply_glb_shadows(edit, search_dirs=shadow_search)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB shadow rigs skipped: %s", exc)
                else:
                    if shadows:
                        logger.info(
                            "Shadow rigs: %d plane(s) published in extras.%s.",
                            len(shadows["planes"]),
                            cls.SHADOW_WEB_KEY,
                        )
                # Reads node names only, so its place in the chain is free of
                # the image and animation passes; self-feeding like the shadow
                # pass: no channel, no-op.
                try:
                    articulated = cls.apply_glb_articulation(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB articulated rigs skipped: %s", exc)
                else:
                    if articulated:
                        logger.info(
                            "Articulated rigs: %d published in extras.%s.",
                            len(articulated["rigs"]),
                            cls.ARTICULATION_WEB_KEY,
                        )
                # Before every animation pass: a curve proxy is an FBX-only
                # transport node whose scale channel would otherwise count as
                # clip content and ship as a stray animated child.
                try:
                    cls.strip_glb_curve_proxies(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB curve-proxy strip skipped: %s", exc)
                # FIRST of the three animation passes: it REPLACES the declared
                # clips, so a gate or a manifest entry written before it would
                # describe clips that no longer exist.
                try:
                    cls.apply_glb_clips(edit, mode=clip_mode)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB clip rebuild skipped: %s", exc)
                # BEFORE the animation manifest, which reports what each clip
                # holds: a shot whose only content is visibility is empty until
                # this has run, and would be reported empty and passed over as
                # the file's default clip.
                try:
                    cls.apply_glb_visibility(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB visibility skipped: %s", exc)
                # After the gate (which makes a fading node present) and before
                # the manifest (which reports a clip carrying only a fade as
                # having content rather than as an empty shot).
                try:
                    faded = cls.apply_glb_fades(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB fades skipped: %s", exc)
                else:
                    if faded:
                        logger.info(
                            "Fades: %d ramp(s) on %d material(s) written as "
                            "%d KHR_animation_pointer channel(s).",
                            faded["nodes"],
                            faded["materials"],
                            faded["channels"],
                        )
                        # A highlighted material gives up its emissive map
                        # (``glb_fades._uncover_emissive``); when that material
                        # was the map's only user the image is dead payload now.
                        try:
                            cls.prune_glb_unreferenced_textures(edit)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("GLB texture sweep skipped: %s", exc)
                # LAST of the writers and BEFORE the manifest: every pass above
                # can only ADD channels (clips, visibility, fades), so what is
                # still hollow here is hollow for good -- and glTF forbids it.
                try:
                    cls.prune_glb_animations(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB animation prune skipped: %s", exc)
                # After the prune, for the same reason the prune is where it is:
                # every pass above can only ADD channels, so the ones that never
                # move are all present and final here. A clip PINS the pose of
                # what it does not animate, which two keys say as well as
                # thousands -- and a baked export writes one key per frame per
                # node per clip (measured on a production assembly: 13,187 of
                # 22,654 channels constant, 16.6 MB spent saying nodes stood
                # still). Unconditional and self-feeding like the passes around
                # it: byte-exact by construction (it collapses only a channel
                # whose every element has the same bit pattern, keeps the clip's
                # own end times so no duration moves, and refuses a layout it
                # cannot prove), so there is no flag for a caller to forget.
                try:
                    cls.compact_glb_animations(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB animation compaction skipped: %s", exc)
                # Skinning data no node binds, then the skeleton roots of the
                # skins that remain. Structural, so correct anywhere after the
                # proxy strip (the last pass that removes nodes); here so the
                # prune's repack copies the smallest BIN this session holds,
                # after the animation passes have released what they drop.
                try:
                    cls.prune_glb_unused_skins(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB skin prune skipped: %s", exc)
                try:
                    cls.fix_glb_skin_skeletons(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB skin skeleton repair skipped: %s", exc)
                # Every shipped tangent, against its UVs: FBX2glTF writes w = +1
                # on every vertex, so each mirrored UV shell read its normal
                # map with green inverted, and passes a DCC's zero tangents
                # through. Structural like the skin repairs, and here for their
                # reason -- its in-place rewrite copies the smallest BIN this
                # session holds.
                try:
                    cls.fix_glb_tangents(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB tangent repair skipped: %s", exc)
                # Unconditional and self-feeding, like the lightmap pass: it
                # reads the take list out of the file and no-ops on a GLB with
                # no animation, so there is no flag for a caller to forget.
                try:
                    animation = cls.apply_glb_animations(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB animation manifest skipped: %s", exc)
                else:
                    if animation:
                        declared = sum(1 for c in animation["clips"] if c["declared"])
                        logger.info(
                            "Animation: %d clip(s) named in extras.%s (%d from "
                            "declared shots); opens on %r.",
                            len(animation["clips"]),
                            cls.ANIMATION_WEB_KEY,
                            declared,
                            animation["default_clip"],
                        )
        except Exception as exc:  # noqa: BLE001 — never let post-process kill a successful conversion
            logger.warning("GLB post-process skipped: %s", exc)

        if report is not None:
            report["timings"] = {
                "fbx2gltf": round(converted_at - started, 3),
                "passes": round(perf_counter() - converted_at, 3),
            }
        return dst_abs

    #: FBX user property that marks a transient curve-proxy node: the per-object
    #: float transport for engines that flatten custom-property curves (a child
    #: transform named ``<node>__<attr>`` whose ``scale.x`` carries the curve).
    #: Both DCC producers stamp it; :meth:`strip_glb_curve_proxies` removes the
    #: node from the GLB, where the ramp already rides ``visibility_tracks``.
    CURVE_PROXY_MARKER = "curveProxy"

    @staticmethod
    def _renumber_manifest_nodes(
        manifest: Any, fields: Sequence[Sequence[str]], remap: Dict[int, int]
    ) -> Any:
        """*manifest* (a dict, or its JSON text -- returned in the same form)
        with the node index at each of *fields* renumbered through *remap*: a
        path of keys, descending through every list on the way. An index to a
        node that is gone becomes None, which every reader takes as absent."""
        text = isinstance(manifest, str)
        if text:
            try:
                manifest = json.loads(manifest)
            except ValueError:
                return manifest

        def walk(value: Any, path: Sequence[str]) -> None:
            if isinstance(value, list):
                for item in value:
                    walk(item, path)
            elif isinstance(value, dict) and path:
                if len(path) == 1:
                    index = value.get(path[0])
                    if isinstance(index, int) and not isinstance(index, bool):
                        value[path[0]] = remap.get(index)
                else:
                    walk(value.get(path[0]), path[1:])

        for path in fields:
            walk(manifest, path)
        return json.dumps(manifest) if text else manifest

    @classmethod
    def strip_glb_curve_proxies(cls, glb: GlbTarget) -> List[str]:
        """Remove every curve-proxy node (and its channels) from a GLB.

        The proxy is an FBX-side transport artifact: Unity flattens animated
        custom properties onto the root Animator with empty paths, so the DCC
        stages a child transform per keyed channel whose ``scale.x`` carries
        the curve, and the Unity importer rebinds and deletes it. FBX2glTF
        carries that child through as a node with a ``scale`` channel --
        measured: ``hlCube__highlight`` arrived with its 0->1 scale intact --
        which in the GLB is a stray animated node under the object. The GLB
        route gets the ramp from ``visibility_tracks`` instead, so the node is
        removed here, BEFORE the clip rebuild counts its channel as content.

        Nodes are identified by the :attr:`CURVE_PROXY_MARKER` user property,
        never by name, so an artist's own ``foo__bar`` transform is safe.

        Returns:
            The removed node names, in file order.
        """
        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            nodes = gltf.get("nodes") or []
            doomed = {
                index
                for index, node in enumerate(nodes)
                if cls._node_user_property(node, cls.CURVE_PROXY_MARKER)
            }
            if not doomed:
                return []
            remap: Dict[int, int] = {}
            kept: List[Dict[str, Any]] = []
            for index, node in enumerate(nodes):
                if index in doomed:
                    continue
                remap[index] = len(kept)
                kept.append(node)

            def renumber(indices: Any) -> List[int]:
                return [remap[i] for i in indices if isinstance(i, int) and i in remap]

            for node in kept:
                if "children" in node:
                    node["children"] = renumber(node["children"])
                    if not node["children"]:
                        del node["children"]
            for scene in gltf.get("scenes") or []:
                if "nodes" in scene:
                    scene["nodes"] = renumber(scene["nodes"])
            for skin in gltf.get("skins") or []:
                if "joints" in skin:
                    skin["joints"] = renumber(skin["joints"])
                if isinstance(skin.get("skeleton"), int):
                    skin["skeleton"] = remap.get(skin["skeleton"])
                    if skin["skeleton"] is None:
                        del skin["skeleton"]
            for animation in gltf.get("animations") or []:
                survivors = []
                for channel in animation.get("channels") or []:
                    target = channel.get("target") or {}
                    node = target.get("node")
                    if isinstance(node, int):
                        if node in doomed:
                            continue
                        target["node"] = remap[node]
                    survivors.append(channel)
                animation["channels"] = survivors
            # The manifests earlier passes bound BY NODE INDEX follow the
            # renumber too: measured, a proxy ahead of a shadow plane left the
            # plane following the node after its source, and a rig posing its
            # joints one node over.
            extras = gltf.get("extras")
            if isinstance(extras, dict):
                for key, fields in (
                    (cls.SHADOW_WEB_KEY, cls.SHADOW_WEB_NODE_FIELDS),
                    (cls.ARTICULATION_WEB_KEY, cls.ARTICULATION_WEB_NODE_FIELDS),
                ):
                    if key in extras:
                        extras[key] = cls._renumber_manifest_nodes(
                            extras[key], fields, remap
                        )
            gltf["nodes"] = kept
            edit.dirty = True
            removed = [str(nodes[i].get("name") or i) for i in sorted(doomed)]
            logger.info(
                "Curve proxies: %d transport node(s) stripped from the GLB (%s).",
                len(removed),
                ", ".join(removed),
            )
            return removed

    @staticmethod
    def _node_user_property(node: Dict[str, Any], key: str) -> Any:
        """One user property off a node's extras, in either on-disk shape.

        FBX2glTF nests user properties under ``extras.fromFBX.userProperties``
        as ``{"type", "value"}`` records; a native glTF writer puts a plain
        value at ``extras[key]``.
        """
        extras = (node or {}).get("extras") or {}
        nested = ((extras.get("fromFBX") or {}).get("userProperties") or {}).get(key)
        if isinstance(nested, dict):
            return nested.get("value")
        if nested is not None:
            return nested
        return extras.get(key)

    #: Primitive extensions that name the primitive's attributes a second
    #: time -- Draco maps each to its compressed stream -- so an attribute
    #: dropped from ``attributes`` alone would leave the extension naming one
    #: that is gone. Same "bail rather than guess" contract as
    #: :attr:`_ACCESSOR_REFERRING_EXTENSIONS`.
    _ATTRIBUTE_REFERRING_EXTENSIONS = frozenset({"KHR_draco_mesh_compression"})

    @classmethod
    def prune_glb_unused_skins(cls, glb: GlbTarget) -> Dict[str, int]:
        """Drop the skinning data no node binds.

        glTF gives a skin meaning only through a node's ``skin``, and
        ``JOINTS_n``/``WEIGHTS_n`` only on a mesh a skinned node instantiates;
        FBX2glTF writes both regardless. Measured on a production assembly
        (2026-09-14): 255 of its 262 skins bound by no node, and JOINTS_0/
        WEIGHTS_0 on 836 meshes no skinned node instantiates -- 0.49 MB of
        vertex data, and a Khronos validator warning per node
        (NODE_SKINNED_MESH_WITHOUT_SKIN x1493).

        Removes every skin no node references (renumbering ``node.skin``) and
        those attributes from each primitive of a mesh no skinned node
        instantiates -- a mesh a skinned node shares keeps them, and so does a
        primitive whose extension names its attributes again
        (:attr:`_ATTRIBUTE_REFERRING_EXTENSIONS`) -- then deletes the accessors
        that leaves unread and repacks the BIN.

        Returns:
            ``{"skins": removed, "attributes": stripped, "bytes": reclaimed}``.
            ``bytes`` stays 0 when the file uses an extension that may hold
            accessor indices of its own (:meth:`_drop_orphaned_accessors`).
        """
        counts = {"skins": 0, "attributes": 0, "bytes": 0}
        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            nodes = gltf.get("nodes") or []
            skins = gltf.get("skins") or []
            bound = {node.get("skin") for node in nodes if "skin" in node}
            released: Set[Any] = set()
            remap: Dict[int, int] = {}
            kept: List[Dict[str, Any]] = []
            for index, skin in enumerate(skins):
                if index in bound:
                    remap[index] = len(kept)
                    kept.append(skin)
                else:
                    released.add((skin or {}).get("inverseBindMatrices"))
            counts["skins"] = len(skins) - len(kept)
            if counts["skins"]:
                for node in nodes:
                    if node.get("skin") in remap:
                        node["skin"] = remap[node["skin"]]
                if kept:
                    gltf["skins"] = kept
                else:
                    del gltf["skins"]  # the schema's minItems is 1
            skinned = {node.get("mesh") for node in nodes if "skin" in node}
            for index, mesh in enumerate(gltf.get("meshes") or []):
                if index in skinned:
                    continue
                for primitive in (mesh or {}).get("primitives") or []:
                    if cls._ATTRIBUTE_REFERRING_EXTENSIONS & set(
                        primitive.get("extensions") or ()
                    ):
                        continue
                    attributes = primitive.get("attributes") or {}
                    for name in list(attributes):
                        prefix, _, set_index = name.partition("_")
                        if prefix in ("JOINTS", "WEIGHTS") and set_index.isdigit():
                            released.add(attributes.pop(name))
                            counts["attributes"] += 1
            if not (counts["skins"] or counts["attributes"]):
                return counts
            edit.dirty = True
            if cls._drop_orphaned_accessors(edit, released):
                counts["bytes"] = cls._compact_bin(edit)
            logger.info(
                "Skins: dropped %d skin(s) no node binds and %d JOINTS/WEIGHTS "
                "attribute(s) from meshes no skinned node instantiates (%.2f MB).",
                counts["skins"],
                counts["attributes"],
                counts["bytes"] / 1048576,
            )
        return counts

    @classmethod
    def fix_glb_skin_skeletons(cls, glb: GlbTarget) -> List[str]:
        """Point every ``skin.skeleton`` at a common root of the skin's joints.

        glTF requires the skeleton node to be the joints' closest common root
        or an ancestor of it. FBX2glTF names a skin's first joint, which holds
        only while the other joints hang below it; an export that re-parents a
        chain's joints side by side under their group (a sheared-chain flatten
        does) breaks it. Measured on a production assembly (2026-09-14): all 7
        skins invalid, rejected by the Khronos validator
        (SKIN_SKELETON_INVALID x7) -- three.js never reads the field, so no
        viewer showed it.

        A valid skeleton is left alone; an invalid one becomes the joints'
        closest common root, or is removed -- the field is optional -- when the
        joints share no root.

        Returns:
            The repaired skins' names (``skin <index>`` when unnamed), in file
            order.
        """
        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            nodes = gltf.get("nodes") or []
            parents = cls._node_parents(gltf)
            repaired: List[str] = []
            outcomes: List[str] = []
            for index, skin in enumerate(gltf.get("skins") or []):
                if not isinstance(skin, dict) or not isinstance(
                    skin.get("skeleton"), int
                ):
                    continue
                joints = [j for j in skin.get("joints") or [] if isinstance(j, int)]
                chains = [cls._node_lineage(parents, joint) for joint in joints]
                if not chains or all(skin["skeleton"] in chain for chain in chains):
                    continue
                shared = set(chains[0]).intersection(*chains[1:])
                root = next((node for node in chains[0] if node in shared), None)
                if root is None:
                    del skin["skeleton"]
                    outcomes.append("removed: no common root")
                else:
                    skin["skeleton"] = root
                    named = 0 <= root < len(nodes) and nodes[root].get("name")
                    outcomes.append(str(named or root))
                repaired.append(str(skin.get("name") or f"skin {index}"))
            if repaired:
                edit.dirty = True
                logger.info(
                    "Skins: %d skeleton(s) were not a common root of their "
                    "joints; re-pointed (%s).",
                    len(repaired),
                    ", ".join(outcomes),
                )
            return repaired

    @classmethod
    def fix_glb_tangents(cls, glb: GlbTarget) -> Dict[str, int]:
        """Point every shipped TANGENT the way its UVs run, and give a
        zero-length one a direction.

        FBX2glTF ships ``w = +1`` on every vertex, and three.js takes a shipped
        TANGENT over its own frame, so every mirrored UV shell rendered its
        normal map with green inverted; a DCC writes a zero tangent where it
        cannot orient one, which glTF forbids and three.js turns into NaN --
        see :class:`~pythontk.file_utils.mesh_convert.glb.tangents.GlbTangents`.

        Parameters:
            glb: Path to a ``.glb``, modified in place, or an open session.

        Returns:
            ``{"primitives": n, "signs": n, "directions": n}`` -- primitives
            reading a rewritten TANGENT, vertices whose ``w`` was set, and
            vertices whose direction was rebuilt.
        """
        from pythontk.file_utils.mesh_convert.glb.tangents import GlbTangents

        return GlbTangents.repair(glb)
