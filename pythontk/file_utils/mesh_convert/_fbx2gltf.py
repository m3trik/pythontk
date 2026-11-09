# !/usr/bin/python
# coding=utf-8
"""The FBX2glTF CLI driver behind :meth:`MeshConvert.fbx_to_glb`, and the
repairs its output needs.

Resolves (or installs) the pinned godotengine/FBX2glTF binary, sizes the
conversion's timeout from the FBX, runs it, and drives every post-conversion
pass on one edit session. The spec-validity repairs of what the converter
itself writes -- curve-proxy nodes, unbound skins, skin skeleton roots,
tangent handedness, non-finite animation keys -- live here with it.

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through the facade it holds (``self._mc``).
"""

import logging
import math
import os
import platform as _platform
import shlex
import shutil
import struct
import subprocess
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit, GlbTarget
from pythontk.file_utils.mesh_convert.glb.fades import GlbFades

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


class _Fbx2Gltf:
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
    #: every take (:attr:`CONVERSION_BAKE_FPS`) whether or not the node is
    #: animated, so a scene's bake cost is nodes x frames. Measured on a
    #: production assembly: 2485 nodes x 1591 frames (3.95M node-frames,
    #: counted at 24 fps) took ~250 s of a ~400 s conversion -- and a 12 MB
    #: TEXTURELESS export of the same scene, budgeted 300 s by size alone,
    #: timed out and lost the push. ~63 us per node-frame measured; this is
    #: that with the same ~5x margin the per-MB term carries (~6x at the real
    #: 30 fps bake).
    TIMEOUT_SECONDS_PER_NODE_FRAME = 3e-4
    #: The converter's default bake rate (``bake30``): measured 2026-10-31, a
    #: 24 fps file and a 30 fps file both bake at 1/30 s.
    CONVERSION_BAKE_FPS = 30.0
    #: ``timeout=AUTO_TIMEOUT`` (the default) derives the budget from the input.
    #: Negative because no real timeout can be, so it cannot collide with a
    #: caller's value -- and unlike ``None`` it is not already meaningful to
    #: ``subprocess.run``, where None means "wait forever".
    AUTO_TIMEOUT = -1.0

    def __init__(self, mc: type) -> None:
        self._mc = mc  # the facade class: every name it exposes is read through it

    def conversion_timeout(self, src: str, fbx: Optional["FbxFile"] = None) -> float:
        try:
            megabytes = os.path.getsize(src) / (1024 * 1024)
        except OSError:
            megabytes = 0.0
        node_frames = self._mc.bake_node_frames(src, fbx=fbx)
        return float(
            max(
                self._mc.DEFAULT_TIMEOUT,
                megabytes * self._mc.TIMEOUT_SECONDS_PER_MB,
                node_frames * self._mc.TIMEOUT_SECONDS_PER_NODE_FRAME,
            )
        )

    def bake_node_frames(self, src: str, fbx: Optional["FbxFile"] = None) -> int:
        from pythontk.file_utils.mesh_convert.fbx_file import FbxFile

        if fbx is None:
            fbx = self._read_curves(src)
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
        return int(nodes * sum(spans.values()) * self._mc.CONVERSION_BAKE_FPS)

    @staticmethod
    def _read_curves(src: str) -> "FbxFile":
        """*src* parsed for what a conversion reads of it -- its census and
        each curve's key extent, never its media -- or an EMPTY file when it
        cannot be read, so each reader degrades to "nothing measured" rather
        than raising: the budget falls back to the size, and no clip is
        noted with its take."""
        from pythontk.file_utils.mesh_convert.fbx_file import FbxFile

        try:
            return FbxFile.load(src, span_arrays=("KeyTime",), raw_payloads=False)
        except (OSError, ValueError) as error:
            logger.debug("FBX curves not read: %s", error)
            return FbxFile(src, 0, [])

    def _platform_exe_name(self) -> str:
        """Return the FBX2glTF binary name for the current platform."""
        plat = _platform.system().lower()
        info = FBX2GLTF_PLATFORMS.get(plat)
        if not info:
            raise LookupError(f"FBX2glTF: unsupported platform '{plat}'")
        return info["executable"]

    def resolve_binary(
        self,
        required: bool = True,
        auto_install: bool = False,
        prompt: Union[bool, Callable[[str], bool]] = True,
    ) -> Optional[str]:
        platform_exe = self._mc._platform_exe_name()
        # Try platform-specific binary name first (matches release zip),
        # then plain "FBX2glTF" for users who renamed it.
        for candidate in (platform_exe, "FBX2glTF"):
            on_path = shutil.which(candidate)
            if on_path:
                return on_path

        from pythontk.core_utils.app_installer import AppInstaller

        managed = AppInstaller.get_path(
            self._mc.TOOL_NAME, executable=platform_exe, add_to_path=True
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
                self._mc.TOOL_NAME,
                platforms=FBX2GLTF_PLATFORMS,
                executable=platform_exe,
                version=FBX2GLTF_VERSION,
            )
        except (RuntimeError, OSError, LookupError) as exc:
            if required:
                raise
            logger.warning("FBX2glTF install failed: %s", exc)
            return None

    def _expand_grayscale_embeds(self, src: str, store: Any) -> str:
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

    def fbx_to_glb(
        self,
        src: str,
        dst: Optional[str] = None,
        *,
        overwrite: bool = False,
        auto_install: bool = True,
        prompt: Union[bool, Callable[[str], bool]] = True,
        timeout: Optional[float] = AUTO_TIMEOUT,
        extra_args: Optional[List[str]] = None,
        sidecar: Optional[Dict[str, Any]] = None,
        rendering: Optional[Mapping[str, Mapping[str, Any]]] = None,
        data_export: Optional[Dict[str, Any]] = None,
        lightmaps: bool = True,
        lightmap_dirs: Sequence[str] = (),
        shadow_dirs: Sequence[str] = (),
        report: Optional[Dict[str, Any]] = None,
        hdr_codec: Any = None,
    ) -> str:
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

        binary = self._mc.resolve_binary(
            required=True, auto_install=auto_install, prompt=prompt
        )

        # The input's curves, parsed ONCE for both readers of them -- the
        # budget and the render effects' curves (``render_effect_curves``) --
        # because a production FBX takes seconds to parse (8 s, a 133 MB
        # Maya assembly, 2026-10-04) and each reader parsed it again. Released
        # before the converter runs.
        auto = timeout is not None and timeout < 0
        fbx = self._read_curves(src_abs)
        # The render effects' curves, off the same parse: each take's curve
        # proxies (``SceneRecords.TRANSPORT``), placed where the converter puts
        # each clip's t=0. Read here, not off the converter's own channels for
        # them, which it resamples and thins (``render_effect_curves``).
        effect_curves: Dict[str, Any] = {}
        try:
            effect_curves = self._mc.render_effect_curves(
                fbx, bake_fps=self._bake_fps(extra_args or ())
            )
        except Exception as exc:  # noqa: BLE001 -- the effects, not the file
            logger.warning("Render-effect curves not read: %s", exc)
        # Each declared take's span, off the same parse: every take is a clip
        # of its name spanning its window, the still ones the converter drops
        # and the ones it cuts short included (``_Animation.take_clip_spans``).
        take_spans: Dict[str, float] = {}
        if fbx is not None:
            try:
                take_spans = self._mc._animation.take_clip_spans(
                    fbx, bake_fps=self._bake_fps(extra_args or ())
                )
            except Exception as exc:  # noqa: BLE001 -- the spans, not the file
                logger.warning("Take windows not read: %s", exc)
        if auto:
            # The default. An explicit number wins outright (a caller that says
            # 60 means 60) and ``None`` still means no limit.
            timeout = self._mc.conversion_timeout(src_abs, fbx=fbx)
        # The file's own materials, off the same parse: the materials pass
        # below needs a few hundred bytes of it, never the parse itself.
        own_materials = None
        if fbx is not None and fbx.roots:
            try:
                own_materials = self._mc._fbx_materials._read_statements(fbx)
            except Exception as exc:  # noqa: BLE001 -- the pass parses it then
                logger.debug("FBX materials not read with the curves: %s", exc)
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
            cmd[cmd.index("-i") + 1] = self._expand_grayscale_embeds(src_abs, staged)
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
            FileUtils.copy_file(
                produced, os.path.dirname(dst_abs), os.path.basename(dst_abs)
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
            with self._mc.open_glb(dst_abs) as edit:
                # FIRST, on the converter's own output: FBX2glTF merges
                # identical vertices across meshes and hands a later skinned
                # mesh an earlier one's weights, which only the FBX it read
                # still holds right. Self-gating: the FBX is read only when a
                # skinned mesh shares a vertex position with another mesh.
                try:
                    self._mc.restore_glb_skin_weights(edit, src_abs)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB skin weight restore skipped: %s", exc)
                # On the converter's own output too, and before the key repair:
                # a node scaled flat on an axis has no rotation to decompose,
                # and the converter DROPS it -- the rest rotation and each
                # channel -- so only the FBX still holds it. It reads the
                # converter's key times (finite: frame indices) and needs the
                # animation the converter wrote for each take, which the repair
                # prunes once it drops that animation's last channel.
                # Self-gating: the FBX is read only when a node is flat.
                try:
                    self._mc.restore_glb_rotations(
                        edit, src_abs, fps=self._bake_fps(cmd)
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB rotation restore skipped: %s", exc)
                # Ahead of every pass that reads a key: the converter bakes
                # each node's local matrix and decomposes it, and a degenerate
                # one (a zero scale) comes out a NaN rotation glTF forbids.
                try:
                    self._mc.fix_glb_nonfinite_keys(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB non-finite key repair skipped: %s", exc)
                try:
                    fixes = self._mc.fix_glb_phantom_opaque_alpha(edit)
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
                    self._mc.dedupe_glb_images(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("dedupe_glb_images skipped: %s", exc)
                # Ahead of EVERY pass that reads an in-band channel (sidecar,
                # lightmaps, clips, gate, fades): each reads "out of the
                # deliverable itself", so an overlay placed after any one of
                # them would build that pass from the scene and the rest from
                # the overlay.
                if data_export:
                    try:
                        overlaid = self._mc.overlay_data_export(edit.gltf, data_export)
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
                # Every take a shot names is a clip of its name, ahead of every
                # pass that writes into clips (a preview's curves, the gates,
                # the fades): FBX2glTF writes none for a take that holds
                # still, and its shot would leave the page. A take no shot
                # names gets none it lacks (a static write's).
                owed: List[str] = []
                if take_spans:
                    try:
                        owed = self._mc._animation._shot_takes(edit.gltf, list(take_spans))
                        if owed:
                            self._mc._animation._synthesize_clips(edit.gltf, owed)
                            edit.dirty = True
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Take clips not made: %s", exc)
                # Right after the overlay, ahead of the passes that read it: a
                # preview states its curves for this build alone, and they are
                # the build's in place of the file's (the overlay's record
                # names them); the key leaves the file here. Clips it states
                # are spans too, held like a take's.
                try:
                    preview, preview_spans = self._mc._visibility.take_preview(edit.gltf)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Previewed render effect skipped: %s", exc)
                else:
                    take_spans = {**take_spans, **preview_spans}
                    owed = [*owed, *preview_spans]
                    if preview or preview_spans:
                        edit.dirty = True
                    if preview:
                        effect_curves = preview
                # The materials, stated once: the ``materials`` record the
                # file carries states each material it names -- produced from
                # the scene as staged for the write, the statement the FBX
                # deliverable itself carries -- and the FBX's own materials,
                # read into the same families, state the rest. Every material
                # takes it whole, its maps read out of the FBX itself, before
                # anything records image indices. A file this reader cannot
                # parse (an ASCII FBX) states nothing: the converter's reading
                # stands. Guarded like every pass here: a failure costs the
                # materials their statement, never the conversion.
                from pythontk.file_utils.mesh_convert.fbx_file import FbxFile

                written = None
                if FbxFile.is_fbx(src_abs):
                    try:
                        _stated, written = self._mc._fbx_materials._apply_fbx_materials(
                            edit, src_abs, lightmap_dirs, read=own_materials
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Materials record skipped: %s", exc)
                    else:
                        if report is not None:
                            report["materials"] = written
                # The handoff: the caller's, else the one the FBX's own record
                # implies -- its source and lighting, in this file's
                # addresses. It names the blocks the file carries once every
                # pass has run (the tail of this session).
                try:
                    self._mc._sidecar._warn_destructive_orms(
                        edit,
                        described=[
                            r["material"] for r in written or () if r.get("orm")
                        ],
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("ORM check skipped: %s", exc)
                try:
                    handoff = (
                        self._mc._sidecar._handoff_of(sidecar)
                        if sidecar
                        else self._mc._sidecar._fbx_handoff(edit.gltf, rendering)
                    )
                    if handoff:
                        # Named last, once every pass has added its block
                        # (_finish_glb_handoff, below).
                        self._mc._write_glb_handoff(edit, handoff)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB handoff skipped: %s", exc)
                # Sweep images no material samples, BEFORE anything pays to
                # re-encode them. An embedded-media export carries every wired
                # file texture, which for a StingrayPBS scene includes
                # Autodesk's own environment maps (``diffuse_cube``,
                # ``specular_cube``, ``ibl_brdf_lut`` -- ~2.6 MB a file);
                # FBX2glTF re-embeds them and glTF has no slot for them to
                # land in. After the materials, which unbind the converter's
                # own maps, and before the lightmap and shadow passes, which
                # add images of their own.
                try:
                    self._mc.prune_glb_unreferenced_textures(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB texture prune skipped: %s", exc)
                if lightmaps:
                    # What the record asks for OF THIS GLB: only the record can
                    # say what was wanted. Only when someone will read it.
                    coverage = None
                    if report is not None:
                        try:
                            coverage = self._mc.lightmap_manifest_coverage(edit)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("GLB lightmap coverage skipped: %s", exc)
                    # Guarded like the alpha repair: a lightmap failure must
                    # never cost the sidecar or the conversion. The files the
                    # record names as the FBX embeds them are searched first:
                    # a finished FBX's drawn probe and copies exist only there.
                    from pythontk.file_utils.temp_artifacts import TempArtifacts

                    store = TempArtifacts("glb_lightmap_media", policy="scoped")
                    try:
                        own = self._mc._media.extract_fbx_files(
                            edit.gltf, src_abs, store.dir_path()
                        )
                        bound = self._mc.apply_glb_lightmaps(
                            edit,
                            search_dirs=[*own, *lightmap_dirs],
                            hdr_codec=hdr_codec,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("GLB lightmaps skipped: %s", exc)
                        bound = []
                    else:
                        if bound:
                            logger.info(
                                "Lightmaps: %d object(s) lit, the record and "
                                "its maps in the file.",
                                len({r["entry"] for r in bound}),
                            )
                    finally:
                        store.cleanup(force=True)
                    if coverage is not None:
                        report["lightmaps"] = self._mc.lightmap_report(coverage, bound)
                # Unconditional and self-feeding: whichever pass or publisher left
                # the lightmaps record, a shipped light would land on top of its
                # own bake.
                try:
                    self._mc.strip_glb_baked_lights(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB baked-light strip skipped: %s", exc)
                # After the lightmaps, whose media writer it shares: the
                # record as it stands and every file it names, by name, so
                # no later pass can renumber what it carries. Unconditional
                # and self-feeding like the lightmap pass: no record, no-op.
                # The files the record names as the FBX embeds them are
                # searched first, then the caller's and the stage's folders.
                store = TempArtifacts("glb_shadow_media", policy="scoped")
                try:
                    own = self._mc._media.extract_fbx_files(
                        edit.gltf, src_abs, store.dir_path(), SceneRecords.SHADOWS
                    )
                    shadows = self._mc.apply_glb_shadows(
                        edit,
                        search_dirs=[
                            *own,
                            *shadow_dirs,
                            *self._mc.fbx_media_dirs(src_abs),
                            *lightmap_dirs,
                        ],
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB shadow rigs skipped: %s", exc)
                else:
                    if shadows:
                        logger.info(
                            "Shadow rigs: the %s record (%d plane(s)) and its "
                            "maps in the file.",
                            SceneRecords.SHADOWS.key,
                            len(shadows.get("planes") or []),
                        )
                finally:
                    store.cleanup(force=True)
                # The record as it stands, by name alone, so its place in the
                # chain is free of the image and animation passes (and of the
                # node strip below); self-feeding like the shadow pass: no
                # channel, no-op.
                try:
                    articulated = self._mc.apply_glb_articulation(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB articulated rigs skipped: %s", exc)
                else:
                    if articulated:
                        logger.info(
                            "Articulated rigs: the %s record (%d rig(s)) in "
                            "the file.",
                            SceneRecords.ARTICULATION.key,
                            len(articulated.get("rigs") or []),
                        )
                # Where each curve proxy sits, read before the strip: the
                # node a curve plays on is its proxy's parent
                # (``SceneRecords.TRANSPORT.join``).
                carriers = GlbFades.carriers(edit.gltf)
                # Before every animation pass: a curve proxy is an FBX-only
                # transport node whose scale channel would otherwise count as
                # clip content and ship as a stray animated child.
                try:
                    self._mc.strip_glb_curve_proxies(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB curve-proxy strip skipped: %s", exc)
                # BEFORE the animation manifest, which reports what each clip
                # holds: a shot whose only content is visibility is empty until
                # this has run, and would be reported empty and passed over as
                # the file's default clip.
                try:
                    self._mc.apply_glb_visibility(
                        edit, curves=effect_curves, carriers=carriers
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB visibility skipped: %s", exc)
                # After the gate (which makes a fading node present) and before
                # the manifest (which reports a clip carrying only a fade as
                # having content rather than as an empty shot).
                try:
                    faded = self._mc.apply_glb_fades(
                        edit, curves=effect_curves, carriers=carriers
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB render effects skipped: %s", exc)
                else:
                    if faded:
                        logger.info(
                            "Render effects: %d node(s) and %d material(s) "
                            "animated by %d KHR_animation_pointer channel(s).",
                            faded["nodes"],
                            faded["materials"],
                            faded["channels"],
                        )
                # After every pass that writes a take's channels, before the
                # prune: a take clip whose channels end before its take does,
                # or that has none (a take that holds still), is held to its
                # end, so the page lists it and plays it for its length.
                if take_spans:
                    try:
                        self._mc._animation._hold_take_spans(edit, take_spans, shots=owed)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Take clips not held: %s", exc)
                # LAST of the writers and BEFORE the manifest: every pass above
                # can only ADD channels (clips, visibility, fades), so what is
                # still hollow here is hollow for good -- and glTF forbids it.
                try:
                    self._mc.prune_glb_animations(edit)
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
                    self._mc.compact_glb_animations(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB animation compaction skipped: %s", exc)
                # Skinning data no node binds, then the skeleton roots of the
                # skins that remain. Structural, so correct anywhere after the
                # proxy strip (the last pass that removes nodes); here so the
                # prune's repack copies the smallest BIN this session holds,
                # after the animation passes have released what they drop.
                try:
                    self._mc.prune_glb_unused_skins(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB skin prune skipped: %s", exc)
                try:
                    self._mc.fix_glb_skin_skeletons(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB skin skeleton repair skipped: %s", exc)
                # Every shipped tangent, against its UVs: FBX2glTF writes w = +1
                # on every vertex, so each mirrored UV shell read its normal
                # map with green inverted, and passes a DCC's zero tangents
                # through. Structural like the skin repairs, and here for their
                # reason -- its in-place rewrite copies the smallest BIN this
                # session holds.
                try:
                    self._mc.fix_glb_tangents(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB tangent repair skipped: %s", exc)
                # Unconditional and self-feeding, like the lightmap pass: it
                # reads the take list out of the file and no-ops on a GLB with
                # no animation, so there is no flag for a caller to forget.
                try:
                    animation = self._mc.apply_glb_animations(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB animation manifest skipped: %s", exc)
                else:
                    if animation:
                        shots = {s["clip"] for s in animation["shots"]}
                        clips = {
                            a.get("name") for a in edit.gltf.get("animations") or []
                        }
                        logger.info(
                            "Animation: the %s record in extras.%s: %d shot(s), "
                            "%d with a clip of their name, at %s fps.",
                            SceneRecords.SHOTS.key,
                            SceneRecords.SHOTS.key,
                            len(shots),
                            len(shots & clips),
                            animation.get("fps", "no stated"),
                        )
                # Every other record a GLB carries as it stands (the audio
                # cues: one layout, two containers), copied off the carrier
                # before the strip below takes it, and before the handoff,
                # which names the blocks the file carries.
                try:
                    # Without its lightmap pass a file holds no map, so it
                    # names none (the record leaves with the carrier below).
                    carried = self._mc._carry_glb_records(
                        edit.gltf,
                        skip=() if lightmaps else (SceneRecords.LIGHTMAPS.key,),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB record carry skipped: %s", exc)
                else:
                    if carried:
                        edit.dirty = True
                        logger.info(
                            "Records: %s carried into the root extras as the "
                            "FBX holds them.",
                            ", ".join(carried),
                        )
                # LAST, once every pass has read the FBX's records and stated
                # them in glTF's own terms: the handoff names the blocks this
                # file carries, and what the records left on the nodes -- the
                # carrier's channels, the baked objects' markers, a carrier
                # left empty -- leaves, so the file states each thing once.
                try:
                    if report is not None:
                        # What the scene's own record of this export keeps:
                        # the record as the file ships it (its probe drawn).
                        report["lightmap_manifest"] = SceneRecords.glb_block(
                            edit.gltf, SceneRecords.LIGHTMAPS
                        )
                    self._mc._finish_glb_handoff(edit)
                    stripped = self._mc.strip_fbx_records(edit.gltf)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB record strip skipped: %s", exc)
                else:
                    if stripped:
                        edit.dirty = True
                        # A carrier left bare goes with its samplers: the bytes
                        # only they read (a no-op when nothing is orphaned).
                        self._mc._compact_bin(edit)
        except Exception as exc:  # noqa: BLE001 — never let post-process kill a successful conversion
            logger.warning("GLB post-process skipped: %s", exc)

        if report is not None:
            report["timings"] = {
                "fbx2gltf": round(converted_at - started, 3),
                "passes": round(perf_counter() - converted_at, 3),
            }
        return dst_abs

    #: FBX user property that marks a transient curve-proxy node: the per-object
    #: float transport for engines that flatten custom-property curves, as
    #: ``SceneRecords.TRANSPORT`` declares it. Both DCC producers stamp it;
    #: :meth:`strip_glb_curve_proxies` removes the node from the GLB, where the
    #: curve it carries rides the property itself (the presence gate and the
    #: ``KHR_animation_pointer`` channels, built from the FBX's own curves).
    CURVE_PROXY_MARKER = SceneRecords.TRANSPORT.marker

    def strip_glb_curve_proxies(self, glb: GlbTarget) -> List[str]:
        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            nodes = gltf.get("nodes") or []
            doomed = {
                index
                for index, node in enumerate(nodes)
                if self._node_user_property(node, self._mc.CURVE_PROXY_MARKER)
            }
            if not doomed:
                return []
            removed = self._mc._remove_glb_nodes(gltf, doomed)
            edit.dirty = True
            # The bytes only the proxies' samplers read: no pass after this
            # one reads a curve the file no longer plays.
            self._mc._compact_bin(edit)
            logger.info(
                "Curve proxies: %d transport node(s) stripped from the GLB (%s).",
                len(removed),
                ", ".join(removed),
            )
            return removed

    def _remove_glb_nodes(self, gltf: Dict[str, Any], doomed: Set[int]) -> List[str]:
        """Remove the nodes at the indices *doomed* names from *gltf*, every
        channel that animates one, and the samplers and accessors only those
        channels read (``GlbEdit._keep_channels``: a sampler left behind is a
        span every later pass reads -- the take holds, the presence gate's
        tail, the compaction -- and the file no longer plays); renumber every
        reference to the rest -- children, scenes, skins and channel targets
        (no record names a node by index). JSON only: the bytes
        the dropped accessors read are :meth:`GlbEdit.compact_bin`'s, which a
        caller holding the session runs. Returns the removed nodes' names, in
        file order."""
        nodes = gltf.get("nodes") or []
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

        def animates_doomed(channel: Dict[str, Any], _animation: Dict[str, Any]) -> bool:
            return (channel.get("target") or {}).get("node") in doomed

        _dropped, released = GlbEdit._keep_channels(gltf, animates_doomed)
        for animation in gltf.get("animations") or []:
            for channel in animation.get("channels") or []:
                target = channel.get("target") or {}
                if isinstance(target.get("node"), int):
                    target["node"] = remap[target["node"]]
        GlbEdit._drop_unreferenced(gltf, released)
        # No record follows the renumber: every record a GLB carries names its
        # nodes, never an index (RecordSpec.glb_verbatim).
        gltf["nodes"] = kept
        return [str(nodes[i].get("name") or i) for i in sorted(doomed)]

    @staticmethod
    def _node_user_property(node: Dict[str, Any], key: str) -> Any:
        """One user property off a node's extras, in either on-disk shape.

        FBX2glTF nests user properties under ``extras.fromFBX.userProperties``
        as ``{"type", "value"}`` records; a native glTF writer puts a plain
        value at ``extras[key]``.
        """
        return GlbEdit.user_property(node, key)

    #: Primitive extensions that name the primitive's attributes a second
    #: time -- Draco maps each to its compressed stream -- so an attribute
    #: dropped from ``attributes`` alone would leave the extension naming one
    #: that is gone. Same "bail rather than guess" contract as
    #: :attr:`_ACCESSOR_REFERRING_EXTENSIONS`.
    _ATTRIBUTE_REFERRING_EXTENSIONS = frozenset({"KHR_draco_mesh_compression"})

    def prune_glb_unused_skins(self, glb: GlbTarget) -> Dict[str, int]:
        counts = {"skins": 0, "attributes": 0, "bytes": 0}
        with self._mc.open_glb(glb) as edit:
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
                    if self._ATTRIBUTE_REFERRING_EXTENSIONS & set(
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
            if self._mc._drop_orphaned_accessors(edit, released):
                counts["bytes"] = self._mc._compact_bin(edit)
            logger.info(
                "Skins: dropped %d skin(s) no node binds and %d JOINTS/WEIGHTS "
                "attribute(s) from meshes no skinned node instantiates (%.2f MB).",
                counts["skins"],
                counts["attributes"],
                counts["bytes"] / 1048576,
            )
        return counts

    def fix_glb_skin_skeletons(self, glb: GlbTarget) -> List[str]:
        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            nodes = gltf.get("nodes") or []
            parents = self._mc._media.node_parents(gltf)
            repaired: List[str] = []
            outcomes: List[str] = []
            for index, skin in enumerate(gltf.get("skins") or []):
                if not isinstance(skin, dict) or not isinstance(
                    skin.get("skeleton"), int
                ):
                    continue
                joints = [j for j in skin.get("joints") or [] if isinstance(j, int)]
                chains = [
                    self._mc._media.node_lineage(parents, joint)
                    for joint in joints
                ]
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

    def fix_glb_nonfinite_keys(self, glb: GlbTarget) -> Dict[str, int]:
        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            accessors = gltf.get("accessors") or []
            readers, times_of, cubic = self._key_readers(gltf)
            blob: Optional[bytearray] = None
            keys = 0
            repaired: Set[Tuple[str, str]] = set()
            hollow: Dict[int, str] = {}  # output -> why it has no key to keep
            placed: Dict[int, List[float]] = {}  # time input -> its times, placed

            # Key times first: every value is placed by them.
            for source in sorted(set(times_of.values())):
                rows = self._floats(edit, source)
                if rows is None:
                    continue
                times = [row[0] for row in rows]
                bad = [k for k, t in enumerate(times) if not math.isfinite(t)]
                if bad:
                    reading = [o for o, s in times_of.items() if s == source]
                    if not self._place_times(times, bad):
                        for output in reading:
                            hollow[output] = "fewer than two finite key times"
                        continue
                    blob = self._splice(
                        edit, blob, source, [(t,) for t in times], bad, 1
                    )
                    accessors[source]["min"] = [min(times)]
                    accessors[source]["max"] = [max(times)]
                    keys += len(bad)
                    who = [r for output in reading for r in readers[output]]
                    repaired.update((clip, label) for clip, label, _ in who)
                    head, labels = self._said(who)
                    logger.warning(
                        "%s: the key times of %s held %d non-finite time(s) of %d, "
                        "which glTF does not allow; each took its place in the "
                        "keys' spacing.",
                        head,
                        labels,
                        len(bad),
                        len(times),
                    )
                placed[source] = times

            for output, who in readers.items():
                if output in hollow:
                    continue
                times = placed.get(times_of[output])
                rows = self._floats(edit, output)
                if rows is None or not times:
                    continue
                count = len(times)
                if len(rows) % count:
                    continue
                group = len(rows) // count  # elements per key (3 per value: a spline)
                values = [list(row) for row in rows]
                finite = [
                    all(
                        math.isfinite(v)
                        for row in values[k * group : (k + 1) * group]
                        for v in row
                    )
                    for k in range(count)
                ]
                if all(finite):
                    continue
                if not any(finite):
                    hollow[output] = "no finite key"
                    continue
                good = [k for k, ok in enumerate(finite) if ok]
                bad = [k for k, ok in enumerate(finite) if not ok]
                rotation = any(path == "rotation" for _, _, path in who)
                self._interpolate(
                    values, times, good, bad, group, output in cubic, rotation
                )
                blob = self._splice(edit, blob, output, values, bad, group)
                accessor = accessors[output]
                if "min" in accessor or "max" in accessor:
                    width = len(values[0])
                    accessor["min"] = [
                        min(row[c] for row in values) for c in range(width)
                    ]
                    accessor["max"] = [
                        max(row[c] for row in values) for c in range(width)
                    ]
                keys += len(bad)
                repaired.update((clip, label) for clip, label, _ in who)
                head, labels = self._said(who)
                logger.warning(
                    "%s: %s held %d non-finite key(s) of %d, which glTF does not "
                    "allow%s; each was interpolated from its finite neighbours by "
                    "time, and held from the nearest at an end.",
                    head,
                    labels,
                    len(bad),
                    count,
                    " (a converter decomposing a degenerate transform, such as a "
                    "zero scale, has no rotation to recover)"
                    if rotation
                    else "",
                )
            if blob is not None:
                edit.replace_rest(blob)

            dropped = 0
            if hollow:
                dropped = self._drop_channels(edit, set(hollow))
                for output, why in hollow.items():
                    head, labels = self._said(readers[output])
                    logger.warning(
                        "%s: %s held %s, which glTF does not allow; the channel is "
                        "dropped (there is no value to give it).",
                        head,
                        labels,
                        why,
                    )
            return {"keys": keys, "channels": len(repaired), "dropped": dropped}

    @classmethod
    def _key_readers(
        cls, gltf: Dict[str, Any]
    ) -> Tuple[Dict[int, List[Tuple[str, str, Any]]], Dict[int, int], Set[int]]:
        """``({output: [(clip, channel label, path)]}, {output: its time input},
        the cubic-spline outputs)``: who reads each animation output, once each."""
        nodes = gltf.get("nodes") or []
        readers: Dict[int, List[Tuple[str, str, Any]]] = {}
        times_of: Dict[int, int] = {}
        cubic: Set[int] = set()
        for position, animation in enumerate(gltf.get("animations") or []):
            clip = str(animation.get("name") or f"animations[{position}]")
            samplers = animation.get("samplers") or []
            for channel in animation.get("channels") or []:
                index = channel.get("sampler")
                if not isinstance(index, int) or not 0 <= index < len(samplers):
                    continue
                sampler = samplers[index] or {}
                output, source = sampler.get("output"), sampler.get("input")
                if not isinstance(output, int) or not isinstance(source, int):
                    continue
                target = channel.get("target") or {}
                reader = (clip, cls._channel_label(nodes, target), target.get("path"))
                if reader not in readers.setdefault(output, []):
                    readers[output].append(reader)
                times_of[output] = source
                if sampler.get("interpolation") == "CUBICSPLINE":
                    cubic.add(output)
        return readers, times_of, cubic

    @staticmethod
    def _channel_label(nodes: List[Dict[str, Any]], target: Dict[str, Any]) -> str:
        """``the <path> channel of node '<name>'`` for a channel *target* (a
        KHR_animation_pointer one by its pointer), as a report names it."""
        pointer = (
            (target.get("extensions") or {}).get("KHR_animation_pointer") or {}
        ).get("pointer")
        node = target.get("node")
        if pointer and node is None:
            return f"the channel animating {pointer!r}"
        named = (
            nodes[node].get("name")
            if isinstance(node, int) and 0 <= node < len(nodes)
            else None
        )
        label = named if named else f"#{node}"
        return f"the {target.get('path')} channel of node {label!r}"

    @staticmethod
    def _said(readers: List[Tuple[str, str, Any]]) -> Tuple[str, str]:
        """``("Animation 'Take'", "the rotation channel of node 'x' and ...")``:
        the clips and the channels *readers* name, each once."""
        clips = list(dict.fromkeys(clip for clip, _, _ in readers))
        labels = list(dict.fromkeys(label for _, label, _ in readers))
        head = ("Animation " if len(clips) == 1 else "Animations ") + ", ".join(
            repr(clip) for clip in clips
        )
        return head, " and ".join(labels)

    def _floats(self, edit: "GlbEdit", index: int) -> Optional[List[Tuple[float, ...]]]:
        """A tightly packed float accessor's elements; ``None`` for any other
        (only a float holds a NaN, and a sparse or interleaved layout is left
        as it is)."""
        accessors = edit.gltf.get("accessors") or []
        if not 0 <= index < len(accessors):
            return None
        if (accessors[index] or {}).get("componentType") != 5126:
            return None
        elements = self._mc._accessor_elements(edit, index)
        if not elements:
            return None
        width = len(elements[0]) // 4
        return [struct.unpack(f"<{width}f", element) for element in elements]

    @staticmethod
    def _place_times(times: List[float], bad: List[int]) -> bool:
        """Give each *bad* key time its place in the finite ones' spacing, in
        place: between two by index, past an end by the end pair's spacing.
        ``False`` (nothing changed) with fewer than two finite times to space
        by. A finite time out of order is not this repair's: it has no one
        right place, and the converter refuses it by name."""
        skip = set(bad)
        good = [k for k in range(len(times)) if k not in skip]
        if len(good) < 2:
            return False
        for k in bad:
            lo = max((j for j in good if j < k), default=None)
            hi = min((j for j in good if j > k), default=None)
            if lo is None:
                lo, hi = hi, min(j for j in good if j > hi)
            elif hi is None:
                lo, hi = max(j for j in good if j < lo), lo
            times[k] = times[lo] + (times[hi] - times[lo]) * (k - lo) / (hi - lo)
        return True

    @staticmethod
    def _interpolate(
        values: List[List[float]],
        times: List[float],
        good: List[int],
        bad: List[int],
        group: int,
        cubic: bool,
        rotation: bool,
    ) -> None:
        """Each non-finite element of the *bad* keys, in place, from the finite
        keys either side by time -- a slerp for a rotation, a lerp otherwise --
        or held from the nearest at an end. A spline's repaired key keeps its
        value's place between its tangents and takes flat ones."""
        from pythontk.math_utils._math_utils import MathUtils
        from pythontk.math_utils.quaternion import Quaternion

        part = group // 3 if cubic else group
        offset = part if cubic else 0
        for k in bad:
            lo = max((j for j in good if j < k), default=None)
            hi = min((j for j in good if j > k), default=None)
            span = times[hi] - times[lo] if lo is not None and hi is not None else 0.0
            u = (times[k] - times[lo]) / span if span > 0.0 else 0.0
            for i in range(offset, offset + part):
                if all(math.isfinite(v) for v in values[k * group + i]):
                    continue
                a = values[lo * group + i] if lo is not None else None
                b = values[hi * group + i] if hi is not None else None
                if a is None or b is None:
                    values[k * group + i] = list(a if b is None else b)
                elif rotation and len(a) == 4:
                    values[k * group + i] = list(Quaternion.slerp(a, b, u))
                else:
                    values[k * group + i] = list(MathUtils.lerp(tuple(a), tuple(b), u))
            if cubic:
                for i in list(range(part)) + list(range(2 * part, group)):
                    values[k * group + i] = [0.0] * len(values[k * group + i])

    @staticmethod
    def _splice(
        edit: "GlbEdit",
        blob: Optional[bytearray],
        index: int,
        values: List[Sequence[float]],
        keys: List[int],
        group: int,
    ) -> bytearray:
        """*keys*' elements of accessor *index* written into a copy of the BIN
        (made on the first write and handed back for the next)."""
        accessor = edit.gltf["accessors"][index]
        view = GlbEdit.bin_view(edit.gltf, accessor) or {}
        start = int(view.get("byteOffset") or 0) + int(accessor.get("byteOffset") or 0)
        if blob is None:
            blob = bytearray(edit.bin_data)
        width = len(values[0])
        for k in keys:
            for i in range(k * group, (k + 1) * group):
                at = start + i * width * 4
                blob[at : at + width * 4] = struct.pack(f"<{width}f", *values[i])
        return blob

    def _drop_channels(self, edit: "GlbEdit", outputs: Set[int]) -> int:
        """Drop every channel reading one of *outputs*, the samplers left
        unread (the rest renumbered: ``GlbEdit._keep_channels``), the
        accessors left orphaned and a clip left empty; how many channels
        went."""

        def reads(channel: Dict[str, Any], animation: Dict[str, Any]) -> bool:
            samplers = animation.get("samplers") or []
            index = channel.get("sampler")
            return (
                isinstance(index, int)
                and 0 <= index < len(samplers)
                and (samplers[index] or {}).get("output") in outputs
            )

        dropped, released = GlbEdit._keep_channels(edit.gltf, reads)
        edit.dirty = True
        self._mc._drop_orphaned_accessors(edit, released)
        self._mc._compact_bin(edit)
        self._mc.prune_glb_animations(edit)
        return dropped

    @staticmethod
    def fbx_media_dirs(fbx_path: str) -> List[str]:
        folder = os.path.dirname(os.path.abspath(fbx_path))
        return [
            folder,
            os.path.join(folder, "sourceimages"),
            os.path.join(os.path.dirname(folder), "sourceimages"),
        ]

    def fix_glb_tangents(self, glb: GlbTarget) -> Dict[str, int]:
        from pythontk.file_utils.mesh_convert.glb.tangents import GlbTangents

        return GlbTangents.repair(glb)

    def restore_glb_skin_weights(self, glb: GlbTarget, fbx: str) -> Dict[str, int]:
        from pythontk.file_utils.mesh_convert.glb.skin_weights import GlbSkinWeights

        return GlbSkinWeights.restore(glb, fbx)

    def restore_glb_rotations(
        self, glb: GlbTarget, fbx: str, fps: Optional[float] = None
    ) -> Dict[str, int]:
        from pythontk.file_utils.mesh_convert.glb.rotations import GlbRotations

        with self._mc.open_glb(glb) as edit:
            # Curve proxies are FBX-only transport (stripped before the
            # animation passes): a ramp at 0 reads as a flat node.
            proxies = {
                index
                for index, node in enumerate(edit.gltf.get("nodes") or [])
                if self._node_user_property(node, self._mc.CURVE_PROXY_MARKER)
            }
            return GlbRotations.restore(
                edit, fbx, fps or self._mc.CONVERSION_BAKE_FPS, skip=proxies
            )

    def _bake_fps(self, args: Sequence[str]) -> float:
        """The rate FBX2glTF bakes at, run with *args*: their
        ``--anim-framerate`` (``bake24`` / ``bake30`` / ``bake60``), else
        :attr:`CONVERSION_BAKE_FPS`."""
        for position, arg in enumerate(args):
            name, _, value = arg.partition("=")
            if name != "--anim-framerate":
                continue
            if not value and position + 1 < len(args):
                value = args[position + 1]
            if value.startswith("bake") and value[4:].isdigit():
                return float(value[4:])
        return float(self._mc.CONVERSION_BAKE_FPS)
