# !/usr/bin/python
# coding=utf-8
"""FBX -> GLB: the one build every GLB deliverable goes through.

The Scene Exporters' GLB output and the live WebXR preview are the same file
made for two audiences -- the target platform and the artist checking it --
and they used to reach it by two chains: the exporter's ``create_glb`` and the
preview deliverer's pass registry, each calling :class:`MeshConvert` in its
own order with its own extras (one downsized the FBX first, one pruned dead
textures, one bound the lightmaps outside the conversion). Every "the preview
shows X but the export ships Y" of the last month was a gap between those two.

:class:`GlbPipeline` is the single chain, and the callers hand it dials only:

    downsize  -- ``FbxMedia.downsize`` the FBX's embedded textures to the
                 texture ceiling FIRST, so the converter and every pass after
                 it read a 2K file rather than a 4K one (measured on a
                 production assembly: the push went from 419 s to 333 s);
    convert   -- ``MeshConvert.fbx_to_glb`` with the scene sidecar, the host's
                 live lightmap folders and a *report*: one edit session running
                 the alpha repair, image dedupe, sidecar, dead-texture sweep,
                 lightmaps, shadow rigs, curve-proxy strip, clips, visibility
                 gates, fades and the animation manifest, in the order their
                 own docstrings justify;
    optimize  -- ``MeshConvert.optimize_glb_textures`` LAST, on the closed
                 file, because a KTX2 payload is opaque to every PIL-based pass
                 and nothing may follow the encode.

A failed conversion or texture pass raises: a deliverable that silently shipped
280 MB where the preview showed 8.71 is the outcome one of the old chains
actually produced, and the callers' own logs are where the reason belongs. The
downsize is the exception -- it is a speed win the texture ceiling re-applies
anyway, so its failure is a warning and the original FBX is read.

:meth:`GlbPipeline.envelope` is the matching single path for the scene
sidecar: every producer reads its host's scene state (a DCC concern) and hands
the reader here, so the envelope's construction, its log line and its failure
policy (a bare GLB still beats no GLB) exist once.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

from pythontk.core_utils.logging_mixin import LoggingMixin
from pythontk.file_utils.temp_artifacts import TempArtifacts


class GlbPipeline(LoggingMixin):
    """FBX -> GLB, with every deliverable's passes, for every deliverable."""

    @staticmethod
    def _mesh_convert():
        """The converter, imported on use: it pulls in the managed-binary installer."""
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        return MeshConvert

    @classmethod
    def envelope(
        cls,
        read_sections: Callable[[], Optional[Dict[str, Any]]],
        *,
        source: Dict[str, str],
        asset: Optional[str] = None,
        logger: Any = None,
    ) -> Dict[str, Any]:
        """The scene-sidecar envelope a build applies, from a host's reader.

        *read_sections* is the DCC's scene-state read (``SceneState.read``,
        bound to the export set) -- passed as a callable so its failure is
        handled HERE, the same way for every producer: a read that raises is
        logged and yields an envelope with no sections. Requested-but-empty
        stays distinguishable from switched-off (the caller attaches nothing
        when the sidecar is off), and a bare GLB still beats no GLB.

        Parameters:
            read_sections: Returns the sidecar sections for the export set.
            source: ``{"application", "version"}`` of the authoring host.
            asset: The payload's basename, recorded in the envelope.
            logger: Where the outcome line goes; the class logger otherwise.
        """
        log = logger or cls.logger
        try:
            sections = read_sections() or {}
        except Exception:  # noqa: BLE001 -- a bare GLB still beats no GLB
            log.warning("Scene sidecar skipped.", exc_info=True)
            sections = {}
        envelope = cls._mesh_convert().build_scene_sidecar(
            sections, source=source, asset=asset
        )
        # Names what the sidecar IS, because "riding the GLB" read as a
        # companion file the consumer has to be handed: the sections are
        # written INTO the GLB's own material JSON and a copy is embedded in
        # `extras` as provenance. Nothing outside the .glb is produced.
        log.info(
            "Scene sidecar (%s) written into the GLB's materials (copy embedded "
            "in extras; no companion file).",
            ", ".join(sorted(sections)) or "no sections",
        )
        return envelope

    @classmethod
    def build(
        cls,
        src: str,
        dst: Optional[str] = None,
        *,
        sidecar: Optional[Dict[str, Any]] = None,
        lightmap_dirs: Sequence[str] = (),
        texture_params: Optional[Dict[str, Any]] = None,
        clip_mode: str = "both",
        downsize: bool = True,
        scratch_path: Optional[Callable[[str], str]] = None,
        release_source: Optional[Callable[[str], Any]] = None,
        progress: Optional[Callable[[str], Any]] = None,
        logger: Any = None,
    ) -> Dict[str, Any]:
        """Build the GLB for *src* and report what each stage did.

        Parameters:
            src: The exported FBX.
            dst: Where the GLB goes; ``None`` writes it beside *src*.
            sidecar: A scene-sidecar envelope (:meth:`envelope`) applied inside
                the conversion and embedded in the file.
            clip_mode: Which animation clips the GLB ships -- ``both`` (the
                declared shots AND the whole-timeline stack they were cut
                from), ``shots``, or ``full``. The two halves hold the same
                performance, so a consumer that plays one never reads the
                other; on a production assembly the stack alone was 66.5 MB.
            lightmap_dirs: Where the host keeps its maps NOW, forwarded to the
                lightmap and shadow appliers as their search directories.
            texture_params: ``optimize_glb_textures`` kwargs (``image_format``,
                ``max_size``, ``ktx2_fallback`` ...); ``None`` takes
                :meth:`MeshConvert.web_delivery_texture_params`. Its ``max_size``
                is also the downsize ceiling, so nothing is lost the texture
                pass would have kept.
            downsize: Shrink the FBX's embedded textures to the ceiling before
                converting. Off when the FBX carries no embedded media anyway.
            scratch_path: ``extension -> path`` allocator for the downsized FBX.
                ``None`` uses a scoped :class:`TempArtifacts` store swept when the
                build returns; a caller passing its own allocator owns the file
                (it is listed in the report's ``"scratch"``).
            release_source: Called with *src* once the downsized copy has
                superseded it and nothing here will read it again -- the caller
                decides whether it may be deleted (a producer's durable file
                must not be). Peak scratch is then one FBX, not two.
            progress: Called with a short message before each stage.
            logger: Logger for the stage lines; the class logger otherwise.

        Returns:
            ``{"glb", "src", "scratch", "downsized", "sidecar", "lightmaps",
            "textures"}`` -- the GLB written, the FBX the converter actually
            read, the scratch files minted, the downsize report (or ``None``),
            the sidecar's per-section outcome, the lightmap coverage
            (:meth:`MeshConvert.lightmap_report`) and the texture-pass summary.

        Raises:
            OSError, RuntimeError, ValueError: from the conversion or the
            texture pass -- the deliverable did not get made.
        """
        MeshConvert = cls._mesh_convert()
        log = logger or cls.logger
        params = dict(texture_params or MeshConvert.web_delivery_texture_params())
        report: Dict[str, Any] = {
            "glb": None,
            "src": src,
            "scratch": [],
            "downsized": None,
            "sidecar": {},
            "lightmaps": None,
            "textures": None,
        }

        def _say(message: str) -> None:
            if progress is not None:
                progress(message)

        own_scratch = (
            None
            if scratch_path is not None
            else TempArtifacts("glb_pipeline", policy="scoped")
        )
        try:
            allocate = scratch_path or (
                lambda extension: own_scratch.path(extension=extension)
            )
            if downsize:
                cls._downsize(src, params, allocate, release_source, report, log, _say)

            _say("GLB: converting the FBX…")
            log.info("Converting FBX to GLB...")
            conversion: Dict[str, Any] = {}
            glb = MeshConvert.fbx_to_glb(
                report["src"],
                dst=dst,
                overwrite=True,
                auto_install=True,
                # A DCC has no tty: a prompting install would raise on the
                # first conversion instead of installing.
                prompt=False,
                sidecar=sidecar,
                lightmap_dirs=lightmap_dirs,
                clip_mode=clip_mode,
                report=conversion,
            )
            report["glb"] = glb
            report["sidecar"] = conversion.get("sidecar") or {}
            report["lightmaps"] = conversion.get("lightmaps")

            carrier = params.get("image_format") or "WEBP"
            _say(f"GLB: {carrier} texture pass…")
            summary = MeshConvert.optimize_glb_textures(glb, **params)
            report["textures"] = summary
            log.info(
                MeshConvert.describe_texture_pass(
                    summary, carrier, params.get("max_size") or 0
                )
            )
        finally:
            if own_scratch is not None:
                own_scratch.cleanup(force=True)
        return report

    @classmethod
    def _downsize(cls, src, params, allocate, release_source, report, log, say) -> None:
        """Rewrite *src*'s embedded textures at the ceiling into a NEW scratch FBX.

        Never in place: a producer that handed over a durable file keeps it
        untouched. A source that is not a binary FBX (a test stub, an ASCII
        export, a path that is not there yet) is left alone without a word;
        a ceiling of zero means nothing to do.
        """
        from pythontk.file_utils.mesh_convert.fbx_file import FbxFile
        from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia

        ceiling = int(params.get("max_size") or 0)
        if not ceiling or not FbxFile.is_fbx(src):
            return
        say("GLB: downsizing embedded textures…")
        scratch = allocate(".fbx")
        try:
            outcome = FbxMedia.downsize(src, scratch, max_size=ceiling)
        except Exception as error:  # noqa: BLE001 -- a speed win, never the build
            log.warning("Payload texture downsize skipped: %s", error)
            return
        report["downsized"] = outcome
        if not outcome["resized"]:
            return  # nothing qualified, so nothing was written
        report["scratch"].append(scratch)
        report["src"] = scratch
        if release_source is not None:
            release_source(src)
        log.info(
            "Payload textures: %d of %d embedded image(s) downsized to %dpx, "
            "%.1f MB -> %.1f MB.",
            outcome["resized"],
            outcome["images"],
            ceiling,
            outcome["before"] / 1e6,
            outcome["after"] / 1e6,
        )
