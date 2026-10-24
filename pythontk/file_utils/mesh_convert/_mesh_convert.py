# !/usr/bin/python
# coding=utf-8
"""FBX -> GLB through FBX2glTF, and the GLB repair and enrichment passes.

:class:`MeshConvert` is the facade. Each job lives in its own module beside
this one as a private mixin, and the facade is composed from them:

- ``_fbx2gltf``    -- the FBX2glTF CLI driver (:meth:`~MeshConvert.fbx_to_glb`)
  and the repairs of what the converter writes
- ``_sidecar``     -- the scene-sidecar envelope, the FBX handoff, the
  ``data_export`` channels, the rendering policy, ``verify_glb``
- ``_lightmaps``   -- a host DCC's committed bake, bound for the web viewer
- ``_highlights``  -- the bake's lights drawn into copies of its reflection
  probe, for the GLB and its FBX twin
- ``_shadow_rigs`` -- shadow-plane maps and the ``shadows`` manifest
- ``_articulation`` -- articulated rigs, the ``articulation`` manifest
- ``_animation``   -- shot clips, the ``animation`` manifest, key compaction
- ``_visibility``  -- keyed visibility, authored fades/highlights, previews
- ``_materials``   -- material checks and the per-channel writers
- ``_fbx_materials`` -- the ``materials`` record an FBX carries (or its own
  materials, read into the record's families) applied to a GLB
- ``_textures``    -- the web-delivery texture pass and its policy
- ``_images``      -- image embedding, de-duplication and pruning
- ``_gltf2fbx``    -- the reverse direction with no DCC: glTF -> FBX and its
  manifest (:meth:`~MeshConvert.gltf_to_fbx`)

The facade itself holds what spans them: :meth:`~MeshConvert.finish_fbx_deliverable`,
the passes a written FBX deliverable takes. A mixin reaches the others through
``cls``, which is what keeps a subclass, a class-level override and
``mock.patch.object(MeshConvert, ...)`` reaching every call site. The GLB container every pass edits is :class:`GlbEdit`
(``glb/edit.py``), bound here as ``MeshConvert.GlbEdit`` along with the private
buffer helpers the passes call it through.
"""

import logging
from typing import Any, Dict, Iterable, Optional, Sequence

from pythontk.core_utils.help_mixin import HelpMixin
from pythontk.file_utils.mesh_convert._animation import _AnimationMixin
from pythontk.file_utils.mesh_convert._articulation import _ArticulationMixin
from pythontk.file_utils.mesh_convert._fbx_materials import _FbxMaterialsMixin
from pythontk.file_utils.mesh_convert._fbx2gltf import (  # noqa: F401
    FBX2GLTF_PLATFORMS,  # published extapps <= 0.2.2 imports these from HERE;
    FBX2GLTF_VERSION,  # drop once its pythontk floor passes this release
    _Fbx2GltfMixin,
)
from pythontk.file_utils.mesh_convert._gltf2fbx import _Gltf2FbxMixin
from pythontk.file_utils.mesh_convert._highlights import _HighlightsMixin
from pythontk.file_utils.mesh_convert._images import _ImagesMixin
from pythontk.file_utils.mesh_convert._lightmaps import _LightmapsMixin
from pythontk.file_utils.mesh_convert._materials import _MaterialsMixin
from pythontk.file_utils.mesh_convert._shadow_rigs import _ShadowRigsMixin
from pythontk.file_utils.mesh_convert._sidecar import _SidecarMixin
from pythontk.file_utils.mesh_convert._textures import _TexturesMixin
from pythontk.file_utils.mesh_convert._visibility import _VisibilityMixin
from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit, GlbTarget


class MeshConvert(
    _Fbx2GltfMixin,
    _Gltf2FbxMixin,
    _SidecarMixin,
    _LightmapsMixin,
    _HighlightsMixin,
    _ShadowRigsMixin,
    _ArticulationMixin,
    _AnimationMixin,
    _VisibilityMixin,
    _FbxMaterialsMixin,
    _MaterialsMixin,
    _TexturesMixin,
    _ImagesMixin,
    HelpMixin,
):
    """3D mesh format conversion via the godotengine/FBX2glTF CLI.

    Currently supports static-mesh FBX -> GLB (binary glTF 2.0), and the
    reverse -- glTF -> FBX -- in pure Python (:meth:`gltf_to_fbx`).
    The FBX2glTF binary is fetched on first use into the pythontk-managed
    tools directory under ``~/.pythontk/tools/`` (overridable via
    ``PYTHONTK_TOOLS_DIR``).

    Note: godotengine/FBX2glTF only ships an x86_64 build for macOS.
    Apple Silicon (arm64) Macs run it transparently via Rosetta 2,
    which must be installed (``softwareupdate --install-rosetta``).
    """

    #: The open-GLB handle every repair operates on (``glb/edit.py``).
    GlbEdit = GlbEdit

    # The container's layout tables and buffer bookkeeping, under the names the
    # passes (and their callers) have always used. Each is the SAME object as
    # on :class:`GlbEdit`, never a copy.
    TEXTURE_CONTAINER_EXTENSIONS = GlbEdit.TEXTURE_CONTAINER_EXTENSIONS
    ACCESSOR_COMPONENT_TYPES = GlbEdit.ACCESSOR_COMPONENT_TYPES
    ACCESSOR_TYPE_COUNT = GlbEdit.ACCESSOR_TYPE_COUNT
    _ACCESSOR_REFERRING_EXTENSIONS = GlbEdit._ACCESSOR_REFERRING_EXTENSIONS
    _read_glb = GlbEdit.read
    _write_glb = staticmethod(GlbEdit.write)
    _append_bin_views = staticmethod(GlbEdit.append_bin_views)
    _compact_bin = staticmethod(GlbEdit.compact_bin)
    _relocate_embedded_images = GlbEdit.relocate_embedded_images
    _bin_view = staticmethod(GlbEdit.bin_view)
    _accessor_elements = GlbEdit.accessor_elements
    _map_accessor_refs = GlbEdit.map_accessor_refs
    _referenced_accessors = GlbEdit.referenced_accessors
    _drop_orphaned_accessors = GlbEdit.drop_orphaned_accessors
    _release_animation_payload = GlbEdit.release_animation_payload

    @classmethod
    def open_glb(cls, glb: GlbTarget):
        """Yield an open :class:`GlbEdit` for *glb*, writing once on close.

        *glb* is a path, **or an already-open session** -- in which case it is
        yielded as-is and its owner keeps responsibility for the write. That
        second form is what lets these repairs compose: each one takes either,
        so running three against a path costs three read/write cycles, while
        wrapping the same three in one ``open_glb`` costs one::

            with MeshConvert.open_glb(path) as glb:
                MeshConvert.set_glb_base_color(glb, base_color)
                MeshConvert.set_glb_emissive(glb, emissive)

        Nothing is written when the body raises -- a half-applied edit must not
        reach disk -- nor when no editor set :attr:`GlbEdit.dirty`. The session
        is :meth:`GlbEdit.open`'s.
        """
        return GlbEdit.open(glb)

    @classmethod
    def finish_fbx_deliverable(
        cls,
        fbx: str,
        glb: Optional[str] = None,
        *,
        search_dirs: Sequence[str] = (),
        walk_root: str = "",
        files: Iterable[Any] = (),
        hdr_codec: Any = None,
        logger: Optional[logging.Logger] = None,
    ) -> Dict[str, Any]:
        """The passes a written FBX deliverable takes, so it reads as its GLB twin does.

        The one sequence every host's FBX hand-off runs (the Scene Exporters,
        the Unity bridges), on the written file and after the GLB built from
        it, whose copies it can share:

        1. :meth:`embed_fbx_highlight_probes` -- the probe's highlight copies
           (*glb*'s, or a build's of *fbx* without its takes: *search_dirs*,
           *hdr_codec*), from the file as written, the one a GLB export reads;
        2. :meth:`FbxMedia.split_pivot_instances` -- an instance whose pivots
           differ from its siblings' gets a mesh of its own, which an engine
           would otherwise move by another's;
        3. :meth:`FbxMedia.drop_host_lighting_media` -- the host's own
           environment maps leave the file;
        4. :meth:`FbxMedia.embed_dependencies` -- every file the records name
           (*search_dirs*, *walk_root*, *files*), the copies among them;
        5. :meth:`FbxMedia.drop_redundant_properties` -- each baked object's
           ``lightmapInfo`` marker, which its ``lightmaps`` entry
           states whole, and the carrier's empty channels: the file states
           each thing once;
        6. :meth:`_finish_fbx_handoff` -- the handoff stated again from the
           records as the passes left them (the copies change what the
           lightmaps row says).

        Each is guarded: a failure is warned and the next still runs, on the
        file as it stands. A file that is not a binary FBX is left alone.

        Parameters:
            fbx: The FBX deliverable, rewritten in place.
            glb: The GLB this export built from *fbx*, if any.
            search_dirs: Where the scene keeps the files its records name.
            walk_root: Where a name those miss is searched for, recursively.
            files: Further files to embed (paths or ``(name, path)`` pairs).
            hdr_codec: The HDR encoder for a highlight build.
            logger: Where the outcome is said; this module's otherwise.

        Returns:
            ``{"instances", "host_lighting", "highlights", "dependencies",
            "redundant", "handoff"}`` -- each pass's report (``highlights``
            the copies' names, ``handoff`` whether it changed), ``None``
            for a pass that failed; ``{}`` for a file that is not a binary FBX.
        """
        from pythontk.file_utils.mesh_convert.fbx_file import FbxFile
        from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia

        log = logger or logging.getLogger(__name__)
        if not FbxFile.is_fbx(fbx):
            return {}
        passes = (
            (
                "highlights",
                lambda: cls.embed_fbx_highlight_probes(
                    fbx,
                    glb,
                    search_dirs=search_dirs,
                    hdr_codec=hdr_codec,
                    logger=log,
                ),
            ),
            ("instances", lambda: FbxMedia.split_pivot_instances(fbx)),
            ("host_lighting", lambda: FbxMedia.drop_host_lighting_media(fbx)),
            (
                "dependencies",
                lambda: FbxMedia.embed_dependencies(
                    fbx,
                    search_dirs=list(search_dirs),
                    walk_root=walk_root,
                    files=files,
                    logger=log,
                ),
            ),
            (
                "redundant",
                lambda: FbxMedia.drop_redundant_properties(
                    fbx, markers=(cls.LIGHTMAP_INFO_KEY,)
                ),
            ),
            ("handoff", lambda: cls._finish_fbx_handoff(fbx)),
        )
        report: Dict[str, Any] = {}
        for name, run in passes:
            try:
                report[name] = run()
            except Exception as error:  # noqa: BLE001 -- the deliverable already shipped
                log.warning(f"FBX deliverable: the {name} pass failed: {error}")
                log.debug("FBX deliverable pass failed.", exc_info=True)
                report[name] = None
        split = report.get("instances") or {}
        if split.get("models"):
            log.info(
                "FBX instances: %s each got a mesh of their own (their pivots "
                "differ from their instances').",
                ", ".join(split["models"]),
            )
        if split.get("deformed"):
            log.warning(
                "FBX instances: %s stay shared though their instances' pivots "
                "differ (a deformer drives them); an engine places all but one "
                "of them off.",
                ", ".join(split["deformed"]),
            )
        dropped = report.get("host_lighting") or {}
        if dropped.get("textures"):
            log.info(
                "FBX: dropped %d host-lighting texture(s) no engine reads (%s).",
                dropped["textures"],
                ", ".join(sorted(set(dropped["media"]))) or "no media",
            )
        return report
