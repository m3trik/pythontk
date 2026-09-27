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
- ``_shadow_rigs`` -- shadow-plane maps and the ``shadow_web`` manifest
- ``_animation``   -- shot clips, the ``animation_web`` manifest, key compaction
- ``_visibility``  -- keyed visibility, authored fades/highlights, previews
- ``_materials``   -- material checks and the per-channel writers
- ``_textures``    -- the web-delivery texture pass and its policy
- ``_images``      -- image embedding, de-duplication and pruning

A mixin reaches the others through ``cls``, which is what keeps a subclass, a
class-level override and ``mock.patch.object(MeshConvert, ...)`` reaching every
call site. The GLB container every pass edits is :class:`GlbEdit`
(``glb/edit.py``), bound here as ``MeshConvert.GlbEdit`` along with the private
buffer helpers the passes call it through.
"""

from pythontk.core_utils.help_mixin import HelpMixin
from pythontk.file_utils.mesh_convert._animation import _AnimationMixin
from pythontk.file_utils.mesh_convert._fbx2gltf import (  # noqa: F401
    FBX2GLTF_PLATFORMS,  # published extapps <= 0.2.2 imports these from HERE;
    FBX2GLTF_VERSION,  # drop once its pythontk floor passes this release
    _Fbx2GltfMixin,
)
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
    _SidecarMixin,
    _LightmapsMixin,
    _ShadowRigsMixin,
    _AnimationMixin,
    _VisibilityMixin,
    _MaterialsMixin,
    _TexturesMixin,
    _ImagesMixin,
    HelpMixin,
):
    """3D mesh format conversion via the godotengine/FBX2glTF CLI.

    Currently supports static-mesh FBX -> GLB (binary glTF 2.0).
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
