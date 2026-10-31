# !/usr/bin/python
# coding=utf-8
"""FBX -> GLB through FBX2glTF, and the GLB repair and enrichment passes.

:class:`MeshConvert` is the facade. Each job lives in its own module beside
this one as a private part, a collaborator the facade builds on the class it
is read through and hands that job's calls to:

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
  manifest (:meth:`~MeshConvert.gltf_to_fbx`), on its builder ``_GltfToFbx``

The facade holds the whole public surface -- each method's signature and
docstring, each part's constants under the names callers use, the retired
names -- and what spans the parts: :meth:`~MeshConvert.finish_fbx_deliverable`,
the passes a written FBX deliverable takes. A part holds the facade
(``self._mc``) and reaches every name it exposes through it, which is what
keeps a subclass, a class-level override and ``mock.patch.object(MeshConvert,
...)`` reaching every call site; its own helpers it reaches through ``self``,
a sibling part's through the facade's accessor (``self._mc._lightmaps``). The
GLB container every pass edits is :class:`GlbEdit` (``glb/edit.py``), bound
here as ``MeshConvert.GlbEdit`` along with the private buffer helpers the
passes call it through.
"""

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np  # loaded with _highlights; its helpers' signatures name it

# Eager: the retired names' tables below are the registry's and the handoff's
# own, read at class definition.
from pythontk.core_utils.class_property import ClassProperty
from pythontk.core_utils.deprecation import Deprecation
from pythontk.core_utils.engines.scene_export.handoff_record import HandoffRecord
from pythontk.core_utils.engines.scene_export.scene_records import (
    RecordSpec,
    SceneRecords,
)
from pythontk.core_utils.help_mixin import HelpMixin
from pythontk.file_utils.mesh_convert._animation import _Animation
from pythontk.file_utils.mesh_convert._articulation import _Articulation
from pythontk.file_utils.mesh_convert._fbx_materials import FbxSource, _FbxMaterials
from pythontk.file_utils.mesh_convert._fbx2gltf import (  # noqa: F401
    FBX2GLTF_PLATFORMS,  # published extapps <= 0.2.2 imports these from HERE;
    FBX2GLTF_VERSION,  # drop once its pythontk floor passes this release
    _Fbx2Gltf,
)
from pythontk.file_utils.mesh_convert._gltf2fbx import (
    GLTF_EXTENSIONS,
    GLTF_UNDECODABLE_EXTENSIONS,
    _GltfToFbx,
)
from pythontk.file_utils.mesh_convert._highlights import _Highlights
from pythontk.file_utils.mesh_convert._images import _Images
from pythontk.file_utils.mesh_convert._lightmaps import _Lightmaps
from pythontk.file_utils.mesh_convert._materials import _Materials
from pythontk.file_utils.mesh_convert._shadow_rigs import _ShadowRigs
from pythontk.file_utils.mesh_convert._sidecar import _Sidecar
from pythontk.file_utils.mesh_convert._textures import _Textures
from pythontk.file_utils.mesh_convert._visibility import _Visibility
from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit, GlbTarget

if TYPE_CHECKING:
    from pythontk.file_utils.mesh_convert.fbx_file import FbxFile

#: The removal window of every name :class:`MeshConvert` retired 2026-10-20:
#: the handoff's, now ``HandoffRecord``'s, and those that restated a record's
#: declaration, now ``SceneRecords``' alone. Passed by name on each call --
#: the spelling the registry's expiry gate reads (it cannot read a ``**``).
_REMOVE_IN = "0.18.0"
_SINCE = "2026-10-24"


class MeshConvert(HelpMixin):
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

    # Each job's part, built on the class it is read through (this one or a
    # subclass) and handed that job's calls.
    _fbx2gltf = ClassProperty(_Fbx2Gltf)
    _sidecar = ClassProperty(_Sidecar)
    _lightmaps = ClassProperty(_Lightmaps)
    _highlights = ClassProperty(_Highlights)
    _shadow_rigs = ClassProperty(_ShadowRigs)
    _articulation = ClassProperty(_Articulation)
    _animation = ClassProperty(_Animation)
    _visibility = ClassProperty(_Visibility)
    _materials = ClassProperty(_Materials)
    _fbx_materials = ClassProperty(_FbxMaterials)
    _textures = ClassProperty(_Textures)
    _images = ClassProperty(_Images)

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
        6. :meth:`_Sidecar._finish_fbx_handoff` -- the handoff stated again from the
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
            ("handoff", lambda: cls._sidecar._finish_fbx_handoff(fbx)),
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

    # ---------------- _fbx2gltf.py: the FBX2glTF driver and its output repairs

    TOOL_NAME = _Fbx2Gltf.TOOL_NAME
    FBX2GLTF_VERSION = _Fbx2Gltf.FBX2GLTF_VERSION
    DEFAULT_TIMEOUT = _Fbx2Gltf.DEFAULT_TIMEOUT
    TIMEOUT_SECONDS_PER_MB = _Fbx2Gltf.TIMEOUT_SECONDS_PER_MB
    TIMEOUT_SECONDS_PER_NODE_FRAME = _Fbx2Gltf.TIMEOUT_SECONDS_PER_NODE_FRAME
    CONVERSION_BAKE_FPS = _Fbx2Gltf.CONVERSION_BAKE_FPS
    AUTO_TIMEOUT = _Fbx2Gltf.AUTO_TIMEOUT
    CURVE_PROXY_MARKER = _Fbx2Gltf.CURVE_PROXY_MARKER

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
        return cls._fbx2gltf.conversion_timeout(src, fbx)

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
        return cls._fbx2gltf.bake_node_frames(src, fbx)

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
        return cls._fbx2gltf.resolve_binary(required, auto_install, prompt)

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
        rendering: Optional[Mapping[str, Mapping[str, Any]]] = None,
        data_export: Optional[Dict[str, Any]] = None,
        lightmaps: bool = True,
        lightmap_dirs: Sequence[str] = (),
        shadow_dirs: Sequence[str] = (),
        clip_mode: str = "both",
        report: Optional[Dict[str, Any]] = None,
        hdr_codec: Any = None,
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
            sidecar:       A handoff block (``HandoffRecord.block("glb", ...)``) to embed in
                           place of the one the FBX's own ``handoff`` record
                           implies; ``None`` -- every export -- reads it off
                           the file, so the GLB says what its FBX says, and a
                           file with no record gets none. An envelope a host
                           built before the materials record
                           (:meth:`build_scene_sidecar`) is still taken: its
                           sections repair a file that carries no record, until
                           they retire.
            rendering:     The export's choices over the lighting recipe
                           (:meth:`rendering_policy`), laid over what the FBX's
                           handoff states -- a preview's own GLB rows.
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
            hdr_codec:     What encodes the lightmaps and the reflection probe,
                           forwarded to :meth:`apply_glb_lightmaps` -- a host
                           with no cv2 (Blender) hands its own image writer.
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
                           caller that reports rather than logs:
                           ``"materials"`` (what :meth:`apply_glb_materials`
                           wrote), ``"sidecar"`` (an old envelope's
                           per-section outcome, ``{}`` otherwise) and
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
        return cls._fbx2gltf.fbx_to_glb(
            src,
            dst,
            overwrite=overwrite,
            auto_install=auto_install,
            prompt=prompt,
            timeout=timeout,
            extra_args=extra_args,
            sidecar=sidecar,
            rendering=rendering,
            data_export=data_export,
            lightmaps=lightmaps,
            lightmap_dirs=lightmap_dirs,
            shadow_dirs=shadow_dirs,
            clip_mode=clip_mode,
            report=report,
            hdr_codec=hdr_codec,
        )

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
        route gets the ramp from ``render_effects`` instead, so the node is
        removed here, BEFORE the clip rebuild counts its channel as content.

        Nodes are identified by the :attr:`CURVE_PROXY_MARKER` user property,
        never by name, so an artist's own ``foo__bar`` transform is safe.

        Returns:
            The removed node names, in file order.
        """
        return cls._fbx2gltf.strip_glb_curve_proxies(glb)

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
        return cls._fbx2gltf.prune_glb_unused_skins(glb)

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
        return cls._fbx2gltf.fix_glb_skin_skeletons(glb)

    @classmethod
    def fix_glb_nonfinite_keys(cls, glb: GlbTarget) -> Dict[str, int]:
        """Give every non-finite animation key time and value a finite one.

        glTF allows no NaN or infinity in an animation's key times or values,
        and its validator refuses a file holding one. FBX2glTF writes one where
        a node's transform is degenerate: it bakes each node's local matrix per
        frame and decomposes it, and a zero scale leaves no rotation to
        recover, so that frame's quaternion comes out NaN
        (``test_assets/fbx/maya_tangents.fbx`` keys its ``tangents`` node at
        scale 0 on frame 1). :meth:`fbx_to_glb` runs this first, on the
        converter's own output, ahead of every pass that reads a key.

        A non-finite key time takes its place in the finite ones' spacing:
        between two by index, past an end by the end pair's spacing. A
        non-finite value is what the motion had there: interpolated by time
        from the finite keys either side (a slerp for a rotation, a lerp for
        anything else), and held from the nearest finite key at an end. A cubic
        spline's repaired key takes flat tangents. Each repaired output is
        warned about once, naming its animations, its channels and how many
        keys it lost. A channel with no finite value, or fewer than two finite
        key times, is dropped with a warning: there is nothing to give it. A
        finite key time out of order is left (it has no one right place;
        :meth:`gltf_to_fbx` refuses it by name). Float accessors only, the one
        kind that can hold a NaN; a sparse or interleaved one is left as it is.

        Parameters:
            glb: ``.glb`` path (modified in place) or an open :class:`GlbEdit`.

        Returns:
            ``{"keys": n, "channels": n, "dropped": n}`` -- the key times and
            values repaired, the channels reading them, and the channels
            dropped.
        """
        return cls._fbx2gltf.fix_glb_nonfinite_keys(glb)

    @staticmethod
    def fbx_media_dirs(fbx_path: str) -> List[str]:
        """Where a file an FBX names loosely is looked for beside it.

        The FBX's own directory, a ``sourceimages`` folder inside it, and one
        beside it (the Maya project layout). A host that converts a STAGED
        copy passes the deliverable's own set as ``shadow_dirs``, so the
        search still reaches the folder the file ships to.

        Parameters:
            fbx_path: The FBX (or the path it will be delivered to).

        Returns:
            The three directories, nearest first.
        """
        return _Fbx2Gltf.fbx_media_dirs(fbx_path)

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
        return cls._fbx2gltf.fix_glb_tangents(glb)

    @classmethod
    def restore_glb_skin_weights(cls, glb: GlbTarget, fbx: str) -> Dict[str, int]:
        """Give every skinned mesh the weights its FBX clusters hold.

        FBX2glTF merges identical vertices across the whole file and compares
        everything but the skin weights, so a duplicated rigged mesh whose
        chain runs another way ships the first mesh's weights read against its
        own joints -- see
        :class:`~pythontk.file_utils.mesh_convert.glb.skin_weights.GlbSkinWeights`.

        Parameters:
            glb: Path to a ``.glb``, modified in place, or an open session.
            fbx: The FBX the GLB was converted from.

        Returns:
            ``{"meshes": n, "vertices": n, "ambiguous": n}`` -- meshes and
            vertices rewritten, and vertices left as shipped because their
            control point could not be told apart.
        """
        return cls._fbx2gltf.restore_glb_skin_weights(glb, fbx)

    # Read beyond the facade (its tests, sibling modules): the part's own.
    @classmethod
    def _platform_exe_name(cls) -> str:
        return cls._fbx2gltf._platform_exe_name()

    _renumber_block_nodes = staticmethod(_Fbx2Gltf._renumber_block_nodes)

    @classmethod
    def _remove_glb_nodes(cls, gltf: Dict[str, Any], doomed: Set[int]) -> List[str]:
        return cls._fbx2gltf._remove_glb_nodes(gltf, doomed)

    # ---------------- _sidecar.py: the handoff block, the data_export channels

    HOST_PROPERTIES = _Sidecar.HOST_PROPERTIES
    DATA_EXPORT_NODE = _Sidecar.DATA_EXPORT_NODE

    # ---------------------------------------------------- retired 2026-10-20
    # The handoff's names, now HandoffRecord's.
    SIDECAR_VERSION = Deprecation.class_attribute(
        2, "HandoffRecord.VERSION", remove_in=_REMOVE_IN, since=_SINCE
    )
    HANDOFF_INSTRUCTIONS = Deprecation.class_attribute(
        HandoffRecord.instructions("glb"),
        "HandoffRecord.instructions('glb')",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    FBX_HANDOFF_VERSION = Deprecation.class_attribute(
        HandoffRecord.VERSION,
        "HandoffRecord.VERSION",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    FBX_HANDOFF_CHANNEL = Deprecation.class_attribute(
        HandoffRecord.KEY, "HandoffRecord.KEY", remove_in=_REMOVE_IN, since=_SINCE
    )
    FBX_HANDOFF_CHANNELS = Deprecation.class_attribute(
        {
            spec.key: spec.description
            for spec in SceneRecords.deliverable()
            if spec is not SceneRecords.HANDOFF
        },
        "SceneRecords.deliverable() -- each RecordSpec's description",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    FBX_HANDOFF_INSTRUCTIONS = Deprecation.class_attribute(
        HandoffRecord.instructions("fbx"),
        "HandoffRecord.instructions('fbx')",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    RENDERING_POLICY = Deprecation.class_attribute(
        HandoffRecord.RENDERING_POLICY,
        "HandoffRecord.RENDERING_POLICY",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    RENDERING_CARRIERS = Deprecation.class_attribute(
        {carrier: {} for carrier in HandoffRecord.CARRIERS},
        "HandoffRecord.lighting(carrier) -- the policy is values alone",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    HANDOFF_ASSEMBLY = Deprecation.class_attribute(
        HandoffRecord.CARRIERS["glb"].own,
        "HandoffRecord.CARRIERS['glb'].own",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    LIGHTING_ASSEMBLY = Deprecation.class_attribute(
        tuple(HandoffRecord.lighting("glb")),
        "HandoffRecord.lighting('glb')",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    # Names that restated a record's declaration: a record is read off
    # SceneRecords alone, and its GLB block takes the record's key.
    FBX_TAKES_KEY = Deprecation.class_attribute(
        "fbx_takes",
        "SceneRecords.SHOTS -- each shot's range rides its clip",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    SHOT_METADATA_KEY = Deprecation.class_attribute(
        SceneRecords.SHOTS.key,
        "SceneRecords.SHOTS.key",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    ANIMATION_WEB_KEY = Deprecation.class_attribute(
        SceneRecords.SHOTS.key,
        "SceneRecords.SHOTS.key",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    ANIMATION_WEB_VERSION = Deprecation.class_attribute(
        SceneRecords.SHOTS.glb.version,
        "SceneRecords.SHOTS.glb.version",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    LIGHTMAP_METADATA_KEY = Deprecation.class_attribute(
        SceneRecords.LIGHTMAPS.key,
        "SceneRecords.LIGHTMAPS.key",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    LIGHTMAP_METADATA_VERSION = Deprecation.class_attribute(
        SceneRecords.LIGHTMAPS.version,
        "SceneRecords.LIGHTMAPS.version",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    LIGHTMAP_WEB_KEY = Deprecation.class_attribute(
        SceneRecords.LIGHTMAPS.key,
        "SceneRecords.LIGHTMAPS.key",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    SHADOW_METADATA_KEY = Deprecation.class_attribute(
        SceneRecords.SHADOWS.key,
        "SceneRecords.SHADOWS.key",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    SHADOW_METADATA_VERSION = Deprecation.class_attribute(
        SceneRecords.SHADOWS.version,
        "SceneRecords.SHADOWS.version",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    SHADOW_WEB_KEY = Deprecation.class_attribute(
        SceneRecords.SHADOWS.key,
        "SceneRecords.SHADOWS.key",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    SHADOW_WEB_NODE_FIELDS = Deprecation.class_attribute(
        SceneRecords.SHADOWS.glb.node_fields,
        "SceneRecords.SHADOWS.glb.node_fields",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    ARTICULATION_KEY = Deprecation.class_attribute(
        SceneRecords.ARTICULATION.key,
        "SceneRecords.ARTICULATION.key",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    ARTICULATION_VERSION = Deprecation.class_attribute(
        SceneRecords.ARTICULATION.version,
        "SceneRecords.ARTICULATION.version",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    ARTICULATION_WEB_KEY = Deprecation.class_attribute(
        SceneRecords.ARTICULATION.key,
        "SceneRecords.ARTICULATION.key",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    ARTICULATION_WEB_NODE_FIELDS = Deprecation.class_attribute(
        SceneRecords.ARTICULATION.glb.node_fields,
        "SceneRecords.ARTICULATION.glb.node_fields",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    VISIBILITY_TRACKS_KEY = Deprecation.class_attribute(
        SceneRecords.VISIBILITY.key,
        "SceneRecords.VISIBILITY.key",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    VISIBILITY_TRACKS_VERSION = Deprecation.class_attribute(
        SceneRecords.VISIBILITY.version,
        "SceneRecords.VISIBILITY.version",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    # The lightmap record's folder hints, which no record has carried since
    # 0.11.0 (the maps' folders ride the private ``lightmap_dirs``).
    LOCATE_HINT_KEY = Deprecation.class_attribute(
        "dir",
        "nothing -- no record carries a folder hint",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    LOCATE_HINT_KEYS = Deprecation.class_attribute(
        ("dir", "dirs"),
        "nothing -- no record carries a folder hint",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    SIDECAR_APPLIERS = Deprecation.class_attribute(
        _Sidecar._SIDECAR_APPLIERS,
        "the FBX's materials record (MeshConvert.apply_glb_materials)",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    #: The per-section outcome a GLB carried beside the envelope -- the
    #: build's own report, not the file's.
    SIDECAR_APPLIED_KEY = Deprecation.class_attribute(
        "scene_sidecar_applied",
        "nothing -- the build reports what it applied (fbx_to_glb's report)",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )

    @ClassProperty
    @Deprecation.symbol(
        "pythontk.geo_utils.shadow_record.ShadowPlane -- every field is required",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    def SHADOW_PLANE_DEFAULTS(cls) -> Dict[str, Any]:
        """A shadow plane's starting values, as 0.16.1 served them:
        ``ShadowPlane``'s defaults, ``None`` for no source or contact (the
        plane spells them ``""``); computed on a read, so no import pays for
        the retired name."""
        from pythontk.geo_utils.shadow_record import ShadowPlane

        plane = {k: v for k, v in vars(ShadowPlane()).items() if v is not None}
        return {**plane, "source": None, "contact": None}

    @classmethod
    @Deprecation.symbol(
        "HandoffRecord.rendering_policy", remove_in=_REMOVE_IN, since=_SINCE
    )
    def rendering_policy(
        cls,
        overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
        carrier: str = "glb",
    ) -> Dict[str, Any]:
        """The lighting recipe with a deliverable's own choices laid over it
        (``HandoffRecord.rendering_policy``); *carrier* is checked, its values the
        same on every carrier."""
        return cls._sidecar.rendering_policy(overrides, carrier)

    @classmethod
    @Deprecation.symbol(
        "HandoffRecord.block('glb', ...) -- a GLB carries extras.handoff, not an envelope",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    def build_scene_sidecar(
        cls,
        sections: Optional[Dict[str, Any]],
        source: Dict[str, str],
        asset: Optional[str] = None,
        rendering: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """The envelope a host built before the materials record: the GLB's
        handoff block with the *sections* a converter applies to a file that
        carries no record (:meth:`fbx_to_glb`)."""
        return cls._sidecar.build_scene_sidecar(sections, source, asset, rendering)

    @classmethod
    def strip_fbx_records(cls, gltf: dict) -> int:
        """Drop what the source FBX's records left in a converted glTF's nodes.

        FBX2glTF transcribes every user property into node extras
        (``extras.fromFBX.userProperties``; a native glTF export writes them
        at the top of the node's extras), so a GLB built from an FBX carried
        the FBX's own account of itself a second time: every channel of the
        ``data_export`` carrier and each baked object's ``lightmapInfo``
        marker. The build consumes them -- the materials stated, the
        lightmaps bound, the clips joined -- and states each in glTF's own
        terms (the materials, ``extras.handoff`` and its sibling blocks), so
        what remains is a repeat, and a record of another file: the handoff
        it held describes the FBX. Run last, once every pass has read them.

        Removes every record of a carrier node and the marker from every
        other node, and the converter's own notes with them -- what a host
        writes on every node of its own accord (:attr:`HOST_PROPERTIES`) and
        FBX2glTF's flags beside the user properties, a material's ``fromFBX``
        (its shading model and ``isTruePBR``), and the clip origin each
        animation carried between the passes (``extras.zero_frame``, which
        ``extras.animation`` states once): nothing reads them once the passes
        ran. A user property the artist authored is content and stays, under
        ``fromFBX.userProperties`` as the converter wrote it, as does another
        program's extras. An extras object left empty goes, and so does a
        carrier node that holds no content (no mesh, camera, skin or child)
        (:meth:`_remove_glb_nodes`).

        Returns:
            How many of the file's objects changed -- nodes, a removed one
            included, materials and animations.
        """
        return cls._sidecar.strip_fbx_records(gltf)

    @classmethod
    def finish_glb(
        cls,
        glb: GlbTarget,
        rendering: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> bool:
        """State a GLB as every deliverable states itself, whatever wrote it.

        What the FBX route's build does at its end, for a GLB a host's own
        exporter wrote (blendertk's ``LightmapWebExport``), run once the file
        is: each block of ours it left in its first scene's extras -- where a
        native exporter puts a scene's custom properties, as JSON text --
        moves to the root's as an object (:meth:`_lift_scene_blocks`); the
        handoff its carrier's ``handoff`` record implies is embedded and named
        against those blocks (:meth:`_fbx_handoff`); and the records leave its
        nodes (:meth:`strip_fbx_records`). A file whose carrier holds no
        handoff record gets none -- a raw export claims no approved scene.
        Idempotent.

        Parameters:
            glb: ``.glb`` path (modified in place) or an open :class:`GlbEdit`.
            rendering: The export's choices over the lighting recipe, laid
                over the record's.

        Returns:
            Whether the file changed.
        """
        return cls._sidecar.finish_glb(glb, rendering)

    @classmethod
    @Deprecation.symbol(
        "MeshConvert.strip_fbx_records -- every record the FBX carried leaves",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    def strip_fbx_handoff(cls, gltf: dict) -> int:
        """Drop the FBX's handoff block from a converted glTF's node extras --
        one of the records :meth:`strip_fbx_records` now drops together.

        Returns:
            How many nodes were stripped.
        """
        return cls._sidecar.strip_fbx_handoff(gltf)

    @classmethod
    @Deprecation.symbol(
        "HandoffRecord.block('fbx', channels, source)",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    def build_fbx_handoff(
        cls,
        channels: Union[Iterable[str], Mapping[str, Any]],
        source: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """The FBX's handoff block for a carrier holding *channels*, or ``{}``
        when it holds no record (``HandoffRecord.block``)."""
        return cls._sidecar.build_fbx_handoff(channels, source)

    @classmethod
    @Deprecation.symbol(
        "MeshConvert.apply_glb_materials -- the FBX's materials record states "
        "every material, so an envelope repairs none (a GLB's statement of "
        "itself: MeshConvert.embed_glb_handoff)",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    def apply_scene_sidecar(
        cls, glb: GlbTarget, sidecar: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Apply a scene-sidecar envelope's sections to a GLB and embed it.

        Retired with the hosts' section readers: what the sections repaired,
        the ``materials`` record states. It still applies an envelope's
        sections, as it did, until the removal (:meth:`_apply_scene_sidecar`).
        """
        return cls._apply_scene_sidecar(glb, sidecar)

    @classmethod
    def embed_glb_handoff(cls, edit: "GlbEdit", handoff: Mapping[str, Any]) -> None:
        """Write *handoff* (``HandoffRecord.block("glb", ...)``) as the open GLB's
        ``extras.handoff``, name the root-``extras`` blocks the file carries
        in it (:meth:`_finish_glb_handoff`), and stamp this pipeline in
        ``asset.generator``.

        Parameters:
            edit: The open session; marked dirty.
            handoff: The block. A copy is written: the caller's stays theirs.
        """
        return cls._sidecar.embed_glb_handoff(edit, handoff)

    @classmethod
    @Deprecation.symbol(
        "MeshConvert.embed_glb_handoff -- a GLB carries extras.handoff",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    def embed_scene_sidecar(
        cls,
        edit: "GlbEdit",
        sidecar: Dict[str, Any],
        summary: Mapping[str, str],
        *,
        described: Iterable[str] = (),
    ) -> None:
        """Embed *sidecar* as the GLB's handoff block (:meth:`embed_glb_handoff`):
        an envelope from a host that predates the block, as the block its
        source and lighting make. *summary* is the build's report, no longer
        the file's."""
        return cls._sidecar.embed_scene_sidecar(
            edit, sidecar, summary, described=described
        )

    @classmethod
    @Deprecation.symbol(
        "MapFactory.foreign_packings over the materials record's maps "
        "(MaterialFamilies.texture_files)",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    def sidecar_foreign_packings(
        cls,
        sidecar: Optional[Dict[str, Any]],
        target: str = "ORM",
        workflow: Optional[str] = None,
    ) -> Dict[str, str]:
        """``{path: map type}`` for envelope textures authored for another engine.

        Retired with the hosts' section readers: an exporter's check now asks
        :meth:`MapFactory.foreign_packings` of the maps its materials record
        names, so no envelope is read for it.

        The pre-flight half of the compatibility story: given a built scene
        sidecar, which of the maps it is about to carry into a glTF deliverable
        are packed for a *different* engine family? Answers before any
        conversion runs, so an exporter can gate on it (see mayatk's /
        blendertk's ``check_material_compatibility``) rather than discovering it
        in the log afterwards.

        Lives here because :meth:`build_scene_sidecar` owns the envelope schema
        and mayatk/blendertk cannot import each other -- written per DCC, the
        walk over ``sections`` would drift the moment a section was added. The
        *judgement* is not duplicated either: it delegates to
        :meth:`MapFactory.foreign_packings`, the single predicate.

        Every string value in every section is offered, so a section added
        later is covered without editing this. That is safe because the
        predicate only reports paths it can resolve to a declared map type.

        Parameters:
            sidecar: A :meth:`build_scene_sidecar` envelope, or ``None``.
            target: The map type the deliverable wants (``"ORM"`` for glTF).
            workflow: A registry workflow name instead of a map-type *target* --
                the form an exporter's texture-template selection arrives in.
                Takes precedence over *target*; see
                :meth:`MapFactory.foreign_packings` for the two judgements.

        Returns:
            ``{path: map type}``; empty for ``None``, an empty envelope, or a
            source set that is already appropriate.
        """
        return cls._sidecar.sidecar_foreign_packings(sidecar, target, workflow)

    @classmethod
    def read_glb_handoff(cls, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        """The handoff block a GLB carries (``extras.handoff``), or ``None``.

        What a downstream tool (or a test) calls to get the file's account of
        itself back out with no side files. Reads only the JSON chunk; the
        geometry is never touched.
        """
        return cls._sidecar.read_glb_handoff(glb)

    @classmethod
    @Deprecation.symbol(
        "MeshConvert.read_glb_handoff -- a GLB carries extras.handoff",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    def read_scene_sidecar(cls, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        """The handoff block a GLB carries (:meth:`read_glb_handoff`)."""
        return cls.read_glb_handoff(glb)

    @classmethod
    def verify_glb(cls, glb: GlbTarget) -> Dict[str, Any]:
        """Check a delivered GLB against what it says of itself.

        The reader a *recipient* runs: read-only and side-file-free, so one on
        another machine, in another DCC, or holding nothing but the ``.glb``
        can run it, and it can never be what damages the asset it inspects.

        Decodes one channel of each ORM image (see
        :meth:`suspect_orm_materials`); everything else reads the JSON chunk.
        That cost is deliberate here and nowhere else -- this is an explicit
        recipient-side call, not an export step.

        Returns:
            A report dict, always carrying ``ok`` (bool), ``problems`` (the
            findings that made ``ok`` False) and ``notes`` (observations that
            did not), plus what was inspected: ``handoff`` (``version`` and
            ``source`` of ``extras.handoff``, or ``None``), ``orm`` (per
            :meth:`suspect_orm_materials`, when it found a destructive
            packing), ``animation`` (clip counts, rate and default clip, when
            the file carries that block), ``extensions`` (``required`` /
            ``used`` -- what a reader must SUPPORT to open the file at all),
            ``lightmap`` and ``generator``. ``problems`` and ``ok`` are kept
            strictly in step: an empty ``problems`` always means ``ok``. A GLB
            with no handoff is reported, not raised -- an asset from another
            producer is a legitimate thing to point this at.
        """
        return cls._sidecar.verify_glb(glb)

    @classmethod
    def data_export_channel(cls, gltf: dict, key: str) -> Optional[Any]:
        """Decoded value of one ``data_export`` channel in a parsed glTF, or ``None``.

        Every in-band metadata system publishes onto the same carrier -- the
        lightmap manifest, the shot definitions, the take list, the audio events
        -- so reading one is the same puzzle every time, and the puzzle is the
        two on-disk SHAPES, not the channel. Hand-rolled per channel, the second
        copy is the one that probes only the nested shape and turns its whole
        feature into a silent no-op on a natively exported GLB.

        Read off a parsed glTF so an applier can read from the session it is
        ALREADY holding -- a second ``open_glb`` on the path
        would re-read and re-parse the file, which is exactly what
        :class:`GlbEdit` exists to avoid.

        Returns whatever the channel decodes to (the producers publish JSON, so
        in practice a dict or a list); a caller wanting one shape checks. An
        unparsable channel is warned and read as absent -- a half-decoded
        manifest is worse than none.
        """
        return cls._sidecar.data_export_channel(gltf, key)

    @classmethod
    def overlay_data_export(cls, gltf: dict, channels: Dict[str, Any]) -> List[str]:
        """Replace ``data_export`` channels in a parsed glTF, ahead of every reader.

        The in-band channels are the passes' only input -- the gate, the fades,
        the clips all read "out of the deliverable itself" -- which is what
        makes them self-feeding, and also what left a build no way to show
        anything but the scene as exported. An overlay is that way: a caller
        states a channel for THIS build (a preview of an effect at a panel's
        settings, say), the passes read it exactly as if the FBX had carried
        it, and nothing upstream -- the scene, the exported file -- is touched.

        Each named channel is removed from every node in BOTH on-disk shapes
        :meth:`data_export_channel` reads (a file may carry several carriers,
        one per referenced module), then written once as top-level node extras.
        ``None`` clears a channel: absent, not empty, so a reader falls back
        exactly as it does for a scene that never published one.

        Parameters:
            gltf: The parsed glTF, edited in place.
            channels: ``{channel key: decoded value or None}``.

        Returns:
            The channel keys that changed, sorted; the caller marks its edit
            session dirty when this is non-empty.
        """
        return cls._sidecar.overlay_data_export(gltf, channels)

    # Read beyond the facade (its tests, sibling modules): the part's own.
    @classmethod
    def _lift_scene_blocks(cls, gltf: dict) -> bool:
        return cls._sidecar._lift_scene_blocks(gltf)

    _sidecar_section_scope = staticmethod(_Sidecar._sidecar_section_scope)

    @classmethod
    def _apply_scene_sidecar(
        cls, glb: GlbTarget, sidecar: Optional[Dict[str, Any]], *, finish: bool = True
    ) -> Dict[str, str]:
        return cls._sidecar._apply_scene_sidecar(glb, sidecar, finish=finish)

    @classmethod
    def _write_glb_handoff(cls, edit: "GlbEdit", handoff: Mapping[str, Any]) -> None:
        return cls._sidecar._write_glb_handoff(edit, handoff)

    @classmethod
    def _finish_glb_handoff(cls, edit: "GlbEdit") -> bool:
        return cls._sidecar._finish_glb_handoff(edit)

    _requirement_note = staticmethod(_Sidecar._requirement_note)

    @classmethod
    def _current_record(cls, gltf: dict, spec: RecordSpec) -> Optional[Dict[str, Any]]:
        return cls._sidecar._current_record(gltf, spec)

    # ------------------ _lightmaps.py: a host's bake, bound for the web viewer

    LIGHTMAP_INFO_KEY = _Lightmaps.LIGHTMAP_INFO_KEY
    LIGHTMAP_CLONE_SUFFIX = _Lightmaps.LIGHTMAP_CLONE_SUFFIX

    @classmethod
    def lightmaps_block(
        cls,
        materials: Dict[str, float],
        carrier: str = "occlusion",
        probe: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """The ``lightmaps`` block a GLB carries (``SceneRecords.LIGHTMAPS.glb``),
        the one statement of its shape whichever exporter writes the file.

        What the file states nowhere else: each lightmapped material's
        intensity -- the scalar restoring its map's range -- the texture slot
        the lightmaps ride (*carrier*), and the bake's embedded *probe*. A
        material's map is its bound texture and the UV set that texture's
        texCoord, which the file already states; sRGB is the one encoding a
        lightmap ships in, so the block states none.

        Parameters:
            materials: ``{material name: intensity}``.
            carrier: The texture slot a lightmap rides (``"occlusion"``: the
                one every viewer here reads; see :meth:`apply_glb_lightmaps`).
            probe: The embedded probe's entry (:meth:`embed_glb_lightmap_probe`).
        """
        return cls._lightmaps.lightmaps_block(materials, carrier, probe)

    @classmethod
    @Deprecation.symbol(
        "nothing -- no record has carried a folder hint since 0.11.0",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    def without_locate_hints(cls, data_export: Dict[str, Any]) -> Dict[str, Any]:
        """*data_export* as given: the folder hints this removed went with
        0.11.0 (the maps' folders ride the private ``lightmap_dirs``)."""
        return data_export

    @classmethod
    @Deprecation.symbol(
        "MeshConvert.data_export_channel(gltf, SceneRecords.LIGHTMAPS.key) -- a "
        "finished GLB carries no record; its lightmaps are its extras.lightmaps "
        "block (SceneRecords.glb_block)",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    def read_glb_lightmap_manifest(cls, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        """The ``lightmaps`` record riding a GLB's carrier node, or ``None`` --
        always ``None`` on a finished GLB, whose records leave it
        (:meth:`strip_fbx_records`); retired with that.

        The manifest travels **in-band**: the host DCC publishes it as a string user
        property on its ``data_export`` carrier node, Maya's FBX exporter writes it as
        an FBX user property, and FBX2glTF's ``--user-properties`` (always passed by
        :meth:`fbx_to_glb`) transcribes it into that node's glTF extras as
        ``extras.fromFBX.userProperties.<key>`` -- probe-verified against v0.13.1.
        So no consumer has to pass anything; the deliverable feeds its own repair.

        A GLB that never made the FBX hop carries the same key as a TOP-LEVEL node
        extra instead (a native glTF export's custom properties), and both shapes
        are read -- the marker walk knows both, and a probe that knew only one
        would make this whole path a silent no-op on half the deliverables.
        Returns ``None`` for a GLB carrying neither, which is a clean no-op for
        every caller.
        """
        return cls._lightmaps.read_glb_lightmap_manifest(glb)

    @classmethod
    @Deprecation.symbol(
        "MeshConvert.finish_glb -- it drops every record and marker a GLB's nodes hold",
        remove_in=_REMOVE_IN,
        since=_SINCE,
    )
    def fix_glb_lightmap_metadata(cls, glb: GlbTarget) -> int:
        """Leave a GLB its one statement of its lightmaps: ``extras.lightmaps``.

        The bake writes two more on the way out -- each baked object's
        ``lightmapInfo`` marker and the ``lightmaps`` record on the
        data_export carrier -- as the scene holds them, BEFORE the web encode
        exists: an ``.exr`` that ships nowhere, at the pre-normalisation
        intensity, beside the authoring folder. A GLB built from an FBX drops
        them once every pass has read them (:meth:`strip_fbx_records`); this is
        the same for a GLB that build never touched -- blendertk's native glTF
        export, whose ``floor`` shipped ``.exr`` @ 1.0 beside a ``.png`` @ 3.19
        (measured in Blender 5.1).

        A GLB with no ``extras.lightmaps`` is left as found: nothing is
        embedded, so the copies may be all it says of its lightmaps.

        Parameters:
            glb: ``.glb`` path (modified in place) or an open :class:`GlbEdit`.

        Returns:
            How many nodes changed; 0 when the file already states them once.
        """
        return cls._lightmaps.fix_glb_lightmap_metadata(glb)

    @classmethod
    def lightmap_manifest_coverage(cls, glb: GlbTarget) -> Dict[str, List[str]]:
        """Split a GLB's bake manifest by whether this GLB actually carries each object.

        The manifest is a **scene** record -- the bake commits to the scene, and
        every export of any subset of it carries the whole thing -- so the
        manifest's length is not the number of objects a given deliverable was
        supposed to light. Reading it as one makes a selection export report
        every unselected object as unlit, which is a false alarm raised
        precisely when the deliverable is correct.

        Returns ``{"present", "ambiguous", "absent"}`` name lists -- one item
        per ENTRY, so a name two objects share is listed once per object --
        where *present* is in scope and bindable, *ambiguous* is in scope but
        matched several nodes nothing tells apart (a real failure), and
        *absent* has no node in this GLB at all (out of scope -- or, if it was
        meant to be exported, a name mismatch this cannot tell apart from a
        scope boundary). Resolved by the binder's own rule
        (:meth:`_resolve_lightmap_entries`).
        """
        return cls._lightmaps.lightmap_manifest_coverage(glb)

    @staticmethod
    def lightmap_report(
        coverage: Dict[str, List[str]], bound: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """``{"expected", "bound", "unbound", "out_of_scope"}`` for one bind.

        *coverage* is :meth:`lightmap_manifest_coverage` read BEFORE the bind
        (a miss leaves no record, so only the manifest can say what was
        wanted), *bound* what :meth:`apply_glb_lightmaps` returned. Scoped to
        this GLB: the bake manifest is a SCENE record every export carries
        whole, and counting it against a selection reported every unselected
        object as unlit. An ambiguous leaf stays in scope -- it IS in the file
        and did not bind. ``out_of_scope`` is kept rather than dropped: "3 of 3
        lit" over a 50-object scene is only reassuring once you can see the
        other 47 were never in the export.
        """
        return _Lightmaps.lightmap_report(coverage, bound)

    @classmethod
    def embed_glb_lightmap_probe(
        cls,
        glb: GlbTarget,
        probe: Optional[Dict[str, Any]],
        search_dirs: Sequence[str],
        hdr_codec: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """Embed a bake's reflection probe in a GLB that ships its lightmaps; its entry, or ``None``.

        The native-export twin of what :meth:`apply_glb_lightmaps` does as it
        binds: a host whose own exporter wrote the ``lightmaps`` block
        (blendertk's ``LightmapWebExport``) embeds the probe AFTER the export
        and :meth:`finish_glb`, since the view it rides exists only once the
        file does. *probe* is the scene's own probe record
        (``LightmapRecords.probe``: its unit beside it); its file is found by
        name in *search_dirs*, embedded whole and published as the block's
        ``probe`` (:meth:`_embed_lightmap_probe`). A GLB with no ``lightmaps``
        block is left as found: nothing in it binds a lightmap, so nothing
        would read a probe. One whose block carries a probe already keeps it
        (idempotent).

        Parameters:
            glb: ``.glb`` path (modified in place) or an open :class:`GlbEdit`.
            probe: ``{"map", "position", "box", "unit_scale"}``; ``None``
                when the bake captured none (nothing is embedded).
            search_dirs: Folders to find the probe's file in, in priority order.
            hdr_codec: The host's HDR codec (see :meth:`apply_glb_lightmaps`);
                its ``encode_hdr_radiance`` encodes the probe.

        Returns:
            The ``lightmaps`` entry embedded, or ``None`` (logged when the
            file has a manifest but the probe could not be embedded).
        """
        return cls._lightmaps.embed_glb_lightmap_probe(
            glb, probe, search_dirs, hdr_codec
        )

    @classmethod
    def strip_glb_baked_lights(cls, glb: GlbTarget) -> List[str]:
        """Drop every ``KHR_lights_punctual`` light from a lightmapped GLB.

        A lightmap already holds every light in the scene -- the bake renders
        all of them -- so a light that also ships is counted twice, and the
        second time without a shadow: twelve bake spots reached a
        moody room's GLB through a saved FBX preset with lights on, and the
        floor under the table read 3x its baked value in WebXR (2026-10-16).
        Self-feeding like the lightmap pass: a GLB with no ``lightmaps``
        block (:meth:`SceneRecords.glb_block`) keeps its lights, which are then
        its only lighting.

        Returns:
            The names of the nodes that carried a light, in file order.
        """
        return cls._lightmaps.strip_glb_baked_lights(glb)

    @classmethod
    def apply_glb_lightmaps(
        cls,
        glb: GlbTarget,
        search_dirs: Sequence[str] = (),
        carrier: str = "occlusion",
        percentile: Optional[float] = None,
        replace_authored: bool = True,
        hdr_codec: Any = None,
    ) -> List[Dict[str, Any]]:
        """Wire a host DCC's committed lightmaps into a GLB for the web viewer.

        The GLB consumer half of the scene-state contract: the bake tool commits to
        the *scene* (per-shape markers -> the ``lightmaps`` manifest on the
        FBX ``data_export`` carrier), and this reads that manifest back **out of the
        GLB itself** (:meth:`_current_record`), encodes each referenced HDR
        EXR for the web (:meth:`ImgUtils.encode_hdr_for_web`), embeds it, and binds it
        as the material's ``occlusionTexture`` on ``TEXCOORD_1`` with the
        ``lightmaps`` root-extras manifest the viewer rebinds from. glTF has no
        lightmap slot; the occlusion carrier is the convention the viewer (and
        blendertk's native exporter) already share -- a naive viewer degrades to grey
        AO rather than to nothing.

        A GLB with no manifest is a clean no-op, which is what makes it safe to run
        unconditionally after every conversion. Name matching is exact first, then
        namespace-tolerant: a name that misses is retried with the ``NS:`` prefix
        stripped from both sides and binds only when that leaf is unambiguous among
        the GLB's nodes (manifests and exports can disagree about namespaces
        without either being wrong -- an older publisher stripped them, some
        exporters flatten them). A name several objects share -- FBX carries leaf
        names only -- is settled by each entry's ``hierarchy`` against where each
        node sits (:meth:`_resolve_lightmap_entries`). Every remaining miss is
        loud: a name nothing tells apart, a name matching no node at all, a
        primitive without ``TEXCOORD_1`` (the FBX was exported without the second
        UV set) and an EXR that cannot be found are each warned and skipped, never
        guessed at. A material worn by objects baked into DIFFERENT maps (a
        secondary material two objects share, or a Per-Object bake of instances)
        binds a copy per object after the first, like a per-instance rect below
        -- it used to be refused, and the object wore the first claimant's
        lighting. The copies are one per (material, map, rect), shared by every
        object baked into that patch of that map.

        Parameters:
            glb: ``.glb`` path (modified in place) or an open :class:`GlbEdit`.
            search_dirs: Directories to resolve the manifest's EXR basenames
                against, in priority order -- the host's own answer to where
                its maps live (a DCC's ``LightmapRecords.search_dirs``: the
                folders the bake markers name first).  Tried after the folders
                a manifest written before 0.11.0 still names, before the GLB's
                directory.
            carrier: ``"occlusion"`` (default) or ``"emissive"`` -- which material
                slot carries the map (mirror of blendertk's ``CARRIERS``).
            percentile: Divide each map by this percentile of its lit values
                instead of its capped peak (see
                :meth:`ImgUtils.encode_hdr_pixels_for_web`).
            replace_authored: Whether a map already sitting in the carrier slot
                gives way to the lightmap. ``True`` (default) displaces it with
                a warning -- a bake IS the deliverable, and its occlusion term
                supersedes a separate AO map. ``False`` keeps the authored map
                and skips the lightmap for that material, warned. Either way, a
                slot holding the material's own ``metallicRoughnessTexture`` is
                NOT authored -- that is the packed-ORM occlusion binding
                (:meth:`set_glb_metallic_roughness`, or FBX2glTF's converted
                packing), whose R channel the bake already contains -- so it is
                displaced silently regardless.
            hdr_codec: What encodes the float images: an object with
                :class:`ImgUtils`' two HDR encoders --
                ``encode_hdr_for_web(path, percentile) -> (png_bytes, scalar)``
                for each lightmap and ``encode_hdr_radiance(path) -> bytes``
                for the probe. ``None`` is :class:`ImgUtils`, whose only
                float-image codec is cv2; a host whose Python has none
                (Blender's) hands its own image writer
                (blendertk's ``LightmapWebExport``) -- without it every map
                was warned "not encoded" and the GLB shipped unlit.

        **Per-instance atlas rects travel as glTF-standard ``KHR_texture_transform``.**
        A manifest record with a non-identity ``scaleOffset`` (an instance's patch of a
        shared atlas over the mesh's shared [0,1] unwrap) cannot bind on the shared
        material -- every sibling would sample the same patch -- so its node gets its
        own MATERIAL clone carrying the rect as a texture transform, and, when the node
        shares its glTF mesh with siblings (FBX2glTF preserves instancing as one mesh
        referenced by many nodes -- probe-measured), its own MESH entry too. Both
        clones are pure JSON referencing the same accessors/bufferViews: zero geometry
        or texture duplication, and any compliant viewer (three.js, model-viewer,
        Babylon, the production WebXR app) renders the rect with no custom code.

        **The bake's reflection probe rides along.** A manifest ``probe`` (the
        HDR the host captured of the room it baked, its capture point and the
        box its reflections project onto, in scene units beside ``unit_scale``)
        is embedded whole and published as ``lightmaps.probe``
        (:meth:`_embed_lightmap_probe`), in metres -- the environment a viewer
        reflects off baked surfaces and lights unbaked ones with.

        Returns:
            One record per (object, material) binding: ``{"material", "object",
            "map", "intensity", "scaleOffset", "entry"}`` -- several objects
            sharing one atlas material each get a record, and ``entry`` is the
            manifest entry's index, which tells apart objects that share a name.
            Empty when there was no manifest or nothing matched.
        """
        return cls._lightmaps.apply_glb_lightmaps(
            glb, search_dirs, carrier, percentile, replace_authored, hdr_codec
        )

    # Read beyond the facade (its tests, sibling modules): the part's own.
    @classmethod
    def _lightmap_manifest(cls, gltf: dict) -> Optional[Dict[str, Any]]:
        return cls._lightmaps._lightmap_manifest(gltf)

    @classmethod
    def _lightmap_final_values(
        cls, gltf: dict, bound: Optional[Dict[str, Dict[str, Any]]]
    ) -> Dict[int, Dict[str, Any]]:
        return cls._lightmaps._lightmap_final_values(gltf, bound)

    @classmethod
    def _correct_record(cls, gltf: dict, final: Optional[dict]) -> bool:
        return cls._lightmaps._correct_record(gltf, final)

    # ---------------- _highlights.py: the bake's lights in copies of its probe

    HIGHLIGHT_ROUGHNESS = _Highlights.HIGHLIGHT_ROUGHNESS
    HIGHLIGHT_MAX_COPIES = _Highlights.HIGHLIGHT_MAX_COPIES
    HIGHLIGHT_SAMPLES_PER_M2 = _Highlights.HIGHLIGHT_SAMPLES_PER_M2
    HIGHLIGHT_MIN_SAMPLES = _Highlights.HIGHLIGHT_MIN_SAMPLES
    _HIGHLIGHT_MAX_SAMPLES = _Highlights._HIGHLIGHT_MAX_SAMPLES
    HIGHLIGHT_MIN_ANGLE = _Highlights.HIGHLIGHT_MIN_ANGLE
    HIGHLIGHT_SUPERSAMPLE = _Highlights.HIGHLIGHT_SUPERSAMPLE
    HIGHLIGHT_EYE_NODE = _Highlights.HIGHLIGHT_EYE_NODE
    FBX_HIGHLIGHT_NAME = _Highlights.FBX_HIGHLIGHT_NAME

    @classmethod
    def highlight_probe_layers(
        cls,
        glb,
        lights: Optional[Sequence[Dict[str, Any]]],
        probe: Optional[Dict[str, Any]] = None,
        width: Optional[int] = None,
        baked: Optional[Sequence[str]] = None,
        eye: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, Any]]:
        """The probe's highlight copies: for each receiver group, the lights it is shown.

        Each layer is what its group's copy adds to the probe as captured
        (radiance in the probe's own units, the three.js equirect layout --
        top row +Y): the bake's lights, each drawn where box projection shows
        it true for that group and scaled so the light it gives the group's
        surfaces is the light's own (see the module).

        Parameters:
            glb: ``.glb`` path or an open :class:`GlbEdit`: the geometry the
                groups are read from (model space, metres), and -- unless given
                -- its ``lightmaps`` manifest's probe and baked materials.
            lights: The bake's lights in metres (:meth:`lights_in_metres`,
                applied here too: a light it leaves out is left out).
            probe: ``{"position", "box"}`` (metres; ``box`` ``{min, max}`` or
                ``None`` for a probe read as distant); default the file's.
            width: The probe's width in texels (the interior's copy is drawn
                at it, an unbaked object's at half); default 1024.
            baked: Material names that carry a lightmap; default the file's.
            eye: Where the room is seen from; default the start camera's node
                (:attr:`HIGHLIGHT_EYE_NODE`), else the probe's position.

        Returns:
            ``[{"materials", "receivers", "image"}]`` -- ``receivers``
            ``"interior"`` or ``"unbaked"``; ``image`` an
            ``(h, w, 3)`` float32 array. Empty when there is nothing to light.
        """
        return cls._highlights.highlight_probe_layers(
            glb, lights, probe, width, baked, eye
        )

    @classmethod
    def lights_in_metres(
        cls,
        lights: Optional[Sequence[Dict[str, Any]]],
        unit_scale: Optional[float] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """The bake's lights in metres and the viewer's units, or ``None``.

        Lights arrive in scene units: the lightmap record's (``LightmapRecord``)
        at the record's *unit_scale* (metres per unit), a host's
        ``specular_lights`` (mayatk ``LightUtils``) with ``unit_scale`` beside
        each -- a light's own wins, then *unit_scale*, then metres.
        This carries metres: a point or spot light's intensity in candela (its
        irradiance falls over metres, so it scales by ``unit_scale`` squared),
        a rect's radiance as it is. One without a position, a type or a
        positive intensity is left out -- a field that is no number leaves its
        light out, never the bake that reads them.
        """
        return cls._highlights.lights_in_metres(lights, unit_scale)

    @classmethod
    def embed_fbx_highlight_probes(
        cls,
        fbx: str,
        glb: Optional[str] = None,
        *,
        search_dirs: Sequence[str] = (),
        hdr_codec: Any = None,
        logger: Optional[logging.Logger] = None,
    ) -> List[str]:
        """Give an FBX deliverable's probe the highlight copies its GLB ships.

        A GLB's probe carries copies with the bake's lights drawn in
        (:meth:`_embed_highlight_probes`). The FBX carried the probe alone, so
        a reader that shades it as the WebXR viewer does (unitytk's
        deliverable shading) lit its metals with the probe as captured, the
        bench's lights missing from them. The copies are *glb*'s -- the GLB
        this export built from *fbx* -- or, without one, those of a GLB built
        here the same way (:meth:`fbx_to_glb`, whose lightmap pass draws
        them) from *fbx* without its takes (:meth:`_fbx_static_scene`), so
        both deliverables carry the same copies to the byte. Each
        goes inside the FBX as its Radiance bytes, under
        :attr:`FBX_HIGHLIGHT_NAME`, and the record's probe lists them:
        ``highlights: [{"map", "objects": [{"name", "hierarchy"}]}]``. An
        engine binds a renderer, not a material, so each mesh node is listed
        under the one copy holding most of its triangles.

        Run on a freshly written FBX: a name the file already embeds keeps its
        bytes (:meth:`FbxMedia.embed`). A no-op on a file whose record has no
        probe or no lights. Never the deliverable's cost: a failure is warned
        and the file stays as written.

        Parameters:
            fbx: The FBX deliverable, rewritten in place.
            glb: A GLB built from *fbx*; ``None`` builds one (FBX2glTF must be
                installed: none is downloaded for this).
            search_dirs: Where the bake's maps and probe are, for that build
                (:meth:`apply_glb_lightmaps`).
            hdr_codec: That build's HDR encoder (:meth:`apply_glb_lightmaps`).
            logger: Where the outcome is said; the ``_highlights`` module's
                logger otherwise.

        Returns:
            The copies' file names, in the record's order; empty when none
            shipped.
        """
        return cls._highlights.embed_fbx_highlight_probes(
            fbx, glb, search_dirs=search_dirs, hdr_codec=hdr_codec, logger=logger
        )

    # Read beyond the facade (its tests, sibling modules): the part's own.
    _fbx_static_scene = staticmethod(_Highlights._fbx_static_scene)

    @classmethod
    def _highlight_receivers(
        cls, gltf: Dict[str, Any], materials: Sequence[set]
    ) -> List[List[Dict[str, Any]]]:
        return cls._highlights._highlight_receivers(gltf, materials)

    @classmethod
    def _highlight_eye(cls, edit) -> Optional[List[float]]:
        return cls._highlights._highlight_eye(edit)

    @classmethod
    def _highlight_meshes(cls, edit) -> List[Dict[str, Any]]:
        return cls._highlights._highlight_meshes(edit)

    @classmethod
    def _highlight_samples(
        cls, meshes: Sequence[Dict[str, Any]], seed: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        return cls._highlights._highlight_samples(meshes, seed)

    @classmethod
    def _highlight_groups(
        cls,
        meshes: Sequence[Dict[str, Any]],
        baked: set,
        lo: Optional[np.ndarray],
        hi: Optional[np.ndarray],
    ) -> List[Dict[str, Any]]:
        return cls._highlights._highlight_groups(meshes, baked, lo, hi)

    _box_exit = staticmethod(_Highlights._box_exit)

    @classmethod
    def _highlight_placement(cls, light, points, normals, eye, lo, hi, irradiance):
        return cls._highlights._highlight_placement(
            light, points, normals, eye, lo, hi, irradiance
        )

    @classmethod
    def _highlight_layer(cls, lights, focal, scales, C, lo, hi, width):
        return cls._highlights._highlight_layer(lights, focal, scales, C, lo, hi, width)

    @classmethod
    def _proxy_radius(cls, light, F, C, lo, hi, width) -> float:
        return cls._highlights._proxy_radius(light, F, C, lo, hi, width)

    # ---------------- _shadow_rigs.py: shadow-plane maps, the shadows manifest

    SHADOW_DATA_SAMPLER = _ShadowRigs.SHADOW_DATA_SAMPLER

    @classmethod
    def apply_glb_shadows(
        cls, glb: GlbTarget, *, search_dirs: Sequence[str] = ()
    ) -> Optional[Dict[str, Any]]:
        """Bind a scene's shadow-rig maps into a GLB; publish ``extras.shadows``.

        The GLB half of the shadow-rig contract (``mayatk/docs/
        shadow_rig_morphing.md``, *Contracts*). The DCC rigs publish
        ``shadows`` on the ``data_export`` carrier -- per plane its
        node, source and contact names, the projection model's constants, the
        silhouette texture and, when in use, the atlas rect and the horizon
        map. This reads that channel back out of the file (like the lightmap
        pass), resolves every name to a glTF node index, binds the loose maps
        the FBX does not carry (the horizon PNG, embedded byte for byte under
        a bilinear / no-mip / clamp sampler; a silhouette the material lost)
        and writes the viewer's manifest (``ShadowBlock``): per plane the
        record's projection model, bound by index in place of every name --
        ``node`` / ``source_node`` / ``contact_node`` (glTF node indices, or
        ``None``), a projected plane's ``texture_index`` (its colour map --
        the material's base colour, which IS the atlas once packed, its tile
        the quad's own UVs) and a horizon plane's map with its
        ``texture_index``. The horizon rect is glTF top-left
        (:meth:`ImgUtils.flip_rect_v` of the record's bottom-left one);
        lengths stay DCC units with ``unit_scale`` beside them, as the record
        has them; every plane states every field its shape declares
        (``pythontk.geo_utils.shadow_record``) and nothing more. The
        packaged viewer script
        (``preview/features/shadow_rig/``) reads the manifest on load,
        evaluates ``ShadowProjection.model`` per frame and drives the planes.

        Names resolve exactly first, then by namespace-stripped leaf against
        EVERY node carrying it: a plane name matching several nodes gets one
        manifest entry per node (a rig can sit under two namespaces in one
        file), while an ambiguous source or contact leaves the plane
        unfollowed rather than following a guess. A plane with no node in the
        file is out of scope (a selection export) and counted once; one whose
        horizon map cannot be found in *search_dirs* -- or a projected plane
        with no colour map at all -- is skipped with a warning and keeps its
        DCC material, which is the silhouette fallback.

        Idempotent: the manifest is replaced wholesale on every run, an
        embedded map is found again by content and reused, and the channel
        itself is left in place. A run that binds nothing removes any stale
        manifest.

        Parameters:
            glb: ``.glb`` path (modified in place) or an open :class:`GlbEdit`.
            search_dirs: Directories to resolve the records' texture basenames
                against, in order; the GLB's own directory is tried last.

        Returns:
            The published manifest, or ``None`` when the file carries no
            channel or nothing in it could be bound.
        """
        return cls._shadow_rigs.apply_glb_shadows(glb, search_dirs=search_dirs)

    # ----------------------------- _articulation.py: the articulation manifest

    @classmethod
    def apply_glb_articulation(cls, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        """Publish ``extras.articulation``: the scene's articulated rigs,
        bound to this file's nodes.

        The GLB half of the articulated-rig contract. The DCC's
        ``ArticulatedRig`` publishes the ``articulation`` record on the
        ``data_export`` carrier -- per rig its joints (node name, parent,
        rest, rotate order, channels and limits) and the parts a hand grabs.
        This reads it back out of the file and resolves every name to a glTF
        node index: a joint to the one node carrying its name (exactly, else
        by namespace-stripped leaf); a grabbed part to the node of that name
        UNDER its joint's node -- a production assembly repeats part names,
        and the joint says which one is meant. The manifest is the record
        with a ``node`` index on every joint and grab entry; the packaged
        viewer script (``preview/features/articulated_rig/``) builds its
        model from it and poses the joints.

        A rig whose joints are not all in the file (a selection export that
        left it out) is out of scope and skipped, counted once; an ambiguous
        joint name skips its rig with a warning -- a runtime must never pose a
        guess. A grab entry that resolves nowhere is dropped with a warning
        (the rig still plays its clips; that part is just not grabbable).

        Idempotent: the manifest is replaced wholesale on every run, and a run
        that binds nothing removes a stale one. The channel is left in place.

        Parameters:
            glb: ``.glb`` path (modified in place) or an open :class:`GlbEdit`.

        Returns:
            The published manifest, or ``None`` when the file carries no
            channel or no rig in it could be bound.
        """
        return cls._articulation.apply_glb_articulation(glb)

    # ----------------------- _animation.py: shot clips, the animation manifest

    SHOT_CLIP_MODE_KEY = _Animation.SHOT_CLIP_MODE_KEY
    ANIMATION_CLIP_MODES = _Animation.ANIMATION_CLIP_MODES
    DEFAULT_CLIP_SPAN = _Animation.DEFAULT_CLIP_SPAN
    CLIP_SPAN_DRIFT_FRAMES = _Animation.CLIP_SPAN_DRIFT_FRAMES
    DEFAULT_CLIP_NAME = _Animation.DEFAULT_CLIP_NAME

    @classmethod
    def apply_glb_clips(
        cls, glb: GlbTarget, *, mode: str = "both"
    ) -> Optional[Dict[str, Any]]:
        """Rebuild the declared shot clips as exact slices of the whole timeline.

        Maya's take split is lossy in a way that does not announce itself: it
        restricts each curve to the take's window and bakes what is left, so a
        curve with no key inside a shot contributes NO channel to it and the
        node plays its rest pose for the shot's whole duration. Measured on a
        12-shot production assembly (358 keys over 2635 frames), Shot_1 through
        Shot_11 were wrong on EVERY frame -- by up to 3.73 m -- while the
        whole-timeline stack the same export retains was right on all 2629.

        So the shots are cut here, from that stack, on the deliverable: no scene
        to reach into, nothing mutated, and the exporter and the preview push
        get identical clips because they run the same pass.
        :mod:`~pythontk.file_utils.mesh_convert.glb.clips` does the cutting;
        this half reads the file (which takes, which rate, which origin).

        Runs FIRST, before :meth:`apply_glb_visibility` writes the presence
        gates -- the gates belong to the rebuilt clips, and a gate written into
        a clip this pass then replaces would be discarded with it.

        Needs to know the authoring frame the source stack places at its own
        ``t=0``, because the converter rebases every stack onto its first key.
        Three ways, in order: the stack's own ``extras`` (there after this pass
        has run once, which is what makes a second run exact rather than
        merely harmless); the file's ``clip_span`` for the whole timeline --
        measured from the FBX by the conversion (:meth:`_stamp_clip_spans`),
        else as the producer published it; and otherwise nothing -- a guessed
        origin would slide every
        shot by the same wrong amount, which is worse than the split this
        replaces, so the clips are left as exported.

        Parameters:
            glb: Path to a ``.glb``, modified in place, or an open session.
            mode: Which clips the deliverable ships, from
                :attr:`ANIMATION_CLIP_MODES`. ``both`` cuts the shots and keeps
                the stack they came from. ``shots`` cuts them and drops it,
                releasing its payload. ``full`` cuts nothing and ships only the
                stack -- still renamed and stamped, so the file describes its
                own origin either way. The two halves hold the same
                performance, so a consumer that plays one never reads the
                other; which one is dead weight is the caller's to say.

        Returns:
            ``{"clips", "channels", "source", "bytes"}``, or ``None`` when
            there was nothing to rebuild.
        """
        return cls._animation.apply_glb_clips(glb, mode=mode)

    @classmethod
    def clip_spans(
        cls,
        frames: Iterable[float],
        takes: Iterable[Any],
        stack_range: Optional[Sequence[float]] = None,
        key_spans: Optional[
            Callable[
                [List[Tuple[Optional[float], Optional[float]]]],
                Sequence[Optional[Sequence[float]]],
            ]
        ] = None,
    ) -> Dict[str, List[float]]:
        """Per take, the first and last authored frame inside its window.

        The producer half of what :meth:`_clip_zero` reads back, kept here
        rather than in each DCC package so the two cannot write the schema
        differently -- the same reason :meth:`build_scene_sidecar` owns the
        sidecar envelope. Each toolkit keeps only the part that needs a scene:
        collecting *frames* (every animated channel's key times, transforms and
        visibility alike, because the converter sizes a take from all of them).

        The :attr:`DEFAULT_CLIP_SPAN` entry is the whole timeline, for the
        full-range stack a take split does not name.

        Parameters:
            frames: Every authored key time in the scene, in frames.
            takes: The declared takes, each ``{"name", "start", "end"}``.
            stack_range: ``(first, last)`` frame the SOURCE STACK will
                actually carry -- the exported/baked range. Pass it whenever
                the caller knows it. The converter rebases every stack onto
                its first key, so the whole-timeline zero must be the first
                frame the FILE holds, not the first frame the SCENE holds: a
                key authored before the exported range never reaches the FBX,
                and letting it set the zero slides every clip cut from that
                stack by the difference. Measured on a production assembly as
                a 33-frame slide (first take at 33, scene's first key at 0),
                which reads as up to 90 cm of apparent distortion while the
                geometry is exact. Also why a bake matters: it writes a key on
                every frame of the range, so the stack's first key IS the
                range start even when the pre-bake scene had none there.

            key_spans: The same question asked per window instead of by listing
                keys -- *frames* is then ignored. Called once with
                ``[(start, end), ...]`` (inclusive frames, ``None`` =
                unbounded: the whole timeline first, then one window per valid
                take) and returns each window's ``(first, last)`` key, or
                ``None`` where no key falls. A DCC that can seek its curves
                passes this: a production bake is millions of keys, and
                listing them cost 10 s a pass.

        Returns:
            ``{take name: [first, last]}``, empty when nothing is animated.
        """
        return cls._animation.clip_spans(frames, takes, stack_range, key_spans)

    @classmethod
    def prune_glb_animations(cls, glb: GlbTarget) -> List[str]:
        """Drop every animation that carries no channels or no samplers.

        glTF requires both arrays on an animation to hold at least one entry,
        so a hollow animation is not a quiet clip -- it is a file a strict
        reader rejects outright. Two writers produce one (both measured on
        Maya 2025 -> FBX2glTF 0.13.1):

        * Maya writes its whole-timeline ``Take 001`` AnimStack whenever
          animation export is armed, keyed scene or not. On a static scene
          that is a stack and a layer with no curves, and the converter
          transcribes it as an animation with neither channels nor samplers
          (two production modules, 2026-09-02).
        * A take split emits an AnimStack per declared range and bakes no
          curve for a range in which nothing exported moves.

        Runs after every pass that can ADD channels (clips, visibility, fades)
        and before the manifest, which reports what the file carries. A
        DECLARED take that is still hollow here is dropped like any other --
        there is no valid way to ship an empty clip -- and the manifest then
        reports it as a take the file lacks.

        Parameters:
            glb: Path to a GLB, or an open :class:`GlbEdit`.

        Returns:
            The names of the animations dropped, in file order; empty when
            the file was left alone.
        """
        return cls._animation.prune_glb_animations(glb)

    @classmethod
    def compact_glb_animations(cls, glb: GlbTarget) -> Dict[str, int]:
        """Collapse every animation channel that never moves to two keys.

        A baked export writes a key on every frame for every node in every
        clip, because a clip must PIN the pose of everything it does not
        animate: a viewer playing clip B after clip A leaves any node B omits
        wherever A left it. That pinning is the reason the channels exist, and
        it needs exactly two keys -- one at each end of the clip -- not one per
        frame. Measured on the PROPS assembly: 13,187 of 22,654 channels (58%)
        carry a value that never changes, costing 16.6 MB of a 215 MB
        deliverable to say a node stood still.

        Byte-exact, so the file plays identically: a channel is collapsed only
        when every one of its elements has the same bit pattern, and the two
        keys it keeps span the clip's own first and last time, so no clip
        changes duration. Channels that move are not touched, and neither is a
        sampler whose output is shared with one that moves.

        Wired into :meth:`fbx_to_glb`'s session after every pass that ADDS
        channels (clips, visibility, fades) and after
        :meth:`prune_glb_animations` -- it can only shrink what is already
        there. The time inputs the collapsed channels stop reading are deleted
        (:meth:`_drop_orphaned_accessors`), and the orphaned payload is
        reclaimed by :meth:`_compact_bin`.
        Also safe to run standalone on any finished GLB.

        Parameters:
            glb: Path to a ``.glb``, modified in place, or an open session.

        Returns:
            ``{"channels": n, "accessors": n, "bytes": n}`` -- channels
            collapsed, accessors rewritten, and BIN payload reclaimed. A file
            with nothing to collapse is not rewritten.
        """
        return cls._animation.compact_glb_animations(glb)

    @classmethod
    def reduce_glb_animations(
        cls,
        glb: GlbTarget,
        tolerance: float,
        rotation_tolerance: Optional[float] = None,
    ) -> Dict[str, int]:
        """Drop the keys a GLB's samplers do not need to reproduce their motion.

        :mod:`~pythontk.file_utils.mesh_convert.glb.key_reduction` does the
        work -- why, and what "need" means, are its docstring; this is the
        session-aware entry the pipeline and a caller with an open edit use.
        Runs after the clips are rebuilt and the constant channels collapsed:
        it can only shrink what is already there.

        Parameters:
            glb: Path to a ``.glb``, modified in place, or an open session.
            tolerance: The deviation any original sample may show under the
                viewer's interpolation, in the sampler's own units (meters for
                translation / scale, quaternion components for rotation; 1e-4
                is 0.1 mm / 0.006 degrees).
            rotation_tolerance: A separate bound for rotations; None takes
                *tolerance*.

        Returns:
            ``{"samplers": n, "keys_before": n, "keys_after": n, "bytes": n}``.
        """
        return cls._animation.reduce_glb_animations(glb, tolerance, rotation_tolerance)

    @classmethod
    def apply_glb_animations(cls, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        """Publish ``extras.animation``: the FBX's shots, inherited whole, and
        the GLB's clips.

        The GLB consumer half of the shots contract. The FBX's ``animation``
        record is the statement of the shots, and this block carries it into
        the GLB as it stands -- every shot, whatever clips the conversion
        produced -- adding only what the GLB alone can say (all measured on
        Maya 2025 -> FBX2glTF 0.13.1):

        * **Which clip plays a shot.** Maya's exporter keeps its whole-timeline
          ``Take 001`` AnimStack alongside the takes it was asked to split out,
          and it converts to ``animations[0]`` -- so the naive
          ``clipAction(animations[0])`` plays the entire timeline rather than
          the first shot. A shot plays the clip its ``clip`` names; a shot the
          file holds no clip for (a ``full`` file cuts none, a take the
          rebuild could not cut) plays its window of the whole-timeline clip,
          the one no shot names. ``default_clip`` names the one to open on.
        * **Where a clip sits on the timeline.** The converter rebases every
          clip to t=0, which is right for playback and loses the sequence: a
          shot authored at frames 20-30 and one at 1-10 both start at zero.
          ``zero_frame`` is the authored frame a clip's t=0 sits on, which is
          what maps clip time back to the frame numbers the shots are quoted
          in. It is measured, not assumed to be the shot's ``start``: a take
          the converter rebased itself starts at its first authored KEY, so a
          clip whose motion starts late carries a lead-in (measured at 43
          frames on a production assembly).
        * **What the frames MEAN.** Frame numbers need the rate they were
          authored at; ``fps`` carries it.

        Inherited rather than derived from the clips (block v2, 2026-10-21):
        v1 listed one entry per glTF animation, so a shot without a clip of
        its own was silently gone from the GLB while its FBX still named it.

        Reads its inputs out of the deliverable itself (the shots record,
        ``animation``, on the ``data_export`` carrier) rather than
        from the host, so it is self-feeding: it runs unconditionally after
        every conversion, and is a clean no-op on a GLB with neither shots nor
        animation. A GLB with animation but no declared shots still gets the
        block -- names, spans and rate are what a player needs whether or not
        shots exist -- and one with shots but no animation states the shots.

        Also the one place the double encoding is undone. The channels arrive
        as JSON *strings* nested under ``extras.fromFBX.userProperties`` (a
        consumer has to parse the JSON it just parsed); this block is the
        decoded reading, derived from them in the same pass, and they leave the
        file at its end with every other record (:meth:`strip_fbx_records`), so
        it states each thing once.

        Parameters:
            glb: ``.glb`` path (modified in place) or an open :class:`GlbEdit`.

        Returns:
            The published block, or ``None`` when the file has neither shots
            nor animation.
        """
        return cls._animation.apply_glb_animations(glb)

    # Read beyond the facade (its tests, sibling modules): the part's own.
    @classmethod
    def _published_origin(
        cls, read: Callable[[str], Any]
    ) -> Optional[Tuple[float, float]]:
        return cls._animation._published_origin(read)

    @classmethod
    def _stamp_clip_spans(
        cls, gltf: Dict[str, Any], measured: Dict[str, Tuple[float, float]]
    ) -> Dict[str, List[float]]:
        return cls._animation._stamp_clip_spans(gltf, measured)

    @classmethod
    def _take_windows(cls, gltf: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        return cls._animation._take_windows(gltf)

    @classmethod
    def _animation_time_span(cls, edit: "GlbEdit", animation: Dict[str, Any]) -> tuple:
        return cls._animation._animation_time_span(edit, animation)

    # ----------------------- _visibility.py: keyed visibility, fades, previews

    VISIBILITY_TRACK_DIGITS = _Visibility.VISIBILITY_TRACK_DIGITS

    @classmethod
    def effect_preview_channels(
        cls,
        nodes: Sequence[str],
        channel: str,
        keys: Sequence[Sequence[float]],
        colors: Optional[Sequence[Optional[Sequence[float]]]] = None,
        fps: float = 30.0,
    ) -> Dict[str, Any]:
        """The :meth:`overlay_data_export` channels that preview ONE render effect.

        What a panel pushes to show an effect at its current settings on the
        objects it names, whatever they carry already: every node gets the same
        *keys* on *channel*, the scene's own tracks are replaced rather than
        merged (an object's existing fade would otherwise play over the
        highlight being judged), and the take list and shot record are cleared
        so the ramp rides one clip over its own extent instead of being cut to
        shots it has nothing to do with. The passes then build exactly what the
        writers would have produced had the keys been authored.

        Parameters:
            nodes: The GLB node names (the DCC's leaf names), deduplicated.
            channel: A :data:`glb.fades.CHANNELS` key (``"opacity"``, ``"highlight"``).
            keys: ``[(frame, value), ...]`` -- a :class:`RampKeys` plan.
            colors: One ``(r, g, b)`` or ``None`` per colour stop of the
                channel, high first; ``None`` leaves that stop to its default.
            fps: The rate *keys* are quoted in.

        Returns:
            ``{render_effects: envelope, animation: None}`` -- the record,
            and no shots to cut.

        Raises:
            KeyError: For a channel with no pointer row.
            ValueError: When there are no nodes, or fewer than two keys.
        """
        return cls._visibility.effect_preview_channels(
            nodes, channel, keys, colors, fps
        )

    @classmethod
    def apply_glb_visibility(cls, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        """Realize keyed visibility as STEP ``scale`` channels the file can play.

        glTF animates four things -- translation, rotation, scale and morph
        weights -- and visibility is none of them. So a DCC's keyed visibility
        survives the FBX (which has a ``Visibility`` property) and dies at the
        glTF hop, silently: the objects are all present, all visible, all the
        time. Measured on a production assembly, that one gap produced three
        separate-looking complaints -- shots that arrived EMPTY (their only
        content was visibility), objects that never left after their shot, and
        "broken" animation in the shots that mixed both.

        The repair writes the one presence channel every glTF viewer already
        plays. Each gated node gets a ``scale`` channel with ``STEP``
        interpolation, driving the node between its authored scale and zero; a
        zero-scale node collapses to a point and rasterizes nothing, which is
        the established glTF idiom for boolean visibility precisely because it
        needs no extension. Nothing here is optional for the consumer, so the
        deliverable a developer loads in their own viewer behaves like the
        preview without being told anything.

        Smooth *fades* are the other half, and they belong to
        :meth:`apply_glb_fades` -- alpha is a material property, so it needs
        ``KHR_animation_pointer`` rather than a node channel. The two have to
        agree about WHEN, and this pass is where that is decided: a node
        carrying a real ramp is gated on the ramp instead of on the DCC's
        mirrored boolean, because the mirror hides it for exactly the frames it
        should be fading over (see :meth:`_presence_keys`). A node without a
        ramp steps exactly where the DCC's own playback steps.

        Runs BEFORE :meth:`apply_glb_animations`, which reports on what the
        clips hold: a shot whose content is entirely visibility is empty until
        this pass has run, and would otherwise be reported empty and skipped
        as the file's default clip.

        Reads its inputs out of the deliverable itself -- the
        ``render_effects`` record on the ``data_export`` carrier, joined to
        the shots record's takes for the clip windows -- so it is
        self-feeding and needs no argument from a caller that may not know the
        scene.

        Parameters:
            glb: ``.glb`` path (modified in place) or an open :class:`GlbEdit`.

        Returns:
            ``{"nodes": n, "channels": n, "clips": n}`` describing what was
            written, or ``None`` when the file carries no tracks to apply.
        """
        return cls._visibility.apply_glb_visibility(glb)

    @classmethod
    def build_visibility_tracks(
        cls,
        tracks: Sequence[Dict[str, Any]],
        fps: Optional[float] = None,
        clip_spans: Optional[Dict[str, List[float]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Wrap *tracks* in the versioned ``render_effects`` envelope.

        The one place that schema exists. Both DCC packages publish through it
        rather than shaping the dict themselves, so the channel
        :meth:`apply_glb_visibility` reads cannot fork between the toolkit that
        wrote it and the one that did not -- and the two cannot import each
        other to share it.

        Every float is rounded to :attr:`VISIBILITY_TRACK_DIGITS` places: the
        values arrive as float32 attributes read back as doubles
        (``0.49952034551044694``), and the extra digits are noise, not data
        -- a millionth of a frame, or of a colour channel, changes nothing a
        consumer can show.

        Returns ``None`` when there is nothing to publish, which is the
        producers' signal to CLEAR the channel rather than stamp an empty one.
        *clip_spans* with no *tracks* is something: the whole-timeline origin
        a shot scene needs whether or not it keys visibility, since
        :meth:`apply_glb_clips` cuts no shot without it.
        """
        return cls._visibility.build_visibility_tracks(tracks, fps, clip_spans)

    @classmethod
    def apply_glb_fades(cls, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        """Realize authored opacity ramps as animated material alpha.

        The other half of :meth:`apply_glb_visibility`. That pass answers
        "is it there", which glTF can express as a scale channel every viewer
        already plays; this one answers "how solid is it", which glTF cannot
        express at all without ``KHR_animation_pointer`` -- so the ramp is
        written through that extension, declared in ``extensionsUsed``, and the
        file fades by itself anywhere the extension is implemented.

        Runs AFTER the gate, and the order is load-bearing in both directions:
        a node has to be PRESENT during its ramp for the ramp to be visible
        (see :meth:`_presence_keys`), and the clip has to end up with a channel
        so it is not reported as an empty shot -- for a shot whose only content
        was a fade, this pass is that channel.

        Reads its inputs out of the deliverable itself, like its neighbours.
        :mod:`~pythontk.file_utils.mesh_convert.glb.fades` does the writing.

        Returns:
            ``{"nodes", "materials", "channels"}``, or ``None`` when the file
            carries no authored ramp.
        """
        return cls._visibility.apply_glb_fades(glb)

    # Read beyond the facade (its tests, sibling modules): the part's own.
    @classmethod
    def _presence_keys(cls, track: Dict[str, Any]) -> List[Sequence[float]]:
        return cls._visibility._presence_keys(track)

    _is_fade = staticmethod(_Visibility._is_fade)

    @classmethod
    def _visibility_runs(
        cls, keys: Sequence[Sequence[float]], window: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        return cls._visibility._visibility_runs(keys, window)

    # ----------------------- _materials.py: checks and the per-channel writers

    INTERMEDIATE_PNG_LEVEL = _Materials.INTERMEDIATE_PNG_LEVEL
    GLTF_ORM_CHANNELS = _Materials.GLTF_ORM_CHANNELS
    _ORM_HARMFUL_CHANNEL = _Materials._ORM_HARMFUL_CHANNEL
    ORM_FINDING_METALLIC_FULL = _Materials.ORM_FINDING_METALLIC_FULL
    ORM_FINDING_UNVALIDATED = _Materials.ORM_FINDING_UNVALIDATED
    UNLIT_EXTENSION = _Materials.UNLIT_EXTENSION

    @classmethod
    def check_glb_materials(cls, glb: GlbTarget) -> List[Dict[str, str]]:
        """Inspect a GLB for materials flagged transparent that should be opaque.

        Catches the Maya/Stingray/OpenPBR/Standard-Surface failure mode where
        a color texture happens to carry an alpha channel (often PNG palette
        transparency) without any actual transparency intent. Maya's FBX
        exporter writes a TransparencyFactor; FBX2glTF then sets
        ``alphaMode: BLEND`` and the renderer disables depth-write —
        producing the "inverted face" / wrong-render-order artifact.

        A material is flagged when its ``alphaMode`` is BLEND or MASK *and*
        its base-color texture's alpha channel is uniformly 255. Genuine
        transparency (varying alpha) is not reported.

        Parameters:
            glb: Path to a binary glTF (.glb) file, or an open
                :class:`GlbEdit` session to inspect. Read-only either way.

        Returns:
            List of findings. Each finding is a dict with keys:
                material   — material name (or '<material[i]>')
                alpha_mode — "BLEND" or "MASK"
                image      — image name / uri / fallback id
                reason     — short human-readable explanation
        """
        return cls._materials.check_glb_materials(glb)

    @classmethod
    def fix_glb_phantom_opaque_alpha(cls, glb: GlbTarget) -> List[Dict]:
        """Repair the Maya phong → FBX → FBX2glTF transparency translation bug.

        When a Maya phong/lambert/blinn shader has its ``.transparency`` fed
        by a file node's ``.outTransparency``, Maya's FBX exporter writes
        ``TransparencyFactor=1.0`` (the texture is meant to modulate
        per-pixel). FBX2glTF then computes
        ``baseColorFactor[3] = 1 - 1 = 0`` — multiplying every fragment's
        alpha by zero and rendering the mesh fully invisible regardless of
        texture content.

        A material is fixed when ALL of:
            - ``alphaMode`` is BLEND or MASK
            - ``baseColorFactor[3]`` is ~0
            - ``baseColorTexture`` exists and references an image with
              *varying* alpha (a real cutout mask, not uniformly 0 or 255)

        On match, ``baseColorFactor[3]`` is reset to 1.0 so per-pixel alpha
        from the texture controls visibility as intended.

        Parameters:
            glb: Path to a binary glTF (.glb), modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.

        Returns:
            List of fix records. Empty when nothing matched. Each record:
                material   — material name
                old_alpha  — original baseColorFactor[3]
                new_alpha  — 1.0
                image      — the baseColorTexture image identifier
        """
        return cls._materials.fix_glb_phantom_opaque_alpha(glb)

    @classmethod
    def set_glb_metallic_roughness(
        cls, glb: GlbTarget, metallic_roughness: Dict[str, Dict[str, Any]]
    ) -> List[Dict]:
        """Pack and write the ORM (metallic/roughness) texture into a GLB, by name.

        The third sibling of :meth:`set_glb_base_color` / :meth:`set_glb_emissive`,
        for the most *destructive* form of the same FBX translation gap. Measured on
        a production room (Maya 2025 StingrayPBS -> FBX2glTF): the converter packed
        the material's roughness+metallic into a **solid-white** ORM -- and since
        glTF reads metallic from the blue channel, that renders the whole room
        metallic=1. A fully metallic surface has no diffuse response, and a baked
        lightmap contributes *only* to diffuse -- so in a lightmap-lit viewer
        (which turns its own lights off) the failure compounds to pure black, the
        single symptom least traceable back to "your roughness texture was lost in
        translation".

        Packing follows the glTF convention (R=occlusion, G=roughness, B=metallic)
        through :meth:`MapFactory.pack_orm_texture` -- the registry's one ORM
        packer -- with R filled white when no AO source resolves, so the
        occlusion binding below stays neutral. The packed image embeds through
        the same session cache as every other writer, so two materials naming the
        same source maps share one embed.

        The packed image is also bound as the material's ``occlusionTexture``
        (glTF's packed-ORM idiom -- occlusion is read from that slot alone, so
        an unbound R channel is dead payload) whenever the slot is free or
        still points at the converted ORM this write replaces; an authored
        separate AO map is never displaced. A lightmap applied afterwards
        recognises the ORM binding by its shared texture index and takes the
        slot silently (:meth:`apply_glb_lightmaps`).

        Parameters:
            glb: Path to a binary glTF (.glb), modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.
            metallic_roughness: ``{material_name: {"metallic": path,
                "roughness": path, "occlusion": path}}``. All keys optional --
                a missing metallic fills black (non-metal), missing roughness
                fills black, missing occlusion fills white -- but an entry with
                no readable map at all writes nothing.

        Returns:
            List of records: ``material``, ``metallic``, ``roughness``.
        """
        return cls._materials.set_glb_metallic_roughness(glb, metallic_roughness)

    @classmethod
    def suspect_orm_materials(
        cls, glb: GlbTarget, *, described: Optional[Iterable[str]] = None
    ) -> Dict[str, Dict[str, str]]:
        """Materials whose delivered ORM binding this pipeline never validated.

        Two findings, one walk, because both are the same question asked of the
        same slot -- *is what a consumer will read here what the scene meant?*

        ``metallic=1 everywhere``
            The measured production failure. FBX2glTF white-fills a grayscale
            ("L"-mode) PBR source; glTF reads metallic from **blue**; a
            solid-white packing therefore renders metallic=1, which has no
            diffuse response, and a baked lightmap contributes to diffuse
            alone -- so a lightmapped viewer renders it pure black.

        ``unvalidated``
            The material carries an ORM binding that the envelope never
            described, so nothing in this pipeline checked its channel
            semantics. This is the case a whiteness test cannot see: a mask map
            packed for another engine (Unity's MaskMap is R=Metallic,
            G=Occlusion, B=Detail) that reaches the GLB unrepaired is read as
            ORM and misinterpreted channel for channel, while looking like
            perfectly ordinary image data. Reported only when the caller says
            what WAS described, since only they know.

        Parameters:
            glb: Path to a binary glTF (.glb) or an open :class:`GlbEdit`. Read
                only -- nothing here marks the session dirty.
            described: Material names the envelope's ``metallic_roughness``
                section covers, i.e. the ones a repair pass validated and
                rewrote. Their packing comes from the authoring maps rather
                than from the converter, so they are exempt from both findings
                (any whiteness in them is a source question
                :meth:`MapFactory.pack_orm_texture` already logs per map).
                A described material covers its LIGHTMAP CLONES too (see
                :attr:`LIGHTMAP_CLONE_SUFFIX`): the envelope was written before
                they existed, so matching literally reports the whole set. Safe
                by construction rather than by assumption -- a clone is a deep
                copy of its base that rebinds only the lightmap slot, and the
                sidecar repair runs BEFORE the lightmap pass, so the binding
                being exempted here is the repaired one the base was checked on.
                ``None`` means "nothing is known to be described", which
                suppresses the ``unvalidated`` finding rather than reporting
                every material.

        Returns:
            ``{material name: {"image": label, "finding": str}}``; empty when
            none, which is the common case and the one that costs no decode.
        """
        return cls._materials.suspect_orm_materials(glb, described=described)

    @classmethod
    def set_glb_emissive(
        cls, glb: GlbTarget, emissive: Dict[str, Dict[str, Any]]
    ) -> List[Dict]:
        """Write emissive color / texture into a GLB's materials, by name.

        Repairs the channel Maya's FBX exporter simply drops. It maps emissive
        only for its own legacy shading models (lambert / blinn / phong via
        ``incandescence``); ``aiStandardSurface``, ``StingrayPBS`` and openPBR
        emission never reach the FBX at all, so the GLB has no ``emissiveFactor``
        to correct and the surface previews as unlit. Measured against Maya
        2025 + MtoA: lambert and blinn arrive with ``emissiveFactor``, the other
        two arrive with the key absent entirely.

        Emissive intensity above 1.0 is preserved rather than clipped, by
        normalizing the color and carrying the magnitude in
        ``KHR_materials_emissive_strength`` — glTF's base ``emissiveFactor`` is
        LDR, so a Maya emission of 5.0 would otherwise flatten to 1.0 and lose
        exactly the over-bright look that motivates using it. The extension is
        declared in ``extensionsUsed`` only when actually applied; loaders that
        don't implement it still get a sensible clamped color.

        Textures are embedded as ``data:`` URIs *for the duration of the
        session*, which keeps every edit inside the JSON chunk — no buffer
        offsets to recompute, which is the part of GLB surgery that silently
        corrupts a file. :meth:`_relocate_embedded_images` then appends them to
        the BIN once on close, so the file that lands pays none of base64's
        ~33% premium. Repeated paths are embedded once per session, so a map
        used as both base colour and emissive costs one copy, not one per
        channel.

        Parameters:
            glb: Path to a binary glTF (.glb), modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.
            emissive: ``{material_name: {"color": (r, g, b), "texture": path}}``.
                Both keys optional; a texture with no color implies white.
                Names not present in the GLB are reported, not raised.

        Returns:
            List of records: ``material``, ``factor``, ``strength``, ``texture``.
        """
        return cls._materials.set_glb_emissive(glb, emissive)

    @classmethod
    def set_glb_alpha_mode(
        cls, glb: GlbTarget, alpha_mode: Dict[str, Dict[str, Any]]
    ) -> List[Dict]:
        """Write ``alphaMode`` / ``alphaCutoff`` into a GLB's materials, by name.

        The sibling of :meth:`set_glb_base_color` for the one material fact
        FBX2glTF has to guess: it derives ``alphaMode`` from the base colour's
        alpha channel alone, so an RGBA base colour is always ``BLEND`` -- a
        cutout material (``MASK``: alpha test, depth writes, the opaque queue)
        cannot be expressed by the FBX at all, and a solid body wearing a
        blended material sorts its own back faces through its front in every
        WebXR viewer. The DCC knows which it authored (Maya's
        ``SceneState._read_alpha_mode`` reads the StingrayPBS graph) and says
        so through this section.

        Parameters:
            glb: Path to a binary glTF (.glb), modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.
            alpha_mode: ``{material_name: {"mode": "OPAQUE"|"MASK"|"BLEND",
                "cutoff": float}}``. ``cutoff`` is written only for ``MASK``
                (absent = glTF's default 0.5); any other mode drops a stale
                one. Names absent from the GLB are reported, an unknown mode
                is skipped with a warning rather than written.

        Returns:
            List of records: ``material``, ``alphaMode``, ``alphaCutoff``.
        """
        return cls._materials.set_glb_alpha_mode(glb, alpha_mode)

    @classmethod
    def set_glb_unlit(
        cls, glb: GlbTarget, unlit: Dict[str, Dict[str, Any]]
    ) -> List[Dict]:
        """Mark a GLB's materials unlit (``KHR_materials_unlit``), by name.

        The one shading fact no FBX can carry: FBX2glTF makes every material it
        meets lit PBR, whatever the DCC authored -- a Maya ``surfaceShader``, or
        the black silhouette of a projected contact shadow. Lit, such a
        surface takes the viewer's environment specular, and a shadow plane
        measured BRIGHTER than the floor it lies on (2026-10-15, WebXR: 24.4
        -> 28.3 mean luminance, off -> on). Unlit, its base colour and alpha
        are what shows. The DCC says which materials it authored unlit
        (mayatk ``SceneState._read_unlit``) through this section.

        Metallic 0 / roughness 1 ride along for a viewer without the
        extension, which then falls back to lit PBR at its most diffuse.

        Parameters:
            glb: Path to a binary glTF (.glb), modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.
            unlit: ``{material_name: {}}`` -- the entries carry nothing yet;
                a dict so the section reads like its siblings. Names absent
                from the GLB are reported.

        Returns:
            List of records: ``material``.
        """
        return cls._materials.set_glb_unlit(glb, unlit)

    @classmethod
    def set_glb_normal_scale(
        cls, glb: GlbTarget, scale: float, lightmapped_only: bool = True
    ) -> int:
        """Write ``normalTexture.scale`` into a GLB's materials.

        How strongly a normal map reads is a DELIVERY decision, not an
        authoring one, and it is felt hardest on baked geometry: a lightmap
        contributes irradiance with no direction in it, so on a lightmapped
        surface the normal map survives only through the environment term --
        which the viewer deliberately dims, because that surface's lighting is
        already in its bake. The result is a correct but flat-looking room, and
        the dial that brings the detail back without touching the bake is this
        one.

        Written as ``normalTexture.scale`` because glTF already has the field:
        it is core (no extension), every runtime honours it, and three.js reads
        it straight into ``material.normalScale``. So the preview's slider and
        the delivered file say the same thing in the same place, and a value
        saved here comes back on the next load with nothing to re-apply it.

        Parameters:
            glb: Path to a binary glTF (.glb), modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.
            scale: The multiplier. ``1.0`` is glTF's default and REMOVES the
                key rather than writing it, so a reset leaves the file as it
                would have been had the dial never moved.
            lightmapped_only: Restrict to the materials
                ``extras.lightmaps`` names -- the ones the flattening
                applies to. ``False`` writes every material carrying a normal
                map. A file with no manifest matches nothing under ``True``.

        Returns:
            How many materials were changed.
        """
        return cls._materials.set_glb_normal_scale(glb, scale, lightmapped_only)

    @classmethod
    def set_glb_base_color(
        cls, glb: GlbTarget, base_color: Dict[str, Dict[str, Any]]
    ) -> List[Dict]:
        """Write base colour / texture into a GLB's materials, by name.

        The sibling of :meth:`set_glb_emissive`, for the same underlying gap.
        Measured against Maya 2025 + MtoA: ``lambert`` / ``blinn`` / ``phong``
        carry their colour through FBX (scaled by Maya's ``diffuse`` weight),
        while ``aiStandardSurface`` and ``standardSurface`` arrive with
        ``baseColorFactor`` at a flat **[1,1,1,1]** -- Maya's exporter does not
        map them at all, so every modern shader previews as white plastic. That
        also swamps emissive: a surface already at white leaves no headroom for
        an additive term to read against.

        Unlike emissive there is no strength extension and no zero-skip: black
        is a legitimate base colour, so values are clamped into [0,1] rather
        than normalized, and an all-zero colour is written as authored.

        Parameters:
            glb: Path to a binary glTF (.glb), modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.
            base_color: ``{material_name: {"color": (r, g, b), "texture": path,
                "alpha": a}}``. All keys optional. ``alpha`` is the material's
                own opacity factor (``baseColorFactor[3]``, which glTF
                multiplies into the texture's alpha): a shader whose opacity
                is ``texture alpha x a scalar`` or a bare constant -- a shadow
                plane's static fade -- that FBX cannot carry. Absent, the
                converter's alpha stands. Names absent from the GLB are
                reported.

        Returns:
            List of records: ``material``, ``factor``, ``texture``.
        """
        return cls._materials.set_glb_base_color(glb, base_color)

    # --------------------------- _fbx_materials.py: the FBX's materials record

    MATERIALS_KEY = _FbxMaterials.MATERIALS_KEY
    FBX_FAMILIES = _FbxMaterials.FBX_FAMILIES
    FBX_MAP_SLOTS = _FbxMaterials.FBX_MAP_SLOTS

    @classmethod
    def fbx_materials(
        cls, fbx: FbxSource, *, logger: Optional[logging.Logger] = None
    ) -> Dict[str, Any]:
        """The ``materials`` record for *fbx*: every material it carries.

        The record the file carries states each material it names -- the
        scene's own ``data_export`` carrier, else a referenced module's, the
        rule every carrier channel is read by -- and a material it does not
        name is read from the file's own properties (:meth:`read_fbx_materials`).
        An entry naming a material the file does not carry is out of scope.

        Parameters:
            fbx: A binary FBX path, or an :class:`FbxFile` already loaded.
            logger: Where what a material's reading cannot state is said;
                the ``_fbx_materials`` module's logger otherwise.

        Returns:
            The record's payload, version stamped; ``{"version", "materials":
            []}`` for a file with no materials.

        Raises:
            ValueError: *fbx* is not a binary FBX this reader can parse.
        """
        return cls._fbx_materials.fbx_materials(fbx, logger=logger)

    @classmethod
    def read_fbx_materials(
        cls, fbx: FbxSource, *, logger: Optional[logging.Logger] = None
    ) -> Dict[str, Any]:
        """*fbx*'s own materials read into the record, whatever it carries.

        Each by its shading family (:attr:`FBX_FAMILIES`), its maps by the
        file names the file binds them under -- the reading a file without a
        record gets, and what a carried record is checked against.

        Parameters:
            fbx: A binary FBX path, or an :class:`FbxFile` already loaded.
            logger: Where what a material's reading cannot state is said.

        Raises:
            ValueError: *fbx* is not a binary FBX this reader can parse.
        """
        return cls._fbx_materials.read_fbx_materials(fbx, logger=logger)

    @classmethod
    def apply_glb_materials(
        cls,
        glb: GlbTarget,
        record: Optional[Mapping[str, Any]],
        textures: Optional[Mapping[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Set each GLB material *record* states to exactly what it states.

        Every material carrying a stated name (a converter's duplicates
        included) takes every field: the factors, the alpha mode, double
        sidedness, unlit -- and each map, where a slot already sampling a file
        of the stated name is kept as bound and any other is rebound from
        *textures*. A stated metal or roughness map is packed with the
        occlusion map into one ORM image (R occlusion, G roughness, B metal;
        a channel with no map white, so its factor is its value) and bound as
        the occlusion map too when the record names one; a slot the record
        leaves empty is cleared. The lightmap pass that follows takes the
        occlusion slot of a baked material.

        A map that cannot be found is not trusted to the converter, whose own
        packing is the one measured to arrive solid white (metal everywhere):
        its channel reads its map type's absent value, as an absent channel
        does in every packing (``MapRegistry.absent_value``: a missing metal
        map dielectric, a missing roughness map matte), and it is warned
        about.

        Parameters:
            glb: Path to a binary glTF, modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.
            record: The ``materials`` record (:meth:`fbx_materials`).
            textures: ``{file name: path}`` -- where each named map's bytes
                are (the FBX's own media, extracted).

        Returns:
            One record per material written: ``material``, ``maps`` (the
            file names bound), ``missing`` (those that could not be) and
            ``orm`` (whether its metal and roughness are a packed map).
        """
        return cls._fbx_materials.apply_glb_materials(glb, record, textures)

    # Read beyond the facade (its tests, sibling modules): the part's own.
    @classmethod
    def _carried_record(cls, parsed) -> Optional[Dict[str, Any]]:
        return cls._fbx_materials._carried_record(parsed)

    # ----------------------------- _textures.py: the web-delivery texture pass

    WEB_DELIVERY_MAX_SIZE = _Textures.WEB_DELIVERY_MAX_SIZE
    WEB_DELIVERY_FORMAT = _Textures.WEB_DELIVERY_FORMAT
    WEB_DELIVERY_KTX2_FALLBACK = _Textures.WEB_DELIVERY_KTX2_FALLBACK
    WEB_DELIVERY_SECONDARY_MAX_SIZE = _Textures.WEB_DELIVERY_SECONDARY_MAX_SIZE
    WEB_DELIVERY_UASTC_RDO = _Textures.WEB_DELIVERY_UASTC_RDO
    WEB_DELIVERY_UASTC_RDO_DICTIONARY = _Textures.WEB_DELIVERY_UASTC_RDO_DICTIONARY
    BASIS_BY_SEMANTIC = _Textures.BASIS_BY_SEMANTIC
    LIGHTMAP_BASIS = _Textures.LIGHTMAP_BASIS
    SECONDARY_SEMANTICS = _Textures.SECONDARY_SEMANTICS
    LOSSY_SAFE_SEMANTICS = _Textures.LOSSY_SAFE_SEMANTICS
    LOSSLESS_WEBP = _Textures.LOSSLESS_WEBP

    @ClassProperty
    def OPTIMIZE_WORKERS(cls) -> int:
        """Deprecated alias of :attr:`ImgUtils.ENCODE_WORKERS`, for one release.

        The encode pass's worker cap moved to :meth:`ImgUtils.encode_workers`,
        the one policy every texture pass shares, and this reads it. Read-only:
        assigning it no longer changes the pass -- set
        ``ImgUtils.ENCODE_WORKERS``, or pass ``workers=``.
        """
        from pythontk.img_utils._img_utils import ImgUtils

        return ImgUtils.ENCODE_WORKERS

    @ClassProperty
    def UASTC_RDO_NORMAL_MAX(cls) -> float:
        """The RDO lambda a normal map is capped at whatever the caller asks:
        toktx's guidance, owned by :attr:`Ktx2Encoder.UASTC_RDO_NORMAL_MAX` so
        the map optimizer applies the same number. Read-only."""
        from pythontk.img_utils.ktx2_encoder import Ktx2Encoder

        return Ktx2Encoder.UASTC_RDO_NORMAL_MAX

    @classmethod
    def drop_glb_texture_fallbacks(cls, glb: GlbTarget) -> Dict[str, int]:
        """Drop the PNG/JPEG twin of every texture that also ships KTX2.

        ``KHR_texture_basisu`` is written as an OPTIONAL extension: the texture
        names a supercompressed image through the extension and an ordinary one
        through ``source``, so a reader without the extension still has a
        picture. The cost is that the deliverable carries both encodings of
        every map -- on the PROPS assembly, 40.66 MB of PNG beside 34.71 MB of
        KTX2, so the KTX2 pass more than doubled the texture budget it was
        meant to cut.

        This drops the fallbacks and moves ``KHR_texture_basisu`` into
        ``extensionsRequired``, which is what the spec asks of a file that can
        no longer be read without it. That is a DELIVERY decision, not a
        cleanup: the result needs a reader with the extension (in three.js, a
        registered ``KTX2Loader``). Callers that cannot promise one must not
        run this pass.

        Deliberately NOT wired into :meth:`fbx_to_glb` -- unlike
        :meth:`compact_glb_animations`, which is byte-exact and therefore
        unconditional. The pipeline owns this decision one step earlier:
        :meth:`web_delivery_texture_params` builds with ``ktx2_fallback=False``,
        so ``optimize_glb_textures`` never writes the twin in the first place.
        This is the RETROFIT, for a finished GLB that already carries twins --
        one built before that policy, or by a caller that asked for them.

        Only a texture that HAS a KTX2 twin loses its fallback -- a texture
        shipping a lone PNG keeps it -- and the orphaned images are collected
        by :meth:`prune_glb_unreferenced_textures`, which owns image
        renumbering and BIN repacking.

        Parameters:
            glb: Path to a ``.glb``, modified in place, or an open session.

        Returns:
            ``{"textures": n, "images": n, "bytes": n}``.
        """
        return cls._textures.drop_glb_texture_fallbacks(glb)

    @classmethod
    def describe_texture_pass(
        cls,
        summary: Dict[str, Any],
        image_format: str,
        max_size: int = 0,
        secondary_max_size: int = 0,
        uastc_rdo: Optional[float] = None,
        uastc_rdo_dictionary: Optional[int] = None,
    ) -> str:
        """Human-readable outcome of :meth:`optimize_glb_textures`, for log lines.

        Twin of :meth:`MapOptimizer.describe_size_clamp`, and here for the same
        reason: the wording is a claim about THIS method's contract, and both
        DCC exporters have to make it. They cannot import each other, so a
        sentence written in both drifts in both -- which it already had, the
        Blender copy trailing Maya's.

        The claim worth centralising is that *max_size* is a **ceiling, not a
        target**. An asset authored at the ceiling resamples nothing, so a
        caller that reports the MODE ("resized") tells its user the exporter
        rescaled textures it never touched -- read, on a 2048 set under a 2048
        ceiling, as "it upscaled my maps to 2K". ``summary["resized"]`` is the
        count that actually happened, and this renders it.

        Parameters:
            summary: What :meth:`optimize_glb_textures` returned. Empty means
                the pass ran and replaced nothing, which gets its own sentence
                -- "asked for and got nothing" must not read like "never ran".
            image_format: The carrier the pass was asked for (``"KTX2"``, ...).
            max_size: The ceiling that was in force; ``0`` = never resample.
            secondary_max_size: The data-map ceiling in force, if any.
            uastc_rdo: The UASTC RDO lambda in force, if any.
            uastc_rdo_dictionary: The RDO dictionary size in force, if any
                (None is toktx's own); named only beside an RDO lambda.

        Returns:
            One complete sentence, ready to log.
        """
        return cls._textures.describe_texture_pass(
            summary,
            image_format,
            max_size,
            secondary_max_size,
            uastc_rdo,
            uastc_rdo_dictionary,
        )

    @classmethod
    def web_delivery_texture_params(
        cls,
        image_format: Optional[str] = None,
        max_size: Optional[int] = None,
        ktx2_fallback: Optional[bool] = None,
        secondary_max_size: Optional[int] = None,
        uastc_rdo: Optional[float] = None,
        uastc_rdo_dictionary: Optional[int] = None,
    ) -> Dict[str, Any]:
        """:meth:`optimize_glb_textures` kwargs for a WEB deliverable.

        The one definition of "finished for the web", so the producers of that
        deliverable cannot each hold their own. They did, and it showed:
        measured on one production assembly, in one session, from one scene --
        the WebXR preview shipped 8.71 MB of WebP, the scene exporter with its
        panel dials untouched shipped 280.13 MB of full-resolution PNG, and the
        same exporter with the dials set to WebP shipped 22.06 MB because its
        size ceiling resolved from an absent template budget to "never
        resample". An artist approves the first and hands a developer the
        second; nothing in either log says they differ.

        Every parameter distinguishes *unspecified* from *chosen*: ``None``
        takes the policy, and any other value -- ``0`` for "keep every pixel"
        included -- is the caller's decision. That matters because the callers
        pass values resolved from UI dials whose own "unset" is falsy, and a
        falsy default reaching :meth:`optimize_glb_textures` as an explicit
        argument stops inheriting anything and starts meaning something.

        ``ktx2_fallback`` is policy too, and off: the PNG/JPEG twin a KTX2 image
        can carry serves only a stock glTF importer, and a web deliverable is
        read by a basisu-capable viewer. Leaving it to each producer ("a
        property of the consumer") let the exporters keep the twins -- a
        production 4K assembly shipped 145.8 MB of them beside 123.1 MB of KTX2,
        48% of the GLB. A caller whose GLB must also open in Blender or Unreal
        asks for the twins here -- the scene exporters' ``KTX2 + PNG/JPEG``
        Texture File Type does.

        Parameters:
            image_format: Container override, as a format id (``"WEBP"``) or a
                file extension (``"jpg"``, the vocabulary of the Scene
                Exporter's Texture File Type row) -- returned as the id
                :meth:`optimize_glb_textures` needs. ``None``/empty takes
                :attr:`WEB_DELIVERY_FORMAT`.
            max_size: Longest-edge ceiling in pixels; ``None`` takes
                :attr:`WEB_DELIVERY_MAX_SIZE`, ``0`` skips resizing.
            ktx2_fallback: Whether each KTX2 image also carries a PNG/JPEG
                twin (KTX2 mode only); ``None`` takes
                :attr:`WEB_DELIVERY_KTX2_FALLBACK` (no twins).
            secondary_max_size: A lower ceiling for the packed data maps
                (:attr:`SECONDARY_SEMANTICS`); ``None`` takes
                :attr:`WEB_DELIVERY_SECONDARY_MAX_SIZE`, ``0`` follows
                ``max_size``.
            uastc_rdo: UASTC RDO lambda for the KTX2 encodes; ``None`` takes
                :attr:`WEB_DELIVERY_UASTC_RDO`, ``0`` is off.
            uastc_rdo_dictionary: RDO dictionary size for those encodes;
                ``None`` takes :attr:`WEB_DELIVERY_UASTC_RDO_DICTIONARY`, ``0``
                is toktx's own.

        Returns:
            ``{"image_format": str, "max_size": int, "ktx2_fallback": bool,
            "secondary_max_size": int, "uastc_rdo": float | None,
            "uastc_rdo_dictionary": int | None}``.
        """
        return cls._textures.web_delivery_texture_params(
            image_format,
            max_size,
            ktx2_fallback,
            secondary_max_size,
            uastc_rdo,
            uastc_rdo_dictionary,
        )

    @classmethod
    def optimize_glb_textures(
        cls,
        glb: GlbTarget,
        max_size: int = WEB_DELIVERY_MAX_SIZE,
        image_format: str = WEB_DELIVERY_FORMAT,
        quality: int = 85,
        workers: Optional[int] = None,
        ktx2_fallback: bool = WEB_DELIVERY_KTX2_FALLBACK,
        secondary_max_size: int = WEB_DELIVERY_SECONDARY_MAX_SIZE,
        uastc_rdo: Optional[float] = WEB_DELIVERY_UASTC_RDO,
        uastc_rdo_dictionary: Optional[int] = WEB_DELIVERY_UASTC_RDO_DICTIONARY,
    ) -> Dict[str, Any]:
        """Downsize and re-encode a GLB's embedded images for web delivery.

        The texture budget IS the file: measured on a production room, the GLB
        was 94.7 MB of which 87.8 MB (93%) was uncompressed source PNG -- a
        24 MB normal map, a 20 MB character texture -- against 2.6 MB of
        geometry. A headset streams that, then holds it decoded in GPU memory.
        blendertk's native web export learned this first
        (``LightmapWebExport``: *"the texture budget is the whole file
        size"*); this is the same policy for the FBX->GLB path the WebXR
        preview and the exporters use, as a separate opt-in pass so plain
        conversions stay byte-stable.

        Every embedded image (bufferView-backed or data URI) is decoded,
        resized so its longest edge is *max_size*, and re-encoded. WebP is the
        default, and because nothing core-readable survives that pass (the
        image IS the WebP), ``EXT_texture_webp`` lands in ``extensionsRequired``
        -- the file says it needs a WebP-capable reader instead of handing a
        core one WebP bytes through a plain ``source``, which glTF 2.0, whose
        core permits image/jpeg and image/png only, does not allow. WebP is
        alpha-capable, universally decoded by WebXR-class browsers, and roughly
        an order of magnitude smaller than PNG at visually equal quality.

        The lossy encoder is used only where the channels ARE colour -- base
        colour and emissive (:attr:`LOSSY_SAFE_SEMANTICS`). Normal and
        metallic-roughness/occlusion maps re-encode LOSSLESS, because WebP's
        lossy mode is YUV 4:2:0 and their X/Z and metalness/occlusion channels
        sit in the half-resolution chroma planes, where they are resampled as
        if the eye were judging them; the same split
        :attr:`BASIS_BY_SEMANTIC` makes for KTX2. Lightmaps are additionally
        exempt from the resize (the bake sized them deliberately) and never
        lossy-encoded whatever they are sampled as; a lightmap is found by its
        binding -- a texCoord-1 occlusion/emissive map -- never by its name, so
        a digest-deduped image whose name lies is still protected. A re-encode
        that comes out larger keeps the original bytes.

        The BIN chunk is repacked -- image payloads replaced, former data-URI
        images relocated into it (dropping base64's 33%), every other
        bufferView's bytes copied verbatim with offsets recomputed. Textures
        sampling a converted image gain the standard ``EXT_texture_webp``
        binding while keeping their plain ``source`` as the fallback the
        extension spec describes.

        ``image_format="KTX2"`` is the GPU-memory half: images are encoded to
        KTX2/Basis Universal (requires the ``toktx`` encoder -- see
        ``pythontk.Ktx2Encoder``; the pass raises upfront when it is missing
        rather than silently shipping WebP) and textures rebind through
        ``KHR_texture_basisu``. Where WebP only shrinks the wire, Basis stays
        block-compressed in GPU memory -- measured on a delivered preview GLB,
        6 MB of WebP decoded to ~740 MB of RGBA+mips, which is the actual
        headset ceiling. Specifics of this mode, chosen deliberately:

        * **Per-slot codecs** -- images sampled as normal / metallic-roughness /
          occlusion encode UASTC + linear; base color / emissive encode ETC1S +
          sRGB; an image shared across slots takes the stricter treatment, and
          one bound to a texture but to no material slot gets UASTC + sRGB. An
          image *no texture samples* is left as found -- nothing could rebind
          it, and a non-core ``image/ktx2`` that no declaration enables is
          invalid glTF. Same rule as ``MapOptimizer.resolve_compression``,
          keyed structurally (by glTF slot) instead of by filename.
        * **Lightmaps take UASTC + sRGB** (:attr:`LIGHTMAP_BASIS`, which says
          why and what it measured) -- never ETC1S, which blotches them as
          lossy WebP does, never RDO, and never resampled: a lightmap whose
          edges are not both powers of two cannot meet the snap below, so it
          keeps a lossless container instead, as every lightmap does with
          :attr:`LIGHTMAP_BASIS` set to None. That container follows
          *ktx2_fallback*: with fallbacks ON it is the core-readable PNG,
          because EXT_texture_webp may only stay out of ``extensionsRequired``
          when the texture carries a PNG/JPEG twin, and WebP + twin costs more
          than the PNG alone (measured: 103 KB + 209 KB vs 209 KB). Pure
          delivery takes lossless WebP and requires the extension.
        * **No core-readable fallback by default** (*ktx2_fallback*, the
          web-delivery policy): the extension lands in
          ``extensionsRequired`` and the deliverable needs a
          ``KHR_texture_basisu``-capable viewer (three.js ``KTX2Loader``; the
          bundled preview page wires it). ``ktx2_fallback=True`` embeds a
          resized PNG/JPEG twin per converted image, bound as the texture's
          plain ``source`` -- the escape hatch the spec defines -- so the
          extension stays in ``extensionsUsed`` and a stock glTF importer
          (Blender, Unreal, Unity) still opens the file. That premium is a
          second copy, not a modest one: UASTC-class images (normals,
          metallic-roughness/occlusion) fall back to PNG, about the size of
          their KTX2, and ETC1S color to JPEG at *quality* (PNG when it
          carries alpha) -- measured on a production 4K assembly, 145.8 MB
          of twins beside 123.1 MB of KTX2. A fallback whose own encode
          fails is logged and dropped, and that image's binding re-tips the
          extension into ``extensionsRequired`` -- never an unreadable
          texture with a declaration claiming otherwise.
        * **Dimensions snap down to power-of-two** -- ``KHR_texture_basisu``
          requires multiple-of-4 dimensions and full mip pyramids (generated at
          encode time; a GPU-compressed texture cannot mip itself), and POT is
          what the WebGL/WebGPU backends want for that anyway. No-op for the
          usual POT sources.
        * **A larger encode is kept** (unlike the transport formats): UASTC can
          exceed a source PNG on the wire and still be the right answer,
          because the win being bought is GPU-resident format, not bytes.

        Parameters:
            glb: Path to a .glb, modified in place, or an open session.
            max_size: Longest edge kept after resize. 0/None skips resizing.
            image_format: ``"WEBP"`` (default), ``"KTX2"``, or any PIL-writable
                format.
            quality: Lossy quality for WEBP/JPEG, and the ETC1S quality dial in
                KTX2 mode (UASTC's tier is fixed by the encoder). Also the
                JPEG quality of KTX2-mode fallback images.
            workers: Concurrent encode threads. Defaults to
                :meth:`ImgUtils.encode_workers` (a memory-capped count shared
                with every other texture pass); 1 forces the serial path.
            ktx2_fallback: KTX2 mode only. ``False`` (default,
                :attr:`WEB_DELIVERY_KTX2_FALLBACK`) ships KTX2 alone and
                hard-requires a basisu-capable viewer
                (``extensionsRequired``). ``True`` embeds a core-readable
                PNG/JPEG twin per converted image and binds it as the
                texture's plain ``source``, keeping the GLB importable
                everywhere (extension in ``extensionsUsed``).
            secondary_max_size: A lower longest-edge ceiling for the packed
                data maps alone (:attr:`SECONDARY_SEMANTICS`: metallic-
                roughness / occlusion), never above *max_size*; ``0`` follows
                *max_size*. Color and normal maps keep the primary ceiling.
            uastc_rdo: KTX2 mode only. UASTC rate-distortion lambda for the
                normal / data encodes (``toktx --uastc_rdo_l``); a normal map
                is capped at :attr:`UASTC_RDO_NORMAL_MAX`. ``None`` is off.
                Measured on a 4K production set: ORM packs -30% at 1.0 (PSNR
                50/44/48 dB), a noisy normal map -3.5%, encode 3-4x slower.
            uastc_rdo_dictionary: KTX2 mode with *uastc_rdo* only. The RDO
                dictionary size (``toktx --uastc_rdo_d``); the default is the
                web-delivery policy (:attr:`WEB_DELIVERY_UASTC_RDO_DICTIONARY`),
                ``None`` toktx's own.

        Returns:
            Summary dict: ``images`` (converted count), ``bytes_before`` /
            ``bytes_after`` (image payload totals), and ``resized`` -- how many
            of those images were actually RESAMPLED. *max_size* is a ceiling,
            never a target: a source already within it keeps its pixels, so
            ``resized`` is 0 for a whole asset authored at the ceiling and a
            caller reporting the mode ("resized") rather than this count tells
            its user their textures were rescaled when nothing was. Empty when
            Pillow is unavailable or there is nothing to do.
        """
        return cls._textures.optimize_glb_textures(
            glb,
            max_size,
            image_format,
            quality,
            workers,
            ktx2_fallback,
            secondary_max_size,
            uastc_rdo,
            uastc_rdo_dictionary,
        )

    # Read beyond the facade (its tests, sibling modules): the part's own.
    @classmethod
    def _uastc_rdo_for(
        cls, semantic: Optional[str], uastc_rdo: Optional[float]
    ) -> Optional[float]:
        return cls._textures._uastc_rdo_for(semantic, uastc_rdo)

    _largest_first = staticmethod(_Textures._largest_first)

    # -------------------------- _images.py: embedding, de-duplication, pruning

    IMAGE_MIME_TYPES = _Images.IMAGE_MIME_TYPES
    GLB_IMAGE_FORMATS = _Images.GLB_IMAGE_FORMATS

    @classmethod
    def decode_glb_textures(
        cls, glb: str, dst: str, *, overwrite: bool = False
    ) -> Optional[str]:
        """Write a copy of *glb* whose KTX2 images are PNG; return its path.

        The way in for an importer that cannot read ``KHR_texture_basisu``:
        Blender 5.1's refuses the WHOLE file when the extension is required,
        which every KTX2 deliverable this package writes declares. Each KTX2
        image is decoded (:meth:`Ktx2Encoder.decode`) and replaced in place --
        same index, so every texture keeps its source -- the extension is
        dropped from each texture and from ``extensionsUsed`` /
        ``extensionsRequired``, and the KTX2 payload is reclaimed. Geometry,
        animation and every other byte are untouched. *glb* itself never is.

        Parameters:
            glb: The source GLB.
            dst: Where the copy goes.
            overwrite: Replace an existing *dst*.

        Returns:
            *dst*, or ``None`` when no texture needs the extension to reach
            its image (none uses KTX2, or each has a core ``source`` fallback):
            every importer reads *glb* as it is.

        Raises:
            FileExistsError: *dst* exists and *overwrite* is False.
            FileNotFoundError: A KTX2 image needs decoding and no ``ktx``
                tool (KTX-Software) is installed.
            RuntimeError: ``ktx`` could not decode an image.
        """
        return cls._images.decode_glb_textures(glb, dst, overwrite=overwrite)

    @classmethod
    def dedupe_glb_images(cls, glb: GlbTarget) -> Dict[str, int]:
        """Collapse byte-identical embedded images onto one copy.

        FBX2glTF embeds per MATERIAL, so two DCC materials wiring one texture
        file arrive as two images with identical bytes -- and glTF has no
        reason to keep both, since a texture names an image by index and any
        number may name the same one. The session's ``image_digests`` dedupe
        cannot reach this: it is write-side, stopping a WRITER from appending a
        payload already embedded, and by the time it runs the converter's own
        pair is already in the file. Measured on a production assembly: two
        duplicate pairs, 154.7 KB of a delivered 5.9 MB -- and, because this
        runs before :meth:`optimize_glb_textures`, two full-size decodes,
        resizes and re-encodes that bought nothing.

        Content-addressed, never name-addressed, for the same reason
        ``image_digests`` is: the copies arrive by different routes and their
        names are the one thing that may lie in either direction -- two
        different maps can share a name, and one map can carry two.

        Every texture pointing at a dropped image is rebound to the survivor
        (``source`` and every entry of :attr:`TEXTURE_CONTAINER_EXTENSIONS`,
        the same set `image_for_texture` resolves a binding through), as is
        every scene-sidecar reference to it; then the orphaned images and their
        exclusive bufferViews are reclaimed by
        :meth:`prune_glb_unreferenced_textures` -- which already owns index
        remapping and BIN repacking, so this pass only has to decide WHAT is
        redundant. Textures themselves are kept: two textures sampling one
        image may still differ by sampler, and collapsing them is a separate
        question this pass does not answer.

        Parameters:
            glb: Path to a ``.glb``, modified in place, or an open session.

        Returns:
            ``{"images": n, "bytes": n}`` -- images dropped and BIN payload
            reclaimed. A file with nothing to collapse is not rewritten.
        """
        return cls._images.dedupe_glb_images(glb)

    @classmethod
    def prune_glb_unreferenced_textures(cls, glb: GlbTarget) -> Dict[str, int]:
        """Drop textures no material samples, and the images/bufferViews only they used.

        The applier-tail sweep. Every channel writer here REBINDS a slot to the
        image it embeds and leaves whatever that slot named before in place --
        which for the ORM repack is FBX2glTF's own ``ao_met_rough_<mat>``
        packing, a full-size PNG in the BIN chunk. Measured on a production
        delivery (HOOKS_PINS.glb): 4 images, 1 unreferenced, 2 MB of dead
        payload the reviewer flagged as an orphaned texture -- one per repacked
        material, every export. Writers cannot prune for themselves: a texture
        is only provably dead once EVERY writer on the session has run, and
        dropping an image renumbers everything after it.

        Referenced means sampled by a material: any ``textureInfo`` in the
        material tree (a dict under a key ending in ``Texture`` with an integer
        ``index`` -- the spec's own naming for the core slots and every
        ``KHR_materials_*`` extension). Images are kept when a surviving
        texture reads them through ``source`` or an extension source
        (``KHR_texture_basisu``, ``EXT_texture_webp``). A bufferView is dropped
        only when the pruned images were its ONLY readers -- FBX2glTF does point
        two images at one view, and a view could in principle be shared with an
        accessor. Every surviving index is remapped in place: texture indices
        across the material tree and the session's embed cache, image indices
        in ``textures``, and ``bufferView`` keys anywhere in the document (the
        key is spec-uniform, so a generic walk covers accessors, sparse
        indices/values, images and compression extensions alike). The BIN is
        rebuilt from the surviving views only, 4-byte padded like every
        repack here.

        Runs after the materials are stated, and again wherever a later
        applier unbinds an image. Safe to run standalone on any GLB; a file
        with nothing to drop is not rewritten.

        Returns:
            ``{"textures": n, "images": n, "bytes": n}`` -- what was dropped;
            ``bytes`` is BIN payload reclaimed (a data-URI image counts 0 here,
            its saving shows up in the JSON chunk).
        """
        return cls._images.prune_glb_unreferenced_textures(glb)

    # ----------------------------------- _gltf2fbx.py: glTF -> FBX with no DCC

    GLTF_EXTENSIONS = GLTF_EXTENSIONS
    GLTF_UNDECODABLE_EXTENSIONS = GLTF_UNDECODABLE_EXTENSIONS

    @classmethod
    def gltf_to_fbx(
        cls,
        src: str,
        dst: Optional[str] = None,
        *,
        texture_dir: Optional[str] = None,
        include_animation: bool = True,
        fps: Optional[float] = None,
        manifest: bool = True,
        overwrite: bool = False,
        report: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Convert a ``.glb`` / ``.gltf`` to a binary FBX -- no DCC involved.

        Writes the payload a host imports a glTF through when it has no glTF
        importer (see the module docstring for what crosses and what does not):
        the FBX, the material images as files in *texture_dir*, and -- with
        *manifest* -- the ``.manifest.json`` sidecar beside the FBX.

        Parameters:
            src: The glTF container.
            dst: Output FBX path; defaults to *src* with ``.fbx``.
            texture_dir: Where the images are written, each named for the map
                type it feeds; defaults to ``<dst stem>_textures`` beside *dst*.
                KTX2 images are decoded to PNG through KTX-Software's ``ktx``
                (:meth:`Ktx2Encoder.decode`); without it they are skipped and
                reported, and the geometry still converts.
            include_animation: Convert the first animation's translation /
                rotation / scale channels.
            fps: The rate keys are resampled at where they must be (cubic
                splines; slerped rotation between sparse keys), and the clock the
                manifest records. ``None`` reads it off the key spacing, else 24.
            manifest: Write the :class:`HandoffManifest` sidecar.
            overwrite: Replace an existing *dst*.
            report: Filled with counts: nodes, meshes, materials, textures,
                textures_missing, skins, animation, fps, keys, skipped.

        Returns:
            The FBX path.

        Raises:
            FileNotFoundError: *src* does not exist.
            FileExistsError: *dst* exists and *overwrite* is False.
            ValueError: *src* is not a glTF container, or requires an
                extension whose geometry cannot be decoded here
                (:data:`GLTF_UNDECODABLE_EXTENSIONS`).
        """
        return _GltfToFbx.gltf_to_fbx(
            src,
            dst,
            texture_dir=texture_dir,
            include_animation=include_animation,
            fps=fps,
            manifest=manifest,
            overwrite=overwrite,
            report=report,
        )
