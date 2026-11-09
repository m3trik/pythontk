# !/usr/bin/python
# coding=utf-8
"""Shadow rigs: carry the ``shadows`` record and the files it names into the GLB.

One layout, two containers: the GLB carries the ``shadows`` record as the
FBX's carrier holds it, in its root ``extras``, and every file the record
names -- each plane's horizon map, its silhouette and the silhouette atlas --
as the bufferView named by that file (``SceneRecords.GLB_MEDIA``), through
the media part every record shares (``_media.py``). Nothing is bound to a material or a node
and no glTF image or sampler is made: a reader resolves each plane, its
source and its contact by name in the file's own scene, decodes each map off
its view and samples it as the record says (a horizon map is data: its
texels read whole -- the WebXR viewer's own texture takes no mipmaps and
clamps).

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through the facade it holds (``self._mc``).
"""

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Eager: the signatures name GlbEdit / GlbTarget; SceneRecords is the light
# declaration module every pass reads.
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit, GlbTarget

logger = logging.getLogger(__name__)


class _ShadowRigs:
    """Shadow rigs: the ``shadows`` record and its files, carried.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    #: The media type a file the record names rides as, by its extension:
    #: each is an image a glTF reader reads, so its bytes ride as they are,
    #: never re-encoded (a horizon map is data) and with no scale.
    MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

    def __init__(self, mc: type) -> None:
        self._mc = mc  # the facade class: every name it exposes is read through it

    def _shadow_record(self, gltf: dict, warn: bool = False) -> Optional[dict]:
        """The ``shadows`` record *gltf* carries in its current shape: its root
        ``extras``' (the record as the file ships it), else its carrier's
        (before the carry; :meth:`_current_record`); ``None`` without one, and
        -- said when *warn* -- for one the declaration refuses, either
        container's (``SceneRecords.problems``: another version, or a field
        the current shape requires missing): a reader refuses rather than
        guesses, and the cure is a re-export."""
        spec = SceneRecords.SHADOWS
        extras = gltf.get("extras") if isinstance(gltf, dict) else None
        carried = extras.get(spec.key) if isinstance(extras, dict) else None
        problems = [] if carried is None else SceneRecords.problems(spec, carried)
        if carried is not None and not problems:
            return carried
        # Before the carry -- or past a root copy the shape refuses -- the
        # carrier's, refused (and said) as every pass refuses it.
        record = self._mc.data_export_channel(gltf, spec.key)
        if record is not None:
            if warn:
                return self._mc._current_record(gltf, spec)
            return record if not SceneRecords.problems(spec, record) else None
        if problems and warn:
            more = f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""
            logger.warning(
                "%s record not applied -- %s%s; re-export the file.",
                spec.key,
                "; ".join(problems[:3]),
                more,
            )
        return None

    @staticmethod
    def plane_files(plane: Any) -> List[str]:
        """The file names one plane of the record names, folder-less, each
        once: its silhouette, the silhouette atlas, its horizon map
        (``FbxMedia.DEPENDENCY_FIELDS``, the walk the FBX embeds them by)."""
        from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia

        if not isinstance(plane, dict):
            return []
        return FbxMedia.named_files({SceneRecords.SHADOWS.key: [{"planes": [plane]}]})

    @classmethod
    def _mime_type(cls, name: str) -> str:
        """The media type the file *name* rides as (:attr:`MIME_TYPES`; PNG,
        what the rigs write, for any other extension)."""
        return cls.MIME_TYPES.get(os.path.splitext(name)[1].lower(), "image/png")

    @staticmethod
    def _image_bytes(edit: GlbEdit, name: str) -> Optional[bytes]:
        """The bytes of the glTF image *edit* already holds as the file *name*,
        or ``None``: an FBX that embedded a plane's map for its material
        arrives with it as an image the converter named by its source file
        (matched on the image's name, with or without its extension, or its
        ``uri``'s file name, without case) -- the FBX's own copy, which the
        pass carries before any loose one."""
        media = SceneRecords.GLB_MEDIA
        wanted = media.fold(name)
        stem = os.path.splitext(wanted)[0]
        for image in edit.images:
            label = str(image.get("name") or "").lower()
            uri = str(image.get("uri") or "")
            candidates = {label, os.path.splitext(label)[0]}
            if uri and not uri.startswith("data:"):
                candidates.add(media.fold(uri.split("?", 1)[0]))
            if wanted in candidates or stem in candidates:
                return edit.image_bytes(image)
        return None

    def apply_glb_shadows(
        self, glb: GlbTarget, *, search_dirs: Sequence[str] = ()
    ) -> Optional[Dict[str, Any]]:
        media = SceneRecords.GLB_MEDIA
        shared = self._mc._media  # the resolution and transport every record shares
        # ONE session for the read and the write: the record is read from the
        # very GLB being edited.
        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            # A plane drawn from a misread record looks like a bad bake rather
            # than a version problem: another schema -- or a record the
            # current shape refuses, in either container -- is refused, said.
            record = self._shadow_record(gltf, warn=True)
            if not record:
                return None
            # The record as it stands, under its key in the root extras: one
            # layout, two containers (``RecordSpec.glb_verbatim``).
            record = SceneRecords.put_glb_block(gltf, SceneRecords.SHADOWS, record)
            edit.dirty = True
            planes = [p for p in record.get("planes") or [] if isinstance(p, dict)]

            # The FBX's own copy first, as the atlas always was: an image the
            # GLB holds under the file's name, then the caller's search paths
            # in its order -- the conversion hands the files the FBX embeds
            # first -- then the file's own folder.
            dirs = [d for d in search_dirs if d]
            dirs.append(os.path.dirname(os.path.abspath(edit.path)))

            #: folded file name -> (the record's spelling, the planes naming
            #: it): each file rides once, however many planes share it (an
            #: atlas), and only for a plane this file carries -- a plane
            #: elsewhere would ship dead bytes.
            wanted: Dict[str, Tuple[str, List[str]]] = {}
            out_of_scope = 0
            for plane, nodes, status, leaves in shared.resolved(gltf, planes):
                name = plane.get("name")
                if status == "ambiguous":
                    logger.warning(
                        "Shadow plane %r: nothing tells its plane apart among the "
                        "GLB node(s) %s -- ambiguous, its maps not embedded.",
                        name,
                        ", ".join(sorted(leaves)),
                    )
                    continue
                if status == "absent" or not nodes:
                    out_of_scope += 1
                    continue
                for file in self.plane_files(plane):
                    entry = wanted.setdefault(media.fold(file), (file, []))
                    entry[1].append(str(name))

            missing: List[Tuple[str, List[str]]] = []
            for file, names in wanted.values():
                if media.views(gltf, file):
                    # The file's own (a second pass adds nothing) -- or a
                    # tie no reader binds, which this says.
                    shared.held(gltf, file, "Shadow map")
                    continue
                data = self._image_bytes(edit, file)
                if data is None:
                    path = shared.locate(dirs, file)
                    if path is not None:
                        with open(path, "rb") as fh:
                            data = fh.read()
                if data is None:
                    missing.append((file, names))
                    continue
                # As it is: a horizon map is data, a silhouette was sized by
                # the rasterizer -- neither is decoded or re-encoded.
                shared.put(edit, file, data, self._mime_type(file), kind="Shadow map")

            for file, names in missing:
                logger.warning(
                    "Shadow map %r not found in the file or %s -- the plane(s) "
                    "naming it (%s) draw without it.",
                    file,
                    dirs,
                    ", ".join(names),
                )
            if out_of_scope:
                logger.info(
                    "Shadow record covers %d plane(s) not in this export "
                    "(scene-wide record, exported subset); ignored.",
                    out_of_scope,
                )
            return record

    def media_findings(self, gltf: dict) -> Tuple[List[str], List[str]]:
        """``(problems, notes)``: the files *gltf*'s ``shadows`` record names,
        held to the media transport (``SceneRecords.GLB_MEDIA``).

        A file several bufferViews hold is held by none; a view a named file
        rides must state the image it is (:attr:`MIME_TYPES`) and no scale --
        a reader decodes by what the view states. A map a plane here reads
        that rides nothing is a note (the build warned of it: that plane draws
        without it).
        """
        from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia

        record = SceneRecords.glb_block(gltf, SceneRecords.SHADOWS)
        if not record:
            return [], []
        media = SceneRecords.GLB_MEDIA
        images = sorted(set(self.MIME_TYPES.values()))

        def an_image(extras: Dict[str, Any]) -> Optional[str]:
            mime = extras.get(media.mime_type)
            if mime in images and media.scale not in extras:
                return None
            return (
                f"states {mime!r}"
                + (" with a scale" if media.scale in extras else "")
                + f", not the image it is ({', '.join(images)}, no scale)"
            )

        named = FbxMedia.named_files({SceneRecords.SHADOWS.key: [record]})
        planes = [p for p in record.get("planes") or [] if isinstance(p, dict)]
        reading = {
            file
            for plane, nodes, status, _leaves in self._mc._media.resolved(gltf, planes)
            if nodes and status not in ("ambiguous", "absent")
            for file in self.plane_files(plane)
        }
        return self._mc._media.findings(
            gltf,
            SceneRecords.SHADOWS,
            {name: an_image for name in named},
            reading,
            readers="planes",
            without="draw without them",
        )
