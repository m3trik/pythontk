# !/usr/bin/python
# coding=utf-8
"""An FBX deliverable's materials, as the ``materials`` record states them.

The record (``MaterialRecord``) rides the file on its ``data_export`` carrier,
produced from the scene as staged for the write (the hosts'
``MaterialRecords``): what each material IS, as a literal glTF 2.0 document.
This part reads it (:meth:`fbx_materials`) and COPIES it into the GLB built
from the file (:meth:`apply_glb_materials`): each GLB material the record
names becomes the record's material, its maps embedded from the files the
record names. So the two deliverables of one export state every material
alike, and a GLB built from the FBX alone (the WebXR preview of a delivered
file) shows what the export did. The GLB carries no second copy of the
record: its materials are the record (one layout, two containers).

A file that carries no record -- a File > Export, a file older than the
record -- is read into the same model (``MaterialFamilies``) from its own
materials, each by the shading family its properties name
(:attr:`FBX_FAMILIES`): what any reader that knows the families would make of
it, its metal and roughness maps packed in the build's scratch folder.
Measured on a production assembly (2026-10-20), that reading of a Maya file
matched what its export's scene stated, material for material.

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through the facade it holds (``self._mc``).
"""

import copy
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from pythontk.core_utils.engines.scene_export.material_record import (
    MaterialFamilies,
)
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.core_utils.engines.scene_export.scene_store import SceneStoreBase
from pythontk.file_utils.mesh_convert.glb.edit import GlbTarget

logger = logging.getLogger(__name__)

#: A path or a parsed file -- what :meth:`fbx_materials` reads.
FbxSource = Any


class _FbxMaterials:
    """The ``materials`` record: read from an FBX, copied into a GLB.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    #: The record's channel on the FBX's ``data_export`` carrier.
    MATERIALS_KEY = SceneRecords.MATERIALS.key

    #: How a file's own material is read into a ``MaterialFamilies`` family,
    #: in match order: ``(family, the property only that family's materials
    #: carry, the method turning its properties into the family's values)``.
    #: A material no row claims is one of FBX's own models (Phong, Lambert)
    #: and reads as ``legacy``. A new dialect is one row and its method.
    FBX_FAMILIES: Tuple[Tuple[str, str, str], ...] = (
        ("stingray_pbs", "Maya|use_color_map", "_fbx_stingray_props"),
        ("pbr", "Maya|specularRoughness", "_fbx_standard_surface_props"),
    )

    #: An FBX material slot -> the logical channel its texture binds.
    FBX_MAP_SLOTS: Dict[str, str] = {
        # StingrayPBS (its graph's own attributes, as Maya writes them).
        "Maya|TEX_color_map": "baseColor",
        "Maya|TEX_metallic_map": "metallic",
        "Maya|TEX_roughness_map": "roughness",
        "Maya|TEX_ao_map": "ambientOcclusion",
        "Maya|TEX_normal_map": "normal",
        "Maya|TEX_emissive_map": "emission",
        "Maya|opacity": "opacity",
        # Standard Surface / aiStandardSurface.
        "Maya|baseColor": "baseColor",
        "Maya|metalness": "metallic",
        "Maya|specularRoughness": "roughness",
        "Maya|emissionColor": "emission",
        "Maya|normalCamera": "normal",
        # FBX's own models.
        "DiffuseColor": "baseColor",
        "TransparentColor": "opacity",
        "EmissiveColor": "emission",
        "NormalMap": "normal",
    }

    #: The extensions a record's materials and maps may carry, which the
    #: GLB's ``extensionsUsed`` states exactly when one of its materials does.
    RECORD_EXTENSIONS: Tuple[str, ...] = (
        MaterialFamilies.STRENGTH_EXTENSION,
        MaterialFamilies.UNLIT_EXTENSION,
        MaterialFamilies.TRANSFORM_EXTENSION,
    )

    def __init__(self, mc: type) -> None:
        self._mc = mc  # the facade class: every name it exposes is read through it

    # ------------------------------------------------------------------ read
    def fbx_materials(
        self, fbx: FbxSource, *, logger: Optional[logging.Logger] = None
    ) -> Dict[str, Any]:
        parsed = self._parsed_fbx(fbx)
        return MaterialFamilies.overlay(
            self._readable(self._mc._carried_record(parsed)),
            self._read_statements(parsed, logger=logger),
        )

    def read_fbx_materials(
        self, fbx: FbxSource, *, logger: Optional[logging.Logger] = None
    ) -> Dict[str, Any]:
        return MaterialFamilies.payload(self._read_statements(fbx, logger=logger))

    def _read_statements(
        self, fbx: FbxSource, *, logger: Optional[logging.Logger] = None
    ) -> List[Dict[str, Any]]:
        """The file's own materials, each as its family states it
        (``MaterialFamilies.entry``: maps by the file name the FBX binds), in
        file order; what they cannot state warned on *logger*."""
        log = logger or globals()["logger"]
        notes: List[str] = []
        entries = []
        for raw in self._parsed_fbx(fbx).materials():
            if raw.get("name"):
                entries.append(self._read_fbx_material(raw, notes))
        for note in dict.fromkeys(notes):
            log.warning(note)
        return entries

    def _read_fbx_material(
        self, raw: Mapping[str, Any], notes: List[str]
    ) -> Dict[str, Any]:
        """One of the file's materials as its family states it."""
        props = raw.get("props") or {}
        maps = {
            self._mc.FBX_MAP_SLOTS[slot]: MaterialFamilies.file_name(name)
            for slot, name in (raw.get("textures") or {}).items()
            if slot in self._mc.FBX_MAP_SLOTS and name
        }
        for family, marker, method in self._mc.FBX_FAMILIES:
            if marker in props:
                # The row's method on the class read through first, as the row
                # is (a subclass's dialect), else this part's own.
                read = getattr(self._mc, method, None) or getattr(self, method)
                values = read(props, notes, raw.get("name"))
                break
        else:
            family = "legacy"
            values = dict(props, ShadingModel=raw.get("shading_model") or "")
        return MaterialFamilies.entry(family, raw["name"], values, maps, notes=notes)

    @staticmethod
    def _fbx_stingray_props(
        props: Mapping[str, Any], notes: List[str], name: Any
    ) -> Dict[str, Any]:
        """A StingrayPBS's own attributes: Maya writes each as ``Maya|<attr>``."""
        return {key[5:]: value for key, value in props.items() if key.startswith("Maya|")}

    @staticmethod
    def _fbx_standard_surface_props(
        props: Mapping[str, Any], notes: List[str], name: Any
    ) -> Dict[str, Any]:
        """A Standard Surface's attributes in the ``pbr`` vocabulary: its
        opacity colour and its transmission (glass) as the family reads them
        (``MaterialFamilies.opacity_scalar``, ``TRANSMISSION_NOTE``)."""
        return {
            "base_weight": props.get("Maya|base", 1.0),
            "base_color": props.get("Maya|baseColor"),
            "metallic": props.get("Maya|metalness"),
            "roughness": props.get("Maya|specularRoughness"),
            "emission_weight": props.get("Maya|emission", 0.0),
            "emission_color": props.get("Maya|emissionColor"),
            "opacity": MaterialFamilies.as_floats(props.get("Maya|opacity"), 3, 1.0),
            "transmission": props.get("Maya|transmission"),
        }

    def _carried_record(self, parsed) -> Optional[Dict[str, Any]]:
        """The ``materials`` record a parsed FBX carries, decoded -- the
        scene's own carrier's (the root-level ``data_export``,
        ``SceneStoreBase.is_scene_carrier``), else the first module's carrier's
        (``NS:data_export``, or one under a group), the precedence every
        carrier channel is read by (``SceneStoreBase.reader_values``; an
        artist's property of the name on another node is no record) -- or
        ``None``."""
        key = self._mc.MATERIALS_KEY
        chosen = SceneStoreBase.reader_values(
            (owner, True, key, bytes(value))
            for owner, value in parsed.user_properties(key, named=True, paths=True)
            if isinstance(value, (bytes, bytearray)) and value.strip()
        ).get(key)
        found = chosen[0] if chosen else None
        try:
            return json.loads(found.decode("utf-8")) if found else None
        except (ValueError, UnicodeDecodeError):
            logger.warning("The file's materials record is unreadable; it is ignored.")
            return None

    @staticmethod
    def _readable(carried: Any) -> Optional[Mapping[str, Any]]:
        """*carried* when a reader reads it, else ``None`` -- a record its
        declaration refuses (another version, the flat version 1 among them,
        or what its shape refuses: ``SceneRecords.problems``, every record's
        one gate) is ignored whole, said with its cure, never guessed at."""
        if carried is None:
            return None
        problems = SceneRecords.problems(SceneRecords.MATERIALS, carried)
        if not problems:
            return carried
        more = f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""
        logger.warning(
            "The file's materials record is not read -- %s%s; re-export the "
            "file to state its materials. They are read from the file instead.",
            "; ".join(problems[:3]),
            more,
        )
        return None

    @staticmethod
    def _parsed_fbx(fbx: FbxSource):
        """*fbx* parsed (a path), or as given (an :class:`FbxFile`)."""
        from pythontk.file_utils.mesh_convert.fbx_file import FbxFile

        if isinstance(fbx, (str, os.PathLike)):
            return FbxFile.load(str(fbx), raw_payloads=False)
        return fbx

    # ----------------------------------------------------------------- apply
    def apply_glb_materials(
        self,
        glb: GlbTarget,
        record: Optional[Mapping[str, Any]],
        textures: Optional[Mapping[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        entries = MaterialFamilies.materials_by_name(record)
        if not entries:
            return []
        lookup = {name.casefold(): path for name, path in (textures or {}).items()}
        records: List[Dict[str, Any]] = []
        with self._mc.open_glb(glb) as edit:
            for name, entry, mat in self._mc._materials._match_glb_materials(
                edit.gltf, entries, "apply_glb_materials"
            ):
                written = self._copy_material(edit, mat, entry, record, lookup)
                records.append({"material": name, **written})
            if not records:
                return []
            gltf = edit.gltf
            # The lists state what the file's materials use, every one of them
            # (a material the record does not name keeps the converter's): an
            # extension a record took off its last material leaves them.
            used = set()
            for material in gltf.get("materials") or []:
                if isinstance(material, dict):
                    used.update(MaterialFamilies.extensions_of(material))
            for extension in self.RECORD_EXTENSIONS:
                for field in ("extensionsUsed", "extensionsRequired"):
                    listed = list(gltf.get(field) or [])
                    if extension not in used:
                        listed = [e for e in listed if e != extension]
                    elif field == "extensionsUsed" and extension not in listed:
                        listed.append(extension)
                    if listed:
                        gltf[field] = listed
                    else:
                        gltf.pop(field, None)
            self._mc._images._prune_empty_containers(gltf)
            edit.dirty = True
        missing = sorted({(r["material"], m) for r in records for m in r["missing"]})
        if missing:
            logger.warning(
                "Materials: %d map(s) the materials record names were not found: "
                "%s. Each reads as absent (MaterialFamilies.absent_map: a metal "
                "map's factor dielectric, a roughness constant kept, a glow "
                "map's emission none).",
                len(missing),
                ", ".join(f"{mat} ({name})" for mat, name in missing),
            )
        return records

    def _apply_fbx_materials(
        self,
        edit,
        fbx: str,
        search_dirs: Iterable[str] = (),
        read: Optional[List[Mapping[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """State *fbx*'s materials on the open GLB converted from it.

        The record the converted file carries (read off the GLB's own
        transcription of the carrier, so a caller's overlay of the channel
        counts) states each material it names; the rest are read from the
        FBX's own materials, their metal and roughness maps packed in this
        build's scratch folder. Copied by :meth:`apply_glb_materials` with
        each map's bytes taken from the FBX itself -- the deliverable's own
        copy, not whatever a disk holds under the name -- and, for a map it
        does not embed, from the pack store (the packed image the export's
        producer wrote: ``MaterialFamilies.packed_file``), then from
        *search_dirs* (the host's live folders). Then
        the record leaves the GLB's node extras: the GLB's materials are the
        record now, and a second account of them could only drift.

        Parameters:
            edit: The open GLB session.
            fbx: The FBX the GLB was converted from.
            search_dirs: Where a map the FBX does not embed is looked for.
            read: The FBX's own materials' statements (:meth:`_read_statements`
                of *fbx*), when a caller parsed it already; read here
                otherwise.

        Returns:
            ``(record, written)`` -- the record stated and
            :meth:`apply_glb_materials`' report.
        """
        from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        carried = self._readable(
            self._mc.data_export_channel(edit.gltf, self._mc.MATERIALS_KEY)
        )
        statements = list(read if read is not None else self._read_statements(fbx))
        named = MaterialFamilies.materials_by_name(carried)
        names = MaterialFamilies.texture_files(carried)
        for statement in statements:
            if statement.get("name") in named:
                continue
            for field in MaterialFamilies.TEXTURE_FIELDS:
                name = statement.get(field)
                if name and MaterialFamilies.file_name(name) not in names:
                    names.append(MaterialFamilies.file_name(name))
        store = TempArtifacts("glb_material_maps", policy="scoped")
        try:
            found = FbxMedia.extract(fbx, store.dir_path(), names) if names else {}
            for name in names:
                if name in found:
                    continue
                path = MaterialFamilies.packed_file(name) or next(
                    (
                        os.path.join(folder, name)
                        for folder in search_dirs
                        if folder and os.path.isfile(os.path.join(folder, name))
                    ),
                    None,
                )
                if path:
                    found[name] = path
            lookup = {name.casefold(): path for name, path in found.items()}

            def pack(*maps: Optional[str], name: Optional[str] = None) -> Optional[str]:
                paths = [
                    lookup.get(MaterialFamilies.file_name(m).casefold()) if m else None
                    for m in maps
                ]
                if any(m and not p for m, p in zip(maps, paths)):
                    return None
                try:
                    packed = MaterialFamilies.pack_metallic_roughness(
                        *paths, output_dir=store.dir_path(), name=name
                    )
                except OSError as error:  # a preview's map: said, read as absent
                    logger.warning("Materials: %s", error)
                    return None
                if packed:
                    found[MaterialFamilies.file_name(packed)] = packed
                return packed

            notes: List[str] = []
            record = MaterialFamilies.overlay(carried, statements, pack=pack, notes=notes)
            for note in dict.fromkeys(notes):
                logger.warning(note)
            written = self._mc.apply_glb_materials(edit, record, found)
        finally:
            store.cleanup(force=True)
        if self._mc._sidecar._drop_data_export_channel(
            edit.gltf, self._mc.MATERIALS_KEY
        ):
            edit.dirty = True
        return record, written

    def _copy_material(
        self,
        edit,
        mat: dict,
        entry: Mapping[str, Any],
        record: Mapping[str, Any],
        lookup: Mapping[str, str],
    ) -> Dict[str, Any]:
        """*mat* (a GLB material) made *entry* (one of *record*'s), whole: its
        maps bound to the GLB's images of the files *record* names, embedded
        from *lookup* where the GLB holds none; ``{"maps", "missing", "orm"}``.

        A map neither holds is named in ``missing`` and read as absent, the
        one rule every reader applies (``MaterialFamilies.absent_map``). The
        material's ``extras`` (another pass's notes on it) are kept."""
        found = copy.deepcopy(dict(entry))
        maps: List[str] = []
        missing: List[str] = []
        for holder, slot, name, sampler in list(
            MaterialFamilies.bindings(record, found)
        ):
            bound = self._glb_texture(edit, name, sampler, lookup) if name else None
            if bound is None:
                if name:
                    missing.append(name)
                MaterialFamilies.absent_map(found, slot)
                continue
            holder[slot]["index"] = bound
            if name not in maps:
                maps.append(name)
        if "extras" in mat:
            found["extras"] = mat["extras"]
        mat.clear()
        mat.update(found)
        return {
            "maps": maps,
            "missing": sorted(set(missing)),
            "orm": "metallicRoughnessTexture" in (found.get("pbrMetallicRoughness") or {}),
        }

    def _glb_texture(
        self,
        edit,
        name: str,
        sampler: Optional[Mapping[str, Any]],
        lookup: Mapping[str, str],
    ) -> Optional[int]:
        """The GLB texture sampling file *name* through *sampler* (``None``:
        how the file already samples the image), or ``None`` when neither the
        GLB nor *lookup* holds the file. An image the GLB holds under the name
        (the converter's embed of the FBX's own map) is reused; else the file
        is embedded once per session."""
        image = next(
            (
                index
                for index in range(len(edit.images))
                if self._same_file(edit.image_label(index), name)
            ),
            None,
        )
        if image is None:
            path = lookup.get(name.casefold())
            texture = self._mc._images._embed_image(edit, path) if path else None
            if texture is None:
                return None
            image = edit.image_for_texture(texture)
            if image is None:
                return None
            if sampler is None:
                return texture
        if sampler is None:
            return edit.texture_for_image(image)
        spec = dict(sampler)
        samplers = edit.gltf.setdefault("samplers", [])
        if spec not in samplers:
            samplers.append(spec)
        wanted = {"source": image, "sampler": samplers.index(spec)}
        textures = edit.gltf.setdefault("textures", [])
        for index, texture in enumerate(textures):
            if {k: texture.get(k) for k in wanted} == wanted and not texture.get(
                "extensions"
            ):
                return index
        textures.append(dict(wanted))
        return len(textures) - 1

    # ---------------------------------------------------------------- values
    @staticmethod
    def _same_file(label: Any, name: str) -> bool:
        """Whether an image *label* (a glTF image name) names file *name*."""
        return MaterialFamilies.file_name(label).casefold() == name.casefold()
