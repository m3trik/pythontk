# !/usr/bin/python
# coding=utf-8
"""An FBX deliverable's materials, as the ``materials`` record states them.

The record (``MaterialRecord``) rides the file on its ``data_export`` carrier,
produced from the scene as staged for the write (the hosts'
``MaterialRecords``): what each material IS, in glTF 2.0's terms. This part
reads it (:meth:`fbx_materials`) and applies it to the GLB built from the file
(:meth:`apply_glb_materials`), so the two deliverables of one export state
every material alike and a GLB built from the FBX alone (the WebXR preview of
a delivered file) shows what the export did.

A file that carries no record -- a File > Export, a file older than the
record -- is read into the same model (``MaterialFamilies``) from its own
materials, each by the shading family its properties name
(:attr:`FBX_FAMILIES`): what any reader that knows the families would make of
it. Measured on a production assembly (2026-10-20), that reading of a Maya
file matched what its export's scene stated, material for material.

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through the facade it holds (``self._mc``).
"""

import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from pythontk.core_utils.engines.scene_export.material_record import (
    ALPHA_MODES,
    MaterialFamilies,
)
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords, Scope
from pythontk.core_utils.engines.scene_export.scene_store import SceneStoreBase
from pythontk.core_utils.engines.textures.map_registry import MapRegistry
from pythontk.file_utils.mesh_convert.glb.edit import GlbTarget

logger = logging.getLogger(__name__)

#: A path or a parsed file -- what :meth:`fbx_materials` reads.
FbxSource = Any


class _FbxMaterials:
    """The ``materials`` record: read from an FBX, applied to a GLB.

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

    def __init__(self, mc: type) -> None:
        self._mc = mc  # the facade class: every name it exposes is read through it

    # ------------------------------------------------------------------ read
    def fbx_materials(
        self, fbx: FbxSource, *, logger: Optional[logging.Logger] = None
    ) -> Dict[str, Any]:
        parsed = self._parsed_fbx(fbx)
        return self._stated_materials(
            self._mc._carried_record(parsed),
            self._mc.read_fbx_materials(parsed, logger=logger),
        )

    def read_fbx_materials(
        self, fbx: FbxSource, *, logger: Optional[logging.Logger] = None
    ) -> Dict[str, Any]:
        log = logger or globals()["logger"]
        notes: List[str] = []
        entries = []
        for raw in self._parsed_fbx(fbx).materials():
            if raw.get("name"):
                entries.append(self._read_fbx_material(raw, notes))
        for note in dict.fromkeys(notes):
            log.warning(note)
        return MaterialFamilies.payload(entries)

    def _read_fbx_material(
        self, raw: Mapping[str, Any], notes: List[str]
    ) -> Dict[str, Any]:
        """One of the file's materials as its family states it."""
        props = raw.get("props") or {}
        maps = {
            self._mc.FBX_MAP_SLOTS[slot]: name
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
        """A Standard Surface's attributes in the ``pbr`` vocabulary; its
        transmission (glass) has no place in the record, and is said."""
        opacity = MaterialFamilies.as_floats(props.get("Maya|opacity"), 3, 1.0)
        if MaterialFamilies.as_float(props.get("Maya|transmission"), 0.0) > 0.0:
            notes.append(
                f"Material {name}: its transmission is not stated (the record "
                "has no transmission); it reads as its opacity alone."
            )
        return {
            "base_weight": props.get("Maya|base", 1.0),
            "base_color": props.get("Maya|baseColor"),
            "metallic": props.get("Maya|metalness"),
            "roughness": props.get("Maya|specularRoughness"),
            "emission_weight": props.get("Maya|emission", 0.0),
            "emission_color": props.get("Maya|emissionColor"),
            "opacity": sum(opacity) / 3.0,
        }

    def _carried_record(self, parsed) -> Optional[Dict[str, Any]]:
        """The ``materials`` record a parsed FBX carries, decoded -- the
        scene's own carrier's, else the first module's carrier's
        (``NS:data_export``), the precedence every carrier channel is read by
        (``FbxMedia.reader_records``) -- or ``None``."""
        import json

        own = SceneStoreBase.name(Scope.DELIVERABLE)
        found = None
        for owner, value in parsed.user_properties(self._mc.MATERIALS_KEY, named=True):
            if not isinstance(value, (bytes, bytearray)) or not value.strip():
                continue
            if not SceneStoreBase.is_carrier(owner):
                continue  # an artist's property of the name: no record
            if owner == own or found is None:
                found = bytes(value)
                if owner == own:
                    break
        try:
            return json.loads(found.decode("utf-8")) if found else None
        except (ValueError, UnicodeDecodeError):
            logger.warning("The file's materials record is unreadable; it is ignored.")
            return None

    def _stated_materials(
        self, carried: Any, read: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """*read* (the file's own materials) with each material *carried*
        names taken from it instead -- a record its declaration refuses
        (another version, or what its shape refuses: ``SceneRecords.problems``,
        every record's one gate) is ignored whole, said, never guessed at."""
        problems = (
            SceneRecords.problems(SceneRecords.MATERIALS, carried)
            if carried is not None
            else []
        )
        stated = self._record_entries(carried) if carried and not problems else {}
        if problems:
            more = f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""
            logger.warning(
                "The file's materials record is not read -- %s%s; its materials "
                "are read from the file instead.",
                "; ".join(problems[:3]),
                more,
            )
        return MaterialFamilies.payload(
            MaterialFamilies.finish(stated.get(entry["name"], entry))
            for entry in read.get("materials") or ()
        )

    @staticmethod
    def _parsed_fbx(fbx: FbxSource):
        """*fbx* parsed (a path), or as given (an :class:`FbxFile`)."""
        from pythontk.file_utils.mesh_convert.fbx_file import FbxFile

        if isinstance(fbx, (str, os.PathLike)):
            return FbxFile.load(str(fbx), raw_payloads=False)
        return fbx

    @staticmethod
    def _record_entries(record: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """``{name: entry}`` of a record payload; the first of a repeated name."""
        entries: Dict[str, Dict[str, Any]] = {}
        for entry in (record or {}).get("materials") or ():
            if isinstance(entry, Mapping) and entry.get("name"):
                entries.setdefault(str(entry["name"]), dict(entry))
        return entries

    # ----------------------------------------------------------------- apply
    def apply_glb_materials(
        self,
        glb: GlbTarget,
        record: Optional[Mapping[str, Any]],
        textures: Optional[Mapping[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        entries = self._record_entries(record)
        if not entries:
            return []
        lookup = {name.casefold(): path for name, path in (textures or {}).items()}
        records: List[Dict[str, Any]] = []
        with self._mc.open_glb(glb) as edit:
            for name, entry, mat in self._mc._materials._match_glb_materials(
                edit.gltf, entries, "apply_glb_materials"
            ):
                written = self._write_material(edit, mat, entry, lookup)
                written.pop("strength")
                records.append({"material": name, **written})
            if not records:
                return []
            gltf = edit.gltf
            # The lists state what the file's materials use, every one of them
            # (a material the record does not name keeps the converter's): an
            # extension a record took off its last material leaves them.
            materials = [m for m in gltf.get("materials") or [] if isinstance(m, dict)]
            for extension in (
                "KHR_materials_emissive_strength",
                self._mc.UNLIT_EXTENSION,
            ):
                used = any(extension in (m.get("extensions") or {}) for m in materials)
                for field in ("extensionsUsed", "extensionsRequired"):
                    listed = list(gltf.get(field) or [])
                    if not used:
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
                "%s. Each reads its map type's absent value (a metal map "
                "dielectric, a roughness map matte).",
                len(missing),
                ", ".join(f"{mat} ({name})" for mat, name in missing),
            )
        return records

    def _apply_fbx_materials(
        self,
        edit,
        fbx: str,
        search_dirs: Iterable[str] = (),
        read: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """State *fbx*'s materials on the open GLB converted from it.

        The record the converted file carries (read off the GLB's own
        transcription of the carrier, so a caller's overlay of the channel
        counts) states each material it names; the rest are read from the
        FBX's own materials. Applied by :meth:`apply_glb_materials` with each
        map's bytes taken from the FBX itself -- the deliverable's own copy,
        not whatever a disk holds under the name -- and, for a map it does not
        embed, from *search_dirs* (the host's live folders). Then the record
        leaves the GLB's node extras: the GLB's materials state it now, and a
        second account of them could only drift.

        Parameters:
            edit: The open GLB session.
            fbx: The FBX the GLB was converted from.
            search_dirs: Where a map the FBX does not embed is looked for.
            read: :meth:`read_fbx_materials` of *fbx*, when a caller parsed
                it already; read here otherwise.

        Returns:
            ``(record, written)`` -- the record stated and
            :meth:`apply_glb_materials`' report.
        """
        from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        record = self._stated_materials(
            self._mc.data_export_channel(edit.gltf, self._mc.MATERIALS_KEY),
            read if read is not None else self._mc.read_fbx_materials(fbx),
        )
        names = MaterialFamilies.texture_files(record)
        store = TempArtifacts("glb_material_maps", policy="scoped")
        try:
            found = FbxMedia.extract(fbx, store.dir_path(), names) if names else {}
            for name in names:
                if name in found:
                    continue
                path = next(
                    (
                        os.path.join(folder, name)
                        for folder in search_dirs
                        if folder and os.path.isfile(os.path.join(folder, name))
                    ),
                    None,
                )
                if path:
                    found[name] = path
            written = self._mc.apply_glb_materials(edit, record, found)
        finally:
            store.cleanup(force=True)
        if self._mc._sidecar._drop_data_export_channel(
            edit.gltf, self._mc.MATERIALS_KEY
        ):
            edit.dirty = True
        return record, written

    def _write_material(
        self, edit, mat: dict, entry: Mapping[str, Any], lookup: Mapping[str, str]
    ) -> Dict[str, Any]:
        """One GLB material set to *entry*; ``{"maps", "missing", "orm",
        "strength"}``."""
        maps: List[str] = []
        missing: List[str] = []

        def bind(container: dict, slot: str, name: Optional[str]) -> None:
            if not name:
                container.pop(slot, None)
                return
            current = (container.get(slot) or {}).get("index")
            image = edit.image_for_texture(current) if current is not None else None
            if image is not None and self._same_file(edit.image_label(image), name):
                maps.append(name)
                return
            path = lookup.get(name.casefold())
            index = self._mc._images._embed_image(edit, path) if path else None
            if index is None:
                missing.append(name)
                return
            container[slot] = {"index": index}
            maps.append(name)

        pbr = mat.setdefault("pbrMetallicRoughness", {})
        pbr["baseColorFactor"] = list(entry.get("baseColorFactor") or (1.0, 1.0, 1.0, 1.0))
        bind(pbr, "baseColorTexture", entry.get("baseColorTexture"))

        metal, rough = entry.get("metallicTexture"), entry.get("roughnessTexture")
        occlusion = entry.get("occlusionTexture")
        metallic = float(entry.get("metallicFactor", 1.0))
        roughness = float(entry.get("roughnessFactor", 1.0))
        sources = {
            name: lookup.get(name.casefold()) for name in (metal, rough, occlusion) if name
        }
        missing.extend(name for name, path in sources.items() if not path)
        if metal and not sources[metal]:
            metal, metallic = None, MapRegistry.absent_value("Metallic")
        if rough and not sources[rough]:
            rough, roughness = None, MapRegistry.absent_value("Roughness")
        pbr["metallicFactor"], pbr["roughnessFactor"] = metallic, roughness
        packed = None
        if metal or rough:
            packed = self._pack_material_orm(
                edit,
                sources.get(occlusion) if occlusion else None,
                sources.get(rough) if rough else None,
                sources.get(metal) if metal else None,
                name=metal or rough,
            )
        if packed is not None:
            pbr["metallicRoughnessTexture"] = {"index": packed}
            maps.extend(n for n in (occlusion, rough, metal) if n and sources.get(n))
        else:
            pbr.pop("metallicRoughnessTexture", None)
            if metal or rough:  # the pack failed: neither map binds
                if metal:
                    pbr["metallicFactor"] = MapRegistry.absent_value("Metallic")
                if rough:
                    pbr["roughnessFactor"] = MapRegistry.absent_value("Roughness")
        if occlusion and packed is not None and sources.get(occlusion):
            mat["occlusionTexture"] = {"index": packed}
        elif occlusion and occlusion in missing:
            mat.pop("occlusionTexture", None)
        else:
            bind(mat, "occlusionTexture", occlusion)

        bind(mat, "normalTexture", entry.get("normalTexture"))
        normal = mat.get("normalTexture")
        if normal is not None:
            scale = float(entry.get("normalScale", 1.0))
            if scale != 1.0:
                normal["scale"] = scale
            else:
                normal.pop("scale", None)

        extensions = mat.setdefault("extensions", {})
        emissive = [float(c) for c in entry.get("emissiveFactor") or (0.0, 0.0, 0.0)]
        peak = max(emissive) if emissive else 0.0
        strength = False
        if peak <= 0.0:
            mat["emissiveFactor"] = [0.0, 0.0, 0.0]
            mat.pop("emissiveTexture", None)
            extensions.pop("KHR_materials_emissive_strength", None)
        else:
            if peak > 1.0:
                emissive = [c / peak for c in emissive]
                extensions["KHR_materials_emissive_strength"] = {"emissiveStrength": peak}
                strength = True
            else:
                extensions.pop("KHR_materials_emissive_strength", None)
            mat["emissiveFactor"] = emissive
            bind(mat, "emissiveTexture", entry.get("emissiveTexture"))

        mode = entry.get("alphaMode") if entry.get("alphaMode") in ALPHA_MODES else "OPAQUE"
        mat["alphaMode"] = mode
        if mode == "MASK":
            mat["alphaCutoff"] = float(entry.get("alphaCutoff", 0.5))
        else:
            mat.pop("alphaCutoff", None)
        mat["doubleSided"] = bool(entry.get("doubleSided"))
        if entry.get("unlit"):
            extensions[self._mc.UNLIT_EXTENSION] = {}
        else:
            extensions.pop(self._mc.UNLIT_EXTENSION, None)
        if not extensions:
            mat.pop("extensions", None)
        return {
            "maps": maps,
            "missing": missing,
            "orm": packed is not None,
            "strength": strength,
        }

    def _pack_material_orm(
        self,
        edit,
        occlusion: Optional[str],
        rough: Optional[str],
        metal: Optional[str],
        name: str,
    ) -> Optional[int]:
        """The ORM texture index for the map paths given, or ``None``.

        A channel with no map fills white: glTF multiplies it by the factor,
        which then carries that channel's value. The packing is cached per
        session by its sources, so materials sharing maps share one image.
        """
        import io

        from pythontk.core_utils.engines.textures.map_factory import MapFactory

        if not (rough or metal):
            return None
        key = "orm|" + "|".join(str(s) for s in (occlusion, rough, metal))
        if key in edit.embedded:
            return edit.embedded[key]
        try:
            image = MapFactory.pack_orm_texture(
                occlusion, rough, metal, save=False, fill_values={"R": 255, "G": 255, "B": 255}
            )
            buffer = io.BytesIO()
            image.save(
                buffer, format="PNG", compress_level=self._mc.INTERMEDIATE_PNG_LEVEL
            )
        except Exception as error:  # noqa: BLE001 -- a bad map must not cost the material
            logger.warning("Materials: the ORM of %s could not be packed: %s", name, error)
            return None
        return self._mc._images._embed_image_bytes(
            edit, key, buffer.getvalue(), "image/png", name=f"orm_{name}"
        )

    # ---------------------------------------------------------------- values
    @staticmethod
    def _same_file(label: Any, name: str) -> bool:
        """Whether an image *label* (a glTF image name) names file *name*."""
        return MaterialFamilies.file_name(label).casefold() == name.casefold()
