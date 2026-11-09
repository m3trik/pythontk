# !/usr/bin/python
# coding=utf-8
"""The ``materials`` record: its payload, declared once, and the one model that
states a material in it.

A deliverable states its materials as LITERAL glTF 2.0 (:class:`MaterialRecord`):
a glTF document holding only ``materials``, ``textures``, ``images`` and
``samplers``, with the ``asset`` and ``extensionsUsed`` glTF requires and the
record's ``version`` beside them. Metal and roughness are PACKED into one
image, glTF's own layout (roughness G, metal B, occlusion R when packed with
them); a UV transform rides ``KHR_texture_transform``, an unlit surface
``KHR_materials_unlit``, and emission above 1 ``KHR_materials_emissive_strength``
(``emissiveFactor`` is capped at 1). Images are named by file name (``uri``),
the files travelling with the delivery. So the Khronos glTF Validator checks
the record as it stands, any glTF reader reads it, and the GLB build copies
it (``MeshConvert.apply_glb_materials``) instead of translating it: an FBX
carries the record by name on its carrier, each material's ``name`` the FBX
material it describes, and a GLB's materials ARE its native glTF materials,
built from the same record (one layout, two containers).

Version 1 (2026-10-20) stated the same values flat, with separate metal and
roughness maps and emission above 1 in its factor, which glTF forbids; every
reader refuses it, and the cure is a re-export.

:class:`MaterialFamilies` is the model every statement comes from: plain values
in, a STATEMENT out (:meth:`MaterialFamilies.entry`: a material's glTF values
flat, its maps by path or file name), and the statements built into the
record (:meth:`MaterialFamilies.payload`), packing each material's metal and
roughness maps into one image through a packer the caller hands it -- the
producer's writes the packed image into the pack store, a delivery file the
export embeds with its other maps and never a file in a source folder
(:meth:`MaterialFamilies.pack_metallic_roughness`), the GLB build's into its
own scratch folder. The hosts' export producers read the live scene into it
(mayatk / blendertk ``MaterialRecords``), and the FBX reader reads a file
that carries no record into the same families (``MeshConvert.fbx_materials``)
-- one set of rules, so a file read back states what its producer did.

The record is a typed :class:`~pythontk.SchemaSpec`: this one definition
validates a payload (``MaterialRecord.validate(payload)``), publishes its JSON
Schema, and is what ``m3trik/scripts/sync_scene_records.py`` generates
unitytk's C# record types from. Change a field together with
:class:`MaterialFamilies` and bump ``SceneRecords.MATERIALS.version``.

Adopted, not built (CODE_STANDARD section 15): glTF 2.0's own material model
and its ratified extensions, rather than a vocabulary of ours; the packing is
``MapFactory.pack_orm_texture``. Weighed and left: keeping the flat shape (the
validator cannot read it, and every reader translated it its own way).
"""

import hashlib
import math
import os
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

from pythontk.core_utils.schema_spec import SchemaSpec

_field = SchemaSpec.spec_field

#: glTF 2.0's alpha modes, the record's vocabulary for a material's opacity.
ALPHA_MODES = ("OPAQUE", "MASK", "BLEND")

#: glTF's sampler vocabulary (WebGL's enums): filters and wraps.
MAG_FILTERS = (9728, 9729)
MIN_FILTERS = (9728, 9729, 9984, 9985, 9986, 9987)
WRAP_MODES = (33071, 33648, 10497)


# ---------------------------------------------------------------- the shape
@dataclass
class MaterialTextureTransform(SchemaSpec):
    """``KHR_texture_transform``: how a map's UVs are offset and scaled, in
    glTF's UV space (origin top left)."""

    TYPED = True

    offset: Tuple[float, float] = _field(
        help="The UV offset, applied after the scale; [0, 0] is none."
    )
    scale: Tuple[float, float] = _field(help="The UV scale; [1, 1] is none.")


@dataclass
class MaterialTextureExtensions(SchemaSpec):
    """The extensions a map's binding carries."""

    TYPED = True

    KHR_texture_transform: MaterialTextureTransform = _field(
        help="The map's UV transform; absent when it has none."
    )


@dataclass
class MaterialTextureInfo(SchemaSpec):
    """A map's binding: the texture it samples, through UV set 0."""

    TYPED = True

    index: int = _field(help="The texture, by its index in 'textures'.", required=True)
    extensions: MaterialTextureExtensions = _field(
        help="KHR_texture_transform, when the map's UVs are transformed."
    )


@dataclass
class MaterialNormalTextureInfo(SchemaSpec):
    """The normal map's binding: tangent space, +Y up (glTF's convention)."""

    TYPED = True

    index: int = _field(help="The texture, by its index in 'textures'.", required=True)
    scale: float = _field(
        help="Scales the normal map's X and Y; absent (1) is the map as authored."
    )
    extensions: MaterialTextureExtensions = _field(
        help="KHR_texture_transform, when the map's UVs are transformed."
    )


@dataclass
class MaterialOcclusionTextureInfo(SchemaSpec):
    """The occlusion map's binding, read from R: it darkens only the
    environment's light. A lightmap holds its object's occlusion, so the
    producer exports a material only lightmapped objects wear without one; a
    material that lightmapped and unbaked objects share keeps it, the export
    warned with the fix: no reader rule applies."""

    TYPED = True

    index: int = _field(
        help=(
            "The texture, by its index in 'textures': the packed "
            "metallicRoughness image when occlusion is packed with them."
        ),
        required=True,
    )
    extensions: MaterialTextureExtensions = _field(
        help="KHR_texture_transform, when the map's UVs are transformed."
    )


@dataclass
class MaterialPbr(SchemaSpec):
    """glTF's metallic-roughness model: each factor linear and multiplied by
    its map."""

    TYPED = True

    baseColorFactor: Tuple[float, float, float, float] = _field(
        help="Linear RGBA, each 0-1; A is the opacity.", required=True
    )
    baseColorTexture: MaterialTextureInfo = _field(
        help="The colour map (sRGB); its alpha is the opacity."
    )
    metallicFactor: float = _field(help="0 dielectric to 1 metal.", required=True)
    roughnessFactor: float = _field(help="0 mirror to 1 matte.", required=True)
    metallicRoughnessTexture: MaterialTextureInfo = _field(
        help=(
            "The packed map (linear): roughness in G, metal in B -- and the "
            "occlusion in R where the material's occlusion map is packed with "
            "them. A channel the material does not map is white, so its factor "
            "carries it."
        )
    )


@dataclass
class MaterialEmissiveStrength(SchemaSpec):
    """``KHR_materials_emissive_strength``: emission above 1."""

    TYPED = True

    emissiveStrength: float = _field(
        help="Multiplies emissiveFactor (whose peak is then 1); above 1.",
        required=True,
    )


@dataclass
class MaterialUnlit(SchemaSpec):
    """``KHR_materials_unlit``: an empty object."""

    TYPED = True


@dataclass
class MaterialExtensions(SchemaSpec):
    """The extensions a material carries."""

    TYPED = True

    KHR_materials_unlit: MaterialUnlit = _field(
        help=(
            "Shows its base colour and opacity alone: no light, reflection or "
            "emission (metallic 0 and roughness 1 for a reader that lights it "
            "anyway)."
        )
    )
    KHR_materials_emissive_strength: MaterialEmissiveStrength = _field(
        help="Emission above 1, carried as the strength."
    )


@dataclass
class MaterialEntry(SchemaSpec):
    """One material, as a glTF 2.0 ``material``."""

    TYPED = True

    name: str = _field(
        help=(
            "The material's name as the file names it, namespace included "
            "('ns:name'): the join key to the imported material (the FBX "
            "material this describes; in a GLB, the material itself). A reader "
            "whose import drops the namespace matches the name after the last "
            "':'."
        ),
        required=True,
    )
    pbrMetallicRoughness: MaterialPbr = _field(
        help="Its metallic-roughness values.", required=True
    )
    normalTexture: MaterialNormalTextureInfo = _field(
        help="The tangent-space normal map (linear)."
    )
    occlusionTexture: MaterialOcclusionTextureInfo = _field(
        help="The ambient-occlusion map (linear), read from R."
    )
    emissiveTexture: MaterialTextureInfo = _field(help="The emission map (sRGB).")
    emissiveFactor: Tuple[float, float, float] = _field(
        help=(
            "Linear RGB emission, each 0-1: emission above 1 is this times "
            "KHR_materials_emissive_strength."
        ),
        required=True,
    )
    alphaMode: str = _field(
        help=(
            "OPAQUE ignores the opacity; MASK cuts at alphaCutoff; BLEND blends "
            "(sorted, no depth write)."
        ),
        required=True,
        choices=ALPHA_MODES,
    )
    alphaCutoff: float = _field(help="MASK only: the opacity below which a pixel is cut.")
    doubleSided: bool = _field(help="Both faces render.", required=True)
    extensions: MaterialExtensions = _field(
        help="KHR_materials_unlit and KHR_materials_emissive_strength, when used."
    )


@dataclass
class MaterialTexture(SchemaSpec):
    """A glTF ``texture``: an image and how it is sampled."""

    TYPED = True

    source: int = _field(help="The image, by its index in 'images'.", required=True)
    sampler: int = _field(
        help="The sampler, by its index in 'samplers'; absent: repeat, auto filtering."
    )


@dataclass
class MaterialImage(SchemaSpec):
    """A glTF ``image``, named by the file the delivery carries."""

    TYPED = True

    uri: str = _field(
        help=(
            "The image's file name (no folder): the file travels with the "
            "delivery (embedded in an FBX under this name; in a GLB, in its "
            "buffer)."
        ),
        required=True,
    )


@dataclass
class MaterialSampler(SchemaSpec):
    """A glTF ``sampler``."""

    TYPED = True

    magFilter: int = _field(help="NEAREST 9728 or LINEAR 9729.", choices=MAG_FILTERS)
    minFilter: int = _field(help="A WebGL minification filter.", choices=MIN_FILTERS)
    wrapS: int = _field(
        help="CLAMP_TO_EDGE 33071, MIRRORED_REPEAT 33648 or REPEAT 10497.",
        choices=WRAP_MODES,
    )
    wrapT: int = _field(help="As wrapS, for V.", choices=WRAP_MODES)


@dataclass
class MaterialAsset(SchemaSpec):
    """glTF's ``asset``: the version of glTF the record is."""

    TYPED = True

    version: str = _field(help="The glTF version: '2.0'.", required=True)


@dataclass
class MaterialRecord(SchemaSpec):
    """The ``materials`` record's payload: a glTF 2.0 document holding every
    material the file's meshes use, and the record's version."""

    TYPED = True

    version: int = _field(
        help=(
            "The record's schema version (SceneRecords.MATERIALS.version): 2. "
            "Not glTF's: a reader reading the record as a glTF document leaves "
            "it out. A reader refuses 1 (the flat shape); the cure is a "
            "re-export."
        ),
        required=True,
    )
    asset: MaterialAsset = _field(help="glTF's asset: version '2.0'.", required=True)
    extensionsUsed: List[str] = _field(
        help="The glTF extensions the materials and textures use; absent when none."
    )
    materials: List[MaterialEntry] = _field(
        help="Every material, in the file's order.",
        required=True,
        nested=[MaterialEntry],
        default_factory=list,
    )
    textures: List[MaterialTexture] = _field(
        help="The textures the materials sample; absent when none.",
        nested=[MaterialTexture],
    )
    images: List[MaterialImage] = _field(
        help="The images the textures read, by file name; absent when none.",
        nested=[MaterialImage],
    )
    samplers: List[MaterialSampler] = _field(
        help="The samplers the textures name; absent when every one repeats.",
        nested=[MaterialSampler],
    )


#: What a packer is handed and returns: ``(occlusion, roughness, metallic)``
#: map paths (``None`` where the material maps none), and a keyword ``name``
#: (the file name to write under, ``None`` for its own) when the record
#: already names another pack the one it would choose -> the packed image's
#: path or file name, or ``None`` when its maps are not to hand (the GLB
#: build's: a map the FBX does not carry). A pack that cannot be WRITTEN
#: raises ``OSError``: the producer's export fails, saying why.
Packer = Callable[..., Optional[str]]


# ---------------------------------------------------------------- the model
class MaterialFamilies:
    """A material, read in its shading family's own terms, as the record states it.

    The host-free model under every statement of a material (module
    docstring). A family is a shading model's vocabulary and the rules that
    turn it into glTF's: :meth:`entry` takes the family's name, the
    material's values in that vocabulary (``props``) and the images bound to
    it by logical channel (``maps``: :attr:`MAP_CHANNELS`, the slot names of
    mayatk's ``ShaderAttributeMap`` and both hosts' ``MatManifest``), and
    returns the material's STATEMENT: its glTF values flat (each map by the
    path or name it was handed, metal and roughness apart, emission whole).
    :meth:`payload` builds statements into the record. A new family is one
    :attr:`FAMILIES` row and its resolver; a new host reads into the families
    that exist.

    The vocabularies:

    - ``pbr`` -- a metallic-roughness surface in logical terms (Standard
      Surface, aiStandardSurface, openPBR, Blender's Principled):
      ``base_color``, ``base_weight``, ``metallic``, ``roughness``,
      ``emission_color``, ``emission_weight``, ``opacity`` (its scalar part,
      or the opacity colour: :meth:`opacity_scalar`), ``opacity_driven``,
      ``transmission`` (its weight: the record states none, and says so), and
      optionally ``alpha_mode`` / ``alpha_cutoff``, ``normal_scale`` and
      ``uv_offset`` / ``uv_scale`` (the maps' UV transform);
    - ``stingray_pbs`` -- a StingrayPBS by its own attribute names
      (``use_color_map``, ``base_color``, ``emissive_intensity``, the opacity
      graph's ``opacity`` or ``mask_threshold``, ``uv_offset``,
      ``uv_scale``, ...);
    - ``legacy`` -- FBX's own models by FBX's property names (``DiffuseColor``,
      ``DiffuseFactor``, ``TransparentColor``, ``TransparencyFactor``,
      ``EmissiveColor``, ``EmissiveFactor``, ``Shininess``) and
      ``ShadingModel``: how a Phong, Blinn or Lambert reaches every engine;
    - ``unlit`` -- a host's unlit model: ``color``, ``opacity``,
      ``opacity_driven``.

    A UV transform (``uv_offset`` / ``uv_scale``) is in the host's UV space,
    origin bottom left, as Maya's and Blender's are: the model turns it into
    glTF's (:meth:`gltf_transform`).
    """

    #: The image channels a reader hands :meth:`entry`, by logical name.
    MAP_CHANNELS: Tuple[str, ...] = (
        "baseColor",
        "metallic",
        "roughness",
        "ambientOcclusion",
        "normal",
        "emission",
        "opacity",
    )

    #: Family -> its resolver, ``(props, maps) -> statement``.
    FAMILIES: Dict[str, str] = {
        "pbr": "_pbr",
        "stingray_pbs": "_stingray_pbs",
        "legacy": "_legacy",
        "unlit": "_unlit",
    }

    #: Family -> the values it reads, by the names a reader hands them under
    #: (the class docstring says what each means). A StingrayPBS's are its
    #: graph's own attribute names, so a host reads exactly these off the
    #: node and the FBX carries the same as ``Maya|<name>``.
    PROPS: Dict[str, Tuple[str, ...]] = {
        "pbr": (
            "base_color",
            "base_weight",
            "metallic",
            "roughness",
            "emission_color",
            "emission_weight",
            "opacity",
            "opacity_driven",
            "transmission",
            "alpha_mode",
            "alpha_cutoff",
            "normal_scale",
            "double_sided",
            "uv_offset",
            "uv_scale",
        ),
        "stingray_pbs": (
            "use_color_map",
            "base_color",
            "use_metallic_map",
            "metallic",
            "use_roughness_map",
            "roughness",
            "use_normal_map",
            "use_emissive_map",
            "emissive",
            "emissive_intensity",
            "use_ao_map",
            "use_opacity_map",
            "opacity",
            "mask_threshold",
            "uv_offset",
            "uv_scale",
        ),
        "legacy": (
            "ShadingModel",
            "DiffuseColor",
            "DiffuseFactor",
            "TransparentColor",
            "TransparencyFactor",
            "EmissiveColor",
            "EmissiveFactor",
            "Shininess",
        ),
        "unlit": ("color", "opacity", "opacity_driven"),
    }

    #: The conversion of FBX's legacy models: FBX2glTF's (Phong and Blinn
    #: metallic 0.4, roughness sqrt(2 / (2 + shininess)); every other model
    #: 0.2 and 0.8). Its constants are arbitrary but they are the look every
    #: approved WebXR deliverable was signed off through, so the record keeps
    #: them and Unity now shows the same; named so a change is a decision
    #: (and a record version), never a drift.
    LEGACY_POLICY = "fbx2gltf"
    LEGACY_SPECULAR_MODELS = frozenset({"phong", "blinn"})

    #: What a transmissive (glass) surface's entry says: the record carries no
    #: transmission, so every reader shows it as its opacity alone.
    TRANSMISSION_NOTE = (
        "its transmission is not stated (the record has no transmission); it "
        "reads as its opacity alone."
    )

    #: The statement fields that name a map, in the order a reader binds them.
    TEXTURE_FIELDS: Tuple[str, ...] = (
        "baseColorTexture",
        "metallicTexture",
        "roughnessTexture",
        "occlusionTexture",
        "normalTexture",
        "emissiveTexture",
    )

    #: Where a record material binds a map: ``(container, slot)``, the
    #: container ``None`` for the material itself.
    TEXTURE_SLOTS: Tuple[Tuple[Optional[str], str], ...] = (
        ("pbrMetallicRoughness", "baseColorTexture"),
        ("pbrMetallicRoughness", "metallicRoughnessTexture"),
        (None, "normalTexture"),
        (None, "occlusionTexture"),
        (None, "emissiveTexture"),
    )

    #: The glTF version the record is.
    GLTF_VERSION = "2.0"
    #: The extensions the record may use, by what states them.
    UNLIT_EXTENSION = "KHR_materials_unlit"
    STRENGTH_EXTENSION = "KHR_materials_emissive_strength"
    TRANSFORM_EXTENSION = "KHR_texture_transform"

    #: The packed image's name: its sources' base name and this suffix -- the
    #: registry's name for glTF's layout (``MapRegistry`` reads it as ORM).
    PACKED_SUFFIX = "_ORM"
    #: The PNG text key a packed image states the content hash of its
    #: sources under (:meth:`pack_metallic_roughness`): what a re-export
    #: compares before it reuses the pack.
    PACKED_FROM_KEY = "glTF metallicRoughness from"
    #: The pack store: the ``TempArtifacts`` prefix packed images live under,
    #: one folder per content hash, swept by age (:attr:`PACK_STORE_DAYS`).
    #: A packed image is a delivery file (CODE_STANDARD sections 10 and 13):
    #: the export embeds it, so nothing keeps it but the next export's reuse.
    PACK_STORE = "material_packs"
    PACK_STORE_DAYS = 7
    #: What a pack is made by, in its content hash: a change to the packing
    #: (its fills, its layout) is a new hash, so no stale pack is reused.
    PACK_FORMAT = "orm/1 fill=255"
    #: ``{file name, casefolded: path}`` of the packs the latest write wrote
    #: or reused in the pack store (:meth:`export_record` with a packer
    #: starts it afresh, so one scene's pack never answers for another's
    #: same-named file): how the delivery's dependency pass and the GLB
    #: build find a pack the record names by file name (:meth:`packed_file`).
    #: A memo of files on disk, never their content.
    _packed: Dict[str, str] = {}

    # ------------------------------------------------------------- statement
    @classmethod
    def entry(
        cls,
        family: str,
        name: str,
        props: Optional[Mapping[str, Any]] = None,
        maps: Optional[Mapping[str, Any]] = None,
        *,
        unlit: bool = False,
        notes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """*name*'s statement, from its *family*'s values.

        Parameters:
            family: A :attr:`FAMILIES` key.
            name: The material's name as the file names it.
            props: Its values in the family's vocabulary (class docstring);
                a value the host does not carry is left out.
            maps: ``{logical channel: image}`` (:attr:`MAP_CHANNELS`) -- a
                path or a file name. The statement names each by file name,
                as the record does, and keeps a path in ``sources`` (``{file
                name: path}``), where a packer reads it.
            unlit: The scene shades it unlit whatever it wears (a projected
                shadow's plane): its base colour and opacity alone.
            notes: Where a sentence about what the record cannot state goes
                (a glass's transmission); dropped when ``None``.

        Raises:
            KeyError: *family* is not a :attr:`FAMILIES` key.
        """
        method = cls.FAMILIES.get(family)
        if method is None:
            raise KeyError(
                f"{family!r} is no material family (known: {', '.join(cls.FAMILIES)})."
            )
        given = {
            channel: str(image)
            for channel, image in (maps or {}).items()
            if channel in cls.MAP_CHANNELS and image
        }
        files = {channel: cls.file_name(image) for channel, image in given.items()}
        found = getattr(cls, method)(props or {}, files)
        found["name"] = name
        sources = {
            cls.file_name(image): image
            for image in given.values()
            if cls.file_name(image) != image
        }
        if sources:
            found["sources"] = sources
        if unlit:
            found["unlit"] = True
        offset, scale = (props or {}).get("uv_offset"), (props or {}).get("uv_scale")
        if offset is not None or scale is not None:
            found["uvTransform"] = {
                "offset": cls.as_floats(offset, 2, 0.0),
                "scale": cls.as_floats(scale, 2, 1.0),
            }
        for sentence in found.pop("_notes", ()):
            if notes is not None:
                notes.append(f"Material {name}: {sentence}")
        return cls.finish(found)

    @classmethod
    def finish(cls, entry: Mapping[str, Any]) -> Dict[str, Any]:
        """*entry* (a statement) normalized: every factor present and in range
        (emission non-negative, unbounded: the record carries what is above 1
        as its strength), a map or an optional value only where it says
        something. Unlit means metallic 0 and roughness 1 (glTF's fallback
        for a reader that lights it anyway)."""
        base = cls.as_floats(entry.get("baseColorFactor"), 4, 1.0)
        mode = (
            entry.get("alphaMode")
            if entry.get("alphaMode") in ALPHA_MODES
            else "OPAQUE"
        )
        unlit = bool(entry.get("unlit"))
        out: Dict[str, Any] = {
            "name": entry.get("name"),
            "baseColorFactor": [cls._unit(c) for c in base],
            "metallicFactor": 0.0
            if unlit
            else cls._unit(cls.as_float(entry.get("metallicFactor"), 1.0)),
            "roughnessFactor": 1.0
            if unlit
            else cls._unit(cls.as_float(entry.get("roughnessFactor"), 1.0)),
            "emissiveFactor": [
                max(0.0, c) for c in cls.as_floats(entry.get("emissiveFactor"), 3, 0.0)
            ],
            "alphaMode": mode,
        }
        for field in cls.TEXTURE_FIELDS:
            if entry.get(field):
                out[field] = str(entry[field])
        if not any(out["emissiveFactor"]):
            out.pop("emissiveTexture", None)
        sources = entry.get("sources")
        if isinstance(sources, Mapping):
            named = {out[f] for f in cls.TEXTURE_FIELDS if out.get(f)}
            kept = {k: str(v) for k, v in sources.items() if k in named and v}
            if kept:
                out["sources"] = kept
        scale = cls.as_float(entry.get("normalScale"), 1.0)
        if out.get("normalTexture") and scale != 1.0:
            out["normalScale"] = scale
        if mode == "MASK":
            out["alphaCutoff"] = cls._unit(cls.as_float(entry.get("alphaCutoff"), 0.5))
        if entry.get("doubleSided"):
            out["doubleSided"] = True
        if unlit:
            out["unlit"] = True
        transform = entry.get("uvTransform")
        if isinstance(transform, Mapping):
            offset = cls.as_floats(transform.get("offset"), 2, 0.0)
            scale_uv = cls.as_floats(transform.get("scale"), 2, 1.0)
            if offset != [0.0, 0.0] or scale_uv != [1.0, 1.0]:
                out["uvTransform"] = {"offset": offset, "scale": scale_uv}
        return out

    # ---------------------------------------------------------------- record
    @classmethod
    def payload(
        cls,
        entries: Iterable[Mapping[str, Any]],
        *,
        pack: Optional[Packer] = None,
        notes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """The record's payload for *entries* (statements; the first of a
        repeated name kept), version stamped: each a glTF material, its maps
        named by file name, its metal and roughness maps packed into one
        image (with its occlusion map, when it has one).

        Parameters:
            entries: Statements (:meth:`entry`).
            pack: Packs a material's ``(occlusion, roughness, metallic)``
                maps into one image and returns its path or name
                (:meth:`pack_metallic_roughness` writes it into the pack store).
                ``None`` writes nothing: the record names the image a pack
                would write (:meth:`packed_name`), for a reader asking what a
                record would hold. A pack that fails (returns ``None``)
                leaves the material its metal and roughness factors over
                their maps' absent values, said in *notes*.
            notes: Where a sentence about what the record cannot state goes.
        """
        return _Document(pack, notes).add_statements(entries).payload()

    @classmethod
    def without_baked_occlusion(
        cls,
        entries: Iterable[Mapping[str, Any]],
        baked: Iterable[str],
        unbaked: Iterable[str],
        *,
        notes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """*entries* as a lightmapped scene states them: the occlusion map
        left out of each material only lightmapped objects wear.

        A lightmap is its object's whole diffuse light, its occlusion
        included (``SceneRecords.LIGHTMAPS``), so the material's own
        occlusion map would darken a lightmapped object twice. A material
        that lightmapped and unbaked objects share keeps it -- the unbaked
        ones need it -- and is said once in *notes*, with the fix: give the
        lightmapped objects their own material. The rule
        :meth:`export_record` applies before :meth:`payload`; a reader has
        no rule of its own for it.

        Parameters:
            entries: Statements (:meth:`entry`).
            baked: The names of the materials lightmapped objects wear.
            unbaked: The names of the materials the write's other objects wear.
            notes: Where the sentence about a shared material goes.

        Returns:
            The statements, each a copy; *entries* are left as they were.
        """
        baked, unbaked = set(baked), set(unbaked)
        out: List[Dict[str, Any]] = []
        for statement in entries:
            statement = dict(statement)
            name = statement.get("name")
            occlusion = statement.get("occlusionTexture")
            if occlusion and name in baked:
                if name not in unbaked:
                    del statement["occlusionTexture"]
                elif notes is not None:
                    notes.append(
                        f"Material {name}: lightmapped and unbaked objects both "
                        f"wear it, so it keeps its occlusion map "
                        f"{cls.file_name(occlusion)}, which darkens the "
                        "lightmapped ones twice (their lightmap holds their "
                        "occlusion); give the lightmapped objects their own "
                        "material."
                    )
            out.append(statement)
        return out

    @classmethod
    def export_record(
        cls,
        ctx: Any,
        entries: Iterable[Mapping[str, Any]],
        worn: Callable[[], Iterable[Tuple[Iterable[Sequence[str]], Iterable[str]]]],
        *,
        store: Any = None,
        pack: Optional[Packer] = None,
        notes: Optional[List[str]] = None,
    ) -> Optional[Any]:
        """The ``materials`` record a write ships, from what its host read:
        the producer both hosts' ``MaterialRecords.export_record`` are, each
        gathering only what needs its host.

        The occlusion rule (:meth:`without_baked_occlusion`) is decided
        against the ``lightmaps`` record THIS write ships, never the baker's
        markers: the one its assembly produced first (the materials record
        is declared ``after`` it) or, shipped as stored (a hand-off), the
        stored copy -- narrowed to the nodes the write ships, and none the
        write withholds (``ExportContext.record``) -- so a refused or absent
        record leaves every material its occlusion. A mesh SHAPE is
        lightmapped when a transform it is instanced under is one of that
        record's objects (by ``hierarchy``, else by name), and the materials
        it wears, per-face assignment included, with it. Then the record is
        built (:meth:`payload`) and every sentence said on *ctx*.

        Parameters:
            ctx: The write's ``ExportContext``.
            entries: The statements of the materials the write ships
                (:meth:`entry`).
            worn: Called only when a lightmaps record ships and some material
                carries an occlusion map: per mesh shape the write ships, the
                hierarchies of the transforms it is instanced under (each its
                path of names from the scene root, as a ``lightmaps`` entry's
                ``hierarchy`` states one) and the names of the materials it
                wears.
            store: The scene's store, read for a lightmaps record the
                assembly did not produce.
            pack: As :meth:`payload` takes it: the hosts' producers pass
                :meth:`pack_metallic_roughness`.
            notes: The host's sentences so far, said on *ctx* with the rest.

        Returns:
            The ``materials`` ``Record``, or ``None`` when nothing is stated.
        """
        from pythontk.core_utils.engines.scene_export.scene_records import (
            Record,
            SceneRecords,
        )

        if pack is not None:
            # A write's packs are its own: the memo the delivery reads a
            # record's packed image off (:meth:`packed_file`) starts afresh,
            # so another scene's same-named pack, from an earlier write of
            # this session, never rides into this one.
            cls._packed.clear()
        notes = list(notes or [])
        entries = list(entries)
        baked, unbaked = cls._lightmapped(ctx, entries, worn, store)
        if baked:
            entries = cls.without_baked_occlusion(entries, baked, unbaked, notes=notes)
        payload = cls.payload(entries, pack=pack, notes=notes) if entries else None
        for note in dict.fromkeys(notes):
            ctx.note(note)
        return Record(SceneRecords.MATERIALS, payload) if payload else None

    @classmethod
    def _lightmapped(
        cls, ctx: Any, entries: List[Mapping[str, Any]], worn: Callable, store: Any
    ) -> Tuple[set, set]:
        """``(baked, unbaked)``: the names of the materials the write's
        lightmapped shapes wear and of those the rest wear
        (:meth:`export_record`); both empty when no lightmaps record ships or
        no material carries an occlusion map, *worn* then never asked."""
        from pythontk.core_utils.engines.scene_export.scene_records import (
            SceneRecords,
        )

        if not any(entry.get("occlusionTexture") for entry in entries):
            return set(), set()
        spec = SceneRecords.LIGHTMAPS
        shipped = ctx.narrowed(spec, ctx.record(spec, store=store))
        if not spec.is_current(shipped):
            return set(), set()
        paths, names = set(), set()
        for obj in shipped.get("objects") or ():
            path = tuple(obj.get("hierarchy") or ())
            if path:
                paths.add(path)
            elif obj.get("name"):
                names.add(obj["name"])
        baked: set = set()
        unbaked: set = set()
        for transforms, materials in worn():
            lit = any(
                tuple(path) in paths or (path and path[-1] in names)
                for path in transforms
            )
            (baked if lit else unbaked).update(materials)
        return baked, unbaked

    @classmethod
    def overlay(
        cls,
        record: Optional[Mapping[str, Any]],
        entries: Iterable[Mapping[str, Any]],
        *,
        pack: Optional[Packer] = None,
        notes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """*entries*' record, each material *record* names taken from it
        instead (copied whole, its maps with it): the file's own materials
        under what its carried record states. *record* must be one a reader
        reads (``SceneRecords.problems`` empty); ``None`` states nothing.
        :meth:`payload`'s *pack* and *notes* build the rest."""
        named = cls.materials_by_name(record)
        document = _Document(pack, notes)
        seen = set()
        for statement in entries:
            name = statement.get("name") if isinstance(statement, Mapping) else None
            if not name or name in seen:
                continue
            seen.add(name)
            if name in named:
                document.add_material(named[name], record)
            else:
                document.add_statement(statement)
        return document.payload()

    @classmethod
    def materials_by_name(
        cls, record: Optional[Mapping[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """``{name: material}`` of a record; the first of a repeated name."""
        found: Dict[str, Dict[str, Any]] = {}
        for material in (record or {}).get("materials") or ():
            if isinstance(material, Mapping) and material.get("name"):
                found.setdefault(str(material["name"]), dict(material))
        return found

    @classmethod
    def texture_files(cls, record: Optional[Mapping[str, Any]]) -> List[str]:
        """The file names *record*'s images name, each once, in record order."""
        names: List[str] = []
        for image in (record or {}).get("images") or ():
            uri = image.get("uri") if isinstance(image, Mapping) else None
            if isinstance(uri, str) and uri and uri not in names:
                names.append(uri)
        return names

    @classmethod
    def source_files(cls, entries: Iterable[Mapping[str, Any]]) -> List[str]:
        """The file names the maps of *entries* (statements) are, each once,
        in statement order: the scene's own maps, before a record packs any
        -- what a check of the authored maps reads (a template's mask
        packing), where :meth:`texture_files` of the record names the packed
        image glTF's layout makes of them."""
        names: List[str] = []
        for statement in entries:
            if not isinstance(statement, Mapping):
                continue
            for field in cls.TEXTURE_FIELDS:
                value = statement.get(field)
                name = cls.file_name(value) if value else None
                if name and name not in names:
                    names.append(name)
        return names

    @classmethod
    def material_files(
        cls, record: Optional[Mapping[str, Any]], material: Mapping[str, Any]
    ) -> Dict[str, str]:
        """``{slot: file name}`` of the maps *material* (one of *record*'s)
        binds, by its glTF slot names (:attr:`TEXTURE_SLOTS`)."""
        return {
            slot: name
            for _holder, slot, name, _sampler in cls.bindings(record, material)
            if name
        }

    @classmethod
    def bindings(
        cls, record: Optional[Mapping[str, Any]], material: Mapping[str, Any]
    ) -> Iterator[Tuple[Any, str, Optional[str], Optional[Dict[str, Any]]]]:
        """Each map *material* (one of *record*'s) binds, in
        :attr:`TEXTURE_SLOTS` order: ``(holder, slot, file name, sampler)`` --
        the mapping holding the slot's texture info (the material, or its
        ``pbrMetallicRoughness``; the caller's own object, so it may rebind
        or drop the slot), the slot, the file name its texture reads
        (``None`` where *record* holds no such texture or image) and a copy
        of the sampler that texture names (``None``: glTF's default). The one
        walk of a record material's maps."""
        textures = (record or {}).get("textures") or []
        samplers = (record or {}).get("samplers") or []
        for container, slot in cls.TEXTURE_SLOTS:
            holder = material.get(container) if container else material
            if not isinstance(holder, Mapping) or not isinstance(holder.get(slot), Mapping):
                continue
            index = holder[slot].get("index")
            name = cls.texture_file(record, index)
            sampler = (textures[index] or {}).get("sampler") if name else None
            spec = (
                dict(samplers[sampler])
                if isinstance(sampler, int) and 0 <= sampler < len(samplers)
                else None
            )
            yield holder, slot, name, spec

    @classmethod
    def extensions_of(cls, material: Mapping[str, Any]) -> List[str]:
        """The glTF extensions *material* uses, its own and its maps', each
        once, in the order met: what a document's ``extensionsUsed`` states."""
        found = list((material.get("extensions") or {}).keys())
        for container, slot in cls.TEXTURE_SLOTS:
            holder = material.get(container) if container else material
            info = holder.get(slot) if isinstance(holder, Mapping) else None
            if isinstance(info, Mapping):
                found.extend((info.get("extensions") or {}).keys())
        return list(dict.fromkeys(found))

    @classmethod
    def absent_map(cls, material: MutableMapping[str, Any], slot: str) -> None:
        """Take the map in *slot* off *material* (a glTF material, changed in
        place), read as ABSENT -- the one rule every reader of a map the
        record names and cannot find applies (the GLB build, and unitytk's
        ``MaterialRecord`` reader, which mirrors it):

        - ``metallicRoughnessTexture``: the metal factor takes its absent
          value (dielectric) only where it is 1 or more -- the map supplied
          it; a constant below 1 is the material's own. The roughness factor
          is kept: a constant is the material's, and 1 is matte, the absent
          value itself.
        - ``emissiveTexture``: no emission -- the factor would light the
          surface the map's colour never reaches (a missing glow map read
          white); its strength goes with it.
        - any other slot: the map alone (a colour its factor, a flat
          normal, no occlusion).
        """
        container = dict((s, c) for c, s in cls.TEXTURE_SLOTS).get(slot)
        holder = material.get(container) if container else material
        if isinstance(holder, MutableMapping):
            holder.pop(slot, None)
        if slot == "metallicRoughnessTexture" and isinstance(holder, MutableMapping):
            if cls.as_float(holder.get("metallicFactor"), 1.0) >= 1.0:
                from pythontk.core_utils.engines.textures.map_registry import (
                    MapRegistry,
                )

                holder["metallicFactor"] = MapRegistry.absent_value("Metallic")
        elif slot == "emissiveTexture":
            material["emissiveFactor"] = [0.0, 0.0, 0.0]
            extensions = material.get("extensions")
            if isinstance(extensions, MutableMapping):
                extensions.pop(cls.STRENGTH_EXTENSION, None)
                if not extensions:
                    material.pop("extensions")

    @classmethod
    def texture_file(
        cls, record: Optional[Mapping[str, Any]], index: Any
    ) -> Optional[str]:
        """The file name texture *index* of *record* reads, or ``None``."""
        textures = (record or {}).get("textures") or []
        images = (record or {}).get("images") or []
        if not isinstance(index, int) or not 0 <= index < len(textures):
            return None
        source = (textures[index] or {}).get("source")
        if not isinstance(source, int) or not 0 <= source < len(images):
            return None
        uri = (images[source] or {}).get("uri")
        return uri if isinstance(uri, str) and uri else None

    @classmethod
    def emission(cls, material: Mapping[str, Any]) -> List[float]:
        """*material*'s emission, whole: ``emissiveFactor`` times its
        ``KHR_materials_emissive_strength`` (1 without one)."""
        factor = cls.as_floats(material.get("emissiveFactor"), 3, 0.0)
        strength = (
            ((material.get("extensions") or {}).get(cls.STRENGTH_EXTENSION) or {})
        ).get("emissiveStrength", 1.0)
        return [c * cls.as_float(strength, 1.0) for c in factor]

    # --------------------------------------------------------------- packing
    @classmethod
    def packed_name(
        cls,
        occlusion: Optional[str],
        roughness: Optional[str],
        metallic: Optional[str],
        *,
        unique: bool = False,
    ) -> Optional[str]:
        """The file name a pack of these maps is written under: their base
        name (``MapRegistry``'s, of the roughness map, else the metal, else the
        occlusion) and :attr:`PACKED_SUFFIX`, ``.png`` -- or ``None`` with no
        metal or roughness map to pack. *unique* puts a digest of the maps'
        paths before the suffix: the name a record gives a second pack its
        base name would collide with."""
        if not (roughness or metallic):
            return None
        base = cls._pack_base(occlusion, roughness, metallic)
        if unique:
            key = "|".join(str(p or "") for p in (occlusion, roughness, metallic))
            base = f"{base}_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]}"
        return f"{base}{cls.PACKED_SUFFIX}.png"

    @classmethod
    def pack_metallic_roughness(
        cls,
        occlusion: Optional[str],
        roughness: Optional[str],
        metallic: Optional[str],
        *,
        output_dir: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Optional[str]:
        """Pack the maps into one image in glTF's layout -- occlusion R,
        roughness G, metal B, a channel with no map white (its factor carries
        it) -- and return its path, or ``None`` when there is nothing to pack.

        The packed image is a DELIVERY file: written into the pack store
        (:attr:`PACK_STORE`, a ``TempArtifacts`` folder per content hash of
        its sources), or into *output_dir* (the GLB build's scratch), never
        into a source folder, as a PNG named :meth:`packed_name` (or *name*)
        that states the hash (:attr:`PACKED_FROM_KEY`). The delivery's
        dependency pass embeds it by its file name (:meth:`packed_file`).
        Re-runnable: a pack of the same content is reused as it is, and a
        source whose content changed packs anew. The packing is
        ``MapFactory.pack_orm_texture``'s (a packed source -- an MSAO, an ORM
        -- gives each channel it carries; a 16-bit or float source at its
        value).

        Parameters:
            occlusion: The occlusion map's path, or ``None``.
            roughness: The roughness map's path, or ``None``.
            metallic: The metal map's path, or ``None``.
            output_dir: Where to write; the pack store when ``None``.
            name: The file name to write under; :meth:`packed_name` when
                ``None``.

        Raises:
            FileNotFoundError: A source map is not a file.
            OSError: A source cannot be read or the pack cannot be written;
                the message says which and what to do.
        """
        sources = (occlusion, roughness, metallic)
        if not (roughness or metallic):
            return None
        for path in sources:
            if path and not os.path.isfile(path):
                raise FileNotFoundError(
                    f"The map {cls.file_name(path)} is not at {path}: relink it, or "
                    "remove it from the material, and export again."
                )
        digest = cls._content_hash(sources)
        file_name = name or cls.packed_name(*sources)
        if output_dir:
            folder = output_dir
        else:
            from pythontk.file_utils.temp_artifacts import TempArtifacts

            folder = TempArtifacts(
                cls.PACK_STORE,
                policy="detached",  # read after this call: by the export's embed
                max_age_days=cls.PACK_STORE_DAYS,
                sweep_interval=3600.0,
            ).dir_path(name=digest[:16])
        path = os.path.join(folder, file_name)
        if os.path.isfile(path) and cls._stamp_of(path) == digest:
            try:
                os.utime(path)  # in use: its age starts again
            except OSError:
                pass
        else:
            cls._write_pack(path, sources, digest)
        if not output_dir:
            cls._packed[file_name.casefold()] = path
        return path

    @classmethod
    def packed_file(cls, name: str) -> Optional[str]:
        """The path of the pack the latest write (:meth:`export_record`) wrote
        or reused in the pack store under file name *name*
        (case-insensitively), while it is on disk; ``None`` otherwise. How the delivery's dependency pass
        (``FbxMedia.embed_dependencies``) and the GLB build find the packed
        image a ``materials`` record names, which lives in no scene folder."""
        path = cls._packed.get(cls.file_name(name).casefold())
        return path if path and os.path.isfile(path) else None

    @classmethod
    def _content_hash(cls, sources: Tuple[Optional[str], ...]) -> str:
        """The SHA-256 of *sources*' bytes by role (occlusion, roughness,
        metal) and :attr:`PACK_FORMAT`: what a pack of them is.

        Raises:
            OSError: A source cannot be read.
        """
        digest = hashlib.sha256(cls.PACK_FORMAT.encode("utf-8"))
        for role, path in zip("ORM", sources):
            digest.update(role.encode("ascii"))
            if not path:
                digest.update(b"-")
                continue
            digest.update(b"+")
            with open(path, "rb") as fh:
                for block in iter(lambda: fh.read(1048576), b""):
                    digest.update(block)
        return digest.hexdigest()

    @classmethod
    def _write_pack(cls, path: str, sources: Tuple[Optional[str], ...], stamp: str) -> None:
        """Write the pack of *sources* to *path*, stamped, atomically.

        Raises:
            OSError: A source cannot be read as an image, or *path* cannot be
                written; the message names the file and the fix.
        """
        from PIL import PngImagePlugin

        from pythontk.core_utils.engines.textures.map_factory import MapFactory
        from pythontk.file_utils._file_utils import FileUtils

        try:
            image = MapFactory.pack_orm_texture(
                *sources, save=False, fill_values={"R": 255, "G": 255, "B": 255}
            )
        except Exception as error:  # noqa: BLE001 -- Pillow raises its own kinds
            raise OSError(
                "The maps "
                + ", ".join(cls.file_name(p) for p in sources if p)
                + f" could not be read as images ({error}): save each as a PNG, "
                "TGA, TIFF or JPEG and export again."
            ) from error
        info = PngImagePlugin.PngInfo()
        info.add_text(cls.PACKED_FROM_KEY, stamp)
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            FileUtils.atomic_write(
                path, lambda part: image.save(part, format="PNG", pnginfo=info)
            )
        except OSError as error:
            raise OSError(
                f"The packed metal / roughness image {path} could not be written "
                f"({error}): free space, or point PYTHONTK_TEMP_ROOT at a writable "
                "folder, and export again."
            ) from error

    @classmethod
    def _stamp_of(cls, path: str) -> Optional[str]:
        """The content hash a packed PNG states (:attr:`PACKED_FROM_KEY`), or ``None``."""
        try:
            from PIL import Image

            with Image.open(path) as image:
                return (getattr(image, "text", None) or {}).get(cls.PACKED_FROM_KEY)
        except Exception:  # noqa: BLE001 -- unreadable: not a pack to reuse
            return None

    @classmethod
    def _pack_base(cls, *paths: Optional[str]) -> str:
        """The base name a pack of *paths* (occlusion, roughness, metal) is
        named by: the roughness map's, else the metal's, else the
        occlusion's, its map-type suffix off (``MapRegistry``)."""
        from pythontk.core_utils.engines.textures.map_registry import MapRegistry

        occlusion, roughness, metallic = paths
        source = roughness or metallic or occlusion
        stem = os.path.splitext(cls.file_name(source))[0]
        try:
            base, suffix = MapRegistry().split_map_suffix(stem)
        except Exception:  # noqa: BLE001 -- an unparsed name packs under its stem
            base, suffix = stem, ""
        return base.rstrip("_.- ") if suffix and base.strip("_.- ") else stem

    @classmethod
    def is_packed_metallic_roughness(cls, path: Optional[str]) -> bool:
        """Whether *path* names an image already in glTF's layout (an ORM,
        by ``MapRegistry``'s reading of its name): one a material uses for
        both its metal and roughness maps is bound as it is, unpacked."""
        if not path:
            return False
        from pythontk.core_utils.engines.textures.map_registry import MapRegistry

        try:
            return MapRegistry().resolve_type_from_path(cls.file_name(path)) == "ORM"
        except Exception:  # noqa: BLE001
            return False

    @classmethod
    def gltf_transform(
        cls, offset: Iterable[float], scale: Iterable[float]
    ) -> Dict[str, List[float]]:
        """A host's UV transform -- ``uv * scale + offset``, origin bottom
        left (Maya's ``place2dTexture`` repeat and offset, Blender's Mapping
        node's scale and location) -- as ``KHR_texture_transform`` states it,
        in glTF's UV space (origin top left, V down): the same scale, and the
        offset ``(u, 1 - scale_v - v)``."""
        o = cls.as_floats(list(offset), 2, 0.0)
        s = cls.as_floats(list(scale), 2, 1.0)
        return {"offset": [o[0], 1.0 - s[1] - o[1]], "scale": s}

    # ------------------------------------------------------------- families
    @classmethod
    def _pbr(cls, props: Mapping[str, Any], maps: Mapping[str, str]) -> Dict[str, Any]:
        """A metallic-roughness surface: the base colour times its weight
        (the weight alone over a colour map), each map standing in for its
        value (factor 1), the emission colour times its weight (the weight
        alone over a map), the opacity's scalar part as the alpha -- its map
        is the colour map's alpha, which glTF reads -- blending wherever the
        opacity is driven or below 1, unless the host states the mode. An
        opacity map apart from the colour map is said: glTF cannot carry it,
        and so is a transmission (:attr:`TRANSMISSION_NOTE`)."""
        weight = cls.as_float(props.get("base_weight"), 1.0)
        color = maps.get("baseColor")
        notes: List[str] = []
        opacity_map = maps.get("opacity")
        if opacity_map and opacity_map != color:
            notes.append(
                f"its opacity map {cls.file_name(opacity_map)} is not its colour "
                f"map ({cls.file_name(color) if color else None}); glTF takes "
                "opacity from the colour map's alpha alone."
            )
        if cls.as_float(props.get("transmission"), 0.0) > 0.0:
            notes.append(cls.TRANSMISSION_NOTE)
        strength = cls.as_float(props.get("emission_weight"), 1.0)
        glow = maps.get("emission")
        opacity = cls._unit(cls.opacity_scalar(props.get("opacity")))
        driven = bool(maps.get("opacity") or props.get("opacity_driven"))
        mode = str(props.get("alpha_mode") or "").upper()
        if mode not in ALPHA_MODES:
            mode = "BLEND" if driven or opacity < 1.0 - 1e-4 else "OPAQUE"
        return {
            "baseColorFactor": (
                [weight] * 3
                if color
                else [c * weight for c in cls.as_floats(props.get("base_color"), 3, 0.8)]
            )
            + [opacity],
            "baseColorTexture": color,
            "metallicFactor": 1.0
            if maps.get("metallic")
            else cls.as_float(props.get("metallic"), 0.0),
            "metallicTexture": maps.get("metallic"),
            "roughnessFactor": 1.0
            if maps.get("roughness")
            else cls.as_float(props.get("roughness"), 0.5),
            "roughnessTexture": maps.get("roughness"),
            "occlusionTexture": maps.get("ambientOcclusion"),
            "normalTexture": maps.get("normal"),
            "normalScale": cls.as_float(props.get("normal_scale"), 1.0),
            "emissiveFactor": [strength] * 3
            if glow
            else [c * strength for c in cls.as_floats(props.get("emission_color"), 3, 0.0)],
            "emissiveTexture": glow,
            "alphaMode": mode,
            "alphaCutoff": props.get("alpha_cutoff"),
            "doubleSided": bool(props.get("double_sided")),
            "_notes": notes,
        }

    @classmethod
    def _stingray_pbs(
        cls, props: Mapping[str, Any], maps: Mapping[str, str]
    ) -> Dict[str, Any]:
        """A StingrayPBS: a map is used while its ``use_*_map`` switch is on,
        and then stands in for its value (the factor is 1); emission is the
        colour (or map) times ``emissive_intensity``. The opacity graph says
        itself through its attributes: ``mask_threshold`` is the masked one,
        ``opacity`` the transparent one, neither the opaque one -- whose
        opacity the graph ignores. Its ``uv_offset`` / ``uv_scale`` are the
        maps' UV transform (:meth:`entry`)."""

        def used(switch: str, channel: str) -> Optional[str]:
            name = maps.get(channel)
            on = cls.as_float(props.get(f"use_{switch}_map"), 0.0) > 0.5
            return name if name and on else None

        color = used("color", "baseColor")
        metal, rough = used("metallic", "metallic"), used("roughness", "roughness")
        intensity = cls.as_float(props.get("emissive_intensity"), 0.0)
        glow = used("emissive", "emission")
        found: Dict[str, Any] = {
            "baseColorFactor": [1.0, 1.0, 1.0]
            if color
            else cls.as_floats(props.get("base_color"), 3, 0.5),
            "baseColorTexture": color,
            "metallicFactor": 1.0 if metal else cls.as_float(props.get("metallic"), 0.0),
            "metallicTexture": metal,
            "roughnessFactor": 1.0 if rough else cls.as_float(props.get("roughness"), 0.5),
            "roughnessTexture": rough,
            "occlusionTexture": used("ao", "ambientOcclusion"),
            "normalTexture": used("normal", "normal"),
            "emissiveFactor": [intensity] * 3
            if glow
            else [c * intensity for c in cls.as_floats(props.get("emissive"), 3, 0.0)],
            "emissiveTexture": glow,
            "_notes": [],
        }
        alpha = 1.0
        if "mask_threshold" in props:
            found["alphaMode"] = "MASK"
            found["alphaCutoff"] = cls.as_float(props.get("mask_threshold"), 0.5)
        elif "opacity" in props or "use_opacity_map" in props:
            found["alphaMode"] = "BLEND"
            opacity_map = used("opacity", "opacity")
            if opacity_map is None:
                alpha = cls.as_float(props.get("opacity"), 1.0)
            elif opacity_map != color:
                found["_notes"].append(
                    f"its opacity map {cls.file_name(opacity_map)} is not its "
                    f"colour map ({cls.file_name(color) if color else None}); "
                    "glTF takes opacity from the colour map's alpha alone."
                )
        found["baseColorFactor"] = found["baseColorFactor"] + [alpha]
        return found

    @classmethod
    def _legacy(
        cls, props: Mapping[str, Any], maps: Mapping[str, str]
    ) -> Dict[str, Any]:
        """An FBX legacy model (Phong, Blinn, Lambert, or anything a reader
        cannot name otherwise) under :attr:`LEGACY_POLICY`: the diffuse colour
        times its factor -- the factor alone over a colour map -- opaque by
        one minus the transparent colour's mean times its factor, emission the
        same way, and Phong's and Blinn's metallic 0.4 with roughness
        sqrt(2 / (2 + shininess)), every other model's 0.2 and 0.8. Unlike
        FBX2glTF, any transparency -- a value, or a map (which only the colour
        map's alpha can carry) -- blends: FBX2glTF judged a lens from the
        colour map's alpha alone and shipped it opaque."""
        model = str(props.get("ShadingModel") or "").lower()
        color = maps.get("baseColor")
        factor = cls.as_float(props.get("DiffuseFactor"), 1.0)
        diffuse = [1.0] * 3 if color else cls.as_floats(props.get("DiffuseColor"), 3, 0.8)
        clear = maps.get("opacity")
        notes: List[str] = []
        if clear:
            if clear != color:
                notes.append(
                    f"its transparency map {cls.file_name(clear)} is not its "
                    f"colour map ({cls.file_name(color) if color else None}); "
                    "glTF takes opacity from the colour map's alpha alone."
                )
            alpha = 1.0
        else:
            transparent = cls.as_floats(props.get("TransparentColor"), 3, 0.0)
            alpha = cls._unit(
                1.0
                - cls.opacity_scalar(transparent)
                * cls.as_float(props.get("TransparencyFactor"), 1.0)
            )
        glow = maps.get("emission")
        glow_factor = cls.as_float(props.get("EmissiveFactor"), 1.0)
        emissive = [
            c * glow_factor
            for c in (
                [1.0] * 3 if glow else cls.as_floats(props.get("EmissiveColor"), 3, 0.0)
            )
        ]
        if model in cls.LEGACY_SPECULAR_MODELS:
            shininess = max(cls.as_float(props.get("Shininess"), 20.0), 0.0)
            metallic, roughness = 0.4, math.sqrt(2.0 / (2.0 + shininess))
        else:
            metallic, roughness = 0.2, 0.8
        return {
            "baseColorFactor": [c * factor for c in diffuse] + [alpha],
            "baseColorTexture": color,
            "metallicFactor": metallic,
            "roughnessFactor": roughness,
            "normalTexture": maps.get("normal"),
            "emissiveFactor": emissive,
            "emissiveTexture": glow,
            "alphaMode": "BLEND" if clear or alpha < 1.0 - 1e-4 else "OPAQUE",
            "_notes": notes,
        }

    @classmethod
    def _unlit(
        cls, props: Mapping[str, Any], maps: Mapping[str, str]
    ) -> Dict[str, Any]:
        """A host's unlit model: its colour (or colour map) and opacity, shown
        as they are, blending wherever the opacity is driven or below 1."""
        color = maps.get("baseColor")
        opacity = cls._unit(cls.opacity_scalar(props.get("opacity")))
        driven = bool(maps.get("opacity") or props.get("opacity_driven"))
        return {
            "baseColorFactor": (
                [1.0] * 3 if color else cls.as_floats(props.get("color"), 3, 1.0)
            )
            + [opacity],
            "baseColorTexture": color,
            "alphaMode": "BLEND" if driven or opacity < 1.0 - 1e-4 else "OPAQUE",
            "unlit": True,
        }

    # ---------------------------------------------------------------- values
    @staticmethod
    def file_name(path: Any) -> str:
        """*path* without its folder (either separator)."""
        return os.path.basename(str(path).replace("\\", "/"))

    @staticmethod
    def _unit(value: float) -> float:
        """*value* clamped to [0, 1]."""
        return min(1.0, max(0.0, float(value)))

    @classmethod
    def opacity_scalar(cls, value: Any, default: float = 1.0) -> float:
        """An opacity as the record states it, one number: a colour's (or a
        transparency colour's) channels' mean, a scalar as it is, *default*
        when it is neither. The one rule every reader of a host's opacity
        colour applies (the hosts' producers and the record-less FBX read)."""
        if isinstance(value, (list, tuple)) and value:
            channels = cls.as_floats(value, len(value), default)
            return sum(channels) / len(channels)
        return cls.as_float(value, default)

    @staticmethod
    def as_float(value: Any, default: float) -> float:
        """*value* as a float; *default* when it is not a number. A host's
        compound value (a tuple, a list) reads as its first component -- what
        a producer hands a scalar field read off a shader attribute."""
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def as_floats(value: Any, size: int, default: float) -> List[float]:
        """*value* as *size* floats; *default* fills what is missing, and a
        scalar fills every component."""
        if isinstance(value, (int, float)):
            items = [float(value)] * size
        elif isinstance(value, (list, tuple)):
            items = []
            for item in value[:size]:
                try:
                    items.append(float(item))
                except (TypeError, ValueError):
                    items.append(float(default))
        else:
            items = []
        return items + [float(default)] * (size - len(items))


class _Document:
    """One record being built: glTF materials appended from statements
    (:meth:`add_statement`) or copied from another record
    (:meth:`add_material`), each image and texture once."""

    def __init__(self, pack: Optional[Packer], notes: Optional[List[str]]) -> None:
        self.pack = pack
        self.notes = notes
        self.materials: List[Dict[str, Any]] = []
        self.textures: List[Dict[str, Any]] = []
        self.images: List[Dict[str, Any]] = []
        self.samplers: List[Dict[str, Any]] = []
        self.names = set()
        self._images: Dict[str, int] = {}
        self._textures: Dict[Tuple[int, Optional[int]], int] = {}
        self._packs: Dict[Tuple[Optional[str], ...], Optional[str]] = {}
        self._pack_names: Dict[str, Tuple[Optional[str], ...]] = {}

    # ------------------------------------------------------------- building
    def add_statements(self, entries: Iterable[Mapping[str, Any]]) -> "_Document":
        for statement in entries:
            if isinstance(statement, Mapping) and statement.get("name"):
                self.add_statement(statement)
        return self

    def add_statement(self, statement: Mapping[str, Any]) -> None:
        """*statement* as a glTF material."""
        mf = MaterialFamilies
        st = mf.finish(statement)
        name = str(st["name"])
        if name in self.names:
            return
        self.names.add(name)
        transform = st.get("uvTransform")
        extension = (
            {
                mf.TRANSFORM_EXTENSION: mf.gltf_transform(
                    transform["offset"], transform["scale"]
                )
            }
            if transform
            else None
        )

        def info(path: str) -> Dict[str, Any]:
            found: Dict[str, Any] = {"index": self.texture(mf.file_name(path))}
            if extension:
                found["extensions"] = {
                    k: {key: list(v) for key, v in value.items()}
                    for k, value in extension.items()
                }
            return found

        pbr: Dict[str, Any] = {
            "baseColorFactor": list(st["baseColorFactor"]),
            "metallicFactor": st["metallicFactor"],
            "roughnessFactor": st["roughnessFactor"],
        }
        material: Dict[str, Any] = {"name": name, "pbrMetallicRoughness": pbr}
        if st.get("baseColorTexture"):
            pbr["baseColorTexture"] = info(st["baseColorTexture"])
        metal, rough = st.get("metallicTexture"), st.get("roughnessTexture")
        occlusion = st.get("occlusionTexture")
        unpacked = False
        if metal or rough:
            sources = st.get("sources") or {}
            maps = ", ".join(mf.file_name(p) for p in (rough, metal) if p)
            try:
                packed, with_occlusion = self.packed(
                    *(sources.get(m, m) if m else None for m in (occlusion, rough, metal))
                )
            except OSError as error:
                raise type(error)(
                    f"Material {name}: its metal and roughness maps ({maps}) could "
                    f"not be packed into one image, glTF's layout -- {error}"
                ) from error
            if packed:
                pbr["metallicRoughnessTexture"] = info(packed)
                if with_occlusion:
                    material["occlusionTexture"] = info(packed)
                    occlusion = None
            else:
                # Its maps are not to hand (a GLB build's map the FBX does
                # not carry): they read as absent, the one rule.
                unpacked = True
                if self.notes is not None:
                    self.notes.append(
                        f"Material {name}: its metal and roughness maps ({maps}) "
                        "are not to hand to pack into one image; they read as "
                        "absent (a metal map's factor dielectric, its roughness "
                        "factor kept)."
                    )
        if occlusion:
            material["occlusionTexture"] = info(occlusion)
        if st.get("normalTexture"):
            material["normalTexture"] = info(st["normalTexture"])
            if st.get("normalScale", 1.0) != 1.0:
                material["normalTexture"]["scale"] = st["normalScale"]
        emissive = list(st["emissiveFactor"])
        peak = max(emissive)
        extensions: Dict[str, Any] = {}
        if peak > 1.0:
            emissive = [c / peak for c in emissive]
            extensions[mf.STRENGTH_EXTENSION] = {"emissiveStrength": peak}
        material["emissiveFactor"] = emissive
        if peak > 0.0 and st.get("emissiveTexture"):
            material["emissiveTexture"] = info(st["emissiveTexture"])
        material["alphaMode"] = st["alphaMode"]
        if st["alphaMode"] == "MASK":
            material["alphaCutoff"] = st.get("alphaCutoff", 0.5)
        material["doubleSided"] = bool(st.get("doubleSided"))
        if st.get("unlit"):
            extensions[mf.UNLIT_EXTENSION] = {}
        if extensions:
            material["extensions"] = extensions
        if unpacked:
            mf.absent_map(material, "metallicRoughnessTexture")
        self.materials.append(material)

    def add_material(self, material: Mapping[str, Any], record: Mapping[str, Any]) -> None:
        """*material* (one of *record*'s) copied whole, its textures with it."""
        import copy

        name = str(material.get("name"))
        if name in self.names:
            return
        self.names.add(name)
        found = copy.deepcopy(dict(material))
        for holder, slot, file_name, sampler in list(
            MaterialFamilies.bindings(record, found)
        ):
            if file_name is None:
                holder.pop(slot)
            else:
                holder[slot]["index"] = self.texture(file_name, sampler)
        self.materials.append(found)

    # ----------------------------------------------------------------- parts
    def packed(
        self, occlusion: Optional[str], rough: Optional[str], metal: Optional[str]
    ) -> Tuple[Optional[str], bool]:
        """The image a material's metal and roughness maps are bound through,
        and whether its occlusion is in it: one map already in glTF's layout
        as it is, else the pack's (once per set of sources; a pack whose
        name another set's pack holds is asked for a unique one, so each
        file name in the record is one image)."""
        mf = MaterialFamilies
        if (
            metal
            and rough
            and metal == rough
            and mf.is_packed_metallic_roughness(metal)
            and occlusion in (None, metal)
        ):
            return metal, occlusion == metal
        key = (occlusion, rough, metal)
        if key not in self._packs:
            packed = self._pack(key, None)
            taken = self._pack_names.get(mf.file_name(packed).casefold()) if packed else None
            if taken is not None and taken != key:
                packed = self._pack(key, mf.packed_name(*key, unique=True))
            if packed:
                self._pack_names[mf.file_name(packed).casefold()] = key
            self._packs[key] = packed
        packed = self._packs[key]
        return packed, bool(packed and occlusion)

    def _pack(
        self, key: Tuple[Optional[str], ...], name: Optional[str]
    ) -> Optional[str]:
        """*key*'s pack under *name* (its own when ``None``): the packer's, or
        -- with none -- the name a pack would take."""
        if self.pack is None:
            return name or MaterialFamilies.packed_name(*key)
        return self.pack(*key, name=name) if name else self.pack(*key)

    def texture(self, file_name: str, sampler: Optional[Mapping[str, Any]] = None) -> int:
        """The texture index of *file_name* sampled by *sampler* (``None``:
        glTF's default), each once."""
        image = self._images.get(file_name)
        if image is None:
            self.images.append({"uri": file_name})
            image = self._images[file_name] = len(self.images) - 1
        sampler_index = None
        if sampler:
            spec = dict(sampler)
            if spec not in self.samplers:
                self.samplers.append(spec)
            sampler_index = self.samplers.index(spec)
        key = (image, sampler_index)
        if key not in self._textures:
            texture: Dict[str, Any] = {"source": image}
            if sampler_index is not None:
                texture["sampler"] = sampler_index
            self.textures.append(texture)
            self._textures[key] = len(self.textures) - 1
        return self._textures[key]

    def payload(self) -> Dict[str, Any]:
        """The record, version stamped (``SceneRecords.MATERIALS.make``)."""
        from pythontk.core_utils.engines.scene_export.scene_records import (
            SceneRecords,
        )

        mf = MaterialFamilies
        used = list(
            dict.fromkeys(e for m in self.materials for e in mf.extensions_of(m))
        )
        body: Dict[str, Any] = {"asset": {"version": mf.GLTF_VERSION}}
        if used:
            body["extensionsUsed"] = sorted(used)
        body["materials"] = self.materials
        for key in ("textures", "images", "samplers"):
            if getattr(self, key):
                body[key] = getattr(self, key)
        return SceneRecords.MATERIALS.make(body).payload
