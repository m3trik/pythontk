# !/usr/bin/python
# coding=utf-8
"""The ``materials`` record: its payload, declared once, and the one model that
states a material in it.

A deliverable states each of its materials in glTF 2.0's metallic-roughness
model (:class:`MaterialRecord`), in the file, so every reader shades it alike:
the GLB build (``MeshConvert.apply_glb_materials``), unitytk, and a reader
holding nothing but the file. An FBX alone states a material in its author's
own terms -- a StingrayPBS's use-map switches, a standardSurface's weights, a
Phong's shininess -- and every reader translated those its own way: measured
on a production assembly (2026-10-20), FBX2glTF multiplied each StingrayPBS
colour map by its inert 0.5 ``base_color``, dropped its emission and left a
Phong lens OPAQUE, and Unity's importer gave the same lens another metal and
roughness than the GLB.

:class:`MaterialFamilies` is the model every statement comes from: plain values
in, a record entry out. A reader hands it a material in its shading FAMILY's
own vocabulary and the images bound to it by logical channel; the family's
rules turn that into glTF's terms. The hosts' export producers read the live
scene into it (mayatk / blendertk ``MaterialRecords``), and the FBX reader
reads a file that carries no record into the same families
(``MeshConvert.fbx_materials``) -- one set of rules, so a file read back states
what its producer did.

It is a typed :class:`~pythontk.SchemaSpec`: this one definition validates a
payload (``MaterialRecord.validate(payload)``), publishes its JSON Schema, and
is what ``m3trik/scripts/sync_scene_records.py`` generates unitytk's C# record
types from. Change a field together with :class:`MaterialFamilies` and bump
``SceneRecords.MATERIALS.version``.
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from pythontk.core_utils.schema_spec import SchemaSpec

_field = SchemaSpec.spec_field

#: glTF 2.0's alpha modes, the record's vocabulary for a material's opacity.
ALPHA_MODES = ("OPAQUE", "MASK", "BLEND")


@dataclass
class MaterialEntry(SchemaSpec):
    """One material as glTF 2.0's metallic-roughness model states it: each
    factor linear and multiplied by its map, each map an image the file
    embeds, named by file name and read through UV set 0."""

    TYPED = True

    name: str = _field(
        help=(
            "The material's name as the file names it, namespace included "
            "('ns:name'): the join key to the imported material. A reader "
            "whose import drops the namespace matches the name after the last "
            "':'."
        ),
        required=True,
    )
    baseColorFactor: Tuple[float, float, float, float] = _field(
        help="Linear RGBA; A is the opacity.",
        required=True,
        default=(1.0, 1.0, 1.0, 1.0),
    )
    baseColorTexture: Optional[str] = _field(
        help="The colour map (sRGB); its alpha is the opacity."
    )
    metallicFactor: float = _field(
        help="0 dielectric to 1 metal.", required=True, default=1.0
    )
    metallicTexture: Optional[str] = _field(help="Metalness map (linear), read from R.")
    roughnessFactor: float = _field(
        help="0 mirror to 1 matte.", required=True, default=1.0
    )
    roughnessTexture: Optional[str] = _field(
        help="Roughness map (linear), read from R."
    )
    occlusionTexture: Optional[str] = _field(
        help=(
            "Ambient-occlusion map (linear), read from R; it darkens only the "
            "environment's light, and a lightmapped object takes none (its bake "
            "holds the occlusion)."
        )
    )
    normalTexture: Optional[str] = _field(
        help="Tangent-space normal map, +Y up (glTF's convention), linear."
    )
    normalScale: float = _field(
        help="Scales the normal map's X and Y; 1 is the map as authored.", default=1.0
    )
    emissiveFactor: Tuple[float, float, float] = _field(
        help="Linear RGB emission; may exceed 1.",
        required=True,
        default=(0.0, 0.0, 0.0),
    )
    emissiveTexture: Optional[str] = _field(help="The emission map (sRGB).")
    alphaMode: str = _field(
        help=(
            "OPAQUE ignores the opacity; MASK cuts at alphaCutoff; BLEND blends "
            "(sorted, no depth write)."
        ),
        required=True,
        choices=ALPHA_MODES,
        default="OPAQUE",
    )
    alphaCutoff: Optional[float] = _field(
        help="MASK only: the opacity below which a pixel is cut."
    )
    doubleSided: bool = _field(help="Both faces render.", default=False)
    unlit: bool = _field(
        help="Shows its base colour and opacity alone: no light, reflection or emission.",
        default=False,
    )


@dataclass
class MaterialRecord(SchemaSpec):
    """The ``materials`` record's payload: every material the file's meshes use."""

    TYPED = True

    version: int = _field(
        help="The record's schema version (SceneRecords.MATERIALS.version).",
        required=True,
    )
    materials: List[MaterialEntry] = _field(
        help="Every material, in the file's order.",
        required=True,
        nested=[MaterialEntry],
        default_factory=list,
    )


class MaterialFamilies:
    """A material, read in its shading family's own terms, as the record states it.

    The host-free model under every statement of a material (module
    docstring). A family is a shading model's vocabulary and the rules that
    turn it into glTF's: :meth:`entry` takes the family's name, the
    material's values in that vocabulary (``props``) and the images bound to
    it by logical channel (``maps``: :attr:`MAP_CHANNELS`, the slot names of
    mayatk's ``ShaderAttributeMap`` and both hosts' ``MatManifest``), and
    returns the finished entry. A new family is one :attr:`FAMILIES` row and
    its resolver; a new host reads into the families that exist.

    The vocabularies:

    - ``pbr`` -- a metallic-roughness surface in logical terms (Standard
      Surface, aiStandardSurface, openPBR, Blender's Principled):
      ``base_color``, ``base_weight``, ``metallic``, ``roughness``,
      ``emission_color``, ``emission_weight``, ``opacity`` (its scalar part),
      ``opacity_driven``, and optionally ``alpha_mode`` / ``alpha_cutoff`` and
      ``normal_scale``;
    - ``stingray_pbs`` -- a StingrayPBS by its own attribute names
      (``use_color_map``, ``base_color``, ``emissive_intensity``, the opacity
      graph's ``opacity`` or ``mask_threshold``, ...);
    - ``legacy`` -- FBX's own models by FBX's property names (``DiffuseColor``,
      ``DiffuseFactor``, ``TransparentColor``, ``TransparencyFactor``,
      ``EmissiveColor``, ``EmissiveFactor``, ``Shininess``) and
      ``ShadingModel``: how a Phong, Blinn or Lambert reaches every engine;
    - ``unlit`` -- a host's unlit model: ``color``, ``opacity``,
      ``opacity_driven``.
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

    #: Family -> its resolver, ``(props, maps) -> entry``.
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
            "alpha_mode",
            "alpha_cutoff",
            "normal_scale",
            "double_sided",
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

    #: The entry fields that name a map, in the order a reader binds them.
    TEXTURE_FIELDS: Tuple[str, ...] = (
        "baseColorTexture",
        "metallicTexture",
        "roughnessTexture",
        "occlusionTexture",
        "normalTexture",
        "emissiveTexture",
    )

    # ------------------------------------------------------------------ state
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
        """*name*'s record entry, from its *family*'s values.

        Parameters:
            family: A :attr:`FAMILIES` key.
            name: The material's name as the file names it.
            props: Its values in the family's vocabulary (class docstring);
                a value the host does not carry is left out.
            maps: ``{logical channel: image}`` (:attr:`MAP_CHANNELS`) -- a
                path or a file name; the entry names it by file name, the
                name the file embeds it under.
            unlit: The scene shades it unlit whatever it wears (a projected
                shadow's plane): its base colour and opacity alone.
            notes: Where a sentence about what the record cannot state goes
                (a tiled map); dropped when ``None``.

        Raises:
            KeyError: *family* is not a :attr:`FAMILIES` key.
        """
        method = cls.FAMILIES.get(family)
        if method is None:
            raise KeyError(
                f"{family!r} is no material family (known: {', '.join(cls.FAMILIES)})."
            )
        files = {
            channel: cls.file_name(image)
            for channel, image in (maps or {}).items()
            if channel in cls.MAP_CHANNELS and image
        }
        found = getattr(cls, method)(props or {}, files)
        found["name"] = name
        if unlit:
            found["unlit"] = True
        for sentence in found.pop("_notes", ()):
            if notes is not None:
                notes.append(f"Material {name}: {sentence}")
        return cls.finish(found)

    @classmethod
    def payload(cls, entries: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        """The record's payload for *entries* (the first of a repeated name
        kept), version stamped."""
        from pythontk.core_utils.engines.scene_export.scene_records import (
            SceneRecords,
        )

        kept: Dict[str, Mapping[str, Any]] = {}
        for found in entries:
            if isinstance(found, Mapping) and found.get("name"):
                kept.setdefault(str(found["name"]), found)
        return SceneRecords.MATERIALS.make({"materials": list(kept.values())}).payload

    @classmethod
    def finish(cls, entry: Mapping[str, Any]) -> Dict[str, Any]:
        """*entry* in the record's shape: every required field, a map or an
        optional value only where it says something, each factor in range.
        Unlit means metallic 0 and roughness 1 (glTF's fallback for a reader
        that lights it anyway)."""
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
                out[field] = cls.file_name(entry[field])
        if not any(out["emissiveFactor"]):
            out.pop("emissiveTexture", None)
        scale = cls.as_float(entry.get("normalScale"), 1.0)
        if out.get("normalTexture") and scale != 1.0:
            out["normalScale"] = scale
        if mode == "MASK":
            out["alphaCutoff"] = cls._unit(cls.as_float(entry.get("alphaCutoff"), 0.5))
        if entry.get("doubleSided"):
            out["doubleSided"] = True
        if unlit:
            out["unlit"] = True
        return out

    @classmethod
    def texture_files(cls, record: Optional[Mapping[str, Any]]) -> List[str]:
        """The file names *record* binds, each once, in record order."""
        names: List[str] = []
        for found in (record or {}).get("materials") or ():
            if not isinstance(found, Mapping):
                continue
            for field in cls.TEXTURE_FIELDS:
                name = found.get(field)
                if isinstance(name, str) and name and name not in names:
                    names.append(name)
        return names

    # ------------------------------------------------------------- families
    @classmethod
    def _pbr(cls, props: Mapping[str, Any], maps: Mapping[str, str]) -> Dict[str, Any]:
        """A metallic-roughness surface: the base colour times its weight
        (the weight alone over a colour map), each map standing in for its
        value (factor 1), the emission colour times its weight (the weight
        alone over a map), the opacity's scalar part as the alpha -- its map
        is the colour map's alpha, which glTF reads -- blending wherever the
        opacity is driven or below 1, unless the host states the mode. An
        opacity map apart from the colour map is said: glTF cannot carry it."""
        weight = cls.as_float(props.get("base_weight"), 1.0)
        color = maps.get("baseColor")
        notes: List[str] = []
        opacity_map = maps.get("opacity")
        if opacity_map and opacity_map != color:
            notes.append(
                f"its opacity map {opacity_map} is not its colour map ({color}); "
                "glTF takes opacity from the colour map's alpha alone."
            )
        strength = cls.as_float(props.get("emission_weight"), 1.0)
        glow = maps.get("emission")
        opacity = cls._unit(cls.as_float(props.get("opacity"), 1.0))
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
        opacity the graph ignores."""

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
                    f"its opacity map {opacity_map} is not its colour map "
                    f"({color}); glTF takes opacity from the colour map's alpha "
                    "alone."
                )
        found["baseColorFactor"] = found["baseColorFactor"] + [alpha]
        offset = cls.as_floats(props.get("uv_offset"), 2, 0.0)
        scale = cls.as_floats(props.get("uv_scale"), 2, 1.0)
        if offset != [0.0, 0.0] or scale != [1.0, 1.0]:
            found["_notes"].append(
                f"its maps tile (uv_scale {scale}, uv_offset {offset}); the "
                "record states none, so every reader samples them untiled."
            )
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
                    f"its transparency map {clear} is not its colour map "
                    f"({color}); glTF takes opacity from the colour map's alpha "
                    "alone."
                )
            alpha = 1.0
        else:
            transparent = cls.as_floats(props.get("TransparentColor"), 3, 0.0)
            alpha = cls._unit(
                1.0
                - sum(transparent)
                / 3.0
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
        opacity = cls._unit(cls.as_float(props.get("opacity"), 1.0))
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
