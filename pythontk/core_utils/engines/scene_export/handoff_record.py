# !/usr/bin/python
# coding=utf-8
"""The handoff: how a deliverable states itself, declared once for every carrier.

A deliverable is routinely handed on alone -- an FBX to an engine, a GLB to a
viewer that is not ours, either to a developer or an agent holding one file
and no conversation -- so each carries one block saying how to read it and
what the approved scene is, under the same keys on every carrier:
``version``, ``instructions``, ``assembly``, ``rendering`` and ``extras``.

:class:`HandoffRecord` is that block's one declaration: the reading instructions,
the approved scene part by part in each carrier's addresses, and the lighting
it was approved under. A carrier's writer says only which records its file
holds and where (:meth:`HandoffRecord.block`); a new carrier (USD) is one
:attr:`HandoffRecord.CARRIERS` row.

Every sentence is declarative about the file and the scene it rebuilds --
never an action outside the file (no command, no fetch, no tool) -- which is
what makes the block safe for an agent to read as untrusted content.
"""

import copy
import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from pythontk.core_utils.engines.scene_export.scene_records import (
    Clause,
    SceneRecords,
    Scope,
)
from pythontk.core_utils.engines.scene_export.scene_store import SceneStoreBase
from pythontk.core_utils.schema_spec import SchemaSpec

#: The FBX's carrier node, named once by the store (``data_export``).
_NODE = SceneStoreBase.name(Scope.DELIVERABLE)


def _filled_rows(
    rows: Iterable[Mapping[str, Any]],
    text: Mapping[str, Optional[str]],
    held: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """*rows* as a file holding *held* reads them: each row's ``spec`` with
    the ``clauses`` *held* calls for (:meth:`Clause.join`; every one for
    ``None``), each ``{placeholder}`` in them replaced by *text*'s, and each
    ``data`` entry that is a placeholder by its address -- left out where
    *text* gives none, so a row never names data the file lacks. A clause
    its placeholder leaves empty, and a row left with no text, are left out.
    A module function so the class body can call it at definition."""

    def fill(spec: str) -> str:
        for placeholder, value in text.items():
            spec = spec.replace(placeholder, value or "")
        return spec

    filled = []
    for row in rows:
        clauses = row.get("clauses", ())
        spec = Clause.join(
            fill(row["spec"]),
            [dataclasses.replace(c, text=fill(c.text)) for c in clauses],
            held,
        )
        if not spec:
            continue
        data = [text.get(d) if d.startswith("{") else d for d in row["data"]]
        filled.append(
            {
                "part": row["part"],
                "data": [d for d in data if d],
                "spec": spec,
            }
        )
    return filled


_field = SchemaSpec.spec_field


@dataclass
class HandoffPart(SchemaSpec):
    """One part of the approved scene: where the file holds it and what it
    makes of the scene."""

    TYPED = True

    part: str = _field(
        help=(
            "The part: 'geometry', 'materials', a record's key, or a lighting "
            "setting's name."
        ),
        required=True,
    )
    data: List[str] = _field(
        help=(
            "Where the file holds it, in its carrier's addresses "
            "('data_export.lightmaps', 'extras.lightmaps', "
            "'rendering.keyLight'); empty for the file's own content."
        ),
        required=True,
        default_factory=list,
    )
    spec: str = _field(
        help="What the part makes of the approved scene, field by field.",
        required=True,
    )


@dataclass
class HandoffSource(SchemaSpec):
    """Where the file came from: provenance, never identity."""

    TYPED = True

    application: str = _field(
        help="The application that wrote it ('maya', 'blender').", required=True
    )
    version: str = _field(help="That application's version.")
    scene: str = _field(help="The scene file it was written from.")


@dataclass
class HandoffExtras(SchemaSpec):
    """What a reader may ignore."""

    TYPED = True

    source: HandoffSource = _field(
        help="Where the file came from.", nested=HandoffSource
    )


@dataclass
class HandoffBlock(SchemaSpec):
    """The ``handoff`` block: how to read the file, the approved scene part by
    part, and the lighting it was approved under (:meth:`HandoffRecord.block`)."""

    TYPED = True

    version: int = _field(
        help="The block's schema version (HandoffRecord.VERSION).", required=True
    )
    instructions: str = _field(
        help="How to read the file and its blocks, for a reader holding nothing else.",
        required=True,
    )
    assembly: List[HandoffPart] = _field(
        help="The approved scene, part by part, then the lighting.",
        required=True,
        nested=[HandoffPart],
        default_factory=list,
    )
    rendering: Dict[str, Dict[str, Any]] = _field(
        help=(
            "The lighting recipe by section -- {section: {setting: value}}, "
            "each value a number, a vector of numbers (colours linear RGB) or "
            "an identifier -- the settings this file's content uses (one "
            "without lightmaps states none of the lightmapped materials'); "
            "the assembly's lighting rows say what each means."
        ),
        required=True,
        default_factory=dict,
    )
    extras: HandoffExtras = _field(
        help="What a reader may ignore; absent when there is nothing to say.",
        nested=HandoffExtras,
    )


@dataclass(frozen=True)
class HandoffCarrier:
    """How one carrier holds a deliverable's data, as its handoff states it.

    Parameters:
        prefix: What the ``data`` address of a record in this carrier is its
            key after.
        blocks: Whether a record rides as its GLB block
            (:attr:`RecordSpec.glb`, whose ``assembly`` its row states) or as
            itself (:attr:`RecordSpec.assembly`) -- one key either way.
        intro: The instructions' opening: what the block describes and where
            the records ride.
        addresses: How this carrier's ``data`` addresses read.
        own: The ``assembly`` rows of the file's own content; no address.
        probe: How the reflection probe's image is stored.
        probe_axes: The axes the probe is mapped in.
        probe_units: What the probe's ``position`` and ``box`` are in.
        intensity: A lightmapped material's level in this carrier; ``{at}``
            is where the carrier holds the lightmaps.
        rules: The carrier's own reading rules, after the shared ones.
        highlights: What the lighting rows say of the probe's highlight
            copies -- empty where the lightmaps' own row says it.
    """

    prefix: str
    blocks: bool
    intro: str
    addresses: str
    own: Tuple[Dict[str, Any], ...]
    probe: str
    probe_axes: str
    probe_units: str
    intensity: str
    rules: str = ""
    highlights: str = ""

    def address(self, key: str) -> str:
        """Where this carrier holds the record *key*: the same key on every
        carrier, after its prefix."""
        return self.prefix + key


class HandoffRecord:
    """The handoff contract, declared once: the block's shape, its reading
    instructions, the approved scene's parts in each carrier's addresses, and
    the lighting it was approved under.

    The block's identity -- the key it rides under on every carrier and its
    schema version -- is its record's (``SceneRecords.HANDOFF``). Version 3
    (2026-10-20) is the shape every block keeps: the fundamentals at the top
    and the rest under ``extras``, glTF's own convention, so a reader ignores
    an extra it does not know and adding one needs no new version.
    """

    KEY: str = SceneRecords.HANDOFF.key
    VERSION: int = SceneRecords.HANDOFF.version

    #: The row a record no declaration knows gets: another tool's.
    UNDECLARED = "A record another tool wrote; this file says nothing more of it."

    #: The reference viewer's lighting setup (``net_utils/preview/viewer.html``),
    #: published as data so a recipient can reproduce the look the asset was
    #: signed off in instead of inferring it. A baked asset needs this more than
    #: an unbaked one, not less: its diffuse lighting is already in its
    #: textures, so the viewer's own rig has to be withheld exactly where the
    #: bake already answers (the key light, and the environment's diffuse) and
    #: kept where it does not (the environment's specular, the only term left
    #: to show a reflection or a roughness map on a baked surface; a normal map
    #: shows there through the relief, ``lightmappedMaterials.normalRelief``,
    #: which is why the key light's direction outlives its intensity). Without
    #: it a recipient reasonably adds a normal key light, or an ambient term,
    #: and washes out every baked surface -- which reads as a bake regression
    #: and sends them back to the baker, where nothing is wrong.
    #:
    #: Values alone -- a number, a vector, or a name a runtime switches on --
    #: and the same on every carrier; a colour is linear RGB, as every colour
    #: the deliverable states. What each means (the formulas, the condition,
    #: where a carrier holds the probe and the lightmaps) is the lighting rows'
    #: ``spec`` (:meth:`lighting`), so a value is never a sentence and a reader
    #: parses one shape. ``test_preview_server`` pins these against the
    #: viewer's literals, and ``test_preview_viewer_live`` against the pixels
    #: it draws, so the published contract cannot drift from what the viewer
    #: does.
    RENDERING_POLICY: Dict[str, Any] = {
        "renderer": {
            "toneMapping": "ACESFilmic",
            "toneMappingExposure": 1.0,
        },
        "environment": {
            "source": "probe",
            "fallback": "RoomEnvironment",
            "prefilter": "PMREM",
            "prefilterBlur": 0.04,
            "intensity": 1.0,
            "probeIntensity": 0.3183099,
        },
        "keyLight": {
            "type": "directional",
            "color": [1.0, 1.0, 1.0],
            "intensity": 0.9,
            "position": [3, 6, 4],
            "disabledWhen": "lightmapped",
        },
        "lightmappedMaterials": {
            "envMapTerms": "specular",
            "envMapIntensity": 1.0,
            "reflectionNormalization": "lightmapLuma",
            "lightMapEnergy": "splitSum",
            "normalRelief": "halfLambertRatio",
        },
    }

    #: The settings only some files use, each with what its file must hold
    #: for it (a path into its records, :meth:`Clause.holds`): a file without
    #: lightmaps has no lightmapped material for these to govern, one without
    #: a probe no probe to choose or level. A file's block leaves out what its
    #: content does not use (:meth:`rendering_policy`), as its rows leave out
    #: the text that would explain it.
    RENDERING_NEEDS: Dict[Tuple[str, ...], Tuple[str, ...]] = {
        ("environment", "source"): (SceneRecords.LIGHTMAPS.key, "probe"),
        ("environment", "probeIntensity"): (SceneRecords.LIGHTMAPS.key, "probe"),
        ("keyLight", "disabledWhen"): (SceneRecords.LIGHTMAPS.key,),
        ("lightmappedMaterials",): (SceneRecords.LIGHTMAPS.key,),
    }

    # ----------------------------------------------------------------- text
    #: The instructions every carrier shares, between its own opening and its
    #: own rules. What each part is -- the file's own content included -- is
    #: its ``assembly`` row's, and a format's conventions are the format's:
    #: the text says how to read the block and nothing a row says.
    _READING = (
        "'assembly' lists every part of that scene: the data it is built from "
        "('data': {addresses}; a part with none is the file's own content) and "
        "what that data means ('spec'). A part left out yields a scene other "
        "than the one approved. 'rendering' is the lighting it was approved "
        "under, part of its look: lit any other way, the same file renders "
        "differently. Every block keeps its fundamentals at the top and the "
        "rest under 'extras' (this one's: 'source', where the file came from); "
        "a reader ignores an extra it does not know."
    )
    _CHECK = "Check 'version' against the schema you expect."

    #: Where the probe is and what its texels mean, whichever carrier holds
    #: it; :attr:`HandoffCarrier` fills in the storage, the axes and the units.
    #: The level is stated for both lightmap conventions, not as a bare
    #: ``probeIntensity``: that number is three.js', whose lightmapped diffuse
    #: is pi below a renderer that composites the texel as ``albedo x
    #: texel``. Such a renderer keeps the balance with the probe at 1 --
    #: unitytk's native path does (``LightmapMetadataController.ApplyProbes``),
    #: where 1/pi alone would reflect a third of the approved room -- but
    #: matches the approved brightness only with its lightmaps AND the probe
    #: at 1/pi, which a cold read of the files found unstated (2026-10-20).
    _PROBE = (
        "{at}.probe: {image}, an equirectangular map in {axes} (u = 0.5 + "
        "atan2(z, x) / 2pi, v = 0.5 + asin(y) / pi, its top row +Y); "
        "'position' is where it was captured and 'box' ({{min, max}}, or null "
        "for an open scene) the box its reflections project onto, both in "
        "{units}. It holds radiance, and a lightmap texel is a white card's "
        "radiance (irradiance / pi): the approved look takes the texel as "
        "irradiance -- three.js, whose baked diffuse is texel x albedo / pi -- "
        "and plays the probe at the same scale, 'probeIntensity' (1/pi). A "
        "renderer that composites the texel as texel x albedo (Unity's "
        "Standard and URP Lit) matches it with its lightmaps scaled by 1/pi "
        "and the probe at 'probeIntensity' too; left at 1, the two keep their "
        "balance but light the scene pi times brighter than approved. Every "
        "lookup is made in those axes. "
        "A reflection is box-projected: followed from the shaded point to "
        "where it leaves the box, and the probe read toward that point from "
        "'position'; diffuse irradiance is read along the normal."
    )

    #: The lighting rows of every handoff's ``assembly``: what each part of
    #: :attr:`RENDERING_POLICY` means, the carrier's addresses filled in
    #: (:meth:`lighting`). Each points into ``rendering`` (an address relative
    #: to the block) rather than restating a number, so an export's override
    #: and the row cannot disagree; the two that read the file's lightmaps --
    #: the probe, a material's map -- list their address too
    #: (``{lightmaps}``), where the file has them. Stated as what the approved
    #: scene HAS, because the failure they exist for is a reader treating the
    #: policy as background: building the geometry faithfully and lighting it
    #: with whatever its engine defaults to. What only a lightmapped file needs
    #: -- a probe, its highlight copies, the lightmapped materials' rule -- is
    #: a :class:`Clause` over the file's records, so a file without them is
    #: told none of it.
    _LIGHTING_ROWS: Tuple[Dict[str, Any], ...] = (
        {
            "part": "environment",
            "data": ["rendering.environment", "{lightmaps}"],
            "spec": (
                "The scene's only ambient and reflected light: a neutral studio "
                "room ('fallback', three.js RoomEnvironment) at 'intensity', "
                "blurred by 'prefilterBlur' (radians) and prefiltered as "
                "'prefilter' says (PMREM: three.js r169's GGX-prefiltered cube-UV "
                "mip chain, a level per roughness). Every unbaked surface takes "
                "it whole, diffuse and specular. No sky, ambient colour or other "
                "environment is part of the approved scene."
            ),
            "clauses": (
                Clause(
                    "A lightmapped surface takes only its reflections, as "
                    "lightmapped_materials says.",
                    (SceneRecords.LIGHTMAPS.key,),
                ),
                Clause(
                    "With 'source' 'probe', the asset's own reflection probe "
                    "takes the studio room's place, prefiltered alike (with no "
                    "blur): the room the bake lit, captured by it as an HDR from "
                    "one point -- {probe} It plays at its own level, never "
                    "dimmed as the studio is: a baked metal has no diffuse for "
                    "its lightmap to hold, so its reflections are all of it, "
                    "and every unbaked object stands in that room's light "
                    "rather than the studio's.",
                    (SceneRecords.LIGHTMAPS.key, "probe"),
                ),
                Clause(
                    "{highlights}",
                    (SceneRecords.LIGHTMAPS.key, "probe", "highlights"),
                ),
            ),
        },
        {
            "part": "key_light",
            "data": ["rendering.keyLight"],
            "spec": (
                "One white directional light ('type', 'color') at 'intensity', "
                "shining from 'position' toward the origin. The approved scene "
                "has no other: no node the file carries lights it, a shadow's "
                "source included."
            ),
            "clauses": (
                Clause(
                    "It is off entirely while any material in the asset is "
                    "lightmapped ('disabledWhen'): scene-wide, it cannot be "
                    "spared the baked geometry it would contradict. Its "
                    "direction still orients lightmappedMaterials.normalRelief, "
                    "and the bake's own lights reach the scene as highlights "
                    "alone, as the lightmaps part says.",
                    (SceneRecords.LIGHTMAPS.key,),
                ),
            ),
        },
        {
            "part": "lightmapped_materials",
            "data": ["rendering.lightmappedMaterials", "{lightmaps}"],
            "spec": "",
            "clauses": (
                Clause(
                    "A lightmapped surface's diffuse light is its lightmap alone, "
                    "at {intensity}: no environment diffuse -- the bake already "
                    "holds every light and the sky, and adding it double-counts "
                    "them and reads as a washed-out bake -- no realtime light, and "
                    "not its material's own ambient-occlusion map, the bake "
                    "already holding the occlusion. It keeps the environment's "
                    "specular ('envMapTerms'), the reflection-probe role a native "
                    "lightmap path such as Unity's gives a lightmapped renderer, "
                    "at 'envMapIntensity' times the environment's own intensity -- "
                    "the studio's level, the export's choice; an asset's own probe "
                    "reflects at its own level, 'probeIntensity', undimmed -- "
                    "scaled per texel as "
                    "'reflectionNormalization' names: lightmapLuma is clamp(luma("
                    "E_bake) / luma(E_env), 0, 1), E_bake the texel's lightmap "
                    "irradiance (map times its intensity), E_env the environment's "
                    "diffuse irradiance about the surface's own normal (both in "
                    "the approved model's units), luma the Rec. 709 weights, so a "
                    "shadow reflects next to nothing and a "
                    "lit surface up to the full level. 'lightMapEnergy' splitSum: "
                    "the lightmap's irradiance enters as an image-based light's "
                    "does in the split-sum model (three.js r169's iblIrradiance, "
                    "RE_IndirectSpecular_Physical), the diffuse it lights weighted "
                    "by one minus the specular's directional albedo (Karis's "
                    "analytic environment-BRDF fit) and the multiscatter energy "
                    "taken from it (Fdez-Aguera's, Favg = F0 + (1 - F0)/21); "
                    "added at full weight beside the reflection, a surface is lit "
                    "twice wherever it reflects most. 'normalRelief' "
                    "halfLambertRatio: where a baked material carries a normal map, "
                    "its lightmap texel is scaled by (dot(N, L) * 0.5 + 0.5) / "
                    "(dot(Ng, L) * 0.5 + 0.5), N the normal-mapped normal, Ng the "
                    "surface's own, L the one unit direction from the origin "
                    "toward keyLight.position "
                    "-- reflected across the surface's plane wherever the surface "
                    "faces away from it, since a bake's light reached the surface "
                    "from its front -- exactly 1 where the map is flat, so "
                    "normalTexture.scale 0 is the pure bake.",
                    (SceneRecords.LIGHTMAPS.key,),
                ),
            ),
        },
        {
            "part": "renderer",
            "data": ["rendering.renderer"],
            "spec": (
                "ACES filmic tone mapping as three.js r169's (Stephen Hill's fit, "
                "its input scaled by 'toneMappingExposure' / 0.6), output in sRGB."
            ),
        },
    )

    # ------------------------------------------------------------- carriers
    #: Each carrier's own account: how it addresses its data, what its reader
    #: must be told about it, the rows of its own content, and where it holds
    #: the probe and the lightmaps. The FBX's records ride its ``data_export``
    #: nodes; the GLB's blocks its root ``extras``. Before the carriers named
    #: their own, the FBX sent its reader after a GLB address an FBX cannot
    #: have, and a coworker's build dropped the lighting environment
    #: (reported 2026-10-14).
    #:
    #: The FBX rules were tightened (2026-10-14) against what a reader holding
    #: only the file actually dropped on a delivered assembly: WHICH carrier
    #: (an assembly ships one ``data_export`` per referenced module beside the
    #: scene's own, each that module's records as it was last exported alone;
    #: ``MeshConvert._data_export_carrier`` and unitytk's ``CarrierImport``
    #: resolve them alike, and the text states the rule), and CASE (a module
    #: spelled a map ``_LightMap`` where the embedded copy is ``_Lightmap``).
    #: Its load-bearing sentence is the one about the files the records name:
    #: they join by FILE NAME to the file's own embedded media -- the maps a
    #: lighting-only bake leaves off every material included, which a
    #: delivered file embeds as media of their own (``FbxMedia.
    #: embed_dependencies``). Delivered, not every: the block rides every
    #: carrier-bearing write (a File > Export restamps it too) while only the
    #: Scene Exporters and unitytk's delivery embed, so the sentence says what
    #: a name joins to where the file holds no copy.
    CARRIERS: Dict[str, HandoffCarrier] = {
        "fbx": HandoffCarrier(
            prefix=f"{_NODE}.",
            blocks=False,
            intro=(
                "This block describes the file that carries it and the "
                "approved scene that file rebuilds. The scene's records ride "
                f"as user properties on nodes named '{_NODE}', each value a "
                "JSON string."
            ),
            addresses=(
                f"an address beginning '{_NODE}.' is a record, one "
                "beginning 'rendering.' a field of this block"
            ),
            rules=(
                f"Per record, the node named exactly '{_NODE}' is the "
                "scene's own and wins; the same record on any other such node "
                f"('{_NODE}1', a namespaced 'NS:{_NODE}') is a "
                "referenced module's, as that module was last exported alone, "
                "and counts only where the scene's own node has none -- a "
                f"'{SceneRecords.HANDOFF.key}' there describes that module, not "
                "this file. Records name files by FILE NAME, never by folder: "
                "a delivered file embeds each as media of its own (a Video "
                "object's Content), the name "
                "the join token to that embedded copy, matched without regard "
                "to case, or, where the file holds no copy, to one provided "
                "with it. A record's lengths times its "
                "'unit_scale' are metres, in the file's axes; angles are in "
                "degrees unless a field says radians, "
                "frames at the record's 'fps'."
            ),
            own=(
                {
                    "part": "geometry",
                    "data": [],
                    "spec": (
                        "The file's hierarchy and meshes, in its own units and "
                        "axes, and its embedded textures, as approved. Each "
                        "material is as the 'materials' part states it; a file "
                        "without that record has only its own materials to go by."
                    ),
                },
            ),
            probe=(
                "'map' names a linear OpenEXR image by FILE NAME, which a "
                "delivered file embeds as media no material uses"
            ),
            probe_axes="the scene's own Y-up axes",
            probe_units=(
                "the units the file's geometry is written in, 'unit_scale' "
                "metres per unit: each converts into a receiving engine exactly "
                "as a vertex of the file does (the box's corners re-sorted per "
                "axis after)"
            ),
            intensity="1, the map applied as stored",
        ),
        "glb": HandoffCarrier(
            prefix="extras.",
            blocks=True,
            intro=(
                f"'extras.{SceneRecords.HANDOFF.key}' describes this glTF 2.0 "
                "file and the approved scene it rebuilds."
            ),
            addresses=(
                "an address beginning 'extras.' is in the file's root extras, "
                "one beginning 'rendering.' a field of this block"
            ),
            own=(
                {
                    "part": "geometry",
                    "data": [],
                    "spec": (
                        "The file's hierarchy and meshes and every texture, "
                        "embedded, as approved."
                    ),
                },
                {
                    "part": SceneRecords.MATERIALS.key,
                    "data": [],
                    "spec": "Every material complete in glTF 2.0's own terms, as approved.",
                    "clauses": (
                        Clause(
                            "A lightmapped one's occlusionTexture is the exception: "
                            "it carries the lightmap ('lightmaps'), not ambient "
                            "occlusion.",
                            (SceneRecords.LIGHTMAPS.key,),
                        ),
                    ),
                },
            ),
            probe=(
                "Radiance bytes (mimeType image/vnd.radiance) in bufferView "
                "'bufferView'"
            ),
            probe_axes="the asset's own axes",
            probe_units="the asset's metres",
            intensity=(
                "that material's 'intensity' in {at}, the scalar that restores "
                "the bake's range"
            ),
            # The copies are a GLB build's (``MeshConvert._embed_highlight_
            # probes``); a finished FBX carries the same copies as media its
            # lightmap record's ``probe.highlights`` names, and that record's
            # own row says so.
            highlights=(
                "'highlights', when present, holds the bake's lights as COPIES "
                "of the probe for groups of receivers: each is a layer "
                "(Radiance in its 'bufferView', the probe's layout and units) "
                "to add to the probe, box-filtered down to the layer's size, "
                "and prefilter as the probe is; the 'materials' it names take "
                "that copy in the probe's place, for their diffuse irradiance "
                "and reflections alike, box-projected from the same 'position' "
                "and 'box'. A copy draws each light where that projection shows "
                "it true for the surfaces it serves, so the lights themselves "
                "never ship -- lighting them as well lights the scene twice."
            ),
        ),
    }

    # ---------------------------------------------------------------- build
    @classmethod
    def carrier(cls, name: str) -> HandoffCarrier:
        """The :attr:`CARRIERS` row named *name*.

        Raises:
            ValueError: *name* is not a known carrier.
        """
        found = cls.CARRIERS.get(name)
        if found is None:
            raise ValueError(
                f"HandoffRecord carrier {name!r} is not one of {', '.join(cls.CARRIERS)}."
            )
        return found

    @classmethod
    def instructions(cls, carrier: str) -> str:
        """The reading contract of *carrier*'s block: its opening, the rules
        every carrier shares (:attr:`_READING`), its own, the version check."""
        found = cls.carrier(carrier)
        reading = cls._READING.format(addresses=found.addresses)
        return " ".join(p for p in (found.intro, reading, found.rules, cls._CHECK) if p)

    @classmethod
    def lighting(
        cls,
        carrier: str,
        held: Union[None, Iterable[str], Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """The lighting rows of *carrier*'s ``assembly`` (:attr:`_LIGHTING_ROWS`)
        for a file holding *held*.

        Parameters:
            carrier: A key of :attr:`CARRIERS`.
            held: The file's records, as :meth:`assembly` takes them: a row
                says what only some files need -- the probe, a lightmapped
                material's rule -- where these call for it, and the two rows
                that read the lightmaps list their address where they are
                held. ``None`` states every row whole (the full statement),
                naming no address: it describes no file.
        """
        found = cls.carrier(carrier)
        lightmaps = SceneRecords.LIGHTMAPS.key
        scope = None if held is None else cls.payloads(held)
        at = found.address(lightmaps)
        probe = cls._PROBE.format(
            at=at, image=found.probe, axes=found.probe_axes, units=found.probe_units
        )
        return _filled_rows(
            cls._LIGHTING_ROWS,
            {
                "{probe}": probe,
                "{highlights}": found.highlights,
                "{intensity}": found.intensity.format(at=at),
                "{lightmaps}": at if scope is not None and lightmaps in scope else None,
            },
            scope,
        )

    @classmethod
    def assembly(
        cls, carrier: str, held: Union[None, Iterable[str], Mapping[str, Any]]
    ) -> List[Dict[str, Any]]:
        """The approved scene part by part, in *carrier*'s addresses: the
        file's own content, each record it holds in declaration order (its row
        under the record's key), then the lighting (:meth:`lighting`). Each
        row states what the file's own record calls for and nothing more
        (:meth:`RecordSpec.assembly_for`).

        Parameters:
            carrier: A key of :attr:`CARRIERS`.
            held: The records the file carries (:meth:`payloads`): its
                carrier's mapping of key to payload, or the keys alone, which
                state each held record's row whole. A key no record declares
                is a record another tool wrote: its row says only that it is
                there. ``None`` when they are not known yet -- a GLB's handoff
                is written before its passes add their blocks: no record row,
                and the lighting rows whole, until the step that knows them
                (``MeshConvert._finish_glb_handoff``) names them.
        """
        found = cls.carrier(carrier)
        if held is None:
            return _filled_rows(found.own, {}, None) + cls.lighting(carrier, None)
        scope = cls.payloads(held)
        declared = {
            spec.key: spec
            for spec in SceneRecords.deliverable()
            if spec is not SceneRecords.HANDOFF
        }
        rows = _filled_rows(found.own, {}, scope)
        for key, spec in declared.items():
            if key not in scope:
                continue
            text = (
                (spec.glb.assembly_for(scope[key]) if spec.glb else "")
                if found.blocks
                else spec.assembly_for(scope[key])
            )
            if text:
                rows.append({"part": key, "data": [found.address(key)], "spec": text})
        for key in sorted(set(scope) - set(declared)):
            rows.append(
                {"part": key, "data": [found.address(key)], "spec": cls.UNDECLARED}
            )
        return rows + cls.lighting(carrier, scope)

    @classmethod
    def payloads(cls, held: Union[Iterable[str], Mapping[str, Any]]) -> Dict[str, Any]:
        """The records a file holds, as ``{key: payload}``: of a mapping of
        key (a carrier's channel) to payload -- the JSON text a carrier stores,
        decoded, or a payload already -- every one but what a carrier's keyable
        attribute holds (a number, a bool, nothing: the animated custom
        properties the records beside them describe) and an empty text. Of an
        iterable, every key it names, its payload :attr:`Clause.UNKNOWN` (so
        each clause looking into it applies), as is a text that is not JSON or
        holds a JSON string (nothing to look into). The handoff itself aside.
        Reading its own result again changes nothing."""
        if not isinstance(held, Mapping):
            return {key: Clause.UNKNOWN for key in held if key != cls.KEY}
        payloads: Dict[str, Any] = {}
        for key, value in held.items():
            if key == cls.KEY:
                continue
            if isinstance(value, str):
                if not value:
                    continue
                try:
                    value = json.loads(value)
                except ValueError:
                    value = Clause.UNKNOWN
                if isinstance(value, str):
                    value = Clause.UNKNOWN
            if value is None or isinstance(value, (bool, int, float)):
                continue
            payloads[key] = value
        return payloads

    @classmethod
    def carried(cls, channels: Union[Iterable[str], Mapping[str, Any]]) -> List[str]:
        """The keys of the records a ``data_export`` carrier holds
        (:meth:`payloads`), the handoff itself aside."""
        return sorted(cls.payloads(channels))

    @classmethod
    def block(
        cls,
        carrier: str,
        held: Union[None, Iterable[str], Mapping[str, Any]],
        source: Optional[Mapping[str, Any]] = None,
        rendering: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """*carrier*'s handoff block for a file holding *held*.

        Parameters:
            carrier: A key of :attr:`CARRIERS`.
            held: The records the file carries, with their payloads where the
                writer has them (:meth:`assembly`); ``None`` when they are not
                known yet, which keeps the whole recipe for the step that knows
                them to gate (:meth:`rendering_policy`).
            source: Where the file came from -- ``{"application", "version",
                "scene"}`` (:class:`HandoffSource`): its declared keys alone,
                each value a string, empty entries dropped; one naming no
                application leaves ``extras`` out.
            rendering: The export's choices over the lighting recipe
                (:meth:`rendering_policy`); ``None`` publishes it as it stands.
        """
        block: Dict[str, Any] = {
            "version": cls.VERSION,
            "instructions": cls.instructions(carrier),
            "assembly": cls.assembly(carrier, held),
            "rendering": cls.rendering_policy(rendering, held),
        }
        # Where the file came from, as its shape declares it: whatever the
        # writer hands over (a version as a number, a key no reader knows),
        # the block is one HandoffSource admits -- and a source naming no
        # application, which it is stated by, is none.
        given = HandoffSource.prune(dict(source)) if isinstance(source, Mapping) else {}
        source = {k: str(v) for k, v in given.items() if v is not None and v != ""}
        if source.get("application"):
            block["extras"] = {"source": source}
        return block

    # ------------------------------------------------------------ rendering
    @classmethod
    def rendering_policy(
        cls,
        overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
        held: Union[None, Iterable[str], Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """:attr:`RENDERING_POLICY` with a deliverable's own choices laid over
        it, as a file holding *held* uses it.

        *overrides* is ``{section: {field: value}}`` -- what the export that
        made the deliverable decided (``ExportContext.rendering``), published in
        its handoff so any reader, the reference viewer included, lights it the
        way it was approved. Only a field the policy declares can be
        overridden: a misspelt one would publish a value no reader looks for
        while the field it meant kept its default.

        *held* is the file's records, as :meth:`assembly` takes them: a
        setting its content does not use (:attr:`RENDERING_NEEDS`) is left
        out. ``None`` keeps every setting (the whole recipe).

        Raises:
            ValueError: an override names a section or field the policy does
                not declare.
        """
        policy = copy.deepcopy(cls.RENDERING_POLICY)
        for section, fields in (overrides or {}).items():
            target = policy.get(section)
            if not isinstance(target, dict):
                raise ValueError(
                    f"Rendering override section {section!r} is not one of "
                    f"HandoffRecord.RENDERING_POLICY's ({', '.join(policy)})."
                )
            unknown = sorted(set(fields or {}) - set(target))
            if unknown:
                raise ValueError(
                    f"Rendering override {section}.{', '.join(unknown)} is "
                    "not a field of HandoffRecord.RENDERING_POLICY."
                )
            target.update(fields or {})
        if held is not None:
            scope = cls.payloads(held)
            for (section, *field), path in cls.RENDERING_NEEDS.items():
                if Clause.holds(scope, path):
                    continue
                if field:
                    policy.get(section, {}).pop(field[0], None)
                else:
                    policy.pop(section, None)
        return policy

    @classmethod
    def overrides(
        cls, rendering: Optional[Mapping[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """The settings of a published *rendering* that differ from the
        policy's (:meth:`rendering_values`): what an export chose, read back
        so a block built from another states the same lighting."""
        policy = cls.rendering_values()
        found: Dict[str, Dict[str, Any]] = {}
        for section, fields in cls.rendering_values(rendering or {}).items():
            for field, value in fields.items():
                if value != policy.get(section, {}).get(field):
                    found.setdefault(section, {})[field] = value
        return found

    @classmethod
    def rendering_values(
        cls, rendering: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """The settings of *rendering* -- :attr:`RENDERING_POLICY` by default,
        or a published block -- by section: each value of the kind its field's
        policy value is (a number, a vector of as many numbers, an identifier
        such as ``"ACESFilmic"``). What a runtime reads and an export can
        choose; a value of another kind is left out, so a block passes its
        settings, never a sentence, to another (``GlbPipeline``), and the C#
        constants unitytk compiles are generated from this
        (``sync_scene_records.py``)."""

        def kind(value: Any) -> Any:
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                return "number"
            if isinstance(value, (list, tuple)):
                numeric = bool(value) and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in value
                )
                return ("vector", len(value)) if numeric else None
            if isinstance(value, str) and value and not any(c.isspace() for c in value):
                return "name"
            return None

        source = cls.RENDERING_POLICY if rendering is None else rendering
        found: Dict[str, Dict[str, Any]] = {}
        for section, fields in cls.RENDERING_POLICY.items():
            stated = source.get(section) if isinstance(source, Mapping) else None
            if not isinstance(fields, Mapping) or not isinstance(stated, Mapping):
                continue
            values = {
                field: stated[field]
                for field, default in fields.items()
                if field in stated
                and kind(default) is not None
                and kind(stated[field]) == kind(default)
            }
            if values:
                found[section] = values
        return found
