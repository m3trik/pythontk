# !/usr/bin/python
# coding=utf-8
"""Scene data carried into the GLB: the scene-sidecar envelope, the FBX
handoff, and the ``data_export`` channels.

The sidecar envelope carries what FBX translation drops (the authored
material state and the rendering policy the deliverable was approved under)
into the glTF root ``extras``; the handoff is the standalone-reader contract;
the ``data_export`` channels are the scene records a DCC publishes as FBX user
properties. :meth:`verify_glb` checks a delivered GLB against the envelope it
carries.

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through ``cls``.
"""

import copy
import hashlib
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Union

# Eager: the channel keys below are SceneRecords' own, read at class
# definition.
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit, GlbTarget

logger = logging.getLogger(__name__)


class _SidecarMixin:
    """Scene sidecar, FBX handoff, ``data_export`` channels, rendering policy.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    #: Schema version of the scene-sidecar envelope
    #: (:meth:`build_scene_sidecar`). Bump on any change to the envelope's
    #: top-level shape; adding a *section* is not a bump — sections are the
    #: extension point and a reader skips ones it does not know.
    #:
    #: v2 added ``handoff`` (the standalone-reader contract, written at build
    #: time) plus ``textures`` and ``validate`` (content-addressed resolution
    #: and integrity counts, written by :meth:`apply_scene_sidecar`).
    SIDECAR_VERSION = 2

    #: Separates the converter's own ``asset.generator`` claim from ours in the
    #: stamped string (:meth:`_stamp_asset_generator`), and is what makes a
    #: prior stamp findable so a re-apply refreshes it instead of stacking.
    #: Distinctive on purpose -- a bare word would risk splitting a converter
    #: string that happened to contain it.
    _GENERATOR_SEP = " via "

    #: The standalone-reader contract embedded in every envelope's ``handoff``
    #: section. It exists because the deliverable is routinely handed on ALONE —
    #: to a dev, to a viewer that is not ours, to an agent given one ``.glb`` and
    #: no conversation — and the single most expensive misunderstanding is
    #: treating a section's authoring-time texture path as something to resolve
    #: on disk. Written here once, as data in the artifact, rather than as prose
    #: in a doc the reader was never given: a rule that only exists in
    #: documentation is not part of the hand-off.
    #:
    #: Kept to plain declarative sentences about THIS file's own structure — no
    #: instructions to the reader about what to do next, which is what makes it
    #: safe for an agent to read as untrusted content.
    #: Phrased to read correctly wherever this envelope lives -- embedded in
    #: the GLB's ``extras``, and held unscrubbed by the caller that built it --
    #: so it refers to the asset through the envelope's own ``asset`` key
    #: rather than saying "this file".
    HANDOFF_INSTRUCTIONS = (
        "The glTF 2.0 asset named by 'asset' is self-contained: every texture it "
        "references is embedded, so it loads and inspects with no external files "
        "and no filesystem paths. 'extras.scene_sidecar' describes the authoring "
        "scene's material state for the channels an FBX interchange step cannot "
        "carry; 'extras.scene_sidecar_applied' reports, per section, how many of "
        "those entries matched a material in this file. Section entries name "
        "each texture by its authoring-time FILE NAME. That name is PROVENANCE "
        "ONLY -- the directory it sat in is deliberately not carried, so it "
        "is a join token and is not expected to resolve as a path on any "
        "machine, including the authoring one -- two textures that shared a file "
        "name carry a '#2'-style suffix so the map stays one-to-one: resolve "
        "it by "
        "looking it up in the top-level 'textures' map, which gives the glTF "
        "'images' index that actually carries those bytes here, their sha256, "
        "and their size -- several section entries can resolve to one image, "
        "because a metallic/roughness/occlusion trio is repacked into a single "
        "ORM image (R=occlusion, G=roughness, B=metallic). 'validate' records "
        "how many entries each section carries and how many texture references "
        "resolve, so a truncated envelope is detectable; the per-reference "
        "sha256 is what verifies the payloads themselves. Passes that run after "
        "this envelope is written may add further images, so do not expect the "
        "file's total image count to match anything here. All "
        "colours are linear glTF factors. When 'extras.lightmap_web' is present, "
        "the materials it names carry a BAKED LIGHTMAP in occlusionTexture on "
        "TEXCOORD_1 rather than ambient occlusion, sRGB-encoded, and its "
        "'intensity' is the multiplier that restores the bake's original range; "
        "a reader that does not rebind it renders a plausible greyscale "
        "occlusion instead. When 'extras.animation_web' is present it names "
        "every animation in this file by its index, says which ones a shot "
        "declared, and gives the frame range and 'offset' (seconds) each "
        "declared clip occupied on the authoring timeline, since every clip's "
        "own keyframe times are rebased to zero; 'default_clip' is the one a "
        "player opens on, which is not necessarily animations[0] -- an FBX "
        "interchange step can retain a whole-timeline take alongside the split "
        "ones. 'handoff.rendering' records the lighting setup the "
        "reference viewer used to produce the look this asset was approved in: "
        "the asset carries no lights of its own, so a viewer that lights it "
        "differently renders something different without either side being "
        "wrong. Check 'version' against the schema you expect."
    )

    #: Schema version of the FBX-side handoff block (:meth:`build_fbx_handoff`).
    #: Separate from :attr:`SIDECAR_VERSION`: that versions the GLB envelope
    #: this block does not live in, and tying them would force a bump on one
    #: carrier every time the other changed.
    FBX_HANDOFF_VERSION = SceneRecords.HANDOFF.version

    #: ``data_export`` channel the FBX handoff block is published on. A channel
    #: like any other, so it rides into the FBX as a user property with no
    #: export-path change and no second carrier -- the deliverable has one
    #: in-band metadata node and this joins it.
    FBX_HANDOFF_CHANNEL = SceneRecords.HANDOFF.key

    #: What each known ``data_export`` channel holds, for the block's ``reads``
    #: map -- each record's own declaration (``SceneRecords``), so a producer
    #: added there is described here with no edit. Descriptions only: the
    #: channel LIST is taken from the carrier at stamp time, so the block never
    #: claims a channel the file lacks, nor omits one it has.
    FBX_HANDOFF_CHANNELS: Dict[str, str] = {
        spec.key: spec.description
        for spec in SceneRecords.deliverable()
        if spec is not SceneRecords.HANDOFF
    }

    #: The standalone-reader contract for an **FBX** deliverable -- the twin of
    #: :attr:`HANDOFF_INSTRUCTIONS`, kept beside it so the two carriers'
    #: accounts of the same pipeline cannot drift.
    #:
    #: Same rules as the glTF text and for the same reasons: plain declarative
    #: sentences about this file's own structure, no instructions to the reader
    #: about what to do next (which is what makes it safe for an agent to read
    #: as untrusted content), and it names the asset through the block's own
    #: ``asset`` key rather than saying "this file".
    #:
    #: The load-bearing sentence is the one about lightmaps NOT being embedded.
    #: An FBX embeds what its MATERIALS reference, and a lighting-only bake
    #: deliberately does not wire its map into a material -- that is the point,
    #: the PBR material survives the bake -- so the maps the manifest names are
    #: the one part of the deliverable that genuinely does not travel with it.
    #: Measured on a delivered room: 23.0 MB of the 23.5 MB file is embedded
    #: material textures and not one byte of it is the lightmap. Saying so in
    #: the file is what stops that being a surprise the recipient has to be
    #: told about out of band.
    #:
    #: Unlike the glTF text this refers to "the FBX carrying this block" rather
    #: than to an ``asset`` key. The block is stamped by an export PREPARER,
    #: which runs before any FBX path is chosen -- the session hook fires for
    #: File > Export and the Game Exporter too -- so an asset name here could
    #: only ever have been the SCENE's, naming a ``.ma`` as though it were the
    #: deliverable. The authoring scene rides under ``source`` instead, where
    #: it is provenance rather than a false identity.
    #: The text itself lives with the record declarations
    #: (``SceneRecords.HANDOFF_INSTRUCTIONS``): the snapshot that commits a
    #: carrier stamps it, and this name is the GLB-side reference to the same
    #: sentence.
    FBX_HANDOFF_INSTRUCTIONS = SceneRecords.HANDOFF_INSTRUCTIONS

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
    #: it a recipient reasonably adds a normal key light, or
    #: an ambient term, and washes out every baked surface -- which reads as a
    #: bake regression and sends them back to the baker, where nothing is
    #: wrong.
    #:
    #: ``test_preview_server`` pins these against the viewer's literals, and
    #: ``test_preview_viewer_live`` against the pixels it draws, so the
    #: published contract cannot drift from what the viewer actually does.
    RENDERING_POLICY: Dict[str, Any] = {
        "renderer": {
            "toneMapping": "ACESFilmic",
            "toneMappingExposure": 1.0,
            "outputColorSpace": "srgb",
        },
        "environment": {
            "source": "three.js RoomEnvironment",
            "prefilter": "PMREM",
            "prefilterBlur": 0.04,
            "intensity": 1.0,
            "note": (
                "The main light for everything that is not baked, and the "
                "source of the reflections on everything that is -- scaled "
                "there by lightmappedMaterials.envMapIntensity."
            ),
        },
        "keyLight": {
            "type": "directional",
            "color": "#ffffff",
            "intensity": 0.9,
            "position": [3, 6, 4],
            "disabled_when": "any material in the asset is lightmapped",
            "note": (
                "Scene-wide, so it cannot be spared the baked geometry it would "
                "contradict; it goes off entirely once anything is baked. Its "
                "direction still orients lightmappedMaterials.normalRelief."
            ),
        },
        "lightmappedMaterials": {
            "envMapTerms": "specular",
            "envMapIntensity": 0.25,
            "lightMapIntensity": (
                "per material, from extras.lightmap_web.materials[<name>].intensity"
            ),
            "normalRelief": (
                "Where a baked material carries a normal map, its lightmap "
                "texel is scaled by (dot(N, L) * 0.5 + 0.5) / "
                "(dot(Ng, L) * 0.5 + 0.5): N the normal-mapped normal, Ng the "
                "surface's own, L the unit direction toward keyLight.position "
                "-- reflected across the surface's plane wherever the surface "
                "faces away from it, since a bake's light reached the surface "
                "from its front. Exactly 1 where the map is flat, so "
                "normalTexture.scale 0 is "
                "the pure bake. A lightmap carries no direction, and without "
                "this a normal map on a baked surface shows only in its "
                "reflections."
            ),
            "note": (
                "The bake already holds the diffuse lighting (every light and "
                "the sky), so the environment's irradiance is NOT added on top: "
                "that double-counts it and reads as a washed-out bake. The "
                "environment keeps only its specular term -- the "
                "reflection-probe role, and what a native lightmap path such as "
                "Unity's does with a lightmapped renderer -- at envMapIntensity "
                "times the environment's own intensity. The environment is a "
                "studio, brighter than the room the bake lit, so at full "
                "strength its reflections lift every dark glossy surface "
                "(measured on a production room: the darkest machine surfaces "
                "at display level 0.06 baked alone, 0.22 with full reflections, "
                "0.11 at a quarter). The level is the export's choice. Un-baked "
                "materials in the same asset take the environment whole, "
                "diffuse and specular."
            ),
        },
    }

    @classmethod
    def rendering_policy(
        cls, overrides: Optional[Mapping[str, Mapping[str, Any]]] = None
    ) -> Dict[str, Any]:
        """:attr:`RENDERING_POLICY` with a deliverable's own choices laid over it.

        *overrides* is ``{section: {field: value}}`` -- what the export that
        made the deliverable decided (``ExportRun.rendering``), published in
        its handoff so any reader, the reference viewer included, lights it
        the way it was approved. Only a field the policy declares can be
        overridden: a misspelt one would publish a value no reader looks for
        while the field it meant kept its default.

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
                    f"MeshConvert.RENDERING_POLICY's ({', '.join(policy)})."
                )
            unknown = sorted(set(fields or {}) - set(target))
            if unknown:
                raise ValueError(
                    f"Rendering override {section}.{', '.join(unknown)} is not a "
                    "field of MeshConvert.RENDERING_POLICY."
                )
            target.update(fields)
        return policy

    #: Sidecar section -> the writer that applies it to a GLB, in application
    #: order. This is the applier column of the scene-data grid: a new kind of
    #: extended scene setup is one more section in the DCC-side reader
    #: (mayatk/blendertk ``SceneState``) plus one more row here — no method
    #: edits anywhere else. Both current rows exist because FBX loses the
    #: channel for every shader that is not the host's own legacy model
    #: (measured on Maya 2025: an aiStandardSurface arrives with no emissive
    #: at all and a flat white base colour). Base colour runs first only for
    #: tidiness; the two touch disjoint fields.
    SIDECAR_APPLIERS: Dict[str, str] = {
        "base_color": "set_glb_base_color",
        "emissive": "set_glb_emissive",
        "metallic_roughness": "set_glb_metallic_roughness",
        "alpha_mode": "set_glb_alpha_mode",
    }
    #: Root-extras key holding :meth:`apply_scene_sidecar`'s per-section
    #: outcome. Its presence is also the fact "the envelope has been applied
    #: to this file", which a later stage reads to avoid applying it twice.
    SIDECAR_APPLIED_KEY = "scene_sidecar_applied"

    @classmethod
    def build_scene_sidecar(
        cls,
        sections: Optional[Dict[str, Any]],
        source: Dict[str, str],
        asset: Optional[str] = None,
        rendering: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Wrap *sections* in the versioned scene-sidecar envelope.

        The one place the envelope schema exists — every producer (the WebXR
        preview bridges, the scene exporters' GLB tasks) builds through this
        rather than shaping the dict itself, so the schema cannot fork between
        packages. The top level is the frozen contract a standalone reader (a
        dev tool holding only the deliverable and this data) parses against::

            {
              "version": 2,              # SIDECAR_VERSION
              "source": {"application": "maya", "version": "2025"},
              "asset": "<payload basename>",
              "color_space": "linear",   # every color below, as the appliers
                                         #   also expect (glTF factors)
              "sections": {...},         # the extension point
              "handoff": {               # the standalone-reader contract
                "instructions": "<HANDOFF_INSTRUCTIONS>",
                "reads": {...}           # extras key -> what it holds
              }
            }

        :meth:`apply_scene_sidecar` adds two more top-level keys when it embeds
        the envelope, because only it can know them: ``textures`` (each
        authoring path -> the glTF image index and sha256 actually carrying it)
        and ``validate`` (the counts the envelope was written against).

        Scope note — this is *not* a second channel registry beside the DCC
        packages' ``DataNodes``. Tool-authored semantic metadata (shots, audio
        events, lightmap manifests, ...) rides **inside** the FBX as user
        properties on the ``data_export`` carrier; the sidecar carries only
        repairs for what the FBX *format* mistranslates about the scene's
        literal content, derived scene-read-only at push/export time. A
        section must never duplicate a ``DataNodes`` channel — one home per
        section per deliverable.

        Parameters:
            sections: ``{section: data}`` from a DCC-side ``SceneState``
                reader. ``None`` or empty still builds an envelope — an empty
                ``sections`` is itself information (nothing needed repair).
            source: Producer identity, e.g.
                ``{"application": "maya", "version": "2025"}``.
            asset: Basename of the deliverable this envelope belongs to.
            rendering: The export's choices over the lighting recipe
                (:meth:`rendering_policy`), published as ``handoff.rendering``.
                ``None`` publishes the policy as it stands.
        """
        return {
            "version": cls.SIDECAR_VERSION,
            "source": source,
            "asset": asset,
            "color_space": "linear",
            "sections": dict(sections or {}),
            "handoff": {
                "instructions": cls.HANDOFF_INSTRUCTIONS,
                # Where the rest of the self-description lives. Derived from the
                # registries rather than listed by hand, so a section or a
                # manifest added later cannot silently fall out of the contract.
                "reads": {
                    "extras.scene_sidecar": "this envelope",
                    "extras.scene_sidecar_applied": "per-section apply outcome",
                    f"extras.{cls.LIGHTMAP_WEB_KEY}": (
                        "materials whose occlusion carrier is a baked lightmap"
                    ),
                    f"extras.{cls.ANIMATION_WEB_KEY}": (
                        "the file's animation clips: which are declared shots, "
                        "their authoring frame ranges, and which to open on. "
                        "A clip's STEP 'scale' channels driving a node to zero "
                        "are its VISIBILITY, not authored scale -- glTF has no "
                        "visibility channel, so keyed visibility is carried "
                        "that way and plays with no extension. Each clip's "
                        "'zero_frame' is the authored frame its t=0 sits on "
                        "(NOT 'start_frame': a take is rebased to its first "
                        "key, so the two differ by the lead-in) -- it is what "
                        "converts a playhead to the frame numbers this block "
                        "quotes. Authored alpha fades are KHR_animation_pointer "
                        "channels on each faded subtree's own material "
                        "(extensionsUsed, never required); a runtime without "
                        "the extension can play them by binding each channel "
                        "to that material's alpha, which is what the preview "
                        "page does"
                    ),
                    f"extras.{cls.SHADOW_WEB_KEY}": (
                        "the scene's shadow-rig planes: per plane the glTF node "
                        "indices of the plane, its source and its contact, the "
                        "projection model's constants (DCC units; unit_scale is "
                        "metres per unit), the texture index of its colour map "
                        "and, when packed or baked, of its atlas and horizon "
                        "map, with rects in glTF top-left space. A runtime "
                        "evaluates ShadowProjection.model per frame from the "
                        "source and contact nodes and places each plane; a "
                        "reader without one leaves the imported keys and the "
                        "silhouette in the base colour, which is the fallback"
                    ),
                    f"extras.{cls.ARTICULATION_WEB_KEY}": (
                        "the scene's articulated rigs: per rig its joints -- the "
                        "glTF node index, the parent joint, the rest translate "
                        "and rotation in the parent's space, the rotate order "
                        "and the channels (rx/ry/rz degrees, tx/ty/tz a slide "
                        "along the rest axes) with their limits -- and the "
                        "parts a hand grabs, each with its node index and the "
                        "joint it rides. A runtime poses a joint as rotation = "
                        "rest (x) Euler(channels), translation = rest + "
                        "rest.(tx, ty, tz), measures its own unit factor off "
                        "the rest translations, and solves a grab with "
                        "pythontk's ArticulationModel (ported in the preview's "
                        "articulated_rig.js); a reader without one plays the "
                        "clips, which are the joints' own keys"
                    ),
                },
                "sections": sorted(cls.SIDECAR_APPLIERS),
                # How to LIGHT what the sections describe. The rest of this
                # envelope says what the asset is; without this, a recipient can
                # rebuild every material correctly and still not reproduce the
                # look, because the lighting lives in the viewer rather than in
                # the file. An input like the sections: what the export chose
                # (a baked-reflection level), never patched on afterwards.
                "rendering": cls.rendering_policy(rendering),
            },
        }

    @classmethod
    def strip_fbx_handoff(cls, gltf: dict) -> int:
        """Drop the FBX's handoff block from a converted glTF's node extras.

        FBX2glTF transcribes every user property on the ``data_export`` carrier
        into ``nodes[N].extras.fromFBX.userProperties`` (probe-measured), so the
        block :meth:`build_fbx_handoff` stamps for the FBX arrives inside the
        GLB as well -- where it is not provenance but a WRONG self-description:
        it states that the lightmaps the manifest names are not embedded, which
        is true of the FBX and false of a GLB that embeds them, and its
        ``reads`` map names ``data_export.<channel>`` paths that do not exist in
        a glTF. A deliverable carrying two accounts of itself, one of them
        wrong, is worse than one carrying none -- and the GLB has its own, in
        ``extras.scene_sidecar.handoff``.

        Removed even when no sidecar was applied (a bare conversion): "no
        self-description" is a recoverable state, "a confidently wrong one" is
        not. Every other transcribed channel is left alone -- ``lightmap_metadata``
        is this applier's own designed input, and the per-object markers are
        read by consumers outside this repo.

        Returns:
            How many nodes were stripped.
        """
        stripped = 0
        for node in gltf.get("nodes") or []:
            props = ((node.get("extras") or {}).get("fromFBX") or {}).get(
                "userProperties"
            )
            if isinstance(props, dict) and props.pop(cls.FBX_HANDOFF_CHANNEL, None):
                stripped += 1
        return stripped

    @classmethod
    def build_fbx_handoff(
        cls,
        channels: Union[Iterable[str], Mapping[str, Any]],
        source: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """The standalone-reader contract for an FBX, ready to publish.

        The FBX twin of the ``handoff`` section :meth:`build_scene_sidecar`
        puts in a GLB. Both deliverables of a pair are routinely handed on
        alone -- the GLB to a web consumer, the FBX to an engine -- and until
        now only one of them could be read without a covering note.

        Deliberately NOT the scene-sidecar envelope. That envelope carries
        repairs for what the FBX *format* mistranslates, which is meaningless
        inside the FBX itself; and this package's standing split (see
        :meth:`build_scene_sidecar`'s scope note) is that FBX-side metadata
        rides as ``data_export`` channels while the sidecar is the GLB's. This
        is one more channel on the carrier that already exists, not a second
        carrier and not a companion file.

        The block itself is built by :meth:`SceneRecords.handoff_block`, beside
        the record declarations it describes; the export snapshot stamps it
        on every commit, and this name is the GLB-side entry point to the
        same builder.

        Parameters:
            channels: The channels actually present on the carrier --
                normally ``DataNodes.dump()["data_export"]``, a mapping of
                name to stored value, of which only the STRING values are
                described (the carrier also holds keyable float attrs, and
                the block's own text says every channel value is a JSON
                string); an iterable of names is taken as given.
            source: Producer identity and provenance, e.g. ``{"application":
                "maya", "version": "2025", "scene": "PROD_ROOM.ma"}``. No
                asset name is carried: this is stamped before any FBX path is
                chosen, so the only name available would be the scene's.

        Returns:
            The block, JSON-serializable. Empty dict when *channels* holds
            nothing to describe -- a carrier with no metadata has no handoff
            to make, and stamping one would put a lone self-referential
            channel in an otherwise empty node.
        """
        return SceneRecords.handoff_block(channels, source)

    @staticmethod
    def _sidecar_section_scope(present: Set[str], data: Any) -> Optional[int]:
        """How many of *data*'s entries name a material *present* in the GLB.

        ``None`` when the section is not a map keyed by material name -- the
        registry is an extension point, so a future section may be keyed on
        anything, and reporting every entry of one as "not in this export"
        would be worse than not scoping it at all. The test is an intersection
        rather than a type check: a material-keyed section on a GLB that
        carries none of its names is indistinguishable from a section keyed on
        something else, and both want the same answer -- do not scope this.
        """
        if not isinstance(data, dict):
            return None
        overlap = len(present.intersection(map(str, data)))
        return overlap or None

    @classmethod
    def apply_scene_sidecar(
        cls, glb: GlbTarget, sidecar: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Apply a scene-sidecar envelope to a GLB and embed it in its extras.

        Every section is dispatched through :attr:`SIDECAR_APPLIERS` against
        **one** open edit session, then the envelope itself (plus the
        per-section outcome summary) is written into the glTF root ``extras``
        — so the artifact leaves self-describing: what the scene authored
        (``extras["scene_sidecar"]``) and what this pass did about it
        (``extras["scene_sidecar_applied"]``), readable by any glTF tool with
        no side files. :meth:`read_scene_sidecar` is the counterpart.

        A section absent from the envelope is simply skipped, and a section
        failure is logged rather than raised: a deliverable missing one
        section still beats no deliverable. Every failure an applier can
        actually reach (an unreadable texture, an image no decoder wants) is
        handled inside it and reported as a skipped image; the per-section
        catch here is the backstop for anything that is not, and the
        container-level catch reports *every* offered section as failed
        rather than letting sections that "applied" claim a success that
        never reached disk.

        Parameters:
            glb: Path to a binary glTF (.glb), modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.
            sidecar: The envelope (:meth:`build_scene_sidecar`). ``None`` (or
                anything falsy) is a true no-op; an envelope with empty
                *sections* still embeds (sidecar was on, the scene had nothing
                to carry) — the distinction is a real envelope vs no envelope.

        Returns:
            ``{section: outcome}`` — ``"N of M"`` (with a
            ``" (K not in this export)"`` clause when the envelope names
            materials this GLB does not carry), ``"0 of M matched"`` or
            ``"failed (...)"`` per offered section; empty when none offered.
        """
        if not sidecar:
            return {}
        sections = sidecar.get("sections") or {}
        summary: Dict[str, str] = {}
        try:
            with cls.open_glb(glb) as edit:
                present = {
                    str(m.get("name"))
                    for m in (edit.gltf.get("materials") or [])
                    if m.get("name")
                }
                for section, method in cls.SIDECAR_APPLIERS.items():
                    data = sections.get(section)
                    if not data:
                        continue
                    apply = getattr(cls, method)
                    try:
                        applied = apply(edit, data)
                    except (OSError, ValueError) as error:
                        logger.warning("Sidecar %r not applied: %s", section, error)
                        summary[section] = f"failed ({error})"
                        continue
                    # The envelope describes the SCENE and the GLB carries the
                    # exported subset, so entries naming a material that is not
                    # in this file were never in scope. Counting them made a
                    # correct deliverable read "10 of 23" on a production
                    # assembly -- 13 reference/ID materials that never
                    # exported, reported as if 13 repairs had failed. The same
                    # distinction ``apply_glb_lightmaps`` draws with
                    # ``out_of_scope``; ``None`` means the section is not keyed
                    # by material name and the plain count is all there is.
                    in_scope = cls._sidecar_section_scope(present, data)
                    # Counted by NAME, not by record: a by-name writer lands
                    # on every material carrying the name (fade clones, the
                    # converter's own duplicates), and each landing is a
                    # record. The outcome the panel reads is "how many of the
                    # scene's materials were repaired", which is names. A
                    # record without a name -- or not a record at all, from a
                    # writer that reports differently -- counts as one.
                    landed = len(
                        {
                            record.get("material", index)
                            if isinstance(record, dict)
                            else index
                            for index, record in enumerate(applied)
                        }
                    )
                    if in_scope is not None and landed > in_scope:
                        # More landed than the exact-name scope allows, so this
                        # applier does not match on the name alone (a
                        # namespace-tolerant or fuzzy resolver would do this).
                        # The scope model does not describe it, and "12 of 10"
                        # is worse than the plain count -- fall back rather
                        # than print a number that cannot be true.
                        in_scope = None
                    out_of_scope = len(data) - in_scope if in_scope is not None else 0
                    scope = (
                        f" ({out_of_scope} not in this export)" if out_of_scope else ""
                    )
                    if not applied and (in_scope is None or in_scope):
                        # The section was read but nothing landed — almost
                        # always a name mismatch, which the applier has just
                        # logged in full. Only reported as a miss when
                        # something WAS in scope to match.
                        logger.warning(
                            "Sidecar %r matched none of its %s entr(ies) in the GLB.",
                            section,
                            len(data),
                        )
                        summary[section] = f"0 of {len(data)} matched"
                        continue
                    denominator = len(data) if in_scope is None else in_scope
                    logger.info(
                        "Sidecar %r applied to %s of %s in this export%s.",
                        section,
                        landed,
                        denominator,
                        f"; {out_of_scope} scene material(s) not in it"
                        if scope
                        else "",
                    )
                    summary[section] = f"{landed} of {denominator}{scope}"
                # Sweep what the writers above unbound (FBX2glTF's converted
                # ORM after the repack, most often) BEFORE the texture map
                # below records image indices -- pruning renumbers them.
                cls.prune_glb_unreferenced_textures(edit)
                extras = edit.gltf.setdefault("extras", {})
                # A COPY, with the resolution keys added: the caller's envelope
                # is theirs to keep (unscrubbed, useful on the authoring machine),
                # and only this pass can know which glTF image each authoring
                # path became.
                embedded = dict(sidecar)
                embedded["textures"] = cls._sidecar_texture_map(edit)
                # Deliberately NOT the file's total image/texture counts. Later
                # passes add images -- the lightmap applier runs after this in
                # every production path -- so a total stamped here is stale on
                # arrival, and a reader using it as an integrity check would
                # reject every lightmapped deliverable. What is stable is what
                # this envelope itself claims: how many entries each section
                # carries and how many texture references resolve. Those, plus
                # the per-reference digests, are the check worth making.
                # Ship file names, not the authoring machine's directory tree.
                # Runs BEFORE `validate` is sized and before the digests are
                # stamped, so both describe the envelope as delivered.
                cls._scrub_sidecar_paths(embedded)
                embedded["validate"] = {
                    "sections": {
                        name: len(entries)
                        for name, entries in sections.items()
                        # A malformed envelope (a null section, say) is skipped
                        # by the dispatch above; sizing it must not be the thing
                        # that raises -- the container catch here only handles
                        # OSError/ValueError, so a TypeError would abort a whole
                        # apply that was otherwise fine.
                        if hasattr(entries, "__len__")
                    },
                    "textures": len(embedded["textures"]),
                }
                extras["scene_sidecar"] = embedded
                extras[cls.SIDECAR_APPLIED_KEY] = dict(summary)
                cls._stamp_asset_generator(edit, sidecar)
                # The materials this envelope did NOT cover still carry the
                # converter's own packing, which is the one measured to arrive
                # solid white. Checked here because the session is already open
                # and the covered names are in hand; a fully-covered export
                # decodes nothing.
                #
                # Only the DESTRUCTIVE finding is raised at export time. The
                # `unvalidated` half is real coverage information but it is not
                # a defect on its own, and an export-time warning an artist
                # cannot act on is how the actionable ones stop being read --
                # it is reported to the recipient instead, by `verify_glb`.
                suspect = cls.suspect_orm_materials(
                    edit, described=sections.get("metallic_roughness") or ()
                )
                harmful = sorted(
                    name
                    for name, found in suspect.items()
                    if found["finding"] == cls.ORM_FINDING_METALLIC_FULL
                )
                if harmful:
                    logger.warning(
                        "Metallic=1 everywhere on %s material(s) not covered by "
                        "the sidecar: %s. glTF reads metallic from the ORM's blue "
                        "channel, so these render with no diffuse response (black "
                        "under a lightmap) -- name them in the scene sidecar's "
                        "metallic_roughness section, or re-export their source "
                        "maps as RGB.",
                        len(harmful),
                        ", ".join(harmful),
                        extra={"preset": "highlight"},
                    )
                # Every reference is meant to carry a content address; one that
                # cannot be digested is an entry a verifying reader has no way
                # to check, so a shortfall is said out loud rather than shipped
                # as a quietly incomplete map.
                stamped = cls._stamp_sidecar_digests(edit)
                if stamped != len(embedded["textures"]):
                    logger.warning(
                        "Sidecar: %d of %d texture reference(s) could not be "
                        "content-addressed (their image payload was unreadable).",
                        len(embedded["textures"]) - stamped,
                        len(embedded["textures"]),
                    )
                edit.dirty = True
        except (OSError, ValueError) as error:
            logger.warning("Sidecar not applied to %s: %s", glb, error)
            return {
                section: f"failed ({error})"
                for section in cls.SIDECAR_APPLIERS
                if sections.get(section)
            }
        return summary

    @classmethod
    def _stamp_asset_generator(
        cls, edit: "GlbEdit", sidecar: Optional[Dict[str, Any]] = None
    ) -> str:
        """Name this pipeline in the glTF's own ``asset.generator``; return it.

        The one provenance field glTF itself defines, and every viewer and
        inspector already displays -- so it reaches a recipient who opens the
        file in a tool that is not ours and reads nothing else we write. The
        converter's own string is kept and ours appended: FBX2glTF really did
        produce the geometry, and replacing that claim would lose the fact that
        matters most when a mesh arrives wrong.

        Deliberately coarse -- the authoring app + version the envelope already
        names, and this package's version. No host, no user, no paths: a
        generator string travels to whoever gets the file (see
        :meth:`_scrub_sidecar_paths` for the same rule applied to the envelope).

        Idempotent: re-applying an envelope to an already-stamped GLB refreshes
        the stamp rather than appending a second one.
        """
        parts = []
        source = (sidecar or {}).get("source") or {}
        # Joined from what is actually present: an f-string over a missing
        # version interpolates the literal "None" into the MIDDLE of the string,
        # where .strip() cannot reach it ("maya None + pythontk 0.9.24"), and a
        # version naming no application used to be dropped outright.
        named = " ".join(
            str(v) for v in (source.get("application"), source.get("version")) if v
        )
        if named:
            parts.append(named)
        try:  # Deferred: the root package imports this module.
            from pythontk import __version__ as ptk_version

            parts.append(f"pythontk {ptk_version}")
        except ImportError:  # pragma: no cover - a broken install, not a case
            pass
        ours = " + ".join(parts)
        asset = edit.gltf.setdefault("asset", {})
        # Drop a stamp from an earlier run before appending this one, so the
        # string stays "<converter> via <ours>" however many passes touch it.
        prior = (asset.get("generator") or "").split(cls._GENERATOR_SEP)[0].strip()
        # A file whose ONLY generator claim is a previous stamp of ours has no
        # separator to split on, so the whole string would come back as "prior"
        # and the next pass would append ours to itself. Our stamp always names
        # this package, which is what makes it recognisable across versions.
        if "pythontk " in prior:
            prior = ""
        asset["generator"] = f"{prior}{cls._GENERATOR_SEP}{ours}" if prior else ours
        edit.dirty = True
        return asset["generator"]

    @classmethod
    def _scrub_sidecar_paths(cls, embedded: Dict[str, Any]) -> int:
        """Strip authoring directories out of *embedded* in place; return the count.

        The envelope names every texture by the absolute path it had on the
        authoring machine -- in both the ``textures`` map's keys and the section
        entries that resolve through them. That is fine locally and is a
        disclosure in a hand-off: a GLB sent to an external developer spells out
        the client folder tree it was built from.

        The directory is what leaks; the file name is not, and the join does not
        need either. Both sides of the join are the SAME string, so rewriting
        both consistently keeps a plain string lookup working -- no schema
        version bump, and every existing reader keeps resolving. (The content
        address stamped by :meth:`_stamp_sidecar_digests` is the identity that
        actually verifies bytes; this name is only a join token.)

        Two distinct paths CAN share a basename, so a collision is disambiguated
        with a ``#N`` suffix assigned in sorted-path order -- deterministic, and
        it keeps the map injective, which a bare basename would not.

        Only the embedded COPY is scrubbed. The caller's own envelope keeps
        full provenance, which is the half that is useful on the authoring
        machine.
        """
        entries = embedded.get("textures")
        if not isinstance(entries, dict):
            return 0

        mapping: Dict[str, str] = {}
        claimed: Dict[str, str] = {}  # scrubbed name -> the path that took it
        for path in sorted(entries):
            if not isinstance(path, str):
                continue
            name = os.path.basename(path.replace("\\", "/").rstrip("/")) or path
            if claimed.get(name, path) != path:
                stem, dot, ext = name.rpartition(".")
                suffix = 2
                while True:
                    candidate = (
                        "{}#{}{}{}".format(stem, suffix, dot, ext)
                        if dot
                        else "{}#{}".format(name, suffix)
                    )
                    if claimed.get(candidate, path) == path:
                        break
                    suffix += 1
                name = candidate
            claimed[name] = path
            if name != path:
                mapping[path] = name

        if not mapping:
            return 0

        def rewrite(node):
            """Replace any string that IS one of the mapped paths, anywhere."""
            if isinstance(node, dict):
                return {
                    mapping.get(k, k) if isinstance(k, str) else k: rewrite(v)
                    for k, v in node.items()
                }
            if isinstance(node, list):
                return [rewrite(v) for v in node]
            if isinstance(node, str):
                return mapping.get(node, node)
            return node

        for key, value in list(embedded.items()):
            embedded[key] = rewrite(value)
        return len(mapping)

    @classmethod
    def _sidecar_texture_map(cls, edit: "GlbEdit") -> Dict[str, Dict[str, Any]]:
        """``{authoring path: {"image": index}}`` for everything this session embedded.

        The join a standalone reader needs and a path cannot give it: a section
        entry names a texture by the path it had on the authoring machine, which
        resolves nowhere else, so the envelope carries the map from that name to
        the glTF ``images`` index actually holding those bytes.
        :meth:`_stamp_sidecar_digests` fills in the content address.

        Derived wholly from :attr:`GlbEdit.embedded` (source key -> texture
        index), which is why it needs no knowledge of any section's shape and a
        section added later is covered for free. Two wrinkles that shape it:

        * The ORM writer's cache key is the three source paths joined by ``|``,
          because the trio collapses into ONE packed image. Splitting it back
          out is what lets each of ``metallic``/``roughness``/``occlusion``
          resolve -- all three to the same index, which is the truth.
        * ``None`` appears in that join for a slot with no map (``"a|None|b"``),
          and is not a path. A real file named ``None`` is indistinguishable
          here, which costs nothing: it would resolve to the image it is
          actually part of.
        """
        resolved: Dict[str, Dict[str, Any]] = {}
        for cache_key, texture_index in edit.embedded.items():
            image_index = edit.image_for_texture(texture_index)
            if image_index is None:
                continue
            for path in str(cache_key).split("|"):
                if path and path != "None":
                    # setdefault, not assignment: one path CAN legitimately
                    # resolve to two images -- embedded whole for one channel
                    # and folded into an ORM for another -- and a single-valued
                    # map can only say one. First wins, which is well defined
                    # because SIDECAR_APPLIERS fixes the order and the whole
                    # embed is the more direct answer than a channel of a pack.
                    # Assignment made the winner depend on applier order, which
                    # is the kind of thing that changes silently.
                    resolved.setdefault(path, {"image": image_index})
        return resolved

    @classmethod
    def _stamp_sidecar_digests(cls, edit: "GlbEdit") -> int:
        """Refresh the embedded envelope's content addresses; return how many.

        Split from :meth:`_sidecar_texture_map` because the two answer questions
        with different lifetimes. *Which* image carries a path is decided when
        the sidecar is applied; a later pass that renumbers images carries the
        reference along (:meth:`_map_sidecar_image_refs`) without touching its
        bytes. *What those bytes are* changes afterwards:
        :meth:`optimize_glb_textures` resizes and re-encodes every image it
        touches, so a digest taken at apply time describes bytes the delivered
        file no longer contains. So the digest is stamped by whoever last wrote
        the payloads, and this is idempotent so both callers can.

        Reads the envelope out of ``extras`` rather than taking it as an
        argument: the optimize pass has no sidecar in hand and must not need
        one -- it just refreshes whatever the file already declares.
        """
        sidecar = (edit.gltf.get("extras") or {}).get("scene_sidecar")
        entries = sidecar.get("textures") if isinstance(sidecar, dict) else None
        if not isinstance(entries, dict):
            return 0
        images = edit.images
        stamped = 0
        for ref in entries.values():
            if not isinstance(ref, dict):
                continue
            index = ref.get("image")
            if not isinstance(index, int) or not 0 <= index < len(images):
                continue
            payload = edit._image_payload(images[index])
            if not payload:
                continue
            ref["sha256"] = hashlib.sha256(payload).hexdigest()
            ref["bytes"] = len(payload)
            mime = images[index].get("mimeType")
            if mime:
                ref["mimeType"] = mime
            stamped += 1
        if stamped:
            edit.dirty = True
        return stamped

    @classmethod
    def sidecar_foreign_packings(
        cls,
        sidecar: Optional[Dict[str, Any]],
        target: str = "ORM",
        workflow: Optional[str] = None,
    ) -> Dict[str, str]:
        """``{path: map type}`` for envelope textures authored for another engine.

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
        from pythontk.core_utils.engines.textures.map_factory import MapFactory

        paths: List[str] = []
        for entry in ((sidecar or {}).get("sections") or {}).values():
            if not isinstance(entry, dict):
                continue
            for spec in entry.values():
                if isinstance(spec, str):
                    paths.append(spec)
                elif isinstance(spec, dict):
                    paths.extend(v for v in spec.values() if isinstance(v, str))
        return MapFactory.foreign_packings(paths, target=target, workflow=workflow)

    @classmethod
    def read_scene_sidecar(cls, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        """The scene-sidecar envelope embedded in a GLB, or ``None``.

        The consumer half of :meth:`apply_scene_sidecar` — what a downstream
        tool (or a test) calls to get the scene description back out of a
        deliverable with no side files. Reads only the JSON chunk; the
        geometry is never touched.
        """
        with cls.open_glb(glb) as edit:
            return (edit.gltf.get("extras") or {}).get("scene_sidecar")

    @staticmethod
    def _requirement_note(required: Sequence[str]) -> str:
        """:meth:`verify_glb`'s note naming a file's ``extensionsRequired``.

        One definition, because :class:`ExportVerifier` recognises it by value:
        its extension gate states a declared requirement, so its envelope gate
        must not restate the same fact as a warning.
        """
        return (
            "requires a reader supporting "
            + ", ".join(required)
            + " -- a viewer without it must refuse this file"
        )

    @classmethod
    def verify_glb(cls, glb: GlbTarget) -> Dict[str, Any]:
        """Check a delivered GLB against the envelope it carries.

        The reader a *recipient* runs. Everything the envelope promises is
        checkable from the file alone -- that is what ``textures`` (content
        addresses) and ``validate`` (the counts the envelope claims for itself)
        are for -- but until this existed nothing in the ecosystem read either
        back, so a truncated envelope or a payload swapped after the digests
        were stamped arrived indistinguishable from a good one.

        Read-only and side-file-free by design: a recipient on another machine,
        in another DCC, or holding nothing but the ``.glb`` can run it, and it
        can never be what damages the asset it is inspecting.

        Decodes one channel of each ORM image the envelope did not describe
        (see :meth:`suspect_orm_materials`); everything else reads the JSON
        chunk and hashes payloads. That cost is deliberate here and nowhere
        else -- this is an explicit recipient-side call, not an export step.

        Returns:
            A report dict, always carrying ``ok`` (bool), ``problems`` (the
            findings that made ``ok`` False) and ``notes`` (observations that
            did not), plus what was inspected: ``envelope``
            (present/version/source/asset), ``textures`` (``checked``,
            ``verified``, ``mismatched``, ``unresolved``), ``sections``
            (declared vs applied), ``orm`` (per
            :meth:`suspect_orm_materials`, when anything was found),
            ``animation`` (clip counts, rate and default clip, when the file
            carries that block), ``extensions`` (``required`` / ``used`` --
            what a reader must SUPPORT to open the file at all), ``lightmap``
            and ``generator``.
            ``problems`` and ``ok`` are kept
            strictly in step: an empty ``problems`` always means ``ok``. A GLB
            with no envelope is reported, not raised -- an asset from another
            producer is a legitimate thing to point this at.
        """
        report: Dict[str, Any] = {"ok": True, "problems": [], "notes": []}

        def fail(message: str) -> None:
            report["ok"] = False
            report["problems"].append(message)

        with cls.open_glb(glb) as edit:
            extras = edit.gltf.get("extras") or {}
            envelope = extras.get("scene_sidecar")
            report["generator"] = (edit.gltf.get("asset") or {}).get("generator")
            report["lightmap"] = bool(cls._lightmap_web_manifest(edit.gltf))
            # What a reader must SUPPORT to open this at all. `extensionsRequired`
            # is not advice: the spec says a reader that does not implement one
            # of these must refuse the file. A web-delivery GLB requires
            # `EXT_texture_webp` (nothing core-readable survives that pass) and
            # a pure-delivery KTX2 one requires `KHR_texture_basisu`, so the
            # deliverable most likely to reach a third party is exactly the one
            # with a hard prerequisite -- and this report, which exists to tell
            # a recipient what they are holding, did not mention it. Reported
            # rather than failed: the requirement is correct, it just has to be
            # readable without parsing the JSON chunk by hand.
            required = sorted(edit.gltf.get("extensionsRequired") or [])
            report["extensions"] = {
                "required": required,
                "used": sorted(edit.gltf.get("extensionsUsed") or []),
            }
            if required:
                report["notes"].append(cls._requirement_note(required))
            # The spec's own subset rule, and a real failure rather than a note:
            # a file demanding a capability it never declares is invalid glTF
            # and stock validators reject it outright.
            undeclared = sorted(set(required) - set(report["extensions"]["used"]))
            if undeclared:
                fail(
                    f"extensionsRequired names {', '.join(undeclared)}, which "
                    "extensionsUsed does not declare -- invalid glTF"
                )
            # Reported before the envelope gate below: an asset can carry
            # clips and no sidecar (another producer's file, or one converted
            # before the envelope existed), and "how many clips, how many of
            # them shots" is exactly what a recipient asks first of an animated
            # deliverable.
            animation = extras.get(cls.ANIMATION_WEB_KEY)
            if isinstance(animation, dict):
                clips = animation.get("clips") or []
                report["animation"] = {
                    "clips": len(clips),
                    "declared": sum(1 for c in clips if c.get("declared")),
                    "default_clip": animation.get("default_clip"),
                    "fps": animation.get("fps"),
                }
            elif edit.gltf.get("animations"):
                # Not a failure -- the block is only written by our own
                # conversion -- but a note, because a consumer that went
                # looking for shot names here found none.
                report["notes"].append(
                    f"{len(edit.gltf['animations'])} animation(s) with no "
                    f"extras.{cls.ANIMATION_WEB_KEY}: clip identity is names only"
                )

            hollow = [
                str(a.get("name") or f"animations[{i}]")
                for i, a in enumerate(edit.gltf.get("animations") or [])
                if not (a.get("channels") and a.get("samplers"))
            ]
            if hollow:
                # glTF requires BOTH arrays to carry at least one entry, so this
                # is not a quality note -- the file fails validation. Measured
                # on a production deliverable: a take split emits a named
                # AnimStack for a shot whose content is entirely visibility,
                # and the converter writes it out with neither channel nor
                # sampler. A strict reader is entitled to reject the whole file.
                fail(
                    f"{len(hollow)} animation(s) carry no channels or samplers "
                    f"({', '.join(hollow)}) -- glTF requires at least one of "
                    "each, so this file does not validate"
                )

            # The file's own handoff promising clips the file does not carry.
            # Measured on a production deliverable (2026-08-30):
            # the export wrote `data_export.fbx_takes` naming 12 shots while the
            # FBX itself was written with animation disarmed, so the GLB shipped
            # 12 declared clips and ZERO animations -- and every check here
            # passed, because textures, sections and envelope were all sound.
            # A recipient reading the handoff plans for clips that do not exist,
            # which makes this a defect of the artifact, not a note.
            declared_takes = cls._declared_takes(edit.gltf)
            if declared_takes and not (edit.gltf.get("animations") or []):
                fail(
                    f"the handoff declares {len(declared_takes)} take(s) "
                    "on its data_export carrier but the file carries no "
                    "animations -- either the FBX was written with animation "
                    "off (bake/takes disarmed at export) or nothing in the "
                    "exported set is keyed, so every take was empty and pruned"
                )
            if not isinstance(envelope, dict):
                report["envelope"] = None
                fail("no scene-sidecar envelope: nothing here declares itself")
                return report

            version = envelope.get("version")
            report["envelope"] = {
                "version": version,
                "source": envelope.get("source"),
                "asset": envelope.get("asset"),
            }
            # Newer is the case worth saying out loud: this reader would skip
            # whatever the newer schema added and report a clean bill on a file
            # it only partly understood.
            if version != cls.SIDECAR_VERSION:
                fail(
                    f"envelope schema v{version} against this reader's "
                    f"v{cls.SIDECAR_VERSION}"
                )

            declared = envelope.get("sections") or {}
            if not isinstance(declared, dict):
                # Said out loud rather than normalised to {}: quietly treating a
                # block this reader cannot parse as an empty one would report a
                # clean bill on an envelope it never read -- the exact false
                # green this method exists to end.
                fail("envelope 'sections' is not an object")
                declared = {}
            applied = extras.get(cls.SIDECAR_APPLIED_KEY)
            report["sections"] = {
                "declared": {
                    name: len(entries)
                    for name, entries in declared.items()
                    if hasattr(entries, "__len__")
                },
                "applied": applied if isinstance(applied, dict) else {},
            }
            for name, outcome in report["sections"]["applied"].items():
                # The apply pass records its own bad news in the artifact; a
                # verifying reader has to repeat it rather than assume the
                # recipient read the log of a run on someone else's machine.
                # Anchored, not `in`: the summary spells a partial apply "10 of
                # 20", and a bare "0 of" substring test calls that a failure.
                text = str(outcome)
                if text.startswith("failed") or text.startswith("0 of"):
                    fail(f"section {name!r} did not land: {outcome}")

            images = edit.images
            refs = envelope.get("textures") or {}
            if not isinstance(refs, dict):
                fail("envelope 'textures' is not an object")
                refs = {}
            checked = verified = 0
            mismatched: List[str] = []
            unresolved: List[str] = []
            for path, ref in refs.items():
                if not isinstance(ref, dict):
                    continue
                checked += 1
                index, digest = ref.get("image"), ref.get("sha256")
                payload = (
                    edit._image_payload(images[index])
                    if isinstance(index, int) and 0 <= index < len(images)
                    else None
                )
                if not payload or not digest:
                    unresolved.append(path)
                elif hashlib.sha256(payload).hexdigest() != digest:
                    mismatched.append(path)
                else:
                    verified += 1
            report["textures"] = {
                "checked": checked,
                "verified": verified,
                "mismatched": mismatched,
                "unresolved": unresolved,
            }
            if mismatched:
                fail(
                    f"{len(mismatched)} texture reference(s) do not match their "
                    "recorded sha256 -- the payload changed after it was stamped"
                )
            if unresolved:
                fail(
                    f"{len(unresolved)} texture reference(s) resolve to no image "
                    "payload in this file"
                )

            # `validate` is the envelope's claim about ITSELF, so a disagreement
            # means the envelope was truncated or rewritten, not that a texture
            # is missing -- worth separating from the digest failures above.
            claims = envelope.get("validate")
            if isinstance(claims, dict):
                if claims.get("sections") != report["sections"]["declared"]:
                    fail("envelope 'validate' section counts disagree with 'sections'")
                if claims.get("textures") != checked:
                    fail(
                        f"envelope 'validate' claims {claims.get('textures')} texture "
                        f"reference(s), found {checked}"
                    )
            else:
                fail("envelope carries no 'validate' block to check against")

            # `described` is the envelope's own section, so a deliverable that
            # documented every ORM it carries reports nothing here -- and one
            # that carries a binding nothing described says so, which is the
            # only way a mask map packed for another engine (read channel for
            # channel as ORM, and perfectly ordinary-looking data) surfaces at
            # all.
            suspect = cls.suspect_orm_materials(
                edit, described=declared.get("metallic_roughness") or {}
            )
            if suspect:
                report["orm"] = suspect
                by_finding: Dict[str, List[str]] = {}
                for material, found in sorted(suspect.items()):
                    by_finding.setdefault(found["finding"], []).append(material)
                harmful = by_finding.get(cls.ORM_FINDING_METALLIC_FULL)
                unvalidated = by_finding.get(cls.ORM_FINDING_UNVALIDATED)
                if harmful:
                    fail(
                        "metallic=1 everywhere on: "
                        + ", ".join(harmful)
                        + " (no diffuse response; black under a lightmap)"
                    )
                if unvalidated:
                    # A NOTE, not a problem: an ORM the producer never had to
                    # repair is perfectly legitimate, and putting this in
                    # `problems` would make the obvious `if report["problems"]`
                    # read as a defect on a sound deliverable. It is said at all
                    # because nothing here can vouch for its channel layout.
                    #
                    # Grouped BY IMAGE, because the finding is a property of the
                    # image and the lightmap pass clones a material per
                    # instance: on a production room one unvalidated ORM
                    # produced 46 identical lines naming 46 clones of one
                    # material, which is the length that stops a note being
                    # read at all. The per-material detail stays in
                    # ``report["orm"]`` for anything that wants it.
                    by_image: Dict[str, List[str]] = {}
                    for material in unvalidated:
                        image = str((suspect[material] or {}).get("image") or "?")
                        by_image.setdefault(image, []).append(material)
                    described = []
                    for image, names in sorted(by_image.items()):
                        shown = ", ".join(sorted(names)[:3])
                        more = ", ..." if len(names) > 3 else ""
                        described.append(
                            f"{image} ({len(names)} material(s): {shown}{more})"
                        )
                    report["notes"].append(
                        "ORM binding not described by the envelope on "
                        + "; ".join(described)
                        + " -- its channel layout is unverified (a mask map "
                        "packed for another engine reads as ORM here)"
                    )
        return report

    @classmethod
    def data_export_channel(cls, gltf: dict, key: str) -> Optional[Any]:
        """Decoded value of one ``data_export`` channel in a parsed glTF, or ``None``.

        Every in-band metadata system publishes onto the same carrier -- the
        lightmap manifest, the shot definitions, the take list, the audio events
        -- so reading one is the same puzzle every time, and the puzzle is the
        two on-disk SHAPES, not the channel. Hand-rolled per channel, the second
        copy is the one that probes only the nested shape and turns its whole
        feature into a silent no-op on a natively exported GLB.

        Split from :meth:`read_glb_lightmap_manifest` so an applier can read from
        the session it is ALREADY holding -- a second ``open_glb`` on the path
        would re-read and re-parse the file, which is exactly what
        :class:`GlbEdit` exists to avoid.

        Returns whatever the channel decodes to (the producers publish JSON, so
        in practice a dict or a list); a caller wanting one shape checks. An
        unparsable channel is warned and read as absent -- a half-decoded
        manifest is worse than none.
        """
        for node in gltf.get("nodes", []) or []:
            extras = node.get("extras") or {}
            # The same TWO on-disk shapes :meth:`_reconcile_node_markers` walks,
            # and for the same reason: FBX2glTF nests user properties under
            # extras.fromFBX.userProperties (the Maya route), a native glTF
            # export writes them as TOP-LEVEL node extras. Probing only the
            # nested shape makes the whole applier a silent no-op on a natively
            # exported deliverable -- and this is public API, pointed at
            # whatever GLB it is handed, so the shape is not knowable from the
            # call site. Nested first: a file carrying both came through the FBX
            # hop, and that copy is the one the converter transcribed.
            for props in (
                (extras.get("fromFBX") or {}).get("userProperties") or {},
                extras,
            ):
                entry = props.get(key)
                if entry is None:
                    continue
                # FBX2glTF wraps each property as {"type": ..., "value": ...}.
                raw = entry.get("value") if isinstance(entry, dict) else entry
                if isinstance(raw, str) and not raw.strip():
                    # A producer with nothing to publish CLEARS its channel to
                    # "" rather than deleting the attribute, so an empty string
                    # is "absent", not "unparsable".
                    return None
                try:
                    return json.loads(raw) if isinstance(raw, str) else raw
                except (TypeError, ValueError) as error:
                    logger.warning(
                        "Unparsable %s on node %r: %s",
                        key,
                        node.get("name"),
                        error,
                    )
                    return None
        return None

    #: What a converted ``data_export`` carrier node is called, and what
    #: :meth:`overlay_data_export` names one it has to create.
    DATA_EXPORT_NODE = "data_export"

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
        changed = set()
        carrier = None
        nodes = gltf.get("nodes") or []
        for key in channels or {}:
            for node in nodes:
                extras = node.get("extras") if isinstance(node, dict) else None
                if not isinstance(extras, dict):
                    continue
                props = (extras.get("fromFBX") or {}).get("userProperties")
                for holder in (props, extras):
                    if isinstance(holder, dict) and key in holder:
                        del holder[key]
                        changed.add(key)
                        carrier = carrier or node

        written = {k: v for k, v in (channels or {}).items() if v is not None}
        if written:
            if carrier is None:
                carrier = next(
                    (
                        node
                        for node in nodes
                        if isinstance(node, dict)
                        and str(node.get("name") or "").split("|")[-1].split(":")[-1]
                        == cls.DATA_EXPORT_NODE
                    ),
                    None,
                )
            if carrier is None:
                # A selection export ships no carrier when the scene has none;
                # the overlay still has to land where the readers look.
                carrier = {"name": cls.DATA_EXPORT_NODE}
                gltf.setdefault("nodes", []).append(carrier)
                scenes = gltf.get("scenes") or []
                index = gltf.get("scene", 0)
                if isinstance(index, int) and 0 <= index < len(scenes):
                    scenes[index].setdefault("nodes", []).append(len(gltf["nodes"]) - 1)
            extras = carrier.setdefault("extras", {})
            for key, value in written.items():
                # JSON text, the shape every producer publishes and the reader
                # decodes.
                extras[key] = value if isinstance(value, str) else json.dumps(value)
                changed.add(key)
        return sorted(changed)
