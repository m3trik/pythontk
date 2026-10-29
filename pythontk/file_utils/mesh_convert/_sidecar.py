# !/usr/bin/python
# coding=utf-8
"""A GLB's account of itself, and the ``data_export`` channels it is read from.

A GLB states itself as every deliverable does: ``extras.handoff``, built by
``HandoffRecord`` (the contract's one declaration) from the FBX's own ``handoff``
record and named against the blocks the file carries. The ``data_export``
channels are the scene records a DCC publishes as FBX user properties, which
the conversion reads and then drops (:meth:`strip_fbx_records`).
:meth:`verify_glb` checks a delivered GLB against what it says of itself.

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through the facade it holds (``self._mc``).
"""

import copy
import json
import logging
from typing import (
    Any,
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

# Eager: the channel keys below are the registry's and the handoff's own,
# read at class definition.
from pythontk.core_utils.engines.scene_export.handoff_record import HandoffRecord
from pythontk.core_utils.engines.scene_export.scene_records import (
    RecordSpec,
    Scope,
    SceneRecords,
)
from pythontk.core_utils.engines.scene_export.scene_store import SceneStoreBase
from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit, GlbTarget

logger = logging.getLogger(__name__)


class _Sidecar:
    """A GLB's handoff block and the ``data_export`` channels it is read from.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    #: Separates the converter's own ``asset.generator`` claim from ours in the
    #: stamped string (:meth:`_stamp_asset_generator`), and is what makes a
    #: prior stamp findable so a re-apply refreshes it instead of stacking.
    #: Distinctive on purpose -- a bare word would risk splitting a converter
    #: string that happened to contain it.
    _GENERATOR_SEP = " via "

    def __init__(self, mc: type) -> None:
        self._mc = mc  # the facade class: every name it exposes is read through it

    def rendering_policy(
        self,
        overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
        carrier: str = "glb",
    ) -> Dict[str, Any]:
        HandoffRecord.carrier(carrier)
        return HandoffRecord.rendering_policy(overrides)

    #: Sidecar section -> the writer that applied it to a GLB, in application
    #: order: the repairs a host read off the live scene (``SceneState``) for
    #: what the FBX mistranslated. Retired 2026-10-20 with those readers --
    #: the ``materials`` record the FBX carries states every material, and no
    #: host reads a section any more. Until the removal it still applies the
    #: sections of an envelope from a host that predates the record
    #: (:meth:`apply_scene_sidecar`).
    _SIDECAR_APPLIERS: Dict[str, str] = {
        "base_color": "set_glb_base_color",
        "emissive": "set_glb_emissive",
        "metallic_roughness": "set_glb_metallic_roughness",
        "alpha_mode": "set_glb_alpha_mode",
        "unlit": "set_glb_unlit",
    }

    def build_scene_sidecar(
        self,
        sections: Optional[Dict[str, Any]],
        source: Dict[str, str],
        asset: Optional[str] = None,
        rendering: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return {
            "version": 2,
            "source": source,
            "asset": asset,
            "sections": dict(sections or {}),
            "handoff": HandoffRecord.block("glb", None, source, rendering),
        }

    def _handoff_of(
        self, given: Optional[Mapping[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """*given* as a GLB handoff block: a block as it is, an envelope from
        a host that predates the block as the block its source and lighting
        choices make; ``None`` for nothing."""
        if not isinstance(given, Mapping):
            return None
        if given.get("version") == HandoffRecord.VERSION and "instructions" in given:
            return copy.deepcopy(dict(given))
        nested = given.get("handoff")
        if (
            isinstance(nested, Mapping)
            and nested.get("version") == HandoffRecord.VERSION
        ):
            return copy.deepcopy(dict(nested))
        rendering = nested.get("rendering") if isinstance(nested, Mapping) else None
        return HandoffRecord.block(
            "glb", None, given.get("source"), HandoffRecord.overrides(rendering) or None
        )

    def _fbx_handoff(
        self, gltf: dict, rendering: Optional[Mapping[str, Mapping[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """The GLB handoff block the FBX's own ``handoff`` record implies, read
        off the converted file's carrier in the version and shape it declares
        (:meth:`_current_record`): its source, and each lighting value it
        chose, with *rendering* laid over them -- the file's blocks not known
        yet, so the whole recipe (:meth:`_finish_glb_handoff` gates it).
        ``None`` for a file with no record -- a raw export claims no approved
        scene -- or one of another version or shape, which is said."""
        record = self._mc._current_record(gltf, SceneRecords.HANDOFF)
        if record is None:
            return None
        source = GlbEdit.extras_of(record).get("source")
        overrides = HandoffRecord.overrides(record.get("rendering"))
        for section, fields in (rendering or {}).items():
            overrides.setdefault(section, {}).update(fields)
        return HandoffRecord.block("glb", None, source, overrides or None)

    def strip_fbx_records(self, gltf: dict) -> int:
        def ours(key: str, carrier: bool) -> bool:
            return (
                key == self._mc.LIGHTMAP_INFO_KEY
                or key in self._mc.HOST_PROPERTIES
                or (carrier and key in self._RECORD_KEYS)
            )

        changed: Set[int] = set()
        for index, node in enumerate(gltf.get("nodes") or []):
            extras = node.get("extras")
            if not isinstance(extras, dict):
                continue
            carrier = SceneStoreBase.is_carrier(node.get("name"))
            # A native glTF export's records and markers sit at the top.
            drop = [key for key in extras if ours(key, carrier)]
            for key in drop:
                del extras[key]
                changed.add(index)
            nested = extras.get("fromFBX")
            if isinstance(nested, dict):
                props = nested.get("userProperties")
                kept = (
                    {k: v for k, v in props.items() if not ours(k, carrier)}
                    if isinstance(props, dict)
                    else {}
                )
                # Only the artist's attributes: FBX2glTF's flags beside them
                # are its own notes.
                if kept:
                    if nested != {"userProperties": kept}:
                        extras["fromFBX"] = {"userProperties": kept}
                        changed.add(index)
                else:
                    del extras["fromFBX"]
                    changed.add(index)
            if not extras:
                del node["extras"]
        notes = 0
        for kind, key in (("materials", "fromFBX"), ("animations", "zero_frame")):
            for item in gltf.get(kind) or []:
                extras = item.get("extras") if isinstance(item, dict) else None
                if isinstance(extras, dict) and key in extras:
                    del extras[key]
                    if not extras:
                        del item["extras"]
                    notes += 1
        bare = {
            index
            for index, node in enumerate(gltf.get("nodes") or [])
            if SceneStoreBase.is_carrier(node.get("name"))
            and not any(k in node for k in ("mesh", "camera", "skin", "children"))
        }
        if bare:
            self._mc._remove_glb_nodes(gltf, bare)
        return len(changed | bare) + notes

    #: Every channel a carrier can hold, read from the declarations: what a
    #: native glTF export's top-level node extras name on a carrier
    #: (:meth:`strip_fbx_records`).
    _RECORD_KEYS = frozenset(spec.key for spec in SceneRecords.deliverable())
    #: The user properties a host writes on every node of its own accord,
    #: never an artist's: Maya's current UV set, 3ds Max's object handle.
    #: Every other user property is content (:meth:`strip_fbx_records`).
    HOST_PROPERTIES = frozenset({"currentUVSet", "MaxHandle"})

    def finish_glb(
        self,
        glb: GlbTarget,
        rendering: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> bool:
        with self._mc.open_glb(glb) as edit:
            changed = self._mc._lift_scene_blocks(edit.gltf)
            handoff = self._fbx_handoff(edit.gltf, rendering)
            if handoff:
                self._mc.embed_glb_handoff(edit, handoff)
                changed = True
            elif changed:
                self._mc._finish_glb_handoff(edit)
            changed = bool(self._mc.strip_fbx_records(edit.gltf)) or changed
            if changed:
                edit.dirty = True
            return changed

    def _lift_scene_blocks(self, gltf: dict) -> bool:
        """Move each block of ours from the first scene's extras to the
        root's, as an object: one place and one encoding, whichever exporter
        wrote the file. A scene property that is no block of ours -- not an
        object stating its version -- stays where it is."""
        scenes = gltf.get("scenes") or []
        first = scenes[0] if scenes and isinstance(scenes[0], dict) else {}
        held = first.get("extras")
        if not isinstance(held, dict):
            return False
        moved = False
        for spec in SceneRecords.glb_records():
            block = held.get(spec.key)
            if block is None:
                continue
            if isinstance(block, str):
                try:
                    block = json.loads(block)
                except ValueError:
                    continue
            if not (isinstance(block, dict) and isinstance(block.get("version"), int)):
                continue
            root = gltf.get("extras")
            if not isinstance(root, dict):
                root = gltf["extras"] = {}
            root[spec.key] = block
            del held[spec.key]
            moved = True
        if moved and not held:
            del first["extras"]
        return moved

    def strip_fbx_handoff(self, gltf: dict) -> int:
        stripped = 0
        for node in gltf.get("nodes") or []:
            props = (GlbEdit.extras_of(node).get("fromFBX") or {}).get("userProperties")
            if isinstance(props, dict) and props.pop(HandoffRecord.KEY, None):
                stripped += 1
        return stripped

    def build_fbx_handoff(
        self,
        channels: Union[Iterable[str], Mapping[str, Any]],
        source: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        held = HandoffRecord.carried(channels)
        return HandoffRecord.block("fbx", channels, source) if held else {}

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

    def _apply_scene_sidecar(
        self, glb: GlbTarget, sidecar: Optional[Dict[str, Any]], *, finish: bool = True
    ) -> Dict[str, str]:
        """Apply a scene-sidecar envelope's sections to a GLB and embed the
        handoff block its source and lighting make.

        What an envelope from a host that predates the ``materials`` record
        gets, until its sections retire (:attr:`SIDECAR_APPLIERS`). Every
        section is dispatched through the applier table against **one** open
        edit session; the images the writers displaced are swept, and the
        file states itself as any GLB does -- ``extras.handoff``
        (:meth:`embed_glb_handoff`), :meth:`read_glb_handoff` the counterpart.
        The per-section outcome is the return value, not the file's.

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
            finish: Name the file's blocks in the handoff now
                (:meth:`embed_glb_handoff`); ``False`` inside a build that
                still adds blocks and finishes it last (:meth:`fbx_to_glb`).

        Returns:
            ``{section: outcome}`` — ``"N of M"`` (with a
            ``" (K not in this export)"`` clause when the envelope names
            materials this GLB does not carry), ``"0 of M matched"`` or
            ``"failed (...)"`` per offered section; empty when none offered.
        """
        if not sidecar:
            return {}
        sections = sidecar.get("sections") or {}
        try:
            with self._mc.open_glb(glb) as edit:
                summary = self._apply_sidecar_sections(edit, sections)
                # What the writers unbound (the converter's ORM, displaced by
                # the repack) must not ship -- nor be decoded by the check.
                self._mc.prune_glb_unreferenced_textures(edit)
                self._warn_destructive_orms(
                    edit, described=sections.get("metallic_roughness") or ()
                )
                write = (
                    self._mc.embed_glb_handoff
                    if finish
                    else self._mc._write_glb_handoff
                )
                write(edit, self._handoff_of(sidecar))
        except (OSError, ValueError) as error:
            logger.warning("Sidecar not applied to %s: %s", glb, error)
            return {
                section: f"failed ({error})"
                for section in self._SIDECAR_APPLIERS
                if sections.get(section)
            }
        return summary

    def _apply_sidecar_sections(
        self, edit: "GlbEdit", sections: Mapping[str, Any]
    ) -> Dict[str, str]:
        """Each section through its :attr:`SIDECAR_APPLIERS` writer, against
        *edit*; ``{section: outcome}`` as :meth:`_apply_scene_sidecar` reports."""
        summary: Dict[str, str] = {}
        present = {
            str(m.get("name"))
            for m in (edit.gltf.get("materials") or [])
            if m.get("name")
        }
        for section, method in self._SIDECAR_APPLIERS.items():
            data = sections.get(section)
            if not data:
                continue
            # The writer on the class read through first, as every caller
            # reaches it, else this part's own (the same lookup as
            # ``_FbxMaterials._read_fbx_material``'s rows).
            apply = getattr(self._mc, method, None) or getattr(self, method)
            try:
                applied = apply(edit, data)
            except (OSError, ValueError) as error:
                logger.warning("Sidecar %r not applied: %s", section, error)
                summary[section] = f"failed ({error})"
                continue
            # Counted by NAME, not by record: a by-name writer lands on every
            # material carrying the name (fade clones, the converter's own
            # duplicates), and each landing is a record. The outcome the panel
            # reads is "how many of the scene's materials were repaired", which
            # is names. A record without a name -- or not a record at all, from
            # a writer that reports differently -- counts as one.
            landed = {
                record.get("material", index) if isinstance(record, dict) else index
                for index, record in enumerate(applied)
            }
            summary[section] = self._sidecar_outcome(section, data, present, landed)
        return summary

    def _sidecar_outcome(
        self,
        section: str,
        data: Any,
        present: Set[str],
        landed: Iterable[Any],
    ) -> str:
        """One section's outcome line: ``"N of M"`` (with a ``" (K not in this
        export)"`` clause), or ``"0 of M matched"`` -- logged as it reads.

        The envelope describes the SCENE and the GLB carries the exported
        subset, so entries naming a material that is not in this file were
        never in scope. Counting them made a correct deliverable read "10 of
        23" on a production assembly -- 13 reference/ID materials that never
        exported, reported as if 13 repairs had failed. The same distinction
        ``apply_glb_lightmaps`` draws with ``out_of_scope``; a section that
        is not keyed by material name reports its plain count.
        """
        landed = len(set(landed))
        in_scope = self._mc._sidecar_section_scope(present, data)
        if in_scope is not None and landed > in_scope:
            # More landed than the exact-name scope allows, so the applier does
            # not match on the name alone (a namespace-tolerant resolver would
            # do this): "12 of 10" is worse than the plain count.
            in_scope = None
        out_of_scope = len(data) - in_scope if in_scope is not None else 0
        scope = f" ({out_of_scope} not in this export)" if out_of_scope else ""
        if not landed and (in_scope is None or in_scope):
            # Read but nothing landed -- almost always a name mismatch, which
            # the applier has just logged in full. Only a miss when something
            # WAS in scope to match.
            logger.warning(
                "Sidecar %r matched none of its %s entr(ies) in the GLB.",
                section,
                len(data),
            )
            return f"0 of {len(data)} matched"
        denominator = len(data) if in_scope is None else in_scope
        logger.info(
            "Sidecar %r applied to %s of %s in this export%s.",
            section,
            landed,
            denominator,
            f"; {out_of_scope} scene material(s) not in it" if scope else "",
        )
        return f"{landed} of {denominator}{scope}"

    def embed_glb_handoff(self, edit: "GlbEdit", handoff: Mapping[str, Any]) -> None:
        self._mc._write_glb_handoff(edit, handoff)
        self._mc._finish_glb_handoff(edit)

    def _write_glb_handoff(self, edit: "GlbEdit", handoff: Mapping[str, Any]) -> None:
        """Write *handoff* as the open GLB's ``extras.handoff`` and stamp this
        pipeline in ``asset.generator``, the blocks it names left to the step
        that knows them (:meth:`_finish_glb_handoff`): what a build still
        adding blocks writes first. Gated then, a lighting choice that only a
        later block calls for -- a lightmapped material's -- would be gone
        before that block exists."""
        extras = edit.gltf.setdefault("extras", {})
        extras[HandoffRecord.KEY] = copy.deepcopy(dict(handoff))
        self._stamp_asset_generator(edit, GlbEdit.extras_of(handoff).get("source"))
        edit.dirty = True

    def _finish_fbx_handoff(self, fbx: str) -> bool:
        """State the FBX's handoff again from the records it carries now --
        the twin of :meth:`_finish_glb_handoff`, the last pass of
        :meth:`finish_fbx_deliverable`.

        The block is written at the commit, from the records the scene
        published; a pass after the write can change one (the highlight
        copies add the probe's ``highlights``), and the row that record calls
        for with it (:meth:`RecordSpec.assembly_for`). Restated from the
        file's records as a reader applies them, its ``extras.source`` and the
        export's lighting choices (:meth:`HandoffRecord.overrides`) kept.

        Returns:
            Whether the handoff changed; ``False`` for a file carrying none.
        """
        from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia

        held = FbxMedia.records(fbx)
        stated = held.pop(HandoffRecord.KEY, None)
        if not isinstance(stated, Mapping):
            return False
        block = HandoffRecord.block(
            "fbx",
            held,
            GlbEdit.extras_of(stated).get("source"),
            HandoffRecord.overrides(stated.get("rendering")) or None,
        )
        if block == stated:
            return False
        FbxMedia.set_record(fbx, HandoffRecord.KEY, block)
        return True

    def _finish_glb_handoff(self, edit: "GlbEdit") -> bool:
        """Make the open GLB's handoff name exactly the root-``extras`` blocks
        the file carries (``HandoffRecord.assembly``).

        A pass that adds a block after the handoff is written calls this
        again, so the block never names one the file lacks, nor omits one it
        has. Idempotent.

        Returns:
            Whether the file carries a handoff to finish.
        """
        extras = edit.gltf.get("extras")
        block = extras.get(HandoffRecord.KEY) if isinstance(extras, dict) else None
        if not isinstance(block, dict):
            return False
        # Each block itself, so its row states what the block calls for and
        # the lighting recipe what the blocks use -- the export's own choices
        # read back off what was published.
        held = {
            spec.key: extras[spec.key]
            for spec in SceneRecords.glb_records()
            if spec.key in extras
        }
        block["assembly"] = HandoffRecord.assembly("glb", held)
        block["rendering"] = HandoffRecord.rendering_policy(
            HandoffRecord.overrides(block.get("rendering")) or None, held
        )
        edit.dirty = True
        return True

    def embed_scene_sidecar(
        self,
        edit: "GlbEdit",
        sidecar: Dict[str, Any],
        summary: Mapping[str, str],
        *,
        described: Iterable[str] = (),
    ) -> None:
        self._warn_destructive_orms(edit, described=described)
        self._mc.embed_glb_handoff(edit, self._handoff_of(sidecar))

    def _warn_destructive_orms(
        self, edit: "GlbEdit", described: Iterable[str] = ()
    ) -> List[str]:
        """Warn of the materials the open GLB renders metallic everywhere --
        the converter's own ORM packing, measured to arrive solid white -- and
        return their names.

        Only the DESTRUCTIVE finding is raised at export time: the rest of
        :meth:`suspect_orm_materials`' report is coverage information, read by
        :meth:`verify_glb` -- an export-time warning an artist cannot act on is
        how the actionable ones stop being read.

        Parameters:
            edit: The open session.
            described: Material names whose metal/roughness packing this
                pipeline wrote from authored maps -- exempt.
        """
        suspect = self._mc.suspect_orm_materials(edit, described=described)
        harmful = sorted(
            name
            for name, found in suspect.items()
            if found["finding"] == self._mc.ORM_FINDING_METALLIC_FULL
        )
        if harmful:
            logger.warning(
                "Metallic=1 everywhere on %s material(s) the file does not "
                "state: %s. glTF reads metallic from the ORM's blue channel, "
                "so these render with no diffuse response (black under a "
                "lightmap) -- export through a Scene Exporter, whose FBX "
                "states every material, or re-export their source maps as "
                "RGB.",
                len(harmful),
                ", ".join(harmful),
                extra={"preset": "highlight"},
            )
        return harmful

    def _stamp_asset_generator(
        self, edit: "GlbEdit", source: Optional[Mapping[str, Any]] = None
    ) -> str:
        """Name this pipeline in the glTF's own ``asset.generator``; return it.

        The one provenance field glTF itself defines, and every viewer and
        inspector already displays -- so it reaches a recipient who opens the
        file in a tool that is not ours and reads nothing else we write. The
        converter's own string is kept and ours appended: FBX2glTF really did
        produce the geometry, and replacing that claim would lose the fact that
        matters most when a mesh arrives wrong.

        Deliberately coarse -- the authoring app + version the handoff's
        *source* names, and this package's version. No host, no user, no
        paths: a generator string travels to whoever gets the file.

        Idempotent: re-stamping an already-stamped GLB refreshes the stamp
        rather than appending a second one.
        """
        parts = []
        source = source or {}
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
        prior = (asset.get("generator") or "").split(self._GENERATOR_SEP)[0].strip()
        # A file whose ONLY generator claim is a previous stamp of ours has no
        # separator to split on, so the whole string would come back as "prior"
        # and the next pass would append ours to itself. Our stamp always names
        # this package, which is what makes it recognisable across versions.
        if "pythontk " in prior:
            prior = ""
        asset["generator"] = f"{prior}{self._GENERATOR_SEP}{ours}" if prior else ours
        edit.dirty = True
        return asset["generator"]

    def sidecar_foreign_packings(
        self,
        sidecar: Optional[Dict[str, Any]],
        target: str = "ORM",
        workflow: Optional[str] = None,
    ) -> Dict[str, str]:
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

    def read_glb_handoff(self, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        with self._mc.open_glb(glb) as edit:
            block = GlbEdit.extras_of(edit.gltf).get(HandoffRecord.KEY)
        return block if isinstance(block, dict) else None

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

    def verify_glb(self, glb: GlbTarget) -> Dict[str, Any]:
        report: Dict[str, Any] = {"ok": True, "problems": [], "notes": []}

        def fail(message: str) -> None:
            report["ok"] = False
            report["problems"].append(message)

        with self._mc.open_glb(glb) as edit:
            extras = GlbEdit.extras_of(edit.gltf)
            report["generator"] = (edit.gltf.get("asset") or {}).get("generator")
            report["lightmap"] = bool(
                SceneRecords.glb_block(edit.gltf, SceneRecords.LIGHTMAPS)
            )
            # What a reader must SUPPORT to open this at all. `extensionsRequired`
            # is not advice: the spec says a reader that does not implement one
            # of these must refuse the file. A web-delivery GLB requires
            # `EXT_texture_webp` (nothing core-readable survives that pass) and
            # a pure-delivery KTX2 one requires `KHR_texture_basisu`, so the
            # deliverable most likely to reach a third party is exactly the one
            # with a hard prerequisite. Reported rather than failed: the
            # requirement is correct, it just has to be readable without
            # parsing the JSON chunk by hand.
            required = sorted(edit.gltf.get("extensionsRequired") or [])
            report["extensions"] = {
                "required": required,
                "used": sorted(edit.gltf.get("extensionsUsed") or []),
            }
            if required:
                report["notes"].append(self._mc._requirement_note(required))
            # The spec's own subset rule, and a real failure rather than a note:
            # a file demanding a capability it never declares is invalid glTF
            # and stock validators reject it outright.
            undeclared = sorted(set(required) - set(report["extensions"]["used"]))
            if undeclared:
                fail(
                    f"extensionsRequired names {', '.join(undeclared)}, which "
                    "extensionsUsed does not declare -- invalid glTF"
                )
            # Reported before the handoff gate below: an asset can carry clips
            # and no handoff (another producer's file), and "how many clips,
            # how many of them shots" is what a recipient asks first of an
            # animated deliverable.
            animation = SceneRecords.glb_block(edit.gltf, SceneRecords.SHOTS)
            if isinstance(animation, dict):
                clips = animation.get("clips") or []
                shots = {s.get("clip") for s in animation.get("shots") or []}
                report["animation"] = {
                    "clips": len(clips),
                    "shots": len(shots),
                    # The shots that play a clip of their own; the rest play
                    # their window of the whole-timeline clip.
                    "declared": sum(1 for c in clips if c.get("name") in shots),
                    "default_clip": animation.get("default_clip"),
                    "fps": animation.get("fps"),
                }
            elif edit.gltf.get("animations"):
                # Not a failure -- the block is only written by our own
                # conversion -- but a note, because a consumer that went
                # looking for shot names here found none.
                report["notes"].append(
                    f"{len(edit.gltf['animations'])} animation(s) with no "
                    f"extras.{SceneRecords.SHOTS.key}: clip identity is names only"
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

            # Every ORM is decoded: a channel layout no reader can vouch for is
            # only worth saying when it is destructive -- metallic everywhere,
            # which renders black under a lightmap.
            suspect = {
                material: found
                for material, found in self._mc.suspect_orm_materials(edit).items()
                if found["finding"] == self._mc.ORM_FINDING_METALLIC_FULL
            }
            if suspect:
                report["orm"] = suspect
                fail(
                    "metallic=1 everywhere on: "
                    + ", ".join(sorted(suspect))
                    + " (no diffuse response; black under a lightmap)"
                )

            handoff = extras.get(HandoffRecord.KEY)
            if not isinstance(handoff, dict):
                report["handoff"] = None
                fail("no extras.handoff: nothing here declares itself")
                return report

            version = handoff.get("version")
            report["handoff"] = {
                "version": version,
                "source": GlbEdit.extras_of(handoff).get("source"),
            }
            # The handoff and every block of ours: in the version read and
            # the shape declared -- a reader would otherwise skip what another
            # schema says and report a clean bill on a file it only partly
            # understood.
            for problem in SceneRecords.problems(SceneRecords.HANDOFF, handoff):
                fail(f"extras.{problem}")
            for spec in SceneRecords.glb_records():
                if spec.key in extras:
                    for problem in SceneRecords.block_problems(spec, extras[spec.key]):
                        fail(problem)
        return report

    def _current_record(self, gltf: dict, spec: RecordSpec) -> Optional[Dict[str, Any]]:
        """*spec*'s record on *gltf*'s carrier in the schema it declares
        (:meth:`RecordSpec.is_current`); ``None`` when the file carries none,
        and -- warned -- when it carries another: a reader refuses rather than
        guesses what another schema's fields mean, and the cure is a
        re-export. Every pass that applies a record reads it here."""
        payload = self._mc.data_export_channel(gltf, spec.key)
        if payload is None:
            return None
        problems = SceneRecords.problems(spec, payload)
        if not problems:
            return payload
        more = f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""
        logger.warning(
            "%s record not applied -- %s%s; re-export the file.",
            spec.key,
            "; ".join(problems[:3]),
            more,
        )
        return None

    def data_export_channel(self, gltf: dict, key: str) -> Optional[Any]:
        found = self._data_export_carrier(gltf, key)
        if found is None:
            return None
        node, props = found
        entry = props[key]
        # FBX2glTF wraps each property as {"type": ..., "value": ...}.
        raw = entry.get("value") if isinstance(entry, dict) else entry
        if isinstance(raw, str) and not raw.strip():
            # A producer with nothing to publish CLEARS its channel to "" rather
            # than deleting the attribute, so an empty string is "absent", not
            # "unparsable".
            return None
        try:
            return json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError) as error:
            logger.warning("Unparsable %s on node %r: %s", key, node.get("name"), error)
            return None

    def _data_export_carrier(self, gltf: dict, key: str) -> Optional[Tuple[dict, dict]]:
        """``(node, holder)`` for the *key* channel :meth:`data_export_channel`
        reads -- the carrier node, and the dict in it that holds the channel
        -- or ``None`` when no node carries one. The one place the reader's
        precedence lives, so a writer can put a channel back where it was read.

        The scene's own carrier (``data_export``, no namespace) wins over a
        referenced module's (``NS:data_export``), which an assembly export
        carries beside it in whatever order the writer put them; a module's
        is read only where the scene's carries no such channel, the first in
        file order. unitytk's importers apply the same rule
        (``CarrierImport``), so the two deliverables of one export read the
        same record.
        """
        first = None
        for node in gltf.get("nodes", []) or []:
            extras = GlbEdit.extras_of(node)
            # TWO on-disk shapes: FBX2glTF nests user properties under
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
                if props.get(key) is not None:
                    name = str(node.get("name") or "").split("|")[-1]
                    if name == self._mc.DATA_EXPORT_NODE:
                        return node, props
                    first = first or (node, props)
                    break
        return first

    @staticmethod
    def _write_data_export(node: dict, key: str, value: Any) -> None:
        """Write *value* as *node*'s *key* channel, in the shape every producer
        publishes and the reader decodes: JSON text in the node's top-level
        extras. A nested copy goes, so the reader cannot meet a stale one
        first."""
        extras = node.setdefault("extras", {})
        props = (extras.get("fromFBX") or {}).get("userProperties")
        if isinstance(props, dict):
            props.pop(key, None)
        extras[key] = value if isinstance(value, str) else json.dumps(value)

    @staticmethod
    def _drop_data_export_channel(gltf: dict, key: str) -> List[dict]:
        """Remove the *key* channel from every node, in both on-disk shapes
        :meth:`data_export_channel` reads; the nodes that held one, in file
        order."""
        held: List[dict] = []
        for node in gltf.get("nodes") or []:
            extras = node.get("extras") if isinstance(node, dict) else None
            if not isinstance(extras, dict):
                continue
            found = False
            for holder in ((extras.get("fromFBX") or {}).get("userProperties"), extras):
                if isinstance(holder, dict) and key in holder:
                    del holder[key]
                    found = True
            if found:
                held.append(node)
        return held

    #: What a converted ``data_export`` carrier node is called, and what
    #: :meth:`overlay_data_export` names one it has to create: the store's.
    DATA_EXPORT_NODE = SceneStoreBase.name(Scope.DELIVERABLE)

    def overlay_data_export(self, gltf: dict, channels: Dict[str, Any]) -> List[str]:
        changed = set()
        carrier = None
        nodes = gltf.get("nodes") or []
        for key in channels or {}:
            held = self._drop_data_export_channel(gltf, key)
            if held:
                changed.add(key)
                carrier = carrier or held[0]

        written = {k: v for k, v in (channels or {}).items() if v is not None}
        if written:
            if carrier is None:
                carrier = next(
                    (
                        node
                        for node in nodes
                        if isinstance(node, dict)
                        and str(node.get("name") or "").split("|")[-1].split(":")[-1]
                        == self._mc.DATA_EXPORT_NODE
                    ),
                    None,
                )
            if carrier is None:
                # A selection export ships no carrier when the scene has none;
                # the overlay still has to land where the readers look.
                carrier = {"name": self._mc.DATA_EXPORT_NODE}
                gltf.setdefault("nodes", []).append(carrier)
                scenes = gltf.get("scenes") or []
                index = gltf.get("scene", 0)
                if isinstance(index, int) and 0 <= index < len(scenes):
                    scenes[index].setdefault("nodes", []).append(len(gltf["nodes"]) - 1)
            for key, value in written.items():
                self._write_data_export(carrier, key, value)
                changed.add(key)
        return sorted(changed)
