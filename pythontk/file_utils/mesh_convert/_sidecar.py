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
        for kind, key in (("materials", "fromFBX"),):
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
            changed = bool(self._carry_glb_records(edit.gltf)) or changed
            handoff = self._fbx_handoff(edit.gltf, rendering)
            if handoff:
                self._mc.embed_glb_handoff(edit, handoff)
                changed = True
            elif changed:
                self._mc._finish_glb_handoff(edit)
            stripped = bool(self._mc.strip_fbx_records(edit.gltf))
            if stripped:
                self._mc._compact_bin(edit)  # what a bare carrier's samplers read
            changed = stripped or changed
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

    def verify_glb(
        self, glb: GlbTarget, *, described: Optional[Iterable[str]] = None
    ) -> Dict[str, Any]:
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
                clips = [a.get("name") for a in edit.gltf.get("animations") or []]
                shots = {s.get("clip") for s in animation.get("shots") or []}
                report["animation"] = {
                    "clips": len(clips),
                    "shots": len(shots),
                    # The shots that play the clip of their name: every one, in
                    # a file whose takes reached the conversion.
                    "declared": len(shots & set(clips)),
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
                for material, found in self._mc.suspect_orm_materials(
                    edit, described=described
                ).items()
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
                    for problem in SceneRecords.problems(spec, extras[spec.key]):
                        fail(f"extras.{problem}")
            # The files the record names, as the media transport carries them
            # (SceneRecords.GLB_MEDIA): a reader decodes each by what its view
            # states, so a view stating something else is unreadable.
            for part in (self._mc._lightmaps, self._mc._shadow_rigs):
                problems, notes = part.media_findings(edit.gltf)
                for problem in problems:
                    fail(problem)
                report["notes"].extend(notes)
            # What a reader of the rigs cannot pose (an ambiguous joint, a
            # rig not wholly in the file, a part not grabbable): notes, as
            # the build said them -- the file is readable, those rigs inert.
            where = f"extras.{SceneRecords.ARTICULATION.key}"
            report["notes"].extend(
                f"{where}: {text}"
                for _level, text in self._mc._articulation.findings(edit.gltf)
            )
        return report

    #: The fields a record states about the FBX container itself, by record
    #: key, which its GLB copy leaves out (:meth:`_carry_glb_records`): the
    #: ``audio`` record's ``media`` digests the sound files the FBX embeds
    #: (``AudioRecord.media``), and a GLB embeds none -- a copy naming them
    #: would vouch for bytes the file does not carry.
    GLB_CONTAINER_FIELDS = {SceneRecords.AUDIO.key: ("media",)}

    def _carry_glb_records(self, gltf: dict, skip: Iterable[str] = ()) -> List[str]:
        """Copy each record a GLB carries as it stands
        (:attr:`RecordSpec.glb_verbatim`) from the FBX's carrier into the root
        ``extras``, under its key -- one layout, two containers -- unless a
        pass has already put it there (:meth:`apply_glb_animations` carries
        ``animation`` with its takes renamed), less the fields that describe
        the FBX container (:attr:`GLB_CONTAINER_FIELDS`). Read in the schema
        it declares (:meth:`_current_record`: another version is refused,
        warned), so the GLB never states what the FBX's record did not.
        Before :meth:`strip_fbx_records` takes the carrier away. A key in
        *skip* is not carried (a build told to leave its files out names none
        of them).

        Returns:
            The keys carried, in declaration order; the caller marks its edit
            session dirty when this is non-empty.
        """
        extras = gltf.get("extras")
        carried: List[str] = []
        skip = set(skip)
        for spec in SceneRecords.glb_records():
            if not spec.glb_verbatim or spec.key in skip:
                continue
            if isinstance(extras, dict) and spec.key in extras:
                continue
            record = self._current_record(gltf, spec)
            if not isinstance(record, dict):
                continue
            carried_record = {
                field: value
                for field, value in copy.deepcopy(record).items()
                if field not in self.GLB_CONTAINER_FIELDS.get(spec.key, ())
            }
            SceneRecords.put_glb_block(gltf, spec, carried_record)
            extras = gltf.get("extras")
            carried.append(spec.key)
        return carried

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

        The scene's own carrier (``data_export``, no namespace, at the
        file's root: :meth:`_root_level_nodes`) wins over a module's
        (``NS:data_export``, or a ``data_export`` under a group), which an
        assembly export carries beside it in whatever order the writer put
        them; a module's is read only where the scene's carries no such
        channel, the first in file order (``SceneStoreBase.is_scene_carrier``),
        and a module's ``handoff`` never (it describes that module, not the
        file: the FBX handoff's own rule) -- ``SceneStoreBase.reader_values``,
        the one statement of it every reader of a written file calls.
        unitytk's importers apply the same rule (``CarrierImport``), so the two
        deliverables of one export read the same record.
        """
        rows = []
        root_level = self._root_level_nodes(gltf)
        for index, node in enumerate(gltf.get("nodes", []) or []):
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
                    rows.append((name, index in root_level, key, (node, props)))
                    break
        chosen = SceneStoreBase.reader_values(rows).get(key)
        return chosen[0] if chosen else None

    #: The node FBX2glTF roots a converted file's hierarchy under: the FBX's
    #: root nodes are its children.
    CONVERTER_ROOT = "RootNode"

    @classmethod
    def _root_level_nodes(cls, gltf: dict) -> set:
        """The indices of the nodes at the root of the file's hierarchy: the
        scene roots (every node no node parents, for a file stating no scene),
        a converter's ``RootNode`` standing for its children -- the level the
        scene's own carrier sits at (``SceneStoreBase.is_scene_carrier``)."""
        nodes = gltf.get("nodes") or []
        roots = [
            i
            for scene in gltf.get("scenes") or []
            for i in (scene.get("nodes") or [])
            if isinstance(i, int)
        ]
        if not roots:
            children = {
                c
                for node in nodes
                if isinstance(node, dict)
                for c in node.get("children") or []
            }
            roots = [i for i in range(len(nodes)) if i not in children]
        level = set()
        for i in roots:
            node = nodes[i] if 0 <= i < len(nodes) else None
            if isinstance(node, dict) and node.get("name") == cls.CONVERTER_ROOT:
                level.update(c for c in node.get("children") or [] if isinstance(c, int))
            else:
                level.add(i)
        return level

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
        nodes = gltf.get("nodes") or []
        for key in channels or {}:
            if self._drop_data_export_channel(gltf, key):
                changed.add(key)

        written = {k: v for k, v in (channels or {}).items() if v is not None}
        if written:
            # The scene's own carrier (the root-level ``data_export``), never a
            # module's that happened to hold a channel; made at the root when
            # the file has none.
            root_level = self._root_level_nodes(gltf)
            carrier = next(
                (
                    node
                    for index, node in enumerate(nodes)
                    if isinstance(node, dict)
                    and SceneStoreBase.is_scene_carrier(
                        str(node.get("name") or "").split("|")[-1], index in root_level
                    )
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
