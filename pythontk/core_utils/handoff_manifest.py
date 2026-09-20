# !/usr/bin/python
# coding=utf-8
"""The hand-off sidecar -- what an FBX or USD payload cannot carry by itself.

Every app-to-app conversion in the ecosystem ships two files: an intermediate
(``.fbx`` / ``.usd*``) and a JSON sidecar named after it, ``<payload>.manifest.json``.
The carrier holds the geometry and the animation; the sidecar holds everything
the carrier has no place for -- which material wore which texture, which Empty
was a group and which a locator, which objects shared one mesh, the source's
clock, its lights, its shots, its rig.

This class is that sidecar's contract, in one place.  Before it, the contract
was a convention: the ``".manifest.json"`` suffix was spelled at ten call sites,
two packages each had their own tolerant reader plus a further seven appliers
that re-opened the same file to read one key out of it, and the section names
were bare string literals in roughly eighty places across four producer
templates, two bridges and two consumers -- a producer writing ``"visibilty"``
would have been caught by nothing.  A manifest now has a *type*: the suffix, the
version, the section vocabulary and the read/write rules live here, and both
sides import them.

**A section's contents stay opaque.**  This class knows that ``shots`` is a
section and nothing about what a shot is -- that belongs to the codec that owns
it (:class:`~pythontk.ShotTransfer` for ``shots``, the rig graph engine for
``rig``).  Adding a section is a constant here plus a codec of its own; it is
never a change to this class's logic.

**Reading is tolerant, by design.**  A missing or malformed sidecar yields an
empty manifest rather than an exception, because a payload whose geometry landed
must not be lost to a damaged JSON file; :attr:`unreadable` separates "damaged"
from "a producer with nothing to say" so the caller can warn about the one that
matters, and a section whose *absence* would leave the scene structurally wrong
(the instance replay) enforces that itself.

The instance was built to be a drop-in for the plain ``dict`` the appliers used
to pass around -- it is a ``Mapping``, so ``manifest.get("rig")``,
``manifest["rig"]``, ``"rig" in manifest`` and ``dict(manifest)`` all behave --
which is what lets a consumer adopt it one applier at a time.

Replaying a manifest is :meth:`plan`; see :class:`~pythontk.ManifestPlan`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Dict, Iterator, Optional

from pythontk.core_utils.manifest_plan import ManifestPlan, OnError
from pythontk.file_utils._file_utils import FileUtils

__all__ = ["HandoffManifest"]


class _HandoffManifestInternal(Mapping):
    """Mapping plumbing and internal helpers for :class:`HandoffManifest`."""

    #: Sentinel telling "the reader could not use this file" apart from a sidecar
    #: that legitimately holds ``null`` -- both of which a plain default hides.
    _UNREAD = object()

    _data: Dict[str, Any]
    _path: Optional[str]
    _unreadable: bool

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        sections = ", ".join(sorted(self._data)) or "empty"
        return f"{type(self).__name__}({sections})"

    @staticmethod
    def _as_dict(loaded: Any) -> Dict[str, Any]:
        """*loaded* when it is a JSON object, else ``{}`` -- a list or a scalar
        is a sidecar written by something else and carries no section.

        Any ``Mapping`` counts, so building one manifest from another (or from
        anything dict-like an applier holds) keeps the sections rather than
        silently emptying them.
        """
        return dict(loaded) if isinstance(loaded, Mapping) else {}


class HandoffManifest(_HandoffManifestInternal):
    """The ``<payload>.manifest.json`` beside a conversion payload.

    Parameters:
        data: The document's sections.  Non-object input is treated as empty.
        path: Where it was read from / will be written to, when known.
        unreadable: A sidecar was there and could not be used (:attr:`unreadable`).
    """

    #: Appended to the payload's own path -- the sidecar's whole naming rule.
    SUFFIX = ".manifest.json"

    # ------------------------------------------------------------ document keys
    #: Key of the document's schema version.  Every producer writes
    #: :attr:`VERSION`, whatever its carrier: a reader may therefore treat this as
    #: naming the SCHEMA, which is what a version is for.
    #:
    #: It did not always.  Until 2026-09-17 the FBX route wrote ``1`` and the USD
    #: route ``2`` although nothing about the document differed except that only
    #: the USD route carries :attr:`INSTANCES`, so the number named the CARRIER's
    #: dialect and a genuine schema change had no number left to turn.  What
    #: actually discriminates the instance spelling is :attr:`FORMAT_KEY`, and
    #: that is what the replay gates on now.
    VERSION_KEY = "version"
    #: The schema version every producer writes.  Bump this when the DOCUMENT
    #: changes shape in a way a reader must branch on -- not when a carrier gains
    #: a section, which :meth:`carries` already answers without a number.
    VERSION = 2
    #: Key naming which side wrote the instance groups, and therefore how their
    #: members are spelled -- :attr:`FORMAT_NAMES` or :attr:`FORMAT_PATHS`.  The
    #: instance replay's real discriminator, and the only thing that ever was: a
    #: Maya producer spells a member as a DAG path, a Blender one as a name, and
    #: replaying one as the other matches nothing.  Written only by a producer
    #: that writes :attr:`INSTANCES`; a document without them has nothing to
    #: spell.
    FORMAT_KEY = "format"

    #: Instance-group spelling written by a Blender-side producer.
    FORMAT_NAMES = "names"
    #: Instance-group spelling written by a Maya-side producer.
    FORMAT_PATHS = "paths"

    # ---------------------------------------------------------------- sections
    #: Per-material texture records the carrier could not express -- the entries
    #: the consumer rebuilds natively.  File-less entries travel too, so a
    #: material whose images never resolved surfaces as a *named* warning
    #: instead of silently gray (or pink) geometry.
    MATERIALS = "materials"
    #: Every material name in the source scene, beyond the textured entries, so a
    #: consumer can claim the ones a carrier renamed on import.
    SCENE_MATERIALS = "scene_materials"
    #: ``{material: [shading group, ...]}`` -- the Maya-side assignment map a USD
    #: stage flattens away.
    SHADING_GROUPS = "shading_groups"
    #: ``[{name, display_type}, ...]`` -- what each Empty *was*.  Every FBX null
    #: imports as a Maya locator and every USD Xform as a shapeless transform, so
    #: the author's intent has to travel beside the carrier.
    EMPTIES = "empties"
    #: ``{node: type}`` -- Maya group / locator identity for the Empties going the
    #: other way.
    TRANSFORMS = "transforms"
    #: Groups of objects that shared one mesh.  Both exporters write flat, so
    #: without this a scene of linked duplicates lands as N independent shapes:
    #: a scene that renders correctly and only betrays itself when an artist
    #: edits one "instance" and its siblings do not follow.  Spelled per
    #: :attr:`FORMAT_KEY`.
    INSTANCES = "instances"
    #: Baked show / hide the carrier drops (FBX carries the curve; Blender's
    #: importer discards it).
    VISIBILITY = "visibility"
    #: The source's lights as data, rebuilt natively on the far side.
    LIGHTS = "lights"
    #: The source's world / environment record (HDRI, sky dome).
    WORLD = "world"
    #: ``{mesh: method}`` -- each skinned mesh's skinning method, so a
    #: dual-quaternion skin does not silently arrive linear.
    SKINS = "skins"
    #: The joint hierarchy the export-time skin flatten removes; neither carrier
    #: stores a bone length, and the flatten is what breaks the one an importer
    #: would otherwise infer.
    BONES = "bones"
    #: The source's time setup -- fps, ranges, current frame.
    SCENE = "scene"
    #: The source's shot store, encoded by :class:`~pythontk.ShotTransfer`:
    #: shots, markers, locked gaps, the edit ledger, render-effect channels and
    #: audio clips.  Replayed last, because its memberships and ledger claims
    #: name objects and curves every other section may have replaced or re-keyed.
    SHOTS = "shots"
    #: The source rig's logic -- the extracted graph, its plan against the
    #: consumer's capability, and the world-space samples the build is verified
    #: against.  Owned by the rig-graph engine.
    RIG = "rig"
    #: ``{node: kind}`` -- the rig apparatus a bake leaves INERT: constraint
    #: nodes, IK handles, control curves, up-vector locators, the groups that
    #: hold only those and joints nothing content is skinned to.  A carrier ships
    #: every node as an object, so without this a baked rig arrives twice over --
    #: its motion in keys, and the whole apparatus that used to produce that
    #: motion, selectable, drawn and driving nothing.  Named by the producer and
    #: dropped by the consumer, never deleted at the source: one route bakes
    #: before it writes and the other samples the live scene, so removing a
    #: constraint there would remove the motion it was about to record.  Both
    #: halves of that judgement are :class:`~pythontk.RigMachinery`; a producer
    #: contributes what each node IS, a consumer what arrived.
    MACHINERY = "machinery"
    #: The scene records that cross a hand-off, keyed by record
    #: (``{"emissive_groups": {...}}``) -- every ``SceneRecords`` record
    #: declared ``portable`` that names no section of its own (the shot store
    #: keeps :attr:`SHOTS`).  Written and read by
    #: :class:`~pythontk.RecordTransfer`, so a new portable record crosses with
    #: no change here or in either bridge.
    RECORDS = "records"

    #: Every section name above, for validation and for tests that pin the
    #: vocabulary.  Not an application order -- that is each consumer's plan.
    SECTIONS = (
        MATERIALS,
        SCENE_MATERIALS,
        SHADING_GROUPS,
        EMPTIES,
        TRANSFORMS,
        INSTANCES,
        VISIBILITY,
        LIGHTS,
        WORLD,
        SKINS,
        BONES,
        SCENE,
        SHOTS,
        RIG,
        MACHINERY,
        RECORDS,
    )

    def __init__(
        self,
        data: Any = None,
        *,
        path: Optional[str] = None,
        unreadable: bool = False,
    ) -> None:
        self._data = self._as_dict(data)
        self._path = str(path) if path else None
        self._unreadable = bool(unreadable)

    # ------------------------------------------------------------------ reading
    @classmethod
    def path_for(cls, payload_path: str) -> str:
        """The sidecar path for *payload_path*.

        Idempotent: a path that already names the sidecar is returned unchanged,
        so a caller holding either one can pass it without checking which.
        """
        path = str(payload_path)
        return path if path.endswith(cls.SUFFIX) else path + cls.SUFFIX

    @classmethod
    def read(cls, payload_path: str) -> "HandoffManifest":
        """The manifest for *payload_path* (the payload's path or the sidecar's).

        Never raises: an absent, unreadable or non-object sidecar yields an empty
        manifest, because a payload whose geometry already landed must not be
        lost to a damaged JSON file.  :attr:`unreadable` separates the two cases
        a caller would report differently, and a section whose absence would
        leave the scene structurally wrong enforces that itself.

        Parsing is :meth:`~pythontk.FileUtils.read_json`, which owns the "cannot
        be read" catch set for the whole package (a missing file and an
        unparseable one are not the same exception, and hand-rolled readers
        disagreed about which they caught).
        """
        path = cls.path_for(payload_path)
        if not os.path.isfile(path):
            # Absent is not damaged: a producer with nothing to say.
            return cls(path=path)
        loaded = FileUtils.read_json(path, default=cls._UNREAD)
        if loaded is cls._UNREAD:
            return cls(path=path, unreadable=True)
        return cls(loaded, path=path)

    @property
    def path(self) -> Optional[str]:
        """Where this manifest was read from, or ``None`` when built in memory."""
        return self._path

    @property
    def payload_path(self) -> Optional[str]:
        """The payload this manifest describes -- :attr:`path` less :attr:`SUFFIX`."""
        if not self._path:
            return None
        return (
            self._path[: -len(self.SUFFIX)]
            if self._path.endswith(self.SUFFIX)
            else self._path
        )

    @property
    def data(self) -> Dict[str, Any]:
        """A deep copy of the document; mutating it never reaches the manifest.

        Deep rather than shallow because a section is a list or a dict: with a
        shallow copy an applier that sorted or filtered a section in place would
        edit the manifest every later step still reads.  Read a section you do
        not intend to change through ``manifest[section]`` instead.
        """
        return deepcopy(self._data)

    @property
    def unreadable(self) -> bool:
        """Whether a sidecar was there and could not be used.

        An absent sidecar is not a failure -- it is a producer with nothing to
        say -- and leaves this ``False``.  ``True`` means the file existed but
        was damaged or was not a JSON object, which is worth a warning at the
        call site: the import will otherwise proceed silently without any of the
        fidelity the producer meant to send.
        """
        return self._unreadable

    @property
    def version(self) -> Any:
        """The document's :attr:`VERSION` value, or ``None``."""
        return self._data.get(self.VERSION_KEY)

    @property
    def format(self) -> Any:
        """The document's :attr:`FORMAT_KEY` value, or ``None``."""
        return self._data.get(self.FORMAT_KEY)

    def carries(self, section: str) -> bool:
        """Whether *section* arrived with something worth replaying.

        Truthiness, not presence: every producer writes "absent means nothing to
        say", so a section that arrived empty asks for the same no-op as one that
        never arrived.  :class:`~pythontk.ManifestPlan` gates its steps by the
        same rule.
        """
        return bool(self._data.get(section))

    # ------------------------------------------------------------------ writing
    @classmethod
    def build(cls, **sections: Any) -> "HandoffManifest":
        """A manifest of *sections*, dropping the ones given as ``None``.

        ``None`` is the "nothing to say" value and is omitted, so the reader's
        truthiness gate and the writer's intent agree.  Anything else is kept
        verbatim -- including an empty list, which is a producer saying "I looked
        and there are none", a distinction the USD route depends on to tell no
        instances from a lost sidecar.
        """
        return cls({k: v for k, v in sections.items() if v is not None})

    def write(self, path: Optional[str] = None, *, indent: Optional[int] = 1) -> str:
        """Write the document beside its payload atomically; return the path written.

        Through :meth:`~pythontk.FileUtils.write_json`, so a reader sees either
        the old sidecar or the complete new one.  That matters more here than
        almost anywhere: a conversion that died mid-write would otherwise leave a
        truncated sidecar beside a perfectly good payload, and the consumer's
        tolerant read would then skip every section the producer meant to send --
        a silently unfaithful import rather than a failed one.

        Parameters:
            path: The payload's path or the sidecar's; defaults to :attr:`path`.
            indent: JSON indent -- ``1`` (the ecosystem's sidecars are read by
                humans when a conversion misbehaves) or ``None`` for compact.

        Raises:
            ValueError: No *path* given and none known.
        """
        target = self.path_for(path) if path else self._path
        if not target:
            raise ValueError("No path given and this manifest has none.")
        FileUtils.write_json(target, self._data, indent=indent)
        self._path = target
        return target

    # ----------------------------------------------------------------- replaying
    def plan(
        self,
        *,
        on_error: Optional[OnError] = None,
        cancel_prefix: Optional[str] = None,
    ) -> ManifestPlan:
        """A :class:`~pythontk.ManifestPlan` gated on this manifest's sections.

        Parameters:
            on_error: ``on_error(label, exception)`` for ``best_effort`` steps.
            cancel_prefix: Lead of the stop message, so a consumer keeps its wording.
        """
        return ManifestPlan(self, on_error=on_error, cancel_prefix=cancel_prefix)
