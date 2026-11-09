# !/usr/bin/python
# coding=utf-8
"""Deliverable verification for exported FBX / GLB pairs.

The permanent form of a verification harness that caught a production GLB
shipping a wire-loom 7.5 cm away from the plug it was constrained to — a
defect no exporter log mentioned. Point it at a deliverable (either file or
both), optionally with its ``.{stem}.scene_data.json`` record and a
previous known-good GLB, and it runs every registered ``check_*`` gate and
reports PASS / WARN / FAIL / SKIP per gate.

Extending it is one method: add ``check_<name>`` to :class:`ExportVerifier`
(or a subclass) returning :class:`Finding` rows — discovery is by prefix, the
same reflection idiom the scene exporters' ``TaskFactory`` uses.

CLI (the deliverable-side twin of the exporters' in-scene checks)::

    python -m pythontk.file_utils.mesh_convert.export_verify ASSET.glb ASSET.fbx
    python -m pythontk.file_utils.mesh_convert.export_verify ASSET.glb \\
        --baseline old/ASSET.glb --json
"""

import glob as _glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from pythontk.core_utils.deprecation import Deprecation
from pythontk.core_utils.engines.scene_export.export_profile import ExportProfile
from pythontk.core_utils.engines.scene_export.handoff_record import HandoffRecord
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert
from pythontk.file_utils.mesh_convert.fbx_file import FbxFile
from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit
from pythontk.file_utils.mesh_convert.glb.reader import GlbReader

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


@dataclass
class Finding:
    """One verification outcome row."""

    status: str
    check: str
    detail: str


@dataclass
class VerificationReport:
    """Every finding from one :meth:`ExportVerifier.run`."""

    rows: List[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no finding FAILed (WARN and SKIP do not fail a report)."""
        return not any(row.status == FAIL for row in self.rows)

    def counts(self) -> Dict[str, int]:
        out = {PASS: 0, WARN: 0, FAIL: 0, SKIP: 0}
        for row in self.rows:
            out[row.status] = out.get(row.status, 0) + 1
        return out

    def summary(self) -> str:
        """Human-readable table plus a one-line verdict."""
        lines = [f"[{row.status}] {row.check}: {row.detail}" for row in self.rows]
        counts = self.counts()
        verdict = "OK" if self.ok else f"{counts[FAIL]} CHECK(S) FAILED"
        lines.append(
            f"RESULT: {verdict}  "
            f"(pass {counts[PASS]}, warn {counts[WARN]}, "
            f"fail {counts[FAIL]}, skip {counts[SKIP]})"
        )
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "ok": self.ok,
                "counts": self.counts(),
                "rows": [row.__dict__ for row in self.rows],
            },
            indent=2,
        )


class _ExportVerifierInternal:
    """Input resolution and small shared predicates."""

    @classmethod
    def _sidecar_beside(cls, path: Optional[str]) -> Optional[str]:
        """The exporter's ``.{stem}.scene_data.json`` record of *path*, if any.

        Looked up in the per-user store of *path*'s folder, where the
        exporters keep it (``SceneDataSidecarBase.store_dir``), then beside
        *path*, where an earlier version left one.  In each, the deliverable's
        own first; failing that, a versioned deliverable's SERIES manifest --
        the exporters key that one to the stem without its trailing ``_v<N>``,
        so every version diffs against one baseline, and a lookup by the
        file's own stem found nothing for any of them.
        """
        if not path:
            return None
        from pythontk.core_utils.engines.scene_export.scene_data_sidecar import (
            SceneDataSidecarBase,
        )

        stem = os.path.splitext(os.path.basename(path))[0]
        series = ExportProfile.VERSION_SUFFIX_RE.sub("", stem)
        for folder in (
            SceneDataSidecarBase.store_dir(path),
            os.path.dirname(os.path.abspath(path)),
        ):
            for candidate in dict.fromkeys((stem, series)):
                sidecar = os.path.join(folder, f".{candidate}.scene_data.json")
                if os.path.isfile(sidecar):
                    return sidecar
        return None

    @staticmethod
    def _declared_clips(sidecar: Optional[dict]) -> Tuple[List[str], Optional[str]]:
        """The shots the exporter declared, as their clip names in timeline
        order -- one take per shot, each named after its shot (the
        ``animation`` record, ``SceneRecords.SHOTS``) -- and why none can be
        read: a record of another version is refused, its cure a re-export.

        Returns:
            ``(clip names, problem)``; ``([], None)`` when the scene declares
            no shots (a file of one whole-timeline take).
        """
        export = (sidecar or {}).get("data_export") or {}
        record = export.get(SceneRecords.SHOTS.key) if isinstance(export, dict) else None
        if not record:
            return [], None
        if not SceneRecords.SHOTS.is_current(record):
            found = record.get("version") if isinstance(record, dict) else None
            return [], (
                f"the {SceneRecords.SHOTS.key!r} record is version {found!r}, and "
                f"this reader reads version {SceneRecords.SHOTS.version}: its "
                "shots cannot be read -- re-export the file"
            )
        return [
            str(s["clip"])
            for s in record.get("shots") or []
            if isinstance(s, dict) and s.get("clip")
        ], None

    @staticmethod
    def _published_fps(sidecar: Optional[dict]) -> Optional[float]:
        """The scene's frame rate, as the exporter recorded it: the
        ``animation`` record's (the ``audio`` and ``render_effects`` records
        state no frame and no rate)."""
        export = (sidecar or {}).get("data_export") or {}
        for channel in (SceneRecords.SHOTS.key,):
            rate = (export.get(channel) or {}).get("fps")
            try:
                if rate and float(rate) > 0:
                    return float(rate)
            except (TypeError, ValueError):
                continue
        return None


class ExportVerifier(_ExportVerifierInternal):
    """Run file-level gates over an exported GLB and/or FBX.

    Parameters:
        glb: Path to the ``.glb`` deliverable (optional).
        fbx: Path to the ``.fbx`` deliverable (optional).
        sidecar: ``"auto"`` (default) finds either input's
            ``.{stem}.scene_data.json`` record (the exporters' per-user store,
            then beside it); a path uses that file; ``None`` disables
            sidecar-dependent gates (they SKIP).
        baseline_glb: A previous known-good GLB to diff structure against
            (counts, image mimes, clip names). ``None`` -> that gate SKIPs.
        fps: Frame rate for converting clip seconds to frames. ``None``
            (default) takes the rate the sidecar publishes, else the FBX's
            own, falling back to 30 -- every frame the gates quote is a
            timeline frame, and a
            24 or 60 fps scene compared at 30 makes each of them wrong by the
            ratio. Pass a number only to override a sidecar that is itself
            suspect.
        huge: World-unit bound for the NaN/garbage scan.
        max_image_bytes: Size past which :meth:`check_glb_image_bytes` warns
            about an embedded image. ``None`` (default) only reports.

    Example:
        >>> report = ExportVerifier(glb="asset.glb", fbx="asset.fbx").run()
        >>> report.ok
        True
        >>> print(report.summary())
    """

    def __init__(
        self,
        glb: Optional[str] = None,
        fbx: Optional[str] = None,
        sidecar: Union[str, None] = "auto",
        baseline_glb: Optional[str] = None,
        fps: Optional[float] = None,
        huge: float = 1e7,
        max_image_bytes: Optional[int] = None,
    ):
        if not glb and not fbx:
            raise ValueError("ExportVerifier needs a glb and/or an fbx path.")
        self.glb_path = glb
        self.fbx_path = fbx
        self.baseline_glb = baseline_glb
        self.huge = huge
        self.max_image_bytes = max_image_bytes

        if sidecar == "auto":
            sidecar = self._sidecar_beside(glb) or self._sidecar_beside(fbx)
        self.sidecar_path = sidecar
        self.sidecar: Optional[dict] = None
        self.sidecar_error: Optional[str] = None
        if sidecar and os.path.isfile(sidecar):
            # A verifier's inputs are exactly the files most likely to be
            # broken — a corrupt sidecar degrades its gates to SKIP with the
            # reason, it must never crash the run before the first gate.
            try:
                with open(sidecar, encoding="utf-8") as handle:
                    self.sidecar = json.load(handle)
            except (OSError, ValueError) as e:
                self.sidecar_error = f"sidecar unreadable: {e}"

        self._reader: Optional[GlbReader] = None
        self._reader_error: Optional[str] = None
        self._fbx: Optional[FbxFile] = None
        self._fbx_error: Optional[str] = None

        # After the sidecar, because that is where the scene's rate is
        # published (else the FBX states its own). Every frame these gates
        # quote is a timeline frame converted from clip SECONDS, so assuming
        # 30 on a 24 or 60 fps scene scales every one of them -- and the
        # caller that matters here (TaskManager.verify_deliverables) passes
        # only paths.
        self.fps = (
            float(fps)
            if fps
            else (
                self._published_fps(self.sidecar)
                or (self.fbx._fps() if self.fbx is not None else None)
                or 30.0
            )
        )

    # ---- lazy inputs ------------------------------------------------------

    @property
    def reader(self) -> Optional[GlbReader]:
        if self._reader is None and self._reader_error is None and self.glb_path:
            try:
                self._reader = GlbReader.load(self.glb_path)
            except (OSError, ValueError) as e:
                self._reader_error = str(e)
        return self._reader

    @property
    def fbx(self) -> Optional[FbxFile]:
        """The FBX, parsed once: no gate reads its embedded media, and the take
        gates read each curve's extent alone (``span_arrays``), so a
        production file's textures and per-frame keys are never held."""
        if self._fbx is None and self._fbx_error is None and self.fbx_path:
            try:
                self._fbx = FbxFile.load(
                    self.fbx_path, raw_payloads=False, span_arrays=("KeyTime",)
                )
            except (OSError, ValueError) as e:
                self._fbx_error = str(e)
        return self._fbx

    # ---- runner -----------------------------------------------------------

    def gate_names(self) -> List[str]:
        """Every registered gate, in run order.

        Named gate_names, NOT check_*: the prefix discovery below
        would otherwise pick the runner's own helper up as a gate. A retired
        gate (PEP 702 ``__deprecated__``) answers through its successor and
        is no gate of its own.
        """
        return sorted(
            name
            for name in dir(self)
            if name.startswith("check_")
            and callable(getattr(self, name))
            and not getattr(getattr(type(self), name, None), "__deprecated__", None)
        )

    def run(self, checks: Optional[Sequence[str]] = None) -> VerificationReport:
        """Run *checks* (default: all) and return the report."""
        report = VerificationReport()
        for name in checks or self.gate_names():
            method = getattr(self, name, None)
            if not callable(method):
                report.rows.append(Finding(FAIL, name, "unknown check"))
                continue
            try:
                report.rows.extend(method())
            except Exception as e:  # a broken gate must not hide the others
                report.rows.append(Finding(FAIL, name, f"check raised: {e}"))
        return report

    # ---- GLB gates --------------------------------------------------------

    def check_glb_container(self) -> List[Finding]:
        """The GLB parses and carries a scene graph."""
        if not self.glb_path:
            return [Finding(SKIP, "glb_container", "no GLB given")]
        if self.reader is None:
            return [Finding(FAIL, "glb_container", self._reader_error or "unreadable")]
        counts = self.reader.counts()
        if not counts["nodes"]:
            return [Finding(FAIL, "glb_container", "no nodes")]
        return [
            Finding(
                PASS,
                "glb_container",
                f"nodes={counts['nodes']} meshes={counts['meshes']} "
                f"materials={counts['materials']} animations={counts['animations']}",
            )
        ]

    def check_glb_extensions(self) -> List[Finding]:
        """``extensionsRequired`` must be a subset of ``extensionsUsed``.

        A PASS also names what is required -- a reader without it must refuse
        the file -- which is why :meth:`check_glb_handoff` does not warn on it.
        """
        if self.reader is None:
            return [Finding(SKIP, "glb_extensions", "no readable GLB")]
        used, required = self.reader.extensions()
        missing = sorted(set(required) - set(used))
        if missing:
            return [
                Finding(FAIL, "glb_extensions", f"required but not used: {missing}")
            ]
        detail = f"used={used or 'none'}"
        if required:
            detail += f"; required={sorted(required)}"
        return [Finding(PASS, "glb_extensions", detail)]

    def check_glb_images(self) -> List[Finding]:
        """Texture sources resolve; basisu usage is declared, and falls back
        unless the file declares the extension REQUIRED."""
        if self.reader is None:
            return [Finding(SKIP, "glb_images", "no readable GLB")]
        gltf = self.reader.gltf
        images = gltf.get("images") or []
        rows: List[Finding] = []
        broken = []
        basisu_used = False
        fallbackless = 0
        for i, texture in enumerate(gltf.get("textures") or []):
            source = texture.get("source")
            basisu = (texture.get("extensions") or {}).get("KHR_texture_basisu")
            if basisu:
                basisu_used = True
                b_source = basisu.get("source")
                if not isinstance(b_source, int) or not 0 <= b_source < len(images):
                    broken.append(f"texture {i} basisu source {b_source!r}")
                if source is None:
                    fallbackless += 1
                    continue
            if source is not None and not (
                isinstance(source, int) and 0 <= source < len(images)
            ):
                broken.append(f"texture {i} source {source!r}")
        if broken:
            rows.append(Finding(FAIL, "glb_images", f"unresolvable: {broken[:5]}"))
        used, required = self.reader.extensions()
        if basisu_used and "KHR_texture_basisu" not in used:
            rows.append(
                Finding(FAIL, "glb_images", "basisu textures but extension undeclared")
            )
        if fallbackless:
            # Missing a fallback is only a defect while the file still claims a
            # reader without the extension can open it. Once basisu is
            # REQUIRED, shipping no PNG twin is the declared contract -- the
            # web-delivery policy writes exactly this shape on purpose
            # (``MeshConvert.drop_glb_texture_fallbacks`` retrofits it), and it
            # is what makes the KTX2 pass a saving rather than a second copy. Warning on a deliberate delivery mode
            # every run trains the reader past the gate that would name a real
            # one; the declaration is what tells the two apart.
            deliberate = "KHR_texture_basisu" in (required or [])
            rows.append(
                Finding(
                    PASS if deliberate else WARN,
                    "glb_images",
                    f"{fallbackless} basisu texture(s) carry no PNG/JPEG fallback"
                    + (
                        " -- KHR_texture_basisu is REQUIRED, so that is the "
                        f"declared contract; mimes={self.reader.image_mimes()}"
                        if deliberate
                        else ""
                    ),
                )
            )
        if not rows:
            rows.append(
                Finding(PASS, "glb_images", f"mimes={self.reader.image_mimes()}")
            )
        return rows

    def check_glb_image_bytes(self) -> List[Finding]:
        """What each embedded image costs, largest first; WARN past
        ``max_image_bytes``.

        The GLB's own bytes, after every resize and re-encode the build
        applied -- the number a source-texture size limit only stands in for:
        a production 57 MB source PNG shipped as a 3.12 MB KTX2. Read from the
        ``bufferViews`` in the JSON chunk, so nothing is decoded; an image
        referenced by ``uri`` is not embedded and is not counted. Never FAILs:
        the file already shipped, and a large image is a cost, not a defect.
        """
        if self.reader is None:
            return [Finding(SKIP, "glb_image_bytes", "no readable GLB")]
        views = self.reader.gltf.get("bufferViews") or []
        sized = []
        for i, image in enumerate(self.reader.gltf.get("images") or []):
            view = image.get("bufferView")
            if not isinstance(view, int) or not 0 <= view < len(views):
                continue
            kind = (image.get("mimeType") or "?").split("/")[-1]
            label = f"{image.get('name') or f'image {i}'} ({kind})"
            sized.append((int((views[view] or {}).get("byteLength") or 0), label))
        if not sized:
            return [Finding(SKIP, "glb_image_bytes", "no embedded images")]
        sized.sort(reverse=True)

        def listing(entries) -> str:
            shown = ", ".join(
                f"{label} {FileUtils.format_bytes(size)}" for size, label in entries[:3]
            )
            return shown + (f" (+{len(entries) - 3} more)" if len(entries) > 3 else "")

        total = f"{FileUtils.format_bytes(sum(size for size, _ in sized))} total"
        limit = self.max_image_bytes
        over = [entry for entry in sized if limit and entry[0] > limit]
        if over:
            detail = (
                f"{len(over)} of {len(sized)} image(s) past "
                f"{FileUtils.format_bytes(limit)}: {listing(over)}; {total}"
            )
            return [Finding(WARN, "glb_image_bytes", detail)]
        detail = f"{len(sized)} image(s), {total}; largest: {listing(sized)}"
        return [Finding(PASS, "glb_image_bytes", detail)]

    def check_glb_skin_weights(self) -> List[Finding]:
        """Every skinned mesh carries the weights its FBX clusters hold.

        FBX2glTF merges identical vertices across meshes and hands a later
        skinned mesh an earlier one's weights -- a duplicated rigged prop whose
        chain runs the other way binds to the wrong end of it
        (:class:`~pythontk.file_utils.mesh_convert.glb.skin_weights.GlbSkinWeights`,
        which the conversion runs). Read back off the shipped pair; the FBX's
        geometry is read only when a skinned mesh shares a vertex position with
        another mesh, the one way the merge can happen.
        """
        from pythontk.file_utils.mesh_convert.glb.skin_weights import GlbSkinWeights

        name = "glb_skin_weights"
        if self.reader is None:
            return [Finding(SKIP, name, "no readable GLB")]
        if not (self.fbx_path and os.path.isfile(self.fbx_path)):
            return [Finding(SKIP, name, "no FBX to compare against")]
        try:
            audit = GlbSkinWeights.audit(self.glb_path, self.fbx_path)
        except (OSError, ValueError) as e:
            return [Finding(SKIP, name, f"not compared: {e}")]
        rows: List[Finding] = []
        if audit["refused"]:
            rows.append(
                Finding(
                    WARN,
                    name,
                    f"{len(audit['refused'])} skinned mesh(es) share vertex "
                    "positions with another mesh but could not be held against "
                    "the FBX (possibly carrying another mesh's weights): "
                    + ", ".join(audit["refused"][:5]),
                )
            )
        if not audit["compared"]:
            if not rows:
                rows.append(
                    Finding(
                        PASS,
                        name,
                        "no skinned mesh shares a vertex position with another "
                        "mesh (nothing the converter could merge)",
                    )
                )
            return rows
        note = (
            f"; {audit['ambiguous']} vertex(es) share a position with a "
            "differently weighted point and were not compared"
            if audit["ambiguous"]
            else ""
        )
        mismatched = audit["mismatched"]
        if mismatched:
            meshes = ", ".join(f"{m} ({n})" for m, n in sorted(mismatched.items())[:5])
            detail = (
                f"{sum(mismatched.values())} vertex(es) on {len(mismatched)} "
                f"skinned mesh(es) disagree with the FBX's clusters: {meshes}{note}"
            )
            return rows + [Finding(FAIL, name, detail)]
        detail = (
            f"{len(audit['compared'])} skinned mesh(es) sharing vertex positions "
            f"match the FBX's clusters{note}"
        )
        return rows + [Finding(PASS, name, detail)]

    def check_glb_skins(self) -> List[Finding]:
        """Referenced skins need inverseBindMatrices and a real skeleton root.

        Converters mint bookkeeping skins nothing references — FBX2glTF
        wrote 56–93 IBM-less stubs on measured production files — and no
        viewer reads a skin no node points at. Failing on those buries the
        real invariant: every skin a mesh node ACTUALLY references resolves
        and carries inverseBindMatrices.

        A referenced skin's ``skeleton``, when present, must be a common root
        of its joints (glTF: the closest one or an ancestor of it). three.js
        never reads the field, so only a validator notices a wrong one: a
        production assembly shipped 7 (2026-09-14).
        """
        if self.reader is None:
            return [Finding(SKIP, "glb_skins", "no readable GLB")]
        gltf = self.reader.gltf
        skins = gltf.get("skins") or []
        if not skins:
            return [Finding(PASS, "glb_skins", "no skins (nothing to check)")]
        referenced = sorted(
            {n.get("skin") for n in gltf.get("nodes") or [] if "skin" in n}
        )
        bad = [
            i
            for i in referenced
            if not (isinstance(i, int) and 0 <= i < len(skins))
            or "inverseBindMatrices" not in skins[i]
        ]

        def rooted(skin: dict) -> bool:
            skeleton = skin.get("skeleton")
            if not isinstance(skeleton, int):
                return True  # optional: a reader finds the root itself
            for joint in skin.get("joints") or []:
                node, seen = joint, set()
                while node is not None and node != skeleton and node not in seen:
                    seen.add(node)
                    node = self.reader.parent_of(node)
                if node != skeleton:
                    return False
            return True

        misrooted = [
            i
            for i in referenced
            if isinstance(i, int) and 0 <= i < len(skins) and not rooted(skins[i])
        ]
        rows: List[Finding] = []
        if misrooted:
            rows.append(
                Finding(
                    FAIL,
                    "glb_skins",
                    f"{len(misrooted)} referenced skin(s) name a skeleton their "
                    f"joints do not all hang under: {misrooted[:5]}",
                )
            )
        if bad:
            rows.append(
                Finding(
                    FAIL,
                    "glb_skins",
                    f"{len(bad)} referenced skin(s) missing "
                    f"inverseBindMatrices: {bad[:5]}",
                )
            )
        referenced_set = set(referenced)
        stubs = sum(
            1
            for i, skin in enumerate(skins)
            if i not in referenced_set and "inverseBindMatrices" not in skin
        )
        if stubs:
            rows.append(
                Finding(
                    WARN,
                    "glb_skins",
                    f"{stubs} unreferenced stub skin(s) without "
                    "inverseBindMatrices (converter bookkeeping; harmless)",
                )
            )
        if not bad and not misrooted:
            rows.append(
                Finding(
                    PASS,
                    "glb_skins",
                    f"{len(referenced)} referenced skin(s), all with "
                    "inverseBindMatrices",
                )
            )
        return rows

    def check_glb_animation_integrity(self) -> List[Finding]:
        """Channels resolve to real nodes/samplers; no NaN/huge output bounds."""
        if self.reader is None:
            return [Finding(SKIP, "glb_animation", "no readable GLB")]
        gltf = self.reader.gltf
        node_count = len(gltf.get("nodes") or [])
        rows: List[Finding] = []
        dangling = []
        for i, anim in enumerate(gltf.get("animations") or []):
            samplers = anim.get("samplers") or []
            for j, channel in enumerate(anim.get("channels") or []):
                target = channel.get("target") or {}
                node = target.get("node")
                if node is not None and not (
                    isinstance(node, int) and 0 <= node < node_count
                ):
                    dangling.append(f"anim {i} channel {j} node {node!r}")
                sampler = channel.get("sampler")
                if not (isinstance(sampler, int) and 0 <= sampler < len(samplers)):
                    dangling.append(f"anim {i} channel {j} sampler {sampler!r}")
        if dangling:
            rows.append(Finding(FAIL, "glb_animation", f"dangling: {dangling[:5]}"))
        nan = self.reader.nan_findings(self.huge, deep=True)
        if nan:
            rows.append(Finding(FAIL, "glb_animation", f"NaN/huge: {nan[:5]}"))
        if not rows:
            clips = self.reader.clip_spans(self.fps)
            rows.append(
                Finding(
                    PASS,
                    "glb_animation",
                    f"{len(clips)} clip(s), longest ends f"
                    f"{max((v[2] for v in clips.values()), default=0)}",
                )
            )
        return rows

    def check_glb_handoff(self) -> List[Finding]:
        """Delegate to :meth:`MeshConvert.verify_glb` when a handoff rides.

        Its notes WARN, except the extension prerequisite, which
        :meth:`check_glb_extensions` states.
        """
        if self.reader is None:
            return [Finding(SKIP, "glb_handoff", "no readable GLB")]
        extras = GlbEdit.extras_of(self.reader.gltf)
        if not extras.get(HandoffRecord.KEY):
            return [Finding(SKIP, "glb_handoff", "no embedded handoff")]
        result = MeshConvert.verify_glb(self.glb_path, described=self._packed_materials())
        rows = [
            Finding(FAIL, "glb_handoff", problem)
            for problem in result.get("problems") or []
        ]
        # A declared requirement is the extension gate's fact, PASSed there as
        # the delivery contract it is; restated here it warned on every WebP
        # and every fallback-free KTX2 deliverable.
        required = (result.get("extensions") or {}).get("required")
        stated = MeshConvert._requirement_note(required) if required else None
        rows.extend(
            Finding(WARN, "glb_handoff", note)
            for note in result.get("notes") or []
            if note != stated
        )
        if not rows:
            rows.append(Finding(PASS, "glb_handoff", "handoff verified"))
        return rows

    def _packed_materials(self) -> Optional[List[str]]:
        """The materials whose metal / roughness packing the export wrote --
        those the FBX's ``materials`` record binds a metallicRoughness image
        on, which the GLB's materials are -- or ``None`` with no readable FBX
        or record: what :meth:`MeshConvert.verify_glb` exempts from its ORM
        finding (a fully metal surface packs blue white, as authored)."""
        if self.fbx is None:
            return None
        from pythontk.core_utils.engines.scene_export.material_record import (
            MaterialFamilies,
        )

        record = MeshConvert._carried_record(self.fbx)
        if record is None or SceneRecords.problems(SceneRecords.MATERIALS, record):
            return None
        return [
            name
            for name, material in MaterialFamilies.materials_by_name(record).items()
            if "metallicRoughnessTexture" in MaterialFamilies.material_files(record, material)
        ]

    @Deprecation.symbol(
        "ExportVerifier.check_glb_handoff -- a GLB carries extras.handoff, "
        "not an envelope",
        remove_in="0.18.0",
        since="2026-10-24",
    )
    def check_glb_envelope(self) -> List[Finding]:
        """Retired 2026-10-20 (removed in 0.18.0): :meth:`check_glb_handoff`."""
        return self.check_glb_handoff()

    # ---- sidecar gates ----------------------------------------------------

    def check_clips_vs_takes(self) -> List[Finding]:
        """The GLB's clips are the declared shots' takes: one per shot, named
        after it, and no other -- a file with shots carries no whole-timeline
        clip beside them -- each as long as its take in the FBX, when the FBX
        is given (one frame of tolerance).

        A record of another version FAILs with its cure (re-export): its
        shots cannot be read.
        """
        if self.reader is None:
            return [Finding(SKIP, "clips_vs_takes", "no readable GLB")]
        clips, problem = self._declared_clips(self.sidecar)
        if problem:
            return [Finding(FAIL, "clips_vs_takes", problem)]
        if not clips:
            detail = self.sidecar_error or "no sidecar shots"
            return [Finding(SKIP, "clips_vs_takes", detail)]
        spans = self.reader.clip_spans(self.fps)
        rows: List[Finding] = []
        missing = [clip for clip in clips if clip not in spans]
        if missing:
            rows.append(
                Finding(FAIL, "clips_vs_takes", f"declared but absent: {missing}")
            )
        named = set(clips)
        extra = [name for name in spans if name not in named]
        if extra:
            rows.append(
                Finding(
                    FAIL,
                    "clips_vs_takes",
                    f"clip(s) no shot names: {extra} -- a file with shots "
                    "carries one take per shot and no whole-timeline take",
                )
            )
        windows = self.fbx.take_windows() if self.fbx is not None else {}
        for clip in clips:
            if clip not in spans or clip not in windows:
                continue
            low, high, _end = spans[clip]
            got = int(round((high - low) * self.fps))
            start, end = windows[clip]
            want = int(round((end - start) * self.fps))
            if abs(got - want) > 1:
                rows.append(
                    Finding(
                        FAIL,
                        "clips_vs_takes",
                        f"{clip}: {got}f vs its take's {want}f",
                    )
                )
        if not rows:
            rows.append(
                Finding(
                    PASS,
                    "clips_vs_takes",
                    f"{len(clips)} shot clip(s), one per take",
                )
            )
        return rows

    def check_glb_shot_clips(self) -> List[Finding]:
        """Every shot the GLB's ``animation`` record names plays the glTF
        animation of its clip's name -- the join every reader of the GLB
        makes (the record rides it as it stands). Read off the GLB alone."""
        name = "glb_shot_clips"
        if self.reader is None:
            return [Finding(SKIP, name, "no readable GLB")]
        record = SceneRecords.glb_block(self.reader.gltf, SceneRecords.SHOTS)
        if not isinstance(record, dict):
            return [Finding(SKIP, name, f"no extras.{SceneRecords.SHOTS.key} record")]
        shots = [
            str(s["clip"])
            for s in record.get("shots") or []
            if isinstance(s, dict) and s.get("clip")
        ]
        clips = set(self.reader.animations())
        missing = [clip for clip in shots if clip not in clips]
        if missing:
            return [
                Finding(
                    FAIL,
                    name,
                    f"shot(s) with no animation of their clip's name: {missing}",
                )
            ]
        return [Finding(PASS, name, f"{len(shots)} shot(s), each its clip by name")]

    def check_animation_rate(self) -> List[Finding]:
        """The rate the ``animation`` record states is the FBX's own (its
        ``GlobalSettings`` time mode): the record states it for a reader of a
        format that holds none (a GLB's), and a reader of the FBX uses the
        file's -- so the two must be one rate. Each record read -- the
        sidecar's, the GLB's -- is held to it."""
        name = "animation_rate"
        rate = self.fbx._fps() if self.fbx is not None else None
        if not rate:
            return [Finding(SKIP, name, "no FBX that states a time mode")]
        stated = []
        export = (self.sidecar or {}).get("data_export") or {}
        held = export.get(SceneRecords.SHOTS.key) if isinstance(export, dict) else None
        if isinstance(held, dict) and SceneRecords.SHOTS.is_current(held):
            stated.append(("the sidecar's record", held.get("fps")))
        if self.reader is not None:
            carried = SceneRecords.glb_block(self.reader.gltf, SceneRecords.SHOTS)
            if isinstance(carried, dict):
                stated.append(("the GLB's record", carried.get("fps")))
        stated = [(where, fps) for where, fps in stated if isinstance(fps, (int, float))]
        if not stated:
            return [Finding(SKIP, name, "no record states a rate")]
        rows = [
            Finding(
                FAIL,
                name,
                f"{where} states {fps:g} fps; the FBX's time mode is {rate:g}",
            )
            for where, fps in stated
            if abs(float(fps) - rate) > 1e-3
        ]
        return rows or [
            Finding(PASS, name, f"{len(stated)} record(s) at the FBX's {rate:g} fps")
        ]

    # ---- FBX gates --------------------------------------------------------

    def check_fbx_container(self) -> List[Finding]:
        """The FBX parses; header version is sane."""
        if not self.fbx_path:
            return [Finding(SKIP, "fbx_container", "no FBX given")]
        if self.fbx is None:
            return [Finding(FAIL, "fbx_container", self._fbx_error or "unreadable")]
        return [
            Finding(
                PASS,
                "fbx_container",
                f"version {self.fbx.version}, {len(self.fbx.roots)} top sections",
            )
        ]

    def check_fbx_dependencies(self) -> List[Finding]:
        """Every file the FBX's records name by file name is inside it.

        ``lightmaps``' maps and reflection probe, ``shadows``' silhouette,
        atlas and horizon maps, ``materials``' maps: the exporters embed each one
        (``FbxMedia.embed_dependencies``) and ship nothing beside the file, so
        a consumer resolves them from the FBX alone. One the file lacks is a
        Unity import without its lightmap, and a GLB built from the file
        without its lighting. The records as a reader applies them
        (``FbxMedia.reader_records``): a referenced module's copy beside the
        scene's own, or naming nodes the file does not carry, binds nothing
        and names no file the FBX needs; an artist's property of a record's
        name on another node is none (``SceneStoreBase.reader_values``). Read
        off the parsed file: no media is held.
        """
        from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia

        name = "fbx_dependencies"
        if self.fbx is None:
            return [Finding(SKIP, name, "no readable FBX")]
        channels = []
        for key in FbxMedia.DEPENDENCY_FIELDS:
            for owner, value in self.fbx.user_properties(key, named=True, paths=True):
                if not isinstance(value, (bytes, bytearray)):
                    continue  # a number: no record's text (bytes(n) is n zeros)
                try:
                    payload = json.loads(bytes(value).decode("utf-8"))
                except (TypeError, ValueError, UnicodeDecodeError):
                    continue
                channels.append((owner, key, payload))
        read = FbxMedia.reader_records(channels, nodes=self.fbx.object_names("Model"))
        named = FbxMedia.named_files({key: [value] for key, value in read.items()})
        if not named:
            return [Finding(SKIP, name, "the records name no file")]
        inside = {media.casefold() for media in self._embedded_names(self.fbx)}
        missing = [n for n in named if n.casefold() not in inside]
        if missing:
            detail = (
                f"{len(missing)} of {len(named)} named file(s) not in the FBX: "
                + ", ".join(missing)
            )
            return [Finding(FAIL, name, detail)]
        return [Finding(PASS, name, f"{len(named)} named file(s) embedded")]

    @staticmethod
    def _embedded_names(fbx: FbxFile) -> List[str]:
        """The file names of the media *fbx* embeds: every ``Video`` whose
        ``Content`` is not empty, named by its ``Filename``'s basename."""
        names: List[str] = []
        for record in fbx.iter_objects():
            if record["name"] != "Video":
                continue
            fields = {child["name"]: child["props"] for child in record["children"]}
            content = (fields.get("Content") or [b""])[0]
            if isinstance(content, tuple):  # ("RAW", length): payloads not read
                content = b"\0" * bool(content[-1])
            if not content:
                continue
            raw = (fields.get("Filename") or fields.get("RelativeFilename") or [b""])[0]
            text = bytes(raw).decode("utf-8", "replace").replace("\\", "/")
            names.append(os.path.basename(text))
        return names

    def check_fbx_records(self) -> List[Finding]:
        """Every record the FBX carries is one the standard declares, in the
        version a reader reads and the shape it declares
        (``SceneRecords.problems``).

        Read off the ``data_export`` carriers -- the scene's and any
        referenced module's. A property there that is a record by its shape
        (a JSON object stating a ``version``) under a key no record declares
        is one a reader passes over: an older file's, which is re-exported; a
        record retired whole (``SceneRecords.RETIRED_FIXES``) is named with
        its fix. A carrier's other properties -- an old file's keyable
        floats -- are not records.
        """
        from pythontk.core_utils.engines.scene_export.scene_store import SceneStoreBase

        name = "fbx_records"
        if self.fbx is None:
            return [Finding(SKIP, name, "no readable FBX")]
        declared = {spec.key for spec in SceneRecords.deliverable()}
        problems: List[str] = []
        read = 0
        for owner, key, value in self.fbx.user_property_items():
            if not SceneStoreBase.is_carrier(owner) or not isinstance(
                value, (bytes, bytearray)
            ):
                continue  # a number is no record's text (bytes(n) is n zeros)
            try:
                payload = json.loads(bytes(value).decode("utf-8"))
            except (TypeError, ValueError, UnicodeDecodeError):
                continue  # not JSON text: no record (a keyable float)
            if key in declared:
                read += 1
                problems += [f"{owner}: {p}" for p in SceneRecords.problems(key, payload)]
            elif key in SceneRecords.RETIRED_FIXES:
                problems.append(
                    f"{owner}: {key!r} is a retired record: "
                    f"{SceneRecords.RETIRED_FIXES[key]}"
                )
            elif isinstance(payload, dict) and "version" in payload:
                problems.append(f"{owner}: {key!r} is no declared record")
        if problems:
            return [Finding(FAIL, name, problem) for problem in problems]
        if not read:
            return [Finding(SKIP, name, "the FBX carries no record")]
        return [Finding(PASS, name, f"{read} record(s) in their declared version and shape")]

    def check_fbx_render_effects(self) -> List[Finding]:
        """The ``render_effects`` record names exactly the curves the FBX
        carries: one curve proxy (``SceneRecords.TRANSPORT``) per channel each
        track and material entry names, and no proxy the record does not name
        -- the record says what carries what, the proxies are the curves, and
        a reader joins the two. A node channel plays on its proxy's parent
        (``SceneRecords.TRANSPORT.join``), so each track's channel has a proxy
        of its own: two tracks whose nodes share a leaf on one proxy ship one
        curve, and two proxies of one name two curves a reader keys as one.
        The record is the one a reader applies: the scene's own carrier's,
        else the first module's (``SceneStoreBase.reader_values``).
        """
        from pythontk.core_utils.engines.scene_export.scene_store import SceneStoreBase

        name = "fbx_render_effects"
        if self.fbx is None:
            return [Finding(SKIP, name, "no readable FBX")]
        spec, transport = SceneRecords.VISIBILITY, SceneRecords.TRANSPORT
        proxies: set = set()
        carried: Dict[Tuple[str, str], int] = {}
        rows = []
        for owner, key, value in self.fbx.user_property_items(paths=True):
            leaf = (owner or "").rsplit("|", 1)[-1]
            if key == transport.marker and value and transport.parse(leaf):
                proxies.add(transport.parse(leaf))
                carried[transport.parse(leaf)] = carried.get(transport.parse(leaf), 0) + 1
            elif key == spec.key and isinstance(value, (bytes, bytearray)):
                rows.append((owner, True, key, value))
        record = None
        chosen = SceneStoreBase.reader_values(rows).get(spec.key)
        if chosen is not None:
            try:
                record = json.loads(bytes(chosen[0]).decode("utf-8"))
            except (TypeError, ValueError, UnicodeDecodeError):
                return [Finding(FAIL, name, f"the {spec.key!r} record is not JSON")]
        if record is None and not proxies:
            return [Finding(SKIP, name, "the FBX carries no render effect")]
        if record is not None and not spec.is_current(record):
            return [
                Finding(
                    FAIL,
                    name,
                    f"the {spec.key!r} record is version {record.get('version')!r}; "
                    f"this reader reads version {spec.version} -- re-export the file",
                )
            ]
        # The record names a target as the delivery (a material, namespace
        # included) or its host (a node) does; its proxy by the transport's
        # name for it. The pair joins on that name; a failure says the
        # record's own.
        named = set()
        said: Dict[Tuple[str, str], str] = {}
        tracked: Dict[Tuple[str, str], List[str]] = {}
        for section, target in (("tracks", "node"), ("materials", "material")):
            for entry in (record or {}).get(section) or ():
                if isinstance(entry, dict) and entry.get(target):
                    for channel in entry.get("channels") or ():
                        pair = (transport.target_name(entry[target]), str(channel))
                        named.add(pair)
                        said.setdefault(pair, str(entry[target]))
                        if section == "tracks" and pair[1] in transport.node_channels:
                            tracked.setdefault(pair, []).append(str(entry[target]))
        problems = [
            f"the record names {said[(target, channel)]}'s {channel} but the "
            f"file carries no {transport.proxy_name(target, channel)!r} proxy"
            for target, channel in sorted(named - proxies)
            if channel in transport.channels
        ] + [
            f"a {transport.proxy_name(target, channel)!r} proxy the record does not name"
            for target, channel in sorted(proxies - named)
        ]
        for (target, channel), nodes in sorted(tracked.items()):
            count = carried.get((target, channel), 0)
            proxy = transport.proxy_name(target, channel)
            if count and len(nodes) > count:
                problems.append(
                    f"the record names {len(nodes)} nodes' {channel} "
                    f"({', '.join(sorted(nodes))}) but the file carries {count} "
                    f"{proxy!r} prox{'y' if count == 1 else 'ies'}: a reader plays "
                    f"each curve on its proxy's parent alone"
                )
            elif count > len(nodes):
                problems.append(
                    f"the file carries {count} {proxy!r} proxies for "
                    f"{len(nodes)} node{'' if len(nodes) == 1 else 's'}: a reader "
                    f"keys their curves by the one name and reads the first"
                )
        if problems:
            return [Finding(FAIL, name, problem) for problem in problems]
        return [Finding(PASS, name, f"{len(named)} channel(s), each on its curve proxy")]

    def check_fbx_takes(self) -> List[Finding]:
        """The FBX carries one take per declared shot, named after it, and no
        other: a file with shots ships no whole-timeline take beside them."""
        if self.fbx is None:
            return [Finding(SKIP, "fbx_takes", "no readable FBX")]
        clips, problem = self._declared_clips(self.sidecar)
        if problem:
            return [Finding(FAIL, "fbx_takes", problem)]
        if not clips:
            detail = self.sidecar_error or "no sidecar shots"
            return [Finding(SKIP, "fbx_takes", detail)]
        stacks = self.fbx.take_names()
        missing = sorted(set(clips) - set(stacks))
        if missing:
            return [Finding(FAIL, "fbx_takes", f"declared but absent: {missing}")]
        named = set(clips)
        extra = [name for name in stacks if name not in named]
        if extra:
            return [
                Finding(
                    FAIL,
                    "fbx_takes",
                    f"take(s) no shot names: {extra} -- a file with shots "
                    "carries one take per shot and no whole-timeline take",
                )
            ]
        return [Finding(PASS, "fbx_takes", f"{len(clips)} shot take(s), one per shot")]

    #: Channels a failing take names before the rest are counted.
    _LISTED_CHANNELS = 3

    def check_fbx_take_channels(self) -> List[Finding]:
        """Every shot's take is an exact slice of the animation: it animates
        each channel any shot's take does, keyed across its own window (the
        span the take declares, :meth:`FbxFile.take_windows`).

        Maya's take split restricts each curve to a take's window before
        writing it unless every curve is resampled, so a curve with no key
        inside a take gives that take NO channel -- the node plays its rest
        pose for the whole shot in every consumer of the takes (Unity builds
        its clips from them) -- and a take's keys stop at its last in-window
        key, a clip shorter than its shot. Measured in Unity 6000.3: a 50-frame
        shot imported 40 frames long with two of three nodes frozen, the next
        shot as a clip with no curves at all. A take whose stack carries no
        curve is judged like any other: it lost every channel.
        """
        name = "fbx_take_channels"
        if self.fbx is None:
            return [Finding(SKIP, name, "no readable FBX")]
        clips, problem = self._declared_clips(self.sidecar)
        if problem:
            return [Finding(FAIL, name, problem)]
        if not clips:
            return [Finding(SKIP, name, self.sidecar_error or "no sidecar shots")]
        curves = self.fbx.take_curves()
        windows = self.fbx.take_windows()
        present = set(self.fbx.take_names())
        shot_takes = [take for take in clips if take in present]
        reference = set().union(*(set(curves.get(take, {})) for take in shot_takes))
        tick = FbxFile.TICKS_PER_SECOND / self.fps

        def listed(keys) -> str:
            keys = sorted(keys)
            shown = ", ".join(
                f"{t}.{p}.{c}" for t, p, c in keys[: self._LISTED_CHANNELS]
            )
            more = len(keys) - self._LISTED_CHANNELS
            return shown + (f" (+{more} more)" if more > 0 else "")

        rows: List[Finding] = []
        checked = 0
        for take in shot_takes:
            # A stack with no curve is not absent: it animates nothing, which
            # take_curves leaves out -- and check_fbx_takes reads names alone.
            channels = curves.get(take, {})
            checked += 1
            missing = reference - set(channels)
            if missing:
                rows.append(
                    Finding(
                        FAIL,
                        name,
                        f"{take}: {len(missing)} channel(s) another shot's take "
                        f"animates are absent -- they play their rest pose for "
                        f"the whole shot: {listed(missing)}",
                    )
                )
            window = windows.get(take)
            if window is None:
                continue
            lo = window[0] * FbxFile.TICKS_PER_SECOND
            hi = window[1] * FbxFile.TICKS_PER_SECOND
            short = [
                key
                for key, (first, last, _count) in channels.items()
                if first is not None and (first > lo + tick or last < hi - tick)
            ]
            if short:
                rows.append(
                    Finding(
                        FAIL,
                        name,
                        f"{take}: {len(short)} channel(s) keyed short of the take's "
                        f"{window[0] * self.fps:g}-{window[1] * self.fps:g} "
                        f"window: {listed(short)}",
                    )
                )
        if not checked:
            return [Finding(SKIP, name, "no declared take is in the file")]
        if rows:
            return rows
        return [
            Finding(
                PASS,
                name,
                f"{checked} take(s) carry all {len(reference)} animated "
                "channel(s) across their windows",
            )
        ]

    # ---- cross / baseline gates -------------------------------------------

    def check_cross_clips(self) -> List[Finding]:
        """GLB clip names exist as FBX stacks (the conversion kept them)."""
        if self.reader is None or self.fbx is None:
            return [Finding(SKIP, "cross_clips", "needs both GLB and FBX")]
        clips = set(self.reader.animations())
        stacks = set(self.fbx.take_names())
        missing = sorted(clips - stacks)
        # One synthesized whole-timeline clip is the converter's own addition.
        if len(missing) > 1:
            return [Finding(FAIL, "cross_clips", f"clips without stacks: {missing}")]
        return [
            Finding(
                PASS, "cross_clips", f"{len(clips)} clip(s) / {len(stacks)} stack(s)"
            )
        ]

    def check_baseline_diff(self) -> List[Finding]:
        """Structural drift vs a previous known-good GLB."""
        if self.reader is None:
            return [Finding(SKIP, "baseline_diff", "no readable GLB")]
        if not self.baseline_glb:
            return [Finding(SKIP, "baseline_diff", "no baseline given")]
        try:
            baseline = GlbReader.load(self.baseline_glb)
        except (OSError, ValueError) as e:
            return [Finding(FAIL, "baseline_diff", f"baseline unreadable: {e}")]
        rows: List[Finding] = []
        ours, theirs = self.reader.counts(), baseline.counts()
        for key in ("meshes", "materials", "skins", "animations", "cameras"):
            if ours[key] != theirs[key]:
                rows.append(
                    Finding(
                        WARN,
                        "baseline_diff",
                        f"{key}: {theirs[key]} -> {ours[key]}",
                    )
                )
        if self.reader.animations() != baseline.animations():
            rows.append(
                Finding(
                    FAIL,
                    "baseline_diff",
                    f"clip names changed: {baseline.animations()} -> "
                    f"{self.reader.animations()}",
                )
            )
        if not rows:
            rows.append(Finding(PASS, "baseline_diff", "structure matches baseline"))
        return rows


# -----------------------------------------------------------------------------
# CLI — private on purpose, like ``pythontk.__main__``.
# -----------------------------------------------------------------------------


def _main(argv: Optional[Sequence[str]] = None) -> int:
    from pythontk.core_utils.cli import CLI

    parser = CLI.get_parser("Verify exported FBX/GLB deliverables.")
    parser.add_argument(
        "paths",
        nargs="+",
        help=".glb and/or .fbx deliverables (extension decides; globs allowed)",
    )
    parser.add_argument("--sidecar", default="auto", help="path | auto | none")
    parser.add_argument("--baseline", help="previous known-good GLB to diff against")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="override the rate; default reads it from the sidecar (else 30)",
    )
    parser.add_argument("--json", action="store_true", help="machine output")
    parser.add_argument("--checks", help="comma-separated subset of gates")
    parser.add_argument(
        "--list-checks", action="store_true", help="print gate names and exit"
    )
    args = parser.parse_args(argv)

    glb = fbx = None
    for pattern in args.paths:
        for path in _glob.glob(pattern) or [pattern]:
            lower = path.lower()
            if lower.endswith(".glb"):
                glb = path
            elif lower.endswith(".fbx"):
                fbx = path
            else:
                parser.error(f"unrecognized deliverable extension: {path}")
    sidecar = None if args.sidecar == "none" else args.sidecar

    verifier = ExportVerifier(
        glb=glb,
        fbx=fbx,
        sidecar=sidecar,
        baseline_glb=args.baseline,
        fps=args.fps,
    )
    if args.list_checks:
        print("\n".join(verifier.gate_names()))
        return 0
    checks = args.checks.split(",") if args.checks else None
    report = verifier.run(checks)
    print(report.to_json() if args.json else report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
