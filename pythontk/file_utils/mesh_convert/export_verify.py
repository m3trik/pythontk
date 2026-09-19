# !/usr/bin/python
# coding=utf-8
"""Deliverable verification for exported FBX / GLB pairs.

The permanent form of a verification harness that caught a production GLB
shipping a wire-loom 7.5 cm away from the plug it was constrained to — a
defect no exporter log mentioned. Point it at a deliverable (either file or
both), optionally beside its ``.{stem}.scene_data.json`` sidecar and a
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
from typing import Dict, List, Optional, Sequence, Union

from pythontk.core_utils.export_profile import ExportProfile
from pythontk.core_utils.scene_records import SceneRecords
from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert
from pythontk.file_utils.mesh_convert.fbx_file import FbxFile
from pythontk.file_utils.mesh_convert.glb_reader import GlbReader

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
        """The exporter's ``.{stem}.scene_data.json`` beside *path*, if any.

        The deliverable's own first; failing that, a versioned deliverable's
        SERIES manifest -- the exporters key that one to the stem without its
        trailing ``_v<N>``, so every version diffs against one baseline, and a
        lookup by the file's own stem found nothing for any of them.
        """
        if not path:
            return None
        folder = os.path.dirname(os.path.abspath(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        series = ExportProfile.VERSION_SUFFIX_RE.sub("", stem)
        for candidate in dict.fromkeys((stem, series)):
            sidecar = os.path.join(folder, f".{candidate}.scene_data.json")
            if os.path.isfile(sidecar):
                return sidecar
        return None

    @staticmethod
    def _declared_takes(sidecar: Optional[dict]) -> List[dict]:
        """The takes the exporter declared: ``shot_metadata.takes`` (the shot
        record carries them since the fold), else the legacy ``fbx_takes``
        channel a sidecar written before it still holds."""
        return SceneRecords.declared_takes(
            ((sidecar or {}).get("data_export") or {}).get
        )

    @classmethod
    def _undeclared_clips(cls, spans, sidecar: Optional[dict]) -> List[str]:
        """Clip names no declared take claims, in file order.

        Exactly one of these is the converter's own whole-timeline stack --
        the clip whose length IS the stack's, and the only one that can show
        a wrong origin (a declared take is cut to its own window). Shared so
        ``clips_vs_takes`` and ``clip_origin`` cannot disagree about which
        clip that is.
        """
        declared = {t.get("name") for t in cls._declared_takes(sidecar)}
        return [name for name in spans if name not in declared]

    @staticmethod
    def _published_fps(sidecar: Optional[dict]) -> Optional[float]:
        """The scene's frame rate, as the exporter recorded it."""
        export = (sidecar or {}).get("data_export") or {}
        for channel in ("shot_metadata", "visibility_tracks"):
            rate = (export.get(channel) or {}).get("fps")
            try:
                if rate and float(rate) > 0:
                    return float(rate)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _published_clip_mode(sidecar: Optional[dict]) -> Optional[str]:
        """The Animation Clips mode the exporter DECLARED, or ``None``.

        Read, never inferred: a Full Sequence Only file carries one stack while
        its takes list every shot, and that is also exactly what a split that
        silently failed leaves.
        """
        export = (sidecar or {}).get("data_export") or {}
        meta = export.get(MeshConvert.SHOT_METADATA_KEY)
        mode = (
            meta.get(MeshConvert.SHOT_CLIP_MODE_KEY) if isinstance(meta, dict) else None
        )
        return mode if isinstance(mode, str) else None

    @staticmethod
    def _published_span(sidecar: Optional[dict]) -> Optional[List[float]]:
        """The whole-timeline ``clip_span`` entry every clip was cut against."""
        tracks = ((sidecar or {}).get("data_export") or {}).get("visibility_tracks")
        spans = (tracks or {}).get("clip_span")
        pair = (spans or {}).get(MeshConvert.DEFAULT_CLIP_SPAN)
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            try:
                return [float(pair[0]), float(pair[1])]
            except (TypeError, ValueError):
                return None
        return None


class ExportVerifier(_ExportVerifierInternal):
    """Run file-level gates over an exported GLB and/or FBX.

    Parameters:
        glb: Path to the ``.glb`` deliverable (optional).
        fbx: Path to the ``.fbx`` deliverable (optional).
        sidecar: ``"auto"`` (default) finds ``.{stem}.scene_data.json``
            beside either input; a path uses that file; ``None`` disables
            sidecar-dependent gates (they SKIP).
        baseline_glb: A previous known-good GLB to diff structure against
            (counts, image mimes, clip names). ``None`` -> that gate SKIPs.
        fps: Frame rate for converting clip seconds to frames. ``None``
            (default) takes the rate the sidecar publishes, falling back to
            30 -- every frame the gates quote is an AUTHORING frame, and a
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

        # After the sidecar, because that is where the scene's rate is
        # published. Every frame these gates quote is an authoring frame
        # converted from clip SECONDS, so assuming 30 on a 24 or 60 fps scene
        # scales every one of them -- and the caller that matters here
        # (TaskManager.verify_deliverables) passes only paths.
        self.fps = float(fps) if fps else (self._published_fps(self.sidecar) or 30.0)

        self._reader: Optional[GlbReader] = None
        self._reader_error: Optional[str] = None
        self._fbx: Optional[FbxFile] = None
        self._fbx_error: Optional[str] = None

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
        if self._fbx is None and self._fbx_error is None and self.fbx_path:
            try:
                self._fbx = FbxFile.load(self.fbx_path)
            except (OSError, ValueError) as e:
                self._fbx_error = str(e)
        return self._fbx

    # ---- runner -----------------------------------------------------------

    def gate_names(self) -> List[str]:
        """Every registered gate, in run order.

        Named gate_names, NOT check_*: the prefix discovery below
        would otherwise pick the runner's own helper up as a gate.
        """
        return sorted(
            name
            for name in dir(self)
            if name.startswith("check_") and callable(getattr(self, name))
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
        the file -- which is why :meth:`check_glb_envelope` does not warn on it.
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

    def check_glb_envelope(self) -> List[Finding]:
        """Delegate to :meth:`MeshConvert.verify_glb` when an envelope rides.

        Its notes WARN, except the extension prerequisite, which
        :meth:`check_glb_extensions` states.
        """
        if self.reader is None:
            return [Finding(SKIP, "glb_envelope", "no readable GLB")]
        if not (self.reader.gltf.get("extras") or {}).get("scene_sidecar"):
            return [Finding(SKIP, "glb_envelope", "no embedded envelope")]
        result = MeshConvert.verify_glb(self.glb_path)
        rows = [
            Finding(FAIL, "glb_envelope", problem)
            for problem in result.get("problems") or []
        ]
        # A declared requirement is the extension gate's fact, PASSed there as
        # the delivery contract it is; restated here it warned on every WebP
        # and every fallback-free KTX2 deliverable.
        required = (result.get("extensions") or {}).get("required")
        stated = MeshConvert._requirement_note(required) if required else None
        rows.extend(
            Finding(WARN, "glb_envelope", note)
            for note in result.get("notes") or []
            if note != stated
        )
        if not rows:
            rows.append(Finding(PASS, "glb_envelope", "envelope verified"))
        return rows

    # ---- sidecar gates ----------------------------------------------------

    def check_clips_vs_takes(self) -> List[Finding]:
        """Each GLB clip's length matches its declared take (±1 frame).

        Exactly one clip with no declared take is treated as the
        whole-timeline clip, measured from its own ``extras.zero_frame`` and
        compared against the takes' overall end -- ending SHORT of them FAILs
        (it cannot play the declared range), while running long is judged on
        the frames it ANIMATES rather than the frames it occupies: a bake pads
        it with held poses, so an inert margin is a WARN and only motion past
        the last take FAILs.
        """
        if self.reader is None:
            return [Finding(SKIP, "clips_vs_takes", "no readable GLB")]
        takes = self._declared_takes(self.sidecar)
        if not takes:
            detail = self.sidecar_error or "no sidecar takes"
            return [Finding(SKIP, "clips_vs_takes", detail)]
        spans = self.reader.clip_spans(self.fps)
        by_name = {t.get("name"): t for t in takes}
        rows: List[Finding] = []
        unmatched = self._undeclared_clips(spans, self.sidecar)
        for clip, (_low, _high, end_frame) in spans.items():
            take = by_name.get(clip)
            if take is None:
                continue
            want = int(take["end"]) - int(take["start"])
            if abs(end_frame - want) > 1:
                rows.append(
                    Finding(
                        FAIL,
                        "clips_vs_takes",
                        f"{clip}: {end_frame}f vs declared {want}f",
                    )
                )
        if len(unmatched) == 1:
            clip = unmatched[0]
            # The whole-timeline stack is REBASED: the converter puts its
            # first key at t=0, so its raw end frame is short of the takes'
            # end by however late the export starts. The clip publishes the
            # authoring frame it sits on; measure from there, or a correct
            # file reads as a failure (33 frames on the PROPS assembly).
            zero = 0.0
            for anim in self.reader.gltf.get("animations") or []:
                if anim.get("name") == clip:
                    value = (anim.get("extras") or {}).get("zero_frame")
                    if isinstance(value, (int, float)):
                        zero = float(value)
                    break
            end_frame = int(round(spans[clip][2] + zero))
            timeline_end = max(int(t["end"]) for t in takes)
            # The two ends fail for different reasons, so judge them apart.
            # SHORT is measured on keys: a clip whose keys stop before the
            # last take cannot play the declared range at all. LONG is
            # measured on motion: the bake writes keys across whatever range
            # it is handed, so a whole-timeline clip routinely ends on a held
            # pose past the last take -- inert padding, not an overrun. Only
            # motion out there is content no shot will ever play. Measured on
            # the PROPS assembly: 48 padded frames moving 76 of 1185 channels
            # by at most 3e-4, against a body moving 1183 of them by up to
            # 3.5. Motion that stops EARLY is never a fault -- a clip closing
            # on a deliberate hold is ordinary animation, not truncation.
            if end_frame < timeline_end - 1:
                rows.append(
                    Finding(
                        FAIL,
                        "clips_vs_takes",
                        f"{clip}: full-timeline clip ends {end_frame}f, "
                        f"short of the takes' {timeline_end}f",
                    )
                )
            elif end_frame > timeline_end + 1:
                motion = self.reader.motion_span(clip)
                motion_end = (
                    int(round(motion[1] * self.fps + zero))
                    if motion is not None
                    else timeline_end
                )
                excess = end_frame - timeline_end
                if motion_end > timeline_end + 1:
                    rows.append(
                        Finding(
                            FAIL,
                            "clips_vs_takes",
                            f"{clip}: full-timeline clip animates to "
                            f"{motion_end}f, past the takes' {timeline_end}f",
                        )
                    )
                else:
                    rows.append(
                        Finding(
                            WARN,
                            "clips_vs_takes",
                            f"{clip}: full-timeline clip ends {end_frame}f vs "
                            f"takes {timeline_end}f, but those {excess:+d}f "
                            f"hold still (bake padding, not content)",
                        )
                    )
        elif unmatched:
            rows.append(
                Finding(WARN, "clips_vs_takes", f"undeclared clips: {unmatched}")
            )
        missing = sorted(set(by_name) - set(spans))
        # A declared Full Sequence Only file keeps the shots as metadata; its
        # whole-timeline clip is still judged against their range above.
        if missing and self._published_clip_mode(self.sidecar) != "full":
            rows.append(
                Finding(FAIL, "clips_vs_takes", f"declared but absent: {missing}")
            )
        if not rows:
            rows.append(
                Finding(
                    PASS,
                    "clips_vs_takes",
                    f"{len(takes)} take(s) match their clips",
                )
            )
        return rows

    def check_clip_origin(self) -> List[Finding]:
        """The stack is as long as the span its clips were cut against.

        ``clip_span["*"]`` is the authoring window the exporter publishes as
        the frame the whole-timeline stack puts at ``t=0``; every clip is cut
        from that stack by offsetting against it
        (``MeshConvert._clip_zero``). The exporter cannot see the written file,
        so it publishes a MEASUREMENT of the curves it is about to serialize --
        and when that measurement is taken from the wrong thing (a bake range
        bounds what the plugin re-bakes, not what an authored curve carries)
        the number describes a file that was never written.

        Nothing else here catches that. Every clip keeps the LENGTH its take
        declares and every declared shot is present, so ``clips_vs_takes`` and
        ``cross_clips`` pass; the clips are simply cut from the wrong PLACE,
        each landing earlier than its window by the difference and playing the
        tail of the shot before it. Measured on a production assembly: a stack
        carrying frames 80-4281 published as 161-4275, so all 18 shots played
        81 frames early while their visibility gates -- written against the
        published span, and therefore correct -- switched on time.

        Compared on the whole-timeline clip, whose length IS the stack's: a
        declared take is cut to its own window and cannot show the drift.
        """
        if self.reader is None:
            return [Finding(SKIP, "clip_origin", "no readable GLB")]
        span = self._published_span(self.sidecar)
        if span is None:
            detail = self.sidecar_error or "no published clip_span"
            return [Finding(SKIP, "clip_origin", detail)]
        spans = self.reader.clip_spans(self.fps)
        if not spans:
            return [Finding(SKIP, "clip_origin", "no clips")]
        loose = self._undeclared_clips(spans, self.sidecar)
        if len(loose) != 1:
            return [
                Finding(
                    SKIP,
                    "clip_origin",
                    f"no single whole-timeline clip ({len(loose)} undeclared)",
                )
            ]
        clip = loose[0]
        carried = spans[clip][2] - round(spans[clip][0] * self.fps)
        published = span[1] - span[0]
        drift = carried - published
        if abs(drift) > 1:
            return [
                Finding(
                    FAIL,
                    "clip_origin",
                    f"{clip}: stack carries {carried:.0f}f but clips were cut "
                    f"against a {published:.0f}f span ({span[0]:g}-{span[1]:g}), "
                    f"{drift:+.0f}f out -- the published origin is not this "
                    "stack's, so every clip is cut from the wrong frame",
                )
            ]
        return [
            Finding(
                PASS,
                "clip_origin",
                f"{clip}: {carried:.0f}f matches the published span "
                f"{span[0]:g}-{span[1]:g}",
            )
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

    def check_fbx_takes(self) -> List[Finding]:
        """Declared sidecar takes all exist as AnimationStacks."""
        if self.fbx is None:
            return [Finding(SKIP, "fbx_takes", "no readable FBX")]
        takes = {t.get("name") for t in self._declared_takes(self.sidecar)}
        if not takes:
            detail = self.sidecar_error or "no sidecar takes"
            return [Finding(SKIP, "fbx_takes", detail)]
        if self._published_clip_mode(self.sidecar) == "full":
            # The takes describe the SCENE's shots in every mode; this file was
            # declared to carry them as metadata inside its one sequence.
            detail = (
                f"Full Sequence Only: the {len(takes)} declared shot(s) ride as "
                "metadata, not as takes"
            )
            return [Finding(SKIP, "fbx_takes", detail)]
        stacks = set(self.fbx.take_names())
        missing = sorted(takes - stacks)
        if missing:
            return [Finding(FAIL, "fbx_takes", f"declared but absent: {missing}")]
        return [Finding(PASS, "fbx_takes", f"{len(takes)} declared take(s) present")]

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
