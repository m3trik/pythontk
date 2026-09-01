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

    @staticmethod
    def _sidecar_beside(path: Optional[str]) -> Optional[str]:
        """The exporter's ``.{stem}.scene_data.json`` beside *path*, if any."""
        if not path:
            return None
        stem = os.path.splitext(os.path.basename(path))[0]
        candidate = os.path.join(
            os.path.dirname(os.path.abspath(path)), f".{stem}.scene_data.json"
        )
        return candidate if os.path.isfile(candidate) else None

    @staticmethod
    def _declared_takes(sidecar: Optional[dict]) -> List[dict]:
        return list(((sidecar or {}).get("data_export") or {}).get("fbx_takes") or [])


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
        fps: Frame rate for converting clip seconds to frames (default 30).
        huge: World-unit bound for the NaN/garbage scan.

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
        fps: float = 30.0,
        huge: float = 1e7,
    ):
        if not glb and not fbx:
            raise ValueError("ExportVerifier needs a glb and/or an fbx path.")
        self.glb_path = glb
        self.fbx_path = fbx
        self.baseline_glb = baseline_glb
        self.fps = fps
        self.huge = huge

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
        """``extensionsRequired`` must be a subset of ``extensionsUsed``."""
        if self.reader is None:
            return [Finding(SKIP, "glb_extensions", "no readable GLB")]
        used, required = self.reader.extensions()
        missing = sorted(set(required) - set(used))
        if missing:
            return [
                Finding(FAIL, "glb_extensions", f"required but not used: {missing}")
            ]
        return [Finding(PASS, "glb_extensions", f"used={used or 'none'}")]

    def check_glb_images(self) -> List[Finding]:
        """Texture sources resolve; basisu usage is declared and falls back."""
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
        used, _ = self.reader.extensions()
        if basisu_used and "KHR_texture_basisu" not in used:
            rows.append(
                Finding(FAIL, "glb_images", "basisu textures but extension undeclared")
            )
        if fallbackless:
            rows.append(
                Finding(
                    WARN,
                    "glb_images",
                    f"{fallbackless} basisu texture(s) carry no PNG/JPEG fallback",
                )
            )
        if not rows:
            rows.append(
                Finding(PASS, "glb_images", f"mimes={self.reader.image_mimes()}")
            )
        return rows

    def check_glb_skins(self) -> List[Finding]:
        """Referenced skins must carry inverseBindMatrices; stubs only WARN.

        Converters mint bookkeeping skins nothing references — FBX2glTF
        wrote 56–93 IBM-less stubs on measured production files — and no
        viewer reads a skin no node points at. Failing on those buries the
        real invariant: every skin a mesh node ACTUALLY references resolves
        and carries inverseBindMatrices.
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
        rows: List[Finding] = []
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
        if not bad:
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
        """Delegate to :meth:`MeshConvert.verify_glb` when an envelope rides."""
        if self.reader is None:
            return [Finding(SKIP, "glb_envelope", "no readable GLB")]
        if not (self.reader.gltf.get("extras") or {}).get("scene_sidecar"):
            return [Finding(SKIP, "glb_envelope", "no embedded envelope")]
        result = MeshConvert.verify_glb(self.glb_path)
        rows = [
            Finding(FAIL, "glb_envelope", problem)
            for problem in result.get("problems") or []
        ]
        rows.extend(
            Finding(WARN, "glb_envelope", note) for note in result.get("notes") or []
        )
        if not rows:
            rows.append(Finding(PASS, "glb_envelope", "envelope verified"))
        return rows

    # ---- sidecar gates ----------------------------------------------------

    def check_clips_vs_takes(self) -> List[Finding]:
        """Each GLB clip's length matches its declared take (±1 frame).

        Exactly one clip with no declared take is treated as the
        whole-timeline clip and compared against the takes' overall end.
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
        unmatched: List[str] = []
        for clip, (_low, _high, end_frame) in spans.items():
            take = by_name.get(clip)
            if take is None:
                unmatched.append(clip)
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
            # file reads as a failure (33 frames on the VDATS assembly).
            zero = 0.0
            for anim in self.reader.gltf.get("animations") or []:
                if anim.get("name") == clip:
                    value = (anim.get("extras") or {}).get("zero_frame")
                    if isinstance(value, (int, float)):
                        zero = float(value)
                    break
            end_frame = spans[clip][2] + zero
            timeline_end = max(int(t["end"]) for t in takes)
            if abs(end_frame - timeline_end) > 1:
                rows.append(
                    Finding(
                        FAIL,
                        "clips_vs_takes",
                        f"{clip}: full-timeline clip ends {end_frame}f, "
                        f"takes end {timeline_end}f",
                    )
                )
        elif unmatched:
            rows.append(
                Finding(WARN, "clips_vs_takes", f"undeclared clips: {unmatched}")
            )
        missing = sorted(set(by_name) - set(spans))
        if missing:
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
    parser.add_argument("--fps", type=float, default=30.0)
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
