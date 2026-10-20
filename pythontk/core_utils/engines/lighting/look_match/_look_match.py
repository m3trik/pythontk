# !/usr/bin/python
# coding=utf-8
"""The look-match facade: one session object, and its command line.

    python -m pythontk.core_utils.engines.lighting.look_match score  session.json
    python -m pythontk.core_utils.engines.lighting.look_match sheet  session.json --out sheet.png
    python -m pythontk.core_utils.engines.lighting.look_match fit    session.json --free key,fill --out fitted.json
    python -m pythontk.core_utils.engines.lighting.look_match rig    session.json --out rig.json
    python -m pythontk.core_utils.engines.lighting.look_match layers session.json
    python -m pythontk.core_utils.engines.lighting.look_match readout session.json --compare exact=exact.png
    python -m pythontk.core_utils.engines.lighting.look_match owners  session.json --region "back wall"
    python -m pythontk.core_utils.engines.lighting.look_match solve   session.json --layer fill --region floor
    python -m pythontk.core_utils.engines.lighting.look_match tiles   session.json --solo key,fill --out tiles.png

A session file names the parts, relative to itself (:class:`LookSession`)::

    {"version": 1, "layers": "layers", "target": "target.json", "mix": "mix.json",
     "transform": {"operator": "aces_filmic", "exposure": 1.0}, "size": [820, 480]}
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a declared dependency
    np = None  # type: ignore

from pythontk.core_utils.cli import CLI
from pythontk.file_utils._file_utils import FileUtils
from pythontk.img_utils._img_utils import ImgUtils
from pythontk.core_utils.engines.lighting.light_layers import LightLayers, LightMix
from pythontk.core_utils.engines.lighting.light_rig_payload import LightRigPayload
from pythontk.core_utils.engines.lighting.light_predictor import (
    LightPredictor,
    ViewerTransform,
)
from pythontk.core_utils.engines.lighting.look_match.fit import FitResult, LookFit
from pythontk.core_utils.engines.lighting.look_match.measures import (
    LookMeasures,
    LookScore,
)
from pythontk.core_utils.engines.lighting.look_match.session import LookSession
from pythontk.core_utils.engines.lighting.look_match.sheet import LookSheet
from pythontk.core_utils.engines.lighting.look_match.target import LookTarget


class LookMatch:
    """A look-match session: a layer store, a reference target and a mix.

    The agent's API and the command line are the same calls: predict a frame,
    score it against the reference, fit weights, render a comparison sheet, and
    turn a mix into a light rig payload a host builds (mayatk's ``LightRig``).

    Parameters:
        layers: A :class:`LightLayers` or its folder.
        target: A :class:`LookTarget` or its JSON file.
        mix: A :class:`LightMix`, its JSON file, or ``None`` (an empty mix).
        transform: The viewer's display transform (default: the WebXR viewer's).
        size: ``(width, height)`` to work at (default: the layers' own).

    Example:
        >>> match = LookMatch.open("session.json")
        >>> match.score().rows()                     # what is off, region by region
        >>> match.fit(["key", "fill"])               # weights fitted, the session moves on
        >>> match.sheet("sheet.png")                 # judge it by eye
        >>> rig = match.to_rig()                     # for mtk.LightRig().apply(rig)
    """

    SESSION_VERSION = LookSession.VERSION

    def __init__(
        self,
        layers: Union[LightLayers, str],
        target: Union[LookTarget, str],
        mix: Union[LightMix, str, None] = None,
        transform: Optional[ViewerTransform] = None,
        size: Optional[Tuple[int, int]] = None,
    ):
        self.layers = layers if isinstance(layers, LightLayers) else LightLayers(layers)
        self.target = (
            target if isinstance(target, LookTarget) else LookTarget.load(target)
        )
        if mix is None:
            mix = LightMix()
        self.mix = mix if isinstance(mix, LightMix) else LightMix.load(mix)
        self.predictor = LightPredictor(self.layers, transform)
        self.size = tuple(size) if size else self.layers.size
        self.measures = LookMeasures.default()

    @classmethod
    def open(cls, session: str) -> "LookMatch":
        """A session from its file (:class:`LookSession`: paths inside are
        relative to it, or ``cache:``)."""
        data = LookSession.read(session)

        def path(key):
            return LookSession.resolve(session, data, data.get(key))

        transform = (
            ViewerTransform.from_dict(data["transform"])
            if data.get("transform")
            else None
        )
        return cls(
            path("layers"), path("target"), path("mix"), transform, data.get("size")
        )

    # ------------------------------------------------------------------ frames and scores
    def predict(
        self, mix: Optional[LightMix] = None, size: Optional[Tuple[int, int]] = None
    ) -> "np.ndarray":
        """The mix's display frame (sRGB 0..1) at *size* (default: the session's)."""
        return self.predictor.frame(mix or self.mix, size or self.size)

    def reference(self, size: Optional[Tuple[int, int]] = None) -> "np.ndarray":
        return self.target.reference_display(size or self.size)

    def shown(
        self,
        mix: Optional[LightMix] = None,
        size: Optional[Tuple[int, int]] = None,
        frame: Optional[Union[str, "np.ndarray"]] = None,
    ) -> "np.ndarray":
        """What is judged: *frame* (a display image -- an exact capture -- file or
        array) at *size*, else the mix's predicted frame."""
        size = size or self.size
        if frame is None:
            return self.predict(mix, size)
        return LookTarget.display_at(frame, size)

    def score(
        self,
        mix: Optional[LightMix] = None,
        size: Optional[Tuple[int, int]] = None,
        frame: Optional[Union[str, "np.ndarray"]] = None,
    ) -> LookScore:
        """The target's measures on the mix's frame (or *frame*, see :meth:`shown`)
        against the reference."""
        size = size or self.size
        return LookScore(
            self.shown(mix, size, frame),
            self.reference(size),
            self.target,
            self.measures,
        )

    def contributions(
        self,
        region: str,
        mix: Optional[LightMix] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, float]:
        """Each layer's share of the light in *region* (our side)."""
        size = size or self.size
        mask = self.target.region(region).mask("ours", size)
        return self.predictor.contributions(mix or self.mix, mask, size)

    # ------------------------------------------------------------------ reading a look
    def _shown(self, item, size) -> "np.ndarray":
        """:meth:`shown` of a mix or of a display image (or its file)."""
        if isinstance(item, LightMix):
            return self.shown(item, size)
        return self.shown(size=size, frame=item)

    def readout(
        self,
        frames: Mapping[str, Union[LightMix, str, "np.ndarray"]],
        regions: Optional[Sequence[str]] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Dict[str, Tuple[float, float, float, float]]]:
        """Each region's mean display colour in each of *frames* and in the
        reference: ``{region: {label: (r, g, b, luma)}}`` in 0..255 codes.

        The plain numbers a person reads a frame by -- what a measure's score
        turns into a ratio -- so a candidate can be put beside the reference
        region by region before (or instead of) any fit.

        Parameters:
            frames: ``{label: mix, display image or its file}`` (an exact
                capture is judged as itself, a mix by its prediction).
            regions: Area and quad regions to read (default: all of them).
        """
        size = size or self.size
        shown = {label: self._shown(frame, size) for label, frame in frames.items()}
        shown["reference"] = self.reference(size)
        out: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {}
        for region in self.target.regions:
            if region.kind == "line" or (
                regions is not None and region.name not in regions
            ):
                continue
            row = {}
            for label, image in shown.items():
                mask = region.mask("ref" if label == "reference" else "ours", size)
                rgb = image[mask].mean(axis=0) * 255.0 if mask.any() else np.zeros(3)
                row[label] = tuple(float(v) for v in rgb) + (
                    float(ImgUtils.luminance(rgb)),
                )
            out[region.name] = row
        return out

    def owners(
        self, region: str, size: Optional[Tuple[int, int]] = None, kind: str = "light"
    ) -> List[Tuple[str, float]]:
        """The layers that light *region* most selectively, best first:
        ``(name, selectivity)`` -- the region's mean linear luminance under the
        layer alone over the whole frame's. 1.0 puts no more light there than
        anywhere; 10 puts ten times its average there.

        What to reach for when a region is off: the light that owns it moves
        it without moving the rest of the frame.
        """
        size = size or self.size
        mask = self.target.region(region).mask("ours", size)
        ranked = []
        for name in self.layers.names(kind):
            lit = ImgUtils.luminance(
                self.predictor.linear(LightMix({name: {"weight": 1.0}}), size)
            )
            mean = float(lit.mean())
            if mean > 0 and mask.any():
                ranked.append((name, float(lit[mask].mean()) / mean))
        return sorted(ranked, key=lambda item: -item[1])

    def solve(
        self,
        layer: str,
        region: str,
        mix: Optional[LightMix] = None,
        kelvin: Optional[float] = None,
        size: Optional[Tuple[int, int]] = None,
        max_weight: float = 1e6,
    ) -> float:
        """The weight of *layer* that brings *region*'s mean display luma to the
        reference's, the rest of *mix* held (bisection on the prediction).

        A layer's weight is in its bake's own units -- a narrow spot may need
        50 where a wide area light needs 0.1 -- so a light is placed by what
        it should do to a region, never by a guessed number.

        Raises:
            ValueError: the region is brighter than the reference without the
                layer, or no weight up to *max_weight* reaches it.
        """
        size = size or self.size
        mix = mix or self.mix
        reg = self.target.region(region)
        goal = float(
            ImgUtils.luminance(self.reference(size)[reg.mask("ref", size)]).mean()
        )
        mask = reg.mask("ours", size)
        entry = {"kelvin": kelvin if kelvin is not None else mix.kelvin(layer)}

        def luma(weight: float) -> float:
            trial = mix.with_entries({layer: dict(entry, weight=weight)})
            return float(ImgUtils.luminance(self.predict(trial, size)[mask]).mean())

        if luma(0.0) > goal:
            raise ValueError(
                f"{region!r} is brighter than the reference without {layer!r}."
            )
        lo, hi = 0.0, 1.0
        while luma(hi) < goal:
            lo, hi = hi, hi * 4.0
            if hi > max_weight:
                raise ValueError(f"{layer!r} cannot bring {region!r} to the reference.")
        for _ in range(30):
            mid = (lo + hi) / 2.0
            lo, hi = (mid, hi) if luma(mid) < goal else (lo, mid)
        return (lo + hi) / 2.0

    def exposed(
        self,
        layer: str,
        level: float = 0.85,
        size: Optional[Tuple[int, int]] = None,
        kelvin: Optional[float] = None,
    ) -> Tuple[float, "np.ndarray"]:
        """*layer* alone at the weight that puts its brightest (99.5th
        percentile) display luma at *level*: ``(weight, display frame)`` --
        what a light does by itself, whatever its scale.
        """
        size = size or self.size

        def frame(weight: float) -> "np.ndarray":
            alone = LightMix({layer: {"weight": weight, "kelvin": kelvin}})
            return self.predict(alone, size)

        lo, hi = 1e-6, 1e6
        for _ in range(40):  # bisection in log space
            mid = math.sqrt(lo * hi)
            if np.percentile(ImgUtils.luminance(frame(mid)), 99.5) > level:
                hi = mid
            else:
                lo = mid
        weight = math.sqrt(lo * hi)
        return weight, frame(weight)

    def fit(
        self,
        free: Sequence[str],
        mix: Optional[LightMix] = None,
        size: Optional[Tuple[int, int]] = None,
        max_iter: int = 30,
        kelvin: Sequence[str] = (),
        **options,
    ) -> FitResult:
        """Fit *free* layers' weights and *kelvin* layers' colour temperatures (see
        :class:`LookFit` for *options*); the session moves on to the result."""
        fit = LookFit(
            self.predictor, self.target, size or self.size, self.measures, **options
        )
        result = fit.run(mix or self.mix, free, max_iter=max_iter, kelvin=kelvin)
        self.mix = result.mix
        return result

    def sheet(
        self,
        path: str,
        mix: Optional[LightMix] = None,
        size: Optional[Tuple[int, int]] = None,
        regions: Optional[Sequence[str]] = None,
        contributions: bool = True,
        title: Optional[str] = None,
        width: int = 1640,
        frame: Optional[Union[str, "np.ndarray"]] = None,
    ) -> str:
        """Write a comparison sheet (:class:`LookSheet`) to *path*: the mix's
        frame, or *frame* (see :meth:`shown`; the light shares stay the mix's)."""
        size = size or self.size
        mix = mix or self.mix
        ours, ref = self.shown(mix, size, frame), self.reference(size)
        score = LookScore(ours, ref, self.target, self.measures)
        chosen = (
            regions if regions is not None else [r.name for r in self.target.regions]
        )
        shares = None
        if contributions:
            masks = {
                name: self.target.region(name).mask("ours", size) for name in chosen
            }
            shares = self.predictor.region_contributions(mix, masks, size)
        return LookSheet(width=width).save(
            path,
            ours,
            ref,
            self.target,
            score=score,
            regions=chosen,
            contributions=shares,
            title=title or mix.notes or "look match",
        )

    # ------------------------------------------------------------------ rig
    def to_rig(
        self,
        mix: Optional[LightMix] = None,
        gain: float = math.pi,
        color_space: str = "ACEScg",
        min_weight: float = 1e-4,
        group: str = "LIGHT_RIG",
    ) -> Dict[str, Any]:
        """The mix as a light rig payload (version 1, look-match spelling).

        Each active light layer contributes its spec's lights at
        ``intensity * gain * weight``, coloured by its tint expressed in the
        scene's rendering space. Emission layers become material overrides
        (their manifest ``target``: ``{"node", "attr", "scale"}``); viewer-only
        panels cannot ship and are listed under ``source.viewer_only``.

        Parameters:
            gain: Light units per layer unit -- what a bake multiplies a light
                by for the viewer to show it at weight 1 (pi for the WebXR
                viewer, which draws baked light pi below the renderer).
            color_space: The scene's rendering space, for the light colours.
            min_weight: Lights below it are left out.
            group: The scene group the rig's lights live under.
        """
        mix = mix or self.mix
        rig: Dict[str, Any] = {
            "version": 1,
            "group": group,
            "lights": [],
            # The material state the layers were baked under (an office's
            # emission off...): applied to a scene that differs, the rig sets it.
            "materials": copy.deepcopy(self.layers.stamp.get("materials") or {}),
            "bake_gain": gain,
        }
        existing = self.layers.stamp.get("existing")
        if existing:
            rig["existing"] = existing
        viewer_only, unbuilt = [], []

        def build_material(rig, name, info, weight) -> bool:
            target = info.get("target")
            if not target:
                return False
            rig["materials"].setdefault(target["node"], {})[target["attr"]] = (
                weight * float(target.get("scale", 1.0))
            )
            return True

        def build_lights(rig, name, info, weight) -> bool:
            if weight < min_weight:
                return True  # off: nothing to build, and nothing missing
            spec = info.get("spec") or {}
            if not spec.get("lights"):
                return False
            tint = np.asarray(ImgUtils.kelvin_tint(mix.kelvin(name)), np.float32)
            color = np.asarray(
                ImgUtils.convert_scene_linear(
                    tint.reshape(1, 1, 3), "scene-linear Rec.709-sRGB", color_space
                )
            ).reshape(3)
            peak = float(color.max())
            for light in spec["lights"]:
                out = dict(light)
                out["intensity"] = (
                    float(light.get("intensity", 1.0)) * gain * weight * peak
                )
                out["color"] = [round(float(c / peak), 5) for c in color]
                if mix.specular(name) != 1.0:  # LightRig sets it as aiSpecular
                    out["specular"] = round(mix.specular(name), 5)
                rig["lights"].append(out)
            return True

        builders = {"material": build_material, "lights": build_lights}
        for name in mix.active():
            weight = mix.weight(name)
            info = self.layers.info(name)
            kind = self.layers.kind(name)
            if not info.get("shippable", kind.shippable):
                viewer_only.append(name)
                continue
            build = builders.get(kind.rig)
            if build is None or not build(rig, name, info, weight):
                unbuilt.append(name)
        rig["source"] = {
            "tool": "look-match",
            "mix": mix.to_dict(),
            "viewer_only": viewer_only,
            "unbuilt": unbuilt,
        }
        checked = LightRigPayload.validate(rig)
        if checked.errors:  # a role's lights spelled outside the contract
            raise ValueError(
                "The rig does not fit LightRigPayload: " + "; ".join(checked.errors)
            )
        return rig

    # ------------------------------------------------------------------ command line
    _FRAME_HELP = (
        "judge this display image (an exact capture) instead of the prediction"
    )
    _COMPARE_HELP = (
        "LABEL=PATH: a mix (.json) or a display image to put beside the reference"
        " (repeat)"
    )

    @classmethod
    def cli(cls, argv: Optional[Sequence[str]] = None) -> int:
        """The command line (``python -m pythontk.core_utils.engines.lighting.look_match``);
        returns the exit code."""
        command = CLI.Command
        frame = ("--frame", {"help": cls._FRAME_HELP})
        compare = (
            "--compare",
            {"action": "append", "default": [], "help": cls._COMPARE_HELP},
        )
        out = ("--out", {"required": True})
        width = ("--width", {"type": int, "default": 1640})
        commands = {
            "score": command(
                "print the scores, region by region",
                cls._cli_score,
                [
                    ("--json", {"action": "store_true", "help": "print JSON rows"}),
                    frame,
                ],
            ),
            "sheet": command(
                "write a comparison sheet",
                cls._cli_sheet,
                [
                    frame,
                    out,
                    ("--regions", {"help": "comma-separated regions to show close up"}),
                    width,
                ],
            ),
            "fit": command(
                "fit layer weights to the reference",
                cls._cli_fit,
                [
                    (
                        "--free",
                        {"required": True, "help": "comma-separated layers to fit"},
                    ),
                    (
                        "--kelvin",
                        {
                            "help": "comma-separated layers whose colour temperature to fit"
                        },
                    ),
                    (
                        "--out",
                        {"required": True, "help": "where to write the fitted mix"},
                    ),
                    ("--tolerance", {"type": float, "default": 0.1}),
                    ("--regularize", {"type": float, "default": 0.3}),
                    ("--max-iter", {"type": int, "default": 30}),
                ],
            ),
            "rig": command(
                "write the mix as a light rig payload",
                cls._cli_rig,
                [
                    out,
                    ("--gain", {"type": float, "default": math.pi}),
                    ("--group", {"default": "LIGHT_RIG"}),
                    ("--color-space", {"default": "ACEScg"}),
                ],
            ),
            "layers": command("list the layer store", cls._cli_layers),
            "readout": command(
                "each region's display colour beside the reference's",
                cls._cli_readout,
                [
                    compare,
                    ("--regions", {"help": "comma-separated regions (default: all)"}),
                ],
            ),
            "owners": command(
                "the lights that light a region most selectively",
                cls._cli_owners,
                [
                    ("--region", {"required": True}),
                    ("--top", {"type": int, "default": 10}),
                ],
            ),
            "solve": command(
                "the weight of a layer that brings a region to the reference",
                cls._cli_solve,
                [
                    ("--layer", {"required": True}),
                    ("--region", {"required": True}),
                    ("--kelvin", {"type": float}),
                ],
            ),
            "tiles": command(
                "candidates and solo layers in a labelled grid",
                cls._cli_tiles,
                [
                    out,
                    compare,
                    (
                        "--solo",
                        {
                            "help": "comma-separated layers, each alone, exposed to be seen"
                        },
                    ),
                    ("--columns", {"type": int, "default": 2}),
                    width,
                ],
            ),
        }
        return CLI.dispatch(
            "Look match: score, fit and compare a lighting mix against a reference.",
            commands,
            argv,
            common=[
                ("session", {"help": "session JSON (layers, target, mix)"}),
                ("--mix", {"help": "a mix JSON to use instead of the session's"}),
                ("--size", {"help": "WIDTHxHEIGHT to work at"}),
            ],
        )

    @classmethod
    def _cli_session(cls, args) -> "LookMatch":
        """The session the command names, with ``--mix`` and ``--size`` applied."""
        match = cls.open(args.session)
        if args.mix:
            match.mix = LightMix.load(args.mix)
        if args.size:
            match.size = tuple(int(v) for v in args.size.lower().split("x"))
        return match

    @staticmethod
    def _cli_names(text: Optional[str]) -> List[str]:
        return [n.strip() for n in (text or "").split(",") if n.strip()]

    @staticmethod
    def _cli_frames(match: "LookMatch", args) -> Dict[str, Any]:
        """The session's mix and each ``--compare`` (a mix or a display image), by label."""
        frames: Dict[str, Any] = {"mix": match.mix}
        for item in args.compare:
            label, _, path = item.partition("=")
            frames[label] = (
                LightMix.load(path) if path.lower().endswith(".json") else path
            )
        return frames

    @classmethod
    def _cli_score(cls, args) -> None:
        rows = cls._cli_session(args).score(frame=args.frame).rows()
        if args.json:
            print(json.dumps(rows, indent=1))
            return
        print(f"{'region':24s} {'measure':12s} {'ours':>9s} {'ref':>9s} {'score':>9s}")
        for row in rows:
            shown = (
                f"x{math.exp(row['score']):.2f}"
                if row["unit"] == "log"
                else f"{row['score']:+.2f}"
            )
            print(
                f"{row['region']:24s} {row['measure']:12s} {row['ours']:9.4f} {row['ref']:9.4f} {shown:>9s}"
            )

    @classmethod
    def _cli_sheet(cls, args) -> None:
        match = cls._cli_session(args)
        regions = cls._cli_names(args.regions) or None
        print(
            match.sheet(args.out, regions=regions, width=args.width, frame=args.frame)
        )

    @classmethod
    def _cli_fit(cls, args) -> None:
        result = cls._cli_session(args).fit(
            cls._cli_names(args.free),
            max_iter=args.max_iter,
            kelvin=cls._cli_names(args.kelvin),
            tolerance=args.tolerance,
            regularize=args.regularize,
        )
        result.mix.save(args.out)
        print(
            f"cost {result.cost_before:.4f} -> {result.cost_after:.4f} in {result.iterations} steps, "
            f"{result.evaluations} frames, {result.seconds:.1f}s"
        )
        for name, (before, after) in result.changes.items():
            print(f"  {name}: {before:.4g} -> {after:.4g}")
        for name, (before, after) in result.kelvins.items():
            print(f"  {name}: {before or 0:.0f} K -> {after:.0f} K")

    @classmethod
    def _cli_rig(cls, args) -> None:
        rig = cls._cli_session(args).to_rig(
            gain=args.gain, group=args.group, color_space=args.color_space
        )
        FileUtils.write_json(args.out, rig, indent=1)
        print(
            f"{args.out}: {len(rig['lights'])} lights; viewer-only: {rig['source']['viewer_only'] or 'none'}"
        )

    @classmethod
    def _cli_layers(cls, args) -> None:
        match = cls._cli_session(args)
        for name, info in match.layers.iter_layers():
            ship = "" if info.get("shippable", True) else "  viewer-only"
            lights = len((info.get("spec") or {}).get("lights") or [])
            print(f"{name:28s} {info.get('kind', 'light'):9s} lights {lights}{ship}")
        for problem in match.layers.validate():
            print("PROBLEM:", problem)

    @classmethod
    def _cli_readout(cls, args) -> None:
        match = cls._cli_session(args)
        frames = cls._cli_frames(match, args)
        read = match.readout(frames, cls._cli_names(args.regions) or None)
        labels = list(frames) + ["reference"]
        print(f"{'region':24s}" + "".join(f"{lb[:19]:>21s}" for lb in labels))
        for region, row in read.items():
            cells = "".join(
                "{:>21s}".format("%d %d %d  %d" % tuple(round(v) for v in row[lb]))
                for lb in labels
            )
            print(f"{region[:24]:24s}{cells}")

    @classmethod
    def _cli_owners(cls, args) -> None:
        for name, share in cls._cli_session(args).owners(args.region)[: args.top]:
            print(f"{name:28s} x{share:.2f}")

    @classmethod
    def _cli_solve(cls, args) -> None:
        weight = cls._cli_session(args).solve(
            args.layer, args.region, kelvin=args.kelvin
        )
        print(
            f"{args.layer} weight {weight:.4g} brings {args.region!r} to the reference"
        )

    @classmethod
    def _cli_tiles(cls, args) -> None:
        match = cls._cli_session(args)
        shown = {
            label: match._shown(frame, match.size)
            for label, frame in cls._cli_frames(match, args).items()
        }
        for name in cls._cli_names(args.solo):
            weight, frame = match.exposed(name)
            shown[f"{name} alone (x{weight:.3g})"] = frame
        shown["reference"] = match.reference()
        LookSheet(width=args.width).tiles(shown, args.columns).save(args.out)
        print(args.out)
