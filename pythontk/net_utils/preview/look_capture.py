# !/usr/bin/python
# coding=utf-8
"""Look capture: the real WebXR viewer, headless, for lighting work.

A :class:`LookCapture` loads a scene's template GLB once in a headless browser
and then, in seconds per frame, swaps any bake in (a mix's summed lightmaps and
reflection probe, ``viewer.lighting.relight``) and shoots what the screen shows
(``lit``) or linear light (``linear``). It is the ground truth the look-match
engine's prediction is held to, and how a newly baked light becomes a layer.

Needs Playwright (``pip install playwright``) and a Chromium-family browser;
Edge, already on every Windows machine, is the default channel. Rendering is
software (SwiftShader) by default: identical on every machine, which is what a
measurement wants.

    python -m pythontk.net_utils.preview.look_capture exact   session.json --out frame.png
    python -m pythontk.net_utils.preview.look_capture capture session.json --layer key [--layer fill]

Loading a large template takes minutes in software; ``serve`` loads it once and
answers ``exact`` / ``capture`` sent with ``--service http://127.0.0.1:8770/``
in seconds:

    python -m pythontk.net_utils.preview.look_capture serve session.json

A session (see ``LookMatch``) names ``template`` (the GLB) and, under
``capture``: ``camera`` (``{pos, target, up, vfov}`` in model space), ``size``,
``black`` (patterns of material or object names whose reflections a layer
capture leaves out; ``!pattern`` keeps a match) and ``emissive`` (``{pattern, intensity}``).
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a declared dependency
    np = None  # type: ignore

from pythontk.core_utils.cli import CLI
from pythontk.core_utils.engines.lighting.light_layers import (
    LightLayerInfo,
    LightLayers,
    LightMix,
)
from pythontk.core_utils.engines.lighting.light_predictor import LightPredictor
from pythontk.file_utils.mesh_convert.glb.reader import GlbReader


class LookCapture:
    """A headless viewer session that relights a template and shoots frames.

    Parameters:
        template: A lightmapped GLB (its bake layout is what bakes swap into).
        camera: ``{pos, target, up, vfov}`` in model space; ``None`` keeps the
            page's own framing.
        size: ``(width, height)`` of the browser viewport.
        channel: The Playwright browser channel (``"msedge"``, ``"chrome"``).
        software: Render with SwiftShader (deterministic across machines).
        timeout: Seconds to wait for the template to load.

    Example:
        >>> with LookCapture("scene.glb", camera) as capture:
        ...     frame = capture.exact(predictor, mix)          # what WebXR shows
        ...     capture.capture_layer(store, "key")            # a baked light, as a layer
    """

    FEATURE = "look_capture"

    def __init__(
        self,
        template: str,
        camera: Optional[Mapping[str, Any]] = None,
        size: Tuple[int, int] = (1638, 960),
        channel: str = "msedge",
        software: bool = True,
        timeout: float = 900.0,
    ):
        self.template = str(template)
        self.camera = dict(camera) if camera else None
        self.size = (int(size[0]), int(size[1]))
        self.channel = channel
        self.software = software
        self.timeout = float(timeout)
        self._manifest: Optional[Dict[str, Any]] = None
        #: The template's own probe as a payload carries it (:meth:`_file_probe`).
        self._probe_payload: Optional[Dict[str, Any]] = None
        self._server = self._playwright = self._browser = self._page = None
        self._module_url = ""

    # ------------------------------------------------------------------ plumbing
    @staticmethod
    def available() -> bool:
        """Whether Playwright is installed (the browser is checked on :meth:`start`)."""
        return importlib.util.find_spec("playwright") is not None

    @property
    def manifest(self) -> Dict[str, Any]:
        """The template's lightmap manifest (``extras.lightmap_web``): materials, probe.

        Read where the viewer and the highlight pass read it
        (``MeshConvert._lightmap_web_manifest``): the first scene's extras --
        a native export's, as a JSON string -- then the root's.
        """
        if self._manifest is None:
            from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

            gltf = GlbReader.load(self.template).gltf
            self._manifest = MeshConvert._lightmap_web_manifest(gltf) or {}
        return self._manifest

    def start(self) -> "LookCapture":
        """Open the browser and load the template; returns self."""
        from pythontk.net_utils.preview.server import PreviewServer
        from playwright.sync_api import sync_playwright

        self._server = PreviewServer(viewer=True, title="look capture", port=0).start()
        self._server.add_script(self.FEATURE)
        self._module_url = next(
            u for u in self._server.manifest()["scripts"] if "look_capture" in u
        )
        self._playwright = sync_playwright().start()
        args = (
            ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"]
            if self.software
            else []
        )
        self._browser = self._playwright.chromium.launch(
            channel=self.channel, headless=True, args=args
        )
        self._page = self._browser.new_page(
            viewport={"width": self.size[0], "height": self.size[1]}
        )
        self._page.goto(
            self._server.url, wait_until="domcontentloaded", timeout=120_000
        )
        self._server.publish(self.template)
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if self._call("status").get("ready"):
                return self
            time.sleep(0.5)
        raise TimeoutError(
            f"The viewer did not settle on {self.template} within {self.timeout:.0f}s."
        )

    def stop(self) -> None:
        """Close the browser and the server (safe to call twice)."""
        for closer in (
            lambda: self._browser and self._browser.close(),
            lambda: self._playwright and self._playwright.stop(),
            lambda: self._server and self._server.stop(),
        ):
            try:
                closer()
            except Exception:
                pass
        self._server = self._playwright = self._browser = self._page = None

    def __enter__(self) -> "LookCapture":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def idle(self, asleep: bool = True) -> None:
        """Stop the page's frame loop (or restart it). An idle viewer still
        draws every frame, and in software GL that is several cores (measured:
        about seven) a bake running beside it waits on. Not a page freeze: a
        frozen headless page's GPU process kept drawing (measured)."""
        self._call("setRendering", not asleep)

    def _call(self, name: str, *args) -> Any:
        """Call the page feature's export *name* (it is imported by its served URL)."""
        script = (
            "async ([url, name, args]) => {"
            " const m = await import(new URL(url, document.baseURI).href);"
            " return m[name](...args); }"
        )
        return self._page.evaluate(script, [self._module_url, name, list(args)])

    # ------------------------------------------------------------------ payloads
    @staticmethod
    def _half_b64(rgb: "np.ndarray") -> str:
        rgba = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), rgb.dtype)], axis=2)
        return base64.b64encode(
            np.ascontiguousarray(rgba, np.float16).tobytes()
        ).decode("ascii")

    @classmethod
    def payload(
        cls,
        bakes: Mapping[str, "np.ndarray"],
        manifest: Mapping[str, Any],
        probe: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """What a relight sends: each material's image, the images, the probe.

        Parameters:
            bakes: ``{file name: (height, width, 3) linear}`` -- atlases matched
                to the template's images by name stem, the probe by ``_Probe``.
            manifest: The template's lightmap manifest.
            probe: ``{"position", "box"}`` overriding the manifest's placement
                of the probe (a panorama taken elsewhere; ``"box": None``
                reads it as distant).
        """
        stems = {os.path.splitext(name)[0]: image for name, image in bakes.items()}
        maps = {
            material: os.path.splitext(entry["map"])[0]
            for material, entry in (manifest.get("materials") or {}).items()
            if os.path.splitext(entry.get("map", ""))[0] in stems
        }
        images = {}
        for stem in sorted(set(maps.values())):
            image = np.asarray(stems[stem], np.float32)
            images[stem] = {
                "width": image.shape[1],
                "height": image.shape[0],
                "b64": cls._half_b64(image),
            }
        placement = dict(probe or {})
        probe = None
        probe_stem = next((s for s in stems if s.endswith("_Probe")), None)
        if probe_stem is not None:
            image = np.asarray(stems[probe_stem], np.float32)
            entry = dict(manifest.get("probe") or {}, **placement)
            probe = {
                "width": image.shape[1],
                "height": image.shape[0],
                "b64": cls._half_b64(image),
                "position": entry.get("position", [0.0, 0.0, 0.0]),
                "box": entry.get("box"),
            }
        return {"maps": maps, "images": images, "probe": probe}

    @staticmethod
    def decode(shot: str) -> "np.ndarray":
        """A shot's return value as an array: ``png`` -> uint8 RGB, ``f16`` -> float32 linear RGB."""
        if shot.startswith("png:"):
            from PIL import Image

            data = base64.b64decode(shot.split(",", 1)[1])
            with Image.open(io.BytesIO(data)) as img:
                return np.asarray(img.convert("RGB"), np.uint8)
        if shot.startswith("f16:"):
            _, size, b64 = shot.split(":", 2)
            width, height = (int(v) for v in size.split("x"))
            return (
                np.frombuffer(base64.b64decode(b64), np.float16)
                .reshape(height, width, 3)
                .astype(np.float32)
            )
        raise ValueError(f"Unknown shot format: {shot[:12]!r}")

    # ------------------------------------------------------------------ frames
    def relight(
        self,
        bakes: Mapping[str, "np.ndarray"],
        emissive: Optional[Mapping[str, Any]] = None,
        probe: Optional[Mapping[str, Any]] = None,
        lights: Optional[Any] = None,
    ) -> int:
        """Swap *bakes* in (see :meth:`payload`, and its *probe*); returns how many
        materials took a map.

        Parameters:
            lights: The bake's lights in metres
                (:meth:`MeshConvert.lights_in_metres`), drawn into the probe's
                highlight copies as the deliverable draws them
                (:meth:`MeshConvert.highlight_probe_layers`) -- added to the
                swap's probe, or to the template's own when the swap has none;
                ``None`` or empty: no highlights.
        """
        payload = self.payload(bakes, self.manifest, probe)
        payload["emissive"] = dict(emissive) if emissive else None
        payload["highlights"] = self._highlights(payload, lights) if lights else None
        return int(self._call("setLighting", payload))

    def _highlights(
        self, payload: Dict[str, Any], lights: Sequence[Mapping[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        """The payload's highlight layers for *lights*, its probe filled in when
        the swap carried none (the copies are a probe plus a layer)."""
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        if payload.get("probe") is None:
            payload["probe"] = self._file_probe()
            if payload["probe"] is None:
                return None
        probe = payload["probe"]
        layers = MeshConvert.highlight_probe_layers(
            self.template,
            [dict(light) for light in lights],
            probe={"position": probe["position"], "box": probe.get("box")},
            width=int(probe["width"]),
        )
        return [
            {
                "materials": layer["materials"],
                "receivers": layer["receivers"],
                "width": int(layer["image"].shape[1]),
                "height": int(layer["image"].shape[0]),
                "b64": self._half_b64(layer["image"]),
            }
            for layer in layers
        ] or None

    def _file_probe(self) -> Optional[Dict[str, Any]]:
        """The template's own reflection probe as a payload's (decoded once), or ``None``."""
        if self._probe_payload is None:
            entry = self.manifest.get("probe") or {}
            view = entry.get("bufferView")
            if not isinstance(view, int):
                return None
            import cv2

            from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit

            edit = GlbEdit.read(self.template)
            spec = (edit.gltf.get("bufferViews") or [])[view]
            start = int(spec.get("byteOffset", 0))
            raw = bytes(edit.bin_data[start : start + int(spec["byteLength"])])
            image = cv2.imdecode(
                np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH
            )
            if image is None:
                return None
            rgb = np.ascontiguousarray(image[..., 2::-1], np.float32)
            self._probe_payload = {
                "width": rgb.shape[1],
                "height": rgb.shape[0],
                "b64": self._half_b64(rgb),
                "position": entry.get("position", [0.0, 0.0, 0.0]),
                "box": entry.get("box"),
            }
        return self._probe_payload

    def shoot(
        self,
        mode: str = "lit",
        camera: Optional[Mapping[str, Any]] = None,
        black: Sequence[str] = (),
        scale: float = 1.0,
        uncap: Sequence[str] = (),
    ) -> "np.ndarray":
        """One frame: ``lit`` (uint8 RGB, as the screen shows it) or ``linear`` (float32 light).

        Parameters:
            camera: Overrides the session's camera for this frame.
            black: Patterns of material or object names (an ancestor's counts)
                whose reflections the frame leaves out; ``!pattern`` keeps a match.
            scale: A linear frame's size against the viewport.
            uncap: Patterns (as *black*) whose baked materials reflect their
                environment whole, without the lightmap's per-texel
                normalization -- a diagnostic: what the environment alone gives.
        """
        options = {
            "mode": mode,
            "camera": dict(camera or self.camera or {}) or None,
            "black": list(black),
            "uncap": list(uncap),
            "scale": scale,
        }
        return self.decode(self._call("shot", options))

    def shoot_ids(
        self, camera: Optional[Mapping[str, Any]] = None, scale: float = 1.0
    ) -> Tuple["np.ndarray", List[Dict[str, Any]]]:
        """Which object every pixel shows: ``(uint8 RGB image, legend)``.

        Each mesh in a flat colour of its own (unlit, unaliased); the legend
        names each colour's object path and material -- what a target's
        ``objects`` regions read (``LookTarget.objects``).
        """
        options = {"camera": dict(camera or self.camera or {}) or None, "scale": scale}
        reply = self._call("ids", options)
        return self.decode("png:" + reply["png"]), list(reply["legend"])

    def exact(
        self,
        predictor: LightPredictor,
        mix: LightMix,
        mode: str = "lit",
        camera: Optional[Mapping[str, Any]] = None,
        black: Sequence[str] = (),
        emissive: Optional[Mapping[str, Any]] = None,
        swap: Optional[Mapping[str, str]] = None,
        probe: Optional[Mapping[str, Any]] = None,
        uncap: Sequence[str] = (),
        lights: Optional[Any] = None,
    ) -> "np.ndarray":
        """*mix* as the viewer shows it: its bake (:meth:`LightPredictor.bakes`) swapped in and shot.

        Parameters:
            swap: ``{file name: path}`` -- bake files to use instead of (or
                beside) the mix's: a what-if (a re-baked probe, a panorama
                taken elsewhere) without capturing a layer.
            probe: The probe's placement instead of the template's (see
                :meth:`payload`).
            uncap: See :meth:`shoot`.
            lights: See :meth:`relight`; default: the mix's own lights
                (:meth:`LightPredictor.lights`), so the frame is what ships.
        """
        bakes = predictor.bakes(mix)
        for file_name, path in (swap or {}).items():
            bakes[file_name] = LightPredictor.read_bake(path)
        if lights is None:  # what ships: the mix's own lights, as highlights
            from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

            lights = MeshConvert.lights_in_metres(predictor.lights(mix)) or []
        self.relight(bakes, emissive, probe, lights)
        return self.shoot(mode, camera, black, uncap=uncap)

    @staticmethod
    def _write_frame(frame: "np.ndarray", mode: str, out: str) -> str:
        """*frame* to *out*: ``.npy`` for linear light, an image for lit."""
        if mode == "linear":
            np.save(out, frame)
        else:
            from PIL import Image

            Image.fromarray(frame).save(out)
        return out

    def file_frame(
        self, mode: str = "lit", camera: Optional[Mapping[str, Any]] = None
    ) -> "np.ndarray":
        """The template as it ships -- its own bake and lights, nothing swapped
        in: the baseline every candidate mix is held to."""
        self._call("restoreLighting")
        deadline = time.time() + self.timeout
        while not self._call("status").get("ready"):  # the file's probe reloads
            if time.time() > deadline:
                raise TimeoutError("The file's own lighting did not settle.")
            time.sleep(0.25)
        return self.shoot(mode, camera)

    def capture_layer(
        self,
        layers: LightLayers,
        name: str,
        black: Sequence[str] = (),
        replace: bool = True,
    ) -> "np.ndarray":
        """Re-capture layer *name* from its bake: white, unit weight, linear, full
        size. The bake stays where the store keeps it (``add`` without ``bakes``)."""
        predictor = LightPredictor(layers)
        frame = self.exact(
            predictor, LightMix({name: {"weight": 1.0}}), "linear", black=black
        )
        info = layers.info(name)
        extra = {
            k: v
            for k, v in info.items()
            if k not in ("kind", "spec", "shippable", "added")
        }
        layers.add(
            name,
            frame,
            kind=info.get("kind", "light"),
            spec=info.get("spec"),
            shippable=info.get("shippable"),
            replace=replace,
            **extra,
        )
        return frame

    #: A bake folder's description of itself (mayatk's ``LightLayerBaker`` writes it).
    LAYER_FILE = LightLayerInfo.FILE

    @classmethod
    def bake_folders(cls, folder: str) -> List[str]:
        """The layer bake folders in *folder*: itself when it is one, else those in it."""
        if os.path.isfile(os.path.join(folder, cls.LAYER_FILE)):
            return [folder]
        return [
            os.path.join(folder, name)
            for name in sorted(os.listdir(folder))
            if os.path.isfile(os.path.join(folder, name, cls.LAYER_FILE))
        ]

    def capture_bake(
        self,
        layers: LightLayers,
        folder: str,
        black: Sequence[str] = (),
        replace: bool = True,
    ) -> "np.ndarray":
        """A new (or re-baked) layer from a bake folder: its maps and probe swapped
        in, its lights drawn in as highlights (``layer.json``'s ``lights``, the
        lightmap record's form), shot linear, stored with its bake and what
        ``layer.json`` says of it (name, kind, spec, gain, settings, lights, the
        ``scene`` save it was baked from, and an emission layer's ``target``:
        what ``LookMatch.to_rig`` scales)."""
        with open(os.path.join(folder, self.LAYER_FILE), encoding="utf-8") as f:
            info = json.load(f)
        checked = LightLayerInfo.validate(info)
        if checked.errors:
            raise ValueError(
                f"{folder}: {self.LAYER_FILE} does not fit LightLayerInfo: "
                + "; ".join(checked.errors)
            )
        name = info.get("name") or os.path.basename(os.path.normpath(folder))
        paths = {
            file_name: os.path.join(folder, file_name)
            for file_name in info.get("files") or []
        }
        # Its lights' highlights, as the deliverable draws them into the
        # probe's copies: linear in the layer's weight (a light's scale is a
        # ratio of its own light), so the layer holds them too.
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        lights = MeshConvert.lights_in_metres(info.get("lights"))
        self.relight(
            {n: LightPredictor.read_bake(p) for n, p in paths.items()}, lights=lights
        )
        frame = self.shoot("linear", black=black)
        extra = {
            k: info[k]
            for k in ("gain", "settings", "lights", "target", "scene")
            if k in info
        }
        layers.add(
            name,
            frame,
            kind=info.get("kind", "light"),
            spec=info.get("spec"),
            replace=replace,
            bakes=list(paths.values()),
            **extra,
        )
        return frame

    # ------------------------------------------------------------------ sessions and service
    @staticmethod
    def open_session(session: str):
        """``(match, settings, template, size)`` for a session file (``LookSession``).

        Raises:
            ValueError: the session names no template.
        """
        from pythontk.core_utils.engines.lighting.look_match._look_match import (
            LookMatch,
        )
        from pythontk.core_utils.engines.lighting.look_match.session import (
            LookSession,
        )

        data = LookSession.read(session)
        if not data.get("template"):
            raise ValueError(f"{session} names no template to capture from.")
        settings = data.get("capture") or {}
        template = LookSession.resolve(session, data, data["template"])
        match = LookMatch.open(session)
        size = tuple(settings.get("size") or match.layers.size or (1638, 960))
        return match, settings, template, size

    # ------------------------------------------------------------------ requests
    def _ops(self) -> Dict[str, Any]:
        """The request ops by name: each ``(match, settings, **arguments) -> reply``."""
        return {
            "status": self._op_status,
            "exact": self._op_exact,
            "file": self._op_file,
            "capture": self._op_capture,
            "ids": self._op_ids,
        }

    def _op_status(self, match, settings) -> Dict[str, Any]:
        return {"template": self.template, "layers": len(match.layers.names())}

    def _op_exact(
        self,
        match,
        settings,
        out: str,
        mix: Optional[Mapping[str, Any]] = None,
        mode: str = "lit",
        camera=None,
        as_layer: bool = False,
        swap: Optional[Mapping[str, str]] = None,
        probe: Optional[Mapping[str, Any]] = None,
        uncap: Sequence[str] = (),
        lights=None,
    ) -> Dict[str, Any]:
        frame = self.exact(
            match.predictor,
            LightMix.from_dict(mix) if mix else match.mix,
            mode,
            camera,
            settings.get("black") or () if as_layer else (),
            settings.get("emissive"),
            swap,
            probe,
            uncap or (),
            lights,
        )
        return {"out": self._write_frame(frame, mode, out)}

    def _op_file(
        self, match, settings, out: str, mode: str = "lit", camera=None
    ) -> Dict[str, Any]:
        return {"out": self._write_frame(self.file_frame(mode, camera), mode, out)}

    def _op_capture(
        self,
        match,
        settings,
        layers: Optional[Sequence[str]] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        black = settings.get("black") or ()
        seconds = {}
        if source:  # new layers from bake folders
            for folder in self.bake_folders(source):
                t0 = time.time()
                self.capture_bake(match.layers, folder, black)
                seconds[os.path.basename(folder)] = round(time.time() - t0, 1)
            return {"captured": seconds}
        for name in layers or [
            n for n in match.layers.names() if match.layers.bake_files(n)
        ]:
            t0 = time.time()
            self.capture_layer(match.layers, name, black)
            seconds[name] = round(time.time() - t0, 1)
        return {"captured": seconds}

    def _op_ids(
        self, match, settings, out: str, camera=None, target: Optional[str] = None
    ) -> Dict[str, Any]:
        from PIL import Image

        image, legend = self.shoot_ids(camera)
        Image.fromarray(image).save(out)
        if target:  # attach it: the target's objects regions read it
            from pythontk.core_utils.engines.lighting.look_match.target import (
                LookTarget,
            )

            loaded = LookTarget.load(target)
            rel = os.path.relpath(out, os.path.dirname(os.path.abspath(target)))
            loaded.ids = {"image": rel.replace(os.sep, "/"), "legend": legend}
            loaded.save(target)
        return {"out": out, "objects": len(legend)}

    def handle(
        self, match, settings: Mapping[str, Any], request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """One request against a loaded session, in process: ``{"op": name, **arguments}``.

        ``status``; ``exact`` (``out``; ``mix`` dict or None, ``mode`` lit or
        linear, ``camera``, ``as_layer`` -- leave out what a layer capture does,
        so the frame can be held to the sum of the mix's layers -- ``swap``
        ``{file: path}``, ``probe`` ``{position, box}``, ``uncap``, ``lights``:
        the frame is written to *out*, PNG for lit, ``.npy`` for linear);
        ``file`` (``out``, ``mode``, ``camera``: the template's own lighting,
        :meth:`file_frame`); ``capture`` (``layers``, default every layer with a
        bake, or ``source``, a bake folder: new layers, :meth:`capture_bake`);
        ``ids`` (``out``, ``camera``, ``target`` to attach the image to). The
        same ops :meth:`serve` answers.
        """
        arguments = dict(request)
        op = arguments.pop("op", None)
        ops = self._ops()
        if op not in ops:
            raise ValueError(f"Unknown request {op!r}; one of {', '.join(ops)}.")
        return ops[op](match, settings, **arguments)

    def serve(self, session: str, host: str = "127.0.0.1", port: int = 8770) -> None:
        """Answer requests over local HTTP until stopped, one at a time, on this thread.

        Speaks ``net_utils.rpc``'s wire contract (an ``RpcClient``, :meth:`client`):
        every :meth:`handle` op, plus ``reload`` (re-read the session after a
        capture changed its layers) and ``stop`` (end the service and close its
        browser -- never a process kill). It renders and writes frames; it never
        runs code it is sent. Bound to *host* (loopback by default): requests
        name files on this machine. On this thread because the browser answers
        only the thread that started it (``RpcPlugin.serve``) -- so it runs as
        its own process (``look_capture serve``), never on a worker thread of a Qt
        application, whose marshaller would move the ops to the main thread. The
        page's frame loop sleeps between requests (:meth:`idle`).
        """
        from pythontk.net_utils.rpc.plugin_core import RpcPlugin

        state = {"session": self.open_session(session)}
        plugin = RpcPlugin(
            "look_capture", None, "PYTHONTK_LOOK_CAPTURE", port, host=host
        )

        def awake(fn):
            def op(**arguments):
                match, settings, _template, _size = state["session"]
                self.idle(False)
                try:
                    return fn(match, settings, **arguments)
                finally:
                    self.idle(True)

            return op

        for name, fn in self._ops().items():
            plugin.registry.register(name)(awake(fn))

        @plugin.registry.register("reload")
        def _reload():
            state["session"] = self.open_session(session)
            return {"reloaded": True}

        @plugin.registry.register("stop")
        def _stop():
            plugin.stop()
            return {"stopping": True}

        self.idle(True)  # asleep until a request wakes it
        print(f"look capture: {self.template}", flush=True)
        plugin.serve(port=port)  # the caller's port, never an environment override

    #: The line a :meth:`serve` process announces its address with (``RpcPlugin.serve``).
    _SERVING = re.compile(r"\[look_capture\] serving on (http://\S+)")

    @classmethod
    def spawn(
        cls,
        session: str,
        python: Optional[str] = None,
        timeout: float = 900.0,
        output: Optional[Any] = None,
    ) -> Tuple[Any, str]:
        """A fresh service on *session*: ``(process, url)`` once it serves.

        A process of its own (``look_capture serve``), bound to this one's
        lifetime (``AppLauncher.spawn``) and on a port the OS picks -- never one
        already running, whoever started it. It loads its template before it
        binds (minutes in software), so this blocks until then: call it off a
        UI thread. End it with its ``stop`` op (it closes its browser), then
        wait on *process*.

        Parameters:
            python: The interpreter to run it with (it needs Playwright);
                default: this process's own, or its host's (``mayapy`` inside
                Maya: ``AppLauncher.companion_python``).
            timeout: Seconds to wait for it to serve.
            output: An ``OutputStream`` its output goes to (a panel's log).

        Raises:
            RuntimeError: it exited first (its last output says why).
            TimeoutError: it did not serve within *timeout* (it is ended).
        """
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher
        from pythontk.core_utils.process_stream import OutputStream, ProcessReader

        python = python or AppLauncher.companion_python()
        if not python:
            raise RuntimeError(
                "No Python to run the capture service with here: pass python=."
            )
        args, env = AppLauncher.python_args_via_env(  # paths survive any code page
            [
                "-m",
                "pythontk.net_utils.preview.look_capture",
                "serve",
                os.path.abspath(session),
                "--port",
                "0",
            ]
        )
        # What it prints (its template's path) reaches the reader as UTF-8: a
        # pipe's default, the ANSI code page, cannot spell every path.
        env["PYTHONIOENCODING"] = "utf-8"
        process = AppLauncher.spawn(python, args, env=env)
        stream = output if output is not None else OutputStream()
        ProcessReader(process.stdout, stream, "look capture").start()
        deadline = time.time() + float(timeout)
        while True:
            hit = stream.wait_for(cls._SERVING, timeout=0.25)
            if hit is not None:
                return process, cls._SERVING.search(hit[1]).group(1).rstrip("/") + "/"
            if process.poll() is not None:
                time.sleep(0.2)  # the reader's last lines
                said = [line for _, line in stream.history()][-12:]
                raise RuntimeError(
                    f"The capture service exited ({process.returncode}) before it served:\n"
                    + "\n".join(said)
                )
            if time.time() > deadline:
                process.kill()  # ours, started just now, never answering
                raise TimeoutError(
                    f"The capture service did not serve {session} within {timeout:.0f}s."
                )

    @staticmethod
    def client(url: str):
        """An ``RpcClient`` of the :meth:`serve` session at *url*."""
        from pythontk.net_utils.rpc.client import RpcClient

        return RpcClient.for_url(url, app_label="look capture")

    # ------------------------------------------------------------------ command line
    @classmethod
    def cli(cls, argv: Optional[Sequence[str]] = None) -> int:
        command = CLI.Command
        session = ("session", {})
        service = ("--service", {"help": "a running session's URL (see serve)"})
        commands = {
            "serve": command(
                "keep the template loaded and answer requests (seconds per frame)",
                cls._cli_serve,
                [session, ("--port", {"type": int, "default": 8770})],
            ),
            "exact": command(
                "the mix as the viewer shows it",
                cls._cli_exact,
                [
                    session,
                    ("--mix", {"help": "a mix JSON instead of the session's"}),
                    ("--mode", {"default": "lit", "choices": ("lit", "linear")}),
                    (
                        "--out",
                        {"required": True, "help": ".png for lit, .npy for linear"},
                    ),
                    (
                        "--as-layer",
                        {
                            "action": "store_true",
                            "help": "leave out what a layer capture does (to check the prediction)",
                        },
                    ),
                    (
                        "--swap",
                        {
                            "action": "append",
                            "default": [],
                            "help": "FILE=PATH: a bake file to use instead of the mix's (repeat)",
                        },
                    ),
                    (
                        "--probe-at",
                        {
                            "help": "x,y,z: where the (swapped) probe was taken, in model space"
                        },
                    ),
                    service,
                ],
            ),
            "capture": command(
                "re-capture layers from their bakes",
                cls._cli_capture,
                [
                    session,
                    (
                        "--layer",
                        {
                            "action": "append",
                            "help": "a layer (repeat); default: every layer with a bake",
                        },
                    ),
                    (
                        "--from",
                        {
                            "dest": "source",
                            "help": "a bake folder (or a folder of them): new layers from their bakes",
                        },
                    ),
                    service,
                ],
            ),
            "ids": command(
                "which object every pixel shows (for objects regions)",
                cls._cli_ids,
                [
                    session,
                    ("--out", {"required": True, "help": "the ID image (.png)"}),
                    ("--target", {"help": "a target JSON to attach it to"}),
                    service,
                ],
            ),
            "stop": command(
                "end a running session (closes its browser)",
                cls._cli_stop,
                [("--service", {"default": "http://127.0.0.1:8770/"})],
            ),
        }
        return CLI.dispatch(
            "Look capture: exact WebXR frames of a mix, and layers captured from their bakes.",
            commands,
            argv,
        )

    @classmethod
    def _cli_serve(cls, args) -> None:
        _, settings, template, size = cls.open_session(args.session)
        with cls(template, settings.get("camera"), size) as capture:
            capture.serve(args.session, port=args.port)

    @classmethod
    def _cli_send(cls, args, request: Mapping[str, Any]) -> None:
        """*request* to the running service, or to a session opened for it alone."""
        if args.service:
            arguments = dict(request)
            client = cls.client(args.service)
            reply = client.invoke(arguments.pop("op"), timeout=3600.0, **arguments)
            if request["op"] == "capture":  # the service's store changed on disk
                client.invoke("reload")
        else:
            match, settings, template, size = cls.open_session(args.session)
            with cls(template, settings.get("camera"), size) as capture:
                reply = capture.handle(match, settings, request)
        print(json.dumps(reply, indent=1))

    @classmethod
    def _cli_exact(cls, args) -> None:
        cls._cli_send(
            args,
            {
                "op": "exact",
                "as_layer": args.as_layer,
                "swap": {
                    file_name: os.path.abspath(path)
                    for file_name, path in (item.split("=", 1) for item in args.swap)
                }
                or None,
                "probe": {"position": [float(v) for v in args.probe_at.split(",")]}
                if args.probe_at
                else None,
                "mix": LightMix.load(args.mix).to_dict() if args.mix else None,
                "mode": args.mode,
                "out": os.path.abspath(args.out),
            },
        )

    @classmethod
    def _cli_capture(cls, args) -> None:
        cls._cli_send(
            args,
            {
                "op": "capture",
                "layers": args.layer,
                "source": os.path.abspath(args.source) if args.source else None,
            },
        )

    @classmethod
    def _cli_ids(cls, args) -> None:
        cls._cli_send(
            args,
            {
                "op": "ids",
                "out": os.path.abspath(args.out),
                "target": os.path.abspath(args.target) if args.target else None,
            },
        )

    @classmethod
    def _cli_stop(cls, args) -> None:
        print(json.dumps(cls.client(args.service).invoke("stop")))


if __name__ == "__main__":
    sys.exit(LookCapture.cli())
