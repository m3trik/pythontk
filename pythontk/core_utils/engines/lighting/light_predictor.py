# !/usr/bin/python
# coding=utf-8
"""The frame of a mix: the tinted sum of its layers, through the viewer's display transform."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a declared dependency
    np = None  # type: ignore

from pythontk.img_utils._img_utils import ImgUtils
from pythontk.core_utils.engines.lighting.light_layers import LightLayers, LightMix


class ViewerTransform:
    """How a viewer turns linear light into what the screen shows.

    Parameters:
        operator: The tone map (:meth:`ImgUtils.tonemap`); the WebXR viewer's is
            ``"aces_filmic"``.
        exposure: The renderer's tone-mapping exposure.
    """

    def __init__(self, operator: str = "aces_filmic", exposure: float = 1.0):
        self.operator = operator
        self.exposure = float(exposure)

    def display_linear(self, linear, exposure: float = 1.0) -> "np.ndarray":
        """Tone-mapped display light, linear 0..1."""
        return ImgUtils.tonemap(linear, self.operator, self.exposure * exposure)

    def display(self, linear, exposure: float = 1.0) -> "np.ndarray":
        """The sRGB-encoded 0..1 value the screen shows."""
        return ImgUtils.linear_to_srgb(self.display_linear(linear, exposure))

    def to_dict(self) -> Dict[str, Any]:
        return {"operator": self.operator, "exposure": self.exposure}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ViewerTransform":
        return cls(data.get("operator", "aces_filmic"), data.get("exposure", 1.0))


class LightPredictor:
    """Renders a :class:`LightMix` from a :class:`LightLayers` store, as the viewer would.

    Lightmaps and the reflection probe are linear in each light, so the linear
    frame of a mix is the sum of its layers, each scaled by its weight and, for
    a light, tinted by its colour temperature. Emission and viewer-only panels
    were captured in their own colour and are scaled, never tinted.

    Parameters:
        layers: The layer store.
        transform: The viewer's display transform (default: the WebXR viewer's).
    """

    def __init__(
        self, layers: LightLayers, transform: Optional[ViewerTransform] = None
    ):
        self.layers = layers
        self.transform = transform or ViewerTransform()
        self._bake_cache: Dict[Tuple[str, float], "np.ndarray"] = {}

    # ------------------------------------------------------------------ colour
    def _scale(self, mix: LightMix, name: str) -> "np.ndarray":
        weight = mix.weight(name)
        tinted = self.layers.kind(name).tinted
        color = ImgUtils.kelvin_tint(mix.kelvin(name)) if tinted else (1, 1, 1)
        return (weight * np.asarray(color, np.float32)).astype(np.float32)

    def _shape(self, size, region) -> Tuple[int, int]:
        if region:
            x0, y0, x1, y1 = region
            return y1 - y0, x1 - x0
        width, height = size or self.layers.size or (0, 0)
        return height, width

    # ------------------------------------------------------------------ frames
    def linear(
        self,
        mix: LightMix,
        size: Optional[Tuple[int, int]] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
        names: Optional[Iterable[str]] = None,
    ) -> "np.ndarray":
        """The mix's linear frame (``(height, width, 3)`` float32).

        Parameters:
            size: ``(width, height)`` -- the layers area-averaged down to it.
            region: ``(x0, y0, x1, y1)`` full-size window instead of the frame.
            names: Only these of the mix's layers (default: every active one).

        Raises:
            KeyError: the mix names a layer the store lacks.
        """
        out = np.zeros(self._shape(size, region) + (3,), np.float32)
        term = None  # one scratch frame for every layer: no temporary per layer
        for name in names if names is not None else mix.active():
            if mix.weight(name) == 0:
                continue
            if term is None:
                term = np.empty_like(out)
            np.multiply(self.layers.load(name, size, region), self._scale(mix, name), out=term)
            out += term
        return out

    def frame(
        self,
        mix: LightMix,
        size: Optional[Tuple[int, int]] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> "np.ndarray":
        """The mix as the screen shows it: sRGB-encoded 0..1 float32."""
        return self.transform.display(self.linear(mix, size, region), mix.exposure)

    def bakes(self, mix: LightMix) -> Dict[str, "np.ndarray"]:
        """The mix's bake: each bake file summed over the active layers, scaled and tinted
        as their frames are -- what a viewer swaps in to show the mix exactly. A
        viewer-only layer (a preview panel, not shippable) is no part of it.

        Returns:
            ``{file name: (height, width, 3) float32}`` linear light (Rec.709).

        Raises:
            KeyError: an active shippable layer has no bake in the store.
            ImportError: OpenCV, which reads the EXR files, is not installed.
        """
        sums: Dict[str, "np.ndarray"] = {}
        for name in mix.active():
            if not self.layers.info(name).get("shippable", True):
                continue  # a viewer-only preview: no bake carries it
            files = self.layers.bake_files(name)
            if not files:
                raise KeyError(f"Layer {name!r} has no bake in {self.layers.root}.")
            scale = self._scale(mix, name)
            for file_name, path in files.items():
                key = (path, os.path.getmtime(path))
                image = self._bake_cache.get(key)
                if image is None:
                    image = self._bake_cache[key] = self.read_bake(path)
                sums[file_name] = sums.get(file_name, 0) + scale * image
        return sums

    def lights(self, mix: LightMix) -> List[Dict[str, Any]]:
        """The mix's lights, as the deliverable draws them into its probe's highlight copies.

        Each active light layer's own lights (its ``lights``: the lightmap
        record's form, baked white at the layer's gain), at the layer's weight
        and tinted by its colour temperature -- as its frame is -- times the
        layer's specular scale (:meth:`LightMix.specular`: these lights light
        only highlights, so it softens them and leaves the bake alone). What an
        exact frame lights (``LookCapture.exact``), so it shows what ships.
        """
        out: List[Dict[str, Any]] = []
        for name in mix.active():
            info = self.layers.info(name)
            if not self.layers.kind(name).lights:
                continue
            weight = mix.weight(name) * mix.specular(name)
            tint = ImgUtils.kelvin_tint(mix.kelvin(name))
            for light in info.get("lights") or []:
                light = dict(light)
                light["intensity"] = float(light.get("intensity", 0.0)) * weight
                colour = light.get("color") or (1.0, 1.0, 1.0)
                light["color"] = [float(c) * float(t) for c, t in zip(colour, tint)]
                out.append(light)
        return out

    @staticmethod
    def read_bake(path: str) -> "np.ndarray":
        """A bake file (a lightmap atlas or a probe, EXR) as ``(height, width, 3)``
        float32 linear RGB.

        Raises:
            OSError: the file cannot be read as an image.
            ImportError: OpenCV, which reads EXR, is not installed.
        """
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # before cv2 is imported
        import cv2  # an optional dependency: only exact frames read bakes

        raw = ImgUtils.cv2_read(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise OSError(f"Unreadable bake {path}")
        if raw.ndim == 2:
            raw = raw[..., None].repeat(3, axis=2)
        return np.ascontiguousarray(raw[..., :3][..., ::-1], dtype=np.float32)

    def contributions(
        self,
        mix: LightMix,
        mask: "np.ndarray",
        size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, float]:
        """Each active layer's share of the linear luminance inside *mask* (sums to 1).

        Answers "which lights light this region", the question behind every
        manual move a fit cannot make.
        """
        return self.region_contributions(mix, {"_": mask}, size)["_"]

    def region_contributions(
        self,
        mix: LightMix,
        masks: Mapping[str, "np.ndarray"],
        size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """:meth:`contributions` for many regions in one pass over the layers.

        Parameters:
            masks: ``{region: boolean mask}`` at *size*.

        Returns:
            ``{region: {layer: share}}``.
        """
        light: Dict[str, Dict[str, float]] = {region: {} for region in masks}
        for name in mix.active():
            image = ImgUtils.luminance(
                self._scale(mix, name) * self.layers.load(name, size)
            )
            for region, mask in masks.items():
                light[region][name] = float(image[mask].sum())
        out = {}
        for region, values in light.items():
            total = sum(values.values()) or 1.0
            out[region] = {name: value / total for name, value in values.items()}
        return out
