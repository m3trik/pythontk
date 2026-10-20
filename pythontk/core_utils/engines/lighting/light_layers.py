# !/usr/bin/python
# coding=utf-8
"""Light layers and mixes: the data a look match works on.

:class:`LightLayers` stores one linear viewer frame per light (float16 ``.npy``
files beside a ``layers.json`` manifest). :class:`LightMix` is a lighting state:
a weight and a colour temperature per layer.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from pythontk.core_utils.schema_spec import SchemaSpec

_field = SchemaSpec.spec_field

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a declared dependency
    np = None  # type: ignore

from pythontk.file_utils._file_utils import FileUtils
from pythontk.img_utils._img_utils import ImgUtils


class LightMix:
    """A lighting state: per layer, a weight and an optional colour temperature.

    Parameters:
        entries: ``{layer: {"weight": float, "kelvin": float | None}}``, and
            optionally ``"specular"``: the layer's highlight level apart from
            its light (Arnold's ``aiSpecular``; default 1). It scales only what
            ships for the highlights (:meth:`LightPredictor.lights`) -- the
            bake, the light's diffuse, is unchanged.
        exposure: The viewer exposure the mix is meant for.
        notes: Free text.

    Example:
        >>> mix = LightMix({"key": {"weight": 2.0, "kelvin": 3200}})
        >>> mix.with_entries({"fill": {"weight": 0.4}}).active()
        ['fill', 'key']
    """

    VERSION = 1

    def __init__(
        self,
        entries: Optional[Mapping[str, Mapping[str, Any]]] = None,
        exposure: float = 1.0,
        notes: str = "",
    ):
        self._entries: Dict[str, Dict[str, Any]] = {}
        for name, entry in (entries or {}).items():
            self._entries[name] = {
                "weight": float(entry.get("weight", 0.0)),
                "kelvin": None
                if entry.get("kelvin") in (None, 0)
                else float(entry["kelvin"]),
            }
            specular = float(entry.get("specular", 1.0))
            if specular < 0:
                raise ValueError(f"{name!r}: a specular scale is never negative.")
            if specular != 1.0:  # the default stays out of the file
                self._entries[name]["specular"] = specular
        self.exposure = float(exposure)
        self.notes = notes

    # ------------------------------------------------------------------ reading
    def names(self) -> List[str]:
        """Every layer the mix names, sorted."""
        return sorted(self._entries)

    def active(self) -> List[str]:
        """The layers with a positive weight, sorted."""
        return sorted(n for n, e in self._entries.items() if e["weight"] > 0)

    def weight(self, name: str) -> float:
        return self._entries.get(name, {}).get("weight", 0.0)

    def kelvin(self, name: str) -> Optional[float]:
        return self._entries.get(name, {}).get("kelvin")

    def specular(self, name: str) -> float:
        """The layer's highlight level (Arnold's ``aiSpecular``; 1 by default)."""
        return self._entries.get(name, {}).get("specular", 1.0)

    def entries(self) -> Dict[str, Dict[str, Any]]:
        """A copy of ``{layer: {"weight", "kelvin"}}``."""
        return copy.deepcopy(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    # ------------------------------------------------------------------ changing
    def with_entries(self, changes: Mapping[str, Mapping[str, Any]]) -> "LightMix":
        """A copy with *changes* merged per layer (a field left out keeps its value)."""
        entries = self.entries()
        for name, change in changes.items():
            entry = entries.setdefault(name, {"weight": 0.0, "kelvin": None})
            entry.update(change)
        return LightMix(entries, self.exposure, self.notes)

    def with_weights(self, weights: Mapping[str, float]) -> "LightMix":
        """A copy with these layers' weights replaced."""
        return self.with_entries({n: {"weight": float(w)} for n, w in weights.items()})

    # ------------------------------------------------------------------ io
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "exposure": self.exposure,
            "notes": self.notes,
            "layers": self.entries(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LightMix":
        version = data.get("version", cls.VERSION)
        if version > cls.VERSION:
            raise ValueError(
                f"Light mix version {version} is newer than this pythontk reads ({cls.VERSION})."
            )
        return cls(
            data.get("layers") or {}, data.get("exposure", 1.0), data.get("notes", "")
        )

    def save(self, path: str) -> str:
        FileUtils.write_json(path, self.to_dict(), indent=1)
        return path

    @classmethod
    def load(cls, path: str) -> "LightMix":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class LightLayerTarget(SchemaSpec):
    """What a mix weight of an emission layer scales: a material's emission level."""

    TYPED = True

    node: str = _field(help="The material the deliverable reads.", required=True)
    attr: str = _field(
        help="Its emission level (a weight, not a colour).", required=True
    )
    scale: float = _field(
        help="The level the layer was baked at: weight 1 is the room at it.",
        required=True,
        default=1.0,
    )


@dataclass
class LightLayerScene(SchemaSpec):
    """The scene file a layer was baked from, as it stood: a layer from an
    older save of it is stale (:meth:`matches`)."""

    TYPED = True

    file: str = _field(help="The scene's file name.", required=True)
    mtime: float = _field(help="Its modification time (epoch seconds).", required=True)
    bytes: int = _field(help="Its size.", required=True)

    @classmethod
    def of(cls, path: str) -> Dict[str, Any]:
        """The record of the scene file at *path* as it stands now."""
        stat = os.stat(path)
        return {
            "file": os.path.basename(path),
            "mtime": stat.st_mtime,
            "bytes": stat.st_size,
        }

    @classmethod
    def matches(cls, record: Optional[Mapping[str, Any]], path: str) -> bool:
        """Whether *record* is the scene file at *path* as it stands now (no
        record, or no file there: False). A file saved since, or another
        scene, does not match."""
        if not record or not os.path.isfile(path):
            return False
        now = cls.of(path)
        return (
            record.get("file") == now["file"]
            and record.get("bytes") == now["bytes"]
            and abs(float(record.get("mtime", 0)) - now["mtime"]) < 1e-3
        )


@dataclass
class LightLayerInfo(SchemaSpec):
    """A bake folder's ``layer.json``: one layer as a host's layer baker wrote it,
    what :meth:`LookCapture.capture_bake` stores (mayatk's ``LightLayerBaker``)."""

    TYPED = True
    FILE = "layer.json"

    name: str = _field(help="The layer's name (a file stem).", required=True)
    kind: str = _field(help="One of LightLayers.KINDS.", required=True)
    spec: Optional[Dict[str, Any]] = _field(
        help="The payload it was baked from (a LightRigPayload's, or an emission's materials)."
    )
    gain: Optional[float] = _field(
        help="What its lights were multiplied by (white layers)."
    )
    settings: Optional[Dict[str, Any]] = _field(help="The baker's settings.")
    files: List[str] = _field(
        help="Its maps and probe, in the folder.", required=True, default_factory=list
    )
    lights: Optional[List[Dict[str, Any]]] = _field(
        help="Its own lights, the lightmap record's form: a capture lights their highlights."
    )
    target: Optional[LightLayerTarget] = _field(
        help="An emission layer's: what a mix weight scales."
    )
    scene: Optional[LightLayerScene] = _field(
        help="The scene file it was baked from, as it stood (LightLayerScene)."
    )
    seconds: Optional[float] = _field(help="How long the bake took.")


@dataclass(frozen=True)
class LayerKind:
    """What a layer of one kind is, and how it reaches the deliverable.

    Parameters:
        tinted: The mix's colour temperature colours it (a light baked white);
            False: it was captured in its own colour (emission, a panel).
        shippable: An export carries it unless its entry says otherwise
            (False: a viewer-only preview).
        lights: Its spec's lights light the deliverable's highlights (its
            entry's ``lights``, the lightmap record's form).
        rig: How a rig built from a mix carries it: ``"lights"`` (its spec's
            lights at the mix's weight), ``"material"`` (its ``target``: a
            material's emission level) or ``None`` (nothing builds it).
    """

    tinted: bool = False
    shippable: bool = True
    lights: bool = False
    rig: Optional[str] = None


class LightLayers:
    """A store of light layers: each light's linear viewer frame, captured alone.

    A layer is scene-referred linear Rec.709 light (before the viewer's tone map)
    from one light at unit weight and white colour. The frame of any mix is the
    weighted, tinted sum of its layers -- exact while every layer comes from the
    same viewer, camera and scene template, which the manifest's ``stamp`` records.

    Layout: ``<root>/layers.json`` (the manifest) and ``<root>/<name>.npy``
    (float16, ``(height, width, 3)``).

    Parameters:
        root: The store's folder (made on the first :meth:`add`).
    """

    MANIFEST = "layers.json"
    VERSION = 1
    #: What a layer is, by kind: a light, the scene's emissive surfaces, or a
    #: viewer-only reflection panel (a preview no export carries).
    #: :meth:`register_kind` adds one.
    KINDS: Dict[str, LayerKind] = {
        "light": LayerKind(tinted=True, lights=True, rig="lights"),
        "emission": LayerKind(rig="material"),
        "panel": LayerKind(shippable=False),
    }

    @classmethod
    def register_kind(cls, name: str, kind: LayerKind) -> None:
        """Add (or replace) a layer kind."""
        cls.KINDS[name] = kind

    #: What a kind this version does not know reads as: inert -- viewer-only,
    #: untinted, never built into a rig (a store a newer tool wrote).
    UNKNOWN_KIND = LayerKind(shippable=False)

    def kind(self, name: str) -> LayerKind:
        """Layer *name*'s :class:`LayerKind` (a manifest without one: a light;
        one this version does not know: :attr:`UNKNOWN_KIND`)."""
        return self.KINDS.get(self.info(name).get("kind", "light"), self.UNKNOWN_KIND)

    def __init__(self, root: str):
        self.root = str(root)
        self._manifest = self._read_manifest()
        self._cache: Dict[Tuple, "np.ndarray"] = {}

    # ------------------------------------------------------------------ manifest
    def _manifest_path(self) -> str:
        return os.path.join(self.root, self.MANIFEST)

    def _read_manifest(self) -> Dict[str, Any]:
        path = self._manifest_path()
        if not os.path.isfile(path):
            return {"version": self.VERSION, "size": None, "stamp": {}, "layers": {}}
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version", 1) > self.VERSION:
            raise ValueError(
                f"Layer store {self.root} is version {data['version']}; this pythontk reads {self.VERSION}."
            )
        return data

    def _refresh(self) -> None:
        """Re-read the manifest before a change: another holder of the store (a
        capture service beside a loop) may have written it since, and a write
        from a stale copy would drop what it added. Frames it replaced are
        forgotten."""
        fresh = self._read_manifest()
        for name in set(self._manifest["layers"]) | set(fresh["layers"]):
            if self._manifest["layers"].get(name) != fresh["layers"].get(name):
                self._forget(name)
        self._manifest = fresh

    def _write_manifest(self) -> None:
        FileUtils.write_json(self._manifest_path(), self._manifest, indent=1)

    @property
    def size(self) -> Optional[Tuple[int, int]]:
        """``(width, height)`` of every layer, or ``None`` while the store is empty."""
        size = self._manifest.get("size")
        return tuple(size) if size else None

    @property
    def stamp(self) -> Dict[str, Any]:
        """What the layers were captured with (viewer, camera, template...): one per store."""
        return dict(self._manifest.get("stamp") or {})

    def set_stamp(self, **fields: Any) -> None:
        """Record what the layers were captured with (merged; ``None`` removes a key)."""
        self._refresh()
        stamp = self._manifest.setdefault("stamp", {})
        for key, value in fields.items():
            if value is None:
                stamp.pop(key, None)
            else:
                stamp[key] = value
        self._write_manifest()

    def names(self, kind: Optional[str] = None) -> List[str]:
        """The layers, sorted; only those of *kind* when given."""
        layers = self._manifest["layers"]
        return sorted(
            n for n, e in layers.items() if kind is None or e.get("kind") == kind
        )

    def info(self, name: str) -> Dict[str, Any]:
        """A layer's manifest entry: ``kind``, ``shippable``, ``spec``, ``added`` and any extra fields."""
        try:
            return copy.deepcopy(self._manifest["layers"][name])
        except KeyError:
            raise KeyError(f"No layer {name!r} in {self.root}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._manifest["layers"]

    # ------------------------------------------------------------------ frames
    def _path(self, name: str) -> str:
        return os.path.join(self.root, f"{name}.npy")

    def add(
        self,
        name: str,
        frame: "np.ndarray",
        kind: str = "light",
        spec: Optional[Mapping[str, Any]] = None,
        shippable: Optional[bool] = None,
        replace: bool = False,
        bakes: Optional[Iterable[str]] = None,
        **info: Any,
    ) -> str:
        """Store *frame* as layer *name*; returns the file written.

        Parameters:
            name: The layer's name (a file stem).
            frame: ``(height, width, 3)`` linear light; the store's size is the
                first layer's, and every layer must match it.
            kind: One of :attr:`KINDS`.
            spec: What made it -- a light spec the rig built from it will use.
            shippable: Whether an export carries it; default: every kind but ``panel``.
            replace: Allow overwriting an existing layer.
            bakes: The bake files the frame was captured from (lightmap atlases
                and the reflection probe, linear EXR): copied into the store,
                so an exact frame of any mix can be rebuilt from them.
                ``None`` keeps the bake the layer has.
            **info: Extra manifest fields (hashes, notes).

        Raises:
            ValueError: an unknown *kind*, or a size other than the store's.
            FileExistsError: *name* exists and *replace* is False.
        """
        if kind not in self.KINDS:
            raise ValueError(
                f"Unknown layer kind {kind!r}; one of {tuple(self.KINDS)}."
            )
        self._refresh()
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"A layer is (height, width, 3); got {frame.shape}.")
        size = (frame.shape[1], frame.shape[0])
        if self.size and size != self.size:
            raise ValueError(
                f"Layer {name!r} is {size}; this store's layers are {self.size}."
            )
        if name in self and not replace:
            raise FileExistsError(
                f"Layer {name!r} exists in {self.root}; pass replace=True to overwrite it."
            )
        os.makedirs(self.root, exist_ok=True)
        path = self._path(name)

        def write(part: str) -> None:  # a handle: np.save appends ".npy" to a path
            with open(part, "wb") as f:
                np.save(f, frame.astype(np.float16))

        FileUtils.atomic_write(path, write)
        entry = {
            "kind": kind,
            "shippable": self.KINDS[kind].shippable
            if shippable is None
            else bool(shippable),
            "spec": copy.deepcopy(dict(spec)) if spec else None,
            "added": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        entry.update(info)
        self._manifest["size"] = list(size)
        self._manifest["layers"][name] = entry
        if bakes is not None:
            folder = self._bake_dir(name)
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder)
            for bake in bakes:
                shutil.copy2(bake, os.path.join(folder, os.path.basename(bake)))
        self._write_manifest()
        self._forget(name)
        return path

    def _bake_dir(self, name: str) -> str:
        return os.path.join(self.root, "bakes", name)

    def bake_files(self, name: str) -> Dict[str, str]:
        """``{file name: path}`` of the bake layer *name* was captured from (``{}`` for none)."""
        self.info(name)
        folder = self._bake_dir(name)
        if not os.path.isdir(folder):
            return {}
        return {f: os.path.join(folder, f) for f in sorted(os.listdir(folder))}

    def remove(self, name: str) -> None:
        """Delete a layer and its file."""
        self._refresh()
        self.info(name)  # names an unknown layer
        try:
            os.remove(self._path(name))
        except FileNotFoundError:
            pass
        del self._manifest["layers"][name]
        shutil.rmtree(self._bake_dir(name), ignore_errors=True)
        if not self._manifest["layers"]:
            self._manifest["size"] = None
        self._write_manifest()
        self._forget(name)

    def _forget(self, name: str) -> None:
        for key in [k for k in self._cache if k[0] == name]:
            del self._cache[key]

    def load(
        self,
        name: str,
        size: Optional[Tuple[int, int]] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> "np.ndarray":
        """A layer as float32 ``(height, width, 3)`` linear light (cached per read).

        Parameters:
            size: ``(width, height)`` to area-average the whole frame down to.
            region: ``(x0, y0, x1, y1)`` in full-size pixels: only that window,
                read from a memory map (a close-up never loads the whole frame).
        """
        self.info(name)
        key = (name, tuple(size) if size else None, tuple(region) if region else None)
        if key not in self._cache:
            path = self._path(name)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Layer {name!r} has no capture: {path}")
            arr = np.load(path, mmap_mode="r")
            if region:
                x0, y0, x1, y1 = region
                out = np.asarray(arr[y0:y1, x0:x1], dtype=np.float32)
            else:
                out = np.asarray(arr, dtype=np.float32)
                if size and tuple(size) != (out.shape[1], out.shape[0]):
                    out = ImgUtils.resample_area(out, size)
            self._cache[key] = out
        return self._cache[key]

    def validate(self) -> List[str]:
        """Problems that would corrupt a prediction or an exact frame: a missing
        capture, a size mismatch, a light layer whose bake recorded none of its
        lights (an exact frame ships none of their highlights)."""
        problems = []
        for name in self.names():
            info = self.info(name)
            if (
                self.kind(name).lights
                and (info.get("spec") or {}).get("lights")
                and not info.get("lights")
            ):
                problems.append(
                    f"{name}: its bake recorded none of its lights, so exact frames"
                    " show none of their highlights -- re-bake it"
                )
            path = self._path(name)
            if not os.path.isfile(path):
                problems.append(f"{name}: no capture ({path})")
                continue
            shape = np.load(path, mmap_mode="r").shape
            if self.size and (shape[1], shape[0]) != self.size:
                problems.append(
                    f"{name}: {shape[1]}x{shape[0]}, the store is {self.size[0]}x{self.size[1]}"
                )
        return problems

    def iter_layers(self, names: Optional[Iterable[str]] = None):
        """``(name, info)`` for the given or all layers."""
        for name in names or self.names():
            yield name, self.info(name)
