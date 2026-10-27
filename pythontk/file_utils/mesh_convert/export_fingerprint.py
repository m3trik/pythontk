# !/usr/bin/python
# coding=utf-8
"""A deterministic structural summary of an exported GLB or FBX, and its diff.

The baseline half of export verification: :class:`ExportVerifier` judges one
deliverable on its own, while a fingerprint pins what an export path wrote so
the next run of the same scene can be compared against it -- the check a code
move that must not change any output needs. It holds what a reader of the file
depends on (hierarchy, transforms, mesh and vertex counts, materials, skins,
takes and their key spans, the extras and user-property names) and leaves out
what legitimately differs run over run: generator strings, buffer packing,
timestamps and paths inside records.

Example:
    >>> fingerprint = ExportFingerprint.of("asset.glb")
    >>> ExportFingerprint.write(fingerprint, "asset.glb.json")
    >>> ExportFingerprint.diff(ExportFingerprint.read("asset.glb.json"), fingerprint)
    []
"""

import json
import os
from typing import Any, Dict, List, Optional

from pythontk.file_utils.mesh_convert.fbx_file import FbxFile
from pythontk.file_utils.mesh_convert.glb.reader import GlbReader


class ExportFingerprint:
    """Fingerprint an exported GLB or FBX, and diff two fingerprints.

    A fingerprint is plain JSON-able data -- dicts, lists, strings and numbers
    rounded to :attr:`DIGITS` -- so it is saved beside a test as a baseline
    and compared within a tolerance (:meth:`diff`).
    """

    #: Decimal places numbers are rounded to; :meth:`diff`'s tolerance absorbs
    #: a value that lands on either side of a rounding edge.
    DIGITS = 4
    #: FBX ticks per second (the unit of an FBX key time).
    FBX_TICKS_PER_SECOND = 46186158000

    @classmethod
    def of(cls, path: str) -> Dict[str, Any]:
        """The fingerprint of the GLB or FBX at *path*, chosen by extension.

        Raises:
            ValueError: *path* is neither a ``.glb`` nor an ``.fbx``.
        """
        extension = os.path.splitext(path)[1].lower()
        if extension == ".glb":
            return cls.of_glb(path)
        if extension == ".fbx":
            return cls.of_fbx(path)
        raise ValueError(f"not a GLB or FBX: {path}")

    @classmethod
    def of_glb(cls, path: str) -> Dict[str, Any]:
        """A GLB's nodes (by name path), meshes, materials, skins, animations
        and extras names."""
        gltf = GlbReader.load(path).gltf
        nodes = gltf.get("nodes") or []
        node_keys = cls._node_keys(nodes)
        mesh_keys = cls._unique([m.get("name") for m in gltf.get("meshes") or []], "mesh")
        material_keys = cls._unique(
            [m.get("name") for m in gltf.get("materials") or []], "material"
        )
        skin_keys = cls._unique([s.get("name") for s in gltf.get("skins") or []], "skin")
        accessors = gltf.get("accessors") or []

        def count(index: Any) -> Optional[int]:
            if isinstance(index, int) and 0 <= index < len(accessors):
                return int(accessors[index].get("count", 0))
            return None

        fingerprint: Dict[str, Any] = {
            "format": "glb",
            "extensions_used": sorted(gltf.get("extensionsUsed") or []),
            "extras": sorted((gltf.get("extras") or {}).keys()),
            "images": sorted(i.get("mimeType") or "?" for i in gltf.get("images") or []),
            "nodes": {},
            "meshes": {},
            "materials": {},
            "skins": {},
            "animations": {},
        }
        for i, node in enumerate(nodes):
            entry: Dict[str, Any] = {}
            for key in ("translation", "rotation", "scale", "matrix"):
                if key in node:
                    entry[key] = cls._round(node[key])
            if "mesh" in node:
                entry["mesh"] = mesh_keys[node["mesh"]]
            if "skin" in node:
                entry["skin"] = skin_keys[node["skin"]]
            if "camera" in node:
                entry["camera"] = True
            if node.get("extras"):
                entry["extras"] = sorted(node["extras"].keys())
            fingerprint["nodes"][node_keys[i]] = entry
        for key, mesh in zip(mesh_keys, gltf.get("meshes") or []):
            fingerprint["meshes"][key] = [
                {
                    "attributes": {
                        name: count(index)
                        for name, index in sorted((p.get("attributes") or {}).items())
                    },
                    "indices": count(p.get("indices")),
                    "material": (
                        material_keys[p["material"]] if "material" in p else None
                    ),
                    "mode": p.get("mode", 4),
                    "targets": len(p.get("targets") or []),
                }
                for p in mesh.get("primitives") or []
            ]
        for key, material in zip(material_keys, gltf.get("materials") or []):
            pbr = material.get("pbrMetallicRoughness") or {}
            fingerprint["materials"][key] = {
                "alpha_mode": material.get("alphaMode", "OPAQUE"),
                "base_color": cls._round(pbr.get("baseColorFactor", [1, 1, 1, 1])),
                "double_sided": bool(material.get("doubleSided", False)),
                "emissive": cls._round(material.get("emissiveFactor", [0, 0, 0])),
                "extensions": sorted((material.get("extensions") or {}).keys()),
                "metallic": cls._round(pbr.get("metallicFactor", 1)),
                "roughness": cls._round(pbr.get("roughnessFactor", 1)),
                "textures": sorted(
                    [k for k in material if k.endswith("Texture")]
                    + [k for k in pbr if k.endswith("Texture")]
                ),
            }
        for key, skin in zip(skin_keys, gltf.get("skins") or []):
            fingerprint["skins"][key] = {
                "inverse_bind_matrices": "inverseBindMatrices" in skin,
                "joints": [node_keys[j] for j in skin.get("joints") or []],
            }
        animation_keys = cls._unique(
            [a.get("name") for a in gltf.get("animations") or []], "animation"
        )
        for key, animation in zip(animation_keys, gltf.get("animations") or []):
            samplers = animation.get("samplers") or []
            channels: Dict[str, Any] = {}
            starts, ends = [], []
            for channel in animation.get("channels") or []:
                target = channel.get("target") or {}
                sampler = samplers[channel.get("sampler", 0)]
                times = accessors[sampler["input"]] if "input" in sampler else {}
                starts += times.get("min") or []
                ends += times.get("max") or []
                node = target.get("node")
                name = node_keys[node] if isinstance(node, int) else "?"
                channels[f"{name}|{target.get('path')}"] = {
                    "interpolation": sampler.get("interpolation", "LINEAR"),
                    "keys": int(times.get("count", 0)),
                }
            fingerprint["animations"][key] = {
                "channels": dict(sorted(channels.items())),
                "span": cls._round([min(starts), max(ends)]) if starts else None,
            }
        return fingerprint

    @classmethod
    def of_fbx(cls, path: str) -> Dict[str, Any]:
        """An FBX's object census, models with their local transforms,
        geometry sizes, materials, takes with each curve's key span, and
        user-property names."""
        fbx = FbxFile.load(path, raw_payloads=False, span_arrays=("KeyTime",))
        fingerprint: Dict[str, Any] = {
            "format": "fbx",
            "census": dict(sorted(fbx.objects_census().items())),
            "models": {},
            "geometry": {},
            "materials": {},
            "takes": {},
            "user_properties": {},
        }
        models: List[Any] = []
        geometry: List[Any] = []
        for record in fbx.iter_objects():
            if record["name"] == "Model":
                kind = record["props"][2] if len(record["props"]) > 2 else b""
                entry = {"type": cls._text(kind), **cls._local_transform(record)}
                models.append((fbx.object_name(record), entry))
            elif record["name"] == "Geometry":
                sizes = {
                    child["name"]: cls._array_length(child["props"])
                    for child in record["children"]
                    if child["name"] in ("Vertices", "PolygonVertexIndex")
                }
                geometry.append((fbx.object_name(record), sizes))
        model_keys = cls._unique([name for name, _ in models], "model")
        for key, (_, entry) in zip(model_keys, models):
            fingerprint["models"][key] = entry
        geometry_keys = cls._unique([name for name, _ in geometry], "geometry")
        for key, (_, sizes) in zip(geometry_keys, geometry):
            fingerprint["geometry"][key] = {
                "polygon_vertices": sizes.get("PolygonVertexIndex"),
                "vertices": (
                    sizes["Vertices"] // 3 if sizes.get("Vertices") is not None else None
                ),
            }
        materials = fbx.materials()
        material_keys = cls._unique([m.get("name") for m in materials], "material")
        for key, material in zip(material_keys, materials):
            fingerprint["materials"][key] = {
                "properties": {
                    name: cls._round(cls._text(value))
                    for name, value in sorted((material.get("props") or {}).items())
                },
                "shading_model": material.get("shading_model"),
                "textures": {
                    slot: os.path.basename(cls._text(file) or "")
                    for slot, file in sorted((material.get("textures") or {}).items())
                },
            }
        for take, curves in sorted(fbx.take_curves().items()):
            fingerprint["takes"][take] = {
                "|".join(key): [
                    cls._round(cls._seconds(first)),
                    cls._round(cls._seconds(last)),
                    count,
                ]
                for key, (first, last, count) in sorted(curves.items())
            }
        for owner, name, value in fbx.user_property_items():
            fingerprint["user_properties"][f"{owner}.{name}"] = cls._kind_of(value)
        fingerprint["user_properties"] = dict(sorted(fingerprint["user_properties"].items()))
        return fingerprint

    @classmethod
    def diff(
        cls, baseline: Any, current: Any, tolerance: float = 1e-3
    ) -> List[str]:
        """Each way *current* differs from *baseline*, one line per leaf:
        ``"nodes/root/arm/translation[0]: 0.0 -> 2.0"``, ``"... : added"``,
        ``"... : removed"``. Numbers within *tolerance* are equal; an empty
        list means the export is unchanged."""
        found: List[str] = []

        def walk(old: Any, new: Any, where: str) -> None:
            if isinstance(old, dict) and isinstance(new, dict):
                for key in sorted(set(old) | set(new), key=str):
                    here = f"{where}/{key}" if where else str(key)
                    if key not in new:
                        found.append(f"{here}: removed")
                    elif key not in old:
                        found.append(f"{here}: added")
                    else:
                        walk(old[key], new[key], here)
            elif isinstance(old, list) and isinstance(new, list):
                if len(old) != len(new):
                    found.append(f"{where}: {len(old)} item(s) -> {len(new)}")
                for i, (a, b) in enumerate(zip(old, new)):
                    walk(a, b, f"{where}[{i}]")
            elif cls._is_number(old) and cls._is_number(new):
                if abs(old - new) > tolerance:
                    found.append(f"{where}: {old} -> {new}")
            elif old != new:
                found.append(f"{where}: {old!r} -> {new!r}")

        walk(baseline, current, "")
        return found

    @classmethod
    def against(
        cls, path: str, baseline: str, update: bool = False, tolerance: float = 1e-3
    ) -> List[str]:
        """:meth:`diff` of the export at *path* against the fingerprint saved at
        *baseline*. *update* (re)writes the baseline instead and returns no
        difference; a missing baseline is itself a difference, so a test never
        passes against nothing."""
        current = cls.of(path)
        if update:
            cls.write(current, baseline)
            return []
        if not os.path.isfile(baseline):
            return [f"no baseline at {baseline}: write one with update=True"]
        return cls.diff(cls.read(baseline), current, tolerance)

    @staticmethod
    def write(fingerprint: Dict[str, Any], path: str) -> str:
        """Save *fingerprint* as sorted, indented JSON (a reviewable diff)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(fingerprint, indent=1, sort_keys=True) + "\n")
        return path

    @staticmethod
    def read(path: str) -> Dict[str, Any]:
        """A fingerprint saved by :meth:`write`."""
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    # ---- helpers ------------------------------------------------------------

    @classmethod
    def _round(cls, value: Any) -> Any:
        """*value* with every float rounded to :attr:`DIGITS` (``-0.0`` -> ``0.0``)."""
        if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
            return value
        if isinstance(value, float):
            return round(value, cls.DIGITS) + 0.0
        if isinstance(value, (list, tuple)):
            return [cls._round(v) for v in value]
        if isinstance(value, dict):
            return {k: cls._round(v) for k, v in value.items()}
        return cls._text(value)

    @staticmethod
    def _text(value: Any) -> Any:
        """Bytes as text; anything else unchanged."""
        if isinstance(value, (bytes, bytearray)):
            return bytes(value).decode("utf-8", "replace")
        return value

    @classmethod
    def _seconds(cls, ticks: Any) -> Optional[float]:
        return None if ticks is None else ticks / cls.FBX_TICKS_PER_SECOND

    @staticmethod
    def _array_length(props: List[Any]) -> Optional[int]:
        """An FBX array property's length, decoded or a skipped placeholder
        (``("ARRAY", type, count)``)."""
        if not props:
            return None
        value = props[0]
        if isinstance(value, tuple) and len(value) >= 3 and value[0] == "ARRAY":
            return int(value[2])
        if isinstance(value, (list, tuple)):
            return len(value)
        return None

    #: The local-transform properties an FBX ``Model`` writes when they differ
    #: from the default; an absent one is the default.
    _LOCAL_TRANSFORM = (b"Lcl Translation", b"Lcl Rotation", b"Lcl Scaling")

    @classmethod
    def _local_transform(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        """A ``Model`` record's ``Lcl`` properties as written, by name."""
        found = {}
        for child in record["children"]:
            if child["name"] != "Properties70":
                continue
            for prop in child["children"]:
                props = prop["props"]
                if len(props) >= 7 and props[0] in cls._LOCAL_TRANSFORM:
                    found[cls._text(props[0])] = cls._round(list(props[4:7]))
        return dict(sorted(found.items()))

    @classmethod
    def _kind_of(cls, value: Any) -> Any:
        """A user property's value as a baseline holds it: a number rounded, a
        string or record by kind alone (it may carry a path or a date)."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return round(float(value), cls.DIGITS) + 0.0
        if isinstance(value, (bytes, bytearray, str)):
            return "text"
        return type(value).__name__

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _unique(names: List[Optional[str]], fallback: str) -> List[str]:
        """*names* made unique and non-empty, in order: an unnamed entry is
        ``<fallback>#<index>``, a repeat gains ``#2``, ``#3`` ..."""
        seen: Dict[str, int] = {}
        out = []
        for i, name in enumerate(names):
            name = name or f"{fallback}#{i}"
            seen[name] = seen.get(name, 0) + 1
            out.append(name if seen[name] == 1 else f"{name}#{seen[name]}")
        return out

    @classmethod
    def _node_keys(cls, nodes: List[Dict[str, Any]]) -> List[str]:
        """Each glTF node's key: its name path from the root (``root/arm``),
        made unique."""
        parent = {
            child: i for i, node in enumerate(nodes) for child in node.get("children") or []
        }
        names = [node.get("name") or f"node#{i}" for i, node in enumerate(nodes)]

        def path(i: int) -> str:
            chain, seen = [], set()
            while i is not None and i not in seen:
                seen.add(i)
                chain.append(names[i])
                i = parent.get(i)
            return "/".join(reversed(chain))

        return cls._unique([path(i) for i in range(len(nodes))], "node")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
