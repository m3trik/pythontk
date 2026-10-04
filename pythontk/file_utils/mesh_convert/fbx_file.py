# !/usr/bin/python
# coding=utf-8
"""Zero-dependency binary-FBX reader: header, node records, objects, takes.

The FBX half of deliverable verification. :meth:`MeshConvert.fbx_to_glb`
*converts* FBX through an external binary but nothing in the ecosystem could
*look inside* one — so a truncated write, a dropped take or a missing
animation stack surfaced only after a GLB conversion or a DCC import. This
reads Kaydara's binary container directly (both the 32-bit pre-7500 and the
64-bit 7500+ record layouts), decodes property records, and answers the
census questions a verifier asks. Array payloads are **skipped by default** —
a 200 MB production FBX parses in well under a second because the geometry is
never inflated.

Read-only by design: there is no writer here, so this can never be what
damages the file it inspects.
"""

import os
import struct
import zlib
from collections import Counter
from typing import Any, Collection, Dict, Iterator, List, Optional, Tuple, Union

#: The 23-byte magic every binary FBX starts with.
FBX_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"

#: FBX splits an object's display name from its class with this token:
#: ``b"Shot_1\\x00\\x01AnimStack"`` -> name ``Shot_1``, class ``AnimStack``.
_NAME_CLASS_SEPARATOR = b"\x00\x01"


class _FbxFileInternal:
    """Record parsing for :class:`FbxFile`."""

    _SCALAR_PROPS = {
        b"Y": ("<h", 2),
        b"C": ("<b", 1),
        b"I": ("<i", 4),
        b"F": ("<f", 4),
        b"D": ("<d", 8),
        b"L": ("<q", 8),
    }
    _ARRAY_PROPS = {
        b"f": ("<f", 4),
        b"d": ("<d", 8),
        b"l": ("<q", 8),
        b"i": ("<i", 4),
        b"b": ("<b", 1),
        b"c": ("<B", 1),
    }

    @classmethod
    def _read_property(
        cls, f, decode_arrays: Union[bool, str], raw_payloads: bool = True
    ) -> Any:
        """One property. *decode_arrays* is True (an array as a list), False
        (a ``("ARRAY", type, count)`` placeholder) or ``"span"`` -- a
        ``("SPAN", type, count, first, last)`` read without building the list,
        ``first``/``last`` ``None`` for an empty array."""
        kind = f.read(1)
        scalar = cls._SCALAR_PROPS.get(kind)
        if scalar:
            fmt, size = scalar
            return struct.unpack(fmt, f.read(size))[0]
        array = cls._ARRAY_PROPS.get(kind)
        if array:
            fmt, size = array
            count, encoding, byte_len = struct.unpack("<III", f.read(12))
            payload = f.read(byte_len)
            if not decode_arrays:
                return ("ARRAY", kind.decode(), count)
            if encoding == 1:
                payload = zlib.decompress(payload)
            if decode_arrays == "span":
                if not count:
                    return ("SPAN", kind.decode(), 0, None, None)
                first = struct.unpack_from(fmt, payload, 0)[0]
                last = struct.unpack_from(fmt, payload, (count - 1) * size)[0]
                return ("SPAN", kind.decode(), count, first, last)
            return list(struct.unpack(f"<{count}{fmt[-1]}", payload))
        if kind in (b"S", b"R"):
            length = struct.unpack("<I", f.read(4))[0]
            if kind == b"R" and not raw_payloads:
                f.seek(length, 1)
                return ("RAW", length)
            return f.read(length)
        raise ValueError(f"Unknown FBX property type {kind!r}")

    @classmethod
    def _read_record(
        cls,
        f,
        wide: bool,
        decode_arrays: Union[bool, Collection[str]],
        raw_payloads: bool = True,
        span_arrays: Collection[str] = (),
    ) -> Optional[Dict[str, Any]]:
        """One node record, or ``None`` at a NULL sentinel.

        *decode_arrays* is a bool for every record, or the names of the records
        whose array properties are inflated (``("KeyTime",)``); *span_arrays*
        names records whose arrays are read as their span alone
        (:meth:`_read_property`)."""
        if wide:
            header = f.read(24)
            if len(header) < 24:
                return None
            end, prop_count, _prop_len = struct.unpack("<QQQ", header)
        else:
            header = f.read(12)
            if len(header) < 12:
                return None
            end, prop_count, _prop_len = struct.unpack("<III", header)
        name_len = struct.unpack("<B", f.read(1))[0]
        name = f.read(name_len).decode("utf-8", "replace")
        if end == 0:
            return None
        decode = (
            decode_arrays if isinstance(decode_arrays, bool) else name in decode_arrays
        )
        if not decode and name in span_arrays:
            decode = "span"
        props = [cls._read_property(f, decode, raw_payloads) for _ in range(prop_count)]
        sentinel = 25 if wide else 13
        children: List[Dict[str, Any]] = []
        while f.tell() < end - sentinel:
            child = cls._read_record(f, wide, decode_arrays, raw_payloads, span_arrays)
            if child is None:
                break
            children.append(child)
        f.seek(end)
        return {"name": name, "props": props, "children": children}

    @staticmethod
    def _display_name(raw: Any) -> Optional[str]:
        """The human half of an FBX object-name property, or ``None``."""
        if not isinstance(raw, (bytes, bytearray)):
            return None
        return bytes(raw).split(_NAME_CLASS_SEPARATOR, 1)[0].decode("utf-8", "replace")

    @staticmethod
    def _merged_span(a: Tuple[Any, ...], b: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Two curves' ``(first, last, count)`` as one channel's: the earlier
        first key, the later last, every key counted. An end a curve was read
        without (``None``) defers to the other's."""
        firsts = [time for time in (a[0], b[0]) if time is not None]
        lasts = [time for time in (a[1], b[1]) if time is not None]
        return (
            min(firsts) if firsts else None,
            max(lasts) if lasts else None,
            a[2] + b[2],
        )


class FbxFile(_FbxFileInternal):
    """A parsed binary FBX, held read-only.

    Example:
        >>> fbx = FbxFile.load("asset.fbx")
        >>> fbx.version
        7700
        >>> fbx.take_names()
        ['Shot_1', 'Shot_2']
        >>> fbx.objects_census()["AnimationCurveNode"]
        27989
    """

    def __init__(self, path: str, version: int, roots: List[Dict[str, Any]]):
        self.path = path
        self.version = version
        self.roots = roots
        self._by_name = {r["name"]: r for r in roots}

    @classmethod
    def load(
        cls,
        path: str,
        decode_arrays: Union[bool, Collection[str]] = False,
        raw_payloads: bool = True,
        span_arrays: Collection[str] = (),
    ) -> "FbxFile":
        """Parse *path*.

        Parameters:
            path: Binary FBX file.
            decode_arrays: When False (default), array properties are read as
                ``("ARRAY", type_char, count)`` placeholders and their
                payloads skipped — the fast census mode. True inflates them
                (zlib where encoded) into python lists. A collection of record
                names inflates only those records' arrays.
            raw_payloads: When False, ``R`` (raw binary) properties — the
                embedded media, hundreds of MB on a textured export — are
                skipped as ``("RAW", length)`` placeholders instead of being
                copied into memory. A census that never looks at the bytes
                (node and take counts) has no reason to hold them.
            span_arrays: Names of records whose arrays are read as their span
                alone -- ``("SPAN", type, count, first, last)``, no list built.
                ``("KeyTime",)`` is what :meth:`take_curves` reads: every
                curve's extent, at a fraction of the memory a full decode of a
                production file's per-frame keys costs.

        Raises:
            ValueError: Not a binary FBX, or truncated or corrupt: a record
                that runs past the end of the file, or data that cannot be
                decoded.
        """
        with open(path, "rb") as f:
            magic = f.read(len(FBX_MAGIC))
            if magic != FBX_MAGIC:
                raise ValueError(f"Not a binary FBX: {path}")
            size = os.fstat(f.fileno()).st_size
            # A cut or corrupt file is this method's documented refusal, never
            # an exception no caller catches: a short read the struct or zlib
            # decode chokes on (``zlib.error`` reached past every caller's
            # ``(OSError, ValueError)``, a conversion's timeout budget among
            # them), or -- where nothing is decoded -- a record that ends past
            # the file, whose remains read as a whole file otherwise.
            try:
                version = struct.unpack("<I", f.read(4))[0]
                wide = version >= 7500
                roots: List[Dict[str, Any]] = []
                while True:
                    record = cls._read_record(
                        f, wide, decode_arrays, raw_payloads, span_arrays
                    )
                    if record is None:
                        break
                    if f.tell() > size:
                        raise ValueError(
                            f"record {record['name']!r} runs past the end of the file"
                        )
                    roots.append(record)
            except (struct.error, zlib.error, ValueError) as error:
                raise ValueError(
                    f"Truncated or corrupt FBX: {path}: {error}"
                ) from error
        return cls(path, version, roots)

    @staticmethod
    def is_fbx(path: str) -> bool:
        """True when *path* starts with the binary-FBX magic."""
        try:
            with open(path, "rb") as f:
                return f.read(len(FBX_MAGIC)) == FBX_MAGIC
        except OSError:
            return False

    # ---- navigation -------------------------------------------------------

    def section(self, name: str) -> Optional[Dict[str, Any]]:
        """Top-level record *name* (``"Objects"``, ``"Connections"`` …)."""
        return self._by_name.get(name)

    def iter_objects(self) -> Iterator[Dict[str, Any]]:
        """Yield every child record of the ``Objects`` section."""
        objects = self.section("Objects")
        if objects:
            yield from objects["children"]

    # ---- census -----------------------------------------------------------

    def objects_census(self) -> Dict[str, int]:
        """``{record name: count}`` over the Objects section.

        Keys are FBX record names — ``Model``, ``Geometry``,
        ``AnimationStack``, ``AnimationLayer``, ``AnimationCurveNode``,
        ``AnimationCurve``, ``Deformer`` … — which is the census a
        deliverable check compares run over run.
        """
        return dict(Counter(record["name"] for record in self.iter_objects()))

    def object_names(self, kind: str) -> List[str]:
        """Display names of every Objects child whose record name is *kind*."""
        names: List[str] = []
        for record in self.iter_objects():
            if record["name"] != kind:
                continue
            for prop in record["props"]:
                name = self._display_name(prop)
                if name is not None:
                    names.append(name)
                    break
        return names

    def take_names(self) -> List[str]:
        """Animation take names — the ``AnimationStack`` display names."""
        return self.object_names("AnimationStack")

    def user_properties(self, name: str, kind: str = "Model") -> List[Any]:
        """Every value a *kind* object carries for the user property *name*.

        User properties live in each object's ``Properties70`` as ``P``
        records -- ``(name, type, label, flags, value...)`` -- which is where a
        DCC's custom attributes (the ``data_export`` channels among them) land.
        Values come back as stored: ``bytes`` for a string property, numbers
        for a numeric one, in file order.
        """
        found: List[Any] = []
        wanted = name.encode("utf-8")
        for record in self.iter_objects():
            if record["name"] != kind:
                continue
            for child in record["children"]:
                if child["name"] != "Properties70":
                    continue
                for prop in child["children"]:
                    props = prop["props"]
                    if len(props) >= 5 and props[0] == wanted:
                        found.append(props[4] if len(props) == 5 else props[4:])
        return found

    #: FBX ticks per second (``FbxTime``'s unit) in every file this reader
    #: has met -- Maya and Blender both write the 7.x default. A file whose
    #: ``OtherFlags/TCDefinition`` is not 127 pins another rate; the times
    #: below are returned as stored, in that file's ticks.
    TICKS_PER_SECOND = 46186158000

    def take_curves(self) -> Dict[str, Dict[Tuple[str, str, str], Tuple[Any, ...]]]:
        """What each take animates: ``{take: {(target, property, channel):
        (first_key, last_key, key_count)}}``, times in FBX ticks
        (:attr:`TICKS_PER_SECOND`).

        The census a take split is judged by. A channel the whole-timeline take
        animates but a shot's take does not is a node that plays its rest pose
        for that whole shot -- what Maya's ``FBXExportSplitAnimationIntoTakes``
        makes of every curve with no key inside a take, unless every curve is
        resampled. *target* is the animated object's name (a ``Model``, or the
        object a custom property lives on) -- ``name#uid`` where more than one
        animated object carries that name, *property* the property animated
        (``"Lcl Translation"``), *channel* the curve's component (``"d|X"``).
        A channel keyed on more than one layer is their union.

        Reads the ``KeyTime`` arrays: load with ``span_arrays=("KeyTime",)``
        (their extent alone -- what this needs) or ``decode_arrays``. A curve
        read without either reports ``(None, None, count)``.
        """
        kind: Dict[Any, str] = {}
        name: Dict[Any, str] = {}
        spans: Dict[Any, Tuple[Any, ...]] = {}
        for record in self.iter_objects():
            props = record["props"]
            if not props or not isinstance(props[0], int):
                continue
            kind[props[0]] = record["name"]
            name[props[0]] = (
                self._display_name(props[1]) if len(props) > 1 else None
            ) or ""
            if record["name"] != "AnimationCurve":
                continue
            times = next(
                (
                    child["props"][0]
                    for child in record["children"]
                    if child["name"] == "KeyTime" and child["props"]
                ),
                None,
            )
            if isinstance(times, list):
                spans[props[0]] = (
                    (times[0], times[-1], len(times)) if times else (None, None, 0)
                )
            elif isinstance(times, tuple) and len(times) == 5:  # SPAN
                spans[props[0]] = (times[3], times[4], times[2])
            elif isinstance(times, tuple) and len(times) == 3:  # ("ARRAY", t, n)
                spans[props[0]] = (None, None, times[2])
            else:
                spans[props[0]] = (None, None, 0)

        parents: Dict[Any, list] = {}
        for _kind, child, parent, prop in self.connections():
            parents.setdefault(child, []).append((parent, prop))

        found: List[Tuple[str, Any, str, str, Tuple[Any, ...]]] = []
        for curve, span in spans.items():
            for node, channel in parents.get(curve, ()):
                if kind.get(node) != "AnimationCurveNode":
                    continue
                links = parents.get(node, ())
                stacks = dict.fromkeys(
                    stack
                    for layer, _ in links
                    if kind.get(layer) == "AnimationLayer"
                    for stack, _ in parents.get(layer, ())
                    if kind.get(stack) == "AnimationStack"
                )
                targets = [(t, prop) for t, prop in links if prop is not None]
                for stack in stacks:
                    for target, prop in targets:
                        found.append((name[stack], target, prop, channel or "", span))

        # Keyed by name alone, one target's curve replaced another's: Maya
        # writes every NodeAttribute nameless, and a referenced module's nodes
        # beside the scene's own short names (478 repeats in one 12-shot
        # production FBX) -- and ``take_spans`` measured whichever was written
        # last. A uid is stable across the takes of one file, the comparison
        # the take gate makes.
        named = Counter(name.get(target, "") for target in {f[1] for f in found})
        takes: Dict[str, Dict[Tuple[str, str, str], Tuple[Any, ...]]] = {}
        for take, target, prop, channel, span in found:
            label = name.get(target, "")
            if named[label] > 1:
                label = f"{label}#{target}"
            curves = takes.setdefault(take, {})
            key = (label, prop, channel)
            curves[key] = (
                self._merged_span(curves[key], span) if key in curves else span
            )
        return takes

    def take_spans(self) -> Dict[str, Tuple[float, float]]:
        """Each take's ``(first, last)`` key in SECONDS, over every curve in it.

        What FBX2glTF sizes a take by: the key extent of all its curves -- a
        camera attribute's or a custom property's as much as a transform's --
        whatever span the take itself declares, and the first of them is what
        it rebases the clip onto (measured on 0.13.1, 2026-10-04: a focal
        length keyed at 3 opened a take whose transforms start at 10, and the
        take's declared 1-120 was ignored). Needs the key times
        (:meth:`take_curves`); a take none of whose curves carries them is
        left out rather than guessed.
        """
        spans: Dict[str, Tuple[float, float]] = {}
        for take, curves in self.take_curves().items():
            firsts = [c[0] for c in curves.values() if c[0] is not None]
            lasts = [c[1] for c in curves.values() if c[1] is not None]
            if firsts and lasts:
                spans[take] = (
                    min(firsts) / self.TICKS_PER_SECOND,
                    max(lasts) / self.TICKS_PER_SECOND,
                )
        return spans

    def connections(self) -> List[Tuple[str, Any, Any, Optional[str]]]:
        """Every ``C`` record as ``(kind, child_id, parent_id, property)``.

        Property-name is ``None`` for object-object (``"OO"``) links and the
        target attribute for object-property (``"OP"``) ones.
        """
        section = self.section("Connections")
        rows: List[Tuple[str, Any, Any, Optional[str]]] = []
        for record in (section or {}).get("children", []):
            props = record["props"]
            if len(props) < 3:
                continue
            kind = (
                props[0].decode("ascii", "replace")
                if isinstance(props[0], (bytes, bytearray))
                else str(props[0])
            )
            prop_name = None
            if len(props) > 3 and isinstance(props[3], (bytes, bytearray)):
                prop_name = props[3].decode("utf-8", "replace")
            rows.append((kind, props[1], props[2], prop_name))
        return rows


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
