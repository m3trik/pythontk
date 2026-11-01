# !/usr/bin/python
# coding=utf-8
"""The published scene, described as data: ``GET /scene.json``.

The page renders the deliverable; this says how it is built and lit, for a
reader that cannot run the page -- a tool, a test, or someone's agent working
out why the same asset renders differently in their own engine. Nothing in it
is a re-description: it is the published GLB's own JSON chunk, verbatim (but
for an embedded ``data:`` payload, cut to its size), and
the lighting recipe the asset carries for exactly that reader
(``extras.handoff.rendering``) -- or, for an asset that carries
none, :attr:`HandoffRecord.RENDERING_POLICY`, the same fallback the viewer
renders it with. A guest gets it too: it is the JSON of the asset a guest can
already download, read for them.

Sized for a reader with a context window. A production GLB's JSON runs to
megabytes (measured: 3.5 MB, two thirds of it accessors and buffer views),
while what decides how it LOOKS -- materials, textures, extras -- is tens of
kilobytes. So the overview inlines the smallest sections up to
:attr:`SCENE_INLINE_BYTES` and indexes every section, and a section is read a
page at a time within the same budget (``?section=<key>&start=<n>``).

A stateless collaborator of :class:`~pythontk.PreviewServer`
(:class:`_SceneDescription`): it reads what is published, and its budgets
(:attr:`PreviewServer.SCENE_INLINE_BYTES` ...), through the server at call
time, so an instance override of one reaches it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from pythontk.net_utils.preview.routes import SCENE_PATH

if TYPE_CHECKING:
    from pythontk.net_utils.preview.server import PreviewServer


class _SceneDescription:
    """:meth:`PreviewServer.describe_scene` and its link.

    A private collaborator of :class:`~pythontk.PreviewServer`; call it through
    the facade, whose methods carry the documentation.
    """

    def __init__(self, server: "PreviewServer"):
        self._server = server

    def scene_url(self) -> Optional[str]:
        url = self._server.url
        return None if url is None else f"{url}{SCENE_PATH}"

    def describe(self, section: Optional[str] = None, start: int = 0) -> Dict[str, Any]:
        """The payload :meth:`PreviewServer.describe_scene` documents."""
        server = self._server
        if start < 0:
            raise ValueError(f"start is a whole number from 0, not {start}")
        with server._lock:
            asset, version, updated = server._asset, server._version, server._updated
            parts = [dict(part) for part in server._parts]
        head = {"schema": server.SCENE_SCHEMA, "asset": asset, "version": version}
        if len(parts) > 1:
            # The scene is several files and this describes the first: the
            # rest are named, each downloadable by its own `url`.
            head["parts"] = parts
        if asset is None:
            if section is not None:
                raise KeyError(section)
            return {**head, "note": "Nothing is published yet."}
        if Path(asset).suffix.lower() != ".glb":
            if section is not None:
                raise KeyError(section)
            return {**head, "note": f"{asset} is not a GLB; it is not described."}
        from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit

        # Read per request rather than held: the parsed JSON of a production
        # GLB is tens of megabytes of Python objects, for a server that lives
        # as long as the DCC; the read is the JSON chunk only.
        gltf = self.without_payloads(GlbEdit.read(str(server.root / asset)).gltf)
        if section is not None:
            return {**head, **self.section(gltf, section, start)}
        sizes = {key: self.json_size(value) for key, value in gltf.items()}
        inline, budget = set(), server.SCENE_INLINE_BYTES
        for key in sorted(gltf, key=sizes.get):
            if sizes[key] > budget:
                break  # sorted by size: nothing after it fits either
            inline.add(key)
            budget -= sizes[key]
        rendering, source = self.rendering(gltf)
        return {
            **head,
            "about": server.SCENE_ABOUT,
            "title": server.title,
            "updated": updated,
            "viewer": self.viewer(),
            "rendering": rendering,
            "renderingSource": source,
            "gltf": {key: value for key, value in gltf.items() if key in inline},
            "sections": {
                key: {
                    "items": len(value) if isinstance(value, list) else None,
                    "bytes": sizes[key],
                    "inline": key in inline,
                    "url": f"{SCENE_PATH}?section={key}",
                }
                for key, value in gltf.items()
            },
        }

    def section(self, gltf: Dict[str, Any], section: str, start: int) -> Dict[str, Any]:
        """One page of *section* (:meth:`describe`), without the head."""
        if section not in gltf:
            raise KeyError(section)
        value = gltf[section]
        if not isinstance(value, list):
            return {"section": section, "value": value}
        items: List[Any] = []
        budget = self._server.SCENE_INLINE_BYTES
        end = start
        while end < len(value):
            size = self.json_size(value[end])
            if items and size > budget:
                break
            items.append(value[end])
            budget -= size
            end += 1
        return {
            "section": section,
            "total": len(value),
            "start": start,
            "items": items,
            "next": (
                f"{SCENE_PATH}?section={section}&start={end}"
                if end < len(value)
                else None
            ),
        }

    @staticmethod
    def without_payloads(gltf: Dict[str, Any]) -> Dict[str, Any]:
        """*gltf* with each base64 ``data:`` URI -- the two places glTF lets
        one ride, ``buffers`` and ``images`` -- cut to its media type and size.

        Bytes, not structure: one embedded texture is megabytes of base64 in a
        single item, the one thing a page's budget cannot split. Edits the
        dict in place; it is this request's own parse.
        """
        for key in ("buffers", "images"):
            for entry in gltf.get(key) or ():
                uri = entry.get("uri") if isinstance(entry, dict) else None
                if isinstance(uri, str) and uri.startswith("data:"):
                    media, _, data = uri.partition(",")
                    entry["uri"] = (
                        f"{media},<{len(data)} base64 characters, not shown: "
                        "read the asset itself>"
                    )
        return gltf

    @staticmethod
    def json_size(value: Any) -> int:
        """Bytes *value* takes as the route sends it (``_send_json``'s
        ``json.dumps``), so a budget holds for what the reader receives."""
        return len(json.dumps(value))

    @staticmethod
    def rendering(gltf: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """The recipe the viewer lights *gltf* with, and where it came from:
        the asset's handoff (``extras.handoff``, the current schema), else
        the policy."""
        from pythontk.core_utils.engines.scene_export.handoff_record import (
            HandoffRecord,
        )

        extras = gltf.get("extras")
        handoff = extras.get(HandoffRecord.KEY) if isinstance(extras, dict) else None
        if (
            isinstance(handoff, dict)
            and handoff.get("version") == HandoffRecord.VERSION
        ):
            rendering = handoff.get("rendering")
            if isinstance(rendering, dict):
                return rendering, "asset"
        return HandoffRecord.rendering_policy(), "viewer default"

    def viewer(self) -> Optional[Dict[str, Any]]:
        """The served page and the three.js release it imports -- read off
        the page actually served, which a caller-owned root may have edited."""
        page = self._server.root / "index.html"
        try:
            text = page.read_text(encoding="utf-8")
        except OSError:
            return None
        match = self._server._THREE_RELEASE.search(text)
        return {"page": "index.html", "three": match.group(1) if match else None}
