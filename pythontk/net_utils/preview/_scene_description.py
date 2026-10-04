# !/usr/bin/python
# coding=utf-8
"""The published scene, described as data: ``GET /scene.json``.

The page renders the deliverable; this says how it is built and lit, for a
reader that cannot run the page -- a tool, a test, or someone's agent working
out why the same asset renders differently in their own engine. Nothing in it
is a re-description: it is the published GLB's own JSON chunk, verbatim (but
for an embedded ``data:`` payload, cut to its size), and
the lighting recipe the asset carries for exactly that reader
(``extras.scene_sidecar.handoff.rendering``) -- or, for an asset that carries
none, :attr:`MeshConvert.RENDERING_POLICY`, the same fallback the viewer
renders it with. A guest gets it too: it is the JSON of the asset a guest can
already download, read for them.

Sized for a reader with a context window. A production GLB's JSON runs to
megabytes (measured: 3.5 MB, two thirds of it accessors and buffer views),
while what decides how it LOOKS -- materials, textures, extras -- is tens of
kilobytes. So the overview inlines the smallest sections up to
:attr:`SCENE_INLINE_BYTES` and indexes every section, and a section is read a
page at a time within the same budget (``?section=<key>&start=<n>``).

One job of :class:`~pythontk.PreviewServer`, composed in ``server.py``; its
methods reach the rest of the server through ``self``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pythontk.net_utils.preview.routes import SCENE_PATH


class _SceneDescriptionMixin:
    """:meth:`describe_scene` and its link.

    A private part of :class:`~pythontk.PreviewServer`; call it through the facade.
    """

    #: Bytes of JSON the overview inlines, smallest section first, and the most
    #: one page of a section carries -- past one item, which goes whatever its
    #: size.
    SCENE_INLINE_BYTES = 128 * 1024

    #: The shape :meth:`describe_scene` answers in, for a reader that keys on it.
    SCENE_SCHEMA = "pythontk.preview.scene/1"

    #: What the overview tells a reader arriving with nothing else -- an agent
    #: handed the link.
    SCENE_ABOUT = (
        "How this preview's scene is built, as data. 'gltf' is the published "
        "asset's own glTF 2.0 JSON, verbatim, for each top-level section that "
        "fit; 'sections' indexes every section with its size, and its 'url' "
        "reads it a page at a time ('next' is the following page). "
        "'rendering' is the lighting recipe the viewer renders this asset with "
        "(three.js 'viewer.three'): tone mapping, environment, key light, and "
        "what a lightmapped material takes from each -- 'renderingSource' "
        "says whether the asset published it or the viewer's default applies. "
        "The asset itself downloads from 'asset'; the page is 'viewer.page'."
    )

    #: The three.js release the served page imports, as its import map pins it.
    _THREE_RELEASE = re.compile(r"three@(\d+\.\d+\.\d+)/build/")

    @property
    def scene_url(self) -> Optional[str]:
        """Where :meth:`describe_scene` is served on this machine, or ``None``
        before :meth:`start`. A share's own is ``share_info()["scene_url"]``."""
        url = self.url
        return None if url is None else f"{url}{SCENE_PATH}"

    def describe_scene(
        self, section: Optional[str] = None, start: int = 0
    ) -> Dict[str, Any]:
        """The published scene as data: the payload served at ``/scene.json``.

        Parameters:
            section: ``None`` for the overview. Else a top-level glTF key
                (``"nodes"``, ``"materials"`` ...): one page of it, from
                *start*, within :attr:`SCENE_INLINE_BYTES` -- ``next`` names
                the following page, ``None`` on the last. A key that is not a
                list comes back whole, as ``value``.
            start: The first item of the page.

        Returns:
            Always ``schema``, ``asset`` (the served name, or ``None`` before a
            publish) and ``version`` -- a page read across a publish shows it.
            The overview adds ``about``, ``title``, ``updated``, ``viewer``
            (``{"page", "three"}``, or ``None`` when no page is served),
            ``rendering`` and ``renderingSource`` (``"asset"`` or
            ``"viewer default"``), ``gltf`` (the inlined sections) and
            ``sections`` (``key -> {"items", "bytes", "inline", "url"}``;
            ``items`` is ``None`` for a key that is not a list). Before a
            publish, or for an asset that is not a GLB, a ``note`` says why
            there is nothing more.

        Raises:
            KeyError: *section* is not a key of the asset's JSON.
            ValueError: The asset is not a readable GLB, or *start* is negative.
            OSError: The asset could not be read.
        """
        if start < 0:
            raise ValueError(f"start is a whole number from 0, not {start}")
        with self._lock:
            asset, version, updated = self._asset, self._version, self._updated
        head = {"schema": self.SCENE_SCHEMA, "asset": asset, "version": version}
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
        gltf = self._without_payloads(GlbEdit.read(str(self.root / asset)).gltf)
        if section is not None:
            return {**head, **self._scene_section(gltf, section, start)}
        sizes = {key: self._json_size(value) for key, value in gltf.items()}
        inline, budget = set(), self.SCENE_INLINE_BYTES
        for key in sorted(gltf, key=sizes.get):
            if sizes[key] > budget:
                break  # sorted by size: nothing after it fits either
            inline.add(key)
            budget -= sizes[key]
        rendering, source = self._scene_rendering(gltf)
        return {
            **head,
            "about": self.SCENE_ABOUT,
            "title": self.title,
            "updated": updated,
            "viewer": self._scene_viewer(),
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

    def _scene_section(
        self, gltf: Dict[str, Any], section: str, start: int
    ) -> Dict[str, Any]:
        """One page of *section* (:meth:`describe_scene`), without the head."""
        if section not in gltf:
            raise KeyError(section)
        value = gltf[section]
        if not isinstance(value, list):
            return {"section": section, "value": value}
        items: List[Any] = []
        budget = self.SCENE_INLINE_BYTES
        end = start
        while end < len(value):
            size = self._json_size(value[end])
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
    def _without_payloads(gltf: Dict[str, Any]) -> Dict[str, Any]:
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
    def _json_size(value: Any) -> int:
        """Bytes *value* takes as the route sends it (``_send_json``'s
        ``json.dumps``), so a budget holds for what the reader receives."""
        return len(json.dumps(value))

    @staticmethod
    def _scene_rendering(gltf: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """The recipe the viewer lights *gltf* with, and where it came from.

        The asset's ``scene_sidecar`` is looked for where the viewer's
        ``readExtras`` looks: the first scene's ``extras``, then the root's,
        either holding an object or a JSON string of one.
        """
        scenes = gltf.get("scenes")
        first = scenes[0] if isinstance(scenes, list) and scenes else None
        for owner in (first, gltf):
            holder = owner.get("extras") if isinstance(owner, dict) else None
            raw = holder.get("scene_sidecar") if isinstance(holder, dict) else None
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except ValueError:
                    continue
            handoff = raw.get("handoff") if isinstance(raw, dict) else None
            rendering = handoff.get("rendering") if isinstance(handoff, dict) else None
            if isinstance(rendering, dict):
                return rendering, "asset"
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        return MeshConvert.rendering_policy(), "viewer default"

    def _scene_viewer(self) -> Optional[Dict[str, Any]]:
        """The served page and the three.js release it imports -- read off
        the page actually served, which a caller-owned root may have edited."""
        page = self.root / "index.html"
        try:
            text = page.read_text(encoding="utf-8")
        except OSError:
            return None
        match = self._THREE_RELEASE.search(text)
        return {"page": "index.html", "three": match.group(1) if match else None}
