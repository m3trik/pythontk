# !/usr/bin/python
# coding=utf-8
"""Live browser / WebXR preview: server, delivery strategy and bridge.

The transport half of the "push the current selection to a headset" loop,
kept as one subpackage because the code ships with the web runtime it serves:

* :mod:`.server` -- :class:`PreviewServer`: the loopback static-file server
  with a live ``/manifest.json``, viewer liveness, and the materialization of
  the web runtime -- the page, its kernel and the active features -- into the
  serve root. The facade over private mixins: ``_serve_root`` (the page, the
  viewer scripts, atomic writes), ``_sharing`` (the read-only guest listener and its tunnel),
  ``_page_outputs`` (recordings and stills), ``_scene_description`` (the
  published scene as data, at ``/scene.json``).
* :mod:`.routes` -- the served surface: the route names the page posts to
  (re-exported by :mod:`.server`) and the HTTP handler behind the loopback
  bind's gates.
* :mod:`.deliverer` -- :class:`PreviewDeliverer`: FBX -> GLB -> publish, the
  GLB built by the shared :class:`~pythontk.GlbPipeline` the Scene Exporters
  also run.
* :mod:`.bridge` -- :class:`PreviewBridge`: the hand-off bridge a DCC package
  binds its export mixin to (``mayatk.WebXrPreview`` / ``blendertk.WebXrPreview``).
* The web runtime, plain ES modules served as written: ``viewer.html`` (the
  page: markup), ``kernel/`` (the viewer -- one part per module, the
  published ``viewer`` API in ``kernel/api.js``, the generated record
  contract in ``kernel/records.js``) and ``features/`` (the optional modules
  it imports when the manifest names them, each a host-free model beside its
  three.js adapter where it has one). Type-checked by
  ``m3trik/scripts/check_js_types.py`` (``jsconfig.json`` at the repo root).

Every class reaches the root (``ptk.PreviewServer`` ...) through
``DEFAULT_INCLUDE``; nothing is imported here. The pipeline, channel by
channel: ``docs/webxr_preview.md``.
"""
