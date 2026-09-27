# !/usr/bin/python
# coding=utf-8
"""Live browser / WebXR preview: server, delivery strategy and bridge.

The transport half of the "push the current selection to a headset" loop,
kept as one subpackage because the code ships with the page and the scripts
it serves:

* :mod:`.server` -- :class:`PreviewServer`: the loopback static-file server
  with a live ``/manifest.json``, viewer liveness, and the materialization of
  ``viewer.html`` plus the active ``scripts/*.js`` into the serve root. The
  facade over private mixins: ``_serve_root`` (the page, the viewer scripts,
  atomic writes), ``_sharing`` (the read-only guest listener and its tunnel),
  ``_page_outputs`` (recordings and stills).
* :mod:`.routes` -- the served surface: the route names the page posts to
  (re-exported by :mod:`.server`) and the HTTP handler behind the loopback
  bind's gates.
* :mod:`.deliverer` -- :class:`PreviewDeliverer`: FBX -> GLB -> publish, the
  GLB built by the shared :class:`~pythontk.GlbPipeline` the Scene Exporters
  also run.
* :mod:`.bridge` -- :class:`PreviewBridge`: the hand-off bridge a DCC package
  binds its export mixin to (``mayatk.WebXrPreview`` / ``blendertk.WebXrPreview``).
* ``viewer.html`` -- the bundled three.js page; ``scripts/`` -- the optional
  ES modules it imports when the manifest names them.

Every class reaches the root (``ptk.PreviewServer`` ...) through
``DEFAULT_INCLUDE``; nothing is imported here. The pipeline, channel by
channel: ``docs/webxr_preview.md``.
"""
