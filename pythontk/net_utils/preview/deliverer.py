# !/usr/bin/python
# coding=utf-8
"""FBX -> GLB -> publish: the hand-off strategy behind every live preview.

:class:`PreviewDeliverer` is the :class:`pythontk.Deliverer` a preview bridge
mounts. It builds the GLB through :class:`pythontk.GlbPipeline` -- the SAME
build the Scene Exporters run for their GLB deliverable (downsize, convert
with the sidecar and the lightmaps, texture pass) -- and publishes the result
to the :class:`~pythontk.PreviewServer` it owns. Nothing about the GLB's
content is decided here: the preview shows what the export ships because the
two are one chain, and the deliverer's own choices stop at the delivery
container (the viewer's texture format) and where the scratch files live.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pythontk.core_utils.app_handoff import Deliverer, HandoffRequest, Payload
from pythontk.net_utils.preview.server import PreviewServer, _mesh_convert


class PreviewDeliverer(Deliverer):
    """Hand-off strategy: build the GLB and publish it to the preview page.

    The delivery half of a preview bridge, so a DCC only has to supply the two
    hooks it already has (``_resolve_objects`` / ``_produce``, both provided by
    the Maya and Blender export mixins) to gain a live preview:

        >>> class WebXrPreview(MayaExportMixin, ptk.PreviewBridge):
        ...     deliverer = ptk.PreviewDeliverer(title="Maya")

    Unlike the script-launch deliverers this sits beside, there is no
    application to discover or launch and no script to render: the "target app"
    is a browser the user already has open, so delivery is a format conversion
    plus a version bump. Keeping it a :class:`Deliverer` rather than a bespoke
    per-DCC chain is what stops the FBX -> GLB -> publish sequence from being
    written twice, once in each engine.

    The server is created on first delivery and reused, because the whole point
    is that a page left open in a headset keeps receiving pushes.

    Parameters:
        server: An existing :class:`PreviewServer`; one is created on demand
            otherwise.
        open_browser: ``"auto"`` (default) opens a tab only when no page is
            currently watching the server -- so the first push opens one, a
            push while the page is still open does not (it picks the new
            version up itself, and a fresh tab every time would both pile up
            and steal focus from the DCC), and a push after the tab was closed
            opens one again. ``True`` always opens, ``False`` never does.
        title: Label shown in the viewer, when creating the server.
        texture_format: Container the texture pass re-encodes to --
            ``"WEBP"`` (default; transport size) or ``"KTX2"`` (GPU-resident
            Basis compression, the headset-memory win; requires the ``toktx``
            encoder -- see :meth:`MeshConvert.optimize_glb_textures`). This is
            the *default*; a single push overrides it per request --
            ``bridge.push(texture_format="KTX2")`` -- so one high-fidelity push
            costs the next quick-iteration one nothing.
        scripts: Viewer scripts to activate on every push (see
            :attr:`PreviewServer.SCRIPTS`). ``None`` -- the default -- leaves
            whatever the server already has alone, so a script registered
            directly on a long-lived server survives; a list replaces the set.
    """

    def __init__(
        self,
        server: Optional[PreviewServer] = None,
        open_browser: Union[bool, str] = "auto",
        title: str = "Preview",
        texture_format: str = "WEBP",
        scripts: Optional[Union[Dict[str, Any], List[str], tuple]] = None,
    ):
        self.server = server
        self.open_browser = open_browser
        self.title = title
        self.texture_format = texture_format
        self.scripts = scripts

    def ensure_server(self) -> PreviewServer:
        """The bridge's server, started, creating it on first use."""
        if self.server is None:
            self.server = PreviewServer(title=self.title)
        return self.server.start()

    def publish(
        self,
        glb: Union[str, Path],
        move: bool = False,
        open_browser: Union[bool, str, None] = None,
        scripts: Optional[Union[Dict[str, Any], List[str], tuple]] = None,
    ) -> Dict[str, Any]:
        """Put *glb* on the server and report what the viewer now sees.

        The tail every delivery shares -- activate the script set, bump the
        version, decide whether a tab needs opening -- factored out of
        :meth:`deliver` so the OTHER way an asset reaches the page
        (:meth:`PreviewBridge.publish_file`, a GLB already on disk) cannot
        answer those three questions differently. A second copy of this is
        exactly how a push and a publish end up disagreeing about whether an
        unticked script box turns a script off.

        Parameters:
            glb: The file to serve.
            move: Move rather than copy. True only for an artifact the caller
                minted -- a file the *user* chose is never moved out from
                under them.
            open_browser: ``True`` / ``False`` / ``"auto"``; ``None`` falls
                back to the deliverer's own setting. ``False`` is meaningful,
                which is why the fallback tests for ``None`` rather than for
                falsiness.
            scripts: The viewer-script set for this publish. ``None`` means
                "leave whatever the server has alone" -- after falling back to
                the deliverer's own default -- so a script registered directly
                on a long-lived server survives a publish that says nothing;
                ``[]`` clears them.

        Returns:
            ``{"url", "version", "asset", "opened_browser"}``.
        """
        server = self.ensure_server()
        if scripts is None:
            scripts = self.scripts
        if scripts is not None:
            server.set_scripts(scripts)

        version = server.publish(glb, move=move)

        # Asked after publishing, so the freshest possible poll counts.
        if open_browser is None:
            open_browser = self.open_browser
        should_open = open_browser is True or (
            open_browser == "auto" and not server.has_viewer()
        )
        # The decision and the outcome are reported separately on purpose:
        # `open_in_browser` says whether a browser actually launched, and
        # returning the decision would claim a tab exists on the one machine
        # where none does.
        opened = should_open and server.open_in_browser()

        return {
            "url": server.url,
            "version": version,
            "asset": server.manifest()["asset"],
            "opened_browser": opened,
        }

    def deliver(
        self, bridge, payload: Payload, request: HandoffRequest
    ) -> Optional[Dict[str, Any]]:
        # Imported here rather than at module scope: the converter pulls in the
        # managed-binary installer, which no other PreviewServer user needs.
        MeshConvert = _mesh_convert()
        from pythontk.file_utils.mesh_convert.glb_pipeline import GlbPipeline

        if not payload.primary:
            bridge.logger.error("Preview delivery got no exported file to convert.")
            return None

        # Eagerly, though `publish` below ensures it too: binding the port is
        # the one failure here that has nothing to do with the model, and it
        # should not arrive after a multi-minute conversion has been paid for.
        self.ensure_server()

        # Request-scoped exactly like `open_browser` below. A deliverer is
        # bound once per bridge *class*, so a format written onto the instance
        # for one high-fidelity push would stick process-wide for every bridge
        # in the session: each later quick-iteration push would then require
        # `toktx` and pay the Basis encode, with no way to opt back out for a
        # single push. The instance attribute stays the default. Falsy falls
        # back rather than overriding (unlike `open_browser`, where False is a
        # meaningful value) -- an empty format is "unspecified", not a request
        # to hand the optimizer nothing, and it lets the caller-facing knobs
        # pass their `None` default straight through. The trailing WEBP is
        # load-bearing: a falsy INSTANCE default would otherwise hand the
        # optimizer None, which raises inside it.
        texture_format = request.get("texture_format") or self.texture_format or "WEBP"

        # KTX2 is the one exception to "the pipeline reports its own failure":
        # docs/webxr_preview.md promises the push raises with the install URL
        # when `toktx` is missing, never silently ships WebP instead -- and the
        # optimizer only reaches its own `resolve_ktx2_encoder(required=True)`
        # once it hits a KTX2 image, minutes into a conversion it would then
        # abandon. Checked BEFORE the build, so the fix-shaped error arrives
        # before the session pays for it.
        if texture_format.upper() == "KTX2":
            from pythontk.img_utils._img_utils import ImgUtils

            ImgUtils.resolve_ktx2_encoder(required=True)

        # Allocate the GLB through the bridge's own payload artifacts rather
        # than deriving a path from the FBX: that keeps it inside the prefix
        # namespace every other bridge artifact joins, so the age-gated sweep
        # reclaims it after a hard DCC crash -- the case no `finally` survives.
        glb = bridge._make_payload_path(extension=".glb")
        extras = payload.extras or {}
        # The host's live texture folders (:meth:`PreviewBridge.lightmap_search_dirs`),
        # read through ``getattr`` because a deliverer is pluggable: mounted on
        # a plain :class:`HandoffBridge` the hook is absent, and an attribute
        # error here would cost the push rather than one lightmap lookup.
        hook = getattr(bridge, "lightmap_search_dirs", None)
        lightmap_dirs = hook() if callable(hook) else ()

        try:
            built = GlbPipeline.build(
                payload.primary,
                dst=glb,
                # Applied INSIDE the conversion, ahead of its fade pass: a
                # material clone copies its source as it stands, so a repair
                # made afterwards would land only on the original and every
                # fade clone would keep the converter's packing (measured).
                sidecar=extras.get("scene_sidecar"),
                lightmap_dirs=lightmap_dirs,
                # The shared web-delivery policy, named rather than inherited:
                # the resolution the preview approves used to be set by a
                # signature default two packages away, where the exporters
                # could not see it to agree with it (8.71 MB here against
                # 280.13 MB from the exporter, same scene, same session).
                # `ktx2_fallback=False`: this GLB is streamed to the viewer
                # page, never re-imported, so the core-readable fallback twins
                # would spend the very bytes the texture pass exists to reclaim
                # (the bundled page wires KTX2Loader, so basisu-only is safe). A
                # property of the consumer, so it stays out of the shared policy
                # -- an exporter's deliverable must stay importable.
                texture_params={
                    **MeshConvert.web_delivery_texture_params(
                        image_format=texture_format
                    ),
                    "ktx2_fallback": False,
                },
                downsize=bool(request.params.get("EMBED_TEXTURES", True)),
                # Scratch through the bridge's own payload store (swept after a
                # crash), and the superseded payload released as soon as the
                # downsized copy exists -- peak footprint one FBX, not two.
                # `_release_payload` refuses what the bridge did not mint, so a
                # producer's durable file is never deleted.
                scratch_path=lambda extension: bridge._make_payload_path(
                    extension=extension
                ),
                release_source=bridge._release_payload,
                logger=bridge.logger,
            )
        except (OSError, RuntimeError, ValueError) as error:
            # Nothing consumed the payload, so it stays on disk as evidence.
            bridge.logger.error("Preview conversion to GLB failed: %s", error)
            return None

        # The FBX has been fully consumed -- the GLB exists and nothing reads
        # the payload again. Released HERE rather than left to the store's
        # age-gated sweep because this bridge is the one shape that store
        # cannot serve: its ``detached`` policy is right for a hand-off whose
        # target app reads the file AFTER we return (no completion signal, so
        # nothing may delete), and wrong for a blocking round trip that
        # converts and publishes inside one call. Measured on a production
        # assembly: 324 MB per push, 3.1 GB of them waiting out ``max_age_days``
        # in the system temp dir.
        if bridge._release_payload(built["src"]):
            bridge.logger.debug("Released the consumed FBX payload: %s", built["src"])

        sidecar = extras.get("scene_sidecar")
        if not (sidecar or {}).get("sections"):
            # Distinguish "switched off" from "on, but the scene had nothing":
            # both produce a bare-FBX preview, and only one is a surprise.
            # Key present means the producer ran and found nothing; absent
            # means it was never asked (the producer attaches an envelope --
            # possibly with empty sections -- whenever the param is on).
            bridge.logger.info(
                "No scene sidecar %s.",
                "produced for this export"
                if "scene_sidecar" in extras
                else "requested",
            )

        published = self.publish(
            glb,
            # The GLB is this bridge's own scratch artifact and nothing reads
            # it again once the server owns a copy.
            move=True,
            # Request-scoped like `texture_format`. `.get`'s default is not
            # enough: `push()` names both knobs explicitly, so the keys are
            # PRESENT and None whenever the caller said nothing -- read with a
            # default here, the deliverer's own settings could never apply.
            # `publish` owns that fallback for both entry points.
            open_browser=request.get("open_browser", self.open_browser),
            scripts=request.get("scripts"),
        )

        return {
            **published,
            # The per-section outcome the conversion recorded (what the panel
            # summarises); ``{}`` when no envelope was offered.
            "sidecar": built.get("sidecar") or {},
            # ``{"expected", "bound", "unbound": [names], "out_of_scope"}``
            # from the conversion's lightmap bind, or ``None`` when it never
            # ran. ``expected`` is what the manifest named of THIS file; an
            # unbound object previews UNLIT, and only this says so -- the
            # viewer renders "no bake", "bake not found" and "bake bound" as
            # three shades of the same dark room.
            "lightmaps": built.get("lightmaps"),
            # Whether one was *offered* -- the caller cannot infer it from an
            # empty summary, which also means "switched off".
            "sidecar_requested": "scene_sidecar" in extras,
        }
