# !/usr/bin/python
# coding=utf-8
"""FBX -> GLB -> publish: the hand-off strategy behind every live preview.

:class:`PreviewDeliverer` is the :class:`pythontk.Deliverer` a preview bridge
mounts. It builds the GLB through :class:`pythontk.GlbPipeline` -- the SAME
build the Scene Exporters run for their GLB deliverable (downsize, convert
with the sidecar and the lightmaps, texture pass) -- and publishes the result
to the :class:`~pythontk.PreviewServer` it owns. Nothing about the GLB's
content is decided here: the preview shows what the export ships because the
two are one chain, and its settings are the Scene Exporter's own GLB rows
(:attr:`pythontk.ExportProfile.GLB_ROWS`) -- the texture pass and the lighting
recipe -- resolved by the methods the exporters call
(:meth:`pythontk.ExportRun.glb_texture_params`, :attr:`pythontk.ExportRun.rendering`).
What the deliverer decides for itself stops at where the scratch files live.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from pythontk.core_utils.app_handoff import Deliverer, HandoffRequest, Payload
from pythontk.core_utils.deprecation import Deprecation
from pythontk.core_utils.export_profile import ExportRun
from pythontk.net_utils.preview.server import PreviewServer


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
        glb_options: The Scene Exporter's GLB rows for every push --
            ``{row key: combo value}`` keyed by :attr:`ExportProfile.GLB_ROWS`
            (the texture rows ``texture_file_type``, ``optimize_textures``,
            ``secondary_max_size``, ``uastc_rdo``, and ``baked_reflections``),
            each value exactly as that row's combo holds it. An absent row is
            the Scene Exporter's default, so ``None`` -- the default -- builds
            the GLB an export with untouched rows ships (the web container,
            WebP, every map at its own resolution, the lighting recipe as
            declared). These are *defaults*; a single push overrides them row
            by row -- ``bridge.push(glb_options={"optimize_textures": 4096})``
            -- so one high-fidelity push costs the next quick-iteration one
            nothing. KTX2 needs the ``toktx`` encoder (see
            :meth:`MeshConvert.optimize_glb_textures`).
        scripts: Viewer scripts to activate on every push (see
            :attr:`PreviewServer.SCRIPTS`). ``None`` -- the default -- leaves
            whatever the server already has alone, so a script registered
            directly on a long-lived server survives; a list replaces the set.
    """

    @Deprecation.parameter(
        "texture_format",
        remove_in="0.12.0",
        new="glb_options",
        transform=lambda value: {"texture_file_type": value},
        reason="The preview now takes every Scene Exporter GLB row, the "
        "container being Texture File Type.",
    )
    def __init__(
        self,
        server: Optional[PreviewServer] = None,
        open_browser: Union[bool, str] = "auto",
        title: str = "Preview",
        glb_options: Optional[Mapping[str, Any]] = None,
        scripts: Optional[Union[Dict[str, Any], List[str], tuple]] = None,
    ):
        self.server = server
        self.open_browser = open_browser
        self.title = title
        self.glb_options: Dict[str, Any] = self._glb_rows(glb_options)
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

    #: Where :meth:`preflight` leaves the resolved texture pass for
    #: :meth:`deliver` -- on the request, which lives exactly one push.
    _TEXTURE_PARAMS_KEY = "_texture_params"
    #: Where :meth:`preflight` leaves the push's choices over the lighting
    #: recipe (:attr:`ExportRun.rendering`) for the host, whose envelope
    #: publishes them (:meth:`PreviewBridge._attach_sidecar` reads it).
    RENDERING_KEY = "rendering"

    @staticmethod
    def _glb_rows(value: Any) -> Dict[str, Any]:
        """*value* as a copy of a row mapping; a bare container is refused, named.

        ``glb_options`` took ``texture_format``'s place in both signatures,
        so a caller that passed the container POSITIONALLY now hands a string
        to a mapping. Said here, with the fix, rather than as a dict-unpacking
        error from inside the push.
        """
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(
                "glb_options takes the Scene Exporter's GLB rows as a mapping "
                f"(ExportProfile.GLB_ROWS), not {value!r}; a container alone "
                f"is {{'texture_file_type': {value!r}}}."
            )
        return dict(value)

    def _glb_run(
        self, request: Optional[HandoffRequest], logger: Any
    ) -> Optional[ExportRun]:
        """The GLB rows *request* builds with, as a run; ``None`` when one
        cannot be honoured (an unknown container -- the export refuses it
        too), with the reason logged.

        The deliverer's :attr:`glb_options` with the request's own laid over
        them row by row, parsed exactly as a Scene Exporter parses the same
        rows (:meth:`ExportRun.for_glb`). Request-scoped like ``open_browser``:
        a deliverer is bound once per bridge *class*, so a row written onto the
        instance for one high-fidelity push would stick process-wide -- every
        later quick push would require ``toktx`` and pay the Basis encode, with
        no way to opt back out for one push.

        Raises:
            TypeError: a request's rows are not a mapping.
        """
        requested = request.get("glb_options") if request is not None else None
        run, notes = ExportRun.for_glb(
            {**self.glb_options, **self._glb_rows(requested)}
        )
        for level, message in notes:
            getattr(logger, level)(message)
        if any(level == "error" for level, _message in notes):
            return None
        return run

    @staticmethod
    def _texture_params(run: ExportRun, logger: Any) -> Dict[str, Any]:
        """*run*'s texture pass (:meth:`ExportRun.glb_texture_params`), with its
        encoder settled.

        KTX2 is settled here, and it is the one exception to "the pipeline
        reports its own failure": docs/webxr_preview.md promises the push
        raises with the install URL when ``toktx`` is missing, never silently
        ships WebP instead -- and the optimizer only reaches its own
        ``resolve_ktx2_encoder(required=True)`` once it hits a KTX2 image,
        minutes into a conversion it would then abandon.

        Raises:
            FileNotFoundError: KTX2 with no ``toktx`` to encode it.
        """
        params = run.glb_texture_params(logger=logger)
        if params["image_format"] == "KTX2":
            from pythontk.img_utils._img_utils import ImgUtils

            ImgUtils.resolve_ktx2_encoder(required=True)
        return params

    def preflight(self, bridge, request: HandoffRequest) -> bool:
        """Resolve this push's GLB rows BEFORE the export is paid for.

        A row the export would refuse, or a KTX2 push with no encoder, costs
        nothing but the check here -- found in :meth:`deliver` it arrived after
        the host had already written the FBX. The results ride the request:
        the texture pass to :meth:`deliver`, and the lighting recipe's
        overrides (:attr:`RENDERING_KEY`) to the host, whose envelope publishes
        them -- so the rows are resolved (and their notes logged) once.
        """
        run = self._glb_run(request, bridge.logger)
        if run is None:
            return False
        request.extras[self._TEXTURE_PARAMS_KEY] = self._texture_params(
            run, bridge.logger
        )
        request.extras[self.RENDERING_KEY] = run.rendering
        return True

    def deliver(
        self, bridge, payload: Payload, request: HandoffRequest
    ) -> Optional[Dict[str, Any]]:
        # Imported here rather than at module scope: the pipeline pulls in the
        # managed-binary installer, which no other PreviewServer user needs.
        from pythontk.file_utils.mesh_convert.glb_pipeline import GlbPipeline

        if not payload.primary:
            bridge.logger.error("Preview delivery got no exported file to convert.")
            return None

        # Eagerly, though `publish` below ensures it too: binding the port is
        # the one failure here that has nothing to do with the model, and it
        # should not arrive after a multi-minute conversion has been paid for.
        self.ensure_server()

        # Resolved by `preflight` on the skeleton's path; a caller driving the
        # deliverer directly skipped it, so it resolves here instead.
        texture_params = request.get(self._TEXTURE_PARAMS_KEY)
        if texture_params is None:
            run = self._glb_run(request, bridge.logger)
            if run is None:
                return None
            texture_params = self._texture_params(run, bridge.logger)

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
                # Request-scoped: a preview's overlay describes THIS push (an
                # effect at the panel's current settings), never the next one.
                data_export=request.get("data_export"),
                lightmap_dirs=lightmap_dirs,
                # The Scene Exporter's texture rows, resolved by its own method:
                # the preview once named a container and inherited the web
                # ceiling, so every push was cut to 2048 px whatever the export
                # was set to. Untouched, the rows resize nothing (OFF).
                texture_params=texture_params,
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
                # Request-scoped like the two knobs above, and for the extra
                # reason that a callback is a live UI object: parked on the
                # deliverer -- a CLASS attribute, shared by every instance of
                # the bridge -- it would outlive the panel that supplied it and
                # keep being called into a closed window.
                progress=request.get("progress"),
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
            # Request-scoped like `glb_options`. `.get`'s default is not
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
            # The in-band channels the request's ``data_export`` overlay
            # replaced in the published GLB, ``[]`` when none did. A caller
            # previewing an effect the scene does not carry checks its channel
            # is HERE before telling anyone the page shows it: a bridge that
            # never learned the knob (an older pythontk still imported in the
            # session) sweeps it into the export bag and publishes the scene
            # as it stands, with no error anywhere -- measured 2026-09-13 as a
            # preview that looked the same whatever the panel was set to.
            "data_export": list(built.get("data_export") or []),
        }
