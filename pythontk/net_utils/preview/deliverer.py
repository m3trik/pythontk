# !/usr/bin/python
# coding=utf-8
"""FBX -> GLB -> publish: the hand-off strategy behind every live preview.

:class:`PreviewDeliverer` is the :class:`pythontk.Deliverer` a preview bridge
mounts. It converts the produced FBX through :class:`pythontk.MeshConvert`,
runs an ordered registry of post-conversion passes -- :attr:`EDIT_PASSES
<PreviewDeliverer.EDIT_PASSES>` against one open GLB edit session,
:attr:`FILE_PASSES <PreviewDeliverer.FILE_PASSES>` against the closed file --
and publishes the result to the :class:`~pythontk.PreviewServer` it owns.
:class:`PreviewPassContext` is the one object every pass reads and reports
into, so a new pass is an entry plus a method rather than a wider signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from pythontk.core_utils.app_handoff import Deliverer, HandoffRequest, Payload
from pythontk.net_utils.preview.server import PreviewServer, _mesh_convert


@dataclass
class PreviewPassContext:
    """What a post-conversion preview pass reads, and reports into.

    One object rather than a widening parameter list, because the passes are a
    *registry* (:attr:`PreviewDeliverer.EDIT_PASSES` /
    :attr:`PreviewDeliverer.FILE_PASSES`) and a registry's entries have to share
    one signature -- a pass added later cannot make the runner grow an argument.

    :attr:`edit` is the open GLB edit session, and is only set while the edit
    passes run: the file passes operate on the closed file, and holding a stale
    handle there is the mistake the ``None`` makes loud instead of silent. A
    payload pass runs before the conversion, so for it :attr:`glb` is the path
    the converter WILL write and :attr:`payload` is what it may rewrite.
    """

    bridge: Any
    glb: Path
    payload: Payload
    request: HandoffRequest
    texture_format: str
    edit: Any = None
    #: Anything a pass wants the deliverer's result dict to carry.
    results: Dict[str, Any] = field(default_factory=dict)

    @property
    def logger(self):
        """The bridge's logger -- every pass reports through the push's own sink."""
        return self.bridge.logger

    @property
    def sidecar(self) -> Optional[Dict[str, Any]]:
        """The scene-sidecar envelope the producer attached, if any."""
        return (self.payload.extras or {}).get("scene_sidecar")

    @property
    def lightmap_search_dirs(self) -> Sequence[str]:
        """The host's live texture folders (:meth:`PreviewBridge.lightmap_search_dirs`).

        Read through ``getattr`` because a deliverer is pluggable — this one is
        paired with :class:`PreviewBridge` in every shipped bridge, but nothing
        stops it being mounted on a plain :class:`HandoffBridge`, and there the
        attribute error would be swallowed by the runner's per-pass guard and
        skip the lightmaps entirely: the exact silent-unlit outcome the hook
        exists to prevent.
        """
        hook = getattr(self.bridge, "lightmap_search_dirs", None)
        return hook() if callable(hook) else ()


class PreviewDeliverer(Deliverer):
    """Hand-off strategy: convert the produced FBX to GLB and publish it.

    The delivery half of a preview bridge, so a DCC only has to supply the two
    hooks it already has (``_resolve_objects`` / ``_produce``, both provided by
    the Maya and Blender export mixins) to gain a live preview:

        >>> class WebXrPreview(MayaExportMixin, ptk.HandoffBridge):
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
        texture_format: Container the web-delivery pass re-encodes textures to
            -- ``"WEBP"`` (default; transport size) or ``"KTX2"`` (GPU-resident
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

    #: Passes run on the produced **payload** -- the FBX -- before the
    #: conversion, in order, ``name -> method on this class``. A pass here may
    #: replace ``context.payload.primary`` with a rewritten scratch file; the
    #: conversion reads the payload after them.
    PAYLOAD_PASSES: Dict[str, str] = {
        "downsize_textures": "_pass_downsize_textures",
    }

    #: Post-conversion passes run against **one** open GLB edit session, in
    #: order, ``name -> method on this class``.
    #:
    #: The order is load-bearing and each entry says why in its own docstring;
    #: what the registry buys is that a new pass (a channel repair, a geometry
    #: rewrite) is an *entry plus a method* rather than another limb grafted
    #: onto one long procedure -- the same shape as
    #: ``MeshConvert.SIDECAR_APPLIERS``, which is the pass column one level
    #: down. Subclasses reorder or extend by overriding the dict.
    EDIT_PASSES: Dict[str, str] = {
        "scene_sidecar": "_pass_scene_sidecar",
        "prune_textures": "_pass_prune_textures",
        "lightmaps": "_pass_lightmaps",
    }

    #: Passes run on the **closed** file, after the edit session, in order.
    #:
    #: Separate from :attr:`EDIT_PASSES` because the distinction is real rather
    #: than stylistic: these rewrite the container itself (repacking the BIN
    #: chunk, re-encoding payloads), which is exactly what an open edit session
    #: cannot have happening underneath it.
    FILE_PASSES: Dict[str, str] = {
        "optimize_textures": "_pass_optimize_textures",
    }

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

    def _run_passes(
        self, passes: Dict[str, str], context: "PreviewPassContext"
    ) -> None:
        """Run *passes* in order, guarding each one separately.

        The guard is per pass, not per chain: a deliverable missing one repair
        still beats no deliverable -- the same rule
        ``MeshConvert.apply_scene_sidecar`` applies one level down -- and the
        alternative failed silently in the worst direction, since an early
        pass raising would take the lightmap wiring down with it and the model
        would simply arrive unlit.
        """
        for name, method in passes.items():
            try:
                getattr(self, method)(context)
            except Exception as error:  # noqa: BLE001 — a pass must not cost the push
                context.logger.warning("Preview %s pass skipped: %s", name, error)

    # -- passes ---------------------------------------------------------
    # Each is one entry of EDIT_PASSES / FILE_PASSES above. They take the
    # context and return nothing; anything the caller needs back goes into
    # `context.results`.

    def _pass_downsize_textures(self, context: "PreviewPassContext") -> None:
        """Shrink the FBX's embedded textures to the delivery ceiling first.

        The ceiling is the shared web-delivery policy the texture pass applies
        to the GLB afterwards, so nothing is lost that the deliverable would
        have kept. Measured quiet on a production assembly (366 MB FBX, 353 MB
        of it 4096x4096 PNG): the push went from 419 s to 333 s. The converter
        itself dropped ~365 -> ~290 s -- a fifth, not the 90% its share of the
        push suggests, because its cost is the animation it bakes over every
        node rather than the images (a 12 MB textureless export of the same
        scene still took over 300 s) -- and everything after it reads 2K
        images instead of 4K (sidecar passes 17 -> 9 s, optimize 21 -> 10 s),
        with ~600 MB less scratch per push (the SDK's extracted ``.fbm`` plus
        the raw GLB). Costs ~7.5 s of its own.

        Rewritten to a NEW scratch path, never in place, and only when the
        bridge minted the original: a producer that handed back a durable
        file keeps it untouched, and ``_release_payload`` refuses to delete
        it. A payload that is not a binary FBX (a test stub, an ASCII export)
        is left alone without a word.
        """
        src = context.payload.primary
        if not src or not context.request.params.get("EMBED_TEXTURES", True):
            return
        from pythontk.file_utils.mesh_convert.fbx_file import FbxFile
        from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia

        if not FbxFile.is_fbx(src):
            return
        ceiling = _mesh_convert().web_delivery_texture_params()["max_size"]
        if not ceiling:
            return
        dst = context.bridge._make_payload_path(extension=".fbx")
        report = FbxMedia.downsize(src, dst, max_size=ceiling)
        context.results["payload_textures"] = report
        if not report["resized"]:
            return  # nothing qualified, so nothing was written
        context.payload.primary = dst
        context.bridge._release_payload(src)
        context.logger.info(
            "Payload textures: %d of %d embedded image(s) downsized to %dpx, "
            "%.1f MB -> %.1f MB.",
            report["resized"],
            report["images"],
            ceiling,
            report["before"] / 1e6,
            report["after"] / 1e6,
        )

    def _pass_scene_sidecar(self, context: "PreviewPassContext") -> None:
        """Record the scene-sidecar outcome, applying the envelope only if the
        conversion did not.

        The apply itself lives on the converter (``MeshConvert`` owns the
        applier registry, the embed, and the outcome summary). ``deliver()``
        hands the envelope to ``fbx_to_glb`` exactly as the exporters do, so
        in the normal run this pass finds the outcome already recorded in the
        file and only threads it back to the panel; the apply below is the
        fallback for a conversion that recorded nothing.
        """
        if not context.sidecar:
            return
        # `deliver()` hands the envelope to the CONVERSION as well, because
        # the conversion's own chain clones materials (one per faded subtree)
        # and a clone copies its source AS IT STANDS -- the same trap the
        # lightmap pass fell into once, which is why it runs after this. So
        # the repairs have to land before the fade pass, i.e. inside
        # `fbx_to_glb`, and what this pass then does is read the outcome back:
        # applying the envelope a second time here would write the authored
        # alpha mode over the fade clones' BLEND and pop every fade.
        applied = (context.edit.gltf.get("extras") or {}).get(
            _mesh_convert().SIDECAR_APPLIED_KEY
        )
        if applied is not None:
            context.results["sidecar"] = dict(applied)
            return
        context.results["sidecar"] = _mesh_convert().apply_scene_sidecar(
            context.edit, context.sidecar
        )

    def _pass_prune_textures(self, context: "PreviewPassContext") -> None:
        """Sweep images no material samples, before anything pays to re-encode them.

        ``EMBED_TEXTURES`` carries every wired file texture into the FBX, which
        for a StingrayPBS scene includes Autodesk's own environment maps
        (``diffuse_cube``, ``specular_cube``, ``ibl_brdf_lut`` -- ~2.6 MB a
        push); FBX2glTF re-embeds them, and glTF has no global environment slot
        for them to land in, so no material ever references them.
        ``apply_scene_sidecar`` already sweeps at its tail, so this is the
        OTHER half: a push whose producer offered no envelope (``SCENE_SIDECAR``
        off, the deliberate probe) published that dead payload and paid the
        texture pass to compress it first.

        Ordered *after* the sidecar for a second reason: pruning renumbers
        image indices, and the envelope's own ``extras.textures`` map is
        recorded at the end of that pass -- running the sweep any earlier would
        stale it.
        """
        _mesh_convert().prune_glb_unreferenced_textures(context.edit)

    def _pass_lightmaps(self, context: "PreviewPassContext") -> None:
        """Bind the baked lightmaps the in-band manifest names.

        Runs after the sidecar, and that order is measured rather than
        stylistic: this pass's per-instance material clones copy each material
        AS IT STANDS, so every repair made afterwards would land only on a base
        material no primitive references anymore. On a production room the 46
        clones the walls actually wear missed the emissive and
        metallic-roughness repairs, and the room rendered black in its own
        preview. (The conversion's own lightmap pass is switched off for the
        same reason -- see ``lightmaps=False`` in :meth:`deliver`.)
        """
        MeshConvert = _mesh_convert()
        # What the manifest asked for OF THIS GLB, read before the bind from the
        # same open session -- so the result can say how many objects came back
        # unlit, which the bound records alone cannot (a miss leaves no record).
        # Scoped, because the bake manifest is a SCENE record that every export
        # carries whole: counting it against a pushed selection reported every
        # unselected object as unlit, crying wolf on a correct preview. An
        # ambiguous leaf stays in scope -- it IS in the file and did not bind.
        coverage = MeshConvert.lightmap_manifest_coverage(context.edit)
        wanted = coverage["present"] + coverage["ambiguous"]
        bound = MeshConvert.apply_glb_lightmaps(
            # The host's live texture folders, so a manifest whose recorded
            # authoring directory has since moved still finds its EXRs --
            # otherwise the push previews unlit and blames the bake.
            context.edit,
            search_dirs=context.lightmap_search_dirs,
        )
        if bound:
            context.logger.info(
                "Lightmaps wired into %d material binding(s).", len(bound)
            )
        bound_objects = {str(record.get("object")) for record in bound}
        context.results["lightmaps"] = {
            "expected": len(wanted),
            "bound": len([name for name in wanted if name in bound_objects]),
            "unbound": [name for name in wanted if name not in bound_objects],
            # Kept rather than dropped: "3 of 3 lit" over a 50-object scene is
            # only reassuring once you can see the other 47 were never in the
            # push. Reported as scope, never as a miss.
            "out_of_scope": len(coverage["absent"]),
        }

    def _pass_optimize_textures(self, context: "PreviewPassContext") -> None:
        """Re-encode and repack the textures for web delivery.

        Last, and on the closed file, so nothing wired above re-embeds a
        full-size copy behind it. Measured on a production room this is
        94.7 MB -> ~15 MB, and its failure must cost quality, never the push --
        which is what the runner's per-pass guard buys.
        """
        # The shared web-delivery policy, named rather than inherited: this
        # pass used to pass a container and let `max_size` fall through to
        # `optimize_glb_textures`' own default, so the resolution the preview
        # approves was set by a signature default two packages away -- and the
        # scene exporters, reading their own dials, could not see it to agree
        # with it. Measured on a production assembly: 8.71 MB here against
        # 280.13 MB from the exporter, same scene, same session.
        #
        # ktx2_fallback=False: this GLB exists to be streamed to the viewer
        # page, never re-imported -- the core-readable fallback twins the
        # optimizer embeds by default would spend the very bytes this pass
        # exists to reclaim (the bundled page wires KTX2Loader, so basisu-only
        # is safe here). Deliberately not part of the shared policy: it is a
        # property of the consumer, and an exporter's deliverable must stay
        # importable.
        MeshConvert = _mesh_convert()
        MeshConvert.optimize_glb_textures(
            context.glb,
            **MeshConvert.web_delivery_texture_params(
                image_format=context.texture_format
            ),
            ktx2_fallback=False,
        )

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
        # pass their `None` default straight through.
        # The trailing WEBP is load-bearing: making this request-scoped turned
        # an absent kwarg into an explicit value, so a falsy INSTANCE default
        # stopped inheriting `optimize_glb_textures`' own "WEBP" default and
        # started handing it None -- which raises inside the optimizer and is
        # swallowed by the pass guard, silently shipping a GLB that skipped the
        # 94.7MB -> ~15MB pass.
        texture_format = request.get("texture_format") or self.texture_format or "WEBP"

        # KTX2 is the one exception: docs/webxr_preview.md promises the push
        # raises with the install URL when `toktx` is missing, never silently
        # ships WebP instead. `optimize_glb_textures` only reaches its own
        # `resolve_ktx2_encoder(required=True)` call once it hits a KTX2 image
        # -- a scene with no images (or an early failure elsewhere in the
        # method) would let the pass guard swallow it and ship the unoptimized
        # GLB. Checked eagerly, BEFORE the conversion rather than beside the
        # optimize call, so the fix-shaped FileNotFoundError arrives before
        # the session pays minutes for a conversion it is going to abandon.
        if texture_format.upper() == "KTX2":
            from pythontk.img_utils._img_utils import ImgUtils

            ImgUtils.resolve_ktx2_encoder(required=True)

        # Allocate the GLB through the bridge's own payload artifacts rather
        # than deriving a path from the FBX: that keeps it inside the prefix
        # namespace every other bridge artifact joins, so the age-gated sweep
        # reclaims it after a hard DCC crash -- the case no `finally` survives.
        glb = bridge._make_payload_path(extension=".glb")
        context = PreviewPassContext(
            bridge=bridge,
            glb=Path(glb),
            payload=payload,
            request=request,
            texture_format=texture_format,
        )
        self._run_passes(self.PAYLOAD_PASSES, context)
        try:
            # prompt=False is required, not a convenience: the confirm-download
            # path reads stdin, and a DCC has no tty -- left prompting, the very
            # first push inside Maya raises instead of installing.
            # lightmaps=False: they are wired below, AFTER the sidecar. The
            # conversion's own pass would run first, and its per-instance
            # material clones copy each material AS IT STANDS -- so every
            # repair the sidecar makes afterwards lands only on the base
            # material no primitive references anymore. Measured on a
            # production room: the 46 clones the walls actually wear missed
            # the emissive and metallic-roughness repairs, and the room
            # rendered black in its own preview.
            # sidecar=: applied INSIDE the conversion, ahead of its fade pass,
            # for the reason the lightmap note above gives -- a material clone
            # copies its source as it stands. Measured on a production
            # assembly with the envelope applied afterwards instead: the
            # metallic-roughness repair reached one fade clone of
            # SCREENS_TEST_CMPTS and the eleven-primitive original kept
            # FBX2glTF's packing, roughness and metalness 255 everywhere.
            # `_pass_scene_sidecar` reads the outcome back rather than
            # applying twice.
            MeshConvert.fbx_to_glb(
                payload.primary,
                dst=glb,
                overwrite=True,
                prompt=False,
                lightmaps=False,
                sidecar=(payload.extras or {}).get("scene_sidecar"),
            )
        except (OSError, RuntimeError, ValueError) as error:
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
        # in the system temp dir. Before the passes, so the peak footprint is
        # one deliverable rather than two. ``_release_payload`` deletes only
        # what the bridge itself minted, which is what keeps this safe on a
        # bridge whose producer hands back a durable path.
        if bridge._release_payload(payload.primary):
            bridge.logger.debug(
                "Released the consumed FBX payload: %s", payload.primary
            )

        # One session for every edit pass: repairs first, then the lightmap
        # wiring that clones the repaired materials. The container guard is for
        # what `open_glb` itself can fail with (an unparseable GLB) -- each
        # individual pass carries its own, so one failing does not cancel the
        # rest of the chain.
        try:
            with _mesh_convert().open_glb(glb) as edit:
                context.edit = edit
                self._run_passes(self.EDIT_PASSES, context)
        except Exception as error:  # noqa: BLE001 — post-process must not cost the push
            bridge.logger.warning("GLB post-process skipped: %s", error)
        finally:
            context.edit = None

        self._run_passes(self.FILE_PASSES, context)

        applied = context.results.get("sidecar", {})
        extras = payload.extras or {}
        if not (context.sidecar or {}).get("sections"):
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
            "sidecar": applied,
            # ``{"expected", "bound", "unbound": [names]}`` from the lightmap
            # pass, or ``None`` when it never ran (a pass failure, a
            # deliverer without it). ``expected`` is what the manifest named;
            # an unbound object previews UNLIT, and only this says so --
            # the viewer renders "no bake", "bake not found" and "bake bound"
            # as three shades of the same dark room.
            "lightmaps": context.results.get("lightmaps"),
            # Whether one was *offered* -- the caller cannot infer it from an
            # empty summary, which also means "switched off".
            "sidecar_requested": "scene_sidecar" in extras,
        }
