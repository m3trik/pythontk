# !/usr/bin/python
# coding=utf-8
"""The hand-off bridge whose target is a live preview page.

:class:`PreviewBridge` specialises :class:`pythontk.HandoffBridge` for one
delivery shape -- geometry pushed to a browser -- and owns everything about it
that is host-independent: the glTF-appropriate export defaults, the scene
sidecar attach, and the ``push`` / ``publish_file`` / ``url`` / ``stop``
surface. A DCC package supplies only the mixin that reads its selection
(``mayatk.WebXrPreview`` / ``blendertk.WebXrPreview``). It lives here rather
than mirrored per engine because mayatk and blendertk cannot import each
other, and anything written in both drifts in both.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from pythontk.core_utils.app_handoff import HandoffBridge, Payload
from pythontk.net_utils.preview.deliverer import PreviewDeliverer


class PreviewBridge(HandoffBridge):
    """Hand-off bridge whose target is a live preview page rather than an application.

    Sibling of :class:`pythontk.ScriptLaunchBridge`: both specialise the
    hand-off skeleton for one delivery shape. Everything about pushing geometry
    to a browser is host-independent -- the glTF-appropriate export defaults,
    the publish call, the URL -- so a DCC package supplies only what pythontk
    cannot know, which is the mixin that reads its selection:

        >>> class WebXrPreview(MayaExportMixin, ptk.PreviewBridge):
        ...     payload_prefix = "maya_webxr_preview"
        ...     deliverer = ptk.PreviewDeliverer(title="Maya")

    It lives here rather than being mirrored into each engine for the usual
    reason: mayatk and blendertk cannot import each other, so anything written
    in both drifts in both.
    """

    deliverer: Optional[PreviewDeliverer] = None

    def lightmap_search_dirs(self) -> Sequence[str]:
        """Extra directories the lightmap pass resolves the manifest's EXRs against.

        The manifest travelling inside the FBX names its maps by BASENAME plus
        the authoring directory recorded when the bake was committed, and
        ``MeshConvert.apply_glb_lightmaps`` tries that hint first. It is history
        rather than a contract: reorganise the project, or open the scene on
        another machine, and every lookup misses -- so the push previews unlit
        with the bake sitting on disk one folder away, which reads as a broken
        bake rather than a stale path.

        Empty here because pythontk cannot know where a host keeps its
        textures. A DCC bridge overrides it with the host's live answer
        (mayatk's returns ``EnvUtils.texture_search_dirs()``), which is the same
        hook shape the rest of this class uses to stay host-independent.
        """
        return ()

    def params_defaults(self) -> Dict[str, Any]:
        """glTF-appropriate export defaults, read by both DCC export mixins.

        ``EMBED_TEXTURES`` because the GLB is served standalone: an FBX
        referencing textures by path previews with every map resolving to
        nothing, since the browser can only fetch what the server hosts.
        Animation stays off -- it multiplies payload size, and a preview is
        aimed at look and scale.

        ``TRIANGULATE`` is deliberately **off**, even though glTF has no
        polygon primitive. Maya's FBX exporter rejects triangulation combined
        with smoothing groups outright -- *"Exporting a mesh with triangulation
        and Smoothing Groups enabled is not supported. The resulting FBX file
        may be invalid."* -- and the export mixins turn smoothing groups on for
        every hand-off, because that is what carries the hard/soft edge
        distinction across. Triangulating in the host would therefore trade a
        correct shading normal for an FBX the exporter itself calls invalid.
        The converter triangulates on the way to glTF regardless, so nothing is
        lost by leaving it to the one step that cannot avoid it.

        ``SCENE_SIDECAR`` carries extended scene setup the FBX cannot express,
        read from the live scene and applied to the GLB after conversion (see
        :meth:`MeshConvert.apply_scene_sidecar`). Turning it off is a
        deliberate probe -- the preview then shows exactly what the FBX itself
        carried, which is the only way to tell something the exporter dropped
        from something it mistranslated.
        """
        return {
            "INCLUDE_MATERIALS": True,
            "EMBED_TEXTURES": True,
            "TRIANGULATE": False,
            "INCLUDE_ANIMATION": False,
            "SCENE_SIDECAR": True,
        }

    def _attach_sidecar(
        self,
        payload: Payload,
        read_sections: Callable[[], Optional[Dict[str, Any]]],
        source: Dict[str, str],
    ) -> Payload:
        """Attach the scene-sidecar envelope for this push to *payload*.

        The envelope is built by :meth:`GlbPipeline.envelope` -- the one path
        every producer of a GLB (this bridge, the Scene Exporters) takes, so
        its schema, its log line and its failure policy cannot fork -- and
        rides on ``Payload.extras`` for the deliverer, which hands it to the
        build. That embedded copy is the handoff, and the only one: a
        `.scene.json` written beside the payload was a second carrier of the
        same envelope that nothing ever read back.

        *read_sections* is the host's scene-state read, called here so a read
        that raises degrades to an envelope with no sections rather than
        failing the push. The producer only calls this when the sidecar param
        is on, so key presence in ``Payload.extras`` is the "was it requested"
        signal :meth:`sidecar_summary` reads -- an empty scene still attaches
        (and writes) an envelope whose ``sections`` is ``{}``.
        """
        from pythontk.file_utils.mesh_convert.glb_pipeline import GlbPipeline

        payload.extras["scene_sidecar"] = GlbPipeline.envelope(
            read_sections,
            source=source,
            asset=os.path.basename(payload.primary) if payload.primary else None,
            logger=self.logger,
        )
        return payload

    @property
    def url(self) -> Optional[str]:
        """The preview URL, or ``None`` before the first push."""
        server = getattr(self.deliverer, "server", None)
        return server.url if server is not None else None

    def scope_objects(self, scope: str = "selected") -> List[Any]:
        """The objects *scope* resolves to, through the host hooks.

        The scope vocabulary is the ecosystem's, not this bridge's:
        ``"selected"`` / ``"all"`` / ``"visible"``, as declared by
        :meth:`uitk.bridge.Parameters.scope_spec` and resolved identically by
        every other hand-off. Public rather than private because a *caller*
        needs the answer too -- a panel that pushes blind cannot tell "nothing
        selected" from "the scene is empty" from "the export failed", and those
        three want three different messages.

        ``"selected"`` is the default AND the fallback for any unknown value:
        an unrecognised scope must never silently WIDEN a push to the whole
        scene. A host that cannot answer a widening hook (either returns
        ``None``) falls back to the selection for the same reason.
        """
        if scope == "all":
            objects = self._scene_objects()
        elif scope == "visible":
            objects = self._visible_objects()
        else:
            objects = None
        # ``None`` from either hook means "this host can't enumerate itself",
        # which the skeleton defines as fall back to the selection -- NOT as an
        # empty scene, which would report a populated scene as nothing to push.
        return self._resolve_objects(objects)

    def push(
        self,
        objects: Optional[List[Any]] = None,
        scope: str = "selected",
        open_browser: Union[bool, str, None] = None,
        texture_format: Optional[str] = None,
        scripts: Optional[Union[Dict[str, Any], List[str], tuple]] = None,
        progress: Optional[Callable[[str], Any]] = None,
        **params: Any,
    ) -> Optional[Dict[str, Any]]:
        """Export and publish, returning the deliverer's result (``None`` on failure).

        Parameters:
            objects: What to preview; ``None`` resolves *scope* against the host.
            scope: ``"selected"`` (default) / ``"all"`` / ``"visible"`` -- the
                shared bridge vocabulary (see :meth:`scope_objects`). Also
                travels to the exporter as the ``SCOPE`` param, which is how
                ``BlenderExportMixin`` knows not to re-add a hidden child of a
                visible parent and defeat the scope.
            open_browser: ``"auto"`` opens a tab only when no page is already
                watching -- so the first push and any push after the tab was
                closed, but not one that an open page will pick up. ``True``
                every push, ``False`` never. ``None`` (the default) defers to
                the deliverer's own setting -- ``"auto"`` unless it was built
                otherwise -- so a deliverer configured once (``open_browser=
                False`` for a test suite or an embedder) is not overruled by a
                caller that said nothing. It was ``"auto"`` here too, and that
                travelled as the request's EXPLICIT answer: every push through
                a deliverer that said ``False`` still opened a tab.
            texture_format: Override the deliverer's texture container for
                *this* push only (``"WEBP"`` / ``"KTX2"``); ``None`` keeps its
                default. Named explicitly rather than left to ``**params``,
                which is the *export* param bag -- swept up there it would be
                handed to the exporter and never reach the deliverer.
            scripts: Viewer scripts to run for this push -- a list of
                :attr:`PreviewServer.SCRIPTS` names, or a ``{name: path}``
                mapping for modules of your own. ``None`` (the default) leaves
                whatever the server already has active alone; ``[]`` clears
                them. Named explicitly for the same reason as
                *texture_format*: ``**params`` is the *export* bag, and swept
                up there it would be handed to the exporter and never reach the
                deliverer.
            progress: Called with a short message before each build stage, for
                a caller that can show one (a panel's footer). Named rather
                than left to ``**params`` for the same reason as the two knobs
                above: that bag goes to the EXPORTER, so a callback swept up
                there would be handed to the FBX write and never reach the
                build. ``None`` (the default) costs nothing.
            **params: Export param overrides (see :meth:`params_defaults`).
        """
        if objects is None:
            objects = self.scope_objects(scope)
        # setdefault, not assignment: an explicit SCOPE in the param bag is a
        # caller who resolved the objects themselves and is naming what they
        # mean, which must outrank the convenience default.
        params.setdefault("SCOPE", scope)
        return self.send(
            objects,
            params=params,
            open_browser=open_browser,
            texture_format=texture_format,
            scripts=scripts,
            progress=progress,
        )

    def publish_file(
        self,
        path: Union[str, Path],
        open_browser: Union[bool, str, None] = None,
        scripts: Optional[Union[Dict[str, Any], List[str], tuple]] = None,
    ) -> Dict[str, Any]:
        """Publish a GLB that already exists on disk, unchanged.

        The one delivery shape with no host in it: nothing is selected,
        exported or converted -- an authored ``.glb`` goes straight to the
        page. That makes the live preview usable as a plain viewer (compare a
        vendor's asset against your own push, or re-open the GLB an exporter
        just wrote) on the very server, port and tab a push already owns, so
        the two alternate in one page rather than needing a second viewer.

        The file is COPIED, never moved: it is the user's own asset, not a
        scratch artifact this bridge minted. It is also published exactly as
        authored -- no sidecar, no lightmap wiring, no texture re-encode --
        because those passes exist to repair what a DCC export loses, and a
        finished GLB has already answered them. What you see is the file.

        Parameters:
            path: The ``.glb`` to serve.
            open_browser: As :meth:`push` -- ``None`` (the default) defers to
                the deliverer's setting; ``"auto"`` opens a tab only when no
                page is already watching.
            scripts: Viewer scripts for this publish (see
                :attr:`PreviewServer.SCRIPTS`); ``None`` leaves the server's
                set alone.

        Returns:
            ``{"url", "version", "asset", "opened_browser", "source"}``.

        Raises:
            RuntimeError: The bridge has no deliverer to publish through.
            FileNotFoundError: *path* is not an existing file.
            ValueError: *path* is not a ``.glb``.
        """
        # The METHOD, not just the attribute: `deliverer` is typed as a
        # PreviewDeliverer but the skeleton lets any Deliverer be mounted, and
        # a plain one would fail here as an AttributeError naming an internal
        # attribute rather than saying what the bridge cannot do.
        if not callable(getattr(self.deliverer, "publish", None)):
            raise RuntimeError(
                f"{type(self).__name__} has no preview deliverer to publish "
                f"through; publish_file needs a PreviewDeliverer."
            )
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No such file to preview: {path}")
        if path.suffix.lower() != ".glb":
            # Not fussiness: the server publishes ONE file into its serve root,
            # so a .gltf's sibling .bin and textures would simply 404 and the
            # page would show an empty scene with nothing to explain it.
            raise ValueError(
                f"The preview serves a single self-contained file, so it needs "
                f"a binary .glb rather than {path.suffix or 'an extensionless file'}: "
                f"{path.name}"
            )

        record = self.deliverer.publish(
            path, move=False, open_browser=open_browser, scripts=scripts
        )
        self.logger.info(
            "Published %s to the preview as v%s.", path.name, record["version"]
        )
        return {**record, "source": str(path)}

    @staticmethod
    def sidecar_summary(result: Optional[Dict[str, Any]]) -> str:
        """One plain-text line describing what the scene sidecar did.

        Lives here rather than in each host's panel because it reads only the
        deliverer's result -- nothing about it is Maya- or Blender-specific,
        and written per panel it would be the same paragraph twice.

        The three outcomes it separates all render as the *same* unlit preview,
        which is what made the feature undebuggable from the UI: switched off,
        on but the scene had nothing to carry, and on but nothing matched.
        """
        if not result:
            return ""
        if not result.get("sidecar_requested"):
            return "Scene sidecar off - showing what the FBX carried."
        applied = result.get("sidecar") or {}
        if not applied:
            # Section-agnostic on purpose: the sections are a registry
            # (MeshConvert.SIDECAR_APPLIERS), so naming one here would go
            # stale with every addition -- it already had, when base colour
            # joined emissive and this line still said "no emissive found".
            return (
                "Scene sidecar: nothing to carry (the scene has nothing the FBX drops)."
            )
        return "Scene sidecar: " + ", ".join(
            f"{name} {outcome}" for name, outcome in sorted(applied.items())
        )

    @staticmethod
    def lightmap_summary(result: Optional[Dict[str, Any]]) -> str:
        """One plain-text line on the lightmaps: bound, or how many came back unlit.

        The sibling of :meth:`sidecar_summary`, for the same reason: a bake
        the pass could not find renders exactly like no bake at all, and
        the one warning the applier logs names a file, not an outcome. Empty
        for a scene with no committed lightmaps -- nothing to report is not
        a line. Names the unbound objects (capped) so the panel's message is
        the whole diagnosis: which maps to find, not just that some are lost.
        """
        if not result:
            return ""
        report = result.get("lightmaps") or {}
        expected = int(report.get("expected") or 0)
        if not expected:
            return ""
        bound = int(report.get("bound") or 0)
        unbound = [str(name) for name in report.get("unbound") or []]
        # SCOPE, never a miss: the bake manifest covers the whole scene, so a
        # pushed selection leaves objects out by definition and they cannot be
        # unlit in a preview they are not in. Said only when there ARE some --
        # a whole-scene push leaves nothing out and the clause would be noise --
        # because "3/3 bound" over a 50-object bake reads as suspiciously few
        # until the other 47 are accounted for.
        left_out = int(report.get("out_of_scope") or 0)
        scope = (
            f" {left_out} more object(s) in the scene's bake were not in this push."
            if left_out
            else ""
        )
        if not unbound:
            return f"Lightmaps: {bound}/{expected} object(s) bound.{scope}"
        listed = ", ".join(unbound[:5])
        if len(unbound) > 5:
            listed += f", +{len(unbound) - 5} more"
        return (
            f"Lightmaps: {bound}/{expected} object(s) bound - {len(unbound)} "
            f"preview UNLIT ({listed}). Their maps were not found where the "
            f"bake markers point; see the log for the folders searched.{scope}"
        )

    def stop(self) -> None:
        """Stop serving and release the port."""
        server = getattr(self.deliverer, "server", None)
        if server is not None:
            server.stop()


class FilePreviewBridge(PreviewBridge):
    """Preview bridge whose source is a file on disk rather than a host selection.

    The third peer of ``mayatk.WebXrPreview`` and ``blendertk.WebXrPreview``,
    and the one that needs no host at all: :meth:`_resolve_objects` takes the
    paths it is handed and :meth:`_produce` wraps one into a :class:`Payload`
    without exporting anything, because the file already IS the export. Every
    stage after that is the shared chain, so a file pushed from here and a
    selection pushed from Maya reach the page through one code path -- the
    same GLB build, the same texture pass, the same version bump.

    Both carriers the pipeline understands are accepted. An ``.fbx`` runs the
    full conversion; a ``.glb`` short-circuits to an identity build and is
    published exactly as authored, which is what makes this usable as a plain
    viewer for a vendor's asset or an exporter's own output.

    The user's file is never at risk. ``_release_payload`` refuses any path
    this bridge did not itself mint, so the deliverer's consumed-payload
    delete and the pipeline's ``release_source`` both decline it silently.

    Example:
        >>> bridge = ptk.FilePreviewBridge()
        >>> bridge.push(objects=["C:/assets/prop.fbx"])
    """

    payload_prefix = "file_webxr_preview"
    deliverer = PreviewDeliverer(title="Preview")

    #: Carriers the GLB pipeline can start from, lowercase with the dot.
    SOURCE_EXTENSIONS = (".fbx", ".glb")

    def __init__(self, app_path: Optional[str] = None):
        super().__init__(app_path)
        #: Extra folders the lightmap applier searches, set by the panel.
        self.texture_dirs: List[str] = []

    def lightmap_search_dirs(self) -> Sequence[str]:
        """Where to look for the EXRs a lightmap manifest names.

        The manifest records map BASENAMES plus the folder they were authored
        in, and that hint goes stale the moment the project moves or the file
        is opened on another machine. A DCC bridge answers this from the live
        host; with no host the only honest answers are the folder the source
        file came from and whatever the user pointed at. Without them a baked
        asset previews UNLIT with its maps one folder away, and nothing on the
        page says why -- which is the failure this hook exists to prevent.
        """
        dirs = [d for d in self.texture_dirs if d]
        source = getattr(self, "_last_source_dir", "")
        if source and source not in dirs:
            dirs.append(source)
        return tuple(dirs)

    def _resolve_objects(self, objects):
        """The paths handed in; this bridge has no host to ask."""
        return list(objects or [])

    def _produce(self, objects, request) -> Optional[Payload]:
        """Hand the chosen file to the deliverer as the payload, unexported.

        Nothing is written: the file on disk is already the artifact every
        other producer spends an export making. Validation is here rather
        than in the panel so a bridge driven from a script gets the same
        fix-shaped refusals a button does.
        """
        if not objects:
            self.logger.error("No file chosen to preview.")
            return None

        path = Path(str(objects[0]))
        if not path.is_file():
            self.logger.error(f"No such file to preview: {path}")
            return None

        extension = path.suffix.lower()
        if extension not in self.SOURCE_EXTENSIONS:
            self.logger.error(
                f"The preview builds from {' or '.join(self.SOURCE_EXTENSIONS)}, "
                f"not {extension or 'an extensionless file'}: {path.name}"
            )
            return None

        # Remembered for `lightmap_search_dirs`, which the deliverer calls
        # after this returns: a bake's maps most often sit beside the file
        # that references them, so the source folder is the best free guess.
        self._last_source_dir = str(path.parent)
        return Payload(primary=str(path))
