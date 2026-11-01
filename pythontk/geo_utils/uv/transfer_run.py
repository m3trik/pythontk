# !/usr/bin/python
# coding=utf-8
"""A Transfer run -- UVs OR textures, source -> targets -- written once for both hosts.

Two transfers that are ALTERNATIVES: the source's UV layout onto the targets
(the host's ``UvUtils.transfer_uvs`` -- exact for identical topology, sampled
by proximity otherwise), or the source's textures re-mapped into each
target's OWN layout (the host's ``TextureTransfer.transfer`` over
:class:`~pythontk.UvTransfer`: exact texel correspondence, no rays, no bleed).
They do not compose -- the texture pass keeps the target's UV channel, so a
source layout copied alongside its maps would land in one nothing references.
*Auto* defers the pick to run time.

:class:`TransferRun` holds the decisions -- which passes a Source / Transfer
pair runs (:meth:`~TransferRun.passes`), Auto's pick
(:meth:`~TransferRun.resolve_auto`), the refusals (:meth:`~TransferRun.gate`)
-- and the run itself (:meth:`~TransferRun.run`: one undo step), over a host
adapter (:class:`TransferHost`; mayatk's ``MayaTransferHost``, blendertk's
``BlenderTransferHost``, which each library's ``TextureTransfer.run`` hands
it; :class:`PureTransferHost` with no DCC: no scene, nothing selected). A run
returns a :class:`TransferResult` -- plain values: what was
decided, what each pass did, what Overwrite target rewrote -- that the Texture
Combine panel words in its host's terms. Nothing here speaks to a user.
"""

import contextlib
import os
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    runtime_checkable,
)

from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.pending_files import PendingFiles


@runtime_checkable
class TransferHost(Protocol):
    """What a :class:`TransferRun` asks of its host: each one of the host
    library's own operations (:class:`PureTransferHost` holds the host-free
    answers). A mesh is the host's (Maya: a long name; Blender: an object)."""

    def selection_sides(
        self, mode: str, stored: Sequence
    ) -> Tuple[str, Optional[list], list]:
        """The selection read into ``(mode, source, targets)`` for the Source
        row's *mode* (``"auto"`` resolved), *stored* the stored source meshes
        (``TextureTransfer.selection_sides``)."""

    def face_materials(self, mesh) -> Tuple[list, Any]:
        """``(materials, per-face ids)`` of what *mesh* wears."""

    def material_maps(self, material) -> Dict[str, str]:
        """``{channel: path}`` of the maps *material* reads."""

    def pair_sources(self, targets: Sequence, sources: Sequence) -> Dict[Any, Any]:
        """Target -> its source: one mesh, or the tuple a combined target was
        built from. Raises ``ValueError`` when they cannot pair."""

    def default_name(self, nodes: Sequence) -> str:
        """What a blank Output Name stands for: *nodes*' common name, else
        ``"transfer"``."""

    def transfer_uvs(self, sources: Sequence, targets: Sequence) -> List[tuple]:
        """Each source's UVs onto the target paired with it, the pairing as
        given: ``[(source, target, how)]``, *how* ``"topology"`` when exact.
        Raises ``ValueError`` on a refusal."""

    def transfer_uvs_to_similar(
        self, source, candidates: Optional[Sequence], tolerance: float
    ) -> list:
        """*source*'s UVs onto every mesh similar to it at *tolerance* (among
        *candidates*; ``None``: the scene): the meshes it reached."""

    def texture_transfer(self) -> Any:
        """A texture-transfer engine: ``transfer(targets, source, **options)``
        returning ``{material: {channel: path}}``, reporting through its
        ``logger`` (a :class:`~pythontk.LoggingMixin` logger)."""

    def has_lightmap(self, mesh) -> bool:
        """Whether *mesh* carries a committed lightmap."""

    def transfer_lightmaps(self, targets, source, **options) -> Dict[Any, dict]:
        """Each source's committed lightmap onto its target:
        ``{target: {"how": "rebound" | "resampled", "path": ...}}``. Raises
        ``ValueError`` on a refusal."""

    def undo_chunk(self, name: str) -> ContextManager:
        """One undo step named *name*."""


class PureTransferHost:
    """The host-free answers (:class:`TransferHost`): no scene, so nothing is
    selected, nothing wears a material and a transfer is refused. A host
    adapter subclasses it and overrides what its host does."""

    def selection_sides(self, mode, stored):
        if mode == "uvset":
            return mode, None, []
        return ("first" if mode == TransferRun.AUTO else mode), [], []

    def face_materials(self, mesh):
        return [], None

    def material_maps(self, material):
        return {}

    def pair_sources(self, targets, sources):
        raise ValueError("No scene to pair meshes in.")

    def default_name(self, nodes):
        return "transfer"

    def transfer_uvs(self, sources, targets):
        raise ValueError("No scene to transfer UVs in.")

    def transfer_uvs_to_similar(self, source, candidates, tolerance):
        return []

    def texture_transfer(self):
        raise ValueError("No scene to transfer textures in.")

    def has_lightmap(self, mesh):
        return False

    def transfer_lightmaps(self, targets, source, **options):
        raise ValueError("No scene to transfer lightmaps in.")

    def undo_chunk(self, name):
        return contextlib.nullcontext()


@dataclass
class TransferResult:
    """What one :meth:`TransferRun.run` decided and did -- plain values.

    ``refusal`` names why the run stopped (``None`` when it went through),
    one of :attr:`TransferRun.REFUSALS`, with the host's own message in
    ``error`` where it gave one. Decided: ``mode`` (the Source row's reading
    of the selection), ``scope`` (the Targets row as run), ``source`` /
    ``targets`` (a Similar scope's targets are the meshes it reached),
    ``source_auto`` / ``transfer_auto`` (Auto read the selection / picked the
    transfer), ``lightmaps`` (Include Lightmaps), ``do_uvs`` /
    ``do_textures`` (the passes). Done: ``uvs`` (``(source, target, how)`` per
    UV transfer); the texture pass's ``maps`` (``{material: {channel:
    path}}``; ``None`` when it failed, ``texture_error`` saying why, or did not
    run), under ``output_name`` (typed when ``typed``, else the source's
    common name) and carrying ``named``; Overwrite target's (``in_place``)
    ``overwritten`` (files rewritten where they were), ``added`` (new files
    the targets' own materials read now) and ``warnings`` (what it left
    alone, and why); the lightmap pass's ``lightmap_results`` (``{target:
    {"how", "path"}}``; ``None`` when it failed, ``lightmap_error`` saying
    why, or did not run) and ``lightmap_folder`` (where a resampled one goes
    once the scene is saved).
    """

    mode: str = ""
    scope: str = "order"
    source: Optional[list] = None
    targets: list = field(default_factory=list)
    source_auto: bool = False
    transfer_auto: bool = False
    lightmaps: bool = False
    do_uvs: bool = False
    do_textures: bool = False
    refusal: Optional[str] = None
    error: str = ""
    uvs: List[tuple] = field(default_factory=list)
    maps: Optional[Dict[str, Dict[str, str]]] = None
    texture_error: str = ""
    output_name: str = ""
    named: str = ""
    typed: bool = False
    in_place: bool = False
    overwritten: List[str] = field(default_factory=list)
    added: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    lightmap_results: Optional[Dict[Any, dict]] = None
    lightmap_error: str = ""
    lightmap_folder: str = ""

    @property
    def map_count(self) -> int:
        """How many maps the texture pass wrote."""
        return sum(len(maps) for maps in (self.maps or {}).values())

    @property
    def maps_folder(self) -> str:
        """The folder the texture pass wrote into (``""`` without a map)."""
        return next(
            (
                os.path.dirname(path)
                for maps in (self.maps or {}).values()
                for path in maps.values()
            ),
            "",
        )

    @property
    def resampled(self) -> List[str]:
        """The lightmaps the lightmap pass wrote (resampled into the target's
        layout; a rebound one writes nothing)."""
        return [
            r["path"]
            for r in (self.lightmap_results or {}).values()
            if r.get("how") == "resampled"
        ]

    @property
    def produced(self) -> bool:
        """Whether the run wrote a file: a map, or a resampled lightmap."""
        return bool(self.map_count or self.resampled)


class TransferRun:
    """Transfer's decisions and its run over a :class:`TransferHost` (see the
    module docstring).

    Parameters:
        host: The host adapter (each library's ``TextureTransfer.run`` builds
            one); ``None``: :class:`PureTransferHost`.
    """

    AUTO = "auto"
    TEXTURES = "textures"
    UVS = "uvs"
    #: Why a run stops (:attr:`TransferResult.refusal`): nothing to transfer
    #: (the same-mesh source with a UV-only Transfer); nothing selected (the
    #: same-mesh source), no targets (a mesh source), no source for the First
    #: / Last pick (``pick_first`` / ``pick_last``); a Similar scope with no
    #: UV pass to find its targets; no stored source meshes; the host refusing
    #: the pairing (``pair_error``) or the UV pass (``uv_error``); a combined
    #: target for the UV pass, which copies one source per target; no mesh
    #: similar enough to the source.
    REFUSALS = (
        "nothing_to_transfer",
        "nothing_selected",
        "no_targets",
        "pick_first",
        "pick_last",
        "similar_needs_uvs",
        "no_stored",
        "pair_error",
        "one_source_per_target",
        "no_similar",
        "uv_error",
    )
    #: The run's one undo step.
    UNDO_NAME = "Texture Transfer"

    def __init__(self, host: Optional[TransferHost] = None):
        self.host = host if host is not None else PureTransferHost()

    # ------------------------------------------------------------ decisions
    @classmethod
    def passes(
        cls, source_mode: str, transfer_mode: Optional[str]
    ) -> Tuple[bool, bool]:
        """``(do_uvs, do_textures)`` for a Source / Transfer pair: Auto
        unresolved is both possible; the same-mesh source (``"uvset"``) has no
        second mesh for UVs; no Transfer reads as UVs."""
        mode = transfer_mode or cls.UVS
        if mode == cls.AUTO:
            return source_mode != "uvset", True
        textures = mode == cls.TEXTURES
        return not textures and source_mode != "uvset", textures

    def resolve_auto(
        self, source_meshes: Sequence, has_lightmap: Optional[Callable] = None
    ) -> str:
        """Auto's run-time pick: Textures once any source material carries a
        map (or, with *has_lightmap*, a source carries a committed lightmap),
        else UVs. An unreadable mesh, material or marker counts as none."""
        host = self.host
        for mesh in source_meshes:
            if has_lightmap is not None:
                try:
                    if has_lightmap(mesh):
                        return self.TEXTURES
                except Exception:  # noqa: BLE001 - unreadable marker = none
                    pass
            try:
                materials, _ = host.face_materials(mesh)
            except Exception:  # noqa: BLE001
                continue
            for material in materials:
                try:
                    if host.material_maps(material):
                        return self.TEXTURES
                except Exception:  # noqa: BLE001
                    continue
        return self.UVS

    def pair(
        self, targets: Sequence, sources: Sequence, do_uvs: bool
    ) -> Tuple[Optional[list], Optional[str], str]:
        """``(pairs, refusal, error)`` for mesh sources: ``[(source, target)]``
        by the host's pairing, else why not (the host's message in *error*).
        The UV pass copies one source per target, so a target combined from
        several refuses it."""
        try:
            grouped = self.host.pair_sources(targets, sources)
        except ValueError as e:
            return None, "pair_error", str(e)
        if do_uvs and any(isinstance(s, tuple) for s in grouped.values()):
            return None, "one_source_per_target", ""
        return [(s, t) for t, s in grouped.items()], None, ""

    def gate(
        self,
        mode: str,
        scope: str,
        source: Optional[list],
        targets: Optional[list],
        do_uvs: bool,
        do_textures: bool,
    ) -> Tuple[Optional[list], Optional[list], Optional[list], Optional[str], str]:
        """``(pairs, targets, others, refusal, error)``: the run's refusal, or
        what it transfers. *pairs* are mesh sources' ``(source, target)``
        (``None`` on one mesh, and for a Similar scope, whose *others* -- the
        selected meshes -- are where it looks)."""
        if not (do_uvs or do_textures):
            return None, targets, None, "nothing_to_transfer", ""
        if mode == "uvset":
            if not targets:
                return None, targets, None, "nothing_selected", ""
            return None, targets, None, None, ""
        if mode == "first" and scope != "order":
            if not source or (scope == "selection" and not targets):
                return None, None, None, "pick_first", ""
            if not do_uvs:
                return None, None, None, "similar_needs_uvs", ""
            return None, None, targets, None, ""
        if mode == "stored" and not source:
            return None, targets, None, "no_stored", ""
        if not (source and targets):
            refusal = {"first": "pick_first", "last": "pick_last"}.get(
                mode, "no_targets"
            )
            return None, targets, None, refusal, ""
        pairs, refusal, error = self.pair(targets, source, do_uvs)
        return pairs, targets, None, refusal, error

    # ------------------------------------------------------------ overwrite
    @staticmethod
    def path_key(path: str) -> str:
        """*path* as the filesystem compares it."""
        return os.path.normcase(os.path.abspath(path))

    def read_files(self, targets: Sequence) -> Set[str]:
        """Path keys of the maps the materials *targets* wear read now."""
        host = self.host
        return {
            self.path_key(path)
            for target in targets or ()
            for material in host.face_materials(target)[0]
            for path in host.material_maps(material).values()
        }

    @classmethod
    def split_written(
        cls, before: Set[str], maps: Dict[str, Dict[str, str]]
    ) -> Tuple[List[str], List[str]]:
        """``(overwritten, added)``: the files *maps* wrote that the targets'
        materials read *before* the run (rewritten where they were), and the
        new ones they read now."""
        written = [path for channels in maps.values() for path in channels.values()]
        over = [path for path in written if cls.path_key(path) in before]
        return over, [path for path in written if cls.path_key(path) not in before]

    # ------------------------------------------------------------ run
    def run(
        self,
        source: str = AUTO,
        transfer: str = AUTO,
        scope: str = "order",
        stored: Sequence = (),
        similarity: float = 0.9,
        lightmaps: bool = False,
        output_name: str = "",
        output_dir: Optional[str] = None,
        source_uv_set: Optional[str] = None,
        target_uv_set: Optional[str] = None,
        size: Optional[int] = None,
        supersample: int = 2,
        padding: int = -1,
        assign: bool = True,
        assign_prefix: str = "",
        assign_suffix: Optional[str] = None,
        assign_from: str = "source",
        assign_shader_type: Optional[str] = None,
        progress: Optional[Callable[[str], Any]] = None,
    ) -> TransferResult:
        """Transfer UVs OR textures from the selection's source onto its
        targets: one pass per run, one undo step (:attr:`UNDO_NAME`); a run
        refused before a pass starts (:meth:`gate`) opens none.

        Parameters:
            source: The Source row: ``"auto"``, ``"first"``, ``"last"``,
                ``"stored"`` or ``"uvset"`` (one mesh, between its UV
                channels) -- read off the selection by the host.
            transfer: The Transfer row: ``"auto"``, ``"textures"`` or
                ``"uvs"``.
            scope: The Targets row for a First pick: ``"order"`` (the rest of
                the selection), ``"selection"`` / ``"scene"`` (the meshes
                similar to the source, at *similarity*). Source: Auto reads
                ``"order"``.
            stored: The stored source meshes (the host's names).
            similarity: The Similar scopes' minimum score (0-1).
            lightmaps: Also carry each source's committed lightmap (mesh
                sources only); Auto then picks Textures for one.
            output_name: The maps' name; blank: the source's common name.
            output_dir: Where the maps go (blank: the host's default).
            source_uv_set: The channel the textures are read through (blank:
                the host's pick).
            target_uv_set: The channel the maps are written for.
            size: Output size per target material (``None``: the largest
                source map feeding it).
            supersample: Sub-samples per texel axis.
            padding: Gutter fill in texels (``-1``: every empty texel).
            assign: Assign a material wired to the new maps.
            assign_prefix: The assigned material's name prefix.
            assign_suffix: Its suffix (``None``: the host's convention).
            assign_from: What it is: ``"source"`` / ``"target"`` (a copy of
                that material) or ``"in_place"`` (Overwrite target: the
                target's own, its maps rewritten where they are).
            assign_shader_type: A shader type to rebuild it on (a host that
                has them).
            progress: Called with ``"uvs"`` / ``"textures"`` as each pass starts.

        Returns:
            The run's :class:`TransferResult`.
        """
        texture_options = dict(
            source_uv_set=source_uv_set or None,
            target_uv_set=target_uv_set or None,
            size=size or None,
            supersample=supersample or 2,
            padding=padding,
            output_dir=output_dir or None,
            assign=bool(assign),
            assign_prefix=assign_prefix,
            assign_suffix=assign_suffix,
            assign_from=assign_from,
        )
        if assign_shader_type:
            texture_options["assign_shader_type"] = assign_shader_type
        result, plan = self._decide(
            source or self.AUTO, transfer, scope or "order", stored, bool(lightmaps)
        )
        if result.refusal:
            # Decided before the step opens: Blender pushes one on its exit
            # whatever ran, so a refusal left an undo step that undid nothing.
            return result
        with self.host.undo_chunk(self.UNDO_NAME):
            self._transfer(
                result,
                plan,
                similarity,
                (output_name or "").strip(),
                texture_options,
                progress or (lambda _stage: None),
            )
        return result

    def _decide(self, requested, transfer, scope, stored, lightmaps):
        """``(result, (pairs, targets, others))``: the selection read, Auto
        resolved and the refusals -- no scene edit."""
        host = self.host
        # The host's reading of the selection: Maya's order, Blender's active
        # object; Auto resolved there too.
        mode, source, targets = host.selection_sides(requested, list(stored or ()))
        result = TransferResult(
            mode=mode,
            source=source,
            targets=list(targets or []),
            source_auto=requested == self.AUTO,
            transfer_auto=transfer == self.AUTO,
            lightmaps=lightmaps,
        )
        if result.source_auto:
            scope = "order"  # Auto's First reads Rest of Selection
        result.scope = scope
        if result.transfer_auto and mode != "uvset":
            transfer = self.resolve_auto(
                source or [], has_lightmap=host.has_lightmap if lightmaps else None
            )
        result.do_uvs, result.do_textures = self.passes(mode, transfer)
        pairs, targets, others, result.refusal, result.error = self.gate(
            mode, scope, source, targets, result.do_uvs, result.do_textures
        )
        return result, (pairs, targets, others)

    def _transfer(self, result, plan, similarity, name, options, progress) -> None:
        """The passes *result*'s decisions run, inside the run's undo step."""
        host = self.host
        pairs, targets, others = plan
        source, scope = result.source, result.scope
        if result.do_uvs:
            progress("uvs")
            try:
                if pairs is None:
                    targets = list(
                        host.transfer_uvs_to_similar(
                            source[0],
                            others if scope == "selection" else None,
                            similarity,
                        )
                        or []
                    )
                    if not targets:
                        result.refusal = "no_similar"
                        return
                    result.uvs = [(source[0], t, "topology") for t in targets]
                else:
                    # The pairing is already decided (selection / stored by
                    # name): re-vetting it by similarity could only drop a
                    # pair the user named.
                    result.uvs = list(
                        host.transfer_uvs([s for s, _ in pairs], [t for _, t in pairs])
                        or []
                    )
            except ValueError as e:
                result.refusal, result.error = "uv_error", str(e)
                return
        result.targets = list(targets or [])
        if result.do_textures:
            progress("textures")
            self._texture_pass(result, targets, source, name, options)

    def _texture_pass(self, result, targets, source, name, options) -> None:
        """The texture pass (and, with Include Lightmaps, the lightmap pass)."""
        host = self.host
        result.typed = bool(name)
        if not result.typed:
            name = host.default_name(list(source) if source else list(targets or []))
        result.output_name = result.named = name
        in_place = options["assign"] and options["assign_from"] == "in_place"
        result.in_place = in_place
        before = self.read_files(targets) if in_place else set()
        engine = left = None
        try:
            engine = host.texture_transfer()
            # What Overwrite target leaves alone is said as a warning; the
            # result repeats it (the engine class's own ring buffer, read
            # before it is dropped).
            if in_place:
                left = engine.logger.enable_log_buffer(level="WARNING")
            maps = engine.transfer(targets, source, output_name=name, **options)
        except ValueError as e:
            result.texture_error = str(e)
            maps = None
        finally:
            warnings = [r.getMessage() for r in left.buffer] if left else []
            if left is not None:
                engine.logger.disable_log_buffer()
        if maps is not None:
            result.maps = maps
            if in_place:
                result.overwritten, result.added = self.split_written(before, maps)
                result.warnings = warnings
            stems = {
                os.path.basename(path).rsplit("_", 1)[0]
                for channels in maps.values()
                for path in channels.values()
            }
            # In place, a map keeps the name it had: no stem is the output's.
            if len(maps) == 1 and len(stems) == 1 and not in_place:
                result.named = stems.pop()
        if source is not None and result.lightmaps:
            self._lightmap_pass(result, targets, source, options)

    def _lightmap_pass(self, result, targets, source, options) -> None:
        try:
            transferred = self.host.transfer_lightmaps(
                targets,
                source,
                output_dir=options["output_dir"],
                output_name=result.named,
                supersample=options["supersample"],
                padding=options["padding"],
            )
        except ValueError as e:
            result.lightmap_error = str(e)
            return
        result.lightmap_results = dict(transferred or {})
        resampled = result.resampled
        # A resampled map waits in the session until the scene is saved
        # (``PendingFiles``); the folder it goes to is its home.
        if resampled:
            result.lightmap_folder = FileUtils.format_path(
                os.path.dirname(PendingFiles.session().home(resampled[0]))
            )
