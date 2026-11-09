# !/usr/bin/python
# coding=utf-8
"""A file delivered by copying it into a folder another application watches.

Many targets ingest whatever lands in a folder: a game engine imports any file
dropped into its project, a renderer watches a hot folder. For those the
session-safe hand-off is the copy itself, and nothing launches or attaches to
the application. :class:`FolderDeliverer` is that :class:`Deliverer`: it copies
the payload's file into the bridge's folder (under a subfolder, by a name),
whole or not at all. Given an *embed* strategy (``FbxMedia.embed_dependencies``,
which the caller hands in: this layer imports nothing above it), an FBX is
delivered carrying every file its scene records name, so the FBX alone is the
hand-off.
:class:`FileDeliveryBridge` delivers a file already on disk through it.

An application shapes it by subclassing: its folder (:meth:`target`), its
default subfolder and its parameter keys, and what it does once the file landed
(:meth:`after`, e.g. open the application). A DCC's export bridge takes the
same deliverer (``deliverer=``), so a scene and a file are delivered alike.

An operation context (``ptk.OpContext``) rides the request as
``extras["ctx"]``: asked before the copy, a dry run lists it and changes nothing.

Weighed (CODE_STANDARD section 15): driving the application instead -- its
scripting or RPC (a command port, a remote-execution bridge such as the ones
DCC-to-engine "send to" tools use) -- needs it running and attaches to a
session a person may have open; the application's own package format (an
engine's import package) still needs the application to import it. A watched
folder needs neither: it is the hand-off such a target already reads, open or
closed, and its own importer stays the one that interprets the file. What the
application does next (open it on the folder, a batch import) is
:meth:`after`'s, which an application adds.

Qt-free and DCC-free.
"""

from __future__ import annotations

import ntpath
import os
import re
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple

from pythontk.core_utils.handoff.app_handoff import (
    Deliverer,
    HandoffBridge,
    HandoffRequest,
    Payload,
)

_INVALID_NAME_CHARS = re.compile(r"[^A-Za-z0-9 ._-]+")


class FolderDeliverer(Deliverer):
    """Copy the payload's file into a folder (see the module docstring).

    Bridge contract: ``bridge.logger``; ``bridge.target_folder``, the folder
    (:meth:`target` may derive another from it); ``payload.primary``, the file;
    ``payload.extras["default_asset_name"]``, the name a request that names
    none takes.
    """

    #: The subfolder a request that names none lands in ("" = the folder itself).
    DEFAULT_SUBDIR = ""
    #: The request parameters naming the subfolder and the file's name.
    SUBDIR_PARAM = "SUBDIR"
    NAME_PARAM = "NAME"
    #: ``(stem, mode label)`` of each delivery: one, the copy.
    DELIVERY_MODES: Tuple[Tuple[str, str], ...] = (("copy", ""),)

    def __init__(self, embed: Optional[Callable[..., Optional[Dict[str, Any]]]] = None):
        """
        Parameters:
            embed: How a file takes the files its records name:
                ``embed(source, dest, search_dirs=..., logger=...)``, called for
                every delivered file, writing *dest* and returning a report
                whose ``"embedded"`` lists what it added; a file it cannot carry
                anything in (for ``ptk.FbxMedia.embed_dependencies``: one that
                is no binary FBX) it leaves, reporting nothing added. ``None``:
                every file is copied as it is.
        """
        self.embed = embed

    def target(self, bridge: Any) -> Optional[str]:
        """The folder the file lands in: ``bridge.target_folder`` when it is
        a folder, else ``None``."""
        folder = getattr(bridge, "target_folder", None)
        return folder if folder and os.path.isdir(folder) else None

    def after(
        self, bridge: Any, result: Dict[str, Any], request: HandoffRequest
    ) -> Dict[str, Any]:
        """What the application does once the file landed (default: nothing);
        returns the result, extended."""
        return result

    def preflight(self, bridge: Any, request: HandoffRequest) -> bool:
        """Refuse before the export when the folder does not exist."""
        if self.target(bridge) is None:
            bridge.logger.error(
                f"Set a folder that exists to deliver into. Got: "
                f"{getattr(bridge, 'target_folder', None)!r}"
            )
            return False
        return True

    def deliver(
        self, bridge: Any, payload: Payload, request: HandoffRequest
    ) -> Optional[Dict[str, Any]]:
        """Copy ``payload.primary`` into ``<folder>/<subfolder>/<name><ext>``.

        The subfolder and the name come from :attr:`SUBDIR_PARAM` and
        :attr:`NAME_PARAM` (defaults: :attr:`DEFAULT_SUBDIR`, the payload's
        ``default_asset_name``), reduced so neither can leave the folder. A file
        of the same name is replaced whole.

        Returns:
            dict | None: ``file`` (where it landed), ``folder``, ``subdir`` and
            ``source``, plus what :meth:`after` adds; ``None`` on a handled
            (logged) failure.
        """
        ctx = request.get("ctx")
        folder = self.target(bridge)
        if folder is None:
            wanted = getattr(bridge, "target_folder", None)
            bridge.logger.error(f"No folder to deliver into: {wanted!r}")
            return None
        params = request.params or {}
        parts = self.subdir_parts(params.get(self.SUBDIR_PARAM), self.DEFAULT_SUBDIR)
        dest_dir = os.path.join(folder, *parts) if parts else folder
        name = str(params.get(self.NAME_PARAM) or "").strip()
        name = self.sanitize_name(name or payload.extras.get("default_asset_name") or "Export")
        ext = os.path.splitext(payload.primary)[1] or ".fbx"
        dest = os.path.join(dest_dir, f"{name}{ext}")
        rel = os.path.relpath(dest, folder)
        if ctx is None or ctx.should_write(dest, f"copy the file in as {rel}"):
            os.makedirs(dest_dir, exist_ok=True)
            try:
                self.copy_self_contained(bridge, payload.primary, dest)
            except OSError as e:
                bridge.logger.error(f"Failed to copy the file in: {e}")
                return None
            bridge.logger.info(
                f'Delivered: <a href="action://open?path={dest}">{rel}</a>'
            )
        result = {
            "file": dest,
            "folder": folder,
            "subdir": "/".join(parts),
            "source": payload.primary,
        }
        return self.after(bridge, result, request)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def subdir_parts(raw: Any, default: str = "") -> List[str]:
        """*raw* (either separator) as folder names that cannot leave the
        folder: no drive, no empty, ``.`` or ``..`` component. Blank: *default*;
        a value that reduces to nothing: the folder itself."""
        text = str(raw or "").strip() or default
        parts = []
        for part in text.replace("\\", "/").split("/"):
            part = ntpath.splitdrive(part.strip())[1].strip()
            if part and part not in (".", ".."):
                parts.append(part)
        return parts

    @staticmethod
    def sanitize_name(name: Any) -> str:
        """*name* reduced to a safe file stem (``"Export"`` when nothing survives)."""
        cleaned = _INVALID_NAME_CHARS.sub("_", str(name or "")).strip(" ._-")
        return re.sub(r"\s+", " ", cleaned) or "Export"

    def copy_self_contained(self, bridge: Any, source: str, dest: str) -> None:
        """Copy *source* to *dest*, an FBX carrying every file its scene
        records name.

        The records name their files (lightmaps, shadow maps) by file name, and
        a reader finds them only beside the model, so they are embedded through
        :attr:`embed`: each name resolves where the bridge's host keeps it (its
        optional ``lightmap_search_dirs`` hook), then beside *source*. Every
        file is handed to the embed, which decides by its content what it can
        carry (a file that is no binary FBX: nothing), so the format's rules
        live with the format's code. The embedding writes a scratch copy, so
        the target never meets a half-written file and *source* is only read.
        A file the embed added nothing to, and every file without an *embed*,
        is copied as it is.

        Raises:
            OSError: The copy failed; *dest* is untouched.
        """
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        if self.embed is None:
            self._place(source, dest)
            return
        hook = getattr(bridge, "lightmap_search_dirs", None)
        try:
            search_dirs = list(hook() or ()) if callable(hook) else []
        except Exception as e:  # noqa: BLE001 -- the source folder still answers
            bridge.logger.warning(f"Map folders unavailable from the host: {e}")
            search_dirs = []
        scratch = TempArtifacts("folder_delivery", policy="scoped")
        try:
            staged = scratch.path(extension=os.path.splitext(source)[1] or ".fbx")
            try:
                report = self.embed(source, staged, search_dirs=search_dirs, logger=bridge.logger)
            except Exception as e:  # noqa: BLE001 -- an addition, never the send
                bridge.logger.warning(
                    f"The FBX's dependencies were not embedded ({e}); it ships as exported."
                )
                report = None
            self._place(staged if report and report.get("embedded") else source, dest)
        finally:
            scratch.cleanup()

    @staticmethod
    def _place(source: str, dest: str) -> None:
        """Copy *source* to *dest* whole or not at all, through a ``.part``
        promoted once complete (``FileUtils.atomic_write``).

        Raises:
            OSError: The copy failed; *dest* is untouched.
        """
        from pythontk.file_utils._file_utils import FileUtils

        FileUtils.atomic_write(dest, lambda part: shutil.copy2(source, part))


class FileDeliveryBridge(HandoffBridge):
    """Deliver a file already on disk into a folder, through a
    :class:`FolderDeliverer` (an application's, or the plain one)::

        bridge = FileDeliveryBridge(target_folder="C:/watch")
        bridge.send("C:/scan/robot.fbx", params={"SUBDIR": "Models"})
    """

    payload_prefix = "file_delivery"
    # The file is given: there is no selection to resolve.
    requires_objects = False

    def __init__(
        self,
        target_folder: Optional[str] = None,
        deliverer: Optional[FolderDeliverer] = None,
    ):
        """
        Parameters:
            target_folder: The folder the file is delivered into.
            deliverer: The delivery (default: a plain :class:`FolderDeliverer`).
        """
        super().__init__()
        self.target_folder = target_folder
        self.deliverer = deliverer or FolderDeliverer()

    def list_template_modes(self) -> List[Tuple[str, str]]:
        """``[(mode_stem, ""), ...]`` for a panel's delivery combo."""
        return list(self.deliverer.DELIVERY_MODES)

    def send(
        self,
        model_path: str,
        *,
        template: str = "copy",
        mode: str = "",
        params: Optional[Dict[str, Any]] = None,
        ctx: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Deliver *model_path* into :attr:`target_folder`.

        Parameters:
            model_path: The file to deliver.
            template: The delivery (:attr:`FolderDeliverer.DELIVERY_MODES`).
            mode: The delivery's mode label.
            params: The request's values (the deliverer's subfolder and name keys).
            ctx: An operation context, asked before the copy: a dry run lists
                it and changes nothing.

        Returns:
            dict | None: The deliverer's result, or ``None`` on a handled
            (logged) failure.
        """
        return self._run(
            [model_path],
            HandoffRequest(
                template=template, mode=mode, params=self.merge_params(params), extras={"ctx": ctx}
            ),
        )

    def _resolve_objects(self, objects):
        return objects or []

    def _produce(self, objects, request) -> Optional[Payload]:
        """The given file as the payload."""
        path = objects[0] if objects else None
        if not path or not os.path.isfile(path):
            self.logger.error(f"File not found: {path}")
            return None
        stem = os.path.splitext(os.path.basename(path))[0]
        return Payload(primary=path, extras={"default_asset_name": stem})
