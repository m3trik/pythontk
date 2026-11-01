# !/usr/bin/python
# coding=utf-8
"""Files the served page asks the server to write: recordings and stills.

The page records a clip through the ``playblast`` routes (the recorder,
:class:`~pythontk.net_utils.preview.playblast.PreviewPlayblast`, is built on
first use) and posts a still to ``snapshot``. Both land by one placement rule
(:meth:`_PageOutputs.page_output`): beside the published deliverable when it
is still on disk and its folder takes the file, else in the serve root, from
which the page downloads it. Whether the folder takes it is found by writing
it -- never a probe file in the user's folder. Every name is composed here,
never taken from the page.

:class:`_PageOutputs` is the server's collaborator that owns the recorder,
the finished recordings and the still's numbering lock; it reads the server's
published source, its root and its limits (:attr:`PreviewServer.SNAPSHOT_TYPES`
...) through it at call time.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING
from urllib.parse import quote

from pythontk.file_utils._file_utils import FileUtils
from pythontk.net_utils.preview.routes import PLAYBLAST_PATH
from pythontk.str_utils._str_utils import StrUtils

if TYPE_CHECKING:  # the recorder is imported on use -- see playblast
    from pythontk.net_utils.preview.playblast import PreviewPlayblast
    from pythontk.net_utils.preview.server import PreviewServer


class _PageOutputs:
    """Page recordings and stills, and where they land.

    A private collaborator of :class:`~pythontk.PreviewServer`; call it through
    the facade, whose methods carry the documentation.
    """

    def __init__(self, server: "PreviewServer"):
        self._server = server
        #: The page's clip recorder, allocated on first use, and a lock of its
        #: own for that one allocation -- see :meth:`playblast`.
        self.recorder: Optional["PreviewPlayblast"] = None
        self.recorder_lock = threading.Lock()
        #: Finished recordings, token -> file, so a download can follow.
        self.recordings: Dict[str, Path] = {}
        #: Held from choosing a still's number to writing it: two stills posted
        #: at once (a desktop tab and a headset) would otherwise both pick the
        #: same free number. Its own lock, so a write never holds up a poll.
        self.snapshot_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Page recordings
    # ------------------------------------------------------------------
    def playblast(self) -> "PreviewPlayblast":
        # Double-checked, on a lock of its OWN. The race being closed is two
        # first-accesses building two recorders (this server answers each
        # request on its own thread), which lasts exactly as long as the first
        # request -- but the recorder is read once per FRAME, and taking the
        # server's lock here put every frame POST of a recording behind the same
        # lock the manifest poll uses. Measured: a 151-frame recording went from
        # 5s to 22s and then died with the accept queue full
        # (ERR_CONNECTION_TIMED_OUT part way through). The common path must not
        # lock at all; the attribute read and write are each atomic.
        if self.recorder is None:
            with self.recorder_lock:
                if self.recorder is None:
                    from pythontk.net_utils.preview.playblast import PreviewPlayblast

                    self.recorder = PreviewPlayblast()
        return self.recorder

    def finish_playblast(
        self, token: str, target: Optional[str] = None
    ) -> Dict[str, Any]:
        server = self._server
        # Named for the deliverable rather than for the clip alone: a folder of
        # shot movies from three different pushes is otherwise unreadable.
        #
        # Composed here and NOT accepted from the caller: this method's caller
        # is the served page, and a name it chose would reach ``os.path.join``.
        # (``finish`` sanitizes a stem as well -- this is the half that keeps
        # the page from naming the file at all.)
        output_dir, base = self.page_output()
        recorder = server.playblast
        name = recorder.clip_name(token)
        stem = f"{base}_{name}" if base and base not in name else name
        # Encoded where it can always be written -- the frames go with the
        # encode, so a refused folder cannot be retried -- then delivered.
        report = recorder.finish(
            token, output_dir=str(server.root), target=target, stem=stem
        )
        path = Path(report["output"])
        if output_dir != server.root:
            placed, reason = FileUtils.deliver_file(
                str(path), str(output_dir / path.name)
            )
            if Path(placed) == path:
                self.fell_back(output_dir, reason)
            else:
                path.unlink(missing_ok=True)
                path = Path(placed)
            report["output"] = str(path)
        with server._lock:
            self.recordings[token] = path
            # Bounded: the map only exists so a download can follow a finish,
            # and a preview server lives for a whole DCC session.
            for stale in list(self.recordings)[: -server.MAX_RECORDINGS]:
                self.recordings.pop(stale, None)
        report["url"] = f"{server.url}{PLAYBLAST_PATH}/{token}"
        report["in_serve_root"] = path.parent == server.root
        server.logger.info("Recording written to %s", path)
        return report

    def recording_path(self, token: str) -> Optional[Path]:
        with self._server._lock:
            path = self.recordings.get(str(token))
        return path if path is not None and path.is_file() else None

    # ------------------------------------------------------------------
    # Page stills
    # ------------------------------------------------------------------
    def save_snapshot(
        self, data: bytes, content_type: str = "image/png"
    ) -> Dict[str, Any]:
        server = self._server
        mime = str(content_type or "").split(";", 1)[0].strip().lower()
        spec = server.SNAPSHOT_TYPES.get(mime)
        if spec is None:
            raise ValueError(
                f"unsupported image type {content_type!r}; expected one of "
                f"{', '.join(sorted(server.SNAPSHOT_TYPES))}"
            )
        extension, signature = spec
        if not data:
            raise ValueError("the image arrived empty")
        if len(data) > server.MAX_SNAPSHOT_BYTES:
            raise ValueError(
                f"the image is {len(data)} bytes; the ceiling is "
                f"{server.MAX_SNAPSHOT_BYTES}"
            )
        if not data.startswith(signature):
            raise ValueError(f"the body is not a {mime} image")

        directory, base = self.page_output()
        stem = f"{base}_view" if base else "view"

        def write(folder: Path) -> Path:
            with self.snapshot_lock:
                path = Path(
                    FileUtils.next_version_path(
                        str(folder / f"{stem}{extension}"),
                        format="{stem}_{n:03d}{ext}",
                    )
                )
                # Whole, then moved into place: the serve root is being read
                # by a page, and beside a deliverable the folder is the user's
                # -- which is also why a failed write takes its partial with it.
                part = path.with_name(path.name + ".part")
                try:
                    part.write_bytes(data)
                    FileUtils.replace_file(part, path)
                except OSError:
                    part.unlink(missing_ok=True)
                    raise
            return path

        try:
            path = write(directory)
        except OSError as e:
            if directory == server.root:
                raise
            self.fell_back(directory, e)
            path = write(server.root)
        in_serve_root = path.parent == server.root
        server.logger.info("Image written to %s", path)
        return {
            "output": str(path),
            "name": path.name,
            "url": quote(path.name) if in_serve_root else None,
            "in_serve_root": in_serve_root,
            "bytes": len(data),
        }

    def page_output(self) -> Tuple[Path, str]:
        """``(directory, base)`` for a file the page asks this server to write.

        The one placement rule every page output follows -- a recording and a
        still alike -- so the two cannot disagree about where "beside the
        deliverable" is: the directory is
        :meth:`PreviewPlayblast.resolve_output_dir`'s answer, and *base* the
        name a file there is labelled with. It is where a write is tried
        first; a folder that refuses the write sends it to the serve root
        (:meth:`fell_back`).

        *base* is the SOURCE's stem, not the served asset's: the asset is
        republished under a stable ``scene.glb`` so a page can keep one URL
        across pushes, which is exactly the property that makes it useless as a
        label -- every output of every deliverable would be called ``scene_*``.
        Sanitized, because the page's outputs are the one place a filename is
        built without the caller naming it.
        """
        from pythontk.net_utils.preview.playblast import PreviewPlayblast

        server = self._server
        with server._lock:
            source = server._source
        directory = PreviewPlayblast.resolve_output_dir(source, server.root)
        # A label only when there IS a deliverable on disk. A scene push's
        # source is the bridge's scratch payload, moved into the serve root on
        # publish: its stem is a random tag, and outputs named after it
        # restarted their numbering under a new unreadable prefix every push.
        stem = source.stem if source is not None and directory != server.root else ""
        return directory, StrUtils.to_legal_name(stem)

    def fell_back(self, directory: Path, reason: Any) -> None:
        """Say that a page output meant for *directory* went to the serve root.

        A deliverable on a read-only share: its outputs still belong to the
        page, and the serve root is where the page can collect them. Found by
        the write itself (``os.access`` answers from mode bits alone, and on
        Windows says yes to a read-only share or a denying ACL)."""
        self._server.logger.warning(
            "%s took no file (%s); page output goes to the preview's own "
            "folder instead, from which the page downloads it.",
            directory,
            reason,
        )
