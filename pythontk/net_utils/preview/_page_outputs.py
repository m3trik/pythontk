# !/usr/bin/python
# coding=utf-8
"""Files the served page asks the server to write: recordings and stills.

The page records a clip through the ``playblast`` routes (the recorder,
:class:`~pythontk.net_utils.preview.playblast.PreviewPlayblast`, is built on
first use) and posts a still to ``snapshot``. Both land by one placement rule
(:meth:`_page_output`): beside the published deliverable when it is still on
disk and writable, else in the serve root, from which the page downloads it.
Every name is composed here, never taken from the page.

One job of :class:`~pythontk.PreviewServer`, composed in ``server.py``; its
methods reach the rest of the server through ``self``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING
from urllib.parse import quote

from pythontk.file_utils._file_utils import FileUtils
from pythontk.net_utils.preview.routes import PLAYBLAST_PATH
from pythontk.str_utils._str_utils import StrUtils

if TYPE_CHECKING:  # the recorder is imported on use -- see playblast
    from pythontk.net_utils.preview.playblast import PreviewPlayblast


class _PageOutputsMixin:
    """Page recordings and stills, and where they land.

    A private part of :class:`~pythontk.PreviewServer`; call it through the facade.
    """

    #: Finished recordings kept addressable for download at once.
    MAX_RECORDINGS: int = 8

    #: Image types the page may post a still as: MIME -> (extension, the
    #: signature its bytes must open with). PNG only -- the page's still is
    #: lossless, and the table is the one place another type would be added.
    #: The signature is checked because this route writes a file under a name
    #: the page does not choose, and a body that is not the image it claims to
    #: be should be refused rather than saved as one.
    SNAPSHOT_TYPES: Dict[str, Tuple[str, bytes]] = {
        "image/png": (".png", b"\x89PNG\r\n\x1a\n"),
    }

    #: Refuse a still larger than this. A 4K PNG of a production scene is tens
    #: of megabytes at most; this is a guard against a runaway body, the same
    #: ceiling a recording's frame gets.
    MAX_SNAPSHOT_BYTES: int = 64 * 1024 * 1024

    # ------------------------------------------------------------------
    # Page recordings
    # ------------------------------------------------------------------
    @property
    def playblast(self) -> "PreviewPlayblast":
        """The page's recorder, created on first use.

        Deferred rather than built in ``__init__`` because it allocates a
        scratch store, and the overwhelming majority of preview servers never
        record anything.
        """
        # Under the lock: this server answers each request on its OWN thread, so
        # two frame POSTs racing the first access would each build a recorder --
        # and a frame would then be filed against one the `begin` never reached.
        # Double-checked, on a lock of its OWN. The race being closed is two
        # first-accesses building two recorders (this server answers each
        # request on its own thread), which lasts exactly as long as the first
        # request -- but this property is read once per FRAME, and taking the
        # server's lock here put every frame POST of a recording behind the same
        # lock the manifest poll uses. Measured: a 151-frame recording went from
        # 5s to 22s and then died with the accept queue full
        # (ERR_CONNECTION_TIMED_OUT part way through). The common path must not
        # lock at all; the attribute read and write are each atomic.
        if self._playblast is None:
            with self._playblast_lock:
                if self._playblast is None:
                    from pythontk.net_utils.preview.playblast import PreviewPlayblast

                    self._playblast = PreviewPlayblast()
        return self._playblast

    def begin_playblast(self, **kwargs: Any) -> Dict[str, Any]:
        """Open a recording (see :meth:`PreviewPlayblast.begin`)."""
        return self.playblast.begin(**kwargs)

    def finish_playblast(
        self, token: str, target: Optional[str] = None
    ) -> Dict[str, Any]:
        """Encode a recording and report where it went.

        The movie lands beside the file that was published, when that file is
        still on disk -- an exporter's GLB, or one chosen with External GLB.
        A scene push has no such file (its GLB is the bridge's own scratch,
        released the moment it is published), so the movie goes to the serve
        root instead and the page's ``url`` is how it is collected.

        Returns:
            :meth:`PreviewPlayblast.finish`'s report plus ``"url"`` -- always
            present, and always the way to fetch the file from the page.
        """
        # Named for the deliverable rather than for the clip alone: a folder of
        # shot movies from three different pushes is otherwise unreadable.
        #
        # Composed here and NOT accepted from the caller: this method's caller
        # is the served page, and a name it chose would reach ``os.path.join``.
        # (``finish`` sanitizes a stem as well -- this is the half that keeps
        # the page from naming the file at all.)
        output_dir, base = self._page_output()
        name = self.playblast.clip_name(token)
        stem = f"{base}_{name}" if base and base not in name else name
        report = self.playblast.finish(
            token, output_dir=str(output_dir), target=target, stem=stem
        )
        path = Path(report["output"])
        with self._lock:
            self._recordings[token] = path
            # Bounded: the map only exists so a download can follow a finish,
            # and a preview server lives for a whole DCC session.
            for stale in list(self._recordings)[: -self.MAX_RECORDINGS]:
                self._recordings.pop(stale, None)
        report["url"] = f"{self.url}{PLAYBLAST_PATH}/{token}"
        report["in_serve_root"] = path.parent == self.root
        self.logger.info("Recording written to %s", path)
        return report

    def recording_path(self, token: str) -> Optional[Path]:
        """The file a finished recording produced, or None."""
        with self._lock:
            path = self._recordings.get(str(token))
        return path if path is not None and path.is_file() else None

    # ------------------------------------------------------------------
    # Page stills
    # ------------------------------------------------------------------
    def save_snapshot(
        self, data: bytes, content_type: str = "image/png"
    ) -> Dict[str, Any]:
        """Write a still of the page's view, and report where it went.

        The page renders its current view and posts the image; this decides
        where it lands, by the rule a recording follows (see
        :meth:`finish_playblast`): beside the published file when that file is
        still on disk, else in the serve root, from which the returned ``url``
        fetches it -- the only copy a scene push's still has.

        Numbered, never overwritten, as ``<deliverable>_view_<NNN>.png``: a
        still is taken again from another angle, and the second must not
        replace the first. The name is composed here and not accepted from the
        page, for the reason a recording's is.

        Parameters:
            data: The encoded image.
            content_type: Its MIME type, one of :attr:`SNAPSHOT_TYPES`.

        Returns:
            ``{"output", "name", "url", "in_serve_root", "bytes"}``. ``url`` is
            the still's served path RELATIVE to the page, or None when it
            landed beside the deliverable, off the serve root -- where it is
            already where its owner keeps it. Relative because the page is the
            one that follows it: a browser honours ``download`` only on the
            page's own origin, and the page is as validly open at
            ``localhost`` as at the ``127.0.0.1`` this server's :attr:`url`
            spells -- an absolute link navigated a ``localhost`` tab to the
            bare PNG instead of saving it.

        Raises:
            ValueError: An unsupported type, an empty or oversized body, or
                bytes that are not the image they claim to be.
            OSError: The write itself failed.
        """
        mime = str(content_type or "").split(";", 1)[0].strip().lower()
        spec = self.SNAPSHOT_TYPES.get(mime)
        if spec is None:
            raise ValueError(
                f"unsupported image type {content_type!r}; expected one of "
                f"{', '.join(sorted(self.SNAPSHOT_TYPES))}"
            )
        extension, signature = spec
        if not data:
            raise ValueError("the image arrived empty")
        if len(data) > self.MAX_SNAPSHOT_BYTES:
            raise ValueError(
                f"the image is {len(data)} bytes; the ceiling is "
                f"{self.MAX_SNAPSHOT_BYTES}"
            )
        if not data.startswith(signature):
            raise ValueError(f"the body is not a {mime} image")

        directory, base = self._page_output()
        stem = f"{base}_view" if base else "view"
        with self._snapshot_lock:
            path = Path(
                FileUtils.next_version_path(
                    str(directory / f"{stem}{extension}"),
                    format="{stem}_{n:03d}{ext}",
                )
            )
            # Whole, then moved into place: the serve root is being read by a
            # page, and beside a deliverable the folder is the user's -- which
            # is also why a failed write takes its partial file with it.
            part = path.with_name(path.name + ".part")
            try:
                part.write_bytes(data)
                FileUtils.replace_file(part, path)
            except OSError:
                part.unlink(missing_ok=True)
                raise
        in_serve_root = directory == self.root
        self.logger.info("Image written to %s", path)
        return {
            "output": str(path),
            "name": path.name,
            "url": quote(path.name) if in_serve_root else None,
            "in_serve_root": in_serve_root,
            "bytes": len(data),
        }

    def _page_output(self) -> Tuple[Path, str]:
        """``(directory, base)`` for a file the page asks this server to write.

        The one placement rule every page output follows -- a recording and a
        still alike -- so the two cannot disagree about where "beside the
        deliverable" is: the directory is
        :meth:`PreviewPlayblast.resolve_output_dir`'s answer, and *base* the
        name a file there is labelled with.

        *base* is the SOURCE's stem, not the served asset's: the asset is
        republished under a stable ``scene.glb`` so a page can keep one URL
        across pushes, which is exactly the property that makes it useless as a
        label -- every output of every deliverable would be called ``scene_*``.
        Sanitized, because the page's outputs are the one place a filename is
        built without the caller naming it.
        """
        from pythontk.net_utils.preview.playblast import PreviewPlayblast

        with self._lock:
            source = self._source
        directory = PreviewPlayblast.resolve_output_dir(source, self.root)
        # A label only when there IS a deliverable on disk. A scene push's
        # source is the bridge's scratch payload, moved into the serve root on
        # publish: its stem is a random tag, and outputs named after it
        # restarted their numbering under a new unreadable prefix every push.
        stem = source.stem if source is not None and directory != self.root else ""
        if directory != self.root and not self._writable(directory):
            # A deliverable on a read-only share: its outputs still belong to
            # the page, and the serve root is where the page can collect them.
            self.logger.warning(
                "%s is not writable; page output goes to the preview's own "
                "folder instead, from which the page downloads it.",
                directory,
            )
            directory = self.root
        return directory, StrUtils.to_legal_name(stem)

    @staticmethod
    def _writable(directory: Path) -> bool:
        """Whether a file can be created in *directory* -- found by trying.

        ``os.access`` answers from mode bits alone, and on Windows says yes to
        a read-only share or a denying ACL; the probe is self-cleaning.
        """
        try:
            with tempfile.NamedTemporaryFile(dir=directory):
                return True
        except OSError:
            return False
