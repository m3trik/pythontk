# !/usr/bin/python
# coding=utf-8
"""Read a file by ``http(s)`` URL with the same surface as a local read.

The one place the ecosystem opens a URL for its bytes. Loaders that accept a
path (the shot-manifest CSV parser, the app installer's download) route a URL
through here instead of growing their own ``urlopen`` call, so the timeout,
the user agent, the sign-in-wall check and the share-link rewrites live once.

Google Sheets is the motivating case: a sheet shared as *anyone with the link*
serves plain CSV from its ``/export?format=csv`` endpoint, so a pasted share
link is a data source with no API client, no OAuth and no dependency.
:meth:`RemoteFile.normalize` turns the share link into that endpoint; a sheet
that is NOT shared answers the export URL with a sign-in page (HTTP 200,
``text/html``), which :meth:`RemoteFile.read_bytes` refuses with a message
that says so. Private sheets stay out of scope by design: an OAuth client
would drag a dependency into pythontk.
"""

from __future__ import annotations

import re
from http.client import HTTPException
from typing import Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

__all__ = ["RemoteFile"]


class _RemoteFileInternal(object):
    """Helpers behind :class:`RemoteFile`."""

    # ``/spreadsheets[/u/<n>]/d/<id>[/<verb>]`` (``/u/<n>`` = the signed-in
    # account index a multi-account browser adds) and the published form
    # ``/d/e/<id>/<verb>``.
    _SHEETS_PATH_RE = re.compile(
        r"^/spreadsheets(?:/u/\d+)?/d/(e/)?([^/]+)(?:/([^/]*))?/?$"
    )
    _SHEETS_HOST = "docs.google.com"
    # Verbs that already return a download; left untouched.
    _SHEETS_DOWNLOAD_VERBS = ("export", "pub")

    @staticmethod
    def _query_gid(parts) -> Optional[str]:
        """The ``gid`` (tab id) from the fragment, else the query, else None."""
        for raw in (parts.fragment, parts.query):
            gid = parse_qs(raw).get("gid")
            if gid:
                return gid[0]
        return None

    @classmethod
    def _normalize_sheets(cls, parts) -> Optional[str]:
        """Rewrite a Google Sheets share link to its CSV download endpoint.

        Returns None when *parts* is not a Sheets document URL, or already
        points at a download form.
        """
        match = cls._SHEETS_PATH_RE.match(parts.path)
        if not match:
            return None
        published, doc_id, verb = match.groups()
        if verb in cls._SHEETS_DOWNLOAD_VERBS:
            return None
        if published:
            path = f"/spreadsheets/d/e/{doc_id}/pub"
            query: Dict[str, str] = {"output": "csv"}
        else:
            path = f"/spreadsheets/d/{doc_id}/export"
            query = {"format": "csv"}
        gid = cls._query_gid(parts)
        if gid is not None:
            query["gid"] = gid
        return urlunsplit((parts.scheme, parts.netloc, path, urlencode(query), ""))

    @classmethod
    def _sharing_hint(cls, url: str) -> str:
        """The remedy for a sign-in wall, specific to the host when known."""
        if urlsplit(url).netloc.lower() == cls._SHEETS_HOST:
            return (
                "For a Google Sheet, share it as 'Anyone with the link can view' "
                "(or File > Share > Publish to web) and reload."
            )
        return "The link is probably behind a sign-in; use a publicly readable URL."


class RemoteFile(_RemoteFileInternal):
    """Fetch the bytes behind an ``http(s)`` URL, with share links normalized.

    Class-level policy (override on a subclass or pass per call):

    Attributes:
        TIMEOUT: Seconds before a stalled connection raises. Short on purpose:
            the DCC panels fetch on their main thread.
        USER_AGENT: Sent with every request.

    Example (a manifest CSV that is a shared Google Sheet):
        >>> raw = RemoteFile.read_bytes(
        ...     "https://docs.google.com/spreadsheets/d/1AbC/edit#gid=0"
        ... )
    """

    TIMEOUT: float = 10.0
    USER_AGENT: str = "pythontk/RemoteFile"

    class Error(OSError):
        """A fetch that did not yield the file: network, HTTP status, or a
        page served where a file was expected. The message is user-facing."""

    @staticmethod
    def is_url(source: str) -> bool:
        """True when *source* is an ``http`` or ``https`` URL.

        Checks the scheme by name: ``urlsplit`` reports a Windows drive letter
        (``C:\\...``) as scheme ``c``, so "has a scheme" is not a URL test.
        """
        if not source:
            return False
        parts = urlsplit(str(source).strip())
        return parts.scheme.lower() in ("http", "https") and bool(parts.netloc)

    @classmethod
    def normalize(cls, url: str) -> str:
        """The download form of *url*; unchanged when no rewrite applies.

        Google Sheets share links (``/spreadsheets/d/<id>/edit#gid=<n>``, the
        bare ``/d/<id>``, the account-indexed ``/spreadsheets/u/<n>/d/<id>``,
        or a published ``/d/e/<id>/pubhtml``) become their CSV export
        endpoint, keeping the tab (``gid``) when the link names one.  A link
        that already points at ``/export`` or ``/pub`` passes through.
        """
        url = str(url).strip()
        parts = urlsplit(url)
        if parts.netloc.lower() == cls._SHEETS_HOST:
            rewritten = cls._normalize_sheets(parts)
            if rewritten:
                return rewritten
        return url

    @classmethod
    def open(
        cls,
        url: str,
        *,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Open *url* (normalized) and return the response for streaming.

        The caller owns the response: use it as a context manager so the
        socket is released on the partial-read path too.

        Parameters:
            url: Address to fetch; passed through :meth:`normalize`.
            timeout: Seconds before a stalled connection fails (default
                :attr:`TIMEOUT`).
            headers: Extra request headers; ``User-Agent`` defaults to
                :attr:`USER_AGENT`.

        Raises:
            RemoteFile.Error: A scheme other than ``http``/``https``,
                DNS/connection/timeout failure, or a non-2xx status.
        """
        target = cls.normalize(url)
        # The one choke point every fetch passes through, so the http(s)
        # contract is enforced here rather than trusted to callers.
        # ``urlopen`` speaks ``file:``, ``ftp:`` and ``data:`` natively, so
        # without this a class documented as "read a file by http(s) URL" --
        # and published as ``ptk.RemoteFile`` -- returns the bytes of a local
        # path handed to it as ``file:///...``. ``is_url`` already answered
        # this correctly and was already tested doing so; nothing asked it.
        if not cls.is_url(target):
            raise cls.Error(
                f"Can't fetch {target!r}: only http and https URLs are "
                "supported. A local path is read with the normal file API."
            )
        merged = {"User-Agent": cls.USER_AGENT}
        merged.update(headers or {})
        request = Request(target, headers=merged)
        try:
            return urlopen(request, timeout=cls.TIMEOUT if timeout is None else timeout)
        except HTTPError as exc:
            raise cls.Error(
                f"Can't fetch {target}: HTTP {exc.code} {exc.reason}."
            ) from exc
        except URLError as exc:
            raise cls.Error(
                f"Can't fetch {target}: {exc.reason}. Check the connection and "
                f"that the link is correct, then reload."
            ) from exc
        except (OSError, ValueError) as exc:  # socket timeout, malformed URL
            raise cls.Error(f"Can't fetch {target}: {exc}") from exc

    @classmethod
    def read_bytes(
        cls,
        url: str,
        *,
        timeout: Optional[float] = None,
        reject_html: bool = True,
    ) -> bytes:
        """The full body behind *url*.

        Parameters:
            url: Address to fetch; passed through :meth:`normalize`.
            timeout: See :meth:`open`.
            reject_html: Refuse a ``text/html`` body. A file endpoint that
                answers with a page is a sign-in wall, and letting that HTML
                reach a parser yields a baffling "header not found" instead
                of the real cause.

        Raises:
            RemoteFile.Error: As :meth:`open`, plus the HTML refusal.
        """
        target = cls.normalize(url)
        with cls.open(target, timeout=timeout) as response:
            if reject_html:
                cls._reject_page(response, target, url)
            # The body read can fail after a clean connect (peer reset, stall,
            # truncated chunked body); keep that inside Error too, so a caller
            # branching on Error vs OSError never gets a bare socket error.
            try:
                return response.read()
            except (OSError, HTTPException) as exc:
                raise cls.Error(f"Can't fetch {target}: {exc}") from exc

    @classmethod
    def probe(cls, url: str, *, timeout: Optional[float] = None) -> Optional[str]:
        """Why *url* would NOT serve a file, or None when it would.

        Opens the URL and closes it without reading the body: a cheap
        reachability plus sign-in-wall check for live validation, sharing
        :meth:`open`'s error messages.  ``None`` means :meth:`read_bytes`
        can be expected to succeed.

        Parameters:
            url: Address to check; passed through :meth:`normalize`.
            timeout: See :meth:`open`.
        """
        target = cls.normalize(url)
        try:
            with cls.open(target, timeout=timeout) as response:
                cls._reject_page(response, target, url)
        except cls.Error as exc:
            return str(exc)
        return None

    @classmethod
    def _reject_page(cls, response, target: str, url: str) -> None:
        """Raise :class:`Error` when *response* is a web page, not a file."""
        if response.headers.get_content_type() == "text/html":
            raise cls.Error(
                f"{target} returned a web page instead of a file. "
                f"{cls._sharing_hint(url)}"
            )
