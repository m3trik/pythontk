# !/usr/bin/python
# coding=utf-8
"""Tests for ``pythontk.net_utils.remote_file.RemoteFile``.

No network: ``urlopen`` is patched at the module seam, which is also what the
app-installer tests patch, so one fake response shape serves every caller.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pythontk.net_utils.remote_file import RemoteFile  # noqa: E402

_URLOPEN = "pythontk.net_utils.remote_file.urlopen"
_SHEET = "https://docs.google.com/spreadsheets/d/1AbC_dEf-9/"


def _response(body=b"a,b\n1,2\n", content_type="text/csv"):
    """A context-manager-compatible stand-in for an ``http.client`` response."""
    resp = MagicMock()
    resp.headers.get_content_type.return_value = content_type
    resp.read.return_value = body
    resp.__enter__.return_value = resp
    return resp


class IsUrlTest(unittest.TestCase):
    def test_http_and_https_are_urls(self):
        self.assertTrue(RemoteFile.is_url("http://example.com/x.csv"))
        self.assertTrue(RemoteFile.is_url("HTTPS://example.com/x.csv"))
        self.assertTrue(RemoteFile.is_url("  https://example.com/x.csv  "))

    def test_paths_are_not_urls(self):
        """A drive letter parses as scheme ``c``; a scheme is not a URL."""
        self.assertFalse(RemoteFile.is_url(r"C:\seq\manifest.csv"))
        self.assertFalse(RemoteFile.is_url("C:/seq/manifest.csv"))
        self.assertFalse(RemoteFile.is_url(r"\\server\share\m.csv"))
        self.assertFalse(RemoteFile.is_url("/home/u/m.csv"))
        self.assertFalse(RemoteFile.is_url("manifest.csv"))
        self.assertFalse(RemoteFile.is_url(""))
        self.assertFalse(RemoteFile.is_url(None))

    def test_other_schemes_are_not_urls(self):
        self.assertFalse(RemoteFile.is_url("file:///C:/seq/m.csv"))
        self.assertFalse(RemoteFile.is_url("ftp://host/m.csv"))
        self.assertFalse(RemoteFile.is_url("https:///no-host.csv"))


class NormalizeTest(unittest.TestCase):
    def test_share_link_with_fragment_gid(self):
        self.assertEqual(
            RemoteFile.normalize(_SHEET + "edit#gid=1234"),
            _SHEET + "export?format=csv&gid=1234",
        )

    def test_share_link_with_query_gid_and_usp(self):
        self.assertEqual(
            RemoteFile.normalize(_SHEET + "edit?usp=sharing&gid=7#gid=7"),
            _SHEET + "export?format=csv&gid=7",
        )

    def test_bare_document_link_defaults_to_first_tab(self):
        self.assertEqual(
            RemoteFile.normalize(_SHEET.rstrip("/")), _SHEET + "export?format=csv"
        )
        self.assertEqual(RemoteFile.normalize(_SHEET), _SHEET + "export?format=csv")

    def test_account_indexed_link_is_rewritten(self):
        """A multi-account browser inserts ``/u/<n>/``; the export endpoint
        must drop it, or a shared sheet reads as an unshared one."""
        self.assertEqual(
            RemoteFile.normalize(
                "https://docs.google.com/spreadsheets/u/1/d/1AbC/edit#gid=2"
            ),
            "https://docs.google.com/spreadsheets/d/1AbC/export?format=csv&gid=2",
        )

    def test_export_and_pub_links_pass_through(self):
        for url in (
            _SHEET + "export?format=csv&gid=0",
            _SHEET + "export?format=xlsx",
            "https://docs.google.com/spreadsheets/d/e/2PACX-abc/pub?output=csv",
        ):
            self.assertEqual(RemoteFile.normalize(url), url)

    def test_published_html_link_becomes_csv(self):
        self.assertEqual(
            RemoteFile.normalize(
                "https://docs.google.com/spreadsheets/d/e/2PACX-abc/pubhtml#gid=5"
            ),
            "https://docs.google.com/spreadsheets/d/e/2PACX-abc/pub?output=csv&gid=5",
        )

    def test_other_hosts_and_other_google_docs_pass_through(self):
        for url in (
            "https://example.com/seq/manifest.csv?rev=3",
            "https://docs.google.com/document/d/1AbC/edit",
            "https://drive.google.com/file/d/1AbC/view",
        ):
            self.assertEqual(RemoteFile.normalize(url), url)


class OpenTest(unittest.TestCase):
    def test_open_sends_normalized_target_agent_and_timeout(self):
        resp = _response()
        with patch(_URLOPEN, return_value=resp) as urlopen:
            got = RemoteFile.open(_SHEET + "edit#gid=3", timeout=4)
        self.assertIs(got, resp)
        request, kwargs = urlopen.call_args.args[0], urlopen.call_args.kwargs
        self.assertEqual(request.full_url, _SHEET + "export?format=csv&gid=3")
        self.assertEqual(request.get_header("User-agent"), RemoteFile.USER_AGENT)
        self.assertEqual(kwargs["timeout"], 4)

    def test_open_default_timeout_and_header_override(self):
        with patch(_URLOPEN, return_value=_response()) as urlopen:
            RemoteFile.open("https://x.test/a.csv", headers={"User-Agent": "me/1"})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("User-agent"), "me/1")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], RemoteFile.TIMEOUT)

    def test_http_error_becomes_remote_error_with_status(self):
        err = HTTPError("https://x.test/a.csv", 404, "Not Found", None, None)
        with patch(_URLOPEN, side_effect=err):
            with self.assertRaises(RemoteFile.Error) as ctx:
                RemoteFile.open("https://x.test/a.csv")
        self.assertIn("HTTP 404", str(ctx.exception))
        self.assertIsInstance(ctx.exception, OSError)

    def test_url_error_becomes_remote_error_with_reason(self):
        with patch(_URLOPEN, side_effect=URLError("Name or service not known")):
            with self.assertRaises(RemoteFile.Error) as ctx:
                RemoteFile.open("https://nope.test/a.csv")
        self.assertIn("Name or service not known", str(ctx.exception))
        self.assertIn("connection", str(ctx.exception).lower())

    def test_timeout_becomes_remote_error(self):
        with patch(_URLOPEN, side_effect=TimeoutError("timed out")):
            with self.assertRaises(RemoteFile.Error):
                RemoteFile.open("https://slow.test/a.csv")


class ProbeTest(unittest.TestCase):
    """``probe`` answers None for a fetchable file, else the reason, without
    reading the body."""

    def test_reachable_file_is_none_and_body_unread(self):
        resp = _response()
        with patch(_URLOPEN, return_value=resp):
            self.assertIsNone(RemoteFile.probe("https://x.test/m.csv"))
        resp.read.assert_not_called()
        resp.__exit__.assert_called_once()

    def test_sign_in_wall_names_the_remedy(self):
        with patch(_URLOPEN, return_value=_response(b"<html>", "text/html")):
            problem = RemoteFile.probe(_SHEET + "edit#gid=0")
        self.assertIn("Anyone with the link", problem)

    def test_unreachable_host_is_the_open_error(self):
        with patch(_URLOPEN, side_effect=URLError("Name or service not known")):
            problem = RemoteFile.probe("https://nope.test/m.csv")
        self.assertIn("Name or service not known", problem)


class ReadBytesTest(unittest.TestCase):
    def test_csv_body_is_returned(self):
        with patch(_URLOPEN, return_value=_response(b"Step,Asset\nA01,geo\n")):
            self.assertEqual(
                RemoteFile.read_bytes("https://x.test/m.csv"), b"Step,Asset\nA01,geo\n"
            )

    def test_html_body_from_a_sheet_names_the_sharing_remedy(self):
        """An unshared sheet answers its export URL with a sign-in page at
        HTTP 200; that must surface as the cause, not as a parser error."""
        with patch(_URLOPEN, return_value=_response(b"<html>", "text/html")):
            with self.assertRaises(RemoteFile.Error) as ctx:
                RemoteFile.read_bytes(_SHEET + "edit#gid=0")
        msg = str(ctx.exception)
        self.assertIn("web page instead of a file", msg)
        self.assertIn("Anyone with the link", msg)

    def test_html_body_elsewhere_gets_the_generic_hint(self):
        with patch(_URLOPEN, return_value=_response(b"<html>", "text/html")):
            with self.assertRaises(RemoteFile.Error) as ctx:
                RemoteFile.read_bytes("https://intranet.test/m.csv")
        self.assertIn("sign-in", str(ctx.exception))
        self.assertNotIn("Google", str(ctx.exception))

    def test_html_accepted_when_not_rejected(self):
        with patch(_URLOPEN, return_value=_response(b"<html>", "text/html")):
            got = RemoteFile.read_bytes("https://x.test/page", reject_html=False)
        self.assertEqual(got, b"<html>")

    def test_body_read_failure_is_a_remote_error(self):
        """A reset or stall AFTER a clean connect must not escape as a bare
        socket OSError: callers branch on Error vs OSError."""
        resp = _response()
        resp.read.side_effect = ConnectionResetError("peer reset")
        with patch(_URLOPEN, return_value=resp):
            with self.assertRaises(RemoteFile.Error) as ctx:
                RemoteFile.read_bytes("https://x.test/m.csv")
        self.assertIn("peer reset", str(ctx.exception))
        resp.__exit__.assert_called_once()

    def test_response_is_closed_after_read(self):
        resp = _response()
        with patch(_URLOPEN, return_value=resp):
            RemoteFile.read_bytes("https://x.test/m.csv")
        resp.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
