# !/usr/bin/python
# coding=utf-8
import sys
import unittest
import os
import json
import shutil
import tempfile
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch, MagicMock
from pythontk.core_utils.app_installer import AppInstaller


class TestAppInstaller(unittest.TestCase):
    """Tests for AppInstaller — all network-free via mocks and local archives."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="appinstaller_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_zip(self, inner_path: str, content: bytes = b"fake-exe") -> str:
        """Create a zip at self.tmp/test.zip containing a file at *inner_path*."""
        zip_path = os.path.join(self.tmp, "test.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(inner_path, content)
        return zip_path

    def _make_platforms(self, url="https://example.com/tool.zip"):
        """Return a platforms dict pointing at a fake URL."""
        return {
            "windows": {"url": url, "type": "zip"},
            "linux": {"url": url, "type": "zip"},
            "darwin": {"url": url, "type": "zip"},
        }

    def _mock_urlopen(self, archive_path):
        """Return a context-manager-compatible mock for urlopen."""
        with open(archive_path, "rb") as fh:
            data = fh.read()

        resp = MagicMock()
        resp.headers = {"Content-Length": str(len(data))}
        resp.read = MagicMock(side_effect=[data, b""])
        return resp

    # ------------------------------------------------------------------
    # _resolve_location
    # ------------------------------------------------------------------

    def test_resolve_location_user(self):
        path = AppInstaller._resolve_location("user")
        self.assertIn(".pythontk", path)
        self.assertTrue(path.endswith("tools"))

    def test_resolve_location_local(self):
        path = AppInstaller._resolve_location("local")
        self.assertTrue(path.endswith("bin"))

    def test_resolve_location_temp(self):
        path = AppInstaller._resolve_location("temp")
        self.assertIn("pythontk_tools", path)

    def test_resolve_location_env_override(self):
        custom = os.path.join(self.tmp, "custom_tools")
        with patch.dict(os.environ, {"PYTHONTK_TOOLS_DIR": custom}):
            self.assertEqual(AppInstaller._resolve_location("user"), custom)
            self.assertEqual(AppInstaller._resolve_location("local"), custom)

    def test_resolve_location_invalid_raises(self):
        with self.assertRaises(ValueError):
            AppInstaller._resolve_location("invalid")

    # ------------------------------------------------------------------
    # _current_platform
    # ------------------------------------------------------------------

    def test_current_platform_returns_lowercase(self):
        plat = AppInstaller._current_platform()
        self.assertIn(plat, ("windows", "linux", "darwin"))

    # ------------------------------------------------------------------
    # _find_executable
    # ------------------------------------------------------------------

    def test_find_executable_nested(self):
        """Exe buried several levels deep is found."""
        nested = os.path.join(self.tmp, "a", "b", "c")
        os.makedirs(nested)
        exe = os.path.join(nested, "mytool.exe")
        with open(exe, "w") as f:
            f.write("x")
        result = AppInstaller._find_executable(self.tmp, "mytool")
        self.assertIsNotNone(result)
        self.assertTrue(result.lower().endswith("mytool.exe"))

    def test_find_executable_exact_name(self):
        """Exact name match without .exe extension."""
        os.makedirs(os.path.join(self.tmp, "bin"))
        exe = os.path.join(self.tmp, "bin", "ffmpeg")
        with open(exe, "w") as f:
            f.write("x")
        result = AppInstaller._find_executable(self.tmp, "ffmpeg")
        self.assertIsNotNone(result)

    def test_find_executable_not_found(self):
        result = AppInstaller._find_executable(self.tmp, "nonexistent")
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # _extract (zip)
    # ------------------------------------------------------------------

    def test_extract_zip(self):
        zip_path = self._make_zip("bin/tool.exe", b"\x00" * 10)
        dest = os.path.join(self.tmp, "extracted")
        os.makedirs(dest)
        AppInstaller._extract(zip_path, dest, "zip")
        self.assertTrue(os.path.isfile(os.path.join(dest, "bin", "tool.exe")))

    # ------------------------------------------------------------------
    # _verify_hash
    # ------------------------------------------------------------------

    def test_verify_hash_pass(self):
        import hashlib

        content = b"hello world"
        path = os.path.join(self.tmp, "file.bin")
        with open(path, "wb") as f:
            f.write(content)
        h = hashlib.sha256(content).hexdigest()
        # Should not raise
        AppInstaller._verify_hash(path, h)

    def test_verify_hash_fail(self):
        path = os.path.join(self.tmp, "file.bin")
        with open(path, "wb") as f:
            f.write(b"data")
        with self.assertRaises(RuntimeError):
            AppInstaller._verify_hash(path, "0" * 64)
        # File should be deleted on mismatch
        self.assertFalse(os.path.exists(path))

    # ------------------------------------------------------------------
    # _download (mocked)
    # ------------------------------------------------------------------

    def test_download_with_callback(self):
        """Progress callback receives (downloaded, total) tuple."""
        zip_path = self._make_zip("bin/x.exe")
        resp = self._mock_urlopen(zip_path)

        progress_calls = []

        def cb(downloaded, total):
            progress_calls.append((downloaded, total))

        dest = os.path.join(self.tmp, "out.zip")
        with patch("pythontk.core_utils.app_installer.urlopen", return_value=resp):
            AppInstaller._download("https://example.com/x.zip", dest, cb)

        self.assertTrue(os.path.isfile(dest))
        self.assertGreater(len(progress_calls), 0)
        # Last call should have downloaded == total
        last_dl, total = progress_calls[-1]
        self.assertEqual(last_dl, total)

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------

    def test_catalog_roundtrip(self):
        AppInstaller._catalog_write(self.tmp, "mytool", "/path/to/exe", "1.0")
        entry = AppInstaller._catalog_read(self.tmp, "mytool")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["path"], "/path/to/exe")
        self.assertEqual(entry["version"], "1.0")

    def test_catalog_read_missing(self):
        entry = AppInstaller._catalog_read(self.tmp, "nothere")
        self.assertIsNone(entry)

    def test_catalog_multiple_tools(self):
        AppInstaller._catalog_write(self.tmp, "a", "/a", "1")
        AppInstaller._catalog_write(self.tmp, "b", "/b", "2")
        self.assertEqual(AppInstaller._catalog_read(self.tmp, "a")["path"], "/a")
        self.assertEqual(AppInstaller._catalog_read(self.tmp, "b")["path"], "/b")

    # ------------------------------------------------------------------
    # get_path
    # ------------------------------------------------------------------

    def test_get_path_finds_on_system_path(self):
        """If shutil.which finds it, get_path returns the result."""
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            result = AppInstaller.get_path("ffmpeg")
        self.assertEqual(result, "/usr/bin/ffmpeg")

    def test_get_path_finds_in_catalog(self):
        """Falls back to managed catalog when not on PATH."""
        exe = os.path.join(self.tmp, "mytool.exe")
        with open(exe, "w") as f:
            f.write("x")
        AppInstaller._catalog_write(self.tmp, "mytool", exe, "1.0")

        with patch("shutil.which", return_value=None):
            with patch.object(AppInstaller, "_resolve_location", return_value=self.tmp):
                result = AppInstaller.get_path("mytool")
        self.assertEqual(result, exe)

    def test_get_path_returns_none_when_not_found(self):
        with patch("shutil.which", return_value=None):
            with patch.object(AppInstaller, "_resolve_location", return_value=self.tmp):
                result = AppInstaller.get_path("nonexistent")
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # add_to_path
    # ------------------------------------------------------------------

    def test_get_path_add_to_path_true(self):
        """add_to_path=True injects the exe's parent dir into os.environ PATH."""
        sub = os.path.join(self.tmp, "deep", "bin")
        os.makedirs(sub)
        exe = os.path.join(sub, "mytool.exe")
        with open(exe, "w") as f:
            f.write("x")
        AppInstaller._catalog_write(self.tmp, "mytool", exe, "1.0")

        original_path = os.environ.get("PATH", "")
        try:
            with patch("shutil.which", return_value=None):
                with patch.object(
                    AppInstaller, "_resolve_location", return_value=self.tmp
                ):
                    AppInstaller.get_path("mytool", add_to_path=True)
            self.assertIn(sub.lower(), os.environ["PATH"].lower())
        finally:
            os.environ["PATH"] = original_path

    def test_get_path_add_to_path_false(self):
        """add_to_path=False (default) does NOT modify os.environ PATH."""
        sub = os.path.join(self.tmp, "notouch", "bin")
        os.makedirs(sub)
        exe = os.path.join(sub, "mytool2.exe")
        with open(exe, "w") as f:
            f.write("x")
        AppInstaller._catalog_write(self.tmp, "mytool2", exe, "1.0")

        original_path = os.environ.get("PATH", "")
        try:
            with patch("shutil.which", return_value=None):
                with patch.object(
                    AppInstaller, "_resolve_location", return_value=self.tmp
                ):
                    AppInstaller.get_path("mytool2")
            self.assertNotIn(sub.lower(), os.environ["PATH"].lower())
        finally:
            os.environ["PATH"] = original_path

    def test_ensure_add_to_path_false(self):
        """ensure(add_to_path=False) skips PATH injection."""
        zip_path = self._make_zip("bin/nopath.exe", b"FAKE")
        resp = self._mock_urlopen(zip_path)
        install_dir = os.path.join(self.tmp, "nopath_dir")

        original_path = os.environ.get("PATH", "")
        try:
            with patch("pythontk.core_utils.app_installer.urlopen", return_value=resp):
                with patch.object(
                    AppInstaller, "_resolve_location", return_value=install_dir
                ):
                    with patch("shutil.which", return_value=None):
                        result = AppInstaller.ensure(
                            "nopath",
                            platforms=self._make_platforms(),
                            add_to_path=False,
                        )
            exe_dir = os.path.dirname(result).lower()
            self.assertNotIn(exe_dir, os.environ["PATH"].lower())
        finally:
            os.environ["PATH"] = original_path

    # ------------------------------------------------------------------
    # ensure (integration, mocked network)
    # ------------------------------------------------------------------

    def test_ensure_downloads_and_extracts(self):
        """Full lifecycle: download → extract → discover → catalog."""
        zip_path = self._make_zip("nested/bin/cooltool.exe", b"FAKEBIN")
        resp = self._mock_urlopen(zip_path)
        install_dir = os.path.join(self.tmp, "managed")

        with patch("pythontk.core_utils.app_installer.urlopen", return_value=resp):
            with patch.object(
                AppInstaller, "_resolve_location", return_value=install_dir
            ):
                with patch("shutil.which", return_value=None):
                    result = AppInstaller.ensure(
                        "cooltool",
                        platforms=self._make_platforms(),
                        version="2.0",
                    )

        self.assertTrue(os.path.isfile(result))
        self.assertIn("cooltool", result.lower())

        # Catalog entry created
        entry = AppInstaller._catalog_read(install_dir, "cooltool")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["version"], "2.0")

    def test_ensure_skips_download_when_on_path(self):
        """If already on PATH, no download occurs."""
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            result = AppInstaller.ensure("ffmpeg", platforms=self._make_platforms())
        self.assertEqual(result, "/usr/bin/ffmpeg")

    def test_ensure_raises_for_unknown_platform(self):
        with self.assertRaises(LookupError):
            AppInstaller.ensure("tool", platforms={"fakeos": {"url": "x"}})

    def test_ensure_with_hash_verification(self):
        """SHA-256 is checked when provided."""
        import hashlib

        zip_path = self._make_zip("bin/verified.exe", b"VERIFIED")
        with open(zip_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()

        resp = self._mock_urlopen(zip_path)
        install_dir = os.path.join(self.tmp, "hashed")
        plat = AppInstaller._current_platform()

        with patch("pythontk.core_utils.app_installer.urlopen", return_value=resp):
            with patch.object(
                AppInstaller, "_resolve_location", return_value=install_dir
            ):
                with patch("shutil.which", return_value=None):
                    result = AppInstaller.ensure(
                        "verified",
                        platforms=self._make_platforms(),
                        sha256={plat: h},
                    )

        self.assertTrue(os.path.isfile(result))

    def test_ensure_update_redownloads(self):
        """update=True triggers a fresh download even if already installed."""
        # First install
        zip_path = self._make_zip("bin/upd.exe", b"V1")
        resp1 = self._mock_urlopen(zip_path)
        install_dir = os.path.join(self.tmp, "upd")

        with patch("pythontk.core_utils.app_installer.urlopen", return_value=resp1):
            with patch.object(
                AppInstaller, "_resolve_location", return_value=install_dir
            ):
                with patch("shutil.which", return_value=None):
                    AppInstaller.ensure(
                        "upd", platforms=self._make_platforms(), version="1"
                    )

        # Update
        zip_path2 = self._make_zip("bin/upd.exe", b"V2")
        resp2 = self._mock_urlopen(zip_path2)

        with patch("pythontk.core_utils.app_installer.urlopen", return_value=resp2):
            with patch.object(
                AppInstaller, "_resolve_location", return_value=install_dir
            ):
                with patch("shutil.which", return_value=None):
                    result = AppInstaller.ensure(
                        "upd",
                        platforms=self._make_platforms(),
                        version="2",
                        update=True,
                    )

        self.assertTrue(os.path.isfile(result))
        entry = AppInstaller._catalog_read(install_dir, "upd")
        self.assertEqual(entry["version"], "2")

    # ------------------------------------------------------------------
    # URL handling
    # ------------------------------------------------------------------

    def test_guess_archive_type_with_query_params(self):
        """Query strings shouldn't confuse archive type detection."""
        url = "https://example.com/tool.tar.gz?token=abc&v=1"
        self.assertEqual(AppInstaller._guess_archive_type(url), "tar.gz")

    def test_filename_from_url_strips_query(self):
        url = "https://example.com/path/tool-1.0.zip?sig=xyz"
        self.assertEqual(AppInstaller._filename_from_url(url), "tool-1.0.zip")

    # ------------------------------------------------------------------
    # Tar path-traversal protection
    # ------------------------------------------------------------------

    def test_extract_tar_rejects_path_traversal(self):
        """Tar members with '../' must be rejected."""
        import io
        import tarfile as _tf

        tar_path = os.path.join(self.tmp, "evil.tar")
        with _tf.open(tar_path, "w") as tf:
            info = _tf.TarInfo(name="../../../etc/passwd")
            info.size = 5
            tf.addfile(info, io.BytesIO(b"owned"))

        dest = os.path.join(self.tmp, "safe_dest")
        os.makedirs(dest)
        with self.assertRaises(RuntimeError):
            AppInstaller._extract(tar_path, dest, "tar")

    def test_extract_zip_rejects_path_traversal(self):
        """Zip members with '../' must be rejected (zip-slip)."""
        zip_path = os.path.join(self.tmp, "evil.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../../../etc/shadow", "owned")

        dest = os.path.join(self.tmp, "safe_dest_zip")
        os.makedirs(dest)
        with self.assertRaises(RuntimeError):
            AppInstaller._extract(zip_path, dest, "zip")

    def test_extract_zip_allows_normal_members(self):
        """Normal zip members still extract fine after zip-slip protection."""
        zip_path = self._make_zip("subdir/tool.exe", b"OK")
        dest = os.path.join(self.tmp, "normal_zip")
        os.makedirs(dest)
        AppInstaller._extract(zip_path, dest, "zip")
        self.assertTrue(os.path.isfile(os.path.join(dest, "subdir", "tool.exe")))

    # ------------------------------------------------------------------
    # resolve_ffmpeg catalog fallback
    # ------------------------------------------------------------------

    def test_resolve_ffmpeg_finds_catalog_install(self):
        """resolve_ffmpeg should find ffmpeg from a previous managed install
        even when auto_install=False and ffmpeg is not on the system PATH."""
        from pythontk.audio_utils._audio_utils import AudioUtils

        exe = os.path.join(self.tmp, "ffmpeg.exe")
        with open(exe, "w") as f:
            f.write("x")

        with patch("shutil.which", return_value=None):
            with patch.object(AppInstaller, "_resolve_location", return_value=self.tmp):
                # Write a catalog entry as if a previous session installed it
                AppInstaller._catalog_write(self.tmp, "ffmpeg", exe, "7.0")
                result = AudioUtils.resolve_ffmpeg(required=False)

        self.assertEqual(result, exe)


if __name__ == "__main__":
    unittest.main()
