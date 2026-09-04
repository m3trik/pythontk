# !/usr/bin/python
# coding=utf-8
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from typing import Callable, Dict, Optional, Union
from urllib.parse import urlparse

from pythontk.net_utils.remote_file import RemoteFile

logger = logging.getLogger(__name__)

# Catalog tracking managed installations
_CATALOG_NAME = ".installed.json"

# ---- Well-known platform definitions (convenience, not required) ----------

FFMPEG_PLATFORMS = {
    "windows": {
        "url": "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
        "type": "zip",
    },
    "linux": {
        "url": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
        "type": "tar.xz",
    },
    "darwin": {
        "url": "https://evermeet.cx/ffmpeg/getrelease/zip",
        "type": "zip",
    },
}


class AppInstaller:
    """Download, extract, and manage external OS-level tool binaries.

    App-agnostic — all tool definitions are supplied by the caller.
    Uses only the Python standard library (no ``requests``).

    Typical usage::

        path = AppInstaller.ensure(
            "ffmpeg",
            platforms={
                "windows": {
                    "url": "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
                    "type": "zip",
                },
                "linux": {
                    "url": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
                    "type": "tar.xz",
                },
            },
        )
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def ensure(
        cls,
        name: str,
        platforms: Dict[str, dict],
        executable: Optional[str] = None,
        version: Optional[str] = None,
        sha256: Optional[Dict[str, str]] = None,
        location: str = "user",
        update: bool = False,
        add_to_path: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """Ensure a tool is available locally, downloading if necessary.

        Parameters:
            name:       Unique tool identifier (e.g. ``"ffmpeg"``).
            platforms:  Mapping of platform name to download info::

                            {"windows": {"url": "...", "type": "zip"},
                             "linux":   {"url": "...", "type": "tar.gz"}}

                        ``"type"`` is an archive kind (``zip``, ``tar.gz``,
                        ``tar.xz``, ``tar.bz2``) or ``"nsis"`` -- a Windows
                        installer, run silently into the tool directory
                        instead of extracted (see
                        :meth:`_run_silent_installer`). Each entry may
                        also include a per-platform ``"executable"``
                        override.
            executable: Binary name to search for after extraction.
                        Defaults to *name*.  On Windows ``.exe`` is
                        appended automatically during search.
            version:    Version string for the tool.  Used to track
                        updates in the install catalog.
            sha256:     Optional per-platform SHA-256 hex digest::

                            {"windows": "abc123...", "linux": "def456..."}

            location:   Where to install.  One of:

                        * ``"user"``  — ``~/.pythontk/tools/`` (default).
                          Override with ``PYTHONTK_TOOLS_DIR`` env-var.
                        * ``"local"`` — ``./bin/`` relative to cwd.
                        * ``"temp"``  — OS temporary directory.
            update:     If *True*, re-download even when already installed
                        (e.g. when *version* is newer).
            add_to_path:
                        If *True* (default), append the executable's
                        parent directory to the current process
                        ``os.environ["PATH"]`` so that
                        ``subprocess.run("tool")`` works immediately.
            progress_callback:
                        ``callback(bytes_downloaded, total_bytes)``.
                        When *None*, a simple ``stdout`` progress line is
                        printed for downloads > 1 MB -- to a terminal only
                        (a DCC script editor gets nothing).

        Returns:
            Absolute path to the tool executable.

        Raises:
            LookupError:  If the current platform has no entry in *platforms*.
            RuntimeError:  If download, extraction, or hash verification fails.
        """
        plat = cls._current_platform()
        plat_info = platforms.get(plat)
        if plat_info is None:
            raise LookupError(
                f"No download defined for platform '{plat}' "
                f"(tool={name!r}, available={list(platforms)})"
            )

        exe_name = plat_info.get("executable", executable or name)

        # Check PATH first (already installed globally).
        existing = cls.get_path(name, location=location, executable=exe_name)
        if existing and not update:
            return existing

        install_dir = cls._resolve_location(location)
        tool_dir = os.path.join(install_dir, f"{name}_{version}" if version else name)

        url = plat_info["url"]
        archive_type = plat_info.get("type", cls._guess_archive_type(url))
        expected_hash = (sha256 or {}).get(plat)

        # Download
        archive_name = cls._filename_from_url(url)
        archive_path = os.path.join(install_dir, archive_name)
        os.makedirs(install_dir, exist_ok=True)
        try:
            cls._download(url, archive_path, progress_callback)
        except Exception:
            # Remove partial download
            if os.path.isfile(archive_path):
                os.remove(archive_path)
            raise

        # Verify integrity
        if expected_hash:
            cls._verify_hash(archive_path, expected_hash)

        # Extract
        if os.path.isdir(tool_dir):
            shutil.rmtree(tool_dir)
        os.makedirs(tool_dir, exist_ok=True)
        # A corrupt or truncated archive raises out of the stdlib as tarfile.ReadError
        # / zipfile.BadZipFile / EOFError -- none of which derive from RuntimeError,
        # OSError or LookupError. Every resolve_* consumer catches that trio because
        # this method's docstring promises RuntimeError, so those escaped all three of
        # them and surfaced to the artist as a raw tar error instead of the fix-shaped
        # 'install KTX-Software' message. Translate once, here, rather than widening
        # three call sites -- and never to a bare except, which would fold a genuine
        # programming error into a misleading 'install failed'.
        try:
            cls._extract(archive_path, tool_dir, archive_type)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Could not extract {archive_name} for '{name}': {exc}"
            ) from exc
        os.remove(archive_path)

        # Discover executable inside extraction
        exe_path = cls._find_executable(tool_dir, exe_name)
        if exe_path is None:
            raise RuntimeError(
                f"Could not locate '{exe_name}' inside extracted archive at {tool_dir}"
            )

        # Ensure executable permission on Unix
        if plat != "windows":
            os.chmod(exe_path, os.stat(exe_path).st_mode | 0o111)

        # Record in the catalog
        cls._catalog_write(install_dir, name, exe_path, version)

        if add_to_path:
            cls._add_to_process_path(exe_path)

        logger.info(f"Installed {name} → {exe_path}")
        return exe_path

    @classmethod
    def get_path(
        cls,
        name: str,
        location: str = "user",
        executable: Optional[str] = None,
        add_to_path: bool = False,
    ) -> Optional[str]:
        """Find a tool without installing.  Checks system PATH then the
        managed install catalog.

        Parameters:
            name:       Tool identifier.
            location:   Which managed location to check (``"user"``,
                        ``"local"``, ``"temp"``).
            executable: Binary name.  Defaults to *name*.
            add_to_path:
                        If *True*, append the executable's parent
                        directory to ``os.environ["PATH"]`` when the
                        tool is found via the managed catalog.  This
                        makes subsequent ``subprocess`` calls work
                        without full paths.  Defaults to *False*.

        Returns:
            Absolute path, or *None* if not found.
        """
        exe_name = executable or name

        # 1. System PATH
        which = shutil.which(exe_name)
        if which:
            return which
        # Also check with .exe on Windows
        if cls._current_platform() == "windows" and not exe_name.endswith(".exe"):
            which = shutil.which(f"{exe_name}.exe")
            if which:
                return which

        # 2. Managed catalog
        install_dir = cls._resolve_location(location)
        entry = cls._catalog_read(install_dir, name)
        if entry:
            path = entry.get("path")
            if path and os.path.isfile(path):
                if add_to_path:
                    cls._add_to_process_path(path)
                return path

        return None

    @staticmethod
    def consent(
        prompt: Union[bool, Callable[[str], bool], None], question: str
    ) -> Optional[bool]:
        """Resolve a caller's *prompt* policy for an install into a yes/no.

        The one consent seam every ``resolve_*(auto_install=True, prompt=...)``
        shares: a GUI plugs its dialog in by passing a callable, a headless
        caller opts out with ``False``, and neither re-implements the console
        fallback.

        Parameters:
            prompt: ``False`` (or None) -- no consent needed (returns True).
                A callable ``(question) -> bool`` -- asked once; its
                truthiness is the verdict (a Qt dialog, a test stub).
                ``True`` -- ask on the console (``[y/N]``).
            question: What to ask, e.g. ``"FBX2glTF v0.13.1 is not installed.
                Download to ~/.pythontk/tools/ now?"``.

        Returns:
            True/False for an answer; **None** when *prompt* is True but no
            interactive console exists (CI, a GUI host, ``pythonw``). That is
            "nobody to ask", not a decline -- callers word it as such and
            point at ``prompt=False``.
        """
        if not prompt:
            return True
        if callable(prompt):
            return bool(prompt(question))
        if not (sys.stdin and sys.stdin.isatty()):
            return None
        sys.stdout.write(f"\n{question} [y/N] ")
        sys.stdout.flush()
        return sys.stdin.readline().strip().lower() in ("y", "yes")

    # ------------------------------------------------------------------
    # Internal — download / verify / extract
    # ------------------------------------------------------------------

    @staticmethod
    def _download(
        url: str,
        dest: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Download *url* to *dest* with chunked reading."""
        try:
            response = RemoteFile.open(
                url, timeout=30, headers={"User-Agent": "pythontk/AppInstaller"}
            )
        except Exception as exc:
            raise RuntimeError(f"Download failed for {url}: {exc}") from exc

        # Close the response (and its underlying socket) deterministically,
        # including on the partial-read/exception path — GC finalization is
        # not guaranteed to reclaim the file descriptor promptly.
        with response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192
            # The ``\r`` progress line is for a terminal only: a DCC's script
            # editor renders every carriage return as its own line (measured:
            # ~600 of them for a 6 MB download) and ``pythonw`` has no stdout.
            use_stdout = (
                progress_callback is None
                and total > 1_048_576
                and bool(getattr(sys.stdout, "isatty", lambda: False)())
            )

            with open(dest, "wb") as fh:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total)
                    elif use_stdout:
                        pct = f"{downloaded * 100 // total}%" if total else "?"
                        mb = downloaded / 1_048_576
                        sys.stdout.write(f"\r  downloading … {mb:.1f} MB ({pct})")
                        sys.stdout.flush()

            if use_stdout:
                sys.stdout.write("\n")

            # http.client.HTTPResponse.read(amt) deliberately does NOT raise
            # IncompleteRead when the peer closes the connection gracefully
            # mid-body, so an empty read is indistinguishable from a clean EOF
            # and a half-received file returns as a successful download.
            # Reconcile against the length the server declared. This matters
            # most for the 'nsis' type, which RUNS the file and re-runs it
            # elevated on WinError 740: without this a truncated installer
            # reached a UAC prompt. A chunked response declares no length, so
            # the check is gated on having one.
            if total and downloaded != total:
                raise RuntimeError(
                    f"Truncated download for {url}: {downloaded} of {total} bytes"
                )

    @staticmethod
    def _verify_hash(file_path: str, expected: str) -> None:
        """Verify SHA-256 digest.  Deletes the file and raises on mismatch."""
        sha = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                sha.update(block)
        digest = sha.hexdigest()
        if digest != expected:
            os.remove(file_path)
            raise RuntimeError(
                f"SHA-256 mismatch for {file_path}: expected {expected}, got {digest}"
            )

    @staticmethod
    def _extract(archive_path: str, dest: str, archive_type: str) -> None:
        """Extract a zip, tar.gz, tar.xz, or tar.bz2 archive -- or, for the
        ``nsis`` type, run the Windows installer silently into *dest*.

        Zip extraction uses :pymethod:`ZipFile.extractall`.
        Tar extraction filters members to prevent path-traversal attacks
        (CVE-2007-4559): absolute paths and ``..`` components are rejected.
        """
        # Deferred because ``file_utils`` itself imports from ``core_utils``,
        # so a module-level import here risks a cycle — the same reason, and
        # the same pattern, as ``preset_store`` and ``package_manager``. (It
        # does not close one today; the convention is what keeps it that way.)
        from pythontk.file_utils._file_utils import FileUtils

        at = archive_type.lower().lstrip(".")
        if at == "zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                # Validate members against zip-slip (path traversal)
                dest_real = os.path.realpath(dest)
                for member in zf.namelist():
                    resolved = os.path.realpath(os.path.join(dest, member))
                    # Both sides are already realpath'd, so is_under's pure
                    # string test is the whole judgement -- and it normalizes
                    # case, which a bare startswith does not: on Windows a
                    # member landing in `C:\Dest\x` under a `C:\dest` root read
                    # as an escape and aborted a legitimate install.
                    if not FileUtils.is_under(resolved, dest_real):
                        raise RuntimeError(
                            f"Zip member {member!r} escapes destination directory"
                        )
                zf.extractall(dest)
        elif at in ("tar.gz", "tgz", "tar.xz", "tar.bz2", "tar"):
            mode = {
                "tar.gz": "r:gz",
                "tgz": "r:gz",
                "tar.xz": "r:xz",
                "tar.bz2": "r:bz2",
                "tar": "r:",
            }.get(at, "r:*")
            with tarfile.open(archive_path, mode) as tf:
                # Use data filter on Python ≥3.12, manual check otherwise
                if hasattr(tarfile, "data_filter"):
                    tf.extractall(dest, filter="data")
                else:
                    dest_real = os.path.realpath(dest)
                    for member in tf.getmembers():
                        resolved = os.path.realpath(os.path.join(dest, member.name))
                        # is_under lands the comparison on a separator
                        # boundary; the bare startswith this replaces let a
                        # sibling directory whose name merely EXTENDS the
                        # destination ("/opt/app-old" under "/opt/app") pass
                        # the traversal guard outright.
                        if not FileUtils.is_under(resolved, dest_real):
                            raise RuntimeError(
                                f"Tar member {member.name!r} escapes "
                                f"destination directory"
                            )
                    tf.extractall(dest)
        elif at == "nsis":
            AppInstaller._run_silent_installer(archive_path, dest)
        else:
            raise RuntimeError(f"Unsupported archive type: {archive_type!r}")

    @classmethod
    def _run_silent_installer(cls, installer: str, dest: str) -> None:
        """Run an NSIS installer unattended, installing into *dest*.

        Stands in for extraction when a tool ships only as an installer
        (KTX-Software's Windows release). ``/S`` silences the wizard and
        ``/D=<dest>`` redirects the install root. NSIS requires ``/D`` to be
        the LAST argument and reads the rest of the command line as the
        path -- spaces included, quotes rejected -- so the command line is
        built as one string: ``list2cmdline`` would quote a path with spaces
        and the installer would land in its default location instead.

        A CPack-built installer requests administrator rights in its
        manifest, so ``CreateProcess`` refuses it from a normal process
        (WinError 740); that case re-runs through :meth:`_run_elevated`,
        which shows the standard UAC consent and waits. The install still
        lands in *dest* -- elevation changes who writes, not where.

        Raises:
            RuntimeError: Off Windows, on a declined UAC prompt, or on a
                non-zero installer exit code.
        """
        name = os.path.basename(installer)
        if cls._current_platform() != "windows":
            raise RuntimeError(
                f"'{name}' is a Windows (NSIS) installer; it cannot run on "
                f"{cls._current_platform()}."
            )
        params = f"/S /D={os.path.abspath(dest)}"
        try:
            code = subprocess.run(f'"{installer}" {params}', check=False).returncode
        except OSError as exc:
            if getattr(exc, "winerror", None) != 740:  # ERROR_ELEVATION_REQUIRED
                raise
            logger.info(f"{name} requests administrator rights; asking (UAC).")
            code = cls._run_elevated(installer, params)
        if code != 0:
            raise RuntimeError(f"Installer '{name}' failed with exit code {code}.")

    @staticmethod
    def _run_elevated(exe: str, params: str, cwd: Optional[str] = None) -> int:
        """Run *exe* with *params* through the Windows UAC ``runas`` verb,
        wait for it, and return its exit code.

        ``ShellExecuteExW`` is the only stdlib-reachable way to request
        elevation: ``subprocess`` cannot (CreateProcess has no elevation
        path) and ``os.startfile(..., "runas")`` neither waits nor reports
        an exit code. ``ctypes`` keeps it dependency-free. *params* is
        passed verbatim -- no re-quoting -- which is what an NSIS ``/D=``
        needs.

        Raises:
            RuntimeError: The consent prompt was declined (``ERROR_CANCELLED``,
                1223) or the launch failed for another reason.
        """
        import ctypes
        from ctypes import wintypes

        class SHELLEXECUTEINFOW(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("fMask", wintypes.ULONG),
                ("hwnd", wintypes.HWND),
                ("lpVerb", wintypes.LPCWSTR),
                ("lpFile", wintypes.LPCWSTR),
                ("lpParameters", wintypes.LPCWSTR),
                ("lpDirectory", wintypes.LPCWSTR),
                ("nShow", ctypes.c_int),
                ("hInstApp", wintypes.HINSTANCE),
                ("lpIDList", ctypes.c_void_p),
                ("lpClass", wintypes.LPCWSTR),
                ("hkeyClass", wintypes.HKEY),
                ("dwHotKey", wintypes.DWORD),
                ("hIconOrMonitor", wintypes.HANDLE),
                ("hProcess", wintypes.HANDLE),
            ]

        SEE_MASK_NOCLOSEPROCESS = 0x00000040
        SEE_MASK_NOASYNC = 0x00000100
        SW_SHOWNORMAL = 1
        INFINITE = 0xFFFFFFFF
        ERROR_CANCELLED = 1223

        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        shell32.ShellExecuteExW.argtypes = [ctypes.POINTER(SHELLEXECUTEINFOW)]
        shell32.ShellExecuteExW.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        info = SHELLEXECUTEINFOW()
        info.cbSize = ctypes.sizeof(SHELLEXECUTEINFOW)
        info.fMask = SEE_MASK_NOCLOSEPROCESS | SEE_MASK_NOASYNC
        info.lpVerb = "runas"
        info.lpFile = exe
        info.lpParameters = params
        info.lpDirectory = cwd
        info.nShow = SW_SHOWNORMAL
        name = os.path.basename(exe)
        if not shell32.ShellExecuteExW(ctypes.byref(info)):
            err = ctypes.get_last_error()
            reason = (
                "the administrator (UAC) prompt was declined"
                if err == ERROR_CANCELLED
                else ctypes.FormatError(err).strip()
            )
            raise RuntimeError(f"Could not run '{name}' elevated: {reason}.")
        if not info.hProcess:
            raise RuntimeError(
                f"'{name}' started elevated but returned no process handle to wait on."
            )
        try:
            kernel32.WaitForSingleObject(info.hProcess, INFINITE)
            code = wintypes.DWORD()
            kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(code))
        finally:
            kernel32.CloseHandle(info.hProcess)
        return int(code.value)

    @staticmethod
    def _find_executable(root: str, exe_name: str) -> Optional[str]:
        """Walk *root* to find a file matching *exe_name*.

        Tries exact match first, then with ``.exe`` appended (Windows).
        """
        candidates = [exe_name.lower()]
        if platform.system().lower() == "windows" and not exe_name.lower().endswith(
            ".exe"
        ):
            candidates.append(f"{exe_name.lower()}.exe")

        for dirpath, _dirs, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower() in candidates:
                    return os.path.join(dirpath, fn)
        return None

    # ------------------------------------------------------------------
    # Internal — location / platform helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_to_process_path(exe_path: str) -> None:
        """Append the parent directory of *exe_path* to ``os.environ["PATH"]``
        if it is not already present."""
        exe_dir = os.path.dirname(exe_path)
        current = os.environ.get("PATH", "")
        if exe_dir.lower() not in current.lower().split(os.pathsep):
            os.environ["PATH"] = f"{current}{os.pathsep}{exe_dir}"

    @staticmethod
    def _current_platform() -> str:
        """Return normalised platform key: ``windows``, ``linux``, or ``darwin``."""
        return platform.system().lower()

    @staticmethod
    def _resolve_location(location: str) -> str:
        env_override = os.environ.get("PYTHONTK_TOOLS_DIR")
        if env_override:
            return env_override

        if location == "user":
            return os.path.join(os.path.expanduser("~"), ".pythontk", "tools")
        elif location == "local":
            return os.path.join(os.getcwd(), "bin")
        elif location == "temp":
            return os.path.join(tempfile.gettempdir(), "pythontk_tools")
        else:
            raise ValueError(f"Unknown location: {location!r}")

    @staticmethod
    def _clean_url(url: str) -> str:
        """Strip query string and fragment from *url*."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    @classmethod
    def _filename_from_url(cls, url: str) -> str:
        """Extract a safe filename from *url*, ignoring query params."""
        clean = cls._clean_url(url)
        return os.path.basename(clean) or "download"

    @classmethod
    def _guess_archive_type(cls, url: str) -> str:
        """Infer archive type from the URL path (ignoring query params)."""
        clean = cls._clean_url(url).lower()
        for ext in ("tar.xz", "tar.gz", "tar.bz2", "tgz", "zip"):
            if clean.endswith(f".{ext}"):
                return ext
        # Deliberately no ``.exe`` -> ``"nsis"`` guess: that type RUNS the
        # download, so a tool author opts in explicitly.
        return "zip"

    # ------------------------------------------------------------------
    # Internal — install catalog (simple JSON sidecar)
    # ------------------------------------------------------------------

    @classmethod
    def _catalog_path(cls, install_dir: str) -> str:
        return os.path.join(install_dir, _CATALOG_NAME)

    @classmethod
    def _catalog_read(cls, install_dir: str, name: str) -> Optional[dict]:
        cat_path = cls._catalog_path(install_dir)
        if not os.path.isfile(cat_path):
            return None
        with open(cat_path, "r", encoding="utf-8") as fh:
            catalog = json.load(fh)
        return catalog.get(name)

    @classmethod
    def _catalog_write(
        cls, install_dir: str, name: str, exe_path: str, version: Optional[str]
    ) -> None:
        cat_path = cls._catalog_path(install_dir)
        catalog: dict = {}
        if os.path.isfile(cat_path):
            with open(cat_path, "r", encoding="utf-8") as fh:
                catalog = json.load(fh)
        catalog[name] = {"path": exe_path, "version": version}
        with open(cat_path, "w", encoding="utf-8") as fh:
            json.dump(catalog, fh, indent=2)
