# !/usr/bin/python
# coding=utf-8
from __future__ import annotations
import os
import socket
import subprocess
import tempfile
from typing import Optional, Dict

try:
    from .credentials import Credentials
except ImportError:
    Credentials = None


class NetUtils:
    """
    General purpose network utilities.
    """

    @staticmethod
    def connect_rdp(
        host: str,
        username: str = None,
        password: str = None,
        width: int = None,
        height: int = None,
        fullscreen: bool = True,
        extra_settings: Dict[str, str] = None,
        save_credentials: bool = True,
    ):
        """
        Connect to a remote desktop using Windows RDP (mstsc.exe).

        Args:
            host (str): Hostname or IP address.
            username (str, optional): Username to log in with.
            password (str, optional): Password. If provided, adds to Windows Credential Manager.
            width (int, optional): Window width.
            height (int, optional): Window height.
            fullscreen (bool): Whether to launch in fullscreen. Defaults to True.
            extra_settings (dict, optional): Dictionary of additional RDP settings (e.g. {'drivestoredirect': '*'})
                                             to merge/override defaults.
            save_credentials (bool, optional): Whether to persist the provided password in Credential Manager.
                                               Defaults to True (standard RDP behavior).
        """
        if os.name != "nt":
            raise OSError("RDP connection is only supported on Windows.")

        # 1. Handle Credentials via Credentials API if available, else fallback to cmdkey
        if username and password and save_credentials:
            target_name = f"TERMSRV/{host}"

            if Credentials:
                Credentials.set_credential(target_name, username, password)
            else:
                # Fallback if pywin32/Credentials not loaded
                subprocess.run(
                    [
                        "cmdkey",
                        "/generic:" + target_name,
                        f"/user:{username}",
                        f"/pass:{password}",
                    ],
                    check=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )

        # 2. Build RDP configuration
        defaults = {
            "full address": f"s:{host}",
            "authentication level": "i:0",  # Connect and don't warn me (default for automated scripts)
            "prompt for credentials": "i:0",
            "administrative session": "i:0",
        }

        if username:
            defaults["username"] = f"s:{username}"

        if fullscreen:
            defaults["screen mode id"] = "i:2"
        elif width and height:
            defaults["screen mode id"] = "i:1"
            defaults["desktopwidth"] = f"i:{width}"
            defaults["desktopheight"] = f"i:{height}"

        if extra_settings:
            defaults.update(extra_settings)

        config_lines = [f"{k}:{v}" for k, v in defaults.items()]

        # 3. Create temp file and launch
        fd, rdp_path = tempfile.mkstemp(suffix=".rdp", prefix="pythontk_rdp_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("\n".join(config_lines))

            subprocess.Popen(["mstsc.exe", rdp_path])
        except Exception as e:
            print(f"Failed to launch RDP: {e}")
            # Clean up immediately if fail, otherwise mstsc needs the file
            if os.path.exists(rdp_path):
                try:
                    os.remove(rdp_path)
                except OSError:
                    pass
            raise

    @staticmethod
    def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
        """
        Check if a TCP port is open on a host.

        Args:
            host (str): Hostname or IP address.
            port (int): Port number.
            timeout (float): Connection timeout in seconds.

        Returns:
            bool: True if port is open, False otherwise.
        """
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False

    @staticmethod
    def get_local_ip() -> Optional[str]:
        """
        Get the local IP address of this machine.
        Returns None if it fails.
        """
        try:
            # Uses a dummy connection to determine the interface IP used for routing
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return None
