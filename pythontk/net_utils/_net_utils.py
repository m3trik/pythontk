# !/usr/bin/python
# coding=utf-8
from __future__ import annotations
import socket
from typing import Optional


class NetUtils:
    """
    General purpose network utilities.
    """

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
