import os
import sys
import time
import socket
import paramiko
from typing import Optional, Tuple, Union

from .credentials import Credentials


class SSHClient:
    """
    A unified SSH Client wrapper around Paramiko that handles:
    1. Secure credential retrieval (OS Store).
    2. Connection logic/policies.
    3. Command execution with streaming or captured output.
    4. PTY allocation for correct buffering on some servers.
    """

    def __init__(
        self,
        host: str,
        user: str = None,
        password: str = None,
        use_secure_store: bool = False,
        port: int = 22,
        credential_target: str = None,
    ):
        """
        Args:
            host (str): Hostname or IP address.
            user (str, optional): SSH Username.
            password (str, optional): SSH Password. If None, keys or secure store will be tried.
            use_secure_store (bool): If True, attempt to fetch password from OS manager using `host` (or `credential_target`) as the target name.
            port (int): SSH port.
            credential_target (str, optional): Specific target name to look up in credential manager if different from host.
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.use_secure_store = use_secure_store
        self.credential_target = credential_target

        self.client = paramiko.SSHClient()
        # Automatically add host keys (useful for dev/internal networks)
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        """Establish the SSH connection."""
        if self._connected:
            return

        password = self.password

        # Attempt secure lookup if configured and no password provided
        if not password and self.use_secure_store:
            target = self.credential_target or self.host
            stored_pass = Credentials.get_password(target)
            if stored_pass:
                password = stored_pass
            else:
                # Could optionally check for self.user in the store if we wanted complex logic,
                # but 'target=host' is the standard integration point.
                pass

        try:
            # Paramiko connect handles:
            # 1. pkey (not passed here, but paramiko looks for ~/.ssh/id_rsa by default if look_for_keys=True)
            # 2. password (passed explicitly)
            # 3. agent keys
            self.client.connect(
                hostname=self.host,
                port=self.port,
                username=self.user,
                password=password,
                look_for_keys=True,
                allow_agent=True,
                timeout=10,
            )
            self._connected = True
        except paramiko.AuthenticationException:
            raise PermissionError(f"Authentication failed for {self.user}@{self.host}")
        except socket.error as e:
            raise ConnectionError(f"Could not connect to {self.host}:{self.port}. {e}")

    def disconnect(self):
        """Close the connection."""
        if self.client:
            self.client.close()
        self._connected = False

    def execute(
        self,
        command: str,
        stream: bool = False,
        use_pty: bool = False,
        timeout: float = None,
    ) -> Union[Tuple[str, str, int], int]:
        """
        Execute a command on the remote server.

        Args:
            command (str): Shell command to run.
            stream (bool): If True, output is printed to sys.stdout/stderr in real-time. Returns exit_code.
                           If False, returns (stdout, stderr, exit_code).
            use_pty (bool): Request a pseudo-terminal. Required for some commands to buffer correctly (e.g. windows hosts).
            timeout (float): Connection timeout.

        Returns:
            Tuple[str, str, int] | int: If stream=False, returns (stdout_str, stderr_str, exit_code).
                                        If stream=True, returns exit_code.
        """
        if not self._connected:
            self.connect()

        if use_pty or stream:
            # Transport-level session execution is better for PTY/streaming
            return self._execute_transport(command, stream, use_pty)
        else:
            # Standard exec_command is simpler for capturing
            stdin, stdout, stderr = self.client.exec_command(
                command, get_pty=use_pty, timeout=timeout
            )

            # Read fully
            out_str = stdout.read().decode("utf-8", errors="replace")
            err_str = stderr.read().decode("utf-8", errors="replace")
            exit_code = stdout.channel.recv_exit_status()

            return out_str.strip(), err_str.strip(), exit_code

    def _execute_transport(self, command: str, stream: bool, use_pty: bool):
        """
        Low-level execution using channel transport, needed for robust PTY streaming.
        """
        transport = self.client.get_transport()
        if not transport or not transport.active:
            raise ConnectionError("SSH Transport is not active.")

        channel = transport.open_session()

        if use_pty:
            channel.get_pty()

        channel.exec_command(command)

        captured_stdout = []
        captured_stderr = (
            []
        )  # PTY merges stderr into stdout usually, but we'll try capture separate if possible

        while True:
            # Reading logic
            if channel.recv_ready():
                data = channel.recv(4096).decode("utf-8", errors="replace")
                if stream:
                    sys.stdout.write(data)
                    sys.stdout.flush()
                else:
                    captured_stdout.append(data)

            if channel.recv_stderr_ready():
                data = channel.recv_stderr(4096).decode("utf-8", errors="replace")
                if stream:
                    sys.stderr.write(data)
                    sys.stderr.flush()
                else:
                    captured_stderr.append(data)

            if channel.exit_status_ready():
                break

            time.sleep(0.05)

        exit_code = channel.recv_exit_status()

        if not stream:
            return (
                "".join(captured_stdout).strip(),
                "".join(captured_stderr).strip(),
                exit_code,
            )
        return exit_code

    def upload_file(self, local_path: str, remote_path: str):
        """Upload a file via SFTP."""
        if not self._connected:
            self.connect()
        sftp = self.client.open_sftp()
        try:
            sftp.put(local_path, remote_path)
        finally:
            sftp.close()

    def download_file(self, remote_path: str, local_path: str):
        """Download a file via SFTP."""
        if not self._connected:
            self.connect()
        sftp = self.client.open_sftp()
        try:
            sftp.get(remote_path, local_path)
        finally:
            sftp.close()
