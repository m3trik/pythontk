import unittest
import sys
import socket
from unittest.mock import MagicMock, patch

# Adjust path for pythontk import if running standalone
try:
    import pythontk.net_utils
except ImportError:
    pass

from pythontk.net_utils.ssh_client import SSHClient
from pythontk.net_utils.credentials import Credentials


class TestCredentials(unittest.TestCase):

    def test_get_windows_creds_mock(self):
        """Test retrieving credentials from a mocked Windows store."""
        with patch("pythontk.net_utils.credentials.win32cred") as mock_win32:
            # Setup mock return
            mock_creds = {
                "CredentialBlob": b"secret_pass".decode("utf-8").encode("utf-16-le")
            }
            mock_win32.CredRead.return_value = mock_creds
            mock_win32.CRED_TYPE_GENERIC = 1

            # Run
            password = Credentials.get_password("some_target")

            # Assert
            self.assertEqual(password, "secret_pass")
            mock_win32.CredRead.assert_called_with("some_target", 1)

    def test_get_windows_creds_none(self):
        """Test handling of missing credentials."""
        with patch("pythontk.net_utils.credentials.win32cred") as mock_win32:
            # Simulate pywintypes.error
            error_class = type("error", (Exception,), {})

            # We need to mock pywintypes.error within the module context or catch Exception
            # Since we import it inside the module, patch the module's view of it
            with patch("pythontk.net_utils.credentials.pywintypes") as mock_types:
                mock_types.error = error_class
                mock_win32.CredRead.side_effect = error_class("Element not found")

                password = Credentials.get_password("missing_target")
                self.assertIsNone(password)


class TestSSHClient(unittest.TestCase):

    @patch("pythontk.net_utils.ssh_client.paramiko.SSHClient")
    def test_connect_with_password_arg(self, mock_ssh_cls):
        """Test connection when password is explicitly provided."""
        mock_client = mock_ssh_cls.return_value

        client = SSHClient(host="test.host", user="user", password="my_password")
        client.connect()

        mock_client.connect.assert_called_with(
            hostname="test.host",
            port=22,
            username="user",
            password="my_password",
            look_for_keys=True,
            allow_agent=True,
            timeout=10,
        )

    @patch("pythontk.net_utils.ssh_client.paramiko.SSHClient")
    @patch("pythontk.net_utils.ssh_client.Credentials.get_password")
    def test_connect_with_secure_store(self, mock_get_pass, mock_ssh_cls):
        """Test connection using secure store lookup."""
        mock_client = mock_ssh_cls.return_value
        mock_get_pass.return_value = "stored_secret"

        client = SSHClient(host="test.host", user="user", use_secure_store=True)
        client.connect()

        # Verify it looked up the password using the host
        mock_get_pass.assert_called_with("test.host")

        # Verify it connected with the retrieved password
        mock_client.connect.assert_called_with(
            hostname="test.host",
            port=22,
            username="user",
            password="stored_secret",
            look_for_keys=True,
            allow_agent=True,
            timeout=10,
        )

    @patch("pythontk.net_utils.ssh_client.paramiko.SSHClient")
    def test_execute_simple(self, mock_ssh_cls):
        """Test simple command execution without streaming."""
        mock_client = mock_ssh_cls.return_value

        # Mock exec_command return values (stdin, stdout, stderr)
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b"output"
        mock_stdout.channel.recv_exit_status.return_value = 0

        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""

        mock_client.exec_command.return_value = (None, mock_stdout, mock_stderr)

        client = SSHClient("test.host")
        # Manually set connected state to avoid connect() call
        client._connected = True

        out, err, code = client.execute("ls -la")

        self.assertEqual(out, "output")
        self.assertEqual(code, 0)
        mock_client.exec_command.assert_called_with(
            "ls -la", get_pty=False, timeout=None
        )


from pythontk.net_utils._net_utils import NetUtils


class TestNetUtils(unittest.TestCase):

    def test_get_local_ip(self):
        """Test retrieving local IP."""
        ip = NetUtils.get_local_ip()
        if ip:
            self.assertRegex(ip, r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")

    @patch("socket.create_connection")
    def test_is_port_open_success(self, mock_connect):
        """Test port open check returns True on success."""
        self.assertTrue(NetUtils.is_port_open("localhost", 80))

    @patch("socket.create_connection")
    def test_is_port_open_failure(self, mock_connect):
        """Test port open check returns False on connection error."""
        mock_connect.side_effect = socket.timeout
        self.assertFalse(NetUtils.is_port_open("localhost", 80))


if __name__ == "__main__":
    unittest.main()
