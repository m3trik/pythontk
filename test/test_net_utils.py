import unittest
import os
import socket
from unittest.mock import MagicMock, patch

# Adjust path for pythontk import if running standalone
try:
    import pythontk.net_utils
except ImportError:
    pass

from pythontk.net_utils.ssh_client import SSHClient, paramiko
from pythontk.net_utils.credentials import Credentials


class TestCredentials(unittest.TestCase):
    
    def setUp(self):
        # Reset the import state for clean testing if needed
        pass

    @patch("pythontk.net_utils.credentials.keyring")
    def test_keyring_priority(self, mock_keyring):
        """Test that keyring is tried first if available."""
        mock_keyring.get_password.return_value = "keyring_pass"
        
        # Act
        result = Credentials.get_credential("some_target")
        
        # Assert
        self.assertEqual(result["password"], "keyring_pass")
        self.assertEqual(result["username"], "keyring_user") # Default for now
        mock_keyring.get_password.assert_called_with("pythontk", "some_target")

    @patch("pythontk.net_utils.credentials.keyring", None) # Simulate keyring missing
    @patch("platform.system")
    @patch("pythontk.net_utils.credentials.win32cred")
    def test_windows_native_fallback(self, mock_win32, mock_platform):
        """Test fallback to Windows Credential Manager if keyring is missing."""
        mock_platform.return_value = "Windows"
        
        # Setup mock return for win32cred
        mock_creds = MagicMock()
        # Mocking the PyWin32 CredentialBlob return logic
        mock_creds.get.side_effect = lambda k, d=None: b"win_pass".decode("utf-8").encode("utf-16-le") if k == "CredentialBlob" else "win_user"
        
        mock_win32.CredRead.return_value = mock_creds
        mock_win32.CRED_TYPE_GENERIC = 1

        # We also need to patch the internal _get_windows_creds decoding logic if we want to be precise,
        # but the logic in the code reads 'CredentialBlob' from dict/object returned by CredRead.
        # The previous test mocked CredentialBlob as a key in a dict, but CredRead returns a dictionary-like object in PyWin32?
        # Let's inspect the code: `blob = creds.get("CredentialBlob", b"")`. So creds is a dict.
        
        # Remock for exact dict match
        mock_creds_dict = {
            "CredentialBlob": b"win_pass".decode("utf-8").encode("utf-16-le"),
            "UserName": "win_user"
        }
        mock_win32.CredRead.return_value = mock_creds_dict

        # Act
        result = Credentials.get_credential("win_target")

        # Assert
        self.assertEqual(result["password"], "win_pass")
        self.assertEqual(result["username"], "win_user")

    @patch("pythontk.net_utils.credentials.keyring", None)
    @patch("platform.system")
    @patch.dict(os.environ, {"MY_TARGET_PASSWORD": "env_pass", "MY_TARGET_USER": "env_user"})
    def test_env_var_fallback(self, mock_platform):
        """Test fallback to Environment Variables on Linux (or when Windows fails/not present)."""
        mock_platform.return_value = "Linux"
        
        # Act
        # Target name "my-target" should map to MY_TARGET_PASSWORD
        result = Credentials.get_credential("my-target")
        
        # Assert
        self.assertIsNotNone(result)
        self.assertEqual(result["password"], "env_pass")
        self.assertEqual(result["username"], "env_user")

    @patch("pythontk.net_utils.credentials.keyring", None)
    @patch("platform.system")
    def test_env_var_patterns(self, mock_platform):
        """Test various environment variable naming patterns."""
        mock_platform.return_value = "Linux"

        cases = [
            ("dev.db", "DEV_DB_PASSWORD"),
            ("prod-server", "PROD_SERVER_SECRET"),
            ("api_key_v1", "API_KEY_V1_KEY"),
        ]

        for target, env_key in cases:
            with patch.dict(os.environ, {env_key: "secret_val"}, clear=True):
                cred = Credentials.get_credential(target)
                self.assertEqual(cred["password"], "secret_val", f"Failed to match {target} to {env_key}")


@unittest.skipIf(paramiko is None, "paramiko not installed")
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

    @patch("pythontk.net_utils.ssh_client.paramiko.SSHClient")
    def test_execute_transport_drains_output_buffered_at_exit(self, mock_ssh_cls):
        """Captured PTY output must include chunks still buffered when the exit
        status arrives.

        Bug: the transport read loop broke on ``exit_status_ready()`` after at
        most one 4096-byte read per stream, dropping any output still queued.
        """
        mock_client = mock_ssh_cls.return_value
        channel = MagicMock()
        transport = mock_client.get_transport.return_value
        transport.active = True
        transport.open_session.return_value = channel

        chunks = [b"first-chunk ", b"second-chunk"]
        channel.recv_ready.side_effect = lambda: bool(chunks)
        channel.recv.side_effect = lambda n: chunks.pop(0)
        channel.recv_stderr_ready.return_value = False
        # Exit status is known from the very first poll — both chunks are
        # already sitting in the receive buffer.
        channel.exit_status_ready.return_value = True
        channel.recv_exit_status.return_value = 0

        client = SSHClient("test.host")
        client._connected = True

        out, err, code = client.execute("cat big.txt", use_pty=True)

        self.assertEqual(out, "first-chunk second-chunk")
        self.assertEqual(err, "")
        self.assertEqual(code, 0)

    @patch("pythontk.net_utils.ssh_client.paramiko.SSHClient")
    def test_execute_transport_multibyte_char_split_across_chunks(self, mock_ssh_cls):
        """A UTF-8 character split across two recv() chunks must decode intact.

        Bug: each chunk was decoded independently with ``errors="replace"``,
        so a multi-byte character straddling a 4096-byte chunk boundary became
        two replacement characters instead of the character itself.
        """
        mock_client = mock_ssh_cls.return_value
        channel = MagicMock()
        transport = mock_client.get_transport.return_value
        transport.active = True
        transport.open_session.return_value = channel

        # "café" with the 2-byte é (b"\xc3\xa9") split across the chunks.
        chunks = [b"caf\xc3", b"\xa9 au lait"]
        channel.recv_ready.side_effect = lambda: bool(chunks)
        channel.recv.side_effect = lambda n: chunks.pop(0)
        channel.recv_stderr_ready.return_value = False
        channel.exit_status_ready.return_value = True
        channel.recv_exit_status.return_value = 0

        client = SSHClient("test.host")
        client._connected = True

        out, err, code = client.execute("cat utf8.txt", use_pty=True)

        self.assertEqual(out, "café au lait")
        self.assertEqual(code, 0)


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
