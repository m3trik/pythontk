import unittest
import os
import socket
from unittest.mock import MagicMock, patch

# Adjust path for pythontk import if running standalone
try:
    import pythontk.net_utils  # noqa: F401  (availability probe, not a use)
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
        self.assertEqual(result["username"], "keyring_user")  # Default for now
        mock_keyring.get_password.assert_called_with("pythontk", "some_target")

    @patch("pythontk.net_utils.credentials.keyring", None)  # Simulate keyring missing
    @patch("platform.system")
    @patch("pythontk.net_utils.credentials.win32cred")
    def test_windows_native_fallback(self, mock_win32, mock_platform):
        """Test fallback to Windows Credential Manager if keyring is missing."""
        mock_platform.return_value = "Windows"

        # Setup mock return for win32cred
        mock_creds = MagicMock()
        # Mocking the PyWin32 CredentialBlob return logic
        mock_creds.get.side_effect = lambda k, d=None: (
            b"win_pass".decode("utf-8").encode("utf-16-le")
            if k == "CredentialBlob"
            else "win_user"
        )

        mock_win32.CredRead.return_value = mock_creds
        mock_win32.CRED_TYPE_GENERIC = 1

        # We also need to patch the internal _get_windows_creds decoding logic if we want to be precise,
        # but the logic in the code reads 'CredentialBlob' from dict/object returned by CredRead.
        # The previous test mocked CredentialBlob as a key in a dict, but CredRead returns a dictionary-like object in PyWin32?
        # Let's inspect the code: `blob = creds.get("CredentialBlob", b"")`. So creds is a dict.

        # Remock for exact dict match
        mock_creds_dict = {
            "CredentialBlob": b"win_pass".decode("utf-8").encode("utf-16-le"),
            "UserName": "win_user",
        }
        mock_win32.CredRead.return_value = mock_creds_dict

        # Act
        result = Credentials.get_credential("win_target")

        # Assert
        self.assertEqual(result["password"], "win_pass")
        self.assertEqual(result["username"], "win_user")

    @patch("pythontk.net_utils.credentials.keyring", None)
    @patch("platform.system")
    @patch.dict(
        os.environ, {"MY_TARGET_PASSWORD": "env_pass", "MY_TARGET_USER": "env_user"}
    )
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
                self.assertEqual(
                    cred["password"],
                    "secret_val",
                    f"Failed to match {target} to {env_key}",
                )


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

    def test_is_port_bindable_free_port(self):
        """A genuinely free (kernel-assigned, then released) port is bindable."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        self.assertTrue(NetUtils.is_port_bindable(port))

    def test_is_port_bindable_detects_zombie_bind(self):
        """Regression (2026-07-10): a bound-but-NOT-listening socket (a hung
        process's leftover) must read as NOT bindable, even though a connect
        probe (is_port_open) sees nothing to connect to there."""
        squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        squatter.bind(("127.0.0.1", 0))  # bound, deliberately never listen()s
        port = squatter.getsockname()[1]
        try:
            self.assertFalse(NetUtils.is_port_bindable(port))
            # The semantic split this helper exists for:
            self.assertFalse(NetUtils.is_port_open("127.0.0.1", port, timeout=0.3))
        finally:
            squatter.close()

    @patch("pythontk.net_utils._net_utils.subprocess.Popen")
    @patch("pythontk.net_utils._net_utils.subprocess.run")
    @patch("pythontk.file_utils.temp_artifacts.TempArtifacts")
    @patch("pythontk.net_utils._net_utils.os.name", "nt")
    def test_connect_rdp_saves_credentials_at_termsrv_target(
        self, mock_temp_artifacts, mock_run, mock_popen
    ):
        """Regression: connect_rdp(save_credentials=True) must write the RDP
        secret to the exact Windows target mstsc reads (TERMSRV/<host>) via
        cmdkey. It previously routed through Credentials.set_credential, whose
        keyring backend files the secret under the service name 'pythontk', so
        mstsc never found it and prompted anyway -- silently defeating
        save_credentials=True."""
        import subprocess as _sp
        import tempfile as _tf
        import shutil as _sh

        if not hasattr(_sp, "CREATE_NO_WINDOW"):
            self.skipTest("subprocess.CREATE_NO_WINDOW is Windows-only")

        # Redirect the generated .rdp file so nothing leaks into the real temp
        # artifacts dir. connect_rdp allocates through ptk.TempArtifacts, never a
        # raw tempfile.mkstemp, so that is the only seam a redirect can bind to --
        # patching tempfile here redirected nothing and silently wrote a live file.
        tmp_dir = _tf.mkdtemp(prefix="pythontk_rdp_test_")
        self.addCleanup(_sh.rmtree, tmp_dir, ignore_errors=True)
        rdp_path = os.path.join(tmp_dir, "test.rdp")
        mock_temp_artifacts.return_value.path.return_value = rdp_path

        NetUtils.connect_rdp(
            "10.0.0.5", username="admin", password="p", save_credentials=True
        )

        # Exactly one credential write, and it goes through cmdkey targeting
        # TERMSRV/<host> -- never through the keyring-first Credentials store.
        self.assertEqual(mock_run.call_count, 1)
        cmd = mock_run.call_args.args[0]
        self.assertEqual(cmd[0], "cmdkey")
        self.assertEqual(cmd[1], "/generic:TERMSRV/10.0.0.5")
        self.assertIn("/user:admin", cmd)
        self.assertIn("/pass:p", cmd)
        # The .rdp config is allocated through the age-swept TempArtifacts
        # primitive (repo temp-artifact rule), and actually written there.
        mock_temp_artifacts.assert_called_once_with("pythontk_rdp")
        mock_temp_artifacts.return_value.path.assert_called_once_with(extension=".rdp")
        self.assertTrue(os.path.exists(rdp_path))
        # mstsc.exe is still launched with the generated .rdp config file.
        mock_popen.assert_called_once()
        self.assertEqual(mock_popen.call_args.args[0], ["mstsc.exe", rdp_path])


class _FakeResolver:
    """A one-port UDP DNS server answering every query the same way.

    *answer* is ``"address"`` (one A record), ``"nxdomain"``, ``"empty"`` (no
    error, no records), ``"servfail"`` or ``"wrong-id"`` (a reply to some
    other query) -- the outcomes ``resolves_publicly`` tells apart.
    """

    def __init__(self, answer, host="127.0.0.1", port=0):
        import threading

        self.answer = answer
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.port = self.sock.getsockname()[1]
        self.queries = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        import struct

        while True:
            try:
                query, peer = self.sock.recvfrom(512)
            except OSError:
                return
            self.queries.append(query)
            query_id = struct.unpack(">H", query[:2])[0]
            if self.answer == "wrong-id":
                query_id ^= 0xFFFF
            rcode = {"nxdomain": 3, "servfail": 2}.get(self.answer, 0)
            count = 1 if self.answer == "address" else 0
            reply = struct.pack(">HHHHHH", query_id, 0x8180 | rcode, 1, count, 0, 0)
            reply += query[12:]
            if count:
                reply += struct.pack(">HHHIH", 0xC00C, 1, 1, 60, 4) + bytes(
                    [192, 0, 2, 1]
                )
            self.sock.sendto(reply, peer)

    def close(self):
        self.sock.close()


class TestResolvesPublicly(unittest.TestCase):
    """A public resolver asked directly -- the check a brand-new tunnel link
    waits on -- told apart into yes, no, and could-not-ask."""

    def _ask(self, *answers, name="fresh-link.example.test"):
        resolvers = [_FakeResolver(a) for a in answers]
        try:
            ports = {r.port for r in resolvers}
            self.assertEqual(len(ports), len(resolvers))
            results = []
            for resolver in resolvers:
                results.append(
                    NetUtils.resolves_publicly(
                        name, servers=("127.0.0.1",), port=resolver.port, timeout=2
                    )
                )
            return results, resolvers
        finally:
            for resolver in resolvers:
                resolver.close()

    def test_an_address_is_a_yes(self):
        (result,), (resolver,) = self._ask("address")
        self.assertIs(result, True)
        # One A query, for the name asked -- the question is echoed back intact.
        self.assertIn(
            b"\x0afresh-link\x07example\x04test\x00\x00\x01\x00\x01",
            resolver.queries[0],
        )

    def test_nxdomain_and_no_address_are_a_no(self):
        results, _ = self._ask("nxdomain", "empty")
        self.assertEqual(results, [False, False])

    def test_a_resolver_with_no_answer_is_no_answer(self):
        """SERVFAIL, and a reply to a different query, say nothing either way."""
        results, _ = self._ask("servfail", "wrong-id")
        self.assertEqual(results, [None, None])

    def test_nobody_to_ask_is_none_not_a_no(self):
        """Outbound DNS blocked or offline must not read as "does not exist":
        a caller waiting on the name would otherwise wait out its whole
        budget for a question that was never asked."""
        closed = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
        closed.close()
        self.assertIsNone(
            NetUtils.resolves_publicly(
                "x.example.test", servers=("127.0.0.1",), port=port, timeout=0.5
            )
        )

    def test_any_resolver_with_the_name_is_a_yes(self):
        """Each resolver caches its own misses, so while a name appears one
        can still say NXDOMAIN after another already has it -- every resolver
        is asked before the answer is no."""
        stale = _FakeResolver("nxdomain")
        try:
            # A second loopback address, so both share the one port the call
            # takes. Every 127/8 address is loopback on Windows and Linux.
            fresh = _FakeResolver("address", host="127.0.0.2", port=stale.port)
        except OSError:
            stale.close()
            self.skipTest("no second loopback address on this host")
        try:
            result = NetUtils.resolves_publicly(
                "fresh-link.example.test",
                servers=("127.0.0.1", "127.0.0.2"),
                port=stale.port,
                timeout=2,
            )
            self.assertIs(result, True)
            self.assertEqual((len(stale.queries), len(fresh.queries)), (1, 1))
        finally:
            stale.close()
            fresh.close()


if __name__ == "__main__":
    unittest.main()
