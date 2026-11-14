import time
import socket
import unittest
from unittest.mock import MagicMock, patch

# Adjust path for pythontk import if running standalone
try:
    import pythontk.net_utils  # noqa: F401
except ImportError:
    pass

from pythontk.net_utils.ssh_client import SSHClient, paramiko


@unittest.skipIf(paramiko is None, "paramiko not installed")
class TestSSHClientStreamTimeout(unittest.TestCase):
    """Regression: execute(..., stream/use_pty, timeout=…) dropped the timeout on
    the transport path, so a never-exiting command (e.g. `tail -f`) pumped
    forever instead of honoring the documented timeout."""

    def _make_client_with_never_exiting_channel(self, mock_ssh_cls):
        """Wire up an SSHClient whose channel produces no output and never
        reports an exit status — the `tail -f` failure shape."""
        mock_client = mock_ssh_cls.return_value
        channel = MagicMock()
        transport = mock_client.get_transport.return_value
        transport.active = True
        transport.open_session.return_value = channel

        channel.recv_ready.return_value = False
        channel.recv_stderr_ready.return_value = False
        channel.exit_status_ready.return_value = False  # never exits
        channel.recv_exit_status.return_value = 0

        client = SSHClient("test.host")
        client._connected = True
        return client, channel

    @patch("pythontk.net_utils.ssh_client.paramiko.SSHClient")
    def test_stream_timeout_raises_and_closes_channel(self, mock_ssh_cls):
        client, channel = self._make_client_with_never_exiting_channel(mock_ssh_cls)

        start = time.time()
        with self.assertRaises(socket.timeout):
            client.execute("tail -f /var/log/app.log", stream=True, timeout=0.1)
        elapsed = time.time() - start

        # Bounded runtime: must return well before a hang, near the deadline.
        self.assertLess(elapsed, 5.0)
        channel.close.assert_called_once()

    @patch("pythontk.net_utils.ssh_client.paramiko.SSHClient")
    def test_pty_capture_timeout_raises(self, mock_ssh_cls):
        """The non-stream PTY-capture path is also bounded by the deadline."""
        client, channel = self._make_client_with_never_exiting_channel(mock_ssh_cls)

        with self.assertRaises(socket.timeout):
            client.execute("cat /dev/urandom", use_pty=True, timeout=0.1)
        channel.close.assert_called_once()

    @patch("pythontk.net_utils.ssh_client.paramiko.SSHClient")
    def test_stream_without_timeout_still_completes(self, mock_ssh_cls):
        """No timeout given: a command that exits promptly returns normally
        (guards against the deadline check firing when timeout is None)."""
        mock_client = mock_ssh_cls.return_value
        channel = MagicMock()
        transport = mock_client.get_transport.return_value
        transport.active = True
        transport.open_session.return_value = channel

        channel.recv_ready.return_value = False
        channel.recv_stderr_ready.return_value = False
        channel.exit_status_ready.return_value = True  # exits immediately
        channel.recv_exit_status.return_value = 0

        client = SSHClient("test.host")
        client._connected = True

        code = client.execute("true", stream=True)
        self.assertEqual(code, 0)
        channel.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
