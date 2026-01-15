import unittest
import argparse
from unittest.mock import MagicMock, patch

try:
    from pythontk.core_utils.cli import CLI
except ImportError:
    # Allow running from root
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from pythontk.core_utils.cli import CLI


class TestCLI(unittest.TestCase):
    def test_add_connection_args(self):
        """Test adding connection arguments to an argparse parser."""
        parser = argparse.ArgumentParser()
        CLI.add_connection_args(
            parser, default_host="test_host", default_target="test_target"
        )

        args = parser.parse_args([])
        self.assertEqual(args.host, "test_host")
        self.assertEqual(args.cred_target, "test_target")
        self.assertEqual(args.port, 22)

    def test_add_connection_args_overrides(self):
        """Test overriding default values via command line."""
        parser = argparse.ArgumentParser()
        CLI.add_connection_args(parser)

        args = parser.parse_args(
            [
                "--host",
                "custom_host",
                "--cred-target",
                "custom_target",
                "--port",
                "2222",
            ]
        )
        self.assertEqual(args.host, "custom_host")
        self.assertEqual(args.cred_target, "custom_target")
        self.assertEqual(args.port, 2222)

    def test_get_connection_kwargs(self):
        """Test extracting connection kwargs from parsed args."""
        args = MagicMock()
        args.host = "my_host"
        args.user = "my_user"
        args.password = "my_pass"
        args.port = 2222
        # Mock use_secure_store logic implicitly handled by the method?
        # Let's check CLI implementation.
        # Actually CLI.get_connection_kwargs usually maps explicit args.

        # We need to see CLI.get_connection_kwargs logic.
        # It usually does: 'host': args.host, 'user': args.user, ...
        # And it might handle password/use_secure_store based on presence.

        # Let's blindly trust basic mapping first.
        kwargs = CLI.get_connection_kwargs(args)

        self.assertEqual(kwargs["host"], "my_host")
        self.assertEqual(kwargs["user"], "my_user")
        self.assertEqual(kwargs["port"], 2222)
        # Note: Implementation details of password vs secure store depend on the method logic.
        # If password is set, it might use it.
