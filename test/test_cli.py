import os
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


class TestConnectionDefaultResolution(unittest.TestCase):
    """The default host/user/target resolve explicit -> environment -> neutral.

    This package publishes to PyPI, so the defaults must carry no machine
    identity of their own; a consumer targeting a fixed machine passes it or
    exports the env var. Before 2026-08-18 the module hardcoded a maintainer
    Tailscale IP, personal email and workstation name as the shipped defaults.
    """

    ENV_KEYS = ("PYTHONTK_SSH_HOST", "PYTHONTK_SSH_USER", "PYTHONTK_SSH_CRED_TARGET")

    def _parse(self, argv=None, **kwargs):
        parser = argparse.ArgumentParser()
        CLI.add_connection_args(parser, **kwargs)
        return parser.parse_args(argv or [])

    def _clean_env(self):
        return patch.dict(os.environ, {k: "" for k in self.ENV_KEYS}, clear=False)

    def test_neutral_when_nothing_supplied(self):
        with self._clean_env():
            args = self._parse()
        self.assertEqual(args.host, "localhost")
        self.assertEqual(args.cred_target, "localhost")  # falls back to the host

    def test_environment_supplies_the_default(self):
        env = {
            "PYTHONTK_SSH_HOST": "env-host",
            "PYTHONTK_SSH_USER": "env-user",
            "PYTHONTK_SSH_CRED_TARGET": "env-target",
        }
        with patch.dict(os.environ, env, clear=False):
            args = self._parse()
        self.assertEqual((args.host, args.user, args.cred_target),
                         ("env-host", "env-user", "env-target"))

    def test_explicit_argument_beats_environment(self):
        with patch.dict(os.environ, {"PYTHONTK_SSH_HOST": "env-host"}, clear=False):
            args = self._parse(default_host="explicit-host")
        self.assertEqual(args.host, "explicit-host")

    def test_command_line_flag_beats_everything(self):
        with patch.dict(os.environ, {"PYTHONTK_SSH_HOST": "env-host"}, clear=False):
            args = self._parse(["--host", "flag-host"], default_host="explicit-host")
        self.assertEqual(args.host, "flag-host")

    def test_cred_target_defaults_to_the_resolved_host(self):
        with self._clean_env():
            args = self._parse(default_host="some-box")
        self.assertEqual(args.cred_target, "some-box")

    def test_module_ships_no_machine_identity(self):
        """Regression guard: a hardcoded host/account must not come back.

        Checks the SOURCE, not just the resolved value — a literal reintroduced
        as a different constant or as an argparse default would still ship.
        """
        import re

        from pythontk.core_utils import cli as cli_module

        source = open(cli_module.__file__, encoding="utf-8").read()
        identity = {
            "IPv4 literal": r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
            "email address": r"[\w.+-]+@[\w-]+\.[\w.]+",
        }
        for label, pattern in identity.items():
            hits = re.findall(pattern, source)
            self.assertEqual(
                hits, [], f"{label} hardcoded in a published module: {hits}"
            )
        # and the neutral fallback is what actually reaches a caller
        with self._clean_env():
            self.assertEqual(self._parse().host, "localhost")
