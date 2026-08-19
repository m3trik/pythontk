import argparse
import getpass
import os
from typing import Any, Dict, Optional

# Connection defaults are resolved at CALL time, in this order: an explicit
# argument, then the environment, then a neutral fallback. Machine identity (a
# host, an account, a credential-store target) is deployment state, not library
# data -- this package publishes to PyPI, so it must not ship anyone's. Consumers
# that target a fixed machine pass it explicitly or export the env var.
ENV_HOST = "PYTHONTK_SSH_HOST"
ENV_USER = "PYTHONTK_SSH_USER"
ENV_CRED_TARGET = "PYTHONTK_SSH_CRED_TARGET"

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 22


class _CLIInternal:
    """Internal helpers for :class:`CLI`."""

    @staticmethod
    def _resolve(explicit: Optional[str], env_var: str, fallback) -> str:
        """First non-empty of: explicit argument, environment variable, fallback.

        Parameters:
            explicit: Value passed by the caller. Wins when truthy.
            env_var: Environment variable consulted next.
            fallback: A value, or a zero-argument callable invoked only if the
                earlier sources are empty (so the cost is not paid when unused).

        Returns:
            The resolved value.
        """
        if explicit:
            return explicit
        value = os.environ.get(env_var)
        if value:
            return value
        return fallback() if callable(fallback) else fallback

    @staticmethod
    def _current_user() -> str:
        """The OS account name, or an empty string where it cannot be determined."""
        try:
            return getpass.getuser()
        except Exception:  # no pwd entry / no USER env (some containers, services)
            return ""


class CLI(_CLIInternal):
    """
    Utilities for standardizing Command Line Interfaces across scripts.
    Designed to be extensible: add new static methods for different argument groups.
    """

    @staticmethod
    def get_parser(description: str = None) -> argparse.ArgumentParser:
        """
        Create a standard ArgumentParser.
        """
        return argparse.ArgumentParser(description=description)

    @staticmethod
    def add_connection_args(
        parser: argparse.ArgumentParser,
        default_host: Optional[str] = None,
        default_user: Optional[str] = None,
        default_target: Optional[str] = None,
    ) -> argparse.ArgumentParser:
        """Add standard SSH connection arguments (host, user, password, cred-target).

        Each default resolves as: the argument given here, else the matching
        environment variable (`PYTHONTK_SSH_HOST` / `PYTHONTK_SSH_USER` /
        `PYTHONTK_SSH_CRED_TARGET`), else a neutral fallback (`localhost`, the
        current OS user, and the resolved host respectively).

        Parameters:
            parser: The parser to extend.
            default_host: Target host for `--host`.
            default_user: Account for `--user`.
            default_target: Credential-store entry name for `--cred-target`.

        Returns:
            The same parser, for chaining.
        """
        host = _CLIInternal._resolve(default_host, ENV_HOST, DEFAULT_HOST)
        user = _CLIInternal._resolve(default_user, ENV_USER, _CLIInternal._current_user)
        target = _CLIInternal._resolve(default_target, ENV_CRED_TARGET, host)

        group = parser.add_argument_group("Connection Settings")
        group.add_argument(
            "--host",
            default=host,
            help=f"Target hostname or IP (default: {host})",
        )
        group.add_argument(
            "--user",
            default=user,
            help=f"SSH Username (default: {user})",
        )
        group.add_argument(
            "--password",
            default=None,
            help="SSH Password. If omitted, will attempt Secure Store lookup.",
        )
        group.add_argument(
            "--cred-target",
            default=target,
            help=f"Windows Credential Manager Target Name (default: {target})",
        )
        group.add_argument(
            "--port",
            type=int,
            default=DEFAULT_PORT,
            help=f"SSH Port (default: {DEFAULT_PORT})",
        )
        return parser

    @staticmethod
    def get_connection_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
        """Convert parsed arguments into a dictionary suitable for SSHClient.__init__.

        Usage:
            args = parser.parse_args()
            kwargs = CLI.get_connection_kwargs(args)
            with SSHClient(**kwargs) as client:
                ...

        Parameters:
            args: The namespace returned by `parse_args()`.

        Returns:
            Keyword arguments for `SSHClient`.
        """
        return {
            "host": args.host,
            "user": args.user,
            "password": args.password,
            "port": args.port,
            "credential_target": args.cred_target,
            # If password is provided explicitly, use_secure_store can be False or True (False is faster),
            # but SSHClient handles this check. We set True to enable fallback.
            "use_secure_store": True,
        }
