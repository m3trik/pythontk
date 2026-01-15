import argparse
from typing import Dict, Any

# Workspace Defaults
DEFAULT_HOST = "100.127.101.7"
DEFAULT_USER = "m3trik@outlook.com"
DEFAULT_CRED_TARGET = "M3TRIK_DESKTOP"


class CLI:
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
        default_host: str = DEFAULT_HOST,
        default_user: str = DEFAULT_USER,
        default_target: str = DEFAULT_CRED_TARGET,
    ) -> argparse.ArgumentParser:
        """
        Add standard SSH connection arguments (host, user, password, cred-target).
        """
        group = parser.add_argument_group("Connection Settings")
        group.add_argument(
            "--host",
            default=default_host,
            help=f"Target hostname or IP (default: {default_host})",
        )
        group.add_argument(
            "--user",
            default=default_user,
            help=f"SSH Username (default: {default_user})",
        )
        group.add_argument(
            "--password",
            default=None,
            help="SSH Password. If omitted, will attempt Secure Store lookup.",
        )
        group.add_argument(
            "--cred-target",
            default=default_target,
            help=f"Windows Credential Manager Target Name (default: {default_target})",
        )
        group.add_argument(
            "--port", type=int, default=22, help="SSH Port (default: 22)"
        )
        return parser

    @staticmethod
    def get_connection_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
        """
        Convert parsed arguments into a dictionary suitable for SSHClient.__init__.

        Usage:
            args = parser.parse_args()
            kwargs = CLI.get_connection_kwargs(args)
            with SSHClient(**kwargs) as client:
                ...
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
