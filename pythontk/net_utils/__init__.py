from ._net_utils import NetUtils
from .credentials import Credentials


def __getattr__(name):
    if name == "SSHClient":
        from .ssh_client import SSHClient

        return SSHClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
