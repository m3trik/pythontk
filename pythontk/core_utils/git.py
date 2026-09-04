import subprocess
import logging
import warnings
from pathlib import Path
from typing import Union, List, Optional

logger = logging.getLogger(__name__)


class Git:
    """A wrapper around git subprocess commands for a specific repository.

    **Deprecated** -- removed in the next release. It has no callers anywhere
    in the ecosystem and no tests of its own, so it is published surface that
    only costs review attention and a name on the index. Shell out to ``git``
    directly, or use a real client library if a project needs one.
    """

    #: Named so the retirement is greppable alongside the JSON key-value one.
    _DEPRECATION = (
        "pythontk.Git is deprecated and will be removed in the next release: "
        "it has no callers in any ecosystem package. Call git through "
        "subprocess directly, or use a dedicated git library."
    )

    def __init__(
        self, path: Union[str, Path], dry_run: bool = False, verbose: bool = True
    ):
        # In __init__ rather than a module-level __getattr__: the class is
        # registered by name through the package resolver, so __getattr__
        # never sees the lookup.
        warnings.warn(self._DEPRECATION, DeprecationWarning, stacklevel=2)
        self.path = Path(path).resolve()
        self.dry_run = dry_run
        self.verbose = verbose

        if not (self.path / ".git").exists() and not dry_run:
            # In dry-run we might be targeting a path that will exist?
            # Ideally the repo should exist even in dry run for these ops.
            logger.warning(
                f"Path '{self.path}' does not appear to be a git repository."
            )

    def execute(
        self, cmd: Union[str, List[str]], desc: str = None, check: bool = True
    ) -> Optional[str]:
        """
        Run a generic shell command in the repository directory.

        Args:
            cmd: Command string or list of args
            desc: Description for logging
            check: Raise exception on failure
        """
        # Formulate command list
        if isinstance(cmd, str):
            args = cmd.split()
        else:
            args = cmd

        cmd_str = " ".join(args)

        # Verbose Logging
        if self.verbose and desc:
            print(f"[{self.path.name}] {desc}...")

        # Dry Run
        if self.dry_run:
            print(f"[{self.path.name}] [DRY RUN] Would execute: {cmd_str}")
            return None

        # Execution
        try:
            result = subprocess.run(
                args, cwd=str(self.path), check=check, capture_output=True, text=True
            )

            if self.verbose and desc:
                print(f"[{self.path.name}] ✅ {desc} Succeeded")

            return result.stdout.strip()

        except subprocess.CalledProcessError as e:
            if self.verbose and desc:
                print(f"[{self.path.name}] ❌ {desc} FAILED")
                print(f"  Error: {e.stderr.strip()}")

            if check:
                raise RuntimeError(
                    f"Command failed: {cmd_str}\n{e.stderr.strip()}"
                ) from e
            return None

    def run(
        self, cmd: Union[str, List[str]], desc: str = None, check: bool = True
    ) -> Optional[str]:
        """
        Run a git command in the repository.
        """
        if isinstance(cmd, str):
            return self.execute(["git"] + cmd.split(), desc, check)
        else:
            return self.execute(["git"] + cmd, desc, check)

    def checkout(self, branch: str):
        """Checkout a branch."""
        return self.run(["checkout", branch], f"Checkout {branch}")

    def pull(self, remote: str = "origin", branch: str = None):
        """Pull changes."""
        cmd = ["pull", remote]
        if branch:
            cmd.append(branch)
        return self.run(cmd, f"Pull {remote} {branch or 'current'}")

    def push(self, remote: str = "origin", branch: str = None):
        """Push changes."""
        cmd = ["push", remote]
        if branch:
            cmd.append(branch)
        return self.run(cmd, f"Push {remote} {branch or 'current'}")

    def merge(self, source_branch: str):
        """Merge a branch into the current branch."""
        return self.run(["merge", source_branch], f"Merge {source_branch}")

    def fetch(self, remote: str = "origin"):
        """Fetch remote."""
        return self.run(["fetch", remote], f"Fetch {remote}")

    def status(self) -> str:
        """Get status output."""
        return self.run("status", "Check Status")

    def current_branch(self) -> str:
        """Get current branch name."""
        if self.dry_run:
            return "dry-run-branch"
        return self.run("rev-parse --abbrev-ref HEAD", check=True)
