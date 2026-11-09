# !/usr/bin/python
# coding=utf-8
"""Generic DCC plugin installer (symlink-first, copytree fallback).

DCC plugin folders sit in OS-specific locations (Toolbag in
``%LOCALAPPDATA%``, Painter in ``%USERPROFILE%\\Documents``, etc.). The
*destination resolution* is DCC-specific; the *install strategy* is not.

This module provides the strategy. Adapters supply the destination via a
plain Path object.

Two install paths, tried in order:
  1. ``os.symlink`` -- zero drift, edits to the package source apply
     immediately. Requires Developer Mode (Win 11) or admin.
  2. ``shutil.copytree`` -- works everywhere, but the copy is a *snapshot*:
     it does not track later edits to the source.

That second path is why installs are content-checked rather than merely
presence-checked. A machine without Developer Mode always lands on
copytree, so a plugin installed once and never refreshed silently keeps
serving whatever ops it shipped with -- a package update that adds an op
leaves the DCC answering "unknown op" for it, which surfaces as the
feature quietly not happening rather than as an install error. So
:meth:`PluginInstaller.install_plugin` reinstalls whenever the
destination has drifted from the source (see
:meth:`PluginInstaller.is_plugin_current`), and ``force=True`` is
reserved for rebuilding an install that already matches.

``__pycache__`` and ``*.pyc`` are filtered from the copy path because the
host DCC's Python runtime may not match the workspace's Python that last
imported the source -- shipping stale bytecode causes obscure import
failures inside the DCC.

A bridge declares its plugin once, as data (:class:`RpcPluginSpec`: its name,
its app, where that app looks for plugins), and every bridge's install,
refresh and removal is the same code (:meth:`PluginInstaller.install`,
:meth:`~PluginInstaller.refresh`, :meth:`~PluginInstaller.uninstall`,
:meth:`~PluginInstaller.status`) -- and a panel's *Install in <app>* is one
action over the same spec, for whichever bridge declares one.
"""

from __future__ import annotations

import filecmp
import fnmatch
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union

__all__ = ["PluginInstaller", "RpcPluginSpec"]


@dataclass(frozen=True)
class RpcPluginSpec:
    """One bridge's plugin in its app, as data: what :class:`PluginInstaller`
    installs, refreshes and removes, and what a panel's *Install in <app>*
    names.

    Parameters:
        name: The plugin's folder (its package) in the app's plugins folder.
        app: The app's name as a person reads it (in *Install in <app>*).
        source: The plugin's source folder (holding its ``__init__.py``).
        user_dir: Where the app looks for a person's plugins (``None`` when
            it cannot be found here).
        enable: Turns on the app's own switch to start the plugin, on a
            person's install only (never a refresh); ``None`` when the app has
            none.
        after_install: What a person does for an open app to load it.
    """

    name: str
    app: str
    source: Path
    user_dir: Callable[[], Optional[Path]]
    enable: Optional[Callable[[], Any]] = None
    after_install: str = ""

    def destination(self) -> Optional[Path]:
        """The plugin's folder in the app's plugins folder, or ``None``."""
        root = self.user_dir()
        return Path(root) / self.name if root else None


class _PluginInstallerInternal(object):
    """Internal helpers for PluginInstaller."""

    #: Copy-path exclusions, in ``shutil.ignore_patterns`` form.
    _IGNORE = ("__pycache__", "*.pyc", "*.pyo")

    @staticmethod
    def _is_ignored(rel: Path) -> bool:
        """True when ``ignore_patterns(*_IGNORE)`` would drop *rel* from a copy.

        Derived from ``_IGNORE`` rather than restating it, so the freshness
        check and the copy cannot disagree about what ships. They must not:
        a pattern the copy honours but the check doesn't would mark every
        install stale forever, rebuilding the plugin on every hand-off.
        ``ignore_patterns`` matches plain *names*, and dropping a directory
        drops its subtree — hence matching against each path part.
        """
        return any(
            fnmatch.fnmatch(part, pattern)
            for part in rel.parts
            for pattern in _PluginInstallerInternal._IGNORE
        )

    @staticmethod
    def _shipped_files(plugin_src: Path) -> Iterator[Path]:
        """Source-relative paths of every file an install would copy."""
        for path in plugin_src.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(plugin_src)
            if not _PluginInstallerInternal._is_ignored(rel):
                yield rel


class PluginInstaller(_PluginInstallerInternal):
    """Install a DCC plugin into its host's folder, symlink-first.

    Plugin folders sit in OS-specific places (Toolbag under
    ``%LOCALAPPDATA%``, Painter under Documents, ...). A symlink keeps a
    development checkout live; a copytree fallback covers hosts and
    filesystems that refuse one.
    """

    @staticmethod
    def is_plugin_current(plugin_src: Union[str, Path], dest: Union[str, Path]) -> bool:
        """True when *dest* already serves the current *plugin_src*.

        A symlink install is current when it still points at *plugin_src*.
        A copytree install is current when every shipped source file is
        present at *dest* with identical content.

        Files at *dest* with no counterpart in the source do **not** make
        an install stale: the DCC writes its own ``__pycache__`` beside the
        plugin, and treating that as drift would rebuild on every check.
        The tradeoff is that a file *deleted* from the source lingers until
        some other change triggers a rebuild -- which then wipes it, since
        the rebuild is a fresh copytree.
        """
        plugin_src = Path(plugin_src)
        dest = Path(dest)
        if not plugin_src.is_dir():
            return False

        if dest.is_symlink():
            try:
                return dest.resolve() == plugin_src.resolve()
            except OSError:
                return False
        if not dest.is_dir():
            return False

        for rel in _PluginInstallerInternal._shipped_files(plugin_src):
            installed = dest / rel
            if not installed.is_file():
                return False
            if not filecmp.cmp(plugin_src / rel, installed, shallow=False):
                return False
        return True

    @staticmethod
    def install_plugin(
        plugin_src: Union[str, Path],
        dest: Union[str, Path],
        force: bool = False,
    ) -> Optional[Path]:
        """Install *plugin_src* at *dest*, refreshing it when it has drifted.

        A destination that already matches the source is left untouched, so
        this stays cheap to call on every hand-off. One that does *not*
        match is rebuilt -- without that, the copytree fallback (every
        machine without Developer Mode) would pin the DCC to whatever the
        plugin shipped the first time it was installed.

        Args:
            plugin_src: Source directory containing the plugin's ``__init__.py``.
            dest: Final install location -- the resolved path inside the DCC's
                plugin folder. Parent dirs are created as needed.
            force: Rebuild even when *dest* already matches the source.

        Returns:
            The destination Path on success, or *None* if *plugin_src* is
            missing.
        """
        plugin_src = Path(plugin_src)
        dest = Path(dest)
        if not plugin_src.is_dir():
            return None

        if not force and PluginInstaller.is_plugin_current(plugin_src, dest):
            return dest

        # Tear down any stale install -- symlink, file, or directory.
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        elif dest.exists():
            shutil.rmtree(dest)

        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            os.symlink(plugin_src, dest, target_is_directory=True)
        except (OSError, NotImplementedError):
            # Symlink rejected (no admin / no Developer Mode) -- fall back.
            # Filter __pycache__/*.pyc; whichever Python last imported the
            # source wrote those, and the DCC's runtime may not match.
            shutil.copytree(
                plugin_src,
                dest,
                ignore=shutil.ignore_patterns(*_PluginInstallerInternal._IGNORE),
            )
        return dest

    @staticmethod
    def uninstall_plugin(dest: Union[str, Path]) -> bool:
        """Remove a plugin install at *dest*. Returns True if anything went.

        Safe to call when nothing is there.
        """
        dest = Path(dest)
        if not dest.exists() and not dest.is_symlink():
            return False
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)
        return True

    @staticmethod
    def is_plugin_installed(dest: Union[str, Path]) -> bool:
        """True if *dest* looks like an installed plugin (has ``__init__.py``)."""
        dest = Path(dest)
        return (dest / "__init__.py").is_file()

    # ------------------------------------------------------- by a bridge's spec
    @staticmethod
    def status(spec: RpcPluginSpec) -> str:
        """*spec*'s plugin in its app: ``"current"``, ``"stale"`` (an older
        copy), ``"absent"`` or ``"unresolved"`` (its app's folder not found)."""
        dest = spec.destination()
        if dest is None:
            return "unresolved"
        if not PluginInstaller.is_plugin_installed(dest):
            return "absent"
        return "current" if PluginInstaller.is_plugin_current(spec.source, dest) else "stale"

    @staticmethod
    def install(spec: RpcPluginSpec, force: bool = False, logger: Any = None) -> Optional[Path]:
        """A person's install of *spec*'s plugin in its app (a panel's *Install
        in <app>*): installed or brought up to date, the app's switch to start
        it turned on (:attr:`RpcPluginSpec.enable`), and said to *logger* with
        what loads it in an open app. The plugin's folder, or ``None``."""
        dest = spec.destination()
        if dest is None:
            if logger is not None:
                logger.warning(f"Could not find {spec.app}'s plugins folder: the {spec.name} plugin was not installed.")
            return None
        installed = PluginInstaller.install_plugin(spec.source, dest, force=force)
        if installed is None:
            if logger is not None:
                logger.warning(f"The {spec.name} plugin's source is missing: {spec.source}.")
            return None
        if spec.enable is not None:
            spec.enable()
        if logger is not None:
            logger.info(f"Installed {spec.app}'s {spec.name} plugin: {installed}. {spec.after_install}".rstrip())
        return installed

    @staticmethod
    def refresh(spec: RpcPluginSpec) -> Optional[Path]:
        """Bring an install of *spec*'s plugin that is there up to date (an
        older copy serves stale ops); never install one that is not, nor touch
        the app's switch. The plugin's folder when it was rewritten, else ``None``."""
        if PluginInstaller.status(spec) != "stale":
            return None
        return PluginInstaller.install_plugin(spec.source, spec.destination())

    @staticmethod
    def uninstall(spec: RpcPluginSpec, logger: Any = None) -> bool:
        """Remove *spec*'s plugin from its app's plugins folder, said to
        *logger*. True when something was removed."""
        dest = spec.destination()
        removed = dest is not None and PluginInstaller.uninstall_plugin(dest)
        if logger is not None:
            logger.info(
                f"Removed {spec.app}'s {spec.name} plugin: {dest}." if removed else f"{spec.app}'s {spec.name} plugin is not installed."
            )
        return removed
