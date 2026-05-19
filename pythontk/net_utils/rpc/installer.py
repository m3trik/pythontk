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
  2. ``shutil.copytree`` -- works everywhere; needs an explicit
     ``force=True`` to refresh after the package is updated.

``__pycache__`` and ``*.pyc`` are filtered from the copy path because the
host DCC's Python runtime may not match the workspace's Python that last
imported the source -- shipping stale bytecode causes obscure import
failures inside the DCC.
"""
import os
import shutil
from pathlib import Path
from typing import Optional, Union

__all__ = ["install_plugin", "uninstall_plugin", "is_plugin_installed"]


def install_plugin(
    plugin_src: Union[str, Path],
    dest: Union[str, Path],
    force: bool = False,
) -> Optional[Path]:
    """Install *plugin_src* at *dest*. Idempotent unless *force* is true.

    Args:
        plugin_src: Source directory containing the plugin's ``__init__.py``.
        dest: Final install location -- the resolved path inside the DCC's
            plugin folder. Parent dirs are created as needed.
        force: When True, rebuild the install (removes any existing
            directory/symlink/file at *dest* first).

    Returns:
        The destination Path on success, or *None* if *plugin_src* is
        missing.
    """
    plugin_src = Path(plugin_src)
    dest = Path(dest)
    if not plugin_src.is_dir():
        return None

    if dest.exists() and not force:
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
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
    return dest


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


def is_plugin_installed(dest: Union[str, Path]) -> bool:
    """True if *dest* looks like an installed plugin (has ``__init__.py``)."""
    dest = Path(dest)
    return (dest / "__init__.py").is_file()
