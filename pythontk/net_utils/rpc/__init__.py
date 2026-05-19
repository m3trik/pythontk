# !/usr/bin/python
# coding=utf-8
"""Generic JSON-RPC plumbing for plugin-hosted RPC servers.

Three modules, one role each:

* :mod:`.client` -- :class:`RpcClient` -- HTTP JSON-RPC client. Subclass
  per host application to bind defaults (port, app finder, label).
* :mod:`.installer` -- ``install_plugin`` / ``uninstall_plugin`` /
  ``is_plugin_installed`` -- symlink-first, copytree-fallback strategy.
  Destination resolution is the adapter's job; strategy lives here.
* :mod:`.job` -- :class:`Call` / :class:`Result` / :func:`run_batch` --
  one-shot batch pipeline over :class:`RpcClient`.

Public symbols are exposed at the top of :mod:`pythontk` via
``DEFAULT_INCLUDE`` (see ``pythontk/__init__.py``). No re-exports here --
this subpackage's ``__init__.py`` is intentionally docstring-only, in
line with the root CLAUDE.md convention.
"""
