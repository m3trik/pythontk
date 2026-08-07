# !/usr/bin/python
# coding=utf-8
"""Generic JSON-RPC plumbing for plugin-hosted RPC servers.

**Both ends of one protocol.** The wire format cannot drift between a client
and a server that ship together, so this subpackage owns the outside half
(:mod:`.client`) and the inside half (:mod:`.plugin_core`) side by side.

Four modules, one role each:

* :mod:`.client` -- :class:`RpcClient` -- HTTP JSON-RPC client. Subclass
  per host application to bind defaults (port, app finder, label).
* :mod:`.plugin_core` -- :class:`RpcPlugin` / :class:`OpRegistry` /
  :class:`MainThreadMarshaller` -- the server that runs *inside* the host
  application. Standard-library only, so an installed plugin payload can
  carry a verbatim copy where ``pythontk`` is not importable (see the
  module docstring; staged by ``m3trik/scripts/sync_rpc_core.py``).
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
