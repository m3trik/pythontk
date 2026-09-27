# !/usr/bin/python
# coding=utf-8
"""App launcher -- find, start and watch other programs, on Windows and Linux.

One feature, formerly one 1,660-line module, split by job behind the
:class:`AppLauncher` facade. Every member still resolves on the class
(``ptk.AppLauncher.find_app``, ``ptk.AppLauncher._posix_environ``), and every
cross-call is spelled ``AppLauncher.<name>``, so patching the facade reaches it:

* :mod:`._app_launcher` -- ``AppLauncher``, the facade: the three launch shapes
  (``launch`` detached, ``run`` blocking, ``spawn`` bound to this process) and
  ``write_batch_script``.
* :mod:`._discovery` -- where an app is installed: ``find_app``,
  ``resolve_app_path``, ``scan_install_dirs``, ``scan_for_executables``, and
  ``companion_python`` / ``looks_like_python`` (the interpreter a host binary
  pairs with).
* :mod:`._environment` -- the environment a child inherits: ``process_environ``
  (the LIVE block, not the ``os.environ`` snapshot), ``handoff_env``,
  ``desktop_env``, and the user's persisted PATH (``append_to_path``,
  ``is_path_persisted``).
* :mod:`._desktop` -- the interactive desktop: Windows sessions
  (``is_interactive_session``, ``launch_in_session``) and a process's windows
  (``wait_for_ready``, ``get_window_titles``).
* :mod:`._processes` -- running processes (``get_running_processes``,
  ``close_process``, ``process_tree``) and the lifetime binding ``spawn`` puts
  a child under.

The concept modules hold private mixins of the facade. ``process_stream`` (the
line streams ``run`` and ``spawn`` output feeds) and ``x11`` (the Xlib client
the window lookup asks) stay flat beside this package: each is a primitive with
users of its own.

The root registers ``AppLauncher`` (``from pythontk import AppLauncher``); this
package path resolves the same class, loading no module until it is first used
(``lazy_exports``, CODE_STANDARD section 4).
"""

from pythontk.core_utils.module_resolver import lazy_exports

lazy_exports(globals(), {"_app_launcher": "AppLauncher"})
