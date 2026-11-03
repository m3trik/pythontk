# !/usr/bin/python
# coding=utf-8
"""Operation catalog -- the named, typed actions every front end calls alike.

One unit of API that people and agents share: an operation has an id
(``mesh.mirror``), typed arguments, a result, an effect (what it may change)
and a dry run, and the command line (``python -m pythontk ops``), MCP and the
panels are thin front ends over it. The vocabulary and the result shape are
published as JSON Schema, so an operation another language declares (a C# one
in a Unity project) and a Python one read alike in one catalog.

* :mod:`.operation` -- ``Operation``, one declaration (an id, the function it
  calls, its effect and host; the function's signature and docstring are its
  schema), and its published forms ``OperationEntry`` and
  ``OperationCatalogDocument``.
* :mod:`.op_context` -- ``OpContext``, what a running operation is handed:
  ``should_change`` / ``should_write`` (the dry run), issues, progress, cancel.
* :mod:`.outcome` -- ``OpRequest`` (one call as the batch protocol carries
  it), ``Outcome``, ``Issue``, ``Change``, ``OpCall``: what a run is asked
  and hands back, as typed payloads.
* :mod:`.operation_host` -- ``OperationHost``, the seam a host's runner
  implements, and ``PureOperationHost``, this process.
* :mod:`.batch_operation_host` -- ``BatchOperationHost``, the runner a host
  library configures (each call in a FRESH instance of the application:
  request file in, outcome file out), and ``SceneFiles`` / ``PureSceneFiles``,
  the scene side it builds inside the application.
* :mod:`.session_operation_host` -- ``SessionOperationHost``, the runner a
  long-lived front end keeps per session instead: one instance per host,
  started on the first call and shared by every runner of the session,
  serving the same run folders from a ``JobQueue``.
* :mod:`.operation_server` -- ``OperationServer``, the catalog as an MCP server
  (``python -m pythontk ops mcp``): one tool per operation, reads and dry runs
  on one server and the changes on another.
* :mod:`._operation_catalog` -- ``OperationCatalog``, the facade: discover
  every provider's committed catalog, validate and run a call, and the ``ops``
  command line.
* :mod:`.operation_provider` -- ``OperationProvider``, one package's cycle:
  onboarding, its catalog written and checked.

No library function is wrapped as an operation: the whole public API is
``ApiSurface``'s (``core_utils/api_surface``, below this package), found by
pythontk's own ``api.search`` / ``api.describe`` and called from code a host's
``python.exec`` runs.

The root registers every public name (``from pythontk import OperationCatalog``);
this package declares none of its own (CODE_STANDARD sections 3 and 4).
"""
