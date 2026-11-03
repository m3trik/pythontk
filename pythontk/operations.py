# !/usr/bin/python
# coding=utf-8
"""pythontk's operations: the named actions agents and front ends run in any Python.

Each one is a pythontk function, declared once with its id and effect;
``ptk.OperationProvider`` turns these declarations into the committed
``operations.json`` beside this file, which ``python -m pythontk ops`` and the
MCP server read. After changing one (or its function's signature or
docstring), rewrite it: ``python -m pythontk ops apply operations.catalog
module=pythontk.operations``. The ``operations.*`` here are the engine's own
upkeep, so a codebase is onboarded and kept in line the way any operation runs;
the ``api.*`` find and read every registered package's whole public API
(``ptk.ApiSurface``, from the source: nothing is imported), which a host's
``python.exec`` then runs: ``python.exec`` itself for a function no
application owns, in a plain Python pythontk registers as the ``python`` host
(``PureSceneFiles.operation_host``).
"""

from pythontk.core_utils.operation_catalog.operation import Operation

OPERATIONS = (
    Operation(
        "api.search",
        "pythontk:ApiSurface.search",
        title="Find a public symbol",
    ),
    Operation(
        "api.describe",
        "pythontk:ApiSurface.describe",
        title="Read a public symbol",
    ),
    Operation(
        "python.exec",
        "pythontk:PythonConsole.run",
        effect="code",
        title="Run Python",
        host="python",
    ),
    Operation(
        "instructions.check",
        "pythontk:AgentInstructions.check",
        title="Check agent instructions",
    ),
    Operation(
        "instructions.sync",
        "pythontk:AgentInstructions.sync",
        effect="files",
        title="Sync agent instructions",
    ),
    Operation(
        "instructions.measure",
        "pythontk:AgentInstructions.measure",
        title="Measure what a session loads",
    ),
    Operation(
        "instructions.duplicates",
        "pythontk:AgentInstructions.duplicates",
        title="Find duplicated rules",
    ),
    Operation(
        "operations.init",
        "pythontk:OperationProvider.init",
        effect="files",
        title="Onboard a package",
    ),
    Operation(
        "operations.catalog",
        "pythontk:OperationProvider.write_catalog",
        effect="files",
        title="Rewrite a package's catalog",
    ),
    Operation(
        "operations.check",
        "pythontk:OperationProvider.check",
        title="Check every package's operations",
    ),
)
