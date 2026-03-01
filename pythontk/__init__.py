# !/usr/bin/python
# coding=utf-8
from pythontk.core_utils.module_resolver import bootstrap_package

__package__ = "pythontk"
__version__ = "0.7.86"

"""Expose toolkit utilities with explicit resolver include maps for clarity."""


DEFAULT_INCLUDE = {
    "audio_utils._audio_utils": "AudioUtils",
    "img_utils._img_utils": "*",
    "img_utils.map_factory": ["MapFactory"],
    "img_utils.map_registry": "MapRegistry",
    "str_utils._str_utils": "*",
    "vid_utils._vid_utils": "*",
    "file_utils._file_utils": "*",
    "file_utils.metadata": "Metadata",
    "iter_utils._iter_utils": "*",
    "math_utils._math_utils": "*",
    "math_utils.progression": "ProgressionCurves",
    "core_utils._core_utils": "*",
    "core_utils.help_mixin": "HelpMixin",
    "core_utils.package_manager": "PackageManager",
    "core_utils.git": "Git",
    "core_utils.class_property": "ClassProperty",
    "core_utils.logging_mixin": "LoggingMixin",
    "core_utils.table_mixin": "TableMixin",
    "core_utils.namespace_handler": "NamespaceHandler",
    "core_utils.namedtuple_container": "NamedTupleContainer",
    "core_utils.hierarchy_diff": "HierarchyDiff",
    "core_utils.singleton_mixin": "SingletonMixin",
    "core_utils.module_reloader": ["ModuleReloader", "reload_package"],
    "core_utils.execution_monitor._execution_monitor": "ExecutionMonitor",
    "core_utils.app_launcher": "AppLauncher",
    "core_utils.cli": "CLI",
    # Hierarchy utils
    "core_utils.hierarchy_utils.hierarchy_indexer": "HierarchyIndexer",
    "core_utils.hierarchy_utils.hierarchy_matching": "HierarchyMatching",
    "core_utils.hierarchy_utils.hierarchy_analyzer": [
        "HierarchyDifference",
        "HierarchyAnalyzer",
    ],
    "net_utils.ssh_client": "SSHClient",
    "net_utils.credentials": "Credentials",
    "net_utils._net_utils": "NetUtils",
    "str_utils.fuzzy_matcher": "FuzzyMatcher",
}


bootstrap_package(globals(), include=DEFAULT_INCLUDE)


__all__ = [
    "CoreUtils",
    "AppLauncher",
    "HelpMixin",
    "LoggingMixin",
    "TableMixin",
    "NamespaceHandler",
    "NamedTupleContainer",
    "SingletonMixin",
    "PackageManager",
    "ClassProperty",
    "HierarchyDiff",
    "ModuleReloader",
    "reload_package",
    "ExecutionMonitor",
    "CLI",
    "SSHClient",
    "Credentials",
    "NetUtils",
    "FileUtils",
    "Metadata",
    "ImgUtils",
    "IterUtils",
    "MathUtils",
    "ProgressionCurves",
    "StrUtils",
    "FuzzyMatcher",
    "AudioUtils",
    "VidUtils",
]
# Test: 222117
