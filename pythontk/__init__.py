# !/usr/bin/python
# coding=utf-8
from pythontk.core_utils.module_resolver import bootstrap_package

__package__ = "pythontk"
__version__ = "0.7.30"

"""Expose toolkit utilities with explicit resolver include maps for clarity."""


DEFAULT_INCLUDE = {
    "core_utils.help_mixin": "HelpMixin",
    "core_utils._core_utils": "CoreUtils",
    "core_utils.pkg_manager": ["PkgManager", "PkgVersionCheck", "PkgVersionUtils"],
    "core_utils.class_property": "ClassProperty",
    "core_utils.logging_mixin": "LoggingMixin",
    "core_utils.namespace_handler": "NamespaceHandler",
    "core_utils.namedtuple_container": "NamedTupleContainer",
    "core_utils.hierarchy_diff": "HierarchyDiff",
    "core_utils.singleton_mixin": "SingletonMixin",
    "file_utils._file_utils": "FileUtils",
    "img_utils._img_utils": "ImgUtils",
    "iter_utils._iter_utils": "IterUtils",
    "math_utils._math_utils": "MathUtils",
    "math_utils.progression": "ProgressionCurves",
    "str_utils._str_utils": "StrUtils",
    "str_utils.fuzzy_matcher": "FuzzyMatcher",
    "vid_utils._vid_utils": "VidUtils",
}


bootstrap_package(globals(), include=DEFAULT_INCLUDE)


__all__ = [
    "CoreUtils",
    "HelpMixin",
    "LoggingMixin",
    "NamespaceHandler",
    "NamedTupleContainer",
    "SingletonMixin",
    "PkgManager",
    "PkgVersionCheck",
    "PkgVersionUtils",
    "ClassProperty",
    "HierarchyDiff",
    "FileUtils",
    "ImgUtils",
    "IterUtils",
    "MathUtils",
    "ProgressionCurves",
    "StrUtils",
    "FuzzyMatcher",
    "VidUtils",
]
