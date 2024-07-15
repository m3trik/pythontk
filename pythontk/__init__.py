# !/usr/bin/python
# coding=utf-8
import inspect
import importlib
import pkgutil

__package__ = "pythontk"
__version__ = "0.7.19"

CLASS_TO_MODULE = {}
METHOD_TO_MODULE = {}
CLASS_METHOD_TO_MODULE = {}
IMPORTED_MODULES = {}


def build_dictionaries():
    for importer, modname, ispkg in pkgutil.walk_packages(__path__, __name__ + "."):
        module = importlib.import_module(modname)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            CLASS_TO_MODULE[name] = modname
            for method_name, _ in inspect.getmembers(obj, inspect.isfunction):
                METHOD_TO_MODULE[method_name] = (modname, name)
            for method_name, _ in inspect.getmembers(obj, inspect.ismethod):
                CLASS_METHOD_TO_MODULE[method_name] = (modname, name)


def import_module(module_name):
    if module_name not in IMPORTED_MODULES:
        IMPORTED_MODULES[module_name] = importlib.import_module(module_name)
    return IMPORTED_MODULES[module_name]


def get_attribute_from_module(module, attribute_name, class_name=None):
    if class_name:
        class_obj = getattr(module, class_name)
        return getattr(class_obj, attribute_name)
    return getattr(module, attribute_name)


def __getattr__(name):
    if name in CLASS_TO_MODULE:
        module = import_module(CLASS_TO_MODULE[name])
        return get_attribute_from_module(module, name)

    if name in METHOD_TO_MODULE:
        module_name, class_name = METHOD_TO_MODULE[name]
        module = import_module(module_name)
        return get_attribute_from_module(module, name, class_name)

    if name in CLASS_METHOD_TO_MODULE:
        module_name, class_name = CLASS_METHOD_TO_MODULE[name]
        module = import_module(module_name)
        return get_attribute_from_module(module, name, class_name)

    raise AttributeError(f"module {__package__} has no attribute '{name}'")


# --------------------------------------------------------------------------------------------
# Build dictionaries at the start
build_dictionaries()

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
