# !/usr/bin/python
# coding=utf-8
import inspect
import importlib
import pkgutil


__package__ = "pythontk"
__version__ = '0.6.9'


# Define dictionaries to map class names, method names, and class method names to their respective modules
CLASS_TO_MODULE = {}
METHOD_TO_MODULE = {}
CLASS_METHOD_TO_MODULE = {}

# Build the dictionaries by iterating over all submodules of the package
for importer, modname, ispkg in pkgutil.walk_packages(__path__, __name__ + "."):
    module = importlib.import_module(modname)
    for name, obj in module.__dict__.items():
        if inspect.isclass(obj):
            CLASS_TO_MODULE[obj.__name__] = modname
            for method_name, method_obj in inspect.getmembers(
                obj, predicate=inspect.isfunction
            ):
                METHOD_TO_MODULE[method_name] = (modname, obj.__name__)
            for method_name, method_obj in inspect.getmembers(
                obj, predicate=inspect.ismethod
            ):
                CLASS_METHOD_TO_MODULE[method_name] = (modname, obj.__name__)

# Define a dictionary to store imported module objects
IMPORTED_MODULES = {}


def __getattr__(name):
    # Check if the requested attribute is a class we need to import
    if name in CLASS_TO_MODULE:
        module_name = CLASS_TO_MODULE[name]
        if module_name not in IMPORTED_MODULES:
            # If the module hasn't been imported yet, import it and add it to the dictionary
            module = importlib.import_module(module_name)
            IMPORTED_MODULES[module_name] = module
        else:
            module = IMPORTED_MODULES[module_name]
        # Return the requested class object from the module
        return getattr(module, name)

    # Check if the requested attribute is a method we need to import
    elif name in METHOD_TO_MODULE:
        module_name, class_name = METHOD_TO_MODULE[name]
        if module_name not in IMPORTED_MODULES:
            # If the module hasn't been imported yet, import it and add it to the dictionary
            module = importlib.import_module(module_name)
            IMPORTED_MODULES[module_name] = module
        else:
            module = IMPORTED_MODULES[module_name]
        # Get the class object and return the requested method object from it
        class_obj = getattr(module, class_name)
        return getattr(class_obj, name)

    # Check if the requested attribute is a class method we need to import
    elif name in CLASS_METHOD_TO_MODULE:
        module_name, class_name = CLASS_METHOD_TO_MODULE[name]
        if module_name not in IMPORTED_MODULES:
            # If the module hasn't been imported yet, import it and add it to the dictionary
            module = importlib.import_module(module_name)
            IMPORTED_MODULES[module_name] = module
        else:
            module = IMPORTED_MODULES[module_name]
        # Get the class object and return the requested class method object from it
        class_obj = getattr(module, class_name)
        return getattr(class_obj, name)

    # If the requested attribute is not a class, method, or class method we handle, raise an AttributeError
    raise AttributeError(f"module {__package__} has no attribute '{name}'")


# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# deprecated:
# --------------------------------------------------------------------------------------------


# visited = set()
# def searchClassesForAttr(module, attr, breakOnMatch=True):
#   """Searches all classes in the given module for the given attribute, excluding any classes starting with an underscore.

#   Parameters:
#       module (str)(module): The module to search for classes and attributes.
#       attr (str): The name of an attribute to search for.
#       breakOnMatch (bool): Return only the first found attribute.

#   Returns:
#       (obj) The found attribute.

#   :raise AttributeError: If the given attribute is not found in any of the classes in the given module.
#   """
#   if module in visited:
#       raise AttributeError("Infinite recursion detected")
#   visited.add(module)

#   found_attrs = []
#   for clss in [o for n, o in inspect.getmembers(module) if inspect.isclass(o) and not n.startswith('_')]:
#       try:
#           if breakOnMatch:
#               found_attrs = getattr(clss, attr)
#               break
#           found_attrs.append(getattr(clss, attr))
#       except AttributeError:
#           continue
#   visited.remove(module)

#   if not found_attrs:
#       raise AttributeError(f"Module '{module.__name__}' has no attribute '{attr}'")
#   return found_attrs


# def import_submodules(package, filetypes=('py', 'pyc', 'pyd'), ignoreStartingWith=('.', '_')):
#   '''Import submodules to the given package.

#   Parameters:
#       package (str/obj): A python package.
#       filetypes (str)(tuple): Filetype extension(s) to include.
#       ignoreStartingWith (str)(tuple): Ignore submodules starting with given chars.

#   Returns:
#       (list) the imported modules.

#   Example: import_submodules(__name__)
#   '''
#   if isinstance(package, str):
#       package = sys.modules[package]
#   if not package:
#       return print(f"# Error: {__file__} in import_submodules\n#\tThe package argument is either not a valid string or it does not correspond to a module in sys.modules.")

#   package_path = os.path.dirname(package.__file__)
#   modules = []
#   for dirpath, _, filenames in os.walk(package_path):
#       for filename in filenames:
#           if any(filename.endswith(f".{ext}") for ext in filetypes):
#               module_name, _ = os.path.splitext(filename)
#               if any(module_name.startswith(ignore) for ignore in ignoreStartingWith):
#                   continue
#               try:
#                   module = importlib.import_module(f"{package.__name__}.{module_name}")
#                   modules.append(module)
#               except ImportError:
#                   pass
#   return modules


# def addMembers(module, ignoreStartingWith='_'):
#   '''Expose class members at module level.

#   Parameters:
#       module (str/obj): A python module.
#       ignoreStartingWith (str)(tuple): Ignore class members starting with given chars.

#   Example: addMembers(__name__)
#   '''
#   if isinstance(module, str):
#       module = sys.modules[module]
#   if not module:
#       return

#   classes = inspect.getmembers(module, inspect.isclass)

#   for cls_name, clss in classes:
#       cls_members = [(o, getattr(clss, o)) for o in dir(clss) if not o.startswith(ignoreStartingWith)]
#       for name, mem in cls_members:
#           vars(module)[name] = mem
