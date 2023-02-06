# !/usr/bin/python
# coding=utf-8
import os, sys
import inspect


__package__ = 'pythontk'
__version__ = '0.5.7'


def __getattr__(attr):
	"""This function dynamically imports a module and returns an attribute from the module. 

	:Parameters:
		attr (str): The name of the attribute to be imported. The name should be in the format 
					'module_name.attribute_name' or just 'attribute_name'.
	:Return:
		(obj) The attribute specified by the `attr` argument.

	:Raises:
		AttributeError: If the specified attribute is not found in either the original module 
						or the 'Core' module within the package.
	:Example:
		<package>.__getattr__('module1.attribute1') #returns: <attribute1 value>
		<package>.__getattr__('attribute1') #returns: <attribute1 value>
	"""
	try:
		module = __import__(f"{__package__}.{attr}", fromlist=[f"{attr}"])
		setattr(sys.modules[__name__], attr, getattr(module, attr))
		return getattr(module, attr)

	except (ValueError, ModuleNotFoundError):
		module = __import__(f"{__package__}.Core", fromlist=["Core"])
		return getattr(module, attr)

	except AttributeError as error:
		raise AttributeError(f"Module '{__package__}' has no attribute '{attr}'") from error


visited = set()
def searchClassesForAttr(module, attr, breakOnMatch=True):
	"""Searches all classes in the given module for the given attribute, excluding any classes starting with an underscore.

	:Parameters:
		module (module): The module to search for classes and attributes.
		attr (str): The name of an attribute to search for.
		breakOnMatch (bool): Return only the first found attribute.

	:Return:
		(obj) The found attribute.

	:raise AttributeError: If the given attribute is not found in any of the classes in the given module.
	"""
	if module in visited:
		raise AttributeError("Infinite recursion detected")
	visited.add(module)

	found_attrs = []
	for clss in [o for n, o in inspect.getmembers(module) if inspect.isclass(o) and not n.startswith('_')]:
		try:
			if breakOnMatch:
				found_attrs = getattr(clss, attr)
				break
			found_attrs.append(getattr(clss, attr))
		except AttributeError:
			continue
	visited.remove(module)

	if not found_attrs:
		raise AttributeError(f"Module '{module.__name__}' has no attribute '{attr}'")
	return found_attrs

def import_submodules(package, filetypes=('py', 'pyc', 'pyd'), ignoreStartingWith=('.', '_')):
	'''Import submodules to the given package.

	:Parameters:
		package (str)(obj): A python package.
		filetypes (str)(tuple): Filetype extension(s) to include.
		ignoreStartingWith (str)(tuple): Ignore submodules starting with given chars.

	:Return:
		(list) the imported modules.

	:Example: import_submodules(__name__)
	'''
	if isinstance(package, str):
		package = sys.modules[package]
	if not package:
		return print(f"# Error: {__file__} in import_submodules\n#\tThe package argument is either not a valid string or it does not correspond to a module in sys.modules.")

	package_path = os.path.dirname(package.__file__)
	modules = []
	for dirpath, _, filenames in os.walk(package_path):
		for filename in filenames:
			if any(filename.endswith(f".{ext}") for ext in filetypes):
				module_name, _ = os.path.splitext(filename)
				if any(module_name.startswith(ignore) for ignore in ignoreStartingWith):
					continue
				try:
					module = importlib.import_module(f"{package.__name__}.{module_name}")
					modules.append(module)
				except ImportError:
					pass
	return modules


def addMembers(module, ignoreStartingWith='_'):
	'''Expose class members at module level.

	:Parameters:
		module (str)(obj): A python module.
		ignoreStartingWith (str)(tuple): Ignore class members starting with given chars.

	:Example: addMembers(__name__)
	'''
	if isinstance(module, str):
		module = sys.modules[module]
	if not module:
		return

	classes = inspect.getmembers(module, inspect.isclass)

	for cls_name, clss in classes:
		cls_members = [(o, getattr(clss, o)) for o in dir(clss) if not o.startswith(ignoreStartingWith)]
		for name, mem in cls_members:
			vars(module)[name] = mem


def lazy_import(importer_name, to_import):
	'''Return the importing module and a callable for lazy importing.

	:Parmameters:
		importer_name (str): Represents the module performing the
				import to help facilitate resolving relative imports.
		to_import (list): An iterable of the modules to be potentially imported (absolute
				or relative). The 'as' form of importing is also supported. e.g. 'pkg.mod as spam'
	:Return:
		(tuple) (importer module, the callable to be set to '__getattr__')

	:Example: mod, __getattr__ = lazy_import(__name__, modules_list)
	'''
	module = importlib.import_module(importer_name)
	import_mapping = {}
	for name in to_import:
		importing, _, binding = name.partition(' as ')
		if not binding:
			_, _, binding = importing.rpartition('.')
		import_mapping[binding] = importing

	def __getattr__(name):
		if name not in import_mapping:
			message = f'module {importer_name!r} has no attribute {name!r}'
			raise AttributeError(message)
		importing = import_mapping[name]
		# imortlib.import_module() implicitly sets submodules on this module as appropriate for direct imports.
		imported = importlib.import_module(importing, module.__spec__.parent)
		setattr(module, name, imported)

		return imported

	return module, __getattr__


def appendPaths(rootDir, ignoreStartingWith=('.', '__'), verbose=False):
	'''Append all sub-directories of the given 'rootDir' to the python path.

	:Parameters:
		rootDir (str): Sub-directories of this directory will be appended to the system path.
		ignoreStartingWith (str)(tuple): Ignore directories starting with the given chars.
		verbose (bool): Output the results to the console. (Debug)
	'''
	path = os.path.dirname(os.path.abspath(rootDir))
	sys.path.insert(0, path)
	if verbose:
		print (path)

	# recursively append subdirectories to the system path.
	for root, dirs, files in os.walk(path):
		dirs[:] = [d for d in dirs if not d.startswith(ignoreStartingWith)]
		for dir_name in dirs:
			dir_path = os.path.join(root, dir_name)
			sys.path.insert(0, dir_path)
			if verbose:
				print (dir_path)

# --------------------------------------------------------------------------------------------









# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# deprecated:
# --------------------------------------------------------------------------------------------