# !/usr/bin/python
# coding=utf-8
import sys
import inspect


__package__ = 'pythontk'
__version__ = '0.5.6'


def __getattr__(attr):
	"""This function dynamically imports a module and returns an attribute from the module. 

	:Parameters:
		attr (str): The name of the attribute to be imported. The name should be in the format 'module_name.attribute_name'.

	:Return:
		(obj) The attribute specified by the `attr` argument.

	:Raises:
		AttributeError: If the specified attribute is not found in either the original module or the 'Core' module within the package.

	:Example:
		<package>.__getattr__('module1.attribute1') #returns: <attribute1 value>
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

# --------------------------------------------------------------------------------------------









# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# deprecated:
# --------------------------------------------------------------------------------------------