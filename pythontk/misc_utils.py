# !/usr/bin/python
# coding=utf-8
import functools
import inspect
import collections.abc
import concurrent.futures
from typing import Any, Callable

# from this package:
from pythontk.iter_utils import Iter


class Misc:
    """ """

    @staticmethod
    def cached_property(func: Callable) -> Any:
        """Decorator that converts a method with a single self argument into a property
        that runs the method only once and stores the result, returning the stored
        result on subsequent accesses.

        This is useful for expensive computations that don't change once computed.

        Parameters:
            func: Method to be converted into a cached property.

        Returns:
            A descriptor object that can be used as a decorator.
        """
        from functools import wraps

        attr_name = "_cached_" + func.__name__

        @property
        @wraps(func)
        def _cached_property(self: Any) -> Any:
            if not hasattr(self, attr_name):
                setattr(self, attr_name, func(self))
            return getattr(self, attr_name)

        return _cached_property

    @staticmethod
    def listify(func=None, arg_name=None, threading=False):
        if func is None:
            return lambda func: Misc.listify(func, arg_name=arg_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_args = inspect.getfullargspec(func).args
            if "self" in func_args or "cls" in func_args:
                func_args = func_args[1:]  # skip 'self' or 'cls' argument for methods

            if arg_name and arg_name in func_args:
                arg_index = func_args.index(arg_name)
                if arg_index < len(args):
                    # Argument is in the positional arguments
                    arg = args[arg_index]
                    args = args[:arg_index] + args[arg_index + 1 :]
                else:
                    raise ValueError(f"No argument named '{arg_name}' provided")
            elif args:
                arg_index = 0
                arg = args[arg_index]
                args = args[arg_index + 1 :]
            else:
                raise ValueError("No argument provided")

            # Check if a single item was provided
            single_item = not isinstance(arg, collections.abc.Iterable) or isinstance(
                arg, (str, bytes, bytearray)
            )

            if single_item:
                arg = [arg]

            results = [
                func(*(args[:arg_index] + (x,) + args[arg_index:]), **kwargs)
                for x in arg
            ]

            # If a single item was provided, return a single result
            if single_item:
                return results[0]

            return results

        return wrapper

    @classmethod
    def format_return(cls, lst, orig=None):
        """Return the list element if the given iterable only contains a single element.
        If the list contains multiple elements, always return the full list.
        If the 'orig' arg is a multi-element type then the original format will always be returned.

        Parameters:
                lst (list): An iterable.
                orig (obj): Optionally; derive the return type form the original value.
                                ie. if it was a multi-value type; do not modify the return value.
        Returns:
                (obj/list) dependant on flags.
        """
        orig = isinstance(orig, (list, tuple, set, dict, range))

        try:
            if len(lst) == 1 and not orig and not isinstance(lst, str):
                return lst[0]

        except Exception as e:
            pass
        return lst

    @staticmethod
    def set_attributes(obj, **attributes):
        """Set attributes for a given object.

        Parameters:
                obj (obj): The object to set attributes for.
                attributes (kwargs) = Attributes and their correponding values as keyword args.
        """
        [
            setattr(obj, attr, value)
            for attr, value in attributes.items()
            if attr and value
        ]

    @staticmethod
    def get_attributes(obj, inc=[], exc=[]):
        """Get attributes for a given object.

        Parameters:
                obj (obj): The object to get the attributes of.
                inc (list): Attributes to include. All other will be omitted. Exclude takes dominance over include. Meaning, if the same attribute is in both lists, it will be excluded.
                exc (list): Attributes to exclude from the returned dictionay. ie. [u'Position',u'Rotation',u'Scale',u'renderable',u'isHidden',u'isFrozen',u'selected']

        Returns:
                (dict) {'string attribute': current value}
        """
        filtered = Iter.filter_list(obj.__dict__, inc, exc)
        return {attr: getattr(obj, attr) for attr in filtered}

    @staticmethod
    def has_attribute(cls, attr):
        """This function checks whether a class has a specific static attribute by using `inspect.getattr_static`.
        It does not invoke the class's `__getattr__` method, so it is useful for checking if an attribute is defined
        on the class itself, rather than on its instances.

        Parameters:
                cls (obj): The class to check for the attribute.
                attr (str): The name of the attribute to check.

        :return:
                (bool) True if the class has the attribute, False otherwise.
        """
        import inspect

        try:
            inspect.getattr_static(cls, attr)
            return True
        except AttributeError:
            return False

    @staticmethod
    def get_derived_type(
        obj,
        return_name=False,
        module=None,
        include=[],
        exclude=[],
        filter_by_base_type=False,
    ):
        """Get the base class of a custom object.
        If the type is a standard object, the derived type will be that object's type.

        Parameters:
            obj (str/obj): Object or its objectName.
            return_name (bool): Return the class or the class name.
            module (str): The name of the base class module to check for.
            include (list): Object types to include. All other will be omitted. Exclude takes dominance over include. Meaning, if the same attribute is in both lists, it will be excluded.
            exclude (list): Object types to exclude.
            filter_by_base_type (bool): When using `include`, or `exclude`; Filter by base class name, or derived class name.

        Returns:
            (obj)(string)(None) class or class name if `return_name`. ie. 'DerivedClass' from a custom object with class name: 'CustomClass'
        """
        for cls in obj.__class__.__mro__:
            if (
                not module
                or cls.__module__ == module
                or cls.__module__.split(".")[-1] == module
            ):
                derived_type = cls.__base__.__name__ if filter_by_base_type else cls
                if not (
                    derived_type in exclude
                    and (
                        derived_type in include
                        if include
                        else derived_type not in include
                    )
                ):
                    return derived_type.__name__ if return_name else derived_type

    CYCLEDICT = {}

    @classmethod
    def cycle(cls, sequence, name=None, query=False):
        """Toggle between numbers in a given sequence.
        Used for maintaining toggling sequences for multiple objects simultaniously.
        Each time this function is called, it returns the next number in the sequence
        using the name string as an identifier key.

        Parameters:
                sequence (list): sequence to cycle through. ie. [1,2,3].
                name (str): identifier. used as a key to get the sequence value from the dict.

        ex. cycle([0,1,2,3,4], 'componentID')
        """
        try:
            if query:  # return the value without changing it.
                return cls.CYCLEDICT[name][-1]  # get the current value ie. 0

            value = cls.CYCLEDICT[
                name
            ]  # check if key exists. if so return the value. ie. value = [1,2,3]

        except KeyError:  # else create sequence list for the given key
            cls.CYCLEDICT[name] = [i for i in sequence]  # ie. {name:[1,2,3]}

        value = cls.CYCLEDICT[name][0]  # get the current value. ie. 1
        cls.CYCLEDICT[name] = cls.CYCLEDICT[name][1:] + [
            value
        ]  # move the current value to the end of the list. ie. [2,3,1]
        return value  # return current value. ie. 1

    @staticmethod
    def are_similar(a, b, tolerance=0.0):
        """Check if the two numberical values are within a given tolerance.
        Supports nested lists.

        Parameters:
                a (obj)(tuple): The first object(s) to compare.
                b (obj)(tuple): The second object(s) to compare.
                tolerance (float) = The maximum allowed variation between the values.

        Returns:
                (bool)

        Example: are_similar(1, 10, 9)" #returns: True
        Example: are_similar(1, 10, 8)" #returns: False
        """
        func = (
            lambda a, b: abs(a - b) <= tolerance
            if isinstance(a, (int, float))
            else True
            if isinstance(a, (list, set, tuple)) and are_similar(a, b, tolerance)
            else a == b
        )
        return all(map(func, Iter.make_iterable(a), Iter.make_iterable(b)))

    @staticmethod
    def randomize(lst, ratio=1.0):
        """Random elements from the given list will be returned with a quantity determined by the given ratio.
        A value of 0.5 will return 50% of the original elements in random order.

        Parameters:
                lst (tuple): A list to randomize.
                ratio (float) = A value of 0.0-1. (default: 100%) With 0 representing 0% and
                                1 representing 100% of the given elements returned in random order.
        Returns:
                (list)

        Example: randomize(range(10), 1.0) #returns: [8, 4, 7, 6, 0, 5, 9, 1, 3, 2]
        Example: randomize(range(10), 0.5) #returns: [7, 6, 4, 2, 8]
        """
        import random

        lower, upper = 0.0, ratio if ratio <= 1 else 1.0  # end result range.
        normalized = lower + (upper - lower) * len(lst)  # returns a float value.
        randomized = random.sample(lst, int(normalized))

        return randomized


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# deprecated:
# --------------------------------------------------------------------------------------------
