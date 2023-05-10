# !/usr/bin/python
# coding=utf-8
# from this package:
from pythontk.itertk import Iter


class Core:
    """ """

    def listify(func):
        """Decorator that allows a function to take a single value or a list of values as its first argument.
        This version executes the function sequentially on the elements of the input list.
        """

        def wrapper(lst, *args, **kwargs):
            input_list = Iter.makeList(lst)

            result = [func(x, *args, **kwargs) for x in input_list]

            return Core.formatReturn(result, lst)

        return wrapper

    def listify_threaded(func):
        """Decorator that allows a function to take a single value or a list of values as its first argument.
        This version uses threading to execute the function in parallel on different elements of the input list.
        """
        from concurrent.futures import ThreadPoolExecutor

        def wrapper(lst, *args, **kwargs):
            input_list = Iter.makeList(lst)

            with ThreadPoolExecutor() as executor:
                result = list(
                    executor.map(lambda x: func(x, *args, **kwargs), input_list)
                )

            return Core.formatReturn(result, lst)

        return wrapper

    def listify_async(func):
        """Decorator that allows a function to take a single value or a list of values as its first argument.

        The decorated function will be called with each element of the input list as its first argument,
        and with any additional positional or keyword arguments passed to the decorated function.

        Parameters:
            func (callable): The function to be decorated.

        Returns:
            The result of each function call will be collected in a list, and the list will be returned if the input argument
            is a list. If the input argument is not a list, the result of the function call will be returned directly.
        """
        import asyncio

        async def wrapper(lst, *args, **kwargs):
            input_list = Iter.makeList(lst)

            result = await asyncio.gather(
                *[func(x, *args, **kwargs) for x in input_list]
            )

            return Core.formatReturn(result, lst)

        return wrapper

    @classmethod
    def formatReturn(cls, rtn, orig=None):
        """Return the list element if the given iterable only contains a single element.
        If the list contains multiple elements, always return the full list.
        If the 'orig' arg is a multi-element type then the original format will always be returned.

        Parameters:
                rtn (list): An iterable.
                orig (obj): Optionally; derive the return type form the original value.
                                ie. if it was a multi-value type; do not modify the return value.
        Returns:
                (obj/list) dependant on flags.
        """
        orig = isinstance(orig, (list, tuple, set, dict, range))

        try:
            if len(rtn) == 1 and not orig and not isinstance(rtn, str):
                return rtn[0]

        except Exception as e:
            pass
        return rtn

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
    def getAttributes(obj, inc=[], exc=[]):
        """Get attributes for a given object.

        Parameters:
                obj (obj): The object to get the attributes of.
                inc (list): Attributes to include. All other will be omitted. Exclude takes dominance over include. Meaning, if the same attribute is in both lists, it will be excluded.
                exc (list): Attributes to exclude from the returned dictionay. ie. [u'Position',u'Rotation',u'Scale',u'renderable',u'isHidden',u'isFrozen',u'selected']

        Returns:
                (dict) {'string attribute': current value}
        """
        filtered = Iter.filterList(obj.__dict__, inc, exc)
        return {attr: getattr(obj, attr) for attr in filtered}

    @staticmethod
    def hasAttribute(cls, attr):
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

    cycleDict = {}

    @staticmethod
    def getDerivedType(
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
                return cls.cycleDict[name][-1]  # get the current value ie. 0

            value = cls.cycleDict[
                name
            ]  # check if key exists. if so return the value. ie. value = [1,2,3]

        except KeyError:  # else create sequence list for the given key
            cls.cycleDict[name] = [i for i in sequence]  # ie. {name:[1,2,3]}

        value = cls.cycleDict[name][0]  # get the current value. ie. 1
        cls.cycleDict[name] = cls.cycleDict[name][1:] + [
            value
        ]  # move the current value to the end of the list. ie. [2,3,1]
        return value  # return current value. ie. 1

    @staticmethod
    def areSimilar(a, b, tol=0.0):
        """Check if the two numberical values are within a given tolerance.
        Supports nested lists.

        Parameters:
                a (obj)(tuple): The first object(s) to compare.
                b (obj)(tuple): The second object(s) to compare.
                tol (float) = The maximum allowed variation between the values.

        Returns:
                (bool)

        Example: areSimilar(1, 10, 9)" #returns: True
        Example: areSimilar(1, 10, 8)" #returns: False
        """
        func = (
            lambda a, b: abs(a - b) <= tol
            if isinstance(a, (int, float))
            else True
            if isinstance(a, (list, set, tuple)) and areSimilar(a, b, tol)
            else a == b
        )
        return all(map(func, Iter.makeList(a), Iter.makeList(b)))

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


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# deprecated:
# --------------------------------------------------------------------------------------------
