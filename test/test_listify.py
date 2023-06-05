#!/usr/bin/python
# coding=utf-8
import os
import unittest
import inspect
from pythontk import Iter


class Misc:
    """ """

    @classmethod
    def listify(cls, func=None, *, arg_name=None, threading=False):
        """Decorator that allows a function to take either a single value or a list of values for a specific argument.

        This decorator enhances a function to handle both individual values and lists for a given argument.
        When the function is called with a list, the function is applied to each element of the list.
        If the 'threading' parameter is set to True, these function calls will be executed in parallel
        using Python's built-in threading.

        The argument to be listified can be specified with the 'arg_name' parameter.
        If no argument is specified, the decorator defaults to listifying the first positional argument.

        Parameters:
            func (Callable): The function to be enhanced.
            arg_name (str, optional): The name of the argument to be listified.
            threading (bool, optional): Whether to use threading to apply the function to
                multiple elements of a list simultaneously.

        Returns:
            Callable: The enhanced function.
        """
        import functools

        if func is None:  # decorator was called with arguments, return a decorator
            return lambda func: cls.listify(
                func, arg_name=arg_name, threading=threading
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine the argument to listify
            if arg_name is None:  # use the first positional argument
                if not args:
                    return func(
                        *args, **kwargs
                    )  # No positional arguments, call the function without listifying
                arg_value = args[0]
                args = args[1:]
            else:  # use the specified keyword argument
                if arg_name not in kwargs:
                    return func(
                        *args, **kwargs
                    )  # Argument not present, call the function without listifying
                arg_value = kwargs.pop(arg_name)

            # Ensure the argument value is a list
            was_single_value = not isinstance(arg_value, (list, tuple, set, range))
            arg_value = Iter.make_list(arg_value)

            # Apply the function to each item in the list
            if threading:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor() as executor:
                    if arg_name is None:
                        results = list(
                            executor.map(lambda x: func(x, *args, **kwargs), arg_value)
                        )
                    else:
                        results = list(
                            executor.map(
                                lambda x: func(*args, **{**kwargs, arg_name: x}),
                                arg_value,
                            )
                        )
            else:
                if arg_name is None:
                    results = [func(x, *args, **kwargs) for x in arg_value]
                else:
                    results = [
                        func(*args, **{**kwargs, arg_name: x}) for x in arg_value
                    ]

            # If the input was a single value, return a single value
            if was_single_value:
                return results[0]

            return results

        return wrapper


class Main(unittest.TestCase):
    """Main test class."""

    def perform_test(self, *cases):
        """Execute the test cases."""
        for case in cases:
            if isinstance(case, dict):
                for expression, expected_result in case.items():
                    method_name = str(expression).split("(")[0]
                    self._test_case(expression, method_name, expected_result)
            elif isinstance(case, tuple) and len(case) == 2:
                expression, expected_result = case
                if isinstance(expression, str):
                    method_name = str(expression).split("(")[0]
                else:
                    method_name = expression.__class__.__name__
                self._test_case(expression, method_name, expected_result)

    def _test_case(self, expression, method_name, expected_result):
        try:
            if isinstance(expression, str):
                path = os.path.abspath(inspect.getfile(eval(method_name)))
            else:
                path = os.path.abspath(inspect.getfile(expression.__class__))
        except (TypeError, IOError):
            path = ""

        if isinstance(expression, str):
            result = eval(expression)
        else:
            result = expression

        self.assertEqual(
            result,
            expected_result,
            f"\n\n# Error: {path}\n#\t{method_name}\n#\tExpected {type(expected_result)}: {expected_result}\n#\tReturned {type(result)}: {result}",
        )


class MiscTest(Main, Misc):
    """Misc test class."""

    def test_listify(self):
        # 1. Standalone function with threading
        @Misc.listify(threading=True)
        def to_string(n):
            return str(n)

        class TestClass:
            # 2. Method within a class with threading
            @Misc.listify(arg_name="n", threading=True)
            def to_string(self, n):
                return str(n)

        # 3. Function with arg_name specified and threading
        @Misc.listify(arg_name="n", threading=True)
        def to_string_arg(n):
            return str(n)

        # 4. Function with arg_name specified and no threading
        @Misc.listify(arg_name="n", threading=False)
        def to_string_no_thread(n):
            return str(n)

        self.assertEqual(to_string(range(4)), ["0", "1", "2", "3"])

        test_obj = TestClass()
        self.assertEqual(
            test_obj.to_string(range(4)),
            ["0", "1", "2", "3"],
        )

        self.assertEqual(to_string_arg(range(4)), ["0", "1", "2", "3"])

        self.assertEqual(
            to_string_no_thread(range(4)),
            ["0", "1", "2", "3"],
        )
