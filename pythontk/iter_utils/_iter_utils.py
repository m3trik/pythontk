# !/usr/bin/python
# coding=utf-8
from typing import Any, Callable, Iterable, List, Dict, Optional, Union

# from this package:
from pythontk import core_utils


class IterUtils(core_utils.HelpMixin):
    """ """

    @staticmethod
    def make_iterable(x: Any, snapshot: bool = False) -> Iterable:
        """Convert the given object to an iterable, unless it's a string, bytes, or bytearray.

        Parameters:
            x (Any): The object to convert to an iterable.
            snapshot (bool): If True, return a list snapshot to avoid modification during iteration.

        Returns:
            Iterable: An iterable or snapshot of the input.
        """
        from collections.abc import Iterable as ABCIterable

        if x is None:
            return ()

        if hasattr(x, "__apimfn__") or isinstance(x, (str, bytes, bytearray)):
            return (x,)

        if isinstance(x, (map, filter, zip)):
            return list(x)

        if isinstance(x, ABCIterable):
            return list(x) if snapshot else x

        return (x,)

    @classmethod
    def nested_depth(cls, lst, typ=(list, set, tuple)):
        """Get the maximum nested depth of any sub-lists of the given list.
        If there is nothing nested, 0 will be returned.

        Parameters:
            lst (list): The list to check.
            typ (type)(tuple): The type(s) to include in the query.

        Returns:
            (int) 0 if none, else the max nested depth.
        """
        d = -1
        for i in lst:
            if isinstance(i, typ):
                d = max(cls.nested_depth(i), d)
        return d + 1

    @classmethod
    def flatten(cls, lst):
        """Flatten arbitrarily nested lists.

        Parameters:
                lst (list): A list with potentially nested lists.

        Returns:
                (generator)
        """
        for i in lst:
            if isinstance(i, (list, tuple, set)):
                for ii in cls.flatten(i):
                    yield ii
            else:
                yield i

    @staticmethod
    def collapse_integer_sequence(lst, limit=None, compress=True, to_string=True):
        """Converts a list of integers into a compressed string representation of sequences.

        This function transforms a list of integers into a compressed representation where consecutive sequences are
        represented as 'start..end'. For example, [19, 22, 23, 24, 25, 26] is transformed to ['19', '22..26'].

        Parameters:
            lst (List[int]): A list of integers to compress.
            limit (Optional[int]): If set, limits the maximum length of the returned elements. If the list exceeds this length, it is truncated and '...' is appended.
            compress (bool): If True, trims redundant characters from the second half of a compressed range. E.g., ['19', '22-32', '1225-6'] instead of ['19', '22..32', '1225..1226']. Defaults to True.
            to_string (bool): If True, joins the list elements into a single string separated by ', '. Defaults to True.

        Returns:
            List[str] or str: The compressed representation of the input list as a list of strings or a single string.
        """
        ranges = []
        prev_x = None
        for x in map(str, lst):  # make sure the list is made up of strings.
            if not ranges:
                ranges.append([x])
            elif int(x) - prev_x == 1:
                ranges[-1].append(x)
            else:
                ranges.append([x])
            prev_x = int(x)

        if compress:  # style: ['19', '22-32', '1225-6']
            collapsed = [
                "-".join(
                    [
                        r[0],
                        r[-1][len(str(r[-1])) - len(str((int(r[-1]) - int(r[0])))) :],
                    ]  # find the difference and use that value to further trim redundant chars from the string
                    if len(r) > 1
                    else r
                )
                for r in ranges
            ]

        else:  # style: ['19', '22..32', '1225..1226']
            collapsed = ["..".join([r[0], r[-1]] if len(r) > 1 else r) for r in ranges]

        if limit and len(collapsed) > limit:
            limited_collapsed = collapsed[:limit]
            limited_collapsed.append("...")
            collapsed = limited_collapsed

        if to_string:
            collapsed = ", ".join(collapsed)

        return collapsed

    @staticmethod
    def bit_array_to_list(bit_array):
        """Convert a binary bit_array to a python list.

        Parameters:
            bit_array () = A bit array or list of bit arrays.

        Returns:
            (list) containing values of the indices of the on (True) bits.
        """
        if len(bit_array):
            if type(bit_array[0]) != bool:  # if list of bitArrays: flatten
                lst = []
                for array in bit_array:
                    lst.append([i + 1 for i, bit in enumerate(array) if bit == 1])
                return [bit for array in lst for bit in array]

            return [i + 1 for i, bit in enumerate(bit_array) if bit == 1]

    @staticmethod
    def insert_into_dict(
        original_dict: Dict[Any, Any],
        key: Any,
        value: Any,
        index: int = 0,
        in_place: bool = False,
    ) -> Dict[Any, Any]:
        """Insert a key-value pair at a specified index in a dictionary.

        This function inserts a key-value pair at the specified index in the dictionary.
        It can either create a new dictionary or modify the original one in place based
        on the `in_place` parameter.

        Parameters:
            original_dict (Dict[Any, Any]): The original dictionary to modify or copy.
            key (Any): The key of the new key-value pair to insert.
            value (Any): The value of the new key-value pair to insert.
            index (int, optional): The index at which to insert the key-value pair.
                Defaults to 0.
            in_place (bool, optional): If True, the original dictionary is modified in place.
                If False, a new dictionary is created. Defaults to False.

        Returns:
            Dict[Any, Any]: The modified dictionary with the new key-value pair inserted.

        Raises:
            IndexError: If the specified index is out of range.

        Example:
            >>> original_dict = {'b': 2, 'c': 3}
            >>> new_dict = insert_into_dict_at_index(original_dict, 'a', 1)
            >>> print(new_dict)
            {'a': 1, 'b': 2, 'c': 3}
            >>> insert_into_dict_at_index(original_dict, 'd', 4, 2, in_place=True)
            >>> print(original_dict)
            {'b': 2, 'c': 3, 'd': 4}
        """
        if not (0 <= index <= len(original_dict)):
            raise IndexError("Index out of range")

        items = list(original_dict.items())
        items.insert(index, (key, value))

        if in_place:
            original_dict.clear()
            original_dict.update(items)
            return original_dict
        else:
            return dict(items)

    @staticmethod
    def rindex(itr, item):
        """Get the index of the first item to match the given item
        starting from the back (right side) of the list.

        Parameters:
            itr (iter): An iterable.
            item () = The item to get the index of.

        Returns:
            (int) -1 if element not found.
        """
        return next(iter(i for i in range(len(itr) - 1, -1, -1) if itr[i] == item), -1)

    @staticmethod
    def indices(itr, value):
        """Get the index of each element of a list matching the given value.

        Parameters:
            itr (iter): An iterable.
            value () = The search value.

        Returns:
            (generator)
        """
        return (i for i, v in enumerate(itr) if v == value)

    @staticmethod
    def remove_duplicates(lst, trailing=True):
        """Removes duplicate entries from a list while maintaining the original order of the remaining items.
        Allows for the choice between preserving the first or the last occurrences of each item.

        Parameters:
            lst (list): The list from which duplicate elements are to be removed.
            trailing (bool): Specifies the strategy for which occurrences to keep:
                             - True (default): Keeps the first occurrence of each item and removes subsequent duplicates.
                             - False: Keeps the last occurrence of each item by removing earlier duplicates.
        Returns:
            list: A new list with duplicates removed, preserving either the first or last occurrences of each item based on the `trailing` parameter.

        Example:
            >>> remove_duplicates([1, 2, 2, 3, 4, 2, 1, 5])
            [1, 2, 3, 4, 5]
            >>> remove_duplicates([1, 2, 2, 3, 4, 2, 1, 5], trailing=False)
            [3, 4, 2, 1, 5]
        """
        try:
            if trailing:
                return list(dict.fromkeys(lst))
            else:
                return list(dict.fromkeys(lst[::-1]))[::-1]
        except TypeError:  # Fallback for unhashable types
            seen = []
            result = []
            # Use reversed list if trailing is False to preserve the last occurrence
            items = reversed(lst) if not trailing else lst

            for item in items:
                if item not in seen:
                    seen.append(item)
                    result.append(item)

            # Reverse the result list if we were preserving the last occurrences
            if not trailing:
                result.reverse()

            return result

    def filter_results(func: Callable) -> Callable:
        """Decorator to filter the results of a function that returns a list or dictionary.

        This decorator can be applied to functions that return a list or a dictionary. It will
        filter the results based on inclusion and exclusion criteria using `filter_list`
        for lists and `filter_dict` for dictionaries.

        Returns:
            Callable: The decorated function with filtering applied to its result.
        """
        import functools
        import inspect

        @functools.wraps(func)
        def wrapper_filter_results(*args, **kwargs) -> Any:
            # Extract filtering arguments
            filter_keys_list = inspect.signature(IterUtils.filter_list).parameters
            filter_keys_dict = inspect.signature(IterUtils.filter_dict).parameters

            filter_args_list = {
                key: kwargs.pop(key, None) for key in filter_keys_list if key in kwargs
            }
            filter_args_dict = {
                key: kwargs.pop(key, None) for key in filter_keys_dict if key in kwargs
            }

            # Call the original function with remaining kwargs
            result = func(*args, **kwargs)

            # Determine the type of the result and apply appropriate filtering
            if isinstance(result, list):
                filtered_result = IterUtils.filter_list(result, **filter_args_list)
            elif isinstance(result, dict):
                filtered_result = IterUtils.filter_dict(result, **filter_args_dict)
            else:
                raise TypeError(
                    f"Unsupported result type: {type(result)}. Only list and dict are supported."
                )

            return filtered_result

        return wrapper_filter_results

    @classmethod
    def filter_list(
        cls,
        lst: List,
        inc: Optional[Union[str, List]] = None,
        exc: Optional[Union[str, List]] = None,
        map_func: Optional[Callable] = None,
        check_unmapped: bool = False,
        nested_as_unit: bool = False,
        basename_only: bool = False,
        ignore_case: bool = False,
    ) -> List:
        """Filters the given list based on inclusion/exclusion criteria using shell-style wildcards. This method can also apply
        the filter to nested structures like lists, tuples, or sets. If 'nested_as_unit' is True, then the entire structure is
        considered as a single unit for inclusion or exclusion based on whether any of its elements match the criteria.

        Parameters:
            lst (list): The list to filter.
            inc (str/list, optional): The pattern(s) or object(s) to include.
                Each item can be a string or integer. Strings can include shell-style wildcards:
                    - '*': Matches any sequence of characters (including none).
                    - '?': Matches any single character.
                    - '[seq]': Matches any character in 'seq'.
                    - '[!seq]': Matches any character not in 'seq'.
                If provided, only items that match any pattern or object in this list are included in the result.
            exc (str/list, optional): The pattern(s) or object(s) to exclude.
                Each item can be a string or integer. Supports the same wildcards as 'inc'.
                If an item matches any pattern or object in this list, it is excluded from the result.
            map_func (callable, optional): The function to apply on each item in the list before matching.
                This function should accept a single argument and return a new value. If the function raises an exception,
                it is ignored, and the original item is used instead for matching.
            check_unmapped (bool, optional): Whether to perform matching checks on the original items if `map_func` is used.
                If True and `map_func` is provided, the function will check both the transformed (mapped) and the original (unmapped)
                items against the inclusion and exclusion criteria. If either the transformed or the original item matches the criteria,
                the original item is included or excluded in the result accordingly. Defaults to False.
            nested_as_unit (bool, optional): Whether to consider the entire nested structure as a single entity for filtering.
                If True, the entire nested structure will be included or excluded if any of its elements match the inclusion or exclusion criteria.
            basename_only (bool, optional): Use only the base name of the file paths when filtering.
            ignore_case (bool, optional): If True, the matching will be case-insensitive.
                This applies to string patterns in both `inc` and `exc` lists.

        Returns:
            list: The filtered list.
        """
        from fnmatch import fnmatchcase
        import os

        inc = list(cls.make_iterable(inc))
        exc = list(cls.make_iterable(exc))

        def match_item(item: Union[str, int], patterns: List[Union[str, int]]) -> bool:
            for pattern in patterns:
                check_item = os.path.basename(str(item)) if basename_only else str(item)

                # Safe case normalization only if both are strings
                if (
                    ignore_case
                    and isinstance(pattern, str)
                    and isinstance(check_item, str)
                ):
                    check_item = check_item.lower()
                    pattern = pattern.lower()

                match_result = (
                    fnmatchcase(check_item, pattern)
                    if isinstance(pattern, str)
                    else item == pattern
                )

                if match_result:
                    return True
            return False

        def check_item(item):
            try:
                mapped_item = map_func(item) if map_func is not None else item
            except Exception:
                mapped_item = item

            if match_item(mapped_item, exc):
                return False

            if not inc or match_item(mapped_item, inc):
                return True

            if check_unmapped and (
                (not inc or match_item(item, inc)) and not match_item(item, exc)
            ):
                return True

            return False

        result = []
        for original_item in lst:
            if isinstance(original_item, (list, tuple, set)):
                if nested_as_unit:
                    match_inc = match_exc = False
                    for i in original_item:
                        if check_item(i):
                            match_inc = True
                        if match_item(i, exc):
                            match_exc = True
                    if match_inc and not match_exc:
                        result.append(original_item)
                else:
                    filtered_sublist = cls.filter_list(
                        original_item,
                        inc,
                        exc,
                        map_func,
                        check_unmapped,
                        nested_as_unit,
                    )
                    # Only append non-empty tuples/lists/sets
                    if filtered_sublist:
                        result.append(type(original_item)(filtered_sublist))
            else:
                if check_item(original_item):
                    result.append(original_item)

        return result

    @classmethod
    def filter_dict(
        cls, dct: Dict, keys: bool = False, values: bool = False, **kwargs
    ) -> Dict:
        """Filter the given dictionary.
        Extends `filter_list` to operate on either the given dict's keys or values.

        Parameters:
            dct (dict): The dictionary to filter.
            keys (bool): Filter the dictionary keys.
            values (bool): Filter the dictionary values.
            **kwargs: Additional arguments to pass to the filter_list method.

        Returns:
            (dict)

        Example:
            dct = {1:'1', 'two':2, 3:'three'}
            filter_dict(dct, exc='*t*', values=True) #returns: {1: '1', 'two': 2}
            filter_dict(dct, exc='t*', keys=True) #returns: {1: '1', 3: 'three'}
            filter_dict(dct, exc=1, keys=True) #returns: {'two': 2, 3: 'three'}
        """
        if keys:
            filtered_keys = cls.filter_list(list(dct.keys()), **kwargs)
            dct = {k: dct[k] for k in filtered_keys}
        if values:
            filtered_values = cls.filter_list(list(dct.values()), **kwargs)
            dct = {k: v for k, v in dct.items() if v in filtered_values}
        return dct

    @staticmethod
    def split_list(lst, into):
        """Split a list into parts.

        Parameters:
            into (str): Split the list into parts defined by the following:
                '<n>parts' - Split the list into n parts.
                        ex. 2 returns:  [[1, 2, 3, 5], [7, 8, 9]] from [1,2,3,5,7,8,9]
                '<n>parts+' - Split the list into n equal parts with any trailing remainder.
                        ex. 2 returns:  [[1, 2, 3], [5, 7, 8], [9]] from [1,2,3,5,7,8,9]
                '<n>chunks' - Split into sublists of n size.
                        ex. 2 returns: [[1,2], [3,5], [7,8], [9]] from [1,2,3,5,7,8,9]
                'contiguous' - The list will be split by contiguous numerical values.
                        ex. 'contiguous' returns: [[1,2,3], [5], [7,8,9]] from [1,2,3,5,7,8,9]
                'range' - The values of 'contiguous' will be limited to the high and low end of each range.
                        ex. 'range' returns: [[1,3], [5], [7,9]] from [1,2,3,5,7,8,9]
        Returns:
            (list)
        """
        from string import digits, ascii_letters, punctuation

        mode = into.lower().lstrip(digits)
        digit = into.strip(ascii_letters + punctuation)
        n = int(digit) if digit else None

        if n:
            if mode == "parts":
                n = len(lst) * -1 // n * -1  # ceil
            elif mode == "parts+":
                n = len(lst) // n
            return [lst[i : i + n] for i in range(0, len(lst), n)]

        elif mode == "contiguous" or mode == "range":
            from itertools import groupby
            from operator import itemgetter

            try:
                contiguous = [
                    list(map(itemgetter(1), g))
                    for k, g in groupby(enumerate(lst), lambda x: int(x[0]) - int(x[1]))
                ]
            except ValueError as error:
                print(
                    "{} in split_list\n\t# Error: {} #\n\t{}".format(
                        __file__, error, lst
                    )
                )
                return lst
            if mode == "range":
                return [[i[0], i[-1]] if len(i) > 1 else (i) for i in contiguous]
            return contiguous


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
