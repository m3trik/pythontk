# !/usr/bin/python
# coding=utf-8


class Iter:
    """ """

    @staticmethod
    def make_list(x):
        """Convert the given obj to a list, unless it's a string, bytes, or bytearray.

        Parameters:
                x () = The object to convert to a list if not already a list, set, or tuple.

        Returns:
                (list)
        """
        return (
            [x]
            if isinstance(x, (str, bytes, bytearray))
            else list(x)
            if isinstance(x, (list, tuple, set, dict, range))
            else [x]
        )

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
            l = collapsed[:limit]
            l.append("...")
            collapsed = l

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
        """Remove all duplicated occurences while keeping the either the first or last.

        Parameters:
                lst (list): The list to remove duplicate elements of.
                trailing (bool): Remove all trailing occurances while keeping the first, else keep last.

        Returns:
                (list)
        """
        if trailing:
            return list(dict.fromkeys(lst))
        else:
            return list(dict.fromkeys(lst[::-1]))[
                ::-1
            ]  # reverse the list when removing from the start of the list.

    @staticmethod
    def filter_mapped_values(lst, filter_func, conversion_func, *args, **kwargs):
        """Filters a list of items based on a filtering function and a conversion function.

        This function first applies the conversion function to each item in the list,
        then filters the converted values using the filtering function. It returns
        the original list items that correspond to the filtered converted values.

        Parameters:
            lst (list): The original list of items to filter.
            filter_func (callable): The filtering function to apply on the converted values.
                                    This function should accept a list of converted values
                                    and any additional arguments or keyword arguments.
            conversion_func (callable): The conversion function to apply on each item in the list.
                                        This function should accept an item from the list and return a converted value.
            *args: Additional positional arguments to pass to the filter_func.
            **kwargs: Additional keyword arguments to pass to the filter_func.

        Returns:
            list: A filtered list of the original items corresponding to the filtered converted values.

        Example:
            # Imagine you have a list of strings representing integers, and you want to filter
            # the list to keep only the even numbers, but you want the final result to still
            # be a list of strings.

            original_list = ["1", "2", "3", "4", "5", "6"]

            # Define a filter function to keep even numbers
            def keep_even_numbers(lst):
                return [x for x in lst if x % 2 == 0]

            # Use a lambda function as the conversion function to convert each string to an integer
            conversion_func = lambda x: int(x)

            # Use the filter_mapped_values function to perform the filtering
            filtered_list = filter_mapped_values(
                original_list, keep_even_numbers, conversion_func
            )

            print(filtered_list)  # Output: ['2', '4', '6']
        """
        item_mapping = {item: conversion_func(item) for item in lst}
        filtered_converted_values = filter_func(
            list(item_mapping.values()), *args, **kwargs
        )

        return [
            item
            for item, mapped_value in item_mapping.items()
            if mapped_value in filtered_converted_values
        ]

    @classmethod
    def filter_list(cls, lst, inc=[], exc=[]):
        """Filter the given list.

        Parameters:
                lst (list): The components(s) to filter.
                inc (str)(int)(obj/list): The objects(s) to include.
                                supports using the '*' operator: startswith*, *endswith, *contains*
                                Will include all items that satisfy ANY of the given search terms.
                                meaning: '*.png' and '*Normal*' returns all strings ending in '.png' AND all
                                strings containing 'Normal'. NOT strings satisfying both terms.
                exc (str)(int)(obj/list): The objects(s) to exclude. Similar to include.
                                exlude take precidence over include.
        Returns:
                (list)

        Example: filter_list([0, 1, 2, 3, 2], [1, 2, 3], 2) #returns: [1, 3]
        """
        exc = cls.make_list(exc)
        inc = cls.make_list(inc)

        def parse_patterns(patterns):
            """Parse patterns and return separate lists for contains, startswith, and endswith."""
            contains, startswith, endswith = [], [], []
            for pattern in patterns:
                if isinstance(pattern, str) and "*" in pattern:
                    if pattern.startswith("*"):
                        if pattern.endswith("*"):
                            contains.append(pattern[1:-1])
                        else:
                            endswith.append(pattern[1:])
                    elif pattern.endswith("*"):
                        startswith.append(pattern[:-1])
            return contains, startswith, endswith

        exc_contains, exc_startswith, exc_endswith = parse_patterns(exc)
        inc_contains, inc_startswith, inc_endswith = parse_patterns(inc)

        def check(item, contains, startswith, endswith):
            """Check if an item matches any of the patterns."""
            return (
                item.startswith(startswith),
                item.endswith(endswith),
                any(substr in item for substr in contains),
            )

        result = []
        inc_startswith, inc_endswith = tuple(inc_startswith), tuple(inc_endswith)
        exc_startswith, exc_endswith = tuple(exc_startswith), tuple(exc_endswith)

        for i in lst:
            if i in exc:
                continue

            if isinstance(i, str):
                check_result = check(i, exc_contains, exc_startswith, exc_endswith)
                if any(check_result):
                    continue

            if not inc or i in inc:
                result.append(i)
            else:
                if isinstance(i, str):
                    check_result = check(i, inc_contains, inc_startswith, inc_endswith)
                    if any(check_result):
                        result.append(i)

        return result

    @classmethod
    def filter_dict(cls, dct, inc=[], exc=[], keys=False, values=False):
        """Filter the given dictionary.
        Extends `filter_list` to operate on either the given dict's keys or values.

        Parameters:
                dct (dict): The dictionary to filter.
                inc (str/obj/list): The objects(s) to include.
                                supports using the '*' operator: startswith*, *endswith, *contains*
                                Will include all items that satisfy ANY of the given search terms.
                                meaning: '*.png' and '*Normal*' returns all strings ending in '.png' AND all
                                strings containing 'Normal'. NOT strings satisfying both terms.
                exc (str/obj/list): The objects(s) to exclude. Similar to include.
                                exlude take precidence over include.
                keys (bool): Filter the dictionary keys.
                values (bool): Filter the dictionary values.

        Returns:
                (dict)

        Example: dct = {1:'1', 'two':2, 3:'three'}
        filter_dict(dct, exc='*t*', values=True) #returns: {1: '1', 'two': 2}
        filter_dict(dct, exc='t*', keys=True) #returns: {1: '1', 3: 'three'}
        filter_dict(dct, exc=1, keys=True) #returns: {'two': 2, 3: 'three'}
        """
        if keys:
            filtered = cls.filter_list(dct.keys(), inc, exc)
            dct = {k: dct[k] for k in filtered}
        if values:
            filtered = cls.filter_list(dct.values(), inc, exc)
            dct = {k: v for k, v in dct.items() if v in filtered}
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


# Deprecated ------------------------------------
