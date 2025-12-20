# !/usr/bin/python
# coding=utf-8
from typing import Union, List, Optional, Dict, Tuple, Callable

# from this package:
from pythontk.core_utils._core_utils import CoreUtils
from pythontk.iter_utils._iter_utils import IterUtils


class StrUtils(CoreUtils):
    """ """

    @staticmethod
    def sanitize(
        text: Union[str, List[str]],
        replacement_char: str = "_",
        char_map: Optional[Dict[str, str]] = None,
        preserve_trailing: bool = False,
        preserve_case: bool = False,
        allow_consecutive: bool = False,
        return_original: bool = False,  # Optionally return original string(s)
    ) -> Union[str, Tuple[str, str], List[str], List[Tuple[str, str]]]:
        """Sanitizes a string or a list of strings by replacing invalid characters.

        Returns:
            (obj/list) dependant on flags.
        """
        import re

        def sanitize_single(text: str) -> Union[str, Tuple[str, str]]:
            original_text = text
            txt = text if preserve_case else text.lower()

            # Apply character mappings if provided
            if char_map:
                for char, replacement in char_map.items():
                    txt = txt.replace(char, replacement)

            # Replace all non-alphanumeric characters
            sanitized_text = re.sub(
                r"[^a-z0-9_]" if not preserve_case else r"[^A-Za-z0-9_]",
                replacement_char,
                txt,
            )

            # Collapse consecutive replacement characters if allow_consecutive is False
            if not allow_consecutive:
                sanitized_text = re.sub(
                    f"{replacement_char}+", replacement_char, sanitized_text
                )

            # Optionally remove trailing illegal characters if preserve_trailing is False
            if not preserve_trailing:
                sanitized_text = re.sub(f"{replacement_char}+$", "", sanitized_text)

            return (
                (sanitized_text, original_text) if return_original else sanitized_text
            )

        # Ensure the input is always iterable using the make_iterable method
        iterable_text = IterUtils.make_iterable(text)

        # Sanitize each item in the iterable
        sanitized_list = [sanitize_single(t) for t in iterable_text]

        # Return the appropriate format using format_return
        return CoreUtils.format_return(sanitized_list, orig=text)

    @staticmethod
    def replace_placeholders(text: str, **kwargs) -> str:
        """Replace placeholders in a string with provided values.

        Supports standard Python string formatting syntax (e.g. {value:03d}).
        Missing keys are preserved as placeholders.

        Args:
            text (str): The string containing placeholders.
            **kwargs: Key-value pairs corresponding to placeholders.

        Returns:
            str: The string with placeholders replaced.

        Example:
            >>> StrUtils.replace_placeholders("File: {name}_{ver:03d}.{ext}", name="shot", ver=5, ext="ma")
            'File: shot_005.ma'
            >>> StrUtils.replace_placeholders("Path: {root}/{missing}", root="C:/Projects")
            'Path: C:/Projects/{missing}'
        """
        import string

        class SafeFormatter(string.Formatter):
            def get_value(self, key, args, kwargs):
                if isinstance(key, str):
                    return kwargs.get(key, "{" + key + "}")
                return super().get_value(key, args, kwargs)

            def format_field(self, value, format_spec):
                if (
                    isinstance(value, str)
                    and value.startswith("{")
                    and value.endswith("}")
                ):
                    return value
                return super().format_field(value, format_spec)

        return SafeFormatter().format(text, **kwargs)

    @staticmethod
    @CoreUtils.listify(threading=True)
    def set_case(string, case="title"):
        """Format the given string(s) in the given case.

        Parameters:
            string (str/list): The string(s) to format.
            case (str): The desired return case. Accepts all python case operators.
                    valid: 'upper', 'lower', 'capitalize' (default), 'swapcase', 'title', 'pascal', 'camel', None.
        Returns:
            (str/list) List if 'string' given as list.
        """
        if (not string) or (not isinstance(string, str)):
            return ""

        if case == "pascal":
            return string[:1].capitalize() + string[1:]  # capitalize the first letter.

        elif case == "camel":
            return string[0].lower() + string[1:]  # lowercase the first letter.

        else:
            try:
                return getattr(string, case)()

            except AttributeError:  # return the original string.
                return string

    @staticmethod
    def get_mangled_name(class_input, attribute_name):
        """Returns the mangled name for a private attribute of a class.

        Parameters:
            class_input (str/type/instance): The class name as a string, the class itself or an instance of the class.
            attribute_name (str): The original name of the attribute.

        Returns:
            str: The mangled name of the attribute.

        Raises:
            TypeError: If class_input is not a string, a type, or an instance of a class, or if attribute_name is not a string.
            ValueError: If attribute_name does not start with double underscore.

        Example:
            get_mangled_name("MyClass", "__attribute") -> "_MyClass__attribute"
            get_mangled_name(MyClass, "__attribute") -> "_MyClass__attribute"
            get_mangled_name(MyClass(), "__attribute") -> "_MyClass__attribute"
        """
        if not isinstance(attribute_name, str):
            raise TypeError("attribute_name must be a string")
        if not attribute_name.startswith("__"):
            raise ValueError("attribute_name must start with double underscore")

        if isinstance(class_input, str):
            class_name = class_input
        elif isinstance(class_input, type):
            class_name = class_input.__name__
        elif hasattr(class_input, "__class__"):
            class_name = class_input.__class__.__name__
        else:
            raise TypeError(
                "class_input must be a string, a type, or an instance of a class"
            )

        return f"_{class_name}{attribute_name}"

    @staticmethod
    def get_matching_hierarchy_items(
        hierarchy_items,
        target,
        upstream=False,
        exact=False,
        downstream=False,
        reverse=False,
        delimiters="|",
    ):
        """Find the closest match(es) for a given 'target' string in a list of hierarchical strings.

        Parameters:
            hierarchy_items (list): A list of strings representing hierarchical items.
            target (str): A string representing the hierarchical item to find a match for.
            upstream (bool, optional): If True, returns items that are one level up in the hierarchy. Default is False.
            exact (bool, optional): If True, returns only items that are an exact match. Default is False.
            downstream (bool, optional): If True, returns items that are one level down in the hierarchy. Default is False.
            reverse (bool, optional): Reverse the result. Default is False.
            delimiters (str/list, optional): A string containing all characters that can act as delimiters in the hierarchy. Default is "|".

        Returns:
            list: A list of matching items ordered by length.

        Example:
            hierarchy_items = [
                "polygons|mesh|submenu",
                "polygons|submenu",
                "polygons",
                "polygons|mesh",
                "polygons|face",
                "polygons|mesh|other",
            ]

            target = "polygons.mesh"
            get_matching_hierarchy_items(hierarchy_items, target, upstream=True) -> ['polygons']
            get_matching_hierarchy_items(hierarchy_items, target, downstream=True) -> ['polygons|mesh|submenu', 'polygons|mesh|other']
            get_matching_hierarchy_items(hierarchy_items, target, exact=True) -> ['polygons|mesh']
        """
        import re

        pattern = "|".join(re.escape(d) for d in IterUtils.make_iterable(delimiters))
        target_parts = re.split(pattern, target)

        def match_hierarchy(item_parts):
            return all(p1 == p2 for p1, p2 in zip(item_parts, target_parts))

        def is_upstream(item_parts):
            return len(item_parts) < len(target_parts)

        def is_downstream(item_parts):
            return len(item_parts) > len(target_parts)

        def filter_items(item):
            item_parts = re.split("|".join(re.escape(d) for d in delimiters), item)

            if exact and item == target:
                return True
            if (
                upstream
                and match_hierarchy(item_parts)
                and is_upstream(item_parts)
                and set(item_parts).issubset(set(target_parts))
            ):
                return True
            if downstream and match_hierarchy(item_parts) and is_downstream(item_parts):
                return True
            return False

        matches = [item for item in hierarchy_items if filter_items(item)]
        return sorted(matches, key=lambda x: len(x), reverse=reverse)

    @staticmethod
    @CoreUtils.listify(threading=True)
    def split_delimited_string(
        string: str,
        delimiter: str = "|",
        max_split: Optional[int] = None,
        occurrence: Optional[int] = None,
        strip_whitespace: bool = False,
        remove_empty: bool = False,
        func: Optional[Callable] = None,
    ) -> Union[List[str], Tuple[str, str]]:
        """Split a delimited string with flexible control over the result format.

        This unified method handles both simple multi-way splitting and binary splitting
        at specific occurrences, with optional preprocessing and post-processing.

        Parameters:
            string (str): The string to split.
            delimiter (str): The delimiter to split on. Default is '|'.
            max_split (int, optional): Maximum number of splits to perform. If None, splits at all delimiters.
            occurrence (int, optional): If specified, returns a 2-tuple split at this specific occurrence.
                - Positive: split at Nth occurrence from left (0-indexed)
                - Negative: split at Nth occurrence from right (-1 = last)
                - If occurrence is specified, returns tuple instead of list.
            strip_whitespace (bool): If True, strip leading/trailing whitespace from each part.
                Default is False.
            remove_empty (bool): If True, remove empty strings from the result after splitting
                and stripping. Default is False.
            func (callable, optional): Function to apply to the result list (not applied to tuples).
                Should take a list and return a transformed list.
                Examples: sorted, reversed, lambda x: [s.upper() for s in x]

        Returns:
            Union[list, tuple]:
                - If occurrence is specified: 2-tuple (left, right)
                - Otherwise: List of string parts

        Example:
            # Multi-way splitting (list output)
            >>> split_delimited_string('a|b|c|d')
            ['a', 'b', 'c', 'd']

            >>> split_delimited_string('  a  | b |  c  ', strip_whitespace=True)
            ['a', 'b', 'c']

            >>> split_delimited_string('a||b||c', remove_empty=True)
            ['a', 'b', 'c']

            >>> split_delimited_string('c|a|b', func=sorted)
            ['a', 'b', 'c']

            >>> split_delimited_string('a|b|c', func=reversed)
            ['c', 'b', 'a']

            >>> split_delimited_string('apple|banana|cherry', func=lambda x: [s.upper() for s in x])
            ['APPLE', 'BANANA', 'CHERRY']

            # Binary splitting at specific occurrence (tuple output)
            >>> split_delimited_string('a|b|c|d', occurrence=-1)
            ('a|b|c', 'd')

            >>> split_delimited_string('a|b|c|d', occurrence=0)
            ('', 'a')

            >>> split_delimited_string('a|b|c|d', occurrence=1)
            ('a', 'b')

            >>> split_delimited_string('string', occurrence=-1)  # No delimiter found
            ('string', '')

            # Max split limiting
            >>> split_delimited_string('a|b|c|d', max_split=2)
            ['a', 'b', 'c|d']
        """
        if not string:
            return ("", "") if occurrence is not None else []

        # Handle binary splitting at specific occurrence (returns tuple)
        if occurrence is not None:
            if delimiter not in string:
                return (string, "")

            parts = string.split(delimiter)

            try:
                # Get the part at the specified occurrence
                right = parts[occurrence]
                # Reconstruct the left part (everything before the occurrence)
                left = delimiter.join(parts[:occurrence])
                return (left, right)
            except IndexError:
                return (string, "")

        # Handle multi-way splitting (returns list)
        if max_split is not None:
            parts = string.split(delimiter, max_split)
        else:
            parts = string.split(delimiter)

        # Strip whitespace if requested
        if strip_whitespace:
            parts = [part.strip() for part in parts]

        # Remove empty strings if requested
        if remove_empty:
            parts = [part for part in parts if part]

        # Apply function if specified
        if func and callable(func):
            parts = func(parts)

        return parts

    @staticmethod
    def get_text_between_delimiters(string, start_delim, end_delim, as_string=False):
        """Get any text between the specified start and end delimiters in the given string. The text can be returned as a
        generator (default behavior) or as a single concatenated string if `as_string` is set to True.

        Parameters:
            string (str): The input string to search for matches.
            start_delim (str): The starting delimiter to search for.
            end_delim (str): The ending delimiter to search for.
            as_string (bool, optional): If True, the function returns a single concatenated string of all matches.
                                                                     If False (default), the function returns a generator that yields each match.

        Returns:
            If as_string is False (default): A generator that yields all matches found in the input string.
            If as_string is True: A single concatenated string containing all matches found in the input string.

        Example:
            input_string = "Here is the <!-- start -->first match<!-- end --> and here is the <!-- start -->second match<!-- end -->"

            # Get the matches as a generator (default behavior)
            matches_generator = get_text_between_delimiters(input_string, '<!-- start -->', '<!-- end -->')
            for match in matches_generator:
                    print(match)  # Output: first match (first iteration), second match (second iteration)

            # Get the matches as a single string
            matches_string = get_text_between_delimiters(input_string, '<!-- start -->', '<!-- end -->', as_string=True)
            print(matches_string)  # Output: "first match second match"
        """
        import re

        def extract_matches(string, start_delim, end_delim, start_index=0):
            pattern = re.compile(
                f"{re.escape(start_delim)}(.*?){re.escape(end_delim)}", re.DOTALL
            )
            match = pattern.search(string, start_index)
            if match:
                yield match.group(1).strip()
                yield from extract_matches(string, start_delim, end_delim, match.end())

        if as_string:
            matches = list(extract_matches(string, start_delim, end_delim))
            return " ".join(matches)
        else:
            return extract_matches(string, start_delim, end_delim)

    @classmethod
    def insert(cls, src, ins, at, occurrence=1, before=False):
        """Insert character(s) into a string at a given location.
        if the character doesn't exist, the original string will be returned.

        Parameters:
            src (str): The source string.
            ins (str): The character(s) to insert.
            at (str)(int): The index or char(s) to insert at.
            occurrence (int): Specify which occurrence to insert at.
                        Valid only when 'at' is given as a string.
                        default: The first occurrence.
                        (A value of -1 would insert at the last occurrence)
            before (bool): Specify inserting before or after. default: after
                        Valid only when 'at' is given as a string.
        Returns:
            (str)
        """
        try:
            return "".join((src[:at], str(ins), src[at:]))

        except TypeError:
            # if 'occurrance' is a negative value, search from the right.
            if occurrence < 0:
                i = src.replace(at, " " * len(at), occurrence - 1).rfind(at)
            else:
                i = src.replace(at, " " * len(at), occurrence - 1).find(at)
            return (
                cls.insert(src, str(ins), i if before else i + len(at))
                if i != -1
                else src
            )

    @staticmethod
    def rreplace(string, old, new="", count=None):
        """Replace occurrances in a string from right to left.
        The number of occurrances replaced can be limited by using the 'count' argument.

        Parameters:
            string (str):
            old (str):
            new (str)(int):
            count (int):

        Returns:
            (str)
        """
        if not string or not isinstance(string, str):
            return string

        if count is not None:
            return str(new).join(string.rsplit(old, count))
        else:
            return str(new).join(string.rsplit(old))

    @staticmethod
    @CoreUtils.listify(threading=True)
    def truncate(string, length=75, mode="start", insert=".."):
        """Shorten the given string to the given length.
        An ellipsis will be added to the section trimmed.

        Parameters:
            string (str): The string to truncate.
            length (int): The maximum allowed length before truncating.
            mode (str): Truncation mode.
                - 'start'/'left': Trim from start (keep end) - default
                - 'end'/'right': Trim from end (keep start)
                - 'middle': Trim from middle (keep start and end)
            insert (str): Characters to add at the trimmed area. (default: ellipsis)

        Returns:
            (str)

        Examples:
            truncate('12345678', 4) #returns: '..5678' (start mode)
            truncate('12345678', 4, 'end') #returns: '1234..' (end mode)
            truncate('12345678', 6, 'middle') #returns: '12..78' (middle mode)
        """
        if not string or not isinstance(string, str):
            return string

        # Normalize mode to lowercase
        mode = mode.lower() if isinstance(mode, str) else "start"

        if len(string) <= length:
            return string

        # Safety nets
        if length <= 0:
            return insert
        if length < len(insert) + 1:
            return insert + string[-1:]

        if mode in ("start", "left"):
            # Keep the last 'length' chars
            return insert + string[-length:]
        elif mode in ("end", "right"):
            # Keep the first 'length' chars
            return string[:length] + insert
        elif mode == "middle":
            # Split around the middle; allocate space for both sides
            # Visible chars excluding insert
            vis = max(1, length)
            # If original code expected length as visible total excluding insert, we mimic prior behavior (it included insert area at trim point)
            # We'll treat 'length' as the number of chars we preserve on each side total minus insert length.
            avail = max(1, length - len(insert))
            if avail <= 1:
                return string[0] + insert
            left = avail // 2
            right = avail - left
            return string[:left] + insert + string[-right:]
        else:
            # Fallback to start trimming (default behavior)
            return insert + string[-length:]

    @staticmethod
    def get_trailing_integers(string, inc=0, as_string=False):
        """Returns any integers from the end of the given string.

        Parameters:
            inc (int): Increment by a step amount. (default: 0)
                    0 does not increment and returns the original number.
            as_string (bool): Return the integers as a string instead of integers.

        Returns:
            (int)

        Example:
            get_trailing_integers('p001Cube1', inc=1) #returns: 2
        """
        import re

        if not string or not isinstance(string, str):
            return string

        m = re.findall(r"\d+\s*$", string)
        result = int(m[0]) + inc if m else None

        if as_string:
            return str(result)
        return result

    @staticmethod
    def find_str(find, strings, regex=False, ignore_case=False):
        """Filter for elements that containing the given string in a list of strings.

        Parameters:
            find (str): The search string. An asterisk denotes startswith*, *endswith, *contains*, and multiple search strings can be separated by pipe chars.
                    wildcards:
                        *chars* - string contains chars.
                        *chars - string endswith chars.
                        chars* - string startswith chars.
                        chars1|chars2 - string matches any of.  can be used in conjuction with other modifiers.
                    regular expressions (if regex True):
                        (.) match any char. ex. re.match('1..', '1111') #returns the regex object <111>
                        (^) match start. ex. re.match('^11', '011') #returns None
                        ($) match end. ex. re.match('11$', '011') #returns the regex object <11>
                        (|) or. ex. re.match('1|0', '011') #returns the regex object <0>
                        (\\A,\\Z) beginning of a string and end of a string. ex. re.match(r'\\A011\\Z', '011') #
                        (\\b) empty string. (\\B matches the empty string anywhere else). ex. re.match(r'\\b(011)\\b', '011 011 011') #
            strings (list): The string list to search.
            regex (bool): Use regular expressions instead of wildcards.
            ignore_case (bool): Search case insensitive.

        Returns:
            (list)

        Example:
            lst = ['invertVertexWeights', 'keepCreaseEdgeWeight', 'keepBorder', 'keepBorderWeight', 'keepColorBorder', 'keepColorBorderWeight']
            find_str('*Weight*', lst) #find any element that contains the string 'Weight'.
            find_str('Weight$|Weights$', lst, regex=True) #find any element that endswith 'Weight' or 'Weights'.
        """
        import re

        # Filter out non-string values
        strings = [s for s in strings if isinstance(s, str)]

        if not find:  # Handle empty search string
            return []

        if not strings:  # Early exit for empty list
            return []

        if regex:
            try:
                flags = re.IGNORECASE if ignore_case else 0
                pattern = re.compile(find, flags)
                return [s for s in strings if pattern.search(s)]
            except re.error as e:
                print(f"# Error find_str: in {find}: {e}. #")
                return []

        # Pre-process: parse all search terms once
        find_parts = find.split("|")
        search_terms = []
        for w in find_parts:
            term = w.strip("*")
            starts = w.startswith("*")
            ends = w.endswith("*")

            if starts and ends:
                mode = "contains"
            elif starts:
                mode = "endswith"
            elif ends:
                mode = "startswith"
            else:
                mode = "exact"

            # Pre-lowercase term if case-insensitive
            search_terms.append((term.lower() if ignore_case else term, mode))

        # Use set for O(1) duplicate checking
        seen = set()
        result = []

        for s in strings:
            if s in seen:
                continue

            check = s.lower() if ignore_case else s

            for term, mode in search_terms:
                matched = False
                if mode == "contains":
                    matched = term in check
                elif mode == "endswith":
                    matched = check.endswith(term)
                elif mode == "startswith":
                    matched = check.startswith(term)
                else:  # exact
                    matched = check == term

                if matched:
                    seen.add(s)
                    result.append(s)
                    break  # Don't check other terms for this string

        return result

    @classmethod
    def find_str_and_format(
        cls,
        strings,
        to,
        fltr="",
        regex=False,
        ignore_case=False,
        return_orig_strings=False,
    ):
        """Expanding on the 'find_str' function: Find matches of a string in a list of strings and re-format them.

        Parameters:
            strings (list): A list of string objects to search.
            to (str): An optional asterisk modifier can be used for formatting. An empty string will attempt to remove the part of the string designated in the from argument.
                    "" - (empty string) - strip chars.
                    *chars* - replace only.
                    *chars - replace suffix.
                    **chars - append suffix.
                    chars* - replace prefix.
                    chars** - append prefix.
            fltr (str): See the 'find_str' function's 'fltr' parameter for documentation.
            regex (bool): Use regular expressions instead of wildcards for the 'find' argument.
            ignore_case (bool): Ignore case when searching. Applies only to the 'fltr' parameter's search.
            return_orig_strings (bool): Return the old names as well as the new.

        Returns:
            (list) if return_orig_strings: list of two element tuples containing the original and modified string pairs. [('frm','to')]
                    else: a list of just the new names.
        """
        import re

        # Filter out non-string values
        strings = [s for s in strings if isinstance(s, str)]

        # If 'fltr' is not empty, filter 'strings' for matches
        if fltr:
            strings = cls.find_str(fltr, strings, regex=regex, ignore_case=ignore_case)

        if not strings:  # Early exit
            return []

        frm_ = fltr.strip("*")
        to_ = to.strip("*")

        # Pre-compile regex pattern if needed for case-insensitive operations
        frm_pattern = None
        if frm_ and ignore_case and not regex:
            try:
                frm_pattern = re.compile(re.escape(frm_), re.IGNORECASE)
            except re.error:
                frm_pattern = None

        # Determine the formatting mode once
        if to.startswith("*") and to.endswith("*") and len(to) > 1:
            mode = "replace_chars"
        elif to.startswith("**"):
            mode = "append_suffix"
        elif to.startswith("*"):
            mode = "replace_suffix"
        elif to.endswith("**"):
            mode = "append_prefix"
        elif to.endswith("*"):
            mode = "replace_prefix"
        elif not to_:
            mode = "strip"
        else:
            mode = "replace_whole"

        # Determine strip sub-mode if applicable
        strip_mode = None
        if mode == "strip" and frm_:
            if fltr.endswith("*") and not fltr.startswith("*"):
                strip_mode = "first"
            elif fltr.startswith("*") and not fltr.endswith("*"):
                strip_mode = "last"
            else:
                strip_mode = "all"

        result = []
        for orig_str in strings:
            s = orig_str  # Default: no change

            if mode == "replace_chars":
                if frm_:
                    if frm_pattern:
                        s = frm_pattern.sub(to_, orig_str)
                    else:
                        s = orig_str.replace(frm_, to_)

            elif mode == "append_suffix":
                s = orig_str + to_

            elif mode == "replace_suffix":
                if frm_:
                    if frm_pattern:
                        match = frm_pattern.search(orig_str)
                        if match:
                            s = orig_str[: match.start()] + to_
                        else:
                            s = orig_str + to_
                    else:
                        parts = orig_str.split(frm_, 1)
                        s = parts[0] + to_ if len(parts) > 1 else orig_str + to_
                else:
                    s = orig_str + to_

            elif mode == "append_prefix":
                s = to_ + orig_str

            elif mode == "replace_prefix":
                if frm_:
                    if frm_pattern:
                        match = frm_pattern.search(orig_str)
                        if match:
                            s = to_ + orig_str[match.end() :]
                        else:
                            s = to_ + orig_str
                    else:
                        parts = orig_str.split(frm_, 1)
                        if len(parts) > 1:
                            s = to_ + frm_ + frm_.join(parts[1:])
                        else:
                            s = to_ + orig_str
                else:
                    s = to_ + orig_str

            elif mode == "strip":
                if frm_:
                    if strip_mode == "first":
                        if frm_pattern:
                            s = frm_pattern.sub("", orig_str, count=1)
                        else:
                            s = orig_str.replace(frm_, "", 1)
                    elif strip_mode == "last":
                        if frm_pattern:
                            # Remove last occurrence
                            matches = list(frm_pattern.finditer(orig_str))
                            if matches:
                                last = matches[-1]
                                s = orig_str[: last.start()] + orig_str[last.end() :]
                        else:
                            s = "".join(orig_str.rsplit(frm_, 1))
                    else:  # all
                        if frm_pattern:
                            s = frm_pattern.sub("", orig_str)
                        else:
                            s = orig_str.replace(frm_, "")

            elif mode == "replace_whole":
                s = to_

            if return_orig_strings:
                result.append((orig_str, s))
            else:
                result.append(s)

        return result

    @staticmethod
    def format_suffix(
        string: str,
        suffix: str = "",
        strip: Union[str, List[str]] = "",
        strip_trailing_ints: bool = False,
        strip_trailing_alpha: bool = False,
    ) -> str:
        """Re-format the suffix for the given string.

        Parameters:
            string (str): The string to format.
            suffix (str): Append a new suffix to the given string.
            strip (str/list): Specific string(s) or regex pattern(s) to strip from the end of the given string.
            strip_trailing_ints (bool): Strip all trailing integers.
            strip_trailing_alpha (bool): Strip all upper-case letters preceded by a non-alphanumeric character.

        Returns:
            (str): The formatted string.
        """
        import re

        def is_regex(pattern: str) -> bool:
            try:
                re.compile(pattern)
                return True
            except re.error:
                return False

        # Always operate on the last pipe segment (Maya naming)
        s = string.split("|")[-1]

        if strip:
            strip_items = IterUtils.make_iterable(strip)
            for pattern in strip_items:
                if isinstance(pattern, str) and is_regex(pattern) and len(pattern) > 1:
                    # Only treat as regex if it is a pattern (not a simple suffix string)
                    s = re.sub(pattern, "", s)
                else:
                    # Standard: strip all occurrences of this suffix from the end
                    while s.endswith(pattern):
                        s = s[: -len(pattern)]

        # Strip trailing ints or uppercase alphas if requested
        while True:
            stripped = False
            if strip_trailing_ints and s and s[-1].isdigit():
                s = re.sub(r"\d+$", "", s)
                stripped = True
            if strip_trailing_alpha and s and s[-1].isupper():
                s = re.sub(r"(?:[^0-9A-Za-z]+)?[A-Z]+$", "", s)
                stripped = True
            if not stripped:
                break

        return s + suffix

    @staticmethod
    @CoreUtils.listify(threading=True)
    def time_stamp(filepath, stamp="%m-%d-%Y  %H:%M"):
        """Attach or detach a modified timestamp and date to/from a given file path.

        Parameters:
            filepath (str): The full path to a file. ie. 'C:/Windows/Temp/__AUTO-SAVE__untitled.0001.mb'
            stamp (str): The time stamp format.

        Returns:
            str: Filepath with attached or detached timestamp, depending on whether it initially had a timestamp.
            ie. '16:46  11-09-2021  C:/Windows/Temp/__AUTO-SAVE__untitled.0001.mb' from 'C:/Windows/Temp/__AUTO-SAVE__untitled.0001.mb'
        """
        from datetime import datetime
        import os.path
        import re
        from pythontk.file_utils._file_utils import FileUtils

        filepath = FileUtils.format_path(filepath)

        # Check if the file path has a timestamp using regular expression
        match = re.match(r"\d{2}:\d{2}  \d{2}-\d{2}-\d{4}", filepath)
        if match:
            # If it does, return the file path without the timestamp
            return "".join(filepath.split()[2:])
        else:
            # If it doesn't, attach a timestamp
            try:
                return "{}  {}".format(
                    datetime.fromtimestamp(os.path.getmtime(filepath)).strftime(stamp),
                    filepath,
                )
            except (FileNotFoundError, OSError) as error:
                print(f"Error: {error}")
                return filepath


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------


# deprecated ---------------------
