# !/usr/bin/python
# coding=utf-8
# from this package:
from pythontk import core_utils
from pythontk import iter_utils
from pythontk import file_utils


class StrUtils:
    """ """

    @staticmethod
    @core_utils.CoreUtils.listify(threading=True)
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

        Examples:
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

        pattern = "|".join(
            re.escape(d) for d in iter_utils.IterUtils.make_iterable(delimiters)
        )
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
    @core_utils.CoreUtils.listify(threading=True)
    def split_at_chars(string, chars="|", occurrence=-1):
        """Split a string containing the given chars at the given occurrence and return
        a two element tuple containing both halves.

        Parameters:
            strings (str/list): The string(s) to operate on.
            chars (str): The chars to split at.
            occurrence (int): The occurrence of the pipe to split at from left.
                    ex. -1 would split at the last occurrence. 0 would split at the first.
                        If the occurrence is out of range, the full string will be
                        returned as: ('original string', '')
        Returns:
            (tuple)(list) two element tuple, or list of two element tuples if multiple strings given.

        Example:
            split_at_chars(['str|ing', 'string']) returns: [('str', 'ing'), ('string', '')]
        """
        split = string.split(chars)

        try:
            s2 = "".join(split[occurrence])
            if chars in string:
                s1 = chars.join(split[:occurrence])
                return (s1, s2)
            else:
                return (s2, "")
        except IndexError:
            return (string, "")

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
    @core_utils.CoreUtils.listify(threading=True)
    def truncate(string, length=75, beginning=True, insert=".."):
        """Shorten the given string to the given length.
        An ellipsis will be added to the section trimmed.

        Parameters:
            length (int): The maximum allowed length before trunicating.
            beginning (bool): Trim starting chars, else; ending.
            insert (str): Chars to add at the trimmed area. (default: ellipsis)

        Returns:
            (str)

        Example:
            truncate('12345678', 4) #returns: '..5678'
        """
        if not string or not isinstance(string, str):
            return string

        if len(string) > length:
            if beginning:  # trim starting chars.
                string = insert + string[-length:]
            else:  # trim ending chars.
                string = string[:length] + insert
        return string

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
                        (\A,\Z) beginning of a string and end of a string. ex. re.match(r'\A011\Z', '011') #
                        (\b) empty string. (\B matches the empty string anywhere else). ex. re.match(r'\b(011)\b', '011 011 011') #
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
        if regex:  # search using a regular expression.
            import re

            try:
                if ignore_case:
                    result = [i for i in strings if re.search(find, i, re.IGNORECASE)]
                else:
                    result = [i for i in strings if re.search(find, i)]
            except Exception as e:
                print("# Error find_str: in {}: {}. #".format(find, e))
                result = []

        else:  # search using wildcards.
            result = []
            for w in find.split("|"):  # split at pipe chars.
                w_ = w.strip("*").rstrip(
                    "*"
                )  # remove any modifiers from the left and right end chars.

                # modifiers
                if w.startswith("*") and w.endswith("*"):  # contains
                    if ignore_case:
                        result += [
                            i for i in strings if w_.lower() in i.lower()
                        ]  # case insensitive.
                    else:
                        result += [i for i in strings if w_ in i]

                elif w.startswith("*"):  # prefix
                    if ignore_case:
                        result += [
                            i for i in strings if i.lower().endswith(w_.lower())
                        ]  # case insensitive.
                    else:
                        result += [i for i in strings if i.endswith(w_)]

                elif w.endswith("*"):  # suffix
                    if ignore_case:
                        result += [
                            i for i in strings if i.lower().startswith(w_.lower())
                        ]  # case insensitive.
                    else:
                        result += [i for i in strings if i.startswith(w_)]

                else:  # exact match
                    if ignore_case:
                        result += [
                            i for i in strings if i.lower() == w_.lower()
                        ]  # case insensitive.
                    else:
                        result += [i for i in strings if i == w_]

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

        # if 'fltr' is not an empty string; fltr 'strings' for matches using 'fltr'.
        if fltr:
            strings = cls.find_str(fltr, strings, regex=regex, ignore_case=ignore_case)

        # re.sub('[^A-Za-z0-9_:]+', '', fltr) #strip any special chars other than '_'.
        frm_ = fltr.strip("*").rstrip("*")
        # remove any modifiers from the left and right end chars.
        to_ = to.strip("*").rstrip("*")

        result = []
        for orig_str in strings:
            # modifiers
            if to.startswith("*") and to.endswith("*"):  # replace chars
                if ignore_case:
                    # remove frm_ from the string (case in-sensitive).
                    s = re.sub(frm_, to_, orig_str, flags=re.IGNORECASE)
                else:
                    s = orig_str.replace(frm_, to_)

            elif to.startswith("**"):  # append suffix
                s = orig_str + to_

            elif to.startswith("*"):  # replace suffix
                if ignore_case:
                    # get the starting index of 'frm_'.
                    index = re.search(frm_, orig_str, flags=re.IGNORECASE).start()
                    s = orig_str[:index] + to_
                else:
                    s = orig_str.split(frm_)[0] + to_

            elif to.endswith("**"):  # append prefix
                s = to_ + orig_str

            elif to.endswith("*"):  # replace prefix
                if ignore_case:
                    # get the ending index of 'frm_'.
                    index = re.search(frm_, orig_str, flags=re.IGNORECASE).end()
                    s = to_ + orig_str[index:]
                else:
                    s = to_ + frm_ + orig_str.split(frm_)[-1]

            elif not to_:  # if 'to_' is an empty string:
                if fltr.endswith("*") and not fltr.startswith(
                    "*"
                ):  # strip only beginning chars.
                    if ignore_case:
                        # remove the first instance of frm_ from the string (case in-sensitive).
                        s = re.sub(frm_, "", orig_str, 1, flags=re.IGNORECASE)
                    else:
                        # remove first instance of frm_ from the string.
                        s = orig_str.replace(frm_, "", 1)

                elif fltr.startswith("*") and not fltr.endswith(
                    "*"
                ):  # strip only ending chars.
                    if ignore_case:
                        # remove the last instance of frm_ from the string (case in-sensitive).
                        s = re.sub(r"(.*)" + frm_, r"\1", orig_str, flags=re.IGNORECASE)
                    else:
                        # remove last instance of frm_ from the string.
                        s = "".join(orig_str.rsplit(frm_, 1))

                else:
                    if ignore_case:
                        # remove frm_ from the string (case in-sensitive).
                        s = re.sub(frm_, "", orig_str, flags=re.IGNORECASE)
                    else:
                        s = orig_str.replace(frm_, "")  # remove frm_ from the string.
            else:  # else; replace whole string.
                s = to_

            if return_orig_strings:
                result.append((orig_str, s))
            else:
                result.append(s)

        return result

    @staticmethod
    def format_suffix(
        string,
        suffix="",
        strip="",
        strip_trailing_ints=False,
        strip_trailing_alpha=False,
    ):
        """Re-format the suffix for the given string.

        Parameters:
            string (str): The string to format.
            suffix (str): Append a new suffix to the given string.
            strip (str/list): Specific string(s) to strip from the end of the given string.
            strip_trailing_ints (bool): Strip all trailing integers.
            strip_trailing_alpha (bool): Strip all upper-case letters preceeded by a non alphanumeric character.

        Returns:
            (str)
        """
        import re

        try:
            s = string.split("|")[-1]
        except Exception:
            s = string.string().split("|")[-1]

        # strip each set of chars in 'strip' from end of string.
        if strip:
            strip = tuple(
                [i for i in iter_utils.IterUtils.make_iterable(strip) if not i == ""]
            )  # assure 'strip' is a tuple and does not contain any empty strings.
            while s.endswith(strip):
                for chars in strip:
                    s = s.rstrip(chars)

        while (
            ((s[-1] == "_" or s[-1].isdigit()) and strip_trailing_ints)
            or ("_" in s and (s == "_" or s[-1].isupper()))
            and strip_trailing_alpha
        ):
            if (
                s[-1] == "_" or s[-1].isdigit()
            ) and strip_trailing_ints:  # trailing underscore and integers.
                s = re.sub(re.escape(s[-1:]) + "$", "", s)

            if (
                "_" in s and (s == "_" or s[-1].isupper())
            ) and strip_trailing_alpha:  # trailing underscore and uppercase alphanumeric char.
                s = re.sub(re.escape(s[-1:]) + "$", "", s)

        return s + suffix

    @staticmethod
    @core_utils.CoreUtils.listify(threading=True)
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

        filepath = file_utils.FileUtils.format_path(filepath)

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
