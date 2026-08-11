# !/usr/bin/python
# coding=utf-8
import re
from typing import Union, List, Optional, Dict, Tuple, Callable, Iterable

# from this package:
from pythontk.core_utils._core_utils import CoreUtils
from pythontk.iter_utils._iter_utils import IterUtils


# ANSI/VT100 control sequences: CSI (``ESC [ … final-byte`` — covers SGR color) and the
# two-character escapes. Compiled at module scope because strip_ansi runs per console
# write, on a streaming hot path.
ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class StrUtils(CoreUtils):
    """ """

    @staticmethod
    def strip_ansi(string: str) -> str:
        """Remove ANSI escape sequences (color/cursor codes) from a string.

        A process teeing a TTY stream picks these up whether or not it wants them:
        CPython emits colored tracebacks when ``sys.stderr.isatty()``, and a tee that
        delegates ``isatty`` keeps that True. Consumers that render text but don't
        interpret VT100 (a Qt view, a log file) then show the raw bytes.

        Parameters:
            string (str): The text to scrub. Non-str input is returned unchanged.

        Returns:
            (str) The text with escape sequences removed.

        Example:
            strip_ansi("\\x1b[35mFile\\x1b[0m") --> 'File'
        """
        if not string or not isinstance(string, str):
            return string
        return ANSI_ESCAPE_RE.sub("", string)

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
                # Preserve unresolved placeholders verbatim, including their
                # format spec, so a second pass can still apply padding etc.
                # (`!r`/`!a` conversions on unresolved keys are not preserved.)
                if (
                    isinstance(value, str)
                    and value.startswith("{")
                    and value.endswith("}")
                ):
                    if format_spec:
                        return value[:-1] + ":" + format_spec + "}"
                    return value
                return super().format_field(value, format_spec)

        return SafeFormatter().format(text, **kwargs)

    @staticmethod
    def resolve_placeholders(text: str, **kwargs) -> dict:
        """Resolve placeholders and report what was substituted vs. left unresolved.

        A verbose companion to :meth:`replace_placeholders`, intended for building
        live previews / diagnostics (e.g. a tooltip that shows the resolved value
        of a user-typed pattern). It parses the ``{field}`` tokens in *text*
        (honouring format specs like ``{n:03d}`` and attribute / index access such
        as ``{obj.attr}`` / ``{seq[0]}`` — only the base name is reported), then
        splits them into the ones supplied in *kwargs* and the ones that are not.

        Args:
            text (str): The string containing placeholders.
            **kwargs: Key-value pairs corresponding to placeholders.

        Returns:
            dict with keys:
                - ``"result"`` (str): *text* with supplied keys substituted and
                  unresolved placeholders preserved verbatim — identical to
                  :meth:`replace_placeholders`.
                - ``"fields"`` (list[str]): every distinct placeholder base name
                  found, in first-seen order (positional ``{}`` / ``{0}`` skipped).
                - ``"resolved"`` (dict[str, str]): base name -> its supplied value
                  rendered as a string, for fields present in *kwargs*.
                - ``"unresolved"`` (list[str]): base names present in *text* but
                  absent from *kwargs*, in first-seen order.

        Raises:
            ValueError: If *text* is a malformed format string (e.g. a lone ``{``),
                matching :meth:`replace_placeholders`.

        Example:
            >>> StrUtils.resolve_placeholders("{root}/{name}_{ver:03d}", root="C:/p", name="shot")
            {'result': 'C:/p/shot_{ver:03d}', 'fields': ['root', 'name', 'ver'], 'resolved': {'root': 'C:/p', 'name': 'shot'}, 'unresolved': ['ver']}
        """
        import string

        fields = []
        for _literal, field_name, _spec, _conv in string.Formatter().parse(text):
            if not field_name:  # None (literal run) or "" (positional auto-number)
                continue
            base = field_name.split(".")[0].split("[")[0]
            if base and not base.isdigit() and base not in fields:
                fields.append(base)

        resolved = {name: format(kwargs[name]) for name in fields if name in kwargs}
        unresolved = [name for name in fields if name not in kwargs]

        return {
            "result": StrUtils.replace_placeholders(text, **kwargs),
            "fields": fields,
            "resolved": resolved,
            "unresolved": unresolved,
        }

    @staticmethod
    def replace_delimited(
        text: str,
        context: dict,
        prefix: str = "__",
        suffix: str = "__",
    ) -> str:
        """Replace delimited placeholders in *text* using *context*.

        Unlike ``replace_placeholders`` (which uses ``{key}`` syntax), this
        method supports **arbitrary** delimiters and is therefore safe inside
        code templates, QSS files, or any format where curly braces carry
        their own meaning.

        Args:
            text: Source string containing placeholders.
            context: Mapping of placeholder names to replacement values.
                     Values are converted to ``str`` automatically.
            prefix: Opening delimiter before each key (default ``"__"``).
            suffix: Closing delimiter after each key  (default ``"__"``).

        Returns:
            The text with all matching placeholders substituted.

        Example:
            >>> StrUtils.replace_delimited(
            ...     'FBX = r"__FBX_PATH__"',
            ...     {"FBX_PATH": "C:/scene.fbx"},
            ... )
            'FBX = r"C:/scene.fbx"'
            >>> StrUtils.replace_delimited(
            ...     "color: {TEXT_COLOR};",
            ...     {"TEXT_COLOR": "rgb(255,255,255)"},
            ...     prefix="{", suffix="}",
            ... )
            'color: rgb(255,255,255);'
        """
        for key, value in context.items():
            text = text.replace(f"{prefix}{key}{suffix}", str(value))
        return text

    @staticmethod
    @CoreUtils.listify(threading=True)
    def set_case(string, case="title"):
        """Format the given string(s) in the given case.

        Parameters:
            string (str/list): The string(s) to format.
            case (str): The desired return case. Accepts all python case operators.
                    valid: 'upper', 'lower', 'capitalize', 'swapcase', 'title' (default), 'pascal', 'camel', None.
        Returns:
            (str/list) List if 'string' given as list.
        """
        if (not string) or (not isinstance(string, str)):
            return ""

        if case == "pascal":
            return string[:1].capitalize() + string[1:]  # capitalize the first letter.

        elif case == "camel":
            return string[0].lower() + string[1:]  # lowercase the first letter.

        elif case is None:  # documented "no transform": return the string unchanged.
            return string

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

        # Iterate `delimiters` directly (char-by-char for a string, element-wise for
        # a list) so the target side splits identically to the item side below.
        pattern = "|".join(re.escape(d) for d in delimiters)
        target_parts = re.split(pattern, target)

        def match_hierarchy(item_parts):
            return all(p1 == p2 for p1, p2 in zip(item_parts, target_parts))

        def is_upstream(item_parts):
            return len(item_parts) < len(target_parts)

        def is_downstream(item_parts):
            return len(item_parts) > len(target_parts)

        def filter_items(item):
            item_parts = re.split(pattern, item)

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
            >>> StrUtils.split_delimited_string('a|b|c|d')
            ['a', 'b', 'c', 'd']

            >>> StrUtils.split_delimited_string('  a  | b |  c  ', strip_whitespace=True)
            ['a', 'b', 'c']

            >>> StrUtils.split_delimited_string('a||b||c', remove_empty=True)
            ['a', 'b', 'c']

            >>> StrUtils.split_delimited_string('c|a|b', func=sorted)
            ['a', 'b', 'c']

            >>> StrUtils.split_delimited_string('a|b|c', func=reversed)
            ['c', 'b', 'a']

            >>> StrUtils.split_delimited_string('apple|banana|cherry', func=lambda x: [s.upper() for s in x])
            ['APPLE', 'BANANA', 'CHERRY']

            # Binary splitting at specific occurrence (tuple output)
            >>> StrUtils.split_delimited_string('a|b|c|d', occurrence=-1)
            ('a|b|c', 'd')

            >>> StrUtils.split_delimited_string('a|b|c|d', occurrence=0)
            ('', 'a')

            >>> StrUtils.split_delimited_string('a|b|c|d', occurrence=1)
            ('a', 'b')

            >>> StrUtils.split_delimited_string('string', occurrence=-1)  # No delimiter found
            ('string', '')

            # Max split limiting
            >>> StrUtils.split_delimited_string('a|b|c|d', max_split=2)
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
            parts = list(func(parts))

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
            # 'at' is a string: locate the requested occurrence by position.
            indices = [m.start() for m in re.finditer(re.escape(at), src)]
            try:
                # positive occurrence is 1-based; negative counts from the right (-1 == last).
                i = indices[occurrence - 1] if occurrence > 0 else indices[occurrence]
            except IndexError:  # occurrence out of range (or char not found).
                return src
            return cls.insert(src, str(ins), i if before else i + len(at))

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
    def collapse_delimiter_runs(string, delimiter="_", strip_trailing=True):
        """Collapse consecutive delimiter runs to a single delimiter.

        Cleans the separator residue left behind when tokens are removed
        from a delimited name: stripping ``tok`` from ``a__tok__tokB``
        yields ``a____B`` — this collapses it to ``a_B``. Leading
        delimiters are preserved (a leading ``_`` can be a deliberate
        legality prefix); trailing runs are stripped by default.

        Parameters:
            string (str): The string to clean.
            delimiter (str): The delimiter whose runs to collapse.
            strip_trailing (bool): Also remove any trailing delimiter run.

        Returns:
            (str)

        Example:
            collapse_delimiter_runs('vdat____Shape702') #returns: 'vdat_Shape702'
            collapse_delimiter_runs('Crate__') #returns: 'Crate'
            collapse_delimiter_runs('_LeadingKept__x') #returns: '_LeadingKept_x'
        """
        if not string or not isinstance(string, str):
            return string

        d = re.escape(delimiter)
        result = re.sub(f"{d}{{2,}}", delimiter, string)
        if strip_trailing:
            result = re.sub(f"{d}+$", "", result)
        return result

    @staticmethod
    @CoreUtils.listify(threading=True)
    def truncate(string, length=75, mode="start", insert="..", head=None):
        """Shorten the given string to the given length.
        An ellipsis will be added to the section trimmed.

        Parameters:
            string (str): The string to truncate.
            length (int): The maximum allowed length before truncating.
            mode (str): Truncation mode.
                - 'start'/'left': Trim from start (keep end) - default
                - 'end'/'right': Trim from end (keep start)
                - 'middle': Trim from middle (keep start and end)
                - 'path': Like 'middle', but cuts only at separators so whole
                  path components survive at both ends (drive/root and leading
                  dirs at the front, filename and its parents at the back).
                  Falls back to 'middle' when there is nothing to drop between
                  a head and a tail.
            insert (str): Characters to add at the trimmed area. (default: ellipsis)
            head (int): 'path' mode only — cap the leading components kept, so
                the budget the head would have taken goes to the tail instead
                (``head=1`` keeps just the drive/root). None grows the head
                greedily with whatever the tail could not use. Ignored by the
                other modes.

        Returns:
            (str)

        Examples:
            truncate('12345678', 4) #returns: '..5678' (start mode)
            truncate('12345678', 4, 'end') #returns: '1234..' (end mode)
            truncate('12345678', 6, 'middle') #returns: '12..78' (middle mode)
            truncate('O:/Cloud/jets/c130j/sourceimages/tex/x_DIFF.png', 36, 'path')
                #returns: 'O:/Cloud/jets/../tex/x_DIFF.png' (path mode)
            truncate('O:/Cloud/jets/c130j/sourceimages/tex/x_DIFF.png', 36, 'path', head=1)
                #returns: 'O:/../sourceimages/tex/x_DIFF.png' (capped head, wider tail)
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
            return StrUtils._truncate_middle(string, length, insert)
        elif mode == "path":
            return StrUtils._truncate_path(string, length, insert, head)
        else:
            # Fallback to start trimming (default behavior)
            return insert + string[-length:]

    @staticmethod
    def _truncate_middle(string, length, insert):
        """Character-count middle cut — the 'middle' mode of :meth:`truncate`.

        Split around the middle; visible chars exclude the insert. Also the
        degenerate-case fallback for 'path' mode.
        """
        avail = max(1, length - len(insert))
        if avail <= 1:
            return string[0] + insert
        left = avail // 2
        right = avail - left
        return string[:left] + insert + string[-right:]

    @staticmethod
    def _truncate_path(string, length, insert, head=None):
        """Component-aware middle truncation — the 'path' mode of :meth:`truncate`.

        A character-count middle cut lands wherever it lands, so a path comes
        back with half-words at the seam (``O:/Cloud/Projects/jets/..ures/x.png``).
        This drops whole components instead, and grows what it keeps in the
        order that carries meaning: the head opens on the drive/root *and* its
        first directory (a bare ``O:/..`` locates nothing), the tail then takes
        the filename and as many of its parents as fit, and any space left over
        goes back to the head. Degenerate shapes — nothing between a head and a
        tail, or a filename that alone overruns the budget — degrade to
        :meth:`_truncate_middle` rather than inventing a boundary.

        ``head`` caps that leading run (``head=1`` = drive/root only). Capping it
        is what lets a caller spend the budget on the end of the path instead:
        the tail is grown first, so a lower cap can only leave the tail wider.
        """
        # Preserve any leading separator run (UNC "//server/share", posix root).
        stripped = string.lstrip("/\\")
        prefix = string[: len(string) - len(stripped)]
        sep = "\\" if stripped.count("\\") > stripped.count("/") else "/"
        parts = stripped.split(sep)
        if len(parts) < 3:  # nothing to drop between a head and a tail
            return StrUtils._truncate_middle(string, length, insert)

        def build(head_count, tail_count):
            head = parts[:head_count]
            tail = parts[len(parts) - tail_count :]
            return prefix + sep.join(head + [insert] + tail)

        # Keeping head + tail must still leave a component to drop, or the
        # insert would mark an elision that never happened ("a/../b/c").
        keepable = len(parts) - 1
        max_head = keepable if head is None else max(1, int(head))

        head_count, tail_count = 1, 1
        if len(build(1, 1)) > length:
            return StrUtils._truncate_middle(string, length, insert)
        if (  # drive/root + 1st dir
            max_head >= 2 and 3 <= keepable and len(build(2, 1)) <= length
        ):
            head_count = 2

        # Tail first — the filename end is what identifies the file.
        while head_count + tail_count < keepable:
            if len(build(head_count, tail_count + 1)) > length:
                break
            tail_count += 1
        # Then spend whatever is left widening the head, up to the cap.
        while head_count < max_head and head_count + tail_count < keepable:
            if len(build(head_count + 1, tail_count)) > length:
                break
            head_count += 1
        return build(head_count, tail_count)

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

        if as_string and result is not None:
            return str(result)
        return result

    @staticmethod
    def _parse_wildcard_terms(find, ignore_case=False):
        """Parse a pipe-separated wildcard filter into (term, mode) pairs.

        Shared by 'find_str' and 'find_str_and_format' so the two can never drift
        on what 'chars', 'chars*', '*chars' and '*chars*' mean. The returned terms
        are index-aligned with ``find.split("|")``, which is what lets the
        formatter map a matched string back to the filter term that selected it.

        Parameters:
            find (str): The search string, e.g. 'chars*|*chars'.
            ignore_case (bool): Case-fold the terms (values must be folded to match).

        Returns:
            (list) [(term, mode)] where mode is one of
                    'contains' / 'endswith' / 'startswith' / 'exact'.
        """
        terms = []
        for w in find.split("|"):
            term = w.strip("*")
            starts, ends = w.startswith("*"), w.endswith("*")

            if starts and ends:
                mode = "contains"
            elif starts:
                mode = "endswith"
            elif ends:
                mode = "startswith"
            else:
                mode = "exact"

            terms.append((term.lower() if ignore_case else term, mode))
        return terms

    @staticmethod
    def _match_wildcard_term(value, term, mode):
        """Test one already case-folded value against one parsed wildcard term.

        Parameters:
            value (str): The candidate, pre-folded when ignoring case.
            term (str): The term from '_parse_wildcard_terms'.
            mode (str): 'contains' / 'endswith' / 'startswith' / 'exact'.

        Returns:
            (bool)
        """
        if mode == "contains":
            return term in value
        elif mode == "endswith":
            return value.endswith(term)
        elif mode == "startswith":
            return value.startswith(term)
        return value == term

    @classmethod
    def find_str(cls, find, strings, regex=False, ignore_case=False):
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
        search_terms = cls._parse_wildcard_terms(find, ignore_case)

        # Use set for O(1) duplicate checking
        seen = set()
        result = []

        for s in strings:
            if s in seen:
                continue

            check = s.lower() if ignore_case else s

            for term, mode in search_terms:
                if cls._match_wildcard_term(check, term, mode):
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

        The asterisk in ``to`` marks *the part of the original that is kept*, so
        the modifier mirrors ``fltr``'s wildcard on the side it preserves. A
        doubled asterisk keeps everything, turning a replace into an append.

        Parameters:
            strings (list): A list of string objects to search.
            to (str): The replacement, with an optional asterisk modifier. An empty
                    string strips the part matched by 'fltr'.
                    "" - (empty string) - strip the matched chars.
                    chars - replace the whole string.
                    *chars* - replace only the matched chars.
                    *chars - replace the suffix (drop from the match onward).
                    **chars - append a suffix (keep the whole string).
                    chars* - replace the prefix (drop through the match).
                    chars** - append a prefix (keep the whole string).
                    When 'fltr' holds pipe-separated terms, 'to' may hold the same
                    number of terms; they pair positionally with the filter term
                    that each string matched (e.g. fltr '*_L|*_R' with to
                    '*_lt|*_rt' renames '_L' names one way and '_R' names another).
                    A single 'to' term applies to every filter term. A pipe in 'to'
                    is literal when 'fltr' has no pipe-separated terms, and always
                    literal when regex is True.
            fltr (str): See the 'find_str' function's 'fltr' parameter for documentation.
                    Each pipe-separated term supplies the "from" text for the strings
                    it matched, so multi-term filters format correctly.
            regex (bool): Use regular expressions instead of wildcards for the 'fltr'
                    argument. The pattern is used for the substitution as well as the
                    search, so '|' stays alternation and asterisks keep their regex
                    meaning. Capture groups are available in 'to' as '\\1', '\\2' or
                    '\\g<name>'; an escape that cannot be expanded is used verbatim.
            ignore_case (bool): Ignore case when searching. Applies to the 'fltr'
                    parameter's search and to the substitution it drives.
            return_orig_strings (bool): Return the old names as well as the new.

        Returns:
            (list) if return_orig_strings: list of two element tuples containing the original and modified string pairs. [('frm','to')]
                    else: a list of just the new names.

        Note:
            'replace_prefix' and 'replace_suffix' fall back to a plain append when
            the filter text is not present in a string (including the no-filter
            case), so '*_GEO' with an empty filter suffixes every string.

        Example:
            find_str_and_format(['pCube1'], '*box*', '*Cube*') #-> ['pbox1']
            find_str_and_format(['arm_L','arm_R'], '*_lt|*_rt', '*_L|*_R') #-> ['arm_lt','arm_rt']
            find_str_and_format(['pCube1'], r'*\\1_box*', r'(p)Cube', regex=True) #-> ['p_box1']
        """
        import re

        # Filter out non-string values
        strings = [s for s in strings if isinstance(s, str)]

        if not strings:  # Early exit
            return []

        flags = re.IGNORECASE if ignore_case else 0

        # Split into paired terms. Wildcard mode only: in regex mode '|' is
        # alternation, so the filter is one pattern and 'to' one template.
        if regex or "|" not in fltr:
            fltr_terms, to_terms = [fltr], [to]
        else:
            fltr_terms = fltr.split("|")
            to_terms = to.split("|") if "|" in to else [to]

        # Compile the regex filter once; it drives the substitution as well as
        # the search, so a bad pattern is fatal here rather than silently literal.
        rx_filter = None
        if regex and fltr:
            try:
                rx_filter = re.compile(fltr, flags)
            except re.error as e:
                print(f"# Error find_str_and_format: in {fltr}: {e}. #")
                return []

        # Pair each surviving string with the index of the filter term that
        # selected it -- that term supplies its "from" text and its paired 'to' --
        # plus, in regex mode, the match itself, so the span is not searched twice.
        # Filtering happens here rather than through 'find_str' because find_str
        # deduplicates, which would collapse same-named inputs (two objects
        # sharing a short name) into a single result and format only one.
        if not fltr:
            matched = [(s, 0, None) for s in strings]
        elif regex:
            matched = []
            for s in strings:
                m = rx_filter.search(s)
                if m:
                    matched.append((s, 0, m))
        else:
            parsed = cls._parse_wildcard_terms(fltr, ignore_case)
            matched = []
            for s in strings:
                check = s.lower() if ignore_case else s
                idx = next(
                    (
                        i
                        for i, (term, mode) in enumerate(parsed)
                        if cls._match_wildcard_term(check, term, mode)
                    ),
                    None,
                )
                if idx is not None:
                    matched.append((s, idx, None))

        if not matched:  # Early exit
            return []

        def infer_mode(t):
            """Map a 'to' term to its formatting mode.

            The '**' tests come first so an all-asterisk term ('**', '***')
            reads as an append of an empty payload rather than a replace.
            """
            if t.startswith("**"):
                return "append_suffix"
            elif t.endswith("**"):
                return "append_prefix"
            elif t.startswith("*") and t.endswith("*") and len(t) > 1:
                return "replace_chars"
            elif t.startswith("*"):
                return "replace_suffix"
            elif t.endswith("*"):
                return "replace_prefix"
            elif not t.strip("*"):
                return "strip"
            return "replace_whole"

        def strip_submode(term):
            """Which occurrences a strip removes, from the filter term's wildcards."""
            if term.endswith("*") and not term.startswith("*"):
                return "first"
            elif term.startswith("*") and not term.endswith("*"):
                return "last"
            return "all"

        term_cache = {}

        def term_data(idx):
            """Resolve the (from, to, mode, ...) bundle for one filter term."""
            if idx not in term_cache:
                frm_term = fltr_terms[idx]
                to_term = to_terms[min(idx, len(to_terms) - 1)]
                frm_ = frm_term if regex else frm_term.strip("*")
                mode = infer_mode(to_term)

                frm_rx = rx_filter
                if not regex and frm_ and ignore_case:
                    try:
                        frm_rx = re.compile(re.escape(frm_), re.IGNORECASE)
                    except re.error:
                        frm_rx = None

                term_cache[idx] = (
                    frm_,
                    to_term.strip("*"),
                    mode,
                    frm_rx,
                    # Regex has no leading/trailing wildcard to read a sub-mode from.
                    ("all" if regex else strip_submode(frm_term))
                    if mode == "strip" and frm_
                    else None,
                )
            return term_cache[idx]

        def expand(match, template):
            """Expand capture-group backrefs; fall back to the literal template."""
            if match is None:
                return template
            try:
                return match.expand(template)
            except (re.error, IndexError):
                return template

        result = []
        for orig_str, idx, match in matched:
            frm_, to_, mode, frm_rx, strip_mode = term_data(idx)

            s = orig_str  # Default: no change (in regex mode every branch below
            # works off 'match', resolved during the filter pass above)

            if mode == "replace_chars":
                if regex:
                    if frm_rx:
                        try:
                            s = frm_rx.sub(to_, orig_str)
                        except (re.error, IndexError):
                            # Unusable escape (re.error) or a backref naming a
                            # group the pattern lacks (IndexError) -> use 'to'
                            # verbatim, matching 'expand' below.
                            s = frm_rx.sub(lambda _m: to_, orig_str)
                elif frm_:
                    if frm_rx:
                        s = frm_rx.sub(to_.replace("\\", "\\\\"), orig_str)
                    else:
                        s = orig_str.replace(frm_, to_)

            elif mode == "append_suffix":
                s = orig_str + (expand(match, to_) if regex else to_)

            elif mode == "append_prefix":
                s = (expand(match, to_) if regex else to_) + orig_str

            elif mode == "replace_suffix":
                if regex:
                    s = (
                        orig_str[: match.start()] + expand(match, to_)
                        if match
                        else orig_str + to_
                    )
                elif frm_:
                    if frm_rx:
                        m = frm_rx.search(orig_str)
                        s = orig_str[: m.start()] + to_ if m else orig_str + to_
                    else:
                        parts = orig_str.split(frm_, 1)
                        s = parts[0] + to_ if len(parts) > 1 else orig_str + to_
                else:
                    s = orig_str + to_

            elif mode == "replace_prefix":
                if regex:
                    s = (
                        expand(match, to_) + orig_str[match.end() :]
                        if match
                        else to_ + orig_str
                    )
                elif frm_:
                    if frm_rx:
                        m = frm_rx.search(orig_str)
                        s = to_ + orig_str[m.end() :] if m else to_ + orig_str
                    else:
                        parts = orig_str.split(frm_, 1)
                        # Drop the matched prefix, mirroring the ignore_case branch.
                        s = to_ + parts[1] if len(parts) > 1 else to_ + orig_str

            elif mode == "strip":
                if regex:
                    if frm_rx:
                        s = frm_rx.sub("", orig_str)
                elif frm_:
                    if strip_mode == "first":
                        s = (
                            frm_rx.sub("", orig_str, count=1)
                            if frm_rx
                            else orig_str.replace(frm_, "", 1)
                        )
                    elif strip_mode == "last":
                        if frm_rx:
                            matches = list(frm_rx.finditer(orig_str))
                            if matches:
                                last = matches[-1]
                                s = orig_str[: last.start()] + orig_str[last.end() :]
                        else:
                            s = "".join(orig_str.rsplit(frm_, 1))
                    else:  # all
                        s = (
                            frm_rx.sub("", orig_str)
                            if frm_rx
                            else orig_str.replace(frm_, "")
                        )

            elif mode == "replace_whole":
                s = expand(match, to_) if regex else to_

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

        s = string

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
                # Only strip digits not preceded by underscore (e.g. CUBE01 -> CUBE,
                # but CUBE_01 stays as-is since _01 is intentional numbering)
                if not re.search(r"_\d+$", s):
                    s = re.sub(r"\d+$", "", s)
                    stripped = True
            if strip_trailing_alpha and s and s[-1].isupper():
                s = re.sub(r"(?:[^0-9A-Za-z]+)?[A-Z]+$", "", s)
                stripped = True
            if not stripped:
                break

        return s + suffix

    @staticmethod
    def strip_known_affix(
        string: str,
        prefix: str = "",
        suffix: str = "",
    ) -> str:
        """Strip a configured prefix and/or suffix from a string, case-insensitively.

        Pure primitive: only literal affix matches (plus adjacent ``_`` separators)
        are removed. Leading/trailing underscores elsewhere in the string are
        preserved — callers that want a fully scrubbed name should chain
        ``.strip("_")`` themselves, or use :py:meth:`apply_affix`.

        Matching rules:
        - Case-insensitive on the affix core (underscores in the supplied affix are
          treated as separators, not part of the token).
        - The match consumes any adjacent ``_`` runs on the affix side and tolerates
          stray ``_`` between the affix and the string boundary
          (``_Mat_brick`` with prefix ``Mat_`` → ``brick``).
        - Boundary required: the core must be followed by ``_`` or end-of-string
          (prefix) / preceded by ``_`` or start-of-string (suffix). This prevents
          false positives like ``Matte_door`` for prefix ``Mat_``.

        Parameters:
            string: The string to strip.
            prefix: Prefix to remove (e.g. ``"Mat_"``). Matched case-insensitively.
            suffix: Suffix to remove (e.g. ``"_MAT"``). Matched case-insensitively.

        Returns:
            The string with the configured affixes removed. If neither matches,
            returns the input unchanged.
        """
        import re

        s = string
        if prefix:
            core = prefix.strip("_")
            if core:
                s = re.sub(
                    rf"^_*(?i:{re.escape(core)})(?:_+|$)",
                    "",
                    s,
                )
        if suffix:
            core = suffix.strip("_")
            if core:
                s = re.sub(
                    rf"(?:_+|^)(?i:{re.escape(core)})_*$",
                    "",
                    s,
                )
        return s

    @staticmethod
    def infer_affix_mode(
        text: str,
        delimiter: str = "_",
        *,
        default: str = "prefix",
    ) -> str:
        """Infer ``"prefix"`` or ``"suffix"`` from *delimiter* placement in *text*.

        Pure primitive for tools that let users type either form
        (``"MAT_"`` vs ``"_MAT"``) and want to interpret intent without
        an explicit toggle.

        Rule:
        - Leading delimiter (e.g. ``"_MAT"`` with ``delimiter="_"``) →
          the token attaches at the END of a base name, so it's a
          ``"suffix"``.
        - Trailing delimiter (e.g. ``"MAT_"``) → it attaches at the
          START, so it's a ``"prefix"``.
        - Both edges or neither edge has the delimiter (or *delimiter*
          is empty) → can't decide → return *default*.

        Parameters:
            text: The affix string.
            delimiter: Boundary token that signals which side the affix
                attaches to. Pass ``""`` to skip detection entirely.
            default: Fallback when detection is ambiguous or disabled.
                Library default is ``"prefix"`` because type-leading
                prefixes (``MAT_brick``, ``GEO_arm``) are the more
                common asset-naming convention.

        Returns:
            ``"prefix"`` or ``"suffix"``.
        """
        fallback = (default or "prefix").lower()
        if fallback not in ("prefix", "suffix"):
            fallback = "prefix"
        if not text or not delimiter:
            return fallback

        starts = text.startswith(delimiter)
        ends = text.endswith(delimiter)
        if starts and not ends:
            return "suffix"
        if ends and not starts:
            return "prefix"
        return fallback

    @staticmethod
    def split_affix(
        text: str,
        mode: str = "auto",
        *,
        default: str = "prefix",
        delimiter: str = "_",
    ) -> Tuple[str, str]:
        """Split an affix string into a ``(prefix, suffix)`` pair per *mode*.

        Pure primitive — turns a user-supplied affix string plus a mode
        declaration into the pair consumed by :py:meth:`apply_affix`.

        Modes:
        - ``"prefix"``: returns ``(text, "")``.
        - ``"suffix"``: returns ``("", text)``.
        - ``"auto"``: delegates to :py:meth:`infer_affix_mode` using
          *delimiter* and *default* — leading delimiter (e.g.
          ``"_MAT"``) → suffix; trailing delimiter (e.g. ``"MAT_"``)
          → prefix; ambiguous → *default*.

        Parameters:
            text: The affix string entered by the user.
            mode: ``"prefix"``, ``"suffix"``, or ``"auto"`` (default).
            default: Fallback mode used when ``mode="auto"`` and *text*
                has no boundary delimiter. Defaults to ``"prefix"``.
            delimiter: Boundary character used by auto-detection.
                Defaults to ``"_"``; pass ``""`` to disable detection
                so auto always falls through to *default*.

        Returns:
            ``(prefix, suffix)`` — at most one element is non-empty. An
            empty *text* returns ``("", "")``.
        """
        if not text:
            return ("", "")

        m = (mode or "auto").lower()
        if m not in ("prefix", "suffix", "auto"):
            m = "auto"
        if m == "auto":
            m = StrUtils.infer_affix_mode(
                text, delimiter=delimiter, default=default
            )

        if m == "prefix":
            return (text, "")
        return ("", text)

    @staticmethod
    def apply_affix(
        string: str,
        prefix: str = "",
        suffix: str = "",
    ) -> str:
        """Idempotently apply a prefix and/or suffix to a string.

        If both ``prefix`` and ``suffix`` are empty, returns ``string`` unchanged
        (no implicit underscore cleanup). Otherwise strips any pre-existing
        occurrence of the configured affixes via :py:meth:`strip_known_affix`,
        cleans dangling separator underscores on the affix side(s), and applies
        the affixes. Safe to call repeatedly without producing ``Mat_Mat_brick``
        duplicates or ``Mat_brick_`` trailing-underscore artifacts.

        Parameters:
            string: The base string.
            prefix: Prefix to apply (e.g. ``"Mat_"``).
            suffix: Suffix to apply (e.g. ``"_MAT"``).

        Returns:
            ``f"{prefix}{core}{suffix}"`` with no duplicate affixes and no
            dangling underscores between the affixes and the core. Internal
            underscores in the core are preserved.
        """
        if not prefix and not suffix:
            return string
        core = StrUtils.strip_known_affix(string, prefix=prefix, suffix=suffix)
        if prefix:
            core = core.lstrip("_")
        if suffix:
            core = core.rstrip("_")
        return f"{prefix}{core}{suffix}"

    @staticmethod
    def alpha_sequence(index: int) -> str:
        """Excel-column-style alphabetic label for a 0-based index.

        ``0 -> "A"``, ``25 -> "Z"``, ``26 -> "AA"``, ``27 -> "AB"``, ``701 -> "ZZ"``,
        ``702 -> "AAA"``. Useful for producing human-friendly sequential suffixes
        for collision groups (e.g. ``mat_A``, ``mat_B``, ...).

        Parameters:
            index: Non-negative 0-based position.

        Returns:
            Uppercase alphabetic label.

        Raises:
            ValueError: if ``index`` is negative.
        """
        if index < 0:
            raise ValueError(f"index must be non-negative, got {index}")
        s = ""
        n = index
        while True:
            s = chr(ord("A") + n % 26) + s
            n = n // 26 - 1
            if n < 0:
                break
        return s

    @staticmethod
    def sequential_suffixes(
        count: int,
        switch_at: int = 26,
        lowercase: bool = False,
    ) -> List[str]:
        """Generate ``count`` sequential labels for naming sibling items.

        Uses single letters (``A, B, ..., Z``) while ``count <= switch_at`` (and
        ``<= 26``); otherwise zero-padded numerics (``01, 02, ...``) with width
        ``max(2, len(str(count)))``. Useful for naming the children of a
        per-group split — e.g. ``mesh_A``, ``mesh_B`` when there are few, or
        ``mesh_001`` … ``mesh_120`` when there are many.

        Parameters:
            count: How many suffixes to produce.
            switch_at: Inclusive upper bound for the letter scheme; counts
                above this fall back to numerics. Capped at 26 (the alphabet
                size) regardless.
            lowercase: Return lowercase letters when in the letter scheme.

        Returns:
            List of ``count`` suffix strings.

        Examples:
            >>> StrUtils.sequential_suffixes(3)
            ['A', 'B', 'C']
            >>> StrUtils.sequential_suffixes(30)[:3]
            ['01', '02', '03']
            >>> StrUtils.sequential_suffixes(3, lowercase=True)
            ['a', 'b', 'c']
        """
        if count <= 0:
            return []
        cap = min(switch_at, 26)
        if count <= cap:
            base = ord("a") if lowercase else ord("A")
            return [chr(base + i) for i in range(count)]
        pad = max(2, len(str(count)))
        return [str(i + 1).zfill(pad) for i in range(count)]

    @staticmethod
    def resolve_name_collisions(
        names: Iterable[str],
        strip: Union[str, List[str]] = "",
        strip_trailing_ints: bool = False,
        strip_trailing_alpha: bool = False,
        collision_suffix: Union[str, Callable[[int, int], str], None] = "alpha",
        suffix_separator: str = "_",
    ) -> Dict[str, str]:
        """Reduce a batch of names to a shared base form, then disambiguate
        same-base groups with sequential suffixes.

        Each name is reduced via :func:`format_suffix` using the ``strip*`` kwargs,
        then names sharing a base are grouped (input order preserved). Within each
        group:

          - **Single-member group**: the name is renamed to the bare base. This is
            unconditional — non-colliding names always strip to base regardless of
            ``collision_suffix``.
          - **Multi-member group**: if ``collision_suffix`` is not ``None``, each
            member is renamed to ``f"{base}{suffix_separator}{suffix}"`` where the
            suffix comes from the chosen scheme. If ``None``, group members keep
            their original names (caller can treat as an unresolved conflict).

        Parameters:
            names: Names to process.
            strip: Forwarded to :func:`format_suffix`.
            strip_trailing_ints: Forwarded to :func:`format_suffix`.
            strip_trailing_alpha: Forwarded to :func:`format_suffix`.
            collision_suffix: Suffix scheme for collision groups. One of:

                - ``"alpha"``: ``A, B, ..., Z, AA, AB, ...`` (Excel-column style).
                - ``"numeric"``: zero-padded, width = ``max(2, len(str(count)))``.
                - ``None``: group members keep their original names.
                - ``callable(index, count) -> str``: custom scheme. Returning the
                  empty string yields the bare base; returning ``None`` keeps the
                  original name.
            suffix_separator: Joined between base and suffix (default ``"_"``).

        Returns:
            Mapping ``original_name -> new_name`` for names that change. No-ops
            (where the new name equals the original) are omitted, so the result
            is directly suitable for driving a rename loop.

        Examples:
            >>> StrUtils.resolve_name_collisions(
            ...     ["mat", "mat1", "mat2", "wood", "wood3"],
            ...     strip_trailing_ints=True,
            ...     collision_suffix="alpha",
            ... )
            {'mat': 'mat_A', 'mat1': 'mat_B', 'mat2': 'mat_C', 'wood3': 'wood'}

            >>> StrUtils.resolve_name_collisions(
            ...     ["mat", "mat1"], strip_trailing_ints=True, collision_suffix=None,
            ... )
            {}
        """

        def _suffix_at(scheme, index, count):
            if scheme is None:
                return None
            if callable(scheme):
                return scheme(index, count)
            if scheme == "alpha":
                return StrUtils.alpha_sequence(index)
            if scheme == "numeric":
                width = max(2, len(str(count)))
                return str(index + 1).zfill(width)
            raise ValueError(f"Unknown collision_suffix scheme: {scheme!r}")

        names = list(names)
        groups: Dict[str, List[str]] = {}
        for name in names:
            base = StrUtils.format_suffix(
                name,
                strip=strip,
                strip_trailing_ints=strip_trailing_ints,
                strip_trailing_alpha=strip_trailing_alpha,
            )
            if not base:
                continue
            groups.setdefault(base, []).append(name)

        result: Dict[str, str] = {}
        for base, members in groups.items():
            if len(members) == 1:
                name = members[0]
                if name != base:
                    result[name] = base
                continue

            count = len(members)
            for i, name in enumerate(members):
                suffix = _suffix_at(collision_suffix, i, count)
                if suffix is None:
                    continue  # keep original name
                new_name = f"{base}{suffix_separator}{suffix}" if suffix else base
                if new_name != name:
                    result[name] = new_name

        return result

    # Matches the prefix produced by the default ``time_stamp`` format:
    # "MM-DD-YYYY  HH:MM  <path>". The path is captured whole so paths
    # containing spaces survive the detach.
    _TIME_STAMP_RE = None  # compiled lazily below

    @staticmethod
    @CoreUtils.listify(threading=True)
    def time_stamp(filepath, stamp="%m-%d-%Y  %H:%M"):
        """Attach or detach a modified timestamp and date to/from a given file path.

        A path that already carries a default-format stamp prefix is returned
        with the stamp removed; otherwise the file's mtime is prepended using
        ``stamp``. Detach only recognizes the default format.

        Parameters:
            filepath (str): The full path to a file. ie. 'C:/Windows/Temp/__AUTO-SAVE__untitled.0001.mb'
            stamp (str): The time stamp format.

        Returns:
            str: Filepath with attached or detached timestamp, depending on whether it initially had a timestamp.
            ie. '11-09-2021  16:46  C:/Windows/Temp/__AUTO-SAVE__untitled.0001.mb' from 'C:/Windows/Temp/__AUTO-SAVE__untitled.0001.mb'
        """
        from datetime import datetime
        import os.path
        import re
        from pythontk.file_utils._file_utils import FileUtils

        if StrUtils._TIME_STAMP_RE is None:
            StrUtils._TIME_STAMP_RE = re.compile(
                r"^\d{2}-\d{2}-\d{4}  \d{2}:\d{2}  (.+)$"
            )

        filepath = FileUtils.format_path(filepath)

        match = StrUtils._TIME_STAMP_RE.match(filepath)
        if match:  # already stamped: return the bare path (spaces preserved).
            return match.group(1)

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
