# !/usr/bin/python
# coding=utf-8
import re
from typing import Union, List, Optional, Dict, Tuple, Callable, Iterable

# from this package:
from pythontk.core_utils._core_utils import CoreUtils
from pythontk.iter_utils._iter_utils import IterUtils

# The facade's bodies, split by concept (CODE_STANDARD section 3): each is a
# StrUtils base; the public methods (signatures + docstrings) stay here for
# the flat ``ptk.<method>`` surface.
from pythontk.str_utils._name_pattern import _StrNamePatternInternal
from pythontk.str_utils._search import _StrSearchInternal
from pythontk.str_utils._affix import _StrAffixInternal


# ANSI/VT100 control sequences: CSI (``ESC [ … final-byte`` — covers SGR color) and the
# two-character escapes. Compiled at module scope because strip_ansi runs per console
# write, on a streaming hot path.
ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# The legal-name character class (see StrUtils.LEGAL_NAME_PATTERN), compiled once:
# a validating field runs it on every keystroke.
_LEGAL_NAME_RE = re.compile(r"[A-Za-z0-9_]+")
_ILLEGAL_NAME_CHAR_RE = re.compile(r"[^A-Za-z0-9_]")

# A run of decimal digits, captured so ``split`` keeps it (StrUtils.natural_sort_key).
_DIGIT_RUN_RE = re.compile(r"(\d+)")


class StrUtils(
    _StrNamePatternInternal,
    _StrSearchInternal,
    _StrAffixInternal,
    CoreUtils,
):
    """ """

    #: What a legal name may hold, as a whole-name pattern.  A name that
    #: becomes an identity somewhere else -- a Qt objectName, an exported clip,
    #: an FBX take -- is legal when every carrier keeps it verbatim, and this
    #: is that set.  :meth:`to_legal_name` converts TO it; :meth:`name_error`
    #: explains a name that breaks it, for a field that refuses rather than
    #: respells (uitk's ``set_validator("name")``).
    LEGAL_NAME_PATTERN = _LEGAL_NAME_RE.pattern
    #: :attr:`LEGAL_NAME_PATTERN` in words, for tooltips and refusals.
    LEGAL_NAME_RULE = "letters, digits and '_'"
    #: The cases :meth:`set_case` applies -- what a case picker offers.
    CASES = ("upper", "lower", "capitalize", "swapcase", "title", "pascal", "camel")
    #: The legal-name rules :meth:`apply_name_rule` applies, by name: ``legal``
    #: (:meth:`to_legal_name`), ``sanitized`` (:meth:`sanitize`, case kept)
    #: and ``filename`` (:meth:`to_legal_filename`).
    NAME_RULES = ("legal", "sanitized", "filename")

    @classmethod
    def apply_name_rule(cls, name: str, rule: str) -> str:
        """*name* made legal by the :attr:`NAME_RULES` rule called *rule* --
        for a caller that lets its user pick the rule by name.

        Parameters:
            name (str): The text to convert.
            rule (str): One of :attr:`NAME_RULES`.

        Returns:
            (str) The converted name.

        Raises:
            ValueError: *rule* is not one of :attr:`NAME_RULES`.

        Example:
            apply_name_rule("Red Door  Lock.", "legal") --> 'Red_Door__Lock_'
            apply_name_rule("Red Door  Lock.", "sanitized") --> 'Red_Door_Lock'
            apply_name_rule("Red Door: Lock", "filename") --> 'Red Door Lock'
        """
        if rule == "legal":
            return cls.to_legal_name(name)
        if rule == "sanitized":
            return cls.sanitize(name, preserve_case=True)
        if rule == "filename":
            return cls.to_legal_filename(name)
        raise ValueError(f"{rule!r} is not one of {list(cls.NAME_RULES)}")

    @staticmethod
    def is_legal_name(name) -> bool:
        """Whether *name* is a non-empty string of :attr:`LEGAL_NAME_PATTERN`
        characters only.

        Example:
            is_legal_name("Shot_01") --> True
            is_legal_name("Step 4.1") --> False
        """
        return isinstance(name, str) and bool(_LEGAL_NAME_RE.fullmatch(name))

    @staticmethod
    def illegal_name_chars(name: str) -> List[str]:
        """The distinct characters of *name* a legal name may not hold, in the
        order they first appear -- what a refusal names.

        Example:
            illegal_name_chars("Step 4.1-b") --> [' ', '.', '-']
        """
        return list(dict.fromkeys(_ILLEGAL_NAME_CHAR_RE.findall(name or "")))

    @classmethod
    def name_error(
        cls, name, subject: str = "names", reason: str = ""
    ) -> Optional[str]:
        """Why *name* is not a legal name, as one sentence, or ``None`` when it is.

        Explains and never repairs: the answer is for a field that keeps what
        the user typed and marks it refused (uitk's ``set_validator("name")``)
        or for a store that raises it -- the opposite of :meth:`to_legal_name`,
        which silently converts and is only right where the result is a key
        both sides derive the same way.

        Parameters:
            name: The candidate.  Anything but a non-empty string is refused.
            subject: What the names are, as the sentence names them after the
                colon ("shot names use letters, digits and '_' only").
            reason: Why the rule applies, appended as a clause
                ("because the name is the exported clip name").

        Returns:
            A sentence for the user, or ``None``.

        Example:
            name_error("Step 4.1") --> "'Step 4.1' has a space, '.': names use
            letters, digits and '_' only."
        """
        if not isinstance(name, str) or not name:
            return f"{subject[:1].upper()}{subject[1:]} cannot be empty."
        bad = cls.illegal_name_chars(name)
        if not bad:
            return None
        shown = ", ".join("a space" if c == " " else repr(c) for c in bad)
        tail = f", {reason}" if reason else ""
        return f"{name!r} has {shown}: {subject} use {cls.LEGAL_NAME_RULE} only{tail}."

    @staticmethod
    def legal_name_matcher(legal_name: str) -> "re.Pattern":
        """A pattern matching every name :meth:`to_legal_name` turns into
        *legal_name* -- the inverse lookup, for finding the original a legal
        identity was derived from (a ``.ui`` file behind a switchboard name).

        Each ``_`` may have been any non-alphanumeric character (an ``_``
        included); every other character matches itself.

        Example:
            legal_name_matcher("my_ui").fullmatch("my ui") --> <match>
        """
        return re.compile(
            "".join(
                "[^0-9a-zA-Z]" if char == "_" else re.escape(char)
                for char in legal_name
            )
        )

    @staticmethod
    def to_legal_name(name: str) -> str:
        """Every non-alphanumeric becomes ``_``: the objectName rule.

        Deliberately NOT :meth:`sanitize`, which is the other, lossier rule in
        this class: that one lowercases, collapses runs of the replacement
        char and strips trailing ones, so ``"a  b"`` comes back ``"a_b"``.
        This one is positional -- one character in, one out -- because the
        result is an IDENTITY two sides derive independently and must agree
        on: the widget uitk's switchboard builds from a definition
        (``SwitchboardNameMixin.convert_to_legal_name``) and the key a
        headless reader looks that widget up by (``ExportProfile.widget_key``).
        Collapsing would make ``"a  b"`` and ``"a b"`` the same objectName.

        It lives here, on the generic string class, rather than on either
        caller: uitk cannot own it (pythontk is downstream of nothing and
        cannot import it) and a Scene-Exporter class is the wrong owner for a
        rule the whole switchboard depends on.

        Parameters:
            name (str): The text to convert.

        Returns:
            (str) The name with every non-alphanumeric replaced by ``_``.

        Example:
            to_legal_name("Convert Textures (2K)") --> 'Convert_Textures__2K_'
        """
        return _ILLEGAL_NAME_CHAR_RE.sub("_", name)

    #: Characters no file name may carry on Windows, and the ones a POSIX tool
    #: is happiest without. Path separators are NOT here: a caller that joins a
    #: name into a directory owns that question, and one that does not has
    #: already decided a slash is legal input.
    ILLEGAL_FILENAME_CHARS = '<>:"|?*'

    @classmethod
    def to_legal_filename(cls, name: str, replacement: str = "", report: bool = False):
        """*name* with every character illegal in a file name removed.

        The permissive counterpart to :meth:`to_legal_name`: that one is an
        identity rule (every non-alphanumeric becomes ``_``, so two sides
        deriving a key agree), this one only removes what an OS will reject --
        a version token, a dash and a dot all survive, because a deliverable
        named ``asset-v2.hero`` is a legitimate file and a mangled one is not.

        Parameters:
            name (str): The candidate file name (no directory part).
            replacement (str): What each illegal character becomes. Empty drops it.
            report (bool): Also return the distinct illegal characters found, in
                the order they appear -- what a caller warns the user about.

        Returns:
            (str) the legal name, or ``(str, list)`` when *report*.

        Example:
            to_legal_filename('a:b?.fbx') --> 'ab.fbx'
        """
        found = []
        for char in name:
            if char in cls.ILLEGAL_FILENAME_CHARS and char not in found:
                found.append(char)
        legal = "".join(
            replacement if c in cls.ILLEGAL_FILENAME_CHARS else c for c in name
        )
        return (legal, found) if report else legal

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
    def expand_wildcard(text: str, key: str = "name", wildcard: str = "*") -> str:
        r"""Rewrite a bare-wildcard template into pure placeholder form.

        Sugar for the one wildcard a single-value template can mean: *stands in
        for the default value*. A blank template is the bare default, so both
        collapse to ``"{key}"`` and the caller is left with a single vocabulary
        -- ``{placeholders}`` -- to resolve (:meth:`replace_placeholders`) and to
        document (uitk's ``TooltipFormat.placeholder_preview``).

        Unlike :meth:`find_str_and_format`'s asterisk, this one is positional
        rather than a mode flag, so a prefix and a suffix compose in one pass.

        Parameters:
            text (str): The user-typed template.
            key (str): The placeholder name the wildcard stands for.
            wildcard (str): The token treated as the default-value marker.

        Only LITERAL text is rewritten. A *wildcard* inside a ``{token}`` belongs
        to that token's own grammar -- ``*`` is the regex "zero or more" in an
        inline modifier (:meth:`split_regex_modifier`) -- and expanding it there
        turns ``{name:_bar.*->}`` into ``{name:_bar.{name}->}``, silently
        breaking every regex that uses ``.*``, ``\d*`` or ``[a-z]*``.

        Returns:
            str: *text* with every *wildcard* in its literal runs replaced by
            ``"{key}"``; ``"{key}"`` when *text* is empty or whitespace.

        Example:
            >>> StrUtils.expand_wildcard("")             # -> '{name}'
            >>> StrUtils.expand_wildcard("*_export")     # -> '{name}_export'
            >>> StrUtils.expand_wildcard("WIP_*")        # -> 'WIP_{name}'
            >>> StrUtils.expand_wildcard("WIP_*_export") # -> 'WIP_{name}_export'
            >>> StrUtils.expand_wildcard("asset")        # -> 'asset'
            >>> StrUtils.expand_wildcard("{name:_bar.*->}")  # -> unchanged
        """
        return _StrNamePatternInternal.expand_wildcard(text, key, wildcard)

    #: Delimiters an inline regex modifier accepts, longest first -- the spec
    #: form a token uses to reshape its OWN value: ``{name:_bar.*->}``. A spec
    #: WITHOUT one of these stays an ordinary format spec (``{n:03d}``), so the
    #: two grammars cannot collide. ``|`` is deliberately not a delimiter: it is
    #: regex alternation, and splitting on it makes ``(foo|bar)->baz``
    #: unwritable.
    REGEX_MODIFIER_DELIMITERS = ("->", "=>")

    @classmethod
    def split_regex_modifier(cls, spec: str) -> Optional[Tuple[str, str]]:
        """``(pattern, replacement)`` when *spec* is a regex modifier, else ``None``.

        The ONE place the modifier's grammar is spelled. A spec with no
        delimiter is not a modifier (``None``), which is what lets a token carry
        either grammar: ``{n:03d}`` pads, ``{name:_bar.*->}`` substitutes.
        A delimiter with nothing after it deletes the match, the common case
        (``_bar.*->`` strips a suffix and everything past it).

        Parameters:
            spec (str): A token's format spec, as ``string.Formatter`` yields it.

        Returns:
            (tuple | None) ``(pattern, replacement)``, both stripped.

        Example:
            >>> StrUtils.split_regex_modifier("_bar.*->")
            ('_bar.*', '')
            >>> StrUtils.split_regex_modifier("03d") is None
            True
        """
        return super().split_regex_modifier(spec)

    @classmethod
    def apply_regex_modifier(cls, value, spec: str) -> Tuple[str, Optional[str]]:
        """*value* reshaped by the regex modifier *spec* -- ``(result, error)``.

        A pattern that will not compile is NOT raised: the text came from a user
        mid-edit, where half a regex is a keystroke rather than a reason to
        abort. The value comes back untouched with the error worded, so the
        caller reports it at its own severity (a tooltip greys it, an export
        logs it).

        Parameters:
            value: The token's value; a non-string is formatted first.
            spec (str): The modifier (see :meth:`split_regex_modifier`). A spec
                that is not a modifier leaves *value* untouched.

        Returns:
            (tuple) ``(result, error)`` -- *error* is ``None`` on success.

        Example:
            >>> StrUtils.apply_regex_modifier("asset_bar_old", "_bar.*->")
            ('asset', None)
            >>> StrUtils.apply_regex_modifier("asset", "(->")[0]
            'asset'
        """
        return super().apply_regex_modifier(value, spec)

    @classmethod
    def attach_modifier(cls, text: str, key: str, spec: str) -> str:
        """Give every bare ``{key}`` in *text* the regex modifier *spec*.

        The migration primitive: it folds a rule that lived in a SEPARATE field
        into the pattern itself, so retiring that field cannot silently drop
        what the user set.

        A ``{key}`` that already carries a spec is left untouched -- what the
        user wrote inline outranks a folded default -- and so is every other
        token. *text* must already be in placeholder form
        (:meth:`expand_wildcard`).

        Parameters:
            text (str): The pattern, in pure placeholder form.
            key (str): The token to modify.
            spec (str): The modifier (see :meth:`split_regex_modifier`).

        Returns:
            (str) *text* with the modifier attached.

        Example:
            >>> StrUtils.attach_modifier("WIP_{scene}", "scene", "_bar.*->")
            'WIP_{scene:_bar.*->}'
            >>> StrUtils.attach_modifier("{scene:^a->b}", "scene", "_bar.*->")
            '{scene:^a->b}'
        """
        return super().attach_modifier(text, key, spec)

    @staticmethod
    def replace_placeholders(text: str, **kwargs) -> str:
        """Replace placeholders in a string with provided values.

        Supports standard Python string formatting syntax (e.g. {value:03d})
        **and the inline regex modifier** -- a spec carrying a delimiter
        (:attr:`REGEX_MODIFIER_DELIMITERS`) reshapes the token's own value
        rather than formatting it: ``{name:_bar.*->}``. A modifier that will not
        compile leaves the value untouched; use :meth:`resolve_placeholders` to
        see WHY (this returns only the text).

        Missing keys are preserved as placeholders -- **including a POSITIONAL
        one** (``{}`` / ``{0}``), which has no value here because this takes only
        keywords. That used to raise ``IndexError`` out of the formatter, which
        is the wrong answer for the callers this exists for: every one of them
        resolves text a USER typed into a pattern field, where a stray brace pair
        is a typo, not a reason to abort an export. A positional field comes back
        as ``{N}`` (auto-numbered ``{}`` included -- Python resolves both to the
        same index before this sees them), so it stays visible in the result.

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
        return _StrNamePatternInternal.replace_placeholders(text, **kwargs)

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
                - ``"regex_errors"`` (list[tuple]): ``(spec, message)`` for every
                  inline regex modifier that would not compile. The value is
                  left untouched, so the result is still usable.

        Raises:
            ValueError: If *text* is a malformed format string (e.g. a lone ``{``),
                matching :meth:`replace_placeholders`.

        Example:
            >>> StrUtils.resolve_placeholders("{root}/{name}_{ver:03d}", root="C:/p", name="shot")
            {'result': 'C:/p/shot_{ver:03d}', 'fields': ['root', 'name', 'ver'], 'resolved': {'root': 'C:/p', 'name': 'shot'}, 'unresolved': ['ver']}
        """
        return _StrNamePatternInternal.resolve_placeholders(text, **kwargs)

    #: The tokens every name pattern gets for free -- token -> meaning, in the
    #: order a tooltip should list them. Universal by construction: nothing here
    #: needs a host, a scene or a file on disk, so a tool adds only the tokens it
    #: alone can value (see :meth:`name_pattern_context`).
    NAME_PATTERN_TOKENS = {
        "date": "YYYY-MM-DD",
        "time": "HH-MM-SS",
        "user": "OS username &mdash; embeds dev identity, so beware on shared output",
    }

    @staticmethod
    def name_pattern_context(**values) -> dict:
        """Live values for :attr:`NAME_PATTERN_TOKENS`, plus the caller's own.

        Sampled once per call so every token a single name resolves against
        reads the same instant -- a pattern using both ``{date}`` and ``{time}``
        cannot straddle midnight.

        Parameters:
            **values: The caller's host-specific tokens. A key given here wins,
                so a tool may override a universal token's value.

        Returns:
            (dict) token -> value, ready for :meth:`resolve_name_pattern`.
        """
        return _StrNamePatternInternal.name_pattern_context(**values)

    @classmethod
    def resolve_name_pattern(
        cls,
        pattern: str,
        context: dict = None,
        wildcard: str = "*",
        key: str = "name",
        keep: Iterable[str] = (),
    ) -> dict:
        """Resolve a user-typed name pattern into a file-name-legal name.

        The one grammar a *single output name* can carry, composed from this
        class's primitives: :meth:`expand_wildcard` (the bare wildcard, and a
        blank pattern, stand for the default value), :meth:`resolve_placeholders`
        (``{token}`` substitution, the inline regex modifier
        ``{token:PATTERN->REPLACEMENT}``, and what it could not fill) and
        :meth:`to_legal_filename`. Anything else in the pattern is literal.

        Diagnostics are *returned*, not logged, so the caller reports them
        through its own logger / UI at its own severity.

        Parameters:
            pattern (str): The user's text. Blank resolves to ``{key}``.
            context (dict): token -> value. ``None`` uses
                :meth:`name_pattern_context` (the universal tokens alone).
            wildcard (str): The bare token standing for *key*.
            key (str): The placeholder the wildcard stands for.
            keep (Iterable[str]): Placeholders a LATER stage owns -- a version
                counter resolved against a folder once the name is known, say.
                They are neither filled from *context* nor reported unresolved,
                and they survive the legalization verbatim, format spec
                included (the ``:`` in ``{n:03d}`` is illegal in a file name).

        Returns:
            (dict) with keys:
                - ``"name"`` (str): the resolved, file-name-legal result, with
                  any kept placeholder left in it verbatim.
                - ``"template"`` (str): ``name`` as a :meth:`str.format`
                  template -- literal braces doubled, kept placeholders intact
                  -- for the stage that owns them (``template.format(n=4)``),
                  where a brace a context value carries cannot read as a field.
                - ``"kept"`` (list): the *keep* placeholders the pattern uses,
                  in first-seen order.
                - ``"expanded"`` (str): *pattern* in pure placeholder form --
                  hand this to a tooltip preview so it resolves what the caller
                  resolved.
                - ``"unresolved"`` (list): tokens the context could not fill;
                  they are left in ``name`` verbatim, as typed.
                - ``"dropped"`` (list): characters removed as illegal in a file
                  name.
                - ``"regex_errors"`` (list): ``(spec, message)`` for every inline
                  regex modifier that would not compile; its token kept its
                  unmodified value.
                - ``"error"`` (str | None): set when *pattern* is not a valid
                  format string, in which case ``name`` is the pattern with its
                  braces taken literally -- still legalized, ``dropped`` saying
                  what went -- and nothing is kept.

        Example:
            >>> StrUtils.resolve_name_pattern("WIP_*", {"name": "myScene"})["name"]
            'WIP_myScene'
            >>> StrUtils.resolve_name_pattern("*_v{n:03d}", {"name": "s"}, keep=["n"])["template"]
            's_v{n:03d}'
        """
        return super().resolve_name_pattern(pattern, context, wildcard, key, keep)

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
    @CoreUtils.listify
    def set_case(string, case="title") -> Union[str, List[str]]:
        """Format the given string(s) in the given case.

        Parameters:
            string (str/list): The string(s) to format.
            case (str): The desired return case. Accepts all python case operators.
                    valid: :attr:`CASES` ('upper', 'lower', 'capitalize', 'swapcase', 'title' (default), 'pascal', 'camel'), or None.
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
    @CoreUtils.listify
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
            collapse_delimiter_runs('prop____Shape702') #returns: 'prop_Shape702'
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
    @CoreUtils.listify
    def truncate(
        string, length=75, mode="start", insert="..", head=None
    ) -> Union[str, List[str]]:
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
            truncate('O:/Cloud/proj/asset01/sourceimages/tex/x_DIFF.png', 36, 'path')
                #returns: 'O:/Cloud/proj/../tex/x_DIFF.png' (path mode)
            truncate('O:/Cloud/proj/asset01/sourceimages/tex/x_DIFF.png', 36, 'path', head=1)
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
        back with half-words at the seam (``O:/Cloud/Projects/proj/..ures/x.png``).
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
    def natural_sort_key(text: str, ignore_case: bool = False) -> Tuple:
        """Sort key that ranks embedded integers by value, not character by character.

        The text is split into alternating runs -- text, digits, text, ... --
        and each digit run compares as an int, so ``Cube2`` sorts before
        ``Cube10`` and ``4.10`` after ``4.9``. The key always starts with a text
        run (``""`` when the string starts with a digit), so two keys never
        compare an int against a str.

        Parameters:
            text: The string to key.
            ignore_case: Lower-case the text runs first (``alpha`` < ``Zeta``).

        Returns:
            (tuple) A hashable key: ``(text, int, text, ..., text)``.

        Example:
            sorted(["Cube10", "Cube2"], key=StrUtils.natural_sort_key)
            #returns: ['Cube2', 'Cube10']
        """
        parts = _DIGIT_RUN_RE.split(text.lower() if ignore_case else text)
        # re.split with one capture group puts every digit run at an odd index;
        # the parity (not str.isdigit, which also accepts superscripts that
        # int() rejects) is what marks a number.
        return tuple(int(p) if i % 2 else p for i, p in enumerate(parts))

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
        return super().find_str(find, strings, regex, ignore_case)

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
        return super().find_str_and_format(
            strings, to, fltr, regex, ignore_case, return_orig_strings
        )

    @staticmethod
    def strip_suffix(name: str, suffixes: Iterable[str]) -> str:
        """*name* without a trailing entry of *suffixes* (case-insensitive; one strip).

        The whitelist counterpart of ``os.path.splitext``: only a LISTED suffix
        comes off, so ``"asset.v2"`` keeps its dotted version token while
        ``"asset.FBX"`` loses its extension -- what an export name typed with
        the wrong format's extension needs (the deliverable's own is appended
        after). The longest matching suffix wins.

        Parameters:
            name (str): The name to strip.
            suffixes (Sequence[str]): The suffixes that count (``[".fbx", ".usd"]``).

        Returns:
            (str) ``name`` without the matched suffix, else ``name`` unchanged.

        Example:
            strip_suffix("asset.fbx", [".fbx", ".usd"]) #returns: 'asset'
            strip_suffix("asset.v2", [".fbx", ".usd"]) #returns: 'asset.v2'
        """
        return _StrAffixInternal.strip_suffix(name, suffixes)

    @staticmethod
    def retain_suffix(
        old_name: str, new_name: str, valid_suffixes: Optional[List[str]] = None
    ) -> str:
        """Carry ``old_name``'s trailing ``_TYPE`` suffix over to ``new_name``.

        The rename tools' "retain suffix" option: a renamed ``Asset_GRP1``
        keeps its ``_GRP`` no matter what the new pattern says. Trailing
        digits are ignored for the comparison (``_GRP1`` → ``_GRP``); a purely
        numeric trailing token (``_01``) is numbering, not a type suffix, and
        is never retained. ``new_name``'s own *recognized* suffix is replaced;
        an unrecognized one is kept and the old suffix appended after it.

        Retaining carries a *lost* suffix over; it never adds one. A ``new_name``
        that still holds the suffix as one of its own ``_`` tokens comes back
        untouched, numbering included -- without that the append patterns, which
        keep the whole old name by definition, doubled it (``Sphere_GEO`` under
        ``**_A`` produced ``Sphere_GEO_A_GEO``) and an auto-numbered
        ``pCube_GEO1`` lost the number that made it unique.

        Parameters:
            old_name (str): The name before the rename.
            new_name (str): The name the rename produced.
            valid_suffixes (list, optional): The suffixes that count as type
                suffixes (``["_GRP", "_GEO"]``). None treats any ``_token`` as
                one -- the DCC rename tools pass their naming convention, so
                only a *defined* suffix is ever carried over there.

        Returns:
            (str) ``new_name`` with the retained suffix.

        Example:
            retain_suffix('Asset_GRP1', 'NewAsset_LOC', ['_GRP', '_LOC']) #returns: 'NewAsset_GRP'
            retain_suffix('Part_GEO', 'Detail_HIGH', ['_GEO']) #returns: 'Detail_HIGH_GEO'
            retain_suffix('Part_GEO', 'Part_GEO_A', ['_GEO']) #returns: 'Part_GEO_A'
            retain_suffix('Screw_01', 'Bolt') #returns: 'Bolt'
        """
        return _StrAffixInternal.retain_suffix(old_name, new_name, valid_suffixes)

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
            strip (str/list): Suffix string(s) to strip from the END of the given
                string -- repeatedly, while it still ends with one. An entry that
                spells a regex (i.e. carries a metacharacter) is instead substituted
                out wherever it matches.
            strip_trailing_ints (bool): Strip all trailing integers.
            strip_trailing_alpha (bool): Strip all upper-case letters preceded by a non-alphanumeric character.

        Returns:
            (str): The formatted string.
        """
        return _StrAffixInternal.format_suffix(
            string, suffix, strip, strip_trailing_ints, strip_trailing_alpha
        )

    @staticmethod
    def strip_known_affix(
        string: str,
        prefix: str = "",
        suffix: str = "",
        *,
        case_sensitive: bool = False,
    ) -> str:
        """Strip a configured prefix and/or suffix from a string.

        Pure primitive: only literal affix matches (plus adjacent ``_`` separators)
        are removed. Leading/trailing underscores elsewhere in the string are
        preserved — callers that want a fully scrubbed name should chain
        ``.strip("_")`` themselves, or use :py:meth:`apply_affix`.

        Matching rules:
        - Case-insensitive on the affix core by default (underscores in the supplied
          affix are treated as separators, not part of the token). Pass
          ``case_sensitive=True`` when the vocabulary is a fixed-case table: folding
          case there buys nothing and costs real words (``_cam`` in ``security_cam``).
        - The match consumes any adjacent ``_`` runs on the affix side and tolerates
          stray ``_`` between the affix and the string boundary
          (``_Mat_brick`` with prefix ``Mat_`` → ``brick``).
        - Boundary required: the core must be followed by ``_`` or end-of-string
          (prefix) / preceded by ``_`` or start-of-string (suffix). This prevents
          false positives like ``Matte_door`` for prefix ``Mat_``.

        Parameters:
            string: The string to strip.
            prefix: Prefix to remove (e.g. ``"Mat_"``).
            suffix: Suffix to remove (e.g. ``"_MAT"``).
            case_sensitive: Require an exact-case match on the affix core.
                Defaults to False, preserving the released behaviour.

        Returns:
            The string with the configured affixes removed. If neither matches,
            returns the input unchanged.
        """
        return _StrAffixInternal.strip_known_affix(
            string, prefix, suffix, case_sensitive=case_sensitive
        )

    @staticmethod
    def strip_any_affix(
        string: str,
        known,
        *,
        exclude=(),
        one: bool = True,
        case_sensitive: bool = True,
    ) -> str:
        """Strip whichever affix in *known* the string carries, from either end.

        The companion to :py:meth:`apply_affix` for a name whose affix comes
        from a *table* rather than a single configured pair: the caller knows
        the whole vocabulary (every type suffix in a naming convention) but not
        which one this particular name is wearing — nor, once a convention can
        be spelled either way, which END it is wearing it on. ``body_GEO`` and
        ``GEO_body`` both yield ``body``, so re-running a rename after the
        convention flips sides corrects the name instead of doubling it.

        Longest-first, so ``"_LSG"`` is tested before ``"_SG"`` — otherwise the
        short token eats the tail of the long one and leaves ``"_L"`` behind.
        Equal-length tokens fall back to alphabetical, which matters more than
        it looks: ordering a *set* by length alone leaves ties to set-iteration
        order, and Python randomizes string hashing per process — so a name
        carrying two known affixes would have had a different one stripped from
        one session to the next.

        Parameters:
            string: The name to strip.
            known: Every affix spelling in the vocabulary. Order is irrelevant
                (this sorts); empty entries are ignored.
            exclude: Affixes to leave in place — normally the one about to be
                applied, so an already-correct name is not churned.
            one: Stop after the first affix that actually matched (the default,
                and what a type-suffix pass wants: a name carries one type
                marker, and stripping further eats real name content). ``False``
                keeps going through the whole vocabulary.
            case_sensitive: Match the vocabulary exactly as spelled. Defaults to
                True, unlike :py:meth:`strip_known_affix`, because a *table* of
                type markers is fixed-case by construction while the words a name
                is made of are not: folding case strips the ``cam`` out of
                ``security_cam`` for a ``_CAM`` entry, the ``set`` out of
                ``tile_set``, and the ``con`` out of ``con_rod``.

        Returns:
            The name with the matched affix (and its adjacent separator)
            removed, or unchanged when it carries none of them.

        Example:
            strip_any_affix("body_GEO", ["_GEO", "_MAT"])            # 'body'
            strip_any_affix("GEO_body", ["_GEO"])                    # 'body'
            strip_any_affix("body_GEO", ["_GEO"], exclude=["_GEO"])  # 'body_GEO'
            strip_any_affix("security_cam", ["_CAM"])                # 'security_cam'
        """
        return _StrAffixInternal.strip_any_affix(
            string, known, exclude=exclude, one=one, case_sensitive=case_sensitive
        )

    @staticmethod
    def common_name(
        paths,
        *,
        sep: str = "|",
        strip=(),
        max_length: int = 32,
    ) -> str:
        """A short name for what a set of hierarchy items is collectively called.

        The default a tool reaches for when the user names nothing -- an
        output named after its source instead of a generic tag:

        * **One item** -- its own name.
        * **Several** -- the leading name tokens they all share
          (``chair_leg`` + ``chair_seat`` -> ``chair``); failing that, the
          deepest group holding them all (``|kit|TABLE_LOC|TABLE`` +
          ``|kit|TABLE_LOC|MAT`` -> ``TABLE_LOC``); failing that, the first
          item's name.

        Each candidate drops one *strip* affix (a naming convention's type
        markers, so ``TABLE_LOC`` reads ``TABLE``) and any namespace. The
        result keeps only name-legal characters and is capped at *max_length*
        on a token boundary.

        Parameters:
            paths: Hierarchy paths (``|grp|mesh``) or plain names.
            sep: The path separator.
            strip: Affixes to drop -- e.g.
                ``pythontk.NamingConvention.all_affixes()``. A name that is
                nothing BUT an affix is kept as is.
            max_length: Longest result; trailing tokens go first.

        Returns:
            The name, or ``""`` when *paths* holds no names.

        Example:
            common_name(["|kit|TABLE_LOC|TABLE", "|kit|TABLE_LOC|MAT"], strip=["_LOC"])
            # 'TABLE'
        """
        import re

        chains = []
        for path in paths:
            parts = [t.rsplit(":", 1)[-1] for t in str(path).split(sep) if t]
            if parts:
                chains.append(parts)
        if not chains:
            return ""
        known = [a for a in strip if a]

        def clean(name):
            bare = StrUtils.strip_any_affix(name, known).strip("_") if known else name
            return bare or name

        leaves = [clean(c[-1]) for c in chains]
        name = leaves[0]
        if len(chains) > 1:
            shared = []
            for column in zip(*(leaf.split("_") for leaf in leaves)):
                if len(set(column)) != 1:
                    break
                shared.append(column[0])
            stem = "_".join(shared).strip("_")
            depth = 0
            for column in zip(*(c[:-1] for c in chains)):
                if len(set(column)) != 1:
                    break
                depth += 1
            if len(stem) >= 3:
                name = stem
            elif depth:
                name = clean(chains[0][depth - 1])
        name = re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_]", "_", name)).strip("_")
        if len(name) > max_length:
            tokens = name.split("_")
            while len(tokens) > 1 and len("_".join(tokens)) > max_length:
                tokens.pop()
            name = "_".join(tokens)[:max_length]
        return name

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
        return _StrAffixInternal.infer_affix_mode(text, delimiter, default=default)

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
        return _StrAffixInternal.split_affix(
            text, mode, default=default, delimiter=delimiter
        )

    @staticmethod
    def delimit_affix(
        text: str,
        mode: str = "suffix",
        *,
        delimiter: str = "_",
    ) -> str:
        """Give a bare affix token its boundary delimiter, on *mode*'s side.

        The forgiving counterpart to :py:meth:`apply_affix`, which concatenates
        VERBATIM — the separator is part of the affix, not something the applier
        inserts. That is the right contract for an engine (it must not guess),
        and the wrong one for a field a user types into: "MAT" with Prefix
        selected would otherwise produce ``"MATbrick"``. Any UI that accepts a
        free-typed affix runs it through here first.

        A token that already carries a delimiter at either edge is returned
        unchanged under ``"auto"`` — there, the delimiter IS the declaration.
        An explicit ``"prefix"``/``"suffix"`` outranks it and the delimiter is
        re-sided, because the picker beside a field is a later statement of intent
        than the spelling the field was pre-filled with: a convention row showing
        ``_GEO`` that the user flips to Prefix means ``GEO_``, not a suffix-spelled
        token stored under a prefix mode (which would then apply as ``_GEObody``).

        Parameters:
            text: The affix as typed. Empty returns empty.
            mode: ``"prefix"`` or ``"suffix"``. Anything else (including
                ``"auto"``, which by definition could not decide) falls back to
                ``"suffix"``, matching :py:meth:`split_affix`'s own default.
            delimiter: Boundary character. Empty disables the whole operation.

        Returns:
            The affix with a single boundary delimiter on the side *mode*
            selects.

        Example:
            delimit_affix("MAT", "prefix")   # 'MAT_'
            delimit_affix("MAT", "suffix")   # '_MAT'
            delimit_affix("_MAT", "prefix")  # 'MAT_'  (re-sided)
            delimit_affix("_MAT", "auto")    # '_MAT'  (already declared)
            delimit_affix("", "prefix")      # ''
        """
        return _StrAffixInternal.delimit_affix(text, mode, delimiter=delimiter)

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
        return _StrAffixInternal.apply_affix(string, prefix, suffix)

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
    @CoreUtils.listify
    def time_stamp(filepath, stamp="%m-%d-%Y  %H:%M") -> Union[str, List[str]]:
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
