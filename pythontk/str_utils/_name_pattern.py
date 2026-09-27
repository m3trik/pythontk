# !/usr/bin/python
# coding=utf-8
"""Name patterns: ``{token}`` placeholders, wildcard expansion and regex
modifiers composed into one resolved name (the bodies behind the
:class:`StrUtils` facade).
"""

from __future__ import annotations

import re
import string
from typing import Optional, Tuple, Iterable


class _SafeFormatter(string.Formatter):
    """The one formatter behind every ``{token}`` this class resolves.

    Two departures from ``string.Formatter``, both because the text it formats
    was typed by a USER into a pattern field rather than written by a
    programmer:

    - An unknown key is preserved verbatim (``{missing}``) instead of raising,
      so one stray brace pair is a typo rather than an aborted export.
    - A format spec carrying a regex delimiter
      (:attr:`StrUtils.REGEX_MODIFIER_DELIMITERS`) is a *regex modifier* on the
      token's own value, not a format spec: ``{name:_bar.*->}``. Errors are
      collected on :attr:`errors`, never raised -- the caller reports them at
      its own severity.
    """

    def __init__(self):
        super().__init__()
        #: ``[(spec, message), ...]`` for every modifier that would not compile.
        self.errors = []

    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            return kwargs.get(key, "{" + key + "}")
        return "{" + str(key) + "}"

    def format_field(self, value, format_spec):
        # Preserve unresolved placeholders verbatim, including their format
        # spec, so a second pass can still apply padding (or a modifier) later.
        # (`!r`/`!a` conversions on unresolved keys are not preserved.)
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            if format_spec:
                return value[:-1] + ":" + format_spec + "}"
            return value
        from pythontk.str_utils._str_utils import StrUtils

        if format_spec and StrUtils.split_regex_modifier(format_spec) is not None:
            result, error = StrUtils.apply_regex_modifier(value, format_spec)
            if error:
                self.errors.append((format_spec, error))
            return result
        return super().format_field(value, format_spec)


class _StrNamePatternInternal:
    """Bodies of :class:`StrUtils`' name patterns (a ``StrUtils`` base).

    The public signatures and docstrings stay on the facade; each delegates
    here.
    """

    @staticmethod
    def expand_wildcard(text: str, key: str = "name", wildcard: str = "*") -> str:
        """Body of :meth:`StrUtils.expand_wildcard`."""
        token = "{" + key + "}"
        text = (text or "").strip()
        if not text:
            return token
        if wildcard not in text:
            return text
        try:
            parsed = list(string.Formatter().parse(text))
        except ValueError:
            # Malformed (a lone brace) -- no token structure to respect, so fall
            # back to the blind rewrite and let the caller report the syntax.
            return text.replace(wildcard, token)
        out = []
        for literal, field, spec, conversion in parsed:
            # `parse` hands back literals already UNescaped, so re-double the
            # braces a user typed to mean one.
            out.append(
                literal.replace("{", "{{").replace("}", "}}").replace(wildcard, token)
            )
            if field is None:
                continue
            out.append(
                "{"
                + field
                + (f"!{conversion}" if conversion else "")
                + (f":{spec}" if spec else "")
                + "}"
            )
        return "".join(out)

    @classmethod
    def split_regex_modifier(cls, spec: str) -> Optional[Tuple[str, str]]:
        """Body of :meth:`StrUtils.split_regex_modifier`."""
        for delim in cls.REGEX_MODIFIER_DELIMITERS:
            if delim in spec:
                pattern, replacement = spec.split(delim, 1)
                return pattern.strip(), replacement.strip()
        return None

    @classmethod
    def apply_regex_modifier(cls, value, spec: str) -> Tuple[str, Optional[str]]:
        """Body of :meth:`StrUtils.apply_regex_modifier`."""
        text = value if isinstance(value, str) else format(value)
        parts = cls.split_regex_modifier(spec)
        if parts is None:
            return text, None
        pattern, replacement = parts
        try:
            return re.sub(pattern, replacement, text), None
        except re.error as e:
            return text, f"invalid regex {pattern!r}: {e}"

    @classmethod
    def attach_modifier(cls, text: str, key: str, spec: str) -> str:
        """Body of :meth:`StrUtils.attach_modifier`."""
        if not spec or not text:
            return text
        try:
            parsed = list(string.Formatter().parse(text))
        except ValueError:
            return text
        out = []
        for literal, field, field_spec, conversion in parsed:
            out.append(literal.replace("{", "{{").replace("}", "}}"))
            if field is None:
                continue
            base = field.split(".")[0].split("[")[0]
            if base == key and not field_spec:
                field_spec = spec
            out.append(
                "{"
                + field
                + (f"!{conversion}" if conversion else "")
                + (f":{field_spec}" if field_spec else "")
                + "}"
            )
        return "".join(out)

    @staticmethod
    def replace_placeholders(text: str, **kwargs) -> str:
        """Body of :meth:`StrUtils.replace_placeholders`."""
        return _SafeFormatter().format(text, **kwargs)

    @staticmethod
    def resolve_placeholders(text: str, **kwargs) -> dict:
        """Body of :meth:`StrUtils.resolve_placeholders`."""
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

        # Format through an OWNED formatter rather than replace_placeholders, so
        # the regex modifiers it applied can be reported rather than swallowed.
        formatter = _SafeFormatter()
        result = formatter.format(text, **kwargs)

        return {
            "result": result,
            "fields": fields,
            "resolved": resolved,
            "unresolved": unresolved,
            "regex_errors": formatter.errors,
        }

    @staticmethod
    def name_pattern_context(**values) -> dict:
        """Body of :meth:`StrUtils.name_pattern_context`."""
        import getpass
        from datetime import datetime

        try:
            user = getpass.getuser()
        except Exception:  # no pwd entry / no USER env (containers, services)
            user = ""
        now = datetime.now()
        return {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H-%M-%S"),
            "user": user,
            **values,
        }

    @classmethod
    def resolve_name_pattern(
        cls,
        pattern: str,
        context: dict = None,
        wildcard: str = "*",
        key: str = "name",
        keep: Iterable[str] = (),
    ) -> dict:
        """Body of :meth:`StrUtils.resolve_name_pattern`."""
        import string

        context = cls.name_pattern_context() if context is None else context
        expanded = cls.expand_wildcard(pattern, key=key, wildcard=wildcard)
        keep = set(keep or ())

        def escape(text: str) -> str:
            return text.replace("{", "{{").replace("}", "}}")

        # Split at the kept placeholders so each run between them resolves and
        # legalizes on its own while the kept ones pass through untouched. A run
        # is rebuilt as format source (literal braces re-doubled, fields as
        # written), so it resolves exactly as it would inside the whole pattern.
        runs, source, kept = [], [], []
        try:
            for literal, field, spec, conversion in string.Formatter().parse(expanded):
                source.append(escape(literal))
                if field is None:
                    continue
                text = (
                    "{"
                    + field
                    + (f"!{conversion}" if conversion else "")
                    + (f":{spec}" if spec else "")
                    + "}"
                )
                base = field.split(".")[0].split("[")[0]
                if base in keep:
                    runs.extend((("".join(source), None), (text, text)))
                    source = []
                    if base not in kept:
                        kept.append(base)
                else:
                    source.append(text)
            runs.append(("".join(source), None))
            runs = [
                (text, verbatim or cls.resolve_placeholders(text, **context))
                for text, verbatim in runs
            ]
        except ValueError as e:  # a malformed format string (a lone brace)
            # Its braces read as literal text, but its illegal characters are
            # still illegal: a name the OS refuses fails only at the write.
            legal, dropped = cls.to_legal_filename(name=expanded, report=True)
            return {
                "name": legal,
                "template": escape(legal),
                "kept": [],
                "expanded": expanded,
                "unresolved": [],
                "dropped": dropped,
                "regex_errors": [],
                "error": str(e),
            }
        name, template, unresolved, dropped, regex_errors = "", "", [], [], []
        for text, resolved in runs:
            if isinstance(resolved, str):  # a kept placeholder, verbatim
                name += resolved
                template += resolved
                continue
            legal, bad = cls.to_legal_filename(name=resolved["result"], report=True)
            name += legal
            template += escape(legal)
            unresolved += [n for n in resolved["unresolved"] if n not in unresolved]
            dropped += [c for c in bad if c not in dropped]
            regex_errors += [
                e for e in resolved["regex_errors"] if e not in regex_errors
            ]
        # A pattern that resolves to NOTHING is not a name -- a lone "?" drops to
        # empty, and the caller would join it onto a directory and write a file
        # that is only an extension. Fall back to the default the wildcard stands
        # for; `dropped` / `unresolved` already say why the typed one vanished.
        if not name:
            name = str(context.get(key, ""))
            template = escape(name)
        return {
            "name": name,
            "template": template,
            "kept": kept,
            "expanded": expanded,
            "unresolved": unresolved,
            "dropped": dropped,
            "regex_errors": regex_errors,
            "error": None,
        }
