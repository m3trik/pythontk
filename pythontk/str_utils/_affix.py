# !/usr/bin/python
# coding=utf-8
"""Prefixes and suffixes: stripping, retaining, formatting, inferring and
applying affixes (the bodies behind the :class:`StrUtils` facade).
"""

from __future__ import annotations

from typing import Union, List, Optional, Tuple, Iterable

from pythontk.iter_utils._iter_utils import IterUtils


class _StrAffixInternal:
    """Bodies of :class:`StrUtils`' prefix / suffix handling (a ``StrUtils`` base).

    The public signatures and docstrings stay on the facade; each delegates
    here.
    """

    @staticmethod
    def strip_suffix(name: str, suffixes: Iterable[str]) -> str:
        """Body of :meth:`StrUtils.strip_suffix`."""
        lowered = name.lower()
        for suffix in sorted(suffixes, key=len, reverse=True):
            if suffix and lowered.endswith(suffix.lower()):
                return name[: -len(suffix)]
        return name

    @staticmethod
    def retain_suffix(
        old_name: str, new_name: str, valid_suffixes: Optional[List[str]] = None
    ) -> str:
        """Body of :meth:`StrUtils.retain_suffix`."""
        digits = "0123456789"

        def trailing_token(name: str) -> str:
            """The name's trailing ``_TOKEN`` (digits ignored), else ``""``."""
            if "_" not in name:
                return ""
            token = name[name.rfind("_") :].rstrip(digits)
            return "" if token == "_" else token

        old_suffix = trailing_token(old_name)
        if not old_suffix:
            return new_name
        if valid_suffixes is not None and old_suffix not in valid_suffixes:
            return new_name
        # Still carried by the new name -- as one of its own '_' tokens, trailing
        # digits ignored -- so nothing was lost and there is nothing to retain.
        body = old_suffix[1:]
        if any(t.rstrip(digits) == body for t in new_name.split("_")[1:]):
            return new_name
        new_suffix = trailing_token(new_name)
        if new_suffix and (valid_suffixes is None or new_suffix in valid_suffixes):
            new_name = new_name[: new_name.rfind("_")]
        return new_name + old_suffix

    @staticmethod
    def format_suffix(
        string: str,
        suffix: str = "",
        strip: Union[str, List[str]] = "",
        strip_trailing_ints: bool = False,
        strip_trailing_alpha: bool = False,
    ) -> str:
        """Body of :meth:`StrUtils.format_suffix`."""
        import re

        def is_regex(pattern: str) -> bool:
            """True only for a token that SPELLS a pattern.

            Compiling is not a test of intent: every plain string compiles, so
            gating on that alone made every literal suffix a global substring
            match -- '_LOC' in the strip list turned 'ITA_LOCKHANDLE' into
            'ITAKHANDLE'. A metacharacter is required first.
            """
            if not re.search(r"[.^$*+?{}\[\]\\|()]", pattern):
                return False
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
                    # A literal token is a SUFFIX, not a substring: strip it only
                    # while the name still ENDS with it.
                    while pattern and s.endswith(pattern):
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
        *,
        case_sensitive: bool = False,
    ) -> str:
        """Body of :meth:`StrUtils.strip_known_affix`."""
        import re

        # The inline flag group is what folds case; an empty group leaves the
        # core matching exactly as written.
        fold = "" if case_sensitive else "?i:"

        s = string
        if prefix:
            core = prefix.strip("_")
            if core:
                s = re.sub(
                    rf"^_*({fold}{re.escape(core)})(?:_+|$)",
                    "",
                    s,
                )
        if suffix:
            core = suffix.strip("_")
            if core:
                s = re.sub(
                    rf"(?:_+|^)({fold}{re.escape(core)})_*$",
                    "",
                    s,
                )
        return s

    @staticmethod
    def strip_any_affix(
        string: str,
        known,
        *,
        exclude=(),
        one: bool = True,
        case_sensitive: bool = True,
    ) -> str:
        """Body of :meth:`StrUtils.strip_any_affix`."""
        skip = {a for a in exclude if a}
        core = string
        for token in sorted({a for a in known if a}, key=lambda a: (-len(a), a)):
            if token in skip:
                continue
            # Cheap necessary condition before the regex. ``strip_known_affix``
            # anchors the token's core at one edge (modulo delimiter runs), so a
            # name whose trimmed edges do not even begin or end with it cannot
            # match -- and a full type vocabulary is ~20 tokens, of which at
            # most one ever does. Skipping the two pattern builds for the other
            # nineteen is a measured 20x on a whole-scene pass; it cannot
            # produce a false negative, and a false positive (``GEOMETRY`` for
            # ``GEO``) still falls through to the regex, which rejects it.
            edge = token.strip("_")
            if not edge:
                continue
            trimmed = core.strip("_")
            if not case_sensitive:
                edge, trimmed = edge.lower(), trimmed.lower()
            if not (trimmed.startswith(edge) or trimmed.endswith(edge)):
                continue
            # prefix= and suffix= both: the vocabulary says WHAT the token is,
            # not which side this name wears it on.
            stripped = _StrAffixInternal.strip_known_affix(
                core, prefix=token, suffix=token, case_sensitive=case_sensitive
            )
            if stripped != core:
                core = stripped
                if one:
                    break
        return core

    @staticmethod
    def infer_affix_mode(
        text: str,
        delimiter: str = "_",
        *,
        default: str = "prefix",
    ) -> str:
        """Body of :meth:`StrUtils.infer_affix_mode`."""
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
        """Body of :meth:`StrUtils.split_affix`."""
        if not text:
            return ("", "")

        m = (mode or "auto").lower()
        if m not in ("prefix", "suffix", "auto"):
            m = "auto"
        if m == "auto":
            m = _StrAffixInternal.infer_affix_mode(
                text, delimiter=delimiter, default=default
            )

        if m == "prefix":
            return (text, "")
        return ("", text)

    @staticmethod
    def delimit_affix(
        text: str,
        mode: str = "suffix",
        *,
        delimiter: str = "_",
    ) -> str:
        """Body of :meth:`StrUtils.delimit_affix`."""
        text = (text or "").strip()
        if not text or not delimiter:
            return text

        mode = (mode or "").lower()
        if text.startswith(delimiter) or text.endswith(delimiter):
            if mode not in ("prefix", "suffix"):
                return text
            # Re-side rather than return: the caller named a side.
            text = text.strip(delimiter)
            if not text:
                return ""

        if mode == "prefix":
            return f"{text}{delimiter}"
        return f"{delimiter}{text}"

    @staticmethod
    def apply_affix(
        string: str,
        prefix: str = "",
        suffix: str = "",
    ) -> str:
        """Body of :meth:`StrUtils.apply_affix`."""
        if not prefix and not suffix:
            return string
        core = _StrAffixInternal.strip_known_affix(string, prefix=prefix, suffix=suffix)
        # The de-duplication strip must never consume the WHOLE name. A node
        # whose name IS its affix ("Cam" under a ``_CAM`` rule, or a literal
        # "GEO" under ``_GEO``) stripped to nothing and came back as the bare
        # affix, losing the name. Guarding on emptiness rather than on case is
        # what keeps the case-folded strip that normalises a sloppy lowercase
        # affix -- "body_geo" still becomes "body_GEO" instead of doubling.
        if not core.strip("_"):
            core = string
        if prefix:
            core = core.lstrip("_")
        if suffix:
            core = core.rstrip("_")
        return f"{prefix}{core}{suffix}"
