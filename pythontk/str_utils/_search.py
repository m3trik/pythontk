# !/usr/bin/python
# coding=utf-8
"""Searching strings by wildcard or regex terms, and search-and-format over a
list (the bodies behind the :class:`StrUtils` facade).
"""

from __future__ import annotations


class _StrSearchInternal:
    """Bodies of :class:`StrUtils`' wildcard / regex search (a ``StrUtils`` base).

    The public signatures and docstrings stay on the facade; each delegates
    here.
    """

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
        """Body of :meth:`StrUtils.find_str`."""
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
        """Body of :meth:`StrUtils.find_str_and_format`."""
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
                else:
                    # No filter text to drop -- plain prepend, mirroring
                    # 'replace_suffix' above (the Note in this docstring
                    # promises the fallback for both sides).
                    s = to_ + orig_str

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
