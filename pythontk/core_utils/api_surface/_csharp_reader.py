# !/usr/bin/python
# coding=utf-8
"""The C# half of :class:`~pythontk.ApiSurface`'s static read: a package's public C# API read from its source.

A tolerant reader, not a compiler: no editor, no ``dotnet``, nothing but the
text. It blanks every comment, string and preprocessor line in one copy of a
file (offsets kept), matches its brackets once, and walks the declarations
namespace by namespace, keeping each one's ``///`` comment. It reads through
``#if`` (both branches: a member declared in each is one), attributes,
generics nested to any depth, expression-bodied members and file-scoped
namespaces. Generated files (``*.g.cs``) are API like any other.

What it records is C#'s rule of the public surface:

* a type (``class``, ``struct``, ``interface``, ``enum``, ``record``) or a
  ``delegate`` declared ``public`` in a namespace, or ``public`` in a public
  type (in an interface, with no access modifier too). One with none is
  internal (nested: private), so neither it nor anything in it is surface;
* a public type's ``public`` members: methods, constructors, properties,
  indexers, fields, constants, events, operators. An interface's members are
  public unless marked otherwise, an enum's always;
* each one's XML documentation (``<summary>``, ``<remarks>``, ``<param>``,
  ``<typeparam>``, ``<returns>``, ``<exception>``), read as a Google
  docstring, and its ``[Obsolete]`` mark: retired, in the release its message
  says it is removed in (the C# pack's ``"Use <new>; removed in <version>."``,
  or ``"goes in unitytk <version>"``) and since the date it says it was
  deprecated (``"deprecated 2026-10-20"``), which the release gate holds to
  the window a Python retirement keeps.

The shape is the Python read's :class:`~pythontk.PackageSurface`: a module per
namespace (its ``relpath`` the namespace, :attr:`_CSharpReader.GLOBAL_NAMESPACE`
for the global one), a :class:`~pythontk.ClassSurface` per type (a nested one
named ``Outer.Inner``, its ``kind`` the keyword), a
:class:`~pythontk.SymbolRecord` per member and per delegate. A member's
``signature`` is its declaration as written, its attributes, comments and body
left out; its kind is ``method`` or ``staticmethod``, ``constructor``,
``property``, ``indexer``, ``field``, ``constant`` (an enum's members too),
``event``, ``operator`` or ``delegate``. Overloads of one name are told apart
by their parameter types (``Json.Write(object, bool)``). Plain attributes ride
beside the fields, as the Python read's ``doc`` and ``path`` do: a type's
``declaration``, and the ``call`` of a static member, a constructor or a
constant -- the C# that calls it (``return M3trik.Unitytk.Core.Json.Parse(text);``).

Where a package's C# is, it declares once, as data its root ``__init__.py``
holds (:attr:`_CSharpReader.CSHARP_DECLARATION`), read with ``ast``: nothing is
imported. One job of :class:`~pythontk.ApiSurface`, composed in
``api_surface.py``; every method is reached through the facade.

Adopted, not built: none fits. Roslyn reads C# exactly, but is .NET, which
pythontk cannot take (every host's Python imports it, and a CI box has no
SDK); tree-sitter's C# grammar is a compiled dependency; Doxygen and DocFX
write documents from a build, not records a search reads.
"""

from __future__ import annotations

import ast
import bisect
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from pythontk.core_utils.api_surface._source_reader import _Parses
from pythontk.core_utils.api_surface.package_surface import (
    ClassSurface,
    ModuleSurface,
    PackageSurface,
)
from pythontk.core_utils.symbol_record import SymbolRecord

#: ``(start, end)`` offsets of a span of a file.
_Span = Tuple[int, int]


class _CSharpText:
    """One C# file, read: its text, a copy with every comment, string, char and
    preprocessor line blanked (offsets and newlines kept, so brackets match in
    it), each bracket's partner, and its ``///`` lines."""

    #: What the blanking pass stops at: a comment, a string or a char, a directive.
    _SCAN = re.compile(r"//|/\*|[$@]{0,2}\"|'|^[ \t]*#", re.M)
    _STRING_OPEN = re.compile(r"[$@]{0,2}\"")
    _BRACKET = re.compile(r"[(){}\[\]]")
    _PAIRS = {")": "(", "]": "[", "}": "{"}

    def __init__(self, text: str) -> None:
        self.text = text
        #: ``(offset, text after the slashes)`` of each ``///`` line, in order.
        self.docs: List[Tuple[int, str]] = []
        self.comments: List[_Span] = []
        #: Each string and char literal, in order.
        self.strings: List[_Span] = []
        blanked: List[_Span] = []
        pos, n = 0, len(text)
        while True:
            m = self._SCAN.search(text, pos)
            if m is None:
                break
            start, token = m.start(), m.group()
            if token == "//":
                end = self._line_end(text, start)
                if text.startswith("///", start) and not text.startswith("////", start):
                    self.docs.append((start, text[start + 3 : end]))
                self.comments.append((start, end))
            elif token == "/*":
                end = text.find("*/", start + 2)
                end = n if end < 0 else end + 2
                self.comments.append((start, end))
            elif token.endswith('"'):
                end = self._string_end(
                    text, start + len(token) - 1, "@" in token, "$" in token
                )
                self.strings.append((start, end))
            elif token == "'":
                end = self._char_end(text, start)
                self.strings.append((start, end))
            else:  # a preprocessor line: read through it
                end = self._line_end(text, start)
            blanked.append((start, end))
            pos = max(end, start + 1)
        pieces, last = [], 0
        for start, end in blanked:
            pieces.append(text[last:start])
            pieces.append("\n".join(" " * len(s) for s in text[start:end].split("\n")))
            last = end
        pieces.append(text[last:])
        #: The text with every comment, literal and directive blanked.
        self.code = "".join(pieces)
        self.match = self._matched(self.code)
        self._lines = [0] + [m.end() for m in re.finditer("\n", text)]
        self._doc_offsets = [offset for offset, _ in self.docs]
        self._string_starts = [start for start, _ in self.strings]
        self._comment_starts = [start for start, _ in self.comments]

    # -- blanking ------------------------------------------------------------------
    @staticmethod
    def _line_end(text: str, start: int) -> int:
        end = text.find("\n", start)
        return len(text) if end < 0 else end

    @classmethod
    def _string_end(
        cls, text: str, quote: int, verbatim: bool, interpolated: bool
    ) -> int:
        """The offset after the string whose opening quote is at *quote*;
        an unclosed one ends with its line (verbatim: with the text)."""
        i, n = quote + 1, len(text)
        while i < n:
            c = text[i]
            if c == '"':
                if verbatim and text.startswith('""', i):
                    i += 2
                    continue
                return i + 1
            if c == "\\" and not verbatim:
                i += 2
                continue
            if c == "\n" and not verbatim:
                return i
            if interpolated and c in "{}":
                if text.startswith(c * 2, i):
                    i += 2
                    continue
                if c == "{":
                    i = cls._hole_end(text, i + 1)
                    continue
            i += 1
        return n

    @classmethod
    def _hole_end(cls, text: str, i: int) -> int:
        """The offset after the ``}`` closing an interpolation hole whose
        expression starts at *i*: nested brackets and literals skipped, a
        top-level ``:`` starting its format, which runs to the ``}``."""
        depth, n = 0, len(text)
        while i < n:
            c = text[i]
            if c in "([{":
                depth += 1
            elif c in ")]":
                depth -= 1
            elif c == "}":
                if depth <= 0:
                    return i + 1
                depth -= 1
            elif c == ":" and depth <= 0:
                end = text.find("}", i)
                return n if end < 0 else end + 1
            elif c == "'":
                i = cls._char_end(text, i)
                continue
            else:
                m = cls._STRING_OPEN.match(text, i)
                if m is not None:
                    token = m.group()
                    i = cls._string_end(text, m.end() - 1, "@" in token, "$" in token)
                    continue
            i += 1
        return n

    @staticmethod
    def _char_end(text: str, start: int) -> int:
        """The offset after the char literal at *start* (``'x'``, ``'\\''``,
        ``'\\u0041'``); a lone quote is one character."""
        i = start + 1
        if text.startswith("\\", i):
            i += 2
        else:
            i += 1
        close = text.find("'", i, i + 8)
        return start + 1 if close < 0 else close + 1

    @classmethod
    def _matched(cls, code: str) -> Dict[int, int]:
        """``{opening offset: closing offset}`` of every bracket, matched by
        kind; a closer with no opener is passed over, and an opener left open
        closes with what closes the bracket around it (or the text)."""
        match: Dict[int, int] = {}
        stack: List[Tuple[str, int]] = []
        for m in cls._BRACKET.finditer(code):
            c, i = m.group(), m.start()
            if c in "([{":
                stack.append((c, i))
                continue
            want = cls._PAIRS[c]
            for k in range(len(stack) - 1, -1, -1):
                if stack[k][0] == want:
                    for _, j in stack[k:]:
                        match[j] = i
                    del stack[k:]
                    break
        for _, j in stack:
            match[j] = len(code) - 1
        return match

    # -- reading ---------------------------------------------------------------------
    def line(self, offset: int) -> int:
        """The 1-based line *offset* is on."""
        return bisect.bisect_right(self._lines, offset)

    def docs_in(self, lo: int, hi: int) -> List[str]:
        """The ``///`` lines in ``[lo, hi)``, each without its slashes."""
        first = bisect.bisect_left(self._doc_offsets, lo)
        last = bisect.bisect_left(self._doc_offsets, hi)
        return [text for _, text in self.docs[first:last]]

    def strings_in(self, lo: int, hi: int) -> List[str]:
        """The source of each string literal in ``[lo, hi)``."""
        first = bisect.bisect_left(self._string_starts, lo)
        last = bisect.bisect_left(self._string_starts, hi)
        return [self.text[s:e] for s, e in self.strings[first:last]]

    def clean(self, lo: int, hi: int, drop: Iterable[_Span] = ()) -> str:
        """The source in ``[lo, hi)`` on one line: its comments and the *drop*
        spans (attributes) left out, whitespace collapsed outside literals."""
        first = bisect.bisect_left(self._comment_starts, lo)
        last = bisect.bisect_left(self._comment_starts, hi)
        cut = [(s, e, False) for s, e in self.comments[first:last]]
        cut += [(s, e, False) for s, e in drop]
        first = bisect.bisect_left(self._string_starts, lo)
        last = bisect.bisect_left(self._string_starts, hi)
        cut += [(s, e, True) for s, e in self.strings[first:last]]
        cut.sort()
        pieces, kept, pos = [], [], lo
        for s, e, keep in cut:
            if s < pos:
                continue  # inside a span already cut (a string in an attribute)
            pieces.append(self.text[pos:s])
            if keep:
                kept.append(self.text[s : min(e, hi)])
                pieces.append(f"\0{len(kept) - 1}\0")
            else:
                pieces.append(" ")
            pos = min(e, hi)
        pieces.append(self.text[pos:hi])
        out = " ".join("".join(pieces).split())
        out = re.sub(r"([(\[])\s+", r"\1", out)
        out = re.sub(r"\s+([)\],])", r"\1", out)
        return re.sub(r"\0(\d+)\0", lambda m: kept[int(m.group(1))], out)


class _CsMember:
    """One public member as a file declares it (a fragment of a type)."""

    __slots__ = (
        "name",
        "kind",
        "signature",
        "summary",
        "doc",
        "line",
        "deprecated",
        "remove_in",
        "since",
        "params",
        "call",
    )

    def __init__(self, **fields) -> None:
        for name in self.__slots__:
            setattr(self, name, fields.get(name, ""))


class _CsType:
    """One type as a file declares it: a partial type is several, merged."""

    __slots__ = (
        "ns",
        "qualname",
        "kind",
        "declaration",
        "bases",
        "summary",
        "doc",
        "line",
        "deprecated",
        "remove_in",
        "since",
        "access",
        "outer",
        "members",
    )

    def __init__(self, **fields) -> None:
        for name in self.__slots__:
            setattr(self, name, fields.get(name, ""))

    @property
    def fqn(self) -> str:
        """Its name as C# code spells it from anywhere (``Ns.Outer.Inner``)."""
        return f"{self.ns}.{self.qualname}" if self.ns else self.qualname

    @property
    def simple(self) -> str:
        """Its own name, without its type parameters (a constructor's name)."""
        return self.qualname.rsplit(".", 1)[-1].split("<", 1)[0]


class _CSharpReader:
    """Reads a package's public C# surface from its source (see the module docstring)."""

    #: The module name of the global namespace (``global::`` in C#).
    GLOBAL_NAMESPACE = "global"
    #: The name a package's root ``__init__.py`` declares its C# folders under:
    #: a string or a tuple of them, relative to the package's folder, the
    #: first that exists read (``("../Packages/com.x.y", "templates")``: a
    #: checkout's, then a wheel's).
    CSHARP_DECLARATION = "CSHARP_SOURCES"
    #: Folders a C# walk never enters, in any case: a test assembly's and the
    #: build's (Unity itself ignores hidden folders and those ending ``~``).
    CS_SKIP_DIR_NAMES = frozenset(
        {"tests", "test", "build", "obj", "bin", "library", "temp", "logs"}
    )

    _CS_TYPES = ("class", "struct", "interface", "enum", "record")
    _CS_ACCESS = frozenset({"public", "private", "protected", "internal", "file"})
    _CS_MODIFIERS = _CS_ACCESS | frozenset(
        {
            "static",
            "readonly",
            "sealed",
            "abstract",
            "virtual",
            "override",
            "new",
            "extern",
            "unsafe",
            "volatile",
            "async",
            "partial",
            "const",
            "event",
            "ref",
            "fixed",
            "required",
        }
    )
    #: Words a parameter list never follows: a call, a constraint, an initializer.
    _CS_NOT_A_NAME = frozenset(
        {"new", "base", "this", "where", "typeof", "sizeof", "default", "nameof"}
    )
    _CS_PARAM_MODIFIERS = ("this", "ref", "out", "in", "params", "scoped")
    _CS_ACCESSORS = ("get", "set", "init", "add", "remove")
    _CS_TOKEN = re.compile(r"@?[A-Za-z_]\w*|\S")
    _CS_IDENT = re.compile(r"@?[A-Za-z_]\w*")
    _CS_SPACE = re.compile(r"\S")
    _CS_HEADER_STOP = re.compile(r"[{};=(\[]")
    _CS_STATEMENT_STOP = re.compile(r"[;({\[}]")
    _CS_ACCESSOR_TOKEN = re.compile(r"[{(\[]|@?[A-Za-z_]\w*|;")
    _CS_OPERATOR = re.compile(r"\boperator\s*[!<>=]?\s*$")
    #: The release an ``[Obsolete]`` message says the name goes in: the C#
    #: pack's ``removed in 0.8.0``, or ``goes in unitytk 0.8.0``.
    _CS_REMOVE_IN = re.compile(
        r"\b(?:removed?|goes)\s+in\s+(?:[A-Za-z][\w.-]*\s+)?v?(\d+\.\d+(?:\.\d+)?)",
        re.I,
    )
    _CS_SINCE = re.compile(r"\b(?:deprecated|since)\s+(\d{4}-\d{2}-\d{2})\b", re.I)
    _CS_DOC_SECTION = re.compile(
        r"<(summary|remarks|returns|value)\b[^>]*>(.*?)</\1\s*>", re.S
    )
    _CS_DOC_NAMED = re.compile(
        r"<(param|typeparam|exception)\s+(?:name|cref)\s*=\s*[\"']([^\"']*)[\"']\s*>"
        r"(.*?)</\1\s*>",
        re.S,
    )

    #: ``{file: (its stamp, what it declares)}``: a re-read after an edit
    #: parses only the files it changed. Holds fragments, never a file's text.
    _CS_PARSED: Dict[str, Tuple[Tuple[int, int], tuple]] = {}
    #: ``{__init__.py: (its stamp, the C# folders it declares)}``.
    _CS_DECLARED: Dict[str, Tuple[Optional[Tuple[int, int]], Tuple[str, ...]]] = {}

    # -- where -------------------------------------------------------------------------
    @classmethod
    def _csharp_declared(cls, init: Path) -> Tuple[str, ...]:
        """The C# folders the ``__init__.py`` at *init* declares
        (:attr:`CSHARP_DECLARATION`, a literal read with ``ast``), as written;
        ``()`` when it declares none or there is no such file."""
        stamp = _Parses.stamp(init)
        kept = cls._CS_DECLARED.get(str(init))
        if kept is not None and kept[0] == stamp:
            return kept[1]
        found: Tuple[str, ...] = ()
        if stamp is not None:
            try:
                text = init.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ()  # kept nowhere: read again next time
            if cls.CSHARP_DECLARATION in text:
                found = cls._declared_in(text)
        cls._CS_DECLARED[str(init)] = (stamp, found)
        return found

    @classmethod
    def _declared_in(cls, text: str) -> Tuple[str, ...]:
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return ()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            if not any(
                isinstance(t, ast.Name) and t.id == cls.CSHARP_DECLARATION
                for t in targets
            ):
                continue
            try:
                declared = ast.literal_eval(value)
            except (ValueError, TypeError, SyntaxError):
                return ()
            if isinstance(declared, str):
                return (declared,)
            if isinstance(declared, (tuple, list)):
                return tuple(d for d in declared if isinstance(d, str))
            return ()
        return ()

    @classmethod
    def _iter_cs_files(cls, root: Path) -> Iterable[Path]:
        """Every ``.cs`` under *root* a walk reads, in POSIX-string order (as
        :meth:`_iter_py_files`): no test assembly's, nothing Unity ignores."""
        for path in sorted(root.rglob("*.cs"), key=lambda p: p.as_posix()):
            folders = path.relative_to(root).parts[:-1]
            if any(
                f.lower() in cls.CS_SKIP_DIR_NAMES
                or f.startswith(".")
                or f.endswith("~")
                for f in folders
            ):
                continue
            yield path

    # -- files -------------------------------------------------------------------------
    @classmethod
    def _cs_parse_kept(cls, path: Path) -> Optional[tuple]:
        """:meth:`_cs_parse_file`, or what it gave while the file is unchanged."""
        stamp = _Parses.stamp(path)
        kept = cls._CS_PARSED.get(str(path))
        if stamp is not None and kept is not None and kept[0] == stamp:
            return kept[1]
        parsed = cls._cs_parse_file(path)
        if parsed is not None and stamp is not None:
            cls._CS_PARSED[str(path)] = (stamp, parsed)
        return parsed

    @classmethod
    def _cs_parse_file(cls, path: Path) -> Optional[tuple]:
        """``(types, delegates)`` the file at *path* declares, each a fragment
        (:class:`_CsType`, :class:`_CsMember` with its namespace); ``None``
        when it cannot be read (a sync client's or a scanner's lock)."""
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            return None
        src = _CSharpText(text)
        types: List[_CsType] = []
        delegates: List[Tuple[str, _CsMember]] = []
        cls._cs_block(src, 0, len(text), "", None, types, delegates)
        return tuple(types), tuple(delegates)

    # -- declarations ------------------------------------------------------------------
    @classmethod
    def _cs_skip(cls, code: str, pos: int, hi: int) -> int:
        """The first offset at or after *pos* that is not whitespace (or *hi*)."""
        m = cls._CS_SPACE.search(code, pos, hi)
        return hi if m is None else m.start()

    @classmethod
    def _cs_block(cls, src, lo, hi, ns, owner, types, delegates) -> None:
        """Walk the declarations in ``[lo, hi)``: a namespace's or a type's body."""
        code, pos = src.code, lo
        while pos < hi:
            start = cls._cs_skip(code, pos, hi)
            if start >= hi:
                return
            if code[start] in ";}":
                pos = start + 1
                continue
            head, term = cls._cs_header_end(src, start, hi)
            end = cls._cs_declaration_end(src, head, term, hi)
            ns = cls._cs_declaration(
                src, pos, start, head, term, end, ns, owner, types, delegates
            )
            pos = max(end, start + 1)

    @classmethod
    def _cs_header_end(cls, src, start: int, hi: int) -> Tuple[int, str]:
        """Where the declaration at *start* stops being its header, and what
        stops it: ``{``, ``;``, ``=`` (an initializer), ``=>`` or a stray ``}``."""
        code, pos = src.code, start
        while True:
            m = cls._CS_HEADER_STOP.search(code, pos, hi)
            if m is None:
                return hi, ""
            j, c = m.start(), m.group()
            if c in "([":
                pos = src.match.get(j, hi - 1) + 1
                continue
            if c == "=":
                if code.startswith("=>", j):
                    return j, "=>"
                if (code.startswith("==", j) or code[j - 1] in "!<>") and (
                    cls._CS_OPERATOR.search(code, start, j)
                ):
                    pos = j + (2 if code.startswith("==", j) else 1)
                    continue  # ``operator ==``: its name, not an initializer
            return j, c

    @classmethod
    def _cs_statement_end(cls, src, pos: int, hi: int) -> int:
        """The offset after the ``;`` ending the statement at *pos*."""
        code = src.code
        while True:
            m = cls._CS_STATEMENT_STOP.search(code, pos, hi)
            if m is None:
                return hi
            k, c = m.start(), m.group()
            if c in "({[":
                pos = src.match.get(k, hi - 1) + 1
                continue
            return k if c == "}" else k + 1

    @classmethod
    def _cs_declaration_end(cls, src, head: int, term: str, hi: int) -> int:
        if term == ";":
            return head + 1
        if term in ("=", "=>"):
            return cls._cs_statement_end(src, head + len(term), hi)
        if term == "{":
            close = src.match.get(head, hi - 1)
            k = cls._cs_skip(src.code, close + 1, hi)
            if k < hi and src.code[k] == "=" and not src.code.startswith("=>", k):
                return cls._cs_statement_end(src, k + 1, hi)  # a property's initializer
            return k + 1 if k < hi and src.code[k] == ";" else close + 1
        return head if term == "}" else hi

    @classmethod
    def _cs_attributes(cls, src, start: int, head: int) -> Tuple[List[_Span], int]:
        """The attribute groups ``[...]`` the header at *start* opens with, and
        where what they mark begins."""
        spans, pos = [], start
        while pos < head and src.code[pos] == "[":
            close = min(src.match.get(pos, head - 1), head - 1)
            spans.append((pos, close + 1))
            pos = cls._cs_skip(src.code, close + 1, head)
        return spans, pos

    @classmethod
    def _cs_declaration(
        cls, src, gap, start, head, term, end, ns, owner, types, delegates
    ) -> str:
        """Record the declaration ``[start, end)`` (its header ``[start,
        head)``); returns the namespace what follows it is in (a file-scoped
        ``namespace X;`` changes it)."""
        code = src.code
        attrs, first = cls._cs_attributes(src, start, head)
        tokens = [
            (m.group(), m.start()) for m in cls._CS_TOKEN.finditer(code, first, head)
        ]
        if not tokens:
            return ns
        word = tokens[0][0]
        if word == "namespace":
            inner = "".join(t for t, _ in tokens[1:])
            inner = f"{ns}.{inner}" if ns else inner
            if term != "{":
                return inner  # file-scoped: the rest of the file
            body = src.match.get(head, end - 1)
            cls._cs_block(src, head + 1, body, inner, None, types, delegates)
            return ns
        if word in ("using", "extern") or (
            word == "global" and len(tokens) > 1 and tokens[1][0] == "using"
        ):
            return ns
        mods = []
        while len(mods) < len(tokens) and tokens[len(mods)][0] in cls._CS_MODIFIERS:
            mods.append(tokens[len(mods)][0])
        rest = tokens[len(mods) :]
        if not rest:
            return ns
        access = cls._cs_access(mods, owner)
        word = rest[0][0]
        if not (
            access == "public"
            or (access == "none" and "partial" in mods and word in cls._CS_TYPES)
        ):
            return ns  # never surface, nor anything in it
        marks = (cls._cs_doc(src.docs_in(gap, first)), cls._cs_obsolete(src, attrs))
        where = (first, head, term, end, attrs)
        if word == "delegate":
            if access == "public":
                member = cls._cs_callable(
                    src, where, mods, rest, owner, marks, "delegate"
                )
                if member is not None:
                    if owner is None:
                        delegates.append((ns, member))
                    else:
                        owner.members.append(member)
            return ns
        if word in cls._CS_TYPES:
            cls._cs_type(
                src, where, mods, rest, ns, owner, access, marks, types, delegates
            )
        elif owner is not None:
            owner.members.extend(cls._cs_members(src, where, mods, rest, owner, marks))
        return ns

    @classmethod
    def _cs_access(cls, mods: List[str], owner: Optional[_CsType]) -> str:
        """``public``, ``none`` (no access modifier: internal or private, an
        interface's member public) or ``other``."""
        access = [m for m in mods if m in cls._CS_ACCESS]
        if "public" in access:
            return "public"
        if access:
            return "other"
        return "public" if owner is not None and owner.kind == "interface" else "none"

    # -- types -------------------------------------------------------------------------
    @classmethod
    def _cs_type(
        cls, src, where, mods, rest, ns, owner, access, marks, types, delegates
    ):
        first, head, term, end, attrs = where
        code = src.code
        kind = rest[0][0]
        k = 1
        if kind == "record" and len(rest) > 1 and rest[1][0] in ("class", "struct"):
            k = 2
        if len(rest) <= k:
            return
        name_at = rest[k][1]
        name_end = cls._cs_after_type_params(code, name_at + len(rest[k][0]), head)
        name = " ".join(code[name_at:name_end].split())
        qualname = f"{owner.qualname}.{name}" if owner is not None else name
        pos = cls._cs_skip(code, name_end, head)
        positional = None
        if pos < head and code[pos] == "(":
            positional = (pos, src.match.get(pos, head - 1))
            pos = cls._cs_skip(code, positional[1] + 1, head)
        bases: List[str] = []
        if pos < head and code[pos] == ":":
            stop = cls._cs_where(code, pos + 1, head)
            bases = [
                " ".join(code[s:e].split())
                for s, e in cls._cs_split(src, pos + 1, stop)
                if code[s:e].strip()
            ]
            bases = [
                re.sub(r"\(.*$", "", b).strip() for b in bases
            ]  # a record's base call
        (summary, doc), (deprecated, remove_in, since) = marks
        entry = _CsType(
            ns=ns,
            qualname=qualname,
            kind=kind,
            declaration=src.clean(first, head, attrs),
            bases=bases,
            summary=summary,
            doc=doc,
            line=src.line(first),
            deprecated=deprecated,
            remove_in=remove_in,
            since=since,
            access=access,
            outer=(owner.ns, owner.qualname) if owner is not None else None,
            members=[],
        )
        types.append(entry)
        if positional is not None:
            entry.members.extend(cls._cs_positional(src, positional, entry))
        if term != "{":
            return
        body = src.match.get(head, end - 1)
        if kind == "enum":
            cls._cs_enum(src, head + 1, body, entry)
        else:
            cls._cs_block(src, head + 1, body, ns, entry, types, delegates)

    @staticmethod
    def _cs_after_type_params(code: str, pos: int, hi: int) -> int:
        """The offset after a ``<...>`` type parameter list at *pos* (spaces
        before it allowed), or *pos* when there is none."""
        k = pos
        while k < hi and code[k] in " \t\r\n":
            k += 1
        if k >= hi or code[k] != "<":
            return pos
        depth = 0
        for j in range(k, hi):
            if code[j] == "<":
                depth += 1
            elif code[j] == ">":
                depth -= 1
                if depth == 0:
                    return j + 1
        return pos

    @classmethod
    def _cs_where(cls, code: str, lo: int, hi: int) -> int:
        """Where a ``where`` constraint clause starts in ``[lo, hi)`` (or *hi*)."""
        for m in re.finditer(r"\bwhere\b", code[lo:hi]):
            before = code[lo : lo + m.start()]
            if before.count("<") == before.count(">") and before.count(
                "("
            ) == before.count(")"):
                return lo + m.start()
        return hi

    @classmethod
    def _cs_split(cls, src, lo: int, hi: int, defaults: bool = False) -> List[_Span]:
        """``[lo, hi)`` split at its top-level commas: brackets skipped, and
        angle brackets counted (in a parameter, only before its ``=`` when
        *defaults*: a default value is no type)."""
        code, spans, start, k, depth, valued = src.code, [], lo, lo, 0, False
        while k < hi:
            c = code[k]
            if c in "([{":
                k = src.match.get(k, hi - 1) + 1
                continue
            if c == "<" and not valued:
                depth += 1
            elif c == ">" and not valued and depth:
                depth -= 1
            elif c == "=" and defaults and depth == 0:
                valued = True
            elif c == "," and depth == 0:
                spans.append((start, k))
                start, valued = k + 1, False
            k += 1
        spans.append((start, hi))
        return spans

    @classmethod
    def _cs_enum(cls, src, lo: int, hi: int, owner: _CsType) -> None:
        """An enum's members: every one public, each a constant."""
        code, gap = src.code, lo
        for s, e in cls._cs_split(src, lo, hi):
            start = cls._cs_skip(code, s, e)
            if start >= e:
                continue
            attrs, first = cls._cs_attributes(src, start, e)
            m = cls._CS_IDENT.match(code, first)
            if m is None:
                continue
            summary, doc = cls._cs_doc(src.docs_in(gap, first))
            deprecated, remove_in, since = cls._cs_obsolete(src, attrs)
            owner.members.append(
                _CsMember(
                    name=m.group().lstrip("@"),
                    kind="constant",
                    signature=src.clean(first, e),
                    summary=summary,
                    doc=doc,
                    line=src.line(first),
                    deprecated=deprecated,
                    remove_in=remove_in,
                    since=since,
                    params=(),
                    call=f"return {owner.fqn}.{m.group()};",
                )
            )
            gap = e

    @classmethod
    def _cs_positional(cls, src, group: _Span, owner: _CsType) -> List[_CsMember]:
        """A positional record's primary constructor and the property each of
        its parameters is."""
        params = cls._cs_params(src, *group)
        args = ", ".join(cls._cs_argument(p) for p in params)
        out = [
            _CsMember(
                name=owner.simple,
                kind="constructor",
                signature=f"public {owner.simple}{src.clean(group[0], group[1] + 1)}",
                summary="",
                doc="",
                line=src.line(group[0]),
                deprecated=False,
                params=tuple(p[3] for p in params),
                call=f"return new {owner.fqn}({args});",
            )
        ]
        for param in params:
            out.append(
                _CsMember(
                    name=param[2],
                    kind="property",
                    signature=f"public {param[1]} {param[2]} {{ get; init; }}",
                    summary="",
                    doc="",
                    line=src.line(group[0]),
                    deprecated=False,
                    params=(),
                )
            )
        return out

    # -- members -----------------------------------------------------------------------
    @classmethod
    def _cs_members(cls, src, where, mods, rest, owner, marks) -> List[_CsMember]:
        """The public member (or the several fields one statement declares)
        at *where*, a member of *owner*."""
        first, head, term, end, attrs = where
        code = src.code
        words = [t for t, _ in rest]
        group = cls._cs_param_group(src, first, head, words)
        if group is not None:
            member = cls._cs_callable(src, where, mods, rest, owner, marks, None, group)
            return [] if member is None else [member]
        (summary, doc), (deprecated, remove_in, since) = marks
        common = dict(
            summary=summary,
            doc=doc,
            deprecated=deprecated,
            remove_in=remove_in,
            since=since,
            params=(),
        )
        static = "static" in mods or "const" in mods
        if "this" in words:
            at = rest[words.index("this")][1]
            k = cls._cs_skip(code, at + 4, head)
            if k < head and code[k] == "[":
                params = cls._cs_params(src, k, src.match.get(k, head - 1))
                signature = src.clean(first, head, attrs) + cls._cs_accessors(
                    src, head, term
                )
                return [
                    _CsMember(
                        name="this",
                        kind="indexer",
                        signature=signature,
                        line=src.line(first),
                        **{**common, "params": tuple(p[3] for p in params)},
                    )
                ]
        if term in ("{", "=>"):
            names = [t for t, _ in rest if cls._CS_IDENT.fullmatch(t)]
            if not names:
                return []
            name = names[-1]
            kind = "event" if "event" in mods else "property"
            signature = src.clean(first, head, attrs)
            if kind == "property":
                signature += cls._cs_accessors(src, head, term)
            call = (
                f"return {owner.fqn}.{name};" if static and kind == "property" else ""
            )
            return [
                _CsMember(
                    name=name.lstrip("@"),
                    kind=kind,
                    signature=signature,
                    line=src.line(first),
                    call=call,
                    **common,
                )
            ]
        return cls._cs_fields(src, where, mods, owner, common)

    @classmethod
    def _cs_param_group(
        cls, src, lo: int, hi: int, words: List[str]
    ) -> Optional[_Span]:
        """The parameter list of the callable whose header is ``[lo, hi)``:
        the first top-level ``(`` a name (or ``operator`` and its symbol)
        precedes; ``None`` when it is no callable."""
        code = src.code
        operator = "operator" in words
        pos = lo
        while True:
            k = code.find("(", pos, hi)
            if k < 0:
                return None
            bracket = code.find("[", pos, k)
            if bracket >= 0:
                pos = src.match.get(bracket, hi - 1) + 1
                continue
            close = src.match.get(k, hi - 1)
            before = code[lo:k].rstrip()
            if operator:
                if re.search(r"\boperator\b", before):
                    return k, close
            else:
                if before.endswith(">"):
                    before = before[: cls._cs_open_angle(before)].rstrip()
                m = re.search(r"@?[A-Za-z_]\w*$", before)
                if (
                    m is not None
                    and m.group() not in cls._CS_NOT_A_NAME | cls._CS_MODIFIERS
                ):
                    return k, close
            pos = close + 1

    @staticmethod
    def _cs_open_angle(text: str) -> int:
        """The offset of the ``<`` matching the ``>`` *text* ends with."""
        depth = 0
        for j in range(len(text) - 1, -1, -1):
            if text[j] == ">":
                depth += 1
            elif text[j] == "<":
                depth -= 1
                if depth == 0:
                    return j
        return len(text)

    @classmethod
    def _cs_callable(cls, src, where, mods, rest, owner, marks, kind, group=None):
        """A method, constructor, operator or delegate: its name, signature
        (to the end of its parameters and constraints; a constructor's
        initializer left out) and call."""
        first, head, term, end, attrs = where
        code = src.code
        words = [t for t, _ in rest]
        if group is None:
            group = cls._cs_param_group(src, first, head, words)
            if group is None:
                return None
        open_, close = group
        before = code[rest[0][1] : open_].rstrip()
        type_params = ""
        if "operator" in words:
            at = rest[words.index("operator")][1]
            lead = (
                rest[words.index("operator") - 1] if words.index("operator") else None
            )
            if lead is not None and lead[0] in ("implicit", "explicit"):
                at = lead[1]
            name = " ".join(code[at:open_].split())
            returns, kind = "", "operator"
        else:
            if before.endswith(">"):
                cut = cls._cs_open_angle(before)
                type_params = " ".join(before[cut:].split())
                before = before[:cut].rstrip()
            m = re.search(r"@?[A-Za-z_]\w*$", before)
            if m is None or before[: m.start()].rstrip().endswith("~"):
                return None  # a finalizer: nothing calls it
            name = m.group()
            returns = before[: m.start()].strip()
            if kind == "delegate":
                returns = returns[len("delegate") :].strip()
                name += type_params
            elif not returns and owner is not None and name == owner.simple:
                kind = "constructor"
            else:
                kind = "staticmethod" if "static" in mods else "method"
        params = cls._cs_params(src, open_, close)
        # To its constraints (``where T : class``); a constructor's initializer
        # (``: base(...)``) is its body's, not its signature.
        stop = close + 1
        if kind != "constructor" and cls._cs_where(code, close + 1, head) < head:
            stop = head
        drop = list(attrs) + [a for p in params for a in p[4]]
        signature = src.clean(first, stop, drop)
        args = ", ".join(cls._cs_argument(p) for p in params)
        call = ""
        if kind == "constructor" and owner is not None:
            call = f"return new {owner.fqn}({args});"
        elif kind == "staticmethod" and owner is not None:
            spelled = f"{owner.fqn}.{name}{type_params}({args})"
            call = f"{spelled};" if returns == "void" else f"return {spelled};"
        (summary, doc), (deprecated, remove_in, since) = marks
        return _CsMember(
            name=name.lstrip("@"),
            kind=kind,
            signature=signature,
            summary=summary,
            doc=doc,
            line=src.line(first),
            deprecated=deprecated,
            remove_in=remove_in,
            since=since,
            params=tuple(p[3] for p in params),
            call=call,
        )

    @classmethod
    def _cs_params(cls, src, open_: int, close: int) -> List[tuple]:
        """Each parameter between the brackets at *open_* and *close*:
        ``(modifiers, type, name, its overload spelling, its attribute spans)``."""
        code, out = src.code, []
        if not code[open_ + 1 : close].strip():
            return out
        for s, e in cls._cs_split(src, open_ + 1, close, defaults=True):
            start = cls._cs_skip(code, s, e)
            attrs, first = cls._cs_attributes(src, start, e)
            stop = code.find("=", first, e)
            stop = e if stop < 0 else stop
            tokens = [
                (m.group(), m.start())
                for m in cls._CS_TOKEN.finditer(code, first, stop)
            ]
            mods = []
            while (
                len(mods) < len(tokens) - 1
                and tokens[len(mods)][0] in cls._CS_PARAM_MODIFIERS
            ):
                mods.append(tokens[len(mods)][0])
            names = [(t, p) for t, p in tokens if cls._CS_IDENT.fullmatch(t)]
            if not names:
                continue
            name, at = names[-1]
            lo = tokens[len(mods)][1] if len(mods) < len(tokens) else at
            kind = " ".join(code[lo:at].split())
            spelled = " ".join([m for m in mods if m != "this"] + [kind])
            out.append((tuple(mods), kind, name, spelled, attrs))
        return out

    @staticmethod
    def _cs_argument(param: tuple) -> str:
        """How a call passes *param*: by its name (``out var x``, ``ref x``)."""
        mods, _, name = param[0], param[1], param[2]
        if "out" in mods:
            return f"out var {name}"
        if "ref" in mods:
            return f"ref {name}"
        return name

    @classmethod
    def _cs_accessors(cls, src, head: int, term: str) -> str:
        """`` { get; set; }`` for a property or indexer whose header ends at
        *head*: its accessors, as written (``private set``)."""
        if term == "=>":
            return " { get; }"
        if term != "{":
            return ""
        code, close = src.code, src.match.get(head, head)
        found, pending, pos = [], [], head + 1
        while True:
            m = cls._CS_ACCESSOR_TOKEN.search(code, pos, close)
            if m is None:
                break
            token, pos = m.group(), m.end()
            if token in "{([":
                pos = src.match.get(m.start(), close) + 1
            elif token in cls._CS_ACCESS:
                pending.append(token)
                continue
            elif token in cls._CS_ACCESSORS:
                found.append(" ".join(pending + [token]))
            pending = []
        return f" {{ {'; '.join(found)}; }}" if found else ""

    @classmethod
    def _cs_fields(cls, src, where, mods, owner, common) -> List[_CsMember]:
        """The fields, constants or events one statement declares (``public
        string status, error;``), each its own record."""
        first, head, term, end, attrs = where
        code = src.code
        stop = end - 1 if end > first and code[end - 1] == ";" else end
        spans: List[_Span] = []
        for s, e in cls._cs_split(src, first, stop, defaults=True):
            if spans and not re.match(r"\s*@?[A-Za-z_]\w*\s*(=|$)", code[s:e]):
                spans[-1] = (spans[-1][0], e)  # a comma inside an initializer
            else:
                spans.append((s, e))
        kind = (
            "event" if "event" in mods else "constant" if "const" in mods else "field"
        )
        static = "static" in mods or kind == "constant"
        out, lead = [], ""
        for index, (s, e) in enumerate(spans):
            eq = code.find("=", s, e)
            decl_end = e if eq < 0 else eq
            m = re.search(r"@?[A-Za-z_]\w*\s*$", code[s:decl_end])
            if m is None:
                continue
            name = m.group().strip()
            at = s + m.start()
            if index == 0:
                lead = src.clean(first, at, attrs)
            signature = f"{lead} {name}"
            if kind == "constant" and eq >= 0:
                signature += f" = {src.clean(eq + 1, e)}"
            out.append(
                _CsMember(
                    name=name.lstrip("@"),
                    kind=kind,
                    signature=signature,
                    line=src.line(at if index else first),
                    call=f"return {owner.fqn}.{name};"
                    if static and kind != "event"
                    else "",
                    **common,
                )
            )
        return out

    # -- documentation and marks -------------------------------------------------------
    @classmethod
    def _cs_obsolete(cls, src, attrs: List[_Span]) -> Tuple[bool, str, str]:
        """``(deprecated, remove_in, since)`` read off ``[Obsolete("...")]``
        among *attrs*: the release and date its message names, if it names them."""
        code = src.code
        for s, e in attrs:
            for a, b in cls._cs_split(src, s + 1, e - 1):
                m = re.match(r"\s*(?:\w+\s*:\s*)?([\w.]+)", code[a:b])
                if m is None or m.group(1).rsplit(".", 1)[-1] not in (
                    "Obsolete",
                    "ObsoleteAttribute",
                ):
                    continue
                message = "".join(cls._cs_literal(lit) for lit in src.strings_in(a, b))
                remove_in = cls._CS_REMOVE_IN.search(message)
                since = cls._CS_SINCE.search(message)
                return (
                    True,
                    remove_in.group(1) if remove_in else "",
                    since.group(1) if since else "",
                )
        return False, "", ""

    @staticmethod
    def _cs_literal(source: str) -> str:
        """The value a C# string literal's *source* spells."""
        if source.startswith("@"):
            return source[2:-1].replace('""', '"')
        if not source.startswith('"'):
            return ""
        escapes = {"n": "\n", "t": "\t", "r": "\r", "0": "\0"}
        return re.sub(
            r"\\(.)", lambda m: escapes.get(m.group(1), m.group(1)), source[1:-1]
        )

    @classmethod
    def _cs_doc(cls, lines: List[str]) -> Tuple[str, str]:
        """``(summary, doc)`` of an XML doc comment's *lines*: the summary as
        ``ApiSurface.summary`` keeps it, the whole as a Google docstring."""
        raw = "\n".join(line[1:] if line.startswith(" ") else line for line in lines)
        if not raw.strip():
            return "", ""
        if not re.search(r"<\w+", raw):
            text = cls._cs_text(raw)
            return cls.summary(text, cls.SUMMARY_MAX), text
        sections: Dict[str, str] = {}
        for m in cls._CS_DOC_SECTION.finditer(raw):
            sections.setdefault(m.group(1), m.group(2))
        named: Dict[str, List[Tuple[str, str]]] = {}
        for m in cls._CS_DOC_NAMED.finditer(raw):
            named.setdefault(m.group(1), []).append(
                (cls._cs_cref(m.group(2)), cls._cs_text(m.group(3), one_line=True))
            )
        summary = cls._cs_text(sections.get("summary", ""))
        parts = [p for p in (summary, cls._cs_text(sections.get("remarks", ""))) if p]
        for title, key in (("Parameters", "param"), ("Type Parameters", "typeparam")):
            if named.get(key):
                rows = "\n".join(f"    {n}: {t}" for n, t in named[key])
                parts.append(f"{title}:\n{rows}")
        returns = cls._cs_text(
            sections.get("returns") or sections.get("value") or "", True
        )
        if returns:
            parts.append(f"Returns:\n    {returns}")
        if named.get("exception"):
            rows = "\n".join(f"    {n}: {t}" for n, t in named["exception"])
            parts.append(f"Raises:\n{rows}")
        return cls.summary(summary, cls.SUMMARY_MAX), "\n\n".join(parts)

    @staticmethod
    def _cs_cref(cref: str) -> str:
        """A ``cref`` as code names it: no ``T:`` prefix, generics in angles."""
        cref = re.sub(r"^[A-Z]:", "", cref.strip())
        return cref.replace("{", "<").replace("}", ">")

    @classmethod
    def _cs_text(cls, xml: str, one_line: bool = False) -> str:
        """An XML doc fragment as Markdown: code and references in backticks,
        ``<para>`` a paragraph, entities decoded; its paragraphs' lines joined."""
        import html  # here: every class that mixes HelpMixin in imports this module

        if not xml:
            return ""
        s = re.sub(
            r"<(?:see|seealso)\s+(?:cref|langword)\s*=\s*[\"']([^\"']*)[\"']\s*/>",
            lambda m: f"`{cls._cs_cref(m.group(1))}`",
            xml,
        )
        s = re.sub(r"<(see|seealso|a)\b[^>]*>(.*?)</\1\s*>", r"\2", s, flags=re.S)
        s = re.sub(r"<see\s+href\s*=\s*[\"']([^\"']*)[\"']\s*/>", r"\1", s)
        s = re.sub(
            r"<(?:paramref|typeparamref)\s+name\s*=\s*[\"']([^\"']*)[\"']\s*/>",
            r"`\1`",
            s,
        )
        s = re.sub(
            r"<(c|code)>(.*?)</\1\s*>",
            lambda m: f"`{' '.join(m.group(2).split())}`",
            s,
            flags=re.S,
        )
        s = re.sub(r"</?para\s*/?>|<br\s*/?>", "\n\n", s)
        s = re.sub(r"<item>", "\n\n- ", s)
        s = html.unescape(re.sub(r"<[^>]+>", "", s))
        paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", s)]
        paragraphs = [p for p in paragraphs if p]
        return (" " if one_line else "\n\n").join(paragraphs)

    # -- the package -------------------------------------------------------------------
    @classmethod
    def _cs_walk(
        cls, source_root: Path, label: str, files: List[Path], keep: bool = True
    ) -> Tuple[PackageSurface, List[Path]]:
        """The C# surface under *source_root*, recorded under *label*, from
        *files*, and the files that could not be read. *keep* parses again
        only the files an edit changed; False parses every one, remembering none."""
        parse = cls._cs_parse_kept if keep else cls._cs_parse_file
        unread: List[Path] = []
        merged: Dict[Tuple[str, str], List[Tuple[Path, _CsType]]] = {}
        delegates: Dict[str, List[Tuple[Path, _CsMember]]] = {}
        for path in files:
            parsed = parse(path)
            if parsed is None:
                unread.append(path)
                continue
            for entry in parsed[0]:
                merged.setdefault((entry.ns, entry.qualname), []).append((path, entry))
            for ns, member in parsed[1]:
                delegates.setdefault(ns, []).append((path, member))

        def surface(key) -> bool:
            parts = merged.get(key)
            if not parts or not any(e.access == "public" for _, e in parts):
                return False
            outer = parts[0][1].outer
            return outer is None or surface(outer)

        classes: Dict[str, List[ClassSurface]] = {}
        for key, parts in merged.items():
            if surface(key):
                classes.setdefault(key[0], []).append(cls._cs_class(parts))
        modules = []
        for ns in sorted(
            set(classes) | set(delegates), key=lambda n: n or cls.GLOBAL_NAMESPACE
        ):
            functions = cls._cs_records("", delegates.get(ns, []))
            modules.append(
                ModuleSurface(
                    relpath=ns or cls.GLOBAL_NAMESPACE,
                    summary="",
                    functions=functions,
                    classes=classes.get(ns, []),
                )
            )
        return (
            PackageSurface(name=source_root.name, source_root=label, modules=modules),
            unread,
        )

    @classmethod
    def _cs_class(cls, parts: List[Tuple[Path, _CsType]]) -> ClassSurface:
        """One type from its fragments: documented by the first that has a doc."""
        path, lead = next(((p, e) for p, e in parts if e.doc), parts[0])
        bases: List[str] = []
        for _, part in parts:
            bases += [b for b in part.bases if b not in bases]
        deprecated = next((e for _, e in parts if e.deprecated), None)
        entry = ClassSurface(
            name=lead.qualname,
            summary=lead.summary,
            line=lead.line,
            bases=bases,
            members=cls._cs_records(
                lead.qualname, [(p, m) for p, e in parts for m in e.members]
            ),
            deprecated=deprecated is not None,
            remove_in=deprecated.remove_in if deprecated is not None else "",
            kind=lead.kind,
        )
        entry.doc, entry.path, entry.declaration = lead.doc, str(path), lead.declaration
        if deprecated is not None:
            entry.since = deprecated.since
        return entry

    @classmethod
    def _cs_records(
        cls, owner: str, members: List[Tuple[Path, _CsMember]]
    ) -> List[SymbolRecord]:
        """*owner*'s members as records, in source order: one declared twice
        alike (``#if`` branches) is one, an overloaded name's each named by
        its parameter types."""
        unique: List[Tuple[Path, _CsMember]] = []
        seen = set()
        for path, member in members:
            if (member.name, member.signature) not in seen:
                seen.add((member.name, member.signature))
                unique.append((path, member))
        count: Dict[str, int] = {}
        for _, member in unique:
            count[member.name] = count.get(member.name, 0) + 1
        out, named = [], set()
        prefix = f"{owner}." if owner else ""
        for path, member in unique:
            qualname = prefix + member.name
            if count[member.name] > 1:
                qualname += f"({', '.join(member.params)})"
            if qualname in named:
                continue  # two overloads whose parameters read alike: the first stands
            named.add(qualname)
            record = SymbolRecord(
                name=member.name,
                qualname=qualname,
                kind=member.kind,
                signature=member.signature,
                summary=member.summary,
                line=member.line,
                deprecated=bool(member.deprecated),
                remove_in=member.remove_in if member.deprecated else "",
            )
            record.doc, record.path = member.doc, str(path)
            if member.call:
                record.call = member.call
            if member.deprecated:
                record.since = member.since
            out.append(record)
        return out
