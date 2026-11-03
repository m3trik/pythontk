# !/usr/bin/python
# coding=utf-8
"""ApiSurface -- a package's public API, read from its source or from live objects, searched and described.

The one engine every reader of the public API goes through, so a symbol reads
the same wherever it is asked about:

* **static** (:meth:`ApiSurface.read`): a package's whole surface from its
  source, importing nothing (:mod:`._source_reader`). :meth:`ApiSurface.search`
  and :meth:`ApiSurface.describe` answer from it (the ``api.search`` and
  ``api.describe`` operations), and ``m3trik/scripts/check_api_surface.py``
  checks it against the last release.
* **live** (:meth:`ApiSurface.members`, :meth:`ApiSurface.record`, ...): a
  loaded class's members as Python resolves them -- its MRO, members a
  metaclass or mixin injects, unwrapped signatures. ``HelpMixin`` renders it
  (``.help()``, ``python -m pythontk <target>``); the runtime-vs-static drift
  gate compares it with the static read by name and kind.
* one docstring summary rule (:meth:`ApiSurface.summary`) for both, and for
  every catalog and search line.

Both produce :class:`~pythontk.SymbolRecord`; the static read groups them as a
:class:`~pythontk.PackageSurface`. A package registers with pythontk under the
entry-point groups named here (:attr:`ApiSurface.PROVIDER_GROUP`,
:attr:`ApiSurface.HOST_GROUP`): :meth:`ApiSurface.registered` lists them and
every package installed editable, so a search with no packages named covers
every codebase onboarded and all the code being worked on.

It imports nothing beyond the standard library at module level, since
``HelpMixin`` -- which nearly every class mixes in -- imports it.

Adopted, not built: the standard library's ``ast`` reads the source, and a
read is kept as JSON in :meth:`UserConfig.cache_dir
<pythontk.UserConfig.cache_dir>`. Weighed: griffe reads a package's API
statically and well, but is a dependency, and pythontk takes none (every
host's Python imports it); ``pydoc`` and ``inspect`` read only what is
imported, and a host's library imports only in its host; ``platformdirs`` and
a JSON store are what ``UserConfig.cache_dir`` already is.
``FileUtils.replace_file`` waits out a reader's hold on a file the same way
:meth:`ApiSurface._replace` does, but sits above this module in pythontk's
layer order.
"""

from __future__ import annotations

import difflib
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from pythontk.core_utils.api_surface._source_reader import _SourceReader
from pythontk.core_utils.api_surface.package_surface import ClassSurface, PackageSurface
from pythontk.core_utils.symbol_record import SymbolRecord

_LOG = logging.getLogger(__name__)


class _ApiSurfaceInternal(_SourceReader):
    """Internal helpers for :class:`ApiSurface`: text, packages and live members."""

    #: ``{(source folder, label): (its source stamp, its surface)}``.
    _READS: Dict[Tuple[str, str], Tuple[Tuple, PackageSurface]] = {}
    #: One read of a package at a time, by the same key: a second asker waits
    #: for it rather than parsing the package again.
    _READ_LOCKS: Dict[Tuple[str, str], threading.Lock] = {}
    _READ_LOCKS_GUARD = threading.Lock()
    #: Days a read kept on disk (:meth:`_cache`) lives unread before a write sweeps it.
    CACHE_DAYS = 14
    #: Minutes a ``.part`` (a write a killed process left) lives before a write sweeps it.
    _PART_MINUTES = 10
    #: Seconds between the tries of a rename Windows refuses while a reader
    #: holds the entry (WinError 5 or 32), then one last try.
    _REPLACE_PAUSES = (0.05, 0.1, 0.2, 0.4)
    _MOMENTARY_LOCKS = (5, 32)
    #: What rides a record or a class beside its fields, which a read
    #: kept on disk carries too (``describe`` reads ``doc`` and ``path``).
    _EXTRAS = ("doc", "path", "since")
    #: This process's :meth:`_reader_key`, worked out once.
    _READER_KEY: Optional[str] = None
    #: The member kinds each of ``HelpMixin``'s filter words keeps.
    _KIND_FILTERS = {
        "methods": ("method", "classmethod", "staticmethod"),
        "properties": ("property",),
        "classmethods": ("classmethod",),
        "staticmethods": ("staticmethod",),
    }
    #: Where one sentence ends and the next starts (a capital, a digit, code),
    #: so "e.g. a material" and "i.e. the header" read on.
    _SENTENCE_END = re.compile(r"(?<=\.)(?<!\be\.g\.)(?<!\bi\.e\.)\s+(?=[A-Z0-9`*])")
    #: The kinds a static record can be called as.
    _CALLABLE_KINDS = ("function", "method", "staticmethod", "classmethod")

    # -- packages --------------------------------------------------------------------
    @staticmethod
    def _top(reference: str) -> str:
        """The top-level package a ``module[:name]`` reference is in."""
        return reference.split(":", 1)[0].split(".", 1)[0]

    @classmethod
    def _source_root(cls, package: Union[str, os.PathLike]) -> Path:
        """The source folder *package* names: a path as given, a name located
        without importing it (:meth:`ApiSurface.locate`).

        Raises:
            FileNotFoundError: A path that is no folder.
            ValueError: A name no package here answers to.
        """
        if isinstance(package, os.PathLike) or os.sep in package or "/" in package:
            source = Path(package)
            if not source.is_dir():
                raise FileNotFoundError(
                    f"Expected package source root at {source}, not found."
                )
            return source
        return cls.locate(package)

    @classmethod
    def _rows(cls, package: str) -> Iterator[Tuple[str, Any]]:
        """``(target, symbol)`` for every symbol of *package* a caller can
        import by its dotted path (a plug-in tree another application loads is
        left out)."""
        source = cls._source_root(package)
        importable: Dict[
            str, bool
        ] = {}  # per module, not per symbol: thousands of stats saved
        for module, symbol in cls.read(package).symbols():
            if module.relpath not in importable:
                importable[module.relpath] = cls._importable(source, module.relpath)
            if not importable[module.relpath]:
                continue
            dotted = ".".join(p for p in (package, module.module) if p)
            yield f"{dotted}:{cls._qualname(symbol)}", symbol

    @classmethod
    def _read_lock(cls, key: Tuple[str, str]) -> threading.Lock:
        with cls._READ_LOCKS_GUARD:
            return cls._READ_LOCKS.setdefault(key, threading.Lock())

    @classmethod
    def _reader_key(cls) -> str:
        """What says a read kept on disk is this reader's: the reader's own
        source files (this folder and the record), as they are now."""
        if cls._READER_KEY is None:
            here = Path(__file__).resolve().parent
            files = sorted(here.glob("*.py")) + [here.parent / "symbol_record.py"]
            stamp = cls._source_stamp(here.parent, [f for f in files if f.is_file()])
            cls._READER_KEY = hashlib.sha1(repr(stamp).encode("utf-8")).hexdigest()
        return cls._READER_KEY

    @staticmethod
    def _cache_file(source: Path, label: str) -> Path:
        """Where a read of *source* (as *label*) is kept between processes:
        one entry per Python version, since Maya's, Blender's and the system's
        Python share the user cache and a file one cannot parse another can."""
        from pythontk.core_utils.user_config import UserConfig

        python = "{}.{}".format(*sys.version_info[:2])
        key = f"{os.path.normcase(str(source))}|{label}|{python}"
        name = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
        return UserConfig.cache_dir("api_surface") / f"{name}.json"

    @staticmethod
    def _keeps_on_disk(source: Path) -> bool:
        """Whether a read of *source* is kept in the per-user cache: not one of
        a tree under a temp folder (the system's, or the ``$PYTHONTK_TEMP_ROOT``
        that :meth:`TempArtifacts.root <pythontk.TempArtifacts.root>` honours) --
        a test's sandbox, a release copied out -- which is gone long before its
        entry would be swept. Such a read is kept in the process alone."""
        here = os.path.normcase(os.path.realpath(source))
        for folder in (tempfile.gettempdir(), os.environ.get("PYTHONTK_TEMP_ROOT")):
            if not folder:
                continue
            folder = os.path.expandvars(os.path.expanduser(folder))
            base = os.path.normcase(os.path.realpath(folder)).rstrip("\\/")
            if here == base or here.startswith(base + os.sep):
                return False
        return True

    @classmethod
    def _cached(
        cls, source: Path, label: str, stamp: Tuple
    ) -> Optional[PackageSurface]:
        """The read an earlier process kept of *source* as it is now (*stamp*),
        by this reader; ``None`` when there is none. A hit refreshes the
        entry's time, so the sweep keeps it; one it cannot refresh is a hit."""
        path = cls._cache_file(source, label)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if (
                data.get("reader") != cls._reader_key()
                or data.get("stamp")
                != hashlib.sha1(repr(stamp).encode("utf-8")).hexdigest()
            ):
                return None
            surface = PackageSurface.from_dict(data["surface"])
            symbols = [symbol for _, symbol in surface.symbols()]
            if len(symbols) != len(data["extras"]):
                return None
            for symbol, extras in zip(symbols, data["extras"]):
                for attr, value in extras.items():
                    setattr(symbol, attr, value)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None
        try:
            os.utime(path)  # read: not stale
        except OSError:
            pass  # unrefreshed, the entry ages toward a sweep; the read stands
        return surface

    @classmethod
    def _cache(
        cls, source: Path, label: str, stamp: Tuple, surface: PackageSurface
    ) -> None:
        """Keep *surface* for the next process, written whole or not at all (a
        ``.part`` renamed onto the entry, removed whatever happens), then sweep
        the folder (:meth:`_sweep`). A cache that cannot be written fails no read."""
        path = cls._cache_file(source, label)
        extras = [
            {
                attr: getattr(symbol, attr)
                for attr in cls._EXTRAS
                if hasattr(symbol, attr)
            }
            for _, symbol in surface.symbols()
        ]
        data = {
            "source": str(source),
            "label": label,
            "reader": cls._reader_key(),
            "stamp": hashlib.sha1(repr(stamp).encode("utf-8")).hexdigest(),
            "surface": surface.to_dict(),
            "extras": extras,
        }
        partial = path.with_name(
            f"{path.stem}.{os.getpid()}.{threading.get_ident()}.part"
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(json.dumps(data, separators=(",", ":")).encode("utf-8"))
            cls._replace(partial, path)
        except OSError:
            pass  # a cache that cannot be written fails no read
        finally:
            try:
                partial.unlink()
            except OSError:
                pass  # renamed onto the entry (or held: the sweep takes it)
        cls._sweep(path.parent)

    @classmethod
    def _replace(cls, partial: Path, path: Path) -> None:
        """``os.replace(partial, path)``, waiting out a reader's brief hold on
        *path*: Windows refuses to rename onto a file another process has open.

        Raises:
            OSError: What the last try raised, or at once anything but a
                momentary lock.
        """
        for pause in cls._REPLACE_PAUSES + (None,):
            try:
                os.replace(partial, path)
                return
            except PermissionError as exc:
                if (
                    pause is None
                    or getattr(exc, "winerror", None) not in cls._MOMENTARY_LOCKS
                ):
                    raise
            time.sleep(pause)

    @classmethod
    def _sweep(cls, folder: Path) -> None:
        """Remove each entry in *folder* left unread :attr:`CACHE_DAYS` and each
        ``.part`` older than :attr:`_PART_MINUTES` (a write a killed process
        left). A file another process swept first, or holds, is passed over."""
        limits = {".json": cls.CACHE_DAYS * 86400, ".part": cls._PART_MINUTES * 60}
        now = time.time()
        try:
            entries = list(folder.iterdir())
        except OSError:
            return
        for kept in entries:
            limit = limits.get(kept.suffix)
            if limit is None:
                continue
            try:
                if now - kept.stat().st_mtime > limit:
                    kept.unlink()
            except OSError:
                continue

    @staticmethod
    def _qualname(symbol: Any) -> str:
        return symbol.name if isinstance(symbol, ClassSurface) else symbol.qualname

    @classmethod
    def _kind(cls, symbol: Any) -> str:
        return "class" if isinstance(symbol, ClassSurface) else symbol.kind

    @classmethod
    def _spelled(cls, symbol: Any) -> str:
        """How a reader calls *symbol*: ``Owner.name(args) -> result``, ``class
        Name(bases)``, or the bare name of a property or constant."""
        if isinstance(symbol, ClassSurface):
            return (
                f"class {symbol.name}({', '.join(symbol.bases)})"
                if symbol.bases
                else f"class {symbol.name}"
            )
        if symbol.kind in cls._CALLABLE_KINDS:
            return f"{symbol.qualname}{symbol.signature}"
        return symbol.qualname

    @classmethod
    def _found(cls, target: str, symbol: Any, host: Optional[str]) -> Dict[str, Any]:
        """The row a search or a description gives for *symbol*."""
        row = {
            "target": target,
            "kind": cls._kind(symbol),
            "signature": cls._spelled(symbol),
            "summary": symbol.summary,
        }
        if host:
            row["host"] = host
        if symbol.deprecated:
            row["deprecated"] = True
            if symbol.remove_in:
                row["remove_in"] = symbol.remove_in
        return row

    # -- live members --------------------------------------------------------------
    @staticmethod
    def _member_kind(owner: type, name: str, member: Any) -> str:
        """*member*'s kind on *owner*, read off the raw descriptor of whichever
        class in the MRO defines it: an inherited property or static method
        keeps its kind (the classes here are composed from capability mixins,
        so most descriptors are inherited)."""
        raw = next(
            (k.__dict__[name] for k in inspect.getmro(owner) if name in k.__dict__),
            None,
        )
        if isinstance(raw, classmethod):
            return "classmethod"
        if isinstance(raw, staticmethod):
            return "staticmethod"
        if isinstance(raw, property):
            return "property"
        return "method" if callable(member) else "attribute"

    @classmethod
    def _member_items(
        cls, owner: type, kind: Optional[str], inherited: bool, private: bool
    ) -> List[Tuple[str, Any, str]]:
        """``(name, member, kind)`` for each member of *owner* the filters keep."""
        own = set(owner.__dict__) if not inherited else None
        wanted = cls._KIND_FILTERS.get(kind) if kind not in (None, "all") else None
        found = []
        for name in dir(owner):
            if not private and name.startswith("_"):
                continue
            if own is not None and name not in own:
                continue
            try:
                member = getattr(owner, name)
            except AttributeError:
                continue
            member_kind = cls._member_kind(owner, name, member)
            if wanted is not None and member_kind not in wanted:
                continue
            found.append((name, member, member_kind))
        return found

    @classmethod
    def _record_of(cls, owner: type, name: str, member: Any, kind: str) -> SymbolRecord:
        doc = inspect.getdoc(member)
        deprecated, remove_in = cls._deprecation(member)
        return SymbolRecord(
            name=name,
            qualname=f"{owner.__name__}.{name}",
            kind=kind,
            signature=cls.signature_of(member, kind),
            summary=cls.summary(doc) if doc else "",
            line=cls._line(member),
            deprecated=deprecated,
            remove_in=remove_in,
        )

    @staticmethod
    def _line(member: Any) -> int:
        """*member*'s 1-based definition line, or ``0`` (a builtin)."""
        try:
            return inspect.getsourcelines(inspect.unwrap(member))[1]
        except (OSError, TypeError, ValueError):
            return 0

    @staticmethod
    def _deprecation_marker(member: Any) -> Any:
        """*member*, or what it unwraps to, whichever carries a ``__deprecated__``
        marker: ``Deprecation`` stamps the wrapper and what it wraps, but a
        member may be wrapped again (``CoreUtils.undoable``), and a property's
        marker is on whichever accessor was decorated."""
        if getattr(member, "__deprecated__", False):
            return member
        if isinstance(member, property):
            for accessor in (member.fget, member.fset, member.fdel):
                if accessor is not None and getattr(accessor, "__deprecated__", False):
                    return accessor
            return member
        try:
            return inspect.unwrap(member) if callable(member) else member
        except Exception:
            return member

    @classmethod
    def _deprecation(cls, member: Any) -> Tuple[bool, str]:
        """``(deprecated, remove_in)`` for a live member: PEP 702's
        ``__deprecated__`` marker (``Deprecation``'s and the stdlib's), and the
        release its ``DeprecationRecord`` names (duck-typed: this module sits
        below ``deprecation`` in the import order)."""
        marker = cls._deprecation_marker(member)
        record = getattr(marker, "__deprecated_record__", None)
        return bool(getattr(marker, "__deprecated__", False)), str(
            getattr(record, "remove_in", "") or ""
        )


class ApiSurface(_ApiSurfaceInternal):
    """A package's public API, read statically or live, searched and described (see the module docstring)."""

    #: The entry-point group a package registers its operations module under.
    PROVIDER_GROUP = "pythontk.operations"
    #: The entry-point group an application's library registers its runner
    #: under: the package whose functions run in that application.
    HOST_GROUP = "pythontk.operation_hosts"

    # -- text ----------------------------------------------------------------------
    @staticmethod
    def markdown(text: str) -> str:
        """*text*'s reST code spelled as Markdown, which is what an agent, a
        palette or a search result reads: a role's or a literal's text in single
        backticks (``:class:`~pkg.Name``` reads ```pkg.Name```)."""
        return re.sub(r":[\w:]+:`~?([^`]+)`", r"`\1`", text).replace("``", "`")

    @classmethod
    def summary(cls, doc: Optional[str], limit: Optional[int] = None) -> str:
        """A docstring's summary: the first sentence of its first paragraph,
        its lines joined (a summary sentence often wraps), as Markdown.

        Parameters:
            doc: The docstring (raw or cleaned); ``None`` reads as empty.
            limit: The most characters to keep; a longer summary is cut and
                ends ``…``. Default: all of it.

        Returns:
            The summary, or ``""`` when there is no docstring.
        """
        if not doc or not doc.strip():
            return ""
        paragraph = " ".join(re.split(r"\n\s*\n", doc.strip(), maxsplit=1)[0].split())
        text = cls._SENTENCE_END.split(cls.markdown(paragraph), maxsplit=1)[0]
        if limit is not None and len(text) > limit:
            text = text[: limit - 1].rstrip() + "…"
        return text

    # -- packages ------------------------------------------------------------------
    @classmethod
    def locate(cls, package: str) -> Path:
        """The source folder of *package*, a top-level package, found without
        importing it. A folder named like it on ``sys.path`` (its repository,
        which imports as an empty namespace) is looked through to the package
        inside.

        Raises:
            ValueError: *package* is dotted, or no package of that name is
                installed or on ``sys.path``.
        """
        if not package or "." in package:
            raise ValueError(f"{package!r} is not a top-level package name")
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            spec = None
        locations = (
            [Path(p) for p in (spec.submodule_search_locations or ())] if spec else []
        )
        for base in locations:
            if (base / "__init__.py").is_file():
                return base
        for base in locations:
            if (base / package / "__init__.py").is_file():
                return base / package
        raise ValueError(f"{package!r}: no package of that name is importable here")

    @classmethod
    def registered(cls) -> List[str]:
        """The packages a search covers when it names none: the top-level
        package of each :attr:`PROVIDER_GROUP` entry point, then of each
        :attr:`HOST_GROUP` one, then each package installed editable here (the
        code being worked on, registered or not), each once (an editable and an
        installed copy register twice)."""
        from pythontk.core_utils.package_manager import PackageManager

        found: List[str] = []
        for group in (cls.PROVIDER_GROUP, cls.HOST_GROUP):
            for ep in PackageManager.entry_points(group):
                package = cls._top(ep.value)
                if package not in found:
                    found.append(package)
        found += [p for p in PackageManager.editable_packages() if p not in found]
        return found

    @classmethod
    def host(cls, package: str) -> Optional[str]:
        """The application *package*'s functions run in: the name of the
        :attr:`HOST_GROUP` entry point whose runner it provides, or ``None``
        (any Python)."""
        from pythontk.core_utils.package_manager import PackageManager

        for ep in PackageManager.entry_points(cls.HOST_GROUP):
            if cls._top(ep.value) == package:
                return ep.name
        return None

    # -- static ----------------------------------------------------------------------
    @classmethod
    def read(
        cls,
        package: Union[str, os.PathLike],
        root: Optional[Union[str, os.PathLike]] = None,
        cache: bool = True,
    ) -> PackageSurface:
        """*package*'s public surface, read from its source (nothing is
        imported); read again only when a source file changes, and kept on disk
        in the per-user cache, so a new process reads an unchanged package
        there instead of parsing it. Reads of one package at once are one
        read. The surface is shared: read it, never change it.

        A tree under a temp folder is kept in the process only. A read that
        could not open a file (a sync client's or a scanner's lock) is kept
        nowhere, so the next read opens it again.

        Parameters:
            package: A top-level package name (located as :meth:`locate`
                does), or the path of its source folder.
            root: The folder its ``source_root`` is written relative to (a
                monorepo root); default: the absolute path.
            cache: False reads a tree read once (a release's source copied
                to temp) without keeping it, in this process or on disk.

        Raises:
            FileNotFoundError: A path that is no folder.
            ValueError: A name no package here answers to.
        """
        source = cls._source_root(package)
        label = (
            Path(os.path.relpath(source, root)).as_posix()
            if root is not None
            else source.as_posix()
        )
        files = list(cls._iter_py_files(source))
        if not cache:
            return cls._walk_package(source, label, files, keep=False)[0]
        stamp = cls._source_stamp(source, files)
        key = (str(source), label)
        with cls._read_lock(key):
            cached = cls._READS.get(key)
            if cached is not None and cached[0] == stamp:
                return cached[1]
            surface = cls._cached(source, label, stamp)
            if surface is None:
                surface, unread = cls._walk_package(source, label, files)
                if unread:
                    return surface
                if cls._keeps_on_disk(source):
                    cls._cache(source, label, stamp, surface)
            cls._READS[key] = (stamp, surface)
        return surface

    @classmethod
    def retirements(
        cls, package: Union[str, os.PathLike]
    ) -> List[Tuple[str, str, str, str]]:
        """Every retirement smaller than a symbol in *package*'s source (a
        keyword, a value, a code path: ``Deprecation.parameter(...)`` and its
        kin), as ``(relpath, what, remove_in, since)``; ``remove_in`` is
        ``""`` where the source names one this read cannot resolve.

        Parameters:
            package: A top-level package name, or the path of its source folder.
        """
        return cls._retirements(cls._source_root(package))

    @classmethod
    def search(
        cls, query: str, packages: Optional[List[str]] = None, limit: Optional[int] = 25
    ) -> List[Dict[str, Any]]:
        """The public symbols whose target or summary names every word of *query*.

        Read from the source, nothing imported: every public function, class,
        method (static, class and instance), property and constant a caller
        can import.

        Parameters:
            query: Words to match, in any case: a name, part of one, a topic
                (``"material texture path"``). Empty lists every one.
            packages: The packages to search; default: every one registered
                with pythontk (:meth:`registered`), but any of those that is
                no package here (logged, then passed over).
            limit: The most to return (``None``: all); those whose name holds
                every word come first.

        Returns:
            One mapping per match: its ``target`` (``"module:qualified.name"``,
            what :meth:`describe` takes), ``kind``, ``signature``, ``summary``,
            the ``host`` application it runs in when it needs one, and
            ``deprecated`` / ``remove_in`` for a retired one.

        Raises:
            ValueError: A package named is not one here.
        """
        words = [w for w in query.lower().split() if w]
        ranked = []
        named = packages is not None
        for package in packages if named else cls.registered():
            if not named:
                try:
                    cls._source_root(package)
                except (ValueError, FileNotFoundError) as exc:
                    _LOG.warning("api search passes over %r: %s", package, exc)
                    continue
            host = cls.host(package)
            for target, symbol in cls._rows(package):
                if not all(w in f"{target} {symbol.summary}".lower() for w in words):
                    continue
                named = all(w in cls._qualname(symbol).lower() for w in words)
                ranked.append((not named, target, cls._found(target, symbol, host)))
        ranked.sort(key=lambda r: r[:2])
        found = [row for _, _, row in ranked]
        return found if limit is None else found[: max(0, limit)]

    @classmethod
    def describe(cls, target: str) -> Dict[str, Any]:
        """What *target* is and how to use it, read from its source (nothing is imported).

        Parameters:
            target: A public symbol as :meth:`search` lists it
                (``"mayatk.mat_utils._mat_utils:MatUtils.get_mat_info"``).

        Returns:
            Its ``target``, ``kind``, ``signature``, ``summary``, ``doc`` (the
            whole docstring), ``file`` (``path:line``), the ``host`` it runs in
            when it needs one, ``deprecated`` / ``remove_in`` for a retired
            one; a class adds its ``bases`` and ``members`` (each one's name,
            kind, signature and summary).

        Raises:
            ValueError: *target* is malformed, private, or names no public
                symbol of a package here.
        """
        module, _, qualname = (target or "").partition(":")
        if not module or not qualname:
            raise ValueError(
                f"target {target!r} is not 'module:qualified.name' (as search lists it)"
            )
        if any(part.startswith("_") for part in qualname.split(".")):
            raise ValueError(f"{target!r} is private")
        package = cls._top(module)
        rows = dict(cls._rows(package))
        symbol = rows.get(target)
        if symbol is None:
            close = difflib.get_close_matches(target, rows, n=3)
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            raise ValueError(f"{target!r} is no public symbol of {package}{hint}")
        found = cls._found(target, symbol, cls.host(package))
        line = symbol.line
        found.update(
            doc=getattr(symbol, "doc", ""), file=f"{getattr(symbol, 'path', '')}:{line}"
        )
        if isinstance(symbol, ClassSurface):
            found["bases"] = list(symbol.bases)
            found["members"] = [
                {
                    "name": m.name,
                    "kind": m.kind,
                    "signature": cls._spelled(m),
                    "summary": m.summary,
                }
                for m in symbol.members
            ]
        return found

    # -- live ------------------------------------------------------------------------
    @classmethod
    def members(
        cls,
        owner: type,
        kind: Optional[str] = None,
        inherited: bool = True,
        private: bool = False,
    ) -> List[SymbolRecord]:
        """*owner*'s members as Python resolves them, one record each, by name.

        Parameters:
            owner: A class.
            kind: Keep only ``"methods"`` (instance, class and static),
                ``"properties"``, ``"classmethods"`` or ``"staticmethods"``;
                ``None`` or ``"all"`` keeps every member.
            inherited: Include what *owner* inherits; ``False``: only what its
                own body defines.
            private: Include names starting with ``_``.
        """
        return [
            cls._record_of(owner, n, m, k)
            for n, m, k in cls._member_items(owner, kind, inherited, private)
        ]

    @classmethod
    def record(cls, owner: type, name: str) -> Optional[SymbolRecord]:
        """The record of *owner*'s member *name*, or ``None`` when it has none."""
        member = getattr(owner, name, None)
        if member is None:
            return None
        return cls._record_of(
            owner, name, member, cls._member_kind(owner, name, member)
        )

    @classmethod
    def enrich(cls, owner: type, record: SymbolRecord) -> Dict[str, Any]:
        """*record*'s fields plus what only a live read knows: its ``flags``
        (:meth:`flags`), the class that ``defined_in`` it, and its ``source``
        (``path:line``, :meth:`location`). Kept off :class:`SymbolRecord`, whose
        fields are the static read's."""
        member = getattr(owner, record.name, None)
        data = record.as_dict()
        data["flags"] = cls.flags(member) if member is not None else []
        data["defined_in"] = cls.defined_in(owner, record.name)
        data["source"] = (
            cls.location(member) if member is not None else None
        ) or cls.NO_LOCATION
        return data

    @staticmethod
    def exports(package: str) -> List[Dict[str, Any]]:
        """What *package*'s root publishes, resolved live (imports it): one row
        per name ``package.<name>`` answers to, which no static read can see
        (a name dropped from the root while its class stays, an alias).

        Two tiers, as the lazy loader publishes them: ``root`` (``__all__``)
        and ``bare`` (a wildcard-exposed function, ``ptk.filter_list``, which
        ``__all__`` leaves out). A name that fails to resolve (an optional
        dependency) is a row too, its ``kind`` saying why.

        Parameters:
            package: A top-level package name.

        Returns:
            ``{"name", "tier", "module", "qualname", "kind"}`` per name, by name
            in any case; ``module`` + ``qualname`` is a dotted path that
            resolves to it.
        """
        import importlib

        pkg = importlib.import_module(package)
        root_names = set(getattr(pkg, "__all__", ()))
        bare_names = set(getattr(pkg, "METHOD_TO_MODULE", {})) - root_names
        # Where the loader REGISTERS each name: for a constant that is the
        # authority, since an instance's __module__ is its class's home.
        owners = {
            **getattr(pkg, "METHOD_TO_MODULE", {}),
            **getattr(pkg, "CLASS_TO_MODULE", {}),
        }
        rows = []
        for name in sorted(root_names | bare_names, key=str.lower):
            row: Dict[str, Any] = {
                "name": name,
                "tier": "root" if name in root_names else "bare",
            }
            try:
                obj = getattr(pkg, name)
            except Exception as exc:  # an optional dependency must not cost the rest
                row.update(module=None, qualname=None, kind=f"unresolvable ({exc})")
                rows.append(row)
                continue
            if isinstance(obj, type) or callable(obj):
                module = getattr(obj, "__module__", None) or owners.get(name)
            else:
                module = owners.get(name) or getattr(obj, "__module__", None)
            row["module"] = module
            row["qualname"] = getattr(obj, "__qualname__", None)
            row["kind"] = (
                "class"
                if isinstance(obj, type)
                else "function"
                if callable(obj)
                else type(obj).__name__
            )
            rows.append(row)
        return rows

    @staticmethod
    def flags(member: Any) -> List[str]:
        """What is special about *member*: ``async``, ``generator``,
        ``async-generator``, ``abstract`` (read through its decorators)."""
        try:
            unwrapped = inspect.unwrap(member) if callable(member) else member
        except Exception:
            unwrapped = member
        flags = []
        if inspect.iscoroutinefunction(unwrapped):
            flags.append("async")
        if inspect.isgeneratorfunction(unwrapped):
            flags.append("generator")
        if inspect.isasyncgenfunction(unwrapped):
            flags.append("async-generator")
        if getattr(unwrapped, "__isabstractmethod__", False):
            flags.append("abstract")
        return flags

    @staticmethod
    def signature_of(member: Any, kind: Optional[str] = None) -> str:
        """*member*'s signature through its decorators (``"(x, y=1)"``); ``""``
        for a property or anything not callable, ``"(...)"`` when it cannot be
        read (a builtin)."""
        if not callable(member) or kind == "property":
            return ""
        try:
            return str(inspect.signature(inspect.unwrap(member)))
        except (ValueError, TypeError):
            return "(...)"

    @staticmethod
    def defined_in(owner: type, name: str) -> Optional[str]:
        """The name of the first class in *owner*'s MRO whose own body defines *name*."""
        for klass in inspect.getmro(owner):
            if name in klass.__dict__:
                return klass.__name__
        return None

    #: What a live read says for a symbol it cannot place (a builtin, an attribute).
    NO_LOCATION = "Unknown location (built-in or C extension)"

    @staticmethod
    def location(obj: Any) -> Optional[str]:
        """Where *obj* is defined, through its decorators: ``path:line``, the
        file alone when its lines cannot be read, ``None`` for a builtin."""
        try:
            unwrapped = inspect.unwrap(obj) if callable(obj) else obj
            path = inspect.getsourcefile(unwrapped)
            return f"{path}:{inspect.getsourcelines(unwrapped)[1]}" if path else None
        except (OSError, TypeError, ValueError):
            try:
                return inspect.getfile(obj)
            except (OSError, TypeError):
                return None

    @staticmethod
    def source_of(obj: Any) -> str:
        """*obj*'s source code, through its decorators.

        Raises:
            OSError: Its source is not available.
            TypeError: It has no source (a builtin, an instance).
        """
        return inspect.getsource(inspect.unwrap(obj) if callable(obj) else obj)
