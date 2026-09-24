# !/usr/bin/python
# coding=utf-8
"""Deprecation - retiring public surface through one mechanism, with a clock.

The public-API contract (``m3trik/docs/CODE_STANDARD.md`` s5) says a renamed,
moved or removed public name keeps working as an alias for **one release**.
Before this module the ecosystem honoured the first half of that rule seven
different ways -- an inline ``warnings.warn`` with a hand-picked ``stacklevel``,
a per-cluster warn helper with a different one, a module ``__getattr__`` that
warned and another that did not, a ``cmds.warning``, a logger call, a silent
binding alias, a docstring line for a renamed keyword -- and the second half not
at all. "Removed in the next release" is unfalsifiable: nothing recorded WHICH
release, so nothing could fail when the release came and went. ``flip_uvs`` was
deprecated 2025-12-17 and has shipped in 50 releases since.

So the point of this module is not that the seven shapes collapse into one. It
is ``remove_in``: every deprecation names the version it stops working in, that
version is parsed (a typo raises at import, because a ``remove_in`` that cannot
be compared is an alias that never expires), and the resulting records are
readable both at runtime (:meth:`Deprecation.registered`) and statically, by
``generate_api_registry.py``, which walks the decorators with ``ast`` and fails
``--check`` once a removal version has passed.

Four shapes, matching what the ecosystem actually retires:

    * :meth:`Deprecation.symbol` - a function, method or class.
    * :meth:`Deprecation.parameter` - one keyword of a live function, including
      the rename-and-remap case (``use_object_axes=False`` ->
      ``axis_frame="world"``).
    * :meth:`Deprecation.attributes` - module attributes that moved, installed
      as a module ``__getattr__`` that chains onto an existing one (the uitk
      ``_LAZY`` loaders have one already, and clobbering it would break the
      package).
    * :meth:`Deprecation.values` - retired members of a value vocabulary, e.g. a
      mode string that now builds as something else.

For anything the four do not reach -- a deprecated branch inside a function
that stays -- :meth:`Deprecation.warn` emits one notice from a function body.
It is the escape hatch, not a fifth shape, and it exists so that an awkward
case cannot justify going back to a hand-rolled ``warnings.warn``.

Two rules the mechanism enforces rather than documents:

    * **A replacement is mandatory.** A deprecation nobody can act on is noise,
      so ``replacement`` must be non-empty -- either a name or the free text
      that stands in for one ("subprocess, or a dedicated git library").
    * **Warning is not breaking.** Every shape leaves this release's behaviour
      exactly as it was; only the notice is new.

The window is counted in calendar days as well as releases. ``since`` names
the date a notice first ships, and a name is due only once the version has
reached ``remove_in`` AND :data:`MIN_WINDOW_DAYS` have passed -- measured
2026-09-19, seven back-to-back releases retired names 15 days after their
first warning. A ``remove_in`` naming a patch release is refused: a patch
never removes a name.

``DeprecationWarning`` is invisible by default outside ``__main__``, which is
every DCC session the ecosystem runs in. :attr:`Deprecation.sink` is the escape
hatch: set it to ``cmds.warning`` (or a logger) and each record is *also*
announced once per process on a channel the user can see. The
``warnings.warn`` call always happens, so ``assertWarns`` and ``-W error`` keep
working regardless of the sink.
"""

from __future__ import annotations

import datetime
import functools
import importlib
import inspect
import re
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

#: Modules whose frames are transparent when a warning is attributed to a
#: caller. This module's own frames obviously, plus the lazy package resolver:
#: a deprecated module attribute reached through ``pythontk.<Name>`` is served
#: by the resolver's ``__getattr__``, and pointing the warning at the resolver
#: tells the reader nothing about which of their lines to change.
_TRANSPARENT_MODULES = frozenset(
    {
        __name__,
        "pythontk.core_utils.module_resolver",
    }
)

#: ``re.ASCII`` deliberately: bare ``\d`` also matches Devanagari and Arabic-
#: Indic digits, so an unnoticed paste of one would build a version key that
#: never compares equal to anything the release tooling writes.
_VERSION_RE = re.compile(r"\d+(?:\.\d+){1,2}\Z", re.ASCII)


def _version_key(version: str) -> Tuple[int, int, int]:
    """Comparable key for a ``MAJOR.MINOR[.PATCH]`` release version.

    Parameters:
        version (str): A release version, e.g. ``"0.11.0"`` or ``"1.4"``.

    Returns:
        (tuple) Three integers, zero-padded, safe to compare with ``>=``.

    Raises:
        ValueError: The string is not a comparable release version. This is
            raised at decoration time -- i.e. at import -- on purpose: a
            ``remove_in`` that cannot be parsed is a deprecation that can never
            expire, which is the exact failure this module exists to prevent.
    """
    text = str(version).strip()
    if not _VERSION_RE.match(text):
        raise ValueError(
            f"remove_in must be a release version like '0.11.0', got {version!r}. "
            "An unparseable version is a deprecation that never expires."
        )
    parts = [int(p) for p in text.split(".")]
    parts += [0] * (3 - len(parts))
    return (parts[0], parts[1], parts[2])


#: The shortest CALENDAR window a notice gets between the release it first
#: warns in (``since``) and its removal. Counted in versions alone, seven
#: releases in two weeks (0.9.35 -> 0.10.1) satisfied "one release of
#: warnings" while an outside caller got 15 days (BACKLOG 2026-09-19).
MIN_WINDOW_DAYS = 30

#: ``YYYY-MM-DD`` and nothing looser: ``date.fromisoformat`` also takes the
#: basic ``20260904`` form on 3.11+, which the static gate's reader must not
#: have to know about.
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}\Z", re.ASCII)


def _date_of(since: str) -> datetime.date:
    """The ``YYYY-MM-DD`` date *since* names.

    Raises:
        ValueError: Not an ISO calendar date -- at decoration time, like an
            unparseable ``remove_in``: a date that cannot be read is a window
            that cannot be counted.
    """
    text = str(since).strip()
    try:
        if not _DATE_RE.match(text):
            raise ValueError
        return datetime.date.fromisoformat(text)
    except ValueError:
        raise ValueError(
            f"since must be the ISO date the notice first ships, like "
            f"'2026-09-23', got {since!r}."
        ) from None


def _window_expired(
    remove_in: str,
    version: str,
    since: str = "",
    today: Optional[datetime.date] = None,
) -> bool:
    """Whether a notice is due: *version* has reached *remove_in* AND, when
    the notice names the date it first shipped, :data:`MIN_WINDOW_DAYS` have
    passed since. A notice naming no date keeps the version rule alone."""
    if _version_key(version) < _version_key(remove_in):
        return False
    if not since:
        return True
    today = today or datetime.date.today()
    return (today - _date_of(since)).days >= MIN_WINDOW_DAYS


@dataclass(frozen=True)
class DeprecationRecord:
    """One retired name, what replaces it, and the release it stops working in.

    Produced by every :class:`Deprecation` entry point and collected in a
    process-wide roster, so "what is currently deprecated, and when does it
    go?" is a question with an answer instead of a grep.
    """

    what: str
    replacement: str
    remove_in: str
    kind: str = "symbol"  # symbol|parameter|attribute|value
    module: str = ""
    reason: str = ""
    #: The ISO date the notice first ships (``"2026-09-23"``); the window runs
    #: :data:`MIN_WINDOW_DAYS` from it. Empty keeps the version rule alone.
    since: str = ""

    def __post_init__(self) -> None:
        """Validate the load-bearing fields at construction.

        Raises:
            ValueError: ``what`` or ``replacement`` is empty, ``remove_in`` is
                not a comparable release version or names a PATCH release (a
                patch never removes a name: a removal is a minor bump's
                business), or ``since`` is not an ISO date.
        """
        if not str(self.what).strip():
            raise ValueError("a deprecation must name what is deprecated")
        if not str(self.replacement).strip():
            raise ValueError(
                f"{self.what}: a deprecation must name a replacement -- a name, "
                "or the free text that stands in for one. A notice the caller "
                "cannot act on is noise."
            )
        if _version_key(self.remove_in)[2]:
            raise ValueError(
                f"{self.what}: remove_in={self.remove_in!r} is a patch release, "
                "and a patch never removes a name -- name the minor it goes in "
                "(e.g. '0.12.0')."
            )
        if self.since:
            _date_of(self.since)

    @property
    def key(self) -> Tuple[str, str]:
        """Identity in the roster: ``(kind, what)``.

        Keyed rather than appended so a module imported twice (a DCC reload
        forks class state -- see the reload notes in ``module_resolver``) does
        not grow the roster, and so re-registering is idempotent.
        """
        return (self.kind, self.what)

    @property
    def package(self) -> str:
        """Top-level package the deprecated name belongs to (may be empty)."""
        return self.module.split(".", 1)[0] if self.module else ""

    @property
    def not_before(self) -> str:
        """The earliest date the name may go (``since`` + :data:`MIN_WINDOW_DAYS`),
        as ``YYYY-MM-DD``; empty when the notice names no ``since``."""
        if not self.since:
            return ""
        due = _date_of(self.since) + datetime.timedelta(days=MIN_WINDOW_DAYS)
        return due.isoformat()

    def message(self) -> str:
        """The warning text: what went, when it goes, what to use instead."""
        where = f"{self.package} {self.remove_in}" if self.package else self.remove_in
        if self.since:
            where = f"{where}, not before {self.not_before}"
        text = (
            f"{self.what} is deprecated and will be removed in {where}; "
            f"use {self.replacement} instead."
        )
        return f"{text} {self.reason}" if self.reason else text

    def expired(self, version: str, today: Optional[datetime.date] = None) -> bool:
        """True once the alias has outlived its window: *version* has reached
        ``remove_in`` AND, for a notice that names ``since``,
        :data:`MIN_WINDOW_DAYS` have passed (:meth:`Deprecation.window_expired`).

        Parameters:
            version (str): The version to test against, normally the package's
                own ``__version__``.
            today (date): The day to judge on; defaults to today.

        Returns:
            (bool) True when the alias has outlived its window.
        """
        return _window_expired(self.remove_in, version, self.since, today)


class _DeprecationInternal:
    """Registry, frame accounting and decoration plumbing for :class:`Deprecation`.

    Split out per the encapsulation rule (helpers on a ``_<Class>Internal``
    base) so the public class reads as the four shapes plus the roster.
    """

    #: Process-wide roster, keyed by :attr:`DeprecationRecord.key`.
    _records: Dict[Tuple[str, str], DeprecationRecord] = {}

    #: Records already announced on :attr:`Deprecation.sink`. The sink is a
    #: user-visible channel in a session that can run for hours, and a
    #: deprecated call inside a per-frame loop would otherwise paper over the
    #: viewport. ``warnings.warn`` is NOT gated by this -- the warnings module
    #: does its own per-location deduplication, and gating it here would break
    #: ``assertWarns`` for the second test that exercises the same call site.
    _announced: set = set()

    # ------------------------------------------------------------------ frames

    @classmethod
    def _own_frames(cls) -> int:
        """Leading stack frames belonging to the deprecation machinery.

        Counted rather than hardcoded because the four shapes reach
        :meth:`_emit` through different depths -- a decorator wrapper, a module
        ``__getattr__``, a value resolver, a direct :meth:`Deprecation.warn`.
        Every one of those frames lives in this module (a ``functools.wraps``
        wrapper keeps *this* module's globals, whatever name it copies), so the
        count is simply the unbroken run of them, which also absorbs the lazy
        package resolver when a deprecated attribute is reached through it.

        Returns:
            (int) The ``stacklevel`` that, used from :meth:`_emit`, attributes
            the warning to the first frame outside this machinery.
        """
        frame = inspect.currentframe()
        count = 0
        while frame is not None:
            if frame.f_globals.get("__name__", "") not in _TRANSPARENT_MODULES:
                break
            count += 1
            frame = frame.f_back
        return count

    # ------------------------------------------------------------------ roster

    @classmethod
    def _register(cls, record: DeprecationRecord) -> DeprecationRecord:
        """Add *record* to the roster (idempotent) and return it."""
        cls._records[record.key] = record
        return record

    @classmethod
    def _emit(cls, record: DeprecationRecord, stacklevel: int = 1) -> None:
        """Warn for *record*, attributed *stacklevel* frames out of the machinery.

        ``stacklevel`` counts exactly as ``warnings.warn``'s own does, but from
        the first frame outside this module: 1 is the caller that tripped the
        deprecation, 2 is that caller's caller. Consumers that wrap a
        :meth:`Deprecation.warn` call in a helper of their own pass the extra
        depth here.
        """
        message = record.message()
        # Clamped: an interpreter without frame support returns no frames at
        # all, and ``warnings.warn(stacklevel=0)`` silently attributes to
        # ``__main__`` instead of raising, which would be a wrong answer that
        # looks like a right one.
        level = max(cls._own_frames() + max(int(stacklevel), 1) - 1, 1)
        warnings.warn(message, DeprecationWarning, stacklevel=level)
        sink = getattr(cls, "sink", None)
        if sink is None or record.key in cls._announced:
            return
        cls._announced.add(record.key)
        try:
            sink(message)
        except Exception:
            # A deprecation notice must not be able to break the call it is
            # attached to. The DCC channels this is pointed at are exactly the
            # ones that raise when called off the main thread or after their
            # host has torn down (``cmds.warning`` does both).
            pass

    # -------------------------------------------------------------- decoration

    @classmethod
    def _describe(cls, target: Any) -> Tuple[str, str]:
        """``(qualname, module)`` for a function or class, for a record's ``what``.

        A ``property`` is looked through to an accessor: the descriptor itself
        carries neither name nor module, so describing it directly produced
        ``"<property object at 0x...>"`` as the deprecated name, an empty
        package in the message, and a roster key that changed every run.
        """
        if isinstance(target, property):
            target = target.fget or target.fset or target.fdel or target
        return (
            getattr(target, "__qualname__", getattr(target, "__name__", repr(target))),
            getattr(target, "__module__", "") or "",
        )

    @classmethod
    def _mark(cls, target: Any, record: DeprecationRecord) -> None:
        """Stamp the PEP 702 marker and the record onto *target*.

        ``__deprecated__`` is set to the message string rather than ``True``
        because that is what :func:`warnings.deprecated` does on 3.13+, so
        anything that already understands the stdlib marker -- including
        ``HelpMixin._is_deprecated`` -- understands this one, and the eventual
        migration is a deletion rather than a rewrite.
        """
        try:
            target.__deprecated__ = record.message()
            target.__deprecated_record__ = record
        except (AttributeError, TypeError):
            # Builtins and some C-level callables refuse attribute assignment.
            # The warning still fires; only the static marker is unavailable.
            pass

    @classmethod
    def _decorate_symbol(cls, target: Any, record: DeprecationRecord) -> Any:
        """Return *target* wrapped so that reaching it warns once per call site."""
        binder = None
        if isinstance(target, (classmethod, staticmethod)):
            # Tolerate either decorator order. Written above ``@classmethod``
            # this sees the descriptor, not the function; rebuilding it after
            # wrapping makes the two orders equivalent instead of making one of
            # them a silent no-op.
            binder = type(target)
            target = target.__func__

        if isinstance(target, property):
            # Written above ``@property`` the decorator sees the descriptor,
            # which is not callable: wrapping it would build a member that
            # raises TypeError on first access. Rebuild the property around
            # decorated accessors instead, so reading AND writing both warn.
            return property(
                cls._decorate_symbol(target.fget, record) if target.fget else None,
                cls._decorate_symbol(target.fset, record) if target.fset else None,
                cls._decorate_symbol(target.fdel, record) if target.fdel else None,
                target.__doc__,
            )

        if inspect.isclass(target):
            cls._decorate_class(target, record)
            return target

        @functools.wraps(target)
        def wrapper(*args, **kwargs):
            cls._emit(record)
            return target(*args, **kwargs)

        # After ``functools.wraps``, which copies ``__dict__`` wholesale. Both
        # the wrapper and the original are marked: ``wraps`` sets
        # ``__wrapped__``, so any consumer that unwraps -- ``HelpMixin`` does,
        # to recover a real signature -- would otherwise land on an unmarked
        # function and report the symbol as live.
        cls._mark(wrapper, record)
        cls._mark(target, record)
        return binder(wrapper) if binder is not None else wrapper

    @classmethod
    def _decorate_class(cls, target: type, record: DeprecationRecord) -> None:
        """Wrap *target*'s ``__new__`` so constructing it warns."""
        original_new = target.__new__

        @functools.wraps(original_new)
        def __new__(subclass, *args, **kwargs):
            cls._emit(record)
            if original_new is object.__new__:
                # ``object.__new__`` rejects extra arguments as soon as the
                # class overrides ``__init__``, which is the common case here.
                return original_new(subclass)
            return original_new(subclass, *args, **kwargs)

        # Wrapped explicitly because CPython's implicit conversion of
        # ``__new__`` happens when the class BODY runs, not on a later
        # attribute assignment. Measured: a bare function assigned here is
        # still *called* correctly, so this is not about the call. It is about
        # the class dict holding what every normally-defined ``__new__`` holds
        # -- without it ``getattr_static`` reports a plain function where every
        # other class reports a staticmethod, and introspection that keys on
        # that difference quietly disagrees about this one class.
        target.__new__ = staticmethod(__new__)
        cls._mark(target, record)


class Deprecation(_DeprecationInternal):
    """Retire a public name with a notice and a removal version.

    Every entry point registers a :class:`DeprecationRecord` when the module
    that uses it is imported -- not when the deprecated path is first taken --
    so :meth:`registered` and :meth:`expired` see an alias nobody has tripped
    yet, which is the only kind that survives a release unnoticed.

    Example::

        class UvUtils:
            @classmethod
            @Deprecation.symbol(
                "UvUtils.mirror_uvs", remove_in="0.16.0", since="2026-09-23"
            )
            def flip_uvs(cls, objects, **kwargs):
                return cls.mirror_uvs(objects, **kwargs)
    """

    #: Optional extra channel for deprecation notices, called once per record
    #: per process with the message. ``DeprecationWarning`` is hidden by
    #: default outside ``__main__``, so in a DCC nobody ever sees one; point
    #: this at ``cmds.warning`` or a logger and they do. Additive -- the
    #: ``warnings.warn`` call happens either way.
    sink: Optional[Callable[[str], None]] = None

    #: The shortest calendar window between a notice's ``since`` and its
    #: removal (see :func:`_window_expired`).
    MIN_WINDOW_DAYS: int = MIN_WINDOW_DAYS

    # ------------------------------------------------------------------- emit

    @staticmethod
    def window_expired(
        remove_in: str,
        version: str,
        since: str = "",
        today: Optional[datetime.date] = None,
    ) -> bool:
        """Whether a retirement is due -- THE rule, for both gates.

        Due once *version* has reached *remove_in* AND, when the notice names
        the date it first shipped (*since*), :attr:`MIN_WINDOW_DAYS` have
        passed; a notice naming no date keeps the version rule alone. Public
        for the reason :meth:`version_key` is: ``generate_api_registry.py``
        loads this module off disk and judges ``--check`` with this function,
        so the static gate and the runtime roster cannot disagree.

        Parameters:
            remove_in (str): The release the name goes in.
            version (str): The version to judge, normally ``__version__``.
            since (str): ISO date the notice first shipped, or empty.
            today (date): The day to judge on; defaults to today.

        Returns:
            (bool) True when the name has outlived its window.
        """
        return _window_expired(remove_in, version, since, today)

    @staticmethod
    def version_key(version: str) -> Tuple[int, int, int]:
        """Comparable key for a ``MAJOR.MINOR[.PATCH]`` release version.

        Public because the expiry rule has to be one implementation: this
        module compares versions at runtime, and ``generate_api_registry.py``
        compares them statically for the CI gate. That generator loads this
        module straight off disk (it must run with nothing installed), so the
        alternative is a second copy of "what counts as a version" in a file
        that cannot import the first.

        Parameters:
            version (str): A release version, e.g. ``"0.11.0"``.

        Returns:
            (tuple) Three integers, zero-padded.

        Raises:
            ValueError: Not a comparable release version.
        """
        return _version_key(version)

    @classmethod
    def warn(
        cls,
        what: str,
        replacement: str,
        *,
        remove_in: str,
        reason: Optional[str] = None,
        module: Optional[str] = None,
        kind: str = "symbol",
        stacklevel: int = 1,
        since: Optional[str] = None,
    ) -> DeprecationRecord:
        """Emit a deprecation notice from inside a function body.

        The escape hatch for what the decorators cannot reach -- a deprecated
        path taken conditionally, or a cluster of entry points that share one
        message. Prefer :meth:`symbol` where the whole callable is retired.

        Parameters:
            what (str): The deprecated name, as a caller would write it.
            replacement (str): What to use instead. Required.
            remove_in (str): The release it stops working in, e.g. ``"0.11.0"``.
            reason (str): Optional extra sentence appended to the message.
            module (str): Owning module; defaults to the caller's ``__name__``,
                which is what names the package in the message.
            kind (str): Roster category. Defaults to ``"symbol"``.
            stacklevel (int): 1 attributes the warning to the caller of this
                method, 2 to that caller's caller. Raise it by one per helper
                the consumer wraps this call in.
            since (str): ISO date the notice first ships; starts the calendar
                window (:meth:`window_expired`).

        Returns:
            (DeprecationRecord) The registered record.
        """
        if module is None:
            frame = inspect.currentframe()
            back = frame.f_back if frame is not None else None
            module = back.f_globals.get("__name__", "") if back is not None else ""
        record = cls._register(
            DeprecationRecord(
                what=what,
                replacement=replacement,
                remove_in=remove_in,
                kind=kind,
                module=module,
                reason=reason or "",
                since=since or "",
            )
        )
        cls._emit(record, stacklevel=stacklevel)
        return record

    # ------------------------------------------------------------- decorators

    @classmethod
    def symbol(
        cls,
        replacement: str,
        *,
        remove_in: str,
        reason: Optional[str] = None,
        since: Optional[str] = None,
    ) -> Callable[[Any], Any]:
        """Deprecate a whole function, method or class.

        Works above or below ``@classmethod``/``@staticmethod``/``@property``.
        On a class it wraps ``__new__``, so subclasses warn too -- deliberately,
        and matching PEP 702: a subclass of a retired class is a live
        dependency on it.

        On a property, the accessors that exist AT DECORATION TIME are wrapped.
        ``@value.setter`` builds a fresh property from the getter, so a setter
        attached afterwards is not covered by a decorator written above
        ``@property``; decorate each accessor function directly (below its
        ``@property`` / ``@value.setter`` line) when a write must warn too.

        Parameters:
            replacement (str): What to use instead. Required.
            remove_in (str): The release it stops working in.
            reason (str): Optional extra sentence appended to the message.
            since (str): ISO date the notice first ships.

        Returns:
            (callable) The decorator.
        """

        def decorate(target: Any) -> Any:
            unwrap = isinstance(target, (classmethod, staticmethod))
            subject = target.__func__ if unwrap else target
            qualname, module = cls._describe(subject)
            record = cls._register(
                DeprecationRecord(
                    what=qualname,
                    replacement=replacement,
                    remove_in=remove_in,
                    kind="symbol",
                    module=module,
                    reason=reason or "",
                    since=since or "",
                )
            )
            return cls._decorate_symbol(target, record)

        return decorate

    @classmethod
    def parameter(
        cls,
        old: str,
        *,
        remove_in: str,
        new: Optional[str] = None,
        transform: Optional[Callable[[Any], Any]] = None,
        drop: bool = False,
        reason: Optional[str] = None,
        since: Optional[str] = None,
    ) -> Callable[[Callable], Callable]:
        """Deprecate one keyword argument of a function that stays.

        Retires the ecosystem's commonest shape: a renamed or remapped flag
        whose old spelling survives only as a docstring line nothing enforces
        (``use_object_axes`` in seven signatures, ``as_strings`` in four).

        Only the KEYWORD spelling is warned on, and that is correct rather than
        a shortcut: a positionally-passed argument carries no name, so there is
        nothing to say the caller meant the retired one. For a pure rename the
        position is unchanged and such a call needs no migration at all.

        Parameters:
            old (str): The retired keyword.
            remove_in (str): The release it stops working in.
            new (str): The live keyword its value moves to. When given, *old*
                is removed from the call and *new* is set.
            transform (callable): Maps the old value onto the new one, for the
                remap case (``use_object_axes=False`` -> ``"world"``).
            drop (bool): With no *new*, discard the value instead of forwarding
                it -- for a parameter documented as having no effect.
            reason (str): Optional extra sentence appended to the message.
            since (str): ISO date the notice first ships.

        Returns:
            (callable) The decorator.
        """

        def decorate(func: Callable) -> Callable:
            unwrap = isinstance(func, (classmethod, staticmethod))
            subject = func.__func__ if unwrap else func
            qualname, module = cls._describe(subject)
            if new:
                replacement = new
            elif drop:
                replacement = "nothing -- the value already has no effect"
            else:
                # Still forwarded this release, so there is nothing to move the
                # value to; the only action a caller can take is to stop
                # passing it. Saying "no effect" here would be false.
                replacement = "nothing -- stop passing it"
            record = cls._register(
                DeprecationRecord(
                    what=f"{qualname} parameter {old!r}",
                    replacement=replacement,
                    remove_in=remove_in,
                    kind="parameter",
                    module=module,
                    reason=reason or "",
                    since=since or "",
                )
            )

            def apply(kwargs: Dict[str, Any]) -> None:
                cls._emit(record)
                value = kwargs.pop(old)
                if new is None:
                    if not drop:
                        kwargs[old] = value
                    return
                if new in kwargs:
                    # Both spellings in one call. The live one wins -- the
                    # alternative, letting the retired spelling override, is
                    # how ``reference_manager`` ended up with a deprecated bool
                    # that silently beat the enum callers were told to use.
                    warnings.warn(
                        f"{qualname} was given both {old!r} and {new!r}; "
                        f"{new!r} wins and {old!r} is ignored.",
                        DeprecationWarning,
                        stacklevel=cls._own_frames(),
                    )
                    return
                kwargs[new] = transform(value) if transform is not None else value

            if isinstance(func, (classmethod, staticmethod)):
                binder, target = type(func), func.__func__
            else:
                binder, target = None, func

            @functools.wraps(target)
            def wrapper(*args, **kwargs):
                if old in kwargs:
                    apply(kwargs)
                return target(*args, **kwargs)

            return binder(wrapper) if binder is not None else wrapper

        return decorate

    # -------------------------------------------------------------- installers

    @classmethod
    def attributes(
        cls,
        module_globals: Dict[str, Any],
        moved: Mapping[str, str],
        *,
        remove_in: str,
        reason: Optional[str] = None,
        since: Optional[str] = None,
    ) -> Dict[str, DeprecationRecord]:
        """Serve module attributes that moved, through a module ``__getattr__``.

        Chains onto any ``__getattr__`` and ``__dir__`` already installed --
        several packages front a lazy loader with one, and replacing it outright
        would make every non-deprecated name on the module vanish.

        The resolved value is deliberately NOT cached back into the module
        globals: a lazy loader caches because the import is the cost, whereas
        here the notice is the point, and one warning per session for a name a
        caller uses in a loop is one the caller can miss.

        Parameters:
            module_globals (dict): The module's ``globals()``.
            moved (Mapping): Retired attribute name -> dotted path of its new
                home, e.g. ``{"PreviewServer": "pythontk.net_utils.preview.server.PreviewServer"}``.
            remove_in (str): The release the aliases stop working in.
            reason (str): Optional extra sentence appended to each message.
            since (str): ISO date the notices first ship.

        Returns:
            (dict) Attribute name -> its registered record.
        """
        module_name = module_globals.get("__name__", "")
        previous_getattr = module_globals.get("__getattr__")
        previous_dir = module_globals.get("__dir__")

        records = {
            name: cls._register(
                DeprecationRecord(
                    what=f"{module_name}.{name}",
                    replacement=path,
                    remove_in=remove_in,
                    kind="attribute",
                    module=module_name,
                    reason=reason or "",
                    since=since or "",
                )
            )
            for name, path in moved.items()
        }

        def __getattr__(name: str) -> Any:
            record = records.get(name)
            if record is None:
                if previous_getattr is not None:
                    return previous_getattr(name)
                raise AttributeError(
                    f"module {module_name!r} has no attribute {name!r}"
                )
            cls._emit(record)
            target_module, _, attribute = record.replacement.rpartition(".")
            return getattr(importlib.import_module(target_module), attribute)

        def __dir__() -> list:
            base = previous_dir() if previous_dir is not None else list(module_globals)
            return sorted(set(base) | set(records))

        module_globals["__getattr__"] = __getattr__
        module_globals["__dir__"] = __dir__
        return records

    @classmethod
    def values(
        cls,
        aliases: Mapping[Any, Any],
        *,
        what: str,
        remove_in: str,
        module: Optional[str] = None,
        reason: Optional[str] = None,
        since: Optional[str] = None,
    ) -> Callable[[Any], Any]:
        """Build a resolver mapping retired members of a value vocabulary onto live ones.

        A factory rather than a plain ``resolve(value, aliases)`` call so the
        records exist from import: registering on first use would mean an alias
        nobody happened to pass was invisible to :meth:`expired`, and an alias
        nobody passes is precisely the one that quietly outlives its release.

        Parameters:
            aliases (Mapping): Retired value -> the live value it resolves to.
            what (str): What the vocabulary names, e.g. ``"ShadowRig mode"``.
            remove_in (str): The release the aliases stop working in.
            module (str): Owning module; defaults to the caller's ``__name__``.
            reason (str): Optional extra sentence appended to each message.
            since (str): ISO date the notices first ship.

        Returns:
            (callable) ``resolve(value)`` -> the live value, warning on a hit
            and returning anything else untouched.
        """
        if module is None:
            frame = inspect.currentframe()
            back = frame.f_back if frame is not None else None
            module = back.f_globals.get("__name__", "") if back is not None else ""

        records = {
            old: cls._register(
                DeprecationRecord(
                    what=f"{what} {old!r}",
                    replacement=repr(live),
                    remove_in=remove_in,
                    kind="value",
                    module=module,
                    reason=reason or "",
                    since=since or "",
                )
            )
            for old, live in aliases.items()
        }

        def resolve(value: Any) -> Any:
            try:
                record = records.get(value)
            except TypeError:  # unhashable: cannot be one of the aliases
                return value
            if record is None:
                return value
            cls._emit(record)
            return aliases[value]

        return resolve

    # ----------------------------------------------------------------- roster

    @classmethod
    def registered(
        cls,
        *,
        module: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> Tuple[DeprecationRecord, ...]:
        """Every deprecation registered so far, sorted by removal version.

        Complete only for modules that have been imported. The static walk in
        ``generate_api_registry.py`` is the complete view of decorated symbols;
        this is the live one, and the only view that sees the shapes a static
        walk cannot resolve (module attributes, value vocabularies).

        Parameters:
            module (str): Keep only records whose module is, or is under, this
                one -- ``"pythontk"`` matches ``pythontk.file_utils._file_utils``.
            kind (str): Keep only this category.

        Returns:
            (tuple) Matching :class:`DeprecationRecord` objects.
        """
        found = [
            record
            for record in cls._records.values()
            if (kind is None or record.kind == kind)
            and (
                module is None
                or record.module == module
                or record.module.startswith(f"{module}.")
            )
        ]
        return tuple(sorted(found, key=lambda r: (_version_key(r.remove_in), r.what)))

    @classmethod
    def expired(
        cls,
        version: str,
        *,
        module: Optional[str] = None,
        today: Optional[datetime.date] = None,
    ) -> Tuple[DeprecationRecord, ...]:
        """Registered deprecations that should already have been deleted.

        Parameters:
            version (str): The version to judge against, normally the package's
                own ``__version__``.
            module (str): Optional module filter, as per :meth:`registered`.
            today (date): The day to judge on; defaults to today.

        Returns:
            (tuple) Records past their window (:meth:`window_expired`): at or
            below *version* AND, where they name ``since``, old enough.
        """
        return tuple(
            record
            for record in cls.registered(module=module)
            if record.expired(version, today)
        )

    @classmethod
    def report(
        cls,
        version: Optional[str] = None,
        *,
        module: Optional[str] = None,
        today: Optional[datetime.date] = None,
    ) -> str:
        """Render the roster as lines, marking anything already overdue.

        Parameters:
            version (str): When given, records past their window are marked
                ``EXPIRED``, and one due by version but not yet by date
                ``HELD until <date>``.
            module (str): Optional module filter, as per :meth:`registered`.
            today (date): The day to judge on; defaults to today.

        Returns:
            (str) One line per record, or a single line saying there are none.
        """
        records = cls.registered(module=module)
        if not records:
            return "No deprecations registered."
        lines = []
        for record in records:
            mark = ""
            if version is not None and record.expired(version, today):
                mark = "EXPIRED "
            elif version is not None and record.expired(version, datetime.date.max):
                mark = f"HELD until {record.not_before} "
            lines.append(
                f"{mark}{record.remove_in}  {record.kind:<9} {record.what} "
                f"-> {record.replacement}"
            )
        return "\n".join(lines)
