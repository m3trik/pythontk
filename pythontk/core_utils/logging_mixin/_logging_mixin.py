# !/usr/bin/python
# coding=utf-8
"""``LoggingMixin``: one ``LoggerExt``-patched logger per class.

The facade of the logging toolkit: ``self.logger`` (class-shared, lazily
created), ``class_logger``, and the class-scoped sink controls (level, file tee,
ring buffer). Tables and groups come from its base, ``TableMixin``.
"""

from __future__ import annotations

import logging as internal_logging
from typing import Optional, Union

from pythontk.core_utils.class_property import ClassProperty
from pythontk.core_utils.logging_mixin.logger_ext import LoggerExt
from pythontk.core_utils.logging_mixin.table_mixin import TableMixin


class LoggingMixin(TableMixin):
    """Mixin class for logging utilities.

    Provides a logger for each class and a shared class logger across instances.
    Includes methods for setting log levels, adding handlers, and redirecting logs.
    """

    _logger: internal_logging.Logger = None
    _class_logger = None

    # Expose formatting constants
    LOG_COLORS = LoggerExt.LOG_COLORS
    HTML_PRESETS = LoggerExt.HTML_PRESETS
    log_link = staticmethod(LoggerExt._log_link)

    def __init__(
        self,
        *args,
        log_level: Optional[Union[int, str]] = None,
        log_file: Optional[str] = None,
        log_buffer: Union[bool, int, None] = None,
        **kwargs,
    ):
        """Forward *args/**kwargs to the next base; the ``log_*`` kwargs
        configure the CLASS-shared logger (``set_log_level`` /
        ``set_log_file`` / ``enable_log_buffer``), not this instance alone.

        Parameters:
            log_level: Level name or int applied via ``set_log_level``.
            log_file: Path to tee this class's records to.
            log_buffer: ``True`` (default capacity) or an int capacity to
                start capturing records into the ring buffer.
        """
        super().__init__(*args, **kwargs)
        if log_level is not None:
            self.set_log_level(log_level)
        if log_file is not None:
            self.set_log_file(log_file)
        if log_buffer:
            # log_buffer may be True (default capacity) or an int capacity.
            # bool is an int subclass, so test it first.
            if isinstance(log_buffer, int) and not isinstance(log_buffer, bool):
                self.enable_log_buffer(capacity=log_buffer)
            else:
                self.enable_log_buffer()

    @ClassProperty
    def logger(cls) -> internal_logging.Logger:
        """The patched logger for this class (one per class, created lazily).

        Built with the ``Logger()`` constructor, detached from the root
        (``propagate=False``, no parent) and NOT registered with the
        logging manager, so host logging config never reaches it.
        ``patch`` attaches the default stderr stream handler.
        """
        if cls.__dict__.get("_logger") is None:
            name = f"{cls.__module__}.{cls.__qualname__}"
            logger = internal_logging.Logger(name, internal_logging.NOTSET)
            logger.propagate = False
            logger.parent = None
            LoggerExt.patch(logger)
            cls._logger = logger

        return cls._logger

    def use_logger(self, logger: Optional[internal_logging.Logger]) -> None:
        """Route THIS instance's log output through *logger*.

        ``self.logger`` is normally the class-shared logger, so a reusable
        component (a manager, an engine) embedded in a host tool logs to its
        own channel -- invisible to the host's sinks (e.g. a panel text
        widget wired via ``setup_logging_redirect``). Adopting the host's
        logger sends every ``self.logger.*`` call on this instance through
        the host's handlers and level instead, while every other instance of
        the class keeps the shared class logger.

        Instance-scoped only: class-level entry points (``cls.set_log_level``,
        ``cls.set_log_file``, ...) still address the class logger. The
        override rides normal attribute lookup shadowing the non-data
        ``ClassProperty`` descriptor, so it requires an instance ``__dict__``.
        The adopted logger must support whatever this class calls on it: any
        stdlib logger covers the standard methods, but the ``LoggerExt``
        extras (``log_group``, ``log_link``, ``success``, ...) exist only on
        patched loggers such as another mixin class's ``.logger``.

        Parameters:
            logger: The host logger to adopt, or ``None`` to revert to the
                shared class logger.
        """
        if logger is None:
            self.__dict__.pop("logger", None)
        else:
            self.__dict__["logger"] = logger

    @ClassProperty
    def class_logger(cls) -> internal_logging.Logger:
        """A manager-registered sibling of ``logger`` (``<name>.class``).

        Unlike ``logger`` it goes through ``getLogger``, so it is visible
        to ``logging.getLogger(name)`` lookups and host logging config.
        """
        if cls.__dict__.get("_class_logger") is None:
            name = f"{cls.__module__}.{cls.__qualname__}.class"
            logger = internal_logging.getLogger(name)
            logger.setLevel(internal_logging.NOTSET)
            logger.propagate = False
            LoggerExt.patch(logger)
            cls._class_logger = logger
        return cls._class_logger

    @ClassProperty
    def logging(cls):
        """Access to Python's internal logging module (aliased)."""
        return internal_logging

    @classmethod
    def set_log_level(cls, level: int | str):
        """Set log level for the class logger and its handlers.

        Delegates to the patched ``logger.setLevel`` (``LoggerExt._set_level``),
        which resolves both standard and custom level NAMES
        (PROGRESS/SUCCESS/RESULT/NOTICE) via ``logging._nameToLevel`` and syncs
        every handler's level — except handlers whose level was explicitly
        pinned (``set_log_file`` / ``enable_log_buffer`` with an explicit
        level). The previous ``getattr(logging, name)`` lookup
        mapped those custom names to WARNING (the ``logging`` module has no such
        attributes — they live in ``_nameToLevel``), silently suppressing the
        very levels the caller asked to enable. Accessing ``cls.logger`` also
        guarantees the custom levels are registered before name resolution.
        """
        cls.logger.setLevel(level)

    @classmethod
    def set_log_file(
        cls, filename: Optional[str], level: Union[int, str] = internal_logging.NOTSET
    ) -> None:
        """Tee this class's log output to *filename* (or ``None`` to stop).

        Off by default; continuous once enabled. See ``LoggerExt._set_log_file``.
        Class-scoped, since ``logger`` is shared across instances.
        """
        cls.logger.set_log_file(filename, level)

    @classmethod
    def enable_log_buffer(
        cls, capacity: int = 2000, level: Union[int, str] = internal_logging.NOTSET
    ) -> None:
        """Capture this class's log records into a capped ring buffer.

        Near-zero cost until ``dump_log`` is called. See
        ``LoggerExt._enable_log_buffer``.
        """
        cls.logger.enable_log_buffer(capacity, level)

    @classmethod
    def disable_log_buffer(cls) -> None:
        """Stop ring-buffer capture and discard buffered records."""
        cls.logger.disable_log_buffer()

    @classmethod
    def clear_log_buffer(cls) -> None:
        """Drop buffered records but keep capturing."""
        cls.logger.clear_log_buffer()

    @classmethod
    def dump_log(
        cls,
        target: Union[str, object, None] = None,
        mode: str = "w",
        encoding: str = "utf-8",
    ) -> str:
        """Render the ring buffer to text, optionally writing it to *target*.

        *target* is a file path, a writable stream, or ``None`` (return only).
        Returns the rendered text. Requires ``enable_log_buffer`` first.
        """
        return cls.logger.dump_log(target, mode, encoding)
