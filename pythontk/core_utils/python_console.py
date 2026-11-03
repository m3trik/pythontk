# !/usr/bin/python
# coding=utf-8
"""PythonConsole -- run code a snippet at a time, in one namespace that lasts.

An agent drives an application's Python the way a person drives its script
editor: one snippet at a time, each seeing what the ones before it defined.
:meth:`PythonConsole.run` runs one snippet in this process's console namespace
and hands back the value of its last expression and what it printed. The host
operations ``maya.python.exec`` and ``blender.python.exec`` declare it, so in
an instance a session keeps (:class:`~pythontk.SessionOperationHost`) the
namespace carries from call to call, and in a fresh instance it lasts one call.

It has no dry run: nothing can be known of code before it runs, so its
operations are run where running them was allowed (an MCP client's change
server), never planned. A cancel interrupts it as Ctrl+C does in a console
(the batch host's job), so the namespace outlives a stopped loop.

Adopted, not built: the interactive interpreter's rule for a value (the last
statement, when it is an expression, is evaluated and shown), with ``ast``.
"""

import ast
import contextlib
import io
import json
import traceback
from typing import Any, Dict, Optional, Tuple

from pythontk.core_utils.cancel_scope import OperationCancelled


class _PythonConsoleInternal:
    """Internal helpers for :class:`PythonConsole`."""

    #: What a snippet's frames are called in a traceback.
    FILENAME = "<console>"
    #: The most characters of output handed back; a longer output keeps its end.
    OUTPUT_MAX = 20000
    #: The most characters a value may take as JSON; a bigger one comes back
    #: as the start of its ``repr``, so one expression cannot flood a client.
    VALUE_MAX = 100000

    #: The namespace every snippet runs in, for the life of the process.
    _namespace: Dict[str, Any] = {"__name__": "__main__"}

    @classmethod
    def _compiled(cls, code: str) -> Tuple[Any, Optional[Any]]:
        """``(body, last)``: *code* compiled, its last statement split off and
        compiled for its value when it is an expression (else ``None``).

        Raises:
            SyntaxError: *code* does not parse.
        """
        tree = ast.parse(code, cls.FILENAME, "exec")
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = compile(ast.Expression(tree.body.pop().value), cls.FILENAME, "eval")
        return compile(tree, cls.FILENAME, "exec"), last

    @classmethod
    def _clip(cls, text: str) -> str:
        """*text*, its end kept when it is longer than :attr:`OUTPUT_MAX`."""
        if len(text) <= cls.OUTPUT_MAX:
            return text
        cut = len(text) - cls.OUTPUT_MAX
        return f"[the first {cut} characters were cut]\n{text[cut:]}"

    @classmethod
    def _bounded(cls, value: Any) -> Any:
        """*value*, or the start of its ``repr`` when its JSON would pass :attr:`VALUE_MAX`."""
        try:
            size = len(json.dumps(value, default=repr))
        except (TypeError, ValueError, RecursionError):
            size = cls.VALUE_MAX + 1
        if size <= cls.VALUE_MAX:
            return value
        text = repr(value)
        return f"{text[: cls.VALUE_MAX]}... [{type(value).__name__} cut: {size} characters as JSON]"

    @classmethod
    def _traceback(cls, exc: BaseException) -> str:
        """*exc*'s traceback from the snippet's own frames on (none of this module's)."""
        if isinstance(exc, SyntaxError):
            return "".join(traceback.format_exception_only(type(exc), exc))
        tb = exc.__traceback__
        while tb is not None and tb.tb_frame.f_code.co_filename != cls.FILENAME:
            tb = tb.tb_next
        return "".join(
            traceback.format_exception(type(exc), exc, tb or exc.__traceback__)
        )


class PythonConsole(_PythonConsoleInternal):
    """A console namespace this process keeps: each snippet sees what earlier ones defined."""

    @classmethod
    def run(cls, code: str) -> Dict[str, Any]:
        """Run *code* in the console namespace, as the next snippet typed into a console.

        Parameters:
            code: Python source: statements, then optionally an expression
                whose value comes back. Names it defines stay for later calls.

        Returns:
            ``{"value": ..., "output": ...}``: the last expression's value
            (``None`` when the code ends in a statement; the start of its
            ``repr`` when huge) and what the code printed (stdout and stderr,
            its end kept when long).

        Raises:
            SyntaxError: The code did not parse, so none of it ran.
            RuntimeError: The code raised: the message is its traceback, then
                what it printed first.
        """
        try:
            body, last = cls._compiled(code)
        except SyntaxError as exc:
            raise SyntaxError(cls._traceback(exc).rstrip()) from None
        except (ValueError, MemoryError, RecursionError) as exc:  # NUL bytes, a too-deep nest
            raise SyntaxError(f"{type(exc).__name__}: {exc}") from None
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                exec(body, cls._namespace)
                value = eval(last, cls._namespace) if last is not None else None
        except OperationCancelled:
            raise
        except (Exception, SystemExit) as exc:
            # SystemExit too: a snippet's sys.exit() must never end the process serving it.
            printed = cls._clip(buffer.getvalue()).rstrip()
            message = cls._traceback(exc).rstrip()
            if printed:
                message += f"\n--- it printed first ---\n{printed}"
            raise RuntimeError(message) from None
        return {"value": cls._bounded(value), "output": cls._clip(buffer.getvalue())}
