# !/usr/bin/python
# coding=utf-8
"""App-agnostic line-stream primitives for launched processes and log files.

Three small, composable pieces (no app knowledge -- pair them with
:class:`pythontk.AppLauncher` / :class:`pythontk.HandoffBridge` app shells):

* :class:`OutputStream` -- thread-safe, multi-consumer ``(source, line)``
  pub/sub with a bounded replayable history. Consumers can react with
  either streaming (``stream.subscribe(callback)`` or
  ``for src, line in stream:``) or blocking (``stream.wait_for(pattern)``).
* :class:`ProcessReader` -- reads a ``subprocess.PIPE`` line-by-line on a
  background thread and pushes into an :class:`OutputStream`.
* :class:`LogTailer` -- polls a log file for new bytes on a background
  thread (rotation-aware) and pushes into an :class:`OutputStream`.
* :class:`TeeStream` -- a synchronous write-through fan-out that can stand in
  for ``sys.stdout`` / ``sys.stderr``, so output a process would otherwise only
  print can be recorded as well.

Extracted from the Substance Painter connection module: the mechanism is
generic process/log plumbing, so it lives here at the bottom of the stack;
the app-specific connection shells (Painter, Toolbag, ...) that compose it
live with their consumers (mayatk / blendertk / extapps).
"""
import os
import time
import queue
import threading
import collections
import logging
from typing import Callable, Iterator, List, Optional, Pattern, Tuple, Union

logger = logging.getLogger(__name__)


_DEFAULT_POLL_INTERVAL = 0.5
_DEFAULT_HISTORY = 5000


class TeeStream:
    """Write text to several streams at once; usable as a ``sys.stdout`` stand-in.

    The synchronous counterpart to :class:`OutputStream`: no threads, no
    subscribers, no line splitting — it passes each write straight through, so
    it can be assigned to ``sys.stdout`` / ``sys.stderr`` around a block whose
    output must be recorded as well as shown. Every ecosystem test runner does
    exactly that, which is why this lives here rather than being written out
    once per runner (the same reason ``StatusBadge`` does).

    Being a stand-in is what shapes the API: :meth:`write` returns a character
    count like a real stream, :meth:`writelines` is defined rather than
    delegated (delegating it would reach the first stream only, silently
    skipping the rest), and every other attribute — ``isatty``, ``fileno``,
    ``encoding`` — is answered by the FIRST stream, which is the real one. Pass
    the real console first and the recorder second.

    Two write failures are contained rather than raised, because a stand-in
    that raises turns an unrelated ``print`` into a test failure: a cp1252
    console cannot encode every character a docstring or log line contains, and
    is retried with replacements; a detached or closed stream (``ValueError`` /
    ``OSError``) is skipped for that write, leaving the other streams intact.
    Anything else propagates.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            try:
                stream.write(text)
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", None) or "ascii"
                stream.write(text.encode(encoding, "replace").decode(encoding))
            except (ValueError, OSError):
                pass  # detached/closed — the other streams still get it
        return len(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except (ValueError, OSError):
                pass

    def __getattr__(self, name):
        # Only reached for attributes this class does not define; `streams` is
        # set in __init__, so there is no recursion here.
        return getattr(self.streams[0], name)


class OutputStream:
    """Thread-safe, multi-consumer text stream with bounded history.

    Each call to :meth:`subscribe` / :meth:`__iter__` / :meth:`wait_for`
    gets its own queue, so multiple consumers can read independently.

    Records are ``(source, line)`` tuples — ``source`` labels like
    ``"stdout"``, ``"stderr"``, ``"log"`` let consumers filter.

    A bounded ring buffer of recent lines is kept so that consumers which
    subscribe after the stream has started can optionally replay history.
    This closes the start-up race where lines pushed between launch and
    the first ``wait_for`` would otherwise be missed.
    """

    def __init__(self, history: int = _DEFAULT_HISTORY):
        self._lock = threading.Lock()
        self._subscribers: List[Callable[[str, str], None]] = []
        self._history: "collections.deque[Tuple[str, str]]" = collections.deque(
            maxlen=history
        )
        self._closed = False

    def push(self, line: str, source: str = "") -> None:
        """Append a line. Called by readers; consumers should not invoke."""
        if self._closed:
            return
        with self._lock:
            self._history.append((source, line))
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(source, line)
            except Exception:
                logger.exception("OutputStream subscriber raised")

    def subscribe(
        self,
        callback: Callable[[str, str], None],
        replay_history: bool = False,
    ) -> Callable[[], None]:
        """Register ``callback(source, line)``.

        If *replay_history* is True, every buffered line is delivered to the
        callback under the same lock that registers it — so no line is lost
        or duplicated relative to future pushes. Because that lock is not
        reentrant, the callback must not call back into this stream
        (``push``/``history``/``clear_history``/``subscribe``) during replay —
        doing so deadlocks. Buffer and defer any such work until replay ends.

        Returns an unsubscribe handle.
        """
        with self._lock:
            if replay_history:
                for src, line in self._history:
                    try:
                        callback(src, line)
                    except Exception:
                        logger.exception("OutputStream replay raised")
            self._subscribers.append(callback)

        def _unsubscribe():
            with self._lock:
                try:
                    self._subscribers.remove(callback)
                except ValueError:
                    pass

        return _unsubscribe

    def history(self) -> List[Tuple[str, str]]:
        """Snapshot the current history buffer."""
        with self._lock:
            return list(self._history)

    def clear_history(self) -> None:
        """Drop buffered lines. Future pushes are unaffected."""
        with self._lock:
            self._history.clear()

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        """Yield buffered + future ``(source, line)`` until the stream is closed."""
        q: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        unsubscribe = self.subscribe(
            lambda src, line: q.put((src, line)), replay_history=True
        )
        try:
            while True:
                try:
                    yield q.get(timeout=_DEFAULT_POLL_INTERVAL)
                except queue.Empty:
                    if self._closed:
                        return
        finally:
            unsubscribe()

    def wait_for(
        self,
        pattern: Union[str, Pattern],
        timeout: Optional[float] = None,
        source: Optional[str] = None,
        include_history: bool = True,
    ) -> Optional[Tuple[str, str]]:
        """Block until a line matches *pattern*, or *timeout* expires.

        Parameters:
            pattern: Substring (``str``) or compiled ``re.Pattern``.
            timeout: Seconds; ``None`` means no limit.
            source: If given, only consider lines from this source.
            include_history: If True (default), buffered lines are checked
                before waiting for new ones. Set False to ignore history
                and only match future events.

        Returns:
            ``(source, line)`` tuple, or ``None`` on timeout / stream closure.
        """
        if isinstance(pattern, str):

            def matches(line: str) -> bool:
                return pattern in line

        else:

            def matches(line: str) -> bool:
                return bool(pattern.search(line))

        q: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        unsubscribe = self.subscribe(
            lambda src, line: q.put((src, line)),
            replay_history=include_history,
        )
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        try:
            while True:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    timeout_chunk = min(remaining, _DEFAULT_POLL_INTERVAL)
                else:
                    timeout_chunk = _DEFAULT_POLL_INTERVAL
                try:
                    src, line = q.get(timeout=timeout_chunk)
                except queue.Empty:
                    if self._closed:
                        return None
                    continue
                if source is not None and src != source:
                    continue
                if matches(line):
                    return (src, line)
        finally:
            unsubscribe()

    def close(self) -> None:
        """Mark the stream closed. Pending iterators and waiters will exit."""
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


class ProcessReader(threading.Thread):
    """Reads a subprocess pipe line-by-line into an :class:`OutputStream`.

    The pipe must be BINARY mode (``Popen`` without ``text=True``/``encoding``):
    the reader uses a ``b""`` sentinel and decodes utf-8 itself. A text-mode
    pipe raises in the reader thread and silently ends the stream early.
    """

    def __init__(self, pipe, target: OutputStream, source: str):
        super().__init__(daemon=True, name=f"stream-{source}-reader")
        self._pipe = pipe
        self._target = target
        self._source = source

    def run(self) -> None:
        try:
            for raw in iter(self._pipe.readline, b""):
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                self._target.push(line, source=self._source)
        finally:
            try:
                self._pipe.close()
            except Exception:
                pass


class LogTailer(threading.Thread):
    """Tails a log file from its current size forward.

    Handles log rotation: if the file shrinks (rotated), reads from 0.
    """

    def __init__(
        self,
        log_path: str,
        target: OutputStream,
        source: str = "log",
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        tail_from_start: bool = False,
    ):
        super().__init__(daemon=True, name="stream-log-tailer")
        self._path = log_path
        self._target = target
        self._source = source
        self._poll = poll_interval
        self._tail_from_start = tail_from_start
        self._stop_event = threading.Event()
        self._pending = b""

    def stop(self) -> None:
        self._stop_event.set()

    @staticmethod
    def _file_id(path: str) -> Optional[int]:
        try:
            return os.stat(path).st_ino
        except OSError:
            return None

    def run(self) -> None:
        if self._tail_from_start:
            position = 0
        else:
            position = (
                os.path.getsize(self._path) if os.path.exists(self._path) else 0
            )
        last_id = self._file_id(self._path)
        while not self._stop_event.is_set():
            try:
                if os.path.exists(self._path):
                    current_id = self._file_id(self._path)
                    if current_id is not None and current_id != last_id:
                        # File was replaced (rotated via rename or delete+recreate).
                        position = 0
                        self._pending = b""
                        last_id = current_id
                    size = os.path.getsize(self._path)
                    if size < position:
                        position = 0
                        self._pending = b""
                    if size > position:
                        with open(self._path, "rb") as f:
                            f.seek(position)
                            chunk = f.read(size - position)
                            position = f.tell()
                        self._emit(chunk)
                else:
                    last_id = None
            except OSError:
                logger.debug("LogTailer read error on %s", self._path, exc_info=True)
            self._stop_event.wait(timeout=self._poll)

    def _emit(self, chunk: bytes) -> None:
        data = self._pending + chunk
        *complete, self._pending = data.split(b"\n")
        for raw in complete:
            line = raw.decode("utf-8", errors="replace").rstrip("\r")
            if line:
                self._target.push(line, source=self._source)


__all__ = ["OutputStream", "ProcessReader", "LogTailer"]
