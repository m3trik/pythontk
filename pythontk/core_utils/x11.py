# !/usr/bin/python
# coding=utf-8
"""A minimal, fully typed ctypes Xlib client: windows, pointer, focus, keys."""

import contextlib
import ctypes
import threading
from typing import Iterable, List, Optional, Tuple


class _X11Internal:
    """Connection, typing and property plumbing behind :class:`X11`."""

    #: libX11 handle and Display*: None until first tried, False when that
    #: failed (not retried -- several of these are polled every few ms).
    _lib = None
    _display = None
    #: Setup and EVERY Xlib call hold this: callers poll from several threads,
    #: and Xlib is not thread-safe on a shared Display without XInitThreads
    #: (which only the process's first Xlib user may call).
    _lock = threading.RLock()
    _atoms = {}
    _error_handler = None
    _previous_handler = None

    _XA_CARDINAL = 6
    _XA_STRING = 31
    _XA_WINDOW = 33
    _ANY_PROPERTY_TYPE = 0
    _IS_VIEWABLE = 2

    #: ``int (*XErrorHandler)(Display*, XErrorEvent*)``
    _ERROR_HANDLER_TYPE = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
    )

    class _WindowAttributes(ctypes.Structure):
        """``XWindowAttributes`` -- read only for ``map_state``, but the whole
        struct must be declared for Xlib to write into it."""

        _fields_ = [
            ("x", ctypes.c_int),
            ("y", ctypes.c_int),
            ("width", ctypes.c_int),
            ("height", ctypes.c_int),
            ("border_width", ctypes.c_int),
            ("depth", ctypes.c_int),
            ("visual", ctypes.c_void_p),
            ("root", ctypes.c_ulong),
            ("c_class", ctypes.c_int),
            ("bit_gravity", ctypes.c_int),
            ("win_gravity", ctypes.c_int),
            ("backing_store", ctypes.c_int),
            ("backing_planes", ctypes.c_ulong),
            ("backing_pixel", ctypes.c_ulong),
            ("save_under", ctypes.c_int),
            ("colormap", ctypes.c_ulong),
            ("map_installed", ctypes.c_int),
            ("map_state", ctypes.c_int),
            ("all_event_masks", ctypes.c_long),
            ("your_event_mask", ctypes.c_long),
            ("do_not_propagate_mask", ctypes.c_long),
            ("override_redirect", ctypes.c_int),
            ("screen", ctypes.c_void_p),
        ]

    @classmethod
    def _load(cls):
        """libX11, every function typed before the handle is published.

        Untyped, ctypes passes a ``Display*`` or ``Window`` as a 32-bit C int:
        the truncated pointer segfaults the process wherever libX11 allocates
        above 4 GB (any worker thread's arena).
        """
        if cls._lib is None:
            cls._lib = False  # a failed load is not retried
            lib = ctypes.cdll.LoadLibrary("libX11.so.6")
            vp, ul, ui, i, cp = (
                ctypes.c_void_p,
                ctypes.c_ulong,
                ctypes.c_uint,
                ctypes.c_int,
                ctypes.c_char_p,
            )
            P = ctypes.POINTER
            signatures = {
                "XOpenDisplay": ([cp], vp),
                "XDefaultRootWindow": ([vp], ul),
                "XDefaultScreen": ([vp], i),
                "XInternAtom": ([vp, cp, i], ul),
                "XGetSelectionOwner": ([vp, ul], ul),
                "XGetWindowProperty": (
                    [vp, ul, ul, ctypes.c_long, ctypes.c_long, i, ul]
                    + [P(ul), P(i), P(ul), P(ul), P(vp)],
                    i,
                ),
                "XQueryTree": ([vp, ul, P(ul), P(ul), P(P(ul)), P(ui)], i),
                "XGetWindowAttributes": ([vp, ul, P(cls._WindowAttributes)], i),
                "XQueryPointer": (
                    [vp, ul, P(ul), P(ul), P(i), P(i), P(i), P(i), P(ui)],
                    i,
                ),
                "XKeysymToKeycode": ([vp, ul], ctypes.c_ubyte),
                "XQueryKeymap": ([vp, ctypes.c_char * 32], i),
                "XFree": ([vp], i),
                "XSetErrorHandler": (
                    [cls._ERROR_HANDLER_TYPE],
                    cls._ERROR_HANDLER_TYPE,
                ),
            }
            for name, (argtypes, restype) in signatures.items():
                fn = getattr(lib, name)
                fn.argtypes, fn.restype = argtypes, restype
            cls._lib = lib
        return cls._lib

    @classmethod
    def _connect(cls):
        """The Display*, or False where there is no X server (cached)."""
        if cls._display is None:
            lib = cls._load()
            # NULL -- no DISPLAY, pure Wayland, SSH -- caches as False: an open
            # against an unreachable server can block.
            cls._display = lib.XOpenDisplay(None) or False
        return cls._display

    @classmethod
    @contextlib.contextmanager
    def _session(cls):
        """Yield ``(lib, display)`` under the lock, or None without a server.

        Xlib's DEFAULT error handler exits the process, and a window can vanish
        between being listed and being read (``BadWindow``): while we talk to
        the server our handler absorbs errors on OUR connection and hands any
        other connection's to whatever handler was installed before.
        """
        with cls._lock:
            try:
                display = cls._connect()
            except Exception:  # no libX11, or it lacks a symbol
                display = None
            if not display:
                yield None
                return
            lib = cls._lib
            if cls._error_handler is None:

                def on_error(dpy, event):
                    if dpy != cls._display and cls._previous_handler:
                        return cls._previous_handler(dpy, event)
                    return 0

                cls._error_handler = cls._ERROR_HANDLER_TYPE(on_error)
            cls._previous_handler = lib.XSetErrorHandler(cls._error_handler)
            try:
                yield lib, display
            finally:
                lib.XSetErrorHandler(cls._previous_handler)
                cls._previous_handler = None

    @classmethod
    def _atom(cls, lib, display, name: str) -> int:
        if name not in cls._atoms:
            cls._atoms[name] = lib.XInternAtom(display, name.encode(), 0)
        return cls._atoms[name]

    @classmethod
    def _property(cls, lib, display, window, name, req_type, max_items=1 << 16):
        """A window property: format-32 items as a list of ints, format-8 as
        bytes, or None when the window lacks it (or is gone)."""
        actual_type = ctypes.c_ulong()
        fmt = ctypes.c_int()
        count = ctypes.c_ulong()
        remaining = ctypes.c_ulong()
        data = ctypes.c_void_p()
        status = lib.XGetWindowProperty(
            display,
            window,
            cls._atom(lib, display, name),
            0,
            max_items,
            0,
            req_type,
            ctypes.byref(actual_type),
            ctypes.byref(fmt),
            ctypes.byref(count),
            ctypes.byref(remaining),
            ctypes.byref(data),
        )
        if status != 0 or not data.value:
            return None
        try:
            if fmt.value == 32:  # delivered as C longs, whatever the wire size
                items = ctypes.cast(data, ctypes.POINTER(ctypes.c_ulong))
                return [items[k] for k in range(count.value)]
            if fmt.value == 8:
                return ctypes.string_at(data, count.value)
            return None
        finally:
            lib.XFree(data)

    @classmethod
    def _top_level_windows(cls, lib, display) -> List[int]:
        """The clients a window manager lists (EWMH ``_NET_CLIENT_LIST``); on a
        bare server with no manager, the root's viewable children."""
        root = lib.XDefaultRootWindow(display)
        clients = cls._property(lib, display, root, "_NET_CLIENT_LIST", cls._XA_WINDOW)
        if clients is not None:
            return clients
        root_ret, parent = ctypes.c_ulong(), ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        n = ctypes.c_uint()
        if not lib.XQueryTree(
            display,
            root,
            ctypes.byref(root_ret),
            ctypes.byref(parent),
            ctypes.byref(children),
            ctypes.byref(n),
        ):
            return []
        try:
            windows = [children[k] for k in range(n.value)]
        finally:
            if children:
                lib.XFree(ctypes.cast(children, ctypes.c_void_p))
        viewable = []
        for window in windows:
            attrs = cls._WindowAttributes()
            if (
                lib.XGetWindowAttributes(display, window, ctypes.byref(attrs))
                and attrs.map_state == cls._IS_VIEWABLE
            ):
                viewable.append(window)
        return viewable

    @classmethod
    def _window_pid(cls, lib, display, window) -> Optional[int]:
        pid = cls._property(lib, display, window, "_NET_WM_PID", cls._XA_CARDINAL, 1)
        return int(pid[0]) if pid else None

    @classmethod
    def _window_title(cls, lib, display, window) -> str:
        utf8 = cls._atom(lib, display, "UTF8_STRING")
        raw = cls._property(lib, display, window, "_NET_WM_NAME", utf8)
        if raw:
            return raw.decode("utf-8", "replace")
        raw = cls._property(lib, display, window, "WM_NAME", cls._ANY_PROPERTY_TYPE)
        return raw.decode("latin-1", "replace") if isinstance(raw, bytes) else ""


class X11(_X11Internal):
    """What an X server can say about windows, the pointer, focus and keys.

    Linux desktops run X11 or Wayland. Under Wayland, X11 clients -- Maya, an
    xcb app, among them -- run on XWayland, and see only XWayland windows.
    Every query answers ``None`` when there is no X server to ask (Windows,
    macOS, pure Wayland, SSH), so callers keep their own fallback. One
    connection per process, opened on first use and never re-tried once it
    failed.
    """

    @classmethod
    def available(cls) -> bool:
        """True when an X server answers."""
        with cls._session() as session:
            return session is not None

    @classmethod
    def window_titles(cls, pids: Iterable[int]) -> Optional[List[str]]:
        """Titles of the visible top-level windows owned by any of *pids*
        (``_NET_WM_PID``); untitled windows are skipped. None without a server.
        """
        wanted = set(pids)
        with cls._session() as session:
            if session is None:
                return None
            lib, display = session
            titles = []
            for window in cls._top_level_windows(lib, display):
                if cls._window_pid(lib, display, window) in wanted:
                    title = cls._window_title(lib, display, window)
                    if title:
                        titles.append(title)
            return titles

    @classmethod
    def active_window_pid(cls) -> Optional[int]:
        """PID owning the focused window (EWMH ``_NET_ACTIVE_WINDOW``), or None
        when there is no server, no window manager, or no focused client."""
        with cls._session() as session:
            if session is None:
                return None
            lib, display = session
            root = lib.XDefaultRootWindow(display)
            active = cls._property(
                lib, display, root, "_NET_ACTIVE_WINDOW", cls._XA_WINDOW, 1
            )
            if not active or not active[0]:
                return None
            return cls._window_pid(lib, display, active[0])

    @classmethod
    def pointer(cls) -> Optional[Tuple[int, int]]:
        """The pointer's root-window (screen) position, or None."""
        with cls._session() as session:
            if session is None:
                return None
            lib, display = session
            root_ret, child = ctypes.c_ulong(), ctypes.c_ulong()
            rx, ry, wx, wy = (ctypes.c_int() for _ in range(4))
            mask = ctypes.c_uint()
            if not lib.XQueryPointer(
                display,
                lib.XDefaultRootWindow(display),
                *(ctypes.byref(v) for v in (root_ret, child, rx, ry, wx, wy, mask)),
            ):
                return None
            return rx.value, ry.value

    @classmethod
    def has_compositor(cls) -> Optional[bool]:
        """True when a compositing manager owns ``_NET_WM_CM_S<screen>`` --
        without one, translucent windows paint opaque (black). None without a
        server."""
        with cls._session() as session:
            if session is None:
                return None
            lib, display = session
            screen = lib.XDefaultScreen(display)
            atom = cls._atom(lib, display, f"_NET_WM_CM_S{screen}")
            return bool(lib.XGetSelectionOwner(display, atom))

    @classmethod
    def key_down(cls, keysym: int) -> Optional[bool]:
        """Whether the key with *keysym* (``0xFF1B`` = Escape) is held right
        now, anywhere on the server (``XQueryKeymap`` ignores focus). None
        without a server."""
        with cls._session() as session:
            if session is None:
                return None
            lib, display = session
            keycode = lib.XKeysymToKeycode(display, keysym)
            if not keycode:
                return False
            keys = (ctypes.c_char * 32)()
            lib.XQueryKeymap(display, keys)
            byte = keys[keycode // 8]  # a c_char item is a length-1 bytes
            value = byte[0] if isinstance(byte, bytes) else byte
            return bool(value & (1 << (keycode % 8)))
