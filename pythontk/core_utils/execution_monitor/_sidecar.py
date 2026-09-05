# !/usr/bin/python
# coding=utf-8
"""Sidecar processes for ``ExecutionMonitor``: indicator, dialog and watchdog.

Run as a script (``python _sidecar.py <command> ...``) under the interpreter
``ExecutionMonitor._get_python_executable`` resolves. A separate process is the
point: it animates, answers and kills while the host's main thread is blocked
inside the operation being watched.

Self-contained on purpose: no ``pythontk`` import (the interpreter may be a
DCC's bundled python with no site access to it) and ``tkinter`` imported
lazily inside the UI commands (the watchdog runs headless, where Tk may be
absent).

Every window watches its parent (``--parent-pid``) and closes itself when the
parent is gone: a borderless topmost overlay has no close button, so an orphan
left behind by a host crash could only be removed from the task manager.

Commands:
    indicator [--gif PATH] [--size N] [--pos=x,y] [--parent-pid N]
    dialog [--force-label L] [--parent-pid N] -- TITLE MESSAGE
    watchdog PID HEARTBEAT --timeout S [--check-interval S] [--kill-tree]
             --stop-file PATH

Dialog exit codes (the monitor reads these):
    0  Keep Waiting        2  Force action (only with ``--force-label``)
    10 Cancel              3  Window closed (treated as Keep Waiting)
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time

CHROMA_KEY = "#f0f0f0"
INDICATOR_SIZE = 48
GIF_SIZE = 128
NUM_DOTS = 12
FRAME_MS = 80
PARENT_POLL_MS = 500
DIALOG_MIN_WIDTH = 450
DIALOG_WRAP = 350

DIALOG_KEEP_WAITING = 0
DIALOG_FORCE = 2
DIALOG_CLOSED = 3
DIALOG_CANCEL = 10

_kernel32 = None


def _win_kernel32():
    global _kernel32
    if _kernel32 is None:
        import ctypes

        _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    return _kernel32


# --------------------------------------------------------------------------- processes
def process_alive(pid: int) -> bool:
    """True while *pid* is running.

    Windows: wait on the process object with a zero timeout. Merely opening a
    handle is not evidence of life — the object outlives the process for as
    long as anyone (a ``Popen`` in the parent) still holds a handle to it.
    """
    if sys.platform == "win32":
        import ctypes

        kernel32 = _win_kernel32()
        synchronize = 0x00100000
        wait_timeout = 0x102
        error_access_denied = 5
        handle = kernel32.OpenProcess(synchronize, False, int(pid))
        if not handle:
            # A process we may not open (another user, elevated) is still there.
            return ctypes.get_last_error() == error_access_denied
        try:
            return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _children_posix(pid: int) -> list:
    """Direct children of *pid* from ``/proc`` (Linux)."""
    kids = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return kids
    for name in entries:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as f:
                stat = f.read()
            # "pid (comm) state ppid ..." — comm may contain spaces or ')',
            # so split after the LAST ')'.
            ppid = int(stat.rsplit(b")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if ppid == pid:
            kids.append(int(name))
    return kids


def _descendants_posix(pid: int) -> list:
    found, stack = [], [pid]
    while stack:
        for child in _children_posix(stack.pop()):
            if child not in found:
                found.append(child)
                stack.append(child)
    return found


#: Seconds ``kill_process`` waits for ``taskkill`` before giving up. Generous
#: for a real kill (a deep tree takes a second or two) and short enough that a
#: wedged one cannot outlive the run that spawned it.
KILL_TIMEOUT = 15.0


def kill_process(pid: int, tree: bool = True) -> None:
    """Force-kill *pid* (and, with *tree*, its descendants).

    POSIX walks ``/proc`` for the descendants rather than killing the process
    group: a host launched from a terminal shares the shell's group, so
    ``killpg`` would have taken the user's terminal session down with it.
    """
    if sys.platform == "win32":
        cmd = ["taskkill", "/PID", str(pid), "/F"] + (["/T"] if tree else [])
        try:
            # Bounded: run_watchdog returns as soon as this does, so a
            # taskkill that never comes back -- an unkillable target stuck in
            # a driver call, a wedged WMI -- would leave the watchdog process
            # alive forever, which is the failure the sidecar exists to
            # prevent. On timeout subprocess.run kills the taskkill child and
            # raises, and the except below absorbs it: the target may survive,
            # but a best-effort kill that gave up beats a watchdog that hangs.
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=KILL_TIMEOUT,
            )
        except Exception:
            pass
        return
    # Resolve the tree BEFORE the first kill: orphans get reparented away.
    targets = [pid] + (_descendants_posix(pid) if tree else [])
    for target in targets:
        try:
            os.kill(target, signal.SIGKILL)
        except OSError:
            pass


def run_watchdog(
    pid: int,
    heartbeat_path: str,
    timeout: float,
    check_interval: float = 1.0,
    kill_tree: bool = True,
    stop_file: str | None = None,
) -> int:
    """Kill *pid* when *heartbeat_path* stops being touched for *timeout* seconds.

    Exits on its own when the process is gone or *stop_file* appears. A missing
    heartbeat gets the same grace period as a stale one, measured from start.
    """
    started = time.time()
    while True:
        if stop_file and os.path.exists(stop_file):
            return 0
        if not process_alive(pid):
            return 0
        now = time.time()
        try:
            age = now - os.path.getmtime(heartbeat_path)
        except OSError:
            age = now - started
        if age > timeout:
            kill_process(pid, kill_tree)
            return 0
        time.sleep(check_interval)


# --------------------------------------------------------------------------- tk helpers
def watch_parent(root, pid: int | None, on_gone=None) -> None:
    """Close *root* (or call *on_gone*) once process *pid* has exited."""
    if not pid:
        return

    def poll():
        if process_alive(pid):
            root.after(PARENT_POLL_MS, poll)
        else:
            (on_gone or root.destroy)()

    root.after(PARENT_POLL_MS, poll)


def _place(root, width: int, height: int, pos=None) -> None:
    """Position a *width* x *height* window centred on *pos*, else on screen."""
    if pos:
        x, y = pos[0] - width // 2, pos[1] - height // 2
    else:
        x = (root.winfo_screenwidth() - width) // 2
        y = (root.winfo_screenheight() - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")


def _overlay_window():
    """A borderless, topmost, chroma-keyed Tk root for an overlay."""
    import tkinter as tk

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    if sys.platform == "win32":
        root.attributes("-transparentcolor", CHROMA_KEY)
    else:
        root.attributes("-alpha", 0.9)
    root.configure(bg=CHROMA_KEY)
    return root


def _load_gif_frames(gif_path: str, size: int):
    """(frames, frame_ms) for *gif_path* scaled to *size*; PIL when available
    (real resampling + the file's own timing), else Tk's native decoder."""
    import tkinter as tk

    try:
        from PIL import Image, ImageSequence, ImageTk

        image = Image.open(gif_path)
        duration = int(image.info.get("duration", 100)) or 100
        frames = [
            ImageTk.PhotoImage(
                frame.convert("RGBA").resize((size, size), Image.LANCZOS)
            )
            for frame in ImageSequence.Iterator(image)
        ]
        if frames:
            return frames, duration
    except Exception:
        pass

    frames = []
    while True:
        try:
            frames.append(
                tk.PhotoImage(file=gif_path, format=f"gif -index {len(frames)}")
            )
        except tk.TclError:
            break
    if frames and frames[0].width() > size:
        factor = max(1, round(frames[0].width() / size))
        frames = [frame.subsample(factor) for frame in frames]
    return frames, 100


def _spinner(root, size: int):
    """Ring of fading dots — resolution-independent, no image file needed."""
    import tkinter as tk

    canvas = tk.Canvas(
        root, width=size, height=size, bg=CHROMA_KEY, highlightthickness=0
    )
    canvas.pack()
    cx = cy = size / 2
    radius = size * 0.35
    dot_r = max(2, size * 0.07)
    dots = []
    for i in range(NUM_DOTS):
        angle = 2 * math.pi * i / NUM_DOTS
        dx, dy = cx + radius * math.cos(angle), cy + radius * math.sin(angle)
        dots.append(
            canvas.create_oval(
                dx - dot_r,
                dy - dot_r,
                dx + dot_r,
                dy + dot_r,
                fill=CHROMA_KEY,
                outline="",
            )
        )

    def color(brightness: float) -> str:
        if brightness <= 0.05:
            return CHROMA_KEY
        v = int(200 * (1 - brightness))  # 200 light gray → 0 black
        return f"#{v:02x}{v:02x}{v:02x}"

    def update(step=0):
        for i, dot in enumerate(dots):
            offset = (step - i) % NUM_DOTS  # steps behind the head
            canvas.itemconfigure(
                dot, fill=color(max(0.0, 1.0 - offset / (NUM_DOTS * 0.5)))
            )
        root.after(FRAME_MS, update, (step + 1) % NUM_DOTS)

    root.after(0, update)


def _gif(root, frames, frame_ms: int):
    import tkinter as tk

    label = tk.Label(root, image=frames[0], bg=CHROMA_KEY)
    label.pack()

    def update(index=0):
        label.configure(image=frames[index])
        root.after(frame_ms, update, (index + 1) % len(frames))

    root.after(0, update)


def run_indicator(
    size: int | None = None,
    pos=None,
    gif_path: str | None = None,
    parent_pid: int | None = None,
) -> int:
    """Show the busy indicator near *pos* until killed or the parent exits."""
    root = _overlay_window()
    frames = []
    if gif_path:
        frames, frame_ms = _load_gif_frames(gif_path, size or GIF_SIZE)
    if frames:
        width, height = frames[0].width(), frames[0].height()
        _gif(root, frames, frame_ms)
    else:
        width = height = size or INDICATOR_SIZE
        _spinner(root, width)
    _place(root, width, height, pos)
    watch_parent(root, parent_pid)
    root.mainloop()
    return 0


def build_dialog(
    root, title: str, message: str, force_label: str | None = None
) -> list:
    """Populate *root* with the VS Code-style dialog; returns the result holder
    (a one-element list the button handlers write the exit code into)."""
    import tkinter as tk
    from tkinter import ttk

    root.title(title)
    root.attributes("-topmost", True)
    root.resizable(False, False)
    result = [DIALOG_CLOSED]

    def choose(code):
        def handler(event=None):
            result[0] = code
            root.destroy()

        return handler

    root.protocol("WM_DELETE_WINDOW", choose(DIALOG_CLOSED))

    main_frame = ttk.Frame(root, padding="20 15 20 15")
    main_frame.pack(fill=tk.BOTH, expand=True)

    msg_frame = ttk.Frame(main_frame)
    msg_frame.pack(fill=tk.X, pady=(0, 15))
    ttk.Label(msg_frame, text="⚠", font=("Segoe UI", 24), foreground="#E8A317").pack(
        side=tk.LEFT, padx=(0, 15)
    )
    ttk.Label(
        msg_frame,
        text=message,
        wraplength=DIALOG_WRAP,
        justify=tk.LEFT,
        font=("Segoe UI", 10),
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
    ttk.Label(btn_frame).pack(side=tk.LEFT, expand=True)  # pushes the buttons right
    if force_label:
        ttk.Button(
            btn_frame, text=force_label, command=choose(DIALOG_FORCE), width=12
        ).pack(side=tk.LEFT, padx=(0, 8))
    ttk.Button(btn_frame, text="Cancel", command=choose(DIALOG_CANCEL), width=12).pack(
        side=tk.LEFT, padx=(0, 8)
    )
    keep = ttk.Button(
        btn_frame, text="Keep Waiting", command=choose(DIALOG_KEEP_WAITING), width=14
    )
    keep.pack(side=tk.LEFT)
    keep.focus_set()

    root.bind("<Return>", choose(DIALOG_KEEP_WAITING))
    root.bind("<Escape>", choose(DIALOG_CANCEL))
    return result


def fit_and_center(root) -> None:
    """Size the dialog to its content and centre it on screen.

    Never a fixed height: the packer silently drops widgets that do not fit,
    and the fullest message (Esc hint + no-checkpoint note + a force button)
    used to lose the entire button row that way.
    """
    root.update_idletasks()
    width = max(DIALOG_MIN_WIDTH, root.winfo_reqwidth())
    height = root.winfo_reqheight()
    _place(root, width, height)


def run_dialog(
    title: str,
    message: str,
    force_label: str | None = None,
    parent_pid: int | None = None,
) -> int:
    """Show the dialog; returns the exit code for the choice made."""
    import tkinter as tk

    root = tk.Tk()
    result = build_dialog(root, title, message, force_label)
    fit_and_center(root)
    watch_parent(root, parent_pid)
    root.mainloop()
    return result[0]


# --------------------------------------------------------------------------- cli
def _parse_pos(text: str | None):
    if not text:
        return None
    try:
        x, y = text.split(",")
        return int(x), int(y)
    except ValueError:
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="_sidecar.py", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    indicator = commands.add_parser("indicator")
    indicator.add_argument("--gif", help="animated GIF to show instead of the spinner")
    indicator.add_argument("--size", type=int)
    indicator.add_argument(
        "--pos", help="x,y cursor position (use --pos=x,y: values may be negative)"
    )
    indicator.add_argument("--parent-pid", type=int)

    dialog = commands.add_parser("dialog")
    dialog.add_argument("title")
    dialog.add_argument("message")
    dialog.add_argument("--force-label")
    dialog.add_argument("--parent-pid", type=int)

    watchdog = commands.add_parser("watchdog")
    watchdog.add_argument("pid", type=int)
    watchdog.add_argument("heartbeat_path")
    watchdog.add_argument("--timeout", type=float, required=True)
    watchdog.add_argument("--check-interval", type=float, default=1.0)
    watchdog.add_argument("--kill-tree", action="store_true")
    watchdog.add_argument("--stop-file")

    args = parser.parse_args(argv)
    if args.command == "indicator":
        return run_indicator(args.size, _parse_pos(args.pos), args.gif, args.parent_pid)
    if args.command == "dialog":
        return run_dialog(args.title, args.message, args.force_label, args.parent_pid)
    return run_watchdog(
        args.pid,
        args.heartbeat_path,
        args.timeout,
        args.check_interval,
        args.kill_tree,
        args.stop_file,
    )


if __name__ == "__main__":
    sys.exit(main())
