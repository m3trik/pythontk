# !/usr/bin/python
# coding=utf-8
"""
Subprocess-based dialog viewer for custom button labels.
This runs in a separate process to avoid blocking issues with the main application.
"""
import sys
import tkinter as tk
from tkinter import ttk


def run(title: str, message: str):
    """Display a dialog with custom buttons matching VS Code style.

    Returns via exit code:
        0: Keep Waiting
        10: Cancel (Stop Operation)
        2: Force Quit
        3: Window closed (treated as Keep Waiting)
    """
    root = tk.Tk()
    root.title(title)
    root.attributes("-topmost", True)
    root.resizable(False, False)

    # Center on screen
    root.update_idletasks()
    width = 450
    height = 180
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - width) // 2
    y = (sh - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")

    # Store result
    result = [3]  # Default: closed without choice

    def on_keep_waiting():
        result[0] = 0
        root.destroy()

    def on_cancel():
        result[0] = 10  # Use 10 for Cancel to distinguish from standard exit code 1 (error)
        root.destroy()

    def on_force_quit():
        result[0] = 2
        root.destroy()

    def on_close():
        result[0] = 3
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # Main frame with padding
    main_frame = ttk.Frame(root, padding="20 15 20 15")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Warning icon and message
    msg_frame = ttk.Frame(main_frame)
    msg_frame.pack(fill=tk.X, pady=(0, 15))

    # Warning icon (using Unicode)
    icon_label = ttk.Label(
        msg_frame, text="⚠", font=("Segoe UI", 24), foreground="#E8A317"
    )
    icon_label.pack(side=tk.LEFT, padx=(0, 15))

    # Message text
    msg_label = ttk.Label(
        msg_frame,
        text=message,
        wraplength=350,
        justify=tk.LEFT,
        font=("Segoe UI", 10),
    )
    msg_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    # Button frame (right-aligned like VS Code)
    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

    # Spacer to push buttons right
    ttk.Label(btn_frame).pack(side=tk.LEFT, expand=True)

    # Buttons (in order: Force Quit, Cancel, Keep Waiting)
    # "Keep Waiting" is the default/primary action
    style = ttk.Style()
    style.configure("Accent.TButton", font=("Segoe UI", 9))

    force_quit_btn = ttk.Button(
        btn_frame, text="Force Quit", command=on_force_quit, width=12
    )
    force_quit_btn.pack(side=tk.LEFT, padx=(0, 8))

    cancel_btn = ttk.Button(btn_frame, text="Cancel", command=on_cancel, width=12)
    cancel_btn.pack(side=tk.LEFT, padx=(0, 8))

    keep_waiting_btn = ttk.Button(
        btn_frame, text="Keep Waiting", command=on_keep_waiting, width=14
    )
    keep_waiting_btn.pack(side=tk.LEFT)
    keep_waiting_btn.focus_set()

    # Bind Enter to Keep Waiting (default action)
    root.bind("<Return>", lambda e: on_keep_waiting())
    root.bind("<Escape>", lambda e: on_cancel())

    root.mainloop()
    sys.exit(result[0])


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        run(sys.argv[1], sys.argv[2])
    else:
        run(
            "Test Dialog",
            "The operation is taking longer than expected.\n\nYou can keep waiting, cancel the operation, or force quit the application.",
        )
