# !/usr/bin/python
# coding=utf-8
"""Lightweight canvas-based spinner for task-indicator overlay.

Draws a ring of fading dots that rotate — resolution-independent,
no external image files required, crisp at any size.
"""
import sys
import math
import tkinter as tk

DEFAULT_SIZE = 48
NUM_DOTS = 12
INTERVAL_MS = 80  # ms between frames


def run(size=DEFAULT_SIZE, pos=None):
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)

    transparent_color = "#f0f0f0"
    if sys.platform == "win32":
        root.attributes("-transparentcolor", transparent_color)
    else:
        root.attributes("-alpha", 0.9)
    root.configure(bg=transparent_color)

    if pos:
        x = pos[0] - size // 2
        y = pos[1] - size // 2
    else:
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        x = (sw - size) // 2
        y = (sh - size) // 2
    root.geometry(f"{size}x{size}+{x}+{y}")

    canvas = tk.Canvas(
        root, width=size, height=size, bg=transparent_color, highlightthickness=0
    )
    canvas.pack()

    cx, cy = size / 2, size / 2
    radius = size * 0.35  # ring radius
    dot_r = max(2, size * 0.07)  # individual dot radius

    # Pre-compute dot positions (angle for each dot)
    angles = [2 * math.pi * i / NUM_DOTS for i in range(NUM_DOTS)]

    # Create dot canvas items (all start invisible)
    dot_ids = []
    for angle in angles:
        dx = cx + radius * math.cos(angle)
        dy = cy + radius * math.sin(angle)
        oid = canvas.create_oval(
            dx - dot_r,
            dy - dot_r,
            dx + dot_r,
            dy + dot_r,
            fill=transparent_color,
            outline="",
        )
        dot_ids.append(oid)

    step = [0]

    def _color(brightness):
        """Map 0.0–1.0 brightness to a hex gray (0=transparent, 1=darkest)."""
        if brightness <= 0.05:
            return transparent_color
        v = int(200 * (1 - brightness))  # 200 → light gray, 0 → black
        return f"#{v:02x}{v:02x}{v:02x}"

    def update():
        for i in range(NUM_DOTS):
            # How many steps behind the "head" is this dot?
            offset = (step[0] - i) % NUM_DOTS
            # Tail fades over ~half the ring
            brightness = max(0, 1.0 - offset / (NUM_DOTS * 0.5))
            canvas.itemconfigure(dot_ids[i], fill=_color(brightness))
        step[0] = (step[0] + 1) % NUM_DOTS
        root.after(INTERVAL_MS, update)

    root.after(0, update)
    root.mainloop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--pos", help="x,y cursor position")
    args = parser.parse_args()

    pos = None
    if args.pos:
        try:
            parts = args.pos.split(",")
            pos = (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            pass

    run(size=args.size, pos=pos)
