# !/usr/bin/python
# coding=utf-8
import sys
import math
import tkinter as tk

DEFAULT_SIZE = 128


def _load_frames_pil(gif_path, target_size):
    """Load GIF frames using PIL (Pillow) if available."""
    try:
        from PIL import Image, ImageTk, ImageSequence

        im = Image.open(gif_path)
        duration = im.info.get("duration", 100)
        frames = []
        for frame in ImageSequence.Iterator(im):
            resized = frame.copy().resize((target_size, target_size), Image.LANCZOS)
            frames.append(ImageTk.PhotoImage(resized))
        return frames, duration
    except Exception:
        return None, None


def _load_frames_tkinter(gif_path, target_size):
    """Load GIF frames using tkinter's native PhotoImage (no PIL)."""
    frames = []
    idx = 0
    while True:
        try:
            frame = tk.PhotoImage(file=gif_path, format=f"gif -index {idx}")
            frames.append(frame)
            idx += 1
        except tk.TclError:
            break
    if not frames:
        return frames, 100
    # Subsample to approximate target_size (integer factor only)
    orig_w = frames[0].width()
    if orig_w > target_size:
        factor = max(1, round(orig_w / target_size))
        frames = [f.subsample(factor) for f in frames]
    duration = 100
    return frames, duration


def run(gif_path, target_size=DEFAULT_SIZE, pos=None):
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)

    # Use a chroma-key color for transparency (Windows supports -transparentcolor)
    transparent_color = "#f0f0f0"
    if sys.platform == "win32":
        root.attributes("-transparentcolor", transparent_color)
    else:
        root.attributes("-alpha", 0.9)
    root.configure(bg=transparent_color)

    # Try PIL first for better quality/duration, fall back to native tkinter
    frames, duration = _load_frames_pil(gif_path, target_size)
    if not frames:
        frames, duration = _load_frames_tkinter(gif_path, target_size)
    if not frames:
        sys.exit(1)

    w, h = frames[0].width(), frames[0].height()
    if pos:
        # Position near cursor, offset so spinner is centered on it
        x = pos[0] - w // 2
        y = pos[1] - h // 2
    else:
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")

    label = tk.Label(root, image=frames[0], bg=transparent_color)
    label.pack()

    def update(ind):
        frame = frames[ind]
        ind += 1
        if ind == len(frames):
            ind = 0
        label.configure(image=frame)
        root.after(duration, update, ind)

    root.after(0, update, 0)
    root.mainloop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("gif_path")
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

    run(args.gif_path, target_size=args.size, pos=pos)
