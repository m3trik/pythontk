# !/usr/bin/python
# coding=utf-8
import sys
import os
import tkinter as tk

try:
    from PIL import Image, ImageTk, ImageSequence
except ImportError:
    sys.exit(0)


def run(gif_path):
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.9)

    try:
        im = Image.open(gif_path)
    except Exception:
        sys.exit(1)

    frames = [ImageTk.PhotoImage(frame.copy()) for frame in ImageSequence.Iterator(im)]
    if not frames:
        sys.exit(1)

    w, h = frames[0].width(), frames[0].height()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - w) // 2
    y = (sh - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")

    label = tk.Label(root, image=frames[0], bg="white")
    label.pack()

    duration = im.info.get("duration", 100)

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
    if len(sys.argv) > 1:
        run(sys.argv[1])
