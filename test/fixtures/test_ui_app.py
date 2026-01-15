import tkinter as tk
import sys


def main():
    root = tk.Tk()
    root.title("TestAppWindow")
    # Auto close after a few seconds so we don't hang if test fails to kill
    root.after(10000, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
