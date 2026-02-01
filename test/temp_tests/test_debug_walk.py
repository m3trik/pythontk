import os
from pathlib import Path

base_dir = r"O:\Cloud\Code\_scripts\pythontk\test\test_files\imgtk_test"

print(f"Scanning: {base_dir}")
for root, dirs, files in os.walk(base_dir):
    print(f"Root: {root}")
    print(f"Files: {files}")
