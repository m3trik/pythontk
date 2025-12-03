"""pythontk package setup.

Configuration (see update_wheel/package_config.py):
    - License: MIT
    - Optional deps excluded: Pillow, numpy
"""

import re
import setuptools
from pathlib import Path

# =============================================================================
# Package metadata (extracted without importing the package)
# =============================================================================

HERE = Path(__file__).parent.resolve()
PACKAGE = "pythontk"

# Read version from __init__.py
_init = (HERE / PACKAGE / "__init__.py").read_text(encoding="utf-8")
VERSION = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', _init).group(1)

# Read README
README = (HERE / "docs" / "README.md").read_text(encoding="utf-8")

# Extract short description from README markers
_desc_match = re.search(
    r"<!-- short_description_start -->(.+?)<!-- short_description_end -->",
    README,
    re.DOTALL,
)
DESCRIPTION = _desc_match.group(1).strip() if _desc_match else "Python toolkit"

# Read requirements, excluding optional system dependencies
EXCLUDE_DEPS = {"Pillow", "numpy"}
REQUIREMENTS = [
    line.strip()
    for line in (HERE / "requirements.txt").read_text().splitlines()
    if line.strip()
    and not line.startswith("#")
    and line.split("==")[0].split(">=")[0] not in EXCLUDE_DEPS
]

# =============================================================================
# Setup
# =============================================================================

setuptools.setup(
    name=PACKAGE,
    version=VERSION,
    author="Ryan Simpson",
    author_email="m3trik@outlook.com",
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type="text/markdown",
    url=f"https://github.com/m3trik/{PACKAGE}",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=REQUIREMENTS,
)
