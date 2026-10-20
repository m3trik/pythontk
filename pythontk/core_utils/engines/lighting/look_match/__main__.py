# !/usr/bin/python
# coding=utf-8
"""``python -m pythontk.core_utils.engines.lighting.look_match`` -- the look-match command line (see ``LookMatch.cli``)."""

import sys

from pythontk.core_utils.engines.lighting.look_match._look_match import LookMatch

if __name__ == "__main__":
    sys.exit(LookMatch.cli())
