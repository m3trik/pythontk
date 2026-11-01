#!/usr/bin/python
# coding=utf-8
"""The host seams' one convention (CODE_STANDARD section 0, Adapters).

Each shared model reaches its host through a ``<Domain>Host`` Protocol declared
beside it, ``runtime_checkable`` so a host's suite can check its adapter with
``isinstance``, and a ``Pure<Domain>Host`` default holding the answers with no
host behind it -- the base each host's adapter is built on. The roster is read
off the package root (every ``*Host`` Protocol it exports), never hand-listed:
a new seam is checked the day it is exported. Each host library's suite checks
its own adapters against these Protocols (mayatk ``test/mock_tests/
test_host_adapters.py``, blendertk ``test/test_host_adapters.py``).
Added: 2026-11-01
"""

import unittest

import pythontk as ptk

from conftest import BaseTestCase


def host_protocols():
    """``{name: Protocol}`` for every ``*Host`` Protocol the root exports."""
    found = {}
    for name in ptk.__all__:
        if name.endswith("Host") and not name.startswith("Pure"):
            value = getattr(ptk, name)
            if isinstance(value, type) and getattr(value, "_is_protocol", False):
                found[name] = value
    return found


class TestHostSeams(BaseTestCase):
    #: The seams with no ``Pure<Domain>Host`` at the root, and why.
    WITHOUT_PURE = {
        "ProbeHost": (
            "every answer reads a scene's geometry: an unbound placement is "
            "refused, never run on an empty scene (ProbeHost's docstring)"
        ),
        "ShotTransferHost": (
            "a ShotsHost role (ShotsHost.transfer): PureShotsHost().transfer() "
            "is its pure answer"
        ),
    }

    def test_the_roster_is_read_off_the_root(self):
        self.assertGreaterEqual(len(host_protocols()), 9, sorted(host_protocols()))
        self.assertIn("EmissiveGroupsHost", host_protocols())

    def test_every_host_protocol_is_runtime_checkable(self):
        for name, protocol in host_protocols().items():
            with self.subTest(protocol=name):
                self.assertTrue(
                    getattr(protocol, "_is_runtime_protocol", False),
                    f"{name}: decorate it @runtime_checkable",
                )

    def test_every_seam_has_a_pure_default_that_satisfies_it(self):
        for name, protocol in host_protocols().items():
            with self.subTest(protocol=name):
                pure = getattr(ptk, f"Pure{name}", None)
                if name in self.WITHOUT_PURE:
                    self.assertIsNone(pure, f"{name} has a Pure default now: drop its row")
                    continue
                self.assertIsNotNone(pure, f"{name}: add Pure{name} beside it")
                self.assertIsInstance(pure(), protocol)

    def test_a_role_without_its_own_pure_default_is_reached_through_one(self):
        self.assertIsInstance(ptk.PureShotsHost().transfer(), ptk.ShotTransferHost)


if __name__ == "__main__":
    unittest.main()
