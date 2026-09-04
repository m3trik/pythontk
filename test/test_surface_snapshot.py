#!/usr/bin/python
# coding=utf-8
"""The published surface is pinned to a checked-in snapshot.

pythontk sits at the bottom of the ecosystem chain, and every name it exports
is a contract: ``pythontk -> uitk -> {mayatk, blendertk} -> tentacle``, plus
``unitytk`` and ``extapps``. A break cascades upward.

Three gates look like they cover this and none of them does. The API registry
gate hashes generated documents, so it reports a *diff* rather than failing on
a removal. ``test_all_packages_namespace_aliases`` checks alias mechanics, not
membership. Nothing at all watched where a name RESOLVES -- so moving a class
between modules, which is exactly what a base-class extraction does, changed
the public surface silently.

This snapshot pins name, tier, defining module and qualname, so a rename, a
removal, a retier and a cross-module move each show up as a named diff. It is
not a freeze: a deliberate change is one regeneration away, and the point is
that it has to be deliberate.

Regenerate after an intended change::

    python test/test_surface_snapshot.py --update
"""

import json
import os
import subprocess
import sys
import unittest

SNAPSHOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "surface_snapshot.json"
)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def live_surface():
    """The surface as the package actually publishes it, via the same
    ``--index --json`` entry point the registry generator uses."""
    out = subprocess.run(
        [sys.executable, "-m", "pythontk", "--index", "--json"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout
    entries = json.loads(out)
    entries.sort(key=lambda e: (e["tier"], e["name"]))
    return entries


def write_snapshot(entries):
    with open(SNAPSHOT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(entries, f, indent=2, sort_keys=True)
        f.write("\n")


class TestPublishedSurface(unittest.TestCase):
    @staticmethod
    def _by_name(entries):
        return {e["name"]: e for e in entries}

    def setUp(self):
        with open(SNAPSHOT, encoding="utf-8") as f:
            self.expected = json.load(f)
        self.actual = live_surface()

    def test_no_name_was_added_or_removed(self):
        want, got = self._by_name(self.expected), self._by_name(self.actual)
        removed = sorted(set(want) - set(got))
        added = sorted(set(got) - set(want))
        self.assertEqual(
            (removed, added),
            ([], []),
            "the published surface changed.\n"
            f"  REMOVED (breaking for consumers): {removed}\n"
            f"  ADDED (new contract to keep): {added}\n"
            "If intended: python test/test_surface_snapshot.py --update, and "
            "note it in CHANGELOG.md (a removal needs an alias for one release).",
        )

    def test_no_name_changed_where_it_resolves(self):
        """A cross-module move is the silent one -- the name still imports, so
        every other gate stays green, while anything importing it by path
        breaks. This is the check that makes a base-class extraction safe."""
        want, got = self._by_name(self.expected), self._by_name(self.actual)
        moved = []
        for name in sorted(set(want) & set(got)):
            for field in ("module", "qualname", "tier", "kind"):
                if want[name][field] != got[name][field]:
                    moved.append(
                        f"  {name}: {field} {want[name][field]!r} -> {got[name][field]!r}"
                    )
        self.assertEqual(
            moved,
            [],
            "published names moved:\n" + "\n".join(moved),
        )

    def test_the_snapshot_is_not_empty_or_stale_shaped(self):
        """A snapshot that failed to generate would otherwise pass forever."""
        self.assertGreater(len(self.expected), 300, "snapshot looks truncated")
        tiers = {e["tier"] for e in self.expected}
        self.assertEqual(tiers, {"root", "bare"}, f"unexpected tiers: {tiers}")
        for e in self.expected:
            self.assertTrue(e.get("module", "").startswith("pythontk."), e)


if __name__ == "__main__":
    if "--update" in sys.argv:
        entries = live_surface()
        write_snapshot(entries)
        tiers = {}
        for e in entries:
            tiers[e["tier"]] = tiers.get(e["tier"], 0) + 1
        print(f"surface_snapshot.json updated: {tiers}, {len(entries)} names")
    else:
        unittest.main()
