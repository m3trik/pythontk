# !/usr/bin/python
# coding=utf-8
"""Tests for the DCC-agnostic key-stash core (``pythontk.core_utils.engines.key_stash``).

Pure-Python, DCC-free.  Covers :class:`StashedClip` derived ranges and
serialization, :class:`KeyStash` CRUD / observer / batch / persistence /
singleton behaviour, the preview record, and the frame-rate rescale.
"""

import os
import sys
import unittest

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pythontk.core_utils.engines.key_stash.key_stash_model import (  # noqa: E402
    KeyStash,
    StashedClip,
)


class _MemoryBackend:
    """In-memory ScenePersistence double."""

    def __init__(self, data=None):
        self.data = data
        self.saves = 0

    def save(self, data):
        self.data = data
        self.saves += 1

    def load(self):
        return self.data


def _curves(*specs):
    """``(("|cube", "tx", [10, 20, 30]), ...)`` -> adapter-shaped curve records."""
    return [
        {"object": obj, "attr": attr, "times": list(times)}
        for obj, attr, times in specs
    ]


class StashedClipTest(unittest.TestCase):
    def test_derived_range_spans_all_curves(self):
        clip = StashedClip(
            1,
            "c",
            ["|cube"],
            _curves(("|cube", "tx", [10, 20]), ("|cube", "ty", [5, 30])),
        )
        self.assertEqual(clip.start, 5)
        self.assertEqual(clip.end, 30)
        self.assertEqual(clip.duration, 25)
        self.assertEqual(clip.key_count, 4)
        self.assertEqual(clip.times, [5, 10, 20, 30])

    def test_empty_clip_has_no_range(self):
        clip = StashedClip(1, "c", [], [])
        self.assertIsNone(clip.start)
        self.assertIsNone(clip.end)
        self.assertEqual(clip.duration, 0.0)

    def test_round_trip_preserves_adapter_payload(self):
        clip = StashedClip(
            3,
            "walk",
            ["|rig|hip"],
            [{"times": [1, 2], "stash": {"name": "n", "uuid": "u"}}],
            created="2026-09-02T10:00:00",
            source_shot_id=7,
            metadata={"note": "x"},
        )
        back = StashedClip.from_dict(clip.to_dict())
        self.assertEqual(back, clip)
        # to_dict copies — mutating the export must not reach the clip.
        d = clip.to_dict()
        d["curves"][0]["times"].append(99)
        self.assertEqual(clip.times, [1, 2])

    def test_rescale_scales_times_only(self):
        clip = StashedClip(1, "c", ["|a"], [{"times": [24, 48], "attr": "tx"}])
        clip.rescale(30 / 24)
        self.assertEqual(clip.times, [30, 60])
        self.assertEqual(clip.curves[0]["attr"], "tx")


class KeyStashCrudTest(unittest.TestCase):
    def setUp(self):
        KeyStash.set_persistence(None)
        KeyStash._active = None
        self.store = KeyStash()

    def test_add_assigns_ids_and_default_label(self):
        a = self.store.add_clip(["|grp|cube"], _curves(("|grp|cube", "tx", [10, 30])))
        b = self.store.add_clip(
            ["|a", "|b"], _curves(("|a", "tx", [1.5])), source_shot_id=4
        )
        self.assertEqual((a.clip_id, b.clip_id), (1, 2))
        self.assertEqual(a.label, "cube 10-30")
        self.assertEqual(b.label, "a +1 1.5")
        self.assertEqual(self.store.clips_for_shot(4), [b])
        self.assertEqual(self.store.clips_for_object("cube"), [a])
        self.assertEqual(self.store.clips_for_object("|grp|cube"), [a])

    def test_add_rejects_keyless_clip(self):
        with self.assertRaises(ValueError):
            self.store.add_clip(["|a"], [{"times": []}])

    def test_remove_returns_record_and_fires_kind(self):
        events = []
        self.store.add_listener(events.append)
        clip = self.store.add_clip(["|a"], _curves(("|a", "tx", [1])))
        gone = self.store.remove_clip(clip.clip_id, kind="retrieved")
        self.assertIs(gone, clip)
        self.assertIsNone(self.store.remove_clip(clip.clip_id))
        self.assertEqual([e.kind for e in events], ["stashed", "retrieved"])
        self.assertTrue(self.store.is_empty())

    def test_ids_never_reused_after_reload(self):
        self.store.add_clip(["|a"], _curves(("|a", "tx", [1])))
        c2 = self.store.add_clip(["|a"], _curves(("|a", "tx", [2])))
        self.store.remove_clip(1)
        reloaded = KeyStash.from_dict(self.store.to_dict())
        c3 = reloaded.add_clip(["|a"], _curves(("|a", "tx", [3])))
        self.assertEqual(c3.clip_id, c2.clip_id + 1)

    def test_ids_never_reused_when_the_highest_was_removed(self):
        """The case the test above cannot reach: it removes the LOWER id, so
        ``max(clip_id) + 1`` still lands past the high-water mark.

        Remove the highest instead and the counter walks backwards on reload,
        handing a live id to a different clip. Stale references then address
        the wrong keys -- a tree item's UserRole captured before the reload, a
        queued ``StashChanged``, or the ``active_preview["clip_id"]`` that
        ``from_dict`` restores verbatim.
        """
        self.store.add_clip(["|a"], _curves(("|a", "tx", [1])))
        c2 = self.store.add_clip(["|a"], _curves(("|a", "tx", [2])))
        self.store.remove_clip(c2.clip_id)
        reloaded = KeyStash.from_dict(self.store.to_dict())
        c3 = reloaded.add_clip(["|a"], _curves(("|a", "tx", [3])))
        self.assertNotEqual(
            c3.clip_id, c2.clip_id, "a removed clip's id was handed out again"
        )
        self.assertEqual(c3.clip_id, c2.clip_id + 1)

    def test_preview_record(self):
        clip = self.store.add_clip(["|a"], _curves(("|a", "tx", [1, 5])))
        with self.assertRaises(KeyError):
            self.store.set_preview(99)
        self.store.set_preview(clip.clip_id, {"layer": "L"})
        self.assertEqual(
            self.store.active_preview, {"clip_id": clip.clip_id, "layer": "L"}
        )
        self.assertFalse(self.store.is_empty())
        # Removing the previewed clip drops the record with it.
        self.store.remove_clip(clip.clip_id)
        self.assertIsNone(self.store.active_preview)
        self.assertIsNone(self.store.clear_preview())

    def test_batch_update_collapses_events_and_flushes_once(self):
        backend = _MemoryBackend()
        KeyStash.set_persistence(backend)
        events = []
        self.store.add_listener(events.append)
        with self.store.batch_update():
            self.store.add_clip(["|a"], _curves(("|a", "tx", [1])))
            self.store.add_clip(["|a"], _curves(("|a", "tx", [2])))
            self.assertEqual(backend.saves, 0)
        self.assertEqual([e.kind for e in events], ["reloaded"])
        self.assertEqual(backend.saves, 1)

    def test_broken_listener_does_not_break_store(self):
        def boom(_event):
            raise RuntimeError("listener bug")

        self.store.add_listener(boom)
        clip = self.store.add_clip(["|a"], _curves(("|a", "tx", [1])))
        self.assertEqual(self.store.get_clip(clip.clip_id), clip)

    def test_helpers(self):
        clip = self.store.add_clip(["|a"], _curves(("|a", "tx", [10, 30])))
        self.assertEqual(KeyStash.offset_for(clip, None), 0.0)
        self.assertEqual(KeyStash.offset_for(clip, 100), 90.0)
        self.assertEqual(KeyStash.gate_range(clip), (10, 30))


class KeyStashPersistenceTest(unittest.TestCase):
    def setUp(self):
        KeyStash._active = None
        KeyStash.set_persistence(None)

    def tearDown(self):
        KeyStash._active = None
        KeyStash.set_persistence(None)

    def test_every_mutation_flushes_through_backend(self):
        backend = _MemoryBackend()
        KeyStash.set_persistence(backend)
        store = KeyStash()
        clip = store.add_clip(["|a"], _curves(("|a", "tx", [1, 2])))
        self.assertEqual(backend.saves, 1)
        self.assertEqual(backend.data["clips"][0]["label"], clip.label)
        store.set_preview(clip.clip_id)
        store.remove_clip(clip.clip_id)
        self.assertEqual(backend.saves, 3)
        self.assertEqual(backend.data["clips"], [])
        self.assertIsNone(backend.data["preview"])

    def test_a_raising_backend_leaves_the_store_dirty(self):
        """Clearing the dirty flag before the write discards the pending
        change: the next ``_flush_dirty`` no-ops and nothing is ever written,
        even once the backend recovers.

        ``ShotStore.save`` -- the sibling this engine mirrors -- clears only
        after a successful write and says why in a comment. Reachable in the
        Maya adapter, whose flush runs under ``evalDeferred``: Maya prints the
        traceback and carries on, so the parked animCurves end up with no
        manifest referencing them and are unreachable on reopen.
        """

        class _RaisingOnce(_MemoryBackend):
            def __init__(self):
                super().__init__()
                self.fail_next = True

            def save(self, data):
                if self.fail_next:
                    self.fail_next = False
                    raise OSError("scene node locked")
                super().save(data)

        backend = _RaisingOnce()
        KeyStash.set_persistence(backend)
        store = KeyStash()
        with self.assertRaises(OSError):
            store.add_clip(["|a"], _curves(("|a", "tx", [1, 2])))
        self.assertTrue(store._dirty, "a failed write was forgotten instead of retried")
        # The backend has recovered; the pending clip must still reach it.
        store._flush_dirty()
        self.assertEqual(backend.saves, 1)
        self.assertEqual(len(backend.data["clips"]), 1)

    def test_active_loads_from_backend_once(self):
        seed = KeyStash()
        seed.add_clip(["|a"], _curves(("|a", "tx", [1])))
        seed.active_preview = {"clip_id": 1, "layer": "L"}
        KeyStash.set_persistence(_MemoryBackend(seed.to_dict()))
        store = KeyStash.active()
        self.assertEqual(len(store.clips), 1)
        self.assertEqual(store.active_preview, {"clip_id": 1, "layer": "L"})
        self.assertIs(KeyStash.active(), store)

    def test_invalidate_drops_active_and_notifies(self):
        hits = []
        KeyStash.add_invalidation_listener(hits.append)
        try:
            first = KeyStash.active()
            KeyStash.invalidate()
            self.assertIsNot(KeyStash.active(), first)
            self.assertEqual([e.kind for e in hits], ["reloaded"])
        finally:
            KeyStash.remove_invalidation_listener(hits.append)

    def test_active_rescales_a_store_saved_at_another_fps(self):
        class Fps30(KeyStash):
            def _scene_fps(self):
                return 30.0

        seed = KeyStash(scene_fps=24.0)
        seed.add_clip(["|a"], _curves(("|a", "tx", [24, 48])))
        Fps30.set_persistence(_MemoryBackend(seed.to_dict()))
        try:
            store = Fps30.active()
            self.assertEqual(store.clips[0].times, [30, 60])
            self.assertEqual(store.scene_fps, 30.0)
        finally:
            Fps30._active = None
            Fps30.set_persistence(None)

    def test_rescale_is_noop_for_same_fps(self):
        store = KeyStash(scene_fps=24.0)
        store.add_clip(["|a"], _curves(("|a", "tx", [24])))
        events = []
        store.add_listener(events.append)
        store.rescale_to_fps(24.004)
        self.assertEqual(events, [])
        self.assertEqual(store.clips[0].times, [24])


if __name__ == "__main__":
    unittest.main()
