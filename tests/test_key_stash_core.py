# !/usr/bin/python
# coding=utf-8
"""Tests for the DCC-agnostic key-stash core (``pythontk.core_utils.engines.key_stash``).

Pure-Python, DCC-free.  Covers :class:`StashedClip` derived ranges and
serialization, :class:`KeyStash` CRUD / observer / batch / persistence /
singleton behaviour, the preview record, the frame-rate rule, the saved
record's golden case, and the store over a fake host adapter
(:class:`KeyStashHost`): what each host differs by, and nothing else.
"""

import os
import sys
import unittest
from contextlib import contextmanager

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pythontk.core_utils.engines.key_stash.key_stash_model import (  # noqa: E402
    KeyStash,
    KeyStashConformance,
    KeyStashHost,
    PureKeyStashHost,
    StashedClip,
)
from pythontk.core_utils.engines.scene_export.scene_records import (  # noqa: E402
    SceneRecords,
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


class _FakeHost(PureKeyStashHost):
    """A host double over the pure defaults: a scene clock, the frame-rate
    flag, and the hooks a test sets."""

    def __init__(self, fps=None, keys_keep_time=True):
        self.fps = fps
        self.keys_keep_time = keys_keep_time

    def scene_fps(self):
        return self.fps


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
            HOST = _FakeHost(fps=30.0)

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


class KeyStashConformanceTest(unittest.TestCase):
    """What the store saves is a contract older scenes hold: the ``key_stash``
    record a filled store writes is pinned byte for byte (every host's store
    is held to the same case)."""

    def setUp(self):
        KeyStash._active = None
        KeyStash.set_persistence(None)

    def test_the_saved_record_is_pinned_byte_for_byte(self):
        store = KeyStashConformance.fill(KeyStash())
        self.assertEqual(
            SceneRecords.KEY_STASH.encode(store.to_dict()),
            KeyStashConformance.payload(),
        )

    def test_a_reloaded_store_saves_the_same_bytes(self):
        payload = KeyStashConformance.payload()
        restored = KeyStash.from_dict(SceneRecords.KEY_STASH.decode(payload))
        self.assertEqual(SceneRecords.KEY_STASH.encode(restored.to_dict()), payload)

    def test_the_backend_receives_the_record_as_saved(self):
        backend = _MemoryBackend()
        KeyStash.set_persistence(backend)
        try:
            KeyStashConformance.fill(KeyStash()).save()
        finally:
            KeyStash.set_persistence(None)
        self.assertEqual(
            SceneRecords.KEY_STASH.encode(backend.data), KeyStashConformance.payload()
        )


class KeyStashFlushPendingTest(unittest.TestCase):
    def tearDown(self):
        KeyStash._active = None
        KeyStash.set_persistence(None)

    def test_flush_pending_stores_what_the_active_stash_holds_unwritten(self):
        """A DCC stash writes on idle, so its record can be a mutation behind;
        a crossing reads the record, and asks for what the stash holds first."""

        class _Idle(_FakeHost):
            def schedule_flush(self, store):
                """Writes on idle, as a DCC stash does."""

        class Deferred(KeyStash):
            HOST = _Idle()

        backend = _MemoryBackend()
        Deferred.set_persistence(backend)
        Deferred._active = None
        try:
            Deferred.active().add_clip(["|a"], _curves(("|a", "tx", [1, 2])))
            self.assertEqual(backend.saves, 0)
            Deferred.flush_pending()
            self.assertEqual(len(backend.data["clips"]), 1)
            Deferred.flush_pending()
            self.assertEqual(backend.saves, 1)  # nothing pending: nothing written
        finally:
            Deferred._active = None
            Deferred.set_persistence(None)


class KeyStashUndoStepTest(unittest.TestCase):
    """The template both DCC adapters run their scene operations through."""

    def setUp(self):
        KeyStash._active = None
        KeyStash.set_persistence(None)

    def tearDown(self):
        KeyStash._active = None
        KeyStash.set_persistence(None)

    def test_an_undo_step_writes_the_record_once_inside_its_chunk(self):
        """The pure default flushes every mutation at once, and so do mayapy's
        ``evalDeferred`` and background Blender. Inside an undo step that flush
        must wait: it would write the new record outside the step, where no
        undo can take it back (measured 2026-09-15 in both adapters).
        Added: 2026-09-15
        """
        events = []

        class _Recording(_MemoryBackend):
            def save(self, data):
                events.append("save")
                super().save(data)

        class _Chunked(_FakeHost):
            @contextmanager
            def undo_chunk(self, name):
                events.append(f"open {name}")
                yield
                events.append("close")

        class Adapter(KeyStash):
            HOST = _Chunked()

        KeyStash.set_persistence(_Recording())
        store = Adapter()
        with store._undo_step("Store Keys"):
            clip = store.add_clip(["|a"], _curves(("|a", "tx", [1, 2])))
            store.set_preview(clip.clip_id)
        self.assertEqual(events, ["open Store Keys", "save", "close"])
        self.assertFalse(store._dirty)

    def test_an_undo_step_that_raises_still_writes_its_record_inside(self):
        """A block that raises after a mutation must still leave its record in
        the step. Otherwise the batch's own exit flush writes it through
        ``save``, which the Maya adapter keeps outside the undo queue, and an
        undo would revert the scene edits while the record still describes them.
        Added: 2026-09-15
        """
        events = []

        class _Recording(_MemoryBackend):
            def save(self, data):
                events.append("save")
                super().save(data)

        class _Chunked(_FakeHost):
            @contextmanager
            def undo_chunk(self, name):
                events.append("open")
                try:
                    yield
                finally:
                    events.append("close")

            def save_in_step(self, backend, data):
                events.append("in-step write")
                return False  # the backend's save already writes in the step

        class Adapter(KeyStash):
            HOST = _Chunked()

        KeyStash.set_persistence(_Recording())
        store = Adapter()
        with self.assertRaises(RuntimeError):
            with store._undo_step("Retrieve Stored Keys"):
                store.add_clip(["|a"], _curves(("|a", "tx", [1, 2])))
                raise RuntimeError("paste refused")
        self.assertEqual(events, ["open", "in-step write", "save", "close"])

    def test_active_rereads_a_record_an_undo_moved(self):
        """An undo or redo moves the record under the loaded store. Asking for
        the store re-reads it in place: a new store's activation would tear
        down the very preview a redo restored.
        Added: 2026-09-15
        """
        activations = []

        class _Movable(_MemoryBackend):
            moved = False

            def record_changed(self):
                return self.moved

            def load(self):
                self.moved = False
                return super().load()

        class _Reconciling(_FakeHost):
            def reconcile(self, store):
                activations.append(store)
                return []

        class Adapter(KeyStash):
            HOST = _Reconciling()

        backend = _Movable()
        Adapter.set_persistence(backend)
        try:
            store = Adapter.active()
            events = []
            store.add_listener(events.append)
            seed = KeyStash()
            clip = seed.add_clip(["|a"], _curves(("|a", "tx", [1, 2])))
            seed.active_preview = {"clip_id": clip.clip_id, "layer": "L"}
            backend.data, backend.moved = seed.to_dict(), True  # the undo
            self.assertIs(Adapter.active(), store)
            self.assertEqual([c.clip_id for c in store.clips], [clip.clip_id])
            self.assertTrue(store.is_previewing(clip.clip_id))
            self.assertEqual([e.kind for e in events], ["reloaded"])
            self.assertEqual(activations, [store])
        finally:
            Adapter._active = None
            Adapter.set_persistence(None)


class KeyStashHostTest(unittest.TestCase):
    """The one store over a host adapter: each member of :class:`KeyStashHost`
    is asked where the scene is reached, and a binding needs nothing else."""

    def setUp(self):
        KeyStash._active = None
        KeyStash.set_persistence(None)

    def tearDown(self):
        KeyStash._active = None
        KeyStash.set_persistence(None)

    def bound(self, host):
        """A store class bound to *host*, as a host library's binding is."""

        class Bound(KeyStash):
            HOST = host

        self.addCleanup(Bound.set_persistence, None)
        return Bound

    def test_the_pure_defaults_are_a_host(self):
        self.assertIsInstance(PureKeyStashHost(), KeyStashHost)
        self.assertIsInstance(KeyStash.HOST, PureKeyStashHost)
        self.assertEqual(KeyStash.ATTR_NAME, SceneRecords.KEY_STASH.key)
        self.assertEqual(KeyStash().scene_fps, 24.0)  # no scene clock

    def test_active_installs_the_hosts_persistence_once(self):
        made = []

        class _Persisting(_FakeHost):
            def persistence(self, store_cls):
                made.append(store_cls)
                return _MemoryBackend()

        Bound = self.bound(_Persisting())
        store = Bound.active()
        self.assertIs(Bound.active(), store)
        self.assertEqual(made, [Bound])
        self.assertIsInstance(Bound._persistence, _MemoryBackend)
        store.add_clip(["|a"], _curves(("|a", "tx", [1])))
        self.assertEqual(Bound._persistence.saves, 1)
        Bound._active = None

    def test_a_host_whose_keys_keep_time_rescales_and_tells_the_listeners(self):
        """Maya: curve keys live in ticks, so a rate change moved the frames."""
        store = self.bound(_FakeHost(fps=24.0, keys_keep_time=True))()
        store.add_clip(["|a"], _curves(("|a", "tx", [24, 48])))
        events = []
        store.add_listener(events.append)
        store.rescale_to_fps(30.0)
        self.assertEqual(store.clips[0].times, [30, 60])
        self.assertEqual(store.scene_fps, 30.0)
        self.assertEqual([e.kind for e in events], ["reloaded"])

    def test_a_host_whose_keys_stay_on_their_frames_records_the_rate_only(self):
        """Blender: keys stay on their frames, so nothing moves and nothing
        repaints; the new rate is recorded and saved."""
        backend = _MemoryBackend()
        Bound = self.bound(_FakeHost(fps=24.0, keys_keep_time=False))
        Bound.set_persistence(backend)
        store = Bound()
        store.add_clip(["|a"], _curves(("|a", "tx", [24, 48])))
        events = []
        store.add_listener(events.append)
        store.rescale_to_fps(48.0)
        self.assertEqual(store.clips[0].times, [24, 48])
        self.assertEqual(store.scene_fps, 48.0)
        self.assertEqual(events, [])
        self.assertEqual(backend.data["scene_fps"], 48.0)
        store.rescale_to_fps(48.004)  # the same rate: nothing to record
        self.assertEqual(backend.saves, 2)

    def test_schedule_flush_goes_through_the_host(self):
        asked = []

        class _Coalescing(_FakeHost):
            def schedule_flush(self, store):
                asked.append(store)

        backend = _MemoryBackend()
        Bound = self.bound(_Coalescing())
        Bound.set_persistence(backend)
        store = Bound()
        store.add_clip(["|a"], _curves(("|a", "tx", [1])))
        self.assertEqual((asked, backend.saves), ([store], 0))
        store._flush_dirty()
        self.assertEqual(backend.saves, 1)

    def test_a_host_that_writes_in_the_step_takes_the_record(self):
        """Maya's backend writes outside the undo queue unless told: the host
        writes the record into the open step, and the store does not save it
        a second time."""
        written = []

        class _InStep(_FakeHost):
            def save_in_step(self, backend, data):
                written.append((backend, data))
                return True

        backend = _MemoryBackend()
        Bound = self.bound(_InStep())
        Bound.set_persistence(backend)
        store = Bound()
        with store._undo_step("Store Keys"):
            store.add_clip(["|a"], _curves(("|a", "tx", [1, 2])))
        self.assertEqual(backend.saves, 0)
        self.assertEqual(len(written), 1)
        self.assertIs(written[0][0], backend)
        self.assertEqual(written[0][1], store.to_dict())
        self.assertFalse(store._dirty)

    def test_reconcile_prunes_dead_clips_and_ends_the_recorded_preview(self):
        """The reconcile is written once: the host answers which parked keys
        still exist and tears its preview down, the store does the rest."""
        ended = []

        class _Scene(_FakeHost):
            def reconcile(self, store):
                return store._reconcile_with(
                    lambda rec: rec.get("alive"), ended.append
                )

        Bound = self.bound(_Scene())
        seed = KeyStash()
        dead = seed.add_clip(["|a"], [{"times": [1], "alive": False}])
        live = seed.add_clip(
            ["|b"], [{"times": [2], "alive": False}, {"times": [3], "alive": True}]
        )
        seed.active_preview = {"clip_id": live.clip_id, "layer": "L"}
        Bound.set_persistence(_MemoryBackend(seed.to_dict()))
        store = Bound.active()  # activation reconciles
        self.assertEqual([c.clip_id for c in store.clips], [live.clip_id])
        self.assertEqual(ended, [{"clip_id": live.clip_id, "layer": "L"}])
        self.assertIsNone(store.active_preview)
        self.assertEqual(Bound._persistence.data["preview"], None)
        events = []
        store.add_listener(events.append)
        self.assertEqual(store.reconcile(), [])  # nothing left to settle
        self.assertEqual(events, [])
        self.assertNotIn(dead.clip_id, [c.clip_id for c in store.clips])
        Bound._active = None

    def test_the_pure_host_prunes_nothing(self):
        store = KeyStash()
        store.add_clip(["|a"], _curves(("|a", "tx", [1])))
        store.active_preview = {"clip_id": 1}
        self.assertEqual(store.reconcile(), [])
        self.assertEqual(len(store.clips), 1)
        self.assertEqual(store.active_preview, {"clip_id": 1})

    def test_watch_undo_subscribes_active_through_the_hosts_event_manager(self):
        """The subscriptions re-ask for the store, which re-reads a record an
        undo moved; an event the host lacks is skipped, never raised."""

        class _Jobs:
            def __init__(self, fail=()):
                self.subscribed, self.cleanups, self.fail = [], [], set(fail)

            def subscribe(self, event, callback, owner=None):
                if event in self.fail:
                    raise RuntimeError("no such event")
                self.subscribed.append((event, callback, owner))
                return len(self.subscribed) + 10

            def connect_cleanup(self, widget, owner=None):
                self.cleanups.append((widget, owner))

        jobs = _Jobs()

        class _Evented(_FakeHost):
            def script_jobs(self):
                return jobs

        Bound = self.bound(_Evented())
        owner, widget = object(), object()
        self.assertEqual(Bound.watch_undo(owner, widget=widget), [11, 12])
        self.assertEqual(
            jobs.subscribed,
            [("Undo", Bound.active, owner), ("Redo", Bound.active, owner)],
        )
        self.assertEqual(jobs.cleanups, [(widget, owner)])
        jobs = _Jobs(fail={"Undo"})
        self.assertEqual(Bound.watch_undo(owner), [11])
        self.assertEqual(jobs.cleanups, [])
        self.assertEqual(KeyStash.watch_undo(owner), [])  # no event manager

    def test_each_store_carries_what_its_host_keeps_for_it(self):
        """``host_state`` is made with each store (Blender: the deferred
        write's pending flag), as the shot store's is."""

        class _Stateful(_FakeHost):
            def store_state(self):
                return {"flush_pending": False}

        Bound = self.bound(_Stateful())
        a, b = Bound(), Bound()
        self.assertEqual(a.host_state, {"flush_pending": False})
        self.assertIsNot(a.host_state, b.host_state)
        self.assertIsNone(KeyStash().host_state)  # the pure host keeps nothing

    def test_merge_carrier_reloads_only_for_the_key_stash_record(self):
        hits = []
        KeyStash.add_invalidation_listener(hits.append)
        try:
            first = KeyStash.active()
            KeyStash.merge_carrier({}, {SceneRecords.SHOT_STORE: {}}, None)
            self.assertIs(KeyStash.active(), first)
            KeyStash.merge_carrier({}, {SceneRecords.KEY_STASH: {"clips": []}}, None)
            self.assertIsNot(KeyStash.active(), first)
            self.assertEqual([e.kind for e in hits], ["reloaded"])
        finally:
            KeyStash.remove_invalidation_listener(hits.append)


if __name__ == "__main__":
    unittest.main()
