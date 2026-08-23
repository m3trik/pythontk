#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk TempArtifacts and CachedArtifact.

Covers (TempArtifacts):
- path() allocation (unique tags, fixed names, prefix scoping)
- register() adoption of externally-created artifacts
- lifetime policies: scoped / session / detached
- context-manager semantics (delete on success, keep on failure)
- on_cleanup callback
- sweep_stale() prefix-scoped GC (age-gated, conservative)

Covers (CachedArtifact):
- key() stability + invalidation on any input file (source AND template)
- hit / miss, scratch-then-atomic-promote, sidecar promotion + stale-sidecar drop
- failure keeps the partial and leaves the cache slot untouched
- use_cache=False bypass

Run with:
    python -m pytest test_temp_artifacts.py -v
    python test_temp_artifacts.py
"""
import os
import time
import shutil
import tempfile
import threading
import unittest
from unittest import mock

from pythontk.file_utils.temp_artifacts import (
    CachedArtifact,
    ScratchTwins,
    TempArtifacts,
)


class TempArtifactsBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ta_test_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def touch(self, path, content=b"x"):
        with open(path, "wb") as f:
            f.write(content)
        return path

    def age(self, path, days):
        """Backdate a file's mtime by *days*."""
        past = time.time() - days * 86400
        os.utime(path, (past, past))


class TestAllocation(TempArtifactsBase):
    def test_path_is_prefix_scoped_and_unique(self):
        ta = TempArtifacts("myprefix", dir=self.dir)
        a = ta.path(extension=".fbx")
        b = ta.path(extension=".fbx")
        self.assertNotEqual(a, b)
        for p in (a, b):
            self.assertEqual(os.path.dirname(p), self.dir)
            self.assertTrue(os.path.basename(p).startswith("myprefix_"))
            self.assertTrue(p.endswith(".fbx"))

    def test_path_is_unique_across_instances_in_one_clock_tick(self):
        """Regression: the tag counter must be process-wide, not per-instance.

        Distinct stores routinely share one dir+prefix namespace -
        ``CachedArtifact`` builds a fresh scoped store per ``get``. The counter
        was reset to 0 in ``__init__``, so two stores allocating inside a single
        ``time_ns`` tick (Windows' ~15ms resolution is far coarser than two
        back-to-back allocations) both minted the same tag and handed back the
        *same path* - the producer's output silently overwriting the other's.
        """
        frozen = 1_700_000_000_000_000_000
        with mock.patch.object(time, "time_ns", return_value=frozen):
            a = TempArtifacts("shared", dir=self.dir).path(extension=".out")
            b = TempArtifacts("shared", dir=self.dir).path(extension=".out")
        self.assertNotEqual(a, b)

    def test_path_is_unique_across_threads(self):
        """The tag counter's read-modify-write must be atomic.

        Concurrent producers share the class-level counter; without the lock two
        threads can read the same value and mint the same tag. Time is frozen so
        the counter - not the clock - is what has to provide the distinctness.
        """
        frozen = 1_700_000_000_000_000_000
        results, lock = [], threading.Lock()

        def allocate():
            ta = TempArtifacts("threaded", dir=self.dir)
            paths = [ta.path(extension=".out") for _ in range(20)]
            with lock:
                results.extend(paths)

        with mock.patch.object(time, "time_ns", return_value=frozen):
            threads = [threading.Thread(target=allocate) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual(len(results), 160)
        self.assertEqual(len(set(results)), 160)

    def test_path_fixed_name_is_deterministic(self):
        ta = TempArtifacts("pfx", dir=self.dir)
        a = ta.path(extension=".lua", name="script")
        b = ta.path(extension=".lua", name="script")
        self.assertEqual(a, b)
        self.assertEqual(os.path.basename(a), "pfx_script.lua")

    def test_default_dir_is_system_temp(self):
        # Assert on .dir rather than allocating: path() on an unsandboxed
        # instance would run a live sweep in the user's real temp dir.
        self.assertEqual(TempArtifacts("pfx").dir, tempfile.gettempdir())

    def test_default_extension_is_generic(self):
        ta = TempArtifacts("pfx", dir=self.dir)
        self.assertTrue(ta.path().endswith(".tmp"))

    def test_empty_prefix_raises(self):
        with self.assertRaises(ValueError):
            TempArtifacts("")

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            TempArtifacts("pfx", policy="bogus")

    def test_register_adopts_and_returns_path(self):
        ta = TempArtifacts("pfx", dir=self.dir, policy="scoped")
        side = self.touch(os.path.join(self.dir, "sidecar.json"))
        self.assertEqual(ta.register(side), side)
        ta.cleanup()
        self.assertFalse(os.path.exists(side))


class TestScopedPolicy(TempArtifactsBase):
    def test_cleanup_removes_tracked_files(self):
        ta = TempArtifacts("pfx", dir=self.dir, policy="scoped")
        p = self.touch(ta.path())
        ta.cleanup()
        self.assertFalse(os.path.exists(p))

    def test_cleanup_ignores_never_created_paths(self):
        ta = TempArtifacts("pfx", dir=self.dir, policy="scoped")
        ta.path()  # allocated but never written
        ta.cleanup()  # must not raise

    def test_context_manager_deletes_on_success(self):
        with TempArtifacts("pfx", dir=self.dir, policy="scoped") as ta:
            p = self.touch(ta.path())
        self.assertFalse(os.path.exists(p))

    def test_context_manager_keeps_on_failure(self):
        try:
            with TempArtifacts("pfx", dir=self.dir, policy="scoped") as ta:
                p = self.touch(ta.path())
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertTrue(os.path.exists(p), "artifacts must survive for debugging")

    def test_on_cleanup_receives_removed_paths(self):
        seen = []
        ta = TempArtifacts(
            "pfx", dir=self.dir, policy="scoped", on_cleanup=lambda paths: seen.extend(paths)
        )
        p = self.touch(ta.path())
        ta.path()  # never written -> not passed to the callback
        ta.cleanup()
        self.assertEqual(seen, [p])

    def test_on_cleanup_exception_does_not_block_removal(self):
        def bad(paths):
            raise RuntimeError("callback boom")

        ta = TempArtifacts("pfx", dir=self.dir, policy="scoped", on_cleanup=bad)
        p = self.touch(ta.path())
        ta.cleanup()  # must not raise
        self.assertFalse(os.path.exists(p))

    def test_allocation_sweeps_stale_kept_on_failure_leftovers(self):
        # Keep-on-failure files have no other reclamation path: a later scoped
        # instance's first allocation must GC stale same-prefix leftovers.
        stale = self.touch(os.path.join(self.dir, "pfx_deadbeef.py"))
        self.age(stale, days=30)
        TempArtifacts("pfx", dir=self.dir, policy="scoped", max_age_days=7).path()
        self.assertFalse(os.path.exists(stale))


class TestDetachedPolicy(TempArtifactsBase):
    def test_cleanup_is_noop_without_force(self):
        ta = TempArtifacts("pfx", dir=self.dir, policy="detached")
        p = self.touch(ta.path())
        ta.cleanup()
        self.assertTrue(os.path.exists(p), "detached payloads must outlive the producer")

    def test_cleanup_force_removes(self):
        ta = TempArtifacts("pfx", dir=self.dir, policy="detached")
        p = self.touch(ta.path())
        ta.cleanup(force=True)
        self.assertFalse(os.path.exists(p))

    def test_allocation_sweeps_stale_same_prefix_files(self):
        stale = self.touch(os.path.join(self.dir, "pfx_deadbeef.fbx"))
        self.age(stale, days=30)
        ta = TempArtifacts("pfx", dir=self.dir, policy="detached", max_age_days=7)
        ta.path()
        self.assertFalse(os.path.exists(stale))

    def test_allocation_never_sweeps_fresh_files(self):
        fresh = self.touch(os.path.join(self.dir, "pfx_cafe.fbx"))
        ta = TempArtifacts("pfx", dir=self.dir, policy="detached", max_age_days=7)
        ta.path()
        self.assertTrue(
            os.path.exists(fresh),
            "a payload may still be read by a launched app — never sweep fresh files",
        )

    def test_sweep_ignores_other_prefixes(self):
        other = self.touch(os.path.join(self.dir, "otherpfx_old.fbx"))
        self.age(other, days=30)
        TempArtifacts("pfx", dir=self.dir, policy="detached", max_age_days=7).sweep_stale()
        self.assertTrue(os.path.exists(other))

    def test_sweep_returns_removed_paths(self):
        stale = self.touch(os.path.join(self.dir, "pfx_old.fbx"))
        self.age(stale, days=30)
        removed = TempArtifacts(
            "pfx", dir=self.dir, policy="detached", max_age_days=7
        ).sweep_stale()
        self.assertEqual(removed, [stale])


class TestSessionPolicy(TempArtifactsBase):
    def test_cleanup_removes_like_scoped(self):
        # atexit wiring can't be unit-tested meaningfully; explicit cleanup must work.
        ta = TempArtifacts("pfx", dir=self.dir, policy="session")
        p = self.touch(ta.path())
        ta.cleanup()
        self.assertFalse(os.path.exists(p))


class CachedArtifactBase(TempArtifactsBase):
    def setUp(self):
        super().setUp()
        self.cache = CachedArtifact("ca", extension=".out", dir=self.dir)
        self.src = self.touch(os.path.join(self.dir, "src.bin"), b"source")
        self.produced = []

    def produce(self, out):
        self.produced.append(out)
        self.touch(out, b"artifact")

    def produce_with_sidecar(self, out):
        self.produce(out)
        self.touch(out + ".manifest.json", b"{}")


class TestCachedArtifactKey(CachedArtifactBase):
    def test_key_is_stable_for_unchanged_inputs(self):
        self.assertEqual(
            CachedArtifact.key("a", files=[self.src]),
            CachedArtifact.key("a", files=[self.src]),
        )

    def test_key_changes_when_a_file_changes(self):
        before = CachedArtifact.key(files=[self.src])
        time.sleep(0.01)
        self.touch(self.src, b"source-edited")
        self.assertNotEqual(before, CachedArtifact.key(files=[self.src]))

    def test_key_changes_when_a_part_changes(self):
        self.assertNotEqual(
            CachedArtifact.key("fbx", files=[self.src]),
            CachedArtifact.key("usd", files=[self.src]),
        )

    def test_key_includes_every_file(self):
        """A template fix must invalidate stale payloads — so the producing script's
        identity has to be part of the key, not just the source's."""
        tpl = self.touch(os.path.join(self.dir, "tpl.py"), b"v1")
        before = CachedArtifact.key(files=[self.src, tpl])
        time.sleep(0.01)
        self.touch(tpl, b"v2")
        self.assertNotEqual(before, CachedArtifact.key(files=[self.src, tpl]))


class TestCachedArtifactGet(CachedArtifactBase):
    def test_miss_produces_then_promotes_into_the_cache_slot(self):
        got = self.cache.get("k1", self.produce)
        self.assertFalse(got.hit)
        self.assertTrue(os.path.isfile(got.path))
        self.assertTrue(os.path.basename(got.path).startswith("ca_cache_"))
        self.assertIsNotNone(got.scratch, "a miss hands back its scratch store to clean up")

    def test_hit_skips_production_entirely(self):
        first = self.cache.get("k1", self.produce)
        second = self.cache.get("k1", self.produce)
        self.assertTrue(second.hit)
        self.assertEqual(second.path, first.path)
        self.assertIsNone(second.scratch, "a hit must never be cleaned up — it IS the cache")
        self.assertEqual(len(self.produced), 1)

    def test_distinct_keys_get_distinct_slots(self):
        a = self.cache.get("k1", self.produce)
        b = self.cache.get("k2", self.produce)
        self.assertNotEqual(a.path, b.path)
        self.assertEqual(len(self.produced), 2)

    def test_sidecars_are_promoted_with_the_artifact(self):
        got = self.cache.get("k1", self.produce_with_sidecar, sidecars=(".manifest.json",))
        self.assertTrue(os.path.isfile(got.path + ".manifest.json"))

    def test_stale_sidecar_is_dropped_when_the_new_run_produced_none(self):
        """A sidecar must never outlive the artifact it describes: re-filling the SAME
        slot from a run that wrote no sidecar has to clear the previous one."""
        first = self.cache.get(
            "k1", self.produce_with_sidecar, sidecars=(".manifest.json",)
        )
        self.assertTrue(os.path.isfile(first.path + ".manifest.json"))

        os.remove(first.path)  # empty the slot so the next call misses and re-produces
        again = self.cache.get("k1", self.produce, sidecars=(".manifest.json",))
        self.assertEqual(again.path, first.path, "same key -> same slot")
        self.assertFalse(os.path.isfile(again.path + ".manifest.json"))

    def test_failure_keeps_the_partial_and_never_poisons_the_slot(self):
        def boom(out):
            self.touch(out, b"partial")
            raise RuntimeError("conversion died")

        with self.assertRaises(RuntimeError):
            self.cache.get("k1", boom)
        # The slot stays empty, so the next attempt re-produces rather than serving junk.
        got = self.cache.get("k1", self.produce)
        self.assertFalse(got.hit)
        self.assertEqual(open(got.path, "rb").read(), b"artifact")

    def test_use_cache_false_produces_into_scratch_every_time(self):
        a = self.cache.get("k1", self.produce, use_cache=False)
        b = self.cache.get("k1", self.produce, use_cache=False)
        self.assertFalse(a.hit)
        self.assertFalse(b.hit)
        self.assertNotEqual(a.path, b.path)
        self.assertNotIn("ca_cache_", os.path.basename(a.path))
        self.assertEqual(len(self.produced), 2)

    def test_empty_name_is_rejected(self):
        with self.assertRaises(ValueError):
            CachedArtifact("", extension=".out")


class TestRootExport(unittest.TestCase):
    def test_registered_on_package_root(self):
        import pythontk as ptk

        self.assertTrue(hasattr(ptk, "TempArtifacts"))
        self.assertTrue(hasattr(ptk, "CachedArtifact"))



class TempArtifactsDirectoryTest(unittest.TestCase):
    """Scratch DIRECTORIES are first-class, not a caller's problem.

    Before ``dir_path`` every site needing scratch space hand-rolled
    ``tempfile.mkdtemp`` + ``finally: shutil.rmtree``, so the lifecycle guarantees
    of this class had to be re-implemented per site -- and were simply absent
    wherever the ``finally`` was forgotten or the process died first.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="ta_dirtest_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _store(self, **kw):
        from pythontk import TempArtifacts

        return TempArtifacts("dtest", dir=self.root, **kw)

    def test_dir_path_creates_a_tracked_directory(self):
        store = self._store(policy="scoped")
        d = store.dir_path()
        self.assertTrue(os.path.isdir(d))
        self.assertTrue(os.path.basename(d).startswith("dtest_"))

    def test_create_false_reserves_without_making_it(self):
        d = self._store(policy="scoped").dir_path(create=False)
        self.assertFalse(os.path.exists(d))

    def test_cleanup_removes_a_NON_EMPTY_directory(self):
        """os.remove cannot delete a directory -- cleanup must recurse."""
        store = self._store(policy="scoped")
        d = store.dir_path()
        os.makedirs(os.path.join(d, "nested"))
        with open(os.path.join(d, "nested", "f.txt"), "w") as fh:
            fh.write("x")
        removed = store.cleanup(force=True)
        self.assertIn(d, removed)
        self.assertFalse(os.path.exists(d))

    def test_context_manager_removes_the_directory_on_clean_exit(self):
        from pythontk import TempArtifacts

        with TempArtifacts("dtest", dir=self.root, policy="scoped") as store:
            d = store.dir_path()
            with open(os.path.join(d, "f.txt"), "w") as fh:
                fh.write("x")
        self.assertFalse(os.path.exists(d))

    def test_failure_keeps_the_directory_and_reports_it(self):
        """A kept DIRECTORY must be reported -- __exit__ once checked isfile only,
        so a failed run said 'nothing kept' while a whole tree sat on disk."""
        from pythontk import TempArtifacts

        store = TempArtifacts("dtest", dir=self.root, policy="scoped")
        d = None
        with self.assertRaises(RuntimeError):
            with store as s:
                d = s.dir_path()
                raise RuntimeError("boom")
        self.assertTrue(os.path.isdir(d))

    def test_sweep_stale_reclaims_an_abandoned_directory(self):
        """The safety net: a process that dies leaves no finally to run, so a
        LATER run must be able to collect the leftover."""
        store = self._store(policy="detached")
        d = store.dir_path()
        f = os.path.join(d, "f.txt")
        with open(f, "w") as fh:
            fh.write("x")
        old = time.time() - 30 * 86400
        os.utime(d, (old, old))
        os.utime(f, (old, old))

        swept = self._store(policy="detached", max_age_days=7).sweep_stale()
        self.assertIn(d, swept)
        self.assertFalse(os.path.exists(d))

    def test_sweep_stale_spares_a_directory_with_a_fresh_file(self):
        """A file rewritten in place never touches its parent's mtime (Maya's .ma
        save), so a scratch dir someone keeps working in must be judged by its
        newest entry, not by the directory itself."""
        store = self._store(policy="detached")
        d = store.dir_path()
        f = os.path.join(d, "work.ma")
        with open(f, "w") as fh:
            fh.write("x")
        old = time.time() - 30 * 86400
        os.utime(d, (old, old))  # the dir looks abandoned; the file inside is fresh

        self.assertEqual(self._store(policy="detached", max_age_days=7).sweep_stale(), [])
        self.assertTrue(os.path.isfile(f))

    def test_sweep_stale_spares_a_FRESH_directory(self):
        store = self._store(policy="detached")
        d = store.dir_path()
        self.assertEqual(self._store(policy="detached").sweep_stale(), [])
        self.assertTrue(os.path.isdir(d))

    def test_sweep_stale_ignores_another_prefix(self):
        from pythontk import TempArtifacts

        other = TempArtifacts("otherprefix", dir=self.root, policy="detached")
        d = other.dir_path()
        old = time.time() - 30 * 86400
        os.utime(d, (old, old))
        self.assertEqual(self._store(policy="detached", max_age_days=7).sweep_stale(), [])
        self.assertTrue(os.path.isdir(d))

    def test_files_and_dirs_coexist_in_one_store(self):
        store = self._store(policy="scoped")
        f = store.path(extension=".txt")
        with open(f, "w") as fh:
            fh.write("x")
        d = store.dir_path()
        removed = store.cleanup(force=True)
        self.assertEqual(sorted(removed), sorted([f, d]))
        self.assertFalse(os.path.exists(f))
        self.assertFalse(os.path.exists(d))


class TestScratchTwins(TempArtifactsBase):
    """Per-source scratch twins: provenance naming, per-source dirs, untouched-only discard."""

    def twins(self, **kw):
        return ScratchTwins("tw", extension=".blend", dir=self.dir, **kw)

    def test_path_carries_source_type_and_is_per_source(self):
        t = self.twins()
        a = t.path_for(os.path.join(self.dir, "projA", "shot.ma"))
        b = t.path_for(os.path.join(self.dir, "projB", "shot.ma"))
        fbx = t.path_for(os.path.join(self.dir, "projA", "shot.FBX"))
        self.assertEqual(os.path.basename(a), "shot_ma.blend")
        self.assertEqual(os.path.basename(fbx), "shot_fbx.blend")
        self.assertTrue(os.path.basename(os.path.dirname(a)).startswith("tw_"))
        self.assertEqual(os.path.dirname(os.path.dirname(a)), self.dir)
        self.assertNotEqual(os.path.dirname(a), os.path.dirname(b))
        # Deterministic, and nothing is created by asking.
        self.assertEqual(t.path_for(os.path.join(self.dir, "projA", "shot.ma")), a)
        self.assertFalse(os.path.exists(os.path.dirname(a)))

    def test_create_copies_and_stamps(self):
        t = self.twins()
        payload = self.touch(os.path.join(self.dir, "bake.blend"), b"bake")
        source = os.path.join(self.dir, "proj", "shot.ma")
        path = t.create(source, payload)
        self.assertEqual(path, t.path_for(source))
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"bake")
        self.assertTrue(t.is_twin(path))
        self.assertFalse(t.is_twin(payload))

    def test_discard_removes_untouched_keeps_saved_into(self):
        t = self.twins()
        payload = self.touch(os.path.join(self.dir, "bake.blend"), b"bake")
        source = os.path.join(self.dir, "proj", "shot.ma")
        path = t.create(source, payload)
        self.assertTrue(t.discard(path))
        self.assertFalse(os.path.exists(path))
        self.assertFalse(t.is_twin(path))

        path = t.create(source, payload)
        self.touch(path, b"the user saved real work here")  # stamp changes
        self.assertFalse(t.discard(path))
        self.assertTrue(os.path.exists(path))
        self.assertFalse(t.is_twin(path))  # forgotten either way
        self.assertFalse(t.discard(path))  # unknown now -> no-op
        self.assertFalse(t.discard(os.path.join(self.dir, "never.blend")))
        self.assertFalse(t.discard(""))

    def test_discard_except_keeps_the_current_file(self):
        t = self.twins()
        payload = self.touch(os.path.join(self.dir, "bake.blend"), b"bake")
        a = t.create(os.path.join(self.dir, "proj", "a.ma"), payload)
        b = t.create(os.path.join(self.dir, "proj", "b.ma"), payload)
        removed = t.discard_except(a.upper() if os.name == "nt" else a)  # case-insensitive on Windows
        # Reported exactly as create() handed them out -- NOT the normcased internal
        # key, which on Windows is lowercased and compares unequal to path_for().
        self.assertEqual(removed, [b])
        self.assertTrue(os.path.exists(a))
        self.assertFalse(os.path.exists(b))
        self.assertEqual(t.discard_except(None), [a])
        self.assertFalse(os.path.exists(a))

    def test_create_never_overwrites_a_twin_the_user_saved_into(self):
        """Re-opening the same source must not destroy work saved into its twin.

        The twin path is deterministic per source, so a second ``create`` for the same
        source lands on the same file. Both routes to that second call are covered:
        in-session (``discard_except`` keeps the saved twin but stops tracking it) and
        across a host restart (the in-memory stamps are gone entirely).
        """
        payload = self.touch(os.path.join(self.dir, "bake.blend"), b"bake")
        work = b"the user saved real work here"

        for label, restart in (("in-session", False), ("after restart", True)):
            with self.subTest(label):
                # A source per case: the twin path is derived from it, and the two
                # cases would otherwise share (and see) each other's preserved file.
                source = os.path.join(self.dir, "proj", f"shot_{int(restart)}.ma")
                t = self.twins()
                path = t.create(source, payload)
                self.touch(path, work)
                if restart:
                    t = self.twins()  # a new process: nothing is tracked
                else:
                    t.discard_except(path)  # keeps it, but forgets it

                again = t.create(source, payload)

                # Same deterministic path -- the panels recompute it to answer
                # "is this row the current scene?", so it must not move.
                self.assertEqual(again, path)
                with open(again, "rb") as f:
                    self.assertEqual(f.read(), b"bake")  # fresh conversion landed
                # ...and the work is still on disk, beside it.
                kept = [
                    os.path.join(os.path.dirname(path), f)
                    for f in os.listdir(os.path.dirname(path))
                    if "_saved" in f
                ]
                self.assertEqual(
                    len(kept), 1, f"expected one preserved file, got {kept}"
                )
                with open(kept[0], "rb") as f:
                    self.assertEqual(f.read(), work)

    def test_create_reuses_the_path_when_the_twin_is_untouched(self):
        """The common case must NOT accumulate _saved copies."""
        t = self.twins()
        payload = self.touch(os.path.join(self.dir, "bake.blend"), b"bake")
        source = os.path.join(self.dir, "proj", "shot.ma")
        path = t.create(source, payload)
        for _ in range(3):
            self.assertEqual(t.create(source, payload), path)
        siblings = [f for f in os.listdir(os.path.dirname(path)) if "_saved" in f]
        self.assertEqual(siblings, [])

    def test_abandoned_twin_dirs_are_age_swept(self):
        t = self.twins(max_age_days=1)
        payload = self.touch(os.path.join(self.dir, "bake.blend"), b"bake")
        old = t.create(os.path.join(self.dir, "proj", "old.ma"), payload)
        # sweep_stale keys off the NEWEST mtime under the dir, so "abandoned" means
        # every file in it is old -- the twin AND the stamp sidecar beside it.
        twin_dir = os.path.dirname(old)
        self.age(twin_dir, 3)
        for name in os.listdir(twin_dir):
            self.age(os.path.join(twin_dir, name), 3)
        ScratchTwins("tw", extension=".blend", dir=self.dir, max_age_days=1).path_for(
            os.path.join(self.dir, "proj", "new.ma")
        )  # a later store's first allocation sweeps
        self.assertFalse(os.path.exists(os.path.dirname(old)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
