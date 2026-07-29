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

from pythontk.file_utils.temp_artifacts import CachedArtifact, TempArtifacts


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
