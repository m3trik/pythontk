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
import stat
import subprocess
import sys
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

    def test_default_dir_is_the_stores_own_folder_under_the_root(self):
        # Assert on .dir rather than allocating: path() on an unsandboxed
        # instance would run a live sweep in the user's real temp dir.
        self.assertEqual(
            TempArtifacts("pfx").dir, os.path.join(TempArtifacts.root(), "pfx")
        )

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


class TestRelease(TempArtifactsBase):
    """One artifact, deleted early because its consumer is provably done.

    The gap `release` fills: a DETACHED store cannot delete (its payloads
    outlive the producer by design), so an intermediate a *blocking* run fully
    consumes -- a preview's FBX, converted to GLB and never read again -- sat
    for the whole `max_age_days` window. Measured on a production assembly:
    324 MB per push, 3.1 GB of them left in the system temp dir.
    """

    def test_release_deletes_a_tracked_path_under_any_policy(self):
        ta = TempArtifacts("pfx", dir=self.dir, policy="detached")
        p = self.touch(ta.path(extension=".fbx"))
        self.assertTrue(ta.release(p))
        self.assertFalse(os.path.exists(p))

    def test_release_untracks_so_cleanup_does_not_re_report_it(self):
        ta = TempArtifacts("pfx", dir=self.dir, policy="scoped")
        p = self.touch(ta.path())
        ta.release(p)
        self.assertEqual(ta.cleanup(), [])

    def test_release_refuses_a_path_this_store_never_minted(self):
        """The ownership guarantee. `release` is called by a *deliverer*, which
        is a pluggable strategy: mounted on a bridge whose producer hands back
        a durable path, an unguarded delete would destroy the caller's file."""
        ta = TempArtifacts("pfx", dir=self.dir, policy="scoped")
        foreign = self.touch(os.path.join(self.dir, "someone_elses.fbx"))
        self.assertFalse(ta.release(foreign))
        self.assertTrue(os.path.exists(foreign))

    def test_release_of_a_never_created_path_is_not_an_error(self):
        ta = TempArtifacts("pfx", dir=self.dir, policy="scoped")
        self.assertFalse(ta.release(ta.path()))

    def test_a_path_it_could_not_remove_stays_tracked_for_cleanup_to_retry(self):
        """A file another process still holds open is the ordinary failure. It
        must stay in the store's ledger, or the scope-exit cleanup that would
        have caught it never looks at it again."""
        ta = TempArtifacts("pfx", dir=self.dir, policy="scoped")
        p = self.touch(ta.path())
        with mock.patch("os.remove", side_effect=OSError("locked")):
            self.assertFalse(ta.release(p))
        self.assertEqual(ta.cleanup(), [p], "the retry never happened")


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
            "pfx",
            dir=self.dir,
            policy="scoped",
            on_cleanup=lambda paths: seen.extend(paths),
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
        self.assertTrue(
            os.path.exists(p), "detached payloads must outlive the producer"
        )

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
        TempArtifacts(
            "pfx", dir=self.dir, policy="detached", max_age_days=7
        ).sweep_stale()
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
        ta = TempArtifacts("pfx", dir=self.dir, policy="session")
        p = self.touch(ta.path())
        ta.cleanup()
        self.assertFalse(os.path.exists(p))

    def _leave_by(self, exit_line):
        """A child fills a session store, then exits by *exit_line*; returns
        the store's directory."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        body = (
            "import os, sys\n"
            f"sys.path.insert(0, {root!r})\n"
            "from pythontk.file_utils.temp_artifacts import TempArtifacts\n"
            "from pythontk.core_utils.process_exit import ProcessExit\n"
            f"d = TempArtifacts('pfx', dir={self.dir!r}, policy='session').dir_path()\n"
            "open(os.path.join(d, 'f.txt'), 'w').close()\n"
            "print(d, end='', flush=True)\n" + exit_line
        )
        proc = subprocess.run(
            [sys.executable, "-c", body], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(proc.stdout, "the child never allocated")
        return proc.stdout

    def test_an_ordinary_exit_removes_the_store(self):
        self.assertFalse(os.path.exists(self._leave_by("sys.exit(0)")))

    def test_a_hard_exit_removes_the_store(self):
        """``ProcessExit.hard_exit`` skips ``atexit`` -- and it is how every
        DCC child and several test runners here end -- so the store has to
        ask it directly. Before it did, each such exit leaked its whole test
        sandbox: 791 of them, 572 MB, in the system temp dir after a week."""
        self.assertFalse(os.path.exists(self._leave_by("ProcessExit.hard_exit(0)")))


class TestSessionOwnership(TempArtifactsBase):
    """A session store names its owner; one whose owner has exited goes at once.

    A session store goes at its owner's exit only as far as it will go: a
    process that is killed never runs that cleanup, and one whose files
    something still holds -- a DCC's helper writing into the TEMP it was
    handed -- leaves the whole store. Seven days of age sweep then kept them
    all: 67 test-sandbox roots in four days (measured 2026-10-05), every one
    free to delete by the next run. The tag carries the owner's id, and the
    sweep any store runs on its first allocation reclaims an entry whose owner
    has exited, at any age.
    """

    def _exited_pid(self):
        """The id of a process that has exited (and was not reused since)."""
        from pythontk.core_utils.execution_monitor._sidecar import process_alive

        for _ in range(5):
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            child.wait()
            if not process_alive(child.pid):
                return child.pid
        self.skipTest("every exited child's id was reused at once")

    def _entry(self, name, days=0):
        path = os.path.join(self.dir, name)
        os.makedirs(path)
        self.touch(os.path.join(path, "held_at_exit.log"))
        if days:
            self.age(os.path.join(path, "held_at_exit.log"), days)
            self.age(path, days)
        return path

    def test_a_session_tag_names_its_owner(self):
        session = TempArtifacts("pfx", dir=self.dir, policy="session")
        self.assertRegex(
            os.path.basename(session.path(extension=".fbx")),
            rf"^pfx_[0-9a-f]{{12,}}-{os.getpid()}\.fbx$",
        )
        self.assertTrue(session.dir_path().endswith(f"-{os.getpid()}"))
        for policy in ("scoped", "detached"):  # outlived on purpose, or owned by no one
            with self.subTest(policy=policy):
                other = TempArtifacts("pfx", dir=self.dir, policy=policy).path()
                self.assertNotIn("-", os.path.basename(other))
        fixed = session.path(name="script", extension=".lua")
        self.assertEqual(os.path.basename(fixed), "pfx_script.lua")

    def test_an_entry_whose_owner_exited_is_reclaimed_at_any_age(self):
        orphan = self._entry(f"pfx_18dba0000000001-{self._exited_pid()}")
        live = self._entry(f"pfx_18dba0000000002-{os.getpid()}")
        unnamed = self._entry("pfx_18dba0000000003")  # before tags named owners
        stale = self._entry("pfx_18dba0000000004", days=10)
        other = self._entry(f"other_18dba0000000005-{self._exited_pid()}")

        TempArtifacts("pfx", dir=self.dir, policy="session").path()

        self.assertFalse(os.path.exists(orphan), "its owner is gone")
        self.assertFalse(os.path.exists(stale), "the age sweep still runs")
        for kept in (live, unnamed, other):
            self.assertTrue(os.path.isdir(kept), kept)

    def test_an_id_no_process_could_have_is_left_to_the_age_sweep(self):
        odd = self._entry("pfx_18dba0000000006-99999999999")
        TempArtifacts("pfx", dir=self.dir, policy="session").path()
        self.assertTrue(os.path.isdir(odd))

    def test_a_killed_owners_store_goes_at_the_next_allocation(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        body = (
            "import os, sys\n"
            f"sys.path.insert(0, {root!r})\n"
            "from pythontk.file_utils.temp_artifacts import TempArtifacts\n"
            f"d = TempArtifacts('pfx', dir={self.dir!r}, policy='session').dir_path()\n"
            "open(os.path.join(d, 'f.txt'), 'w').close()\n"
            "print(d, end='', flush=True)\n"
            "os.kill(os.getpid(), 9)  # no exit cleanup runs\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", body], capture_output=True, text=True, timeout=60
        )
        left = proc.stdout
        self.assertTrue(left and os.path.isdir(left), "the killed store must remain")

        TempArtifacts("pfx", dir=self.dir, policy="session").path()

        self.assertFalse(os.path.exists(left))


class TestHeldEntries(TempArtifactsBase):
    """An entry that cannot be deleted costs only itself, never its tree.

    ``shutil.rmtree`` stops at the first entry it cannot delete, and on Windows
    that is routine: a helper a DCC spawns outlives it holding a file in the
    DCC's temp dir. One held file kept a whole test sandbox where only that
    file had to stay.
    """

    def _tree(self, store):
        """A scratch dir three levels deep; returns (dir, [files])."""
        d = store.dir_path()
        os.makedirs(os.path.join(d, "a", "b"))
        names = ("0.txt", os.path.join("a", "1.txt"), os.path.join("a", "b", "2.txt"))
        return d, [self.touch(os.path.join(d, n)) for n in names + ("z.txt",)]

    @staticmethod
    def _holding(held):
        """Refuse to delete *held*, as a handle another process holds does.

        Matched by name: POSIX ``rmtree`` unlinks a bare name against a
        directory descriptor, never the full path."""
        real = os.unlink

        def unlink(path, *args, **kwargs):
            if os.path.basename(path) == os.path.basename(held):
                raise PermissionError(13, "held by another process", path)
            return real(path, *args, **kwargs)

        return mock.patch("os.unlink", side_effect=unlink)

    @staticmethod
    def _present(files):
        return [f for f in files if os.path.exists(f)]

    def test_cleanup_removes_everything_but_the_held_entry(self):
        ta = TempArtifacts("pfx", dir=self.dir, policy="scoped")
        d, files = self._tree(ta)
        with self._holding(files[1]):
            self.assertEqual(ta.cleanup(), [])
        self.assertEqual(self._present(files), [files[1]])
        # Still tracked: once the holder lets go, the next cleanup finishes it.
        self.assertEqual(ta.cleanup(), [d])
        self.assertFalse(os.path.exists(d))

    def test_release_removes_everything_but_the_held_entry(self):
        ta = TempArtifacts("pfx", dir=self.dir, policy="detached")
        d, files = self._tree(ta)
        with self._holding(files[2]):
            self.assertFalse(ta.release(d))
        self.assertEqual(self._present(files), [files[2]])
        self.assertTrue(ta.release(d))

    def test_sweep_removes_everything_but_the_held_entry(self):
        d, files = self._tree(TempArtifacts("pfx", dir=self.dir, policy="detached"))
        for parent, dirs, names in os.walk(d):
            for name in dirs + names:
                self.age(os.path.join(parent, name), 30)
        self.age(d, 30)
        with self._holding(files[0]):
            swept = TempArtifacts("pfx", dir=self.dir, max_age_days=7).sweep_stale()
        self.assertEqual(swept, [], "reported as gone while an entry remains")
        self.assertEqual(self._present(files), [files[0]])

    def test_a_read_only_file_does_not_stop_removal(self):
        """Windows refuses to delete a read-only entry (a git object, a file
        copied off read-only media); it goes once made writable."""
        ta = TempArtifacts("pfx", dir=self.dir, policy="scoped")
        d, files = self._tree(ta)
        os.chmod(files[1], stat.S_IREAD)
        self.assertEqual(ta.cleanup(), [d])
        self.assertFalse(os.path.exists(d))

    @unittest.skipUnless(os.name == "nt", "only Windows refuses to delete an open file")
    def test_a_file_held_open_costs_only_itself(self):
        """The real case the patched ones stand in for."""
        ta = TempArtifacts("pfx", dir=self.dir, policy="scoped")
        d, files = self._tree(ta)
        with open(files[1], "rb"):
            self.assertEqual(ta.cleanup(), [])
            self.assertEqual(self._present(files), [files[1]])
        self.assertEqual(ta.cleanup(), [d])


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
        self.assertIsNotNone(
            got.scratch, "a miss hands back its scratch store to clean up"
        )

    def test_hit_skips_production_entirely(self):
        first = self.cache.get("k1", self.produce)
        second = self.cache.get("k1", self.produce)
        self.assertTrue(second.hit)
        self.assertEqual(second.path, first.path)
        self.assertIsNone(
            second.scratch, "a hit must never be cleaned up — it IS the cache"
        )
        self.assertEqual(len(self.produced), 1)

    def test_distinct_keys_get_distinct_slots(self):
        a = self.cache.get("k1", self.produce)
        b = self.cache.get("k2", self.produce)
        self.assertNotEqual(a.path, b.path)
        self.assertEqual(len(self.produced), 2)

    def test_sidecars_are_promoted_with_the_artifact(self):
        got = self.cache.get(
            "k1", self.produce_with_sidecar, sidecars=(".manifest.json",)
        )
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

        self.assertEqual(
            self._store(policy="detached", max_age_days=7).sweep_stale(), []
        )
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
        self.assertEqual(
            self._store(policy="detached", max_age_days=7).sweep_stale(), []
        )
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
        removed = t.discard_except(
            a.upper() if os.name == "nt" else a
        )  # case-insensitive on Windows
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


def _norm(path):
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _age_tree(path, days):
    """Backdate *path* and everything under it by *days*."""
    past = time.time() - days * 86400
    for parent, dirs, files in os.walk(path):
        for name in dirs + files:
            os.utime(os.path.join(parent, name), (past, past))
    os.utime(path, (past, past))


class _RootedBase(TempArtifactsBase):
    """A store with no ``dir`` lives under the temp ROOT: point it here.

    ``PYTHONTK_TEMP_ROOT`` names the root; the system temp dir (where the
    pre-root flat entries live) is pointed at a folder of its own, so the
    one-release legacy sweep scans only what a test put there.
    """

    def setUp(self):
        super().setUp()
        self.root = os.path.join(self.dir, "root")
        self.legacy = os.path.join(self.dir, "legacy")
        os.makedirs(self.legacy)
        for patcher in (
            mock.patch.dict(os.environ, {TempArtifacts.ROOT_ENV_VAR: self.root}),
            mock.patch.object(tempfile, "tempdir", self.legacy),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)


class TestRoot(TempArtifactsBase):
    """ONE temp root: every store with no ``dir`` lives under it."""

    def test_the_env_var_names_the_root_created_on_demand(self):
        target = os.path.join(self.dir, "named root")
        with mock.patch.dict(os.environ, {TempArtifacts.ROOT_ENV_VAR: target}):
            self.assertEqual(_norm(TempArtifacts.root()), _norm(target))
        self.assertTrue(os.path.isdir(target))

    def test_without_it_the_root_is_one_folder_in_the_system_temp_dir(self):
        with (
            mock.patch.dict(os.environ),
            mock.patch.object(tempfile, "tempdir", self.dir),
        ):
            os.environ.pop(TempArtifacts.ROOT_ENV_VAR, None)
            self.assertEqual(
                _norm(TempArtifacts.root()), _norm(os.path.join(self.dir, "ptk"))
            )

    @unittest.skipUnless(os.name == "nt", "the ANSI code page is Windows'")
    def test_the_root_is_spelled_so_an_ansi_program_can_open_it(self):
        """Maya opens files through the ANSI code page: a root under a folder
        it cannot spell must reach every store in its 8.3 form."""
        from pythontk.core_utils.app_launcher import AppLauncher

        odd = os.path.join(self.dir, "Zoë ★")
        os.makedirs(odd)
        if not (AppLauncher._short_name(odd) or "").isascii():
            self.skipTest("no 8.3 short names on this volume")
        target = os.path.join(odd, "root")
        with mock.patch.dict(os.environ, {TempArtifacts.ROOT_ENV_VAR: target}):
            root = TempArtifacts.root()
        self.assertTrue(AppLauncher._ansi_encodable(root), ascii(root))
        self.assertTrue(os.path.samefile(root, target))


class TestStoreFolders(_RootedBase):
    """Each store owns ``<root>/<prefix>/``: one store's sweep can never reach
    another's entries -- a prefix that extends another (``glb_pipeline`` /
    ``glb_pipeline_media``, a cache's ``<name>`` / ``<name>_cache``) was swept
    by the shorter one's age, so a cache's own ``max_age_days`` never held."""

    def test_a_default_store_owns_one_folder_under_the_root(self):
        store = TempArtifacts("pfx", policy="scoped")
        path = store.path(extension=".fbx")
        folder = os.path.join(self.root, "pfx")
        self.assertEqual(_norm(store.dir), _norm(folder))
        self.assertEqual(_norm(os.path.dirname(path)), _norm(folder))
        self.assertTrue(os.path.basename(path).startswith("pfx_"))
        self.assertTrue(os.path.isdir(folder), "the caller writes straight into it")

    def test_a_longer_prefix_is_another_store_entirely(self):
        media = TempArtifacts("glb_pipeline_media", max_age_days=30).dir_path()
        _age_tree(media, 10)  # inside ITS age, past the shorter prefix's
        TempArtifacts("glb_pipeline", max_age_days=7).path()
        self.assertTrue(os.path.isdir(media), "swept by another store's age")

    def test_a_caches_own_max_age_is_authoritative(self):
        cache = CachedArtifact("ca", extension=".out", max_age_days=30)
        got = cache.get("k1", lambda out: self.touch(out))
        got.scratch.cleanup()
        self.age(got.path, 10)
        TempArtifacts("ca", max_age_days=7).path()  # the scratch prefix
        self.assertTrue(os.path.isfile(got.path))

    def test_the_store_folder_records_the_age_its_owner_declared(self):
        import json

        TempArtifacts("declared", max_age_days=3).path()
        TempArtifacts("forever", max_age_days=None).path()
        for prefix, age in (("declared", 3), ("forever", None)):
            marker = os.path.join(self.root, prefix, TempArtifacts.STORE_MARKER)
            with open(marker, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["max_age_days"], age, prefix)

    def test_the_marker_keeps_the_longest_age_any_alias_declared(self):
        """Last writer won: a store spelled with a shorter default (an alias,
        a caller that forgot ``max_age_days``) re-declared the folder and the
        daily ``sweep_all`` aged out the longer-lived cache by it. ``None``
        (never aged) is the longest."""
        import json

        def declared(prefix):
            marker = os.path.join(self.root, prefix, TempArtifacts.STORE_MARKER)
            with open(marker, encoding="utf-8") as fh:
                return json.load(fh)["max_age_days"]

        for prefix, ages, want in (
            ("alias_long", (30, 7), 30),
            ("alias_grow", (7, 30), 30),
            ("alias_none", (None, 7), None),
            ("alias_none_late", (7, None), None),
        ):
            for age in ages:
                TempArtifacts(prefix, max_age_days=age).path()
            self.assertEqual(declared(prefix), want, prefix)
            TempArtifacts._markers.clear()  # a new process reads the file
            TempArtifacts(prefix, max_age_days=1).path()
            self.assertEqual(declared(prefix), want, f"{prefix}, next process")

    def test_max_age_none_is_never_age_swept(self):
        store = TempArtifacts("keepers", max_age_days=None)
        old = store.dir_path()
        _age_tree(old, 3650)
        TempArtifacts("keepers", max_age_days=None).sweep_stale()
        TempArtifacts.sweep_all()
        self.assertTrue(os.path.isdir(old))

    def test_allocation_never_rescans_the_system_temp_dir(self):
        """Store construction is per call all over the ecosystem; each first
        allocation scanned the whole system temp dir (24,500 entries on one
        workstation). The one-release legacy sweep reads it once a process."""
        scanned = []
        real = os.scandir

        def spy(path=".", *args, **kwargs):
            scanned.append(_norm(os.fspath(path)))
            return real(path, *args, **kwargs)

        with mock.patch("os.scandir", side_effect=spy):
            for _ in range(3):
                TempArtifacts("cheap", policy="scoped").path()
        self.assertLessEqual(scanned.count(_norm(self.legacy)), 1, scanned)


class TestLegacyFlatSweep(_RootedBase):
    """For one release a store's first sweep also reclaims what the pre-root
    layout left in the system temp dir (``<temp>/<prefix>_*``), by its own age."""

    def test_stale_legacy_entries_go_and_fresh_ones_stay(self):
        old = self.touch(os.path.join(self.legacy, "legacy_pfx_deadbeef.fbx"))
        self.age(old, 30)
        fresh = self.touch(os.path.join(self.legacy, "legacy_pfx_cafe.fbx"))
        other = self.touch(os.path.join(self.legacy, "other_deadbeef.fbx"))
        self.age(other, 30)
        TempArtifacts("legacy_pfx", max_age_days=7).path()
        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.path.exists(fresh), "a fresh payload may still be read")
        self.assertTrue(os.path.exists(other), "another prefix's")

    def test_a_store_folder_sitting_in_the_temp_dir_is_never_a_legacy_entry(self):
        # A root that IS the system temp dir (PYTHONTK_TEMP_ROOT pointed at it)
        # holds every store folder beside the flat leftovers.
        folder = os.path.join(self.legacy, "legacy_pfx_media")
        os.makedirs(folder)
        self.touch(os.path.join(folder, TempArtifacts.STORE_MARKER), b"{}")
        _age_tree(folder, 30)
        TempArtifacts("legacy_pfx", max_age_days=7).path()
        self.assertTrue(os.path.isdir(folder))

    def test_the_legacy_flat_sweep_is_retired_on_schedule(self):
        from pythontk import __version__
        from pythontk.core_utils.deprecation import Deprecation

        self.assertFalse(
            Deprecation.window_expired(
                TempArtifacts._LEGACY_SWEEP_REMOVE_IN,
                __version__,
                TempArtifacts._LEGACY_SWEEP_SINCE,
            ),
            "the one-release migration window is over: delete "
            "TempArtifacts._sweep_legacy, the legacy sweep in TestSandbox.temp, "
            "the system temp dir in TempArtifacts.contains and this test",
        )

    def test_contains_reads_the_root_and_the_legacy_temp_dir(self):
        """One spelling of "a path TempArtifacts holds" for a host: the root,
        and for the legacy window the system temp dir its flat layout used (a
        scratch twin minted before the root). Was ``_in_temp`` in both DCCs."""
        inside = os.path.join(TempArtifacts.root(), "pfx", "twin.ma")
        flat = self.touch(os.path.join(self.legacy, "pfx_deadbeef.ma"))
        outside = os.path.join(self.dir, "proj", "scenes", "hero.ma")
        self.assertTrue(TempArtifacts.contains(inside))
        self.assertTrue(TempArtifacts.contains(TempArtifacts.root()))
        self.assertTrue(TempArtifacts.contains(flat))
        self.assertFalse(TempArtifacts.contains(outside))
        self.assertFalse(TempArtifacts.contains(self.root + "_sibling"))
        self.assertFalse(TempArtifacts.contains(""))
        self.assertFalse(TempArtifacts.contains("bad\x00path"))


class TestUniqueTags(TempArtifactsBase):
    """Two processes in one clock tick never mint one path.

    The tag was the allocation clock alone, and Python 3.11's clock on Windows
    ticks every ~15 ms: two processes allocating at once got the same path,
    and ``dir_path`` (``makedirs(exist_ok=True)``) silently shared the folder.
    """

    FROZEN = 1_700_000_000_000_000_000

    def test_two_processes_in_one_clock_tick_get_distinct_paths(self):
        with mock.patch.object(TempArtifacts, "_next_tag_ns", return_value=self.FROZEN):
            a = TempArtifacts("pfx", dir=self.dir).path()
            with mock.patch("os.getpid", return_value=os.getpid() + 1):
                b = TempArtifacts("pfx", dir=self.dir).path()
        self.assertNotEqual(a, b)

    def test_a_reused_process_id_in_one_tick_still_gets_a_fresh_path(self):
        with mock.patch.object(TempArtifacts, "_next_tag_ns", return_value=self.FROZEN):
            paths = {TempArtifacts("pfx", dir=self.dir).path() for _ in range(8)}
        self.assertGreater(len(paths), 1, "no random part")

    def test_dir_path_never_shares_a_directory_that_already_exists(self):
        taken = os.path.join(self.dir, "pfx_taken")
        os.makedirs(taken)
        store = TempArtifacts("pfx", dir=self.dir, policy="scoped")
        with mock.patch.object(
            TempArtifacts, "_mint_tag", side_effect=["taken", "free"]
        ):
            got = store.dir_path()
        self.assertEqual(os.path.basename(got), "pfx_free")
        store.cleanup()
        self.assertTrue(os.path.isdir(taken), "the other owner's folder was taken")


class TestKeepAndScoped(TempArtifactsBase):
    def test_keep_disowns_a_path_so_scope_exit_leaves_it(self):
        with TempArtifacts("pfx", dir=self.dir, policy="scoped") as store:
            kept = store.dir_path()
            gone = store.dir_path()
            self.assertTrue(store.keep(kept))
            self.assertFalse(store.keep(os.path.join(self.dir, "not_ours")))
        self.assertTrue(os.path.isdir(kept), "the stage holds the only copy")
        self.assertFalse(os.path.exists(gone))

    def test_scoped_spells_the_scoped_policy(self):
        with TempArtifacts.scoped("pfx", dir=self.dir) as store:
            self.assertEqual(store.policy, "scoped")
            path = self.touch(store.path())
        self.assertFalse(os.path.exists(path))

    def test_a_sweep_interval_spares_a_folder_a_rescan(self):
        """``atomic_write`` builds one store per write into the TARGET's folder:
        a batch of sidecars saved into one folder scanned it once per file."""
        first = self.touch(os.path.join(self.dir, "pfx_first.part"))
        self.age(first, 30)
        TempArtifacts("pfx", dir=self.dir, sweep_interval=600).path()
        self.assertFalse(os.path.exists(first))
        second = self.touch(os.path.join(self.dir, "pfx_second.part"))
        self.age(second, 30)
        TempArtifacts("pfx", dir=self.dir, sweep_interval=600).path()
        self.assertTrue(os.path.exists(second), "rescanned within the interval")
        TempArtifacts("pfx", dir=self.dir).path()
        self.assertFalse(os.path.exists(second))


class TestSweepAll(_RootedBase):
    """A prefix that never allocates again is swept by no one but this."""

    def _store_entry(self, prefix, max_age_days, days):
        entry = TempArtifacts(prefix, max_age_days=max_age_days).dir_path()
        self.touch(os.path.join(entry, "f.txt"))
        _age_tree(entry, days)
        return entry

    def test_an_orphaned_prefix_goes_by_the_age_its_owner_declared(self):
        orphan = self._store_entry("orphan", 1, days=2)
        young = self._store_entry("young", 30, days=10)
        forever = self._store_entry("forever", None, days=3650)
        removed = TempArtifacts.sweep_all()
        self.assertEqual([_norm(p) for p in removed], [_norm(orphan)])
        self.assertTrue(os.path.isdir(young))
        self.assertTrue(os.path.isdir(forever))

    def test_an_unreadable_declaration_is_read_conservatively(self):
        store = TempArtifacts("garbled", max_age_days=1)
        fresh, stale = store.dir_path(), store.dir_path()
        _age_tree(fresh, 10)
        _age_tree(stale, TempArtifacts.UNKNOWN_MAX_AGE_DAYS + 5)
        with open(
            os.path.join(self.root, "garbled", TempArtifacts.STORE_MARKER), "w"
        ) as fh:
            fh.write("not json")
        TempArtifacts.sweep_all()
        self.assertTrue(os.path.isdir(fresh), "inside the conservative default")
        self.assertFalse(os.path.exists(stale))

    def test_a_folder_that_is_no_store_is_never_touched(self):
        foreign = os.path.join(self.root, "someone", "someone_else")
        os.makedirs(foreign)
        _age_tree(os.path.dirname(foreign), 3650)
        TempArtifacts.sweep_all()
        self.assertTrue(os.path.isdir(foreign))

    def test_it_runs_on_its_own_about_once_a_day(self):
        first = os.path.join(self.root, "abandoned")
        os.makedirs(first)
        with open(os.path.join(first, TempArtifacts.STORE_MARKER), "w") as fh:
            fh.write('{"max_age_days": 1}')
        entry = os.path.join(first, "abandoned_1")
        os.makedirs(entry)
        _age_tree(entry, 5)
        TempArtifacts("active", policy="scoped").path()
        self.assertFalse(os.path.exists(entry), "the first allocation sweeps")
        os.makedirs(entry)
        _age_tree(entry, 5)
        TempArtifacts("active2", policy="scoped").path()
        self.assertTrue(os.path.isdir(entry), "swept again within the day")


class TestCachedArtifactSlots(_RootedBase):
    """The cache's slot model, public: what a key maps to, whether it holds a
    complete entry, and refreshing its age -- what mayatk rebuilt by hand
    from the private naming (``_bake_slot`` / ``_touch_bake``)."""

    def produce(self, out):
        self.touch(out, b"artifact")

    def test_slot_names_a_keys_path_and_creates_nothing(self):
        project = os.path.join(self.dir, "project", "cache", "bake")
        cache = CachedArtifact("bk", extension=".ma", dir=project)
        slot = cache.slot("abc")
        self.assertEqual(_norm(os.path.dirname(slot)), _norm(project))
        self.assertEqual(os.path.basename(slot), "bk_cache_abc.ma")
        self.assertIsNone(cache.peek("abc"))
        self.assertFalse(os.path.exists(os.path.join(self.dir, "project")))

    def test_the_spelling_mayatk_rebuilt_by_hand_still_names_the_slot(self):
        by_hand = TempArtifacts("bk_cache", policy="detached").path(
            extension=".ma", name="abc"
        )
        self.assertEqual(
            _norm(CachedArtifact("bk", extension=".ma").slot("abc")), _norm(by_hand)
        )

    def test_peek_answers_only_for_a_complete_entry(self):
        cache = CachedArtifact("pk", extension=".out")
        self.assertIsNone(cache.peek("k"))
        os.makedirs(cache.cache_dir, exist_ok=True)
        self.touch(cache.slot("k"), b"")
        self.assertIsNone(cache.peek("k"), "an empty file is no entry")
        got = cache.get("k", self.produce)
        self.assertEqual(cache.peek("k"), got.path)

    def test_touch_refreshes_an_entry_and_its_sidecars(self):
        cache = CachedArtifact("tc", extension=".ma")
        got = cache.get("k", self.produce)
        side = self.touch(got.path + ".src.json")
        for path in (got.path, side):
            self.age(path, 5)
        self.assertTrue(cache.touch("k", sidecars=(".src.json",)))
        for path in (got.path, side):
            self.assertGreater(os.path.getmtime(path), time.time() - 60, path)
        self.assertFalse(cache.touch("missing"))

    def test_a_hit_refreshes_the_entrys_age(self):
        """A hit writes nothing, so a bake in daily use went at its age."""
        cache = CachedArtifact("hit", extension=".ma")
        got = cache.get("k", self.produce)
        self.age(got.path, 5)
        self.assertTrue(cache.get("k", self.produce).hit)
        self.assertGreater(os.path.getmtime(got.path), time.time() - 60)

    def test_scratch_is_always_in_the_temp_root_never_the_cache_dir(self):
        project = os.path.join(self.dir, "project")
        cache = CachedArtifact("pc", extension=".ma", dir=project, max_age_days=None)
        seen = []
        got = cache.get("k", lambda out: (seen.append(out), self.produce(out)))
        got.scratch.cleanup()
        self.assertTrue(_norm(seen[0]).startswith(_norm(self.root) + os.sep))
        self.assertEqual(os.listdir(project), [os.path.basename(got.path)])

    def test_a_project_resident_cache_is_never_age_swept(self):
        project = os.path.join(self.dir, "project")
        got = CachedArtifact("pc", extension=".ma", dir=project, max_age_days=None).get(
            "k", self.produce
        )
        self.age(got.path, 3650)
        again = CachedArtifact(
            "pc", extension=".ma", dir=project, max_age_days=None
        ).get("k", self.produce)
        self.assertTrue(again.hit, "a referenced bake is never stale")

    def test_a_directory_slot_is_produced_whole_then_reused(self):
        cache = CachedArtifact("cd", directory=True)
        calls = []

        def produce(folder):
            calls.append(folder)
            self.touch(os.path.join(folder, "map_BaseColor.png"))

        got = cache.get("k", produce)
        self.assertEqual(_norm(got.path), _norm(cache.slot("k")))
        self.assertTrue(os.path.isfile(os.path.join(got.path, "map_BaseColor.png")))
        self.assertTrue(cache.get("k", produce).hit)
        self.assertEqual(len(calls), 1)

    def test_promote_moves_a_finished_artifact_into_its_slot(self):
        files = CachedArtifact("pf", extension=".out")
        made = self.touch(os.path.join(self.dir, "made.out"), b"one")
        slot = files.promote("k", made)
        self.assertFalse(os.path.exists(made))
        with open(slot, "rb") as fh:
            self.assertEqual(fh.read(), b"one")

        folders = CachedArtifact("pd", directory=True)
        for name in ("a.png", "b.png"):
            made = os.path.join(self.dir, f"made_{name}")
            os.makedirs(made)
            self.touch(os.path.join(made, name))
            slot = folders.promote("k", made)
        self.assertEqual(os.listdir(slot), ["b.png"], "a slot is replaced whole")


if __name__ == "__main__":
    unittest.main(verbosity=2)
