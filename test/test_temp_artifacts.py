#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk TempArtifacts.

Covers:
- path() allocation (unique tags, fixed names, prefix scoping)
- register() adoption of externally-created artifacts
- lifetime policies: scoped / session / detached
- context-manager semantics (delete on success, keep on failure)
- on_cleanup callback
- sweep_stale() prefix-scoped GC (age-gated, conservative)

Run with:
    python -m pytest test_temp_artifacts.py -v
    python test_temp_artifacts.py
"""
import os
import time
import shutil
import tempfile
import unittest

from pythontk.file_utils.temp_artifacts import TempArtifacts


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


class TestRootExport(unittest.TestCase):
    def test_registered_on_package_root(self):
        import pythontk as ptk

        self.assertTrue(hasattr(ptk, "TempArtifacts"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
