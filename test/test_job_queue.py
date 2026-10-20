# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.file_utils.job_queue: a folder of JSON jobs one worker takes in order."""

import json
import os
import shutil
import tempfile
import threading
import time
import unittest

from pythontk.file_utils.job_queue import JobQueue


class TestJobQueue(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="job_queue_")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.queue = JobQueue(self.root)

    def serve(self, handler, **kwargs):
        """Run a worker on a thread; stopped and joined at cleanup."""
        kwargs.setdefault("poll", 0.02)
        kwargs.setdefault("beat", 0.05)
        thread = threading.Thread(
            target=self.queue.serve, args=(handler,), kwargs=kwargs, daemon=True
        )
        thread.start()

        def stop():
            self.queue.stop()
            thread.join(10)

        self.addCleanup(stop)
        return thread

    def test_a_job_is_queued_until_a_worker_takes_it(self):
        job = self.queue.submit({"n": 2})
        self.assertEqual(self.queue.status(job), "queued")
        with self.assertRaises(LookupError):
            self.queue.result(job)
        self.serve(lambda j: {"twice": j["n"] * 2})
        self.assertEqual(self.queue.wait(job, timeout=10), {"twice": 4})
        self.assertEqual(self.queue.status(job), "done")

    def test_jobs_run_in_the_order_they_were_submitted(self):
        ran = []
        ids = [self.queue.submit({"i": i}) for i in range(5)]
        self.serve(lambda j: ran.append(j["i"]) or {})
        for job in ids:
            self.queue.wait(job, timeout=10)
        self.assertEqual(ran, [0, 1, 2, 3, 4])

    def test_a_failed_job_reports_its_error_and_the_worker_goes_on(self):
        def handler(job):
            if job.get("boom"):
                raise ValueError("the light has no shape")
            return {"ok": True}

        bad = self.queue.submit({"boom": True})
        good = self.queue.submit({})
        self.serve(handler)
        with self.assertRaises(RuntimeError) as ctx:
            self.queue.wait(bad, timeout=10)
        self.assertIn("the light has no shape", str(ctx.exception))
        self.assertEqual(self.queue.status(bad), "failed")
        self.assertEqual(self.queue.wait(good, timeout=10), {"ok": True})

    def test_stop_ends_the_worker_and_says_so(self):
        thread = self.serve(lambda j: {})
        self.queue.wait(self.queue.submit({}), timeout=10)
        self.assertTrue(self.queue.alive())
        self.queue.stop()
        thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.queue.worker()["state"], "stopped")
        self.assertFalse(self.queue.alive())

    def test_setup_runs_first_and_the_worker_beats_through_it(self):
        started = threading.Event()
        release = threading.Event()

        def setup():
            started.set()
            release.wait(10)

        self.serve(lambda j: {"after": True}, setup=setup)
        self.assertTrue(started.wait(10))
        time.sleep(0.2)
        self.assertEqual(self.queue.worker()["state"], "starting")
        self.assertTrue(self.queue.alive(stale=1.0))
        job = self.queue.submit({})
        release.set()
        self.assertEqual(self.queue.wait(job, timeout=10), {"after": True})

    def test_waiting_with_no_worker_times_out(self):
        job = self.queue.submit({})
        with self.assertRaises(TimeoutError):
            self.queue.wait(job, timeout=0.2, poll=0.02)

    def test_waiting_on_a_worker_that_stopped_beating_fails_fast(self):
        job = self.queue.submit({})
        os.replace(  # it died running this job
            os.path.join(self.root, f"{job}.job.json"),
            os.path.join(self.root, f"{job}.running.json"),
        )
        with open(os.path.join(self.root, JobQueue.WORKER), "w") as f:
            json.dump({"pid": 1, "state": "serving", "beat": time.time() - 3600}, f)
        with self.assertRaises(RuntimeError) as ctx:
            self.queue.wait(job, timeout=10, poll=0.02, stale=1.0)
        self.assertIn("stopped", str(ctx.exception))

    def test_a_crashed_workers_heartbeat_does_not_fail_a_job_a_new_worker_takes(self):
        """A worker that crashed left its heartbeat saying serving: a job still
        queued failed at once, while the worker launched for it was starting."""
        job = self.queue.submit({"n": 3})
        with open(os.path.join(self.root, JobQueue.WORKER), "w") as f:
            json.dump({"pid": 1, "state": "serving", "beat": time.time() - 3600}, f)
        late = threading.Timer(0.3, self.serve, args=(lambda j: {"n": j["n"]},))
        late.start()
        self.addCleanup(late.cancel)
        self.assertEqual(self.queue.wait(job, timeout=10, poll=0.02, stale=5.0), {"n": 3})

    def test_a_momentarily_locked_job_file_does_not_end_the_worker(self):
        """A sync client or a scanner holding the queued file refused the move
        to running once (WinError 32), and that ended serve()."""
        from unittest import mock

        job = self.queue.submit({"n": 1})
        real, refused = os.replace, []

        def replace(src, dst):
            if str(src).endswith(".job.json") and not refused:
                refused.append(src)
                error = PermissionError(13, "held by another process", src)
                error.winerror = 32
                raise error
            return real(src, dst)

        self.queue.begin(beat=0.05)
        try:
            with mock.patch("os.replace", side_effect=replace):
                self.assertEqual(self.queue.step(lambda j: {"n": j["n"]}), job)
        finally:
            self.queue.end()
        self.assertEqual(len(refused), 1)
        self.assertEqual(self.queue.result(job), {"n": 1})

    def test_ids_sort_in_submission_order_within_one_clock_tick(self):
        """A coarse clock (Windows) gives a burst of submits one time: the
        count after it decides, and "10" sorted before "2"."""
        from unittest import mock

        with mock.patch("time.time_ns", return_value=1_760_000_000_000_000_000):
            ids = [self.queue.submit({"i": i}) for i in range(12)]
        self.assertEqual(sorted(ids), ids)

    def test_a_second_worker_on_a_live_queue_is_refused(self):
        self.serve(lambda j: {})
        self.queue.wait(self.queue.submit({}), timeout=10)
        with self.assertRaises(RuntimeError):
            JobQueue(self.root).serve(lambda j: {}, poll=0.02, beat=0.05)

    def test_a_host_with_its_own_loop_serves_step_by_step(self):
        """begin / step / stopping / end: a GUI host takes jobs from its timer."""
        first, second = self.queue.submit({"n": 1}), self.queue.submit({"n": 2})
        self.queue.begin(beat=0.05)
        try:
            self.assertEqual(self.queue.worker()["state"], "serving")
            self.assertEqual(self.queue.step(lambda j: {"n": j["n"]}), first)
            self.assertEqual(self.queue.result(first), {"n": 1})
            self.assertEqual(self.queue.status(second), "queued")
            self.assertEqual(self.queue.step(lambda j: {"n": j["n"]}), second)
            self.assertIsNone(self.queue.step(lambda j: {}))  # nothing queued
            self.assertFalse(self.queue.stopping())
            self.queue.stop()
            self.assertTrue(self.queue.stopping())
        finally:
            self.queue.end()
        self.assertEqual(self.queue.worker()["state"], "stopped")
        self.assertFalse(self.queue.stopping())  # the request is spent

    def test_a_lapsed_lease_ends_the_worker_once_nothing_is_queued(self):
        """A caller renews a lease while it wants the worker; stopping renewing
        (done with it, or gone) lets the worker end by itself."""
        self.queue.lease(1.5)
        thread = self.serve(lambda j: {"ran": True})
        job = self.queue.submit({})
        self.assertEqual(self.queue.wait(job, timeout=10, poll=0.02), {"ran": True})
        self.assertTrue(self.queue.alive())  # leased: still up, idle
        thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.queue.worker()["state"], "stopped")
        self.assertLess(self.queue.leased(), 0)

    def test_a_queued_job_outlives_the_lease(self):
        self.queue.begin(beat=0.05)
        try:
            self.queue.lease(0.05)
            job = self.queue.submit({"n": 1})
            time.sleep(0.15)
            self.assertFalse(self.queue.stopping())  # lapsed, but a job waits
            self.assertEqual(self.queue.step(lambda j: {}), job)
            self.assertTrue(self.queue.stopping())  # lapsed and idle
        finally:
            self.queue.end()

    def test_a_lapsed_lease_from_a_caller_that_is_gone_is_dropped_at_start(self):
        """Left by a caller that died before its worker came up: a worker
        started later serves until stopped, as an unleased queue does."""
        self.queue.lease(-1.0)
        self.queue.begin(beat=0.05)
        try:
            self.assertIsNone(self.queue.leased())
            self.assertFalse(self.queue.stopping())
        finally:
            self.queue.end()

    def test_a_withdrawn_lease_serves_until_stopped(self):
        self.queue.lease(0.05)
        self.queue.lease(None)
        time.sleep(0.1)
        self.assertIsNone(self.queue.leased())
        self.assertFalse(self.queue.stopping())

    def test_a_setup_that_fails_leaves_no_worker_beating(self):
        """A scene that will not open: the worker says it stopped, and why."""

        def setup():
            raise OSError("the scene will not open")

        with self.assertRaises(OSError):
            self.queue.serve(lambda j: {}, setup=setup, poll=0.02, beat=0.05)
        worker = self.queue.worker()
        self.assertEqual(worker["state"], "stopped")
        self.assertIn("will not open", worker["error"])
        time.sleep(0.2)  # its heartbeat is not still running
        self.assertEqual(self.queue.worker()["beat"], worker["beat"])

    def test_waiting_with_nobody_serving_fails_once_stale(self):
        """A job queued to a stopped (or never started) worker fails, not hangs."""
        job = self.queue.submit({})
        with self.assertRaises(RuntimeError) as ctx:
            self.queue.wait(job, poll=0.02, stale=0.2)
        self.assertIn("no worker", str(ctx.exception))

    def test_a_job_a_dead_worker_was_running_fails_at_the_next_start(self):
        job = self.queue.submit({})
        os.replace(
            os.path.join(self.root, f"{job}.job.json"),
            os.path.join(self.root, f"{job}.running.json"),
        )
        self.serve(lambda j: {})
        with self.assertRaises(RuntimeError) as ctx:
            self.queue.wait(job, timeout=10)
        self.assertIn("worker", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(exit=False)
