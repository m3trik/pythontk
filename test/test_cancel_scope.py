# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.CancelScope — cooperative cancellation primitive."""
import os
import sys
import time
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythontk import CancelScope, OperationCancelled  # noqa: E402


class TestCancelScopeBasics(unittest.TestCase):
    def test_starts_live(self):
        scope = CancelScope("test")
        self.assertFalse(scope.cancelled)
        self.assertIsNone(scope.reason)
        self.assertFalse(scope.has_ticked)

    def test_cancel_sets_flag_and_reason(self):
        scope = CancelScope("test")
        scope.cancel("dialog")
        self.assertTrue(scope.cancelled)
        self.assertEqual(scope.reason, "dialog")

    def test_cancel_is_idempotent_keeping_first_reason(self):
        scope = CancelScope("test")
        scope.cancel("first")
        scope.cancel("second")
        self.assertEqual(scope.reason, "first")

    def test_cancel_from_another_thread(self):
        scope = CancelScope("test")
        t = threading.Thread(target=lambda: scope.cancel("worker"))
        t.start()
        t.join(timeout=2)
        self.assertTrue(scope.cancelled)

    def test_reset_clears_state(self):
        scope = CancelScope("test")
        scope.cancel("x")
        scope.poll()
        scope.reset()
        self.assertFalse(scope.cancelled)
        self.assertIsNone(scope.reason)
        self.assertEqual(scope.tick_count, 0)


class TestCheckpoints(unittest.TestCase):
    def test_tick_returns_true_while_live(self):
        scope = CancelScope("test")
        self.assertTrue(scope.tick())

    def test_tick_returns_false_after_cancel(self):
        scope = CancelScope("test")
        scope.cancel()
        self.assertFalse(scope.tick())

    def test_checkpoint_raises_after_cancel(self):
        scope = CancelScope("test")
        scope.cancel("esc")
        with self.assertRaises(OperationCancelled) as ctx:
            scope.checkpoint()
        self.assertIs(ctx.exception.scope, scope)
        self.assertEqual(ctx.exception.reason, "esc")

    def test_operation_cancelled_is_base_exception(self):
        """Must survive a broad ``except Exception`` in tool code."""
        self.assertTrue(issubclass(OperationCancelled, BaseException))
        self.assertFalse(issubclass(OperationCancelled, Exception))

        scope = CancelScope("test")
        scope.cancel()
        swallowed = False
        try:
            try:
                scope.checkpoint()
            except Exception:  # noqa: BLE001 - the point of the test
                swallowed = True
        except OperationCancelled:
            pass
        self.assertFalse(swallowed)

    def test_ticks_are_counted(self):
        scope = CancelScope("test")
        for _ in range(3):
            scope.tick()
        self.assertEqual(scope.tick_count, 3)
        self.assertTrue(scope.has_ticked)
        self.assertIsNotNone(scope.elapsed_since_tick)

    def test_iterate_stops_at_cancel(self):
        scope = CancelScope("test")
        seen = []
        with self.assertRaises(OperationCancelled):
            for item in scope.iterate(range(10)):
                seen.append(item)
                if item == 2:
                    scope.cancel("mid-loop")
        self.assertEqual(seen, [0, 1, 2])


class TestPullSources(unittest.TestCase):
    def test_source_triggers_cancel_on_poll(self):
        flag = {"v": False}
        scope = CancelScope("test", poll_min_interval=0)
        scope.add_source(lambda: flag["v"])

        self.assertTrue(scope.tick())
        flag["v"] = True
        self.assertFalse(scope.tick())
        self.assertEqual(scope.reason, "source")

    def test_flag_read_does_not_poll_sources(self):
        """``cancelled`` must stay side-effect free and thread-safe."""
        calls = []

        def source():
            calls.append(1)
            return True

        scope = CancelScope("test", poll_min_interval=0)
        scope.add_source(source)
        self.assertFalse(scope.cancelled)
        self.assertEqual(calls, [])

    def test_failing_source_is_ignored(self):
        def boom():
            raise RuntimeError("probe failed")

        scope = CancelScope("test", poll_min_interval=0)
        scope.add_source(boom)
        self.assertTrue(scope.tick())
        self.assertFalse(scope.cancelled)

    def test_sources_not_polled_off_owner_thread(self):
        """A DCC probe called from a worker thread would crash the host."""
        calls = []
        scope = CancelScope("test", poll_min_interval=0)
        scope.add_source(lambda: calls.append(1) or False)

        def worker():
            scope.poll()

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=2)
        self.assertEqual(calls, [])

        scope.poll()  # owner thread does poll
        self.assertEqual(len(calls), 1)

    def test_poll_interval_throttles_sources(self):
        calls = []
        scope = CancelScope("test", poll_min_interval=5.0)
        scope.add_source(lambda: calls.append(1) or False)
        for _ in range(10):
            scope.tick()
        self.assertEqual(len(calls), 1)

    def test_remove_source(self):
        src = lambda: True  # noqa: E731
        scope = CancelScope("test", poll_min_interval=0)
        scope.add_source(src).remove_source(src)
        self.assertTrue(scope.tick())


class TestListeners(unittest.TestCase):
    def test_listener_fires_once_on_cancel(self):
        seen = []
        scope = CancelScope("test")
        scope.add_listener(lambda s: seen.append(s))
        scope.cancel()
        scope.cancel()
        self.assertEqual(len(seen), 1)

    def test_failing_listener_does_not_break_cancel(self):
        scope = CancelScope("test")
        scope.add_listener(lambda s: 1 / 0)
        scope.cancel()
        self.assertTrue(scope.cancelled)


class TestAmbientActivation(unittest.TestCase):
    def tearDown(self):
        self.assertIsNone(CancelScope.current(), "ambient scope leaked")

    def test_current_is_none_by_default(self):
        self.assertIsNone(CancelScope.current())

    def test_activate_sets_and_restores(self):
        scope = CancelScope("outer")
        with scope.activate():
            self.assertIs(CancelScope.current(), scope)
        self.assertIsNone(CancelScope.current())

    def test_context_manager_protocol(self):
        scope = CancelScope("outer")
        with scope as active:
            self.assertIs(active, scope)
            self.assertIs(CancelScope.current(), scope)
        self.assertIsNone(CancelScope.current())

    def test_ambient_check_is_noop_without_scope(self):
        CancelScope.check()  # must not raise
        self.assertTrue(CancelScope.proceed())
        self.assertFalse(CancelScope.is_cancelled())

    def test_ambient_check_raises_when_cancelled(self):
        scope = CancelScope("outer")
        with scope.activate():
            CancelScope.check()
            scope.cancel("esc")
            self.assertTrue(CancelScope.is_cancelled())
            self.assertFalse(CancelScope.proceed())
            with self.assertRaises(OperationCancelled):
                CancelScope.check()

    def test_activation_restores_on_exception(self):
        scope = CancelScope("outer")
        with self.assertRaises(ValueError):
            with scope.activate():
                raise ValueError("boom")
        self.assertIsNone(CancelScope.current())

    def test_nested_outer_cancel_propagates_to_inner(self):
        outer = CancelScope("outer")
        inner = CancelScope("inner")
        with outer.activate():
            with inner.activate():
                self.assertTrue(inner.tick())
                outer.cancel("outer-esc")
                self.assertTrue(inner.cancelled)
                self.assertFalse(inner.tick())
                self.assertEqual(inner.reason, "outer-esc")

    def test_inner_cancel_does_not_leak_to_outer(self):
        outer = CancelScope("outer")
        inner = CancelScope("inner")
        with outer.activate():
            with inner.activate():
                inner.cancel("inner-only")
            self.assertFalse(outer.cancelled)
            self.assertTrue(outer.tick())

    def test_reentrant_with_does_not_leak_the_ambient_scope(self):
        """``with scope:`` nested on the SAME scope must still unwind cleanly.

        A single-slot activation record let the inner ``__enter__`` overwrite
        the outer one, so the outer ``__exit__`` reset nothing and the scope
        stayed ambient forever — and once cancelled, cancelled every later
        operation on that thread.
        """
        scope = CancelScope("reentrant")
        with scope:
            with scope:
                self.assertIs(CancelScope.current(), scope)
            self.assertIs(CancelScope.current(), scope)
        self.assertIsNone(CancelScope.current())

    def test_reentrant_activate_preserves_the_parent_link(self):
        outer = CancelScope("outer")
        inner = CancelScope("inner")
        with outer.activate():
            with inner.activate():
                with inner.activate():  # re-entrant
                    pass
                # The inner block must not have stripped inner's parent link.
                outer.cancel("outer-esc")
                self.assertTrue(inner.cancelled)

    def test_reactivating_an_outer_scope_does_not_form_a_cycle(self):
        """A → B → A would make ``cancelled`` recurse forever."""
        a = CancelScope("a")
        b = CancelScope("b")
        with a.activate():
            with b.activate():
                with a.activate():
                    self.assertFalse(a.cancelled)  # must not RecursionError
                    self.assertFalse(b.cancelled)
                    a.cancel("x")
                    self.assertTrue(a.cancelled)
                    self.assertTrue(b.cancelled)

    def test_ambient_is_not_inherited_by_new_threads(self):
        """Pull sources must never be polled from a worker thread."""
        scope = CancelScope("outer")
        seen = []
        with scope.activate():
            t = threading.Thread(target=lambda: seen.append(CancelScope.current()))
            t.start()
            t.join(timeout=2)
        self.assertEqual(seen, [None])


class TestMetrics(unittest.TestCase):
    def test_elapsed_since_request(self):
        """Asserted as invariants, not durations: ``time.monotonic`` has ~15.6ms
        granularity on Windows, so any short-sleep threshold is a flake."""
        scope = CancelScope("test")
        self.assertIsNone(scope.elapsed_since_request)

        scope.cancel()
        elapsed = scope.elapsed_since_request
        self.assertIsNotNone(elapsed)
        self.assertGreaterEqual(elapsed, 0)
        # The request came after the scope was created, so it cannot have been
        # pending for longer than the scope has existed.
        self.assertLessEqual(elapsed, scope.elapsed)

    def test_elapsed_since_request_grows(self):
        scope = CancelScope("test")
        scope.cancel()
        first = scope.elapsed_since_request
        time.sleep(0.05)
        self.assertGreaterEqual(scope.elapsed_since_request, first)

    def test_has_ticked_reports_uncancellable_operation(self):
        scope = CancelScope("monolith")
        self.assertFalse(scope.has_ticked)
        self.assertIsNone(scope.elapsed_since_tick)


if __name__ == "__main__":
    unittest.main(verbosity=2)
