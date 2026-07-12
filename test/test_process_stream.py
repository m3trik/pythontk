# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.process_stream.

Covers OutputStream, ProcessReader, and LogTailer against in-process
fixtures (BytesIO pipes, temp log files). The app-specific connection
shells that compose these primitives (e.g. the Substance Painter
``SubstanceConnection`` in mayatk/blendertk) are tested with their
consumers.
"""
import io
import os
import re
import sys
import time
import tempfile
import threading
import unittest

# Ensure package is importable when running standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythontk import OutputStream, ProcessReader, LogTailer


class TestOutputStream(unittest.TestCase):
    def test_subscribe_receives_pushed_lines(self):
        stream = OutputStream()
        received = []
        stream.subscribe(lambda src, line: received.append((src, line)))

        stream.push("hello", source="stdout")
        stream.push("world", source="stderr")

        self.assertEqual(received, [("stdout", "hello"), ("stderr", "world")])

    def test_unsubscribe_stops_callbacks(self):
        stream = OutputStream()
        received = []
        unsub = stream.subscribe(lambda src, line: received.append(line))

        stream.push("first")
        unsub()
        stream.push("second")

        self.assertEqual(received, ["first"])

    def test_wait_for_substring(self):
        stream = OutputStream()

        def producer():
            time.sleep(0.05)
            stream.push("starting up", source="stdout")
            time.sleep(0.05)
            stream.push("project loaded", source="stdout")

        threading.Thread(target=producer, daemon=True).start()

        result = stream.wait_for("project loaded", timeout=2.0)
        self.assertIsNotNone(result)
        self.assertEqual(result[1], "project loaded")

    def test_wait_for_regex(self):
        stream = OutputStream()
        threading.Thread(
            target=lambda: (time.sleep(0.05), stream.push("ERROR: 42 fails")),
            daemon=True,
        ).start()

        result = stream.wait_for(re.compile(r"ERROR:\s*\d+"), timeout=2.0)
        self.assertIsNotNone(result)
        self.assertIn("ERROR:", result[1])

    def test_wait_for_timeout_returns_none(self):
        stream = OutputStream()
        result = stream.wait_for("never", timeout=0.2)
        self.assertIsNone(result)

    def test_wait_for_source_filter(self):
        stream = OutputStream()

        def producer():
            time.sleep(0.05)
            stream.push("noise", source="stdout")
            time.sleep(0.05)
            stream.push("target", source="log")

        threading.Thread(target=producer, daemon=True).start()

        result = stream.wait_for("target", timeout=2.0, source="log")
        self.assertEqual(result, ("log", "target"))

    def test_iter_yields_lines_until_close(self):
        stream = OutputStream()
        collected = []

        def consumer():
            for record in stream:
                collected.append(record)

        t = threading.Thread(target=consumer, daemon=True)
        t.start()

        time.sleep(0.05)
        stream.push("a", source="stdout")
        stream.push("b", source="stderr")
        time.sleep(0.2)
        stream.close()
        t.join(timeout=2.0)

        self.assertEqual(collected, [("stdout", "a"), ("stderr", "b")])

    def test_close_unblocks_wait_for(self):
        stream = OutputStream()
        result_holder = []

        def waiter():
            result_holder.append(stream.wait_for("nope", timeout=None))

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        time.sleep(0.1)
        stream.close()
        t.join(timeout=2.0)

        self.assertEqual(result_holder, [None])

    def test_wait_for_matches_pre_subscription_history(self):
        stream = OutputStream()
        stream.push("project ready", source="stdout")
        stream.push("idle", source="stdout")

        # Default include_history=True should find the buffered line.
        result = stream.wait_for("project ready", timeout=0.5)
        self.assertIsNotNone(result)
        self.assertEqual(result[1], "project ready")

    def test_wait_for_can_ignore_history(self):
        stream = OutputStream()
        stream.push("project ready", source="stdout")

        # Opt out of replay: no future "project ready" should arrive.
        result = stream.wait_for("project ready", timeout=0.3, include_history=False)
        self.assertIsNone(result)

    def test_iter_replays_buffered_lines(self):
        stream = OutputStream()
        stream.push("pre 1", source="stdout")
        stream.push("pre 2", source="stdout")

        collected = []

        def consumer():
            for record in stream:
                collected.append(record)

        t = threading.Thread(target=consumer, daemon=True)
        t.start()

        time.sleep(0.1)
        stream.push("live", source="stdout")
        time.sleep(0.2)
        stream.close()
        t.join(timeout=2.0)

        self.assertEqual(
            collected,
            [("stdout", "pre 1"), ("stdout", "pre 2"), ("stdout", "live")],
        )

    def test_history_bounded(self):
        stream = OutputStream(history=3)
        for i in range(10):
            stream.push(str(i))
        snapshot = stream.history()
        self.assertEqual(snapshot, [("", "7"), ("", "8"), ("", "9")])

    def test_clear_history(self):
        stream = OutputStream()
        stream.push("gone")
        stream.clear_history()
        self.assertEqual(stream.history(), [])
        result = stream.wait_for("gone", timeout=0.2)
        self.assertIsNone(result)

    def test_push_after_close_is_dropped(self):
        # Closing the stream must stop history accumulation -- otherwise
        # background readers writing after teardown would leak memory.
        stream = OutputStream()
        received = []
        stream.subscribe(lambda src, line: received.append(line))
        stream.close()
        stream.push("post-close")
        self.assertEqual(received, [])
        self.assertEqual(stream.history(), [])


class TestProcessReader(unittest.TestCase):
    def test_reads_pipe_into_stream(self):
        # Mimic a subprocess pipe with BytesIO-like buffer
        pipe = io.BytesIO(b"line one\nline two\nline three\n")
        stream = OutputStream()
        received = []
        stream.subscribe(lambda src, line: received.append((src, line)))

        reader = ProcessReader(pipe, stream, source="stdout")
        reader.start()
        reader.join(timeout=2.0)

        self.assertEqual(
            received,
            [("stdout", "line one"), ("stdout", "line two"), ("stdout", "line three")],
        )


class TestLogTailer(unittest.TestCase):
    def setUp(self):
        fd, self.log_path = tempfile.mkstemp(prefix="log_tail_test_", suffix=".txt")
        os.close(fd)

    def tearDown(self):
        try:
            os.remove(self.log_path)
        except OSError:
            pass

    def test_detects_appended_lines(self):
        stream = OutputStream()
        received = []
        stream.subscribe(lambda src, line: received.append(line))

        tailer = LogTailer(self.log_path, stream, poll_interval=0.05)
        tailer.start()
        time.sleep(0.1)

        with open(self.log_path, "ab") as f:
            f.write(b"appended line 1\nappended line 2\n")

        # Allow the tailer poll cycle to pick up the change
        time.sleep(0.3)
        tailer.stop()
        tailer.join(timeout=2.0)

        self.assertIn("appended line 1", received)
        self.assertIn("appended line 2", received)

    def test_partial_lines_are_buffered(self):
        stream = OutputStream()
        received = []
        stream.subscribe(lambda src, line: received.append(line))

        tailer = LogTailer(self.log_path, stream, poll_interval=0.05)
        tailer.start()
        time.sleep(0.1)

        with open(self.log_path, "ab") as f:
            f.write(b"partial ")
        time.sleep(0.15)
        # No complete line yet
        self.assertEqual(received, [])

        with open(self.log_path, "ab") as f:
            f.write(b"complete\n")
        time.sleep(0.2)
        tailer.stop()
        tailer.join(timeout=2.0)

        self.assertIn("partial complete", received)

    def test_tail_from_start_reads_existing_content(self):
        with open(self.log_path, "wb") as f:
            f.write(b"existing 1\nexisting 2\n")

        stream = OutputStream()
        received = []
        stream.subscribe(lambda src, line: received.append(line))

        tailer = LogTailer(
            self.log_path, stream, poll_interval=0.05, tail_from_start=True
        )
        tailer.start()
        time.sleep(0.2)
        tailer.stop()
        tailer.join(timeout=2.0)

        self.assertIn("existing 1", received)
        self.assertIn("existing 2", received)

    def test_handles_truncate_without_rotate(self):
        # If the writer (or the filesystem) shrinks the log in place -- same
        # inode, smaller size -- the tailer must reset position to 0 and
        # re-read from the start.
        with open(self.log_path, "wb") as f:
            f.write(b"line A\nline B\nline C\n")

        stream = OutputStream()
        received = []
        stream.subscribe(lambda src, line: received.append(line))

        tailer = LogTailer(self.log_path, stream, poll_interval=0.05)
        tailer.start()
        time.sleep(0.15)

        # Open + truncate in place (preserves inode on most filesystems).
        with open(self.log_path, "r+b") as f:
            f.truncate(0)
            f.write(b"truncated content\n")

        time.sleep(0.3)
        tailer.stop()
        tailer.join(timeout=2.0)

        self.assertIn("truncated content", received)

    def test_handles_rotation(self):
        # Pre-populate file (matches what we'll see at tailer startup)
        with open(self.log_path, "wb") as f:
            f.write(b"old content\n")

        stream = OutputStream()
        received = []
        stream.subscribe(lambda src, line: received.append(line))

        tailer = LogTailer(self.log_path, stream, poll_interval=0.05)
        tailer.start()
        time.sleep(0.15)

        # Delete + recreate (apps that rotate by rename produce a new
        # NTFS file ID; this is the realistic case).
        os.remove(self.log_path)
        time.sleep(0.1)
        with open(self.log_path, "wb") as f:
            f.write(b"after rotation\n")

        time.sleep(0.3)
        tailer.stop()
        tailer.join(timeout=2.0)

        self.assertIn("after rotation", received)


if __name__ == "__main__":
    unittest.main()
