# !/usr/bin/python
# coding=utf-8
"""Tests for the preview page's clip recorder.

The WebXR page records the clip its transport is on by stepping it, posting one
rendered frame per request, and asking the server to encode them -- so what is
checked here is the server's half of that conversation: the routes, the refusals
(a page reaches a file on disk through them), the frame bookkeeping that makes a
gap loud rather than silent, and where the movie lands.

The browser half is covered live in ``test_preview_viewer_live.py``.
"""

import json
import os
import re
import struct
import sys
import unittest
import unittest.mock
import urllib.error
import urllib.request
import zlib
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythontk.file_utils.temp_artifacts import TempArtifacts
from pythontk.net_utils.preview.playblast import PreviewPlayblast
from pythontk.net_utils.preview.server import (
    PLAYBLAST_ACTIONS,
    PLAYBLAST_PATH,
    PreviewServer,
)
from pythontk.vid_utils._vid_utils import VidUtils


def _png(width=16, height=16, value=200):
    """A real, decodable PNG — ffmpeg is asked to read these for real."""
    raw = b"".join(
        b"\x00" + bytes([value, value, value] * width) for _ in range(height)
    )

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _fake_encode(self, capture, output, **kwargs):
    """Stand in for the ffmpeg call, keeping what it was handed assertable."""
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_bytes(b"movie")
    return str(output)


def _script():
    return (PreviewServer.SCRIPTS_DIR / "playblast.js").read_text(encoding="utf-8")


def _viewer_api():
    """Names the viewer page publishes on the object it hands each script.

    Read out of the page rather than listed here, so a rename of an API member
    fails the scripts that used it instead of drifting past them.
    """
    page = (PreviewServer.SCRIPTS_DIR.parent / "viewer.html").read_text(
        encoding="utf-8"
    )
    block = page.split("const viewer = {", 1)[1].split("\n};", 1)[0]
    return set(re.findall(r"^  (?:get )?(\w+)\s*[({:,]", block, re.M))


class PlayblastRoutesTestCase(unittest.TestCase):
    """The recorder driven the way the page drives it: over HTTP."""

    def setUp(self):
        self.temp = TempArtifacts("test_preview_playblast", policy="scoped")
        self.root = Path(self.temp.dir_path())
        self.assets = Path(self.temp.dir_path())
        self.server = PreviewServer(root=self.root, port=0, viewer=False).start()
        # The encode itself is ffmpeg's, and is covered by test_vid / the real
        # round trip below. Stubbing it keeps every route test fast and keeps
        # them meaningful on a machine with no ffmpeg.
        self._encode = unittest.mock.patch.object(
            PreviewPlayblast, "encode_sequence", autospec=True, side_effect=_fake_encode
        )
        self.encoded = self._encode.start()

    def tearDown(self):
        self._encode.stop()
        self.server.stop()
        self.temp.cleanup()

    # -- helpers --------------------------------------------------------

    def _post(self, route, body=b"", content_type="application/json", origin=None):
        headers = {"Content-Type": content_type}
        if origin:
            headers["Origin"] = origin
        request = urllib.request.Request(
            self.server.url + route, data=body, headers=headers
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = response.read()
        return json.loads(payload) if payload else {}

    def _json(self, action, payload, **kwargs):
        return self._post(
            f"{PLAYBLAST_PATH}/{action}", json.dumps(payload).encode(), **kwargs
        )

    def _begin(self, frames=3, **kwargs):
        payload = {"name": "shot_010", "fps": 24, "start_frame": 101, "frames": frames}
        payload.update(kwargs)
        return self._json("begin", payload)

    def _frame(self, token, index, data=None):
        return self._post(
            f"{PLAYBLAST_PATH}/frame?token={token}&index={index}",
            data if data is not None else _png(),
            content_type="image/png",
        )

    def _publish(self, name="cube.glb"):
        source = self.assets / name
        source.write_bytes(b"glTF-stub-0")
        self.server.publish(source)
        return source

    # -- the happy path -------------------------------------------------

    def test_a_recording_round_trips_and_reports_where_it_landed(self):
        source = self._publish()
        token = self._begin()["token"]
        for index in range(3):
            progress = self._frame(token, index)
        self.assertEqual(progress, {"received": 3, "expected": 3})

        report = self._json("finish", {"token": token})
        self.assertEqual(report["frames"], 3)
        self.assertEqual(report["start_frame"], 101)
        # Beside the published deliverable, and named for it as well as for the
        # clip: a folder of shot movies from several pushes is otherwise
        # unreadable.
        self.assertEqual(Path(report["output"]).parent, source.parent)
        self.assertIn("cube", Path(report["output"]).name)
        self.assertIn("shot_010", Path(report["output"]).name)
        self.assertFalse(report["in_serve_root"])

    def test_a_scene_push_with_no_file_on_disk_falls_back_to_the_serve_root(self):
        """A scene push's GLB is the bridge's own scratch and is released the
        moment it is published, so there is nothing to sit beside -- and the
        page's download is then the only copy that survives."""
        source = self._publish()
        source.unlink()
        token = self._begin(frames=1)["token"]
        self._frame(token, 0)
        report = self._json("finish", {"token": token})
        self.assertEqual(Path(report["output"]).parent, self.root)
        self.assertTrue(report["in_serve_root"])

    def test_a_finished_recording_is_downloadable_as_an_attachment(self):
        self._publish()
        token = self._begin(frames=1)["token"]
        self._frame(token, 0)
        report = self._json("finish", {"token": token})
        with urllib.request.urlopen(report["url"], timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertIn("attachment", response.headers["Content-Disposition"])
            self.assertEqual(response.read(), b"movie")

    def test_the_report_states_the_clips_own_frame_range(self):
        """What a reviewer reads back to the animator. The scratch sequence is
        numbered from zero (see below); the RANGE is the clip's."""
        self._publish()
        token = self._begin()["token"]
        for index in range(3):
            self._frame(token, index)
        report = self._json("finish", {"token": token})

        self.assertEqual((report["start_frame"], report["end_frame"]), (101, 103))

    def test_the_scratch_sequence_is_numbered_from_zero(self):
        """Not from the authoring frame. The frames are deleted the moment the
        movie exists, so nothing carries their numbering anywhere -- and see the
        pre-roll case below for what numbering them the timeline's way costs."""
        token = self._begin()["token"]
        for index in range(3):
            self._frame(token, index)

        self.assertEqual(self.server.playblast._get(token).frame_numbers, [0, 1, 2])

    def test_a_clip_that_starts_before_frame_zero_still_encodes(self):
        """A scene with pre-roll declares a shot at a NEGATIVE frame.

        Numbered the timeline's way that is `shot.-010.png`, which matches no
        printf pattern -- and ffmpeg will not take a negative `-start_number`
        either -- so the run failed at the encode having already paid for the
        whole capture, with nothing in the message to say why.
        """
        self._publish()
        token = self._begin(frames=2, start_frame=-10)["token"]
        for index in range(2):
            self._frame(token, index)
        report = self._json("finish", {"token": token})

        self.assertEqual((report["start_frame"], report["end_frame"]), (-10, -9))
        # The pattern handed to ffmpeg is readable, and starts where it says.
        capture = self.encoded.call_args[0][1]
        self.assertEqual(capture.start, 0)
        self.assertNotIn("-", Path(capture.pattern).name)

    def test_two_pages_recording_at_once_do_not_share_a_sequence(self):
        first = self._begin(frames=2, name="a")["token"]
        second = self._begin(frames=2, name="b")["token"]
        self.assertNotEqual(first, second)
        self._frame(first, 0)
        self.assertEqual(len(self.server.playblast._get(first).received), 1)
        self.assertEqual(len(self.server.playblast._get(second).received), 0)

    # -- refusals -------------------------------------------------------

    def test_a_gap_in_the_frames_fails_the_encode_loudly(self):
        """ffmpeg reads a printf pattern straight through and STOPS at the
        first gap, so a missing frame would otherwise encode as a short movie
        with nothing to say it was wrong."""
        token = self._begin()["token"]
        self._frame(token, 0)
        self._frame(token, 2)  # 1 never arrives
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._json("finish", {"token": token})
        self.assertEqual(caught.exception.code, 500)
        self.assertIn("incomplete", caught.exception.reason)

    def test_a_recording_past_the_frame_ceiling_is_refused_before_any_frame(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._begin(frames=self.server.playblast.max_frames + 1)
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(self.server.playblast.active(), [])

    def test_a_frame_outside_the_declared_count_is_refused(self):
        token = self._begin(frames=2)["token"]
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._frame(token, 5)
        self.assertEqual(caught.exception.code, 400)

    def test_an_empty_frame_is_refused(self):
        token = self._begin()["token"]
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._frame(token, 0, data=b"")
        self.assertEqual(caught.exception.code, 400)

    def test_an_unknown_token_is_a_conflict_not_a_404(self):
        """The route exists; the recording does not. The page's remedy is to
        start a new one, not to retry -- which a 404 would not say."""
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._json("finish", {"token": "nope"})
        self.assertEqual(caught.exception.code, 409)

    def test_an_unsupported_frame_type_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._begin(content_type="image/gif")
        self.assertEqual(caught.exception.code, 400)

    def test_a_zero_frame_recording_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._begin(frames=0)
        self.assertEqual(caught.exception.code, 400)

    def test_an_unknown_playblast_action_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._json("obliterate", {})
        self.assertEqual(caught.exception.code, 404)

    def test_a_cross_origin_recording_is_rejected(self):
        """These routes write a file; they are held to the same origin check as
        the settings write, so a page the user happens to have open cannot
        record through this server."""
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._json(
                "begin",
                {"name": "x", "fps": 24, "frames": 1},
                origin="http://evil.example",
            )
        self.assertEqual(caught.exception.code, 403)

    def test_the_page_cannot_name_the_output_file(self):
        """The stem reaches ``os.path.join``. Composed server-side from the
        deliverable and the (sanitized) clip name, and NOT accepted from the
        request -- a page that could name it could write outside the output
        directory through a route whose whole job is to write a file.
        """
        self._publish()
        token = self._begin(frames=1)["token"]
        self._frame(token, 0)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._json("finish", {"token": token, "stem": "../../../pwned"})
        self.assertEqual(caught.exception.code, 400)

    def test_a_stem_handed_in_by_code_is_still_sanitized(self):
        """Defence in depth for the other callers of a public method."""
        recorder = PreviewPlayblast()
        opened = recorder.begin(name="shot", fps=24, frames=1)
        recorder.add_frame(opened["token"], 0, _png())
        report = recorder.finish(
            opened["token"], output_dir=str(self.root), stem="../../pwned"
        )

        self.assertEqual(Path(report["output"]).parent, self.root)

    def test_a_clip_name_cannot_escape_the_scratch_directory(self):
        token = self._begin(name="../../../etc/passwd")["token"]
        recording = self.server.playblast._get(token)
        self.assertNotIn("..", recording.name)
        self._frame(token, 0)
        written = next(iter(recording.received.values()))
        self.assertEqual(
            Path(written).parent.resolve(), Path(recording.directory).resolve()
        )

    # -- lifecycle ------------------------------------------------------

    def test_cancel_drops_the_recording_and_its_frames(self):
        token = self._begin()["token"]
        self._frame(token, 0)
        directory = self.server.playblast._get(token).directory
        self.assertTrue(self._json("cancel", {"token": token})["cancelled"])
        self.assertEqual(self.server.playblast.active(), [])
        self.assertFalse(os.path.exists(directory))

    def test_cancelling_nothing_is_not_an_error(self):
        self.assertFalse(self._json("cancel", {"token": "nope"})["cancelled"])

    def test_finishing_drops_the_scratch_frames(self):
        self._publish()
        token = self._begin(frames=1)["token"]
        self._frame(token, 0)
        directory = self.server.playblast._get(token).directory
        self._json("finish", {"token": token})
        self.assertFalse(os.path.exists(directory))
        self.assertEqual(self.server.playblast.active(), [])

    def test_a_failed_encode_still_drops_the_frames(self):
        """A gigabyte of PNGs nobody will look at again is not worth keeping
        for a run that produced nothing."""
        token = self._begin(frames=1)["token"]
        self._frame(token, 0)
        directory = self.server.playblast._get(token).directory
        with unittest.mock.patch.object(
            PreviewPlayblast, "encode_sequence", side_effect=RuntimeError("no ffmpeg")
        ):
            with self.assertRaises(urllib.error.HTTPError):
                self._json("finish", {"token": token})
        self.assertFalse(os.path.exists(directory))

    def test_only_movie_targets_can_be_recorded(self):
        """The page already HAS the frames; posting them here to be handed back
        as a PNG sequence is a round trip with no purpose."""
        token = self._begin(frames=1)["token"]
        self._frame(token, 0)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._json("finish", {"token": token, "target": "png_sequence"})
        self.assertEqual(caught.exception.code, 400)


class PlayblastRegistrationTestCase(unittest.TestCase):
    """How the button gets onto the page."""

    def test_the_script_is_registered_and_packaged(self):
        self.assertIn("playblast", PreviewServer.SCRIPTS)
        source = PreviewServer.SCRIPTS_DIR / PreviewServer.SCRIPTS["playblast"]
        self.assertTrue(source.is_file())

    def test_a_deliverable_that_ships_clips_turns_the_recorder_on(self):
        """No checkbox: the button would otherwise be missing on exactly the
        push a reviewer just watched and wants to send on."""
        self.assertEqual(PreviewServer.AUTO_SCRIPTS.get("playblast"), "animation_web")

    def test_the_script_only_talks_to_routes_the_server_answers(self):
        """Every action the script posts is one the handler's allow-list names —
        a typo here is a 404 the user meets in a headset."""
        source = _script()
        posted = set(re.findall(r"post\(\s*'(\w+)'", source))
        self.assertTrue(posted, "the script posts nothing")
        self.assertLessEqual(posted, set(PLAYBLAST_ACTIONS))
        self.assertIn("playblast/frame?", source)

    def test_the_script_uses_only_the_published_viewer_api(self):
        """A script is an ES module handed the page's API object; reaching past
        it is what the seam exists to prevent."""
        source = _script()
        published = _viewer_api()
        used = set(re.findall(r"viewer[.](\w+)", source))
        self.assertTrue(used)
        missing = sorted(used - published)
        self.assertEqual(missing, [], f"not on the viewer API: {missing}")


@unittest.skipUnless(VidUtils.resolve_ffmpeg(required=False), "needs ffmpeg on PATH")
class PlayblastEncodeTestCase(unittest.TestCase):
    """One real round trip: posted PNGs out the other side as a playable movie."""

    def setUp(self):
        self.temp = TempArtifacts("test_preview_playblast_encode", policy="scoped")
        self.out = Path(self.temp.dir_path())
        self.recorder = PreviewPlayblast()

    def tearDown(self):
        self.temp.cleanup()

    def test_posted_frames_encode_to_a_movie_at_the_clips_rate(self):
        opened = self.recorder.begin(name="shot 010", fps=24, start_frame=101, frames=6)
        for index in range(6):
            self.recorder.add_frame(opened["token"], index, _png())
        report = self.recorder.finish(opened["token"], output_dir=str(self.out))

        movie = Path(report["output"])
        self.assertTrue(movie.is_file() and movie.stat().st_size > 0)
        self.assertEqual(movie.suffix, ".mp4")
        self.assertEqual(report["fps"], 24)
        self.assertAlmostEqual(report["duration"], 0.25)
        # A name off a page reaches a filesystem: the space is gone, the shot is
        # still identifiable.
        self.assertNotIn(" ", movie.name)
        self.assertIn("010", movie.name)
        self.assertAlmostEqual(VidUtils.get_video_frame_rate(str(movie)), 24, places=1)


if __name__ == "__main__":
    unittest.main()
