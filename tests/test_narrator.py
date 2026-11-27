# !/usr/bin/python
# coding=utf-8
"""Narrator: lines spoken into WAV files, a narration folder's lock, a sheet's narration.

Most tests speak through a stand-in engine (a short tone per line: nothing downloaded, nothing
run); the default engine's install and command line are tested with the installer and the
program stood in for. :class:`RealEngineTest` runs the real engine, which no test installs: it
skips where the engine is not installed -- and fails under CI (``CI`` set, as the glTF
Validator's proof), whose job installs it first, so a green build always ran it.
"""

import array
import contextlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

from pythontk.audio_utils import narrator as narrator_module
from pythontk.audio_utils.narrator import NarrationEngine, Narrator, SherpaKokoro
from pythontk.core_utils.app_installer import AppInstaller
from pythontk.core_utils.operation_catalog.op_context import OpContext


def _write_tone(path, seconds=0.2, rate=24000):
    """A mono 16-bit WAV of a tone, *seconds* long, at *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frames = int(seconds * rate)
    samples = array.array(
        "h", (int(8000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(frames))
    )
    with wave.open(path, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(rate)
        clip.writeframes(samples.tobytes())


class _ToneEngine:
    """A stand-in :class:`NarrationEngine`: a tenth of a second of tone per word, each call
    logged in order (``("prepare" | said, thread)``)."""

    name = "tone 1"
    model = "tone"
    license = "a test tone (no terms)"
    credits = "Narration: a test tone.\n"

    def __init__(self):
        self.events = []
        self._lock = threading.Lock()

    def prepare(self):
        with self._lock:
            self.events.append(("prepare", threading.current_thread()))

    def speak(self, said, path, voice, lang, speed, threads):
        with self._lock:
            self.events.append((said, threading.current_thread()))
        _write_tone(path, 0.1 * max(1, len(said.split())))

    def spoken(self):
        return [said for said, _ in self.events if said != "prepare"]


@contextlib.contextmanager
def _tone_sherpa():
    """The default engine with its install and program stood in for by a tone; its name and
    model kept, so a reader's lookup names the files a render made."""

    def speak(self, said, path, voice, lang, speed, threads):
        _write_tone(path)

    with (
        mock.patch.object(SherpaKokoro, "prepare", lambda self: None),
        mock.patch.object(SherpaKokoro, "speak", speak),
    ):
        yield


class _Stop(Exception):
    """Raised by a stand-in to stop a run where a test has seen what it needs."""


class NarratorTest(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp(prefix="narrator_")
        self.addCleanup(shutil.rmtree, self.out, True)
        self.engine = _ToneEngine()
        self.narrator = Narrator(pronounce={"PCB": "P C B"}, engine=self.engine)

    def _lock(self):
        with open(os.path.join(self.out, Narrator.LOCK), encoding="utf-8") as fh:
            return json.load(fh)["lines"]

    # ---- one line ----------------------------------------------------------------------------------------------

    def test_a_line_said_again_reuses_its_file_and_a_changed_one_gets_its_own(self):
        self.assertIsInstance(self.engine, NarrationEngine)
        first = self.narrator.say("Remove the retaining nut.", self.out)
        self.assertEqual(
            (first["synthetic"], first["engine"], first["model"]),
            (True, "tone 1", "tone"),
        )
        self.assertGreater(first["seconds"], 0.3)
        again = self.narrator.say("Remove  the retaining nut.", self.out)
        self.assertEqual(
            again["file"],
            first["file"],
            "the same words: the same file, not spoken again",
        )
        self.assertEqual(len(self.engine.spoken()), 1)
        other = self.narrator.say("Remove the retaining nuts.", self.out)
        self.assertNotEqual(other["file"], first["file"])
        self.assertEqual(
            [n for n in os.listdir(self.out) if not n.endswith(".wav")],
            [],
            "no partial file left",
        )

    def test_another_engine_voices_its_own_files(self):
        """An engine that names itself otherwise keys its files apart, so changing engine re-voices
        every line and nothing else does. Added: 2026-11-22"""

        class Retuned(_ToneEngine):
            name = _ToneEngine.name + " (retuned)"

        narrator = Narrator(engine=Retuned())
        record = narrator.say("Close the door.", self.out)
        self.assertEqual(
            (record["engine"], os.path.isfile(record["file"])), (Retuned.name, True)
        )
        self.assertNotEqual(
            narrator.key("Close the door."),
            Narrator(engine=_ToneEngine()).key("Close the door."),
        )

    def test_a_word_is_said_as_the_pronunciation_list_says_and_names_its_file(self):
        self.assertEqual(
            self.narrator.spoken("Check the PCB lights; PCBs too."),
            "Check the P C B lights; PCBs too.",
            "whole words only",
        )
        plain = Narrator(engine=_ToneEngine())
        self.assertNotEqual(
            self.narrator.key("Check the PCB lights."),
            plain.key("Check the PCB lights."),
        )
        with self.assertRaises(ValueError):
            self.narrator.say("   ", self.out)

    # ---- a folder of lines -------------------------------------------------------------------------------------

    def test_render_narrates_the_missing_lines_once_and_keeps_a_lock_and_credits(self):
        lines = {"A01": "Open the door.", "A02": "Close the door."}
        first = self.narrator.render(lines, self.out)
        self.assertEqual(sorted(first["rendered"]), ["A01", "A02"])
        again = self.narrator.render(lines, self.out)
        self.assertEqual(
            (again["rendered"], sorted(again["reused"])), ([], ["A01", "A02"])
        )
        lock = self._lock()
        self.assertEqual(
            {k: v["kind"] for k, v in lock.items()},
            {"A01": "synthetic", "A02": "synthetic"},
        )
        self.assertEqual(
            lock["A01"]["sha256"],
            Narrator._digest(os.path.join(self.out, lock["A01"]["file"])),
        )
        self.assertEqual(
            Path(self.out, Narrator.CREDITS).read_text(encoding="utf-8"),
            self.engine.credits,
        )
        self.assertGreater(again["seconds"]["A01"], 0.2)

    def test_render_speaks_every_line_it_is_given_but_a_blank_one(self):
        """What a line holds is its caller's to filter (a sheet's "N/A" is the sheet reader's):
        the narrator skips only a line with nothing to say. Added: 2026-11-23"""
        result = self.narrator.render(
            {"A01": "N/A", "A02": "  ", "A03": "Open the door."}, self.out
        )
        self.assertEqual(sorted(result["rendered"]), ["A01", "A03"])

    def test_the_lock_is_versioned_and_one_without_a_version_still_reads(self):
        """Added: 2026-11-23"""
        self.narrator.render({"A01": "Open the door."}, self.out)
        path = Path(self.out, Narrator.LOCK)
        body = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(body["version"], 1)
        del body["version"]
        path.write_text(json.dumps(body), encoding="utf-8")
        again = self.narrator.render({"A01": "Open the door."}, self.out)
        self.assertEqual(
            (again["reused"], again["rendered"]),
            (["A01"], []),
            "an unversioned lock is read as it is",
        )
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8"))["version"],
            1,
            "and written back with its version",
        )

    def test_an_edited_line_is_narrated_anew_and_prune_removes_only_untouched_narration(
        self,
    ):
        self.narrator.render(
            {"A01": "Open the door.", "A02": "Close the door."}, self.out
        )
        before = self._lock()
        replaced = os.path.join(self.out, before["A02"]["file"])
        _write_tone(replaced, 1.5)  # a person's take written over the narration file
        ctx = OpContext()
        after = self.narrator.render(
            {"A01": "Open the red door.", "A02": "Close the door."},
            self.out,
            prune=True,
            ctx=ctx,
        )
        self.assertEqual(after["rendered"], ["A01"])
        self.assertEqual(
            after["pruned"],
            [before["A01"]["file"]],
            "the old narration of the edited line goes",
        )
        self.assertEqual(after["replaced"], ["A02"])
        self.assertTrue(
            os.path.isfile(replaced),
            "a file a person replaced is never removed or overwritten",
        )
        self.assertIn(Narrator.REPLACED, [i.code for i in ctx.issues])
        pruned = self.narrator.render(
            {"A01": "Open the red door."}, self.out, prune=True
        )
        self.assertEqual(pruned["pruned"], [], "nor removed once no line uses it")
        self.assertTrue(os.path.isfile(replaced))

    def test_a_replaced_file_stays_a_persons_through_later_runs(self):
        """A file a person wrote over a narration file is reported on every run, never adopted as
        narration again, so an edit of its line with ``prune`` never removes it; one that is not
        a WAV file goes unmeasured. Added: 2026-11-23"""
        lines = {"A01": "Open the door.", "A02": "Close the door."}
        self.narrator.render(lines, self.out)
        theirs = os.path.join(self.out, self._lock()["A02"]["file"])
        Path(theirs).write_bytes(
            b"ID3\x04\x00\x00" + bytes(256)
        )  # an MP3, under the WAV's name
        for run in (1, 2):
            again = self.narrator.render(lines, self.out)
            self.assertEqual(
                (again["replaced"], again["reused"]),
                (["A02"], ["A01"]),
                f"run {run}: still theirs",
            )
            self.assertEqual(self._lock()["A02"]["kind"], "replaced", f"run {run}")
            self.assertEqual(sorted(again["seconds"]), ["A01"])
        edited = self.narrator.render(
            {"A01": "Open the door.", "A02": "Close the red door."},
            self.out,
            prune=True,
        )
        self.assertEqual(edited["pruned"], [])
        self.assertTrue(os.path.isfile(theirs), "a person's file is never removed")

    def test_a_recording_is_stamped_with_its_line_and_reported_once_the_line_changes(
        self,
    ):
        take = os.path.join(self.out, "studio", "A03_take.wav")
        _write_tone(take, 0.8)
        first = self.narrator.render(
            {"A03": "Check the flap."}, self.out, recordings={"A03": take}
        )
        self.assertEqual(
            (first["recordings"], first["rendered"], first["stale"]), (["A03"], [], [])
        )
        self.assertAlmostEqual(first["seconds"]["A03"], 0.8, places=2)
        ctx = OpContext()
        edited = self.narrator.render(
            {"A03": "Check both flaps."}, self.out, recordings={"A03": take}, ctx=ctx
        )
        self.assertEqual(
            edited["stale"], ["A03"], "a recording made for an older line is reported"
        )
        self.assertIn(Narrator.STALE_RECORDING, [i.code for i in ctx.issues])
        accepted = self.narrator.render(
            {"A03": "Check both flaps."},
            self.out,
            recordings={"A03": take},
            accept=["A03"],
        )
        self.assertEqual(accepted["stale"], [])
        self.assertEqual(
            self.narrator.render(
                {"A03": "Check both flaps."}, self.out, recordings={"A03": take}
            )["stale"],
            [],
            "accepted for the new line, it stays accepted",
        )
        self.assertEqual(self.engine.spoken(), [], "a recorded line is never narrated")

    def test_a_recording_keeps_its_stamp_when_the_project_moves_and_a_new_take_is_current(
        self,
    ):
        """The lock names a recording by its file name and content, never a machine path: the same
        take found elsewhere (the project moved) keeps its stamp; a take recorded again is made for
        the line it is recorded for. Added: 2026-11-23"""
        take = os.path.join(self.out, "here", "A03_take.wav")
        _write_tone(take, 0.8)
        self.narrator.render(
            {"A03": "Check the flap."}, self.out, recordings={"A03": take}
        )
        entry = self._lock()["A03"]
        self.assertEqual(entry["file"], "A03_take.wav")
        self.assertFalse(
            [v for v in entry.values() if isinstance(v, str) and os.path.isabs(v)],
            "no machine path",
        )
        moved = os.path.join(self.out, "there", "A03_take.wav")
        os.makedirs(os.path.dirname(moved))
        shutil.copy(take, moved)
        edited = self.narrator.render(
            {"A03": "Check both flaps."}, self.out, recordings={"A03": moved}
        )
        self.assertEqual(
            edited["stale"],
            ["A03"],
            "the same take, moved: still made for the older line",
        )
        _write_tone(moved, 1.1)  # recorded again, for the new line
        again = self.narrator.render(
            {"A03": "Check both flaps."}, self.out, recordings={"A03": moved}
        )
        self.assertEqual(again["stale"], [])

    def test_a_recording_the_wave_module_cannot_read_is_kept_without_its_length(self):
        """An MP3 take is a recording like any other; only its length goes unmeasured. Added: 2026-11-23"""
        take = os.path.join(self.out, "studio", "take.mp3")
        os.makedirs(os.path.dirname(take))
        Path(take).write_bytes(b"ID3\x04\x00\x00" + bytes(256))
        result = self.narrator.render(
            {"A01": "Open the door.", "A02": "Close it."},
            self.out,
            recordings={"A01": take},
        )
        self.assertEqual((result["recordings"], result["rendered"]), (["A01"], ["A02"]))
        self.assertNotIn("A01", result["seconds"])

    def test_a_dry_run_plans_the_writes_and_makes_nothing(self):
        ctx = OpContext(dry_run=True)
        result = self.narrator.render({"A01": "Open the door."}, self.out, ctx=ctx)
        self.assertEqual(result["rendered"], [])
        self.assertEqual(os.listdir(self.out), [])
        self.assertEqual(
            self.engine.events, [], "nothing prepared (installed) or spoken"
        )
        self.assertTrue(
            any(
                c.kind == "write" and str(c.target).endswith(".wav")
                for c in ctx.changes
            )
        )

    def test_the_engine_is_prepared_once_on_the_calling_thread_before_any_line(self):
        """An install (and its question) happens once, before the workers start, never in each
        worker at once. Added: 2026-11-23"""
        lines = {f"A0{i}": f"Line number {i}." for i in range(1, 5)}
        result = self.narrator.render(lines, self.out, workers=4)
        self.assertEqual(len(result["rendered"]), 4)
        self.assertEqual(self.engine.events[0], ("prepare", threading.current_thread()))
        self.assertEqual([e for e, _ in self.engine.events].count("prepare"), 1)

    def test_lines_of_the_same_words_are_spoken_once_and_each_reported(self):
        """Added: 2026-11-23"""
        result = self.narrator.render(
            {"A01": "Open the door.", "B01": "Open  the door."}, self.out, workers=2
        )
        self.assertEqual(sorted(result["rendered"]), ["A01", "B01"])
        self.assertEqual(len(self.engine.spoken()), 1, "one file, spoken once")
        lock = self._lock()
        self.assertEqual(lock["A01"]["file"], lock["B01"]["file"])
        self.assertEqual(sorted(result["seconds"]), ["A01", "B01"])

    # ---- a sheet -----------------------------------------------------------------------------------------------

    def _sheet(self, rows):
        sheet = Path(self.out) / "Sheet (3).csv"
        sheet.write_text(
            "SECTION A: INTRO\nStep,Step Contents,Asset Names,Voice Support,Audio Files\n"
            + rows,
            encoding="utf-8",
        )
        return str(sheet)

    def test_a_sheet_is_narrated_where_reading_it_finds_the_narration_and_a_recording_wins(
        self,
    ):
        from pythontk.core_utils.engines.shots.manifest.mapping import Mapping

        project = Path(self.out) / "project"
        sheet = self._sheet(
            "A01.),Door opens,door,Open the door.,\n"
            "A02.),Lid opens,lid,Lift the lid.,a02_take.wav\n"
            "A03.),Nothing said,lid,N/A,\n"
        )
        _write_tone(str(project / "recordings" / "a02_take.wav"))
        options = {"audio": "narrate"}
        with _tone_sherpa():
            result = Mapping.narrate(
                sheet, mapping="default", options=options, base=str(project)
            )
        self.assertEqual(
            (result["steps"], result["rendered"], result["recordings"]),
            (2, ["A01"], ["A02"]),
        )
        steps = Mapping.resolve(
            sheet, name="default", options=options, base=str(project)
        )
        found = {
            s.step_id: [
                Path(o.source_path).parent.name for o in s.objects if o.kind == "audio"
            ]
            for s in steps
        }
        self.assertEqual(
            found["A01"], ["narration"], "the reader finds what the render made"
        )
        self.assertEqual(found["A02"], ["recordings"])
        with self.assertRaises(ValueError):
            Mapping.narrate(
                sheet, mapping="default", options={"audio": "text"}, base=str(project)
            )

    def test_a_recording_in_a_folder_named_like_the_narration_folder_is_a_recording(
        self,
    ):
        """``narration_takes/`` is not inside ``narration/``. Added: 2026-11-23"""
        from pythontk.core_utils.engines.shots.manifest import mapping as mapping_pkg
        from pythontk.core_utils.engines.shots.manifest.mapping import Mapping

        template = json.loads(
            (Path(mapping_pkg.__file__).parent / "default.json").read_text(
                encoding="utf-8"
            )
        )
        for method in template["options"]["audio"]["choices"]["narrate"]["set"][
            "audio_resolve"
        ]:
            if method["method"] != "narrate":
                method["directory"] = "narration_takes"
        custom = Path(self.out) / "takes.json"
        custom.write_text(json.dumps(template), encoding="utf-8")
        project = Path(self.out) / "project"
        sheet = self._sheet(
            "A01.),Door opens,door,Open the door.,\nA02.),Lid opens,lid,Lift the lid.,a02_take.wav\n"
        )
        _write_tone(str(project / "narration_takes" / "a02_take.wav"))
        with _tone_sherpa():
            result = Mapping.narrate(
                sheet,
                mapping=str(custom),
                options={"audio": "narrate"},
                base=str(project),
            )
        self.assertEqual((result["rendered"], result["recordings"]), (["A01"], ["A02"]))

    def test_a_sheets_narration_asks_consent_as_its_caller_says(self):
        """``prompt`` (a plain bool an operation takes) and ``confirm`` (a panel's dialog) reach the
        engine's install; the operation exposes the bool alone. Added: 2026-11-23"""
        from pythontk.core_utils.engines.shots.manifest.mapping import Mapping

        project = Path(self.out) / "project"
        sheet = self._sheet("A01.),Door opens,door,Open the door.,\n")
        seen = []

        def stop(engine):
            seen.append(engine.prompt)
            raise _Stop()

        def asker(question):
            return False

        with mock.patch.object(
            SherpaKokoro, "prepare", autospec=True, side_effect=stop
        ):
            for kwargs in ({}, {"prompt": False}, {"confirm": asker}):
                with self.assertRaises(_Stop):
                    Mapping.narrate(
                        sheet,
                        mapping="default",
                        options={"audio": "narrate"},
                        base=str(project),
                        **kwargs,
                    )
        self.assertEqual(seen, [True, False, asker])
        catalog = json.loads(
            (Path(narrator_module.__file__).parents[1] / "operations.json").read_text(
                encoding="utf-8"
            )
        )
        args = next(o for o in catalog["operations"] if o["id"] == "narration.render")[
            "args"
        ]["properties"]
        self.assertEqual(
            (args["prompt"]["type"], args["prompt"]["default"]), ("boolean", True)
        )
        self.assertNotIn("confirm", args, "code: a panel passes it, no caller sends it")


class SherpaKokoroTest(unittest.TestCase):
    """The default engine's install and command line, the installer and the program stood in
    for: nothing is downloaded or run."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="sherpa_")
        self.addCleanup(shutil.rmtree, self.root, True)
        # Never a person's tools folder, whatever a stand-in misses: an install lands in this
        # test's own folder, and a download fails the test.
        for guard in (
            mock.patch.dict(
                os.environ, {"PYTHONTK_TOOLS_DIR": os.path.join(self.root, "tools")}
            ),
            mock.patch.object(
                AppInstaller,
                "_download",
                side_effect=AssertionError("a test downloaded"),
            ),
        ):
            guard.start()
            self.addCleanup(guard.stop)
        self.program = os.path.join(self.root, "bin", "sherpa-onnx-offline-tts.exe")
        self.voice = os.path.join(self.root, "voice", "model.int8.onnx")
        for path in (self.program, self.voice):
            os.makedirs(os.path.dirname(path))
            open(path, "wb").close()
        self.installed, self.asked, self.ensured, self.runs = {}, [], [], []
        self.out = os.path.join(self.root, "out")

    def _get_path(self, name, location="user", executable=None, add_to_path=False):
        return self.installed.get(name)

    def _consent(self, prompt, question):
        self.asked.append(question)
        return True

    def _ensure(self, name, platforms, executable=None, **kwargs):
        self.ensured.append(name)
        time.sleep(
            0.2
        )  # a download: long enough for workers started at once to meet it
        self.installed[name] = (
            self.voice if executable == "model.int8.onnx" else self.program
        )
        return self.installed[name]

    def _run(self, cmd, **kwargs):
        """``subprocess.run``'s stand-in: the engine writes the file it is told to."""
        self.runs.append((list(cmd), kwargs))
        _write_tone(
            next(
                a.split("=", 1)[1]
                for a in cmd
                if str(a).startswith("--output-filename=")
            )
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    @contextlib.contextmanager
    def _fakes(self, ensure=True):
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(AppInstaller, "get_path", side_effect=self._get_path)
            )
            stack.enter_context(
                mock.patch.object(AppInstaller, "consent", side_effect=self._consent)
            )
            if ensure:
                stack.enter_context(
                    mock.patch.object(AppInstaller, "ensure", side_effect=self._ensure)
                )
            stack.enter_context(mock.patch("subprocess.run", side_effect=self._run))
            yield

    def _speak(self, said):
        os.makedirs(self.out, exist_ok=True)
        SherpaKokoro(prompt=False).speak(
            said, os.path.join(self.out, "line.wav"), 0, "en", 1.0, 2
        )

    def test_lines_spoken_at_once_install_the_engine_and_the_voice_once(self):
        """Added: 2026-11-23"""
        with self._fakes():
            narrator = Narrator(
                engine=SherpaKokoro(prompt=True)
            )  # consent stood in for: yes
            result = narrator.render(
                {f"A0{i}": f"Line number {i}." for i in range(1, 5)},
                self.out,
                workers=4,
            )
        self.assertEqual(len(result["rendered"]), 4)
        self.assertEqual(
            (len(self.asked), len(self.ensured)),
            (2, 2),
            "asked and installed once each: engine, voice",
        )

    def test_the_engine_runs_with_no_window_or_input_and_any_output_reads(self):
        """Added: 2026-11-23"""
        with self._fakes():
            self._speak("Open the door.")
        cmd, kwargs = self.runs[0]
        self.assertEqual(cmd[0], self.program)
        self.assertEqual(
            kwargs.get("stdin"), subprocess.DEVNULL, "it can prompt no one"
        )
        self.assertEqual(
            kwargs.get("errors"), "replace", "a message in another encoding still reads"
        )
        if os.name == "nt":
            self.assertTrue(
                kwargs.get("creationflags", 0) & subprocess.CREATE_NO_WINDOW,
                "no console over a DCC",
            )

    def test_an_engine_run_past_its_time_is_a_runtime_error(self):
        """Added: 2026-11-23"""
        with (
            self._fakes(),
            mock.patch(
                "subprocess.run", side_effect=subprocess.TimeoutExpired("engine", 300)
            ),
        ):
            with self.assertRaises(RuntimeError):
                self._speak("Open the door.")

    def test_a_line_led_by_dashes_is_given_as_words_not_a_flag(self):
        """Added: 2026-11-23"""
        with self._fakes():
            self._speak("--Open the door.")
        self.assertEqual(self.runs[0][0][-2:], ["--", "--Open the door."])

    def test_a_word_processors_quotes_dashes_and_accents_reach_the_engine_plain(self):
        """REGRESSION: on Windows the engine reads its command line in the system's code page
        (measured: a curly apostrophe arrived as the byte 0x92, not UTF-8), the only way it reads
        text, so a sheet's typographic marks were spoken as noise. They are given plain, on every
        platform alike. Added: 2026-11-26"""
        with self._fakes():
            self._speak("Don’t open it – the café’s “door”…")
        self.assertEqual(self.runs[0][0][-1], "Don't open it - the cafe's \"door\"...")

    def test_a_character_the_command_line_cannot_carry_is_refused_by_name(self):
        """Where the engine's command line is read in a code page, a character with no plain
        form cannot reach it: refused, named, never spoken as noise; elsewhere it is given as
        written. Added: 2026-11-26"""
        with self._fakes(), mock.patch.object(SherpaKokoro, "CODE_PAGE_ARGV", True):
            with self.assertRaisesRegex(RuntimeError, "°"):
                self._speak("Turn it 90°.")
        self.assertEqual(self.runs, [], "the engine never ran")
        with self._fakes(), mock.patch.object(SherpaKokoro, "CODE_PAGE_ARGV", False):
            self._speak("Turn it 90°.")
        self.assertEqual(self.runs[0][0][-1], "Turn it 90°.")

    def test_a_bumped_pin_installs_anew(self):
        """The installer finds a tool by name alone, so the name carries the pinned version: an
        older pin's install is not taken for this one. Added: 2026-11-23"""
        self.installed.update(
            {"sherpa-onnx": self.program, "sherpa-onnx-0.0.1": self.program}
        )
        with self._fakes():
            SherpaKokoro(prompt=False).program()
        self.assertEqual(len(self.ensured), 1)
        self.assertIn(narrator_module.ENGINE_VERSION, self.ensured[0])

    def test_the_linux_engine_installs_only_on_x86_64(self):
        """Added: 2026-11-23"""
        with (
            self._fakes(ensure=False),
            mock.patch("platform.system", return_value="Linux"),
            mock.patch("platform.machine", return_value="aarch64"),
        ):
            with self.assertRaises(LookupError):
                SherpaKokoro(prompt=False).program()


class RealEngineTest(unittest.TestCase):
    """The real engine, as installed: never installed here."""

    @classmethod
    def setUpClass(cls):
        try:
            SherpaKokoro(
                prompt=lambda question: False
            ).prepare()  # finds it, or declines the download
        except (FileNotFoundError, LookupError) as exc:
            reason = f"the narration engine is not installed ({exc}); SherpaKokoro(prompt=False).prepare() installs it"
            if os.environ.get("CI"):
                raise AssertionError(f"CI runs the narration engine, but {reason}")
            raise unittest.SkipTest(reason)

    def setUp(self):
        self.out = tempfile.mkdtemp(prefix="narrator_")
        self.addCleanup(shutil.rmtree, self.out, True)

    def test_lines_are_spoken_at_once_into_mono_wavs_one_led_by_dashes_too(self):
        narrator = Narrator(prompt=False)
        lines = {"A01": "Select the highlighted door to open it.", "A02": "--Close it."}
        result = narrator.render(lines, self.out, workers=2)
        self.assertEqual(sorted(result["rendered"]), ["A01", "A02"])
        lock = json.loads(Path(self.out, Narrator.LOCK).read_text(encoding="utf-8"))[
            "lines"
        ]
        for line_id, low, high in (("A01", 1.0, 8.0), ("A02", 0.3, 5.0)):
            with wave.open(os.path.join(self.out, lock[line_id]["file"]), "rb") as clip:
                self.assertEqual((clip.getnchannels(), clip.getsampwidth()), (1, 2))
                samples = array.array("h", clip.readframes(clip.getnframes()))
            self.assertTrue(
                low < result["seconds"][line_id] < high,
                f"{line_id}: {result['seconds'][line_id]} s",
            )
            self.assertGreater(
                max(abs(s) for s in samples), 1000, f"{line_id} is not silent"
            )
        self.assertEqual(
            [n for n in os.listdir(self.out) if n.endswith(".part")],
            [],
            "no partial file left",
        )

    def test_a_line_with_typographic_marks_sounds_as_its_plain_spelling(self):
        """REGRESSION: on Windows the marks reached the engine as code-page bytes, and it dropped
        what they spelled (0.25 s of a 1.3 s line). Added: 2026-11-26"""
        lines = {
            "A01": "Don’t open it – the café’s door.",
            "A02": "Don't open it - the cafe's door.",
        }
        seconds = Narrator(prompt=False).render(lines, self.out, workers=2)["seconds"]
        self.assertAlmostEqual(seconds["A01"], seconds["A02"], delta=0.05, msg=seconds)

    def test_a_narration_folder_named_outside_the_code_page_is_written(self):
        """REGRESSION: on Windows a path the system's code page cannot spell reached the engine
        as its nearest spelling there (a folder 'narration Łódź' as 'narration Lódz'), a
        folder that does not exist, and it could not write there. Added: 2026-11-26"""
        out = os.path.join(self.out, "narration Łódź")
        result = Narrator(prompt=False).render({"A01": "Open the door."}, out)
        self.assertEqual(result["rendered"], ["A01"])


if __name__ == "__main__":
    unittest.main()
