# !/usr/bin/python
# coding=utf-8
"""Narration: lines of text spoken into audio files by an open-source engine on this machine.

The engine (sherpa-onnx, Apache-2.0) and its voice (Kokoro, Apache-2.0 weights; its training
audio is CC BY, so credit it where the narration ships) are installed on first use through
:class:`AppInstaller`, pinned and checked, after the caller's consent. No service is called:
the text never leaves the machine.

A narration folder holds one file per line, named by what decides its sound, and a lock
file (:data:`Narrator.LOCK`) only :meth:`Narrator.render` writes: what each line's file is,
its digest, and for a recording the line it was made for -- so a recording is told apart
from narration, a recording made for an older line is reported, and a file a person
replaced is never overwritten or removed.

Weighed: the ``sherpa-onnx`` wheel (the same engine in-process, its model loaded once for
every line) is a compiled dependency each host's interpreter would need installed, where the
engine's own program is one pinned download every host runs alike (pythontk prefers no
dependency); the operating systems' speech (SAPI, ``say``, espeak-ng) is a different voice
on each, so one machine's narration is not what another's would make; Piper's voices are
VITS models this same engine runs (``--vits-model``), so they stay in reach without a second
program; a cloud service sends the text off the machine and needs an account. The program
speaks one text into one file per run, so each line is a run of its own, and each run loads
the model first: measured on one machine, 11-35 s a line for 1-4 s of speech as its load
varied, about half of it spent before the engine starts speaking. Lines therefore run several
at once.
"""

import hashlib
import json
import os
import re
import subprocess
import threading
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
    runtime_checkable,
)

#: The engine's release, pinned (its flags are read against this version).
ENGINE_VERSION = "2.0.1"
_ENGINE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/v{0}/sherpa-onnx-v{0}-{1}.tar.bz2"
#: :class:`AppInstaller` platform table of the engine (x64 Windows and Linux, universal macOS).
ENGINE_PLATFORMS = {
    "windows": {
        "url": _ENGINE_URL.format(ENGINE_VERSION, "win-x64-shared-MT-Release"),
        "type": "tar.bz2",
    },
    "linux": {
        "url": _ENGINE_URL.format(ENGINE_VERSION, "linux-x64-shared"),
        "type": "tar.bz2",
        "arch": "x86_64",  # no arm64 Linux build is pinned
    },
    "darwin": {
        "url": _ENGINE_URL.format(ENGINE_VERSION, "osx-universal2-shared"),
        "type": "tar.bz2",
    },
}
#: SHA-256 of each release asset above, by platform; bumping :data:`ENGINE_VERSION` updates all three.
ENGINE_SHA256 = {
    "windows": "dc703575a68594435f53dec93df775ff4913af851a196c0537163f21d8d2364e",
    "linux": "2e823376987c3b6960d50726b6560155e2ad92da0b8b3886325773225ec6bdec",
    "darwin": "bf6718f708e15390f1da133dd888661a68df270791b178aaf765d4e3f0baa9fe",
}

#: The default voice model: Kokoro v1.1, int8, many languages (English included). Its name
#: carries its version, so a new model installs anew.
VOICE_MODEL = "kokoro-int8-multi-lang-v1_1"
_VOICE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{}.tar.bz2".format(
    VOICE_MODEL
)
VOICE_SHA256 = "259c9c8bc906d6078e4e0570421885dabb68b539c2f970fcbb890a58eeaf3e9c"
#: What the narration's files carry as their source and terms.
VOICE_LICENSE = (
    "Kokoro (Apache-2.0 weights; training audio CC BY) via sherpa-onnx (Apache-2.0)"
)


@runtime_checkable
class NarrationEngine(Protocol):
    """What speaks a line into a WAV file for a :class:`Narrator`: the default is
    :class:`SherpaKokoro`; a project passes its own (another open engine, a studio's service).

    Its ``name`` and ``model`` are part of each file's name (a new engine re-voices every line,
    nothing else does), and are recorded with ``license`` on each file; ``credits`` is what
    :meth:`Narrator.render` writes beside narration it voiced, to ship with it.

    An engine may also have ``prepare()``: called once by :meth:`Narrator.render`, on the
    calling thread, before it speaks lines several at once -- where an engine installs what it
    needs (and asks first), so its workers never race to.
    """

    name: str
    model: str
    license: str
    credits: str

    def speak(
        self, said: str, path: str, voice: Any, lang: str, speed: float, threads: int
    ) -> None:
        """Speak *said* into the WAV file *path* (raise on failure, leaving no file there).

        Parameters:
            said: The line, its words as they are to be read.
            path: The WAV file to write.
            voice: Which of the engine's voices.
            lang: The line's language (``en``).
            speed: How fast (1: as trained).
            threads: The CPU threads it may use.
        """


class SherpaKokoro:
    """The default :class:`NarrationEngine`: sherpa-onnx (Apache-2.0) speaking with the Kokoro voice
    model (Apache-2.0 weights), both installed on first use into pythontk's tools folder (pinned,
    checksummed, asked first).

    Parameters:
        prompt: Consent policy for the install -- ``True`` asks on the console, ``False`` needs none,
            a callable ``(question) -> bool`` is asked instead (a panel's dialog).
    """

    #: The engine's name in :class:`AppInstaller`'s catalog, which finds a tool by name alone: the
    #: pinned version is part of it, so a bumped pin installs anew instead of reusing an older one.
    TOOL_NAME = f"sherpa-onnx-{ENGINE_VERSION}"
    #: Seconds one line may take.
    TIMEOUT = 300
    name = f"sherpa-onnx {ENGINE_VERSION}"
    model = VOICE_MODEL
    license = VOICE_LICENSE
    credits = (
        f"Narration voiced by {VOICE_MODEL} ({VOICE_LICENSE}).\n"
        "Kokoro's training audio is licensed CC BY: credit it where this narration ships.\n"
    )
    #: Held through each install's look, question and download: one install at a time.
    _INSTALLING = threading.Lock()
    #: Whether the engine reads its command line in the system's code page rather than UTF-8, as
    #: on Windows (measured, 2.0.1: a curly apostrophe arrived as the byte 0x92, a folder named
    #: in another script as '?'). The command line is the only way it reads its text, so there
    #: a character with no plain form is refused (:meth:`_text_arg`), and a path is given by its
    #: short form (:meth:`_path_arg`).
    CODE_PAGE_ARGV = os.name == "nt"
    #: A word processor's quotes and dashes, as the plain marks the engine reads alike everywhere.
    _PLAIN_MARKS = str.maketrans(
        {
            **dict.fromkeys("‘’‚‛ʼ", "'"),
            **dict.fromkeys("“”„", '"'),
            **dict.fromkeys("‐‑‒–—―−", "-"),
        }
    )

    def __init__(self, prompt: Union[bool, Callable[[str], bool]] = True):
        self.prompt = prompt

    @classmethod
    def _text_arg(cls, said: str) -> str:
        """*said* as the engine is given it: a word processor's quotes and dashes as plain ones,
        compatibility forms as theirs (an ellipsis as three stops, a no-break space as a space),
        and a letter's accents dropped (``café``: ``cafe``) -- on every platform alike, so a line
        sounds the same on each.

        Raises:
            RuntimeError: Where the command line is read in a code page (:attr:`CODE_PAGE_ARGV`),
                a character left outside ASCII, which cannot reach the engine there.
        """
        import unicodedata

        plain = "".join(
            c
            for c in unicodedata.normalize("NFKD", said.translate(cls._PLAIN_MARKS))
            if not unicodedata.combining(c)
        )
        left = sorted({c for c in plain if not c.isascii()})
        if left and cls.CODE_PAGE_ARGV:
            raise RuntimeError(
                f"the narration engine reads its text in this system's code page, which cannot "
                f"carry {' '.join(left)} in '{said[:60]}': write {'it' if len(left) == 1 else 'them'} in words"
            )
        return plain

    @classmethod
    def _path_arg(cls, path: str) -> str:
        """*path* as the engine's command line carries it: where that is read in a code page
        (:attr:`CODE_PAGE_ARGV`), a path outside ASCII by its short (8.3) form -- its folder's,
        for a file still to be written. Unchanged where the volume keeps no short names: the
        engine then names the path it could not open."""
        if not cls.CODE_PAGE_ARGV or path.isascii():
            return path
        import ctypes

        folder, name = (path, "") if os.path.exists(path) else os.path.split(path)
        short = ctypes.create_unicode_buffer(32768)
        if not ctypes.windll.kernel32.GetShortPathNameW(folder, short, len(short)):
            return path
        return os.path.join(short.value, name) if name else short.value

    @classmethod
    def _ensure(cls, name, platforms, executable, version, sha256, prompt, what) -> str:
        from pythontk.core_utils.app_installer import AppInstaller

        with cls._INSTALLING:
            found = AppInstaller.get_path(name, executable=executable)
            if found:
                return found
            answer = AppInstaller.consent(
                prompt,
                f"{what} is not installed. Download it into the pythontk tools folder now?",
            )
            if not answer:
                raise FileNotFoundError(
                    f"{what} is not installed"
                    + (
                        " and the download was declined"
                        if answer is False
                        else ", and no console is available to confirm the download (pass prompt=False)"
                    )
                )
            return AppInstaller.ensure(
                name,
                platforms=platforms,
                executable=executable,
                version=version,
                sha256=sha256,
                add_to_path=False,
            )

    def program(self) -> str:
        """The engine's program, installed on first use (:data:`ENGINE_VERSION`)."""
        return self._ensure(
            self.TOOL_NAME,
            ENGINE_PLATFORMS,
            "sherpa-onnx-offline-tts",
            ENGINE_VERSION,
            ENGINE_SHA256,
            self.prompt,
            f"The narration engine (sherpa-onnx {ENGINE_VERSION}, about 25 MB)",
        )

    def voices(self) -> str:
        """The voice model's folder, installed on first use (:data:`VOICE_MODEL`, the same on every platform)."""
        table = {
            p: {"url": _VOICE_URL, "type": "tar.bz2"}
            for p in ("windows", "linux", "darwin")
        }
        digest = {p: VOICE_SHA256 for p in table}
        onnx = self._ensure(
            VOICE_MODEL,
            table,
            "model.int8.onnx",
            "1.1",
            digest,
            self.prompt,
            f"The narration voice ({VOICE_MODEL}, about 140 MB)",
        )
        return os.path.dirname(onnx)

    def prepare(self) -> None:
        """Install the engine and the voice where missing, asked first (:attr:`prompt`).

        Raises:
            FileNotFoundError: Not installed, and the download was declined or nobody could be asked.
            LookupError: No build is pinned for this platform.
            RuntimeError: The download or its check failed.
        """
        self.program()
        self.voices()

    def speak(
        self, said: str, path: str, voice: Any, lang: str, speed: float, threads: int
    ) -> None:
        """Speak *said* into *path* with the Kokoro model (``voice``: its speaker number), its
        text and paths as the engine's command line carries them (:meth:`_text_arg`,
        :meth:`_path_arg`).

        Raises:
            RuntimeError: The engine failed, or took longer than :attr:`TIMEOUT`, or *said* holds
                a character its command line cannot carry.
        """
        from pythontk.core_utils.app_launcher import AppLauncher

        text = self._text_arg(said)
        program, model = self.program(), self.voices()

        def file(name: str) -> str:
            return self._path_arg(os.path.join(model, name))

        lexicon = ",".join(
            file(f)
            for f in ("lexicon-us-en.txt", "lexicon-gb-en.txt")
            if os.path.isfile(os.path.join(model, f))
        )
        args = [
            f"--kokoro-model={file('model.int8.onnx')}",
            f"--kokoro-voices={file('voices.bin')}",
            f"--kokoro-tokens={file('tokens.txt')}",
            f"--kokoro-lexicon={lexicon}",
            f"--kokoro-lang={lang}",
            f"--kokoro-length-scale={1.0 / max(speed, 0.1):.4f}",
            f"--sid={int(voice)}",
            f"--num-threads={max(1, threads)}",
            f"--output-filename={self._path_arg(path)}",
            "--",  # the end of the flags: a line that starts with "--" is still words
            text,
        ]
        try:
            done = AppLauncher.run(
                program, args=args, timeout=self.TIMEOUT, hide_window=True
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"the narration engine took over {self.TIMEOUT} s on '{said[:60]}'"
            ) from None
        if (
            done.returncode != 0
            or not os.path.isfile(path)
            or os.path.getsize(path) <= 44
        ):
            message = ((done.stderr or "") + (done.stdout or "")).strip()[-400:]
            raise RuntimeError(
                f"the narration engine failed on '{said[:60]}' (exit {done.returncode}): {message}"
            )


#: A line the planner left to speak: ``(said, file name, [(line id, text key), ...])`` by path --
#: lines of the same words share one file, spoken once.
_Todo = Dict[str, Tuple[str, str, List[Tuple[str, str]]]]


class _NarratorInternal:
    """Internal helpers for :class:`Narrator`: digests, the lock file, the engine call, and the
    parts of :meth:`Narrator.render` (plan, speak, prune, credits, lock)."""

    @staticmethod
    def _digest(path: str) -> str:
        """The SHA-256 of the file at *path*."""
        sha = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                sha.update(block)
        return sha.hexdigest()

    @staticmethod
    def _text_key(text: str) -> str:
        """A line's words, as written, digested: what a recording is stamped with (no engine,
        voice or pronunciation in it, so a new engine never makes a recording look stale)."""
        return hashlib.sha256(
            " ".join((text or "").split()).encode("utf-8")
        ).hexdigest()[:16]

    @staticmethod
    def _length(path: str) -> Optional[float]:
        """How long the audio file at *path* plays; None where it is not a WAV file (an MP3 take)."""
        try:
            return Narrator.seconds(path)
        except (wave.Error, EOFError, OSError):
            return None

    @classmethod
    def _read_lock(cls, out_dir: str) -> Dict[str, Any]:
        """The lock file in *out_dir* (one written before the lock had a ``version`` reads as it
        is); an empty one where there is none."""
        path = os.path.join(out_dir, Narrator.LOCK)
        try:
            with open(path, encoding="utf-8") as fh:
                lock = json.load(fh)
        except (OSError, ValueError):
            lock = {}
        lock.setdefault("lines", {})
        return lock

    def _synthesize(self, said: str, path: str, threads: int) -> None:
        """Speak *said* into *path* through a temporary file beside it, promoted only once the
        engine finished: a stopped or failed run never leaves a file that counts as done."""
        from pythontk.file_utils._file_utils import FileUtils

        FileUtils.atomic_write(
            path,
            lambda part: self.engine.speak(
                said, part, self.voice, self.lang, self.speed, threads
            ),
        )

    @staticmethod
    def _owned(old: Dict[str, Any]) -> Tuple[Set[str], Dict[str, str]]:
        """``(theirs, ours)`` of a lock's *old* lines: the file names a person replaced -- theirs
        on every later run, never adopted or pruned -- and ``{file name: digest}`` of the
        narration the lock rendered."""
        theirs = {e.get("file") for e in old.values() if e.get("kind") == "replaced"}
        ours = {
            e.get("file"): e.get("sha256")
            for e in old.values()
            if e.get("kind") == "synthetic" and e.get("file") not in theirs
        }
        return theirs, ours

    @staticmethod
    def _issue(ctx, code: str, message: str, location: str) -> None:
        if ctx is not None:
            ctx.issue(code, message, severity="warning", location=location)

    def _plan(
        self,
        lines: Mapping[str, str],
        out_dir: str,
        recordings: Mapping[str, str],
        accept: Set[str],
        old: Dict[str, Any],
        ctx,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], _Todo]:
        """Classify each line: a recording, a narration file there already (reused, or replaced
        by a person), or one to speak (asked). ``(entries, out, todo)``: the lock's entries so
        far, the result being built, and what is left to speak."""
        theirs, ours = self._owned(old)
        entries: Dict[str, Dict[str, Any]] = {}
        out: Dict[str, Any] = {
            k: []
            for k in ("rendered", "reused", "recordings", "stale", "replaced", "pruned")
        }
        out["seconds"] = {}
        todo: _Todo = {}
        for line_id, text in lines.items():
            if not (text or "").strip():
                continue
            words = self._text_key(text)
            recording = recordings.get(line_id)
            if recording:
                self._plan_recording(
                    line_id,
                    words,
                    recording,
                    old.get(line_id, {}),
                    line_id in accept,
                    entries,
                    out,
                    ctx,
                )
                continue
            name = self.key(text) + ".wav"
            path = os.path.join(out_dir, name)
            if path in todo:
                todo[path][2].append((line_id, words))
            elif os.path.isfile(path):
                now = self._digest(path)
                kept = ours.get(name)
                if name in theirs or (kept and kept != now):
                    entries[line_id] = {
                        "kind": "replaced",
                        "file": name,
                        "text_key": words,
                        "sha256": now,
                    }
                    out["replaced"].append(line_id)
                    self._issue(
                        ctx,
                        Narrator.REPLACED,
                        f"{line_id}: its narration file {name} was replaced in place: it is kept as it "
                        f"is; keep a recording under a name of its own instead, so an edit of the line cannot orphan it",
                        line_id,
                    )
                else:
                    entries[line_id] = {
                        "kind": "synthetic",
                        "file": name,
                        "text_key": words,
                        "sha256": kept or now,
                    }
                    out["reused"].append(line_id)
                length = self._length(path)  # a person's file need not be a WAV
                if length is not None:
                    out["seconds"][line_id] = length
            elif ctx is None or ctx.should_write(path, f"narrate {line_id}"):
                todo[path] = (self.spoken(text), name, [(line_id, words)])
        return entries, out, todo

    def _plan_recording(
        self, line_id, words, recording, before, accepted, entries, out, ctx
    ) -> None:
        """Stamp a line's recording with the line it was first seen for: known by its content
        (and named by its file name, never a machine path), so the same take found elsewhere --
        the project moved -- keeps its stamp, and a take recorded again is made for its line."""
        sha = self._digest(recording) if os.path.isfile(recording) else None
        same = (
            before.get("kind") == "recording"
            and sha is not None
            and before.get("sha256") == sha
        )
        stamp = words if (accepted or not same) else before.get("accepted_key", words)
        entries[line_id] = {
            "kind": "recording",
            "file": os.path.basename(recording),
            "sha256": sha,
            "text_key": words,
            "accepted_key": stamp,
        }
        out["recordings"].append(line_id)
        if stamp != words:
            out["stale"].append(line_id)
            self._issue(
                ctx,
                Narrator.STALE_RECORDING,
                f"{line_id}: its recording ({os.path.basename(recording)}) was made for an "
                f"older line: re-record it, or accept it for the new line",
                line_id,
            )
        length = None if sha is None else self._length(recording)
        if length is not None:
            out["seconds"][line_id] = length

    def _speak_all(
        self, todo: _Todo, out_dir: str, workers: int, old, entries, out, ctx
    ) -> None:
        """Speak each planned file, several at once, after the engine's ``prepare`` (an install,
        asked once, here); the lock records what was made even when the run stops."""
        prepare = getattr(self.engine, "prepare", None)
        if callable(prepare):
            prepare()
        os.makedirs(out_dir, exist_ok=True)
        cores = os.cpu_count() or 2
        workers = workers or max(1, min(4, cores // 4))
        threads = max(1, cores // (2 * workers))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {
                pool.submit(self._synthesize, said, path, threads): (path, name, users)
                for path, (said, name, users) in todo.items()
            }
            try:
                for done, job in enumerate(as_completed(jobs), 1):
                    path, name, users = jobs[job]
                    job.result()
                    sha, seconds = self._digest(path), self.seconds(path)
                    for line_id, words in users:
                        entries[line_id] = {
                            "kind": "synthetic",
                            "file": name,
                            "text_key": words,
                            "sha256": sha,
                        }
                        out["rendered"].append(line_id)
                        out["seconds"][line_id] = seconds
                    if ctx is not None:
                        ctx.progress(
                            done / len(jobs),
                            "narrated " + ", ".join(i for i, _ in users),
                        )
            except BaseException:
                for job in jobs:
                    job.cancel()
                raise
            finally:
                # What was made stays recorded, so a stopped run resumes where it stopped.
                self._write_lock(out_dir, old, entries, ctx, partial=True)

    def _prune(self, out_dir: str, old, entries, out, ctx) -> None:
        """Remove the narration files no line uses that are still exactly as rendered."""
        ours = self._owned(old)[1]
        used = {e["file"] for e in entries.values() if e.get("kind") != "recording"}
        for name in sorted(os.listdir(out_dir)):
            rendered = ours.get(name)
            if not name.lower().endswith(".wav") or name in used or rendered is None:
                continue
            path = os.path.join(out_dir, name)
            if rendered != self._digest(path):
                continue  # a person's file now: never removed
            if ctx is None or ctx.should_delete(path, "remove narration no line uses"):
                os.remove(path)
                out["pruned"].append(name)

    def _write_credits(self, out_dir: str, ctx) -> None:
        """Write the engine's credits beside the narration it voiced, where they changed."""
        credits = os.path.join(out_dir, Narrator.CREDITS)
        text = self.engine.credits
        current = (
            Path(credits).read_text(encoding="utf-8")
            if os.path.isfile(credits)
            else None
        )
        if current != text and (
            ctx is None or ctx.should_write(credits, "name the voice's terms")
        ):
            with open(credits, "w", encoding="utf-8") as fh:
                fh.write(text)

    def _write_lock(
        self,
        out_dir: str,
        old: Dict[str, Any],
        entries: Dict[str, Any],
        ctx,
        partial: bool = False,
    ) -> str:
        """Write the folder's lock (atomically): *entries*, and, for a stopped run (*partial*), the
        lines it did not reach as they were."""
        from pythontk.file_utils._file_utils import FileUtils

        path = os.path.join(out_dir, Narrator.LOCK)
        lines = dict(old, **entries) if partial else entries
        body = json.dumps(
            {
                "version": Narrator.LOCK_VERSION,
                "engine": self.engine.name,
                "model": self.engine.model,
                "lines": lines,
            },
            indent=1,
            sort_keys=True,
        )
        current = (
            Path(path).read_text(encoding="utf-8") if os.path.isfile(path) else None
        )
        if (
            current != body
            and os.path.isdir(out_dir)
            and (
                ctx is None or partial or ctx.should_write(path, "record the narration")
            )
        ):
            FileUtils.atomic_write_text(path, body)
        return path


class Narrator(_NarratorInternal):
    """Speak lines of text into WAV files, each named by what it says and how, so a script
    edited and narrated again re-voices only the lines that changed.

    Parameters:
        voice: The voice's speaker number in the model (its README lists them).
        lang: The language the lines are in (``en``).
        speed: How fast it speaks (1: as trained).
        pronounce: Words and how to say them (``{"PCB": "P C B"}``), whole words, before the
            engine reads the line: an abbreviation, a product name, an acronym.
        prompt: Consent policy for the first install -- ``True`` asks on the console, ``False``
            needs none, a callable ``(question) -> bool`` is asked instead (a panel's dialog).
        engine: What speaks the lines (a :class:`NarrationEngine`); none: :class:`SherpaKokoro`.

    Example:
        >>> narrator = Narrator(pronounce={"PCB": "P C B"})
        >>> narrator.say("Check the PCB lights.", "audio")["seconds"]  # doctest: +SKIP
    """

    #: The lock file in a narration folder, written only by :meth:`render`.
    LOCK = "narration.lock.json"
    #: The lock file's format (its ``version``; a lock without one reads as this).
    LOCK_VERSION = 1
    #: The credits file :meth:`render` writes beside narration: the voice's terms, to ship with it.
    CREDITS = "NARRATION_CREDITS.txt"
    #: Issue codes :meth:`render` reports.
    STALE_RECORDING = "narration.stale_recording"
    REPLACED = "narration.replaced_in_place"

    def __init__(
        self,
        voice: int = 0,
        lang: str = "en",
        speed: float = 1.0,
        pronounce: Optional[Dict[str, str]] = None,
        prompt: Union[bool, Callable[[str], bool]] = True,
        engine: Optional[NarrationEngine] = None,
    ):
        self.voice = voice
        self.lang = lang
        self.speed = float(speed)
        self.pronounce = dict(pronounce or {})
        self.prompt = prompt
        self.engine: NarrationEngine = (
            engine if engine is not None else SherpaKokoro(prompt=prompt)
        )

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        prompt: Union[bool, Callable[[str], bool]] = True,
    ) -> "Narrator":
        """A narrator from a narration settings block (``voice``, ``lang``, ``speed``, ``pronounce``;
        other keys, such as ``directory``, ignored): one reading of such a block, so a reader that
        looks a line's file up and the render that made it name the file alike.

        Parameters:
            settings: The block (saved data: it holds no consent policy).
            prompt: Consent policy for the first install, as :class:`Narrator` takes it.
        """
        return cls(
            voice=settings.get("voice", 0),
            lang=settings.get("lang", "en"),
            speed=settings.get("speed", 1.0),
            pronounce=settings.get("pronounce"),
            prompt=prompt,
        )

    # ---- speak -----------------------------------------------------------------------------------------------

    def spoken(self, text: str) -> str:
        """*text* as the engine is given it: each word of :attr:`pronounce` replaced, whole words only."""
        said = " ".join((text or "").split())
        for word, how in self.pronounce.items():
            said = re.sub(r"(?<!\w)%s(?!\w)" % re.escape(word), how, said)
        return said

    def key(self, text: str) -> str:
        """The name a line's file takes: a short slug of it and a digest of what decides its sound."""
        said = self.spoken(text)
        digest = hashlib.sha256(
            json.dumps(
                [
                    said,
                    self.engine.model,
                    self.engine.name,
                    self.voice,
                    self.lang,
                    self.speed,
                ]
            ).encode("utf-8")
        ).hexdigest()[:12]
        slug = re.sub(r"[^a-z0-9]+", "_", said.lower()).strip("_")[:32] or "line"
        return f"{slug}_{digest}"

    def say(self, text: str, out_dir: str, ctx=None) -> Dict[str, object]:
        """Speak *text* into a WAV file in *out_dir* (made where missing), unless a file of the same
        line, voice and settings is there already.

        Parameters:
            text: The line.
            out_dir: The folder the file is written to.
            ctx: The run's context (an ``OpContext``), asked before the file is written; none: write.

        Returns:
            The line's record: ``file`` (its path), ``seconds`` (None where a dry run made no file),
            ``text``, ``voice``, ``engine``, ``model``, ``license``, ``synthetic`` (True: a stand-in
            until a recorded line replaces it).

        Raises:
            ValueError: *text* says nothing.
            FileNotFoundError: The engine or the voice is not installed and may not be.
            RuntimeError: The engine failed (its message); no file is left behind.
        """
        said = self.spoken(text)
        if not said:
            raise ValueError("nothing to say: the line is empty")
        path = os.path.join(out_dir, self.key(text) + ".wav")
        if not os.path.isfile(path) and (
            ctx is None or ctx.should_write(path, f"speak '{said[:48]}'")
        ):
            os.makedirs(out_dir, exist_ok=True)
            self._synthesize(said, path, max(1, (os.cpu_count() or 2) // 2))
        return self._record(text, path)

    def _record(self, text: str, path: str) -> Dict[str, object]:
        return {
            "file": path,
            "seconds": self.seconds(path) if os.path.isfile(path) else None,
            "text": text,
            "voice": self.voice,
            "engine": self.engine.name,
            "model": self.engine.model,
            "license": self.engine.license,
            "synthetic": True,
        }

    def narrate(self, lines: Iterable[str], out_dir: str) -> List[Dict[str, object]]:
        """:meth:`say` each line of *lines* (blank ones skipped) into *out_dir*; their records, in order.

        Parameters:
            lines: The lines, in order.
            out_dir: The folder the files are written to.
        """
        return [self.say(line, out_dir) for line in lines if (line or "").strip()]

    # ---- render ----------------------------------------------------------------------------------------------

    def render(
        self,
        lines: Mapping[str, str],
        out_dir: str,
        recordings: Optional[Mapping[str, str]] = None,
        accept: Iterable[str] = (),
        prune: bool = False,
        workers: int = 0,
        ctx=None,
    ) -> Dict[str, Any]:
        """Narrate every line of *lines* that has no file yet into *out_dir*, several at once, and
        keep the folder's lock file: what each line's file is and its digest.

        A line with a recording (*recordings*) is not narrated: the recording is stamped with the
        line it was first seen for, and a later edit of that line reports it as made for an older
        line (:attr:`STALE_RECORDING`) until it is accepted (*accept*) or recorded again. A
        narration file whose digest no longer matches the lock was replaced in place by a person:
        it is kept and reported on every run, never overwritten or removed (:attr:`REPLACED`).
        With *prune*, narration files no line uses are removed -- only those still exactly as
        rendered. Lines of the same words share one file, spoken once. A credits file names the
        voice's terms. Before it speaks, the engine's ``prepare`` runs here, once (an install,
        asked first).

        Parameters:
            lines: ``{id: text}``, every line with words in it narrated (a blank one skipped).
            out_dir: The narration folder.
            recordings: ``{id: recorded file}`` for the lines a person recorded (a WAV file is
                measured; another format is kept unmeasured).
            accept: Ids whose recordings are accepted for their current text.
            prune: Remove narration files no line uses.
            workers: Engine runs at once (0: about a quarter of the cores, at most 4).
            ctx: The run's context (an ``OpContext``): asked before each write and removal,
                told the progress, its cancel honored between lines; none: make the changes.

        Returns:
            ``{"rendered", "reused", "recordings", "stale", "replaced", "pruned"}`` (ids, or
            file names for ``pruned``), ``"seconds"`` ``{id: length}`` and ``"lock"`` (its path).

        Raises:
            FileNotFoundError: The engine is not installed and may not be (its ``prepare``).
            RuntimeError: The engine failed on a line; the lines made before it stay recorded.
        """
        old = self._read_lock(out_dir)["lines"]
        entries, out, todo = self._plan(
            lines, out_dir, dict(recordings or {}), set(accept), old, ctx
        )
        if todo:
            self._speak_all(todo, out_dir, workers, old, entries, out, ctx)
        if prune and os.path.isdir(out_dir):
            self._prune(out_dir, old, entries, out, ctx)
        if any(e["kind"] == "synthetic" for e in entries.values()):
            self._write_credits(out_dir, ctx)
        out["lock"] = self._write_lock(out_dir, old, entries, ctx)
        return out

    @staticmethod
    def seconds(path: str) -> float:
        """How long the WAV file at *path* plays, in seconds.

        Parameters:
            path: A WAV file.
        """
        with wave.open(path, "rb") as clip:
            return clip.getnframes() / float(clip.getframerate())
