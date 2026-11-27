# Narration

`ptk.Narrator` speaks lines of text into WAV files with an open-source engine on this machine: a
scene's or a step's narration, heard before anyone records it. No service is called, and the text
never leaves the machine.

```python
import pythontk as ptk

narrator = ptk.Narrator(pronounce={"PCB": "P C B"})
record = narrator.say("Check the PCB lights on the solenoid.", "audio")
record["file"], record["seconds"]          # audio/check_the_p_c_b_lights_on_the_so_<digest>.wav, 2.9
```

## The engine and its voice

| | |
|:---|:---|
| Engine | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 2.0.1 (Apache-2.0): Windows x64, Linux x64, macOS universal |
| Voice | Kokoro v1.1, int8, many languages (Apache-2.0 weights; its training audio is CC BY, so credit it where the narration ships) |
| Install | On first use, through `AppInstaller`: pinned releases, SHA-256 checked, after the caller's consent (`prompt=True` asks on the console, `False` needs none, a callable is a panel's dialog) |

That pair is the default engine, `SherpaKokoro`. A project passes its own as `Narrator(engine=...)`:
anything with the `NarrationEngine` shape (`name`, `model`, `license`, `credits`, and
`speak(said, path, voice, lang, speed, threads)` writing a WAV file) -- another open engine, a
studio's service. Its name and model are part of each file's name, so a new engine re-voices every
line and nothing else does. An engine may also have `prepare()`: `render` calls it once, on the
calling thread, before it speaks lines several at once, so an install is asked for and made once
(`SherpaKokoro.prepare()` installs both halves). A bumped pin installs anew: the engine's name in
the tools catalog carries its version, the voice's name its model's.

A line's file is named by what decides its sound -- its words as spoken, the voice, language,
speed, model and engine -- so narrating an edited script again re-voices only the lines that
changed. `pronounce` says whole words as written there (an acronym, a product name). The engine
writes through a temporary file, promoted only once it finished: a stopped run never leaves a file
that counts as done.

## A sheet's narration

A sheet is narrated by the **`narration.render`** operation (the sheet reader's `Mapping.narrate`), never while it
is read. Choose the templates' audio option **Recordings, else narration** (`{"audio": "narrate"}`):

```python
from pythontk.core_utils.engines.shots.manifest.mapping import Mapping

Mapping.narrate("course.csv", mapping="interactive", options={"audio": "narrate"}, base="project")
steps = Mapping.resolve("course.csv", name="interactive", options={"audio": "narrate"}, base="project")
```

Each step's audio is then found by a list of methods, the first to find a file winning: the file
its **Audio Files** cell names in `recordings/`, a recording named by its step ID there, else the
narration in `narration/`. A recording replaces narration with no bookkeeping; a step whose voice
text is `N/A` says nothing. `base` is the project's settings folder: relative folders resolve
there, never beside the sheet (an exported sheet lands somewhere new each time). The sheet itself
is never written.

The first run installs the engine and voice (about 170 MB), asked first: `prompt=True` asks on the
console, or through `confirm` -- a panel's dialog, `(question) -> bool`; `prompt=False` installs
unasked. The operation takes the plain `prompt`: where nobody can be asked (an agent, an MCP
session), nothing is downloaded and the run stops; `prompt=False` is the caller's consent.

`render` keeps a lock file, `narration.lock.json`, beside the narration -- written by it alone:

- **A recording** is stamped with the line it was first seen for; once the line is edited it is
  reported (`narration.stale_recording`) until re-recorded or accepted (`accept=["A03"]`). It is
  known by its file name and content, never a machine path: a moved project keeps its stamps, and
  a take recorded again is made for the line it was recorded for. A take that is not a WAV file
  (an MP3) is kept as it is, its length unmeasured.
- **A narration file a person replaced in place** is kept, reported on every run and never
  overwritten or removed (`narration.replaced_in_place`): name recordings by their step instead.
- **`prune=True`** removes narration no line uses -- only files still exactly as rendered.
- **`NARRATION_CREDITS.txt`** names the voice's terms, to ship with the narration.

The lock carries its format's `version` (1; a lock without one reads as it). Lines render several
at once, report progress and honor a cancel between lines; lines of the same words share one file,
spoken once. A dry run lists the writes and makes none (and installs nothing). A stopped run
resumes where it stopped: run it again.
