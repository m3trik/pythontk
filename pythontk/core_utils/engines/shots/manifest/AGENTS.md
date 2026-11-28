# pythontk/core_utils/engines/shots/manifest

**Role**: The build sheet (a production's sequence doc) as every tool reads it: the templates that declare its format, the reader (`ManifestModel`), the one check (`ManifestActions`) and the Shot Manifest's engine.

**Nav**: [← pythontk](../../../../../AGENTS.md) · [← root](../../../../../../AGENTS.md) · the format, for authors: [build sheets](../../../../../docs/build_sheets.md)

**What the format is for** (hold every change to these): one sheet is the truth for the whole pipeline; each tool that reads it works without the others; nothing in it is implied or guessed; a new need extends it and never reshapes it.

## Rules

- **One declaration.** A word (its cells, the ones it needs, its settings, its moment, the sentence it reads as) is template data (`mapping/<template>.json`, `actions`), read through `ManifestActions`. Code names no word; a starter, a doc and every other reader restate the declaration, each held to it by a test.
- **A word is named for its family at its most general** (`indicate`, `show`, `travel`, `picture`). A form of it (an arrow, a fade, a teleport, a picture-in-picture) is a setting's value or the reader's to build, never a word, a builder or a setting's name of its own.
- **Two kinds of row.** A step's row (the one with an ID) is the step alone: name, voice, placard, audio, its own To and Settings. A row with no ID is one action. A cell holds one kind of thing; no cell's prose is read, and Notes never is.
- **Names are exact.** A name is a node's as the delivery spells it; where two share it, the end of its path that tells it apart (`ptk.NodeNames.ends`: the fewest trailing names no other node's path ends with), never only the whole path. A strict template infers, strips and matches nothing; the tolerant `default` template is a person's choice, never a fallback.
- **A step, a shot, a take.** A step's ID is written as every tool spells it (`A01`; one added later, `B03A`) and never changes. A shot is its step: named by the step's ID, holding its narration and the takes its rows play. A take is a span inside its step's shot, what one row plays and one clip of an export; its name starts with its step's ID and never changes. Clip is the take a row belongs to. Nothing is split or named by hand.
- **Order is declared, never positional.** A step has three moments (its start, the user's, its end); a word's `when` says which. Order lives between steps; rows sharing a moment run top to bottom; a row waits on another only for what it needs (its With in hand, `after=` a take). Row position means nothing else.
- **A word is made of five primitives**: trigger, requirement, effect, look, place. A setting is declared once, in the template's `settings` (its type, its `primitive`, its bounds), so its name means one thing in every word (`when`, `hold`: the trigger's; `until`, `state`, `speed`, `reply`: the effect's; `color`, `size`, `style`, `fade`: the look's); a word lists the ones it takes with only its own choices, default and `required` (`ManifestActions.resolve`, held by `check_declared`). A new setting takes the name its primitive already has.
- **Extend in this order**: a setting on a word that exists; a word composed of the primitives; a project's own template. A new column, a second meaning for a cell, a branch on a word's name or a setting that means two things changes the format: stop and ask.
- **One meaning per field, whatever object it holds.** Target is what a row is about; `where` (a setting of type `object`) is the object a placing word happens at; With is the tool in hand. Nothing branches on the kind of object a field is given.
- **Placement is no setting.** A placed thing (a viewpoint, a pointer's tip, a picture's camera) is an object in the scene, named in `where`; empty, the tools make the row's own helper once. Where a thing sits is never a number or a hint in the sheet.
- **A shown thing has a span** (`until`: the step, or a step's ID), never a second row that hides it.
- **Every problem is an issue at its cell, with its fix**, the cell named by the sheet's own header, from the one check (`check_row`, `check_step`, `check_step_row`) every tool reports alike. Never a silent drop, a guess, or a default that hides it.
- **A row reads back as a sentence** made from its cells (a word's `says`); no tool stores one.
- **A starter is the sheet a new one is copied from**: `mapping/starters/<template>.csv` uses every word its template declares and passes its strict read with nothing to report.
- **A reader builds what it can and leaves the rest alone**: the Shot Manifest keys the looks (`highlight`, `show`, `hide`), makes the helper a placing row names in `where` where the scene has none (`ManifestPlacement`: a plain first pose, never moved again), gives a row's Target the state it names (`set_states`), and shows every other word untouched.
- **Fixtures are synthetic**: a client's sheet, names or paths never enter a file or a test here.

**Proof**: `python pythontk/tests/run_tests.py test_shots_manifest_actions test_shots_manifest_core` after any change here. A change to a template, a word or a starter also runs each reader's own spec test: `python m3trik/scripts/affected_tests.py <files> --run`.
