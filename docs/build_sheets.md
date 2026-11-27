# Build sheets

A build sheet is the procedure a production is made from (its sequence doc): a CSV file or a Google
Sheet, written by an author and reviewed by a client. The Shot Manifest reads it into shots, highlights,
fades and voice clips; other readers (a game engine's task kit) read the same sheet for their own work,
and each works without the others. This guide is the sheet's format: what a sheet holds so that every reader
takes it the same way, with nothing guessed.

**Contents:** [Templates](#templates) · [Rows](#rows-a-step-then-what-happens-in-it) · [Columns](#columns) ·
[Words](#words) · [Settings](#settings) · [Steps and sections](#steps-and-sections) ·
[Order within a step](#order-within-a-step) · [Names](#names) · [Shots and takes](#shots-and-takes) ·
[What the check says](#what-the-check-says) · [A sheet of sentences](#a-sheet-of-sentences) ·
[A sheet written before the format](#a-sheet-written-before-the-format)

## Templates

A template is a mapping file: the column headers it knows, the words its Action cells may hold, the
columns it passes through and its option defaults. Three ship built in, and a user's own go in the
templates folder the Shot Manifest's column picker opens.

| Template | For | Columns |
|:---|:---|:---|
| `linear` | Animated procedures: rendered video, a web click-through | Step, Step Name, Voice, Audio, Action, Target, Settings, Notes |
| `interactive` | A step-by-step task in desktop or VR (work instructions, a demo, a walk-through) | Linear's, plus Placard, With, Clip, Members, Label and To |
| `default` | Any sheet, including one written before this format | Whatever it finds; object names inferred from the prose where none are listed |

Linear and Interactive are formats a reader can hold a sheet to: a row nothing reads, or a name the
scene lacks, is an error to fix, never a guess (how strictly is each reader's own setting). The tolerant
`default` template is a choice a person makes for an older sheet, never what a strict read falls back
to. Each strict template ships a **starter sheet**: a small task that uses every word its template
declares and reads with nothing to report. A new sheet is a copy of it, filled:
`python -m pythontk ops apply sheet.new out=<file.csv> template=interactive` (or `linear`) writes one
and never replaces a file (`Mapping.new_sheet`; the starter itself: `Mapping.starter_sheet`).

## Rows: a step, then what happens in it

A sheet is one table with two kinds of row.

- A **step row** has a Step ID and is the step alone: its name, its narration, what the user reads,
  and where it goes by itself. It holds no action.
- An **action row** has no Step ID, belongs to the step above it, and says one thing that happens: one
  word in Action, and the cells that word takes.

| Step | Step Name | Voice | Action | Target | With | Clip | Label | To | Settings |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `A01` | Unlock the latches | Unlock both latches with the key. | | | | | | | |
| | | | `teleport` | `UNIT` | | | | | `where=VIEW_FRONT` |
| | | | `highlight` | `LATCHES` | | | | | |
| | | | `take` | `KEY` | | | | | |
| | | | `use` | `LATCH_UPPER` | `KEY` | `A01_1` | | | |
| | | | `use` | `LATCH_LOWER` | `KEY` | `A01_2` | | | |
| | | | `return` | `KEY` | | | | | |
| `B04` | Did it light? | Did the lamp light? | | | | | | | |
| | | | `button` | | | | YES | `B05` | |
| | | | `button` | | | | NO | `B04A` | |

Every cell holds one kind of thing, and no cell's prose is read. Filtering a sheet to its step rows
gives the client's script: ID, name, voice, placard. A step row that also fills an action's cells is an
issue (*a step's row holds no action: write this one on the row below it*).

## Columns

| Column | Row | Holds |
|:---|:---|:---|
| Step | step | The step's ID; a row with none belongs to the step above |
| Step Name | step | Its title |
| Voice | step | The narration, as spoken |
| Placard | step | (Interactive) what the user reads; empty: nothing shown |
| Audio | step | The narration's clip file, where one exists |
| Action | action | One word from the template's list (`use`, `highlight`, `button`); `Fade In` reads as `fade_in` |
| Target | action | The parts the action is about, by scene name, one per line |
| With | action | What it is done with: the tool in hand |
| Clip | action | The take the row belongs to: the one a `use` or `play` row plays, or the one a look is keyed into |
| Members | action | What the take animates; empty: the row's Target and With |
| Label | action | The words it shows: a button's, a typed answer, a checklist item, a caption |
| To | both | Where it goes: `next`, a step's ID, `restart`, `exit`, `retry`. On a step row: where the step goes by itself once done (empty: the next step) |
| Settings | both | `name=value; name=value`, over the word's defaults. On a step row: the step's own, where its template declares any (`step_settings`) |
| Notes | both | Comments for people. Never read; a sheet may leave the column out |

Step Name, Voice and Placard are what a viewer meets: free prose, never parsed, the columns a client
reviews and rewords. Header aliases are template data, so a sheet that calls its narration "Narration" or
"VO" reads the same; a header no template field takes is reported (`ColumnMap.unmapped`), never dropped
silently. Each row's own cells are kept with it (`BuilderStep.rows`), and a step row's To and Settings
with its step (`BuilderStep.to`, `BuilderStep.settings`).

A sheet is in this format when its header carries one of its own columns (Target, With, Members, Label,
To, Settings: a template's `cell_marks`); any other is [a sheet of sentences](#a-sheet-of-sentences).

## Words

A template declares its words (its `actions`): for each, the cells it takes and needs, its settings,
the moment it happens at and the sentence a row of it reads as (`ManifestActions`; the declaration's
reference is `MAPPING_FORMAT.md`, written beside a user's templates). Nothing in code knows a word: a
project's own template adds its own.

The Interactive template's seventeen:

| Family | Word | Happens | Cells (needed in bold) | Settings |
|:---|:---|:---|:---|:---|
| Attention | `highlight` | as the step starts | **Target**, Clip | `style` (auto, pulse, steady), `color` |
| | `indicate` | start | **Target** | `size`, `color`, `where` (the arrow) |
| Visibility | `fade_in` | start | **Target**, Clip | `fade` (seconds) |
| | `fade_out` | as the step ends | **Target**, Clip | `fade` |
| State | `set` | start | **Target** | `state` (required) |
| Hand | `take` | the user | **Target** | |
| | `return` | end | **Target** | |
| Animation | `use` | the user | **Target**, With (the tool in hand), **Clip**, Members | `trigger` (select, hold, near), `hold` (seconds held) |
| | `play` | start | **Clip**, **Members** | `after` (the take it follows), `speed` |
| Flow | `button` | the user | **Label**, **To** | `when` (end, start), `reply` |
| | `select` | the user | **Target**, To | |
| | `answer` | the user | **Label** (`*`: any other), **To** | |
| Movement | `teleport` | start | Target, what the user is brought to face; or `where` alone | `when` (start, end), `ring` (a span), `where` (the viewpoint) |
| Screen | `pip` | start | **Target**, Label | `until` (a span), `where` (the camera) |
| | `checklist` | start | Label, an item a line | `until` (a span: the step it leaves the placard at) |
| | `check` | end | **Label** | |
| Note | `note` | never | | |

The Linear template declares the three the Shot Manifest keys (`highlight`, `fade_in`, `fade_out`), and
a row with a Target and no Action lists a member of the step's shot that nothing is keyed on. A word a
reader does not build is left alone: the Shot Manifest keys the three and reads the rest as the step's
script.

Every row is also said back as a sentence made from its cells (`row["says"]`: "Selecting LATCH_UPPER
with KEY in hand plays A01_1"), which is what the Shot Manifest shows as the step's description, so a
person reads the procedure without anyone having written it twice.

## Settings

A word is a combination of five parts: what starts it (a trigger), what must be true first (a
requirement: its tool in hand, a take finished), what it does (an effect), what a shown thing looks like
(a look) and where a placed thing sits (a place). A setting belongs to its part, so a name means one
thing in every word:

| Part | Its settings |
|:---|:---|
| Trigger | `when` (start or end, where a word may move), `trigger` (select, hold, near), `hold` (seconds a press is held) |
| Requirement | `after` (a take of the step that must finish first); the tool in hand is the row's With |
| Effect | `until` and `ring` (how long a shown thing lasts), `state`, `speed`, `reply` |
| Look | `color`, `size`, `style`, `fade` (seconds) |
| Place | `where`: the object a placing word happens at (a viewpoint, an arrow, a camera), used exactly as it stands; empty, the row's own helper, made once. Never a position or a hint. Empty, the row's own helper is named from what the step says, alike in every tool (the Shot Manifest makes it under that name; a reader that runs the steps takes the delivered one, else makes its own under it): `CAM_<step>_<target>` for a picture, `VIEW_<step>_<target>` for a viewpoint, `ARROW_<step>_<target>` for an arrow, the target by its node's own name, or, where another target of the step ends with that name, by the fewest names of its path no other ends with (`PANEL\|LATCH` beside `DOOR\|LATCH`: `CAM_A01_PANEL_LATCH`). Two rows of one purpose about one target in a step must name theirs. A name the scene does not have yet is made by the Shot Manifest's build under that name: a camera for a picture, a locator for a viewpoint or an arrow's tip, at a plain first pose (beside what the row is about, read from its bounds alone), there to be moved; one the scene has is never moved by a build |

A value is held to its setting's type: text, number, flag (`on`, `off`), choice, color (`#rgb` or
`#rrggbb`), clip (a take its step plays), step (a step's ID), span, object (one object's exact name, or
the end of its path where two share it), state (a state of its targets' indicator light, as the delivery's
emissive groups name them: `off`, `amber`). A **span** says how long a shown
thing lasts: `step` (while its step runs) or a step's ID (until that step starts): a picture's
`until=B06`, a viewpoint's ring `ring=step`. So no second row is written to hide it.

## Steps and sections

Step IDs are letters, digits and underscore, written as every tool spells them: `A01`, `IN_01`, a quiz
question `Q01`. A step ID is an identity: once used it never changes, and a step added later takes a
letter (`B03A`) rather than a renumbering, so everything pointing at a step keeps pointing at it. An
older sheet's dotted ID (`B03.5`, shot `B03_5`) still reads. A row `SECTION X: Title` starts a section,
as do `INTRO`, `OUTRO` and `QUIZ` rows.

A step with no `button`, `select` or `answer` row goes on by itself once done, to its step row's To
(empty: the next step): a wrong-answer step `B04A` whose To is `B04` needs no button. A quiz question is
a step whose buttons go `next` (right) or `retry` (wrong: the step stays and says the row's `reply`).

## Order within a step

A step has three moments: as it starts, while the user acts, as it ends. Each word belongs to one
(the table's *Happens*; `when=` moves a button or a teleport).

- **Order lives between steps.** Inside a step the user's rows are all open at once and the step
  waits for all of them (two latches, either order). To force an order between two actions, make them
  two steps.
- **A row waits for another only when it needs something from it**: its tool in hand (With), or, for a
  take no user triggers, the take named in `after=`.
- **Rows that share a moment happen top to bottom.** Row order means nothing else.

## Names

The sheet declares the names and the scene matches them: the sheet is written before the scene exists,
and the artist builds from it. In this format a name is a node's own, exactly as the scene spells it
(`KEY_GRP`), or, where two nodes share a name, the end of its path that tells it apart: the fewest names
from the node upward that no other node's path ends with (`TABLE|KEY_GRP`, never the whole path). Nothing is
inferred from a row's wording, and a name that is no node, or that two nodes answer to, is an issue at its
cell, which says the ends that tell them apart.

## Shots and takes

A shot is its step: named by the step's ID, in sheet order, as long as the step needs (its narration,
and every take its rows play). A take is what one trigger plays: a span inside its step's shot and one
clip of an export. Its name starts with its step's ID (`A02`; `A01_1`, `A01_2` where a step has several)
and, once used, never changes: a take added beside `A05` is `A05_2`. Another tool's delivery keeps its
own take names: the check warns, never refuses.

**Clip says which take a row belongs to.** A `use` or `play` row plays it. A look (`highlight`,
`fade_in`, `fade_out`) with a Clip is keyed into that take; with none it belongs to its step. A row
that moves nothing leaves the cell empty.

A look that belongs to its step happens at the step's own moment. In a sheet played straight through
(Linear) the step is its shot, so the DCC keys the look into it. In an Interactive sheet a step waits
on its user and its takes play when they act, so the look is the task's to show as the step starts or
ends (the tool that runs the steps builds it), and the DCC keys nothing for it: keyed into a take, a
part that fades in for the user to pick up would stay hidden until they had. A template says which
(`step_looks`: `key` or `leave`).

The cells also say which steps have a shot: a template's **Shots** option.

| Choice | Gives |
|:---|:---|
| *Every step* | Each step a shot (every template's default: a step's narration needs its time whether or not anything moves). A step whose rows play takes holds them inside its shot (`ShotStore.TAKES`); one that plays none is as long as its narration and exports no clip |
| *Steps that play a take* | A shot only for a step one of whose rows names a take: a step that moves nothing has none, and its narration is the task's to play |

A scene whose takes were cut by hand as shots of their own, before its sheet said so, is found as it
stands: Build joins a step's take shots into the step's one shot, each take keeping its frames
(`ShotManifest.join_takes`). Where another shot stands between them, or one is locked, they are left as
they are and it says so.

## What the check says

One check holds a sheet to its template (`ManifestActions.check_row`, `check_step`, `check_step_row`),
and every reader reports it alike: each problem is an issue at its cell, named by the sheet's own
header, with its fix.

- A word the template lacks; a cell the word does not take, or one it needs left empty; a column the
  sheet's words need and its header lacks (said once, for the sheet).
- A setting the word does not have; a value its type refuses; a required setting missing; two
  settings split by a comma (a `;` or a line splits them).
- A choice (To) going to no step; a step row going to no step; two rows of a word sharing a cell the
  word declares `distinct` (two buttons of one Label, two answers taking any other value); two rows
  whose own helpers would share a name (one purpose, one target: give one its `where`).
- A take two rows play; a look or an `after` naming a take its step does not play; and, as a warning, a
  take not named for its step (its ID, or its ID and `_<n>`), and a span (`until`, `ring`) naming a step the sheet is past by then
  (what the row shows would stay).
- A step row holding an action; a `teleport` naming neither a Target nor a `where`.
- A row no step reads, said with its row in the sheet: a second step of one ID, a Step cell that is
  neither a step's ID nor a section (`a02`, `A 02`, `Section C - Close`), an action row above its
  section's first step. Such a row and the rows under it are left out, never read into the step above.

Every message ends in its fix after one colon, so a tool shows the reason and the fix apart.

## A sheet of sentences

A sheet with none of the format's own columns is read by its sentences, one action per row: the form
sheets took before the cells, still read so that one can be rewritten in its own time.

| Column | Holds |
|:---|:---|
| Step Contents | One action per row, in the sheet's phrasing |
| Asset Names | The row's objects by their scene names, one per line in the cell |
| Action | (Interactive) the row's action from a short list, for a row whose wording no phrase reads |
| Animation Clip Names | (Interactive) the take the row plays |
| Text On Placard | (Interactive) what the user reads during the step |
| Hint | (Interactive) guidance, one line each: `-Bouncing arrow pointing at ...` |

The phrases the Shot Manifest knows are the behavior templates' (highlight, fade in, fade out): their
`detect` patterns find a behavior anywhere in a row.

| Phrase | Keys |
|:---|:---|
| `Highlight <objects>` | A highlight across the step |
| `<object> fades in` | A fade in at the step's start |
| `<object> fades out` | A fade out at the step's end |

Names here are matched, not exact: scene names are `UPPER_CASE` with underscores (`RED_DOOR_LOCKS`), the
build suffixes `_GRP`, `_GEO`, `_LOC` and the like are ignored (`KEYS` names the chain
`KEYS_GRP > KEYS > KEYS_GEO`), and a row's words are matched only against the names that row lists. The
inline-name pattern, the build suffixes, the viewpoint pattern and the arrow pattern are each reader's
own name rules, data beside the reader. A reader's own phrases (a task kit's: taking a tool,
playing a take, branching) are that reader's, in its own documentation.

## A sheet written before the format

The `default` template reads it as it is: object names inferred from the prose where the asset cells are
empty, either step-ID style. A reader's converter rewrites it in the format once (a step's own row, then
a row per action, each cell filled, a dotted ID respelled with a letter, what it could not decide listed
by cell); a person reviews that once, and the strict template then reports whatever is left.
