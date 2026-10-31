# Look match

**Light a baked scene to match a reference image, by eye and region by region.** Each light
of a rig is baked once as a *layer*; light adds, so any mix of the layers -- each weighted and
tinted -- is their sum, predicted in milliseconds and held up against the reference. When a
mix looks right it is saved as a light rig a DCC builds and bakes for real.

**Nav**: [← pythontk docs](README.md) · Related: [Live WebXR preview](webxr_preview.md) (the
viewer a look match is judged in) · panel: [tentacle](https://github.com/m3trik/tentacle)

---

## The pieces

| Piece | What it is |
|---|---|
| `LightLayers`, `LightMix` | The layer store (one linear frame per light, captured alone) and a mix of it: a weight, a colour temperature and a highlight level per layer. `LayerKind` says what a kind of layer is (a light, the scene's emission, a viewer-only panel); `register_kind` adds one. |
| `LightPredictor` | A mix's frame: the tinted, weighted sum of its layers, through the viewer's display transform. |
| `LookTarget`, `LookRegion` | The reference image and the regions measured on both sides (rectangles, ellipses, polygons, quads, lines, painted markers, objects -- `RegionShape` registers a new shape). |
| `LookScore` | The target's measures on a frame against the reference. `summary()` reads each region the way a person does: within tolerance or not, how much brighter on screen, and a word for a colour that is off. |
| `LookFit`, `LookMatch.solve` | Weights fitted to the measures; one layer's weight that brings a region's brightness to the reference's. |
| `LookSheet` | Comparison sheets a person judges by; `region_row` is one region close up, `difference` where one frame is brighter than another. |
| `LookMatch` | The facade over a session, and the command line. |
| `LookSession` | The session file, declared once. |
| `LookCapture` | The real WebXR viewer, headless: the ground truth a prediction is held to, and how a bake becomes a layer. |
| `LookLoop` | A round from a rig by role -- bake what changed (a DCC worker), capture it, fit, judge -- and `rebake`, the layers an older save of the scene made. |
| `LightRigPayload`, `LightLayerInfo` | The two contracts a host's tools trade with: the rig a mix becomes, and a bake folder's `layer.json`. |

## The session file

One JSON file names everything, relative to itself (`LookSession`):

```json
{"version": 1, "id": "room_moody",
 "layers": "cache:layers", "target": "target.json", "mix": "mixes/h4.json",
 "template": "cache:template/ROOM.glb", "size": [820, 480],
 "capture": {"camera": {"pos": [0, 2, 3], "target": [0, 1.6, 2.9], "up": [0, 1, 0], "vfov": 48}},
 "loop": {"queue": "cache:bake_queue",
          "worker": {"module": "mayatk.light_utils.light_layer_baker", "scene": "../scenes/ROOM.ma"}}}
```

A path spelled `cache:<path>` lives in the session's folder in the per-user cache
(`UserConfig.cache_dir("look_match", id)`): derived data -- the captured layers, bakes, renders --
rebuilt from its sources and kept off a synced project drive. Everything else (the reference,
the regions, the mixes) is authored and travels with the project. A session from before the
contract -- no version, no id, plain relative paths -- still reads.

Its saved mixes are one `<name>.json` each and nothing else (the file `LightMix.save` writes; a
name's punctuation other than `-`, `_` and spaces becomes `_`) in the folder its
`mix` sits in, else `mixes/` beside it: `LookSession.mixes(path, data)` is that folder as a
`PresetStore`, which lists, saves and deletes them -- `delete(name, trash=True)` to the
platform's trash, where a mix can be restored.

## The panel

**Look Match** (a tentacle standalone panel) is the panel over all of this. Open a session from the *Session*
combo's option box; the panel loads its layers off the UI thread and keeps the session it had
open usable meanwhile.

- **The mix** is a table of layers, grouped by kind. Type a weight, drag it with the middle
  button (doubling every 80 px), or roll the wheel over it (Shift: finer) -- a drag or a roll is
  one undo step. Weights are drawn on a log scale (a narrow spot may need 50 where a wide area
  light needs 0.1). **S** solos and **M** mutes a layer in the picture without changing the mix;
  **F** marks what Fit may change. *All layers* shows the whole store: raise a layer's weight to
  add it.
- **The comparison** shows two of: the preview, the reference, a WebXR render, a saved mix --
  split (drag the line), side by side (zoomed together), blinked (hold Space), or their
  difference (red where A is brighter, blue darker). During a drag the picture follows at half
  size; the full frame, the scores and the close-up follow when the gesture pauses.
- **The regions** say how many are within tolerance, and for each: ✓ within, ▲ brighter,
  ▼ darker, ● off in colour; the brightness difference *on screen* (display values after the
  tone map, not light: +40% reads 40% brighter); and a colour word. Pick one (here or on the
  picture) to see it close up -- the same row a sheet exports -- and the lights that own it.
- **Solve** proposes the weight of one layer (the region's owner, first in the list) that
  matches the region's brightness; **Fit** proposes weights for the layers ticked F, scored on
  the selected region (or all), staying close to where they are and stopping at *good enough*
  (its option box). Fitting colour temperature works one light at a time -- several at once
  trade one colour for another. A proposal shows its weights as markers beside the current ones
  and changes nothing until **Accept** (one undo step); any edit discards it.
- **Render in WebXR** renders the mix in the real viewer -- highlights and a specular scale
  included -- and shows it as *WebXR frame*. The first render starts a fresh capture service
  for this panel (minutes, while the template loads) on a port the OS picks; it is never one
  already running, and it ends with the panel.
- **Update layers (N)** appears when the session names a bake worker: N layers were baked from
  an older save of the scene. It bakes them again from what the store recorded (their payloads,
  settings and state) in a Maya it starts minimized, which closes a few minutes after the last
  layer, then captures them; the mix and the undo history stay.
- **Save to Rig Library** writes the mix as a light rig into the library every host shares;
  apply it in Maya's *Light Rigs* panel to build and bake it.
- **Edit regions** turns the regions pane into an editor: draw a rectangle, an ellipse, a
  polygon (a click per corner, double-click or Enter to close) or a line on the picture and name
  it; double-click a region's name or weight in the table to change it; delete the selected one.
  **Paint...** writes the reference's area regions as a marker layer (one colour each) with a
  guide, for any image editor: repaint the layer, save it transparent, and **Reload**. Every
  edit is saved to the session's target at once and scored.

### What the preview can show

Held against the WebXR viewer on a production scene, a mix's preview is within about 1% on
matte surfaces. Metal reads 10-15% darker in the preview: the viewer normalizes a baked
reflection per texel by its lightmap, which is not linear in a sum of layers. A layer's specular
scale changes only what ships for the highlights, so it shows in a WebXR render, not the
preview. The trust line under the picture says so, and says when a WebXR frame was rendered
from an earlier mix, when layers come from an older save of the scene, and when the template
GLB is older than the scene.

## Rigs, fixtures and the library

`LookMatch.to_rig` writes a `LightRigPayload` and validates it: lights in the host-neutral
spelling (`spot`, `cone`, `radius`...), overrides on nodes the rig does not own, and **fixtures**
(`LightRigFixture`) -- a light fixture in the scene and which of three ways lights the room from
it, exactly one at a time:

| Mode | The camera sees its emission | Its emission lights the room | Its lamps light the room |
|---|---|---|---|
| `glow` | yes | no | no |
| `emissive` | yes | yes | no |
| `lamp` | yes | no | yes, each at its `lamps` intensity |

A variant made in a mode other than glow says which (`MOODY_LAMP`, `MOODY_EMISSIVE`). mayatk's
`LightRig` sets a fixture's mode as ordinary overrides on its meshes (diffuse visibility) and
lamps (intensity), so apply, compare, swap back and undo treat them like any other.

`LightRigPayload.library()` is the rig library: one per user, unnamespaced, every host's and
tool's. Rigs saved where mayatk kept them before (`<config>/mayatk/light_rigs`) are copied in
on first use; the old folder stays.

## Baking layers

mayatk's `LightLayerBaker` bakes each layer alone, white, at one gain, in a worker -- a fresh,
minimized GUI Maya (`python -m mayatk.light_utils.light_layer_baker launch --queue Q --scene S`)
fed jobs through a `JobQueue` folder; no command port, and the scene is never saved. Each bake
folder's `layer.json` records the save of the scene it was baked from (`LightLayerScene`); the
worker opens its scene again before a job when the file was saved since. A caller that leases
the queue (`JobQueue.lease`) has the worker end once the lease lapses with nothing queued.

## Command line

```text
python -m pythontk.core_utils.engines.lighting.look_match score|sheet|fit|rig|layers|readout|owners|solve|tiles session.json ...
python -m pythontk.core_utils.engines.lighting.look_match.loop iterate|status session.json --rig rig.json ...
python -m pythontk.net_utils.preview.look_capture serve|exact|capture session.json ...
```
