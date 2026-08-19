# Live WebXR preview

**Select geometry in Maya or Blender, press one button, and see it in a headset — then keep
pressing it.** The page does not reload, is not re-opened, and needs no interaction: a tab left
open in a PC-tethered headset picks up each new push on its next poll, so the loop is *change the
scene, push, look up*.

This is the shared preview pipeline: how the deliverable is built, what survives the trip, and
which parts are load-bearing when it looks wrong.

**Nav**: [← pythontk docs](README.md) · Related: [mayatk scene data nodes](https://github.com/m3trik/mayatk/blob/main/docs/data_nodes.md)

---

## The one-paragraph description

> A live glTF preview bridge. The DCC exports the selection to FBX, `MeshConvert` converts it to
> GLB and repairs the channels FBX translation loses, a loopback HTTP server publishes it under a
> version number, and a bundled three.js page polls that version and hot-swaps the model. Because
> the server binds `127.0.0.1` it is a *secure context*, which is what makes `navigator.xr`
> available — so the same page is an orbit-controls preview on a desktop and a real `immersive-vr`
> session with a headset attached. Baked lightmaps ride along in-band, so a baked scene previews
> lit with no extra step.

The design rule worth repeating to devs: **the deliverable is self-describing.** Everything the
viewer needs — the lightmap manifest, the scene sidecar, what the sidecar changed — is embedded in
the GLB's own `extras`. There are no side files to keep together, and no viewer-specific data
format: a third-party glTF tool opens it and gets a sane, if plainer, result.

## The chain

```
  Maya / Blender selection
        |
        |  MayaExportMixin / BlenderExportMixin      (host-specific: read selection, export FBX)
        v
  FBX  ---------------------------------------------  carries geometry, UVs (incl. lightmap UV),
        |                                             materials, embedded textures, and the
        |                                             `data_export` node's user properties
        |  MeshConvert.fbx_to_glb  ->  FBX2glTF --binary --user-properties
        v
  GLB   (raw conversion)
        |
        |  PreviewDeliverer.EDIT_PASSES   (one open edit session, in this order)
        |    scene_sidecar    repair base colour / emissive / metallic-roughness
        |    prune_textures   drop images no material samples (Maya's env cubes)
        |    lightmaps        bind baked maps + write the `lightmap_web` manifest
        |  PreviewDeliverer.FILE_PASSES   (the closed file)
        |    optimize_textures  resize to 2048, re-encode WebP (or KTX2/basis, opt-in),
        |                       repack the BIN chunk
        v
  GLB   (deliverable)  --  PreviewServer.publish()  ->  version += 1
        |
        |  page polls /manifest.json every 1s, reloads only when `version` changes
        |  and imports whatever `scripts` the manifest names
        v
  three.js viewer  (localhost => secure context => WebXR)
        |
        |  reads `handoff.rendering` out of the GLB it just loaded
        v
  the lighting the asset was signed off in
```

Ownership, because it decides where a fix goes:

| Layer | Owns |
|---|---|
| `pythontk.PreviewServer` | the loopback server, `/manifest.json` versioning, viewer liveness, materializing the page **and the active viewer scripts** |
| `pythontk.PreviewDeliverer` | FBX → GLB → publish, and the ordered **pass registry** (`EDIT_PASSES` / `FILE_PASSES`) that runs between them |
| `pythontk.PreviewBridge` | the glTF-appropriate export defaults and the `push()` / `url` / `stop()` surface |
| `pythontk.MeshConvert` | every GLB edit, the sidecar envelope schema, the lightmap binding, **the published rendering policy** |
| `preview_viewer.html` | rebinding the carrier slot to a real `lightMap`, scale/framing, and **spending** the rendering policy it reads out of the file |
| `preview_scripts/*.js` | optional behaviour the page gains by activation, never by being edited |
| `mayatk` / `blendertk` | reading the host's selection, exporting the FBX, reading scene state |

Both DCC bridges are under 80 lines, most of that docstring. Everything else is shared, because
mayatk and blendertk cannot import each other and anything written twice drifts twice.

## Using it

```python
import mayatk as mtk
preview = mtk.WebXrPreview()
preview.push()      # first call opens a tab; later calls swap the model in the open one
preview.url         # the localhost URL — paste into the headset's browser
preview.stop()      # release the port
```

`btk.WebXrPreview` is the Blender twin, same surface. One `PreviewDeliverer` is shared per class,
so the port — and the tab pointed at it — survives across pushes and panel reopens for the life of
the session.

In the page: **Scale** toggles fitted (normalized to 1.5 m) vs. true scale, **Frame** re-frames the
camera, **Light** appears for a lightmapped model and toggles the dim environment on and off
(`r` / `f` / `l`). Fitted mode exists because exported units are rarely metres — a centimetre scene
arrives 100× too large, and "my model is invisible because I am standing inside it" is the most
common first-run failure.

Export defaults (`PreviewBridge.params_defaults`): materials on, **textures embedded** (the browser
can only fetch what the server hosts, so a path-referencing FBX previews with every map missing),
animation off, sidecar on, and **triangulation off** — Maya's FBX exporter refuses triangulation
combined with smoothing groups, and the converter triangulates on the way to glTF anyway.

## Extending it

Two seams, and the rule for choosing is where the work happens: **in the delivery** (a pass) or
**in the page** (a script). Both are registries, so extending means an entry plus a file — never an
edit to the path every DCC bridge already runs through.

### Passes — work on the deliverable

`PreviewDeliverer` runs its post-conversion work as an ordered registry, `name → method`:

```python
class DracoPreview(ptk.PreviewDeliverer):
    FILE_PASSES = {**ptk.PreviewDeliverer.FILE_PASSES, "draco": "_pass_draco"}

    def _pass_draco(self, context):
        encode(context.glb)          # context: .glb .edit .payload .request .results .logger
```

`EDIT_PASSES` run inside **one** open GLB edit session (sidecar → prune → lightmaps); `FILE_PASSES`
run on the **closed** file (optimize). The split is real rather than stylistic: a file pass rewrites
the container — repacking the BIN chunk, re-encoding payloads — which is exactly what an open edit
session cannot have happening underneath it, and `context.edit` is `None` there so a stale handle
fails loudly instead of writing through dead buffers.

Each pass is guarded **individually**. A deliverable missing one repair still beats no deliverable,
and the alternative failed in the worst direction: one early failure took the lightmap wiring down
with it, so the model arrived unlit with nothing naming the pass that actually broke.

### Scripts — work in the page

The viewer page is the stable path. It gains behaviour when the server *activates* an ES module,
which the page imports and calls once with its own API object:

```python
bridge.push(scripts=["turntable", "inspect"])            # this push only
bridge.push()                                            # leaves whatever is active alone
bridge.push(scripts=[])                                  # clears them
bridge.push(scripts={"mine": "C:/tools/overlay.js"})     # your own module, nothing vendored
```

The mapping form is the one to reach for from a DCC: a bridge creates its server lazily on the
first delivery, so there is no `PreviewServer` to call `add_script()` on until a push has happened.
`add_script` / `remove_script` / `set_scripts` are for code that owns a server directly (a test, a
tool serving a GLB it produced itself), and what they register persists across pushes.

`scripts=None` (the default) deliberately means *leave the server's set alone* rather than *use the
default* — the server outlives every push, so a script registered once must not be dropped by the
next push that simply says nothing about scripts. An explicit `[]` is still an instruction.

A module's default export receives the viewer API: `THREE`, `scene`, `renderer`, `camera`,
`controls`, `pivot`, `model`, `bounds`, `policy`, `setStatus`, `addButton(label, onClick)`, and
`on(event, fn)` for `'load'` / `'frame'` / `'key'`. A script that throws is logged and contained —
an optional module must never make a good preview *look* broken, because the one place this is read
is a headset where the console is not visible.

Two ship in the box: **`turntable`** (hands-free rotation, on the pivot so it survives a push) and
**`inspect`** (draw calls, materials and *decoded* texture memory read off the renderer — the two
numbers a GLB's size does not tell you).

## Does WebXR use OpenGL?

Effectively yes: WebXR renders through **WebGL 2, which is an OpenGL ES 3.0 profile**. That settles
the normal-map convention — **+Y green (OpenGL), which is also what glTF mandates**. A DirectX
(-Y) normal map previews with its lighting inverted on every sloped surface. The maps in a
correct set are named accordingly (`*_Normal_OpenGL`, `*_NRML_OGL`).

## What travels, channel by channel

| glTF slot | Source | Notes |
|---|---|---|
| `baseColorTexture` | FBX, or the sidecar's `base_color` | A packed `Albedo_Transparency` map passes through **as-is** — its RGB+A layout already *is* glTF's base-colour layout |
| `normalTexture` | FBX, `texCoord` 0 | Wired by the converter; nothing repairs it because nothing loses it |
| `metallicRoughnessTexture` | the sidecar's `metallic_roughness`, repacked | glTF ORM: **R=occlusion, G=roughness, B=metallic** |
| `emissiveTexture` / factor | FBX, or the sidecar's `emissive` | Emission weight folded in; magnitude above 1 preserved via `KHR_materials_emissive_strength` |
| `occlusionTexture` | the packed ORM (`texCoord` 0), displaced by **the lightmap** (`texCoord` 1) on baked materials | glTF has no lightmap slot; see below |

### Normal maps are wired

Measured on a production interior scene: **54 of 57 materials carry a `normalTexture` at
`texCoord` 0**, all pointing at OpenGL-convention maps. The three without simply have no normal map
in their source set. Every primitive ships `TEXCOORD_0` and `TEXCOORD_1`.

There is no `TANGENT` attribute, and that is fine — three.js derives the tangent frame from
screen-space derivatives when one is absent.

### MSAO / mask maps work; you do not have to author ORM

glTF only understands ORM, but the writer does not require it. `MapFactory.pack_orm_texture`
**decomposes any registered packing** — ORM, MRAO, MSAO, Metallic\_Smoothness,
Albedo\_Transparency — into channels and repacks to the glTF convention, converting smoothness to
roughness (`1 - x`) on the way. So an HDRP mask map (R=metallic, G=AO, B=detail, A=smoothness)
delivers correct metallic *and* roughness.

This matters because getting it wrong is catastrophic and silent. When the converter cannot resolve
the real maps it writes a **solid-white ORM**, and glTF reads metallic from blue — so the whole
scene becomes `metallic=1`. A fully metallic surface has no diffuse response, a lightmap
contributes only to diffuse, and a lightmapped viewer turns its own lights off: three correct
behaviours compounding into a black room, with nothing naming the lost roughness map. That is the
failure the sidecar's `metallic_roughness` section exists to prevent.

Repacking is still reconstruction, not authoring, so the pass logs one highlighted summary naming
the counts (`2 MSAO maps`, …). The scene exporters also offer a **texture-template** selector that
converts the source set to a chosen registry workflow up front and then gates on what did not
convert — the same registry definition drives both the conversion and the check.

## Lightmaps

### How they are carried

glTF 2.0 has **no lightmap slot**, so a baked map travels disguised as a real one:

1. **Bake** (in the DCC) commits to the *scene*: per-object markers plus a `lightmap_metadata`
   manifest on the shared `data_export` carrier node. Platform-agnostic — the Unity/FBX path and
   the Maya round trip read the same commitment.
2. **Carry.** Maya's FBX exporter writes that manifest as an FBX user property; `FBX2glTF
   --user-properties` transcribes it into node `extras`. So the manifest arrives *inside the GLB*
   and the deliverable feeds its own repair — callers pass nothing.
3. **Encode.** A lightmap's dynamic range does not fit in a PNG (a real interior bake measured
   0 → 129.8, mean 0.33 — emissive fixtures against a dim room). Clamping at 1.0 blows out the
   fixtures and crushes everything else, so the encoder divides by the **99.5th percentile**,
   records that divisor, and lets the few texels above it clip — they are the light sources.
4. **Bind.** The PNG is embedded and set as `occlusionTexture` on **`TEXCOORD_1`**, with a
   `lightmap_web` manifest in the root `extras` naming which materials wear the disguise and each
   one's divisor.
5. **Rebind.** The viewer moves the texture to a real `material.lightMap`, sets
   `lightMapIntensity` to the divisor, applies the colour space the manifest declares (`srgb` from
   both producers — the loader treats occlusion as linear *data*, and left linear the map renders
   far too dark), points it at UV2, and clears the slot it arrived in.

Occlusion is the carrier rather than emissive because it leaves the emissive slot free for the
scene's authored emissive map — which matters precisely when the light sources *are* emissive
geometry — and because it **degrades sanely**: a third-party viewer applies it as grey AO rather
than showing nothing.

Per-instance atlas rects (one object's patch of a shared atlas) cannot bind on a shared material,
so they ride a **material clone carrying `KHR_texture_transform`** — pure JSON referencing the same
accessors and the same embedded texture, so any compliant viewer renders the rect with no custom
code.

### Why a lightmap and not a fused unlit bake?

The alternative — bake albedo × lighting into `baseColorTexture` and ship
`KHR_materials_unlit` — is cheaper, more portable, and genuinely the right answer for some
deliverables. Both bake tools once had that level and **removed it deliberately**. One question
decides which applies:

> **Does every surface have its own unique albedo texels?**

- **No** — tiling, instancing, shared materials, i.e. most environment art → **lightmap.** Fusing
  is structurally impossible without an albedo explosion: an instance sharing a material would need
  its own copy of a map it currently shares.
- **Yes** — uniquely unwrapped, non-instanced → **fused is viable and often preferable.**

The reason is *frequency*, not texture count. Irradiance is smooth and cheap: a whole environment's
lighting fits one atlas whose canvas **is** the bake resolution, so the bake tier alone sets that
budget. Albedo carries text, decals, seams and grain, wants 2048² per material, and is reused across
many objects. Separate maps let each be sampled at its natural frequency and multiplied at runtime;
fusing forces one resolution and one UV layout on both, so you either pay albedo density for
lighting or accept lighting density for albedo.

Three secondary advantages, all load-bearing here: normal / roughness / specular response survives
(a fused unlit asset cannot show a normal map at all, by construction); the HDR range survives, via
the divisor described above, where an 8-bit fused base colour would have to tonemap it away
permanently; and the albedo stays shareable across instances.

**Where fused wins** and is worth reaching for: standalone mobile targets where fragment cost is the
bottleneck (unlit does no lighting maths at all); maximum portability (one texture per material, no
second UV set, no rebind, no manifest — every viewer renders it identically); and texture memory,
where collapsing normal + ORM + lightmap to one map is the largest single reduction available.
Architectural walkthroughs and product turntables are usually fused for exactly these reasons.

It is not a binary, either. The usual middle ground is baked lighting for static geometry plus some
cheap analytic term for everything it cannot cover — here that is the dimmed environment described
in the next section, which is what carries specular and normal response rather than a probe system
(there is none). Other points on the spectrum: fuse *indirect* only and keep direct lighting
dynamic, or vertex-bake the low-frequency term.

### Are they combined with the viewer's default lighting?

**Partly, and the details are load-bearing.**

- **The key light goes off.** A baked scene already contains its diffuse lighting. A second
  directional rig leaves the baked shadows in place while everything around them lifts, which reads
  as a washed-out model rather than as double lighting — easy to misdiagnose as a bad bake.
- **The environment stays on, dimmed to 25%.** This is deliberate and it is the fix for a real bug.
  three.js adds lightmap irradiance through `BRDF_Lambert`, which has **no normal term** — a bake
  supplies light that does not vary with the surface normal at all. Switch the viewer's own lighting
  fully off and *nothing left in the render samples the normal*: every correctly-bound normal map,
  all roughness variation and every specular highlight go inert, and the model reads dead flat.
  (Measured: 51 of 57 materials lightmapped, 54 normal maps correctly bound, no surface detail
  visible anywhere.) The environment is the right thing to keep because it is omnidirectional — it
  cannot contradict the bake's shadow *direction* the way a key light would — while its view- and
  normal-dependent **specular** term (`getIBLRadiance`, which takes the normal) is exactly what
  makes a normal map legible.
- The dimming is **per material**, so a partly-baked scene stops under-lighting its un-baked props.
  Only a lightmapped material already contains its diffuse lighting and wants the viewer's own light
  held back; a prop carrying no bake is an ordinary PBR surface and gets the **full** environment.
  This matters because scenes are routinely partly baked — measured on a production room, 51 of 57
  materials — so a scene-wide dim left every un-baked prop cooler than it should be for a reason
  invisible from the model.

  The obstacle was real and is solved rather than accepted: three.js overwrites a *material's*
  `envMapIntensity` from the scene value for any `MeshStandardMaterial` whose `envMap` is `null` —
  which is every material GLTFLoader produces — so setting the property alone does nothing. The
  opt-out is that `envMap === null` clause, so a baked material is handed the **shared** session
  environment as its own `envMap`. That costs no extra upload (it is the texture the scene was
  already lighting with) and stops the override. `disposeModel` is guarded to spare
  `scene.environment` accordingly: it frees every texture-valued property of every material, and
  without the guard the second push would render unlit — a failure no first-push check would catch.

The **Light** button toggles between `bake + env` and `bake only`. That comparison is the one that
tells a flat-looking model caused by a bad bake apart from one caused by the viewer's lighting
policy.

### Are they resource intensive?

**At runtime, no — they are a rounding error.** On the same scene: 5 unique lightmaps totalling
**0.12 MB of a 5.97 MB texture payload**. They are also exempt from the downsizing pass (the bake
sized them deliberately) and re-encode **lossless** — lossy WebP's 4:2:0 chroma blotches
magenta/green on near-black texels and smears colour across atlas rect borders.

The cost is **offline**: the bake itself, in the DCC, at production sample counts. That is the
minutes-to-hours step, which is exactly why the bake commits to the scene and every consumer reads
that one commitment rather than re-baking.

Sampling cost is one extra texture fetch on UV2 and one multiply — cheaper than the analytic lights
it replaces.

### Do we retain the original AO maps?

**Yes — with one deliberate ownership rule for the occlusion slot.**

- The AO map is packed into **R of the ORM** and the packed image is bound as `occlusionTexture`
  too — the spec's own packed-ORM idiom (same image, both slots; glTF reads occlusion *only* from
  that slot, so an unbound R channel would be dead payload). A separate hand-authored AO map
  already sitting in the slot is never displaced by the ORM writer.
- On a **lightmapped** material the lightmap then *takes* the slot. This is the right trade: a bake
  already contains occlusion, computed with real bounce, so re-multiplying a separate AO map on top
  would double-darken every crevice. Displacing the ORM binding is silent (its R channel is the
  same AO the bake supersedes — recognised by the shared texture index); displacing a genuinely
  authored AO map warns.
- That claim is **gated**: `apply_glb_lightmaps(..., replace_authored=False)` keeps an authored
  occlusion (or emissive-carrier) map and stands the lightmap down for that material instead —
  when the authored map is the one you need to review.

So un-baked materials sample their authored AO through the ORM, and baked materials get the
bake's own occlusion — nothing authored is lost, and which one wins is a parameter.

## The scene sidecar

FBX translation silently drops or mistranslates parts of a modern shader. The sidecar is the
envelope that carries the repairs — read read-only from the live scene at push time, applied to the
GLB after conversion, and **embedded in the GLB's own `extras`**.

```json
{
  "version": 2,
  "source": {"application": "maya", "version": "2025"},
  "asset": "<payload basename>",
  "color_space": "linear",
  "sections": {
    "base_color":         {"<material>": {"color": [r, g, b], "texture": "<path>"}},
    "emissive":           {"<material>": {"color": [r, g, b], "texture": "<path>"}},
    "metallic_roughness": {"<material>": {"metallic": "<path>", "roughness": "<path>",
                                          "occlusion": "<path>"}}
  },
  "handoff":  {"instructions": "<the standalone-reader contract>",
               "reads": {"extras.<key>": "<what it holds>"},
               "sections": ["base_color", "emissive", "metallic_roughness"]},
  "textures": {"<path>": {"image": 7, "sha256": "…", "bytes": 262144,
                          "mimeType": "image/webp"}},
  "validate": {"sections": {"base_color": 9, "metallic_roughness": 8}, "textures": 17}
}
```

Alongside it the applier writes `extras["scene_sidecar_applied"]` — the per-section outcome
(`"9 of 9"`, `"0 of 8 matched"`, `"failed (...)"`). So the artifact records both *what the scene
authored* and *what this pass did about it*, which is the difference between "my emissive is
missing" and "the channel was never read".

**Paths are provenance; `textures` is the reference.** A section names each texture by its
authoring-machine path, which will not resolve anywhere else — so the envelope carries the map from
that name to the glTF `images` index actually holding those bytes, plus their **sha256**. That is
the OCI/Docker idea applied here: reference content by digest, not by location. Several entries can
resolve to one image, which is the truth — a metallic/roughness/occlusion trio is repacked into a
single ORM. The digest is re-stamped by whichever pass last wrote the payloads (the texture
optimizer re-encodes everything, so a digest taken at apply time would describe bytes the delivered
file no longer holds).

`validate` records what the envelope itself claims — entries per section, and how many texture
references resolve — deliberately *not* the file's total image count: later passes add images (the
lightmap applier runs after the sidecar in every production path), so a total would be stale on
arrival and a reader checking it would reject a perfectly good deliverable. The per-reference
sha256 is what verifies payloads.

`handoff.instructions` is the reading contract *as data in the artifact* rather than prose in a doc
the recipient was never given — a rule that only exists in documentation is not part of the
hand-off. It states what the file is, that section paths are provenance only, how to resolve them,
and what a lightmap in the occlusion slot means. It is deliberately declarative about the file's own
structure and contains no directives, so an agent can read it as untrusted content safely.

Why these three sections: `aiStandardSurface` / `standardSurface` reach the GLB with
`baseColorFactor` flat `[1,1,1,1]` (Maya's exporter does not map them), emission is gated behind a
separate scalar that legacy shaders do not have, and the metallic/roughness **maps** are lost
outright. Legacy models (`lambert`/`blinn`/`phong`) are deliberately *left alone* — the exporter
already folds Maya's `diffuse` weight in, and re-asserting the raw colour would preview brighter
than the FBX intends.

**Scope boundary, worth stating to devs:** the sidecar is *not* a second metadata channel. Tool-
authored semantic metadata (shots, audio events, lightmap manifests) rides **inside** the FBX on the
`data_export` carrier. The sidecar carries only repairs for what the FBX *format* mistranslates
about the scene's literal content. One home per section per deliverable.

### How useful is the GLB + sidecar as a handoff on its own?

**Fully standalone.** Reading it back needs no side files and no pythontk:

```python
sidecar = ptk.MeshConvert.read_scene_sidecar("scene.glb")   # or any glTF JSON reader

print(sidecar["handoff"]["instructions"])                   # how to read the rest
ref = sidecar["textures"][sidecar["sections"]["base_color"]["MAT_x"]["texture"]]
image = gltf["images"][ref["image"]]                        # the bytes that path became
assert hashlib.sha256(payload).hexdigest() == ref["sha256"] # …and they are intact
```

**What a dev — or an agent — holding only the GLB gets:** a complete, self-describing,
standards-compliant scene (geometry, materials, embedded WebP textures, per-instance atlas
transforms as standard `KHR_texture_transform`), the three `extras` manifests, an embedded
statement of how to read them, a content-addressed resolution for every texture the sidecar names,
and integrity counts to check against. Every viewer-specific behaviour degrades rather than breaks:
without the `lightmap_web` rebind the lightmap still applies as grey occlusion.

**What it still is not:** the *source scene*. Shader graphs, modifiers, rigs and history do not
travel — the sidecar describes the material state that reached the deliverable, not how it was
authored. Colours are glTF-convention **linear**, and the envelope is a versioned contract
(`version: 2`) a reader should check rather than assume.

So: a genuine standalone hand-off and audit trail, not a substitute for the DCC file.

## Cost and budget

Timings, measured end to end on a 231 MB production FBX carrying 224 MB of embedded PNG:

| Stage | Cost |
|---|---|
| FBX2glTF conversion | **44–57 s** — ~80% of the wall clock, and noisy run to run |
| Sidecar + lightmap wiring | 8.7 s |
| Texture optimize | 31.8 s → **7.1 s** (parallelized; byte-identical output) |
| Publish | 0.03 s |

The external converter dominates, and it is a third-party binary — so the honest statement is that
the pipeline's own work is now a small fraction of the push.

**Texture budget is the whole file.** Before the optimize pass a delivery measured 94.7 MB, of which
87.8 MB (93%) was uncompressed source PNG — a 24 MB normal map, a 20 MB character texture — against
2.6 MB of geometry. Downsizing to 2048 and re-encoding WebP took that room to **~15 MB**.

**The remaining constraint is GPU memory, not download — and KTX2 mode is the fix.** Probed on a
delivered preview GLB (a different, larger selection — 57 materials, 514 primitives): **9.5 MB on
the wire, 5.97 MB of it images, which decode to ~555 MB of RGBA and ~740 MB with mipmaps.** Every
one of its 38 images is 2048², because `max_size` is a per-image ceiling and nothing budgets the
total. On a headset that, not the download, is what limits how large a scene can be previewed.

The fix is opt-in and request-scoped: `bridge.push(texture_format="KTX2")` re-encodes that delivery to
KTX2/Basis (`KHR_texture_basisu`), which the GPU keeps block-compressed — the viewer's `KTX2Loader`
transcodes it to ASTC on a standalone headset, BC7 on desktop. It needs KTX-Software's `toktx` on
the authoring machine (the push raises with the install URL when it is missing, never silently
ships WebP). Codecs are chosen per glTF slot — UASTC for normals and ORM/occlusion data, ETC1S for
base color and emissive — and baked lightmaps deliberately stay on the lossless-WebP path; the
details live on `MeshConvert.optimize_glb_textures`. Per-slot resolution ceilings (a normal map
needs 2048 far more than a mask does) remain the cheaper complementary lever and are still open.

## Gotchas worth knowing

- **The page needs `unpkg.com`.** three.js loads from a CDN. Behind a default-deny outbound
  firewall the module never executes — no error event, just a dark page. A classic-script watchdog
  says so after 8 s rather than leaving it silent.
- **WebXR needs localhost or HTTPS.** Opened over a plain-HTTP LAN address, `navigator.xr` is
  absent and the VR button simply never appears; the page says which case it is in.
- **A missing `TEXCOORD_1` means no lightmap.** The FBX was exported without the lightmap UV set;
  the applier warns per primitive rather than binding something wrong.
- **Per-object maps on a shared material cannot both ship.** A glTF material carries one lightmap.
  Atlas packing is what normally prevents this; reaching it means one object wears another's
  lighting — which looks like a bad bake. It is warned, loudly.
- **Draco-compressed GLBs do not load** in the bundled viewer (no decoder wired in). Don't pass
  `--draco`. If that changes it should arrive as a *script*, not a viewer edit.
- **A viewer script that names a hook the page does not emit is inert, silently.** Nothing throws;
  the callback simply never fires. `test_preview_server` checks the packaged scripts against the
  page's `emit()` calls, which is the whole of what can be checked without a JS runtime.
- **A script's own errors are logged, not surfaced.** Deliberate — an optional module must not make
  a good preview look broken on a device where the console is invisible — so check the browser
  console when a script seems to do nothing.
- **Namespaces can disagree.** Manifest and export can differ about `NS:leaf` without either being
  wrong, so matching is exact first, then namespace-stripped — but only when the leaf is
  unambiguous. An ambiguous leaf is skipped rather than guessed.
