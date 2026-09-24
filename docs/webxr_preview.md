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
  Maya / Blender selection            (scope: selected / visible / all)
        |
        |  MayaExportMixin / BlenderExportMixin._export_fbx  (the hand-off mixin's
        |  FBX write, inside the export bracket: the visibility and render-effects
        |  producers refresh the data_export carrier, viewport material bindings are
        |  suspended and one curve proxy per keyed channel is staged for the write,
        |  then everything is put back). PreviewBridge attaches the scene sidecar
        |  SceneState read from the same objects.
        v
  FBX   (payload: geometry + embedded textures + the carrier + the curve proxies)
        |
        |  GlbPipeline.build  --  the SAME build TaskManager.create_glb runs for the
        |  Scene Exporter's GLB deliverable; the two callers hand it dials only:
        |    downsize   FbxMedia.downsize the embedded textures to the texture ceiling
        |    convert    MeshConvert.fbx_to_glb: FBX2glTF reads the FBX with its grayscale
        |               embeds expanded (FbxMedia.expand_grayscale -- it packs a gray
        |               roughness/metallic map as white); then alpha repair, image
        |               dedupe, scene sidecar, dead-texture sweep, lightmaps (the host's
        |               live map folders),
        |               shadow rigs, curve-proxy strip, clips, visibility gates, fades,
        |               skin + tangent repairs, animation manifest -- one edit
        |               session, and a report back
        |    reduce     MeshConvert.reduce_glb_animations when a key tolerance is
        |               named: each clip keeps only the keys its interpolation needs
        |    optimize   MeshConvert.optimize_glb_textures, last, on the closed file
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

The preview's job is to show what the target platform will get, so the GLB **content** is never
the preview's own: the sidecar, the lightmaps, the render effects, the texture policy all come out
of `GlbPipeline`, and a fix to any of them lands in the export and the preview at once. Its texture
settings are the Scene Exporter's own rows, resolved by the exporters' own method (see *Preview and
export are one deliverable*). What the preview decides for itself stops at where its scratch files
live, and the fact that it runs **no export tasks or validation checks** -- those
belong to the Scene Exporter, and the preview must stay fast enough to press after every tweak.
When you want the exporter's *whole* run checked, export the GLB and publish that file (see
*Publishing a GLB you already have*).

Ownership, because it decides where a fix goes:

| Layer | Owns |
|---|---|
| `pythontk.PreviewServer` | the loopback server, `/manifest.json` versioning, viewer liveness, materializing the page **and the active viewer scripts**, and the **read-only guest listener** a share fronts |
| `pythontk.ShareTunnel` | a loopback port at a public HTTPS link: the tunnel CLIs (registry), their lifetime, the stable alias -- knows nothing about the preview |
| `pythontk.GlbPipeline` | **the GLB build** -- downsize, convert, optimize, in that order, for the preview AND the Scene Exporters' GLB output |
| `pythontk.PreviewDeliverer` | the build's dials (container, scratch, release) and the publish |
| `pythontk.PreviewBridge` | the glTF-appropriate export defaults, the sidecar attach and the `push()` / `publish_file()` / `url` / `share()` / `stop()` surface |
| `pythontk.MeshConvert` | every GLB edit, the sidecar envelope schema, the lightmap binding, **the published rendering policy** |
| `net_utils/preview/viewer.html` | rebinding the carrier slot to a real `lightMap`, scale/framing, and **spending** the rendering policy it reads out of the file |
| `net_utils/preview/scripts/*.js` | optional behaviour the page gains by activation, never by being edited |
| `mayatk` / `blendertk` | reading the host's selection, exporting the FBX inside the export bracket, reading scene state |

Both DCC bridges are under 80 lines, most of that docstring. Everything else is shared, because
mayatk and blendertk cannot import each other and anything written twice drifts twice.

## Using it

```python
import mayatk as mtk
preview = mtk.WebXrPreview()
preview.push()      # first call opens a tab; later calls swap the model in the open one
preview.url         # the localhost URL — paste into the headset's browser
preview.share()     # a view-only HTTPS link anyone can open -- see *Sharing a link*
preview.stop()      # release the port (and end any share)
```

`btk.WebXrPreview` is the Blender twin, same surface. One `PreviewDeliverer` is shared per class,
so the port — and the tab pointed at it — survives across pushes and panel reopens for the life of
the session.

**Scope** is the ecosystem's, not this bridge's — the same `selected` / `all` / `visible`
vocabulary every other hand-off offers, declared once by `uitk.bridge.Parameters.scope_spec` and
resolved here through the export mixins' host hooks (`_scene_objects` / `_visible_objects`):

```python
preview.push(scope="visible")     # every visible mesh — what the artist actually sees
preview.push(scope="all")         # the whole scene, hidden objects included
preview.scope_objects("visible")  # what that scope resolves to, without pushing
```

`scope_objects` is public because a *caller* needs the answer: pushing blind collapses "nothing
selected", "the scene is empty" and "the export failed" into one failure message, and the first two
are the user's own next action. An unknown scope resolves to the selection — a scope must never
silently *widen* a push.

### Publishing a GLB you already have

```python
preview.publish_file("C:/assets/vendor_chair.glb")   # no host, no export, no conversion
```

The one delivery shape with no DCC in it: an authored `.glb` goes straight to the page, on the
**same** server, port and tab a push already owns — so an outside asset and your own export
alternate in one viewer instead of needing a second one. It is what makes the preview usable as a
plain GLB viewer (check what an exporter actually wrote; compare a vendor's asset against your
push).

Two rules it does not bend. The file is **copied, never moved** — it is your asset, not a scratch
artifact the bridge minted. And it is published **exactly as authored**: no sidecar, no lightmap
wiring, no texture re-encode. Those passes exist to repair what a DCC export loses, and a finished
GLB has already answered them; what you see is the file. A `.gltf` is refused rather than served,
because the server publishes one file into its root and the sibling `.bin` and textures would
simply 404 — an empty scene with nothing to explain it.

In the Rendering panel this is the **External GLB** entry on the WebXR Preview option box's scope
combo. Nothing is wired to the combo itself — pressing the button is what opens the browse
dialog, so the option box can be configured without a modal interrupting it, and so a restored
External scope cannot greet the next panel open with a file browser. It asks per push rather than
remembering: the combo is already sitting on External, so re-picking it could never say "same
source, different file", and a remembered path goes stale on its own. The folder carries over, so
the second ask opens where the first one landed. It lives on the panel rather than in `scope_spec`:
it is a *source*, not a scope, and every other hand-off bridge would inherit a choice it has no way
to honour. The viewer-script rows still apply — those describe the page, and the page is the same
one either source publishes to.

In the page: **Scale** toggles fitted (normalized to 1.5 m) vs. true scale and **Frame** re-frames
the camera (`r` / `f`). A third slot, `#lookdev`, is the area for dials that tune how the model
*reads*: the **Normals** dial (`normalTexture.scale`, saved into the GLB through `POST /settings`).
Each dial hides itself when the model gives it nothing to do, and the area follows them;
`LOOKDEV_ENABLED` holds the whole area back until there is a SET of dials worth a permanent
seat. Adding the next dial is markup inside `#lookdev`, a sync called from `syncLookdev`, and
its element in that function's `controls` list.

Fitted mode exists because exported units are rarely metres — a centimetre scene
arrives 100× too large, and "my model is invisible because I am standing inside it" is the most
common first-run failure.

Export defaults (`PreviewBridge.params_defaults`): materials on, **textures embedded** (the browser
can only fetch what the server hosts, so a path-referencing FBX previews with every map missing),
animation off, sidecar on, and **triangulation off** — Maya's FBX exporter refuses triangulation
combined with smoothing groups, and the converter triangulates on the way to glTF anyway.

## Sharing a link

**Press Share Link, send the link, keep pushing.** The person who built the scene serves it; anyone
with the link opens it in a browser -- desktop, phone, or a standalone headset's browser, which
gets its VR button because the link is HTTPS. The share is **view-only** and **live**: every push
reaches every guest on their next poll, and nothing a guest does writes to this machine. Each
guest downloads the GLB and renders it on their own device; this machine serves files and renders
nothing for anyone. Guests need no install -- only the builder does (a tunnel client).

```python
info = preview.share()               # {"url": the link to send, "guests": 0, ...}
preview.share(provider="tailscale_funnel")
preview.share_url                    # None once the share ends -- or once its tunnel dies
preview.unshare()                    # the owner's page and server are untouched
```

In the WebXR Preview panel: the header menu's **Share Link** (copies the link) and **Stop
Sharing**, the **Share Via** row, and the footer, which shows the link and how many guests are
watching.

### How it works

```
  owner's tab ──► 127.0.0.1:8118  ─┐  the owner listener: every route, as always
                                   ├─ one serve root, one manifest, one version
  guest ──HTTPS──► tunnel ──► 127.0.0.1:<ephemeral>  the GUEST listener: read-only
```

**A second door, and the role is the socket's.** `PreviewServer.share()` opens a second listener
on loopback (`start_guest`) and fronts *that one* with a `ShareTunnel`. Who may write is decided by
which listener a request arrived on -- never by a header or a token a client could claim -- so the
owner's loop is untouched: its tab, its settings saves, its recordings and stills work exactly as
before. The guest listener:

- **reads an allow-list** -- the page, its manifest, and exactly the files that manifest names (the
  asset, the active scripts). A scene push's stills and recordings land in the serve root too, a
  publish passes through a `.part` file, and `scripts/` would list itself; none of it is the share.
  It never lists a folder -- not even `/` on a root that holds no page.
- **refuses every write** (`/settings`, `/snapshot`, `/playblast/*`: 403), and takes no body past
  4 KiB on any route (413, unread): its one write, the close beacon, carries none, and a still's
  64 MB is the owner's. Its manifest says `"guest": true`, and the page hides
  what it cannot do: the owner-only scripts (`PreviewServer.OWNER_SCRIPTS`: `playblast`, `snapshot`)
  are not even served, and the Normals **Save** is hidden. `xrRuntime` is left out -- it describes
  the builder's machine, not the guest's.
- **counts guests per tab** (`guest_count`, from the `?id=` each page polls with), never toward
  `has_viewer` -- a guest watching must not stop the next push from reopening the owner's closed tab.
  A tab's close beacon retires that tab only, and a poll that crosses its own beacon (each request
  has its own thread) is ignored for `CLOSED_LINGER` seconds rather than reviving it for 90.
- **answers only the names admitted**: loopback, plus the tunnel's public name, admitted *before* the
  link is announced. The owner listener never admits a public name, so even a misconfigured tunnel
  pointed at it gets 403 for every route.

**The tunnel is generic.** `pythontk.ShareTunnel` exposes any loopback port at an HTTPS link through a
tunnel CLI and knows nothing about the preview. Its providers are a registry of plain values
(`ShareTunnel.PROVIDERS`: arguments, the pattern the link is printed in, what "ready" looks like,
where to install) -- a provider is an entry, not a branch:

| Provider | Link | Trade-off |
|---|---|---|
| `cloudflared` (default when installed) | random `https://<words>.trycloudflare.com` per share | no account; fast; TLS ends at Cloudflare's edge, which sees the traffic. The panel offers the download (one executable, `AppInstaller`'s `binary` type) |
| `tailscale_funnel` | stable `https://<machine>.<tailnet>.ts.net` | bookmark once; TLS ends on this machine; relayed, so large pushes load slower; Funnel must be enabled for the tailnet |
| `tailscale_serve` | the same name, tailnet only | nothing public; a headset needs the Tailscale app |

`provider=None` takes `PYTHONTK_SHARE_PROVIDER`, else the first installed of `ShareTunnel.PREFERENCE`
(public providers only). A start returns only once guests can actually reach the link: after the
provider's own ready line, and -- for a quick tunnel, whose name is new every time -- after a public
resolver answers it (measured: NXDOMAIN for 6-18 s after the tunnel registers). The resolver is asked
directly (`NetUtils.resolves_publicly`), never through this machine's DNS, because a lookup made too
early caches the miss for minutes -- on the builder's machine and, worse, on a guest's.

**Lifetime.** The client runs through `AppLauncher.spawn`, which on Windows puts it in a kill-on-close
Job Object: it dies with the DCC however the DCC ends, a crash included. An orphaned tunnel would keep
a public link pointed at a port that whatever binds it next would be published through. `stop()`
ends any share first, and `unshare()` / `stop()` landing while a share still waits on its provider
end that one too: its tunnel stops the moment its link arrives, and the share raises. Off Windows a
crash can still orphan the client; a normal exit is covered by `atexit`.

### A link that never changes (the alias)

A quick tunnel's link changes every share, and typing forty characters into a headset is where
sharing dies in practice. Point two environment variables at a page your own web server serves:

```
PYTHONTK_PREVIEW_ALIAS      = me@myserver:/srv/www/vr/index.html   (or a local/UNC path)
PYTHONTK_PREVIEW_ALIAS_URL  = https://example.com/vr
```

Every share rewrites that page to redirect to the live link (over the system `scp`, batch mode --
a missing key fails rather than prompting), and every stop rewrites it to "nothing is being shared",
which reloads itself -- so a guest who bookmarked `https://example.com/vr` in the headset once opens
it, waits, and lands on the next share by itself. The alias address is what `share()` hands out,
but only while its last update landed; a stale alias is never what gets sent (`alias_error` says why).

### First-time setup, and what a failed share says

Each of these fails the share at once with the next step, rather than waiting out a timeout:

- **Tailscale Serve/Funnel not enabled for the tailnet** -- the CLI prints an enable link and waits on
  it; the share fails naming that link. Enabling it is a tailnet admin change.
- **An outbound firewall that denies by default** -- the client dies with WinError 10013; the share
  says the firewall blocked it and names the executable to allow.
- **The CLI is missing** -- the message names the install; the panel offers cloudflared's download
  (`ShareTunnel.settle`, the same shape as the KTX2 encoder's).

What a share does not change: the link **is** the deliverable -- a guest can save the GLB, so stop
sharing when the review is over. The Cloudflare quick tunnel's name is unguessable but not secret
once sent; the Funnel name is stable and guessable. There is no guest authentication; a share is for
people you send the link to.

## Extending it

Two seams, and the rule for choosing is where the work happens: **in the deliverable** (the shared
GLB build) or **in the page** (a script). Neither is an edit to a DCC bridge: the build is one chain
both producers run, and the scripts are a registry the page imports from.

### The build — work on the deliverable

The GLB's content is made by `GlbPipeline.build`, which is the same call `TaskManager.create_glb`
makes in both Scene Exporters. A step that belongs in the deliverable -- a channel repair, a Draco
encode, a per-slot resolution ceiling -- goes into that chain (`MeshConvert.fbx_to_glb`'s edit
session for anything that reads the JSON chunk, the pipeline's stages for anything that rewrites the
container), and the preview shows it on the next push because it never had a chain of its own. The
rule of thumb: if a fix would need to be made twice, it is in the wrong place.

A build can also be shown something the scene does not carry. `push(data_export={...})` overlays
the GLB's in-band channels for that push only (`MeshConvert.overlay_data_export`, applied ahead of
every pass that reads one; `None` clears a channel), so the passes build as if the FBX had carried
it and nothing upstream -- scene or exported file -- is written. `MeshConvert.effect_preview_channels`
states one render effect that way: the Render Effects option boxes' **Preview in WebXR** button
pushes the selection with the fade or pulse at the box's settings -- keys planned by `RampKeys`, the
same plan the key tools write -- with the objects' own tracks replaced and the take list and shot
record cleared, so the effect plays alone on one clip over its own extent. The push's result lists
the channels the overlay replaced under `data_export` (`[]` when none landed -- a finished-GLB
source, or a bridge that predates the knob and swept it into the export bag), and the panel reads
that before telling anyone the page shows the effect.

The three stages have three failure policies, and they are the exporters' policies: the downsize is
a speed win the texture ceiling re-applies anyway, so its failure is a warning; a failed conversion
or texture pass raises and the push reports it, rather than publishing a GLB that quietly skipped
the 94.7 MB -> ~15 MB pass.

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
`controls`, `pivot`, `model`, `bounds`, `policy`, `guest` (true on a view-only share, where every
write is refused -- see *Sharing a link*), `setStatus`, `addButton(label, onClick)`,
`showDialog({title, fields, confirm})`, and `on(event, fn)` for `'load'` / `'frame'` / `'key'`.
A script that writes a file through the server gets the page's half of that too:
`captureSize(maxEdge)` — the size to capture at and the pixel ratio to *render* at for it, clamped
to what the GPU allocates, the one sizing rule a playblast and a still share — `refusal(response)`,
the server's own reason for a refused request, for the status line, and `download(url)`, which
saves a served file rather than navigating the page to it. Hand `download` a path **relative to the
page** (or a route that answers `Content-Disposition: attachment`): a browser honours `download`
only on the page's own origin, and the page is as validly open at `localhost` as at the
`127.0.0.1` the server's `url` spells — an absolute link navigates a `localhost` tab to the file.
`showDialog` is the page's one modal — a script asks for options through it rather than through
`window.confirm` for the reason it adds buttons through `addButton` rather than `createElement`:
one look, one place. Its `fields` are `{key, label, type, value, title}` — `'check'` by default,
`'choice'` for a picker over `choices: [{value, label, title}]`, `'note'` for a read-only line —
and it resolves to `{key: answer}` on confirm (`checked` for a check, the chosen `value` for a
choice) and to **null** on cancel, so a caller branches on the answer rather than on a flag inside
it. While it is up the page behind it is inert: no shortcut fires, and focus cannot leave the
prompt. The transport is on it too: `mixer`, `playClip(name)`, `playing` / `setPlaying(state)`,
`poseAt(seconds)`,
`shotAt(seconds)`, `descriptionAt()` — the shot record's own note for whatever the model is posed
at — and `clip`, the selection as
`{name, duration, fps, startFrame, endFrame, sequence}`, which is what lets a script address a clip
in the AUTHORING frames the DCC and the picker quote rather than in seconds. A script that throws is
logged and contained —
an optional module must never make a good preview *look* broken, because the one place this is read
is a headset where the console is not visible.

Five ship in the box: **`turntable`** (hands-free rotation, on the pivot so it survives a push),
**`inspect`** (draw calls, materials and *decoded* texture memory read off the renderer — the two
numbers a GLB's size does not tell you) and **`shadow_rig`** (the runtime half of the DCC shadow
rigs: reads the `extras.shadow_web` manifest `MeshConvert.apply_glb_shadows` writes during the
conversion, gives every plane one `ShaderMaterial` — projected silhouettes and horizon maps in one
program — batches the projected planes that share an atlas and carry no fade into an
`InstancedMesh`, and re-places each plane from its source and contact nodes every frame with a port
of `ShadowProjection.model`; the contract is `mayatk/docs/shadow_rig_morphing.md`), and
**`snapshot`** (an **Export Image** button: the current view saved as a PNG — see *Exporting a
still* below). `turntable`, `inspect` and `snapshot` are checkboxes on the WebXR Preview option box,
which passes an explicit list every push: the panel
is authoritative, so a script registered on the server by other code is cleared by the next push
from there. **`playblast`** records the clip the transport is on to a movie file (see *Recording a clip*
below). `shadow_rig` and `playblast` are **on by themselves**: `PreviewServer.AUTO_SCRIPTS` maps it to the extras key
each reads, and `publish()` activates it — appended to whatever the push named — for any GLB whose
root extras carry that key (`shadow_web` and `animation_web` respectively; the JSON chunk is probed,
never the geometry). Opt out by removing the registry entry or with `remove_script(name)` after the
push. One caveat of the page's
loading order: scripts and the asset load concurrently and the first `load` is not held for the
imports, so a deliverable small enough to parse before a 40 KB module arrives shows still planes
until the next push (the script says so in the console); a production GLB is never that small.

## Does WebXR use OpenGL?

Effectively yes: WebXR renders through **WebGL 2, which is an OpenGL ES 3.0 profile**. That settles
the normal-map convention — **+Y green (OpenGL), which is also what glTF mandates**. A DirectX
(-Y) normal map previews with its lighting inverted on every sloped surface. The maps in a
correct set are named accordingly (`*_Normal_OpenGL`, `*_NRML_OGL`).

## What travels, channel by channel

| glTF slot | Source | Notes |
|---|---|---|
| `baseColorTexture` | FBX, or the sidecar's `base_color` | A packed `Albedo_Transparency` map passes through **as-is** — its RGB+A layout already *is* glTF's base-colour layout |
| `normalTexture` | FBX, `texCoord` 0 | Wired by the converter; the map itself travels intact, and its `TANGENT` handedness is repaired (below) |
| `metallicRoughnessTexture` | the sidecar's `metallic_roughness`, repacked | glTF ORM: **R=occlusion, G=roughness, B=metallic** |
| `emissiveTexture` / factor | FBX, or the sidecar's `emissive` | Emission weight folded in; magnitude above 1 preserved via `KHR_materials_emissive_strength`. A **highlighted** object's isolated copy drops the map: glTF emission is factor × map, so a mapped material (factor at white) would clamp the additive `highlight` channel to nothing and mask its colour where the map is black — the copy glows from the channel alone |
| `occlusionTexture` | the packed ORM (`texCoord` 0), displaced by **the lightmap** (`texCoord` 1) on baked materials | glTF has no lightmap slot; see below |

### Normal maps are wired

Measured on a production interior scene: **54 of 57 materials carry a `normalTexture` at
`texCoord` 0**, all pointing at OpenGL-convention maps. The three without simply have no normal map
in their source set. Every primitive ships `TEXCOORD_0` and `TEXCOORD_1`.

Every normal-mapped primitive also ships a `TANGENT` (the hand-off mixins pin
`FBXExportTangents` / `use_tspace`): left to the receiver, three.js swaps in a screen-space
derivative frame and flips green to compensate, which other viewers do not, so the same file read
differently in each. One repair rides with it. FBX2glTF carries the FBX's tangents but not its
binormals, where the handedness lives, and writes `w = +1` on every vertex — right on a plain UV
shell, and green-inverted on every **mirrored** one, because three.js trusts a shipped tangent over
its own frame. Measured on the first production push to carry tangents: 7 of 61 primitives, each
wrong on exactly its mirrored triangles (a button and four light fixtures whole, 29% of a table),
reported as "bad normals" on the button — and 8 zero-length tangents on UV slivers the DCC could
not orient, which glTF forbids and three.js turns into NaN. `MeshConvert.fix_glb_tangents`
(`GlbTangents.repair`) sets each vertex's `w` from the way its UVs run, the rule Blender's
MikkTSpace glTF export follows, and rebuilds a zero tangent along its UVs, inside the conversion
session both producers share. If a normal map reads inverted on one object only, check whether its
UV shell is mirrored before suspecting the map.

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

### Where the maps are found — and what happens when they are not

The manifest names each map by **basename**; the folder it records is the one the bake was
committed from, and that is history, not a contract (a reorganised project, a scene migrated to
another module, a teammate's machine). The applier resolves each basename against, in order: the
manifest's recorded folder, the `search_dirs` the host hands it, and the GLB's own folder. Both
DCC hosts pass `LightmapRecords.search_dirs()` — the project's texture folders plus wherever the
bake markers' maps were actually found — so a map that moved into a subfolder still binds. A map
found nowhere is **not guessed at**: its objects are left unbound (they render unlit, exactly like
no bake), the applier logs one line naming the file and the folders searched, and the push result
carries `lightmaps: {"expected", "bound", "unbound": [...]}` which `PreviewBridge.lightmap_summary`
renders for the panel. The fix lives in the DCC, not here: the Texture Path Editor lists lightmap
dependencies beside the textures (Find & Copy relocates them and repoints the markers; Normalize
re-spells the recorded folder workspace-relative so another machine resolves it), and the Scene
Exporter's *Check For Valid Paths* fails on a lightmap the markers name but no folder holds.

Per-instance atlas rects (one object's patch of a shared atlas) cannot bind on a shared material,
so they ride a **material clone carrying `KHR_texture_transform`** — pure JSON referencing the same
accessors and the same embedded texture, so any compliant viewer renders the rect with no custom
code. A map the shared material cannot carry takes the same route: when objects baked into
DIFFERENT maps share a material (a secondary material on two machine bodies, or a Per-Object bake
of instances), each object after the first binds its own clone.

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
cheap analytic term for everything it cannot cover — here that is the environment's specular term
described in the next section, which plays the reflection-probe role (there is no probe system) and
is what carries specular and normal response. Other points on the spectrum: fuse *indirect* only and
keep direct lighting dynamic, or vertex-bake the low-frequency term.

### Are they combined with the viewer's default lighting?

**Only where the bake has nothing to say, and the details are load-bearing.**

- **The key light goes off.** A baked scene already contains its diffuse lighting. A second
  directional rig leaves the baked shadows in place while everything around them lifts, which reads
  as a washed-out model rather than as double lighting — easy to misdiagnose as a bad bake. It is
  scene-wide (it cannot be withheld per material without render layers), so it goes off the moment
  *anything* is baked; a room is routinely partly baked (measured: 51 of 57 materials).
- **The environment stays at full strength, and a baked material takes only its specular.** A
  lightmap holds the surface's diffuse lighting, every light and the sky included, so adding the
  environment's irradiance on top double-counts it: measured on a production office, a quarter of
  the environment on top of the bake doubled every shadow and lifted every surface — the washed-out
  look that gets reported as "blown out". Switching the environment off instead renders the model
  dead flat: three.js adds lightmap irradiance through `BRDF_Lambert`, which has **no normal
  term**, so with the environment gone *nothing left in the render samples the normal* and every
  normal map, roughness map and specular highlight goes inert (measured: 54 normal maps correctly
  bound, no surface detail visible anywhere). So on a baked material the viewer drops the one line
  of three.js's `lights_fragment_maps` chunk that adds the environment's irradiance and keeps the
  specular term — the role a reflection probe plays for a lightmapped surface in a runtime, and
  what Unity's native lightmap path does with a lightmapped renderer (no ambient, probes for
  specular). It is a shader rule rather than a material setting because three.js scales the
  environment's diffuse and specular with one number (`envMapIntensity`) and no glTF-level
  setting can express the split, which is why the published policy states it as
  `lightmappedMaterials.envMapTerms: "specular"` for a recipient to act on.
- **A baked normal map is relieved along the key light.** Specular alone barely moves a matte
  baked surface (measured: `normalScale` 1 → 4 moved ~1% of pixels by ~0.4/255), so the page
  also scales each baked texel by how the normal-mapped normal faces the key light's direction
  against how the surface's own normal does (half-Lambert) — from the surface's own side of its
  plane: where a surface faces away from the key, the direction is mirrored across that plane
  first, because a bake's light reached it from its front. A flat map is then exactly the pure
  bake on every surface, and a bump reads the same on a ceiling as on the floor. Unmirrored (as
  it first shipped) a face turned straight away from the key rendered black, and one bump moved
  a ceiling 40 levels where it moved the floor 6 — a lighting artefact that reads as broken
  normals. The key light's intensity is off on a baked model; its direction still orients this,
  and the published policy states the rule as `lightmappedMaterials.normalRelief`.
- **A baked material reflects at the level the export chose.** The environment is a bright studio,
  not the room the bake lit, so at full strength its reflections lift every dark glossy baked
  surface (measured on a production room: the darkest machine surfaces at 0.06 of display baked
  alone, 0.22 with full reflections, 0.11 at a quarter). The Scene Exporter's **Baked Reflections**
  row sets the level -- Off (the pure bake), Quarter (the default), Half, Full -- and the deliverable
  publishes it as `lightmappedMaterials.envMapIntensity`; the page scales everything the
  environment gives a baked material by it, with one uniform the baked shaders share.
- **Un-baked materials in the same asset are ordinary PBR surfaces** and take the full environment,
  diffuse and specular.

There is no page-local lighting mode: the deliverable's own `handoff.rendering` is the one rig, and
this page is one of its readers. The earlier **Light** toggle (`bake only` / `bake + env`) existed
to tell a flat bake from the viewer's lighting; with the environment's diffuse withheld from baked
materials that question no longer arises, and a mode the export cannot express is a preview that
lies about the deliverable.

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

## Preview and export are one deliverable

The preview is the approval gate, so the asset a developer receives has to be the asset the artist
signed off. That is a *shared definition*, not a convention: `MeshConvert.web_delivery_texture_params`
is the one statement of what a web deliverable's textures are (`WEB_DELIVERY_FORMAT`,
`WEB_DELIVERY_MAX_SIZE`), and the Scene Exporter's texture rows **override** it — Texture File Type
the container, Optimize Textures the ceiling, Secondary Map Size and KTX2 RDO the two GLB-only
levers — rather than being the only thing that turns the pass on.

**The preview offers those same rows, and resolves them with the same methods.**
`ExportProfile.GLB_ROWS` declares the rows that decide a GLB deliverable -- its images
(`GLB_TEXTURE_ROWS`) and the lighting recipe it publishes (`GLB_LIGHTING_ROWS`: Baked
Reflections). `ExportRun.glb_texture_params` resolves the texture rows against the policy and
`ExportRun.rendering` the lighting ones, and both Scene Exporters and the preview call them (each
exporter used to carry a private copy of the texture half, and the preview none: it named a
container and inherited the web ceiling, so every push was cut to 2048 px whatever the export was
set to). The WebXR Preview panel's **Textures** and **Lighting** rows are built from the same
tables with the same labels and defaults (`ExportProfile.glb_options`, `glb_defaults`), and a push
hands their values to the deliverer as `glb_options`, keyed as the export button keys them:

```python
preview.push(glb_options={"optimize_textures": 4096, "texture_file_type": "ktx2"})
```

Set the two panels the same and the GLB texture pass is the same in both: same container, same
ceilings. An export can still do more to its maps BEFORE that pass -- a Texture Template re-authors
the materials, and Optimize Textures corrects each scene map's mode and bit depth -- and those are
export tasks a preview does not run. The panel offers only what a
GLB-only push can honour: the containers a GLB can carry (a TGA or EXR choice means the web default
inside a GLB anyway), and no *Template Budget* — that ceiling comes from the export's Texture
Template row, whose material conversion the preview does not run, so pick the template's ceiling
instead. Optimize Textures at **OFF** resizes nothing, in the GLB too: every map keeps its own
resolution, re-encoded to the container. A plain **Optimize** names no ceiling, so the GLB takes
the web ceiling (`WEB_DELIVERY_MAX_SIZE`, 2048 px); *Optimize + Max N* caps at N.

The lighting row rides the same way, into the file rather than the texture pass: the export
publishes its level in the GLB's `handoff.rendering` (and the FBX's handoff record, through its
`ExportContext`), and the page reads it from the GLB it loaded. The preview's Lighting row reaches
the page through the scene sidecar, so it needs Scene Sidecar on, and a file-source push -- which
builds no sidecar -- does not offer it.

The dependency runs one way. An export reads its own panel and the scene; nothing it needs to be
configured or handed off comes from a preview run.

This is worth stating because the alternative was measured. Running both legs over one production
assembly in one Maya session and diffing 24 observable properties of the two GLBs: geometry,
materials, lightmaps and the sidecar matched exactly, and the textures did not — **8.71 MB of WebP
from the preview against 280.13 MB of full-resolution PNG from the exporter**, with nothing in
either log saying so. Setting the exporter's dials to WebP still gave 22.06 MB, because its ceiling
resolved from an absent template budget to "never resample".

After the fix, the same comparison on the same scene reports **full semantic parity — all 24
properties equal** (8.71 MB vs 8.72 MB, 1925 nodes over 537 instanced meshes, 28 WebP images, 47/47
lightmaps, 13 clips opening on `Shot_2`), `verify_glb` returns an identical report for both, and
both render identically in headless WebXR. Two things had to be pinned to get there beyond the
texture policy: the Maya exporter's FBX write inherits **sticky global plugin state** when no preset
names a configuration (factory is instancing OFF, smoothing groups OFF, embedded media OFF — a
clean session would have shipped an untextured, de-instanced GLB), and cameras, which the preview
mixin drops and the factory keeps.

The policy also ships KTX2 **without fallback twins** (`ktx2_fallback=False`). A twin is a PNG/JPEG
copy of a KTX2 image that only a stock glTF importer reads. It used to be each producer's call, the
exporters kept the twins, and a production 4K assembly shipped 145.8 MB of them beside 123.1 MB of
KTX2 — 48% of the GLB. A GLB that must also open in Blender or Unreal asks for the twins
(`ktx2_fallback=True`), which the scene exporters offer as the **KTX2 + PNG/JPEG** Texture File Type.

The other axis to check on a handoff is the **take split**. A scene that declares shots must have
them realized into FBX AnimStacks before conversion, or the deliverable carries one continuous clip
where the preview showed twelve. `apply_glb_animations` warns with both counts when the `fbx_takes`
channel names takes the file has no clips for — that warning is the one to read.

## Watching shots

A deliverable that ships clips grows a picker and a transport. Shots the DCC declared are listed
first, each with its authoring range; a clip marked *(full range)* is the whole timeline the
exporter kept beside them. On a deliverable that ships **only** shots the page rebuilds
`FULL SEQUENCE` from them — every shot placed at its authored frame, not concatenated, so the gaps
between them are real and hold the previous shot's last pose exactly as the whole-timeline clip did.
It is offered first, because "watch the whole thing" is what a reviewer opens.

The readout beside the playhead names **which shot the playhead is standing in** while the sequence
plays, followed by the time and the authoring frame:

```
  SHOT_B · 3.40 / 5.00s  f102          inside a shot
  SHOT_A (hold) · 2.50 / 5.00s  f75    in the gap after it
```

The picker cannot answer that question — it is sitting on `FULL SEQUENCE` the whole way through —
and scrubbing a five-second sequence without it says only how far in you are. A gap is labelled as
a **hold** rather than named outright: the pose on screen is the previous shot's last frame, and
naming that shot plainly would claim it plays through frames it does not cover, which is a bug
report waiting to be filed against a shot that is behaving correctly. A single clip is not labelled
at all, since the picker already names it.

## Recording a clip

**Export Playblast** writes the clip the transport is on to a movie file — one declared shot, the
whole-timeline clip, or `FULL SEQUENCE` (every shot laid back onto the timeline it was cut from),
which records as one continuous movie exactly as it plays. The button appears whenever the
deliverable ships clips, because that is when the clip picker does: `playblast` is an
`AUTO_SCRIPTS` entry keyed on `animation_web`, so there is no checkbox to have forgotten on the push
a reviewer just watched.

**It is a playblast, not a screen recording.** The page *steps* the clip: pose frame N, render,
hand the pixels over, then ask for N+1. A throttled tab, a headset and a desktop therefore all
produce the same file, at the deliverable's own authoring frame rate — which is what makes the
result comparable with a viewport playblast of the same shot rather than merely similar to it.
Sampling the display instead would drop and duplicate frames wherever the device was busy, and the
busiest moment is always the one worth reviewing.

```
  page          poseAt(i / fps) -> render -> snapshot          (main thread, one per frame)
  workers       snapshot -> PNG -> POST /playblast/frame       (several frames at once)
  server        each frame straight to a scratch sequence, numbered from the clip's START frame
  encode        pythontk.SequenceEncoder -> ffmpeg -> <deliverable>_<clip>.mp4
```

The encode is **the same code Maya's playblast exporter runs**: `pythontk.SequenceEncoder` owns the
target registry, the CRF mapping, the even-dimension rule H.264 needs and the audio mux, and
`mayatk.PlayblastExporter` is a `pythontk.SequenceExporter` — the same core plus the capture plan —
that supplies a viewport and a timeline. So "MP4 (H.264)" means one thing in this toolkit, and a fix
to the encode lands in both tools at once.

Where the movie goes:

| The push was | Lands |
|---|---|
| an exporter's GLB, or one chosen with **External GLB** | beside that file, as `<glb stem>_<clip>.mp4` |
| a scene push | in the serve root, and the page downloads it — a scene push's GLB is the bridge's own scratch and is released the moment it is published, so there is nothing to sit beside |

Either way the finished file is fetchable from the page at `/playblast/<token>` as an attachment,
which is the half that matters in a headset: there is no containing folder to open there.

Details worth knowing:

- The report names the clip's **authoring frame range** — the numbers the DCC's timeline and the
  picker's shot ranges quote. The scratch frames themselves are numbered from zero: they are deleted
  the moment the movie exists, and numbering them the timeline's way breaks on a scene with pre-roll,
  where a shot's start frame is negative (`shot.-010.png` matches no printf pattern, and ffmpeg will
  not take a negative `-start_number` either).
- Contiguity is checked before the encode. ffmpeg reads a printf pattern straight through and stops
  at the first gap, so a dropped frame would otherwise encode silently as a short movie.
- **Frames compress on worker threads.** Compressing a frame to PNG is essentially the entire cost of
  a recording — measured per frame, everything else together (pose, render, readback, POST, the
  server's disk write) is under 5% of it — so the page hands each snapshot to a pool of workers and
  gets on with rendering the next one. Measured interleaved on a 151-frame recording from a
  5120x2880 buffer: **~5x** on the capture itself, **2.7x** on the whole button press (39.4s → 14.7s;
  the rest is the server's ffmpeg pass, which did not change). A *single* worker is already 2.7x,
  so most of the win is simply not blocking the render loop. The pool is half the machine's cores,
  clamped to 2–8; past 4 the curve is flat (2 workers 4.4x, 4 workers 5.2x, 8 workers 5.8x).
  A browser without `Worker` or `OffscreenCanvas` falls back to compressing on the main thread.
- **Quality is a preset chosen in the prompt** — `QUALITY_PRESETS` in `scripts/playblast.js`:

  | Preset | Long edge | CRF (`quality`) |
  |---|---|---|
  | Draft — 720p | 1280 | 23 (70) |
  | Standard — 1080p | 1920 | 20 (85) |
  | **High — 1440p** (default) | 2560 | 16 (100) |
  | Maximum — 4K | 3840 | 16 (100) |

  Before presets every recording was 1280 px at CRF 16. The preset's `quality` travels with
  `POST /playblast/begin` — it describes the recording, as the rate does — and reaches
  `SequenceEncoder.encode_sequence`; a `begin` naming none encodes at the recorder's own `quality`,
  and one outside 0–100 is refused before a frame is sent rather than clamped at the encode. The
  frame is **rendered** at the preset's size, not upscaled to it: a view whose drawing buffer is
  smaller has its pixel ratio raised for the recording and put back as soon as the last frame is
  in, so High from a 1280-wide window is a real 2560-wide render. A larger view is downsampled by
  the capture. The GPU's `MAX_RENDERBUFFER_SIZE` / `MAX_VIEWPORT_DIMS` cap it: a canvas asked for
  more silently allocates less, and the capture would read that stretched. Size is no longer the
  wall-clock dial it was when frames compressed on the main thread (1920 → 1280 then measured 2.1x
  faster); with the pool **1920 and 1280 cost the same** (46.7ms and 45.1ms per frame). What a
  bigger preset still costs is file size and wire time — what a reviewer on a headset over Wi-Fi
  waits for, and what Draft is for. The saved status line states the size that was written.
- **Pressing the button opens an export prompt** rather than recording at once. It states what is
  about to be written (clip, frame count, rate — the clip is the picker's to choose and the rate is
  the deliverable's, so the prompt only shows them) and offers the quality preset and the two
  burn-ins below. Cancel,
  Escape or a click on the backdrop writes nothing. The answers are remembered for the tab, so a
  second recording re-offers them rather than the defaults, and they are snapshotted when the
  recording starts — a file is annotated the way the prompt that started it was answered, never
  half of it. A push landing while the prompt is up **dismisses** it: what it was asking about is
  no longer on screen, and a stale summary over a different scene is worse than asking again.
- **Burn-in: shot name and frame** draws the **shot name, the DCC frame number and the clip time**
  into the recorded frames — the shot the playhead is inside on the whole-timeline clip,
  `'<name> (hold)'` through a gap, and the clip's own name on a single shot, so the movie says the
  same thing the transport readout did. The frame number is the **authoring** one — the clip's
  start frame plus the offset — so a note about "frame 112" names the frame an animator will open.
- **Burn-in: shot description** draws the note the DCC's Shots panel carries for that shot
  (`extras.animation_web`'s `description`) on the line above the name — the half of a shot record a
  reviewer watching the movie cannot otherwise see. It follows the playhead across the sequence, and
  a shot that states no description records without one rather than holding a blank line open for
  it. The two burn-ins are independent: with only the description ticked the note lands alone on
  the foot of the frame.
- Both are opt-in and off by default: a burn-in is drawn into the pixels and cannot be taken out
  again, so a recording is what the reviewer saw unless someone asked for the annotation. Left
  text is elided with an ellipsis rather than run under the frame counter, because a description is
  prose and can be longer than the frame.
- A push landing mid-recording **drops** it. Every frame after the swap would be of a different
  scene, and the file would silently be a cut between two versions.
- The frame size is fixed when the recording starts, so resizing the window part way through cannot
  change the movie's dimensions (ffmpeg answers a mixed-size sequence with a garbled encode rather
  than an error). Recording from **inside** an immersive session is refused: `render` targets the XR
  framebuffer there, so a canvas readback would capture the mirror.
- The button doubles as the progress readout (`Recording 42/151 ✕`) and cancels on a second click;
  the status line is left saying what the page is showing.

## Exporting a still

**Export Image** saves the current view — the camera where the reviewer left it, the pose the
transport is holding — as a PNG. It is the `snapshot` viewer script, a checkbox on the panel's
Viewer Scripts row; the HUD and the control bar are page markup over the canvas and are not in it.

```
  page     [raise pixel ratio] -> render -> drawImage onto a pad -> [restore]   (one task)
           pad.toBlob(PNG) -> POST /snapshot
  server   PreviewServer.save_snapshot -> <deliverable>_view_001.png, _002 ...
```

- **Sizes** are the prompt's: *As shown* (the drawing buffer exactly as the page draws it), or
  Standard / High / Maximum at the playblast presets' long edges (1920 / 2560 / 3840 px; High is the
  default and the choice is remembered for the tab). A sized still is **rendered** at that size via
  `viewer.captureSize` — the playblast's rule — never upscaled, and the ratio is put back in the
  same task, with a re-render so the page never composites a blank frame.
- **Render and readback happen in one task**, for the playblast's reason: without
  `preserveDrawingBuffer` the drawing buffer is gone once the browser composites, and a readback
  after an `await` saves an empty canvas without complaint.
- **Where it lands** is the playblast's table, decided by the one rule the server holds for every
  file the page writes: beside the published file when it is on disk, else in the serve root, which
  the page then downloads. Stills
  are **numbered, never overwritten** (`FileUtils.next_version_path`) — the next angle must not
  replace the last — and the name is composed server-side from the deliverable's sanitized stem;
  nothing in the request can name or place the file.
- The route takes **PNG only**, checked by signature as well as by `Content-Type`
  (`PreviewServer.SNAPSHOT_TYPES`), under `MAX_SNAPSHOT_BYTES`, and is held to the Host and Origin
  checks every writing route carries.
- Refused inside an immersive session (the canvas is the mirror, not the device), with a lost GPU
  context (the page's own sticky status says why), and with no model loaded.

## Cost and budget

Timings, measured end to end on a production assembly (366 MB FBX, 2485 nodes over 757 mesh
transforms and 537 instanced shapes, 98k triangles, one 1591-frame take, 47 baked objects, 353 MB
of embedded 4096² PNG), with the shipped methods wrapped and the machine otherwise idle:

| Stage | Cost | Share |
|---|---|---|
| FBX write (Maya) | 6.7 s | 2% |
| **FBX2glTF conversion** | **~365 s** | **87%** |
| Passes inside the conversion (sidecar incl. ORM repack 17 s; dedupe, clips, fades) | 22 s | 5% |
| Lightmap wiring | 3.1 s | <1% |
| Texture optimize (WebP) | 20.9 s | 5% |
| Publish | 0.03 s | — |
| **Push** | **419 s** | |

**The converter is the push, and its cost is the bake, not the bytes.** FBX2glTF evaluates every
node at every frame of every take (24 fps) whether or not the node is animated. Measured on the
same payload: strip every animation object and it converts in **148 s**; a 12 MB *textureless*
export of the same scene still takes **over 300 s** (it timed out under the old size-derived budget
and lost the push — `MeshConvert.conversion_timeout` now also reads nodes × frames out of the file's
own census, `bake_node_frames`); resize every embedded texture to 2048 *before* the conversion (now the payload pass) and
the converter drops to ~290 s — real, but a fifth. So the larger levers are on the DCC side, in what the
export ships: the take and timeline range (~250 s of this push, ~63 µs per node-frame) and the node
count (602 hidden rig helpers are 24% of the nodes). Both are priced maintainer decisions in
`.claude/BACKLOG.md`.

What each option box row costs on the same scene (ratios rather than absolutes for the last three:
those runs shared the machine with the experiments above):

| Option | Effect |
|---|---|
| Export Preset | the push's tasks, checks, textures, animation and sidecar are the chosen Scene Exporter preset's rows (2026-09-05; the dials below were the option box's own before that and are kept for the costs they measured) |
| Textures **off** | FBX 366 → 12 MB, no sidecar textures, no optimize pass — and the conversion still takes minutes |
| Scene Sidecar **off** | saves the 17 s ORM repack; the preview then shows FBX2glTF's own packing, which reads metallic 1 on grayscale source sets |
| Texture File Type **KTX2** | texture pass 57 s instead of 21 s (`toktx`, UASTC for data maps / ETC1S for colour) and 42 MB instead of 35 MB on the wire — the win is GPU memory, not the push |
| Include Animation **on** | push 719 s: Maya's FBX write becomes 84 s (it bakes complex animation), the FBX 482 MB, the conversion 616 s, and the GLB 101 MB — 73 MB of it animation accessors |
| Scope **visible**, both viewer scripts | push 306 s: the hidden rig helpers (24% of the nodes) stay behind, and with them about a quarter of the bake; the two viewer scripts cost nothing measurable |
| External GLB (`publish_file`) | milliseconds: no export, no conversion |

The server half is a rounding error everywhere: publishing a 50 MB GLB is 8 ms moved / 75 ms
copied, a manifest poll round-trips in 1.5 ms, the page re-sync on publish is 0.3 ms, a viewer
script is one file copy, and loopback serves the asset at ~370 MB/s.

Cheaper wins taken on the Python side (~44 s of the push): `dedupe_glb_images` collapses the
byte-identical copies FBX2glTF emits per material before anything pays to encode them; the preview
releases its consumed FBX (and the SDK's extracted `.fbm`) as soon as the GLB exists instead of
leaving ~700 MB per push to the 7-day sweep; `GlbPipeline`'s first stage downsizes the FBX's
embedded textures to the delivery ceiling first (`FbxMedia.downsize`, 7.5 s) so the raw GLB the
converter hands back is 96 MB rather than 340 MB and every later pass reads a 2K file
(re-measured quiet, end to end, the push went from 419 s to 333 s: the converter ~365 → ~290 s, the sidecar's inline passes 17 → 9 s, the optimize pass 21 → 10 s); the lossless-WebP effort is 75 rather than 100 (identical bytes, 14% less of the
optimize pass, whose critical path is one 2K normal map's encode); and the packed ORM the sidecar
embeds is written at PNG level 1, since the texture pass re-encodes it anyway.

**Texture budget is the whole file.** Before the optimize pass a delivery measured 94.7 MB, of which
87.8 MB (93%) was uncompressed source PNG — a 24 MB normal map, a 20 MB character texture — against
2.6 MB of geometry. Downsizing to 2048 and re-encoding WebP took that room to **~15 MB**.

**The remaining constraint is GPU memory, not download — and KTX2 mode is the fix.** Probed on a
delivered preview GLB (a different, larger selection — 57 materials, 514 primitives): **9.5 MB on
the wire, 5.97 MB of it images, which decode to ~555 MB of RGBA and ~740 MB with mipmaps.** Every
one of its 38 images is 2048², because `max_size` is a per-image ceiling and nothing budgets the
total. On a headset that, not the download, is what limits how large a scene can be previewed.

The fix is opt-in and request-scoped: `bridge.push(glb_options={"texture_file_type": "ktx2"})`
— the **Texture File Type** row on the WebXR Preview panel, the Scene Exporter's own — re-encodes
that delivery to
KTX2/Basis (`KHR_texture_basisu`), which the GPU keeps block-compressed — the viewer's `KTX2Loader`
transcodes it to ASTC on a standalone headset, BC7 on desktop. It needs KTX-Software's `toktx` on
the authoring machine (the push raises with the install URL when it is missing, never silently
ships WebP). Codecs are chosen per glTF slot — UASTC for normals and ORM/occlusion data, ETC1S for
base color and emissive — and baked lightmaps deliberately stay on the lossless-WebP path; the
details live on `MeshConvert.optimize_glb_textures`. Two complementary levers ride the same call:
a **secondary ceiling** (`secondary_max_size`) for the packed data maps alone -- a mask reads the
same at half the resolution a normal map needs, so the Scene Exporter's *Secondary Map Size* row
caps metallic-roughness / occlusion below the primary ceiling while color and normals keep it --
and **UASTC RDO** (`uastc_rdo`, the exporter's *KTX2 RDO* row), which steers the UASTC blocks
toward what Zstandard compresses at a measured quality cost (ORM packs -30% at lambda 1; normals
are capped at 0.75). The deliverable's per-frame animation keys are the third lever
(`GlbPipeline.build(key_tolerance=...)`, the exporter's *GLB Key Tolerance* row).

## Gotchas worth knowing

- **The page needs `unpkg.com`.** three.js loads from a CDN. Behind a default-deny outbound
  firewall the module never executes — no error event, just a dark page. A classic-script watchdog
  says so after 8 s rather than leaving it silent.
- **WebXR needs localhost or HTTPS.** Opened over a plain-HTTP LAN address, `navigator.xr` is
  absent and the VR button simply never appears; the page says which case it is in. Anything
  that is not this machine -- a standalone headset, a reviewer -- gets HTTPS from a share.
- **A first share can need one-time setup, and says which.** A tailnet that has not enabled
  Serve/Funnel, and a firewall that denies outbound by default, both fail the share at once with
  the step to take first (see *Sharing a link*).
- **A missing `TEXCOORD_1` means no lightmap.** The FBX was exported without the lightmap UV set;
  the applier warns per primitive rather than binding something wrong.
- **Per-object maps on a shared material each bind a copy.** A glTF material carries one lightmap,
  so the first object binds the shared material and every later object with a different map binds
  its own clone of it (and its own mesh entry when instanced). This used to be refused — and the
  refused object wore the first one's lighting anyway, which reads as a bad bake.
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
- **Same-named objects are told apart by where they sit.** FBX carries leaf names only, so two
  objects sharing one (`…|MACHINE_B|BODY|BODY`, `…|MACHINE_A|BODY`) arrive as two `BODY` nodes. The
  manifest publishes each object's `hierarchy`, and a record binds only to the node whose lineage
  it is the unique best match for; a stale record whose node belongs to a better match counts as
  out of scope. A manifest without hierarchies (built before they were published) cannot tell
  them apart, so they are skipped and warned — re-export to publish one.
