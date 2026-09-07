/*
  Shadow rig — drive the DCC shadow planes from the model's own source and
  contact nodes, per frame, with the maps the conversion bound.

  A shadow rig (mayatk / blendertk `ShadowRig`) is a ground quad under a prop
  whose transform a DCC expression re-places from the light every frame, and
  whose texture is either a rasterized silhouette (the PROJECTED type: the
  quad's UVs read one tile of an atlas) or a HORIZON map (a height-field bake
  of the prop's footprint -- solid spans per pixel, a distance field and a
  min/max pyramid -- that the shader marches toward the light). Neither
  survives the FBX -> glTF hop as behaviour: the expression is baked to keys at
  export and the horizon map is a data texture no material references.

  `MeshConvert.apply_glb_shadows` (pythontk) binds the maps into the file and
  writes root `extras.shadow_web`: the v2 record per plane plus the glTF NODE
  indices of the plane, its source and its contact, the texture index of each
  map, and rects in glTF top-left space. This script is the runtime half:

    load   read the manifest; map node indices to Object3Ds through the
           loader's association table (names are not unique); give every plane
           one ShaderMaterial (both types in ONE program, `uMode`); merge the
           projected planes that share a colour map and carry no fade track
           into an InstancedMesh with per-instance rect and opacity/intensity
           attributes. Horizon planes stay one mesh each (their per-frame
           uniforms are a frame and a source, which instancing does not buy
           enough to be worth a second program).
    frame  evaluate the projection model (a port of ShadowProjection.model /
           ShadowModel.placement, pinned against it by test_shadow_web.py)
           from the source and contact nodes and place each plane -- or its
           instance matrix -- the way the Maya expression does. A horizon
           plane is fed its source in the contact frame just before it draws,
           off world matrices the renderer has already refreshed.

  Everything is placed in MODEL space (the glTF root's frame, metres): the page
  fits and spins the pivot and the model, and a placement in that frame is
  unaffected. The record's lengths are DCC units times `unit_scale`. A plane's
  UVs are brought back to the unit square first (the DCC remaps a packed plane's
  UVs into its atlas rect so a fallback viewer needs no transform; here the
  rect is applied per plane, per instance).

  Textures this script creates are its own: the page's disposeModel frees what
  hangs off a model's materials, so the horizon maps (never on a material) and
  the originals detached from the planes it takes over are disposed HERE, on
  the next load. The InstancedMesh is a child of the model and dies with it.
  A plane's fade keeps working: the page binds the KHR_animation_pointer ramp
  to the ORIGINAL material, whose opacity this script copies every frame.

  Activate:  automatic -- PreviewServer.AUTO_SCRIPTS turns it on for any
             deliverable whose root extras carry `shadow_web`;
             or bridge.push(scripts=["shadow_rig"]).
*/

const MANIFEST_KEY = 'shadow_web';
//: Lift above the ground in DCC units (the Maya rig's GROUND_OFFSET), so the
//: plane never z-fights the floor; times unit_scale at run time.
const GROUND_OFFSET = 0.01;
//: ShadowProjection.OVERHEAD_BEARING / FAR_FACTOR / EPS, verbatim.
const OVERHEAD_BEARING = [0, 1];
const FAR_FACTOR = 1.0e6;
const EPS = 1.0e-6;
//: The axis a directional source node shines along, in its own frame, as the
//: FBX hop delivers it: FBX lights point down local -Y, and the DCC exporters
//: bake that pre-rotation into the light node's own rotation (a Maya light
//: shines down its local -Z in Maya; a locator's frame passes through
//: unchanged). Measured with Maya 2025 + FBX2glTF 0.13.1: a directionalLight
//: at rotate (-50, 35, 0) arrives as a node whose local -Y, rotated by its
//: quaternion, IS Maya's world shining direction to 1e-7, while its local -Z
//: points up. Pinned by test_shadow_web.py on an asymmetric rotation.
const DIRECTIONAL_AXIS = [0, -1, 0];
const ALPHA_POINTER = /^\/materials\/(\d+)\/pbrMetallicRoughness\/baseColorFactor$/;
const RECT_IDENTITY = [1, 1, 0, 0];
const CANVAS_DEFAULT = [-1, 1, -0.5, 0.5];

// One program for both types. The instanced path carries the rect and the
// opacity/intensity pair per instance; the per-mesh path carries them as
// uniforms; both land in the same varyings.
const VERTEX = /* glsl */ `
uniform vec4 uRect;
uniform vec2 uParams;
varying vec2 vUv;
varying vec3 vWorld;
varying vec4 vRect;
varying vec2 vParams;
#ifdef USE_INSTANCING
attribute vec4 iRect;
attribute vec2 iParams;
#endif
void main() {
  vec4 local = vec4(position, 1.0);
#ifdef USE_INSTANCING
  local = instanceMatrix * local;
  vRect = iRect;
  vParams = iParams;
#else
  vRect = uRect;
  vParams = uParams;
#endif
  vec4 world = modelMatrix * local;
  vWorld = world.xyz;
  vUv = uv;
  gl_Position = projectionMatrix * viewMatrix * world;
}
`;

// Mode 0 (projected): the silhouette's alpha through the plane's rect, black
// RGB times the material colour. Mode 1 (horizon): the contract's evaluation
// -- a ray march through the prop's height-field map -- which is NOT written
// here: the body between the markers below is spliced in from
// `pythontk/geo_utils/shadow_horizon.glsl` by
// `m3trik/scripts/sync_shadow_shaders.py`, and `HeightFieldMap.alpha` beside
// it is the oracle `test_shadow_web.py` renders this page against. What IS
// this file's own is the host half: the uniforms, and `SH_Fetch` -- glTF's
// top-left texture origin is a per-engine fact and the shared body never
// computes a texture address.
const FRAGMENT = /* glsl */ `
uniform int uMode;
uniform vec3 uColor;
uniform sampler2D uMap;
uniform sampler2D uHorizonMap;
uniform vec4 uHorizonRect;     // the tile block inside its texture, glTF top-left: sx, sy, ox, oy
uniform vec4 uHorizonParams;   // footprint pixels per side (S), spans per column (K), pyramid levels, 0
uniform vec4 uHorizonBounds;   // the footprint in the map's frame (metres): a0, a1, b0, b1
uniform float uHorizonScale;   // the height (metres) a 16-bit channel value of 65535 stands for
uniform float uGround;         // the ground plane's height in the map's frame
uniform vec3 uOrigin;          // the contact frame in WORLD space: origin, then its three axes
uniform vec3 uAxisA;
uniform vec3 uAxisB;
uniform vec3 uAxisUp;
uniform vec4 uSource;          // xyz world; w 1 = a position, 0 = the direction the source SHINES
uniform vec2 uSourceSize;      // diameter (metres), angular diameter (radians) -- FULL widths, halved in the shader
varying vec2 vUv;
varying vec3 vWorld;
varying vec4 vRect;
varying vec2 vParams;

// The shared body's texel hook. The map's tiles sit side by side inside a
// block the atlas packer placed at uHorizonRect.zw, each S texels square; row
// 0 of a tile is the PNG's top row (the frame's b0 edge) -- and three.js
// uploads a data texture unflipped, so a texel address is the block origin
// plus the tile plus the texel.
vec4 SH_Fetch(int tile, int xi, int yi) {
  ivec2 size = textureSize(uHorizonMap, 0);
  int S = int(uHorizonParams.x + 0.5);
  ivec2 origin = ivec2(uHorizonRect.zw * vec2(size) + 0.5);
  return texelFetch(uHorizonMap, origin + ivec2(tile * S + xi, yi), 0);
}

// >>> BEGIN GENERATED shadow_horizon body
// ===========================================================================
//  shadow_horizon.glsl -- the height-field shadow march.
//
//  SINGLE SOURCE OF TRUTH. Every consumer -- the WebXR viewer's GLSL, Unity's
//  HLSL, Maya's .ogsfx/.fx and Blender's gpu shader -- runs THIS text. A
//  consumer that is a Python process when it needs the shader assembles it
//  through ShadowHorizon.shader_source(); one that is not (a browser, a
//  Unity project) carries a generated mirror that sync_shadow_shaders.py
//  writes and --check guards. NEVER hand-edit a mirror.
//
//  The normative behaviour is HeightFieldMap.alpha in shadow_horizon.py:
//  the exact walk over every footprint pixel the ray crosses. This text
//  walks the map's min/max pyramid instead and skips a cell only when it can
//  prove the ray clears it by the source's angular margin -- so it lands on
//  the same pixels, computes the same gaps, and returns the same alpha.
//  Every engine test compares pixels against that reference; a change here
//  is a change there and both move together.
//
//  ---------------------------------------------------------------- the host
//  Define these BEFORE pasting this body:
//
//    SH_HLSL                 -- when the target speaks HLSL (else GLSL)
//    vec4 SH_Fetch(int tile, int xi, int yi)
//                            -- one texel of the map image: tile 0 .. K - 1
//                               the span tiles, K the aux tile, K + 1 the
//                               pyramid tile; (xi, yi) inside the tile with
//                               (0, 0) its top-left, the frame's (a0, b0)
//                               corner (row 0 = ib 0). The body never
//                               computes a texture address: glTF's top-left
//                               origin, Unity's double row flip and Maya's own
//                               convention are irreducibly per-engine, and
//                               this is where that divergence belongs.
//
//  The host also supplies the uniforms and calls ShAlpha. Nothing in here
//  samples a texture, reads a uniform or names an engine.
// ===========================================================================

#ifdef SH_HLSL
#define vec2  float2
#define vec3  float3
#define vec4  float4
#define mix   lerp
#else
#endif

//  The traversal's hard cap. Measured 7-16 steps on average and under 110 at
//  the 98th percentile on chair- and table-sized props at 128 pixels; a ray
//  that still walks at the cap returns the penumbra it has so far.
#define SH_MAX_STEPS 256

// ---------------------------------------------------------------- the map
//  What the layout needs, built once per fragment. ground is the height of
//  the ground plane along the map's up axis IN THE MAP'S FRAME: the spans
//  are heights above it, so the fragment's own height is replaced by it
//  (see ShAlpha) and every ray starts at height 0.
struct ShField
{
    int   size;        // footprint pixels per side, S (a power of two)
    int   spans;       // solid spans per column, K
    int   levels;      // pyramid levels above level 0: log2(S)
    vec4  bounds;      // the footprint in the frame: a0, a1, b0, b1
    float heightScale; // the height a 16-bit channel value of 65535 stands for
    float ground;
};

ShField ShMakeField(int size, int spans, int levels, vec4 bounds,
                    float heightScale, float ground)
{
    ShField g;
    g.size = max(1, size);
    g.spans = max(1, spans);
    g.levels = max(0, levels);
    g.bounds = bounds;
    g.heightScale = max(heightScale, 1e-6);
    g.ground = ground;
    return g;
}

// ------------------------------------------------------------- decoding
//  A channel in [0, 1] back to its byte, then two bytes to their word.
float ShByte(float c) { return floor(c * 255.0 + 0.5); }
float ShWord(float hiByte, float loByte) { return ShByte(hiByte) * 256.0 + ShByte(loByte); }
//  A 16-bit height, an 8-bit height, and the distance field's 1/64 pixel.
float ShHeight16(ShField g, float hiByte, float loByte)
{
    return ShWord(hiByte, loByte) / 65535.0 * g.heightScale;
}
float ShHeight8(ShField g, float c) { return ShByte(c) / 255.0 * g.heightScale; }
float ShDist16(float hiByte, float loByte) { return ShWord(hiByte, loByte) / 64.0; }

//  The tiles. A span texel: R, G the 16-bit lo, B, A the 16-bit hi (hi == 0
//  = no span). The aux texel: R, G the distance to the nearest solid column
//  in 1/64 pixel, B that column's hull top and A its bottom (8-bit). A
//  pyramid texel at level l: R the dilated hull's bottom and G its top
//  (8-bit, G == 0 = nothing within a cell), B, A the cell's least distance.
vec4 ShSpanTexel(ShField g, int k, int xi, int yi) { return SH_Fetch(k, xi, yi); }
vec4 ShAuxTexel(ShField g, int xi, int yi) { return SH_Fetch(g.spans, xi, yi); }
vec4 ShCellTexel(ShField g, int level, int xi, int yi)
{
    // level l occupies S >> l rows from row S - (S >> (l - 1)) of its tile
    int y0 = g.size - (g.size >> (level - 1));
    return SH_Fetch(g.spans + 1, xi, y0 + yi);
}

// --------------------------------------------------------------- helpers
//  How far the ray's height range [h0, h1] misses a span [lo, hi]; 0 when
//  they overlap -- the hit test and the penumbra's gap in one.
float ShGap(float h0, float h1, float lo, float hi)
{
    if (h1 < lo) return lo - h1;
    if (h0 > hi) return h0 - hi;
    return 0.0;
}

//  The coarsest level whose cell edges include the integer pixel boundary
//  b: the count of its trailing zero bits, capped (b == 0 is every level).
int ShAlignment(int b, int levels)
{
    int a = 0;
    while (a < levels && ((b >> a) & 1) == 0) a++;
    return a;
}

// ------------------------------------------------------------- the march
//  o and d are the ray in PIXEL space (origin, pixels per unit t; t runs in
//  frame units along the horizontal), slope its rise per unit t, rho the
//  source's angular radius, [tEnter, tExit] its passage through the field,
//  pixelSize the smaller pixel pitch in frame units and aspect the larger
//  over it: the distance field is measured in the smaller pitch, so a
//  pixel step along the other axis is aspect units of it.
//
//  Level by level: the cell at the ray's position is skipped whole when the
//  ray clears the cell's bound by more than the margin rho * t (the height a
//  gap has to have at that distance to lie outside the source disc);
//  otherwise the level drops until level 0, where the K spans decide
//  exactly. After a cell the level climbs back as far as the boundary just
//  crossed is aligned. HeightFieldMap._march is the same walk over every
//  level-0 pixel; the skipped cells are exactly the ones that could not
//  have changed its answer.
float ShMarch(ShField g, vec2 o, vec2 d, float slope, float rho,
              float tEnter, float tExit, float pixelSize, float aspect)
{
    int level = g.levels;
    float t = tEnter;
    float clearance = 1e30;
    bool hit = false;
    bool zx = abs(d.x) < 1e-12;
    bool zy = abs(d.y) < 1e-12;
    vec2 nudge = vec2(d.x >= 0.0 ? 1e-4 : -1e-4, d.y >= 0.0 ? 1e-4 : -1e-4);
    // The ray's position in pixels. After a cell it is rebuilt from the
    // boundary just crossed -- an exact integer -- never from o + d * t,
    // which float32 lands a hair short of the boundary at any size the
    // nudge does not cover, re-entering the cell it left until the cap.
    vec2 p = o + d * t + nudge;
    for (int iter = 0; iter < SH_MAX_STEPS; ++iter)
    {
        if (t >= tExit || hit) break;
        int cell = 1 << level;
        int n = g.size >> level;
        int cx = int(floor(p.x / float(cell)));
        int cy = int(floor(p.y / float(cell)));
        if (cx < 0 || cy < 0 || cx >= n || cy >= n) break;
        // the cell's exit along the ray
        float bx = float((cx + (d.x > 0.0 ? 1 : 0)) * cell);
        float by = float((cy + (d.y > 0.0 ? 1 : 0)) * cell);
        float tx = zx ? 1e30 : (bx - o.x) / d.x;
        float ty = zy ? 1e30 : (by - o.y) / d.y;
        float te = min(min(tx, ty), tExit);
        float h0 = slope * t;
        float h1 = slope * te;
        float margin = rho * te;
        if (level > 0)
        {
            vec4 c = ShCellTexel(g, level, cx, cy);
            float distC = ShDist16(c.b, c.a);
            bool descend;
            if (distC <= 0.0)
            {
                // holds a solid column: its neighbourhood's hull bounds every
                // span and every nearest-column hull a pixel in it can meet
                descend = ShGap(h0, h1, ShHeight8(g, c.r), ShHeight8(g, c.g)) <= margin;
            }
            else
            {
                // empty: only the lateral distance can bring a pixel's gap
                // under the margin -- the exact test reads the field
                // bilinearly, one pixel past the cell at most, and its own
                // half-pixel comes off too (both in the larger pitch)
                descend = max(distC - 1.5 * aspect, 0.0) * pixelSize <= margin;
            }
            if (descend)
            {
                level -= 1;
                continue;
            }
        }
        else
        {
            bool solid = false;
            float gap = 1e30;
            for (int k = 0; k < g.spans; ++k)
            {
                vec4 s = ShSpanTexel(g, k, cx, cy);
                if (ShWord(s.b, s.a) <= 0.0) continue;
                solid = true;
                float gk = ShGap(h0, h1, ShHeight16(g, s.r, s.g), ShHeight16(g, s.b, s.a));
                if (gk <= 0.0) { hit = true; break; }
                gap = min(gap, gk);
            }
            if (hit) break;
            float tmid = max(0.5 * (t + te), 1e-9);
            if (!solid)
            {
                // The distance field and the nearest column's hull, read
                // BILINEARLY at the ray's midpoint through the pixel: per-
                // pixel values would make the penumbra jump a quarter of its
                // width when a boundary-hugging ray lands one pixel over in
                // float32 (measured in the viewer: 0.21 against 0.07).
                vec2 m = o + d * tmid - 0.5;
                vec2 f = floor(m);
                vec2 w = clamp(m - f, 0.0, 1.0);
                int x0 = clamp(int(f.x), 0, g.size - 1);
                int y0 = clamp(int(f.y), 0, g.size - 1);
                int x1 = min(x0 + 1, g.size - 1);
                int y1 = min(y0 + 1, g.size - 1);
                vec4 a00 = ShAuxTexel(g, x0, y0);
                vec4 a10 = ShAuxTexel(g, x1, y0);
                vec4 a01 = ShAuxTexel(g, x0, y1);
                vec4 a11 = ShAuxTexel(g, x1, y1);
                // x the distance in pixels, y the hull's bottom, z its top
                vec3 v00 = vec3(ShDist16(a00.r, a00.g), ShHeight8(g, a00.a), ShHeight8(g, a00.b));
                vec3 v10 = vec3(ShDist16(a10.r, a10.g), ShHeight8(g, a10.a), ShHeight8(g, a10.b));
                vec3 v01 = vec3(ShDist16(a01.r, a01.g), ShHeight8(g, a01.a), ShHeight8(g, a01.b));
                vec3 v11 = vec3(ShDist16(a11.r, a11.g), ShHeight8(g, a11.a), ShHeight8(g, a11.b));
                vec3 v = mix(mix(v00, v10, w.x), mix(v01, v11, w.x), w.y);
                float lateral = max(v.x - 0.5 * aspect, 0.0) * pixelSize;
                float vertical = ShGap(h0, h1, v.y, v.z);
                gap = sqrt(lateral * lateral + vertical * vertical);
            }
            clearance = min(clearance, gap / tmid);
        }
        // leave the cell, and climb as far as the boundary crossed allows
        t = te;
        p = o + d * te;
        if (tx <= ty) p.x = bx + nudge.x;
        if (ty <= tx) p.y = by + nudge.y;
        int a = 0;
        if (tx <= ty) a = max(a, ShAlignment(int(bx + 0.5), g.levels));
        if (ty <= tx) a = max(a, ShAlignment(int(by + 0.5), g.levels));
        if (a > level) level = a;
    }
    if (hit) return 1.0;
    if (rho <= 1e-12) return 0.0;
    return clamp(1.0 - clearance / rho, 0.0, 1.0);
}

// ------------------------------------------------------------- the alpha
//  Shadow alpha at a world-space fragment, for one source.
//
//  origin, axisA, axisB, axisUp are the contact frame in WORLD space:
//  its origin and its three axes rotated by whatever transform carries the
//  prop. Working through them rather than a world-to-contact matrix keeps
//  every binding a vec3 -- which is all a Maya .ogsfx uniform, a Blender
//  push constant and a Unity instanced property are all proven to carry --
//  and takes a float4x4 out of Unity's instancing buffer.
//
//  source.w selects the kind: 1 = source.xyz is a world position, 0 =
//  it is the direction the source SHINES (the reference's vocabulary; the
//  shader negates). sourceSize.x is a positional source's diameter in frame
//  units, sourceSize.y a directional source's angular diameter in radians
//  -- the same halving the reference does.
float ShAlpha(ShField g, vec3 worldPos, vec3 origin,
              vec3 axisA, vec3 axisB, vec3 axisUp,
              vec4 source, vec2 sourceSize)
{
    vec3 rel = worldPos - origin;
    // the fragment in the frame, its height REPLACED by the ground plane's
    vec2 P = vec2(dot(rel, axisA), dot(rel, axisB));
    vec3 L;
    float rho;
    if (source.w > 0.5)
    {
        vec3 sv = source.xyz - origin;
        vec3 sf = vec3(dot(sv, axisA), dot(sv, axisUp), dot(sv, axisB));
        vec3 lv = sf - vec3(P.x, g.ground, P.y);
        float dist = max(length(lv), 1e-12);
        L = lv / dist;
        rho = asin(clamp(0.5 * sourceSize.x / dist, 0.0, 1.0));
    }
    else
    {
        vec3 sh = vec3(dot(source.xyz, axisA), dot(source.xyz, axisUp), dot(source.xyz, axisB));
        L = -sh / max(length(sh), 1e-12);
        rho = 0.5 * sourceSize.y;
    }
    float horiz = length(L.xz);
    if (horiz < 1e-9 || L.y <= 0.0) return 0.0;
    vec2 dir = L.xz / horiz;
    float slope = L.y / horiz;
    // into pixel space
    float S = float(g.size);
    vec2 px = vec2(g.bounds.y - g.bounds.x, g.bounds.w - g.bounds.z) / S;
    vec2 o = vec2((P.x - g.bounds.x) / px.x, (P.y - g.bounds.z) / px.y);
    vec2 d = dir / px;
    bool zx = abs(d.x) < 1e-12;
    bool zy = abs(d.y) < 1e-12;
    bool inx = o.x >= 0.0 && o.x < S;
    bool iny = o.y >= 0.0 && o.y < S;
    float tx0 = zx ? 0.0 : (0.0 - o.x) / d.x;
    float tx1 = zx ? 0.0 : (S - o.x) / d.x;
    float ty0 = zy ? 0.0 : (0.0 - o.y) / d.y;
    float ty1 = zy ? 0.0 : (S - o.y) / d.y;
    float txLo = zx ? (inx ? -1e30 : 1e30) : min(tx0, tx1);
    float txHi = zx ? (inx ? 1e30 : -1e30) : max(tx0, tx1);
    float tyLo = zy ? (iny ? -1e30 : 1e30) : min(ty0, ty1);
    float tyHi = zy ? (iny ? 1e30 : -1e30) : max(ty0, ty1);
    float tEnter = max(max(txLo, tyLo), 0.0);
    float tExit = min(txHi, tyHi);
    if (tExit <= tEnter) return 0.0;
    float pitch = min(px.x, px.y);
    return ShMarch(g, o, d, slope, rho, tEnter, tExit, pitch,
                   max(px.x, px.y) / max(pitch, 1e-12));
}
// <<< END GENERATED shadow_horizon body

void main() {
  float opacity = vParams.x;
  float intensity = vParams.y;
  if (uMode == 1) {
    ShField g = ShMakeField(
      int(uHorizonParams.x + 0.5), int(uHorizonParams.y + 0.5), int(uHorizonParams.z + 0.5),
      uHorizonBounds, uHorizonScale, uGround);
    float a = ShAlpha(g, vWorld, uOrigin, uAxisA, uAxisB, uAxisUp, uSource, uSourceSize);
    gl_FragColor = vec4(uColor, a * opacity * intensity);
  } else {
    vec4 tex = texture2D(uMap, vUv * vRect.xy + vRect.zw);
    gl_FragColor = vec4(tex.rgb * uColor, tex.a * opacity * intensity);
  }
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

/* ------------------------------------------------------- the model, ported --- */

// ShadowProjection.model for up = Y, term for term: the bounding cylinder's
// base and top disks project through the light at k = (L - G) / (L - h); the
// reach is capped at maxStretch heights and no factor exceeds 1 + maxStretch.
function shadowModel(contact, light, ground, radius, height, maxStretch) {
  const [cx, cy, cz] = contact;
  const [lx, ly, lz] = light;
  const dx = cx - lx;
  const dz = cz - lz;
  const dist = Math.hypot(dx, dz);
  let bearing = OVERHEAD_BEARING;
  let overhead = true;
  if (dist > EPS) {
    bearing = [dx / dist, dz / dist];
    overhead = false;
  }
  const kMax = 1 + maxStretch;
  const kBase = Math.min(Math.max((ly - ground) / Math.max(1e-4, ly - cy), 0), kMax);
  const kCap = Math.min(kMax, kBase + (maxStretch * height) / Math.max(dist, EPS));
  const kTop = Math.min(Math.max((ly - ground) / Math.max(1e-4, ly - cy - height), 0), kCap);
  const reach = Math.max(0, dist * (kTop - kBase));
  const width = 2 * radius * Math.max(kTop, kBase);
  return {
    anchor: [lx + dx * kBase, lz + dz * kBase],
    bearing,
    kBase,
    kTop,
    reach,
    base: radius * kBase,
    top: radius * kTop,
    width,
    overhead,
  };
}

// ShadowModel.placement: the canvas fractions (far edge in top radii from
// the head, near edge as a fraction of the far edge -- so the anchor, a
// grounded target's feet, keeps its place in the texture -- sides as
// fractions of the width) to the quad's centre and its extents along and
// across the bearing.
function placement(model, fractions) {
  const [u0, u1, w0, w1] = fractions;
  const uHi = Math.max(0, model.reach + u1 * model.top);
  const uLo = u0 * uHi;
  const wLo = w0 * model.width;
  const wHi = w1 * model.width;
  const cu = 0.5 * (uLo + uHi);
  const cw = 0.5 * (wLo + wHi);
  const [ux, uz] = model.bearing;
  const wx = uz;
  const wz = -ux;
  return {
    centre: [model.anchor[0] + ux * cu + wx * cw, model.anchor[1] + uz * cu + wz * cw],
    along: uHi - uLo,
    across: wHi - wLo,
  };
}

// ShadowProjection.far_point: a directional source written as a point a long
// way back along the way it shines, so the one model body serves both.
function farPoint(contact, direction, scale) {
  const far = FAR_FACTOR * Math.max(scale, 1e-3);
  return [
    contact[0] - direction[0] * far,
    contact[1] - direction[1] * far,
    contact[2] - direction[2] * far,
  ];
}

/* ------------------------------------------------------------- the file --- */

function readManifest(parser) {
  let raw = parser?.json?.extras?.[MANIFEST_KEY];
  if (typeof raw === 'string') {
    try { raw = JSON.parse(raw); } catch { return null; }
  }
  return raw && typeof raw === 'object' ? raw : null;
}

// glTF material indices a KHR_animation_pointer alpha ramp drives -- read off
// the file the same way the page's fade shim reads it, so the two agree about
// which planes fade. A fading plane keeps its own mesh: the page drives ONE
// mesh per faded material, and an instance has no material of its own.
function fadedMaterials(parser) {
  const faded = new Set();
  for (const animation of parser.json.animations || []) {
    for (const channel of animation.channels || []) {
      const target = channel.target || {};
      if (target.path !== 'pointer') continue;
      const pointer = target.extensions?.KHR_animation_pointer?.pointer || '';
      const match = ALPHA_POINTER.exec(pointer);
      if (match) faded.add(Number(match[1]));
    }
  }
  return faded;
}

// glTF node index -> Object3D, through the loader's association table rather
// than by name (the production assembly ships duplicate names). A name-verified
// entry wins over an unverified one, and an index the table lacks (a mesh
// shared by several nodes shares one association record) falls back to the
// unique object carrying that glTF name in its userData.
function nodeResolver(THREE, parser, model) {
  const defs = parser.json.nodes || [];
  const table = new Map();
  for (const [object, assoc] of parser.associations) {
    if (!object?.isObject3D || assoc?.nodes === undefined) continue;
    const index = assoc.nodes;
    const def = defs[index];
    const named = !def?.name || object.name === THREE.PropertyBinding.sanitizeNodeName(def.name);
    if (named || !table.has(index)) table.set(index, object);
  }
  return (index) => {
    if (!Number.isInteger(index)) return null;
    if (table.has(index)) return table.get(index);
    const name = defs[index]?.name;
    if (!name) return null;
    const matches = [];
    model.traverse((object) => { if (object.userData?.name === name) matches.push(object); });
    return matches.length === 1 ? matches[0] : null;
  };
}

/* ----------------------------------------------------------- the planes --- */

const finite = (value, fallback) => (Number.isFinite(Number(value)) ? Number(value) : fallback);
const flatRect = (rect) => (Array.isArray(rect) && rect.length === 4 ? rect.map(Number) : RECT_IDENTITY.slice());
const vec3 = (value, fallback) => (Array.isArray(value) && value.length === 3 ? value.map(Number) : fallback.slice());

// The record's horizon block, validated and in metres; null when this shim
// cannot read it (an encoding or mapping it does not know -- the coverage
// maps of encoding 1 included, which fall back to the silhouette).
function horizonSpec(record, unit) {
  const h = record.horizon;
  if (!h || typeof h !== 'object') return null;
  const size = Number(h.size);
  const spans = Number(h.spans);
  const levels = h.levels === undefined ? Math.round(Math.log2(size)) : Number(h.levels);
  const bounds = Array.isArray(h.bounds) && h.bounds.length === 4
    ? h.bounds.map((v) => Number(v) * unit)
    : null;
  const heightScale = Number(h.height_scale) * unit;
  const ok = Number.isInteger(h.texture_index) && size >= 8 && spans >= 1 && bounds
    && (h.encoding ?? 1) === 2 && h.mapping === 'heightfield'
    && heightScale > 0 && bounds[1] > bounds[0] && bounds[3] > bounds[2];
  if (!ok) return null;
  return {
    textureIndex: h.texture_index,
    size,
    spans,
    levels,
    bounds,
    heightScale,
    rect: flatRect(h.rect),
    frameA: vec3(h.frame_a, [1, 0, 0]),
    frameB: vec3(h.frame_b, [0, 0, 1]),
  };
}

// A packed plane's UVs were remapped into its atlas rect by the DCC; the rect
// is applied here per plane (and per instance, over one shared quad), so the
// UVs come back to the unit square first. Idempotent, and the identity for an
// unpacked plane. A degenerate range is left alone.
function normaliseUv(geometry) {
  if (geometry.userData.shadowRigUv) return;
  geometry.userData.shadowRigUv = true;
  const uv = geometry.getAttribute('uv');
  if (!uv) return;
  let minU = Infinity;
  let minV = Infinity;
  let maxU = -Infinity;
  let maxV = -Infinity;
  for (let i = 0; i < uv.count; i += 1) {
    const u = uv.getX(i);
    const v = uv.getY(i);
    if (u < minU) minU = u;
    if (u > maxU) maxU = u;
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
  }
  const du = maxU - minU;
  const dv = maxV - minV;
  if (du < 1e-6 || dv < 1e-6) return;
  const unit = Math.abs(minU) < 1e-5 && Math.abs(minV) < 1e-5 && Math.abs(du - 1) < 1e-5 && Math.abs(dv - 1) < 1e-5;
  if (unit) return;
  for (let i = 0; i < uv.count; i += 1) {
    uv.setXY(i, (uv.getX(i) - minU) / du, (uv.getY(i) - minV) / dv);
  }
  uv.needsUpdate = true;
}

// The quad's own size, which the DCC expression divides by (basePlaneSize).
function quadExtent(geometry) {
  if (!geometry.boundingBox) geometry.computeBoundingBox();
  const box = geometry.boundingBox;
  const w = box.max.x - box.min.x;
  const d = box.max.z - box.min.z;
  return { w: w > 1e-9 ? w : 1, d: d > 1e-9 ? d : 1 };
}

function prepareDataTexture(THREE, texture) {
  texture.colorSpace = THREE.NoColorSpace;
  texture.generateMipmaps = false;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.flipY = false;
  texture.premultiplyAlpha = false;
  texture.needsUpdate = true;
}

// The horizon map decoded from its own bytes rather than through the loader:
// GLTFLoader decodes images with createImageBitmap at the browser's default
// premultiplication, and a texel whose alpha is a small occupancy mask has
// its other channels multiplied down and quantised on the way -- bits lost
// before the shader ever sees them. Decoded here with premultiplication and
// colour conversion both off; the loader's own texture is the fallback where
// createImageBitmap does not exist.
async function loadDataTexture(THREE, parser, textureIndex) {
  const def = parser.json.textures?.[textureIndex];
  const source = def?.extensions?.EXT_texture_webp?.source ?? def?.source;
  const image = parser.json.images?.[source];
  if (!image || typeof createImageBitmap !== 'function') {
    const texture = await parser.getDependency('texture', textureIndex);
    prepareDataTexture(THREE, texture);
    return texture;
  }
  let blob;
  if (image.bufferView !== undefined) {
    const buffer = await parser.getDependency('bufferView', image.bufferView);
    blob = new Blob([buffer], { type: image.mimeType || 'image/png' });
  } else {
    const url = THREE.LoaderUtils.resolveURL(image.uri, parser.options.path);
    blob = await (await fetch(url)).blob();
  }
  const bitmap = await createImageBitmap(blob, {
    premultiplyAlpha: 'none',
    colorSpaceConversion: 'none',
    imageOrientation: 'none',
  });
  const texture = new THREE.Texture(bitmap);
  prepareDataTexture(THREE, texture);
  return texture;
}

const colorOf = (THREE, material) => (material.color ? material.color.clone() : new THREE.Color(1, 1, 1));

function makeMaterial(THREE, mode, map, color, side) {
  return new THREE.ShaderMaterial({
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    blending: THREE.NormalBlending,
    side,
    uniforms: {
      uMode: { value: mode },
      uColor: { value: color },
      uMap: { value: map },
      uRect: { value: new THREE.Vector4(1, 1, 0, 0) },
      uParams: { value: new THREE.Vector2(1, 1) },
      uHorizonMap: { value: null },
      uHorizonRect: { value: new THREE.Vector4(1, 1, 0, 0) },
      uHorizonParams: { value: new THREE.Vector4(1, 1, 0, 0) },
      uHorizonBounds: { value: new THREE.Vector4(-1, 1, -1, 1) },
      uHorizonScale: { value: 1 },
      uGround: { value: 0 },
      uOrigin: { value: new THREE.Vector3(0, 0, 0) },
      uAxisA: { value: new THREE.Vector3(1, 0, 0) },
      uAxisB: { value: new THREE.Vector3(0, 0, 1) },
      uAxisUp: { value: new THREE.Vector3(0, 1, 0) },
      uSource: { value: new THREE.Vector4(0, 1, 0, 0) },
      uSourceSize: { value: new THREE.Vector2(0, 0) },
    },
  });
}

// Replaces the page's depth-from-opacity rule on a plane this script takes
// over: that rule writes depth at full alpha, and a ground shadow never
// should -- an alpha-0 fragment writing depth punches a hole in anything
// blended behind it.
function keepDepthOff() {
  this.material.depthWrite = false;
}

// The colour map a plane samples, onto that material's uMap. Taken off the
// material when it already carries exactly this texture (no second decode),
// otherwise fetched through the parser -- which is the normal case on a real
// export, where the manifest's atlas is in the file but no material points at
// it. A fetched texture is given the colour space the loader would have given
// it in a base-colour slot; getDependency hands back the loader's own cached
// instance, so a material already using it sees no change.
function attachMap(THREE, session, parser, plane, material) {
  if (plane.useOwnMap) {
    material.uniforms.uMap.value = plane.original.map;
    return;
  }
  if (plane.mapIndex === null) return;
  session.pending.push(parser.getDependency('texture', plane.mapIndex).then((texture) => {
    texture.colorSpace = THREE.SRGBColorSpace;
    material.uniforms.uMap.value = texture;
    session.owned.push(texture);
  }));
}

function build(THREE, model, gltf, environment) {
  const parser = gltf.parser;
  const manifest = readManifest(parser);
  if (!manifest || !Array.isArray(manifest.planes) || !manifest.planes.length) return null;
  const unit = finite(manifest.unit_scale, 1) > 0 ? finite(manifest.unit_scale, 1) : 1;
  const resolve = nodeResolver(THREE, parser, model);
  const faded = fadedMaterials(parser);
  const materialDefs = parser.json.materials || [];
  const session = {
    model,
    unit,
    environment,
    planes: [],
    batches: [],
    owned: [],
    pending: [],
    ready: null,
  };

  for (const record of manifest.planes) {
    const node = resolve(record.node);
    const mesh = node && (node.isMesh ? node : node.children.find((child) => child.isMesh));
    if (!mesh || Array.isArray(mesh.material) || !mesh.geometry) {
      console.warn(`shadow_rig: plane ${record.name} (node ${record.node}) has no mesh to drive`);
      continue;
    }
    const mode = record.type === 'horizon' ? 1 : 0;
    const horizon = mode === 1 ? horizonSpec(record, unit) : null;
    if (mode === 1 && !horizon) {
      console.warn(`shadow_rig: plane ${record.name}: horizon record not readable by this viewer`);
      continue;
    }
    const mapIndex = Number.isInteger(record.texture_index) ? record.texture_index : null;
    if (mode === 0 && mapIndex === null) {
      console.warn(`shadow_rig: plane ${record.name} has no colour map`);
      continue;
    }
    const original = mesh.material;
    const ownIndex = materialDefs[record.material]?.pbrMetallicRoughness?.baseColorTexture?.index;
    const atlasIndex = record.atlas?.texture_index;
    const source = resolve(record.source_node);
    const contact = resolve(record.contact_node);
    const plane = {
      record,
      node,
      mesh,
      original,
      mode,
      horizon,
      source,
      contact,
      follow: !!record.follow_source && !!source && !!contact,
      directional: record.source_type === 'directional',
      intensity: finite(record.intensity, 1),
      // The rect applies only when the colour map IS the atlas; a silhouette
      // bound on its own is the whole tile.
      rect: mode === 0 && Number.isInteger(atlasIndex) && atlasIndex === mapIndex
        ? flatRect(record.atlas.rect)
        : RECT_IDENTITY.slice(),
      mapIndex,
      useOwnMap: mode === 0 && ownIndex === mapIndex && !!original.map,
      faded: faded.has(record.material),
      extent: quadExtent(mesh.geometry),
      ground: finite(record.ground, 0) * unit,
      radius: finite(record.radius, 0.5) * unit,
      height: finite(record.height, 1) * unit,
      maxStretch: finite(record.max_stretch, 6),
      sourceSize: finite(record.source_size, 0) * unit,
      sourceAngle: finite(record.source_angle, 0),
      canvas: Array.isArray(record.canvas) && record.canvas.length === 4
        ? record.canvas.map(Number)
        : CANVAS_DEFAULT.slice(),
      material: null,
      batch: null,
      instance: -1,
      lastOpacity: NaN,
    };
    normaliseUv(mesh.geometry);
    session.planes.push(plane);
  }
  if (!session.planes.length) return null;

  // Batches: projected planes sharing one colour map, none of them faded.
  // Keyed on the TEXTURE THE SHADER SAMPLES (the manifest's texture_index),
  // never on whether the plane's material happens to carry it: a Maya
  // standardSurface loses its file texture through the FBX hop, so on a real
  // export every plane's material has no map at all while all three sample
  // one atlas -- gating on the material left three separate meshes where the
  // whole point of an atlas is one draw call.
  const groups = new Map();
  for (const plane of session.planes) {
    if (plane.mode !== 0 || plane.faded || plane.mapIndex === null) continue;
    if (!groups.has(plane.mapIndex)) groups.set(plane.mapIndex, []);
    groups.get(plane.mapIndex).push(plane);
  }
  for (const [key, members] of groups) {
    if (members.length < 2) continue;
    const first = members[0];
    const count = members.length;
    // The first plane's quad, cloned so the instance attributes are its own;
    // the DCC builds every plane centred on its pivot, which is what an
    // instance matrix places.
    const geometry = first.mesh.geometry.clone();
    const iRect = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4);
    const iParams = new THREE.InstancedBufferAttribute(new Float32Array(count * 2), 2);
    iParams.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('iRect', iRect);
    geometry.setAttribute('iParams', iParams);
    const material = makeMaterial(
      THREE, 0, first.useOwnMap ? first.original.map : null,
      colorOf(THREE, first.original), first.original.side,
    );
    attachMap(THREE, session, parser, first, material);
    const instanced = new THREE.InstancedMesh(geometry, material, count);
    instanced.name = `shadow_rig:${key}`;
    // Instances are placed anywhere on the ground; the geometry's own sphere
    // is one quad at the origin, so culling by it would drop them.
    instanced.frustumCulled = false;
    instanced.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    const batch = { key, mesh: instanced, members, iRect, iParams };
    members.forEach((plane, i) => {
      plane.batch = batch;
      plane.instance = i;
      plane.extent = first.extent;
      plane.mesh.visible = false;
      iRect.setXYZW(i, plane.rect[0], plane.rect[1], plane.rect[2], plane.rect[3]);
      iParams.setXY(i, plane.original.opacity, plane.intensity);
      plane.lastOpacity = plane.original.opacity;
    });
    model.add(instanced);
    session.batches.push(batch);
    session.owned.push(geometry, material);
  }

  // Everything else: one ShaderMaterial per plane, on the plane's own mesh.
  for (const plane of session.planes) {
    if (plane.batch) continue;
    const map = plane.useOwnMap ? plane.original.map : null;
    const material = makeMaterial(THREE, plane.mode, map, colorOf(THREE, plane.original), plane.original.side);
    if (plane.mode === 0) attachMap(THREE, session, parser, plane, material);
    const u = material.uniforms;
    u.uRect.value.fromArray(plane.rect);
    u.uParams.value.set(plane.original.opacity, plane.intensity);
    material.opacity = plane.original.opacity;
    plane.lastOpacity = plane.original.opacity;
    if (plane.horizon) {
      const h = plane.horizon;
      u.uHorizonRect.value.fromArray(h.rect);
      u.uHorizonParams.value.set(h.size, h.spans, h.levels, 0);
      u.uHorizonBounds.value.fromArray(h.bounds);
      // The map's own height scale (metres per 65535), never anything of
      // the plane's: the record's top-level max_stretch is a placement cap
      // the artist retunes without a re-bake.
      u.uHorizonScale.value = h.heightScale;
      // FULL widths: the reference halves the source's size once, in the
      // shader. Halving here as well would quarter every penumbra.
      u.uSourceSize.value.set(plane.sourceSize, plane.sourceAngle);
      session.pending.push(loadDataTexture(THREE, parser, h.textureIndex).then((texture) => {
        u.uHorizonMap.value = texture;
        session.owned.push(texture);
      }));
    }
    plane.material = material;
    plane.mesh.material = material;
    plane.mesh.onBeforeRender = plane.horizon
      ? function beforeHorizonDraw() {
        this.material.depthWrite = false;
        feedSource(THREE, plane);
      }
      : keepDepthOff;
    // The original is detached: this script now owns its disposal.
    session.owned.push(material, plane.original);
  }
  session.ready = Promise.all(session.pending).then(() => session);
  return session;
}

function teardown(session) {
  if (!session) return;
  for (const item of session.owned) {
    if (!item) continue;
    if (item.isTexture) {
      if (item !== session.environment) item.dispose();
    } else if (item.isMaterial) {
      for (const value of Object.values(item)) {
        if (value?.isTexture && value !== session.environment) value.dispose();
      }
      item.dispose();
    } else if (item.isBufferGeometry) {
      item.dispose();
    }
  }
  session.owned.length = 0;
  for (const batch of session.batches) batch.mesh.removeFromParent();
}

/* ----------------------------------------------------------- per frame --- */

const _m = { a: null, b: null, c: null, s: null, q: null, pos: null, scl: null, dir: null, up: null };

function scratch(THREE) {
  if (!_m.a) {
    _m.a = new THREE.Matrix4();
    _m.b = new THREE.Matrix4();
    _m.c = new THREE.Vector3();
    _m.s = new THREE.Vector3();
    _m.q = new THREE.Quaternion();
    _m.pos = new THREE.Vector3();
    _m.scl = new THREE.Vector3();
    _m.dir = new THREE.Vector3();
    _m.up = new THREE.Vector3(0, 1, 0);
  }
  return _m;
}

// An object's matrix in MODEL space, composed from fresh local matrices: the
// mixer writes position/quaternion/scale, and the renderer only recomposes
// them at draw time, which is after this hook.
function modelMatrixOf(object, model, out) {
  out.identity();
  for (let o = object; o && o !== model; o = o.parent) {
    if (o.matrixAutoUpdate) o.updateMatrix();
    out.premultiply(o.matrix);
  }
  return out;
}

// The Maya expression, applied: translate to the canvas centre, lifted off
// the ground; rotateY = atan2(ux, uz) so local +Z is the bearing (the -Z edge
// is the near edge, toward the light) and local +X is across it; scale the
// quad's own size to the canvas extents.
function place(THREE, session, plane) {
  const t = scratch(THREE);
  const { model, unit } = session;
  modelMatrixOf(plane.contact, model, t.a);
  t.c.setFromMatrixPosition(t.a);
  const contact = [t.c.x, t.c.y, t.c.z];
  modelMatrixOf(plane.source, model, t.a);
  let light;
  if (plane.directional) {
    t.dir.fromArray(DIRECTIONAL_AXIS).transformDirection(t.a);
    light = farPoint(contact, [t.dir.x, t.dir.y, t.dir.z], Math.max(plane.height, 2 * plane.radius));
  } else {
    t.s.setFromMatrixPosition(t.a);
    light = [t.s.x, t.s.y, t.s.z];
  }
  const shadow = shadowModel(contact, light, plane.ground, plane.radius, plane.height, plane.maxStretch);
  const { centre, along, across } = placement(shadow, plane.canvas);
  const yaw = Math.atan2(shadow.bearing[0], shadow.bearing[1]);
  t.pos.set(centre[0], plane.ground + GROUND_OFFSET * unit, centre[1]);
  t.q.setFromAxisAngle(t.up, yaw);
  t.scl.set(Math.max(1e-4, across / plane.extent.w), 1, Math.max(1e-4, along / plane.extent.d));
  t.a.compose(t.pos, t.q, t.scl);
  if (plane.batch) {
    plane.batch.mesh.setMatrixAt(plane.instance, t.a);
    plane.batch.mesh.instanceMatrix.needsUpdate = true;
    return;
  }
  const node = plane.node;
  if (node.parent && node.parent !== model) {
    modelMatrixOf(node.parent, model, t.b).invert();
    t.a.premultiply(t.b);
  }
  t.a.decompose(node.position, node.quaternion, node.scale);
  node.updateMatrix();
}

// The horizon shader's frame and source, in WORLD space, off world matrices
// the renderer has just refreshed (this runs from the mesh's onBeforeRender).
//
// The frame travels as an origin and three axes rather than a world-to-contact
// matrix: every binding is then a vec3, which is what a Maya .ogsfx uniform, a
// Blender push constant and a Unity instanced property are all proven to
// carry, and it takes a float4x4 out of Unity's instancing buffer. The axes
// come out normalised, which is the frame the map was baked in -- a scaled
// contact locator would not be one.
function feedSource(THREE, plane) {
  if (!plane.source || !plane.contact) return;
  const t = scratch(THREE);
  const u = plane.material.uniforms;
  const contact = plane.contact.matrixWorld;
  u.uOrigin.value.setFromMatrixPosition(contact);
  u.uAxisA.value.fromArray(plane.horizon.frameA).transformDirection(contact);
  u.uAxisB.value.fromArray(plane.horizon.frameB).transformDirection(contact);
  // The frame's up is the contact's own +Y, NOT a cross product of A and B.
  // The two DCCs bake with opposite bearing senses -- Maya's frame is (X, Z),
  // Blender's arrives as (X, -Z) -- so cross(B, A) is +Y for one and -Y for
  // the other, which puts every Blender-exported source below the horizon and
  // draws nothing. The map's up is the exporter's up, whatever the bearing
  // sense; the rig already assumes the contact stands on the world ground.
  u.uAxisUp.value.set(0, 1, 0).transformDirection(contact);
  // The ground plane's height along that up, from any point on it. The map's
  // intervals were baked as elevations seen FROM this plane, and the shared
  // body projects every fragment onto it.
  u.uGround.value = t.c.set(0, plane.ground, 0).sub(u.uOrigin.value).dot(u.uAxisUp.value);
  if (plane.directional) {
    // The uniform holds the direction the source SHINES (w = 0) -- the
    // reference's vocabulary; the shader negates it.
    t.dir.fromArray(DIRECTIONAL_AXIS).transformDirection(plane.source.matrixWorld);
    u.uSource.value.set(t.dir.x, t.dir.y, t.dir.z, 0);
  } else {
    t.s.setFromMatrixPosition(plane.source.matrixWorld);
    u.uSource.value.set(t.s.x, t.s.y, t.s.z, 1);
  }
}

function update(THREE, session) {
  const t = scratch(THREE);
  for (const plane of session.planes) {
    // The fade lives on the ORIGINAL material (the page bound the ramp there
    // before this script swapped it); follow it whenever it moves, and let
    // anything driving this material's own opacity directly win otherwise.
    const opacity = plane.original.opacity;
    if (plane.batch) {
      if (opacity !== plane.lastOpacity) {
        plane.batch.iParams.setX(plane.instance, opacity);
        plane.batch.iParams.needsUpdate = true;
        plane.lastOpacity = opacity;
      }
    } else {
      if (opacity !== plane.lastOpacity) {
        plane.material.opacity = opacity;
        plane.lastOpacity = opacity;
      }
      plane.material.uniforms.uParams.value.x = plane.material.opacity;
    }
    if (plane.follow) {
      place(THREE, session, plane);
    } else if (plane.batch) {
      // Not following (or unable to): the instance rides the node's own
      // imported keys, exactly as the hidden mesh would.
      modelMatrixOf(plane.node, session.model, t.a);
      plane.batch.mesh.setMatrixAt(plane.instance, t.a);
      plane.batch.mesh.instanceMatrix.needsUpdate = true;
    }
  }
}

export default function shadowRig(viewer) {
  const { THREE } = viewer;
  let session = null;

  // The page imports its scripts and loads the asset concurrently and does
  // not await the imports first, so a very small deliverable can be on screen
  // before this module has arrived -- and everything here needs the loader's
  // parser, which only the 'load' event carries. Said once, in the console,
  // rather than left to look like a broken export: the next push is caught.
  if (viewer.model) {
    console.warn('shadow_rig: a model loaded before this script; its shadow rigs stay still until the next push');
  }

  viewer.on('load', ({ model, gltf }) => {
    teardown(session);
    session = null;
    try {
      session = build(THREE, model, gltf, viewer.scene.environment);
    } catch (error) {
      console.warn('shadow_rig: could not read the shadow manifest', error);
    }
    // Hung on the model so it dies with it, and so a test can read what this
    // script made of the file without the script exposing anything global.
    if (session) model.userData.shadowRig = session;
  });

  viewer.on('frame', () => {
    if (session) update(THREE, session);
  });
}
