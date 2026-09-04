/*
  Shadow rig — drive the DCC shadow planes from the model's own source and
  contact nodes, per frame, with the maps the conversion bound.

  A shadow rig (mayatk / blendertk `ShadowRig`) is a ground quad under a prop
  whose transform a DCC expression re-places from the light every frame, and
  whose texture is either a rasterized silhouette (the PROJECTED type: the
  quad's UVs read one tile of an atlas) or a HORIZON map (a log-polar bake of
  the prop's horizon as seen from every ground point, in azimuth bins, two
  layers per bin: the members touching the ground and the overhangs). Neither
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
// RGB times the material colour. Mode 1 (horizon): the contract's evaluation,
// which is NOT written here -- the body between the markers below is spliced
// in from `pythontk/geo_utils/shadow_horizon.glsl` by
// `m3trik/scripts/sync_shadow_shaders.py`, and `HorizonMap.alpha` beside it is
// the oracle `test_shadow_web.py` renders this page against. What IS this
// file's own is the host half: the uniforms, and `SH_Fetch` -- glTF's
// top-left texture origin is a per-engine fact and the shared body never
// computes a texture address.
const FRAGMENT = /* glsl */ `
uniform int uMode;
uniform vec3 uColor;
uniform sampler2D uMap;
uniform sampler2D uHorizonMap;
uniform vec4 uHorizonRect;     // the tile block inside its texture, glTF top-left: sx, sy, ox, oy
uniform vec4 uHorizonParams;   // bins, cols, rows, layers
uniform vec4 uHorizonRange;    // r_min, r_max (metres), tile width, tile height (texels)
uniform float uMaxStretch;     // the BAKE's max_stretch: the cotangent scale the R/G channels were divided by
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

// The shared body's texel hook. The map is a grid of tiles inside a block the
// atlas packer placed at uHorizonRect.zw; row 0 of a tile is its r_min ring,
// which is the PNG's top row -- and three.js uploads a data texture unflipped,
// so a texel address is the block origin plus the cell plus the texel.
vec4 SH_Fetch(int col, int row, int xi, int yi) {
  ivec2 size = textureSize(uHorizonMap, 0);
  ivec2 tile = ivec2(uHorizonRange.zw);
  ivec2 origin = ivec2(uHorizonRect.zw * vec2(size) + 0.5);
  return texelFetch(uHorizonMap, origin + ivec2(col * tile.x + xi, row * tile.y + yi), 0);
}

// >>> BEGIN GENERATED shadow_horizon body
// ===========================================================================
//  shadow_horizon.glsl -- the coverage-aware horizon evaluation.
//
//  SINGLE SOURCE OF TRUTH. Every consumer -- the WebXR viewer's GLSL, Unity's
//  HLSL, Maya's .ogsfx/.fx and Blender's gpu shader -- runs THIS text. A
//  consumer that is a Python process when it needs the shader assembles it
//  through ShadowHorizon.shader_source(); one that is not (a browser, a
//  Unity project) carries a generated mirror that sync_shadow_shaders.py
//  writes and --check guards. NEVER hand-edit a mirror.
//
//  The normative behaviour is HorizonMap.alpha in shadow_horizon.py, term
//  for term -- that reference is the oracle every engine test compares
//  against, so a change here is a change there and both move together.
//
//  ---------------------------------------------------------------- the host
//  Define these BEFORE pasting this body:
//
//    SH_HLSL                 -- when the target speaks HLSL (else GLSL)
//    vec4 SH_Fetch(int col, int row, int xi, int yi)
//                            -- one texel of the tile at grid cell
//                               (col, row), texel (xi, yi) inside it, with
//                               (0, 0) the tile's r_min / bearing-zero
//                               corner. The body never computes a texture
//                               address: glTF's top-left origin, Unity's
//                               double row flip and Maya's own convention are
//                               irreducibly per-engine, and this is where
//                               that divergence belongs.
//
//  The host also supplies the uniforms and calls ShAlpha. Nothing in here
//  samples a texture, reads a uniform or names an engine.
// ===========================================================================

#ifdef SH_HLSL
#define vec2  float2
#define vec3  float3
#define vec4  float4
#define mix   lerp
#define SH_ATAN2(y, x) atan2(y, x)
#else
#define SH_ATAN2(y, x) atan(y, x)
#endif

//  mod for floats, spelled once: HLSL's fmod truncates toward zero and so
//  disagrees with GLSL's on a negative dividend -- which is exactly the case
//  the bearing wrap hits.
#define SH_MOD(x, y) ((x) - (y) * floor((x) / (y)))

#define SH_TWO_PI  6.283185307179586
#define SH_SUBBINS 16
//  Sub-bins of the three bins the coverage integral spans: k - 1, k, k + 1.
//  A source disc may straddle a bin edge, and truncating that span clips a
//  wide penumbra -- so the whole three-bin window is walked.
#define SH_SPAN    48

// ---------------------------------------------------------------- the grid
//  Everything the layout needs, built once per fragment. ground is the
//  height of the ground plane along the map's up axis IN THE MAP'S FRAME:
//  the intervals were baked as elevations seen from that plane, so the
//  fragment's own height is replaced by it (see ShAlpha).
struct ShGrid
{
    int   bins;
    int   cols;        // tiles per row of the PNG's tile grid
    int   layers;
    int   tileW;
    int   tileH;
    float rMin;
    float rMax;
    float maxStretch;  // the BAKE's cot scale: R, G hold cot(angle) / this
    float ground;
};

ShGrid ShMakeGrid(int bins, int cols, int layers, int tileW, int tileH,
                  float rMin, float rMax, float maxStretch, float ground)
{
    ShGrid g;
    g.bins = max(1, bins);
    g.cols = max(1, cols);
    g.layers = max(1, layers);
    g.tileW = max(1, tileW);
    g.tileH = max(1, tileH);
    g.rMin = max(rMin, 1e-6);
    g.rMax = max(rMax, g.rMin * 1.000001);
    g.maxStretch = max(maxStretch, 1e-3);
    g.ground = ground;
    return g;
}

// -------------------------------------------------------------- primitives
float ShOverlap(float a0, float a1, float b0, float b1)
{
    return max(0.0, min(a1, b1) - max(a0, b0));
}

int ShClampI(int x, int lo, int hi) { return max(lo, min(hi, x)); }

//  cot(angle) for an elevation in (0, pi/2]. Deliberately UNCAPPED: the
//  blocked interval is already bounded by maxStretch, and clamping the
//  source's own cotangent instead would widen a grazing disc's overlap with
//  it and darken the far field.
float ShCot(float angle)
{
    float a = max(angle, 1e-6);
    return cos(a) / sin(a);
}

//  The 16 sub-bin occupancy bits of a texel: B holds sub-bins 0..7, A 8..15.
int ShMask(vec4 texel)
{
    return int(floor(texel.b * 255.0 + 0.5))
         | (int(floor(texel.a * 255.0 + 0.5)) << 8);
}

int ShBit(int mask, int j) { return (mask >> j) & 1; }

//  Sub-bin j of the three-bin window, j in [0, SH_SPAN): bin k - 1 first,
//  then k, then k + 1.
int ShBit48(int mP, int mA, int mN, int j)
{
    if (j < SH_SUBBINS) return ShBit(mP, j);
    if (j < 2 * SH_SUBBINS) return ShBit(mA, j - SH_SUBBINS);
    return ShBit(mN, j - 2 * SH_SUBBINS);
}

//  The coverage centre of a mask as a bin fraction: the midpoint of the first
//  and last set sub-bins. An empty mask reports 0.5 -- the bin's own centre,
//  which leaves side decided by the bearing alone.
float ShMaskMid(int mask)
{
    if (mask == 0) return 0.5;
    int first = 0;
    int last = 0;
    bool seen = false;
    for (int j = 0; j < SH_SUBBINS; ++j)
    {
        if (ShBit(mask, j) == 1)
        {
            if (!seen) { first = j; seen = true; }
            last = j;
        }
    }
    return 0.5 * float(first + last + 1) / float(SH_SUBBINS);
}

//  Runs of set bits (0 -> 1 transitions) in a mask.
int ShRunCount(int mask)
{
    int runs = 0;
    int prev = 0;
    for (int j = 0; j < SH_SUBBINS; ++j)
    {
        int bit = ShBit(mask, j);
        if (bit == 1 && prev == 0) runs += 1;
        prev = bit;
    }
    return runs;
}

//  Which run sub-bin j falls in: 1 for the first, 2 for any later one, and
//  0 BEFORE the first run -- never floored to 1. A sub-bin no run covers is
//  not "in run 1"; it is uncovered, and the coverage term already says so.
int ShRunAt(int mask, int j)
{
    int runs = 0;
    int prev = 0;
    for (int i = 0; i < SH_SUBBINS; ++i)
    {
        if (i > j) break;
        int bit = ShBit(mask, i);
        if (bit == 1 && prev == 0) runs += 1;
        prev = bit;
    }
    return runs;
}

//  Bilinear R, G over the taps covered selects, falling back to the NEAREST
//  tap's own value when none is. An empty texel encodes the zenith, so
//  letting it into the blend pulls a neighbour's interval toward it and
//  lengthens the shadow -- which is what covered exists to prevent.
vec2 ShInterval(vec4 taps[4], vec4 w, int nearest, vec4 covered)
{
    vec2 acc = vec2(0.0, 0.0);
    float total = 0.0;
    int i;
    for (i = 0; i < 4; ++i)
    {
        float wi = w[i] * covered[i];
        acc += wi * vec2(taps[i].r, taps[i].g);
        total += wi;
    }
    if (total > 1e-9) return acc / total;
    return vec2(taps[nearest].r, taps[nearest].g);
}

//  One texel of one tile. The tile index is layer * bins + k with k
//  wrapped; the grid cell it lands in is this body's business, the texture
//  address the host's.
vec4 ShTexel(ShGrid g, int layer, int k, int xi, int yi)
{
    // GLSL leaves % UNDEFINED when either operand is negative, and the
    // caller asks for k - 1. k is never more than one bin outside [0, bins),
    // so wrap it by comparison instead of trusting a driver's remainder.
    int kk = k < 0 ? k + g.bins : (k >= g.bins ? k - g.bins : k);
    int tile = layer * g.bins + kk;
    return SH_Fetch(tile % g.cols, tile / g.cols, xi, yi);
}

// ------------------------------------------------------------ one layer
//  HorizonMap._layer_alpha, term for term: the coverage of the source's
//  sub-bin span over the four taps, times the source's overlap with the
//  occluder's elevation interval.
float ShLayerAlpha(ShGrid g, int layer, float u, float v, int k, float s,
                   float elev, float rho, float stepA)
{
    // -- the four texels around (u, v): x wraps, y clamps to the tile's rows
    float x = u * float(g.tileW) - 0.5;
    float xf = floor(x);
    float fx = x - xf;
    // u is in [0, 1), so floor(x) lands in [-1, tileW - 1]: the column wrap
    // is one comparison, not a modulus. Going through a float mod and back
    // through int() would round a whole texel the wrong way at the seam.
    int xi0 = int(xf);
    int x0 = xi0 < 0 ? xi0 + g.tileW : xi0;
    int x1 = x0 + 1 >= g.tileW ? 0 : x0 + 1;
    float yc = clamp(v * float(g.tileH) - 0.5, 0.0, float(g.tileH) - 1.0);
    int y0 = int(floor(yc));
    int y1 = min(y0 + 1, g.tileH - 1);
    float fy = yc - float(y0);
    vec4 w = vec4((1.0 - fx) * (1.0 - fy), fx * (1.0 - fy),
                  (1.0 - fx) * fy, fx * fy);
    // Ties break toward the HIGHER texel. Not cosmetic: the nearest tap alone
    // decides the grounded run index, so a tie broken the other way selects a
    // different BRANCH, not merely a rounded value.
    int nearest = (fx >= 0.5 ? 1 : 0) + (fy >= 0.5 ? 2 : 0);
    // Declared once for the whole function: HLSL hoists a for-loop's control
    // variable to the enclosing scope, so a second for-loop declaring i here is a
    // redeclaration it warns about.
    int i;
    int cx[4];
    int cy[4];
    cx[0] = x0; cy[0] = y0;
    cx[1] = x1; cy[1] = y0;
    cx[2] = x0; cy[2] = y1;
    cx[3] = x1; cy[3] = y1;

    // -- bin k, whose coverage centre decides which neighbour matters
    vec4 tA[4];
    int mA[4];
    for (i = 0; i < 4; ++i)
    {
        tA[i] = ShTexel(g, layer, k, cx[i], cy[i]);
        mA[i] = ShMask(tA[i]);
    }
    int maskA = mA[nearest];
    float midA = ShMaskMid(maskA);
    bool hasA = maskA != 0;
    int side = s > midA ? 1 : -1;

    // -- the neighbour on the light's side: the one the interval lerps toward
    vec4 tB[4];
    int mB[4];
    for (i = 0; i < 4; ++i)
    {
        tB[i] = ShTexel(g, layer, k + side, cx[i], cy[i]);
        mB[i] = ShMask(tB[i]);
    }
    int maskB = mB[nearest];
    bool hasB = maskB != 0;
    float midB = ShMaskMid(maskB);

    // -- coverage: the source disc's sub-bin span against each tap's bits,
    //    over the 48 sub-bins of k - 1, k, k + 1
    float rhoSub = (rho / stepA) * float(SH_SUBBINS);
    float centre = s * float(SH_SUBBINS) + float(SH_SUBBINS);
    float discLo = centre - rhoSub;
    float discHi = centre + rhoSub;
    float width = 2.0 * rhoSub;
    // NOT named point: that is a reserved word in HLSL.
    bool isPoint = width <= 1e-9;
    // A point source's sub-bin is always inside bin k -- centre lands in
    // [16, 32) -- and only bins k and k + side carry an interval, so the bin
    // on the far side is read ONLY for a disc that can spill past the edge.
    // That is what keeps a point source at sixteen texel loads per fragment
    // and a disc at twenty-four, the budget the contract states.
    int mC[4];
    mC[0] = 0; mC[1] = 0; mC[2] = 0; mC[3] = 0;
    if (!isPoint)
    {
        for (i = 0; i < 4; ++i)
            mC[i] = ShMask(ShTexel(g, layer, k - side, cx[i], cy[i]));
    }
    int centreBin = ShClampI(int(floor(centre)), 0, SH_SPAN - 1);
    float covPhi = 0.0;
    for (i = 0; i < 4; ++i)
    {
        // The window runs k - 1, k, k + 1; B sits on the light's side of k
        // and C opposite it, so which of them is which follows side.
        int mLo = side > 0 ? mC[i] : mB[i];
        int mHi = side > 0 ? mB[i] : mC[i];
        float cov;
        if (isPoint)
        {
            cov = float(ShBit48(mLo, mA[i], mHi, centreBin));
        }
        else
        {
            float total = 0.0;
            for (int j = 0; j < SH_SPAN; ++j)
            {
                total += float(ShBit48(mLo, mA[i], mHi, j))
                       * ShOverlap(discLo, discHi, float(j), float(j + 1));
            }
            cov = total / width;
        }
        covPhi += w[i] * cov;
    }
    covPhi = clamp(covPhi, 0.0, 1.0);
    // Every tap of every neighbouring bin is empty here: no occluder, and no
    // interval worth reading. Gating on the NEAREST tap alone would instead
    // zero a texel its neighbours cover.
    if (covPhi <= 0.0) return 0.0;

    // -- interval: bin k's sample (taken at its coverage centre) lerped
    //    toward the neighbour on the light's side when that bin is covered
    //    too; a hold, never a fade, when only one of them is
    float gap = (midB + float(side)) - midA;
    float t = clamp((s - midA) / (abs(gap) > 1e-9 ? gap : 1.0), 0.0, 1.0);
    t = (hasA && hasB) ? t : (hasA ? 0.0 : 1.0);

    vec4 covA = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 covB = vec4(0.0, 0.0, 0.0, 0.0);
    for (i = 0; i < 4; ++i)
    {
        covA[i] = mA[i] != 0 ? 1.0 : 0.0;
        covB[i] = mB[i] != 0 ? 1.0 : 0.0;
    }
    vec2 intA = ShInterval(tA, w, nearest, covA);
    vec2 intB = ShInterval(tB, w, nearest, covB);

    float cotLo;
    float cotHi;
    if (layer == 0)
    {
        // Grounded: lo is always the ground, R carries the FIRST run's top
        // and G any later run's -- a second leg in the same bin, which keeps
        // its own shadow length. The light's sub-bin picks the run.
        int j = ShClampI(int(floor(s * float(SH_SUBBINS))), 0, SH_SUBBINS - 1);
        int run = ShRunAt(maskA, j);
        // G is written 0 on a one-run texel, so the later-run top blends only
        // over the taps that HAVE a second run; letting a one-run neighbour
        // in drags it toward the zenith and over-lengthens the shadow.
        vec4 multi = vec4(0.0, 0.0, 0.0, 0.0);
        for (i = 0; i < 4; ++i)
            multi[i] = ShRunCount(mA[i]) >= 2 ? 1.0 : 0.0;
        float laterG = ShInterval(tA, w, nearest, multi).y;
        bool later = (run >= 2) && (laterG > 0.0);
        float hiFirst = mix(intA.x, intB.x, t);
        cotLo = g.maxStretch;
        cotHi = (later ? laterG : hiFirst) * g.maxStretch;
    }
    else
    {
        // Floating: R = cot(lo), G = cot(hi) -- the overhang's own band.
        vec2 lohi = mix(intA, intB, t);
        cotLo = lohi.x * g.maxStretch;
        cotHi = lohi.y * g.maxStretch;
    }

    // -- the elevation test in cotangent space: blocked while
    //    cot(hi) <= cot(e) <= cot(lo); a disc spans [cot(e + rho), cot(e - rho)]
    float cNear = ShCot(elev + rho);
    float cFar = ShCot(elev - rho);
    float spanE = cFar - cNear;
    float covE;
    if (spanE > 1e-9)
    {
        covE = ShOverlap(cNear, cFar, cotHi, cotLo) / spanE;
    }
    else
    {
        float mid = 0.5 * (cNear + cFar);
        covE = (mid >= cotHi && mid <= cotLo && cotLo > cotHi) ? 1.0 : 0.0;
    }
    return covPhi * covE;
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
//  -- full widths, halved here and nowhere else.
float ShAlpha(ShGrid g, vec3 worldPos, vec3 origin,
              vec3 axisA, vec3 axisB, vec3 axisUp,
              vec4 source, vec2 sourceSize)
{
    vec3 d = worldPos - origin;
    float px = dot(d, axisA);
    float pz = dot(d, axisB);
    float r = length(vec2(px, pz));
    if (r > g.rMax) return 0.0;

    // The fragment's own height is REPLACED by the ground, not merely
    // ignored: the map's intervals are elevations as seen from the ground
    // plane (the bake marches from height 0), while a rig lifts its plane by
    // its own ground offset. Forming L from the fragment measures against a
    // plane the map was never baked on.
    vec3 P = vec3(px, g.ground, pz);
    vec3 sv = source.w > 0.5 ? (source.xyz - origin) : source.xyz;
    vec3 sf = vec3(dot(sv, axisA), dot(sv, axisUp), dot(sv, axisB));
    vec3 L = source.w > 0.5 ? (sf - P) : -sf;

    float dist = max(length(L), 1e-12);
    float elev = SH_ATAN2(L.y, length(vec2(L.x, L.z)));
    if (elev <= 0.0) return 0.0;   // the source is at or below the ground
    float rho = source.w > 0.5
        ? asin(clamp(0.5 * sourceSize.x / dist, 0.0, 1.0))
        : 0.5 * sourceSize.y;

    float phi = SH_ATAN2(L.z, L.x);
    phi = SH_MOD(phi, SH_TWO_PI);
    float theta = SH_ATAN2(pz, px);
    theta = SH_MOD(theta, SH_TWO_PI);

    float u = theta / SH_TWO_PI;
    float v = clamp(log(max(r, 1e-12) / g.rMin) / log(g.rMax / g.rMin),
                    0.0, 1.0);

    float stepA = SH_TWO_PI / float(g.bins);
    float fk = phi / stepA;
    int k = int(floor(fk));
    float s = fk - float(k);

    // The two layers block independently: a table top spans a whole bin while
    // its legs are thin, so one interval per bin cannot serve both.
    float grounded = ShLayerAlpha(g, 0, u, v, k, s, elev, rho, stepA);
    float floating = g.layers >= 2
        ? ShLayerAlpha(g, 1, u, v, k, s, elev, rho, stepA)
        : 0.0;
    return 1.0 - (1.0 - grounded) * (1.0 - floating);
}
// <<< END GENERATED shadow_horizon body

void main() {
  float opacity = vParams.x;
  float intensity = vParams.y;
  if (uMode == 1) {
    ShGrid g = ShMakeGrid(
      int(uHorizonParams.x + 0.5), int(uHorizonParams.y + 0.5), int(uHorizonParams.w + 0.5),
      int(uHorizonRange.z + 0.5), int(uHorizonRange.w + 0.5),
      uHorizonRange.x, uHorizonRange.y, uMaxStretch, uGround);
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

// ShadowModel.placement: the canvas fractions (near edge in base radii from
// the anchor, far edge in top radii from the head, sides as fractions of the
// width) to the quad's centre and its extents along and across the bearing.
function placement(model, fractions) {
  const [u0, u1, w0, w1] = fractions;
  const uLo = u0 * model.base;
  const uHi = model.reach + u1 * model.top;
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
// cannot read it (an encoding, mapping or layer count it does not know).
function horizonSpec(record, unit) {
  const h = record.horizon;
  if (!h || typeof h !== 'object') return null;
  const bins = Number(h.bins);
  const layers = h.layers === undefined ? 2 : Number(h.layers);
  const tile = Array.isArray(h.tile) && h.tile.length === 2 ? h.tile.map(Number) : null;
  const cols = Math.ceil(Math.sqrt(2 * bins));
  const layout = Array.isArray(h.layout) && h.layout.length === 2
    ? h.layout.map(Number)
    : [cols, Math.ceil((2 * bins) / cols)];
  const rMin = Number(h.r_min) * unit;
  const rMax = Number(h.r_max) * unit;
  const ok = Number.isInteger(h.texture_index) && bins >= 1 && layers === 2 && tile
    && (h.encoding ?? 1) === 1 && (h.mapping ?? 'logpolar') === 'logpolar'
    && rMin > 0 && rMax > rMin;
  if (!ok) return null;
  return {
    textureIndex: h.texture_index,
    bins,
    layers,
    tile,
    layout,
    rMin,
    rMax,
    // The cotangent scale the map was BAKED at. Null when the block does not
    // carry it (a map baked before the field existed), and the record's
    // top-level max_stretch stands in.
    maxStretch: Number(h.max_stretch) > 0 ? Number(h.max_stretch) : null,
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
      uHorizonParams: { value: new THREE.Vector4(1, 1, 1, 2) },
      uHorizonRange: { value: new THREE.Vector4(0.1, 1, 1, 1) },
      uMaxStretch: { value: 6 },
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
      u.uHorizonParams.value.set(h.bins, h.layout[0], h.layout[1], h.layers);
      u.uHorizonRange.value.set(h.rMin, h.rMax, h.tile[0], h.tile[1]);
      // The DECODE scale, which is the map's own and not the plane's: R and G
      // hold cot(elevation) / max_stretch as of the BAKE, while the record's
      // top-level max_stretch is the live placement cap a keyable attribute
      // an artist can retune afterwards -- decoding with that one scales
      // every shadow length by the ratio between them. An older map that
      // carries no bake scale falls back to it, which is what it was baked
      // with.
      u.uMaxStretch.value = h.maxStretch === null ? plane.maxStretch : h.maxStretch;
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
  // The frame's up: bearing runs from A toward B, so B x A is the vertical
  // (X x Z is -Y in a right-handed Y-up file).
  u.uAxisUp.value.crossVectors(u.uAxisB.value, u.uAxisA.value).normalize();
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
