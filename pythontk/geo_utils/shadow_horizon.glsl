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
