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
