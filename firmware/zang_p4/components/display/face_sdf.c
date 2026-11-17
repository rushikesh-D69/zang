/*
 * face_sdf.c - float32 port of ai_stack/zang_ai/face/render.py. Pure C (no IDF).
 *
 * The Python file is canonical. Keep the structure and the constants identical: each shape below is one signed
 * distance d (negative inside), its coverage is clamp(0.5 - d, 0, 1), and the pixel's ink alpha is the maximum of
 * the shape coverages (render.py: a = max(a, cover(d))). Because cover() is monotonic, taking the max of several
 * covers equals the cover of the min distance, so render.py's "np.minimum" unions become separate shapes here.
 */
#include "face_sdf.h"

#include <math.h>
#include <string.h>

#define PANEL 240
#define HALF  120.0f
#define DEG2RAD 0.017453292519943295f

typedef enum { K_EYE, K_HEART, K_SEG, K_ELL, K_OVAL, K_ARC } shape_kind_t;

typedef struct {
    uint8_t kind;
    int x0, y0, x1, y1; /* half-open pixel box outside which coverage is exactly 0 */
    float weight;       /* multiplies the coverage (brows) */
    union {
        struct { /* K_EYE */
            float cx, cy, w, h, r;
            uint8_t lid_top, lid_bot;
            float k, y_line, inv_n; /* slanted upper lid */
            float R, c2y;           /* squint arc */
        } eye;
        struct { /* K_HEART */
            float cx, cy, s;
        } heart;
        struct { /* K_SEG */
            float ax, ay, ex, ey, r, den;
        } seg;
        struct { /* K_ELL, K_OVAL */
            float cx, cy, rx, ry, m;
            uint8_t flat;
            float flat_y;
        } ell;
        struct { /* K_ARC */
            float ex, my, t, R, ccy, e_y, sgn;
        } arc;
    } u;
} shape_t;

#define MAX_SHAPES 20

typedef struct {
    shape_t sh[MAX_SHAPES];
    int n;
    shape_t blush[2];
    int n_blush;
    float blush_k;      /* 0.5 * blush */
    face_rect_t bbox;
    float bg[3], ink[3], pink[3];
    uint8_t has_ink; /* any shape or blush at all */
} scene_t;

/* ------------------------------------------------------------------ small helpers */
static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline float cover(float d) { return clampf(0.5f - d, 0.0f, 1.0f); }
static inline float hyp(float a, float b) { return sqrtf(a * a + b * b); }

static void hsv_to_rgb(float h, float s, float v, float out[3])
{ /* colorsys.hsv_to_rgb */
    if (s == 0.0f) {
        out[0] = out[1] = out[2] = v;
        return;
    }
    float h6 = h * 6.0f;
    int i = (int)h6;
    float f = h6 - (float)i;
    float p = v * (1.0f - s), q = v * (1.0f - s * f), t = v * (1.0f - s * (1.0f - f));
    switch (i % 6) {
    case 0: out[0] = v, out[1] = t, out[2] = p; break;
    case 1: out[0] = q, out[1] = v, out[2] = p; break;
    case 2: out[0] = p, out[1] = v, out[2] = t; break;
    case 3: out[0] = p, out[1] = q, out[2] = v; break;
    case 4: out[0] = t, out[1] = p, out[2] = v; break;
    default: out[0] = v, out[1] = p, out[2] = q; break;
    }
}

/* Pixel box covering coordinate range [lo, hi] (panel-centre coordinates), padded for safety. */
static void set_box(shape_t *s, float lox, float hix, float loy, float hiy)
{
    s->x0 = (int)floorf(lox + HALF - 0.5f) - 1;
    s->x1 = (int)ceilf(hix + HALF - 0.5f) + 2;
    s->y0 = (int)floorf(loy + HALF - 0.5f) - 1;
    s->y1 = (int)ceilf(hiy + HALF - 0.5f) + 2;
    if (s->x0 < 0) s->x0 = 0;
    if (s->y0 < 0) s->y0 = 0;
    if (s->x1 > PANEL) s->x1 = PANEL;
    if (s->y1 > PANEL) s->y1 = PANEL;
}

/* ------------------------------------------------------------------ shape constructors */
static shape_t *new_shape(scene_t *sc, shape_kind_t k, float weight)
{
    if (sc->n >= MAX_SHAPES) return NULL;
    shape_t *s = &sc->sh[sc->n++];
    memset(s, 0, sizeof(*s));
    s->kind = k;
    s->weight = weight;
    return s;
}

static void add_seg(scene_t *sc, float ax, float ay, float bx, float by, float r, float weight)
{
    shape_t *s = new_shape(sc, K_SEG, weight);
    if (!s) return;
    s->u.seg.ax = ax, s->u.seg.ay = ay, s->u.seg.ex = bx - ax, s->u.seg.ey = by - ay, s->u.seg.r = r;
    s->u.seg.den = fmaxf(s->u.seg.ex * s->u.seg.ex + s->u.seg.ey * s->u.seg.ey, 1e-9f);
    float m = r + 0.5f;
    set_box(s, fminf(ax, bx) - m, fmaxf(ax, bx) + m, fminf(ay, by) - m, fmaxf(ay, by) + m);
}

static void fill_ell(shape_t *s, float cx, float cy, float rx, float ry)
{
    s->u.ell.cx = cx, s->u.ell.cy = cy, s->u.ell.rx = rx, s->u.ell.ry = ry, s->u.ell.m = fminf(rx, ry);
    /* d = (k - 1) * m < 0.5  <=>  k < 1 + 0.5 / m  (exact on the axes) */
    float g = 1.0f + 0.5f / s->u.ell.m;
    set_box(s, cx - rx * g, cx + rx * g, cy - ry * g, cy + ry * g);
}

static void add_ell(scene_t *sc, float cx, float cy, float rx, float ry)
{
    shape_t *s = new_shape(sc, K_ELL, 1.0f);
    if (s) fill_ell(s, cx, cy, rx, ry);
}

static void add_eye(scene_t *sc, const face_frame_t *f, int side, float open_scale)
{
    const float *p = f->p;
    shape_t *s = new_shape(sc, K_EYE, 1.0f);
    if (!s) return;
    float cx = (float)side * p[FP_EYE_DX] + p[FP_GAZE_X], cy = p[FP_EYE_Y] + p[FP_GAZE_Y];
    float w = p[FP_EYE_W];
    float h = fmaxf(p[FP_EYE_H] * p[FP_EYE_OPEN] * open_scale, 4.0f);
    float r = p[FP_EYE_ROUND] * fminf(w, h) / 2.0f;
    s->u.eye.cx = cx, s->u.eye.cy = cy, s->u.eye.w = w, s->u.eye.h = h, s->u.eye.r = r;
    if (p[FP_LID_TOP] > 0.001f) {
        float k = tanf(p[FP_LID_ANGLE] * DEG2RAD) * (float)(-side);
        s->u.eye.lid_top = 1;
        s->u.eye.k = k;
        s->u.eye.y_line = cy - h / 2.0f + p[FP_LID_TOP] * h;
        s->u.eye.inv_n = 1.0f / sqrtf(1.0f + k * k);
    }
    if (p[FP_LID_BOT] > 0.001f) {
        float R = w * 0.9f;
        s->u.eye.lid_bot = 1;
        s->u.eye.R = R;
        s->u.eye.c2y = cy + h / 2.0f + R - p[FP_LID_BOT] * h;
    }
    /* both lid cuts only ever remove area from the rounded box, so its expanded box bounds the coverage */
    set_box(s, cx - w / 2.0f - 0.5f, cx + w / 2.0f + 0.5f, cy - h / 2.0f - 0.5f, cy + h / 2.0f + 0.5f);
}

static void add_heart(scene_t *sc, float cx, float cy, float size)
{
    shape_t *s = new_shape(sc, K_HEART, 1.0f);
    if (!s) return;
    s->u.heart.cx = cx, s->u.heart.cy = cy, s->u.heart.s = size;
    /* lobes reach |dx| = 0.604 s and 0.604 s above the centre; the tip is 0.5 s below it (exact distance) */
    float m = 0.61f * size + 0.5f;
    set_box(s, cx - m, cx + m, cy - m, cy + 0.5f * size + 0.5f);
}

static void add_mouth(scene_t *sc, const float *p)
{
    float w0 = p[FP_MOUTH_W], my = p[FP_MOUTH_Y], t = p[FP_MOUTH_THICK];
    if (w0 < 1.0f) return;
    float op = p[FP_MOUTH_OPEN];
    float sg = p[FP_MOUTH_CURVE] * w0 * 0.35f; /* sagitta; + = middle lower = smile */
    float w = w0 * fmaxf(0.0f, 1.0f - 1.6f * op);
    if (w >= 1.0f) {
        float sga = sg * (w / w0), ex = w / 2.0f;
        if (fabsf(sga) < 0.5f) {
            add_seg(sc, -ex, my, ex, my, t / 2.0f, 1.0f);
        } else {
            shape_t *s = new_shape(sc, K_ARC, 1.0f);
            if (s) {
                float asg = fabsf(sga);
                float R = (ex * ex + sga * sga) / (2.0f * asg);
                s->u.arc.ex = ex, s->u.arc.my = my, s->u.arc.t = t, s->u.arc.R = R;
                s->u.arc.sgn = copysignf(1.0f, sga);
                s->u.arc.ccy = my + sga / 2.0f - copysignf(R, sga);
                s->u.arc.e_y = my - sga / 2.0f;
                /* render.py colours every point with |x| <= ex on the arc's side of the circle centre whose radial
                 * distance to the circle is < t/2 + 0.5: that band reaches from the circle's far point
                 * (middle of the mouth) back towards its centre height until it leaves |x| <= ex. */
                float band = t / 2.0f + 0.5f;
                float rin = R - band;
                float y_near = s->u.arc.ccy + s->u.arc.sgn * (rin > ex ? sqrtf(rin * rin - ex * ex) : 0.0f);
                float y_far = s->u.arc.ccy + s->u.arc.sgn * (R + band);
                float lo = fminf(y_near, y_far), hi = fmaxf(y_near, y_far);
                float e_y = s->u.arc.e_y; /* the two round end caps */
                lo = fminf(lo, e_y - band), hi = fmaxf(hi, e_y + band);
                set_box(s, -ex - band, ex + band, lo - 1.0f, hi + 1.0f);
            }
        }
    }
    if (op > 0.02f) {
        float ry = 4.0f + 22.0f * op;
        float rx = w0 * (0.30f + 0.2f * op);
        float cy = my + sg * 0.3f;
        shape_t *s = new_shape(sc, K_OVAL, 1.0f);
        if (s) {
            fill_ell(s, 0.0f, cy, rx, ry);
            if (p[FP_MOUTH_CURVE] > 0.3f && op > 0.3f) { /* laugh: cut the top flat */
                s->u.ell.flat = 1;
                s->u.ell.flat_y = cy - ry * 0.25f;
            }
        }
    }
}

static void add_symbol(scene_t *sc, const float *p, int sym)
{
    float cyy = p[FP_EYE_Y] + p[FP_GAZE_Y];
    if (sym == 3) { /* zzz: two Z glyphs top right */
        static const float z[2][3] = {{62.0f, -82.0f, 16.0f}, {86.0f, -102.0f, 11.0f}};
        for (int i = 0; i < 2; i++) {
            float zx = z[i][0], zy = z[i][1], zs = z[i][2];
            add_seg(sc, zx, zy, zx + zs, zy, 2.5f, 1.0f);
            add_seg(sc, zx + zs, zy, zx, zy + zs, 2.5f, 1.0f);
            add_seg(sc, zx, zy + zs, zx + zs, zy + zs, 2.5f, 1.0f);
        }
    } else if (sym == 4) { /* tear under the viewer's left eye */
        float tx = -p[FP_EYE_DX] + p[FP_GAZE_X];
        float ty = cyy + p[FP_EYE_H] * p[FP_EYE_OPEN] / 2.0f + 16.0f;
        add_ell(sc, tx, ty, 6.0f, 7.0f);
        add_seg(sc, tx, ty - 12.0f, tx, ty - 4.0f, 2.5f, 1.0f);
    } else if (sym == 5) { /* sparkle by the viewer's right eye */
        float sx = p[FP_EYE_DX] + p[FP_GAZE_X] + p[FP_EYE_W] / 2.0f + 12.0f;
        float sy = cyy - p[FP_EYE_H] / 2.0f - 6.0f;
        add_seg(sc, sx - 11.0f, sy, sx + 11.0f, sy, 2.0f, 1.0f);
        add_seg(sc, sx, sy - 11.0f, sx, sy + 11.0f, 2.0f, 1.0f);
        add_seg(sc, sx - 5.0f, sy - 5.0f, sx + 5.0f, sy + 5.0f, 1.5f, 1.0f);
    }
}

static void scene_build(scene_t *sc, const face_frame_t *f, int use_boxes)
{
    const float *p = f->p;
    sc->n = 0;
    sc->n_blush = 0;
    sc->bbox = (face_rect_t){0, 0, 0, 0};
    sc->bg[0] = FACE_BG_R, sc->bg[1] = FACE_BG_G, sc->bg[2] = FACE_BG_B;
    float hue = fmodf(p[FP_HUE], 360.0f);
    if (hue < 0.0f) hue += 360.0f; /* Python's % is non-negative */
    hsv_to_rgb(hue / 360.0f, clampf(p[FP_SAT], 0.0f, 1.0f), 1.0f, sc->ink);
    for (int i = 0; i < 3; i++) sc->ink[i] *= 255.0f;
    hsv_to_rgb(340.0f / 360.0f, 0.55f, 1.0f, sc->pink);
    for (int i = 0; i < 3; i++) sc->pink[i] *= 255.0f;

    int sym = (int)lrintf(p[FP_SYMBOL]); /* Python round(): half to even, like lrintf in the default mode */
    for (int side = -1; side <= 1; side += 2) {
        float cx = (float)side * p[FP_EYE_DX] + p[FP_GAZE_X], cyy = p[FP_EYE_Y] + p[FP_GAZE_Y];
        if (sym == 1) {
            add_heart(sc, cx, cyy, p[FP_EYE_W] * 1.25f);
        } else if (sym == 2) {
            float L = p[FP_EYE_W] * 0.45f;
            add_seg(sc, cx - L, cyy - L, cx + L, cyy + L, 4.0f, 1.0f);
            add_seg(sc, cx - L, cyy + L, cx + L, cyy - L, 4.0f, 1.0f);
        } else {
            float wink = (f->wink && side == 1) ? 0.07f : 1.0f;
            add_eye(sc, f, side, f->blink * wink);
        }
        if (p[FP_BROW] > 0.02f) {
            float by = cyy - p[FP_EYE_H] / 2.0f + p[FP_BROW_Y];
            float L = p[FP_EYE_W] * 0.55f;
            float dy = tanf(p[FP_BROW_ANGLE] * DEG2RAD) * L * (float)(-side);
            add_seg(sc, cx - L, by - dy, cx + L, by + dy, 4.0f, p[FP_BROW]);
        }
    }
    add_mouth(sc, p);
    add_symbol(sc, p, sym);

    if (p[FP_BLUSH] > 0.01f) {
        sc->blush_k = 0.5f * p[FP_BLUSH];
        for (int side = -1; side <= 1; side += 2) {
            shape_t *s = &sc->blush[sc->n_blush++];
            memset(s, 0, sizeof(*s));
            s->kind = K_ELL;
            s->weight = 1.0f;
            fill_ell(s, (float)side * (p[FP_EYE_DX] + 8.0f), p[FP_EYE_Y] + p[FP_EYE_H] / 2.0f + 18.0f, 16.0f, 8.0f);
        }
    }
    if (!use_boxes) { /* test reference: every shape everywhere */
        for (int i = 0; i < sc->n; i++) sc->sh[i].x0 = sc->sh[i].y0 = 0, sc->sh[i].x1 = sc->sh[i].y1 = PANEL;
        for (int i = 0; i < sc->n_blush; i++) {
            sc->blush[i].x0 = sc->blush[i].y0 = 0, sc->blush[i].x1 = sc->blush[i].y1 = PANEL;
        }
    }
    for (int i = 0; i < sc->n; i++) {
        face_rect_t r = {sc->sh[i].x0, sc->sh[i].y0, sc->sh[i].x1, sc->sh[i].y1};
        sc->bbox = face_rect_union(sc->bbox, r);
    }
    for (int i = 0; i < sc->n_blush; i++) {
        face_rect_t r = {sc->blush[i].x0, sc->blush[i].y0, sc->blush[i].x1, sc->blush[i].y1};
        sc->bbox = face_rect_union(sc->bbox, r);
    }
}

/* ------------------------------------------------------------------ distance functions (render.py order) */
static float eval_shape(const shape_t *s, float x, float y)
{
    switch (s->kind) {
    case K_EYE: {
        const float cx = s->u.eye.cx, cy = s->u.eye.cy;
        float qx = fabsf(x - cx) - (s->u.eye.w / 2.0f - s->u.eye.r);
        float qy = fabsf(y - cy) - (s->u.eye.h / 2.0f - s->u.eye.r);
        float d = hyp(fmaxf(qx, 0.0f), fmaxf(qy, 0.0f)) + fminf(fmaxf(qx, qy), 0.0f) - s->u.eye.r;
        if (s->u.eye.lid_top) d = fmaxf(d, -(y - s->u.eye.y_line - s->u.eye.k * (x - cx)) * s->u.eye.inv_n);
        if (s->u.eye.lid_bot) d = fmaxf(d, s->u.eye.R - hyp(x - cx, y - s->u.eye.c2y));
        return d;
    }
    case K_HEART: {
        float sz = s->u.heart.s;
        float px = fabsf(x - s->u.heart.cx) / sz;
        float py = -(y - s->u.heart.cy) / sz + 0.5f;
        float d;
        if (py + px > 1.0f) {
            d = hyp(px - 0.25f, py - 0.75f) - 0.35355339059327373f; /* sqrt(2) / 4 */
        } else {
            float m = fmaxf(px + py, 0.0f) * 0.5f;
            float a = px * px + (py - 1.0f) * (py - 1.0f), b = (px - m) * (px - m) + (py - m) * (py - m);
            float sgn = (px - py) > 0.0f ? 1.0f : ((px - py) < 0.0f ? -1.0f : 0.0f); /* np.sign */
            d = sqrtf(fminf(a, b)) * sgn;
        }
        return d * sz;
    }
    case K_SEG: {
        float px = x - s->u.seg.ax, py = y - s->u.seg.ay;
        float t = clampf((px * s->u.seg.ex + py * s->u.seg.ey) / s->u.seg.den, 0.0f, 1.0f);
        return hyp(px - s->u.seg.ex * t, py - s->u.seg.ey * t) - s->u.seg.r;
    }
    case K_ELL:
    case K_OVAL: {
        float k = hyp((x - s->u.ell.cx) / s->u.ell.rx, (y - s->u.ell.cy) / s->u.ell.ry);
        float d = (k - 1.0f) * s->u.ell.m;
        if (s->kind == K_OVAL && s->u.ell.flat) d = fmaxf(d, s->u.ell.flat_y - y);
        return d;
    }
    case K_ARC: {
        float ex = s->u.arc.ex, t2 = s->u.arc.t / 2.0f;
        float dy = y - s->u.arc.ccy;
        float sy = dy > 0.0f ? 1.0f : (dy < 0.0f ? -1.0f : 0.0f); /* np.sign */
        if (fabsf(x) <= ex && sy == s->u.arc.sgn) return fabsf(hyp(x, dy) - s->u.arc.R) - t2;
        float ey = y - s->u.arc.e_y;
        return fminf(hyp(x + ex, ey), hyp(x - ex, ey)) - t2;
    }
    }
    return 1e9f;
}

/* ------------------------------------------------------------------ per-pixel colour */
typedef struct {
    float r, g, b;
} rgbf_t;

static inline rgbf_t shade(const scene_t *sc, const shape_t *const *act, int n_act, float xc, float yc, int px, int py)
{
    rgbf_t out = {sc->bg[0], sc->bg[1], sc->bg[2]};
#if FACE_PANEL_ROUND
    if (xc * xc + yc * yc > HALF * HALF) return out;
#endif
    float a = 0.0f;
    for (int i = 0; i < n_act; i++) {
        const shape_t *s = act[i];
        if (px < s->x0 || px >= s->x1) continue;
        float c = cover(eval_shape(s, xc, yc));
        if (s->weight != 1.0f) c *= s->weight;
        if (c > a) a = c;
    }
    float r = sc->bg[0] * (1.0f - a) + sc->ink[0] * a;
    float g = sc->bg[1] * (1.0f - a) + sc->ink[1] * a;
    float b = sc->bg[2] * (1.0f - a) + sc->ink[2] * a;
    if (sc->n_blush) {
        float ba = 0.0f;
        for (int i = 0; i < sc->n_blush; i++) {
            const shape_t *s = &sc->blush[i];
            if (px < s->x0 || px >= s->x1 || py < s->y0 || py >= s->y1) continue;
            float c = cover(eval_shape(s, xc, yc));
            if (c > ba) ba = c;
        }
        ba *= sc->blush_k * (1.0f - a);
        r = r * (1.0f - ba) + sc->pink[0] * ba;
        g = g * (1.0f - ba) + sc->pink[1] * ba;
        b = b * (1.0f - ba) + sc->pink[2] * ba;
    }
    out.r = r, out.g = g, out.b = b;
    return out;
}

static inline uint8_t to_u8(float v)
{
    long i = lrintf(v); /* np.round: half to even */
    return (uint8_t)(i < 0 ? 0 : (i > 255 ? 255 : i));
}

static face_rect_t clip_rect(face_rect_t r, face_rect_t bound)
{
    if (r.x0 < bound.x0) r.x0 = bound.x0;
    if (r.y0 < bound.y0) r.y0 = bound.y0;
    if (r.x1 > bound.x1) r.x1 = bound.x1;
    if (r.y1 > bound.y1) r.y1 = bound.y1;
    return r;
}

typedef void (*put_fn)(void *dst, int x, int y, int stride, uint8_t r, uint8_t g, uint8_t b);

static void put888(void *dst, int x, int y, int stride, uint8_t r, uint8_t g, uint8_t b)
{
    uint8_t *p = (uint8_t *)dst + 3 * ((size_t)y * (size_t)stride + (size_t)x);
    p[0] = r, p[1] = g, p[2] = b;
}

static void put565be(void *dst, int x, int y, int stride, uint8_t r, uint8_t g, uint8_t b)
{
    uint16_t v = (uint16_t)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
    ((uint16_t *)dst)[(size_t)y * (size_t)stride + (size_t)x] = (uint16_t)((v << 8) | (v >> 8));
}

static void render(const face_frame_t *f, face_rect_t r, void *dst, int stride, put_fn put, int use_boxes)
{
    scene_t sc;
    scene_build(&sc, f, use_boxes);
    r = clip_rect(r, (face_rect_t){0, 0, PANEL, PANEL});
    uint8_t bgr = to_u8(sc.bg[0]), bgg = to_u8(sc.bg[1]), bgb = to_u8(sc.bg[2]);
    for (int y = r.y0; y < r.y1; y++) {
        const shape_t *act[MAX_SHAPES];
        int n_act = 0;
        for (int i = 0; i < sc.n; i++) {
            if (y >= sc.sh[i].y0 && y < sc.sh[i].y1) act[n_act++] = &sc.sh[i];
        }
        int blush_row = 0;
        for (int i = 0; i < sc.n_blush; i++) blush_row |= (y >= sc.blush[i].y0 && y < sc.blush[i].y1);
        float yc = (float)y + 0.5f - HALF;
        if (n_act == 0 && !blush_row) {
            for (int x = r.x0; x < r.x1; x++) put(dst, x, y, stride, bgr, bgg, bgb);
            continue;
        }
        for (int x = r.x0; x < r.x1; x++) {
            rgbf_t c = shade(&sc, act, n_act, (float)x + 0.5f - HALF, yc, x, y);
            put(dst, x, y, stride, to_u8(c.r), to_u8(c.g), to_u8(c.b));
        }
    }
}

face_rect_t face_sdf_bbox(const face_frame_t *f)
{
    scene_t sc;
    scene_build(&sc, f, 1);
    return sc.bbox;
}

void face_sdf_render_rgb888(const face_frame_t *f, face_rect_t r, uint8_t *rgb, int stride)
{
    render(f, r, rgb, stride, put888, 1);
}

void face_sdf_render_rgb565be(const face_frame_t *f, face_rect_t r, uint16_t *px, int stride)
{
    render(f, r, px, stride, put565be, 1);
}

void face_sdf_render_rgb888_ref(const face_frame_t *f, face_rect_t r, uint8_t *rgb, int stride)
{
    render(f, r, rgb, stride, put888, 0);
}

uint16_t face_bg_rgb565(void)
{
    return (uint16_t)(((FACE_BG_R & 0xF8) << 8) | ((FACE_BG_G & 0xFC) << 3) | (FACE_BG_B >> 3));
}

/* ------------------------------------------------------------------ parameter blending (render.py blend/resolve) */
void face_blend(const float a[FACE_N_PARAMS], uint8_t wink_a, const float b[FACE_N_PARAMS], uint8_t wink_b, float u,
                float out[FACE_N_PARAMS], uint8_t *wink_out)
{
    for (int i = 0; i < FACE_N_PARAMS; i++) {
        float va = a[i], vb = b[i];
        if (u >= 1.0f || u <= 0.0f) { /* exact end points (a + (b - a) * 1 can differ from b by one float ulp) */
            out[i] = u >= 1.0f ? vb : va;
        } else if (i == FP_SYMBOL) {
            out[i] = u >= 0.5f ? vb : va;
        } else if (i == FP_HUE) {
            float dh = fmodf(vb - va + 180.0f, 360.0f);
            if (dh < 0.0f) dh += 360.0f;
            dh -= 180.0f;
            float h = fmodf(va + dh * u, 360.0f);
            if (h < 0.0f) h += 360.0f;
            out[i] = h;
        } else {
            out[i] = va + (vb - va) * u;
        }
    }
    if (wink_out) *wink_out = u >= 0.5f ? wink_b : wink_a;
}

void face_emotion_params(int emotion, float intensity, float out[FACE_N_PARAMS], uint8_t *wink)
{
    if (emotion < 0 || emotion >= FACE_N_EMOTIONS) emotion = 0;
    uint8_t w;
    face_blend(face_emotions[0].p, face_emotions[0].wink, face_emotions[emotion].p, face_emotions[emotion].wink,
               clampf(intensity, 0.0f, 1.0f), out, &w);
    if (wink) *wink = w;
}
