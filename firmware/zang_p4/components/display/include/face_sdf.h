/*
 * face_sdf.h - the face renderer: a float32 port of ai_stack/zang_ai/face/render.py (the canonical algorithm).
 * Pure C (host-tested against the goldens rendered by render.py: components/display/test/faces/).
 *
 * Coordinates: panel centre = origin, +x right, +y down, pixel centres at i + 0.5 - 120 (240 px panel).
 * Speed: every shape gets a conservative bounding box once per frame; a pixel only evaluates the shapes whose box
 * contains it, and face_sdf_bbox() gives the union so a caller can redraw only the region that can change.
 */
#pragma once

#include <stdint.h>
#include "faces_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float p[FACE_N_PARAMS]; /* parameter values in faces.json units, indexed by face_param_id_t */
    float blink;            /* scales eye_open: 1 = as given, 0.05 = mid-blink */
    uint8_t wink;           /* 1 = viewer's right eye winks (scale 0.07) */
} face_frame_t;

/* Half-open pixel rectangle [x0, x1) x [y0, y1). Empty when x1 <= x0 or y1 <= y0. */
typedef struct {
    int x0, y0, x1, y1;
} face_rect_t;

static inline int face_rect_empty(face_rect_t r) { return r.x1 <= r.x0 || r.y1 <= r.y0; }

static inline face_rect_t face_rect_union(face_rect_t a, face_rect_t b)
{
    if (face_rect_empty(a)) return b;
    if (face_rect_empty(b)) return a;
    face_rect_t r = {a.x0 < b.x0 ? a.x0 : b.x0, a.y0 < b.y0 ? a.y0 : b.y0, a.x1 > b.x1 ? a.x1 : b.x1,
                     a.y1 > b.y1 ? a.y1 : b.y1};
    return r;
}

/* Pixels outside this rectangle are guaranteed to be pure background for this frame. */
face_rect_t face_sdf_bbox(const face_frame_t *f);

/* Render rect r (clipped to the panel) into an RGB888 buffer: pixel (x, y) at rgb[3 * (y * stride + x)]. */
void face_sdf_render_rgb888(const face_frame_t *f, face_rect_t r, uint8_t *rgb, int stride);
/* Same into RGB565 with the bytes swapped (big-endian on the wire, as SPI panels expect): px[y * stride + x]. */
void face_sdf_render_rgb565be(const face_frame_t *f, face_rect_t r, uint16_t *px, int stride);
/* Test hook: same as face_sdf_render_rgb888 but every shape is evaluated at every pixel (no bounding boxes). */
void face_sdf_render_rgb888_ref(const face_frame_t *f, face_rect_t r, uint8_t *rgb, int stride);

/* Background colour as RGB565 (native byte order). */
uint16_t face_bg_rgb565(void);

/* Parameter set of emotion e (resolved) blended from neutral by intensity 0..1: blend(neutral, e, intensity). */
void face_emotion_params(int emotion, float intensity, float out[FACE_N_PARAMS], uint8_t *wink);
/* render.py blend(): linear; symbol and wink switch at u >= 0.5; hue the short way round. */
void face_blend(const float a[FACE_N_PARAMS], uint8_t wink_a, const float b[FACE_N_PARAMS], uint8_t wink_b, float u,
                float out[FACE_N_PARAMS], uint8_t *wink_out);

#ifdef __cplusplus
}
#endif
