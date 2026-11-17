/*
 * face_engine.h - runtime behaviour of the face (pure C, host-tested). Same behaviour as FaceAnimator in
 * ai_stack/zang_ai/studio/web/face.js:
 *   FACE_SET     blend from the current parameters to neutral->emotion at intensity/255 over blend_ms
 *   FACE_PARAMS  override individual parameters (sticky until the next FACE_SET), blended over blend_ms
 *   FACE_TRACK   emotion keys played on the motion stream's clock
 *   idle layer   blink every 2.5..5.5 s for 140 ms (eye_open x 0.05), small gaze drift; hold flag turns it off
 *   talking      mouth_open = max(base, 0.15 + 0.5 |sin(t / 90 ms)|)
 * All times are milliseconds on one monotonic clock chosen by the caller (the display task passes esp_timer ms).
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "face_sdf.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FACE_ENGINE_MAX_KEYS 256

/* FACE_SET flags (same bits as ZLink) */
#define FACE_FLAG_TALKING (1u << 0)
#define FACE_FLAG_BLINK   (1u << 1)
#define FACE_FLAG_HOLD    (1u << 2)

typedef struct {
    uint32_t t_ms; /* stream clock */
    uint8_t emotion, intensity;
    uint16_t blend_ms;
} face_key_t;

typedef struct {
    float from[FACE_N_PARAMS], to[FACE_N_PARAMS];
    uint8_t from_wink, to_wink;
    uint32_t blend_t0, blend_ms;
    uint8_t emotion;       /* reported emotion id */
    int8_t gaze_x, gaze_y; /* FACE_SET gaze, px, added to the emotion's own gaze */
    uint8_t flags;
    /* idle layer */
    uint32_t rng;
    uint32_t next_blink_ms, blink_until_ms, next_drift_ms, last_ms;
    float drift_tx, drift_ty, drift_x, drift_y;
    /* face track */
    face_key_t keys[FACE_ENGINE_MAX_KEYS];
    uint16_t n_keys;
    uint32_t track_stream;
    bool track_active;
    /* last track output, to blend back from when the stream ends */
    float track_last[FACE_N_PARAMS];
    uint8_t track_last_wink;
} face_engine_t;

void face_engine_init(face_engine_t *e, uint32_t now_ms, uint32_t seed);
void face_engine_set(face_engine_t *e, uint32_t now_ms, uint8_t emotion, uint8_t intensity, uint16_t blend_ms,
                     int8_t gaze_x, int8_t gaze_y, uint8_t flags);
/* ids >= FACE_N_PARAMS are ignored; values are clamped to the parameter's [min, max]. */
void face_engine_params(face_engine_t *e, uint32_t now_ms, uint16_t blend_ms, int n, const uint8_t *ids,
                        const float *values);
/* Keys for stream_id (append when the id equals the stored one, else replace). Returns keys stored. */
int face_engine_track_add(face_engine_t *e, uint32_t stream_id, const face_key_t *keys, int n);
/* Call every frame with the motion stream state: starts/stops playing the stored track. */
void face_engine_track_sync(face_engine_t *e, uint32_t now_ms, bool streaming, uint32_t stream_id, uint32_t stream_ms);
/* Parameters at stream time t (pure function of the keys), for tests and the track player. */
void face_engine_track_params(const face_engine_t *e, uint32_t t_ms, float out[FACE_N_PARAMS], uint8_t *wink,
                              uint8_t *emotion);
/* Everything the renderer needs at now_ms (base blend + idle + talking). Zero-pads the struct. */
void face_engine_frame(face_engine_t *e, uint32_t now_ms, face_frame_t *out);

#ifdef __cplusplus
}
#endif
