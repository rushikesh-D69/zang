/* face_engine.c - see face_engine.h. Pure C. */
#include "face_engine.h"

#include <math.h>
#include <string.h>

#define BLINK_MS      140u
#define BLINK_SCALE   0.05f
#define BLINK_MIN_MS  2500u
#define BLINK_SPAN_MS 3000u
#define TRACK_RETURN_MS 500u

static uint32_t rnd(face_engine_t *e)
{
    uint32_t x = e->rng ? e->rng : 0x9E3779B9u; /* xorshift32 */
    x ^= x << 13, x ^= x >> 17, x ^= x << 5;
    return e->rng = x;
}

static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* Blend position of the stored from -> to pair at now. */
static void current(const face_engine_t *e, uint32_t now_ms, float out[FACE_N_PARAMS], uint8_t *wink)
{
    float u = e->blend_ms ? (float)(int32_t)(now_ms - e->blend_t0) / (float)e->blend_ms : 1.0f;
    face_blend(e->from, e->from_wink, e->to, e->to_wink, clampf(u, 0.0f, 1.0f), out, wink);
}

static void schedule_idle(face_engine_t *e, uint32_t now_ms)
{
    e->next_blink_ms = now_ms + BLINK_MIN_MS + rnd(e) % BLINK_SPAN_MS;
}

void face_engine_init(face_engine_t *e, uint32_t now_ms, uint32_t seed)
{
    memset(e, 0, sizeof(*e));
    e->rng = seed;
    uint8_t w;
    face_emotion_params(0, 1.0f, e->to, &w);
    memcpy(e->from, e->to, sizeof(e->from));
    e->to_wink = e->from_wink = w;
    e->blend_ms = 1;
    e->blend_t0 = now_ms;
    e->next_blink_ms = now_ms + 2000;
    e->next_drift_ms = now_ms + 1500;
    e->last_ms = now_ms;
    memcpy(e->track_last, e->to, sizeof(e->track_last));
}

/* Start a blend from whatever is on screen now. */
static void restart_from_current(face_engine_t *e, uint32_t now_ms, uint16_t blend_ms)
{
    float cur[FACE_N_PARAMS];
    uint8_t w;
    if (e->track_active) {
        memcpy(cur, e->track_last, sizeof(cur));
        w = e->track_last_wink;
    } else {
        current(e, now_ms, cur, &w);
    }
    memcpy(e->from, cur, sizeof(cur));
    e->from_wink = w;
    e->blend_t0 = now_ms;
    e->blend_ms = blend_ms ? blend_ms : 1;
}

void face_engine_set(face_engine_t *e, uint32_t now_ms, uint8_t emotion, uint8_t intensity, uint16_t blend_ms,
                     int8_t gaze_x, int8_t gaze_y, uint8_t flags)
{
    if (emotion >= FACE_N_EMOTIONS) emotion = 0;
    restart_from_current(e, now_ms, blend_ms);
    face_emotion_params(emotion, (float)intensity / 255.0f, e->to, &e->to_wink);
    e->track_active = false; /* the host took manual control */
    e->n_keys = 0;
    e->emotion = emotion;
    e->gaze_x = gaze_x, e->gaze_y = gaze_y;
    e->flags = (uint8_t)(flags & (FACE_FLAG_TALKING | FACE_FLAG_HOLD));
    if (flags & FACE_FLAG_BLINK) e->blink_until_ms = now_ms + BLINK_MS;
    if (flags & FACE_FLAG_HOLD) {
        e->drift_tx = e->drift_ty = 0.0f;
    }
}

void face_engine_params(face_engine_t *e, uint32_t now_ms, uint16_t blend_ms, int n, const uint8_t *ids,
                        const float *values)
{
    if (e->track_active) { /* freeze what the track shows and edit that */
        memcpy(e->to, e->track_last, sizeof(e->to));
        e->to_wink = e->track_last_wink;
        e->blend_ms = 1;
        e->blend_t0 = now_ms;
        memcpy(e->from, e->to, sizeof(e->from));
        e->from_wink = e->to_wink;
        e->track_active = false;
        e->n_keys = 0;
    }
    restart_from_current(e, now_ms, blend_ms);
    for (int i = 0; i < n; i++) {
        if (ids[i] >= FACE_N_PARAMS || !isfinite(values[i])) continue;
        e->to[ids[i]] = clampf(values[i], face_params_info[ids[i]].min, face_params_info[ids[i]].max);
    }
}

int face_engine_track_add(face_engine_t *e, uint32_t stream_id, const face_key_t *keys, int n)
{
    if (stream_id != e->track_stream || e->n_keys == 0) {
        e->n_keys = 0;
        e->track_stream = stream_id;
    }
    for (int i = 0; i < n && e->n_keys < FACE_ENGINE_MAX_KEYS; i++) {
        if (e->n_keys && keys[i].t_ms <= e->keys[e->n_keys - 1].t_ms) continue; /* keys must increase */
        face_key_t k = keys[i];
        if (k.emotion >= FACE_N_EMOTIONS) k.emotion = 0;
        e->keys[e->n_keys++] = k;
    }
    return e->n_keys;
}

void face_engine_track_params(const face_engine_t *e, uint32_t t_ms, float out[FACE_N_PARAMS], uint8_t *wink,
                              uint8_t *emotion)
{
    if (e->n_keys == 0) {
        face_emotion_params(0, 1.0f, out, wink);
        if (emotion) *emotion = 0;
        return;
    }
    int i = 0;
    while (i + 1 < e->n_keys && e->keys[i + 1].t_ms <= t_ms) i++;
    const face_key_t *k = &e->keys[i], *prev = i ? &e->keys[i - 1] : k;
    float pa[FACE_N_PARAMS], pb[FACE_N_PARAMS];
    uint8_t wa, wb;
    face_emotion_params(prev->emotion, (float)prev->intensity / 255.0f, pa, &wa);
    face_emotion_params(k->emotion, (float)k->intensity / 255.0f, pb, &wb);
    if (emotion) *emotion = (t_ms < k->t_ms) ? prev->emotion : k->emotion;
    if (t_ms < k->t_ms) {
        memcpy(out, pa, sizeof(pa));
        *wink = wa;
        return;
    }
    float u = clampf((float)(t_ms - k->t_ms) / (float)(k->blend_ms ? k->blend_ms : 1), 0.0f, 1.0f);
    face_blend(pa, wa, pb, wb, u, out, wink);
    if (emotion && u < 0.5f) *emotion = prev->emotion;
}

void face_engine_track_sync(face_engine_t *e, uint32_t now_ms, bool streaming, uint32_t stream_id, uint32_t stream_ms)
{
    bool want = streaming && e->n_keys > 0 && stream_id == e->track_stream;
    if (want) {
        uint8_t emo = e->emotion;
        face_engine_track_params(e, stream_ms, e->track_last, &e->track_last_wink, &emo);
        e->emotion = emo;
        e->track_active = true;
    } else if (e->track_active) { /* stream ended: ease back to the last explicit face */
        e->track_active = false;
        memcpy(e->from, e->track_last, sizeof(e->from));
        e->from_wink = e->track_last_wink;
        e->blend_t0 = now_ms;
        e->blend_ms = TRACK_RETURN_MS;
        /* e->to keeps the face the host last set with FACE_SET / FACE_PARAMS */
    }
}

void face_engine_frame(face_engine_t *e, uint32_t now_ms, face_frame_t *out)
{
    memset(out, 0, sizeof(*out));
    float *p = out->p;
    uint8_t wink;
    if (e->track_active) {
        memcpy(p, e->track_last, sizeof(e->track_last));
        wink = e->track_last_wink;
    } else {
        current(e, now_ms, p, &wink);
    }
    out->wink = wink;
    out->blink = 1.0f;

    float dt = (float)(now_ms - e->last_ms);
    e->last_ms = now_ms;
    if (!(e->flags & FACE_FLAG_HOLD)) {
        if ((int32_t)(now_ms - e->next_blink_ms) >= 0) {
            e->blink_until_ms = now_ms + BLINK_MS;
            schedule_idle(e, now_ms);
        }
        if ((int32_t)(now_ms - e->next_drift_ms) >= 0) {
            e->drift_tx = ((float)(rnd(e) % 1000) / 1000.0f - 0.5f) * 6.0f;
            e->drift_ty = ((float)(rnd(e) % 1000) / 1000.0f - 0.5f) * 4.0f;
            e->next_drift_ms = now_ms + 1200 + rnd(e) % 2000;
        }
    } else {
        e->drift_tx = e->drift_ty = 0.0f;
    }
    float a = 1.0f - expf(-dt / 80.0f); /* ease the drift so the gaze glides instead of snapping */
    e->drift_x += (e->drift_tx - e->drift_x) * a;
    e->drift_y += (e->drift_ty - e->drift_y) * a;
    if ((int32_t)(e->blink_until_ms - now_ms) > 0) out->blink = BLINK_SCALE;

    p[FP_GAZE_X] += (float)e->gaze_x + e->drift_x;
    p[FP_GAZE_Y] += (float)e->gaze_y + e->drift_y;
    if (e->flags & FACE_FLAG_TALKING) {
        /* |sin(t / 90 ms)| has period pi * 90 = 282.74 ms; wrapping at 10 periods keeps float precision after days */
        float m = 0.15f + 0.5f * fabsf(sinf((float)(now_ms % 2827u) / 90.0f));
        if (m > p[FP_MOUTH_OPEN]) p[FP_MOUTH_OPEN] = m;
    }
}
