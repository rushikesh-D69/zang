/* engine_test.c - face_engine.c: blending, idle layer, talking, FACE_PARAMS, FACE_TRACK. */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "face_engine.h"

static int fails;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);             \
            fails++;                                                          \
        }                                                                     \
    } while (0)
#define NEAR(a, b, tol) (fabsf((a) - (b)) <= (tol))

static face_engine_t E;

static void test_blend_and_intensity(void)
{
    face_engine_init(&E, 0, 1);
    face_frame_t f;
    face_engine_frame(&E, 0, &f);
    CHECK(memcmp(f.p, face_emotions[0].p, sizeof(f.p)) == 0 && f.blink == 1.0f && f.wink == 0);

    face_engine_set(&E, 1000, FACE_EMO_HAPPY, 255, 200, 0, 0, FACE_FLAG_HOLD);
    face_engine_frame(&E, 1100, &f); /* halfway */
    CHECK(NEAR(f.p[FP_MOUTH_CURVE], (0.25f + 0.8f) / 2, 1e-4f));
    CHECK(NEAR(f.p[FP_LID_BOT], 0.175f, 1e-4f));
    face_engine_frame(&E, 1200, &f);
    CHECK(memcmp(f.p, face_emotions[FACE_EMO_HAPPY].p, sizeof(f.p)) == 0);
    face_engine_frame(&E, 5000, &f);
    CHECK(memcmp(f.p, face_emotions[FACE_EMO_HAPPY].p, sizeof(f.p)) == 0);

    /* intensity 128 of "excited": symbol (5) switches at the midpoint u = 128/255 >= 0.5, size is 50 % */
    face_engine_set(&E, 6000, FACE_EMO_EXCITED, 128, 0, 0, 0, FACE_FLAG_HOLD);
    face_engine_frame(&E, 6001, &f);
    CHECK(f.p[FP_SYMBOL] == 5.0f && NEAR(f.p[FP_EYE_H], 56 + 8 * (128 / 255.0f), 1e-3f));
    face_engine_set(&E, 7000, FACE_EMO_EXCITED, 127, 0, 0, 0, FACE_FLAG_HOLD);
    face_engine_frame(&E, 7001, &f);
    CHECK(f.p[FP_SYMBOL] == 0.0f);

    /* a new FACE_SET during a blend starts from what is on screen (no jump) */
    face_engine_set(&E, 10000, FACE_EMO_SAD, 255, 1000, 0, 0, FACE_FLAG_HOLD);
    face_engine_frame(&E, 10500, &f);
    float mid = f.p[FP_MOUTH_CURVE];
    face_engine_set(&E, 10500, FACE_EMO_ANGRY, 255, 1000, 0, 0, FACE_FLAG_HOLD);
    face_engine_frame(&E, 10500, &f);
    CHECK(NEAR(f.p[FP_MOUTH_CURVE], mid, 1e-4f));

    /* hue takes the short way: neutral 192 -> angry 4: 192 down to 4, no wrap; sad 215 -> angry 4 likewise */
    /* wink emotion sets the wink flag once past the midpoint */
    face_engine_set(&E, 20000, FACE_EMO_WINK, 255, 100, 0, 0, FACE_FLAG_HOLD);
    face_engine_frame(&E, 20040, &f);
    CHECK(f.wink == 0);
    face_engine_frame(&E, 20060, &f);
    CHECK(f.wink == 1);
    /* gaze offset is added to the emotion's gaze */
    face_engine_set(&E, 30000, FACE_EMO_THINKING, 255, 0, 5, -3, FACE_FLAG_HOLD);
    face_engine_frame(&E, 30010, &f);
    CHECK(NEAR(f.p[FP_GAZE_X], -14 + 5, 1e-4f) && NEAR(f.p[FP_GAZE_Y], -16 - 3, 1e-4f));
}

static void test_idle(void)
{
    face_engine_init(&E, 0, 42);
    face_engine_set(&E, 0, FACE_EMO_NEUTRAL, 255, 0, 0, 0, 0);
    int blinks = 0, in_blink = 0, frames_in = 0, maxf = 0;
    uint32_t last_start = 0, min_gap = 1u << 30, max_gap = 0;
    float gmin = 0, gmax = 0;
    for (uint32_t t = 0; t < 120000; t += 33) {
        face_frame_t f;
        face_engine_frame(&E, t, &f);
        if (f.blink < 1.0f) {
            CHECK(f.blink == 0.05f);
            if (!in_blink) {
                in_blink = 1, frames_in = 0;
                blinks++;
                if (last_start) {
                    uint32_t gap = t - last_start;
                    if (gap < min_gap) min_gap = gap;
                    if (gap > max_gap) max_gap = gap;
                }
                last_start = t;
            }
            frames_in++;
            if (frames_in > maxf) maxf = frames_in;
        } else {
            in_blink = 0;
        }
        if (f.p[FP_GAZE_X] < gmin) gmin = f.p[FP_GAZE_X];
        if (f.p[FP_GAZE_X] > gmax) gmax = f.p[FP_GAZE_X];
    }
    printf("  idle: %d blinks in 120 s, gap %u..%u ms, blink %d frames, gaze_x %.2f..%.2f\n", blinks, min_gap, max_gap,
           maxf, gmin, gmax);
    CHECK(blinks >= 120000 / 5600 && blinks <= 120000 / 2500 + 1);
    CHECK(min_gap >= 2500 + 100 - 33 && max_gap <= 5500 + 140 + 66); /* gap is measured between 33 ms frames */
    CHECK(maxf >= 4 && maxf <= 5);                                  /* 140 ms at 33 ms/frame */
    CHECK(gmin < -0.5f && gmax > 0.5f && gmin >= -3.1f && gmax <= 3.1f);

    /* hold: no blinking, no drift, for a long time */
    face_engine_set(&E, 130000, FACE_EMO_NEUTRAL, 255, 0, 0, 0, FACE_FLAG_HOLD);
    int any = 0;
    for (uint32_t t = 130000; t < 190000; t += 33) {
        face_frame_t f;
        face_engine_frame(&E, t, &f);
        if (f.blink != 1.0f || f.p[FP_GAZE_X] != 0.0f || f.p[FP_GAZE_Y] != 0.0f) any++;
    }
    CHECK(any == 0);
    /* blink-now works even with hold, and lasts 140 ms */
    face_engine_set(&E, 200000, FACE_EMO_NEUTRAL, 255, 0, 0, 0, FACE_FLAG_HOLD | FACE_FLAG_BLINK);
    face_frame_t f;
    face_engine_frame(&E, 200010, &f);
    CHECK(f.blink == 0.05f);
    face_engine_frame(&E, 200139, &f);
    CHECK(f.blink == 0.05f);
    face_engine_frame(&E, 200141, &f);
    CHECK(f.blink == 1.0f);
}

static void test_talking(void)
{
    face_engine_init(&E, 0, 7);
    face_engine_set(&E, 0, FACE_EMO_HAPPY, 255, 0, 0, 0, FACE_FLAG_TALKING | FACE_FLAG_HOLD);
    float lo = 9, hi = 0;
    for (uint32_t t = 10; t < 5000; t += 7) {
        face_frame_t f;
        face_engine_frame(&E, t, &f);
        if (f.p[FP_MOUTH_OPEN] < lo) lo = f.p[FP_MOUTH_OPEN];
        if (f.p[FP_MOUTH_OPEN] > hi) hi = f.p[FP_MOUTH_OPEN];
    }
    CHECK(lo >= 0.15f - 1e-4f && hi <= 0.65f + 1e-4f && hi > 0.6f);
    /* max(base, ...): an emotion with the mouth fully open keeps it open */
    face_engine_set(&E, 6000, FACE_EMO_SURPRISED, 255, 0, 0, 0, FACE_FLAG_TALKING | FACE_FLAG_HOLD);
    face_frame_t f;
    face_engine_frame(&E, 6100, &f);
    CHECK(f.p[FP_MOUTH_OPEN] == 1.0f);
    /* exact formula */
    face_engine_set(&E, 7000, FACE_EMO_NEUTRAL, 255, 0, 0, 0, FACE_FLAG_TALKING | FACE_FLAG_HOLD);
    face_engine_frame(&E, 7045, &f);
    CHECK(NEAR(f.p[FP_MOUTH_OPEN], 0.15f + 0.5f * fabsf(sinf(7045.0f / 90.0f)), 1e-2f));
    /* talking off again with the next FACE_SET */
    face_engine_set(&E, 8000, FACE_EMO_NEUTRAL, 255, 0, 0, 0, FACE_FLAG_HOLD);
    face_engine_frame(&E, 8045, &f);
    CHECK(f.p[FP_MOUTH_OPEN] == 0.0f);
}

static void test_params(void)
{
    face_engine_init(&E, 0, 3);
    face_engine_set(&E, 0, FACE_EMO_NEUTRAL, 255, 0, 0, 0, FACE_FLAG_HOLD);
    uint8_t ids[4] = {FP_MOUTH_W, 99, FP_EYE_H, FP_HUE};
    float vals[4] = {80.0f, 5.0f, 9999.0f, 10.0f};
    face_engine_params(&E, 1000, 100, 4, ids, vals);
    face_frame_t f;
    face_engine_frame(&E, 1050, &f);
    CHECK(NEAR(f.p[FP_MOUTH_W], 63.0f, 1e-3f)); /* 46 -> 80 halfway */
    face_engine_frame(&E, 1100, &f);
    CHECK(f.p[FP_MOUTH_W] == 80.0f);
    CHECK(f.p[FP_EYE_H] == 90.0f); /* clamped to the parameter max */
    CHECK(NEAR(f.p[FP_HUE], 10.0f, 1e-3f));
    /* sticky across time, replaced by the next FACE_SET */
    face_engine_frame(&E, 9000, &f);
    CHECK(f.p[FP_MOUTH_W] == 80.0f);
    face_engine_set(&E, 9000, FACE_EMO_NEUTRAL, 255, 0, 0, 0, FACE_FLAG_HOLD);
    face_engine_frame(&E, 9001, &f);
    CHECK(f.p[FP_MOUTH_W] == 46.0f);
    /* NaN is ignored */
    float nanv = NAN;
    uint8_t id = FP_MOUTH_W;
    face_engine_params(&E, 9100, 0, 1, &id, &nanv);
    face_engine_frame(&E, 9200, &f);
    CHECK(f.p[FP_MOUTH_W] == 46.0f);
}

static void test_track(void)
{
    face_engine_init(&E, 0, 5);
    face_engine_set(&E, 0, FACE_EMO_CALM, 255, 0, 0, 0, FACE_FLAG_HOLD);
    face_key_t k[3] = {{0, FACE_EMO_HAPPY, 255, 100}, {1000, FACE_EMO_SAD, 255, 200}, {500, FACE_EMO_ANGRY, 255, 0}};
    CHECK(face_engine_track_add(&E, 7, k, 3) == 2); /* third key goes backwards in time: ignored */
    face_key_t k2[1] = {{2000, FACE_EMO_NEUTRAL, 255, 100}};
    CHECK(face_engine_track_add(&E, 7, k2, 1) == 3); /* same stream: appended */
    face_frame_t f;
    /* not streaming: the FACE_SET face (calm) stays */
    face_engine_track_sync(&E, 10, false, 0, 0);
    face_engine_frame(&E, 10, &f);
    CHECK(memcmp(f.p, face_emotions[FACE_EMO_CALM].p, sizeof(f.p)) == 0);
    /* another stream id: not ours */
    face_engine_track_sync(&E, 20, true, 8, 500);
    face_engine_frame(&E, 20, &f);
    CHECK(memcmp(f.p, face_emotions[FACE_EMO_CALM].p, sizeof(f.p)) == 0);
    /* stream 7 at 500 ms: happy */
    face_engine_track_sync(&E, 30, true, 7, 500);
    face_engine_frame(&E, 30, &f);
    CHECK(memcmp(f.p, face_emotions[FACE_EMO_HAPPY].p, sizeof(f.p)) == 0);
    /* 1100 ms: halfway through the 200 ms blend to sad */
    face_engine_track_sync(&E, 40, true, 7, 1100);
    face_engine_frame(&E, 40, &f);
    CHECK(NEAR(f.p[FP_MOUTH_CURVE], (0.8f + -0.7f) / 2, 1e-3f));
    face_engine_track_sync(&E, 50, true, 7, 1500);
    face_engine_frame(&E, 50, &f);
    CHECK(memcmp(f.p, face_emotions[FACE_EMO_SAD].p, sizeof(f.p)) == 0);
    CHECK(E.emotion == FACE_EMO_SAD);
    /* stream ends: ease back to calm over 500 ms starting from sad */
    face_engine_track_sync(&E, 60, false, 0, 0);
    face_engine_frame(&E, 60, &f);
    CHECK(memcmp(f.p, face_emotions[FACE_EMO_SAD].p, sizeof(f.p)) == 0);
    face_engine_frame(&E, 310, &f);
    CHECK(NEAR(f.p[FP_MOUTH_CURVE], (-0.7f + 0.45f) / 2, 1e-3f));
    face_engine_frame(&E, 560, &f);
    CHECK(memcmp(f.p, face_emotions[FACE_EMO_CALM].p, sizeof(f.p)) == 0);
    /* a FACE_SET while a track plays cancels it */
    face_engine_track_sync(&E, 600, true, 7, 100);
    face_engine_set(&E, 610, FACE_EMO_JOY, 255, 0, 0, 0, FACE_FLAG_HOLD);
    face_engine_track_sync(&E, 620, true, 7, 300);
    face_engine_frame(&E, 700, &f);
    CHECK(memcmp(f.p, face_emotions[FACE_EMO_JOY].p, sizeof(f.p)) == 0);
    /* a new stream id replaces the keys */
    face_key_t k3[1] = {{0, FACE_EMO_LOVE, 255, 0}};
    CHECK(face_engine_track_add(&E, 9, k3, 1) == 1);
}

int main(void)
{
    test_blend_and_intensity();
    test_idle();
    test_talking();
    test_params();
    test_track();
    printf("%s (%d failures)\n", fails ? "ENGINE TESTS FAILED" : "face engine tests ok", fails);
    return fails ? 1 : 0;
}
