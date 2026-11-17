/* face_test.c - face_sdf.c against the golden renders of ai_stack/zang_ai/face/render.py, plus self-consistency:
 *  - every golden case: mean |diff| <= 2.0 and >= 99 % of pixels within 24 levels on every channel
 *  - the bounding-box fast path equals the evaluate-everything reference exactly (goldens + random fuzz)
 *  - pixels outside face_sdf_bbox() are background; a partial render equals the same rectangle of the full one
 *  - RGB565 output is the RGB888 output truncated and byte-swapped
 *  - blend rules (hue the short way round, symbol/wink switch at the midpoint)
 */
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "face_sdf.h"
#include "faces_goldens.h"

#define N (240 * 240)
static int fails;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);             \
            fails++;                                                          \
        }                                                                     \
    } while (0)

static int load_ppm(const char *name, uint8_t *rgb)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", GOLDEN_DIR, name);
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int w, h, mx;
    if (fscanf(f, "P6 %d %d %d", &w, &h, &mx) != 3 || w != 240 || h != 240 || mx != 255 || fgetc(f) == EOF) {
        fclose(f);
        return -2;
    }
    size_t n = fread(rgb, 1, N * 3, f);
    fclose(f);
    return n == N * 3 ? 0 : -3;
}

static face_frame_t frame_of(const golden_case_t *c)
{
    face_frame_t f;
    memset(&f, 0, sizeof(f));
    /* golden param_order == faces_lib param order (checked below) */
    memcpy(f.p, c->p, sizeof(f.p));
    f.blink = c->blink;
    f.wink = (uint8_t)c->wink;
    return f;
}

static uint8_t *A, *B, *C;

static void test_goldens(void)
{
    CHECK(GOLDEN_N_PARAMS == FACE_N_PARAMS);
    for (int i = 0; i < FACE_N_PARAMS; i++) CHECK(strcmp(golden_param_order[i], face_params_info[i].name) == 0);
    int ok = 0;
    for (int k = 0; k < GOLDEN_N_CASES; k++) {
        const golden_case_t *c = &golden_cases[k];
        face_frame_t f = frame_of(c);
        face_sdf_render_rgb888(&f, (face_rect_t){0, 0, 240, 240}, A, 240);
        if (load_ppm(c->file, B) != 0) {
            printf("  FAIL cannot read %s/%s\n", GOLDEN_DIR, c->file);
            fails++;
            continue;
        }
        double sum = 0;
        int close = 0, maxd = 0;
        for (int i = 0; i < N; i++) {
            int worst = 0;
            for (int ch = 0; ch < 3; ch++) {
                int d = abs((int)A[3 * i + ch] - (int)B[3 * i + ch]);
                sum += d;
                if (d > worst) worst = d;
            }
            if (worst <= 24) close++;
            if (worst > maxd) maxd = worst;
        }
        double mean = sum / (N * 3.0), pct = 100.0 * close / N;
        int pass = mean <= 2.0 && pct >= 99.0;
        printf("  %-20s mean|d| %.4f  within24 %.3f%%  max %3d  %s\n", c->name, mean, pct, maxd, pass ? "ok" : "FAIL");
        if (pass) ok++;
        else fails++;
    }
    printf("faces %d/%d\n", ok, GOLDEN_N_CASES);
}

static void test_fastpath(const face_frame_t *f, const char *what)
{
    face_sdf_render_rgb888(f, (face_rect_t){0, 0, 240, 240}, A, 240);
    face_sdf_render_rgb888_ref(f, (face_rect_t){0, 0, 240, 240}, B, 240);
    if (memcmp(A, B, N * 3) != 0) {
        int bad = 0, fx = -1, fy = -1;
        for (int i = 0; i < N; i++) {
            if (memcmp(&A[3 * i], &B[3 * i], 3) && !bad++) fx = i % 240, fy = i / 240;
        }
        printf("  FAIL bbox fast path differs from reference for %s: %d px, first (%d,%d)\n", what, bad, fx, fy);
        fails++;
    }
    face_rect_t bb = face_sdf_bbox(f);
    int outside_ok = 1;
    for (int y = 0; y < 240 && outside_ok; y++) {
        for (int x = 0; x < 240; x++) {
            int inside = x >= bb.x0 && x < bb.x1 && y >= bb.y0 && y < bb.y1;
            if (!inside && (B[3 * (y * 240 + x)] != FACE_BG_R || B[3 * (y * 240 + x) + 1] != FACE_BG_G ||
                            B[3 * (y * 240 + x) + 2] != FACE_BG_B)) {
                outside_ok = 0;
                break;
            }
        }
    }
    if (!outside_ok) printf("  FAIL non-background pixel outside bbox for %s\n", what);
    CHECK(outside_ok);
}

static float frand(float lo, float hi) { return lo + (hi - lo) * (float)rand() / (float)RAND_MAX; }

static void test_fuzz(void)
{
    srand(12345);
    for (int it = 0; it < 150; it++) {
        face_frame_t f;
        memset(&f, 0, sizeof(f));
        for (int i = 0; i < FACE_N_PARAMS; i++) f.p[i] = frand(face_params_info[i].min, face_params_info[i].max);
        f.p[FP_SYMBOL] = (float)(rand() % 6);
        if (rand() % 3 == 0) f.p[FP_LID_TOP] = 0;
        if (rand() % 3 == 0) f.p[FP_LID_BOT] = 0;
        if (rand() % 4 == 0) f.p[FP_MOUTH_OPEN] = 0;
        if (rand() % 4 == 0) f.p[FP_BLUSH] = 0;
        if (rand() % 5 == 0) f.p[FP_MOUTH_W] = 0;
        f.blink = (rand() % 4 == 0) ? 0.05f : 1.0f;
        f.wink = (uint8_t)(rand() % 5 == 0);
        char what[48];
        snprintf(what, sizeof(what), "fuzz #%d", it);
        test_fastpath(&f, what);
        if (fails > 5) return;
    }
}

static void test_partial_and_565(void)
{
    const golden_case_t *c = &golden_cases[3]; /* excited: sparkle, big mouth */
    face_frame_t f = frame_of(c);
    face_sdf_render_rgb888(&f, (face_rect_t){0, 0, 240, 240}, A, 240);
    memset(B, 0x55, N * 3);
    face_rect_t r = {50, 70, 190, 160};
    face_sdf_render_rgb888(&f, r, B, 240);
    int same = 1, untouched = 1;
    for (int y = 0; y < 240; y++) {
        for (int x = 0; x < 240; x++) {
            int in = x >= r.x0 && x < r.x1 && y >= r.y0 && y < r.y1;
            const uint8_t *a = &A[3 * (y * 240 + x)], *b = &B[3 * (y * 240 + x)];
            if (in && memcmp(a, b, 3)) same = 0;
            if (!in && (b[0] != 0x55 || b[1] != 0x55 || b[2] != 0x55)) untouched = 0;
        }
    }
    CHECK(same);
    CHECK(untouched);
    /* out-of-range rectangles are clipped, not written */
    face_sdf_render_rgb888(&f, (face_rect_t){-20, -20, 300, 10}, B, 240);

    uint16_t *px = malloc(N * 2);
    face_sdf_render_rgb565be(&f, (face_rect_t){0, 0, 240, 240}, px, 240);
    int ok565 = 1;
    for (int i = 0; i < N; i++) {
        uint16_t v = (uint16_t)(((A[3 * i] & 0xF8) << 8) | ((A[3 * i + 1] & 0xFC) << 3) | (A[3 * i + 2] >> 3));
        if (px[i] != (uint16_t)((v << 8) | (v >> 8))) ok565 = 0;
    }
    CHECK(ok565);
    free(px);
    CHECK(face_bg_rgb565() == (uint16_t)(((4 & 0xF8) << 8) | ((6 & 0xFC) << 3) | (10 >> 3)));
}

static void test_blend(void)
{
    float a[FACE_N_PARAMS], b[FACE_N_PARAMS], o[FACE_N_PARAMS];
    uint8_t w;
    memcpy(a, face_base, sizeof(a));
    memcpy(b, face_base, sizeof(b));
    a[FP_HUE] = 350.0f, b[FP_HUE] = 10.0f;
    face_blend(a, 0, b, 1, 0.5f, o, &w);
    CHECK(fabsf(o[FP_HUE] - 0.0f) < 1e-3f || fabsf(o[FP_HUE] - 360.0f) < 1e-3f);
    CHECK(w == 1);
    face_blend(a, 0, b, 1, 0.25f, o, &w);
    CHECK(fabsf(o[FP_HUE] - 355.0f) < 1e-3f && w == 0);
    a[FP_HUE] = 10.0f, b[FP_HUE] = 350.0f; /* the other way: 10 -> 350 goes down through 0 */
    face_blend(a, 0, b, 0, 0.5f, o, &w);
    CHECK(fabsf(o[FP_HUE] - 0.0f) < 1e-3f || fabsf(o[FP_HUE] - 360.0f) < 1e-3f);
    a[FP_SYMBOL] = 0, b[FP_SYMBOL] = 5;
    face_blend(a, 0, b, 0, 0.49f, o, &w);
    CHECK(o[FP_SYMBOL] == 0.0f);
    face_blend(a, 0, b, 0, 0.5f, o, &w);
    CHECK(o[FP_SYMBOL] == 5.0f);
    /* emotion at intensity 1 is the resolved emotion; at 0 it is neutral; the wink emotion flags wink */
    uint8_t wk;
    face_emotion_params(FACE_EMO_SAD, 1.0f, o, &wk);
    CHECK(memcmp(o, face_emotions[FACE_EMO_SAD].p, sizeof(o)) == 0 && wk == 0);
    face_emotion_params(FACE_EMO_SAD, 0.0f, o, &wk);
    CHECK(memcmp(o, face_emotions[0].p, sizeof(o)) == 0);
    face_emotion_params(FACE_EMO_WINK, 1.0f, o, &wk);
    CHECK(wk == 1);
    face_emotion_params(FACE_EMO_WINK, 0.4f, o, &wk);
    CHECK(wk == 0);
    face_emotion_params(99, 1.0f, o, &wk); /* unknown id -> neutral */
    CHECK(memcmp(o, face_emotions[0].p, sizeof(o)) == 0);
    CHECK(FACE_N_EMOTIONS == 21 && FACE_N_PARAMS == 23);
}

int main(void)
{
    A = malloc(N * 3), B = malloc(N * 3), C = malloc(N * 3);
    test_goldens();
    for (int k = 0; k < GOLDEN_N_CASES; k++) {
        face_frame_t f = frame_of(&golden_cases[k]);
        test_fastpath(&f, golden_cases[k].name);
    }
    test_fuzz();
    test_partial_and_565();
    test_blend();
    free(A), free(B), free(C);
    printf("%s (%d failures)\n", fails ? "FACE TESTS FAILED" : "face tests ok", fails);
    return fails ? 1 : 0;
}
