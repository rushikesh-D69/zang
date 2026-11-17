/* face_bench.c - host render time per 240x240 frame (built -O2, no sanitizers, like the firmware's -O2). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "face_sdf.h"
#include "faces_goldens.h"

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int main(void)
{
    uint16_t *px = malloc(240 * 240 * 2);
    const int reps = 20;
    double tot = 0, worst = 0, tot_bb = 0, worst_bb = 0, sum_area = 0;
    const char *worst_name = "";
    for (int k = 0; k < GOLDEN_N_CASES; k++) {
        face_frame_t f;
        memset(&f, 0, sizeof(f));
        memcpy(f.p, golden_cases[k].p, sizeof(f.p));
        f.blink = golden_cases[k].blink, f.wink = (uint8_t)golden_cases[k].wink;
        double t0 = now_s();
        for (int i = 0; i < reps; i++) face_sdf_render_rgb565be(&f, (face_rect_t){0, 0, 240, 240}, px, 240);
        double full = (now_s() - t0) / reps;
        face_rect_t bb = face_sdf_bbox(&f);
        t0 = now_s();
        for (int i = 0; i < reps; i++) face_sdf_render_rgb565be(&f, bb, px, 240);
        double part = (now_s() - t0) / reps;
        tot += full, tot_bb += part;
        sum_area += (double)(bb.x1 - bb.x0) * (bb.y1 - bb.y0);
        if (full > worst) worst = full, worst_name = golden_cases[k].name;
        if (part > worst_bb) worst_bb = part;
    }
    printf("host -O2, RGB565, %d golden cases\n", GOLDEN_N_CASES);
    printf("  full 240x240 frame : mean %.3f ms, worst %.3f ms (%s)\n", 1e3 * tot / GOLDEN_N_CASES, 1e3 * worst,
           worst_name);
    printf("  bbox-only frame    : mean %.3f ms, worst %.3f ms (mean bbox %.0f px = %.0f %% of the panel)\n",
           1e3 * tot_bb / GOLDEN_N_CASES, 1e3 * worst_bb, sum_area / GOLDEN_N_CASES,
           100.0 * sum_area / GOLDEN_N_CASES / (240.0 * 240.0));
    free(px);
    return 0;
}
