/*
 * motion_math.h - pure-C motion building blocks (no IDF; host-tested in test_host/):
 *   calibration   model angle (rad) <-> servo ticks
 *   limiter       last-line position / velocity / acceleration limiter (limits from robot_joints.h)
 *   trajectory    keyframe buffer on a stream clock, linear or cubic-Hermite interpolation
 *   zact          SD-card action file format (firmware/docs/ZLINK.md, "Action files on SD")
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "robot_joints.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MOTION_N ROBOT_N_JOINTS

/* ------------------------------------------------------------------ calibration */
typedef struct {
    int8_t sign;            /* +1 / -1: servo tick direction vs model joint direction */
    int16_t zero_ticks;     /* servo ticks at model angle 0 */
    uint16_t ticks_per_rev; /* ticks per 360 deg */
    uint16_t tick_max;      /* highest valid tick */
} motion_calib_joint_t;

typedef struct {
    motion_calib_joint_t j[MOTION_N];
    bool from_nvs; /* false = defaults: robot is NOT calibrated */
} motion_calib_t;

void motion_calib_defaults(motion_calib_t *c);
/* Clamped to [1, tick_max] (Zing's firmware refused position 0). *clamped set if clamping happened. */
uint16_t motion_calib_to_ticks(const motion_calib_joint_t *j, float rad, bool *clamped);
float motion_calib_to_rad(const motion_calib_joint_t *j, uint16_t ticks);

/* ------------------------------------------------------------------ limiter */
typedef struct {
    float q[MOTION_N];
    float v[MOTION_N];
} motion_limiter_t;

void motion_limiter_reset(motion_limiter_t *l, const float q_now[MOTION_N]);
/*
 * One step of dt seconds toward target: clamp to [lower, upper], then accelerate/brake so |v| <= vmax,
 * |dv/dt| <= amax and the joint can still stop at the target (v <= sqrt(2 a |e|)). Writes q_out.
 * Returns a bitmask of joints whose TARGET was outside the position limits.
 */
uint32_t motion_limiter_step(motion_limiter_t *l, const float target[MOTION_N], float dt, float q_out[MOTION_N]);

/* ------------------------------------------------------------------ trajectory */
#define MOTION_TRAJ_CAP 128

typedef enum { MOTION_INTERP_LINEAR = 0, MOTION_INTERP_CUBIC = 1 } motion_interp_t;

typedef struct {
    int64_t t_ms; /* stream clock */
    float q[MOTION_N];
} motion_key_t;

typedef struct {
    motion_key_t keys[MOTION_TRAJ_CAP]; /* ring buffer of future keys */
    size_t head, count;
    motion_key_t prev;  /* segment start (last key passed, or the anchor pose) */
    motion_key_t prev2; /* the key before prev, for cubic tangents */
    bool has_prev2;
    motion_interp_t interp;
} motion_traj_t;

typedef enum { MOTION_TRAJ_OK = 0, MOTION_TRAJ_FULL = -1, MOTION_TRAJ_PAST = -2 } motion_traj_status_t;

/* Start from pose q_now at stream time t_now. */
void motion_traj_reset(motion_traj_t *t, int64_t t_now, const float q_now[MOTION_N], motion_interp_t interp);
/*
 * Append a key. Keys must be strictly increasing in time. A key at or before the last key is MOTION_TRAJ_PAST,
 * except when the buffer is empty: then it replaces the anchor pose (the limiter smooths the step).
 */
motion_traj_status_t motion_traj_push(motion_traj_t *t, int64_t t_ms, const float q[MOTION_N]);
/* Sample at t_now. Returns false once every key has been passed (q_out = final pose). */
bool motion_traj_sample(motion_traj_t *t, int64_t t_now, float q_out[MOTION_N]);
static inline size_t motion_traj_pending(const motion_traj_t *t) { return t->count; }
static inline size_t motion_traj_room(const motion_traj_t *t) { return MOTION_TRAJ_CAP - t->count; }
int64_t motion_traj_end_ms(const motion_traj_t *t);

/* ------------------------------------------------------------------ .zact files (little-endian) */
#define ZACT_MAGIC        "ZACT"
#define ZACT_VERSION      1
#define ZACT_HEADER_SIZE  44
#define ZACT_AUDIO_LEN    32
#define ZACT_FLAG_AUDIO   0x0001
#define ZACT_FRAME_SIZE(n_joints) (4 + 2 * (size_t)(n_joints))

typedef struct {
    uint8_t version;
    uint8_t n_joints;
    uint16_t flags;
    uint32_t n_frames;
    char audio[ZACT_AUDIO_LEN + 1];
} zact_header_t;

typedef enum {
    ZACT_OK = 0,
    ZACT_ERR_SHORT = -1,
    ZACT_ERR_MAGIC = -2,
    ZACT_ERR_VERSION = -3,
    ZACT_ERR_JOINTS = -4, /* n_joints != ROBOT_N_JOINTS */
    ZACT_ERR_ARG = -5,
} zact_status_t;

zact_status_t zact_parse_header(const uint8_t *b, size_t n, zact_header_t *h);
zact_status_t zact_parse_frame(const uint8_t *b, size_t n, const zact_header_t *h, uint32_t *t_ms,
                               int16_t q_cdeg[MOTION_N]);
zact_status_t zact_write_header(const zact_header_t *h, uint8_t *b, size_t cap);
zact_status_t zact_write_frame(uint32_t t_ms, const int16_t q_cdeg[MOTION_N], uint8_t *b, size_t cap);

#ifdef __cplusplus
}
#endif
