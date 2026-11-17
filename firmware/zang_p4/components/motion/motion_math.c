/* motion_math.c - calibration, limiter, trajectory and .zact codec. Pure C; see motion_math.h. */
#include "motion_math.h"

#include <math.h>
#include <string.h>

#define TWO_PI_F 6.28318530718f

/* ================================================================== calibration */
void motion_calib_defaults(motion_calib_t *c)
{
    memset(c, 0, sizeof(*c));
    for (int i = 0; i < MOTION_N; i++) {
        c->j[i].sign = 1;
        c->j[i].tick_max = ROBOT_SERVO_TICK_MAX_DEFAULT;
        c->j[i].ticks_per_rev = ROBOT_SERVO_TICKS_PER_REV_DEFAULT;
        c->j[i].zero_ticks = (int16_t)((ROBOT_SERVO_TICK_MAX_DEFAULT + 1) / 2);
    }
    c->from_nvs = false;
}

uint16_t motion_calib_to_ticks(const motion_calib_joint_t *j, float rad, bool *clamped)
{
    float t = (float)j->zero_ticks + (float)j->sign * rad * ((float)j->ticks_per_rev / TWO_PI_F);
    bool c = false;
    if (t < 1.0f) {
        t = 1.0f;
        c = true;
    } else if (t > (float)j->tick_max) {
        t = (float)j->tick_max;
        c = true;
    }
    if (clamped) *clamped = c;
    return (uint16_t)lroundf(t);
}

float motion_calib_to_rad(const motion_calib_joint_t *j, uint16_t ticks)
{
    return (float)j->sign * ((float)ticks - (float)j->zero_ticks) * (TWO_PI_F / (float)j->ticks_per_rev);
}

/* ================================================================== limiter */
void motion_limiter_reset(motion_limiter_t *l, const float q_now[MOTION_N])
{
    memcpy(l->q, q_now, sizeof(l->q));
    memset(l->v, 0, sizeof(l->v));
}

uint32_t motion_limiter_step(motion_limiter_t *l, const float target[MOTION_N], float dt, float q_out[MOTION_N])
{
    uint32_t clamped = 0;
    if (dt <= 0.0f) {
        memcpy(q_out, l->q, sizeof(l->q));
        return 0;
    }
    for (int i = 0; i < MOTION_N; i++) {
        const robot_joint_def_t *d = &robot_joints[i];
        float tgt = target[i];
        if (isnan(tgt)) tgt = l->q[i]; /* NaN = hold */
        if (tgt < d->lower_rad) {
            tgt = d->lower_rad;
            clamped |= 1u << i;
        } else if (tgt > d->upper_rad) {
            tgt = d->upper_rad;
            clamped |= 1u << i;
        }
        const float a = d->amax_rad_s2, vmax = d->vmax_rad_s;
        float e = tgt - l->q[i];
        float ae = fabsf(e);
        /* Desired velocity: toward the target, capped by vmax and by the braking curve. */
        float v_des = fminf(vmax, sqrtf(2.0f * a * ae));
        if (v_des * dt > ae) v_des = ae / dt; /* do not plan past the target within one step */
        v_des = copysignf(v_des, e);
        float dv = v_des - l->v[i];
        float dv_max = a * dt;
        if (dv > dv_max) dv = dv_max;
        if (dv < -dv_max) dv = -dv_max;
        l->v[i] += dv;
        float q_next = l->q[i] + l->v[i] * dt;
        /* Arrived / crossed the target: snap and stop. A last-line limiter must never overshoot a target
         * (it may be a joint limit); the braking curve keeps the implied deceleration small in practice. */
        if ((e >= 0.0f && q_next >= tgt) || (e <= 0.0f && q_next <= tgt)) {
            q_next = tgt;
            l->v[i] = 0.0f;
        }
        /* Position limits are absolute, whatever the dynamics did. */
        if (q_next < d->lower_rad) q_next = d->lower_rad;
        if (q_next > d->upper_rad) q_next = d->upper_rad;
        l->q[i] = q_next;
        q_out[i] = q_next;
    }
    return clamped;
}

/* ================================================================== trajectory */
void motion_traj_reset(motion_traj_t *t, int64_t t_now, const float q_now[MOTION_N], motion_interp_t interp)
{
    t->head = t->count = 0;
    t->prev.t_ms = t_now;
    memcpy(t->prev.q, q_now, sizeof(t->prev.q));
    t->has_prev2 = false;
    t->interp = interp;
}

static motion_key_t *key_at(motion_traj_t *t, size_t k) { return &t->keys[(t->head + k) % MOTION_TRAJ_CAP]; }

int64_t motion_traj_end_ms(const motion_traj_t *t)
{
    if (t->count == 0) return t->prev.t_ms;
    return t->keys[(t->head + t->count - 1) % MOTION_TRAJ_CAP].t_ms;
}

motion_traj_status_t motion_traj_push(motion_traj_t *t, int64_t t_ms, const float q[MOTION_N])
{
    if (t->count == 0 && t_ms <= t->prev.t_ms) {
        memcpy(t->prev.q, q, sizeof(t->prev.q)); /* immediate target; the limiter smooths it */
        t->has_prev2 = false;
        return MOTION_TRAJ_OK;
    }
    if (t_ms <= motion_traj_end_ms(t)) return MOTION_TRAJ_PAST;
    if (t->count == MOTION_TRAJ_CAP) return MOTION_TRAJ_FULL;
    motion_key_t *k = &t->keys[(t->head + t->count) % MOTION_TRAJ_CAP];
    k->t_ms = t_ms;
    memcpy(k->q, q, sizeof(k->q));
    t->count++;
    return MOTION_TRAJ_OK;
}

bool motion_traj_sample(motion_traj_t *t, int64_t t_now, float q_out[MOTION_N])
{
    while (t->count > 0 && key_at(t, 0)->t_ms <= t_now) {
        t->prev2 = t->prev;
        t->has_prev2 = true;
        t->prev = *key_at(t, 0);
        t->head = (t->head + 1) % MOTION_TRAJ_CAP;
        t->count--;
    }
    if (t->count == 0) {
        memcpy(q_out, t->prev.q, sizeof(t->prev.q));
        return false;
    }
    const motion_key_t *p0 = &t->prev, *p1 = key_at(t, 0);
    float h = (float)(p1->t_ms - p0->t_ms); /* ms, > 0 by construction */
    float u = (float)(t_now - p0->t_ms) / h;
    if (u < 0.0f) u = 0.0f;
    if (u > 1.0f) u = 1.0f;
    if (t->interp == MOTION_INTERP_LINEAR) {
        for (int i = 0; i < MOTION_N; i++) q_out[i] = p0->q[i] + (p1->q[i] - p0->q[i]) * u;
        return true;
    }
    /* Cubic Hermite with Catmull-Rom tangents (per ms); zero tangent at the ends of the known keys. */
    const motion_key_t *pm = t->has_prev2 ? &t->prev2 : NULL;
    const motion_key_t *p2 = t->count >= 2 ? key_at(t, 1) : NULL;
    float u2 = u * u, u3 = u2 * u;
    float h00 = 2 * u3 - 3 * u2 + 1, h10 = u3 - 2 * u2 + u, h01 = -2 * u3 + 3 * u2, h11 = u3 - u2;
    for (int i = 0; i < MOTION_N; i++) {
        float m0 = pm ? (p1->q[i] - pm->q[i]) / (float)(p1->t_ms - pm->t_ms) : 0.0f;
        float m1 = p2 ? (p2->q[i] - p0->q[i]) / (float)(p2->t_ms - p0->t_ms) : 0.0f;
        q_out[i] = h00 * p0->q[i] + h10 * h * m0 + h01 * p1->q[i] + h11 * h * m1;
    }
    return true;
}

/* ================================================================== .zact */
static uint16_t le16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t le32(const uint8_t *p) { return (uint32_t)le16(p) | ((uint32_t)le16(p + 2) << 16); }
static void put16(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
}
static void put32(uint8_t *p, uint32_t v)
{
    put16(p, (uint16_t)v);
    put16(p + 2, (uint16_t)(v >> 16));
}

zact_status_t zact_parse_header(const uint8_t *b, size_t n, zact_header_t *h)
{
    if (!b || !h) return ZACT_ERR_ARG;
    if (n < ZACT_HEADER_SIZE) return ZACT_ERR_SHORT;
    if (memcmp(b, ZACT_MAGIC, 4) != 0) return ZACT_ERR_MAGIC;
    h->version = b[4];
    h->n_joints = b[5];
    h->flags = le16(&b[6]);
    h->n_frames = le32(&b[8]);
    memcpy(h->audio, &b[12], ZACT_AUDIO_LEN);
    h->audio[ZACT_AUDIO_LEN] = '\0';
    if (h->version != ZACT_VERSION) return ZACT_ERR_VERSION;
    if (h->n_joints != MOTION_N) return ZACT_ERR_JOINTS;
    return ZACT_OK;
}

zact_status_t zact_parse_frame(const uint8_t *b, size_t n, const zact_header_t *h, uint32_t *t_ms,
                               int16_t q_cdeg[MOTION_N])
{
    if (!b || !h || !t_ms || !q_cdeg) return ZACT_ERR_ARG;
    if (n < ZACT_FRAME_SIZE(h->n_joints)) return ZACT_ERR_SHORT;
    *t_ms = le32(b);
    for (int i = 0; i < MOTION_N; i++) q_cdeg[i] = (int16_t)le16(&b[4 + 2 * i]);
    return ZACT_OK;
}

zact_status_t zact_write_header(const zact_header_t *h, uint8_t *b, size_t cap)
{
    if (!h || !b) return ZACT_ERR_ARG;
    if (cap < ZACT_HEADER_SIZE) return ZACT_ERR_SHORT;
    memcpy(b, ZACT_MAGIC, 4);
    b[4] = h->version;
    b[5] = h->n_joints;
    put16(&b[6], h->flags);
    put32(&b[8], h->n_frames);
    memset(&b[12], 0, ZACT_AUDIO_LEN);
    memcpy(&b[12], h->audio, strnlen(h->audio, ZACT_AUDIO_LEN));
    return ZACT_OK;
}

zact_status_t zact_write_frame(uint32_t t_ms, const int16_t q_cdeg[MOTION_N], uint8_t *b, size_t cap)
{
    if (!q_cdeg || !b) return ZACT_ERR_ARG;
    if (cap < ZACT_FRAME_SIZE(MOTION_N)) return ZACT_ERR_SHORT;
    put32(b, t_ms);
    for (int i = 0; i < MOTION_N; i++) put16(&b[4 + 2 * i], (uint16_t)q_cdeg[i]);
    return ZACT_OK;
}
