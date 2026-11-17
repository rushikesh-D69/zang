/* zlink.c - ZLink v1.1 codec. Pure C; see zlink.h and firmware/docs/ZLINK.md. */
#include "zlink.h"

#include <string.h>

/* ---------------------------------------------------------------- little-endian cursor helpers */
typedef struct {
    uint8_t *p;
    size_t cap, n;
    bool overflow;
} wr_t;

typedef struct {
    const uint8_t *p;
    size_t len, n;
    bool underflow;
} rd_t;

static void w_u8(wr_t *w, uint8_t v)
{
    if (w->n + 1 > w->cap) {
        w->overflow = true;
        return;
    }
    w->p[w->n++] = v;
}
static void w_u16(wr_t *w, uint16_t v)
{
    w_u8(w, (uint8_t)v);
    w_u8(w, (uint8_t)(v >> 8));
}
static void w_u32(wr_t *w, uint32_t v)
{
    w_u16(w, (uint16_t)v);
    w_u16(w, (uint16_t)(v >> 16));
}
static void w_i16(wr_t *w, int16_t v) { w_u16(w, (uint16_t)v); }
static void w_f32(wr_t *w, float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    w_u32(w, u);
}
static void w_bytes(wr_t *w, const void *src, size_t n)
{
    if (n == 0) return;
    if (!src || w->n + n > w->cap) {
        w->overflow = true;
        return;
    }
    memcpy(&w->p[w->n], src, n);
    w->n += n;
}
static void w_i16s(wr_t *w, const int16_t *v, size_t n)
{
    for (size_t i = 0; i < n; i++) w_i16(w, v[i]);
}

static uint8_t r_u8(rd_t *r)
{
    if (r->n + 1 > r->len) {
        r->underflow = true;
        return 0;
    }
    return r->p[r->n++];
}
static uint16_t r_u16(rd_t *r)
{
    uint16_t lo = r_u8(r);
    return (uint16_t)(lo | ((uint16_t)r_u8(r) << 8));
}
static uint32_t r_u32(rd_t *r)
{
    uint32_t lo = r_u16(r);
    return lo | ((uint32_t)r_u16(r) << 16);
}
static int16_t r_i16(rd_t *r) { return (int16_t)r_u16(r); }
static float r_f32(rd_t *r)
{
    uint32_t u = r_u32(r);
    float f;
    memcpy(&f, &u, sizeof(f)); /* IEEE-754 binary32, little-endian on the wire */
    return f;
}
static void r_i16s(rd_t *r, int16_t *v, size_t n)
{
    for (size_t i = 0; i < n; i++) v[i] = r_i16(r);
}
/* Remaining bytes as a borrowed slice. */
static const uint8_t *r_rest(rd_t *r, uint16_t *n)
{
    *n = (uint16_t)(r->len - r->n);
    const uint8_t *p = &r->p[r->n];
    r->n = r->len;
    return p;
}

static zlink_status_t w_done(const wr_t *w, uint16_t *len)
{
    if (w->overflow) return ZLINK_ERR_NOSPACE;
    if (w->n > ZLINK_MAX_PAYLOAD) return ZLINK_ERR_LEN;
    *len = (uint16_t)w->n;
    return ZLINK_OK;
}
/* Exact-length messages: underflow or leftover bytes are both length errors. */
static zlink_status_t r_done(const rd_t *r) { return (r->underflow || r->n != r->len) ? ZLINK_ERR_LEN : ZLINK_OK; }

const char *zlink_status_str(zlink_status_t st)
{
    switch (st) {
    case ZLINK_OK: return "ok";
    case ZLINK_ERR_ARG: return "bad argument";
    case ZLINK_ERR_SHORT: return "incomplete";
    case ZLINK_ERR_MAGIC: return "bad magic";
    case ZLINK_ERR_VERSION: return "bad version";
    case ZLINK_ERR_LEN: return "bad length";
    case ZLINK_ERR_CRC: return "crc mismatch";
    case ZLINK_ERR_NOSPACE: return "buffer too small";
    case ZLINK_ERR_TYPE: return "wrong type";
    case ZLINK_ERR_VALUE: return "value out of range";
    }
    return "?";
}

/* ---------------------------------------------------------------- CRC */
uint16_t zlink_crc16_update(uint16_t crc, const uint8_t *data, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int b = 0; b < 8; b++) crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
    return crc;
}

/* ---------------------------------------------------------------- frames */
zlink_status_t zlink_encode(uint8_t type, uint16_t seq, const uint8_t *payload, uint16_t len, uint8_t *out,
                            size_t cap, size_t *out_len)
{
    if (!out || (len && !payload)) return ZLINK_ERR_ARG;
    if (len > ZLINK_MAX_PAYLOAD) return ZLINK_ERR_LEN;
    size_t total = ZLINK_HDR_LEN + (size_t)len + ZLINK_CRC_LEN;
    if (cap < total) return ZLINK_ERR_NOSPACE;
    out[0] = ZLINK_MAGIC0;
    out[1] = ZLINK_MAGIC1;
    out[2] = ZLINK_VERSION;
    out[3] = type;
    out[4] = (uint8_t)seq;
    out[5] = (uint8_t)(seq >> 8);
    out[6] = (uint8_t)len;
    out[7] = (uint8_t)(len >> 8);
    if (len) memmove(&out[ZLINK_HDR_LEN], payload, len); /* memmove: payload may already sit in out */
    uint16_t crc = zlink_crc16(&out[2], ZLINK_HDR_LEN - 2 + (size_t)len);
    out[total - 2] = (uint8_t)crc;
    out[total - 1] = (uint8_t)(crc >> 8);
    if (out_len) *out_len = total;
    return ZLINK_OK;
}

zlink_status_t zlink_decode(const uint8_t *buf, size_t n, zlink_frame_t *f, size_t *consumed)
{
    if (!buf || !f) return ZLINK_ERR_ARG;
    if (n >= 1 && buf[0] != ZLINK_MAGIC0) return ZLINK_ERR_MAGIC;
    if (n >= 2 && buf[1] != ZLINK_MAGIC1) return ZLINK_ERR_MAGIC;
    if (n < ZLINK_HDR_LEN) return ZLINK_ERR_SHORT;
    if (buf[2] != ZLINK_VERSION) return ZLINK_ERR_VERSION;
    uint16_t len = (uint16_t)(buf[6] | (buf[7] << 8));
    if (len > ZLINK_MAX_PAYLOAD) return ZLINK_ERR_LEN;
    size_t total = ZLINK_HDR_LEN + (size_t)len + ZLINK_CRC_LEN;
    if (n < total) return ZLINK_ERR_SHORT;
    uint16_t want = (uint16_t)(buf[total - 2] | (buf[total - 1] << 8));
    if (zlink_crc16(&buf[2], ZLINK_HDR_LEN - 2 + (size_t)len) != want) return ZLINK_ERR_CRC;
    f->type = buf[3];
    f->seq = (uint16_t)(buf[4] | (buf[5] << 8));
    f->len = len;
    f->payload = &buf[ZLINK_HDR_LEN];
    if (consumed) *consumed = total;
    return ZLINK_OK;
}

/* ---------------------------------------------------------------- stream */
void zlink_stream_init(zlink_stream_t *s) { memset(s, 0, sizeof(*s)); }

static void stream_drop(zlink_stream_t *s, size_t k)
{
    if (k > s->n) k = s->n;
    memmove(s->buf, &s->buf[k], s->n - k);
    s->n -= k;
}

size_t zlink_stream_push(zlink_stream_t *s, const uint8_t *data, size_t n)
{
    if (s->pending_drop) {
        stream_drop(s, s->pending_drop);
        s->pending_drop = 0;
    }
    size_t room = sizeof(s->buf) - s->n;
    if (n > room) n = room;
    memcpy(&s->buf[s->n], data, n);
    s->n += n;
    return n;
}

bool zlink_stream_next(zlink_stream_t *s, zlink_frame_t *f)
{
    if (s->pending_drop) {
        stream_drop(s, s->pending_drop);
        s->pending_drop = 0;
    }
    while (s->n > 0) {
        size_t used = 0;
        zlink_status_t st = zlink_decode(s->buf, s->n, f, &used);
        if (st == ZLINK_OK) {
            s->pending_drop = used;
            return true;
        }
        if (st == ZLINK_ERR_SHORT) {
            if (s->n == sizeof(s->buf)) { /* cannot happen with a valid frame: buffer holds 2 max frames */
                s->dropped_bytes++;
                stream_drop(s, 1);
                continue;
            }
            return false;
        }
        if (st == ZLINK_ERR_CRC) s->crc_errors++;
        /* Resync: drop one byte and look for the next magic. */
        size_t k = 1;
        while (k < s->n && s->buf[k] != ZLINK_MAGIC0) k++;
        s->dropped_bytes += (uint32_t)k;
        stream_drop(s, k);
    }
    return false;
}

/* ---------------------------------------------------------------- messages */
#define PACK_BEGIN(cap_)                                                                                      \
    if (!m || !p || !len) return ZLINK_ERR_ARG;                                                               \
    wr_t w = {.p = p, .cap = (cap_)}
#define UNPACK_BEGIN()                                                                                        \
    if (!m || (!p && len)) return ZLINK_ERR_ARG;                                                              \
    rd_t r = {.p = p, .len = len}

zlink_status_t zlink_pack_hello(const zlink_hello_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u32(&w, m->host_ms);
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_hello(const uint8_t *p, uint16_t len, zlink_hello_t *m)
{
    UNPACK_BEGIN();
    m->host_ms = r_u32(&r);
    return r_done(&r);
}

zlink_status_t zlink_pack_hello_ack(const zlink_hello_ack_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u8(&w, m->proto_ver);
    w_u8(&w, m->n_joints);
    w_u16(&w, m->fw_ver);
    char name[ZLINK_NAME_LEN] = {0}; /* NUL-padded, not necessarily terminated on the wire */
    memcpy(name, m->name, strnlen(m->name, ZLINK_NAME_LEN));
    w_bytes(&w, name, ZLINK_NAME_LEN);
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_hello_ack(const uint8_t *p, uint16_t len, zlink_hello_ack_t *m)
{
    UNPACK_BEGIN();
    m->proto_ver = r_u8(&r);
    m->n_joints = r_u8(&r);
    m->fw_ver = r_u16(&r);
    memset(m->name, 0, sizeof(m->name));
    for (int i = 0; i < ZLINK_NAME_LEN; i++) m->name[i] = (char)r_u8(&r);
    return r_done(&r);
}

zlink_status_t zlink_pack_ping(const zlink_ping_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u32(&w, m->host_us);
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_ping(const uint8_t *p, uint16_t len, zlink_ping_t *m)
{
    UNPACK_BEGIN();
    m->host_us = r_u32(&r);
    return r_done(&r);
}

zlink_status_t zlink_pack_pong(const zlink_pong_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u32(&w, m->host_us);
    w_u32(&w, m->robot_us);
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_pong(const uint8_t *p, uint16_t len, zlink_pong_t *m)
{
    UNPACK_BEGIN();
    m->host_us = r_u32(&r);
    m->robot_us = r_u32(&r);
    return r_done(&r);
}

zlink_status_t zlink_pack_joint_targets(const zlink_joint_targets_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u32(&w, m->t_robot_ms);
    w_u16(&w, m->mask);
    w_u16(&w, m->move_ms);
    w_i16s(&w, m->q_cdeg, ZLINK_N_JOINTS);
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_joint_targets(const uint8_t *p, uint16_t len, zlink_joint_targets_t *m)
{
    UNPACK_BEGIN();
    m->t_robot_ms = r_u32(&r);
    m->mask = r_u16(&r);
    m->move_ms = r_u16(&r);
    r_i16s(&r, m->q_cdeg, ZLINK_N_JOINTS);
    return r_done(&r);
}

zlink_status_t zlink_pack_traj_chunk(const zlink_traj_chunk_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    if (m->n_frames > ZLINK_TRAJ_MAX_FRAMES || m->interp > ZLINK_INTERP_CUBIC) return ZLINK_ERR_VALUE;
    w_u32(&w, m->stream_id);
    w_u32(&w, m->t0_ms);
    w_u8(&w, m->interp);
    w_u8(&w, m->n_frames);
    for (int i = 0; i < m->n_frames; i++) {
        w_u16(&w, m->frames[i].dt_ms);
        w_i16s(&w, m->frames[i].q_cdeg, ZLINK_N_JOINTS);
    }
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_traj_chunk(const uint8_t *p, uint16_t len, zlink_traj_chunk_t *m)
{
    UNPACK_BEGIN();
    m->stream_id = r_u32(&r);
    m->t0_ms = r_u32(&r);
    m->interp = r_u8(&r);
    m->n_frames = r_u8(&r);
    if (r.underflow) return ZLINK_ERR_LEN;
    if (m->n_frames > ZLINK_TRAJ_MAX_FRAMES || m->interp > ZLINK_INTERP_CUBIC) return ZLINK_ERR_VALUE;
    for (int i = 0; i < m->n_frames; i++) {
        m->frames[i].dt_ms = r_u16(&r);
        r_i16s(&r, m->frames[i].q_cdeg, ZLINK_N_JOINTS);
    }
    return r_done(&r);
}

zlink_status_t zlink_pack_stream_start(const zlink_stream_start_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u32(&w, m->stream_id);
    w_u8(&w, m->clock);
    w_u8(&w, m->flags);
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_stream_start(const uint8_t *p, uint16_t len, zlink_stream_start_t *m)
{
    UNPACK_BEGIN();
    m->stream_id = r_u32(&r);
    m->clock = r_u8(&r);
    m->flags = r_u8(&r);
    return r_done(&r);
}

zlink_status_t zlink_pack_stream_end(const zlink_stream_end_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u32(&w, m->stream_id);
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_stream_end(const uint8_t *p, uint16_t len, zlink_stream_end_t *m)
{
    UNPACK_BEGIN();
    m->stream_id = r_u32(&r);
    return r_done(&r);
}

zlink_status_t zlink_pack_audio_chunk(const zlink_audio_chunk_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u32(&w, m->stream_id);
    w_u32(&w, m->sample_offset);
    w_u8(&w, m->codec);
    w_u8(&w, 0); /* rsv */
    w_bytes(&w, m->data, m->n_data);
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_audio_chunk(const uint8_t *p, uint16_t len, zlink_audio_chunk_t *m)
{
    UNPACK_BEGIN();
    m->stream_id = r_u32(&r);
    m->sample_offset = r_u32(&r);
    m->codec = r_u8(&r);
    (void)r_u8(&r);
    if (r.underflow) return ZLINK_ERR_LEN;
    m->data = r_rest(&r, &m->n_data);
    return ZLINK_OK;
}

zlink_status_t zlink_pack_mic_chunk(const zlink_mic_chunk_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u32(&w, m->robot_ms);
    w_u8(&w, m->codec);
    w_u8(&w, 0);
    w_bytes(&w, m->data, m->n_data);
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_mic_chunk(const uint8_t *p, uint16_t len, zlink_mic_chunk_t *m)
{
    UNPACK_BEGIN();
    m->robot_ms = r_u32(&r);
    m->codec = r_u8(&r);
    (void)r_u8(&r);
    if (r.underflow) return ZLINK_ERR_LEN;
    m->data = r_rest(&r, &m->n_data);
    return ZLINK_OK;
}

zlink_status_t zlink_pack_state(const zlink_state_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u32(&w, m->t_robot_ms);
    w_u16(&w, m->flags);
    w_i16s(&w, m->q_meas_cdeg, ZLINK_N_JOINTS);
    w_i16s(&w, m->q_cmd_cdeg, ZLINK_N_JOINTS);
    w_i16s(&w, m->rpy_cdeg, 3);
    w_i16s(&w, m->gyro_ddps, 3);
    w_i16s(&w, m->acc_mg, 3);
    w_u16(&w, m->vbat_mv);
    w_u8(&w, m->temp_max_c);
    w_u8(&w, m->fault_servo_id);
    if (!m->legacy_88) {
        w_u16(&w, m->stream_buffered_ms);
        w_u32(&w, m->stream_pos_ms);
        w_u8(&w, m->display_mode);
        w_u8(&w, m->emotion);
    }
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_state(const uint8_t *p, uint16_t len, zlink_state_t *m)
{
    UNPACK_BEGIN();
    m->t_robot_ms = r_u32(&r);
    m->flags = r_u16(&r);
    r_i16s(&r, m->q_meas_cdeg, ZLINK_N_JOINTS);
    r_i16s(&r, m->q_cmd_cdeg, ZLINK_N_JOINTS);
    r_i16s(&r, m->rpy_cdeg, 3);
    r_i16s(&r, m->gyro_ddps, 3);
    r_i16s(&r, m->acc_mg, 3);
    m->vbat_mv = r_u16(&r);
    m->temp_max_c = r_u8(&r);
    m->fault_servo_id = r_u8(&r);
    if (r.underflow) return ZLINK_ERR_LEN;
    /* v1.1 tail; a v1.0 sender stops at 88 bytes. Longer payloads (future tails) are accepted and ignored. */
    m->legacy_88 = len < ZLINK_STATE_PAYLOAD_LEN;
    m->stream_buffered_ms = 0;
    m->stream_pos_ms = 0;
    m->display_mode = 0;
    m->emotion = 0;
    if (!m->legacy_88) {
        m->stream_buffered_ms = r_u16(&r);
        m->stream_pos_ms = r_u32(&r);
        m->display_mode = r_u8(&r);
        m->emotion = r_u8(&r);
    }
    return r.underflow ? ZLINK_ERR_LEN : ZLINK_OK;
}

zlink_status_t zlink_pack_cmd(const zlink_cmd_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u8(&w, m->cmd);
    w_bytes(&w, m->data, m->n_data);
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_cmd(const uint8_t *p, uint16_t len, zlink_cmd_t *m)
{
    UNPACK_BEGIN();
    m->cmd = r_u8(&r);
    if (r.underflow) return ZLINK_ERR_LEN;
    m->data = r_rest(&r, &m->n_data);
    return ZLINK_OK;
}

zlink_status_t zlink_pack_ack(const zlink_ack_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u16(&w, m->seq_acked);
    w_u8(&w, m->status);
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_ack(const uint8_t *p, uint16_t len, zlink_ack_t *m)
{
    UNPACK_BEGIN();
    m->seq_acked = r_u16(&r);
    m->status = r_u8(&r);
    return r_done(&r);
}

zlink_status_t zlink_pack_log(const zlink_log_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u8(&w, m->level);
    w_bytes(&w, m->text, m->n_text);
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_log(const uint8_t *p, uint16_t len, zlink_log_t *m)
{
    UNPACK_BEGIN();
    m->level = r_u8(&r);
    if (r.underflow) return ZLINK_ERR_LEN;
    m->text = (const char *)r_rest(&r, &m->n_text);
    return ZLINK_OK;
}

zlink_status_t zlink_pack_event(const zlink_event_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u8(&w, m->kind);
    w_bytes(&w, m->data, m->n_data);
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_event(const uint8_t *p, uint16_t len, zlink_event_t *m)
{
    UNPACK_BEGIN();
    m->kind = r_u8(&r);
    if (r.underflow) return ZLINK_ERR_LEN;
    m->data = r_rest(&r, &m->n_data);
    return ZLINK_OK;
}

/* ---------------------------------------------------------------- CMD data helpers */
zlink_status_t zlink_cmd_go_home_ms(const zlink_cmd_t *c, uint16_t *move_ms)
{
    if (!c || !move_ms || c->cmd != ZLINK_CMD_GO_HOME) return ZLINK_ERR_ARG;
    rd_t r = {.p = c->data, .len = c->n_data};
    *move_ms = r_u16(&r);
    return r_done(&r);
}

zlink_status_t zlink_cmd_set_rate_hz(const zlink_cmd_t *c, uint8_t *hz)
{
    if (!c || !hz || c->cmd != ZLINK_CMD_SET_RATE) return ZLINK_ERR_ARG;
    rd_t r = {.p = c->data, .len = c->n_data};
    *hz = r_u8(&r);
    return r_done(&r);
}

zlink_status_t zlink_cmd_calib_set(const zlink_cmd_t *c, zlink_calib_set_t *out)
{
    if (!c || !out || c->cmd != ZLINK_CMD_CALIB_SET) return ZLINK_ERR_ARG;
    rd_t r = {.p = c->data, .len = c->n_data};
    out->index = r_u8(&r);
    out->zero_ticks = r_i16(&r);
    out->sign = (int8_t)r_u8(&r);
    out->has_ticks_per_rev = c->n_data >= 6;
    out->ticks_per_rev = out->has_ticks_per_rev ? r_u16(&r) : 0;
    zlink_status_t st = r_done(&r);
    if (st != ZLINK_OK) return st;
    if (out->index >= ZLINK_N_JOINTS || (out->sign != 1 && out->sign != -1)) return ZLINK_ERR_VALUE;
    if (out->has_ticks_per_rev && out->ticks_per_rev < 256) return ZLINK_ERR_VALUE;
    return ZLINK_OK;
}

zlink_status_t zlink_cmd_play_file_path(const zlink_cmd_t *c, char *out, size_t cap)
{
    if (!c || !out || cap == 0 || c->cmd != ZLINK_CMD_PLAY_FILE) return ZLINK_ERR_ARG;
    size_t n = c->n_data;
    while (n > 0 && c->data[n - 1] == '\0') n--; /* tolerate a trailing NUL */
    if (n == 0) return ZLINK_ERR_LEN;
    if (n + 1 > cap) return ZLINK_ERR_NOSPACE;
    memcpy(out, c->data, n);
    out[n] = '\0';
    if (memchr(out, '\0', n)) return ZLINK_ERR_VALUE; /* embedded NUL */
    return ZLINK_OK;
}

zlink_status_t zlink_cmd_mic_stream(const zlink_cmd_t *c, zlink_mic_stream_t *out)
{
    if (!c || !out || c->cmd != ZLINK_CMD_MIC_STREAM) return ZLINK_ERR_ARG;
    rd_t r = {.p = c->data, .len = c->n_data};
    out->on = r_u8(&r);
    out->codec = r_u8(&r);
    return r_done(&r);
}

zlink_status_t zlink_cmd_camera_stream(const zlink_cmd_t *c, zlink_camera_stream_t *out)
{
    if (!c || !out || c->cmd != ZLINK_CMD_CAMERA_STREAM) return ZLINK_ERR_ARG;
    rd_t r = {.p = c->data, .len = c->n_data};
    out->on = r_u8(&r);
    out->size = r_u8(&r);
    out->fps = r_u8(&r);
    out->jpeg_quality = r_u8(&r);
    return r_done(&r);
}

zlink_status_t zlink_cmd_percent(const zlink_cmd_t *c, uint8_t *percent)
{
    if (!c || !percent || (c->cmd != ZLINK_CMD_SET_VOLUME && c->cmd != ZLINK_CMD_SET_BRIGHTNESS)) return ZLINK_ERR_ARG;
    rd_t r = {.p = c->data, .len = c->n_data};
    *percent = r_u8(&r);
    zlink_status_t st = r_done(&r);
    return (st == ZLINK_OK && *percent > 100) ? ZLINK_ERR_VALUE : st;
}

/* ---------------------------------------------------------------- v1.1 messages */
zlink_status_t zlink_pack_camera_frame(const zlink_camera_frame_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u32(&w, m->frame_id);
    w_u32(&w, m->robot_ms);
    w_u16(&w, m->w);
    w_u16(&w, m->h);
    w_u8(&w, m->format);
    w_u8(&w, 0);
    w_u32(&w, m->total_len);
    w_u32(&w, m->offset);
    w_bytes(&w, m->data, m->n_data);
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_camera_frame(const uint8_t *p, uint16_t len, zlink_camera_frame_t *m)
{
    UNPACK_BEGIN();
    m->frame_id = r_u32(&r);
    m->robot_ms = r_u32(&r);
    m->w = r_u16(&r);
    m->h = r_u16(&r);
    m->format = r_u8(&r);
    (void)r_u8(&r);
    m->total_len = r_u32(&r);
    m->offset = r_u32(&r);
    if (r.underflow) return ZLINK_ERR_LEN;
    m->data = r_rest(&r, &m->n_data);
    if ((uint64_t)m->offset + m->n_data > m->total_len) return ZLINK_ERR_VALUE;
    return ZLINK_OK;
}

zlink_status_t zlink_pack_calib_report(const zlink_calib_report_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    for (int i = 0; i < ZLINK_N_JOINTS; i++) {
        w_i16(&w, m->joints[i].zero_ticks);
        w_u8(&w, (uint8_t)m->joints[i].sign);
        w_u16(&w, m->joints[i].ticks_per_rev);
    }
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_calib_report(const uint8_t *p, uint16_t len, zlink_calib_report_t *m)
{
    UNPACK_BEGIN();
    for (int i = 0; i < ZLINK_N_JOINTS; i++) {
        m->joints[i].zero_ticks = r_i16(&r);
        m->joints[i].sign = (int8_t)r_u8(&r);
        m->joints[i].ticks_per_rev = r_u16(&r);
    }
    return r_done(&r);
}

zlink_status_t zlink_pack_face_set(const zlink_face_set_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u8(&w, m->emotion);
    w_u8(&w, m->intensity);
    w_u16(&w, m->blend_ms);
    w_u8(&w, (uint8_t)m->gaze_x);
    w_u8(&w, (uint8_t)m->gaze_y);
    w_u8(&w, m->flags);
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_face_set(const uint8_t *p, uint16_t len, zlink_face_set_t *m)
{
    UNPACK_BEGIN();
    m->emotion = r_u8(&r);
    m->intensity = r_u8(&r);
    m->blend_ms = r_u16(&r);
    m->gaze_x = (int8_t)r_u8(&r);
    m->gaze_y = (int8_t)r_u8(&r);
    m->flags = r_u8(&r);
    return r_done(&r);
}

zlink_status_t zlink_pack_face_params(const zlink_face_params_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    if (m->n > ZLINK_FACE_MAX_PARAMS) return ZLINK_ERR_VALUE;
    w_u16(&w, m->blend_ms);
    w_u8(&w, m->n);
    for (int i = 0; i < m->n; i++) {
        w_u8(&w, m->params[i].id);
        w_f32(&w, m->params[i].value);
    }
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_face_params(const uint8_t *p, uint16_t len, zlink_face_params_t *m)
{
    UNPACK_BEGIN();
    m->blend_ms = r_u16(&r);
    m->n = r_u8(&r);
    if (r.underflow) return ZLINK_ERR_LEN;
    if (m->n > ZLINK_FACE_MAX_PARAMS) return ZLINK_ERR_VALUE;
    for (int i = 0; i < m->n; i++) {
        m->params[i].id = r_u8(&r);
        m->params[i].value = r_f32(&r);
    }
    return r_done(&r);
}

zlink_status_t zlink_pack_face_track(const zlink_face_track_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    if (m->n > ZLINK_FACE_MAX_KEYS) return ZLINK_ERR_VALUE;
    w_u32(&w, m->stream_id);
    w_u8(&w, m->n);
    for (int i = 0; i < m->n; i++) {
        w_u32(&w, m->keys[i].t_ms);
        w_u8(&w, m->keys[i].emotion);
        w_u8(&w, m->keys[i].intensity);
        w_u16(&w, m->keys[i].blend_ms);
    }
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_face_track(const uint8_t *p, uint16_t len, zlink_face_track_t *m)
{
    UNPACK_BEGIN();
    m->stream_id = r_u32(&r);
    m->n = r_u8(&r);
    if (r.underflow) return ZLINK_ERR_LEN;
    if (m->n > ZLINK_FACE_MAX_KEYS) return ZLINK_ERR_VALUE;
    for (int i = 0; i < m->n; i++) {
        m->keys[i].t_ms = r_u32(&r);
        m->keys[i].emotion = r_u8(&r);
        m->keys[i].intensity = r_u8(&r);
        m->keys[i].blend_ms = r_u16(&r);
    }
    return r_done(&r);
}

zlink_status_t zlink_pack_display_image(const zlink_display_image_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u16(&w, m->image_id);
    w_u8(&w, m->format);
    w_u8(&w, m->flags);
    w_u32(&w, m->total_len);
    w_u32(&w, m->offset);
    w_bytes(&w, m->data, m->n_data);
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_display_image(const uint8_t *p, uint16_t len, zlink_display_image_t *m)
{
    UNPACK_BEGIN();
    m->image_id = r_u16(&r);
    m->format = r_u8(&r);
    m->flags = r_u8(&r);
    m->total_len = r_u32(&r);
    m->offset = r_u32(&r);
    if (r.underflow) return ZLINK_ERR_LEN;
    m->data = r_rest(&r, &m->n_data);
    if ((uint64_t)m->offset + m->n_data > m->total_len) return ZLINK_ERR_VALUE;
    return ZLINK_OK;
}

zlink_status_t zlink_pack_display_text(const zlink_display_text_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u8(&w, m->size);
    w_u16(&w, m->fg);
    w_u16(&w, m->bg);
    w_u16(&w, m->duration_ms);
    w_bytes(&w, m->text, m->n_text);
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_display_text(const uint8_t *p, uint16_t len, zlink_display_text_t *m)
{
    UNPACK_BEGIN();
    m->size = r_u8(&r);
    m->fg = r_u16(&r);
    m->bg = r_u16(&r);
    m->duration_ms = r_u16(&r);
    if (r.underflow) return ZLINK_ERR_LEN;
    m->text = (const char *)r_rest(&r, &m->n_text);
    return (m->size < 1 || m->size > 4) ? ZLINK_ERR_VALUE : ZLINK_OK;
}

zlink_status_t zlink_pack_display_mode_msg(const zlink_display_mode_msg_t *m, uint8_t *p, size_t cap, uint16_t *len)
{
    PACK_BEGIN(cap);
    w_u8(&w, m->mode);
    w_u16(&w, m->image_id);
    return w_done(&w, len);
}
zlink_status_t zlink_unpack_display_mode_msg(const uint8_t *p, uint16_t len, zlink_display_mode_msg_t *m)
{
    UNPACK_BEGIN();
    m->mode = r_u8(&r);
    m->image_id = r_u16(&r);
    zlink_status_t st = r_done(&r);
    return (st == ZLINK_OK && m->mode > ZLINK_DISPLAY_STATUS) ? ZLINK_ERR_VALUE : st;
}
