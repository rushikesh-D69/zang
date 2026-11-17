/*
 * Host unit tests for the pure-C firmware modules (built with ASan/UBSan):
 *   zlink codec (CRC known answer, golden vectors from ai_stack, stream resync, length checks)
 *   DS servo packets (byte-exact vs Zing's format, header-check fix, echo skipping, sync write)
 *   last-line limiter, trajectory interpolation, calibration, .zact, app state machine, WAV parser
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "app_sm.h"
#include "ds_proto.h"
#include "motion_math.h"
#include "robot_joints.h"
#include "wav.h"
#include "zlink.h"
#include "zlink_vectors.h"

static int g_fail, g_checks;
#define CHECK(cond)                                                                                              \
    do {                                                                                                         \
        g_checks++;                                                                                              \
        if (!(cond)) {                                                                                           \
            g_fail++;                                                                                            \
            printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                             \
        }                                                                                                        \
    } while (0)
#define CHECK_NEAR(a, b, eps) CHECK(fabs((double)(a) - (double)(b)) <= (eps))

static size_t unhex(const char *hex, uint8_t *out, size_t cap)
{
    size_t n = strlen(hex) / 2;
    if (n > cap) return 0;
    for (size_t i = 0; i < n; i++) {
        unsigned v;
        sscanf(hex + 2 * i, "%2x", &v);
        out[i] = (uint8_t)v;
    }
    return n;
}

/* ====================================================================== zlink */

static void test_crc(void)
{
    CHECK(zlink_crc16((const uint8_t *)"123456789", 9) == 0x29B1);
#if ZV_HAVE_VECTORS
    CHECK(zlink_crc16((const uint8_t *)ZV_CRC_INPUT, strlen(ZV_CRC_INPUT)) == ZV_CRC_VALUE);
#endif
    /* incremental == one shot */
    uint16_t c = zlink_crc16_update(0xFFFF, (const uint8_t *)"1234", 4);
    CHECK(zlink_crc16_update(c, (const uint8_t *)"56789", 5) == 0x29B1);
}

#if ZV_HAVE_VECTORS
/* Pack a vector's fields with the C codec. */
static zlink_status_t pack_vec(const zv_vector_t *v, uint8_t *p, size_t cap, uint16_t *len)
{
    switch (v->type) {
    case ZLINK_HELLO: return zlink_pack_hello(&v->u.hello, p, cap, len);
    case ZLINK_HELLO_ACK: return zlink_pack_hello_ack(&v->u.hello_ack, p, cap, len);
    case ZLINK_PING: return zlink_pack_ping(&v->u.ping, p, cap, len);
    case ZLINK_PONG: return zlink_pack_pong(&v->u.pong, p, cap, len);
    case ZLINK_JOINT_TARGETS: return zlink_pack_joint_targets(&v->u.joint_targets, p, cap, len);
    case ZLINK_TRAJ_CHUNK: return zlink_pack_traj_chunk(&v->u.traj_chunk, p, cap, len);
    case ZLINK_STREAM_START: return zlink_pack_stream_start(&v->u.stream_start, p, cap, len);
    case ZLINK_STREAM_END: return zlink_pack_stream_end(&v->u.stream_end, p, cap, len);
    case ZLINK_STATE: return zlink_pack_state(&v->u.state, p, cap, len);
    case ZLINK_CMD: return zlink_pack_cmd(&v->u.cmd, p, cap, len);
    case ZLINK_ACK: return zlink_pack_ack(&v->u.ack, p, cap, len);
    case ZLINK_CAMERA_FRAME: return zlink_pack_camera_frame(&v->u.camera_frame, p, cap, len);
    case ZLINK_CALIB_REPORT: return zlink_pack_calib_report(&v->u.calib_report, p, cap, len);
    case ZLINK_FACE_SET: return zlink_pack_face_set(&v->u.face_set, p, cap, len);
    case ZLINK_FACE_PARAMS: return zlink_pack_face_params(&v->u.face_params, p, cap, len);
    case ZLINK_FACE_TRACK: return zlink_pack_face_track(&v->u.face_track, p, cap, len);
    case ZLINK_DISPLAY_IMAGE: return zlink_pack_display_image(&v->u.display_image, p, cap, len);
    case ZLINK_DISPLAY_TEXT: return zlink_pack_display_text(&v->u.display_text, p, cap, len);
    case ZLINK_DISPLAY_MODE: return zlink_pack_display_mode_msg(&v->u.display_mode_msg, p, cap, len);
    }
    return ZLINK_ERR_TYPE;
}

/* Unpack into the matching struct and compare every field with the vector. */
static int unpack_matches(const zv_vector_t *v, const uint8_t *p, uint16_t len)
{
    int ok = 1;
#define EQ(field) ok &= (got.field == v->u.MSG.field)
#define EQA(field) ok &= (memcmp(got.field, v->u.MSG.field, sizeof(got.field)) == 0)
    switch (v->type) {
    case ZLINK_HELLO: {
#define MSG hello
        zlink_hello_t got;
        ok &= zlink_unpack_hello(p, len, &got) == ZLINK_OK;
        EQ(host_ms);
#undef MSG
        break;
    }
    case ZLINK_HELLO_ACK: {
#define MSG hello_ack
        zlink_hello_ack_t got;
        ok &= zlink_unpack_hello_ack(p, len, &got) == ZLINK_OK;
        EQ(proto_ver), EQ(n_joints), EQ(fw_ver);
        ok &= strcmp(got.name, v->u.hello_ack.name) == 0;
#undef MSG
        break;
    }
    case ZLINK_PING: {
#define MSG ping
        zlink_ping_t got;
        ok &= zlink_unpack_ping(p, len, &got) == ZLINK_OK;
        EQ(host_us);
#undef MSG
        break;
    }
    case ZLINK_PONG: {
#define MSG pong
        zlink_pong_t got;
        ok &= zlink_unpack_pong(p, len, &got) == ZLINK_OK;
        EQ(host_us), EQ(robot_us);
#undef MSG
        break;
    }
    case ZLINK_JOINT_TARGETS: {
#define MSG joint_targets
        zlink_joint_targets_t got;
        ok &= zlink_unpack_joint_targets(p, len, &got) == ZLINK_OK;
        EQ(t_robot_ms), EQ(mask), EQ(move_ms), EQA(q_cdeg);
#undef MSG
        break;
    }
    case ZLINK_TRAJ_CHUNK: {
#define MSG traj_chunk
        static zlink_traj_chunk_t got;
        ok &= zlink_unpack_traj_chunk(p, len, &got) == ZLINK_OK;
        EQ(stream_id), EQ(t0_ms), EQ(interp), EQ(n_frames);
        for (int k = 0; k < got.n_frames && ok; k++) {
            ok &= got.frames[k].dt_ms == v->u.traj_chunk.frames[k].dt_ms;
            ok &= memcmp(got.frames[k].q_cdeg, v->u.traj_chunk.frames[k].q_cdeg, sizeof(got.frames[k].q_cdeg)) == 0;
        }
#undef MSG
        break;
    }
    case ZLINK_STREAM_START: {
#define MSG stream_start
        zlink_stream_start_t got;
        ok &= zlink_unpack_stream_start(p, len, &got) == ZLINK_OK;
        EQ(stream_id), EQ(clock), EQ(flags);
#undef MSG
        break;
    }
    case ZLINK_STREAM_END: {
#define MSG stream_end
        zlink_stream_end_t got;
        ok &= zlink_unpack_stream_end(p, len, &got) == ZLINK_OK;
        EQ(stream_id);
#undef MSG
        break;
    }
    case ZLINK_STATE: {
#define MSG state
        zlink_state_t got;
        ok &= zlink_unpack_state(p, len, &got) == ZLINK_OK;
        EQ(t_robot_ms), EQ(flags), EQA(q_meas_cdeg), EQA(q_cmd_cdeg), EQA(rpy_cdeg), EQA(gyro_ddps), EQA(acc_mg);
        EQ(vbat_mv), EQ(temp_max_c), EQ(fault_servo_id);
        EQ(stream_buffered_ms), EQ(stream_pos_ms), EQ(display_mode), EQ(emotion), EQ(legacy_88);
#undef MSG
        break;
    }
    case ZLINK_CMD: {
#define MSG cmd
        zlink_cmd_t got;
        ok &= zlink_unpack_cmd(p, len, &got) == ZLINK_OK;
        EQ(cmd), EQ(n_data);
        if (got.n_data) ok &= memcmp(got.data, v->u.cmd.data, got.n_data) == 0;
        if (got.cmd == ZLINK_CMD_GO_HOME) {
            uint16_t ms = 0;
            ok &= zlink_cmd_go_home_ms(&got, &ms) == ZLINK_OK && ms == 1500;
        }
        if (got.cmd == ZLINK_CMD_CAMERA_STREAM) {
            zlink_camera_stream_t cs;
            ok &= zlink_cmd_camera_stream(&got, &cs) == ZLINK_OK && cs.on == 1 && cs.size == 0 && cs.fps == 10 &&
                  cs.jpeg_quality == 60;
        }
#undef MSG
        break;
    }
    case ZLINK_CAMERA_FRAME: {
#define MSG camera_frame
        zlink_camera_frame_t got;
        ok &= zlink_unpack_camera_frame(p, len, &got) == ZLINK_OK;
        EQ(frame_id), EQ(robot_ms), EQ(w), EQ(h), EQ(format), EQ(total_len), EQ(offset), EQ(n_data);
        ok &= memcmp(got.data, v->u.MSG.data, got.n_data) == 0;
#undef MSG
        break;
    }
    case ZLINK_CALIB_REPORT: {
#define MSG calib_report
        zlink_calib_report_t got;
        ok &= zlink_unpack_calib_report(p, len, &got) == ZLINK_OK;
        for (int i = 0; i < ZLINK_N_JOINTS; i++) {
            EQ(joints[i].zero_ticks), EQ(joints[i].sign), EQ(joints[i].ticks_per_rev);
        }
#undef MSG
        break;
    }
    case ZLINK_FACE_SET: {
#define MSG face_set
        zlink_face_set_t got;
        ok &= zlink_unpack_face_set(p, len, &got) == ZLINK_OK;
        EQ(emotion), EQ(intensity), EQ(blend_ms), EQ(gaze_x), EQ(gaze_y), EQ(flags);
#undef MSG
        break;
    }
    case ZLINK_FACE_PARAMS: {
#define MSG face_params
        static zlink_face_params_t got;
        ok &= zlink_unpack_face_params(p, len, &got) == ZLINK_OK;
        EQ(blend_ms), EQ(n);
        for (int i = 0; i < got.n; i++) EQ(params[i].id), EQ(params[i].value);
#undef MSG
        break;
    }
    case ZLINK_FACE_TRACK: {
#define MSG face_track
        static zlink_face_track_t got;
        ok &= zlink_unpack_face_track(p, len, &got) == ZLINK_OK;
        EQ(stream_id), EQ(n);
        for (int i = 0; i < got.n; i++) EQ(keys[i].t_ms), EQ(keys[i].emotion), EQ(keys[i].intensity), EQ(keys[i].blend_ms);
#undef MSG
        break;
    }
    case ZLINK_DISPLAY_IMAGE: {
#define MSG display_image
        zlink_display_image_t got;
        ok &= zlink_unpack_display_image(p, len, &got) == ZLINK_OK;
        EQ(image_id), EQ(format), EQ(flags), EQ(total_len), EQ(offset), EQ(n_data);
        ok &= memcmp(got.data, v->u.MSG.data, got.n_data) == 0;
#undef MSG
        break;
    }
    case ZLINK_DISPLAY_TEXT: {
#define MSG display_text
        zlink_display_text_t got;
        ok &= zlink_unpack_display_text(p, len, &got) == ZLINK_OK;
        EQ(size), EQ(fg), EQ(bg), EQ(duration_ms), EQ(n_text);
        ok &= memcmp(got.text, v->u.MSG.text, got.n_text) == 0;
#undef MSG
        break;
    }
    case ZLINK_DISPLAY_MODE: {
#define MSG display_mode_msg
        zlink_display_mode_msg_t got;
        ok &= zlink_unpack_display_mode_msg(p, len, &got) == ZLINK_OK;
        EQ(mode), EQ(image_id);
#undef MSG
        break;
    }
    default:
        ok = 0;
    }
#undef EQ
#undef EQA
    return ok;
}
#endif

static void test_golden_vectors(void)
{
#if !ZV_HAVE_VECTORS
    printf("  SKIP golden vectors: %s not found (run ai_stack/tools/gen_zlink_vectors.py)\n", ZV_SOURCE);
#else
    for (size_t i = 0; i < ZV_N; i++) {
        const zv_vector_t *v = &zv_vectors[i];
        uint8_t want_payload[ZLINK_MAX_PAYLOAD], want_frame[ZLINK_MAX_FRAME], payload[ZLINK_MAX_PAYLOAD],
            frame[ZLINK_MAX_FRAME];
        size_t np = unhex(v->payload_hex, want_payload, sizeof(want_payload));
        size_t nf = unhex(v->frame_hex, want_frame, sizeof(want_frame));
        /* (a) encode fields -> payload -> frame, byte-exact */
        uint16_t len = 0;
        int ok_pack = pack_vec(v, payload, sizeof(payload), &len) == ZLINK_OK && len == np &&
                      memcmp(payload, want_payload, np) == 0;
        size_t flen = 0;
        int ok_frame = zlink_encode(v->type, v->seq, payload, len, frame, sizeof(frame), &flen) == ZLINK_OK &&
                       flen == nf && memcmp(frame, want_frame, nf) == 0;
        /* (b) decode frame -> fields */
        zlink_frame_t f;
        size_t used = 0;
        int ok_decode = zlink_decode(want_frame, nf, &f, &used) == ZLINK_OK && used == nf && f.type == v->type &&
                        f.seq == v->seq && unpack_matches(v, f.payload, f.len);
        if (!(ok_pack && ok_frame && ok_decode)) {
            printf("  vector %-14s pack=%d frame=%d decode=%d\n", v->name, ok_pack, ok_frame, ok_decode);
        }
        CHECK(ok_pack && ok_frame && ok_decode);
    }
    printf("  %zu golden vectors checked\n", (size_t)ZV_N);
#endif
}

static void test_zlink_errors_and_stream(void)
{
    uint8_t frame[ZLINK_MAX_FRAME], p[64];
    uint16_t len;
    size_t n;
    zlink_ping_t ping = {.host_us = 42};
    CHECK(zlink_pack_ping(&ping, p, sizeof(p), &len) == ZLINK_OK && len == 4);
    CHECK(zlink_encode(ZLINK_PING, 7, p, len, frame, sizeof(frame), &n) == ZLINK_OK && n == 14);
    zlink_frame_t f;
    size_t used;
    CHECK(zlink_decode(frame, n - 1, &f, &used) == ZLINK_ERR_SHORT);
    frame[9] ^= 1;
    CHECK(zlink_decode(frame, n, &f, &used) == ZLINK_ERR_CRC);
    frame[9] ^= 1;
    frame[2] = 2;
    CHECK(zlink_decode(frame, n, &f, &used) == ZLINK_ERR_VERSION);
    frame[2] = 1;
    CHECK(zlink_unpack_ping(p, 3, &ping) == ZLINK_ERR_LEN);
    CHECK(zlink_unpack_ping(p, 5, &ping) == ZLINK_ERR_LEN); /* exact length */
    uint8_t too_big_traj[16] = {0};
    too_big_traj[9] = 41; /* n_frames = 41 > 40 */
    static zlink_traj_chunk_t tc;
    CHECK(zlink_unpack_traj_chunk(too_big_traj, 10, &tc) == ZLINK_ERR_VALUE);

    /* stream: garbage, a frame, a corrupted frame, a frame split across pushes */
    static zlink_stream_t s;
    zlink_stream_init(&s);
    const uint8_t junk[] = {0x00, 0x5A, 0x11, 0xFF, 0x5A};
    uint8_t bad[ZLINK_MAX_FRAME];
    memcpy(bad, frame, n);
    bad[10] ^= 0x55;
    int got = 0;
    zlink_stream_push(&s, junk, sizeof(junk));
    while (zlink_stream_next(&s, &f)) got++;
    zlink_stream_push(&s, frame, n);
    while (zlink_stream_next(&s, &f)) got += (f.type == ZLINK_PING && f.seq == 7);
    zlink_stream_push(&s, bad, n);
    while (zlink_stream_next(&s, &f)) got += 100; /* must not happen */
    zlink_stream_push(&s, frame, 5);
    while (zlink_stream_next(&s, &f)) got += 100;
    zlink_stream_push(&s, frame + 5, n - 5);
    while (zlink_stream_next(&s, &f)) got += (f.type == ZLINK_PING);
    CHECK(got == 2);
    CHECK(s.crc_errors >= 1);

    /* PLAY_FILE path helper */
    zlink_cmd_t c = {.cmd = ZLINK_CMD_PLAY_FILE, .data = (const uint8_t *)"actions/wave.zact\0", .n_data = 18};
    char path[32];
    CHECK(zlink_cmd_play_file_path(&c, path, sizeof(path)) == ZLINK_OK && strcmp(path, "actions/wave.zact") == 0);
    zlink_cmd_t cs = {.cmd = ZLINK_CMD_CALIB_SET, .data = (const uint8_t[]){3, 0x00, 0x08, 0xFF}, .n_data = 4};
    zlink_calib_set_t cal;
    CHECK(zlink_cmd_calib_set(&cs, &cal) == ZLINK_OK && cal.index == 3 && cal.zero_ticks == 2048 && cal.sign == -1 &&
          !cal.has_ticks_per_rev);
    zlink_cmd_t cs2 = {.cmd = ZLINK_CMD_CALIB_SET, .data = (const uint8_t[]){3, 0x00, 0x08, 0x01, 0x55, 0x15}, .n_data = 6};
    CHECK(zlink_cmd_calib_set(&cs2, &cal) == ZLINK_OK && cal.has_ticks_per_rev && cal.ticks_per_rev == 5461);
    /* STATE: a v1.0 (88 B) payload still decodes; v1.1 packs 96 B */
    zlink_state_t st = {.t_robot_ms = 5, .emotion = 2};
    uint8_t sp[128];
    uint16_t sl;
    CHECK(zlink_pack_state(&st, sp, sizeof(sp), &sl) == ZLINK_OK && sl == 96);
    zlink_state_t st2;
    CHECK(zlink_unpack_state(sp, 88, &st2) == ZLINK_OK && st2.legacy_88 && st2.emotion == 0);
    CHECK(zlink_unpack_state(sp, 96, &st2) == ZLINK_OK && !st2.legacy_88 && st2.emotion == 2);
    CHECK(zlink_unpack_state(sp, 87, &st2) == ZLINK_ERR_LEN);
}

/* ====================================================================== DS servo protocol */

static void test_ds_proto(void)
{
    uint8_t b[DS_MAX_PACKET];
    /* Byte-exact against Zing's DS_ServoMove: FF FF id 07 03 2A posH posL timeH timeL chk */
    int n = ds_build_goal(b, sizeof(b), 3, 2048, 20);
    const uint8_t want[] = {0xFF, 0xFF, 0x03, 0x07, 0x03, 0x2A, 0x08, 0x00, 0x00, 0x14, 0x00};
    uint8_t chk = (uint8_t)~(0x03 + 0x07 + 0x03 + 0x2A + 0x08 + 0x00 + 0x00 + 0x14);
    CHECK(n == 11);
    CHECK(memcmp(b, want, 10) == 0 && b[10] == chk);
    /* Zing's torque-enable broadcast: FF FF FE 04 03 28 01 chk */
    n = ds_build_torque(b, sizeof(b), DS_BROADCAST_ID, true);
    CHECK(n == 8 && b[2] == 0xFE && b[3] == 0x04 && b[4] == 0x03 && b[5] == 0x28 && b[6] == 1 &&
          b[7] == ds_checksum(b, 8));
    /* read position: FF FF id 04 02 38 02 chk */
    n = ds_build_read(b, sizeof(b), 5, DS_REG_PRESENT_POSITION, 2);
    CHECK(n == 8 && b[4] == 0x02 && b[5] == 0x38 && b[6] == 0x02);

    /* header check: fixed version rejects FF 00 (Zing's accepted anything starting with FF) */
    CHECK(ds_header_ok((const uint8_t[]){0xFF, 0xFF}));
    CHECK(ds_header_ok((const uint8_t[]){0xFF, 0xF5}));
    CHECK(!ds_header_ok((const uint8_t[]){0xFF, 0x00}));
    CHECK(!ds_header_ok((const uint8_t[]){0x00, 0xFF}));

    /* single-wire echo + reply: request echo (8 B) then reply FF FF 05 04 00 07 D0 chk (pos 2000) */
    uint8_t rx[32];
    int req = ds_build_read(rx, sizeof(rx), 5, DS_REG_PRESENT_POSITION, 2);
    uint8_t reply[] = {0xFF, 0xFF, 0x05, 0x04, 0x00, 0x07, 0xD0, 0x00};
    reply[7] = ds_checksum(reply, sizeof(reply));
    memcpy(rx + req, reply, sizeof(reply));
    ds_reply_t r;
    CHECK(ds_find_reply(rx, (size_t)req + sizeof(reply), (size_t)req, 5, &r) == DS_OK);
    CHECK(r.error == 0 && r.n_params == 2 && ds_be16(r.params) == 2000);
    /* without skipping, the echo itself parses as a frame from id 5 with "error" = 0x02: why skip exists */
    CHECK(ds_find_reply(rx, (size_t)req + sizeof(reply), 0, 5, &r) == DS_OK && r.error == 0x02);
    /* incomplete reply */
    CHECK(ds_find_reply(rx, (size_t)req + 5, (size_t)req, 5, &r) == DS_ERR_SHORT);
    /* corrupted checksum */
    rx[req + 7] ^= 0xFF;
    CHECK(ds_find_reply(rx, (size_t)req + sizeof(reply), (size_t)req, 5, &r) == DS_ERR_CHECKSUM);

    /* sync write for 15 servos: 5 header + 2 + 15*5 params + 1 checksum = 83 bytes */
    uint8_t ids[15];
    uint16_t pos[15];
    for (int i = 0; i < 15; i++) ids[i] = (uint8_t)(i + 1), pos[i] = (uint16_t)(1000 + i);
    n = ds_build_sync_goal(b, sizeof(b), ids, pos, 20, 15);
    CHECK(n == 83 && b[2] == 0xFE && b[4] == 0x83 && b[5] == 0x2A && b[6] == 4);
    CHECK(b[7] == 1 && ds_be16(&b[8]) == 1000 && ds_be16(&b[10]) == 20 && b[12] == 2);
    CHECK(b[82] == ds_checksum(b, 83));
}

/* ====================================================================== motion */

static void test_calib(void)
{
    motion_calib_t c;
    motion_calib_defaults(&c);
    CHECK(!c.from_nvs && c.j[0].zero_ticks == 2048 && c.j[0].ticks_per_rev == ROBOT_SERVO_TICKS_PER_REV_DEFAULT);
    for (float rad = -1.5f; rad <= 1.5f; rad += 0.25f) {
        uint16_t t = motion_calib_to_ticks(&c.j[0], rad, NULL);
        CHECK_NEAR(motion_calib_to_rad(&c.j[0], t), rad, 0.0015);
    }
    c.j[1].sign = -1;
    CHECK(motion_calib_to_ticks(&c.j[1], 0.5f, NULL) < 2048);
    bool clamped = false;
    CHECK(motion_calib_to_ticks(&c.j[0], 10.0f, &clamped) == c.j[0].tick_max && clamped);
    CHECK(motion_calib_to_ticks(&c.j[0], -10.0f, &clamped) == 1 && clamped); /* never 0 (Zing refused 0) */
}

static void test_limiter(void)
{
    motion_limiter_t l;
    float q0[MOTION_N] = {0}, tgt[MOTION_N], q[MOTION_N], prev[MOTION_N];
    motion_limiter_reset(&l, q0);
    for (int i = 0; i < MOTION_N; i++) tgt[i] = 1.0f;
    tgt[0] = 5.0f; /* beyond the upper limit (+1.5708) */
    const float dt = 0.02f;
    float prev_v[MOTION_N] = {0};
    memcpy(prev, q0, sizeof(prev));
    uint32_t clamped = 0;
    int vel_ok = 1, acc_ok = 1, overshoot = 0;
    for (int k = 0; k < 400; k++) {
        clamped |= motion_limiter_step(&l, tgt, dt, q);
        for (int i = 0; i < MOTION_N; i++) {
            const robot_joint_def_t *d = &robot_joints[i];
            float v = (q[i] - prev[i]) / dt;
            if (fabsf(v) > d->vmax_rad_s * 1.001f) vel_ok = 0;
            /* accel limit holds except for the final snap onto the target */
            if (fabsf(v - prev_v[i]) / dt > d->amax_rad_s2 * 1.001f && fabsf(q[i] - (i ? 1.0f : 1.5708f)) > 1e-4f)
                acc_ok = 0;
            if (q[i] > (i ? 1.0f : 1.5708f) + 1e-6f) overshoot = 1;
            prev_v[i] = v;
        }
        memcpy(prev, q, sizeof(q));
    }
    CHECK(vel_ok);
    CHECK(acc_ok);
    CHECK(!overshoot);
    CHECK(clamped & 1u);
    CHECK_NEAR(q[0], robot_joints[0].upper_rad, 1e-6);
    for (int i = 1; i < MOTION_N; i++) CHECK_NEAR(q[i], 1.0, 1e-6);
    /* NaN target = hold */
    float nan_t[MOTION_N];
    for (int i = 0; i < MOTION_N; i++) nan_t[i] = NAN;
    motion_limiter_step(&l, nan_t, dt, q);
    CHECK_NEAR(q[3], 1.0, 1e-6);
}

static void fill(float *q, float v)
{
    for (int i = 0; i < MOTION_N; i++) q[i] = v;
}

static void test_traj(void)
{
    static motion_traj_t t;
    float q0[MOTION_N], q1[MOTION_N], q2[MOTION_N], out[MOTION_N];
    fill(q0, 0), fill(q1, 1), fill(q2, 0);
    motion_traj_reset(&t, 0, q0, MOTION_INTERP_LINEAR);
    CHECK(motion_traj_push(&t, 100, q1) == MOTION_TRAJ_OK);
    CHECK(motion_traj_push(&t, 200, q2) == MOTION_TRAJ_OK);
    CHECK(motion_traj_push(&t, 150, q1) == MOTION_TRAJ_PAST); /* not after the last key */
    CHECK(motion_traj_sample(&t, 50, out) && fabsf(out[0] - 0.5f) < 1e-6f);
    CHECK(motion_traj_sample(&t, 150, out) && fabsf(out[4] - 0.5f) < 1e-6f);
    CHECK(!motion_traj_sample(&t, 250, out) && fabsf(out[0]) < 1e-6f); /* finished, holds the last key */

    /* cubic: passes through keys, zero velocity at the start (anchor), smooth in between */
    motion_traj_reset(&t, 0, q0, MOTION_INTERP_CUBIC);
    motion_traj_push(&t, 100, q1);
    motion_traj_push(&t, 200, q2);
    motion_traj_sample(&t, 100, out);
    CHECK_NEAR(out[0], 1.0, 1e-5);
    motion_traj_reset(&t, 0, q0, MOTION_INTERP_CUBIC);
    motion_traj_push(&t, 100, q1);
    float a[MOTION_N], b[MOTION_N];
    motion_traj_sample(&t, 1, a);
    motion_traj_sample(&t, 2, b);
    CHECK(a[0] < 0.001f && b[0] - a[0] < 0.002f); /* starts with ~zero velocity */

    /* empty buffer: a key in the past replaces the anchor (immediate target, limiter smooths it) */
    motion_traj_reset(&t, 1000, q0, MOTION_INTERP_LINEAR);
    CHECK(motion_traj_push(&t, 900, q1) == MOTION_TRAJ_OK);
    CHECK(!motion_traj_sample(&t, 1000, out) && fabsf(out[0] - 1.0f) < 1e-6f);

    /* capacity */
    motion_traj_reset(&t, 0, q0, MOTION_INTERP_LINEAR);
    int full = 0;
    for (int k = 1; k <= MOTION_TRAJ_CAP + 1; k++) full |= motion_traj_push(&t, k * 10, q1) == MOTION_TRAJ_FULL;
    CHECK(full && motion_traj_pending(&t) == MOTION_TRAJ_CAP);
}

static void test_zact(void)
{
    zact_header_t h = {.version = ZACT_VERSION, .n_joints = MOTION_N, .flags = ZACT_FLAG_AUDIO, .n_frames = 2};
    strcpy(h.audio, "wave.wav");
    uint8_t buf[ZACT_HEADER_SIZE + 2 * ZACT_FRAME_SIZE(MOTION_N)];
    CHECK(zact_write_header(&h, buf, sizeof(buf)) == ZACT_OK);
    int16_t q[MOTION_N], q2[MOTION_N];
    for (int i = 0; i < MOTION_N; i++) q[i] = (int16_t)(i * 100 - 700);
    CHECK(zact_write_frame(1234, q, buf + ZACT_HEADER_SIZE, ZACT_FRAME_SIZE(MOTION_N)) == ZACT_OK);
    zact_header_t g;
    CHECK(zact_parse_header(buf, sizeof(buf), &g) == ZACT_OK && g.n_frames == 2 && strcmp(g.audio, "wave.wav") == 0);
    uint32_t t;
    CHECK(zact_parse_frame(buf + ZACT_HEADER_SIZE, ZACT_FRAME_SIZE(MOTION_N), &g, &t, q2) == ZACT_OK && t == 1234);
    CHECK(memcmp(q, q2, sizeof(q)) == 0);
    buf[5] = 16;
    CHECK(zact_parse_header(buf, sizeof(buf), &g) == ZACT_ERR_JOINTS);
    buf[0] = 'X';
    CHECK(zact_parse_header(buf, sizeof(buf), &g) == ZACT_ERR_MAGIC);
}

/* ====================================================================== app / audio */

static void test_app_sm(void)
{
    app_state_t s = APP_BOOT;
    CHECK(!app_sm_step(&s, APP_EV_STREAM_START) && s == APP_BOOT);
    CHECK(app_sm_step(&s, APP_EV_BOOT_DONE) && s == APP_IDLE);
    CHECK(!app_sm_can_move(s));
    CHECK(!app_sm_step(&s, APP_EV_PLAY_FILE) && s == APP_IDLE); /* no motion without torque */
    CHECK(app_sm_step(&s, APP_EV_TORQUE_ON) && s == APP_HOLD && app_sm_can_move(s));
    CHECK(app_sm_step(&s, APP_EV_STREAM_START) && s == APP_STREAMING);
    CHECK(app_sm_step(&s, APP_EV_WATCHDOG) && s == APP_HOLD);
    CHECK(app_sm_step(&s, APP_EV_PLAY_FILE) && s == APP_PLAYING_FILE);
    CHECK(app_sm_step(&s, APP_EV_FILE_DONE) && s == APP_HOLD);
    CHECK(app_sm_step(&s, APP_EV_ESTOP) && s == APP_ESTOP);
    CHECK(!app_sm_step(&s, APP_EV_HOLD) && s == APP_ESTOP); /* latched */
    CHECK(app_sm_step(&s, APP_EV_TORQUE_ON) && s == APP_HOLD);
    for (int st = 0; st < APP_STATE_COUNT; st++) { /* ESTOP is accepted from every state */
        app_state_t x = (app_state_t)st;
        CHECK(app_sm_step(&x, APP_EV_ESTOP) && x == APP_ESTOP);
    }
}

static void test_wav(void)
{
    uint8_t h[44] = "RIFF\0\0\0\0WAVEfmt ";
    h[16] = 16;
    h[20] = 1;               /* PCM */
    h[22] = 1;               /* mono */
    h[24] = 0x80, h[25] = 0x3E; /* 16000 */
    h[34] = 16;              /* bits */
    memcpy(&h[36], "data", 4);
    h[40] = 0x00, h[41] = 0x7D; /* 32000 bytes */
    wav_info_t w;
    CHECK(wav_parse(h, sizeof(h), &w) == 0 && w.sample_rate == 16000 && w.channels == 1 && w.data_offset == 44 &&
          w.data_bytes == 32000);
    h[34] = 8;
    CHECK(wav_parse(h, sizeof(h), &w) < 0); /* 8-bit rejected */
}

static void test_face_track_100(void)
{
    /* ai_stack/zang_ai/link/zlink.py face_track() sends up to 100 keys per message: the C codec must take them */
    static zlink_face_track_t a, b;
    a.stream_id = 7, a.n = 100;
    for (int i = 0; i < 100; i++) a.keys[i].t_ms = (uint32_t)i * 100, a.keys[i].emotion = (uint8_t)(i % 21),
                                  a.keys[i].intensity = 200, a.keys[i].blend_ms = 150;
    uint8_t p[ZLINK_MAX_PAYLOAD];
    uint16_t len;
    CHECK(zlink_pack_face_track(&a, p, sizeof(p), &len) == ZLINK_OK && len == 5 + 8 * 100);
    CHECK(zlink_unpack_face_track(p, len, &b) == ZLINK_OK && b.n == 100 && b.keys[99].t_ms == 9900 &&
          b.keys[99].emotion == 99 % 21);
    a.n = 101;
    CHECK(zlink_pack_face_track(&a, p, sizeof(p), &len) == ZLINK_ERR_VALUE);
}

int main(void)
{
    struct {
        const char *name;
        void (*fn)(void);
    } tests[] = {
        {"zlink crc", test_crc},
        {"zlink golden vectors", test_golden_vectors},
        {"zlink errors + stream resync", test_zlink_errors_and_stream},
        {"ds servo protocol", test_ds_proto},
        {"calibration", test_calib},
        {"limiter", test_limiter},
        {"trajectory", test_traj},
        {".zact", test_zact},
        {"app state machine", test_app_sm},
        {"wav parser", test_wav},
        {"face track 100 keys", test_face_track_100},
    };
    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        int before = g_fail;
        tests[i].fn();
        printf("%-32s %s\n", tests[i].name, g_fail == before ? "ok" : "FAILED");
    }
    printf("\n%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
